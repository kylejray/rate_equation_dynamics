"""
Continuous-Time Markov Chain (CTMC) dynamics and entropy production analysis.

Implements the framework from:
    "Large Interconnected Thermodynamic Systems Nearly Minimize Entropy Production"
    K. J. Ray and A. B. Boyd (2025), arXiv:2507.10476

Provides tools for:
    - Constructing CTMCs from various generator functions (Arrhenius, cyclic, etc.)
    - Computing the nonequilibrium steady state (NESS)
    - Computing the minimum entropy production state (MEPS)
    - Calculating entropy production rates (EPR) and related thermodynamic quantities
    - Supporting arbitrary time-reversal involutions for non-time-symmetric states
"""

import numpy as np

# Optional JAX support for improved MEPS optimization
try:
    import jax
    import jax.numpy as jnp
    from jax.scipy.special import logsumexp as jax_logsumexp
    jax.config.update("jax_enable_x64", True)  # match numpy float64 precision
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# Optional jaxopt support for vmappable L-BFGS
try:
    import jaxopt
    HAS_JAXOPT = True
except ImportError:
    HAS_JAXOPT = False

# Suppress noisy jaxopt linesearch diagnostics (harmless near-convergence warnings).
# jaxopt uses jax.debug.print which bypasses Python logging, so we filter here.
if HAS_JAX and HAS_JAXOPT:
    _original_jax_debug_print = jax.debug.print
    def _filtered_jax_debug_print(fmt, *args, **kwargs):
        if isinstance(fmt, str) and 'jaxopt' in fmt:
            return
        return _original_jax_debug_print(fmt, *args, **kwargs)
    jax.debug.print = _filtered_jax_debug_print


# ---------------------------------------------------------------------------
# Rate matrix generators
# ---------------------------------------------------------------------------

def uniform_generator(S=10, N=1, min_rate_tol=6):
    """Generate rate matrices with uniformly distributed off-diagonal rates.

    Parameters
    ----------
    S : int
        Number of states.
    N : int
        Number of independent rate matrices (batch size).
    min_rate_tol : int
        Rates are drawn from U(10^{-min_rate_tol}, 1) and rounded.

    Returns
    -------
    np.ndarray of shape (N, S, S)
    """
    return np.random.uniform(10**(-min_rate_tol), 1, (N, S, S)).round(decimals=min_rate_tol)


def normal_generator(S=10, N=1, mu=0, sigma=1):
    """Generate rate matrices with |Normal(mu, sigma)| off-diagonal rates."""
    return np.abs(np.random.normal(mu, sigma, (N, S, S)))


def gamma_generator(S=10, N=1, mu=1, sigma=0.1):
    """Generate rate matrices with Gamma-distributed off-diagonal rates."""
    return np.random.gamma(mu**2 / sigma**2, sigma**2 / mu, (N, S, S))


def cyclic_generator(S=5, N=1, mu=0, sigma=1, max_jump=None, max_reverse_rate=1.5):
    """Generate rate matrices with 1-D ring (cyclic) connectivity.

    Forward rates decrease exponentially with jump distance. Reverse rates
    are drawn as a fraction of the forward rates to create a preferred
    direction. The ends of the ring are connected by a fast "catalytic" jump.

    Parameters
    ----------
    S : int
        Number of states.
    N : int
        Batch size.
    mu, sigma : float
        Parameters for the Normal noise added to log-forward rates.
    max_jump : int or None
        Maximum jump distance on the ring (None = S-1, fully connected).
    max_reverse_rate : float
        Reverse rates are drawn as U(0, max_reverse_rate) * forward rate.
    """
    if max_jump is None or max_jump > S - 1:
        max_jump = S - 1

    R = np.zeros((N, S, S))

    for i in range(1, max_jump + 1):
        # Forward rates: decrease exponentially in distance, with Normal noise
        new_vals_p = np.exp(-i + np.random.normal(mu, sigma, size=(N, S)))
        # Reverse rates: damped relative to forward to set preferred direction
        new_vals_m = new_vals_p * np.random.uniform(0, max_reverse_rate, size=(N, S))

        # Set transition rates one diagonal at a time
        R += np.multiply(new_vals_p[..., None], np.eye(S, k=i))
        R += np.multiply(new_vals_m[..., None], np.eye(S, k=-i))

    # Connect the ends of the ring with a fast catalytic jump
    R[..., -1, 0] = R.max(axis=-1).max(axis=-1)
    R[..., 0, -1] = R[..., -1, 0]

    return R


def detailed_balance_generator(S=5, N=1, energy=None, energy_gen=np.random.uniform,
                               beta=1, gen_args=[0.1, 1]):
    """Generate rate matrices satisfying detailed balance for given energies.

    R_{s->s'} = exp(beta * (E_s - E_{s'})), guaranteeing the Boltzmann
    distribution as the steady state.
    """
    if energy is None:
        energy = energy_gen(*gen_args, size=(N, S))
    assert energy.shape == (N, S)
    return np.exp(beta * (energy[:, :, None] - energy[:, None, :]))


def arrhenius_pump_generator(S=5, N=1, energy=None, barrier=None,
                             energy_gen=np.random.uniform, beta=1,
                             gen_args=[-1, 1], n_pumps=0, pump_strength=5):
    """Generate Arrhenius rate matrices with optional nonequilibrium pumps.

    Transition rates follow the Arrhenius form (Eq. 20-21 in the paper):
        R_{s->s'} = K * exp(-(E_barrier - E_s) / k_B T)
    with additional pump forces F_{s->s'} on selected transitions.

    Parameters
    ----------
    S : int
        Number of states.
    N : int
        Batch size.
    energy : np.ndarray or None
        State energies of shape (N, S). Generated if None.
    barrier : np.ndarray or None
        Energy barriers of shape (N, n_pairs). Generated if None.
    n_pumps : int
        Number of transitions to drive with nonequilibrium pumps.
    pump_strength : float
        Maximum pump force magnitude (in units of energy scale).
    """
    n_pairs = int((S**2 - S) / 2)

    # Choose minimal integer type for memory efficiency
    int_type = 'uint32'
    largest_int = max(S, N)
    int_bytes = np.ceil(np.log2(max(largest_int, 1)))
    for precision in [8, 16, 32]:
        if int_bytes < precision:
            int_type = f'uint{precision}'
            break

    if energy is None:
        energy = energy_gen(*gen_args, size=(N, S))
    assert energy.shape == (N, S)

    if barrier is None:
        barrier = energy_gen(*gen_args, size=(N, n_pairs))
    assert barrier.shape == (N, n_pairs)

    barrier = np.abs(barrier)
    delta_E = -energy[:, :, None] + energy[:, None, :]
    # Find transitions where delta_E is negative (high-to-low energy)
    idx = np.argwhere(delta_E < 0).astype(int_type).reshape(N, -1, 3)

    # High-to-low energy transitions: just overcome the barrier
    delta_E[idx[..., 0], idx[..., 1], idx[..., 2]] = barrier
    # Low-to-high energy transitions: overcome energy difference + barrier
    delta_E[idx[..., 0], idx[..., 2], idx[..., 1]] += barrier

    barrier = None  # free memory

    # Add catalytic pumps to randomly selected one-way transitions
    if n_pumps > 0:
        idx = np.zeros(shape=(N, n_pumps, 3), dtype=int_type)
        idx[..., 1:] = np.random.randint(0, S, size=(N, n_pumps, 2), dtype=int_type)
        idx[..., 0] = np.floor(np.arange(0, N * n_pumps) / n_pumps).reshape(N, -1).astype(int_type)
        delta_E[tuple(idx.T)] += (energy_gen(*gen_args, size=(N, n_pumps)) * pump_strength).T

    delta_E = np.exp(-delta_E)
    return delta_E


def exponential_generator(S=5, N=1, scale=1, slowstate=False):
    """Generate rate matrices with exponentially distributed rates.

    If slowstate=True, transitions to/from state 0 are slowed by 100x.
    """
    R = np.random.exponential(scale, (N, S, S))
    if slowstate:
        R[..., 0, :] *= 0.01
        R[..., :, 0] *= 0.01
    return R


# ---------------------------------------------------------------------------
# JAX helper functions for MEPS optimization (module-level for JIT)
# ---------------------------------------------------------------------------

if HAS_JAX:
    @jax.jit
    def _jax_epr_single(theta, R, statewise_Q):
        """EPR as a function of unconstrained theta (single system).

        Uses softmax parameterization: p = softmax(theta), so theta lives
        in unconstrained R^S and we minimize without simplex constraints.
        """
        log_p = theta - jax_logsumexp(theta)
        p = jnp.exp(log_p)
        sigma_s = -R @ log_p + statewise_Q
        return jnp.dot(p, sigma_s)

    _jax_epr_grad = jax.jit(jax.grad(_jax_epr_single, argnums=0))

    def _jax_solve_single_lbfgs(theta0, R, statewise_Q, max_iter, tol):
        """Solve MEPS for a single system using jaxopt L-BFGS."""
        solver = jaxopt.LBFGS(
            fun=_jax_epr_single,
            maxiter=max_iter,
            tol=tol,
            linesearch='zoom',
        )
        result = solver.run(theta0, R=R, statewise_Q=statewise_Q)
        return result.params

    def _jax_solve_single_scipy(theta0, R, statewise_Q, max_iter, tol):
        """Solve MEPS for a single system using jax.scipy L-BFGS-B."""
        def objective(theta):
            return _jax_epr_single(theta, R, statewise_Q)
        result = jax.scipy.optimize.minimize(
            objective, theta0, method='BFGS',
            options={'maxiter': max_iter, 'gtol': tol},
        )
        return result.x

    def _jax_solve_batch_lbfgs(theta0_batch, R_batch, Q_batch, max_iter, tol):
        """Solve MEPS for a batch of systems using vmapped L-BFGS."""
        solver = jaxopt.LBFGS(
            fun=_jax_epr_single, maxiter=max_iter, tol=tol,
            linesearch='zoom', verbose=False,
        )
        def solve_one(theta0, R, statewise_Q):
            result = solver.run(theta0, R=R, statewise_Q=statewise_Q)
            return result.params
        return jax.vmap(solve_one)(theta0_batch, R_batch, Q_batch)


# ---------------------------------------------------------------------------
# Core CTMC class
# ---------------------------------------------------------------------------

class ContinuousTimeMarkovChain():
    """Continuous-time Markov chain with entropy production analysis.

    Supports both single rate matrices and batched computation over N
    independent systems. Computes the nonequilibrium steady state (NESS),
    minimum entropy production state (MEPS), and various thermodynamic
    quantities.

    Time-reversal symmetry can be configured via:
        - time_even_states=True: identity involution (all states time-symmetric)
        - time_even_states=False: arbitrary involution via involution_indices

    Parameters
    ----------
    R : np.ndarray or None
        Rate matrix of shape (S, S) or (N, S, S) for batched mode.
        If None, a rate matrix is generated using `generator`.
    generator : callable
        Generator function to create rate matrices.
    **gen_kwargs
        Keyword arguments passed to the generator.

    Attributes
    ----------
    R : np.ndarray
        The (normalized) rate matrix.
    rev_R : np.ndarray
        The time-reversed rate matrix.
    S : int
        Number of states.
    batch : bool
        Whether operating in batched mode.
    statewise_Q : np.ndarray
        Precomputed state-wise environmental entropy flow rates.
    forbidden_mask : np.ndarray
        Boolean mask of forbidden (zero-rate) transitions.
    """

    # Convergence tolerances (override these on the instance if needed)
    CONVERGENCE_ATOL = 1E-4       # absolute tolerance for derivative convergence
    CONVERGENCE_FLOOR = 1E-12     # floor for relative derivative normalization
    DT_DECAY_RATE = 0.95          # multiplicative decay for adaptive timestep
    EPR_INCREASE_TOL = 1E-1       # tolerance for EPR increase detection in MEPS
    EPR_ATOL = 1E-15              # EPR values this small are at float64 machine noise
                                  # (~4.5e-16 worst case).  Used only as an optimizer
                                  # convergence criterion — get_epr() always reports
                                  # the raw computed value.

    def __init__(self, R=None, generator=uniform_generator, **gen_kwargs):
        self.scale = 1
        self.timescale = 1
        self.batch = False
        self.time_even_states = True
        self.analytic_threshhold = 200
        self.min_rate = 1E-12
        self.min_state = 1E-32
        self.verbose = False
        self.forbidden_mask = None
        if R is not None:
            self.set_rate_matrix(R)
        else:
            self.generator = generator
            self.gen_kwargs = gen_kwargs
            self.set_rate_matrix(self.generator(**self.gen_kwargs))

    def __repr__(self):
        mode = 'batch' if self.batch else 'single'
        n_batch = self.R.shape[0] if self.batch else 1
        sym = 'time-even' if self.time_even_states else 'involution'
        return f'ContinuousTimeMarkovChain(S={self.S}, N={n_batch}, mode={mode}, symmetry={sym})'

    # -------------------------------------------------------------------
    # Rate matrix setup
    # -------------------------------------------------------------------

    def set_rate_matrix(self, R, max_rate=1):
        """Set (or reset) the rate matrix, clearing cached NESS/MEPS.

        Validates, normalizes, identifies forbidden transitions, builds
        the time-reversed matrix, and precomputes statewise_Q.
        """
        for attr in ('ness', 'meps'):
            if hasattr(self, attr):
                delattr(self, attr)
        R = self.verify_rate_matrix(R)

        self.R = R
        if self.time_even_states:
            self.rev_R = self.R
        else:
            P = self.get_reversal_matrix()
            self.rev_R = P @ self.R @ P.T
            # Zero out forbidden transitions in rev_R
            if self.batch:
                rev_R_T = self.rev_R.transpose(0, 2, 1)
                rev_R_T[self.forbidden_mask] = 0
                self.rev_R = rev_R_T.transpose(0, 2, 1)
            else:
                rev_R_T = self.rev_R.T
                rev_R_T[self.forbidden_mask] = 0
                self.rev_R = rev_R_T.T

        self._set_statewise_Q()

    def _R_info(self, R):
        """Return (num_states, shape, size) for the per-system rate matrix."""
        if self.batch:
            return len(R[0, ...]), R[0, ...].shape, R[0, ...].size
        else:
            return len(R), R.shape, R.size

    def _set_diags(self, R):
        """Set diagonal entries so that each row sums to zero."""
        R = np.abs(R)
        if self.batch:
            R -= R.sum(axis=-1)[:, :, np.newaxis] * np.identity(self.S)
        else:
            R -= np.identity(self.S) * R.sum(axis=-1)
        return R

    def verify_rate_matrix(self, R):
        """Validate and prepare a rate matrix.

        - Converts to numpy array if needed
        - Detects batch mode from shape
        - Normalizes rates
        - Identifies and symmetrically enforces forbidden transitions
        - Sets diagonal entries for row-sum = 0
        """
        if not isinstance(R, np.ndarray):
            R = np.array(R)
        R = np.squeeze(R)
        if len(R.shape) == 3:
            self.batch = True

        R = self.normalize_R(R)

        S, shape, size = self._R_info(R)

        assert len(shape) == 2 and size == S**2, 'each R must be a 2D square matrix'
        assert (np.sign(R - R * np.identity(S)) >= np.zeros(shape) - np.identity(S)).all(), \
            'R_ij must be >= 0 for i != j'

        self.S = S

        # Identify forbidden transitions (those below min_rate threshold)
        forbidden = np.abs(R) <= self.min_rate
        self.forbidden_mask = forbidden & (~np.eye(S, dtype=bool))

        if self.time_even_states:
            # Time-even: if R[i,j] is forbidden, R[j,i] must be too
            if self.batch:
                self.forbidden_mask = self.forbidden_mask | self.forbidden_mask.transpose(0, 2, 1)
            else:
                self.forbidden_mask = self.forbidden_mask | self.forbidden_mask.T
        else:
            # Time-odd: enforce involution symmetry for forbidden transitions
            # If s->s' is forbidden, then inv(s')->inv(s) must also be forbidden
            P = self.get_reversal_matrix()
            involution_mapped = P @ self.forbidden_mask @ P.T
            if self.batch:
                self.forbidden_mask = self.forbidden_mask | involution_mapped.transpose(0, 2, 1)
            else:
                self.forbidden_mask = self.forbidden_mask | involution_mapped.T

        R[self.forbidden_mask] = 0

        if not (R.sum(axis=-1) == np.zeros(self.S)).all():
            R = self._set_diags(R)

        return R

    def normalize_R(self, R):
        """Normalize rate matrix so the max absolute rate equals self.timescale."""
        if self.batch:
            R_max = np.abs(R).max(axis=-1).max(axis=-1)
        else:
            R_max = np.abs(R).max()

        self.scale = R_max / self.timescale

        if self.batch:
            R = (R.T / self.scale).T
        else:
            R = R / self.scale

        return R

    def _set_statewise_Q(self):
        """Precompute the state-wise environmental entropy flow rates.

        This computes the heat-like contribution to entropy production
        from the rate matrix structure (Eq. 3 in the paper), handling
        forbidden (zero-rate) transitions safely.
        """
        if self.batch:
            rev_R_T = self.rev_R.transpose(0, 2, 1)
        else:
            rev_R_T = self.rev_R.T

        # Verify zero-structure consistency between R and rev_R^T
        off_diag = ~np.eye(self.S, dtype=bool)
        R_zero = (self.R == 0) & off_diag
        rev_R_T_zero = (rev_R_T == 0) & off_diag
        assert np.all(R_zero == rev_R_T_zero), \
            "Inconsistency: wherever R is zero, rev_R_T must also be zero and vice versa."

        # Compute log(R / rev_R^T) only where both are nonzero
        log_ratio = np.zeros_like(self.R)
        nonzero_mask = (self.R != 0) & off_diag
        log_ratio[nonzero_mask] = np.log(self.R[nonzero_mask] / rev_R_T[nonzero_mask])

        R_diags = self.R * np.identity(self.S)
        rev_R_diags = self.rev_R * np.identity(self.S)

        self.statewise_Q = ((self.R * log_ratio) + R_diags - rev_R_diags).sum(axis=-1)

        if np.any(np.isnan(self.statewise_Q)):
            raise ValueError(
                'statewise_Q contains NaNs -- check rate matrix for inconsistent zero structure'
            )

    # -------------------------------------------------------------------
    # Time-reversal involution
    # -------------------------------------------------------------------

    def get_reversal_matrix(self):
        """Build the time-reversal permutation matrix from involution_indices.

        If involution_indices is not set, generates a random involution
        (product of disjoint transpositions). The involution sigma satisfies
        sigma^2 = identity, meaning each state either maps to itself (time-even)
        or swaps with exactly one partner (time-reversed pair).

        Any valid involution can be used -- set self.involution_indices to a
        numpy array where involution[s] gives the time-reversed partner of state s.
        """
        try:
            involution = self.involution_indices
        except AttributeError:
            self.involution_indices = involution_builder(self.S)
            involution = self.involution_indices
        rev_matrix = np.zeros((self.S, self.S), dtype=bool)
        rev_matrix[range(self.S), involution] = 1
        return rev_matrix

    # -------------------------------------------------------------------
    # Dynamics
    # -------------------------------------------------------------------

    def get_time_deriv(self, state, R=None):
        """Compute dp/dt = p @ R (the master equation time derivative)."""
        if R is None:
            R = self.R
        if self.batch:
            return np.einsum('ni,nik->nk', state, R)
        else:
            return np.matmul(state, R)

    def get_meps_deriv(self, state, R=None):
        """Compute the MEPS gradient flow (Eq. 9 in the paper).

        d_tau p(s) = d_t p(s) - p(s) * (sigma(R,p,s) - sigma(R,p))

        This drives the distribution toward the minimum entropy production state.
        """
        if R is None:
            R = self.R
        if self.batch:
            return self.get_time_deriv(state, R) + \
                state * (self.get_epr(state, R) - self.get_statewise_epr(state, R).T).T
        else:
            return self.get_time_deriv(state, R) + \
                state * (self.get_epr(state, R) - self.get_statewise_epr(state, R))

    def evolve_state(self, state, dt):
        """Forward-Euler step of the master equation with positivity clipping."""
        next_state = state + dt * self.get_time_deriv(state)
        if np.any(next_state < 0):
            next_state[next_state < 0] = state[next_state < 0] / 10
        return self.normalize_state(next_state)

    # -------------------------------------------------------------------
    # Thermodynamic quantities
    # -------------------------------------------------------------------

    def get_statewise_epr(self, state, R=None):
        """Compute state-wise entropy production rate sigma(R, p, s).

        This is Eq. A5 in the paper: the contribution of each state to
        the total EPR, such that sigma(R,p) = sum_s p(s) * sigma(R,p,s).
        """
        if R is None:
            R = self.R
        if self.batch:
            surprisal_rate = -np.einsum('nik,nk->ni', R, np.log(state))
        else:
            surprisal_rate = -np.matmul(R, np.log(state))
        return surprisal_rate + self.statewise_Q

    def get_epr(self, state, R=None):
        """Compute the total entropy production rate sigma(R, p).

        This is Eq. 3 in the paper.  Returns the raw computed value;
        note that float64 arithmetic noise is ~1e-16, so values at or
        below that scale should be interpreted with caution.
        """
        if R is None:
            R = self.R
        if self.batch:
            return np.einsum('nk,nk->n', state, self.get_statewise_epr(state, R))
        else:
            return np.matmul(state, self.get_statewise_epr(state, R))

    def get_statewise_activity_in(self, state, R=None):
        """Compute inflow activity for each state (off-diagonal incoming rates)."""
        if R is None:
            R = self.R
        return self.get_time_deriv(state, R=R * (1 - np.identity(self.S)))

    def get_statewise_activity_out(self, state, R=None):
        """Compute outflow activity for each state (diagonal escape rates)."""
        if R is None:
            R = self.R
        R_diag = R * np.identity(self.S)
        if self.batch:
            return -np.einsum('nik,nk->ni', R_diag, state)
        else:
            return -np.matmul(R_diag, state)

    def get_activity(self, state):
        """Compute total activity (sum of inflow over all states)."""
        return self.get_statewise_activity_in(state).sum(axis=-1)

    def get_statewise_prob_current(self, state):
        """Compute absolute probability current for each state."""
        return np.abs(self.get_time_deriv(state))

    def get_prob_current(self, state):
        """Compute total probability current (sum over states)."""
        return self.get_statewise_prob_current(state).sum(axis=-1)

    # -------------------------------------------------------------------
    # State construction
    # -------------------------------------------------------------------

    def get_uniform(self):
        """Return the uniform distribution over S states."""
        if self.batch:
            return np.ones((len(self.R), self.S)) / self.S
        else:
            return np.ones(self.S) / self.S

    def get_random_state(self):
        """Return a random state drawn from the Dirichlet distribution."""
        if self.batch:
            state = np.random.dirichlet(np.ones(self.S), self.R.shape[0])
        else:
            state = np.random.dirichlet(np.ones(self.S))
        return self.normalize_state(state)

    def get_local_state(self, mu=None, sigma=None):
        """Return a localized (Gaussian-like) state on the cyclic state space."""
        if self.batch:
            N = self.R.shape[0]
        else:
            N = 1

        state = np.mgrid[0:N, 0:self.S][1]
        if mu is None:
            mu = np.random.choice(self.S, size=N)
        if sigma is None:
            sigma = np.random.uniform(0.1, int(self.S / 3), (N))

        state = np.exp(-cyclic_distance(state, mu)**2 / (2 * sigma[:, None]**2))
        state = state / np.sum(state, axis=-1)[:, None]
        return state

    # -------------------------------------------------------------------
    # Steady state solvers
    # -------------------------------------------------------------------

    def get_meps(self, dt0=1, dt_min=1E-8, state=None, max_iter=5_000,
                 dt_iter=50, diagnostic=False, method='auto', **jax_kwargs):
        """Find the minimum entropy production state (MEPS).

        By default, uses JAX L-BFGS when JAX is available, falling back to
        forward-Euler gradient descent otherwise. The method can be selected
        explicitly via the ``method`` parameter.

        Parameters
        ----------
        dt0 : float
            Initial timestep (Euler only).
        dt_min : float
            Minimum allowed timestep (Euler only).
        state : np.ndarray or None
            Initial guess (defaults to cached meps, then ness, then uniform).
        max_iter : int
            Maximum number of iterations (Euler only; ignored when method='jax').
        dt_iter : int
            Number of iterations between timestep decay steps (Euler only).
        diagnostic : bool
            If True, return full history of EPRs, states, and convergence flags
            (Euler only; ignored when method='jax').
        method : str
            Optimization method. Options:

            - ``'auto'`` (default): Use JAX L-BFGS if available, else Euler.
            - ``'euler'``: Forward-Euler gradient descent (numpy only).
            - ``'jax'``: JAX L-BFGS softmax optimization (requires JAX).
        **jax_kwargs
            Additional keyword arguments forwarded to :meth:`get_meps_jax`
            when ``method='jax'``. Common options: ``tol`` (float),
            ``num_restarts`` (int), ``max_iter`` (int, default 500).

        Returns
        -------
        np.ndarray or tuple
            The MEPS distribution (or diagnostic tuple if diagnostic=True).
        """
        # --- Method dispatch ---
        if method == 'auto':
            method = 'jax' if HAS_JAX else 'euler'

        if method == 'jax':
            if not HAS_JAX:
                raise ImportError(
                    "JAX is required for method='jax'. "
                    "Install with: pip install 'jax[cpu]' jaxopt"
                )
            jax_kw = dict(
                state=state,
                max_iter=jax_kwargs.pop('max_iter', 500),
                tol=jax_kwargs.pop('tol', 1e-10),
                num_restarts=jax_kwargs.pop('num_restarts', 1),
            )
            return self.get_meps_jax(**jax_kw)

        if method != 'euler':
            raise ValueError(
                f"Unknown MEPS method '{method}'. Use 'auto', 'euler', or 'jax'."
            )
        if state is None:
            if hasattr(self, 'meps'):
                state = self.meps
            elif hasattr(self, 'ness'):
                state = self.ness
            else:
                state = self.get_uniform()

        eprs = [self.get_epr(state)]
        states = [state]
        doneBool = np.array(False)
        doneEprBool = np.array(False)
        negativeBool = False
        nanStateBool = False
        i = 0
        if self.batch:
            j = -1.0 * np.ones(self.R.shape[0])
        else:
            j = np.array(-1.0)

        while (not np.all(doneBool) or np.any(negativeBool) or np.any(nanStateBool)) and i < max_iter:
            if i % dt_iter == 0:
                j += 1
            dt = np.maximum(dt0 * self.DT_DECAY_RATE**j, dt_min)

            if self.batch:
                new_state = state + dt[:, None] * self.get_meps_deriv(state)
            else:
                new_state = state + dt * self.get_meps_deriv(state)

            negativeBool = np.any(new_state < 0, axis=-1)
            nanStateBool = np.any(np.isnan(new_state), axis=-1)

            if np.any(negativeBool):
                if diagnostic:
                    print(f'rewinding {np.sum(negativeBool)} states to avoid negative values at iter {i}')
                new_state[negativeBool] = state[negativeBool]
                j[negativeBool] += 1

            if np.any(nanStateBool):
                if diagnostic:
                    print(f'rewinding {np.sum(nanStateBool)} states to avoid NaN at iter {i}')
                new_state[nanStateBool] = state[nanStateBool]
                j[nanStateBool] += 1

            new_state = self.normalize_state(new_state)

            epr = np.asarray(self.get_epr(state))

            eprIncreaseBool = (epr - eprs[-1]) / eprs[-1] > self.EPR_INCREASE_TOL
            if np.any(eprIncreaseBool):
                if diagnostic:
                    print(f'partially rewinding {np.sum(eprIncreaseBool)} states for EPR increase at iter {i}')
                alpha = np.random.randint(50, 70) / 100
                new_state[eprIncreaseBool] = (
                    alpha * state[eprIncreaseBool] + (1 - alpha) * new_state[eprIncreaseBool]
                )
                epr = np.asarray(self.get_epr(state))
                j[eprIncreaseBool] += 1

            eprs.append(epr)
            states.append(new_state)
            state = new_state

            if not diagnostic:
                eprs = eprs[-3:]
                states = states[-3:]

            if not (np.any(negativeBool) or np.any(nanStateBool)):
                meps_deriv = self.get_meps_deriv(state)
                doneBool = np.all(
                    np.isclose(0, np.abs(meps_deriv) / np.maximum(self.CONVERGENCE_FLOOR, state),
                               atol=self.CONVERGENCE_ATOL),
                    axis=-1
                )
                doneEprBool = np.isclose(eprs[-1], eprs[-2], rtol=1E-3 * dt, atol=1E-14)
                # EPR at machine-precision noise floor — no point iterating further
                doneBool = doneBool | (np.abs(epr) < self.EPR_ATOL)

            i += 1

        if i == max_iter:
            print(f'MEPS did not converge after {i} iterations in {np.sum(~doneBool)} machines')
            print(f'MEPS EPR did not converge after {i} iterations in {np.sum(~doneEprBool)} machines')

        self.meps = states[-1]
        if not diagnostic:
            return self.meps
        else:
            return np.array(eprs), np.array(states), doneBool, doneEprBool

    def get_meps_jax(self, state=None, max_iter=500, tol=1e-10, num_restarts=1):
        """Find the minimum entropy production state (MEPS) via JAX L-BFGS.

        Uses softmax parameterization (p = softmax(theta)) to convert the
        constrained optimization on the probability simplex into unconstrained
        optimization over R^S. This avoids the boundary-trapping issues that
        plague the forward Euler method in get_meps().

        Requires JAX to be installed. Falls back to jaxopt.LBFGS when available
        (supports vmap for batch mode), otherwise uses jax.scipy.optimize.

        Parameters
        ----------
        state : np.ndarray or None
            Initial guess (defaults to cached meps, then ness, then uniform).
        max_iter : int
            Maximum number of L-BFGS iterations per restart.
        tol : float
            Gradient tolerance for convergence.
        num_restarts : int
            Number of random restarts. The best solution (lowest EPR) is kept.
            The first restart always uses the provided/default initial state.

        Returns
        -------
        np.ndarray
            The MEPS distribution. Also cached as self.meps.

        Raises
        ------
        ImportError
            If JAX is not installed.
        """
        if not HAS_JAX:
            raise ImportError(
                "JAX is required for get_meps_jax(). "
                "Install with: pip install 'jax[cpu]' jaxopt"
            )

        # Determine initial state (same fallback chain as get_meps)
        if state is None:
            if hasattr(self, 'meps'):
                state = self.meps.copy()
            elif hasattr(self, 'ness'):
                state = self.ness.copy()
            else:
                state = self.get_uniform()

        ms = self.min_state

        if self.batch:
            # --- Batch mode ---
            N = self.R.shape[0]
            best_states = np.empty_like(state)
            best_eprs = np.full(N, np.inf)

            R_jax = jnp.array(self.R)
            Q_jax = jnp.array(self.statewise_Q)

            def _theta_from_state(init_state):
                """Convert probability distributions to centered log-space."""
                p_init = np.maximum(init_state, ms)
                p_init = p_init / p_init.sum(axis=-1, keepdims=True)
                theta0 = jnp.array(np.log(p_init))
                return theta0 - theta0.mean(axis=-1, keepdims=True)

            def _theta_to_probs(theta_batch):
                """Convert optimized theta back to probability distributions."""
                log_p = theta_batch - jax_logsumexp(
                    theta_batch, axis=-1, keepdims=True)
                p_opt = np.array(jnp.exp(log_p))
                p_opt = np.maximum(p_opt, ms)
                p_opt /= p_opt.sum(axis=-1, keepdims=True)
                return p_opt

            for restart in range(num_restarts):
                if restart == 0:
                    init_state = state
                else:
                    init_state = self.get_random_state()

                theta0_batch = _theta_from_state(init_state)

                if HAS_JAXOPT:
                    # Vmapped L-BFGS: solve all N systems in parallel
                    theta_opt = _jax_solve_batch_lbfgs(
                        theta0_batch, R_jax, Q_jax, max_iter, tol)
                else:
                    # Fallback: sequential scipy solver
                    theta_opt = jnp.stack([
                        _jax_solve_single_scipy(
                            theta0_batch[n], R_jax[n], Q_jax[n],
                            max_iter, tol)
                        for n in range(N)
                    ])

                p_opt = _theta_to_probs(theta_opt)
                eprs = self.get_epr(p_opt)

                improved = eprs < best_eprs
                best_states[improved] = p_opt[improved]
                best_eprs[improved] = eprs[improved]

            self.meps = best_states
            return self.meps

        else:
            # --- Single system mode ---
            R_jax = jnp.array(self.R)
            Q_jax = jnp.array(self.statewise_Q)

            best_p = None
            best_epr = np.inf

            for restart in range(num_restarts):
                # Already at machine-precision noise — no point trying more restarts
                if best_epr < self.EPR_ATOL:
                    break

                if restart == 0:
                    init_state = state
                else:
                    init_state = self.get_random_state()

                # Initialize theta = log(p) centered
                p_init = np.maximum(init_state, ms)
                p_init = p_init / p_init.sum()
                theta0 = jnp.array(np.log(p_init))
                theta0 = theta0 - theta0.mean()

                # Optimize
                if HAS_JAXOPT:
                    theta_opt = _jax_solve_single_lbfgs(
                        theta0, R_jax, Q_jax, max_iter, tol)
                else:
                    theta_opt = _jax_solve_single_scipy(
                        theta0, R_jax, Q_jax, max_iter, tol)

                # Convert back to probability
                log_p = theta_opt - jax_logsumexp(theta_opt)
                p_opt = np.array(jnp.exp(log_p))
                p_opt = np.maximum(p_opt, ms)
                p_opt /= p_opt.sum()

                epr_opt = float(self.get_epr(p_opt))
                if epr_opt < best_epr:
                    best_epr = epr_opt
                    best_p = p_opt

            self.meps = best_p
            return self.meps

    def _ness_estimate(self):
        """Estimate NESS from the eigenvector with smallest eigenvalue of R^T.

        This is a fallback when the analytic method fails (e.g., singular matrix).
        """
        if self.batch:
            vals, vects = np.linalg.eig(self.R.transpose(0, 2, 1))
        else:
            vals, vects = np.linalg.eig(self.R.T)

        vects = np.abs(vects)
        vals = np.abs(vals)
        min_ix = np.argmin(vals, axis=-1)

        if self.batch:
            ness = vects[range(len(vects)), :, min_ix]
        else:
            ness = vects[:, min_ix].ravel()

        return self.normalize_state(ness)

    def ness_analytic(self, n=0):
        """Compute NESS analytically by solving R^T pi = 0 with normalization.

        Replaces column n of R^T with ones (normalization constraint) and
        inverts. This is the preferred method for small-to-moderate systems.
        """
        M = self.R.copy()
        M[..., n] = 1
        ness = np.linalg.inv(M)[..., n, :]
        return self.normalize_state(ness)

    def get_ness(self, dt0=1, dt_iter=50, dt_min=1E-8, max_iter=5_000,
                 force_analytic=False, diagnostic=False):
        """Find the nonequilibrium steady state (NESS).

        Strategy:
        1. Try analytic inversion (for small enough systems or if forced)
        2. Fall back to eigenvalue method
        3. Fall back to numeric integration of the master equation

        Parameters
        ----------
        dt0 : float
            Initial timestep for numeric integration.
        dt_iter : int
            Iterations between timestep decay steps.
        dt_min : float
            Minimum allowed timestep.
        max_iter : int
            Maximum number of iterations for numeric integration.
        force_analytic : bool
            If True, always try the analytic method first regardless of system size.
        diagnostic : bool
            If True, return full integration history.

        Returns
        -------
        np.ndarray or tuple
            The NESS distribution (or diagnostic tuple if diagnostic=True).
        """
        ness_list = []
        if hasattr(self, 'ness'):
            ness = self.ness
        else:
            batch_size = self.R.shape[0] if self.batch else 1
            use_analytic = (
                (1 + np.log(batch_size)) * np.sqrt(self.S) < self.analytic_threshhold
                or force_analytic
            )
            if use_analytic:
                try:
                    ness = self.ness_analytic()
                except (np.linalg.LinAlgError, ValueError):
                    if self.verbose:
                        print('analytic NESS failed, trying eigenvalue method')
                    try:
                        ness = self._ness_estimate()
                    except (np.linalg.LinAlgError, ValueError):
                        if self.verbose:
                            print('eigenvalue method failed, defaulting to numeric solution')
                        ness = self.meps if hasattr(self, 'meps') else self.get_uniform()
            else:
                if self.verbose:
                    print('system too large for analytic, defaulting to numeric solution')
                ness = self.meps if hasattr(self, 'meps') else self.get_uniform()

        dt = dt0
        i = 0
        negativeBool = False
        doneBool = False

        if self.batch:
            j = -1.0 * np.ones(self.R.shape[0])
        else:
            j = np.array(-1.0)

        while ((not np.all(doneBool)) or np.any(negativeBool)) and i < max_iter:
            if i % dt_iter == 0:
                j += 1
            dt = np.maximum(dt0 * self.DT_DECAY_RATE**j, dt_min)

            ness_list.append(ness)
            if self.batch:
                ness = self.evolve_state(ness, dt[:, None])
            else:
                ness = self.evolve_state(ness, dt)

            negativeBool = np.any(ness < 0, axis=-1)
            if np.any(negativeBool):
                if diagnostic:
                    print(f'rewinding {np.sum(negativeBool)} states to avoid negative values at iter {i}')
                ness[negativeBool] = ness_list[-1][negativeBool]
                j[negativeBool] += 1

            doneBool = np.all(
                np.isclose(0, np.abs(self.get_time_deriv(ness)) / np.maximum(self.CONVERGENCE_FLOOR, ness),
                           atol=self.CONVERGENCE_ATOL),
                axis=-1
            )

            i += 1

            if not diagnostic:
                ness_list = ness_list[-1:]

        if i == max_iter:
            print(f'NESS did not converge after {i} iterations in {np.sum(~doneBool)} machines')

        self.ness = ness

        if diagnostic:
            return ness, ness_list, doneBool
        else:
            return self.ness

    # -------------------------------------------------------------------
    # Validation and utilities
    # -------------------------------------------------------------------

    def validate_state(self, state):
        """Check that state has the correct shape for this CTMC."""
        shape = state.shape
        assert shape[-1] == self.S, f'state dimension expected {self.S}, got {shape[-1]}'

        if self.batch:
            assert len(shape) == 2, f'expected 2D state array, got shape {shape}'
            assert shape[0] == self.R.shape[0], \
                f'got {shape[0]} state vectors for {self.R.shape[0]} machines'

        return True

    def normalize_state(self, state):
        """Normalize a probability distribution, clamping non-positive values."""
        assert self.validate_state(state), 'found invalid state shape'

        if np.any(state <= 0):
            if self.verbose:
                print('found negative or zero state, setting to min state prob')
            state[state <= 0] = self.min_state

        if np.all(state.sum(axis=-1) == 1):
            return state
        else:
            if self.batch:
                return (state.T / np.sum(state, axis=-1)).T
            else:
                return state / state.sum()

    def dkl(self, p, q):
        """Compute the Kullback-Leibler divergence D_KL(p || q).

        Both distributions must have the same support (nonzero entries).
        """
        assert (np.array([p, q]) >= 0).all(), 'all probabilities must be non-negative'
        p_dom, q_dom = p != 0, q != 0
        assert np.all(np.equal(p_dom, q_dom)), 'p and q must have the same support'

        p, q = (p.T / np.sum(p, axis=-1)).T, (q.T / np.sum(q, axis=-1)).T
        p[np.where(~p_dom)] = 1
        q[np.where(~q_dom)] = 1
        return np.sum(p * np.log(p / q), axis=-1)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def cyclic_distance(x, i):
    """Compute the signed distance on a cyclic (ring) state space."""
    L = x.shape[-1]
    try:
        x_new = x - i
    except (ValueError, TypeError):
        x_new = (x.T - i).T

    wrap = np.where(abs(x_new) > int(L / 2))
    x_new[wrap] = -np.sign(x_new[wrap]) * (L - abs(x_new[wrap]))
    return x_new


def involution_builder(N_states, N_swaps=None):
    """Build a random involution (permutation that is its own inverse).

    An involution is a product of disjoint transpositions: some states
    swap pairwise, and the rest are fixed points. This is the correct
    structure for time-reversal symmetry in CTMCs.

    Parameters
    ----------
    N_states : int
        Total number of states.
    N_swaps : int or None
        Number of pairwise swaps (default: floor(N_states/2), maximum swaps).

    Returns
    -------
    np.ndarray
        Array of length N_states where involution[s] is the time-reversed
        partner of state s.
    """
    if N_swaps is None or N_swaps > int(N_states / 2):
        N_swaps = int(N_states / 2)

    swaps = np.array(range(N_states))
    np.random.shuffle(swaps)
    swap_list = [[swaps[2 * i], swaps[2 * i + 1]] for i in range(N_swaps)]
    involution = np.array(range(N_states))
    for i1, i2 in swap_list:
        involution[i1] = i2
        involution[i2] = i1
    return involution


# ---------------------------------------------------------------------------
# Graph connectivity and sparsification
# ---------------------------------------------------------------------------

def _bfs_reachable(adj, start):
    """Return the set of nodes reachable from `start` via BFS on adjacency matrix.

    Parameters
    ----------
    adj : np.ndarray of shape (S, S)
        Boolean or numeric adjacency matrix. Nonzero off-diagonal entries
        are treated as edges.
    start : int
        Starting node index.

    Returns
    -------
    set of int
        All node indices reachable from start (including start itself).
    """
    S = adj.shape[0]
    visited = set()
    queue = [start]
    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        for neighbor in range(S):
            if neighbor != node and adj[node, neighbor] != 0 and neighbor not in visited:
                queue.append(neighbor)
    return visited


def is_irreducible(R):
    """Check whether a rate matrix defines a strongly connected (irreducible) CTMC.

    A CTMC is irreducible iff every state can reach every other state via
    some sequence of transitions with nonzero rates. This is required for
    the existence of a unique steady-state distribution.

    Uses BFS on both R and R^T: the graph is strongly connected iff every
    node is reachable from node 0 in both the forward and reverse graphs.

    Parameters
    ----------
    R : np.ndarray of shape (S, S) or (N, S, S)
        Rate matrix or batch of rate matrices.

    Returns
    -------
    bool or np.ndarray of bool
        True if the corresponding rate matrix is irreducible.
        For batch input, returns an array of shape (N,).
    """
    R = np.asarray(R)
    single = (R.ndim == 2)
    if single:
        R = R[np.newaxis, ...]

    N, S, _ = R.shape
    results = np.empty(N, dtype=bool)

    for n in range(N):
        Rn = R[n]
        # Forward reachability from state 0
        fwd = _bfs_reachable(Rn, 0)
        if len(fwd) < S:
            results[n] = False
            continue
        # Reverse reachability (can state 0 be reached from all others?)
        rev = _bfs_reachable(Rn.T, 0)
        results[n] = (len(rev) == S)

    return results[0] if single else results


def _strongly_connected_components(adj):
    """Find strongly connected components using Tarjan's algorithm.

    Parameters
    ----------
    adj : np.ndarray of shape (S, S)
        Adjacency matrix. Nonzero off-diagonal entries are edges.

    Returns
    -------
    list of list of int
        Each inner list is one SCC (list of node indices).
    """
    S = adj.shape[0]
    index_counter = [0]
    stack = []
    on_stack = np.zeros(S, dtype=bool)
    indices = -np.ones(S, dtype=int)
    lowlinks = -np.ones(S, dtype=int)
    sccs = []

    def strongconnect(v):
        indices[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack[v] = True

        for w in range(S):
            if w == v or adj[v, w] == 0:
                continue
            if indices[w] == -1:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif on_stack[w]:
                lowlinks[v] = min(lowlinks[v], indices[w])

        if lowlinks[v] == indices[v]:
            component = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                component.append(w)
                if w == v:
                    break
            sccs.append(component)

    for v in range(S):
        if indices[v] == -1:
            strongconnect(v)

    return sccs


def _repair_connectivity(R_single, rng):
    """Add minimal edges to make a single rate matrix strongly connected.

    For each pair of SCCs that need linking, restores a random original-weight
    edge from the dense original. If the original weight was zero, assigns
    the median nonzero off-diagonal rate as a reasonable default.

    Parameters
    ----------
    R_single : np.ndarray of shape (S, S)
        A single (possibly disconnected) rate matrix. Modified in-place.
    rng : np.random.Generator
        Random number generator for reproducible edge selection.

    Returns
    -------
    int
        Number of edges added.
    """
    S = R_single.shape[0]
    sccs = _strongly_connected_components(R_single)

    if len(sccs) <= 1:
        return 0

    # Build a condensation: assign each node to its SCC index
    node_to_scc = np.empty(S, dtype=int)
    for idx, comp in enumerate(sccs):
        for node in comp:
            node_to_scc[node] = idx

    n_sccs = len(sccs)
    edges_added = 0

    # Determine a fallback rate for edges that were originally zero
    nonzero_off_diag = R_single[~np.eye(S, dtype=bool) & (R_single > 0)]
    fallback_rate = np.median(nonzero_off_diag) if len(nonzero_off_diag) > 0 else 1.0

    # Connect SCCs in a cycle: scc_0 -> scc_1 -> ... -> scc_{k-1} -> scc_0
    for i in range(n_sccs):
        j = (i + 1) % n_sccs
        src_nodes = sccs[i]
        dst_nodes = sccs[j]

        # Pick a random (src, dst) pair between these two components
        src = rng.choice(src_nodes)
        dst = rng.choice(dst_nodes)

        # Add both directions so verify_rate_matrix's symmetric forbidden-
        # transition logic doesn't zero out the repair edge.
        if R_single[src, dst] == 0:
            R_single[src, dst] = fallback_rate
            edges_added += 1
        if R_single[dst, src] == 0:
            R_single[dst, src] = fallback_rate
            edges_added += 1

    return edges_added


def sparsify(R, p=None, avg_degree=None, ensure_connected=True, seed=None):
    """Sparsify a dense rate matrix using Erdos-Renyi edge removal.

    Takes a fully-connected rate matrix and randomly removes off-diagonal
    transitions, keeping each edge independently with probability p. This
    produces a sparse CTMC that can then be passed to ContinuousTimeMarkovChain.

    Parameters
    ----------
    R : np.ndarray of shape (S, S) or (N, S, S)
        Dense rate matrix or batch of rate matrices. Off-diagonal entries
        are the transition rates; diagonal entries are ignored (they will
        be recomputed by ContinuousTimeMarkovChain.verify_rate_matrix).
    p : float or None
        Per-edge keep probability (standard Erdos-Renyi G(n,p)).
        Must be in (0, 1]. Exactly one of p or avg_degree must be specified.
    avg_degree : float or None
        Target average outgoing degree per state. Internally computes
        p = avg_degree / (S - 1). Takes precedence over p if both given.
    ensure_connected : bool
        If True (default), checks strong connectivity after sparsification
        and repairs disconnected graphs by adding back minimal edges
        between strongly connected components.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Sparsified rate matrix, same shape as input. Diagonal entries are
        preserved from the original (they will be reset by
        ContinuousTimeMarkovChain anyway).

    Examples
    --------
    >>> R_dense = arrhenius_pump_generator(S=50, N=10, n_pumps=100)
    >>> R_sparse = sparsify(R_dense, avg_degree=5)
    >>> machine = ContinuousTimeMarkovChain(R=R_sparse)
    >>> ness = machine.get_ness()

    >>> # Equivalently, using edge probability directly:
    >>> R_sparse = sparsify(R_dense, p=0.1, seed=42)
    """
    R = np.asarray(R, dtype=float)
    single = (R.ndim == 2)
    if single:
        R = R[np.newaxis, ...].copy()
    else:
        R = R.copy()

    N, S, _ = R.shape
    assert S >= 2, f'need at least 2 states, got {S}'

    # --- resolve parameterization ---
    if avg_degree is not None:
        if p is not None:
            import warnings
            warnings.warn('both p and avg_degree given; using avg_degree')
        assert 0 < avg_degree <= S - 1, \
            f'avg_degree must be in (0, {S-1}], got {avg_degree}'
        p = avg_degree / (S - 1)
    else:
        assert p is not None, 'must specify either p or avg_degree'

    assert 0 < p <= 1, f'p must be in (0, 1], got {p}'

    # --- generate masks ---
    rng = np.random.default_rng(seed)
    off_diag = ~np.eye(S, dtype=bool)

    for n in range(N):
        # Bernoulli mask: keep each undirected pair with probability p.
        # Symmetric so that verify_rate_matrix's forbidden-transition logic
        # (which zeros both directions if either is zero) doesn't kill extra edges.
        upper = rng.random((S, S)) < p
        mask = np.triu(upper, k=1)
        mask = mask | mask.T          # symmetric: same decision for i->j and j->i
        mask |= np.eye(S, dtype=bool) # always keep diagonal
        R[n] *= mask

    # --- connectivity repair ---
    if ensure_connected:
        for n in range(N):
            if not is_irreducible(R[n]):
                edges_added = _repair_connectivity(R[n], rng)
                if edges_added > 0:
                    # Verify repair worked
                    assert is_irreducible(R[n]), \
                        f'connectivity repair failed for matrix {n}'

    if single:
        return R[0]
    return R
