"""
Comprehensive test suite for the refactored CTMC framework.

Tests:
1. Basic functionality: CTMC creation, NESS, MEPS, EPR calculations
2. Time symmetry: time_even_states=True/False, involutions, forbidden transitions
3. Physics checks: EPR relationships, detailed balance, NESS stability
4. Batch mode: verify batch operations match single-system results
5. Edge cases: forbidden transitions, very small rates
"""

import os, sys
sys.path.insert(0, os.path.expanduser('~/source/discrete_states'))

import numpy as np
from ctmc import (
    ContinuousTimeMarkovChain,
    uniform_generator,
    cyclic_generator,
    detailed_balance_generator,
    involution_builder,
)


def assert_close(a, b, rtol=1e-5, atol=1e-8, msg=""):
    """Assert that a and b are close within tolerances."""
    if np.isscalar(a) and np.isscalar(b):
        assert np.isclose(a, b, rtol=rtol, atol=atol), \
            f"{msg} {a} not close to {b} (rtol={rtol}, atol={atol})"
    else:
        assert np.allclose(a, b, rtol=rtol, atol=atol), \
            f"{msg} arrays not close (rtol={rtol}, atol={atol})\nmax diff: {np.max(np.abs(a - b))}"


def test_basic_ctmc_creation():
    """Test basic CTMC creation with various generators."""
    print("\n=== Test: Basic CTMC Creation ===")
    
    # Test with uniform generator (single)
    ctmc_single = ContinuousTimeMarkovChain(generator=uniform_generator, S=5, N=1)
    assert ctmc_single.S == 5
    assert not ctmc_single.batch
    assert ctmc_single.R.shape == (5, 5)
    print("✓ Single system with uniform generator")
    
    # Test with batch
    ctmc_batch = ContinuousTimeMarkovChain(generator=uniform_generator, S=5, N=3)
    assert ctmc_batch.S == 5
    assert ctmc_batch.batch
    assert ctmc_batch.R.shape == (3, 5, 5)
    print("✓ Batch mode (N=3) with uniform generator")
    
    # Test with cyclic generator
    ctmc_cyclic = ContinuousTimeMarkovChain(generator=cyclic_generator, S=6, N=1)
    assert ctmc_cyclic.S == 6
    print("✓ Cyclic generator")
    
    # Test with explicit rate matrix
    R_explicit = np.random.uniform(0, 1, (4, 4))
    ctmc_explicit = ContinuousTimeMarkovChain(R=R_explicit)
    assert ctmc_explicit.S == 4
    print("✓ Explicit rate matrix")


def test_rate_matrix_validation():
    """Test rate matrix validation and normalization."""
    print("\n=== Test: Rate Matrix Validation ===")
    
    # Test that rows sum to zero
    ctmc = ContinuousTimeMarkovChain(generator=uniform_generator, S=5, N=1)
    row_sums = ctmc.R.sum(axis=-1)
    assert_close(row_sums, np.zeros_like(row_sums), atol=1e-10, 
                 msg="Rate matrix rows should sum to zero")
    print("✓ Rate matrix rows sum to zero")
    
    # Test that off-diagonal elements are non-negative
    off_diag = ctmc.R - ctmc.R * np.eye(5)
    assert np.all(off_diag >= -1e-10), "Off-diagonal elements must be non-negative"
    print("✓ Off-diagonal elements are non-negative")


def test_time_even_states():
    """Test time_even_states=True configuration."""
    print("\n=== Test: Time-Even States (time_even_states=True) ===")
    
    ctmc = ContinuousTimeMarkovChain(generator=uniform_generator, S=5, N=1)
    # Time-even states is the default
    assert ctmc.time_even_states == True
    
    # For time-even states, rev_R should equal R
    assert_close(ctmc.rev_R, ctmc.R, atol=1e-10,
                 msg="For time_even_states=True, rev_R should equal R")
    print("✓ rev_R == R for time_even_states=True")
    
    # When time_even_states=True, we don't generate random involutions unnecessarily
    # but get_reversal_matrix will generate identity by default
    ctmc.time_even_states = True
    ctmc.set_rate_matrix(ctmc.R)
    assert ctmc.time_even_states == True
    print("✓ time_even_states=True configuration works")


def test_custom_involution():
    """Test time_even_states=False with custom involution."""
    print("\n=== Test: Custom Involution (time_even_states=False) ===")
    
    S = 6
    ctmc = ContinuousTimeMarkovChain(generator=uniform_generator, S=S, N=1)
    ctmc.time_even_states = False
    
    # Set a specific involution: (0,1)(2,3)(4,5) - three 2-cycles
    custom_involution = np.array([1, 0, 3, 2, 5, 4])
    ctmc.involution_indices = custom_involution
    ctmc.set_rate_matrix(ctmc.R)
    
    # Involution squared should be identity
    P = ctmc.get_reversal_matrix()
    P_squared = P @ P
    assert_close(P_squared, np.eye(S), atol=1e-10,
                 msg="P^2 should be identity")
    print("✓ P^2 = identity (involution property)")
    
    # rev_R should be P @ R @ P^T (but not equal to R in general)
    expected_rev_R = P @ ctmc.R @ P.T
    assert_close(ctmc.rev_R, expected_rev_R, atol=1e-10,
                 msg="rev_R = P @ R @ P^T")
    print("✓ rev_R = P @ R @ P^T for custom involution")
    
    # Check the involution is correctly applied
    involution_from_P = np.argmax(P, axis=1)
    assert np.allclose(involution_from_P, custom_involution), \
        "Involution from P should match custom_involution"
    print("✓ Custom involution correctly applied")


def test_arbitrary_involution():
    """Test that arbitrary involutions (not just nearest-neighbor) are supported."""
    print("\n=== Test: Arbitrary Involutions ===")
    
    S = 8
    ctmc = ContinuousTimeMarkovChain(generator=uniform_generator, S=S, N=1)
    ctmc.time_even_states = False
    
    # Test several different involutions
    involutions = [
        np.array([1, 0, 2, 3, 4, 5, 6, 7]),  # Just swap first two
        np.array([7, 6, 5, 4, 3, 2, 1, 0]),  # Reverse (palindrome)
        np.array([3, 2, 1, 0, 7, 6, 5, 4]),  # Multiple swaps
        np.array([4, 5, 6, 7, 0, 1, 2, 3]),  # Cyclic shift by 4
    ]
    
    for inv in involutions:
        ctmc.involution_indices = inv
        ctmc.set_rate_matrix(ctmc.R)
        
        # Check it's a valid involution (σ² = identity)
        P = ctmc.get_reversal_matrix()
        P_squared = P @ P
        assert_close(P_squared, np.eye(S), atol=1e-10)
        
        # Check that all involutions map each state correctly
        reconstructed_inv = np.argmax(P, axis=1)
        assert np.array_equal(reconstructed_inv, inv)
    
    print(f"✓ All {len(involutions)} arbitrary involutions work correctly")


def test_forbidden_transitions_time_even():
    """Test forbidden transitions with time_even_states=True."""
    print("\n=== Test: Forbidden Transitions (time_even_states=True) ===")
    
    S = 5
    R = np.random.uniform(0.01, 1, (S, S))
    
    # Make some transitions explicitly forbidden (very small rates)
    R[0, 1] = 1e-13
    R[1, 0] = 1e-13
    R[2, 3] = 1e-13
    R[3, 2] = 1e-13
    
    ctmc = ContinuousTimeMarkovChain(R=R)
    ctmc.time_even_states = True
    ctmc.set_rate_matrix(R)
    
    # Check that forbidden transitions are symmetric
    mask = ctmc.forbidden_mask
    # For time-even, if [i,j] is forbidden, [j,i] should be too
    for i in range(S):
        for j in range(i+1, S):
            if mask[i, j]:
                assert mask[j, i], f"Forbidden mask not symmetric: [{i},{j}] but not [{j},{i}]"
    
    print("✓ Forbidden transitions are symmetric for time_even_states=True")


def test_forbidden_transitions_involution():
    """Test forbidden transitions with involution symmetry."""
    print("\n=== Test: Forbidden Transitions (with Involution) ===")
    
    S = 6
    R = np.random.uniform(0.01, 1, (S, S))
    
    # Make some transitions forbidden
    R[0, 1] = 1e-13
    R[2, 3] = 1e-13
    R[4, 5] = 1e-13
    
    ctmc = ContinuousTimeMarkovChain(R=R)
    ctmc.time_even_states = False
    
    # Custom involution that swaps (0,1), (2,3), (4,5)
    custom_involution = np.array([1, 0, 3, 2, 5, 4])
    ctmc.involution_indices = custom_involution
    ctmc.set_rate_matrix(R)
    
    # Check forbidden transitions respect involution symmetry
    P = ctmc.get_reversal_matrix()
    mask = ctmc.forbidden_mask
    
    # If s->s' is forbidden, then inv(s')->inv(s) should be too
    for i in range(S):
        for j in range(S):
            if mask[i, j] and i != j:
                inv_i = custom_involution[i]
                inv_j = custom_involution[j]
                assert mask[inv_j, inv_i], \
                    f"Forbidden [{i},{j}] but involution map [{inv_j},{inv_i}] not forbidden"
    
    print("✓ Forbidden transitions respect involution symmetry")


def test_ness_computation():
    """Test NESS computation and stability."""
    print("\n=== Test: NESS Computation ===")
    
    ctmc = ContinuousTimeMarkovChain(generator=uniform_generator, S=5, N=1)
    ness = ctmc.get_ness(diagnostic=False)
    
    # NESS should be a valid probability distribution
    assert_close(ness.sum(), 1.0, atol=1e-8, msg="NESS should sum to 1")
    assert np.all(ness >= 0), "NESS should be non-negative"
    print("✓ NESS is a valid probability distribution")
    
    # NESS should satisfy R^T @ pi = 0
    residual = ctmc.R.T @ ness
    assert_close(residual, np.zeros_like(residual), atol=1e-6,
                 msg="R^T @ ness should be approximately zero")
    print("✓ NESS satisfies R^T @ π = 0")
    
    # Time derivative at NESS should be ~0
    time_deriv = ctmc.get_time_deriv(ness)
    assert_close(time_deriv, np.zeros_like(time_deriv), atol=1e-6,
                 msg="dp/dt at NESS should be ~0")
    print("✓ dp/dt at NESS is ~0")


def test_meps_computation():
    """Test MEPS computation."""
    print("\n=== Test: MEPS Computation ===")
    
    ctmc = ContinuousTimeMarkovChain(generator=uniform_generator, S=5, N=1)
    meps = ctmc.get_meps(diagnostic=False)
    
    # MEPS should be a valid probability distribution
    assert_close(meps.sum(), 1.0, atol=1e-8, msg="MEPS should sum to 1")
    assert np.all(meps >= 0), "MEPS should be non-negative"
    print("✓ MEPS is a valid probability distribution")


def test_epr_calculations():
    """Test EPR calculations and basic properties."""
    print("\n=== Test: EPR Calculations ===")
    
    ctmc = ContinuousTimeMarkovChain(generator=uniform_generator, S=5, N=1)
    
    # Test with uniform state
    uniform_state = ctmc.get_uniform()
    epr_uniform = ctmc.get_epr(uniform_state)
    assert epr_uniform >= 0, "EPR should be non-negative"
    print(f"✓ EPR at uniform state is non-negative (EPR={epr_uniform:.6f})")
    
    # Test with random state
    random_state = ctmc.get_random_state()
    epr_random = ctmc.get_epr(random_state)
    assert epr_random >= 0, "EPR should be non-negative"
    print(f"✓ EPR at random state is non-negative (EPR={epr_random:.6f})")
    
    # Test statewise EPR
    # Note: Statewise EPR can be negative for individual states
    # because EPR is a net quantity measuring heat dissipation
    statewise_epr = ctmc.get_statewise_epr(uniform_state)
    total_epr = np.dot(uniform_state, statewise_epr)
    assert_close(total_epr, epr_uniform, atol=1e-8,
                 msg="Total EPR should equal sum of statewise EPR")
    print("✓ Total EPR = sum of statewise EPR (statewise can be negative per state)")


def test_epr_meps_less_than_ness():
    """Test that EPR at MEPS <= EPR at NESS."""
    print("\n=== Test: EPR(MEPS) <= EPR(NESS) ===")
    
    ctmc = ContinuousTimeMarkovChain(generator=cyclic_generator, S=6, N=1)
    ness = ctmc.get_ness(diagnostic=False)
    meps = ctmc.get_meps(state=ness, diagnostic=False)
    
    epr_ness = ctmc.get_epr(ness)
    epr_meps = ctmc.get_epr(meps)
    
    # MEPS should minimize EPR
    assert epr_meps <= epr_ness + 1e-6, \
        f"EPR(MEPS)={epr_meps:.6f} should be <= EPR(NESS)={epr_ness:.6f}"
    print(f"✓ EPR(MEPS)={epr_meps:.6f} <= EPR(NESS)={epr_ness:.6f}")


def test_detailed_balance_system():
    """Test that detailed balance systems have ~0 EPR."""
    print("\n=== Test: Detailed Balance Systems ===")
    
    # Create a detailed balance system
    ctmc = ContinuousTimeMarkovChain(
        generator=detailed_balance_generator,
        S=5,
        N=1,
        beta=2.0
    )
    
    ness = ctmc.get_ness(diagnostic=False)
    epr_ness = ctmc.get_epr(ness)
    
    # At detailed balance (equilibrium), EPR should be ~0
    assert epr_ness < 1e-6, \
        f"EPR at detailed balance should be ~0, got {epr_ness:.2e}"
    print(f"✓ Detailed balance system has EPR ≈ 0 (EPR={epr_ness:.2e})")


def test_batch_vs_single():
    """Test that batch mode relationships hold (distributions, not absolute values)."""
    print("\n=== Test: Batch vs Single (Distribution Relationships) ===")
    
    np.random.seed(42)
    
    # Create batch system
    ctmc_batch = ContinuousTimeMarkovChain(generator=uniform_generator, S=5, N=3)
    
    # Test NESS: verify it's a probability distribution
    ness_batch = ctmc_batch.get_ness(diagnostic=False)
    for i in range(3):
        assert_close(ness_batch[i].sum(), 1.0, atol=1e-8,
                     msg=f"NESS batch[{i}] should sum to 1")
        assert np.all(ness_batch[i] >= 0), f"NESS batch[{i}] should be non-negative"
    print("✓ All batch NESS distributions are valid")
    
    # Test MEPS: verify it's a probability distribution
    meps_batch = ctmc_batch.get_meps(diagnostic=False)
    for i in range(3):
        assert_close(meps_batch[i].sum(), 1.0, atol=1e-8,
                     msg=f"MEPS batch[{i}] should sum to 1")
        assert np.all(meps_batch[i] >= 0), f"MEPS batch[{i}] should be non-negative"
    print("✓ All batch MEPS distributions are valid")
    
    # Test EPR: verify property EPR(MEPS) <= EPR(NESS) holds for each system
    epr_ness_batch = ctmc_batch.get_epr(ness_batch)
    epr_meps_batch = ctmc_batch.get_epr(meps_batch)
    for i in range(3):
        assert epr_meps_batch[i] <= epr_ness_batch[i] + 1e-6, \
            f"Batch[{i}]: EPR(MEPS) should be <= EPR(NESS)"
    print("✓ EPR(MEPS) <= EPR(NESS) holds for all batch systems")


def test_small_rate_tolerance():
    """Test handling of very small rates."""
    print("\n=== Test: Small Rate Tolerance ===")
    
    S = 4
    R = np.ones((S, S)) * 0.1
    
    # Add some extremely small rates
    R[0, 1] = 1e-15
    R[1, 2] = 1e-14
    
    ctmc = ContinuousTimeMarkovChain(R=R)
    
    # Small rates should be treated as forbidden
    assert ctmc.forbidden_mask[0, 1], "Rate 1e-15 should be forbidden"
    assert ctmc.forbidden_mask[1, 2], "Rate 1e-14 should be forbidden"
    print("✓ Very small rates are correctly identified as forbidden")
    
    # NESS should still be computable
    ness = ctmc.get_ness(diagnostic=False)
    assert_close(ness.sum(), 1.0, atol=1e-8)
    print("✓ NESS computable even with forbidden transitions")


def test_time_reversal_symmetry_involution():
    """Test that rev_R respects involution symmetry."""
    print("\n=== Test: Time-Reversal Involution Symmetry ===")
    
    S = 6
    ctmc = ContinuousTimeMarkovChain(generator=uniform_generator, S=S, N=1)
    ctmc.time_even_states = False
    
    # Custom involution
    custom_involution = np.array([1, 0, 3, 2, 5, 4])
    ctmc.involution_indices = custom_involution
    ctmc.set_rate_matrix(ctmc.R)
    
    # rev_R should be the time-reversed (and involution-permuted) version
    P = ctmc.get_reversal_matrix()
    
    # rev_R should relate to R through the involution
    # Specifically: rev_R = P @ R @ P^T
    expected_rev_R = P @ ctmc.R @ P.T
    assert_close(ctmc.rev_R, expected_rev_R, atol=1e-10,
                 msg="rev_R should equal P @ R @ P^T")
    print("✓ rev_R correctly implements involution symmetry")


def test_batch_forbidden_transitions():
    """Test forbidden transitions in batch mode."""
    print("\n=== Test: Batch Mode Forbidden Transitions ===")
    
    S = 5
    N = 3
    R = np.random.uniform(0.01, 1, (N, S, S))
    
    # Make some transitions forbidden in batch
    R[0, 0, 1] = 1e-13
    R[0, 1, 0] = 1e-13
    R[1, 2, 3] = 1e-13
    R[1, 3, 2] = 1e-13
    
    ctmc = ContinuousTimeMarkovChain(R=R)
    
    # Check forbidden mask is properly set in batch mode
    assert ctmc.batch, "Should be in batch mode"
    assert ctmc.forbidden_mask.shape == (N, S, S), "Forbidden mask should match R shape"
    
    # For time-even, forbidden transitions should be symmetric
    for n in range(N):
        for i in range(S):
            for j in range(i+1, S):
                if ctmc.forbidden_mask[n, i, j]:
                    assert ctmc.forbidden_mask[n, j, i], \
                        f"Batch[{n}]: forbidden [{i},{j}] but not [{j},{i}]"
    
    print("✓ Batch mode forbidden transitions are correctly symmetric")


def test_dynamics_conserve_probability():
    """Test that master equation dynamics conserve probability."""
    print("\n=== Test: Probability Conservation ===")
    
    ctmc = ContinuousTimeMarkovChain(generator=uniform_generator, S=5, N=1)
    state = ctmc.get_random_state()
    
    # Time derivative should not change total probability
    time_deriv = ctmc.get_time_deriv(state)
    prob_change = time_deriv.sum()
    assert_close(prob_change, 0.0, atol=1e-10,
                 msg="Sum of time derivatives should be zero")
    print("✓ Master equation conserves probability")


def test_batch_dynamics():
    """Test dynamics in batch mode."""
    print("\n=== Test: Batch Mode Dynamics ===")
    
    ctmc = ContinuousTimeMarkovChain(generator=uniform_generator, S=5, N=3)
    states = ctmc.get_random_state()
    
    # Batch time derivatives
    time_derivs = ctmc.get_time_deriv(states)
    assert time_derivs.shape == (3, 5), "Batch time derivatives shape"
    
    # Each should conserve probability
    for i in range(3):
        prob_change = time_derivs[i].sum()
        assert_close(prob_change, 0.0, atol=1e-10)
    
    print("✓ Batch mode dynamics conserve probability")


def test_involution_generator():
    """Test the involution_builder utility."""
    print("\n=== Test: Involution Builder ===")
    
    for S in [4, 6, 8]:
        inv = involution_builder(S)
        
        # Should be a valid involution: σ(σ(s)) = s
        inv_inv = inv[inv]
        assert np.array_equal(inv_inv, np.arange(S)), \
            f"Involution builder failed for S={S}"
    
    print("✓ Involution builder generates valid involutions")


def test_ness_time_derivative():
    """Test that NESS has ~zero time derivative."""
    print("\n=== Test: NESS Time Derivative ===")
    
    ctmc = ContinuousTimeMarkovChain(generator=cyclic_generator, S=6, N=1)
    ness = ctmc.get_ness(diagnostic=False)
    
    time_deriv_at_ness = ctmc.get_time_deriv(ness)
    
    # Should be very close to zero
    max_deriv = np.max(np.abs(time_deriv_at_ness))
    assert max_deriv < 1e-5, \
        f"NESS time derivative too large: {max_deriv:.2e}"
    print(f"✓ NESS time derivative is small (max={max_deriv:.2e})")


def test_meps_from_various_initializations():
    """Test that MEPS can be reached from various initial conditions."""
    print("\n=== Test: MEPS from Various Initializations ===")
    
    ctmc = ContinuousTimeMarkovChain(generator=uniform_generator, S=5, N=1)
    
    # Initialize from different starting points
    initial_states = [
        ctmc.get_uniform(),
        ctmc.get_random_state(),
        ctmc.get_ness(diagnostic=False),
    ]
    
    meps_solutions = []
    for state in initial_states:
        meps = ctmc.get_meps(state=state, diagnostic=False)
        meps_solutions.append(meps)
    
    # All should converge to similar EPR values (though not exactly the same state)
    eprs = [ctmc.get_epr(m) for m in meps_solutions]
    for i in range(1, len(eprs)):
        # EPR values should be close
        ratio = eprs[i] / eprs[0] if eprs[0] > 1e-10 else eprs[i]
        assert 0.99 < ratio < 1.01 or abs(eprs[i] - eprs[0]) < 1e-8, \
            f"EPR values differ: {eprs[0]:.6f} vs {eprs[i]:.6f}"
    
    print(f"✓ MEPS converges from various initializations (EPRs: {[f'{e:.6f}' for e in eprs]})")


def test_arrhenius_pump():
    """Test Arrhenius pump generator."""
    print("\n=== Test: Arrhenius Pump Generator ===")
    
    from ctmc import arrhenius_pump_generator
    ctmc = ContinuousTimeMarkovChain(
        generator=arrhenius_pump_generator,
        S=6,
        N=1,
        beta=2.0,
        n_pumps=2,
        pump_strength=1.0
    )
    
    # Should have valid structure
    assert ctmc.S == 6
    row_sums = ctmc.R.sum(axis=-1)
    assert_close(row_sums, np.zeros_like(row_sums), atol=1e-10)
    print("✓ Arrhenius pump rate matrix is valid")
    
    # Should compute NESS
    ness = ctmc.get_ness(diagnostic=False)
    assert_close(ness.sum(), 1.0, atol=1e-8)
    print("✓ Arrhenius pump system has well-defined NESS")


# ---------------------------------------------------------------------------
# Sparsification and connectivity tests
# ---------------------------------------------------------------------------

def test_is_irreducible():
    """Test the is_irreducible utility on known graphs."""
    from ctmc import is_irreducible

    # Fully connected -> irreducible
    R_full = np.random.uniform(0.1, 1.0, (5, 5))
    np.fill_diagonal(R_full, 0)
    assert is_irreducible(R_full) == True, 'fully connected graph should be irreducible'
    print("  - fully connected: irreducible")

    # Disconnected: two isolated blocks
    R_disc = np.zeros((6, 6))
    R_disc[0, 1] = R_disc[1, 0] = 1.0
    R_disc[1, 2] = R_disc[2, 1] = 1.0
    R_disc[2, 0] = R_disc[0, 2] = 1.0
    R_disc[3, 4] = R_disc[4, 3] = 1.0
    R_disc[4, 5] = R_disc[5, 4] = 1.0
    R_disc[5, 3] = R_disc[3, 5] = 1.0
    assert is_irreducible(R_disc) == False, 'disconnected graph should not be irreducible'
    print("  - disconnected blocks: not irreducible")

    # Single state (trivially irreducible)
    R_one = np.zeros((1, 1))
    # For a 1-state system, BFS from 0 reaches {0}, which equals S=1
    # But let's just make sure it doesn't crash
    result = is_irreducible(R_one)
    assert result == True, '1-state system should be irreducible'
    print("  - single state: irreducible")

    # Batch mode
    R_batch = np.stack([R_full[:4, :4], R_disc[:4, :4]])
    results = is_irreducible(R_batch)
    assert results.shape == (2,)
    assert results[0] == True, 'first batch element should be irreducible'
    # R_disc[:4,:4] has states 0-2 connected but state 3 isolated
    assert results[1] == False, 'second batch element should not be irreducible'
    print("  - batch mode works correctly")

    # Directed cycle (strongly connected)
    R_cycle = np.zeros((4, 4))
    R_cycle[0, 1] = R_cycle[1, 2] = R_cycle[2, 3] = R_cycle[3, 0] = 1.0
    assert is_irreducible(R_cycle) == True, 'directed cycle should be irreducible'
    print("  - directed cycle: irreducible")

    # Directed chain (NOT strongly connected)
    R_chain = np.zeros((4, 4))
    R_chain[0, 1] = R_chain[1, 2] = R_chain[2, 3] = 1.0
    assert is_irreducible(R_chain) == False, 'directed chain should not be irreducible'
    print("  - directed chain: not irreducible")

    print("✓ is_irreducible works correctly on all test cases")


def test_sparsify_basic():
    """Test basic sparsification: shapes, edge removal, diagonal preservation."""
    from ctmc import sparsify

    S = 20
    R_dense = np.random.uniform(0.1, 1.0, (S, S))
    np.fill_diagonal(R_dense, -R_dense.sum(axis=1))

    R_sparse = sparsify(R_dense, p=0.3, seed=42)

    # Shape preserved
    assert R_sparse.shape == R_dense.shape, 'shape should be preserved'
    print("  - shape preserved")

    # Some edges removed (statistically certain at p=0.3, S=20)
    off_diag = ~np.eye(S, dtype=bool)
    n_orig = np.sum(R_dense[off_diag] != 0)
    n_sparse = np.sum(R_sparse[off_diag] != 0)
    assert n_sparse < n_orig, f'expected fewer edges: got {n_sparse} vs {n_orig}'
    print(f"  - edges removed: {n_orig} -> {n_sparse}")

    # Nonzero entries in sparse are a subset with correct values
    kept = R_sparse[off_diag] != 0
    orig_vals = R_dense[off_diag][kept]
    sparse_vals = R_sparse[off_diag][kept]
    assert_close(sparse_vals, orig_vals, atol=1e-15)
    print("  - kept edge values match original")

    print("✓ Basic sparsification works correctly")


def test_sparsify_avg_degree():
    """Test avg_degree parameterization."""
    from ctmc import sparsify

    S = 50
    N = 5
    R = np.random.uniform(0.1, 1.0, (N, S, S))
    k_target = 8.0

    R_sparse = sparsify(R, avg_degree=k_target, seed=123)

    # Measure actual average degree across batch
    off_diag = ~np.eye(S, dtype=bool)
    degrees = []
    for n in range(N):
        Rn = R_sparse[n]
        # Count nonzero outgoing edges per state
        out_degree = np.sum((Rn != 0) & off_diag, axis=1)
        degrees.append(out_degree.mean())

    avg_measured = np.mean(degrees)
    p_expected = k_target / (S - 1)
    # With N*S*S samples, the measured degree should be close
    assert abs(avg_measured - k_target) < 2.0, \
        f'measured avg degree {avg_measured:.1f} too far from target {k_target}'
    print(f"  - target degree {k_target}, measured {avg_measured:.1f}")

    print("✓ avg_degree parameterization works correctly")


def test_sparsify_connectivity():
    """Test that ensure_connected repairs disconnected graphs."""
    from ctmc import sparsify, is_irreducible

    S = 30
    R = np.random.uniform(0.1, 1.0, (S, S))

    # Very aggressive sparsification - high chance of disconnection
    R_sparse = sparsify(R, p=0.05, ensure_connected=True, seed=99)
    assert is_irreducible(R_sparse), 'ensure_connected should guarantee irreducibility'
    print("  - low p (0.05): connectivity repaired")

    # Even more aggressive
    R_sparse2 = sparsify(R, p=0.02, ensure_connected=True, seed=77)
    assert is_irreducible(R_sparse2), 'ensure_connected should work at very low p'
    print("  - very low p (0.02): connectivity repaired")

    # Without repair, low p should often produce disconnected graphs
    n_disconnected = 0
    for s in range(20):
        R_no_repair = sparsify(R, p=0.03, ensure_connected=False, seed=s)
        if not is_irreducible(R_no_repair):
            n_disconnected += 1
    assert n_disconnected > 0, 'at p=0.03, S=30, some samples should be disconnected'
    print(f"  - without repair, {n_disconnected}/20 were disconnected (expected)")

    print("✓ Connectivity repair works correctly")


def test_sparsify_batch():
    """Test that batch mode produces independent masks per matrix."""
    from ctmc import sparsify, is_irreducible

    S = 15
    N = 10
    R = np.random.uniform(0.1, 1.0, (N, S, S))

    R_sparse = sparsify(R, avg_degree=4, ensure_connected=True, seed=42)

    assert R_sparse.shape == (N, S, S), 'batch shape should be preserved'
    print("  - batch shape preserved")

    # Each matrix should have a different sparsity pattern
    off_diag = ~np.eye(S, dtype=bool)
    patterns = [R_sparse[n][off_diag] != 0 for n in range(N)]
    all_same = all(np.array_equal(patterns[0], p) for p in patterns[1:])
    assert not all_same, 'batch elements should have independent masks'
    print("  - independent masks per batch element")

    # All should be connected
    connected = is_irreducible(R_sparse)
    assert np.all(connected), 'all batch elements should be connected'
    print("  - all batch elements connected")

    print("✓ Batch sparsification works correctly")


def test_sparsify_integration():
    """Integration test: sparsify -> CTMC -> NESS -> MEPS -> physics checks."""
    from ctmc import sparsify, is_irreducible, arrhenius_pump_generator

    S = 15
    N = 5
    R_dense = arrhenius_pump_generator(S=S, N=N, n_pumps=20, pump_strength=3)
    R_sparse = sparsify(R_dense, avg_degree=5, ensure_connected=True, seed=42)

    # Feed into CTMC
    machine = ContinuousTimeMarkovChain(R=R_sparse)
    print(f"  - created CTMC: {machine}")

    ness = machine.get_ness()
    meps = machine.get_meps()

    # NESS should be valid probability distributions
    assert_close(ness.sum(axis=-1), np.ones(N), atol=1e-6)
    print("  - NESS sums to 1")

    # EPR(MEPS) <= EPR(NESS)
    epr_ness = machine.get_epr(ness)
    epr_meps = machine.get_epr(meps)
    assert np.all(epr_meps <= epr_ness + 1e-6), \
        f'MEPS should minimize EPR: {epr_meps} vs {epr_ness}'
    print(f"  - EPR(MEPS) <= EPR(NESS) verified for all {N} systems")

    # NESS time derivative should be ~0
    deriv = machine.get_time_deriv(ness)
    assert np.all(np.abs(deriv) < 1e-3), 'NESS time derivative should be near zero'
    print("  - NESS is stationary")

    print("✓ Full integration test passed (sparsify -> CTMC -> NESS/MEPS)")


def test_sparsify_seed_reproducibility():
    """Test that seed produces reproducible results."""
    from ctmc import sparsify

    R = np.random.uniform(0.1, 1.0, (10, 10))

    R1 = sparsify(R, p=0.5, seed=42)
    R2 = sparsify(R, p=0.5, seed=42)
    R3 = sparsify(R, p=0.5, seed=99)

    assert np.array_equal(R1, R2), 'same seed should give same result'
    assert not np.array_equal(R1, R3), 'different seeds should give different results'
    print("✓ Seed reproducibility works correctly")


# ---------------------------------------------------------------------------
# JAX MEPS tests
# ---------------------------------------------------------------------------

from ctmc import HAS_JAX

import pytest

def _skip_unless_jax():
    if not HAS_JAX:
        pytest.skip("JAX not installed")


def test_jax_meps_easy_case():
    """Test that JAX MEPS agrees with Euler MEPS on easy cases."""
    _skip_unless_jax()
    from ctmc import arrhenius_pump_generator
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    np.random.seed(42)
    R = arrhenius_pump_generator(S=5, N=1, n_pumps=6, pump_strength=4)[0]

    m1 = ContinuousTimeMarkovChain(R=R)
    m1.get_ness()
    meps_euler = m1.get_meps(method='euler', max_iter=5000)
    epr_euler = float(m1.get_epr(meps_euler))

    m2 = ContinuousTimeMarkovChain(R=R)
    m2.get_ness()
    meps_jax = m2.get_meps_jax()
    epr_jax = float(m2.get_epr(meps_jax))

    # Both should find similar EPR on easy cases
    assert abs(epr_euler - epr_jax) / max(epr_euler, 1e-15) < 0.01, \
        f"Easy case: Euler EPR={epr_euler:.8f} vs JAX EPR={epr_jax:.8f} differ by >1%"
    print(f"✓ JAX MEPS agrees with Euler on easy case (EPR={epr_jax:.8f})")


def test_jax_meps_hard_case():
    """Test that JAX MEPS finds lower EPR than Euler on hard cases."""
    _skip_unless_jax()
    from ctmc import arrhenius_pump_generator
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    np.random.seed(42)
    R = arrhenius_pump_generator(S=9, N=1, n_pumps=28, pump_strength=12)[0]

    m1 = ContinuousTimeMarkovChain(R=R)
    m1.get_ness()
    meps_euler = m1.get_meps(method='euler', max_iter=10000)
    epr_euler = float(m1.get_epr(meps_euler))

    m2 = ContinuousTimeMarkovChain(R=R)
    m2.get_ness()
    meps_jax = m2.get_meps_jax()
    epr_jax = float(m2.get_epr(meps_jax))

    # JAX should find significantly lower EPR
    assert epr_jax < epr_euler, \
        f"Hard case: JAX EPR={epr_jax:.8f} should be < Euler EPR={epr_euler:.8f}"
    # The improvement should be substantial (we saw ~80% in benchmarking)
    improvement = (epr_euler - epr_jax) / epr_euler
    assert improvement > 0.5, \
        f"Hard case: JAX improvement only {improvement*100:.1f}%, expected >50%"
    print(f"✓ JAX MEPS finds lower EPR on hard case: {epr_jax:.8f} vs {epr_euler:.8f} ({improvement*100:.1f}% better)")


def test_jax_meps_batch_mode():
    """Test JAX MEPS in batch mode matches individual results."""
    _skip_unless_jax()
    from ctmc import arrhenius_pump_generator
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    np.random.seed(42)
    N = 5
    R = arrhenius_pump_generator(S=8, N=N, n_pumps=20, pump_strength=5)

    # Batch mode
    m_batch = ContinuousTimeMarkovChain(R=R)
    m_batch.get_ness()
    meps_batch = m_batch.get_meps_jax()
    epr_batch = m_batch.get_epr(meps_batch)

    # Individual single-system mode
    epr_singles = np.zeros(N)
    for n in range(N):
        m_single = ContinuousTimeMarkovChain(R=R[n])
        m_single.get_ness()
        meps_single = m_single.get_meps_jax()
        epr_singles[n] = float(m_single.get_epr(meps_single))

    # Results should be very close (both find global minimum)
    assert_close(epr_batch, epr_singles, rtol=0.01,
                 msg="Batch vs single JAX MEPS EPR mismatch")
    print(f"✓ JAX MEPS batch mode matches individual results (max diff: {np.max(np.abs(epr_batch - epr_singles)):.2e})")


def test_jax_meps_epr_leq_ness():
    """Test that JAX MEPS EPR ≤ NESS EPR (basic sanity)."""
    _skip_unless_jax()
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    np.random.seed(42)
    for S in [5, 10, 20]:
        m = ContinuousTimeMarkovChain(S=S)
        ness = m.get_ness()
        meps = m.get_meps_jax()
        epr_ness = float(m.get_epr(ness))
        epr_meps = float(m.get_epr(meps))
        assert epr_meps <= epr_ness + 1e-10, \
            f"S={S}: JAX MEPS EPR={epr_meps:.8f} > NESS EPR={epr_ness:.8f}"
    print(f"✓ JAX MEPS EPR ≤ NESS EPR for all tested sizes")


def test_jax_meps_detailed_balance():
    """Test that JAX MEPS finds near-zero EPR for detailed balance systems."""
    _skip_unless_jax()
    from ctmc import detailed_balance_generator
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    np.random.seed(42)
    R = detailed_balance_generator(S=8, N=1)[0]
    m = ContinuousTimeMarkovChain(R=R)
    m.get_ness()
    meps = m.get_meps_jax()
    epr = float(m.get_epr(meps))
    assert epr < 1e-6, f"Detailed balance: JAX MEPS EPR={epr:.2e} should be near zero"
    print(f"✓ JAX MEPS on detailed balance: EPR={epr:.2e} ≈ 0")


def test_jax_meps_multi_restart():
    """Test that multi-restart finds same or better solution."""
    _skip_unless_jax()
    from ctmc import arrhenius_pump_generator
    import warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    np.random.seed(42)
    R = arrhenius_pump_generator(S=9, N=1, n_pumps=28, pump_strength=12)[0]

    m1 = ContinuousTimeMarkovChain(R=R)
    m1.get_ness()
    meps_1 = m1.get_meps_jax(num_restarts=1)
    epr_1 = float(m1.get_epr(meps_1))

    m2 = ContinuousTimeMarkovChain(R=R)
    m2.get_ness()
    meps_3 = m2.get_meps_jax(num_restarts=3)
    epr_3 = float(m2.get_epr(meps_3))

    assert epr_3 <= epr_1 + 1e-10, \
        f"Multi-restart EPR={epr_3:.8f} should be ≤ single start EPR={epr_1:.8f}"
    print(f"✓ Multi-restart: 1-start EPR={epr_1:.8f}, 3-start EPR={epr_3:.8f}")


def run_all_tests():
    """Run all tests and report results."""
    tests = [
        ("Basic CTMC Creation", test_basic_ctmc_creation),
        ("Rate Matrix Validation", test_rate_matrix_validation),
        ("Time-Even States", test_time_even_states),
        ("Custom Involution", test_custom_involution),
        ("Arbitrary Involutions", test_arbitrary_involution),
        ("Forbidden Transitions (time-even)", test_forbidden_transitions_time_even),
        ("Forbidden Transitions (involution)", test_forbidden_transitions_involution),
        ("NESS Computation", test_ness_computation),
        ("MEPS Computation", test_meps_computation),
        ("EPR Calculations", test_epr_calculations),
        ("EPR(MEPS) <= EPR(NESS)", test_epr_meps_less_than_ness),
        ("Detailed Balance Systems", test_detailed_balance_system),
        ("Batch vs Single", test_batch_vs_single),
        ("Small Rate Tolerance", test_small_rate_tolerance),
        ("Time-Reversal Involution Symmetry", test_time_reversal_symmetry_involution),
        ("Batch Forbidden Transitions", test_batch_forbidden_transitions),
        ("Probability Conservation", test_dynamics_conserve_probability),
        ("Batch Dynamics", test_batch_dynamics),
        ("Involution Builder", test_involution_generator),
        ("NESS Time Derivative", test_ness_time_derivative),
        ("MEPS from Various Initializations", test_meps_from_various_initializations),
        ("Arrhenius Pump Generator", test_arrhenius_pump),
        # Sparsification and connectivity tests
        ("is_irreducible", test_is_irreducible),
        ("Sparsify Basic", test_sparsify_basic),
        ("Sparsify avg_degree", test_sparsify_avg_degree),
        ("Sparsify Connectivity Repair", test_sparsify_connectivity),
        ("Sparsify Batch", test_sparsify_batch),
        ("Sparsify Integration", test_sparsify_integration),
        ("Sparsify Seed Reproducibility", test_sparsify_seed_reproducibility),
        # JAX MEPS tests (skipped if JAX not installed)
        ("JAX MEPS Easy Case", test_jax_meps_easy_case),
        ("JAX MEPS Hard Case", test_jax_meps_hard_case),
        ("JAX MEPS Batch Mode", test_jax_meps_batch_mode),
        ("JAX MEPS EPR ≤ NESS", test_jax_meps_epr_leq_ness),
        ("JAX MEPS Detailed Balance", test_jax_meps_detailed_balance),
        ("JAX MEPS Multi-Restart", test_jax_meps_multi_restart),
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((test_name, str(e)))
            print(f"✗ FAILED: {test_name}")
            print(f"  Error: {e}\n")
    
    print("\n" + "="*70)
    print(f"Test Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("="*70)
    
    if failed > 0:
        print("\nFailed tests:")
        for test_name, error in errors:
            print(f"\n  {test_name}:")
            print(f"    {error[:200]}")
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_tests()
    exit(0 if failed == 0 else 1)
