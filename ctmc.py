import numpy as np

def uniform_generator(S = 10, N=1, min_rate_tol=6):
    return np.random.uniform(10**(-min_rate_tol), 1, (N,S,S)).round(decimals=min_rate_tol)

def normal_generator(S=10, N=1, mu=0, sigma=1):
    return np.abs(np.random.normal(mu, sigma, (N,S,S)))

def gamma_generator(S=10, N=1, mu=1, sigma=.1):
    return np.random.gamma(mu**2/sigma**2, sigma**2/mu, (N,S,S))

def cyclic_generator(S=5, N=1, mu=0, sigma=1, max_jump=None, min_rate=6):
    # imposes a kind of 1-D ring connectivity

    # if max_jump is 1, then only nearest neighbor jumps are allowed (1->2, 2->3, 5->4 etc..)
    if max_jump is None or max_jump > S-1:
        max_jump = S-1

    R = np.ones((N,S,S)) * 10**(-min_rate)
    
    
    for i in range(1,max_jump+1):
        # sets forward rates to be decreasing in distance from initial state plus a random normal distribution, here N(mu,sigma).
        # for example, the jump rate from 2->3 in a 5 state system would be 4+N(mu, sigma) where 2->5 would be 1+N(mu,sigma)
        new_vals_p = np.abs((S-i)+np.random.normal(mu, sigma, size=(N,S)))
        # in order to be less on average, and set a preferred direciton the reverse rates are set to be R_{3->2} = N(.6,.3)*R_{2->3}
        new_vals_m = np.abs(new_vals_p*np.random.normal(.6, .3, size=(N,S)))

        #set the transition rates, one disgonal at a time
        R += np.multiply(new_vals_p[..., None], np.eye(S, k=i))
        R += np.multiply(new_vals_m[..., None], np.eye(S, k=-i))

    return R

def detailed_balance_generator(S=5, N=1, energy = None, energy_gen = np.random.uniform, beta=1, gen_args=[.1,1]):
    if energy is None:
        energy = energy_gen(*gen_args, size=(N,S))
    assert energy.shape == (N,S)    
    return np.exp(beta*(energy[:,:,None]-energy[:,None,:]))

from sys import getsizeof
from time import sleep
def arrhenius_pump_generator(S=5, N=1, energy=None, barrier=None, energy_gen=np.random.uniform, beta=1, gen_args=[.1,1], n_pumps=0, pump_strength=10):
    n_pairs = int((S**2-S)/2)

    if energy is None:
        energy = energy_gen(*gen_args, size=(N,S))
    assert energy.shape == (N,S)
    
    if barrier is None:
        barrier = energy_gen(*gen_args, size=(N,n_pairs)) 
    assert barrier.shape == (N, n_pairs)
    barrier = np.abs(barrier)
    delta_E = -energy[:,:,None] + energy[:,None,:]
    # find transitions where delta_E is negative
    idx = np.argwhere(delta_E < 0).reshape(N, -1, 3)

    #delta_E = np.where(delta_E>0, barrier+delta_E[delta_E>0], delta_E)
    # transitions from high energy to low energy just need to overcome the barrier
    delta_E[idx[...,0],idx[...,1],idx[...,2]] = barrier
    sleep(10)
    # low to high energy transitions must overcome the energy difference and the barrier as well
    delta_E[idx[...,0],idx[...,2],idx[...,1]] += barrier

    barrier = None
    idx = None


    # adds catalytic pumps to some one way transitions
    if n_pumps > 0:
        idx = np.zeros( shape=(N, n_pumps, 3), dtype='int')
        pump_offsets = energy_gen(*gen_args, size=(N,n_pumps)) * pump_strength
        idx[...,1:] =  np.random.randint(0,S, size = (N, n_pumps, 2))
        idx[...,0] = np.floor(np.arange(0,N*n_pumps)/(n_pumps)).reshape(N,-1)
        delta_E[tuple(idx.T)] += pump_offsets.T
    #print([getsizeof(item)/2**30 for item in [energy, barrier, delta_E, idx, pump_offsets]])
    delta_E = np.exp(-delta_E)

    return delta_E




class ContinuousTimeMarkovChain():
    def __init__(self, R=None, generator=uniform_generator, **gen_kwargs):
        self.scale = 1
        self.timescale = 1
        self.batch = False
        self.time_even_states = True
        self.analytic_threshhold = 65
        self.min_rate = 1E-32
        self.verbose = False
        if R is not None:
            self.set_rate_matrix(R)
        else:
            self.generator = generator
            self.gen_kwargs = gen_kwargs
            self.set_rate_matrix(self.generator(**self.gen_kwargs))     
    
    def set_rate_matrix(self, R, max_rate=1):
        R = self.verify_rate_matrix(R)
        R = self.normalize_R(R)
        self.R = R
        if self.time_even_states is True:
            self.rev_R = self.R
        if self.time_even_states is False:
            self.rev_R = self.get_reversal_matrix() @ self.R @ self.get_reversal_matrix().T
        self.__set_statewise_Q()
        return 
    
    def __R_info(self, R):
        if self.batch:
            return len(R[0,...]), R[0,...].shape, R[0,...].size
        else:
            return len(R), R.shape, R.size
    
    def __set_diags(self, R):
        R = np.abs(R)
        if self.batch:
            R -= R.sum(axis=-1)[:,:,np.newaxis]*np.identity(self.S)
        else:
            R -= np.identity(self.S)*R.sum(axis=-1)

        return R

    def verify_rate_matrix(self,R):
        if not type(R) is np.ndarray:
            R = np.array(R)
        R = np.squeeze(R)
        if len(R.shape) == 3:
            self.batch = True
        
        R = self.normalize_R(R)

        S, shape, size = self.__R_info(R)
        
        assert len(shape) == 2 and size == S**2, 'each R must be a 2D square matrix'
        assert ( np.sign(R-R*np.identity(S)) == np.ones(shape)-np.identity(S)).all(), 'R_ij must be >0 for i!=j'

        self.S = S

        R[np.abs(R) <= self.min_rate] = self.min_rate
        
        if not (R.sum(axis=-1) == np.zeros(self.S)).all():
            R = self.__set_diags(R)

        return R
    
    def normalize_R(self, R):
        if self.batch:
            R_max = np.abs(R).max(axis=-1).max(axis=-1)
        else:
            R_max = np.abs(R).max()

        self.scale = R_max / self.timescale

        #this is if we want to allow slower scale processes, only cutting the fast ones down
        #scale = R_max/ (np.minimum(R_max,self.timescale))

        if self.batch:
            R = (R.T / self.scale).T
        else:
            R = R / self.scale

        return R
    
    def __set_statewise_Q(self):
        if self.batch:
            rev_R_T = self.rev_R.transpose(0,2,1)
        else:
            rev_R_T = self.rev_R.T
        
        #this makes sure the rev transpose gives 0 for all diagonal elements when taking the log ratio
        rev_R_T = rev_R_T - rev_R_T*np.identity(self.S) + self.R*np.identity(self.S)
        
        R_diags, rev_R_diags = [ R*np.identity(self.S) for R in [self.R, self.rev_R]]

        self.statewise_Q = ((self.R * np.log(self.R/rev_R_T) )+ R_diags - rev_R_diags).sum(axis=-1)

        if np.any(np.isnan(self.statewise_Q)):
            print('statewise heat contained NANs, re-veryfying matrix')
            self.set_rate_matrix(self.R)

        return



    def get_reversal_matrix(self):
        try: involution = self.involution_indices
        except:
            self.involution_indices = involution_builder(self.S)
            involution = self.involution_indices
        rev_matrix = np.zeros((self.S,self.S))
        rev_matrix[range(self.S),involution]=1
        return rev_matrix

    def get_time_deriv(self, state, R = None):
        if R is None:
            R = self.R

        if self.batch:
            return np.einsum('ni,nik->nk',state, R)
        else:
            return np.matmul(state, R)
    
    def get_meps_deriv(self, state, R = None):
        if R is None:
            R = self.R

        if self.batch:
            return self.get_time_deriv(state) + state * ( self.get_epr(state) - self.get_statewise_epr(state).T ).T
        else:
            return self.get_time_deriv(state) + state * ( self.get_epr(state) - self.get_statewise_epr(state) )
    

    def get_statewise_activity_in(self, state):
        return self.get_time_deriv(state, R=self.R*(1-np.identity(self.S)))
        #return self.get_time_deriv(state) - state *((self.R*np.identity(self.S)).sum(axis=-1))

    def get_statewise_activity_out(self, state, R=None):
        if R is None:
            R = self.R
        R = R*np.identity(self.S)
        if self.batch:
            return -np.einsum('nik,nk->ni', R , state )
        else:
            return -np.matmul(R, state)


    def get_activity(self, state):
        return self.get_statewise_activity_in(state).sum(axis=-1)
    
    def get_statewise_prob_current(self, state):
        return np.abs(self.get_time_deriv(state))
    
    def get_prob_current(self, state):
        return self.get_statewise_prob_current(state).sum(axis=-1)

    def evolve_state(self, state, dt):
        return self.normalize_state(state + dt*self.get_time_deriv(state))
    
    def get_statewise_epr(self, state):
        if self.batch:
            surprisal_rate = -np.einsum('nik,nk->ni', self.R , np.log(state) )
        else:
            surprisal_rate = -np.matmul(self.R, np.log(state))

        return surprisal_rate + self.statewise_Q
    
    def get_epr(self,state):
        if self.batch:
            return np.einsum('nk,nk->n', state, self.get_statewise_epr(state))
        else:
            return np.matmul(state, self.get_statewise_epr(state))
    
    def get_uniform(self):
        if self.batch:
            return np.ones((len(self.R),self.S))/self.S
        else:
            return np.ones(self.S)/self.S
    
    def get_random_state(self):
        if self.batch:
            state = np.random.uniform(1E-16, 1, (self.R.shape[0],self.S))
        else:
            state = np.random.uniform(1E-16, 1, self.S)
        return (state.T/ state.sum(axis=-1)).T
    
    def get_local_state(self, mu=None, sigma=None):
        if self.batch:
            N = self.R.shape[0]
        else:
            N = 1
        
        state = np.mgrid[0:N, 0:self.S][1]
        if mu is None:
            mu = np.random.choice(self.S,size=N)
        if sigma is None:
            sigma = np.random.uniform(.1,int(self.S/3),(N))

        state = np.exp(-cyclic_distance(state, mu)**2/(2*sigma[:,None]**2))
        state = state / np.sum(state, axis=-1)[:,None]
        return state
    

    def get_meps(self, dt0=1, dtmin=.001 , state=None, max_iter=50_000, dt_iter=150, diagnostic=False):
        if state is None:
            try:
                state = self.meps
            except:
                try:
                    state = self.ness
                except:
                    state = self.get_uniform()

        eprs = [self.get_epr(state)]
        states = [state]
        doneBool = np.array(False)
        doneEprBool = np.array(False)
        negativeBool = False
        nanStateBool = False
        i=0
        if self.batch:
            j = -1*np.ones(self.R.shape[0])
        else:
            j = np.array(-1)
        #dt = dt0 * (2/(2+j))
    
        while (not np.all(doneBool) or np.any(negativeBool) or np.any(nanStateBool)) and i < max_iter:
            if i % dt_iter == 0:
                j += 0 
            dt = dt0 * (2/(2+j))

            if self.batch:
                new_state = state + dt[:,None]*self.get_meps_deriv(state)
                #new_state[doneBool] = state[doneBool]  
            else:
                new_state = state + dt * self.get_meps_deriv(state)
            
            negativeBool = np.any(new_state < 0, axis=-1)
            nanStateBool = np.any(np.isnan(new_state), axis=-1)

            if np.any(negativeBool):
                num_neg = np.sum(negativeBool)
                if diagnostic:
                    print(f'rewinding {num_neg} states to avoid negative states at iteration {i}')
                new_state[negativeBool] = state[negativeBool]
                j[negativeBool] += 1
                #if diagnostic:
                #    print(f'dt changed at iteration{i}')
            if np.any(nanStateBool):
                num_nan = sum(nanStateBool)
                if diagnostic:
                    print(f'rewinding {num_nan} states to avoid nan at iteration {i}')
                new_state[nanStateBool] = state[nanStateBool]
                j[nanStateBool] += 1
            
            new_state = (new_state.T / new_state.sum(axis=-1)).T

            epr = self.get_epr(state)
            eprs.append(epr)

            states.append(new_state)
            state = new_state

            if not diagnostic:
                eprs = eprs[-3:]
                states = states[-3:]
            if i >= dt_iter:
                doneBool = np.all(np.isclose(0, np.abs(self.get_meps_deriv(state))/np.maximum(1E-12,state), atol=1E-4), axis=-1)
                #doneBool = np.all(np.isclose(states[-1],states[-2], rtol=1E-4, atol=1E-14), axis=-1)
                doneEprBool = np.isclose(eprs[-1],eprs[-2], rtol=1E-3*dt, atol=1E-14)

            i += 1
        if i == max_iter:
            print(f'meps didnt converge after {i} iterations in {np.sum(~doneBool)} machines')
            print(f'mepr didnt converge after {i} iterations in {np.sum(~doneEprBool)} machines')     
        self.meps = states[-1]
        if diagnostic is False:
            return self.meps
        else:
            return np.array(eprs), np.array(states), doneBool, doneEprBool
    
    def __ness_estimate(self):
        if self.batch:
            vals, vects = np.linalg.eig(self.R.transpose(0,2,1))
        else:
            vals, vects = np.linalg.eig(self.R.T)

        vects = np.abs(vects)
        vals = np.abs(vals)
        # finds the positive eigenvector with the minimum time derivative
        #min_idx = np.abs(np.array([ self.get_time_deriv(vects[:,i,:]) for i in range(self.S) ])).sum(axis=-1).argmin(axis=0)
        # finds the eigenvector with the minimum magnitude eigenvalue
        min_ix = np.argmin(vals, axis=-1)

        if self.batch:
            ness = vects[range(len(vects)),:, min_idx]
        else:
            ness = vects[:,min_idx].ravel()
        
        return self.normalize_state(ness)
    

    def validate_state(self, state):
        shape = state.shape
        assert shape[-1] == self.S, f'state shape expected to be {self.S}, not {shape[-1]}'

        if self.batch:
            assert len(shape) == 2, f'expected 2D shape, but state is shape {shape}'
            assert shape[0] == self.R.shape[0], 'got {shape[0]} state vectors for {self.R.shape[0]} machines'

        return True

    def normalize_state(self, state):
        assert self.validate_state, 'found invalid state shape'

        if np.all(state.sum(axis=-1)==1):
            return state
        else:
            if self.batch:
                return (state.T/np.sum(state,axis=-1)).T
            else:
                return state/state.sum()



    def get_ness(self, dt0=1, dt_iter=150, dtmin=.001, max_iter=50_000, force_analytic=False, diagnostic=False):
        ness_list=[]
        try:
            ness = self.ness
        except:
            if np.log(self.R.shape[0])*np.sqrt(self.S) < self.analytic_threshhold or force_analytic: 
                try:
                    ness = self.__ness_estimate()
                except:
                    if self.verbose:
                        print('defaulting to numeric solution')
                    try:
                        ness = self.meps
                    except:
                        ness = self.get_uniform()
            else:
                if self.verbose:
                    print('defaulting to numeric solution')
                try:
                    ness = self.meps
                except:
                    ness = self.get_uniform()

        dt = dt0
        i = 0
        negativeBool = False
        doneBool = False

        if self.batch:
            j = -1*np.ones(self.R.shape[0])
        else:
            j = np.array(-1)

        while ( (not np.all(doneBool)) or np.any(negativeBool) )  and i < max_iter:
            if i % dt_iter == 0:
                j += 1 
            dt = dt0 * (2/(2+j))

            ness_list.append(ness)
            if self.batch:
                ness = self.evolve_state(ness,dt[:,None])
            else:
                ness = self.evolve_state(ness,dt)

            negativeBool = np.any(ness < 0, axis=-1)
            if np.any(negativeBool):
                if diagnostic:
                    num_neg = sum(negativeBool)
                    print(f'rewinding {num_neg} states to avoid negative states at iteration {i}')
                ness[negativeBool] = ness_list[-1][negativeBool]
                j[negativeBool] += 1
            
            #doneBool = np.all(np.isclose(0, self.get_time_deriv(ness)))
            doneBool = np.all(np.isclose(0, np.abs(self.get_time_deriv(ness))/np.maximum(1E-12,ness), atol=1E-4), axis=-1)
            #doneBool = np.all(np.isclose(ness,ness_list[-1], rtol=1E-4, atol=1E-16), axis=-1)
            
            i += 1

            if not diagnostic:
                ness_list = ness_list[-1:]
        
        if i == max_iter:
            print(f'ness didnt converge after {i} iterations in {np.sum(~doneBool)} machines')

        self.ness = ness

        if diagnostic:
            return ness, ness_list, doneBool
        else:
            return self.ness
        
    def dkl(self, p, q):
        assert (np.array([p,q])>=0).all, 'all probs must be non-negative'
        p_dom, q_dom = p!=0, q!=0 
        assert np.all(np.equal(p_dom, q_dom)), 'p and z must have the same support'

        p, q = (p.T/np.sum(p, axis=-1)).T, (q.T/np.sum(q, axis=-1)).T
        p[np.where(~p_dom)] = 1
        q[np.where(~q_dom)] = 1
        return np.sum( p*np.log(p/q), axis=-1 )
    

def cyclic_distance(x, i):
    L = x.shape[-1]
    try:
        x_new = x - i
    except:
        x_new = (x.T - i).T
    
    i_d = np.where( abs(x_new) > int(L/2))
    x_new[i_d] = -np.sign(x_new[i_d])*(L-abs(x_new[i_d]))
    return x_new

def involution_builder(N_states, N_swaps=None):
    if N_swaps is None or N_swaps > int(N_states/2) :
        N_swaps = int(N_states/2)

    swaps = np.array(range(N_states))
    np.random.shuffle(swaps)
    swap_list = [ [swaps[2*i],swaps[2*i+1]] for i in range(N_swaps)]
    involution = np.array(range(N_states))
    for i1,i2 in swap_list[:]:
        involution[i1] = i2
        involution[i2] = i1
    return involution


    

