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


class ContinuousTimeMarkovChain():
    def __init__(self, R=None, generator=uniform_generator, **gen_kwargs):
        self.scale = 1
        self.batch = False
        self.time_even_states = True
        self.analytic_threshhold = 65
        if R is not None:
            self.set_rate_matrix(R)
        else:
            self.generator = generator
            self.gen_kwargs = gen_kwargs
            self.set_rate_matrix(self.generator(**self.gen_kwargs))     
    
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
        
        S, shape, size = self.__R_info(R)
        
        assert len(shape) == 2 and size == S**2, 'each R must be a 2D square matrix'
        assert ( np.sign(R-R*np.identity(S)) == np.ones(shape)-np.identity(S)).all(), 'R_ij must be + for i!=j'

        self.S = S

        if not (R.sum(axis=-1) == np.zeros(self.S)).all():
            R = self.__set_diags(R)

        return R
    
    def __normalize_R(self, R, max_val):
        if self.batch:
            R_max = np.abs(R).max(axis=-1).max(axis=-1)
        else:
            R_max = np.abs(R).max()

        self.scale = R_max/ (np.minimum(R_max,max_val))

        if self.batch:
            R = (R.T / self.scale).T
        else:
            R = R/self.scale

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

        return


    def set_rate_matrix(self, R, max_rate=1):
        R = self.verify_rate_matrix(R)

        if self.scale is not None:
            R = self.__normalize_R(R, max_rate)

        self.R = R
        if self.time_even_states is True:
            self.rev_R = self.R
        if self.time_even_states is False:
            self.rev_R = self.get_reversal_matrix() @ self.R @ self.get_reversal_matrix().T

        self.__set_statewise_Q()

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
        return state + dt*self.get_time_deriv(state)
    
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

    def get_meps(self, dt0=.5 , state=None, max_iter=500, dt_iter=5, diagnostic=False):
        if state is None:
            try:
                state = self.ness
            except:
                state = self.get_uniform()

        eprs = []
        states = [state]
        doneBool = np.array(False)
        negativeBool = False
        i=0
        j=-1
        while (not np.all(doneBool) or np.any(negativeBool)) and i < max_iter:
            if i % dt_iter == 0:
                j += 1 
            dt = dt0 * (2/(2+j))
            epr = self.get_epr(state)
            eprs.append(epr)
            if self.batch:
                new_state = (state + dt*(self.get_time_deriv(state) + state*(epr-self.get_statewise_epr(state).T).T))
                new_state[doneBool] = state[doneBool]  
            else:
                new_state = state + dt * (self.get_time_deriv(state) + state*(epr - self.get_statewise_epr(state)))
            
            negativeBool = np.any(new_state < 0, axis=-1)
            if np.any(negativeBool):
                new_state[negativeBool] = state[negativeBool]
                j += 1
                if diagnostic:
                    print(f'dt changed at iteration{i}')
            
            states.append(new_state)
            state = new_state
            doneBool = np.all(np.isclose(states[-2],states[-1], atol=1E-6), axis=-1)
            i += 1
        if i == max_iter:
            print(f'meps didnt converge after {i} iterations in {np.sum(~doneBool)} machines')    
        self.meps = states[-1]
        if diagnostic is False:
            return self.meps
        else:
            return np.array(eprs), np.array(states), doneBool
    
    def __ness_estimate(self):
        if self.batch:
            vals, vects = np.linalg.eig(self.R.transpose(0,2,1))
        else:
            vals, vects = np.linalg.eig(self.R.T)
        zero_vals = np.isclose(vals, 0)

        assert np.all(np.sum(zero_vals, axis=-1)) == 1, 'found none or more than one zero eigenval when finding the NESS'

        if self.batch:
            ness = np.abs(vects[np.where(zero_vals)[0],:,np.where(zero_vals)[-1]])
            return (ness.T/np.sum(ness,axis=-1)).T
        else:
            ness = np.abs( vects[:,np.where(zero_vals)] ).ravel()
            return ness/ness.sum()

    def get_ness(self, dt=.1, max_iter=500, force_analytic=False):
        try:
            return self.ness
        except:
            ness_list=[]

        if np.log(self.R.shape[0])*np.sqrt(self.S) < self.analytic_threshhold or force_analytic: 
            try:
                ness = self.__ness_estimate()
            except:
                print('defaulting to numeric solution')
                ness = self.get_uniform()
        else:
            print('defaulting to numeric solution')
            ness = self.get_uniform()

        i=0
        while not np.all(np.isclose(self.get_time_deriv(ness),0, rtol=1E-5)) and i < max_iter:
            ness_list.append(ness)
            ness = self.evolve_state(ness,dt)
            i += 1
        if i == max_iter:
            print(f'ness didnt converge after {i} iterations')    
        self.ness = ness
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


    

