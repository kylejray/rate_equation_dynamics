from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()  # number of MPI prdocs
rank = comm.Get_rank()  # i.d. for local proc

import sys, os

#change the path to the directory that ctmc.py is located in on your system
sys.path.append(os.path.expanduser('~/source/discrete_states/'))

from ctmc import ContinuousTimeMarkovChain as MC
from ctmc import normal_generator, gamma_generator, uniform_generator, cyclic_generator, detailed_balance_generator, arrhenius_pump_generator

import numpy as np
from datetime import datetime

sys.path.append(os.path.expanduser('~/source/'))
import kyle_tools as kt

base_dir = ''
strength = 15

if rank == 0:
    base_dir = kt.generate_save_dir()+f'strength{strength:02}/'

base_dir = comm.bcast(base_dir, root=0)
rank_dir = base_dir + f'ratio{rank+1}/'
os.makedirs(rank_dir, exist_ok=True)


s_vals = [5, 10, 25, 50, 100, 250, 500, 1_000, 2_000]
s_machines = [1000,500,300,300,100,100,50,40,30]

ratio = 1/(rank+1)

name = int(datetime.now().timestamp())

output = [ [] for i in range(len(s_vals)) ]

for i,[s,n] in enumerate(zip(s_vals,s_machines)):


    print(f'rank {rank} at n,s : {n,s}', flush=True)
    num_pump = int(ratio * (s**2-s) )
    machine = MC(generator=arrhenius_pump_generator, S=s, N=n, n_pumps=num_pump, pump_strength=strength, gen_args=[0,1])
    machine.verbose = False
    ness = machine.get_ness()
    meps = machine.get_meps()
    unif = machine.get_uniform()
    np.savez(rank_dir+f'S{s:05}.npz', R=machine.R, ness = ness, meps = meps, unif = unif)

    for j, item in enumerate([meps, ness, unif]):
        output[i].append(machine.get_epr(item))

save_dict = {f"{s:05}":d for s,d in zip(s_vals,output)}

np.savez(base_dir+'{summary_data}.npz', s_values = s_vals, s_trials=s_machines, **save_dict)
