# packages
from tqdm import tqdm
from pennylane import numpy as np
import time
import itertools
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
# custom
from setup import N, J, h, timestep, nsteps, times, fully_connected
from Circuits import evolve
from Observables import hamiltonian, levels_nofield
from Utils import plot_timeprobabilities, plot_statesprobabilities, animation_states

running_plot=True

levels, (Eg, ground_states) = levels_nofield(J, h, fc=fully_connected)
energies = [level[0] for level in levels]
print("Ground energy:", Eg, "Degeneracy:", len(ground_states))

# A0s = [1e3]
# B0s = [1e3]
# gammaBs = [7]
# gammaAs = [5]
A0s = [1e3]
B0s = [1e3]
gammaBs = [6,7,8]
gammaAs = [4,5,6]
search_space = list(itertools.product(*[gammaBs, gammaAs, A0s, B0s]))

pg_best = np.zeros(nsteps)
probs_animation_best = np.zeros([nsteps, 2**N])
state_best = None

initial_state=np.ones(2**N)/np.sqrt(2**N)

start = time.time()
for  gammaB, gammaA, A0, B0 in tqdm(search_space, colour="red"):
    
    ##########
    ## CORE ##
    ##########
    # time changing coeffs
    # ising
    A = A0*(1-np.exp(-gammaA*times)) 
    # magnetic field
    B = B0*np.exp(-gammaB*times)
    # A = times
    # B = 1-times
    #state
    state = initial_state
    #store probabilities
    p_ground = np.zeros([nsteps])
    probs_animation = np.zeros([nsteps, 2**N])
    for step in tqdm(range(nsteps), colour="green", total=nsteps-1):
        # hamiltonian
        H = hamiltonian(J, h, A[step].item(), B[step].item(), fc=fully_connected)

        if state is not None:
            for g in ground_states:
                p_ground[step] = np.square(np.absolute(np.vdot(g, state).item())).item()
        probs_animation[step,:] = np.square(np.absolute(state))

        ## TROTTERIZATION
        if step == nsteps-1: break
        # update state
        state = evolve(H=H, t=timestep, init=state) 

        ## RUNNING PLOT
        if running_plot and (step+1)%int(nsteps/10) == 0:
            plot_timeprobabilities(range(step), p_ground[:step])


    ### SAVE BEST
    if p_ground[-1] > 1/2**N and p_ground[-1] > pg_best[-1]:
        pg_best = p_ground
        state_best = state
        probs_animation_best = probs_animation
        tqdm.write(f"Best configuration: {[A0, B0, gammaA, gammaB]} with P_g={np.round(p_ground[-1],4)}")
        plot_timeprobabilities(times, p_ground, name=f"ProbabilityGround_N={N}_A0={A0}_B0={B0}_gammaA={gammaA}_gammaB={gammaB}")


end = time.time()
print(f"Time elapsed: {end-start} seconds")

if state_best is not None:
    plot_timeprobabilities(times, pg_best, name="BEST_ProbabilityGround")
    plot_statesprobabilities(state_best, energies, name="BEST_StatesProbabilities") 
    animation_states(probs_animation_best[::int(nsteps/100)], energies)


