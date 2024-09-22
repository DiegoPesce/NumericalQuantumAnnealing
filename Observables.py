import pennylane as qml
from pennylane import numpy as np
import itertools
from Utils import neighbors
from setup import N1, N2, N3, N


def hamiltonian(J, Bz, A, Bx, fc=False):
    h0 = 0*qml.I(0)
    h1 = 0*qml.I(0)
    for s in range(N):
        # ising
        neigh = neighbors(s, fc)
        for ndim1, ndim2, ndim3 in neigh:
            ns = np.ravel_multi_index((ndim1,ndim2,ndim3), (N1,N2,N3))
            h0 += -A*J[s,ns]*qml.Z(s)@qml.Z(ns)
        # magnetic field     
        h0 += -Bz[s]*qml.Z(wires=[s]) 
        h1 += -Bx*qml.X(wires=[s])

    # convert in single hamiltonian
    H = h0 + h1
    return H

def energy_nofield(config, J, Bz, fc=False):
    E = 0
    for s in range(N):
        neigh = neighbors(s, fc)
        for nx, ny, nz in neigh:
            ns = np.ravel_multi_index((nx, ny, nz), (N1,N2,N3))
            E += -J[s,ns]*config[s]*config[ns]
        E += -Bz[s]*config[s]
    return E

def levels_nofield(J, Bz, fc=False):
    configurations = itertools.product([1,-1], repeat=N)
    Eg = np.inf
    all_states = []
    for idx, cfg in enumerate(configurations):
        # create state and energy
        E = energy_nofield(cfg, J, Bz, fc)
        state = np.zeros(2**N, dtype=complex) 
        state[idx] = 1 
        # save it
        all_states.append((E,state))

        # save ground states
        if E < Eg:
            states_ground = [] # clean ground states
            Eg = E # set new ground level
            states_ground.append(state) # insert state
        elif E == Eg:
            states_ground.append(state) # insert state

    return all_states, (Eg, states_ground)