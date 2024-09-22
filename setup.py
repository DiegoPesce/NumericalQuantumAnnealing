from pennylane import numpy as np
import pennylane as qml

np.random.seed(1811)

## TIME EVOLUTION
nsteps=100000 # high -> smoother decrease, low -> stepwise decrease
maxtime=1
times, timestep = np.linspace(0, 1, num=nsteps, retstep=True)

## SYSTEM
N1 = 5
N2 = 1
N3 = 1
N = N1*N2*N3
fully_connected = True
# fixed random interaction
J = np.random.normal(0, 1, [N,N])
J = (J + J.T)/2
# longitudinal magnetic field (remove degeneration)
h = np.random.normal(0, 1, [N])


## SIMULATOR
dev = qml.device("default.qubit", wires=N)
niter=3