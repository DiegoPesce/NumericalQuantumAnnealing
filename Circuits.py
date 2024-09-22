import pennylane as qml
from setup import N, dev, niter

@qml.qnode(dev)
def evolve(H, t, init):
    # set the initial state
    if init is not None:
        qml.QubitStateVector(init, wires=list(range(N)))
    
    # apply trotterization  
    qml.adjoint(qml.TrotterProduct(hamiltonian=H, time=t, n=niter, order=2))
    return qml.state()
