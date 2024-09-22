from pennylane import numpy as np
from setup import N1, N2, N3, N, nsteps
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def neighbors(linear_idx, fc):
    neighbors=[]
    dim1, dim2, dim3 = np.unravel_index(linear_idx, (N1,N2,N3)) # indeces of the spin
    if fc == False:
        if dim1 > 0:
            neighbors.append((dim1-1, dim2, dim3))
        if dim1 < N1-1:
            neighbors.append((dim1+1, dim2, dim3))
        if dim2 > 0:
            neighbors.append((dim1, dim2-1, dim3))
        if dim2 < N2-1:
            neighbors.append((dim1, dim2+1, dim3))
        if dim3 > 0:
            neighbors.append((dim1, dim2, dim3-1))
        if dim3 < N3-1:
            neighbors.append((dim1, dim2, dim3+1))
    else:
        for x in range(N1):
            for y in range(N2):
                for z in range(N3):
                    if dim1==x and dim2==y and dim3==z: continue # exclude auto interaction
                    neighbors.append((x,y,z))
    return neighbors
    
def plot_timeprobabilities(times, p, name=None):
    plt.figure(figsize=(15,5))
    plt.title("Ground state probability")
    plt.plot(times, p)
    plt.xlabel("Time")
    plt.ylabel("$P_g(t)$")
    plt.grid()
    if name == None:
        plt.savefig("Media/ProbabilityGround.png")
    else:
        plt.savefig(f"Media/{name}.png")
    plt.close()

def plot_statesprobabilities(state, energies, name=None):
    p = np.square(np.absolute(state))
    plt.figure(figsize=(15,5))
    plt.title("States probabilities")
    plt.bar(energies, p, width=0.4)
    plt.xlabel("Energies")
    plt.ylabel("Probability")
    for E in energies:
        plt.axvline(x=E, ls="--", c="gray")

    if name == None:
        plt.savefig("Media/ProbabilitiesStates.png")
    else:
        plt.savefig(f"Media/{name}.png")
    plt.close()

def animation_states(probs, energies):
    fig=plt.figure(figsize=(15,5))
    plt.title("State evolution in the canonical basis")
    plt.xlabel("State")
    plt.ylabel("Probability")
    plt.ylim([0,1])
    for E in energies:
        plt.axvline(x=E, ls="--", c="gray")
    barcollection = plt.bar(energies, probs[0], width=0.4)
    def animate(i):
        y=probs[i]
        for j, b in enumerate(barcollection):
            b.set_height(y[j])
    anim=animation.FuncAnimation(fig, animate, repeat=False, blit=False, frames=probs.shape[0], interval=500)
    writervideo = animation.FFMpegWriter(fps=5) 
    anim.save("Media/states.mp4", writer=writervideo)
    plt.close()
    HTML(anim.to_jshtml())
