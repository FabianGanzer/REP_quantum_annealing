import numpy as np
import matplotlib.pyplot as plt
from telecom import generate_signal, get_couplings, CCR, signal_norm
from scipy.stats import uniform_direction


def main():
    # --------- Parameters ---------
    N = 5               # number of users
    M = 7               # length of id-sequence for every user
    K = 10              # number of antennas
    xi = 0              # std of thermal noise

    T = 0.6             # Thresold for the CCR

    width_plt = 5
    height_plt = width_plt

    # ----------- Program -----------
    # activity pattern
    alpha = [1, 0, 0, 0, 0]
    print(f"activity pattern: {alpha}")
    alpha = np.array(alpha)
    if len(alpha) != N:
        print(f"ERROR: wrong initial activity pattern! Length must be N = {N} but is {len(alpha)}.")
    
    
    P = uniform_direction.rvs(dim=M, size=N).transpose()
    Y = generate_signal(alpha, P, xi, K)
    J, b = get_couplings(Y, P)
    
    print(f"N = {N}, M = {M}, K = {K}")
    print(f"verification of normalization of pilots: {signal_norm(P)}")
    print(f"np.shape(P) = {np.shape(P)}")
    print(f"np.shape(Y) = {np.shape(Y)}")        


    # conventional correlation receiver (CCR)
    alpha_CCR, Y_normed = CCR(Y, P, T, True)
    

    # -------------- Plotting ---------------

    N_row = 2
    N_col = 2
    N_plt = N_row*N_col
    fig = plt.figure(figsize=((N_col*width_plt, N_row*height_plt)))
    ax = []
    for i in range(N_plt):
        ax.append(fig.add_subplot(N_row, N_col, i+1))

    # unnormalized signals
    for i in range(K):
        ax[0].plot(np.arange(M), Y[:, i])
    ax[0].set_ylabel(r"$y_{i}(t)$")
    ax[0].set_title("received signals")

    # pilots
    for i in range(N):
        ax[1].plot(np.arange(M), P[:, i], label=i)
    ax[1].set_ylabel(r"$p_i(t)$")
    ax[1].legend()
    ax[1].set_title("pilots")

    # normalized signals
    for i in range(N):
        ax[2].plot(np.arange(M), Y_normed[:, i])
    ax[2].set_ylabel(r"$\tilde{y}_i(t)$")
    ax[2].set_title("normalized signals")

    for i in range(N_plt):
        ax[i].set_xlabel(r"$t$")
        ax[i].grid()
    plt.show()
    



if __name__ == "__main__":
    main()