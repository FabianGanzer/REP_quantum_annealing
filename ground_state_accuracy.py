import numpy as np
from haming_distance_distribution import hamming_distance_distribution
import matplotlib.pyplot as plt
from telecom import get_ising_parameters, generate_signal, get_couplings
from scipy.stats import uniform_direction

def main():
    # --------- Parameters ---------
    N_repeat = 20

    N = 5               # number of users
    M = 4               # length of id-sequence for every user
    K = 100              # number of antennas
    xi = 0              # std of thermal noise

    T = 0.01             # Thresold for the CCR

    which_ctl_fct = 0   # 0: linear control function, 1: optimal control function
    N_thres = 30        # how many different neglection thresolds between 0 and 1

    nb_pts_gap = 20     # number of points for the gap computation
    nb_pts_time = 30    # number of points for resolution of the time dependant Schrodinger's equation
    epsilon = 0.1       # precision level for the control function (valid for both, linear and optimal scheduling)
    gamma = 1           # strength of the transverse field, irrelevant for us 


    # ----------- Program -----------
    # activity pattern
    alpha = np.zeros(N)
    alpha[0] = 1
    print(f"activity pattern: {alpha}")
    
    
    P = uniform_direction.rvs(dim=M, size=N).transpose()
    print(np.sqrt(np.sum(P**2, axis=0)))
    Y = generate_signal(alpha, P, xi, K)
    J, b = get_couplings(Y, P)
    
    print(np.shape(Y))
    print(np.shape(P))
    
    # convenional correlation receiver (CCR)
    alpha_CCR = np.zeros(N)
    for n in range(N):
        f = np.zeros(K)
        for k in range(K):
            f[k] = np.sum(Y[:, k]*P[:, n])
        f_mean = np.mean(f)                     # average over all antennas
        print(f_mean)
        if np.abs(f_mean) >= T:
            alpha_CCR[n] = 1
    
    print(alpha_CCR)
    

    



if __name__ == "__main__":
    main()