"""set of functions tracking the haming distance between the result of an 
annealing process and the actual activity pattern while sweeping the 
neglection parameter (thresold, number or max. degree of nodes)"""

import numpy as np
from scipy.stats import uniform_direction
from telecom import get_ising_parameters, CCR, generate_signal
from annealing import solve_annealing, modify_coupling_matrix, get_groundstate, adjacency_from_couplings, exhaustive_search
from graph import is_connected

def hamming_distance(array0, array1):
    """compute the hamming distance between two arrays"""
    return np.sum(np.bitwise_xor(array0, array1))


def hamming_distance_distribution_CCR(N_repeat, N, M, alpha, K, xi, thres_CCR, verbose=True):
    """computes N_repeat times a problem instance and determines the activity pattern using a CCR.
        Computes the hamming distance between the actual and the determined activity pattern

    Returns:
    - d         hamming distance
    - n         number of occurences of hamming distance"""

    d = np.arange(N+1)
    n = np.zeros(N+1)

    alpha = np.array(alpha, dtype=int)
    
    for i in range(N_repeat):
        P = uniform_direction.rvs(dim=M, size=N).transpose()
        Y = generate_signal(alpha, P, xi, K)
        alpha_CCR, _ = CCR(Y, P, thres_CCR)
        
        d_Hamming = hamming_distance(alpha, alpha_CCR)  # np.sum(np.bitwise_xor(gs_array, gs_array1))
        n[d_Hamming] += 1
        
        if verbose:
            print(f"{i:} alpha: {alpha}     CCR estimation: {alpha_CCR}   d_Hamming = {d_Hamming}")

    return d, n


def hamming_distance_distribution(N_repeat, N, M, alpha, K, xi, neglection_rule, neglection_thres, gamma, epsilon, which_ctl_fct, nb_pts_gap, nb_pts_time, verbose=True):
    """computes N_repeat times an annealing process with the original and with the modified coupling matrix. Then, the Hamming distance between the 
    groundstates of the final hamiltonian with and without modification is computed.
    
    Returns:
    - d                 Hamming distance
    - n                 number of occurences of Hamming distance d
    - N_n               number of neglected matrix elements in J averaged over all N_repeat runs of the annealing process
    - connected_counter count of the number of instances in which the graph with neglected matrix elements is connected 
    """

    d = np.arange(N+1)
    n = np.zeros(N+1)
    N_n= np.zeros(N_repeat)             # number of neglected matrix elements
    connected_counter = 0

    for i in range(N_repeat):
        J, b, *_ = get_ising_parameters(N, M, alpha, K, xi, False)
        J_n, where_n = modify_coupling_matrix(J, neglection_rule, neglection_thres, False)
        N_n[i] = len(where_n[0])
        connected_counter += int(is_connected(adjacency_from_couplings(J_n)))

        _, _, _, _, _, _, _, _, Hscheduled = solve_annealing(J, b, gamma, epsilon, which_ctl_fct, nb_pts_gap, nb_pts_time, verbose=False, time_evolution=False)
        _, _, _, _, _, _, _, _, Hscheduled1 = solve_annealing(J_n, b, gamma, epsilon, which_ctl_fct, nb_pts_gap, nb_pts_time, verbose=False, time_evolution=False)
        
        _, gs_string, gs_array = get_groundstate(Hscheduled, 1)
        _, gs_string1, gs_array1 = get_groundstate(Hscheduled1, 1)
        
        
        d_Hamming = hamming_distance(gs_array, gs_array1)  # np.sum(np.bitwise_xor(gs_array, gs_array1))
        n[d_Hamming] += 1
        
        if verbose:
            print(f"{i:} gs: {gs_string}     gs1: {gs_string1}   d_Hamming = {d_Hamming}")

    N_n = np.mean(N_n)

    return d, n, N_n, connected_counter


def hamming_distance_distribution_exhaustive(N_repeat, N, M, alpha, K, xi, neglection_rule, neglection_thres, verbose=True):
    """computes N_repeat times the lowest energy state by exhaustive search. Then, the Hamming distance between the 
    groundstates of the final hamiltonian with and without modification is computed.
    
    Returns:
    - d                 Hamming distance
    - n                 number of occurences of Hamming distance d
    - N_n               number of neglected matrix elements in J averaged over all N_repeat runs of the annealing process
    - connected_counter count of the number of instances in which the graph with neglected matrix elements is connected 
    """

    d = np.arange(N+1)
    n = np.zeros(N+1)
    N_n= np.zeros(N_repeat)             # number of neglected matrix elements
    connected_counter = 0

    for i in range(N_repeat):
        J, b, *_ = get_ising_parameters(N, M, alpha, K, xi, False)
        J_n, where_n = modify_coupling_matrix(J, neglection_rule, neglection_thres, False)
        N_n[i] = len(where_n[0])
        connected_counter += int(is_connected(adjacency_from_couplings(J_n)))

        minimum_energy_states = exhaustive_search(J_n, b)
        gs_array = minimum_energy_states[0]

        #print(gs_array, type(gs_array), alpha, type(alpha))
        
        d_Hamming = hamming_distance(np.array(gs_array, dtype=int), np.array(alpha, dtype=int))  # np.sum(np.bitwise_xor(gs_array, gs_array1))
        n[d_Hamming] += 1
        
        if verbose:
            print(f"{i:} gs: {gs_array}     alpha: {alpha}   d_Hamming = {d_Hamming}")

    N_n = np.mean(N_n)

    return d, n, N_n, connected_counter




def main():
    """main for test purposes"""
    # --------- Parameters ---------
    N_repeat = 100      # how many samples per neglection thresold

    N = 5               # number of users
    M = 4               # length of id-sequence for every user
    K = 100              # number of antennas
    xi = 0              # std of thermal noise

    which_ctl_fct = 0   # 0: linear control function, 1: optimal control function
    neglection_thres = 0.2

    nb_pts_gap = 20     # number of points for the gap computation
    nb_pts_time = 30    # number of points for resolution of the time dependant Schrodinger's equation
    epsilon = 0.1       # precision level for the control function (valid for both, linear and optimal scheduling)
    gamma = 1           # strength of the transverse field, irrelevant for us 

    # ------------- Program ----------------
    import time


    # activity pattern
    alpha = np.zeros(N)
    alpha[0] = 1
    alpha[2] = 1

    # runtime estimation
    t0 = time.time()
    distr = hamming_distance_distribution(N_repeat, N, M, alpha, K, xi, 1, neglection_thres, gamma, epsilon, which_ctl_fct, nb_pts_gap, nb_pts_time, False)
    t1 = time.time()
    print(f"runtime of hamming_distance_distribution: {t1-t0:.3f}s")

if __name__ == "__main__":
    main()