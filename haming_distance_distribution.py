import numpy as np
from telecom import get_ising_parameters
from annealing import solve_annealing, modify_coupling_matrix, get_groundstate


def hamming_distance_distribution(N_repeat, N, M, alpha, K, xi, neglection_rule, neglection_thres, gamma, epsilon, which_ctl_fct, nb_pts_gap, nb_pts_time, verbose=True):
    """computes N_repeat times an annealing process with the original and with the modified coupling matrix. Then, the Hamming distance between the 
    groundstates of the final hamiltonian with and without modification is computed.
    
    Returns:
    - d     Hamming distance
    - n     number of occurences of Hamming distance d
    """
    d = np.arange(N+1)
    n = np.zeros(N+1)
    
    for i in range(N_repeat):
        J, b = get_ising_parameters(N, M, alpha, K, xi, False)
        J_n, _ = modify_coupling_matrix(J, neglection_rule, neglection_thres, False)

        _, _, _, _, _, _, _, _, Hscheduled = solve_annealing(J, b, gamma, epsilon, which_ctl_fct, nb_pts_gap, nb_pts_time, False)
        _, _, _, _, _, _, _, _, Hscheduled1 = solve_annealing(J_n, b, gamma, epsilon, which_ctl_fct, nb_pts_gap, nb_pts_time, False)
        
        _, gs_string, gs_array = get_groundstate(Hscheduled, 1)
        _, gs_string1, gs_array1 = get_groundstate(Hscheduled1, 1)
        
        
        d_Hamming = np.sum(np.bitwise_xor(gs_array, gs_array1))
        n[d_Hamming] += 1
        
        if verbose:
            print(f"{i:} gs: {gs_string}     gs1: {gs_string1}   d_Hamming = {d_Hamming}")

    return d, n