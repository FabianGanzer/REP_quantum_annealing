import numpy as np
import qutip as qt
from itertools import product
import matplotlib.pyplot as plt
from telecom import decimal_to_binary

from gap_tools_ed.specific_fct.annealing_time import annealing_time
from gap_tools_ed.specific_fct.hamiltonian import hamiltonian
from gap_tools_ed.specific_fct.evolve_state import evolve_state
from gap_tools_ed.specific_fct.get_control_function import get_control_function
from gap_tools_ed.specific_fct.gap import gap, spectrum


def solve_annealing(J, b, gamma, epsilon, ctl_fct, nb_pts_gap, nb_pts_time, verbose=True, time_evolution=True):
    """
    Parameters:
    - J             coupling matrix of the Ising Hamiltonian
    - b             magnetic fields of the Ising Hamiltonian
    - gamma         strength of the transverse field
    - epsilon       precision level for the control function (valid for both, linear and optimal scheduling)
    - ctl_fct       0: linear control fct, 1: optimal control fct
    - nb_pts_gap    number of points for the gap computation
    - nb_pts_time   number of points for resolution of the time dependant Schrodinger's equation
    - verbose       print annealing times

    Returns:
    - proba_coef    probability distribution at the different time steps. The ordering is the same as the order of the eigenstates. numpy array, shape (2**N, len(timesTab))
    - sigma_z_exp   expectation value of the sigma_z operator at the different time steps. numpy array, shape (N, len(timesTab))
    - eigenbasis_end    eigenvectors in the final state
    - times_tab     times for the probability distribution
    - tp            times at which the control function is known
    - up            values of the controlfunciton at times tp
    - spectrum_tab  eigenvalues at times tp
    - squared_gap   squared gap between the 0th and the 1st eigenvalue at times tp
    - Hscheduled    qt.QuObj operator of the Hamiltonian scheduled with the selected control function
    """
    
    
    # find the gap for each value of u(t)
    up = np.linspace(1,0,nb_pts_gap) 
    tp_linear = np.linspace(0,1,nb_pts_gap)
    times_tab_linear = np.linspace(0,1,nb_pts_gap)

    Hlinear = hamiltonian(J, b, gamma, times_tab_linear, tp_linear, up)
    spectrum_tab = spectrum(Hlinear, times_tab_linear, progressBar=False)
    gap_tab = spectrum_tab[:, 1] - spectrum_tab[:, 0]
    squared_gap = gap_tab**2


    # control function
    Tint, Tlin = annealing_time(up, gap_tab, epsilon)

    if ctl_fct == 0:
        if verbose:
            print(f"linear scheduling with Tlin = {Tlin:.2f} = 1/(epsilon*DeltaMin**2) = {1/epsilon} * 1/DeltaMin**2   (for reference: Tint = {Tint})")
        tp = Tlin*(1-up)
    elif ctl_fct == 1:
        if verbose:
            print(f"optimal scheduling with Topt = {Tint}   (for reference: Tlin = {Tlin:.2f})")
        tp = get_control_function(up, squared_gap, epsilon)
    else:
        print("[solve_annealing]: invalid value of parameter ctl_fct")
        tp = None


    # compute time evolution using Schrödinger's equation
    times_tab = np.linspace(0, tp[-1], nb_pts_time)
    Hscheduled = hamiltonian(J, b, gamma, times_tab, tp, up)
   
    if time_evolution:
        N = np.shape(J)[0]
        state_list = [(qt.basis(2, 0) + qt.basis(2, 1))/np.sqrt(2)]*N
        psi0 = qt.tensor(state_list)    # |+>^tensor(N) eigenstate of control hamiltonian
        proba_coef, sigma_z_exp, eigenbasis_end = evolve_state(Hscheduled, times_tab, psi0)        # eigenbasis is the eigenbasis of the problem Hamiltonian

    else:
        proba_coef, sigma_z_exp, eigenbasis_end = None, None, None
    
    return proba_coef, sigma_z_exp, eigenbasis_end, times_tab, tp, up, spectrum_tab, squared_gap, Hscheduled


def modify_coupling_matrix(J, neglection_rule, neglection_thres=0.1, verbose=True):
    """
    Parameters:
    - J                     coupling matrix of the Ising Hamiltonian
    - neglection_rule       0: set matrix element of lowest absolute value to zero
                            1: set all matrix elements of absolute value lower than neglection_thres to zero
                            2: set a certain total number of matrix elements to zero starting with the smallest one
                            3: set matrix elements to zero such that each node only has a certain maximum degree. Algorithm starts checking from the smallest matrix element
    - neglection_thres      only used if neglection_rule = 1
    
    Returns:
    - J_n                   coupling matrix with neglected matrix elements
    - where_n               np.where object containing the indices of all neglected matrix elements
    """
    J_n = np.copy(J)

    if verbose:
        print(f"Matrix J before neglection of matrix elements: \n{J}")

    if neglection_rule == 0:
        where_n = np.where(J_n==-np.min(np.abs(J_n[J_n!=0])))       # complicated construction because the matrix elements of J are negative 
    
    elif neglection_rule == 1:
        J_n[J_n==0] = -np.inf
        where_n = np.where(J_n > -neglection_thres)   
        J_n[J_n==-np.inf] = 0        
    
    elif neglection_rule == 2:            # neglecting a certain number of couplings
        sorted_1d_indices = np.argsort(np.abs(J), axis=None)
        sorted_2d_indices = np.unravel_index(sorted_1d_indices, np.shape(J))
        N = np.shape(J)[0]
        i_lower = int(N*(N+1)/2)
        i_upper = i_lower + int(neglection_thres)
        where_n = (sorted_2d_indices[0][i_lower:i_upper], sorted_2d_indices[1][i_lower:i_upper])

    elif neglection_rule == 3:          # for every qubit only allow a maximum number of couplings 
        sorted_1d_indices = np.argsort(np.abs(J), axis=None)
        #sorted_2d_indices = np.unravel_index(sorted_1d_indices, np.shape(J))
        N = np.shape(J)[0]
        where_n1 = []           # first index of the 2d where object indicating which couplings to neglect
        where_n2 = []           # second index of the 2d where object indicating which couplings to neglect
        where_n = (np.array(where_n1, dtype=int), np.array(where_n2, dtype=int))
        J_n = np.copy(J)

        for i in range(int(N*(N-1)/2)):
            j = int(N*(N+1)/2) + i              # 1d index of index of matrix element that should be checked
            discard = degree_too_large(J_n, sorted_1d_indices[j], neglection_thres)
            if discard:
                i1, i2 = np.unravel_index(sorted_1d_indices[j], np.shape(J))
                where_n1.append(i1)
                where_n2.append(i2)
                where_n = (np.array(where_n1, dtype=int), np.array(where_n2, dtype=int))
                J_n[where_n] = 0
            
    else:
        print("unknown neglection rule!")
    
    if verbose:
        print(f"\nmatrixelement(s) set to zero: {J[where_n]}")
    J_n[where_n] = 0

    if verbose:
        print("\nMatrix J after neglection of matrix elements:")
        print(J_n)

    return J_n, where_n


def degree_too_large(J, j, max_degree):
    """checks if a coupling in J is between nodes whose degree is larger than a max degree
    Returns:
    - True      if one of both nodes connected by coupling j has a degree larger than the given max degree
    - False     otherwise
    """
    # calculate degrees
    ix, iy = np.unravel_index(j, np.shape(J))
    degreex = np.sum(adjacency_from_couplings(J)[ix, :])
    degreey = np.sum(adjacency_from_couplings(J)[:, iy])

    # check if a degree is too large
    if degreex > max_degree:
        return True
    if degreey > max_degree:
        return True
    
    return False


def get_groundstate(Hscheduled, t):
    """
    Parameters:
    - Hscheduled        Hamiltonian scheduled with a control function
    - t                 time at which the ground state should be given
    - N                 number of qubits"""
    dim = Hscheduled(0).shape[0]       # dimension of Hilbert space
    N = int(np.log2(dim))           # number of qubits
    states_string = list(product([0,1], repeat = N))
    states_string = ["".join([str(i) for i in state]) for state in states_string]
    
    energy_gs, gs = Hscheduled(t).groundstate()
    idx = np.argmax(np.abs(gs.full()))      # question: why only the maximum?? As I understand it, the ground state is the superposition of all those states
    gs_string = states_string[idx]

    gs_array = np.zeros(N, dtype=int)
    for i in range(len(gs_string)):
        gs_array[i] = int(gs_string[i])
    
    return energy_gs, gs_string, gs_array


def get_state(pattern):
    """returns the qutip.Qobj state that corresponds to the given list 'pattern' of 0 and 1"""
    zero = qt.basis(2, 0)
    one = qt.basis(2, 1)
    psi_list = [zero]*len(pattern)
    for i in range(len(pattern)):
        if pattern[i] == 1:
            psi_list[i] = one
    return qt.tensor(psi_list)


def adjacency_from_couplings(J):
    """returns the adjacency matrix assuming that the graph described by J is symmetric.
    Parameters:
    - J         Couplings of an Ising Hamiltonian (typically only upper right of matrix populated)
    Returns:
    - A         adjacency matrix
    """
    A = np.zeros_like(J, dtype=int)
    A[np.where((J+J.T)!=0)] = 1
    return A

def find_state_index(state, state_set):
    """ Find the index of a state within a given set of states
    Parameters:
    - state     qt.Qobj         (state to find in a set of states)
    - basis     list of qt.Qobj (set in which to find the state)
    """
    for i in range(len(state_set)):
        if state_set[i] == state:
            return i
    return None



def Ising_energy(J, b, basisstate):
    """Evaluates the Ising Hamiltonian for a given BASISSTATE.
    Attention: J[i, j] has to be zero for i>j since the factor 1/2 from the 
    Ising model is not implemented!"""
    sigma = 1 - 2*basisstate
    return -sigma@J@sigma - b@sigma         # factor 1/2 can be omitted because J_n is already a matrix which for i>j is zero (values only in upper right)


def exhaustive_search(J, b):
    """finds the basis state leading to the lowest energy in the Ising Model
    by searching over all possible basis states
    Attention: J[i, j] has to be zero for i>j since the factor 1/2 from the 
    Ising model is not implemented!
    """

    N = np.shape(J)[0]
    
    # find energies
    E = np.zeros(2**N)
    for k in range(2**N):                       # search the whole basis states set
        state_array = decimal_to_binary(k, N)   # 
        E[k] = Ising_energy(J, b, state_array)       

    # find minimum energy states
    indices_E_min = np.where(E==np.min(E))[0]           # avoids usage of np.argmin because np.argmin only returns first occurence of the minimum value. If however there are two states that yield the minimum value equally, only one of them would be detected.

    minimum_energy_states = np.zeros(shape=(len(indices_E_min), N))
    for i in range(len(indices_E_min)):
        minimum_energy_states[i] = decimal_to_binary(indices_E_min[i], N)
    
    return minimum_energy_states



def main():
    """main for testing purposes"""
    from telecom import get_ising_parameters
    import time
    from graph import degree_of_nodes, draw_graph
    #J, b, *_ = get_ising_parameters(5, 4, [1, 1, 0, 0, 0], 100, 0)
    #t0 = time.time()
    #solve_annealing(J, b, 1, 0.1, 1, 20, 30, False, False)
    #t1 = time.time()
    #print(f"{t1-t0} s")

    J = 0.1*np.array([[0, -7, 0, 0], 
                  [0, 0, -10, -9],
                  [0, 0, 0, -6],
                  [0, 0, 0, 0]])
    
    print(f"degrees before neglection: {degree_of_nodes(adjacency_from_couplings(J))}")
    max_degree = 1
    print(degree_too_large(J, j=1, max_degree=max_degree))
    J_n, where_n = modify_coupling_matrix(J, neglection_rule=3, neglection_thres=max_degree)
    print(f"degrees after neglection: {degree_of_nodes(adjacency_from_couplings(J_n))}")
    
    fig = plt.figure()
    ax = fig.add_subplot()
    draw_graph(ax, J_n)
    plt.show()

if __name__ == "__main__":
    main()