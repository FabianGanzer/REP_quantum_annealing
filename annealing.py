import numpy as np
import qutip as qt
from itertools import product

from gap_tools_ed.specific_fct.annealing_time import annealing_time
from gap_tools_ed.specific_fct.hamiltonian import hamiltonian
from gap_tools_ed.specific_fct.evolve_state import evolve_state
from gap_tools_ed.specific_fct.get_control_function import get_control_function
from gap_tools_ed.specific_fct.gap import gap, spectrum


def solve_annealing(J, b, gamma, epsilon, ctl_fct, nb_pts_gap, nb_pts_time, verbose=True):
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
    N = np.shape(J)[0]
    state_list = [(qt.basis(2, 0) + qt.basis(2, 1))/np.sqrt(2)]*N
    psi0 = qt.tensor(state_list)    # |+>^tensor(N) eigenstate of control hamiltonian

    times_tab = np.linspace(0, tp[-1], nb_pts_time)
    Hscheduled = hamiltonian(J, b, gamma, times_tab, tp, up)
   
    proba_coef, sigma_z_exp, eigenbasis_end = evolve_state(Hscheduled, times_tab, psi0)        # eigenbasis is the eigenbasis of the problem Hamiltonian
    
    return proba_coef, sigma_z_exp, eigenbasis_end, times_tab, tp, up, spectrum_tab, squared_gap, Hscheduled


def modify_coupling_matrix(J, neglection_rule, neglection_thres=0.1, verbose=True):
    """
    Parameters:
    - J                     coupling matrix of the Ising Hamiltonian
    - neglection_rule       0: set matrix element of lowest absolute value to zero
                            1: set all matrix elements of absolute value lower than neglection_thres to zero
    - neglection_thres      only used if neglection_rule = 1
    
    Returns:
    - J_n                   coupling matrix with neglected matrix elements
    - where_n               np.where object containing the indices of all neglected matrix elements
    """
    J_n = np.copy(J)
    if neglection_rule == 0:
        where_n = np.where(J_n==-np.min(np.abs(J_n[J_n!=0])))       # complicated construction because the matrix elements of J are negative 
    elif neglection_rule == 1:
        J_n[J_n==0] = -np.inf
        where_n = np.where(J_n > -neglection_thres)   
        J_n[J_n==-np.inf] = 0        
    else:
        print("unknown neglection rule!")
    if verbose:
        print(f"\nmatrxelement(s) set to zero: {J_n[where_n]}")
    J_n[where_n] = 0

    if verbose:
        print("\nMatrix J after neglection of matrix elements:")
        print(J_n)

    return J_n, where_n


def get_groundstate(Hscheduled, t):
    """
    Parameters:
    - Hscheduled        Hamiltonian scheduled with a control function
    - t                 time at which the ground state should be given
    - N                 number of qubits"""
    dim = Hscheduled(0).shape[0]       # dimension of Hilber space
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


def main():
    print(get_state([0, 0, 1]))


if __name__ == "__main__":
    main()