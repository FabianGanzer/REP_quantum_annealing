import numpy as np
import qutip as qt

def evolve_state(H, psi0):
    """
    Compute the state of the system at different times.

    Parameters
    ----------
    H : qutip.qobj
        The Hamiltonian of the system.

    psi0 : qutip.qobj
        The initial state of the system.

    Returns
    -------
    overlapTab : numpy array
        The overlap between the current state and the targeted state at different times.

    """
    N = int(np.log2(len(psi0.full())))

    # Setup operators for individual qubits
    sx_list, sy_list, sz_list = [], [], []
    for i in range(N):
        op_list = [qt.qeye(2)] * N
        op_list[i] = qt.sigmax()
        sx_list.append(qt.tensor(op_list))
        op_list[i] = qt.sigmay()
        sy_list.append(qt.tensor(op_list))
        op_list[i] = qt.sigmaz()
        sz_list.append(qt.tensor(op_list))
    
    #times at which the state is computed
    timesTab = H.tlist
    #Problem Hamiltonian
    H0 = H(timesTab[-1])
    #Eigenbasis of the problem Hamiltonian
    projectors_list = []
    states_list = []
    for i in range(2**N):
        eigenstate = H0.eigenstates()[1][i]
        states_list.append(eigenstate)
        projectors_list.append(eigenstate.proj())

    exp_values_list = projectors_list + sz_list


    try:
        result = qt.sesolve(H, psi0, timesTab, exp_values_list, options=qt.Options(nsteps=20000))

        expectation_values = result.expect
        proba_coef = expectation_values[0:2**N]
        sigma_z_exp = expectation_values[2**N:]

    except:
        print("Error : QuTip has not converged")
        proba_coef = [np.nan*np.zeros(len(timesTab)) for i in range(2**N)]
        sigma_z_exp = [np.nan*np.zeros(len(timesTab)) for i in range(N)]
    
    return proba_coef, sigma_z_exp, states_list