import numpy as np
import qutip as qt

def hamiltonian(J,h,gamma,timesTab,tp,up):
    """
    Compute the Hamiltonian of the system.

    Parameters
    ----------
    J : numpy array
        The coupling matrix.
    h : numpy array
        The local fields.
    gamma : float
        Stenght of the transverse field.
    timesTab : numpy array
        Discretization of the annealing period [0,T]
    tp : numpy array
        Points where the control function is known
    uTab : numpy array
        Values of the control function at points tp

    Returns
    -------
    H : qutip.qobj
        The Hamiltonian of the system for a given value u of the control function

    """
    N = len(h)
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
    
    #Problem Hamiltonian
    H0 = 0
    for i in range(N):
        H0 -=  h[i] * sz_list[i]
    for i in range(N):
        for j in range(N):
            H0 -=  J[i,j] * sz_list[i] * sz_list[j]
            
    #Control Hamiltonian
    H1 = 0
    for i in range(N):
        H1 -= sx_list[i]
    H1 = gamma * H1
    
    #time-dependant coefficients
    def A(t,args):
        return 1 - np.interp(t,tp,up) #interpolation of the control function

    def B(t,args):
        return np.interp(t,tp,up) #interpolation of the control function

    #Hamiltonian
    H = qt.QobjEvo([[H0,A], [H1,B]],tlist=timesTab)

    return H

    