import qutip as qt
import numpy as np

def final_distrib(psi):
    """
    Compute the final distribution of the state psi.

    Parameters
    ----------
    psi : qutip.Qobj
        The state of the system.
    Returns
    -------
    numpy.ndarray
        The distribution of the state psi over the computational basis
    """
    N = int(np.log2(len(psi.full())))
    coef_tab = psi.full().reshape(2**N)
    for i in range(2**N):
        coef_tab[i] = np.real(np.abs(coef_tab[i])**2)
    
    if np.abs(np.sum(coef_tab) - 1) > 1e-5 :
        print("Error : the sum of the coefficients is not 1")

    return coef_tab