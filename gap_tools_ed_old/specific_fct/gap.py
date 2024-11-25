import numpy as np
import qutip as qt
from tqdm.notebook import tqdm


def spectrum(H, progressBar = True):
    """
    Compute the spectrum of the system.
    
    Parameters
    ----------
    H : qutip.qobj
        The Hamiltonian of the system.
    progressBar : bool, optional
        If True, display a progress bar. The default is True.
    Returns
    -------
    spectrum : numpy array
        The spectrum of the system at control parameters u contained in H.tlist .

    """
    spectrum = np.zeros((len(H.tlist),len(H(0).eigenenergies())))
    timesTab = H.tlist
    for i in tqdm(range(len(timesTab)),disable = not progressBar):
        spectrum[i] = H(timesTab[i]).eigenenergies()
    
    return spectrum

def gap(H, progressBar = True):
    """
    Compute the gap of the system.
    
    Parameters
    ----------
    H : qutip.qobj
        The Hamiltonian of the system.
    progressBar : bool, optional
        If True, display a progress bar. The default is True.
    Returns
    -------
    gap : numpy array
        The gap of the system at control parameters u contained in H.tlist .

    """

    gap = np.zeros(len(H.tlist))
    timesTab = H.tlist
    for i in tqdm(range(len(timesTab)),disable = not progressBar):
        energiesTab = H(timesTab[i]).eigenenergies()
        gap[i] = energiesTab[1] - energiesTab[0]
    
    return gap

