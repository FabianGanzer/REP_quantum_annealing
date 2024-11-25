import numpy as np
import qutip as qt
from tqdm.notebook import tqdm


def spectrum(H, timesTab, progressBar = True):
    """
    Compute the spectrum of the system.

    Parameters
    ----------
    H : qutip.qobj
        The Hamiltonian of the system.
    timesTab : numpy array
        Times at which we know the Hamiltonian
    progressBar : bool, optional
        If True, display a progress bar. The default is True.


    Returns
    -------
    spectrum : numpy array
        The spectrum of the system at control parameters u contained in timesTab

    """
    spectrum = np.zeros((len(timesTab),len(H(0).eigenenergies())))
    for i in tqdm(range(len(timesTab)),disable = not progressBar):
        spectrum[i] = H(timesTab[i]).eigenenergies()

    return spectrum

def gap(H, timesTab, progressBar = True):
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

    gap = np.zeros(len(timesTab))
    for i in tqdm(range(len(timesTab)),disable = not progressBar):
        energiesTab = H(timesTab[i]).eigenenergies()
        gap[i] = energiesTab[1] - energiesTab[0]

    return gap

