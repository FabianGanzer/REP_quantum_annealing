import numpy as np

def annealing_time(uTab, gapTab, epsilon):
    """
    Compute the annealing time given the gap of the system at various times.

    Parameters
    ----------
    uTab : numpy array
        Discretization of the control function
    gapTab : numpy array
        The gap of the system at control parameters u contained in uTab.
    epsilon : float
        The target precision.

    Returns
    -------
    Tint : float
        The annealing time evaluated with an integral (non linear control function).
    Tlin : float
        The annealing time required for a linear control function.
    """

    gapMin = np.min(gapTab)

    if gapMin < 1e-10:
        print("The gap vanishes at some point. The annealing time is infinite.")
        return np.inf, np.inf

    Tlin = 1/epsilon * 1/gapMin**2

    Tint = -1/epsilon * np.trapz(1/gapTab**2, uTab)

    return Tint, Tlin