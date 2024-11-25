import numpy as np

def get_control_function(uTab, gap_squared, epsilon):
    """
    Compute the control function given the gap of the system at various times.

    Parameters
    ----------
    uTab : numpy array
        Discretization of the control function
    gap_squared : numpy array
        The gap Delta^2 of the system at control parameters u contained in uTab.
    epsilon : float
        The target precision.

    Returns
    -------
    timesTab : numpy array
        The times such that u(t[i]) = uTab[i] for i in range(len(tTab)).

    """

    gapMin = np.min(gap_squared)
    nbPts = len(uTab)
    timesTab = np.zeros(nbPts)


    if gapMin < 1e-10:
        print("The gap vanishes at some point. The annealing time is infinite.")
        return np.nan*timesTab

    timesTab[0] = 0

    for i in range(1,nbPts):
        timesTab[i] = timesTab[i-1] + 1/epsilon * (uTab[i-1]-uTab[i])/gap_squared[i]
  
    return timesTab
    
    

   