import numpy as np
import pandas as pd

def confidence(gap_csv, universal_fct_tab):
    """
    Compute the probability Pr(universal_fct < gap^2) for a given universal function.
    
    Parameters
    ----------
    gap_csv : str
        The path to the csv file containing the gap of the system.
    universal_fct_tab : numpy array
        The "universal gap" to test the confidence
    Returns
    -------
    confidence : numpy array
       Proba(universal_fct < gap) for u contained in gap_csv
    """

    gap_tab = pd.read_csv(gap_csv).to_numpy()
    nb_pts = len(gap_tab[0])
    N_samples = len(gap_tab)

    if nb_pts != len(universal_fct_tab):
        raise ValueError("The gap and the universal function must have the same number of points")
    
    gap_squared = gap_tab**2
    confidence_tab = np.zeros((N_samples, nb_pts))
    
    for i in range(N_samples):
        confidence_tab[i] = 1/2*(1+np.sign(gap_squared[i] - universal_fct_tab))
        confidence_tab[i][0] = 1 #Because at u=0 the gap has always the same value
    
    confidence_fct = np.mean(confidence_tab, axis = 0)

    return confidence_fct

