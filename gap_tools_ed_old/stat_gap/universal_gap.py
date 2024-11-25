import numpy as np
import pandas as pd

def moment_gap(gap_csv, n):
    """
    Compute the n-th momentum of the mean gap of the system.
    
    Parameters
    ----------
    gap_csv : str
        The path to the csv file containing the gap of the system.
    n : int
        The moment to compute. (the mean gap corresponds to n = 1)
    Returns
    -------
    mean_gap : numpy array
        The n-th momentum of the mean gap of the system.
    """
    gap = pd.read_csv(gap_csv).to_numpy()
    gap_n = gap**n
    moment_estimator = np.mean(gap_n, axis=0)
    
    return moment_estimator

def quantile(gap_csv, alpha):
    """
    Compute the alpha-quantile of the gap of the system.

    Parameters
    ----------
    gap_csv : str
        The path to the csv file containing the gap of the system.
    alpha : float
        The quantile to compute.
    Returns
    -------
    quantile : numpy array
        The alpha-quantile of the gap of the system.
    """
    gap = pd.read_csv(gap_csv).to_numpy()
    quantile_estimator = np.quantile(gap, alpha, axis=0)

    return quantile_estimator