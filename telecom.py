import numpy as np
from scipy.stats import uniform_direction
from itertools import product

def generate_signal(alpha, P, xi, K):
    """
    Generates the signal matrix Y given the required parameters.

    Parameters
    ----------
    alpha : numpy array
        Activity pattern
    P : numpy array
        Pilot matrix
    xi : float
        Level of noise at the AP
    K : int
        Number of antennas

    Returns
    -------
    Y : numpy array
        Signal matrix
    """

    M, N = P.shape
    Z = np.random.normal(0, xi, (M,K))
    H_tilde = np.random.normal(0, 1, (N,K))

    alpha_diag = np.diag(alpha)
    signal = P @ alpha_diag @ H_tilde + Z

    return signal

def get_couplings(Y, P):
    """
    Computes the couplings J and b for the Ising model.

    Parameters
    ----------
    Y : numpy array
        Signal matrix
    P : numpy array
        Pilot matrix

    Returns
    -------
    J : numpy array
        Couplings
    b : numpy array
        Local fields
    """

    M,K = Y.shape
    N = P.shape[1]

    J = np.zeros((N,N))
    b = np.zeros(N)

    projectors_list = np.zeros((N,M,M))
    for i in range(N):
        p_vector = np.matrix(P[:,i]).T
        projectors_list[i] = p_vector @ p_vector.T

    sampled_cov = 1/K * Y @ np.conjugate(Y.T)

    for i in range(N):
        for j in range(i+1, N):
            J[i,j] = -1/2 * np.trace(projectors_list[i] @ projectors_list[j])

    for i in range(N):
        b[i] = - np.trace(projectors_list[i] @ sampled_cov)
        for j in range(N):
           b[i] += 1/2* np.trace(projectors_list[i] @ projectors_list[j])

    return J, b


def get_ising_parameters(N, M, alpha, K, xi, verbose=True):
    """
    Parameters:
    - N         number of users of the network
    - M         length of pilot sequences 
    - alpha     activity pattern
    - K         number of antennas
    - xi        standard deviation of thermal noise

    Returns:
    - J         coupling matrix of the Ising Hamiltonian
    - b         magnetic field paramters of the Ising Hamiltonian
    """
    
    P = uniform_direction.rvs(dim=M, size=N).transpose()
    Y = generate_signal(alpha, P, xi, K)
    J, b = get_couplings(Y, P)
    
    if verbose: 
        print("Coupling matrix of the Ising Hamiltonian:")
        print(f"{J}")

    return J, b


def binary_to_decimal(bin):
    """returns decimal representation of the binary list bin"""
    bin = np.array(bin)
    powers = len(bin)-1 - np.arange(len(bin))
    dec_places = 2**powers
    return np.sum(dec_places * bin)
   

def probability_histogram(basis, proba_coef, order_by="binary"):
    """
    Parameters:
    - basis         eigenbasis of the final Hamiltonian
    - proba_coef    probability distribution at the different time steps. The ordering is the same as the order of the eigenstates. numpy array, shape (2**N, len(timesTab))
    - order_by      binary or energy

    Returns:
    - xlabels       binary representation of the eigenstates in the selected order
    - coef_to_plot  probabilities of the different eigenstates
    - [not anymore] states_string binary representation of the eigenstates, always in binary order, i.e. for order_by=binary it is identical to xlabels
    """
    # list of labels
    N = int(np.log2(len(basis)))      # basis is a list of QuObj, i.e. each element is a basis vector. There are 2**N basis vectors where N is the number of qubits
    states_string = list(product([0,1], repeat = N))
    states_string = ["".join([str(i) for i in state]) for state in states_string]
    xlabels = states_string.copy()
    coef_to_plot = np.zeros(2**N)

    if order_by == "binary":
        print("order binary")
        for i in range(2**N):
            current_state = np.real(basis[i].full().reshape(2**N))
            index_state = np.where(current_state > 0.99)[0][0]
            coef_to_plot[index_state] = proba_coef[i][-1]
    
    elif order_by == "energy":
        for i in range(2**N):
            current_state = np.real(basis[i].full().reshape(2**N))  # without .full(), .reshape() doesn't work. full() adds an imaginary part of 0j to every number
            index_state = np.where(np.abs(current_state) > 0.99)[0][0]
            xlabels[i] = states_string[index_state]
            coef_to_plot[i] = proba_coef[i][-1]
    
    else:
        print("invalid argument for parameter order_by")

    return xlabels, coef_to_plot


def signal_norm(signal):
    """Returns array of the norms of the signals that are in the COLUMNS of a signal matrix.
    The used norm is ||X|| = sqrt(x1²+x2²+...+xn²)."""
    return np.sqrt(np.sum(signal**2, axis=0))


def CCR(Y, P, T, verbose=False):
    """Conventional correlation receiver (CCR)
    Parameters:
    - Y             matrix of shape (M, K) containing signal of length M received by K antennas
    - P             matrix of shape (M, N) containing N pilot signals of length M
    - T             detection thresold:
                    if the correlation value for a certain pilot is greater than T, the corresponding bit in the activity pattern is set to 1

    Returns:
    - alpha_CCR     CCR estimation of the activity pattern
    - Y_normed      signal matrix Y normalized such that the signal norm is 1 for every antenna
    """

    K = np.shape(Y)[1]
    N = np.shape(P)[1]

    # normalize each received signal
    if verbose:
        print(f"norms of received signals before normalization: {signal_norm(Y)}")
    Y_normed = Y / signal_norm(Y)
    if verbose: 
        print(f"norms of received signals after normalization: {signal_norm(Y_normed)}")

    # compute correlation with all the pilots
    f = np.zeros(shape=(N, K))             # correlation measure
    for n in range(N):
        for k in range(K):
            f[n, k] = np.sum(Y_normed[:, k]*P[:, n])

    # average the correlations over the channels
    f = np.abs(f)
    f = np.mean(f, axis=1)

    # find the activity pattern according to the thresold
    alpha_CCR = np.zeros(N)
    alpha_CCR[f>T] = 1

    if verbose:
        print(f"correlation measure averaged over all K channels: {f}")
        print(f"identified activity pattern: {alpha_CCR}")

    return alpha_CCR, Y_normed




def main():
    """main function for testing purposes"""
    print(list(product([0,1], repeat = 10)))

if __name__ == "__main__":
    main()


