import numpy as np

def initialize_parameters(C, b0, w, gaussianNoise, lbda, print_params=False):
    """
    Parameters
    ----------
    C : numpy array
        The code words of the sequences. The code words are the columns of the matrix and are assumed to be normalized.
    b0 : numpy array
        The initial activity pattern.
    w : numpy array
        The channel coefficients.
    gaussianNoise : numpy array
        The gaussian noise at the BS
    lbda : float
        The regularization parameter.
    print_params : bool, optional
        Print the generated J_ij and h_i. The default is False.

    Returns
    -------
    J : numpy array
        The coupling matrix.
    h : numpy array
        The local fields.

    """
    N = C.shape[1] #number of nodes
    M = C.shape[0] #size of the sequences

    codesCopy = C.copy()
    #Take into account the channel coeffcients
    for i in range(N):
        codesCopy[:,i] = w[i]*codesCopy[:,i]

    #received signal
    y = codesCopy.dot(b0.T) + gaussianNoise #received signal
    power = np.linalg.norm(y)**2

    #defining the couplings J_ij
    J = np.zeros((N,N))
    for i in range(N):
        for j in range(i+1,N):
            J[i,j] = - 1/2*codesCopy[:,i].dot(codesCopy[:,j])

    #defining the local fields h_i
    h = np.zeros(N)
    for i in range(N):
        h[i] = -1*np.dot(codesCopy[:,i],y)
        for j in range(N):
            h[i] += 1/2 * codesCopy[:,i].dot(codesCopy[:,j])
        h[i] += lbda/2

    if print_params:
        print("Received power : ",power)
        print("Number of nodes : ",N)
        print("Size of the sequences: ",M)
        print("Active users: ",b0)
        print("-----------------")
        print("Local fields : \n",h)
        print("-----------------")
        print("Couplings matrix : \n",J)

    return J,h,power