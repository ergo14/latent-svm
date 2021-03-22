import numpy as np



def compute_target(X, y, alpha, nu, beta):
    """
    Calculating the target from equation (1).

    Parameters
    ----------
    X : ndarary
        An n by d array of n observation in an d-dimensional space.
    y : ndarray
        An n by 1 array of n responses.
    alpha : float
        Parameter used for L-regularization.
    nu : float
        Constant used for L-regularization.
    beta : ndarray
        An d by 1 array.
    
    Returns
    -------
    loss : float
    """
    
    std = X.std(axis=0)
    std[np.isclose(std, 0)] = 1
    std = std.reshape(-1, 1)
    return np.maximum(0, 1 - (y * X) @ beta).sum() + ((np.abs(beta) / (nu * std)) ** alpha).sum()


def compute_std_and_var(X):
    """
    Return standard deviation and variance of features. Replaces 0 by 1 for constant features.

    Parameters
    ----------
    X : ndarary
        An n by d array of n observation in an d-dimensional space.
    
    Returns
    -------
    std : ndarary
        An d by 1 array of d standard deviations.
    var : ndarary
        An d by 1 array of d standard variances.
    """
    std = X.std(axis=0)
    std[np.isclose(std, 0)] = 1
    std = std.reshape(-1, 1)
    var = std ** 2
    return std, var
    