import numpy as np

from utils import compute_target, compute_std_and_var



def compute_mcmc_svm(X, y, nu, abs_eps, max_iterations, noisy=False, noise_period=20, seed=None):
    """
    Implementation of algorithm MCMC-SVM for a value alpha of 1.

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
    abs_eps : float
        Absolute tolerance used to confirm convergence.
    max_iterations : int
        Maximum number of iterations to run.
    noisy : bool
        Flag indicating whether or not to produce logs.
    noise_period : in
        Wait noise_period iterations before logging.
    seed : int or None
        Random seed used to seed numpy.random. No seeding is performed if None is passed.
    
    Returns
    -------
    betas : ndarray
        An N x d x 1 array of the N successive beta values. N is the number of iterations 
    b_values : ndarray
        An N x d x 1 array of the N successive prior mean values of beta. N is the number of iterations 
    """

    if seed is not None:
        np.random.seed(seed)

    if noisy:
        n_fill = int(np.log10(max_iterations)) + 1

    _, d = X.shape
    
    yX = y * X
    std, var = compute_std_and_var(X)
    inv_var_matrix = np.diagflat(1 / var)
    
    beta = np.random.randn(d, 1)
    betas = [beta]
    omega_inv = (nu * std) / np.abs(beta)
    lam_inv = np.abs(1 - yX @ beta)
    b_values = []
    iteration = 1
    
    while True:
        if noisy and iteration % noise_period == 0:
            loss = compute_target(X, y, 1, nu, beta)
            print(f'At iteration ({str(iteration).zfill(n_fill)}/{str(max_iterations).zfill(n_fill)}), loss = {loss:.3f}')
        
        # prepping step 1
        B = np.linalg.inv(inv_var_matrix @ np.diagflat(omega_inv) / (nu ** 2) + yX.T @ np.diagflat(lam_inv) @ yX)
        b = B @ yX.T @ (1 + lam_inv)
        b_values.append(b)
        
        # step 1
        new_beta = np.random.multivariate_normal(b.squeeze(), B).reshape(-1, 1)
        
        if np.abs(beta - new_beta).max() <= abs_eps or iteration == max_iterations:
            if noisy:
                loss = compute_target(X, y, 1, nu, beta)
                print(f'Finished run ({str(iteration).zfill(n_fill)}/{str(max_iterations).zfill(n_fill)}), loss = {loss:.3f}')
            return np.array(betas), np.array(b_values)

        beta = new_beta
        betas.append(beta)
        
        # step 2
        gap = np.abs(1 - yX @ beta)
        is_zero = np.isclose(gap, 0)
        lam_inv = np.zeros_like(gap)
        lam_inv[is_zero] = 1 / (np.random.normal(size=is_zero.sum()) ** 2)
        lam_inv[~is_zero] = np.random.wald(1 / gap[~is_zero], 1)
        
        # step 3
        gap = np.abs(beta) / (nu * std)
        is_zero = np.isclose(gap, 0)
        omega_inv = np.zeros_like(gap)
        omega_inv[is_zero] = 1 / (np.random.normal(size=is_zero.sum()) ** 2)
        omega_inv[~is_zero] = np.random.wald(1 / gap[~is_zero], 1)

        iteration += 1



def compute_mcmc_svm_with_nu(X, y, anu, bnu, abs_eps, max_iterations, noisy=False, noise_period=20, seed=None):
    """
    Implementation of algorithm MCMC-SVM for a value alpha of 1.

    Parameters
    ----------
    X : ndarary
        An n by d array of n observation in an d-dimensional space.
    y : ndarray
        An n by 1 array of n responses.
    alpha : float
        Parameter used for L-regularization.
    anu : float
        Constant used for the prior of nu.
    bnu : float
        Constant used for the prior of nu.
    abs_eps : float
        Absolute tolerance used to confirm convergence.
    max_iterations : int
        Maximum number of iterations to run.
    noisy : bool
        Flag indicating whether or not to produce logs.
    noise_period : in
        Wait noise_period iterations before logging.
    seed : int or None
        Random seed used to seed numpy.random. No seeding is performed if None is passed.
    
    Returns
    -------
    betas : ndarray
        An N x d x 1 array of the N successive beta values. N is the number of iterations 
    b_values : ndarray
        An N x d x 1 array of the N successive prior mean values of beta. N is the number of iterations 
    """
    if seed is not None:
        np.random.seed(seed)

    if noisy:
        n_fill = int(np.log10(max_iterations)) + 1
    
    _, d = X.shape
    
    yX = y * X
    std, var = compute_std_and_var(X)
    inv_var_matrix = np.diagflat(1 / var)
    
    beta = np.random.randn(d, 1)
    betas = [beta]
    nu = 1
    nu_values = [nu]
    omega_inv = (nu * std) / np.abs(beta)
    lam_inv = np.abs(1 - yX @ beta)
    b_values = []
    iteration = 1
    
    while True:
        if noisy and iteration % noise_period == 0:
            loss = compute_target(X, y, 1, nu, beta)
            print(f'At iteration ({str(iteration).zfill(n_fill)}/{str(max_iterations).zfill(n_fill)}), loss = {loss:.3f}')
        
        # prepping step 1
        B = np.linalg.inv(inv_var_matrix @ np.diagflat(omega_inv) / (nu ** 2) + yX.T @ np.diagflat(lam_inv) @ yX)
        b = B @ yX.T @ (1 + lam_inv)
        b_values.append(b)
        
        # step 1
        new_beta = np.random.multivariate_normal(b.squeeze(), B).reshape(-1, 1)
        
        if np.abs(beta - new_beta).max() <= abs_eps or iteration == max_iterations:
            if noisy:
                loss = compute_target(X, y, 1, nu, beta)
                print(f'Finished run ({str(iteration).zfill(n_fill)}/{str(max_iterations).zfill(n_fill)}), loss = {loss:.3f}')
            return np.array(betas), np.array(b_values), np.array(nu_values).reshape(-1, 1)

        beta = new_beta
        betas.append(beta)
        
        # step 2
        gap = np.abs(1 - yX @ beta)
        is_zero = np.isclose(gap, 0)
        lam_inv = np.zeros_like(gap)
        lam_inv[is_zero] = 1 / (np.random.normal(size=is_zero.sum()) ** 2)
        lam_inv[~is_zero] = np.random.wald(1 / gap[~is_zero], 1)
        
        # step 3
        gap = np.abs(beta) / (nu * std)
        is_zero = np.isclose(gap, 0)
        omega_inv = np.zeros_like(gap)
        omega_inv[is_zero] = 1 / (np.random.normal(size=is_zero.sum()) ** 2)
        omega_inv[~is_zero] = np.random.wald(1 / gap[~is_zero], 1)

        # step 4
        nu = 1 / np.random.gamma(shape=anu + d, scale=1 / (bnu + np.abs(beta).sum()))
        nu_values.append(nu)

        iteration += 1