import numpy as np

from utils import compute_target, compute_std_and_var



def compute_unstable_em_svm(X, y, alpha, nu, abs_eps, max_iterations, noisy=False, noise_period=20, seed=None):
    """
    Unstable implementation of algorithm EM-SVM.

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
    """

    if seed is not None:
        np.random.seed(seed)
    
    if noisy:
        n_fill = int(np.log10(max_iterations)) + 1

    _, d = X.shape

    yX = y * X
    std, _ = compute_std_and_var(X)
    std_inv_mat = np.diagflat(1 / std)
    
    beta = np.random.randn(d, 1)
    betas = [beta]
    iteration = 1
    
    while True:
        if noisy and iteration % noise_period == 0:
            loss = compute_target(X, y, alpha, nu, beta)
            print(f'At iteration ({str(iteration).zfill(n_fill)}/{str(max_iterations).zfill(n_fill)}), loss = {loss:.3f}')

        # E-step
        lam_inv = 1 / np.abs(1 - yX @ beta)
        lam_inv_mat = np.diagflat(lam_inv)
        
        omega_inv = alpha * ((nu * std / np.abs(beta)) ** (2 - alpha))
        omega_inv_mat = np.diagflat(omega_inv)
        
        # M-step
        new_beta = np.linalg.solve(
            std_inv_mat @ omega_inv_mat / (nu ** 2) + yX.T @ lam_inv_mat @ yX,
            yX.T @ (1 + lam_inv),
        )

        if np.abs(beta - new_beta).max() <= abs_eps or iteration == max_iterations:
            if noisy:
                loss = compute_target(X, y, alpha, nu, beta)
                print(f'Finished run ({str(iteration).zfill(n_fill)}/{str(max_iterations).zfill(n_fill)}), loss = {loss:.3f}')
            return np.array(betas)
        
        beta = new_beta
        betas.append(beta)
        iteration += 1
        