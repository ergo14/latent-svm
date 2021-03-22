import numpy as np

from utils import compute_target, compute_std_and_var



def compute_ecme_svm(X, y, alpha, anu, bnu, abs_eps, max_iterations, noisy=False, noise_period=20, seed=None):
    """
    Stable implementation of algorithm ECME-SVM.

    Parameters
    ----------
    X : ndarary
        An n by d array of n observation in an d-dimensional space.
    y : ndarray
        An n by 1 array of n responses.
    alpha : float
        Parameter used for L-regularization.
    anu : float
        Constant used for L-regularization.
    bnu : float
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
    nu_values : ndarray
        An N x 1 array of the N successive nu values. N is the number of iterations 
    """

    if seed is not None:
        np.random.seed(seed)
        
    if noisy:
        n_fill = int(np.log10(max_iterations)) + 1

    _, d = X.shape

    std, var = compute_std_and_var(X)
    
    beta = np.random.randn(d, 1)
    betas = [beta]
    nu = np.random.randn()
    nu_values = [nu]
    iteration = 1
    
    while True:
        if noisy and iteration % noise_period == 0:
            loss = compute_target(X, y, alpha, nu, beta)
            print(f'At iteration ({str(iteration).zfill(n_fill)}/{str(max_iterations).zfill(n_fill)}), loss = {loss:.3f}')

        # E step
        lam = np.abs(1 - (y * X) @ beta)
        omega = ((np.abs(beta) / (nu * std)) ** (2 - alpha)) / alpha
        
        nonzero_beta = ~np.isclose(omega, 0).squeeze()
        is_sv = np.isclose(lam, 0).squeeze()
        
        Xs = (y * X)[is_sv[:, None] & nonzero_beta[None, :]].reshape(is_sv.sum(), nonzero_beta.sum())
        Xns = (y * X)[(~is_sv[:, None]) & nonzero_beta[None, :]].reshape((~is_sv).sum(), nonzero_beta.sum())
        
        lam_inv = 1 / lam[~is_sv]
        omega_inv = 1 / omega[nonzero_beta]
        var_inv = 1 / var[nonzero_beta]
        
        # CM step
        Bns = np.diagflat(var_inv * omega_inv) / (nu ** 2) + Xns.T @ np.diagflat(lam_inv) @ Xns
        
        partitioned_matrix = np.block([
            [Bns, Xs.T],
            [Xs, np.zeros((Xs.shape[0], Xs.shape[0]))]
        ])
        
        target = np.r_[
            Xns.T @ (1 + lam_inv),
            np.ones((is_sv.sum(), 1)),
        ]
            
        beta_and_lagrange_multipliers =  np.linalg.pinv(partitioned_matrix) @ target
        new_beta = np.zeros_like(beta)
        new_beta[nonzero_beta, :] = beta_and_lagrange_multipliers[:nonzero_beta.sum(), :]
        
        # CME step
        new_nu = np.power((bnu + ((np.abs(new_beta) / std) ** alpha).sum()) / (d / alpha + anu - 1), -1 / alpha)
        
        if np.abs(beta - new_beta).max() <= abs_eps or iteration == max_iterations:
            if noisy:
                loss = compute_target(X, y, alpha, nu, beta)
                print(f'Finished run ({str(iteration).zfill(n_fill)}/{str(max_iterations).zfill(n_fill)}), loss = {loss:.3f}')
            return np.array(betas), np.array(nu_values).reshape(-1, 1)

        beta = new_beta
        betas.append(beta)
        nu = new_nu
        nu_values.append(nu)
        iteration += 1
