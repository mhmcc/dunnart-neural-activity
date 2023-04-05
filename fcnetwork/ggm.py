"""Functional connectivity estimation via Gaussian graphical modelling

This module contains tools for estimating functional connectivity
networks from neural data via Gaussian graphical modelling.
"""

__version__ = '0.1'
__author__ = 'Michael H McCullough'
__status__ = 'Development'

import numpy as np
import time
from inverse_covariance import QuicGraphicalLassoEBIC 


def GridSearchBIC(
        X,
        lims,
        n_val,
        max_iter=10,
        tol=1e-2,
        add_noise=False,
        noise_std=1e-6,
        fit_tol=1e-6,
        fit_max_iter=1000,
        verbose=0
    ):
    """ Estimate the precision matrix for array X with shape (N,D) via
    Gaussian graphical modelling.
    
    This function is a grid search wrapper around the
    QuicGraphicalLassoEBIC function from the skggm package:
    
    @misc{laska_narayan_2017_830033,
        author      = {Jason Laska and
                          Manjari Narayan},
        title       = {{skggm 0.2.7: A scikit-learn compatible package
                          for Gaussian and related Graphical Models}},
        month       = jul,
        year        = 2017,
        doi         = {10.5281/zenodo.830033},
        url         = {https://doi.org/10.5281/zenodo.830033}
    }
    
    Uses Bayesian information criterion to optimise the regularisation
    parameter lambda (i.e. gamma=0).
    
    Input
    -----
    X : array_like, shape (N, D)
        dff time series for each voxel with shape N observations by
        D dimensions.
        
    lims : array_like, shape (2,)
        Upper and lower bounds of the grid search for lambda.
        
    n_val : int
        Number of intermediate values in the grid search.
        
    max_iter : int, optional
        Maximum number of iterations
        
    tol : float, optional
        Convergence threshold
        
    add_noise : bool, optional
        If True add low level Gaussian noise to any dimension for
        which all values are zero. Dimensions with an non-zero
        elements are ignored.
        
    noise_std : float, optional
        Standard deviation of the Gaussian noise applied if add_noise
        is True.
    
    fit_tol : float, optional
        Tolerance for QuicGraphicalLassoEBIC fit.
    
    fit_max_iter : int, optional
        Maximum number of Newton iterations for QuicGraphicalLassoEBIC
        fit.
        
    verbose : int, optional
        If 1 then print iteration details.
    
    Returns
    -------
    lam : float
        Estimated optimal value for regularisation parameter lambda.
        
    mdl : object
        Model for optimal lambda.
    
    """
    
    if add_noise:
        idx = (X==0).all(axis=0)
        if idx.any():
            noise = np.random.normal(
                        loc=0,
                        scale=noise_std,
                        size=X.shape
                    )
            noise = noise*idx.T
            X += noise
   
    # Initialise variables
    delta = tol+1 # a number bigger than tol
    lam_ = lims[1]+delta # a number such that the loop will continue to
                         # iterate more than once unless max_iter=1
    lims_ = lims
    iter_ = 0
    
    # Fit GGM optimised over lambda by gridsearch and BIC
    while (delta>=tol) & (iter_<max_iter):
        # Track iteration time
        t = time.time()
        
        # Generate query points
        lims_ = np.log10(lims_) # convert limits to logspace
        lams = np.logspace(lims_[0],lims_[1],n_val)[::-1]

        # Fit 
        mdl = QuicGraphicalLassoEBIC(
                  lam=1.0,
                  path=lams,
                  tol=fit_tol,
                  max_iter=fit_max_iter,
              )
        mdl.fit(X)
        
        # Compute change in optimal hyperparameter value
        delta = np.abs(lam_-mdl.lam_)
        
        # Update limits
        lam_ = mdl.lam_
        idx = np.nonzero(lam_==lams)[0]
        if idx==0:
            lims_[0] = lams[idx+1]
            lims_[1] = lams[idx]
        elif idx==(n_val-1):
            lims_[0] = lams[idx]
            lims_[1] = lams[idx-1]
        else:
            lims_[0] = lams[idx+1]
            lims_[1] = lams[idx-1]
            
        # Increment interation counter
        iter_ += 1
        elapsed = time.time() - t
        
        # Print iteration details
        if verbose==1:
            msg = (
                   'Iteration '
                   + str(iter_)
                   + '     '
                   + '\u03bb = '
                   + '{:.3f}'.format(lam_)
                   + '     '
                   + 'Elapsed time: '
                   + '{:.2f}'.format(elapsed)
                   + ' seconds'
                  )
            print(msg)
        
    return lam_,mdl


def partial_corr_from_precision(T):
    """ Computes partial correlations from precision matrix
    
    Input
    -----
    T : array_like, shape (D, D)
        Precision matrix.
    
    Returns
    -------
    P : array_like, shape (D, D)
        Partial correlation matrix.
        
    """
    diag = T.diagonal()
    P = -T/np.sqrt(np.matmul(diag.reshape(-1,1),diag.reshape(1,-1)))
    return P