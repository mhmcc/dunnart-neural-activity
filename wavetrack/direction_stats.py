"""Directional statistics

This module contains routines for computing various directional
statistics.
"""

__version__ = '0.1'
__author__ = 'Michael H McCullough'
__status__ = 'Development'

import numpy as np
import scipy as sp

        
def wrap2pi(theta):
    """ Wrap angle theta (radians) to (-pi,pi].
    
    """
    return (( -theta + np.pi) % (2.0 * np.pi ) - np.pi) * -1.0


def omnibus(
        alpha,
        w=None,
        sz=np.radians(1),
        axis=None,
    ):
    """
    Computes omnibus test for non-uniformity of circular data. The test
    is also known as Hodges-Ajne test.
    
    ***NOTE: This is a modified version of code from the pycircstat
    package: https://github.com/circstat/pycircstat

    MIT License

    Copyright (c) 2017 circstat

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    
    H0: the population is uniformly distributed around the circle
    HA: the population is not distributed uniformly around the circle

    Alternative to the Rayleigh and Rao's test. Works well for
    unimodal, bimodal or multimodal data. If requirements of the
    Rayleigh test are met, the latter is more powerful.

    :param alpha: sample of angles in radian
    :param w:      number of incidences in case of binned angle data
    :param sz:    step size for evaluating distribution, default 1 deg
    :param axis:  compute along this dimension, default is None
                  if axis=None, array is raveled
    :return pval: two-tailed p-value
    :return m:    minimum number of samples falling in one half of the
                  circle

    References: [Fisher1995]_, [Jammalamadaka2001]_, [Zar2009]_
    """

    if w is None:
        w = np.ones_like(alpha)

    assert w.shape == alpha.shape, (
        "Dimensions of alpha and w must match"
    )

    alpha = alpha % (2 * np.pi)
    n = np.sum(w, axis=axis)

    dg = np.arange(0, np.pi, np.radians(1))

    m1 = np.zeros((len(dg),) + alpha.shape[1:])
    m2 = np.zeros((len(dg),) + alpha.shape[1:])

    for i, dg_val in enumerate(dg):
        m1[i, ...] = np.sum(
                         w * (
                                 (alpha > dg_val)
                                 & (alpha < np.pi + dg_val)
                             ),
                         axis=axis
                     )
        m2[i, ...] = n - m1[i, ...]

    m = np.concatenate((m1, m2), axis=0).min(axis=axis)

    n = np.atleast_1d(n)
    m = np.atleast_1d(m)
    A = np.empty_like(n)
    pval = np.empty_like(n)
    idx50 = (n > 50)

    if np.any(idx50):
        A[idx50] = (
                       np.pi
                       * np.sqrt(n[idx50])
                       / 2
                       / (n[idx50] - 2 * m[idx50])
                   )
        pval[idx50] = (
                          np.sqrt(2 * np.pi) / A[idx50]
                          * np.exp(-np.pi ** 2 / 8 / A[idx50] ** 2)
                      )

    if np.any(~idx50):
        pval[~idx50] = (
                           2 ** (1 - n[~idx50])
                           * (n[~idx50]
                           - 2 * m[~idx50])
                           * sp.special.comb(n[~idx50], m[~idx50])
                       )

    return pval.squeeze(), m


def hermans_rasson(
        alpha,
        N=1000
    ):
    """Computes the p-value and test statistic T for the Hermans-Rasson
    test.
    
    Implemented based on Hermans, M., & Rasson, J. P. (1985). A new
    Sobolev test for uniformity on the circle. Biometrika, 72(3),
    698-702.
    
    The Hermans-Rasson test is more powerful than the Rayleigh test
    for bi-modal and multi-modal deviations from a uniform circular
    distribution. It also has similar power to a Rayleigh test for
    unimodal distributions.
    
    Input
    -----
    alpha : array_like, shape (n,)
        Sample of angles in radians.
    
    N : int, optional
        Number of iterations for computing the test statistic.
    
    Returns
    -------
    pval : float
        Significance level.
    
    T : float
        The test statistic.
    
    """
    def hrt(alpha):
        # Return the test statistic for the sample alpha in radians
        t = 0
        beta = 2.895 # constant for beta_2; see paper for details
        
        A = (alpha - alpha.reshape(-1,1)).flatten()
        T = np.pi -np.abs(np.pi - np.abs(A)) + beta*np.abs(np.sin(A))
        T = T.sum()/len(alpha)
        
        return T
    
    # Get test statistic for sample
    T = hrt(alpha)
    
    # Get test statistic for null hypothesis
    k = len(alpha)
    T0 = np.zeros(N)
    for n in range(N):
        alpha_null = 2*np.pi*np.random.rand(k)
        T0[n] = hrt(alpha_null)
    
    pval = (T0<T).sum()/N
    
    return pval,T