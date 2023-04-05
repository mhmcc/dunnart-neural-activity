"""Calcium imaging data voxel-based processing tools

This module contains tools for basic processing of 2d calcium imaging
data for voxel-based analysis.
"""

__version__ = '0.1'
__author__ = 'Michael H McCullough'
__status__ = 'Development'

import numpy as np
from scipy.ndimage.filters import  percentile_filter, uniform_filter
from scipy.stats import iqr
from sklearn.neighbors import KernelDensity


def get_voxel_dff(
        im,
        bin_edge_len,
        win_size,
        filter_percentile=10,
        low_activity_thr=None,
        eps=1e-7,
    ):
    """ Extract voxel dff time series from 2d calcium imaging data
    
    Input
    -----
    im : array_like, shape (num_frames, height, width)
        Imaging data where num_frames is the number of frames, and
        height and width are the pixel dimensions of each 2d frame.
    
    bin_edge_length : int
        Edge length in pixels of each square voxel. Image dimensions
        must be divisible by bin_edge_length.
    
    win_size : int
        Length of the percentile filter and uniform filter for
        estimating the baseline fluorescence 
    
    filter_percentile : float
        The percentile for estimating baseline fluorescence
    
    low_activity_thr : float
        Low activity threshold for voxels. If None (default) then value
        is automatically computed as equivalent to a single pixel
        active on average in a given bin over duration win_size.
    
    eps : float
        Regularisation to avoid divide by zero error in dff calculation
    
    Returns
    -------
    dff : array_like, shape (num_frames, m, k)
        dff time series for each voxel arranged as per imaging data.
        Array has dimensions (num_frames, m, k) where
        m = height/bin_edge_length and k = width/bin_edge_length.
    
    """
    # Input check
    cond1 = (im.shape[1]%bin_edge_len) != 0
    cond2 = (im.shape[2]%bin_edge_len) != 0
    if cond1 or cond2:
        raise ValueError(
                  'The height and width of the frame must both be '
                  'divisible by bin_edge_len.'
              )

    # Set parameters
    win_size = win_size.astype(int)
    if low_activity_thr==None:
        # If not specified then set low activity threshold to be
        # equivalent to a single pixel active on average in a
        # given bin over duration win_size
        low_activity_thr = bin_edge_len**(-2)
    
    # Extract voxel time series
    ox = im.shape[1]//bin_edge_len
    oy = im.shape[2]//bin_edge_len
    f = im.reshape(
               (im.shape[0], ox, bin_edge_len, oy, bin_edge_len)
           ).mean(4).mean(2)
    
    # Percentile filter to get baseline fluorescence
    k = np.ones((win_size,1,1)) # filter footprint
    f0 = percentile_filter(
            f,
            filter_percentile,
            footprint=k,
            mode='mirror',
         )
    
    # Smooth f0
    f0 = uniform_filter(
            f0,
            size=(win_size,1,1),
            mode='mirror',
         )
    
    # Detrend and scale
    dff = (f-f0)/(f0+eps)
    
    # Any dff voxels with low baseline (below threshold) are set to set
    # to zero throughout (i.e. assume ROI has no fluorescence activity)
    dff = np.nan_to_num(dff)*(f0>low_activity_thr).all(axis=0)
    
    return dff


def binarise_dff(dff,thr=3):
    """ Binarise dff time series by thresholding.
    
    Input
    -----
    dff : array_like, shape (num_frames, m, k)
        dff time series for each voxel arranged as per imaging data.
        Array has dimensions (num_frames, m, k) where num_frames is the
        number of frames, m is the number of rows in the voxel array
        and k is the number of columns in the voxel array.
    
    thr : float
        The number of standard deviations for the activity threshold.
    
    Returns
    -------
    b : array_like, shape (num_frames, m, k)
        binary activity time series for each voxel with same dimensions
        as dff input array.
    
    """
    b = np.zeros(dff.shape,dtype=bool)
    # Loop over voxels
    for i in np.arange(dff.shape[1]):
        for j in np.arange(dff.shape[2]):
            x = dff[:,i,j]
            if x.any(): # only process ROIs that are not blank
                n = x.shape[0]
                # Silverman's rule to get KDE bandwidth
                h = 0.9*np.min([x.std(),iqr(x)/1.34])*n**(-1/5)
                kde = KernelDensity(bandwidth=h)
                kde.fit(x.reshape(-1,1))
                # Evaluate kde to find peak
                mu = x.mean()
                std = x.std()
                xq = np.linspace(mu-3*std,mu+3*std,100)
                pdf = np.exp(kde.score_samples(xq.reshape(-1,1)))
                mu = xq[pdf.argmax()]
                # Extract negative deflections from the peak and mirror
                y = x[x<mu]
                y = np.append(y,2*mu-y) # reflect
                std = y.std()
                b[:,i,j] = x>(mu+thr*std) # threshold
    return b

