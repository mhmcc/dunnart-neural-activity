"""Wave tracking by connected components

This module contains tools for tracking waves in binary neural
activity data estimated from 2-dimensional calcium imaging data.
"""

__version__ = '0.1'
__author__ = 'Michael H McCullough'
__status__ = 'Development'

import os
import numpy as np
from scipy import ndimage
from skimage.morphology import remove_small_objects
from scipy.linalg import block_diag
from filterpy import kalman
from filterpy.common import Q_discrete_white_noise
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib import cm
from matplotlib.collections import LineCollection
# from matplotlib.transforms import TransformedBbox, Affine2D
import pylab as pl
from IPython import display

@dataclass(frozen=True)
class CCProps:
    """ Dataclass to store cc properties
    
    Constructor arguments
    ---------------------
    duration : float
        Duration of cc (length of cc along axis 0)
    
    total_size : float
        Total number of activations in cc (volumse of cc)
        
    voxels : float
        Total number of voxels participating in the cc (area of the
        projection of the cc onto axes 1 and 2)
        
    size : numpy.ndarray
        area of the cc w.r.t. time (sum over axes 1 and 2)
        
    """
    duration : float
    total_size : float
    voxels : float
    size : float


@dataclass(frozen=True)
class ActivityThresholds:
    """ Dataclass to store activity thresholds
    
    Constructor arguments
    ---------------------
    min_duration : float
        Lower threshold for CC duration.
        
    min_total_size : float
        Lower threshold for total CC size in number of activations.

    min_voxels : float
        Lower threshold for the number of voxels participating in the
        CC.
    
    min_size :  float
        Lower threshold for the size (area in number of voxels) of a CC
        in any given frame.
    
    """
    min_duration : float
    min_total_size : float
    min_voxels : float
    min_size : float


def circ_mean(theta):
    """ Compute the circular mean.
    
    Based on https://en.wikipedia.org/wiki/Directional_statistics
    
    Input
    -----
    theta : array_like, shape(n,)
        Sample of n angles in radians.
        
    Returns
    -------
    circ_mean : float
        Circular mean
        
    """
    v = np.exp(1j*theta)
    circ_mean = np.angle(v.mean())
    return circ_mean


        
def get_cc_properties(
        labels,
        cc,
        num_cc,
    ):
    """ Computes basic properties of spatio-temporal connected
    components.
    
    Input
    -----
    labels : array_like, shape (num_frames, m, k)
        Labelled connected components array. Array must conform to the
        output format of the function scipy.ndimage.label. The array
        has dimensions (num_frames, m, k) where num_frames is the
        number of frames, m is the number of rows in the voxel array
        and k is the number of columns in the voxel array.
        
    cc : list
        List of slices. Must conform to the output of the function
        scipy.ndimage.find_objects(labels).
        
    num_cc : int
        The number of connected components.
    
    Returns
    -------
    cc_props : dataclass
        Lower thresholds for cc duration, total_size, voxels and size.
    
    """
    # Duration (length along axis 0)
    duration = np.array([labels[c].shape[0] for c in cc],dtype=float)
    
    # Total size
    total_size = ndimage.sum(
                     labels>0,
                     labels=labels,
                     index=np.arange(1,num_cc+1),
                 )
    
    # Number of participating voxels (2d projection of cc onto axes 1
    # and 2
    voxels = np.array(
                 [
                     (labels[c]==i).any(axis=0).sum()
                     for i,c
                     in enumerate(cc,1)
                 ],
                 dtype=float,
             )
    
    # Size w.r.t. axis 0
    size = []
    for i,c in enumerate(cc,1):
        sz = (labels[c]==i).sum(axis=(1,2))
        size.append(sz)
    size = np.concatenate(size,axis=0,dtype=float)
    
    # Output
    cc_props = CCProps(
                   duration=duration,
                   total_size=total_size,
                   voxels=voxels,
                   size=size,
               )
    
    return cc_props
    
    
def activity_thresholds(
        b,
        n_shuffles=1000,
        alpha=0.05,
    ):
    """ Estimate spatiotemporal activity thresholds for binary neural
    activity data.
    
    Input
    -----
    b : array_like, shape (num_frames, m, k)
        binary activity time series for each voxel arranged as per
        imaging data. Array has dimensions (num_frames, m, k) where
        num_frames is the number of frames, m is the number of rows
        in the voxel array and k is the number of columns in the voxel
        array.
        
    n_shuffles : int, optional
        Number of circular shuffles to estimate null distribution.
        
    alpha : float, optional
        Significance level.
    
    Returns
    -------
    thr : dataclass (Activity_Thresholds)
        Dataclass containing min_duration, min_total_size, min_voxels
        and min_size
    
    """
    k = b.shape[0]
    y = np.zeros_like(b,dtype='bool')
    T = []
    S = []
    V = []
    U = []
    
    # Shuffle
    for n in range(n_shuffles):
        for i in range(b.shape[1]):
            for j in range(b.shape[2]):
                idx = np.arange(b.shape[0])
                shift = np.random.randint(b.shape[0])
                idx = np.roll(idx,shift)
                y[:,i,j] = b[idx,i,j]
                
        # CC analysis
        labels,num_cc = ndimage.label(y)
        cc = ndimage.find_objects(labels)
        
        
        # Enumerate duration, total size, voxels and size w.r.t. time
        cc_props = get_cc_properties(
                        labels,
                        cc,
                        num_cc,
                    )
        
        # Store CC properties
        T.append(cc_props.duration)
        S.append(cc_props.total_size)
        V.append(cc_props.voxels)
        U.append(cc_props.size)
        
    # Significance thresholding
    percent = (1-alpha)*100
    T = np.concatenate(T)
    S = np.concatenate(S)
    V = np.concatenate(V)
    U = np.concatenate(U)
    min_duration = np.percentile(T,percent)
    min_total_size = np.percentile(S,percent)
    min_voxels = np.percentile(V,percent)
    min_size = np.percentile(U,percent)
    
    thr = ActivityThresholds(
              min_duration=min_duration,
              min_total_size=min_total_size,
              min_voxels=min_voxels,
              min_size=min_size,
          )
    
    return thr


def label_cc(
        b,
        n_shuffles=1000,
        alpha=0.05,
    ):
    """ Label all connected components and filter out cc's which are
    not significant.
    
    This function finds contiguous spatio-temporal events with size,
    duration, voxel participation and area per frame above chance level
    as determined by circular shuffles in the temporal dimension.
    
    Input
    -----
    b : array_like, shape (num_frames, m, k)
        binary activity time series for each voxel arranged as per
        imaging data. Array has dimensions (num_frames, m, k) where
        num_frames is the number of frames, m is the number of rows
        in the voxel array and k is the number of columns in the voxel
        array.
        
    n_shuffles : int, optional
        Number of circular shuffles to estimate null distribution.
        
    alpha : float, optional
        Significance level.
    
    Returns
    -------
    results_dict : dict
        Dictionary containing the connected components and labels
        corresponding to above-chance-level events, and thresholds for
        duration, size, area and size per frame.
        
    """

    # Compute significant coactivity
    thr = activity_thresholds(
              b,
              n_shuffles,
              alpha,
          )
    
    # Break cc's if size for a given time step is < min_duration
    bw = np.zeros_like(b,dtype='bool')
    for k in range(b.shape[0]):
        bw[k,:,:] = remove_small_objects(
                        b[k,:,:],
                        min_size=thr.min_size+1,
                    )
    
    # Label cc's
    labels,num_cc = ndimage.label(bw)
    cc = ndimage.find_objects(labels)
    
    # Compute the duration and area of all cc's
    cc_props = get_cc_properties(
                        labels,
                        cc,
                        num_cc,
                    )
    idx0 = cc_props.duration>thr.min_duration
    idx1 = cc_props.total_size>thr.min_total_size
    idx2 = cc_props.voxels>thr.min_voxels
    mask_indicies = idx0*idx1*idx2
    mask = np.zeros_like(bw,dtype=bool)
    mask[mask_indicies[labels-1]] = True
    
    # Relabel cc's
    labels,num_cc = ndimage.label(bw*mask)
    cc = ndimage.find_objects(labels)
    
    # Store output
    results_dict ={
                       'labels':labels,
                       'cc':cc,
                       'num_cc':num_cc,
                       'min_duration':thr.min_duration,
                       'min_total_size':thr.min_total_size,
                       'min_voxels':thr.min_voxels,
                       'min_size':thr.min_size,
                  }
    
    return results_dict


def rts_smoother_2d(
        z,
        dt=1,
        var=0.1,
        noise=10,
        P_init=100
    ):
    """ Rauch–Tung–Striebel Kalman smoother for 2-dimensional data.
    
    Assumes Newtonian mechanics with constant acceleration. Operates on
    2-dimensional data. Intended for smoothing estimated wave peak
    trajectories - the default parameters have been tuned by inspection
    for this purpose.
    
    Uses the the filterpy package: github.com/rlabbe/filterpy created
    by Roger R. Labbe Jr (distributed under the MIT License).
    
    Input
    -----
    z : array_like, shape (n, 2)
        2-dimensional observations to be smoothed
    
    dt : float, optional
        Sampling period
    
    var : float, optional
        Process noise variance
        
    noise : float, optional
        Measurement noise variance
        
    P_init : float, optional
        Initial value for covariance matrix diagonal (other elements
        set to zero)
        
    Returns
    -------
    zs : array_like, shape (n, 2)
        2-dimensional smoothed trajectory
    
    """

    # Construct filter
    kf = kalman.KalmanFilter(dim_x=6,dim_z=2)
    # State transition matrix (constant acceleration)
    kf.F = np.array(
               [
                   [1, dt, dt**2/2, 0, 0, 0],
                   [0, 1, dt, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 1, dt, dt**2/2],
                   [0, 0, 0, 0, 1, dt],
                   [0, 0, 0, 0, 0, 1]
               ]
           )
    
    # Process noise
    Q = Q_discrete_white_noise(dim=3, dt=dt, var=var)
    Q = block_diag(Q, Q)
    kf.Q = Q
    # Measurement function
    H = np.array(
            [
                [1.,0.,0.,0.,0.,0.],
                [0.,0.,0.,1.,0.,0.]
            ]
        )
    kf.H = H
    # Measurement noise
    kf.R = np.array([[noise, 0.],
                     [0., noise]])
    # Initial conditions
    kf.x = np.array([z[0,0], 0., 0., z[0,1], 0., 0.]).T
    kf.P = np.eye(6)*P_init

    # Run filter
    mu, cov, _, _ = kf.batch_filter(z)

    # Smooth
    xs, _, _, _ = kf.rts_smoother(mu,cov)
    
    zs = np.stack((xs[:,0],xs[:,3])).T
    
    return zs


def mean_trajectory_angle(z):
    """ Computes mean angle long trajectory in radians.
    
    Input
    -----
    z : array_like, shape(n, 2)
        2-dimensional trajectory.
    
    Returns
    -------
    theta : float
        Circular mean angle along trajectory z in radians.
        
    """
    dz = np.diff(z,axis=0)
    dtheta = np.arctan2(-dz[:,0],dz[:,1]) # axis 1 dir is negative
    theta = circ_mean(dtheta)
    return theta
    

def trajectory_length(z):
    """ Sums length along 2-dimensional trajectory.
    
    Input
    -----
    z : array_like, shape(n, 2)
        2-dimensional trajectory
    
    Returns
    -------
    d : float
        length along trajectory z
        
    """
    x = np.diff(z,axis=0)
    x = np.linalg.norm(x,axis=1)
    d = x.sum()
    return d


def wave_peak_tracker(
        dff,
        labels,
        cc,
        min_duration,
        min_total_size,
        min_voxels,
        max_speed,
        generate_movie=False,
        save_frames=False,
        im=None,
        **props,
    ):
    """Get trajectories of spatio-temporal events.
    
    Tracks the peaks of spatially smoothed df/f for spatio-temporal
    events given by connected components labels.
    
    Input
    -----
    dff : array_like, shape (num_frames, m, k)
        dff time series for each voxel arranged as per imaging data.
        Array has dimensions (num_frames, m, k) where num_frames is the
        number of frames, m is the number of rows in the voxel array
        and k is the number of columns in the voxel array.
        
    labels : array_like, shape (num_frames, m, k)
        Labelled connected components array. Array must conform to the
        output format of the function scipy.ndimage.label. The array
        has dimensions (num_frames, m, k) where num_frames is the
        number of frames, m is the number of rows in the voxel array
        and k is the number of columns in the voxel array.
        
    cc : list
        List of slices. Must conform to the output of the function
        scipy.ndimage.find_objects(labels).
        
    min_duration : float
        Minimum event duration.
    
    min_total_size : float
        Minimum wave size.
    
    min_voxels : float
        Minimum wave voxels (area).

    max_speed : float
        The maximum instantaneous speed of a wave with units of bins
        edge length (i.e. how many voxels can a wave traverse in one
        time step). Waves with greater speed will be split into two
        events.
    
    generate_movie : bool, optional
        Generate movie illustrating the tracking if True.
    
    save_frames : bool, optional
        Save still frames from movie as images.
    
    im : array_like, shape (num_frames, w, h), optional
        imaging data array.
    
    props : dict, optional
        Dictionary of movie rendering properties.
    
    Returns
    -------
    waves : dataclass (Waves)
        Dataclass containing wave trajectory (x), wave size (size) and
        wave area (area).
    
    """
    # Keyword arguments
    defaultprops = {
                       "movie_path" : '', # save path for movie
                       "frame_path" : '', # save path for still frames
                       "vmax" : None, # image vmax
                       "figsize" : (8,6),
                       "fontsize" : 14,
                       "bin_edge_len" : 1,
                       "cmap" : cm.viridis,
                       "line_alpha" : 0.75,
                       "line_width" : 4,
                       "marker_size" : 20,
                       "marker_color" : 'r',
                       "scale_bar" : False,
                       "scale_bar_length" : 0,
                       "scale_bar_pos" : [0,0],
                       "scale_bar_width" : 0,
                       "scale_bar_units" : '',
                       "scale" : 0,
                       "scale_bar_color" : 'k',
                       "fontsize" : 12,
                       "text_pad" : 0,
                       "imaging_rate" : 1,
                       "playback_annotation" : False,
                       "playback_ann_pos" : [0,0],
                       "playback_multiplier" : 1,
                   }
    defaultprops.update(props)
    props = defaultprops
    
    # Input check
    if generate_movie:
        if (props["movie_path"]==''):
            raise ValueError(
                    'Path to save movie is undefined.'
                )
        if not os.path.exists(props["movie_path"].parent):
            os.makedirs(props["movie_path"].parent)
    if save_frames:
        if (props["frame_path"]==''):
            raise ValueError(
                    'Path to save frames is undefined.'
                )
        if not os.path.exists(props["frame_path"]):
            os.makedirs(props["frame_path"])
        
    
    # Initialise output list
    waves = []
    
    # Initialise movie generation
    if generate_movie:
        # Set up axes
        fig,axs = plt.subplots(
                     nrows=1,
                     ncols=3,
                     figsize=props['figsize'],
                     tight_layout=True,
                 )
        for ax in axs:
            ax.axis('off')
        
        # Set up moviewriter
        fps = props['imaging_rate'] * props['playback_multiplier']
        moviewriter = FFMpegWriter(
                          fps=fps,
                          codec='h264',
                      )
        moviewriter.setup(fig, props['movie_path'], dpi=100)
    
    # Loop over significant connected components
    for i,c in enumerate(cc):
        # First pass tracking
        mask = labels[c[0],:,:]==(i+1)
        event_dff=dff[c[0],:,:]*mask
        x,_ = track_dff_peak_2d(event_dff,mask)
        
        # Tracking refinement: Split trajectory if com jumps too far in
        # one step. This can happen when there are multiple waves and
        # also occurs due to noisy transients at the ends of events.
        dx = np.linalg.norm(np.diff(x,axis=0),axis=1)
        split_idx = np.where(dx>max_speed)[0]
        c_split = []
        if split_idx.shape[0]==0:
            c_split.append(c)
        else:
            # add start point and align with global time
            split_idx = np.concatenate([[-1],split_idx]) + c[0].start
            # add endpoint
            split_idx = np.concatenate([split_idx,[c[0].stop]])
            for c1,c2 in zip(split_idx[:-1],split_idx[1:]):
                c1+=1
                if c1<c2:
                    c_split.append(
                        (
                            slice(c1,c2, None),
                            slice(0, 12, None),
                            slice(0, 16, None),
                        )
                    )
        
        # Second pass tracking
        for cs in c_split:
            mask = labels[cs[0],:,:]==(i+1)
            N = mask.shape[0]
            size = mask.sum(axis=(1,2))
            total_size = mask.size
            n_voxels = mask.any(axis=0).sum()
            
            if (
                    (N<=min_duration)
                    or (total_size<=min_total_size)
                    or (n_voxels<=min_voxels)
               ):
                continue # cc too short or small after split
            event_dff=dff[cs[0],:,:]*mask
            x,smoothed = track_dff_peak_2d(event_dff,mask)
            
            # Kalman smoother
            xs = rts_smoother_2d(x)

            # Append to output
            w = {
                    'x' : xs,
                    'size' : size,
                    'area' : n_voxels
                }
            waves.append(w)
            
            if generate_movie:
                # Import the imaging data
                raw = im[cs[0],:,:]
                
                # Reset and render gap between waves
                for ax in axs:
                    ax.clear()
                    ax.axis('off')
                    dummy_img = np.zeros_like(event_dff[0,:,:])
                    dummy_img[0,0] = 1;
                    ax.imshow(dummy_img,cmap='gray')
                axs[0].set_title('Imaging data')
                axs[1].set_title(r'Masked $\frac{\Delta f}{f}$')
                axs[2].set_title('Smoothed')
                display.clear_output(wait=True) 
                display.display(pl.gcf())

                # Loop over frames
                for n in range(N):
                    # Clear axes
                    for ax in axs:
                        ax.clear()
                        ax.axis('off')
                        
                    # Plot imaging data
                    ax = axs[0]
                    ax.imshow(
                        raw[n,:,:],
                        cmap='gray',
                        vmax=props['vmax'],
                    )
                    ax.set_title(
                        'Imaging data',
                        fontsize=props['fontsize'],
                    )
                    
                    # Plot masked df/f
                    ax = axs[1]
                    ax.imshow(
                        event_dff[n,:,:],cmap='gray',
                        vmax=event_dff.max(),
                    )
                    ax.set_title(
                        r'Event masked $\frac{\Delta f}{f}$',
                        fontsize=props['fontsize'],
                    )
                    
                    # PLot smoothed df/f
                    ax = axs[2]
                    ax.imshow(
                        smoothed[n,:,:],
                        cmap='gray',
                        vmax=smoothed.max(),
                    )
                    ax.set_title(
                        'Smoothed',
                        fontsize=props['fontsize'],          
                    )
                    
                    # Create line segments for tracked trajectory 
                    xtmp = xs[:n+1,:]
                    z = xtmp[:,1].T
                    y = xtmp[:,0].T
                    points = np.array([z, y]).T.reshape(-1, 1, 2)
                    segments = np.concatenate(
                                   [points[:-1],points[1:]],
                                   axis=1,
                               )
                    
                    # Scale and plot trajectory segments on imaging
                    # data
                    ax = axs[0]
                    lc = LineCollection(
                             (
                                 segments*props['bin_edge_len']
                                 + props['bin_edge_len']*0.5
                             ),
                             cmap=props['cmap'],
                             norm=plt.Normalize(0, N),
                             alpha=props['line_alpha'],
                             zorder=1,
                          )
                    lc.set_array(np.arange(n+1))
                    lc.set_linewidth(props['line_width'])
                    ax.add_collection(lc)
                    ax.scatter(
                        (
                            xs[n,1]*props['bin_edge_len']
                            + props['bin_edge_len']*0.5
                        ),
                        (
                            xs[n,0]*props['bin_edge_len']
                            + props['bin_edge_len']*0.5
                        ),
                        s=props['marker_size'],
                        color=props['marker_color'],
                        zorder=2,
                    )
                    
                    # Plot trajectory segments on masked df/f
                    ax = axs[1]
                    lc = LineCollection(
                             segments,
                             cmap=props['cmap'],
                             norm=plt.Normalize(0, N),
                             alpha=props['line_alpha'],
                             zorder=1,
                         )
                    lc.set_array(np.arange(n+1))
                    lc.set_linewidth(props['line_width'])
                    ax.add_collection(lc)
                    ax.scatter(
                        xs[n,1],
                        xs[n,0],
                        s=props['marker_size'],
                        color=props['marker_color'],
                        zorder=2,
                    )
                    
                    
                    # Plot trajectory segments on smoothed df/f
                    ax = axs[2]
                    lc = LineCollection(
                             segments,
                             cmap=props['cmap'],
                             norm=plt.Normalize(0, N),
                             alpha=props['line_alpha'],
                             zorder=1,
                         )
                    lc.set_array(np.arange(n+1))
                    lc.set_linewidth(props['line_width'])
                    ax.add_collection(lc)
                    ax.scatter(
                        xs[n,1],
                        xs[n,0],
                        s=props['marker_size'],
                        color=props['marker_color'],
                        zorder=2,
                    )
                    
                    # Plot scale bar
                    if props['scale_bar']:
                        ax = axs[0]
                        xx = props['scale_bar_pos'][0]
                        data_to_axis = (
                                           ax.transData
                                           + ax.transAxes.inverted()
                                       )
                        delta_xx = (
                                       (
                                           props['scale_bar_length']
                                           / props['scale']
                                       ),
                                       0
                                   )
                        delta_xx = data_to_axis.transform(delta_xx)[0]
                        xx = np.array(
                                 [
                                     xx,
                                     xx + delta_xx,
                                 ]
                             )
                        yy = props['scale_bar_pos'][1]
                        yy = (
                                 np.array([yy,yy])
                                 + props['scale_bar_width']/2
                             )
                        ax.fill_between(
                            xx,
                            yy-props['scale_bar_width'],
                            yy,
                            linewidth=0,
                            color=props['scale_bar_color'],
                            zorder=1,
                            transform=ax.transAxes,
                        )
                        pltstr = (
                                     f"{props['scale_bar_length']:.0f}"
                                     + props['scale_bar_units']
                                 )
                        ax.text(
                            xx.mean(),
                            yy[0]+props['text_pad'],
                            pltstr,
                            horizontalalignment='center',
                            verticalalignment='bottom',
                            fontsize=props['fontsize'],
                            fontstyle='normal',
                            color=props['scale_bar_color'],
                            zorder=2,
                            transform=ax.transAxes,
                        )

                    # Plot playback rate text
                    if props['playback_annotation']:
                        ax = axs[0]
                        pltstr = (
                                     f"{props['playback_multiplier']}"
                                     r'$ \times$ speed'
                                 )
                        ax.text(
                            props['playback_ann_pos'][0],
                            props['playback_ann_pos'][1],
                            pltstr,
                            horizontalalignment='center',
                            verticalalignment='bottom',
                            fontsize=props['fontsize'],
                            fontstyle='normal',
                            color=props['scale_bar_color'],
                            transform=ax.transAxes,
                        )
                    
                    # Render frame
                    display.clear_output(wait=True) 
                    display.display(pl.gcf())
                    moviewriter.grab_frame()
                    if save_frames:
                        description = [
                                   'img',
                                   'maskeddff',
                                   'smoothed',
                               ]
                        for ax,desc in zip(axs,description):
                            ax.set_title(None)
                            bbox = ax.get_tightbbox(
                                       fig.canvas.get_renderer()
                                       )
                            bbox = bbox.transformed(
                                             fig.
                                             dpi_scale_trans.
                                             inverted()
                                   )
                            savestr = f"{desc}_frame{n:04d}.pdf"
                            fig.savefig(
                                props['frame_path'] / savestr,
                                format='pdf',
                                bbox_inches=bbox,
                            )
    
    # Finish recording movie
    if generate_movie:
        moviewriter.finish()

    return waves


def track_dff_peak_2d(dff,mask,sigma=2):
    """ Track the peak in fluorescence activity for a spatio-temporal
    event over the 2-dimensional imaging area with masking and
    smoothing.
    
    Input
    -----
    dff : array_like, shape (N, m, k)
        dff time series per voxel for the spatio-temporal event,
        arranged as per imaging data. Array has dimensions (N, m, k)
        where N is the duration of the event, m is the number of rows
        in the voxel array and k is the number of columns in the voxel
        array.
    
    mask : array_like, shape (N, m, k)
        binary mask for the spatio-temporal event. Dimensions should
        match the dff array.
    
    sigma : float
        Standard deviation of the 2d Gaussian smoothing kernel.
    
    Returns
    -------
    x : array_like, shape(N, 2)
        Trajectory of the smoothed dff peak.
    
    img : array_like, shape (N, m, k)
        dff after masking and smoothing.
    
    """
    N = mask.shape[0]
    x = np.zeros((N,2))
    img = np.zeros(dff.shape)
    for n in range(N):
        img[n,:,:] = ndimage.gaussian_filter(
                         dff[n,:,:],
                         sigma=sigma,
                         mode='constant',
                         cval=0,
                     )
        x[n,:] = np.unravel_index(
                     img[n,:,:].argmax(),
                     img[n,:,:].shape
                 )
    return x,img
    
