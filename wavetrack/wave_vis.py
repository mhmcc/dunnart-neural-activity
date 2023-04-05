"""Wave trajectory visualisation tools

This module contains custom plotting tools for wave trajectory
visualisation based on Matplotlib.

"""

__version__ = '0.1'
__author__ = 'Michael H McCullough'
__status__ = 'Development'

import numpy as np
from scipy.stats import binned_statistic_2d
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection


def wave_trajectories(
        waves,
        timescale_bar=False,
        ax=None,
        **props,
    ):
    """
    
    """
    # Keyword arguments
    defaultprops = {
                       "figsize" : (2,1.5),
                       "xlim" : [],
                       "ylim" : [],
                       "cmap" : cm.viridis,
                       "linewidth" : 1,
                       "linealpha" : 0.5,
                       "scale_bar" : False,
                       "scale_bar_length" : 0,
                       "scale_bar_pos" : [0,0],
                       "scale_bar_width" : 0,
                       "scale_bar_units" : '',
                       "scale" : 0,
                       "scale_bar_color" : 'k',
                       "fontsize" : 12,
                       "text_pad" : 0,
                       "title" : "",
#                        "axcolor" : "k",
                       "timescale_figsize" : (1,1.5),
                       "timescale_labelpad" : -20,
                   }
    defaultprops.update(props)
    props = defaultprops
    
    # Set up figure
    if not ax:
        fig,ax = plt.subplots(figsize=props['figsize'])
        ax.set_position([0, 0, 1, 1])
    else:
        fig = ax.figure
    
    for x in waves:
        z = x[:,1].T + 0.5
        y = (x[:,0].T + 0.5)
        points = np.array([z, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        dur = x.shape[0]
        lc = LineCollection(
                 segments,
                 cmap=props['cmap'],
                 norm=plt.Normalize(0, dur-1),
                 alpha=props['linealpha'],
             )
        lc.set_array(np.arange(dur))
        lc.set_linewidth(props['linewidth'])
        ax.add_collection(lc)
    
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which='both',length=0)
    if props['xlim']:
        # Manually set xlim
        ax.set_xlim(props['xlim'])
    if props['ylim']:
        # Manually set ylim
        ax.set_ylim(props['ylim'])
    if props['title']:
        ax.set_title(props['title'],fontsize=props['fontsize'])
    ax.invert_yaxis()
        
    # Plot scale bar
    if props['scale_bar']:
        xx = props['scale_bar_pos'][0]
        data_to_axis = ax.transData + ax.transAxes.inverted()
        delta_xx = (props['scale_bar_length']/props['scale'],0)
        delta_xx = data_to_axis.transform(delta_xx)[0]
        xx = np.array(
                 [
                     xx,
                     xx + delta_xx,
                 ]
             )
        yy = props['scale_bar_pos'][1]
        yy = np.array([yy,yy])+props['scale_bar_width']/2
        ax.fill_between(
            xx,
            yy-props['scale_bar_width'],
            yy,
            linewidth=0,
            color=props['scale_bar_color'],
            zorder=10,
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
            zorder=11,
            transform=ax.transAxes,
        )
    
    if timescale_bar:
        # Plot separate timescale bar
        fig_ts,ax_ts = plt.subplots(figsize=props['timescale_figsize'])
        sm = plt.cm.ScalarMappable(
                 cmap=props['cmap'],
                 norm=plt.Normalize(0, 1)
             )
        cbar = plt.colorbar(sm,cax=ax_ts)
        cbar.set_ticks([0,1])
        cbar.set_ticklabels(['start','end'])
        cbar.ax.tick_params(labelsize=props['fontsize'])
        cbar.set_label(
            'Relative time',
            fontsize=props['fontsize'],
            labelpad=props['timescale_labelpad']
        )
        cbar.ax.set_position([0.1,0.02,0.1,0.96])
    else:
        fig_ts = None
        ax_ts = None

    return fig, ax, fig_ts, ax_ts


def vector_field(
        X,
        Y,
        K,
        U,
        V,
        xbins,
        ybins,
    ):
    """
    
    """
    nx = xbins.size - 1
    ny = ybins.size - 1
#     x_edges = np.linspace(0,16,nx+1)
#     y_edges = np.linspace(0,12,ny+1)
    u = np.zeros((ny,nx,np.unique(K).shape[0]))
    v = np.zeros((ny,nx,np.unique(K).shape[0]))
    c = np.zeros((ny,nx,np.unique(K).shape[0]))
    for k in np.unique(K):
        idx = K==k
        bs = binned_statistic_2d(X[idx],Y[idx],U[idx],
                                 statistic='mean',bins=[xbins,ybins])
        u[:,:,k] = bs.statistic.T
        bs = binned_statistic_2d(X[idx],Y[idx],V[idx],
                                 statistic='mean',bins=[xbins,ybins])
        v[:,:,k] = bs.statistic.T
        bs = binned_statistic_2d(X[idx],Y[idx],U[idx],
                                 statistic='count',bins=[xbins,ybins])
        c[:,:,k] = bs.statistic.T
            
    # Average over the waves
    u = np.nanmean(u,axis=2)
    v = np.nanmean(v,axis=2)
    c = (c>0).sum(axis=2)

    x = bs.x_edge
    x = (x[:-1]+x[1:])/2
    y = bs.y_edge
    y = (y[:-1]+y[1:])/2

    return x,y,u,v,c


def vector_field_plot(
        X,
        Y,
        U,
        V,
        C,
        histscale_bar=False,
        ax=None,
        **props,
    ):
    """
    
    """
    # Keyword arguments
    defaultprops = {
                       "figsize" : (2,1.5),
                       "xlim" : [],
                       "ylim" : [],
                       "cmap" : cm.Oranges,
                       "norm" : None,
                       "quiver_scale" : 0.1,
                       "quiver_width" : 0.02,
                       "linewidth" : 1,
                       "linealpha" : 0.5,
                       "scale_bar" : False,
                       "scale_bar_length" : 0,
                       "scale_bar_pos" : [0,0],
                       "scale_bar_width" : 0,
                       "scale_bar_units" : '',
                       "scale" : 0,
                       "scale_bar_color" : 'k',
                       "fontsize" : 12,
                       "text_pad" : 0,
                       "title" : "",
#                        "axcolor" : "k",
                       "histscale_figsize" : (1,1.5),
                       "histscale_labelpad" : -10,
                   }
    defaultprops.update(props)
    props = defaultprops
    
    # Set up figure
    if not ax:
        fig,ax = plt.subplots(figsize=props['figsize'])
        ax.set_position([0, 0, 1, 1])
    else:
        fig = ax.figure
    
        fig,ax = plt.subplots(figsize=(4,3),tight_layout=False)

    # Plot quiver
    X,Y = np.meshgrid(X,Y)
    ax.quiver(
        X,
        -Y,
        U,
        -V,
        C,
        norm=props['norm'],
        cmap=props['cmap'],
        scale_units='x',
        scale=props['quiver_scale'],
        width=props['quiver_width'],
    )
        
    # Format
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which='both',length=0)
    if props['xlim']:
        # Manually set xlim
        ax.set_xlim(props['xlim'])
    if props['ylim']:
        # Manually set ylim
        ax.set_ylim(props['ylim'])
    if props['title']:
        ax.set_title(props['title'],fontsize=props['fontsize'])
        
    # Plot scale bar
    if props['scale_bar']:
        xx = props['scale_bar_pos'][0]
        data_to_axis = ax.transData + ax.transAxes.inverted()
        delta_xx = (props['scale_bar_length']/props['scale'],0)
        delta_xx = data_to_axis.transform(delta_xx)[0]
        xx = np.array(
                 [
                     xx,
                     xx + delta_xx,
                 ]
             )
        yy = props['scale_bar_pos'][1]
        yy = np.array([yy,yy])+props['scale_bar_width']/2
        ax.fill_between(
            xx,
            yy-props['scale_bar_width'],
            yy,
            linewidth=0,
            color=props['scale_bar_color'],
            zorder=10,
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
            zorder=11,
            transform=ax.transAxes,
        )

    if histscale_bar:
        # Plot separate timescale bar
        fig_hs,ax_hs = plt.subplots(figsize=props['histscale_figsize'])
        sm = plt.cm.ScalarMappable(
                 cmap=props['cmap'],
                 norm=plt.Normalize(0, 1)
             )
        cbar = plt.colorbar(sm,cax=ax_hs)
        cbar.set_ticks([0,1])
        cbar.set_ticklabels([0,props['norm'].vmax])
        cbar.ax.tick_params(labelsize=props['fontsize'])
        cbar.set_label(
            'Number of\nevents',
            fontsize=props['fontsize'],
            labelpad=props['histscale_labelpad']
        )
        cbar.ax.set_position([0.1,0.02,0.1,0.96])
    else:
        fig_hs = None
        ax_hs = None

    return fig, ax, fig_hs, ax_hs