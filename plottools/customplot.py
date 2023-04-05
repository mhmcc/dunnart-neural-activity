"""Custom plot functions

This module contains custom plotting and data visualisation routines
based on Matplotlib.

"""

__version__ = '0.1'
__author__ = 'Michael H McCullough'
__status__ = 'Development'

import numpy as np
from scipy.stats import sem,ranksums
import matplotlib.pyplot as plt
from matplotlib import cm, lines
from plottools.format import plot_overline


def bar_plot3(
       x,
       y,
       scatter_width=0.05,
       bar_width=0.6,
       pval_format="asterisk",
       **props,
    ):
    """ Bar plot for 3 ordered samples with SEM error bars and pairwise
    statistical tests.
    
    Input
    -----
    x : array_like, shape (n,)
        Group labels for the samples. Must be integers.
    
    y : array_like, shape (n,)
        Observation to be plotted.
    
    scatter_width : float, optional
        Standard deviation of point scatter in x-direction.
    
    bar_width : float, optional
        Width of bars.
        
    props : dict, optional
        Dictionary of figure and axes properties.
    
    Returns
    -------
    fig : figure (matplotlib.figure.Figure)
        Figure handle to plot
    
    ax : axes (matplotlib.axes._subplots.AxesSubplot)
        Axes handle to plot
    
    """
    # Input check
    cond1 = np.unique(x).size != 3
    if cond1:
        raise ValueError(
                  'There must be 3 unique group labels in x.'
              )
    cond2 = x.shape != y.shape
    if cond2:
        raise ValueError(
                  'x and y must have the same shape (n,).'
              )
    cond3 = (x.ndim != 1) or (y.ndim != 1)
    if cond3:
        raise ValueError(
                  'x and y must both be 1d arrays with shape (n,).'
              )
    
    # Keyword arguments
    defaultprops = {
                       "figsize" : (1.5,1.5),
                       "xlim" : [-0.5, 2.5],
                       "ymax" : None,
                       "xlabel": '',
                       "ylabel": '',
                       "xticklabels" : np.unique(x),
                       "cmap" : cm.Set1(range(3)),
                       "fontsize" : 12,
                       "linewidth" : 1,
                       "overline_height" : 0.7, # ax.transAxes
                       "overline_vspacing" : 0.14, # ax.transAxes
                       "overline_pad" : 0.05, # ax.transAxes
                       "axcolor" : 'k',
                   }
    defaultprops.update(props)
    props = defaultprops
    
    # Set up figure
    fig,ax = plt.subplots(figsize=props['figsize'],tight_layout=False)

    # Plot
    grps = np.unique(x)
    z = [] # stores data for statistical tests
    for j,grp in enumerate(grps):
        idx = x==grp
        
        # Scatter datapoints
        x_scatter = j*np.ones_like(y[idx])
        x_scatter += (np.random.randn(y[idx].size))*scatter_width
        ax.scatter(
            x_scatter,
            y[idx],
            color=props['cmap'][j,:],
        )
        
        # Plot bars
        ymean = y[idx].mean()
        yerr = sem(y[idx])
        ax.bar(
            x=j,
            height=ymean,
            yerr=yerr,
            width=bar_width,
            color='None',
            edgecolor=props['axcolor'],
            ecolor=props['axcolor'],
            linewidth=props['linewidth'],
            error_kw={'linewidth':props['linewidth']},
        )
        z.append(y[idx])
        
    # Format figure
    ax.set_xticks(np.arange(3))
    ax.set_xlim(props['xlim'])
    if props['ymax']:
        ymax = props['ymax']
    else:
        ymax = max([max(item) for item in z])
    ax.set_ylim(
        [
            0,
            ymax*1.1/(props['overline_height']-props['overline_pad'])
        ]
    )
    xticklabels = [
                      f'{item}'
                      for item
                      in np.unique(x)
                  ]
    ax.set_xticklabels(xticklabels)    
    ax.set_xlabel(props['xlabel'],fontsize=props['fontsize'])
    ax.set_ylabel(props['ylabel'],fontsize=props['fontsize'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_bounds(0, ymax*1.1)
    yticks = [yt for yt in ax.get_yticks() if yt <= ymax*1.1]
    ax.set_yticks(yticks)
    ax.tick_params(axis='both',labelsize=props['fontsize'])
    
    # Plot overlines 
    axis_to_data = ax.transAxes + ax.transData.inverted()
    pairs = [(0,1),(1,2),(0,2)]
    for j,pr in enumerate(pairs):
        _,pval = ranksums(z[pr[0]],z[pr[1]])
        lineheight = (
                         props['overline_height']
                         + props['overline_vspacing']*j
                     )
        lineheight = axis_to_data.transform([0,lineheight])[1]
        plot_overline(
            ax=ax,
            lims=[pr[0],pr[1]],
            height = lineheight,
            text_offset=0,
            pval=pval,
            color=props['axcolor'],
            print_as=pval_format,
            font_size=props['fontsize']
        )
        
    return fig,ax


def stacked_histogram(
        x,
        z,
        xbins,
        zbins,
        polar=False,
        legend=False,
        **props,
    ):
    """ Stacked histogram plot
    
    Input
    -----
    x : array_like, shape(n,)
        Sample data.
    
    z : array_like, shape(n,)
        Grouping data, paired with x.
        
    xbins : array_like, shape(m,)
        Bin edges for x.
        
    zbins : array_like, shape(k,)
        Bin edges for y.
    
    polar : bool, optional
        Flag for plotting to a polar projection. This creates a
        windrose plot.
        
    legend : bool, optional
        Flag for plotting legend.
        
    props : dict, optional
        Dictionary of figure and axes properties.
        
    
    Returns
    -------
    fig : figure (matplotlib.figure.Figure)
        Figure handle to plot
    
    ax : axes (matplotlib.axes._subplots.AxesSubplot)
        Axes handle to plot
    
    """
    
    # Keyword arguments
    defaultprops = {
                       "figsize" : (2,2),
                       "xlim" : [],
                       "ylim" : [],
                       "xlabel" : '',
                       "ylabel" : '',
                       "xtick_rotation" : 45,
                       "cmap" : cm.magma,
                       "zlabel" : '',
                       "zunits" : '',
                       "legend_pos" : (1.1, 0.5),
                       "fontsize" : 12,
                   }
    defaultprops.update(props)
    props = defaultprops
    
    n_val = x.shape[0]
    
    # Plot parameters
    bar_width = np.ptp(xbins)/(xbins.shape[0]-1)
    cmap = props['cmap'](np.linspace(0,1,zbins.shape[0]))[1:]
    
    # Compute 2d histogram
    h,xedges,zedges = np.histogram2d(x,z,bins=[xbins,zbins])
    
    # Create axes
    if polar:
        fig,ax = plt.subplots(
                 figsize=props['figsize'],
                 subplot_kw=dict(projection='polar'),
                 constrained_layout=True,
             )
    else:
        fig,ax = plt.subplots(figsize=props['figsize'])
    
    # Compute bin centres
    bin_centres = (xedges[:-1] + xedges[1:])/2
    
    # Loop over the z dimension to plot
    for i in range(zedges.shape[0]-1):
        if i==0:
            bottom=None
        else:
            bottom=np.sum(h[:,:i],axis=1)
        heights = h[:,i]
        label_str = (
                        '{0:.0f} - {1:.0f}'
                        .format(zedges[i],zedges[i+1])
                        + props['zunits']
                    )
        ax.bar(
            bin_centres,
            height=heights,
            bottom=bottom,
            width=bar_width,
            color=cmap[i,:],
            label=label_str,
            edgecolor='none',
        )
    
    # Format
    if props['xlim']:
        # Manually set xlim
        xlim = props['xlim']
    else:
        xlim = (xbins.min(),xbins.max())
    ax.set_xlim(xlim)
    if props['ylim']:
        # Manually set ylim
        ylim = props['ylim']
        ax.set_ylim(ylim)
    ax.set_xticks(xbins[::2])
    ax.set_xticklabels(
        ax.get_xticks().astype(int),
        rotation=45,
        fontsize=props["fontsize"],
    )
    ax.set_yticks(ax.get_yticks().tolist())
    ax.set_yticklabels(
        ax.get_yticks().astype(int),
        rotation=0,
        fontsize=props["fontsize"],
    )
    ax.set_ylabel(props['xlabel'],fontsize=props["fontsize"])
    ax.set_xlabel(props['ylabel'],fontsize=props["fontsize"])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if legend:
        hleg = ax.legend(
                   loc='center left',
                   bbox_to_anchor=props['legend_pos'],
                   frameon=False,
                   fontsize=props['fontsize'],
               )
        hleg.set_title(
            props['zlabel'],
            prop={'size':props['fontsize']}
        )

    return fig,ax

