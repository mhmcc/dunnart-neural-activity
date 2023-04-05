"""Network visualisation tools

This module contains tools for network visualisation created for the
investigation of functional connectivity networks from voxel-based
analysis of calcium imaging data.

Some functions build on the networkx package:

@InProceedings{SciPyProceedings_11,
    author      = {Aric A. Hagberg and Daniel A. Schult and Pieter J.
                      Swart},
    title       = {Exploring Network Structure, Dynamics, and Function
                      using NetworkX},
    booktitle   = {Proceedings of the 7th Python in Science Conference},
    pages       = {11 - 15},
    address     = {Pasadena, CA USA},
    year        = {2008},
    editor      = {Ga\"el Varoquaux and Travis Vaught and Jarrod
                      Millman},
}

"""

__version__ = '0.1'
__author__ = 'Michael H McCullough'
__status__ = 'Development'

from itertools import chain
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import itertools


def adjacency_communities(
        G,
        c,
        cmap=np.empty(0),
        vmin=None,
        vmax=None,
        ax=None,
        **props,
    ):
    """ Plot network adjacency matrix with coloured communities.
    
    Input
    -----
    G : networkx.classes.graph.Graph
        A networkx graph.
        
    c : list
        List of communities. Each list element contains a list of node
        labels for the corresponding community.
    
    cmap : array_like, shape (M,4), optional
        RGBA array. Each row corresponds to a community.
        
    vmin : float, optional
        Minimum value for adjacency matrix visualisation.
        
    vmax : float, optional
        Maximum value for adjacency matrix visualisation.
    
    ax : axes (matplotlib.axes._subplots.AxesSubplot), optional
        Axes handle to plot.
        
    props : dict, optional
        Dictionary of figure and axes properties.
    
    Returns
    -------
    fig : figure (matplotlib.figure.Figure)
        Figure handle to plot.
    
    ax : axes (matplotlib.axes._subplots.AxesSubplot)
        Axes handle to plot.
    
    """
    # Keyword arguments
    defaultprops = {
                       "figsize" : (6,4),
                       "zlabel" : "$w_{i,j}$",
                       "fontsize" : 12,
                       "labelpad" : 12,
                   }
    defaultprops.update(props)
    props = defaultprops
    
    # Set up figure and axes
    if not ax:
        fig,ax = plt.subplots(figsize=props['figsize'])
        ax2 = inset_axes(
                  ax,
                  width="5%",  # width = 8% of parent_bbox width
                  height="100%",  # height : 50%
                  loc='center left',
                  bbox_to_anchor=(-0.075, 0, 1, 1),
                  bbox_transform=ax.transAxes,
                  borderpad=0,
              )
    else:
        fig = ax.figure
    
    # Get adjacency
    A = nx.to_numpy_array(G)
    
    # Sort adjacency by communities
    idx = list(chain.from_iterable(c))
    A = A[np.ix_(idx,idx)]
    
    # Plot adjacency matrix
    im = ax.imshow(
             A,
             vmin=vmin,
             vmax=vmax,
             interpolation='none',
             cmap=cm.gray
         )

    # Specify colormap
    if cmap.size==0:
        cmap = cm.tab20(range(len(c)))
        
    # Plot community structure
    offset = -0.5
    for i,cc in enumerate(c):
        # Plot bar on y-axis
        ax2.fill_betweenx(
            [offset,offset+len(cc)],
            [0,0],
            [1,1],
            facecolor=cmap[i,:],
            alpha=1,
            linewidth=0,
        )
        # Plot transparent overlay 
        ax.fill_between(
            [offset,offset+len(cc)],
            [offset,offset],
            [offset+len(cc),offset+len(cc)],
            A.shape,
            facecolor=cmap[i,:],
            alpha=0.2,
            linewidth=0
        )
        offset+=len(cc)
    
    # Format adjacency image
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    
    # Format communities axis
    ax2.set_xlim([0,1])
    ax2.set_ylim(ax.get_ylim())

    ax2.axis('off')
    
    axins = inset_axes(
                ax,
                width="8%",  # width = 8% of parent_bbox width
                height="50%",  # height : 50%
                loc='center left',
                bbox_to_anchor=(1.05, 0., 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )
    cbar = fig.colorbar(im,ax=ax,cax=axins)
    cbar.outline.set_visible(False)
    if not vmin:
        vmin = A.min()
    if not vmax:
        vmax = A.max()
    cticks = np.linspace(vmin,vmax,5)
    labels = [f'{item:.2}' for item in cticks]
    if vmin>A.min():
        labels[0] = r'$\leq$' + labels[0] 
    if vmax<A.max():
        labels[-1] = r'$\geq$' + labels[-1] 
    cbar.set_ticks(cticks)
    cbar.ax.set_yticklabels(labels,rotation=0)
    cbar.set_label(
        props['zlabel'],
        rotation=0,
        fontsize=props['fontsize'],
        labelpad=props['labelpad'],
    )
    
    return fig, ax

def community_map(
        c,
        pos,
        cmap=np.empty(0),
        img=np.empty(0),
        ax=None,
        **props,
    ):
    """ Plot spatial map of network community structure.
    
    Input
    -----
    c : list
        List of communities. Each list element contains a list of node
        labels for the corresponding community.
        
    pos : array_like, shape (N,2)
        Array specifying node positions (row and column voxel
        indicies)
    
    cmap : array_like, shape (M,4), optional
        RGBA array. Each row corresponds to a community.
    
    img : array_like, optional
        Grayscale image to render under the community map plot.
    
    ax : axes (matplotlib.axes._subplots.AxesSubplot)
        Axes handle to plot
        
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
                       "figsize" : (4,3),
                       "xlim" : (pos[:,0].min(),pos[:,0].max()+1),
                       "ylim" : (pos[:,1].max()+1,pos[:,1].min()),
                       "node_size" : 250,
                       "node_alpha" : 1,
                       "othernode_alpha" : 0.1,
                       "vmin" : None, # image vmin
                       "vmax" : None, # image vmax
                       "scale_bar" : False,
                       "scale_bar_length" : 0,
                       "scale_bar_pos" : [0,0],
                       "scale_bar_width" : 0,
                       "scale_bar_units" : '',
                       "scale" : 0,
                       "scale_bar_color" : 'k',
                       "fontsize" : 12,
                       "text_pad" : 0,
                   }
    defaultprops.update(props)
    props = defaultprops

    # Set up figure
    if not ax:
        fig,ax = plt.subplots(figsize=props['figsize'],frameon=False)
        ax.set_aspect('equal')
        ax.set_position([0, 0, 1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(length=0)
        ax.set_xlim(props['xlim'])
        ax.set_ylim(props['ylim'])
    else:
        fig = ax.figure
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(length=0)
    ax.set_xlim(props['xlim'])
    ax.set_ylim(props['ylim'])
    
    # Plot background image
    if img.size>0:
        ax.imshow(
            img,
            cmap='gray',
            extent= np.hstack([props['xlim'],props['ylim']]),
            vmin=props['vmin'],
            vmax=props['vmax'],
        )
    
    # Specify colormap
    if cmap.size==0:
        cmap = cm.tab20(range(len(c)))
    
    # Plot nodes in communites
    plotted = []
    for i,nodes in enumerate(c):
        x = pos[nodes][:,0]+0.5
        y = pos[nodes][:,1]+0.5
        ax.scatter(
            x,
            y,
            s=props['node_size'],
            color=cmap[i,:],
            edgecolors='none',
            alpha=props['node_alpha'],
        )
        plotted+=nodes
    
    # Plot nodes not in communities
    other_nodes = [
                      node
                      for node in range(pos.shape[0])
                      if not node in plotted
                  ]
    if (img.size==0) and (other_nodes):
        x = pos[other_nodes][:,0]+0.5
        y = pos[other_nodes][:,1]+0.5
        ax.scatter(
            x,
            y,
            s=props['node_size'],
            color='gray',
            edgecolors='none',
            alpha=props['othernode_alpha']
        )
    
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
        
    return fig, ax


def network_embedding(
        G,
        c,
        cmap=np.empty(0),
        seed=None,
        ax=None,
        **props,
    ):
    """ Plot 2-dimensional spring network embedding with nodes coloured
    by communities.
    
    Input
    -----
    G : networkx.classes.graph.Graph
        A networkx graph.
        
    c : list
        List of communities. Each list element contains a list of node
        labels for the corresponding community.
    
    cmap : array_like, shape (M,4), optional
        RGBA array. Each row corresponds to a community.
        
    seed : int, optional
        Random seed for spring embedding.
    
    ax : axes (matplotlib.axes._subplots.AxesSubplot), optional
        Axes handle to plot.
        
    props : dict, optional
        Dictionary of figure and axes properties.
    
    Returns
    -------
    fig : figure (matplotlib.figure.Figure)
        Figure handle to plot.
    
    ax : axes (matplotlib.axes._subplots.AxesSubplot)
        Axes handle to plot.
        
    """
    # Keyword arguments
    defaultprops = {
                       "figsize" : (4,4),
                       "node_size" : 50,
                       "node_alpha" : 1,
                       "othernode_alpha" : 1,
                       "edge_alpha" : 0.2,
                   }
    defaultprops.update(props)
    props = defaultprops
    
    # Set up figure
    if not ax:
        fig,ax = plt.subplots(figsize=props['figsize'])
        ax.set_position([0, 0, 1, 1])
    else:
        fig = ax.figure
    ax.axis('off')
    
    # Specify colormap
    if cmap.size==0:
        cmap = cm.tab20(range(len(c)))
    
    # Plot network
    pos = nx.spring_layout(G,seed=seed)
    nx.draw_networkx_edges(G,pos=pos,alpha=0.2,edge_color='gray')

    # Plot nodes not in communities
    nodes = list(itertools.chain(*c))
    other_nodes = [
                      node
                      for node in range(max(nodes))
                      if node not in nodes
                  ]
    
    if other_nodes:
        nx.draw_networkx_nodes(
            G,
            ax=ax,
            pos=pos,
            node_size=props['node_size'],
            nodelist=other_nodes,
            node_color='darkgray',
            edgecolors=None,
            alpha=props['othernode_alpha'],
        )
    
    # Plot nodes in communities
    for i,nodes in enumerate(c):
        nx.draw_networkx_nodes(
            G,
            ax=ax,
            pos=pos,
            node_size=props['node_size'],
            nodelist=nodes,
            node_color=cmap[i,:].reshape(1,4),
            edgecolors=None,
            alpha=props['node_alpha'],
        )
        
    return fig, ax

