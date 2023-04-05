"""Network statistics for voxel-based functional connectivity analysis

This module contains various functions and utilities for measuring
properties of functional connectivity networks estimated from calcium
imaging data.

Builds on the networkx package:

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

import numpy as np
import networkx as nx
import warnings


def average_compactness(
        c,
        pos,
        nrows,
        ncols,
    ):
    """ Measure average spatial compactness of communities where nodes
    have a known 2-dimensional spatial grid layout, independent of
    connectivity.
    
    Input
    -----
    c : list
        List of communities. Each list element contains a list of node
        labels for the corresponding community.
        
    pos : array_like, shape (N,2)
        Spatial positions of each of the N nodes in the 2-dimensional
        grid.
        
    nrows : int
        Number of rows in the grid
        
    ncols : int
        Number of columns in the grid
    
    Returns
    -------
    compactness : float
        Average (mean) compactness over the set of communities in the
        network. Returns NaN if c contains no communities.
    
    """
    # Return nan for no clusters
    if len(c)==0:
        warnings.warn('c contains no communities. Returning NaN.')
        return np.nan
    
    cmp = []
    for cluster in c:
        cmp.append(
                community_grid_compactness(
                    cluster,
                    pos,
                    nrows,
                    ncols,
                )
            )
    compactness = np.mean(cmp)
    
    return compactness


def community_grid_compactness(
        comm,
        pos,
        nrows,
        ncols,
    ):
    """ Measures spatial compactness of a network community normalised
    by a minimal configuration on a 2d 4-connected lattice graph.
    
    Input
    -----
    comm : list
        List of node labels for the community.
        
    pos : array_like, shape (N,2)
        Spatial positions of each of the N nodes in the complete
        network in the 2-dimensional grid.
        
    nrows : int
        Number of rows in the grid
        
    ncols : int
        Number of columns in the grid
    
    Returns
    -------
    cmp : float
        Normalised spatial compactness over of the community.
        
    """
    
    if comm==[]:
        raise ValueError(
                  'Community comm is empty.'
              )
    
    # Create 2d lattice network
    # Note that lattice includes additional boundary nodes beyond the
    # observed area such that the cluster subgraph boundary is
    # correctly computed. 
    H = nx.grid_2d_graph(nrows+2,ncols+2)
    
    # Relabel nodes
    node_label = 0
    for i in range(1,nrows+1):
        for j in range(1,ncols+1):
            mapping = {(i,j): node_label}
            H = nx.relabel_nodes(H, mapping, copy=False)
            node_label+=1
    
    # Compute border length of optimal (minimal) configuration
    size = len(comm)
    rows=1
    cols=1
    while (rows*cols)<size:
        if rows==cols:
            cols+=1
        else:
            rows+=1
    optimal = 2*(rows+cols)
    out_connected_edges = []
    
    # Compute border length of actual configuration
    for node in comm:
        N = list(nx.neighbors(H,node))        
        out_connected_edges += list(set(N) - set(comm))
    actual = len(out_connected_edges)   
    
    # Compute compactness measure
    cmp = actual/optimal
    
    return(cmp)


def edge_weights_inverse(G):
    """ Compute inverse edge weights for network.
    
    Input
    -----
    G : networkx.classes.graph.Graph
        A networkx graph.

    Returns
    -------
    G : networkx.classes.graph.Graph
        Same as input G but with additional node property inv_weights.
    
    """
    
    for e in G.edges:
        w = G[e[0]][e[1]]['weight']
        G[e[0]][e[1]]['inv_weight'] = 1/w
    return G


def list_communities(p,keep_all=False):
    """ Convert networkx dict of communities to list.
    
    Input
    -----
    p : dict
        Dictionary where keys are network nodes and values are
        community indicies.
    
    keep_all : bool, optional
        If keep_all then then include communities with only one node
        otherwise only keep communities with 2 or more nodes.
        
    Returns
    -------
    c : list
        List of communities. Each list element contains a list of node
        labels for the corresponding community.
    """
    
    c = [] # assemblies (partition elements with size of at least 2)
    for j in range(max(p.values())+1):
        comm = [n for n in p if p[n]==j]
        if len(comm)>1:
            c.append(comm)
    return c