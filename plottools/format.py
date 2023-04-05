"""Misc formatting tools for Matplotlib figures

This module contains miscellaneous tools for formatting plots generated
by Matplotlib.
"""

__version__ = '0.1'
__author__ = 'Michael H McCullough'
__status__ = 'Development'

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import lines


def pvalue_str(
        pval,
        print_as='numeric',
        sig_digits=3
    ):
    """ Format p-value float as readable string or asterisks.
    
    Input
    -----
    pval : float
        p-value to format.
        
    print_as : str, optional
        Format type for plot string. Either 'numeric' or 'asterisk'.
        
    sig_digits : int, optional
        Number of significant digits to render for 'numeric' format.
        Not used for 'asterisk' format.
    
    Returns
    -------
    plotstr : str
        Formatted p-value string.
        
    """
    
    # Input check
    if not (print_as in ['numeric','asterisk']):
        raise ValueError(
                  "Argument print_as must be either"
                  "'numeric' or 'asterisk'" 
              )
    
    if type(sig_digits)!=int:
        raise ValueError(
                  "Argument sig_digits must be integer."
              )
    
    # Format string
    if print_as=='asterisk':
        if pval<0.001:
            plotstr = u"\u2217\u2217\u2217"
        elif pval<0.01:
            plotstr = u"\u2217\u2217"
        elif pval<0.05:
            plotstr = u"\u2217"
        else:
            plotstr = 'n.s.'
    elif print_as=='numeric':
        fstr = r'{0:.' + f'{sig_digits}' + r'f}'
        if pval<10**(-sig_digits):
            plotstr = ''.join(['p<',fstr.format(10**(-sig_digits))])
        else:
            plotstr = ''.join(['p=',fstr.format(pval)])
    
    return plotstr


def plot_overline(
        ax,
        lims,
        height,
        text_offset,
        pval,
        linewidth=1,
        font_size=10,
        color='black',
        print_as="numeric",
        sig_digits=3,
        endcap_length=0.05,
    ):
    """ Plot overline and p-value.

    Input
    -----
    ax : axes (matplotlib.axes._subplots.AxesSubplot)
        Axes on which to plot overline.
        
    lims : array_like, shape (2,)
        Start-point and end-point of the overline
        (i.e [x_start,x_end]).
        
    height : float
        The height of the overline.
        
    text_offset : float
        The offset between the overline and the p-value string.
    
    pval : float
        p-value to format.
    
    linewidth : float, optional
        Width of overline.
    
    font_size : float, optional
        Font size of the p-value string
    
    color : color, optional
        Overline and text color
    
    print_as : str, optional
        Format type for plot string. Either 'numeric' or 'asterisk'.
    
    sig_digits : int, optional
        Number of significant digits to render for 'numeric' format.
        Not used for 'asterisk' format.
    
    endcap_length :  float, optional
        Relative length of overline bracket endcaps.
    
    """
    
    # Input check
    if not (isinstance(ax,mpl.axes.Axes)):
        raise ValueError(
                  "Must pass a valid Matplotlib axes." 
              )
    
    if not isinstance(lims,np.ndarray):
        lims = np.array(lims)
    if not (lims.shape==(2,)):
        raise ValueError(
                  "Argument lims must have shape (2,)."
              )
        
    if not (print_as in ['numeric','asterisk']):
        raise ValueError(
                  "Argument print_as must be either"
                  "'numeric' or 'asterisk'" 
              )
    
    if type(sig_digits)!=int:
        raise ValueError(
                  "Argument sig_digits must be integer."
              )
    
    # Plot overline
#     ax.plot(
#         lims,
#         [height,height],
#         color=color,
#         linewidth=linewidth,
#     )
    
    line = lines.Line2D(
        lims,
        [height,height],
        color=color,
        linewidth=linewidth,
        transform=ax.transData,
    )
    ax.add_line(line)

    
    # Plot overline endcaps
    axis_to_data = ax.transAxes + ax.transData.inverted()
    endcap_scaled = axis_to_data.transform([
                                               (0,0),
                                               (0,endcap_length),
                                           ])[:,1]
    endcap_scaled = np.abs(np.diff(endcap_scaled))
    ax.vlines(
        lims,
        ymin=height-endcap_scaled,
        ymax=height,
        color=color,
        linewidth=linewidth,
    )
    
    
    # Plot p-value string
    plotstr = pvalue_str(
        pval,
        print_as=print_as,
        sig_digits=sig_digits,
    )
    mean_lims = sum(lims)/len(lims)
    ax.text(
        mean_lims,
        height+text_offset,
        plotstr,
        verticalalignment='bottom',
        horizontalalignment='center',
        color=color,
        fontdict={'fontsize':font_size},
    )