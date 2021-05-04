import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib

from typing import Tuple, List, NewType
from functools import reduce

Figure = NewType('Figure', matplotlib.figure.Figure)
Axis   = NewType('Figure', matplotlib.axes.Axes)

l_multiply = lambda x, y: x * y
l_sum      = lambda x, y: x + y

def stacked_frames(num_rows: int, num_cols: int, size: Tuple[int],
                   names_left: List[str] = None,
                   names_right: List[str] = None,
                   x_axis_name: str = None,
                   title: str = None,
                   names_size: int = 15,
                   title_size: int = 20)         -> Tuple[Figure, Axis]:
    fig, axs = plt.subplots(nrows = num_rows, ncols = num_cols)
    fig.set_size_inches(size)
    fig.subplots_adjust(hspace = 0.01)

    # remove the x - axis from all frames, minus the bottom one
    for ax in axs[:-1]:
        ax.xaxis.set_major_formatter(plt.NullFormatter())
        
    # remove the y - axis from all frames
    for ax in axs:
        ax.yaxis.set_major_formatter(plt.NullFormatter())
        ax.set_yticks([])
        
    # limiting the number of ticks in the bottom axis
    axs[-1].locator_params('x', nbins = 4)
    
    # setting names at the left side of each frame
    if names_left != None and len(names_left) == len(axs):
        for ax, name in zip(axs, names_left):
            ax.text(-.03, 0.5, name, fontsize = names_size, rotation = "vertical", 
                    transform = ax.transAxes, va = 'center', family = 'serif')
            
    if names_right != None and len(names_right) == len(axs):
        for ax, name in zip(axs, names_right):
            ax.text(1.01, 0.5, name, fontsize = names_size, rotation = "vertical",
                    transform = ax.transAxes, va = 'center', family = 'serif')
    
    # setting name at the bottom of the last frame
    if x_axis_name != None:
        axs[-1].set_xlabel(x_axis_name, fontsize = names_size, family = 'serif')    
        
    # setting title
    if title != None:
        axs[0].set_title(title, fontsize = 20, family = 'serif')
        
    return fig, axs

def grid_frames(num_rows: int, num_cols: int, size: Tuple[int] = None,
                spacing: Tuple[float] = None,
                remove_all_axis: bool = False,
                x_names: List[str] = None,
                y_names: List[str] = None,
                axs_titles: List[str] = None,
                title: str = None,
                names_size: int = 12,
                title_size: int = 18) -> Tuple[Figure, Axis]:
    fig, axs = plt.subplots(nrows = num_rows, ncols = num_cols)
    if size == None: 
        fig.set_size_inches(3*num_cols, 2.5*num_rows)
    else: 
        fig.set_size_inches(size)

    if spacing == None:
        fig.subplots_adjust(hspace = max(0.35, 0.15*num_rows),
                            wspace = max(0.5, 0.07*num_cols))
    else:
        fig.subplots_adjust(spacing)
    
    # Remove or format the axis's ticks
    if remove_all_axis:
        # remove the y - axis from all frames
        for ax in axs.flatten():
            ax.yaxis.set_major_formatter(plt.NullFormatter())
            ax.set_yticks([])
            ax.xaxis.set_major_formatter(plt.NullFormatter())
            ax.set_xticks([])
    else:
        # limiting the number of ticks in the bottom axis
        for ax in axs.flatten():
            ax.locator_params('x', nbins = 4)
            ax.locator_params('y', nbins = 4)
            ax.xaxis.set_tick_params(labelsize = 7)
            ax.yaxis.set_tick_params(labelsize = 7)
            ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%g'))
            ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%g'))
    
#     # Set titles for the rows - left
#     if left_side_names != None and len(left_side_names) == axs.shape[0]:
#         for ax_rows, name in zip(axs, left_side_names):
#             ax_rows[0].text(.8, .5, name, fontsize = names_size, rotation = "vertical", 
#                             transform = ax.transAxes, va = 'center', family = 'serif')
    
#     # Set titles for the rows - right
#     if right_side_names != None and len(right_side_names) == axs.shape[0]:
#         for ax_rows, name in zip(axs, right_side_names):
#             ax_rows[-1].text(1.1, 0.5, name, fontsize = 25, rotation = "vertical", 
#                              transform = ax.transAxes, va = 'center', family = 'serif')
        
    # Set x labels
    if x_names != None and len(x_names) == reduce(l_multiply, axs.shape):
        for ax, name in zip(axs.flatten(), x_names):
            ax.set_xlabel(name, fontsize = names_size, family = 'serif') 
        
    # Set y labels
    if y_names != None and len(y_names) == reduce(l_multiply, axs.shape):
        for ax, name in zip(axs.flatten(), y_names):
            ax.set_ylabel(name, fontsize = names_size, family = 'serif')
    
    # Set title
    if title != None:
        fig.suptitle(title, fontsize= title_size)
    
    return fig, axs

def simple_plot(size: Tuple[int]) -> Tuple[Figure, Axis]:
    fig, ax = plt.subplots(1, 1)
    
    if size != None:
        fig.set_size_inches(size)
    else:
        fig.set_size_inches(w = 10, h = 10)
        
def heat_plot(X: np.array, size: Tuple[int] = None,
              x_tick_labels: List[str] = None,
              y_tick_labels: List[str] = None,
              cmap: str = None) -> Tuple[Figure, Axis]:
    fig, ax = plt.subplots(1, 1)
    
    if size != None:
        fig.set_size_inches(size)
    else:
        fig.set_size_inches(w = 15, h = 10)
    
    sns.heatmap(X, linewidth = 0, ax = ax, cmap = cmap)
    
    ax.locator_params('x', nbins = 3)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.0e'))
    
    if x_tick_labels != None:
        ax.set_xticklabels(x_tick_labels)
    if y_tick_labels != None:
        ax.set_yticklabels(y_tick_labels)
    
    return fig, ax