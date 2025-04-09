import numpy as np
import matplotlib.pyplot as plt

def dim_2_network_topology_axes_formatter(
        ax: plt.axes,
        core_square: np.ndarray,
        core_square_color: str,
        core_square_linewidth: float,
        xlim: np.ndarray,
        ylim: np.ndarray,
        xticks: np.ndarray,
        yticks: np.ndarray,
        xlabel: str,
        ylabel: str,
        grid_alpha: float,
        grid_zorder: int) -> plt.axes:
    ax.plot(
        core_square[:, 0], core_square[:, 1],
        color=core_square_color, linewidth=core_square_linewidth)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=grid_alpha, zorder=grid_zorder)
    return ax

def dim_3_network_topology_axes_formatter(
        ax: plt.axes,
        core_cube: np.ndarray,
        core_cube_color: str,
        core_cube_linewidth: float,
        xlim: np.ndarray,
        ylim: np.ndarray,
        zlim: np.ndarray,
        xticks: np.ndarray,
        yticks: np.ndarray,
        zticks: np.ndarray,
        xlabel: str,
        ylabel: str,
        zlabel: str,
        grid_alpha: float,
        grid_zorder: int) -> plt.axes:
    for face in np.arange(6):
        ax.plot(
            core_cube[face, :, 0], core_cube[face, :, 1], core_cube[face, :, 2],
            color=core_cube_color, linewidth=core_cube_linewidth)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks(zticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.grid(True, alpha=grid_alpha, zorder=grid_zorder)
    return ax
