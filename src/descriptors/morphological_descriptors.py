import numpy as np
from scipy.optimize import curve_fit
from src.helpers.simulation_box_utils import A_or_V_arg_L_func
from src.helpers.network_utils import rho_func
from src.helpers.network_topology_initialization_utils import (
    core_node_tessellation,
    box_neighborhood_id
)
from src.descriptors.general_topological_descriptors import l_naive_func

def exp_decay_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Exponential decay function.

    Exponential decay function.

    Args:
        x (np.ndarray): Independent variable.
        a (float): Exponential pre-factor.
        b (float): Exponential decay.
    
    Returns:
        np.ndarray: Exponential decay function.
    
    """
    return a * np.exp(-x/b)

def g_r_nrmlzd_func(
        b: float,
        L: np.ndarray,
        coords: np.ndarray,
        core_nodes: np.ndarray,
        inv_bin_width: int=100,
        inv_bin_shift: int=10) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Radial distribution function.

    This function constructs the radial distribution function for a
    given set of nodes in a simulation box.

    Args:
        b (float): Chain segment and/or cross-linker diameter.
        L (np.ndarray): Simulation box side lengths.
        coords (np.ndarray): Coordinates of the core nodes.
        core_nodes (np.ndarray): Core nodes.
        inv_bin_width (int): Inverse of the radial domain defining the bin width. Default value is 100 (and the bin width is 1/100th of the radial domain).
        inv_bin_shift (int): Inverse of the radial increment defining the bin shift. Default value is 10 (and the bin shift is 1/10th of the bin width).
    
    Returns:
        tuple[np.ndarray, np.ndarray, float, float]: Radial distribution
        function, normalized radial distance bins, normalized bin
        width/radial increment, and normalized shift increment.
    
    """
    # Simulation box parameters
    n, dim = np.shape(coords)
    rho = rho_func(n, A_or_V_arg_L_func(L))
    box_center = L / 2
    L_min = np.min(L)
    r_max = L_min / 2

    # Initialize bin width/radial increment, radial distance bins, and
    # the radial distribution function histogram
    dr = (r_max-b) / inv_bin_width
    bin_shift = dr / inv_bin_shift
    num_bins = inv_bin_shift * (inv_bin_width-1) + 1
    start_bins = np.linspace(b, r_max-dr, num_bins)
    end_bins = start_bins + dr
    bins = np.stack((start_bins, end_bins), axis=1)
    r_bins = bins[:, 0] + dr / 2
    g_r = np.zeros(num_bins)
    
    # Initialize normalization shell volumes
    if dim == 2:
        shell_V = np.pi * ((r_bins+(dr/2))**2-(r_bins-(dr/2))**2)
    elif dim == 3:
        shell_V = 4 / 3 * np.pi * ((r_bins+(dr/2))**3-(r_bins-(dr/2))**3)

    # Determine the core nodes in the simulation box about which to
    # compute the radial distribution function
    g_r_box_indcs, g_r_box_n = box_neighborhood_id(
        dim, coords, box_center, r_max, inclusive=True, indices=True)
    g_r_box_nodes = core_nodes[g_r_box_indcs]

    # Tessellate the core node coordinates
    tsslltd_coords, _ = core_node_tessellation(dim, core_nodes, coords, L)
    tsslltd_nodes = np.arange(np.shape(tsslltd_coords)[0], dtype=int)

    # Determine the box that minimally encompasses the tessellated nodes
    # about which the radial distribution function will be computed
    tsslltd_g_r_box_indcs, _ = box_neighborhood_id(
        dim, tsslltd_coords, box_center, L_min, inclusive=True, indices=True)
    tsslltd_g_r_box_nodes = tsslltd_nodes[tsslltd_g_r_box_indcs]

    # Calculate the radial distribution function by calculating the
    # Euclidean length separating specified nodes, and then
    # histogramming the lengths with respect to radial distance bins
    for g_r_box_node in np.nditer(g_r_box_nodes):
        edges = np.column_stack(
            (np.full_like(tsslltd_g_r_box_nodes, int(g_r_box_node)), tsslltd_g_r_box_nodes))
        r = l_naive_func(edges, tsslltd_coords)
        r = r[r>0]
        hist = np.asarray(
            [np.histogram(r, bins=[start, end])[0][0] for start, end in bins])
        g_r += hist
    
    # Normalize the radial distribution function, the radial
    # distance bins, the radial increment, and the shift increment
    g_r_nrmlzd = g_r / (g_r_box_n*shell_V*rho)
    r_nrmlzd_bins = r_bins / b
    dr_nrmlzd = dr / b
    bin_shift_nrmlzd = bin_shift / b

    return g_r_nrmlzd, r_nrmlzd_bins, dr_nrmlzd, bin_shift_nrmlzd

def g_r_nrmlzd_fit_func(
        b: float,
        L: np.ndarray,
        coords: np.ndarray,
        core_nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exponential decay fit to the radial distribution function.

    This function constructs the exponential decay fit to the radial
    distribution function for a given set of nodes in a simulation box.

    Args:
        b (float): Chain segment and/or cross-linker diameter.
        L (np.ndarray): Simulation box side lengths.
        coords (np.ndarray): Coordinates of the core nodes.
        core_nodes (np.ndarray): Core nodes.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Exponential decay fit to the
        radial distribution function and normalized radial distance
        bins.
    
    """
    # Calculate radial distribution function
    g_r_nrmlzd, r_nrmlzd_bins, _, _ = g_r_nrmlzd_func(b, L, coords, core_nodes)
    
    # Construct and return the exponential decay fit to the radial
    # distribution function
    try:
        popt, _ = curve_fit(exp_decay_func, r_nrmlzd_bins, g_r_nrmlzd-1)
        g_r_nrmlzd_fit = exp_decay_func(r_nrmlzd_bins, *popt) + 1
        return g_r_nrmlzd_fit, r_nrmlzd_bins
    except RuntimeError as e: return g_r_nrmlzd, r_nrmlzd_bins

def xi_corr_func(
        b: float,
        L: np.ndarray,
        coords: np.ndarray,
        core_nodes: np.ndarray) -> float:
    """Correlation length.

    This function calculates the correlation length to the radial
    distribution function for a given set of nodes in a simulation box.

    Args:
        b (float): Chain segment and/or cross-linker diameter.
        L (np.ndarray): Simulation box side lengths.
        coords (np.ndarray): Coordinates of the core nodes.
        core_nodes (np.ndarray): Core nodes.
    
    Returns:
        float: Correlation length.
    
    """
    # Calculate radial distribution function
    g_r_nrmlzd, r_nrmlzd_bins, _, _ = g_r_nrmlzd_func(b, L, coords, core_nodes)
    
    # Construct the exponential decay fit to the radial distribution
    # function, then calculate and return the correlation length
    try:
        popt, _ = curve_fit(exp_decay_func, r_nrmlzd_bins, g_r_nrmlzd-1)
        return b * (1+popt[1])
    except RuntimeError as e: return b

def power_func(
        x: np.ndarray,
        a: float,
        b: float) -> np.ndarray:
    """Power function.

    Power function.

    Args:
        x (np.ndarray): Independent variable.
        a (float): Exponential pre-factor.
        b (float): Power.
    
    Returns:
        np.ndarray: Power function.
    
    """
    return a * x**b

def M_r_nrmlzd_func(
        b: float,
        L: np.ndarray,
        coords: np.ndarray,
        core_nodes: np.ndarray,
        num_bins: int=100) -> tuple[np.ndarray, np.ndarray, float]:
    """Box growing mass distribution function.

    This function constructs the box growing mass distribution function
    for a given set of nodes in a simulation box.

    Args:
        b (float): Chain segment and/or cross-linker diameter.
        L (np.ndarray): Simulation box side lengths.
        coords (np.ndarray): Coordinates of the core nodes.
        core_nodes (np.ndarray): Core nodes.
        num_bins (int): Number of bins subdividing the radial domain. Default value is 100 (and the bin width is 1/100th of the radial domain).
    
    Returns:
        tuple[np.ndarray, np.ndarray, float]: Box growing mass
        distribution function, normalized radial distance bins, and
        normalized bin width/radial increment.
    
    """
    # Simulation box parameters
    dim = np.shape(L)[0]
    box_center = L / 2
    L_min = np.min(L)
    r_max = L_min / 2

    # Initialize bin width/radial increment, radial distance bins, and
    # the box growing mass distribution function histogram
    dr = (r_max-b) / num_bins
    bins = np.linspace(b, r_max, num_bins+1)
    r_bins = bins[1:] - dr / 2
    M_r = np.zeros(num_bins)

    # Determine the core nodes in the simulation box about which
    # to compute the box growing mass distribution function
    M_r_box_indcs, M_r_box_n = box_neighborhood_id(
        dim, coords, box_center, r_max, inclusive=True, indices=True)
    M_r_box_nodes = core_nodes[M_r_box_indcs]

    # Tessellate the core node coordinates
    tsslltd_coords, _ = core_node_tessellation(dim, core_nodes, coords, L)
    tsslltd_nodes = np.arange(np.shape(tsslltd_coords)[0], dtype=int)

    # Determine the box that minimally encompasses the tessellated nodes
    # about which the box growing mass distribution function will be
    # computed
    tsslltd_M_r_box_indcs, _ = box_neighborhood_id(
        dim, tsslltd_coords, box_center, L_min, inclusive=True, indices=True)
    tsslltd_M_r_box_nodes = tsslltd_nodes[tsslltd_M_r_box_indcs]

    # Calculate the box growing mass distribution function by
    # calculating the Euclidean length separating specified nodes, and
    # then histogramming the lengths with respect to radial distance
    # bins
    for M_r_box_node in np.nditer(M_r_box_nodes):
        edges = np.column_stack(
            (np.full_like(tsslltd_M_r_box_nodes, int(M_r_box_node)), tsslltd_M_r_box_nodes))
        r = l_naive_func(edges, tsslltd_coords)
        r = r[r>0]
        hist, _ = np.histogram(r, bins=bins)
        hist_cumsum = np.cumsum(hist)
        M_r += hist_cumsum

    # Normalize the box growing mass distribution function, the radial
    # distance bins, and the radial increment
    M_r_nrmlzd = M_r / M_r_box_n
    r_nrmlzd_bins = r_bins / b
    dr_nrmlzd = dr / b

    return M_r_nrmlzd, r_nrmlzd_bins, dr_nrmlzd

def M_r_nrmlzd_fit_func(
        b: float,
        L: np.ndarray,
        coords: np.ndarray,
        core_nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalized radius-based power fit to the box growing mass
    distribution function.

    This function constructs the normalized radius-based power fit
    to the box growing mass distribution function for a given set of
    nodes in a simulation box.

    Args:
        b (float): Chain segment and/or cross-linker diameter.
        L (np.ndarray): Simulation box side lengths.
        coords (np.ndarray): Coordinates of the core nodes.
        core_nodes (np.ndarray): Core nodes.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Normalized radius-based
        power fit to the box growing mass distribution and normalized
        radial distance bins.
    
    """
    # Calculate box growing mass distribution function
    M_r_nrmlzd, r_nrmlzd_bins, _= M_r_nrmlzd_func(b, L, coords, core_nodes)
    
    # Construct and return the normalized radius-based power fit to the
    # box growing mass distribution function
    popt, _ = curve_fit(power_func, r_nrmlzd_bins, M_r_nrmlzd)
    M_r_nrmlzd_fit = power_func(r_nrmlzd_bins, *popt)
    return M_r_nrmlzd_fit, r_nrmlzd_bins

def n_fractal_dim_func(
        b: float,
        L: np.ndarray,
        coords: np.ndarray,
        core_nodes: np.ndarray) -> float:
    """Node-based fractal dimension.

    This function calculates the node-based fractal dimension for a
    given set of nodes in a simulation box.

    Args:
        b (float): Chain segment and/or cross-linker diameter.
        L (np.ndarray): Simulation box side lengths.
        coords (np.ndarray): Coordinates of the core nodes.
        core_nodes (np.ndarray): Core nodes.
    
    Returns:
        float: Node-based fractal dimension.
    
    """
    # Calculate box growing mass distribution function
    M_r_nrmlzd, r_nrmlzd_bins, _= M_r_nrmlzd_func(b, L, coords, core_nodes)

    # Construct the normalized radius-based power fit to the box growing
    # mass distribution function, then calculate and return the
    # node-based fractal dimension
    popt, _ = curve_fit(power_func, r_nrmlzd_bins, M_r_nrmlzd)
    n_fractal_dim = popt[1]
    dim = np.shape(L)[0]
    if n_fractal_dim > dim: n_fractal_dim = dim*1.0
    return n_fractal_dim