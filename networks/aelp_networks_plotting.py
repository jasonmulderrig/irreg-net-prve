import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from file_io.file_io import L_filename_str
from helpers.graph_utils import (
    lexsorted_edges,
    add_nodes_from_numpy_array,
    add_edges_from_numpy_array,
    extract_edges_to_numpy_array,
    elastically_effective_end_linked_graph
)
from descriptors.general_topological_descriptors import (
    core_pb_edge_id
)
from helpers.plotting_utils import (
    dim_2_network_topology_axes_formatter,
    dim_3_network_topology_axes_formatter
)
from networks.aelp_networks import aelp_filename_str

def aelp_network_core_node_marker_style(
        core_node: int,
        core_nodes_type: int,
        selfloop_edges: list[tuple[int, int]]) -> tuple[str, str, str]:
    marker = ""
    markerfacecolor = ""
    markeredgecolor = ""
    if core_nodes_type == 1:
        markerfacecolor = "black"
        markeredgecolor = "black"
        if len(selfloop_edges) == 0: marker = "."
        else:
            core_node_selfloop_edge_order = 0
            for selfloop_edge in selfloop_edges:
                if (selfloop_edge[0] == core_node) and (selfloop_edge[1] == core_node):
                    core_node_selfloop_edge_order += 1
            if core_node_selfloop_edge_order == 0: marker = "."
            elif core_node_selfloop_edge_order == 1: marker = "s"
            elif core_node_selfloop_edge_order == 2: marker = "h"
            elif core_node_selfloop_edge_order >= 3: marker = "8"
    elif core_nodes_type == 0:
        marker = "."
        markerfacecolor = "red"
        markeredgecolor = "red"
    return (marker, markerfacecolor, markeredgecolor)

def aelp_network_edge_alpha(
        core_node_0: int,
        core_node_1: int,
        conn_graph: nx.Graph | nx.MultiGraph) -> float:
    alpha = 0.25 * conn_graph.number_of_edges(core_node_0, core_node_1)
    alpha = np.minimum(alpha, 1.0)
    return alpha

def aelp_network_topology_plotter(
        plt_pad_prefactor: float,
        core_tick_inc_prefactor: float,
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int) -> None:
    # Generate filenames
    aelp_filename = aelp_filename_str(network, date, batch, sample, config)
    coords_filename = aelp_filename + ".coords"
    core_nodes_type_filename = aelp_filename + "-core_nodes_type" + ".dat"
    conn_edges_filename = aelp_filename + "-conn_edges" + ".dat"
    conn_edges_type_filename = aelp_filename + "-conn_edges_type" + ".dat"

    # Load simulation box side lengths
    L = np.loadtxt(L_filename_str(network, date, batch, sample))
    L_max = np.max(L)
    L_x = L[0]
    L_y = L[1]

    # Load node coordinates
    coords = np.loadtxt(coords_filename)
    n, dim = np.shape(coords)

    # Load fundamental graph constituents
    core_nodes = np.arange(n, dtype=int)
    core_nodes_type = np.loadtxt(core_nodes_type_filename, dtype=int)
    conn_edges = np.loadtxt(conn_edges_filename, dtype=int)
    conn_edges_type = np.loadtxt(conn_edges_type_filename, dtype=int)
    m = np.shape(conn_edges)[0]
    
    # Lexicographically sort the edges and the edges type
    conn_edges, lexsort_indcs = lexsorted_edges(conn_edges, return_indcs=True)
    conn_edges_type = conn_edges_type[lexsort_indcs]

    # Create nx.MultiGraph and add nodes before edges
    conn_graph = nx.MultiGraph()
    conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
    conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

    # Extract list of self-loop edges
    conn_graph_selfloop_edges = list(nx.selfloop_edges(conn_graph))

    # Plot formatting parameters
    plt_pad = plt_pad_prefactor * L_max
    core_tick_inc = core_tick_inc_prefactor * L_max
    min_core = -plt_pad
    max_core = L_max + plt_pad
    core_tick_steps = int(np.around((max_core-min_core)/core_tick_inc)) + 1

    xlim = np.asarray([min_core, max_core])
    ylim = np.asarray([min_core, max_core])
    zlim = np.asarray([min_core, max_core])

    xticks = np.linspace(min_core, max_core, core_tick_steps)
    yticks = np.linspace(min_core, max_core, core_tick_steps)
    zticks = np.linspace(min_core, max_core, core_tick_steps)

    xlabel = "x"
    ylabel = "y"
    zlabel = "z"

    grid_alpha = 0.25
    grid_zorder = 0

    core_color = "red"
    core_linewidth = 0.5

    # Core simulation box coordinates
    if dim == 2:
        core_box = np.asarray(
            [
                [0, 0], [L_x, 0], [L_x, L_y], [0, L_y], [0, 0]
            ]
        )
    elif dim == 3:
        L_z = L[2]
        core_box = np.asarray(
            [
                [[0, 0, 0], [L_x, 0, 0], [L_x, L_y, 0], [0, L_y, 0], [0, 0, 0]],
                [[0, 0, L_z], [L_x, 0, L_z], [L_x, L_y, L_z], [0, L_y, L_z], [0, 0, L_z]],
                [[0, 0, 0], [L_x, 0, 0], [L_x, 0, L_z], [0, 0, L_z], [0, 0, 0]],
                [[L_x, 0, 0], [L_x, L_y, 0], [L_x, L_y, L_z], [L_x, 0, L_z], [L_x, 0, 0]],
                [[L_x, L_y, 0], [0, L_y, 0], [0, L_y, L_z], [L_x, L_y, L_z], [L_x, L_y, 0]],
                [[0, L_y, 0], [0, 0, 0], [0, 0, L_z], [0, L_y, L_z], [0, L_y, 0]]

            ]
        )

    def core_pb_graph_topology_plotting_func(colored=False, eeel_ntwrk=False):
        """Plot of the core and periodic boundary cross-linkers and
        edges for the graph capturing the spatial topology of the core
        and periodic boundary nodes and edges. Here, the edges could all
        be represented as blue lines, or the core and periodic boundary
        edges could each be represented by purple or olive lines,
        respectively.
        
        """
        if dim == 2:
            fig, ax = plt.subplots()
            for edge in range(m):
                conn_edge_type = conn_edges_type[edge]
                core_node_0 = conn_edges[edge, 0]
                core_node_1 = conn_edges[edge, 1]
                alpha = aelp_network_edge_alpha(
                    core_node_0, core_node_1, conn_graph)
                core_node_0_type = core_nodes_type[core_node_0]
                core_node_1_type = core_nodes_type[core_node_1]
                core_node_0_x = coords[core_node_0, 0]
                core_node_0_y = coords[core_node_0, 1]
                core_node_1_x = coords[core_node_1, 0]
                core_node_1_y = coords[core_node_1, 1]

                if conn_edge_type == 1:
                    edge_x = np.asarray(
                        [
                            core_node_0_x,
                            core_node_1_x
                        ]
                    )
                    edge_y = np.asarray(
                        [
                            core_node_0_y,
                            core_node_1_y
                        ]
                    )
                    ax.plot(
                        edge_x, edge_y,
                        color="tab:purple" if colored else "tab:blue",
                        linewidth=1.5, alpha=alpha)
                    marker, markerfacecolor, markeredgecolor = (
                        aelp_network_core_node_marker_style(
                            core_node_0, core_node_0_type,
                            conn_graph_selfloop_edges)
                    )
                    ax.plot(
                        core_node_0_x, core_node_0_y, marker=marker,
                        markersize=1.5, markerfacecolor=markerfacecolor,
                        markeredgecolor=markeredgecolor)
                    marker, markerfacecolor, markeredgecolor = (
                        aelp_network_core_node_marker_style(
                            core_node_1, core_node_1_type,
                            conn_graph_selfloop_edges)
                    )
                    ax.plot(
                        core_node_1_x, core_node_1_y, marker=marker,
                        markersize=1.5, markerfacecolor=markerfacecolor,
                        markeredgecolor=markeredgecolor)
                elif conn_edge_type == 0:
                    core_node_0_coords = np.asarray(
                        [
                            core_node_0_x,
                            core_node_0_y
                        ]
                    )
                    core_node_1_coords = np.asarray(
                        [
                            core_node_1_x,
                            core_node_1_y
                        ]
                    )
                    pb_node_0_coords, _ = core_pb_edge_id(
                        core_node_1_coords, core_node_0_coords, L)
                    pb_node_1_coords, _ = core_pb_edge_id(
                        core_node_0_coords, core_node_1_coords, L)
                    pb_node_0_x = pb_node_0_coords[0]
                    pb_node_0_y = pb_node_0_coords[1]
                    pb_node_1_x = pb_node_1_coords[0]
                    pb_node_1_y = pb_node_1_coords[1]
                    edge_x = np.asarray(
                        [
                            core_node_0_x,
                            pb_node_1_x
                        ]
                    )
                    edge_y = np.asarray(
                        [
                            core_node_0_y,
                            pb_node_1_y
                        ]
                    )
                    ax.plot(
                        edge_x, edge_y,
                        color="tab:olive" if colored else "tab:blue",
                        linewidth=1.5, alpha=alpha)
                    edge_x = np.asarray(
                        [
                            pb_node_0_x,
                            core_node_1_x
                        ]
                    )
                    edge_y = np.asarray(
                        [
                            pb_node_0_y,
                            core_node_1_y
                        ]
                    )
                    ax.plot(
                        edge_x, edge_y,
                        color="tab:olive" if colored else "tab:blue",
                        linewidth=1.5, alpha=alpha)
                    marker, markerfacecolor, markeredgecolor = (
                        aelp_network_core_node_marker_style(
                            core_node_0, core_node_0_type,
                            conn_graph_selfloop_edges)
                    )
                    ax.plot(
                        core_node_0_x, core_node_0_y, marker=marker,
                        markersize=1.5, markerfacecolor=markerfacecolor,
                        markeredgecolor=markeredgecolor)
                    ax.plot(
                        pb_node_0_x, pb_node_0_y, marker=marker,
                        markersize=1.5, markerfacecolor=markerfacecolor,
                        markeredgecolor=markeredgecolor)
                    marker, markerfacecolor, markeredgecolor = (
                        aelp_network_core_node_marker_style(
                            core_node_1, core_node_1_type,
                            conn_graph_selfloop_edges)
                    )
                    ax.plot(
                        core_node_1_x, core_node_1_y, marker=marker,
                        markersize=1.5, markerfacecolor=markerfacecolor,
                        markeredgecolor=markeredgecolor)
                    ax.plot(
                        pb_node_1_x, pb_node_1_y, marker=marker,
                        markersize=1.5, markerfacecolor=markerfacecolor,
                        markeredgecolor=markeredgecolor)
            ax = dim_2_network_topology_axes_formatter(
                ax, core_box, core_color, core_linewidth,
                xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha,
                grid_zorder)
        elif dim == 3:
            fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
            for edge in range(m):
                conn_edge_type = conn_edges_type[edge]
                core_node_0 = conn_edges[edge, 0]
                core_node_1 = conn_edges[edge, 1]
                alpha = aelp_network_edge_alpha(
                    core_node_0, core_node_1, conn_graph)
                core_node_0_type = core_nodes_type[core_node_0]
                core_node_1_type = core_nodes_type[core_node_1]
                core_node_0_x = coords[core_node_0, 0]
                core_node_0_y = coords[core_node_0, 1]
                core_node_0_z = coords[core_node_0, 2]
                core_node_1_x = coords[core_node_1, 0]
                core_node_1_y = coords[core_node_1, 1]
                core_node_1_z = coords[core_node_1, 2]

                if conn_edge_type == 1:
                    edge_x = np.asarray(
                        [
                            core_node_0_x,
                            core_node_1_x
                        ]
                    )
                    edge_y = np.asarray(
                        [
                            core_node_0_y,
                            core_node_1_y
                        ]
                    )
                    edge_z = np.asarray(
                        [
                            core_node_0_z,
                            core_node_1_z
                        ]
                    )
                    ax.plot(
                        edge_x, edge_y, edge_z,
                        color="tab:purple" if colored else "tab:blue",
                        linewidth=1.5, alpha=alpha)
                    marker, markerfacecolor, markeredgecolor = (
                        aelp_network_core_node_marker_style(
                            core_node_0, core_node_0_type,
                            conn_graph_selfloop_edges)
                    )
                    ax.plot(
                        core_node_0_x, core_node_0_y, core_node_0_z,
                        marker=marker, markersize=1.5,
                        markerfacecolor=markerfacecolor,
                        markeredgecolor=markeredgecolor)
                    marker, markerfacecolor, markeredgecolor = (
                        aelp_network_core_node_marker_style(
                            core_node_1, core_node_1_type,
                            conn_graph_selfloop_edges)
                    )
                    ax.plot(
                        core_node_1_x, core_node_1_y, core_node_1_z,
                        marker=marker, markersize=1.5,
                        markerfacecolor=markerfacecolor,
                        markeredgecolor=markeredgecolor)
                elif conn_edge_type == 0:
                    core_node_0_coords = np.asarray(
                        [
                            core_node_0_x,
                            core_node_0_y,
                            core_node_0_z
                        ]
                    )
                    core_node_1_coords = np.asarray(
                        [
                            core_node_1_x,
                            core_node_1_y,
                            core_node_1_z
                        ]
                    )
                    pb_node_0_coords, _ = core_pb_edge_id(
                        core_node_1_coords, core_node_0_coords, L)
                    pb_node_1_coords, _ = core_pb_edge_id(
                        core_node_0_coords, core_node_1_coords, L)
                    pb_node_0_x = pb_node_0_coords[0]
                    pb_node_0_y = pb_node_0_coords[1]
                    pb_node_0_z = pb_node_0_coords[2]
                    pb_node_1_x = pb_node_1_coords[0]
                    pb_node_1_y = pb_node_1_coords[1]
                    pb_node_1_z = pb_node_1_coords[2]
                    edge_x = np.asarray(
                        [
                            core_node_0_x,
                            pb_node_1_x
                        ]
                    )
                    edge_y = np.asarray(
                        [
                            core_node_0_y,
                            pb_node_1_y
                        ]
                    )
                    edge_z = np.asarray(
                        [
                            core_node_0_z,
                            pb_node_1_z
                        ]
                    )
                    ax.plot(
                        edge_x, edge_y, edge_z,
                        color="tab:olive" if colored else "tab:blue",
                        linewidth=1.5, alpha=alpha)
                    edge_x = np.asarray(
                        [
                            pb_node_0_x,
                            core_node_1_x
                        ]
                    )
                    edge_y = np.asarray(
                        [
                            pb_node_0_y,
                            core_node_1_y
                        ]
                    )
                    edge_z = np.asarray(
                        [
                            pb_node_0_z,
                            core_node_1_z
                        ]
                    )
                    ax.plot(
                        edge_x, edge_y, edge_z,
                        color="tab:olive" if colored else "tab:blue",
                        linewidth=1.5, alpha=alpha)
                    marker, markerfacecolor, markeredgecolor = (
                        aelp_network_core_node_marker_style(
                            core_node_0, core_node_0_type,
                            conn_graph_selfloop_edges)
                    )
                    ax.plot(
                        core_node_0_x, core_node_0_y, core_node_0_z,
                        marker=marker, markersize=1.5,
                        markerfacecolor=markerfacecolor,
                        markeredgecolor=markeredgecolor)
                    ax.plot(
                        pb_node_0_x, pb_node_0_y, pb_node_0_z, marker=marker,
                        markersize=1.5, markerfacecolor=markerfacecolor,
                        markeredgecolor=markeredgecolor)
                    marker, markerfacecolor, markeredgecolor = (
                        aelp_network_core_node_marker_style(
                            core_node_1, core_node_1_type,
                            conn_graph_selfloop_edges)
                    )
                    ax.plot(
                        core_node_1_x, core_node_1_y, core_node_1_z,
                        marker=marker, markersize=1.5,
                        markerfacecolor=markerfacecolor,
                        markeredgecolor=markeredgecolor)
                    ax.plot(
                        pb_node_1_x, pb_node_1_y, pb_node_1_z, marker=marker,
                        markersize=1.5, markerfacecolor=markerfacecolor,
                        markeredgecolor=markeredgecolor)
            ax = dim_3_network_topology_axes_formatter(
                ax, core_box, core_color, core_linewidth,
                xlim, ylim, zlim, xticks, yticks, zticks,
                xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        core_pb_graph_topology = (
            "-eeel_core_pb_graph" if eeel_ntwrk else "-core_pb_graph"
        )
        core_pb_graph_topology = (
            core_pb_graph_topology+"_colored_topology" if colored
            else core_pb_graph_topology+"_topology"
        )
        fig.savefig(aelp_filename+core_pb_graph_topology+".png")
        plt.close()
        
        return None
    
    def conn_graph_topology_plotting_func(colored=False, eeel_ntwrk=False):
        """Plot of the core and periodic boundary cross-linkers and
        edges for the graph capturing the periodic connections between
        the core nodes. Here, the edges could all be represented as blue
        lines, or the core and periodic boundary edges could each be
        represented by purple or olive lines, respectively.
        
        """
        if dim == 2:
            fig, ax = plt.subplots()
            for edge in range(m):
                conn_edge_type = conn_edges_type[edge]
                core_node_0 = conn_edges[edge, 0]
                core_node_1 = conn_edges[edge, 1]
                alpha = aelp_network_edge_alpha(
                    core_node_0, core_node_1, conn_graph)
                core_node_0_type = core_nodes_type[core_node_0]
                core_node_1_type = core_nodes_type[core_node_1]
                core_node_0_x = coords[core_node_0, 0]
                core_node_0_y = coords[core_node_0, 1]
                core_node_1_x = coords[core_node_1, 0]
                core_node_1_y = coords[core_node_1, 1]
                edge_x = np.asarray(
                    [
                        core_node_0_x,
                        core_node_1_x
                    ]
                )
                edge_y = np.asarray(
                    [
                        core_node_0_y,
                        core_node_1_y
                    ]
                )
                if conn_edge_type == 1: color = "tab:purple"
                elif conn_edge_type == 0: color = "tab:olive"
                ax.plot(
                    edge_x, edge_y, color=color if colored else "tab:blue",
                    linewidth=1.5, alpha=alpha)
                marker, markerfacecolor, markeredgecolor = (
                    aelp_network_core_node_marker_style(
                        core_node_0, core_node_0_type,
                        conn_graph_selfloop_edges)
                )
                ax.plot(
                    core_node_0_x, core_node_0_y, marker=marker, markersize=1.5,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
                marker, markerfacecolor, markeredgecolor = (
                    aelp_network_core_node_marker_style(
                        core_node_1, core_node_1_type,
                        conn_graph_selfloop_edges)
                )
                ax.plot(
                    core_node_1_x, core_node_1_y, marker=marker, markersize=1.5,
                    markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
            ax = dim_2_network_topology_axes_formatter(
                ax, core_box, core_color, core_linewidth,
                xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha,
                grid_zorder)
        elif dim == 3:
            fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
            for edge in range(m):
                conn_edge_type = conn_edges_type[edge]
                core_node_0 = conn_edges[edge, 0]
                core_node_1 = conn_edges[edge, 1]
                alpha = aelp_network_edge_alpha(
                    core_node_0, core_node_1, conn_graph)
                core_node_0_type = core_nodes_type[core_node_0]
                core_node_1_type = core_nodes_type[core_node_1]
                core_node_0_x = coords[core_node_0, 0]
                core_node_0_y = coords[core_node_0, 1]
                core_node_0_z = coords[core_node_0, 2]
                core_node_1_x = coords[core_node_1, 0]
                core_node_1_y = coords[core_node_1, 1]
                core_node_1_z = coords[core_node_1, 2]
                edge_x = np.asarray(
                    [
                        core_node_0_x,
                        core_node_1_x
                    ]
                )
                edge_y = np.asarray(
                    [
                        core_node_0_y,
                        core_node_1_y
                    ]
                )
                edge_z = np.asarray(
                    [
                        core_node_0_z,
                        core_node_1_z
                    ]
                )
                if conn_edge_type == 1: color = "tab:purple"
                elif conn_edge_type == 0: color = "tab:olive"
                ax.plot(
                    edge_x, edge_y, edge_z,
                    color=color if colored else "tab:blue",
                    linewidth=1.5, alpha=alpha)
                marker, markerfacecolor, markeredgecolor = (
                    aelp_network_core_node_marker_style(
                        core_node_0, core_node_0_type,
                        conn_graph_selfloop_edges)
                )
                ax.plot(
                    core_node_0_x, core_node_0_y, core_node_0_z, marker=marker,
                    markersize=1.5, markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
                marker, markerfacecolor, markeredgecolor = (
                    aelp_network_core_node_marker_style(
                        core_node_1, core_node_1_type,
                        conn_graph_selfloop_edges)
                )
                ax.plot(
                    core_node_1_x, core_node_1_y, core_node_1_z, marker=marker,
                    markersize=1.5, markerfacecolor=markerfacecolor,
                    markeredgecolor=markeredgecolor)
            ax = dim_3_network_topology_axes_formatter(
                ax, core_box, core_color, core_linewidth,
                xlim, ylim, zlim, xticks, yticks, zticks,
                xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        conn_graph_topology = (
            "-eeel_conn_graph" if eeel_ntwrk else "-conn_graph"
        )
        conn_graph_topology = (
            conn_graph_topology+"_colored_topology" if colored
            else conn_graph_topology+"_topology"
        )
        fig.savefig(aelp_filename+conn_graph_topology+".png")
        plt.close()
        
        return None
    
    core_pb_graph_topology_plotting_func(colored=False, eeel_ntwrk=False)
    core_pb_graph_topology_plotting_func(colored=True, eeel_ntwrk=False)
    conn_graph_topology_plotting_func(colored=False, eeel_ntwrk=False)
    conn_graph_topology_plotting_func(colored=True, eeel_ntwrk=False)
    
    # Extract elastically-effective end-linked network
    eeel_conn_graph = elastically_effective_end_linked_graph(conn_graph)

    # Extract edges from the elastically-effective end-linked
    # network
    eeel_conn_edges = extract_edges_to_numpy_array(eeel_conn_graph)
    del eeel_conn_graph, conn_graph
    
    # Lexicographically sort the edges for the elastically-effective
    # end-linked network
    eeel_conn_edges = lexsorted_edges(eeel_conn_edges)
    eeel_m = np.shape(eeel_conn_edges)[0]

    # Downselect edges type for the elastically-effective end-linked
    # network from the as-provided network
    eeel_conn_edges_type = np.empty(eeel_m, dtype=int)
    eeel_edge = 0
    edge = 0
    while eeel_edge < eeel_m:
        if np.array_equal(eeel_conn_edges[eeel_edge], conn_edges[edge]):
            eeel_conn_edges_type[eeel_edge] = conn_edges_type[edge]
            eeel_edge += 1
        edge += 1

    conn_edges = eeel_conn_edges.copy()
    conn_edges_type = eeel_conn_edges_type.copy()
    m = np.shape(conn_edges)[0]
    
    # Create nx.MultiGraph and add nodes before edges
    conn_graph = nx.MultiGraph()
    conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
    conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

    # Extract list of self-loop edges
    conn_graph_selfloop_edges = list(nx.selfloop_edges(conn_graph))

    core_pb_graph_topology_plotting_func(colored=False, eeel_ntwrk=True)
    core_pb_graph_topology_plotting_func(colored=True, eeel_ntwrk=True)
    conn_graph_topology_plotting_func(colored=False, eeel_ntwrk=True)
    conn_graph_topology_plotting_func(colored=True, eeel_ntwrk=True)