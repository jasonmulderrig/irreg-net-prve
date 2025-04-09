# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from file_io.file_io import (
    L_filename_str,
    filepath_str
)
from helpers.simulation_box_utils import A_or_V_arg_L_func
from helpers.network_utils import rho_func
from helpers.graph_utils import (
    lexsorted_edges,
    add_nodes_from_numpy_array,
    add_edges_from_numpy_array,
    add_edges_and_edge_attributes_from_numpy_arrays,
    extract_edges_to_numpy_array,
    elastically_effective_end_linked_graph
)
from descriptors.general_topological_descriptors import l_func
from networks.aelp_networks import aelp_filename_str
from networks.apelp_networks_config import (
    apelpConfig,
    sample_config_params_arr_func
)

# Hydra ConfigStore initialization
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(name="config", node=apelpConfig)

@hydra.main(version_base=None, config_path=".", config_name="apelp_networks_config")
def main(cfg: apelpConfig) -> None:
    sample_config_params_arr, sample_config_num = sample_config_params_arr_func(cfg)
    filepath = filepath_str(cfg.label.network)

    dim_2_sample_config_num = np.count_nonzero(sample_config_params_arr[:, 1]==2)
    dim_3_sample_config_num = np.count_nonzero(sample_config_params_arr[:, 1]==3)

    b = cfg.topology.b[0]

    k_max = int(np.max(sample_config_params_arr[:, 5]))
    k_list = list(range(k_max+1))
    en_max = int(np.max(sample_config_params_arr[:, 8]))
    nu_max = en_max - 1
    
    # Initialization
    dim_2_l_chns = np.asarray([])
    dim_2_nu_chns = np.asarray([], dtype=int)
    dim_2_prop_dnglng_chns = np.empty(dim_2_sample_config_num)
    dim_2_k_clnkr_rho = np.empty((dim_2_sample_config_num, k_max+1))

    dim_3_l_chns = np.asarray([])
    dim_3_nu_chns = np.asarray([], dtype=int)
    dim_3_prop_dnglng_chns = np.empty(dim_3_sample_config_num)
    dim_3_k_clnkr_rho = np.empty((dim_3_sample_config_num, k_max+1))
    
    dim_2_dnglng_n_tot = 0
    dim_2_m_tot = 0
    
    dim_3_dnglng_n_tot = 0
    dim_3_m_tot = 0

    dim_2_indx = 0
    dim_3_indx = 0
    for indx in range(sample_config_num):
        sample = int(sample_config_params_arr[indx, 0])
        dim = int(sample_config_params_arr[indx, 1])
        k = int(sample_config_params_arr[indx, 5])
        config = int(sample_config_params_arr[indx, 9])
        
        # Generate filenames
        L_filename = L_filename_str(
            cfg.label.network, cfg.label.date, cfg.label.batch, sample)
        aelp_filename = aelp_filename_str(
            cfg.label.network, cfg.label.date, cfg.label.batch, sample, config)
        coords_filename = aelp_filename + ".coords"
        core_nodes_type_filename = aelp_filename + "-core_nodes_type" + ".dat"
        conn_edges_filename = aelp_filename + "-conn_edges" + ".dat"
        conn_edges_type_filename = aelp_filename + "-conn_edges_type" + ".dat"
        l_cntr_conn_edges_filename = (
            aelp_filename + "-l_cntr_conn_edges" + ".dat"
        )

        # Load simulation box side lengths and node coordinates
        L = np.loadtxt(L_filename)
        coords = np.loadtxt(coords_filename)

        # Load fundamental graph constituents
        core_nodes_type = np.loadtxt(core_nodes_type_filename, dtype=int)
        core_nodes = np.arange(np.shape(core_nodes_type)[0], dtype=int)
        conn_edges = np.loadtxt(conn_edges_filename, dtype=int)
        conn_edges_type = np.loadtxt(conn_edges_type_filename, dtype=int)
        l_cntr_conn_edges = np.loadtxt(l_cntr_conn_edges_filename)
        nu_chns = l_cntr_conn_edges / b
        nu_chns = nu_chns.astype(int)
        m = np.shape(conn_edges)[0]

        # Calculate end-to-end chain length (Euclidean edge length)
        l_chns = l_func(conn_edges, conn_edges_type, coords, L)

        # Create nx.MultiGraph, and add nodes before edges
        conn_graph = nx.MultiGraph()
        conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
        conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

        # Number of dangling chains
        dnglng_n = np.count_nonzero(core_nodes_type==0)

        if dim == 2:
            # End-to-end chain length for each/every chain
            dim_2_l_chns = np.concatenate((dim_2_l_chns, l_chns))

            # Chain segment number for each/every chain
            dim_2_nu_chns = np.concatenate((dim_2_nu_chns, nu_chns), dtype=int)

            # Proportion of dangling chains
            dim_2_prop_dnglng_chns[dim_2_indx] = dnglng_n / m

            # Count cross-linker node degree occurances
            dim_2_k_clnkr_count = np.zeros(k_max+1, dtype=int)
            for node, k in list(conn_graph.degree()):
                # If dangling chain node, then continue to next node
                if core_nodes_type[node] == 0: continue
                # Update cross-linker node degree occurance count
                dim_2_k_clnkr_count[k] += 1
            dim_2_k_clnkr_count[0] = np.sum(dim_2_k_clnkr_count[1:])

            # Calculate degree-wise cross-linker number density
            A = A_or_V_arg_L_func(L)
            for k in range(k_max+1):
                dim_2_k_clnkr_rho[dim_2_indx, k] = rho_func(
                    dim_2_k_clnkr_count[k], A)
            
            # Update total values for various parameters
            dim_2_dnglng_n_tot += dnglng_n
            dim_2_m_tot += m
            dim_2_indx += 1
        elif dim == 3:
            # End-to-end chain length for each/every chain
            dim_3_l_chns = np.concatenate((dim_3_l_chns, l_chns))

            # Chain segment number for each/every chain
            dim_3_nu_chns = np.concatenate((dim_3_nu_chns, nu_chns), dtype=int)

            # Proportion of dangling chains
            dim_3_prop_dnglng_chns[dim_3_indx] = dnglng_n / m

            # Count cross-linker node degree occurances
            dim_3_k_clnkr_count = np.zeros(k_max+1, dtype=int)
            for node, k in list(conn_graph.degree()):
                # If dangling chain node, then continue to next node
                if core_nodes_type[node] == 0: continue
                # Update cross-linker node degree occurance count
                dim_3_k_clnkr_count[k] += 1
            dim_3_k_clnkr_count[0] = np.sum(dim_3_k_clnkr_count[1:])

            # Calculate degree-wise cross-linker number density
            V = A_or_V_arg_L_func(L)
            for k in range(k_max+1):
                dim_3_k_clnkr_rho[dim_3_indx, k] = rho_func(
                    dim_3_k_clnkr_count[k], V)
            
            # Update total values for various parameters
            dim_3_dnglng_n_tot += dnglng_n
            dim_3_m_tot += m
            dim_3_indx += 1
    
    # Total proportion of dangling chains
    dim_2_prop_dnglng_chns_tot = dim_2_dnglng_n_tot / dim_2_m_tot
    dim_3_prop_dnglng_chns_tot = dim_3_dnglng_n_tot / dim_3_m_tot

    print("Two-dimensional apelp network dangling chain proportion = {}".format(dim_2_prop_dnglng_chns_tot))
    print("Three-dimensional apelp network dangling chain proportion = {}".format(dim_3_prop_dnglng_chns_tot))

    # Total cross-linker number density
    dim_2_clnkr_rho_tot = np.max(dim_2_k_clnkr_rho[:, 0])
    dim_3_clnkr_rho_tot = np.max(dim_3_k_clnkr_rho[:, 0])

    print("Two-dimensional apelp network cross-linker number density = {}".format(dim_2_clnkr_rho_tot))
    print("Three-dimensional apelp network cross-linker number density = {}".format(dim_3_clnkr_rho_tot))
    
    # Density histogram of number density of k=1, k=2, k=3, k=4 cross-linker nodes
    xlabel = "rho"
    dim_2_clnkr_rho_first_bin = 0.0
    dim_2_clnkr_rho_last_bin = dim_2_clnkr_rho_tot
    
    dim_3_clnkr_rho_first_bin = 0.0
    dim_3_clnkr_rho_last_bin = dim_3_clnkr_rho_tot
    
    # Density histogram preformatting
    dim_2_clnkr_rho_bin_steps = 81
    dim_2_clnkr_rho_bins = np.linspace(
        dim_2_clnkr_rho_first_bin, dim_2_clnkr_rho_last_bin,
        dim_2_clnkr_rho_bin_steps)
    dim_2_clnkr_rho_steps = 5
    dim_2_xticks = np.linspace(
        dim_2_clnkr_rho_first_bin, dim_2_clnkr_rho_last_bin,
        dim_2_clnkr_rho_steps)
    dim_2_clnkr_rho_dnstyhist_filename = (
        filepath + "dim_2" + "-" + "clnkr_rho" + "-" + "dnstyhist" + ".png"
    )
    
    dim_3_clnkr_rho_bin_steps = 81
    dim_3_clnkr_rho_bins = np.linspace(
        dim_3_clnkr_rho_first_bin, dim_3_clnkr_rho_last_bin,
        dim_3_clnkr_rho_bin_steps)
    dim_3_clnkr_rho_steps = 5
    dim_3_xticks = np.linspace(
        dim_3_clnkr_rho_first_bin, dim_3_clnkr_rho_last_bin,
        dim_3_clnkr_rho_steps)
    dim_3_clnkr_rho_dnstyhist_filename = (
        filepath + "dim_3" + "-" + "clnkr_rho" + "-" + "dnstyhist" + ".png"
    )
    
    fig, axs = plt.subplots(k_max)
    for k in range(k_max+1):
        if k == 0: continue
        else:
            axs[k-1].hist(
                dim_2_k_clnkr_rho[:, k], bins=dim_2_clnkr_rho_bins,
                density=True, color="tab:blue", edgecolor="black", zorder=3)
            axs[k-1].set_xticks(dim_2_xticks)
            axs[k-1].set_title("k = {}".format(k))
            axs[k-1].grid(True, alpha=0.25, zorder=0)
    axs[-1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(dim_2_clnkr_rho_dnstyhist_filename)
    plt.close()

    fig, axs = plt.subplots(k_max)
    for k in range(k_max+1):
        if k == 0: continue
        else:
            axs[k-1].hist(
                dim_3_k_clnkr_rho[:, k], bins=dim_3_clnkr_rho_bins,
                density=True, color="tab:blue", edgecolor="black", zorder=3)
            axs[k-1].set_xticks(dim_3_xticks)
            axs[k-1].set_title("k = {}".format(k))
            axs[k-1].grid(True, alpha=0.25, zorder=0)
    axs[-1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(dim_3_clnkr_rho_dnstyhist_filename)
    plt.close()

    # Density histogram of chain segment number
    xlabel = "nu"
    dim_2_nu_first_bin = 1
    dim_2_nu_last_bin = nu_max
    dim_2_nu_bin_inc = 1

    dim_3_nu_first_bin = 1
    dim_3_nu_last_bin = nu_max
    dim_3_nu_bin_inc = 1
    
    # Density histogram preformatting
    dim_2_nu_bin_steps = (
        int(np.around((dim_2_nu_last_bin-dim_2_nu_first_bin)/dim_2_nu_bin_inc))
        + 1
    )
    dim_2_nu_bins = np.linspace(
        dim_2_nu_first_bin, dim_2_nu_last_bin, dim_2_nu_bin_steps)
    dim_2_xticks = np.linspace(0, dim_2_nu_last_bin, 11)
    dim_2_nu_dnstyhist_filename = (
        filepath + "dim_2" + "-" + "nu" + "-" + "dnstyhist" + ".png"
    )

    dim_3_nu_bin_steps = (
        int(np.around((dim_3_nu_last_bin-dim_3_nu_first_bin)/dim_3_nu_bin_inc))
        + 1
    )
    dim_3_nu_bins = np.linspace(
        dim_3_nu_first_bin, dim_3_nu_last_bin, dim_3_nu_bin_steps)
    dim_3_xticks = np.linspace(0, dim_3_nu_last_bin, 11)
    dim_3_nu_dnstyhist_filename = (
        filepath + "dim_3" + "-" + "nu" + "-" + "dnstyhist" + ".png"
    )
    
    fig, axs = plt.subplots()
    axs.hist(
        dim_2_nu_chns, bins=dim_2_nu_bins, density=True, color="tab:blue",
        zorder=3)
    axs.set_xticks(dim_2_xticks)
    axs.grid(True, alpha=0.25, zorder=0)
    axs.set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(dim_2_nu_dnstyhist_filename)
    plt.close()

    fig, axs = plt.subplots()
    axs.hist(
        dim_3_nu_chns, bins=dim_3_nu_bins, density=True, color="tab:blue",
        zorder=3)
    axs.set_xticks(dim_3_xticks)
    axs.grid(True, alpha=0.25, zorder=0)
    axs.set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(dim_3_nu_dnstyhist_filename)
    plt.close()

    # Density histogram of end-to-end chain length for all chains
    xlabel = "l"
    dim_2_l_first_bin = 0
    dim_2_l_last_bin = np.max(dim_2_l_chns)
    dim_2_l_bin_steps = 101

    dim_3_l_first_bin = 0
    dim_3_l_last_bin = np.max(dim_3_l_chns)
    dim_3_l_bin_steps = 101
    
    # Density histogram preformatting
    dim_2_l_bins = np.linspace(
        dim_2_l_first_bin, dim_2_l_last_bin, dim_2_l_bin_steps)
    dim_2_xticks = np.linspace(0, dim_2_l_last_bin, 11)
    dim_2_l_dnstyhist_filename = (
        filepath + "dim_2" + "-" + "l" + "-" + "dnstyhist" + ".png"
    )

    dim_3_l_bins = np.linspace(
        dim_3_l_first_bin, dim_3_l_last_bin, dim_3_l_bin_steps)
    dim_3_xticks = np.linspace(0, dim_3_l_last_bin, 11)
    dim_3_l_dnstyhist_filename = (
        filepath + "dim_3" + "-" + "l" + "-" + "dnstyhist" + ".png"
    )

    fig, axs = plt.subplots()
    axs.hist(
        dim_2_l_chns, bins=dim_2_l_bins, density=True, color="tab:blue",
        zorder=3)
    axs.set_xticks(dim_2_xticks)
    axs.grid(True, alpha=0.25, zorder=0)
    axs.set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(dim_2_l_dnstyhist_filename)
    plt.close()

    fig, axs = plt.subplots()
    axs.hist(
        dim_3_l_chns, bins=dim_3_l_bins, density=True, color="tab:blue",
        zorder=3)
    axs.set_xticks(dim_3_xticks)
    axs.grid(True, alpha=0.25, zorder=0)
    axs.set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(dim_3_l_dnstyhist_filename)
    plt.close()

    # Initialization
    dim_2_eeel_l_chns = np.asarray([])
    dim_2_eeel_nu_chns = np.asarray([], dtype=int)
    dim_2_eeel_k_clnkr_l_chns = dict.fromkeys(k_list, np.asarray([]))
    dim_2_eeel_k_clnkr_nu_chns = dict.fromkeys(k_list, np.asarray([], dtype=int))
    dim_2_eeel_k_clnkr_rho = np.empty((dim_2_sample_config_num, k_max+1))

    dim_3_eeel_l_chns = np.asarray([])
    dim_3_eeel_nu_chns = np.asarray([], dtype=int)
    dim_3_eeel_k_clnkr_l_chns = dict.fromkeys(k_list, np.asarray([]))
    dim_3_eeel_k_clnkr_nu_chns = dict.fromkeys(k_list, np.asarray([], dtype=int))
    dim_3_eeel_k_clnkr_rho = np.empty((dim_3_sample_config_num, k_max+1))

    dim_2_indx = 0
    dim_3_indx = 0
    for indx in range(sample_config_num):
        sample = int(sample_config_params_arr[indx, 0])
        dim = int(sample_config_params_arr[indx, 1])
        k = int(sample_config_params_arr[indx, 5])
        config = int(sample_config_params_arr[indx, 9])
        
        # Generate filenames
        L_filename = L_filename_str(
            cfg.label.network, cfg.label.date, cfg.label.batch, sample)
        aelp_filename = aelp_filename_str(
            cfg.label.network, cfg.label.date, cfg.label.batch, sample, config)
        coords_filename = aelp_filename + ".coords"
        core_nodes_type_filename = aelp_filename + "-core_nodes_type" + ".dat"
        conn_edges_filename = aelp_filename + "-conn_edges" + ".dat"
        conn_edges_type_filename = aelp_filename + "-conn_edges_type" + ".dat"
        l_cntr_conn_edges_filename = (
            aelp_filename + "-l_cntr_conn_edges" + ".dat"
        )

        # Load simulation box side lengths and node coordinates
        L = np.loadtxt(L_filename)
        coords = np.loadtxt(coords_filename)

        # Load fundamental graph constituents
        core_nodes_type = np.loadtxt(core_nodes_type_filename, dtype=int)
        core_nodes = np.arange(np.shape(core_nodes_type)[0], dtype=int)
        conn_edges = np.loadtxt(conn_edges_filename, dtype=int)
        conn_edges_type = np.loadtxt(conn_edges_type_filename, dtype=int)
        l_cntr_conn_edges = np.loadtxt(l_cntr_conn_edges_filename)
        nu_chns = l_cntr_conn_edges / b
        nu_chns = nu_chns.astype(int)
        m = np.shape(conn_edges)[0]

        # Calculate end-to-end chain length (Euclidean edge length)
        l_chns = l_func(conn_edges, conn_edges_type, coords, L)

        # Lexicographically sort the edges, end-to-end chain length, and
        # chain segment number
        conn_edges, lexsort_indcs = lexsorted_edges(
            conn_edges, return_indcs=True)
        l_chns = l_chns[lexsort_indcs]
        nu_chns = nu_chns[lexsort_indcs]
        
        # Create nx.MultiGraph, and add nodes before edges
        conn_graph = nx.MultiGraph()
        conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
        conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

        # Extract elastically-effective end-linked network
        eeel_conn_graph = elastically_effective_end_linked_graph(conn_graph)

        # Extract edges from the elastically-effective end-linked
        # network
        eeel_conn_edges = extract_edges_to_numpy_array(eeel_conn_graph)
        del eeel_conn_graph

        # Lexicographically sort the edges for the elastically-effective
        # end-linked network
        eeel_conn_edges = lexsorted_edges(eeel_conn_edges)
        eeel_m = np.shape(eeel_conn_edges)[0]

        # Downselect end-to-end chain length and chain segment number
        # for the elastically-effective end-linked network from the
        # as-provided network
        eeel_l_chns = np.empty(eeel_m)
        eeel_nu_chns = np.empty(eeel_m, dtype=int)
        eeel_edge = 0
        edge = 0
        while eeel_edge < eeel_m:
            if np.array_equal(eeel_conn_edges[eeel_edge], conn_edges[edge]):
                eeel_l_chns[eeel_edge] = l_chns[edge]
                eeel_nu_chns[eeel_edge] = nu_chns[edge]
                eeel_edge += 1
            edge += 1
        
        # Create nx.MultiGraph, and add nodes before edges and edge
        # attributes
        eeel_conn_graph = nx.MultiGraph()
        eeel_conn_graph = add_nodes_from_numpy_array(
            eeel_conn_graph, core_nodes)
        eeel_conn_graph = add_edges_and_edge_attributes_from_numpy_arrays(
            eeel_conn_graph, eeel_conn_edges, ["l", "nu"],
            eeel_l_chns, eeel_nu_chns)
        
        # Remove isolate nodes in order to maintain the extracted
        # elastically-effective end-linked network
        eeel_conn_graph.remove_nodes_from(list(nx.isolates(eeel_conn_graph)))
        
        if dim == 2:
            # End-to-end chain length for each/every chain in the
            # elastically-effective end-linked network
            dim_2_eeel_l_chns = np.concatenate((dim_2_eeel_l_chns, eeel_l_chns))
            
            # Chain segment number for each/every chain in the
            # elastically-effective end-linked network
            dim_2_eeel_nu_chns = np.concatenate(
                (dim_2_eeel_nu_chns, eeel_nu_chns), dtype=int)
            
            # Extract end-to-end chain length and chain segment number
            # for each chain connected to each node in the
            # elastically-effective end-linked network in a degree-wise
            # fashion
            for node in list(eeel_conn_graph.nodes()):
                # Initialize arrays
                dim_2_eeel_k_clnkr_l = np.asarray([])
                dim_2_eeel_k_clnkr_nu = np.asarray([], dtype=int)
                # Degree of node
                k = eeel_conn_graph.degree(node)
                # Unique edges connected to the node
                edges = np.unique(
                    np.sort(np.asarray(list(eeel_conn_graph.edges(node)), dtype=int), axis=1),
                    axis=0)
                for edge in range(np.shape(edges)[0]):
                    # Node numbers
                    node_0 = int(edges[edge, 0])
                    node_1 = int(edges[edge, 1])
                    # Get edge data
                    edge_data = eeel_conn_graph.get_edge_data(node_0, node_1)
                    # Determine how many multiedges begin and end with
                    # node_0 and node_1
                    multiedge_num = eeel_conn_graph.number_of_edges(
                        node_0, node_1)
                    for multiedge in range(multiedge_num):
                        # Store the end-to-end chain length and chain
                        # segment number for each multiedge
                        dim_2_eeel_k_clnkr_l = np.concatenate(
                            (dim_2_eeel_k_clnkr_l, np.asarray([edge_data[multiedge]["l"]])))
                        dim_2_eeel_k_clnkr_nu = np.concatenate(
                            (dim_2_eeel_k_clnkr_nu, np.asarray([edge_data[multiedge]["nu"]], dtype=int)),
                            dtype=int) 
                dim_2_eeel_k_clnkr_l_chns[k] = np.concatenate(
                    (dim_2_eeel_k_clnkr_l_chns[k], dim_2_eeel_k_clnkr_l))
                dim_2_eeel_k_clnkr_nu_chns[k] = np.concatenate(
                    (dim_2_eeel_k_clnkr_nu_chns[k], dim_2_eeel_k_clnkr_nu),
                    dtype=int)
            
            # Count elastically-effective end-linked network
            # cross-linker node degree occurances
            dim_2_eeel_k_clnkr_count = np.zeros(k_max+1, dtype=int)
            for node, k in list(eeel_conn_graph.degree()):
                # If dangling chain node, then continue to next node.
                # This should never occur in the elastically-effective
                # end-linked network.
                if core_nodes_type[node] == 0: continue
                # Update elastically-effective end-linked network
                # cross-linker node degree occurance count
                dim_2_eeel_k_clnkr_count[k] += 1
            dim_2_eeel_k_clnkr_count[0] = np.sum(dim_2_eeel_k_clnkr_count[1:])

            # Calculate degree-wise cross-linker number density
            A = A_or_V_arg_L_func(L)
            for k in range(k_max+1):
                dim_2_eeel_k_clnkr_rho[dim_2_indx, k] = rho_func(
                    dim_2_eeel_k_clnkr_count[k], A)
                        
            dim_2_indx += 1
        elif dim == 3:
            # End-to-end chain length for each/every chain in the
            # elastically-effective end-linked network
            dim_3_eeel_l_chns = np.concatenate((dim_3_eeel_l_chns, eeel_l_chns))
            
            # Chain segment number for each/every chain in the
            # elastically-effective end-linked network
            dim_3_eeel_nu_chns = np.concatenate(
                (dim_3_eeel_nu_chns, eeel_nu_chns), dtype=int)
            
            # Extract end-to-end chain length and chain segment number
            # for each chain connected to each node in the
            # elastically-effective end-linked network in a degree-wise
            # fashion
            for node in list(eeel_conn_graph.nodes()):
                # Initialize arrays
                dim_3_eeel_k_clnkr_l = np.asarray([])
                dim_3_eeel_k_clnkr_nu = np.asarray([], dtype=int)
                # Degree of node
                k = eeel_conn_graph.degree(node)
                # Unique edges connected to the node
                edges = np.unique(
                    np.sort(np.asarray(list(eeel_conn_graph.edges(node)), dtype=int), axis=1),
                    axis=0)
                for edge in range(np.shape(edges)[0]):
                    # Node numbers
                    node_0 = int(edges[edge, 0])
                    node_1 = int(edges[edge, 1])
                    # Get edge data
                    edge_data = eeel_conn_graph.get_edge_data(node_0, node_1)
                    # Determine how many multiedges begin and end with
                    # node_0 and node_1
                    multiedge_num = eeel_conn_graph.number_of_edges(
                        node_0, node_1)
                    for multiedge in range(multiedge_num):
                        # Store the end-to-end chain length and chain
                        # segment number for each multiedge
                        dim_3_eeel_k_clnkr_l = np.concatenate(
                            (dim_3_eeel_k_clnkr_l, np.asarray([edge_data[multiedge]["l"]])))
                        dim_3_eeel_k_clnkr_nu = np.concatenate(
                            (dim_3_eeel_k_clnkr_nu, np.asarray([edge_data[multiedge]["nu"]], dtype=int)),
                            dtype=int) 
                dim_3_eeel_k_clnkr_l_chns[k] = np.concatenate(
                    (dim_3_eeel_k_clnkr_l_chns[k], dim_3_eeel_k_clnkr_l))
                dim_3_eeel_k_clnkr_nu_chns[k] = np.concatenate(
                    (dim_3_eeel_k_clnkr_nu_chns[k], dim_3_eeel_k_clnkr_nu),
                    dtype=int)
            
            # Count elastically-effective end-linked network
            # cross-linker node degree occurances
            dim_3_eeel_k_clnkr_count = np.zeros(k_max+1, dtype=int)
            for node, k in list(eeel_conn_graph.degree()):
                # If dangling chain node, then continue to next node.
                # This should never occur in the elastically-effective
                # end-linked network.
                if core_nodes_type[node] == 0: continue
                # Update elastically-effective end-linked network
                # cross-linker node degree occurance count
                dim_3_eeel_k_clnkr_count[k] += 1
            dim_3_eeel_k_clnkr_count[0] = np.sum(dim_3_eeel_k_clnkr_count[1:])

            # Calculate degree-wise cross-linker number density
            V = A_or_V_arg_L_func(L)
            for k in range(k_max+1):
                dim_3_eeel_k_clnkr_rho[dim_3_indx, k] = rho_func(
                    dim_3_eeel_k_clnkr_count[k], V)
                        
            dim_3_indx += 1
    
    # Total cross-linker number density
    dim_2_eeel_clnkr_rho_tot = np.max(dim_2_eeel_k_clnkr_rho[:, 0])
    dim_3_eeel_clnkr_rho_tot = np.max(dim_3_eeel_k_clnkr_rho[:, 0])

    print("Two-dimensional eeel apelp network cross-linker number density = {}".format(dim_2_eeel_clnkr_rho_tot))
    print("Three-dimensional eeel apelp network cross-linker number density = {}".format(dim_3_eeel_clnkr_rho_tot))

    # Density histogram of number density of k=3 and k=4 cross-linker nodes
    xlabel = "rho"
    dim_2_eeel_clnkr_rho_first_bin = 0.0
    dim_2_eeel_clnkr_rho_last_bin = dim_2_eeel_clnkr_rho_tot
    
    dim_3_eeel_clnkr_rho_first_bin = 0.0
    dim_3_eeel_clnkr_rho_last_bin = dim_3_eeel_clnkr_rho_tot
    
    # Density histogram preformatting
    dim_2_eeel_clnkr_rho_bin_steps = 81
    dim_2_eeel_clnkr_rho_bins = np.linspace(
        dim_2_eeel_clnkr_rho_first_bin, dim_2_eeel_clnkr_rho_last_bin,
        dim_2_eeel_clnkr_rho_bin_steps)
    dim_2_eeel_clnkr_rho_steps = 5
    dim_2_xticks = np.linspace(
        dim_2_eeel_clnkr_rho_first_bin, dim_2_eeel_clnkr_rho_last_bin,
        dim_2_eeel_clnkr_rho_steps)
    dim_2_eeel_clnkr_rho_dnstyhist_filename = (
        filepath + "dim_2" + "-" + "eeel_clnkr_rho" + "-" + "dnstyhist" + ".png"
    )
    
    dim_3_eeel_clnkr_rho_bin_steps = 81
    dim_3_eeel_clnkr_rho_bins = np.linspace(
        dim_3_eeel_clnkr_rho_first_bin, dim_3_eeel_clnkr_rho_last_bin,
        dim_3_eeel_clnkr_rho_bin_steps)
    dim_3_eeel_clnkr_rho_steps = 5
    dim_3_xticks = np.linspace(
        dim_3_eeel_clnkr_rho_first_bin, dim_3_eeel_clnkr_rho_last_bin,
        dim_3_eeel_clnkr_rho_steps)
    dim_3_eeel_clnkr_rho_dnstyhist_filename = (
        filepath + "dim_3" + "-" + "eeel_clnkr_rho" + "-" + "dnstyhist" + ".png"
    )

    fig, axs = plt.subplots(k_max-2)
    for k in range(k_max+1):
        if k == 0 or k == 1 or k == 2: continue
        else:
            axs[k-3].hist(
                dim_2_eeel_k_clnkr_rho[:, k], bins=dim_2_eeel_clnkr_rho_bins,
                density=True, color="tab:blue", edgecolor="black", zorder=3)
            axs[k-3].set_xticks(dim_2_xticks)
            axs[k-3].set_title("k = {}".format(k))
            axs[k-3].grid(True, alpha=0.25, zorder=0)
    axs[-1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(dim_2_eeel_clnkr_rho_dnstyhist_filename)
    plt.close()

    fig, axs = plt.subplots(k_max-2)
    for k in range(k_max+1):
        if k == 0 or k == 1 or k == 2: continue
        else:
            axs[k-3].hist(
                dim_3_eeel_k_clnkr_rho[:, k], bins=dim_3_eeel_clnkr_rho_bins,
                density=True, color="tab:blue", edgecolor="black", zorder=3)
            axs[k-3].set_xticks(dim_3_xticks)
            axs[k-3].set_title("k = {}".format(k))
            axs[k-3].grid(True, alpha=0.25, zorder=0)
    axs[-1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(dim_3_eeel_clnkr_rho_dnstyhist_filename)
    plt.close()

    # Density histogram of chain segment number for chains connected to
    # k=3 and k=4 eeel cross-linker nodes
    xlabel = "nu"
    dim_2_eeel_clnkr_nu_first_bin = 1
    dim_2_eeel_clnkr_nu_last_bin = nu_max
    dim_2_eeel_clnkr_nu_bin_inc = 1

    dim_3_eeel_clnkr_nu_first_bin = 1
    dim_3_eeel_clnkr_nu_last_bin = nu_max
    dim_3_eeel_clnkr_nu_bin_inc = 1
    
    # Density histogram preformatting
    dim_2_eeel_clnkr_nu_bin_steps = (
        int(np.around((dim_2_eeel_clnkr_nu_last_bin-dim_2_eeel_clnkr_nu_first_bin)/dim_2_eeel_clnkr_nu_bin_inc))
        + 1
    )
    dim_2_eeel_clnkr_nu_bins = np.linspace(
        dim_2_eeel_clnkr_nu_first_bin, dim_2_eeel_clnkr_nu_last_bin, dim_2_eeel_clnkr_nu_bin_steps)
    dim_2_xticks = np.linspace(0, dim_2_eeel_clnkr_nu_last_bin, 11)
    dim_2_eeel_clnkr_nu_dnstyhist_filename = (
        filepath + "dim_2" + "-" + "eeel_clnkr_nu" + "-" + "dnstyhist" + ".png"
    )

    dim_3_eeel_clnkr_nu_bin_steps = (
        int(np.around((dim_3_eeel_clnkr_nu_last_bin-dim_3_eeel_clnkr_nu_first_bin)/dim_3_eeel_clnkr_nu_bin_inc))
        + 1
    )
    dim_3_eeel_clnkr_nu_bins = np.linspace(
        dim_3_eeel_clnkr_nu_first_bin, dim_3_eeel_clnkr_nu_last_bin, dim_3_eeel_clnkr_nu_bin_steps)
    dim_3_xticks = np.linspace(0, dim_3_eeel_clnkr_nu_last_bin, 11)
    dim_3_eeel_clnkr_nu_dnstyhist_filename = (
        filepath + "dim_3" + "-" + "eeel_clnkr_nu" + "-" + "dnstyhist" + ".png"
    )
    
    fig, axs = plt.subplots(k_max-2)
    for k in range(k_max+1):
        if k == 0 or k == 1 or k == 2: continue
        else:
            axs[k-3].hist(
                dim_2_eeel_k_clnkr_nu_chns[k], bins=dim_2_eeel_clnkr_nu_bins,
                density=True, color="tab:blue", zorder=3)
            axs[k-3].set_xticks(dim_2_xticks)
            axs[k-3].set_title("k = {}".format(k))
            axs[k-3].grid(True, alpha=0.25, zorder=0)
    axs[-1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(dim_2_eeel_clnkr_nu_dnstyhist_filename)
    plt.close()

    fig, axs = plt.subplots(k_max-2)
    for k in range(k_max+1):
        if k == 0 or k == 1 or k == 2: continue
        else:
            axs[k-3].hist(
                dim_3_eeel_k_clnkr_nu_chns[k], bins=dim_3_eeel_clnkr_nu_bins,
                density=True, color="tab:blue", zorder=3)
            axs[k-3].set_xticks(dim_3_xticks)
            axs[k-3].set_title("k = {}".format(k))
            axs[k-3].grid(True, alpha=0.25, zorder=0)
    axs[-1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(dim_3_eeel_clnkr_nu_dnstyhist_filename)
    plt.close()

    # Density histogram of chain segment number for all chains
    dim_2_eeel_nu_bins = dim_2_eeel_clnkr_nu_bins.copy()
    dim_2_eeel_nu_dnstyhist_filename = (
        filepath + "dim_2" + "-" + "eeel_nu" + "-" + "dnstyhist" + ".png"
    )

    dim_3_eeel_nu_bins = dim_3_eeel_clnkr_nu_bins.copy()
    dim_3_eeel_nu_dnstyhist_filename = (
        filepath + "dim_3" + "-" + "eeel_nu" + "-" + "dnstyhist" + ".png"
    )
    
    fig, axs = plt.subplots()
    axs.hist(
        dim_2_eeel_nu_chns, bins=dim_2_eeel_nu_bins, density=True,
        color="tab:blue", zorder=3)
    axs.set_xticks(dim_2_xticks)
    axs.grid(True, alpha=0.25, zorder=0)
    axs.set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(dim_2_eeel_nu_dnstyhist_filename)
    plt.close()

    fig, axs = plt.subplots()
    axs.hist(
        dim_3_eeel_nu_chns, bins=dim_3_eeel_nu_bins, density=True,
        color="tab:blue", zorder=3)
    axs.set_xticks(dim_3_xticks)
    axs.grid(True, alpha=0.25, zorder=0)
    axs.set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(dim_3_eeel_nu_dnstyhist_filename)
    plt.close()

    # Density histogram of end-to-end chain length for all chains
    xlabel = "l"
    dim_2_eeel_l_first_bin = 0
    dim_2_eeel_l_last_bin = np.max(dim_2_eeel_l_chns)
    dim_2_eeel_l_bin_steps = 101

    dim_3_eeel_l_first_bin = 0
    dim_3_eeel_l_last_bin = np.max(dim_3_eeel_l_chns)
    dim_3_eeel_l_bin_steps = 101
    
    # Density histogram preformatting
    dim_2_eeel_l_bins = np.linspace(
        dim_2_eeel_l_first_bin, dim_2_eeel_l_last_bin, dim_2_eeel_l_bin_steps)
    dim_2_xticks = np.linspace(0, dim_2_eeel_l_last_bin, 11)
    dim_2_eeel_l_dnstyhist_filename = (
        filepath + "dim_2" + "-" + "eeel_l" + "-" + "dnstyhist" + ".png"
    )

    dim_3_eeel_l_bins = np.linspace(
        dim_3_eeel_l_first_bin, dim_3_eeel_l_last_bin, dim_3_eeel_l_bin_steps)
    dim_3_xticks = np.linspace(0, dim_3_eeel_l_last_bin, 11)
    dim_3_eeel_l_dnstyhist_filename = (
        filepath + "dim_3" + "-" + "eeel_l" + "-" + "dnstyhist" + ".png"
    )

    fig, axs = plt.subplots()
    axs.hist(
        dim_2_eeel_l_chns, bins=dim_2_eeel_l_bins, density=True,
        color="tab:blue", zorder=3)
    axs.set_xticks(dim_2_xticks)
    axs.grid(True, alpha=0.25, zorder=0)
    axs.set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(dim_2_eeel_l_dnstyhist_filename)
    plt.close()

    fig, axs = plt.subplots()
    axs.hist(
        dim_3_eeel_l_chns, bins=dim_3_eeel_l_bins, density=True,
        color="tab:blue", zorder=3)
    axs.set_xticks(dim_3_xticks)
    axs.grid(True, alpha=0.25, zorder=0)
    axs.set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(dim_3_eeel_l_dnstyhist_filename)
    plt.close()

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Artificial polydisperse end-linked polymer network statistical analysis took {execution_time} seconds to run")