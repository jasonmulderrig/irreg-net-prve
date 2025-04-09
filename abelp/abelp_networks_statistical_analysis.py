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
from networks.abelp_networks_config import (
    abelpConfig,
    sample_config_params_arr_func
)

# Hydra ConfigStore initialization
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(name="config", node=abelpConfig)

@hydra.main(version_base=None, config_path=".", config_name="abelp_networks_config")
def main(cfg: abelpConfig) -> None:
    sample_config_params_arr, sample_config_num = sample_config_params_arr_func(cfg)
    filepath = filepath_str(cfg.label.network)

    dim_2_sample_config_num = np.count_nonzero(sample_config_params_arr[:, 1]==2)
    dim_3_sample_config_num = np.count_nonzero(sample_config_params_arr[:, 1]==3)

    b = cfg.topology.b[0]

    k_max = int(np.max(sample_config_params_arr[:, 5]))
    k_list = list(range(k_max+1))
    en_min = int(np.max(sample_config_params_arr[:, 8]))
    en_max = int(np.max(sample_config_params_arr[:, 9]))
    en_chn = np.asarray([en_min, en_max], dtype=int)
    nu_min = en_min - 1
    nu_max = en_max - 1
    nu_chn = en_chn - 1
    
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
        config = int(sample_config_params_arr[indx, 10])
        
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
    nu_bins = np.asarray([nu_min-0.5, nu_min+0.5, nu_max-0.5, nu_max+0.5])
    dim_2_nu_dnstyhist_filename = (
        filepath + "dim_2" + "-" + "nu" + "-" + "dnstyhist" + ".png"
    )
    dim_3_nu_dnstyhist_filename = (
        filepath + "dim_3" + "-" + "nu" + "-" + "dnstyhist" + ".png"
    )
    
    fig, axs = plt.subplots()
    axs.hist(
        dim_2_nu_chns, bins=nu_bins, density=True, align="mid", color="tab:blue", zorder=3)
    axs.set_xticks(nu_chn)
    axs.grid(True, alpha=0.25, zorder=0)
    axs.set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(dim_2_nu_dnstyhist_filename)
    plt.close()

    fig, axs = plt.subplots()
    axs.hist(
        dim_3_nu_chns, bins=nu_bins, density=True, align="mid", color="tab:blue", zorder=3)
    axs.set_xticks(nu_chn)
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

    dim_2_eeel_k_clnkr_type_rho = dict.fromkeys(k_list, np.asarray([]))
    dim_3_eeel_k_clnkr_type_rho = dict.fromkeys(k_list, np.asarray([]))
    for k in range(k_max+1):
        dim_2_eeel_k_clnkr_type_rho[k] = np.empty((dim_2_sample_config_num, k+2))
        dim_3_eeel_k_clnkr_type_rho[k] = np.empty((dim_3_sample_config_num, k+2))

    dim_2_indx = 0
    dim_3_indx = 0
    for indx in range(sample_config_num):
        sample = int(sample_config_params_arr[indx, 0])
        dim = int(sample_config_params_arr[indx, 1])
        k = int(sample_config_params_arr[indx, 5])
        config = int(sample_config_params_arr[indx, 10])
        
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
            # fashion. In addition, count elastically-effective
            # end-linked network cross-linker type occurances
            dim_2_eeel_k_clnkr_type_count = dict.fromkeys(k_list, np.asarray([]))
            for k in range(k_max+1):
                dim_2_eeel_k_clnkr_type_count[k] = np.zeros(k+2, dtype=int)
            
            for node in list(eeel_conn_graph.nodes()):
                # Initialize arrays and maximum chain segment number
                # count
                dim_2_eeel_k_clnkr_l = np.asarray([])
                dim_2_eeel_k_clnkr_nu = np.asarray([], dtype=int)
                nu_max_count = 0
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
                        if int(edge_data[multiedge]["nu"]) == nu_max:
                            nu_max_count += 1
                dim_2_eeel_k_clnkr_l_chns[k] = np.concatenate(
                    (dim_2_eeel_k_clnkr_l_chns[k], dim_2_eeel_k_clnkr_l))
                dim_2_eeel_k_clnkr_nu_chns[k] = np.concatenate(
                    (dim_2_eeel_k_clnkr_nu_chns[k], dim_2_eeel_k_clnkr_nu),
                    dtype=int)
                # Cross-linker type is equal to nu_max_count+1
                dim_2_eeel_k_clnkr_type_count[k][nu_max_count+1] += 1
            
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

            # Calculate degree-wise cross-linker number density and
            # degree-wise cross-linker type number density
            A = A_or_V_arg_L_func(L)
            for k in range(k_max+1):
                dim_2_eeel_k_clnkr_rho[dim_2_indx, k] = rho_func(
                    dim_2_eeel_k_clnkr_count[k], A)
                for clnkr_type in range(k+2):
                    dim_2_eeel_k_clnkr_type_rho[k][dim_2_indx, clnkr_type] = (
                        rho_func(
                            dim_2_eeel_k_clnkr_type_count[k][clnkr_type], A)
                    )
       
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
            # fashion. In addition, count elastically-effective
            # end-linked network cross-linker type occurances
            dim_3_eeel_k_clnkr_type_count = dict.fromkeys(k_list, np.asarray([]))
            for k in range(k_max+1):
                dim_3_eeel_k_clnkr_type_count[k] = np.zeros(k+2, dtype=int)
            
            for node in list(eeel_conn_graph.nodes()):
                # Initialize arrays and maximum chain segment number
                # count
                dim_3_eeel_k_clnkr_l = np.asarray([])
                dim_3_eeel_k_clnkr_nu = np.asarray([], dtype=int)
                nu_max_count = 0
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
                        if int(edge_data[multiedge]["nu"]) == nu_max:
                            nu_max_count += 1
                dim_3_eeel_k_clnkr_l_chns[k] = np.concatenate(
                    (dim_3_eeel_k_clnkr_l_chns[k], dim_3_eeel_k_clnkr_l))
                dim_3_eeel_k_clnkr_nu_chns[k] = np.concatenate(
                    (dim_3_eeel_k_clnkr_nu_chns[k], dim_3_eeel_k_clnkr_nu),
                    dtype=int)
                # Cross-linker type is equal to nu_max_count+1
                dim_3_eeel_k_clnkr_type_count[k][nu_max_count+1] += 1
            
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

            # Calculate degree-wise cross-linker number density and
            # degree-wise cross-linker type number density
            V = A_or_V_arg_L_func(L)
            for k in range(k_max+1):
                dim_3_eeel_k_clnkr_rho[dim_3_indx, k] = rho_func(
                    dim_3_eeel_k_clnkr_count[k], V)
                for clnkr_type in range(k+2):
                    dim_3_eeel_k_clnkr_type_rho[k][dim_3_indx, clnkr_type] = (
                        rho_func(
                            dim_3_eeel_k_clnkr_type_count[k][clnkr_type], A)
                    )
       
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

    # Density histogram of number density of k=3 and k=4 cross-linker types
    dim_2_eeel_clnkr_type_rho_bins = dim_2_eeel_clnkr_rho_bins.copy()
    dim_3_eeel_clnkr_type_rho_bins = dim_3_eeel_clnkr_rho_bins.copy()
    dim_2_k_3_eeel_clnkr_type_rho_dnstyhist_filename = (
        filepath + "dim_2" + "-" + "k_3" + "-" + "eeel_clnkr_type_rho" + "-"
        + "dnstyhist" + ".png"
    )
    dim_2_k_4_eeel_clnkr_type_rho_dnstyhist_filename = (
        filepath + "dim_2" + "-" + "k_4" + "-" + "eeel_clnkr_type_rho" + "-"
        + "dnstyhist" + ".png"
    )
    dim_3_k_3_eeel_clnkr_type_rho_dnstyhist_filename = (
        filepath + "dim_3" + "-" + "k_3" + "-" + "eeel_clnkr_type_rho" + "-"
        + "dnstyhist" + ".png"
    )
    dim_3_k_4_eeel_clnkr_type_rho_dnstyhist_filename = (
        filepath + "dim_3" + "-" + "k_4" + "-" + "eeel_clnkr_type_rho" + "-"
        + "dnstyhist" + ".png"
    )

    k = 3
    fig, axs = plt.subplots(k+1)
    for clnkr_type in range(k+2):
        if clnkr_type == 0: continue
        axs[clnkr_type-1].hist(
            dim_2_eeel_k_clnkr_type_rho[k][:, clnkr_type],
            bins=dim_2_eeel_clnkr_type_rho_bins, density=True, color="tab:blue",
            edgecolor="black", zorder=3)
        axs[clnkr_type-1].set_xticks(dim_2_xticks)
        axs[clnkr_type-1].set_title("Type {}".format(clnkr_type))
        axs[clnkr_type-1].grid(True, alpha=0.25, zorder=0)
    axs[-1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(dim_2_k_3_eeel_clnkr_type_rho_dnstyhist_filename)
    plt.close()

    fig, axs = plt.subplots(k+1)
    for clnkr_type in range(k+2):
        if clnkr_type == 0: continue
        axs[clnkr_type-1].hist(
            dim_3_eeel_k_clnkr_type_rho[k][:, clnkr_type],
            bins=dim_3_eeel_clnkr_type_rho_bins, density=True, color="tab:blue",
            edgecolor="black", zorder=3)
        axs[clnkr_type-1].set_xticks(dim_3_xticks)
        axs[clnkr_type-1].set_title("Type {}".format(clnkr_type))
        axs[clnkr_type-1].grid(True, alpha=0.25, zorder=0)
    axs[-1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(dim_3_k_3_eeel_clnkr_type_rho_dnstyhist_filename)
    plt.close()

    k = 4
    fig, axs = plt.subplots(k+1)
    for clnkr_type in range(k+2):
        if clnkr_type == 0: continue
        axs[clnkr_type-1].hist(
            dim_2_eeel_k_clnkr_type_rho[k][:, clnkr_type],
            bins=dim_2_eeel_clnkr_type_rho_bins, density=True, color="tab:blue",
            edgecolor="black", zorder=3)
        axs[clnkr_type-1].set_xticks(dim_2_xticks)
        axs[clnkr_type-1].set_title("Type {}".format(clnkr_type))
        axs[clnkr_type-1].grid(True, alpha=0.25, zorder=0)
    axs[-1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(dim_2_k_4_eeel_clnkr_type_rho_dnstyhist_filename)
    plt.close()

    fig, axs = plt.subplots(k+1)
    for clnkr_type in range(k+2):
        if clnkr_type == 0: continue
        axs[clnkr_type-1].hist(
            dim_3_eeel_k_clnkr_type_rho[k][:, clnkr_type],
            bins=dim_3_eeel_clnkr_type_rho_bins, density=True, color="tab:blue",
            edgecolor="black", zorder=3)
        axs[clnkr_type-1].set_xticks(dim_3_xticks)
        axs[clnkr_type-1].set_title("Type {}".format(clnkr_type))
        axs[clnkr_type-1].grid(True, alpha=0.25, zorder=0)
    axs[-1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(dim_3_k_4_eeel_clnkr_type_rho_dnstyhist_filename)
    plt.close()
    
    # Density histogram of chain segment number for chains connected to
    # k=3 and k=4 eeel cross-linker nodes, and density histogram of
    # chain segment number for all chains
    xlabel = "nu"
    nu_bins = np.asarray([nu_min-0.5, nu_min+0.5, nu_max-0.5, nu_max+0.5])
    dim_2_eeel_clnkr_nu_dnstyhist_filename = (
        filepath + "dim_2" + "-" + "eeel_clnkr_nu" + "-" + "dnstyhist" + ".png"
    )
    dim_3_eeel_clnkr_nu_dnstyhist_filename = (
        filepath + "dim_3" + "-" + "eeel_clnkr_nu" + "-" + "dnstyhist" + ".png"
    )
    dim_2_eeel_nu_dnstyhist_filename = (
        filepath + "dim_2" + "-" + "eeel_nu" + "-" + "dnstyhist" + ".png"
    )
    dim_3_eeel_nu_dnstyhist_filename = (
        filepath + "dim_3" + "-" + "eeel_nu" + "-" + "dnstyhist" + ".png"
    )

    fig, axs = plt.subplots(k_max-2)
    for k in range(k_max+1):
        if k == 0 or k == 1 or k == 2: continue
        else:
            axs[k-3].hist(
                dim_2_eeel_k_clnkr_nu_chns[k], bins=nu_bins, density=True,
                align="mid", color="tab:blue", zorder=3)
            axs[k-3].set_xticks(nu_chn)
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
                dim_3_eeel_k_clnkr_nu_chns[k], bins=nu_bins, density=True,
                align="mid", color="tab:blue", zorder=3)
            axs[k-3].set_xticks(nu_chn)
            axs[k-3].set_title("k = {}".format(k))
            axs[k-3].grid(True, alpha=0.25, zorder=0)
    axs[-1].set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(dim_3_eeel_clnkr_nu_dnstyhist_filename)
    plt.close()

    fig, axs = plt.subplots()
    axs.hist(
        dim_2_eeel_nu_chns, bins=nu_bins, density=True, align="mid",
        color="tab:blue", zorder=3)
    axs.set_xticks(nu_chn)
    axs.grid(True, alpha=0.25, zorder=0)
    axs.set_xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(dim_2_eeel_nu_dnstyhist_filename)
    plt.close()

    fig, axs = plt.subplots()
    axs.hist(
        dim_3_eeel_nu_chns, bins=nu_bins, density=True, align="mid",
        color="tab:blue", zorder=3)
    axs.set_xticks(nu_chn)
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
    print(f"Artificial bimodal end-linked polymer network statistical analysis took {execution_time} seconds to run")