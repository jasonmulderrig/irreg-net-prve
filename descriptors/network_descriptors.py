import numpy as np
import networkx as nx
from helpers.graph_utils import (
    lexsorted_edges,
    add_nodes_from_numpy_array,
    add_edges_from_numpy_array,
    add_edges_and_edge_attributes_from_numpy_arrays,
    extract_nodes_to_numpy_array,
    extract_edges_to_numpy_array,
    largest_connected_component,
    elastically_effective_end_linked_graph
)
from descriptors.general_topological_descriptors import (
    l_func,
    l_inv_func,
    gamma_func,
    gamma_inv_func
)
import descriptors.nodal_degree_topological_descriptors
import descriptors.shortest_path_topological_descriptors
import descriptors.general_topological_descriptors
import descriptors.morphological_descriptors

def network_local_topological_descriptor(
        b: float,
        L: np.ndarray,
        coords: np.ndarray,
        core_nodes: np.ndarray,
        conn_edges: np.ndarray,
        conn_edges_type: np.ndarray,
        l_cntr_conn_edges: np.ndarray,
        multigraph: bool,
        result_filename: str,
        tplgcl_dscrptr: str,
        eeel_ntwrk: bool,
        save_result: bool,
        return_result: bool) -> np.ndarray | float | int | None:
    """Local network topological descriptor.
    
    This function calculates the result of a local topological
    descriptor for a supplied network. Several options can be activated
    to this end: the elastically-effective end-linked network from the
    supplied network may be extracted and the resulting local
    topological descriptor can be saved and returned.

    Args:
        b (float): Chain segment and/or cross-linker diameter.
        L (np.ndarray): Simulation box side lengths.
        coords (np.ndarray): Coordinates of the core nodes.
        core_nodes (np.ndarray): Core nodes.
        conn_edges (np.ndarray): Edges from the graph capturing the periodic connections between the core nodes.
        conn_edges_type (np.ndarray): Type label for the edges from the graph capturing the periodic connections between the core nodes. Core edges are of type 1, and periodic boundary edges are of type 0.
        l_cntr_conn_edges (np.ndarray): Contour length of the edges from the graph capturing the periodic connections between the core nodes.
        multigraph (bool): Boolean indicating if a nx.Graph (False) or nx.MultiGraph (True) object is needed to represent the supplied network.
        result_filename (str): Filename for the local topological descriptor result.
        tplgcl_dscrptr (str): Topological descriptor name.
        eeel_ntwrk (bool): Boolean indicating if the elastically-effective end-linked network ought to be supplied for the local topological descriptor calculation.
        save_result (bool): Boolean indicating if the result ought to be saved.
        return_result (bool): Boolean indicating if the result ought to be returned.
    
    Returns:
        np.ndarray | float | int | None: Local topological descriptor
        result.
    
    """
    # Nodewise topological descriptors
    node_tplgcl_dscrptr_list = [
        "k", "avrg_nn_k", "avrg_k_diff", "c", "lcl_avrg_kappa", "epsilon",
        "l_attr_epsilon", "l_inv_attr_epsilon", "gamma_attr_epsilon",
        "gamma_inv_attr_epsilon", "avrg_d", "avrg_l_attr_d",
        "avrg_l_inv_attr_d", "avrg_gamma_attr_d", "avrg_gamma_inv_attr_d",
        "avrg_e", "avrg_l_attr_e", "avrg_l_inv_attr_e", "avrg_gamma_attr_e",
        "avrg_gamma_inv_attr_e", "bc", "l_attr_bc", "l_inv_attr_bc",
        "gamma_attr_bc", "gamma_inv_attr_bc", "cc", "l_attr_cc",
        "l_inv_attr_cc", "gamma_attr_cc", "gamma_inv_attr_cc"
    ]
    # Edgewise topological descriptors
    edge_tplgcl_dscrptr_list = [
        "l", "l_inv", "l_cmpnts", "l_naive", "l_cntr", "gamma", "gamma_inv",
        "k_diff", "ebc", "l_attr_ebc", "l_inv_attr_ebc", "gamma_attr_ebc",
        "gamma_inv_attr_ebc"
    ]
    # Edge length topological descriptors
    l_tplgcl_dscrptr_list = ["l", "l_inv", "l_cmpnts"]
    # Chain/edge stretch topological descriptors
    gamma_tplgcl_dscrptr_list = ["gamma", "gamma_inv"]
    # Edge length attributed topological descriptors
    l_attr_tplgcl_dscrptr_list = [
        "l_attr_epsilon", "avrg_l_attr_d", "avrg_l_attr_e", "l_attr_bc",
        "l_attr_ebc", "l_attr_cc"
    ]
    # Inverse edge length attributed topological descriptors
    l_inv_attr_tplgcl_dscrptr_list = [
        "l_inv_attr_epsilon", "avrg_l_inv_attr_d", "avrg_l_inv_attr_e",
        "l_inv_attr_bc", "l_inv_attr_ebc", "l_inv_attr_cc"
    ]
    # Edge stretch attributed topological descriptors
    gamma_attr_tplgcl_dscrptr_list = [
        "gamma_attr_epsilon", "avrg_gamma_attr_d", "avrg_gamma_attr_e",
        "gamma_attr_bc", "gamma_attr_ebc", "gamma_attr_cc"
    ]
    # Inverse edge stretch attributed topological descriptors
    gamma_inv_attr_tplgcl_dscrptr_list = [
        "gamma_inv_attr_epsilon", "avrg_gamma_inv_attr_d",
        "avrg_gamma_inv_attr_e", "gamma_inv_attr_bc", "gamma_inv_attr_ebc",
        "gamma_inv_attr_cc"
    ]
    # Integer-based topological descriptors
    dtype_int_tplgcl_dscrptr_list = ["k", "k_diff", "epsilon"]
    
    # Extract number of nodes, number of edges, and dimensionality of
    # the as-provided network
    n = np.shape(core_nodes)[0]
    m = np.shape(conn_edges)[0]
    dim = np.shape(L)[0]

    # Exit if the topological descriptor is incompatible with the
    # possible nodewise and edgewise topological descriptors
    if (tplgcl_dscrptr not in node_tplgcl_dscrptr_list) \
        and (tplgcl_dscrptr not in edge_tplgcl_dscrptr_list):
            error_str = (
                "The topological descriptor is incompatible with the "
                + "possible nodewise and edgewise topological "
                + "descriptors. Please modify the requested "
                + "topological descriptor accordingly."
            )
            print(error_str)
            return None
    
    # Modify topological descriptor string to match function name
    # convention
    tplgcl_dscrptr_func_str = tplgcl_dscrptr + "_func"

    # Probe each topological descriptors module to identify the
    # topological descriptor calculation function
    if hasattr(descriptors.nodal_degree_topological_descriptors, tplgcl_dscrptr_func_str):
        tplgcl_dscrptr_func = getattr(
            descriptors.nodal_degree_topological_descriptors, tplgcl_dscrptr_func_str)
    elif hasattr(descriptors.shortest_path_topological_descriptors, tplgcl_dscrptr_func_str):
        tplgcl_dscrptr_func = getattr(
            descriptors.shortest_path_topological_descriptors, tplgcl_dscrptr_func_str)
    elif hasattr(descriptors.general_topological_descriptors, tplgcl_dscrptr_func_str):
        tplgcl_dscrptr_func = getattr(
            descriptors.general_topological_descriptors, tplgcl_dscrptr_func_str)
    else:
        error_str = (
            "The topological descriptor ``" + tplgcl_dscrptr + "'' is "
            + "not implemented!"
        )
        print(error_str)
        return None
    
    # Determine if the topological descriptor is a nodewise topological
    # descriptor or an edgewise topological descriptor
    if tplgcl_dscrptr in node_tplgcl_dscrptr_list:
        tplgcl_dscrptr_type = "nodewise"
    elif tplgcl_dscrptr in edge_tplgcl_dscrptr_list:
        tplgcl_dscrptr_type = "edgewise"

    # Determine if the topological descriptor is integer-based or not
    if tplgcl_dscrptr in dtype_int_tplgcl_dscrptr_list: dtype_int = True
    else: dtype_int = False

    # For the as-provided network, sort the node numbers in ascending
    # order, and correspondingly sort both the core nodes type and the
    # coordinates
    argsort_indcs = np.argsort(core_nodes)
    core_nodes = core_nodes[argsort_indcs]
    coords = coords[argsort_indcs]

    # For the as-provided network, lexicographically sort the edges,
    # edges type, and edges contour length
    conn_edges, lexsort_indcs = lexsorted_edges(conn_edges, return_indcs=True)
    conn_edges_type = conn_edges_type[lexsort_indcs]
    l_cntr_conn_edges = l_cntr_conn_edges[lexsort_indcs]

    # Create an array to store the eventual topological descriptor
    # result
    if tplgcl_dscrptr_type == "nodewise":
        if dtype_int: tplgcl_dscrptr_result = np.zeros(n, dtype=int)
        else: tplgcl_dscrptr_result = np.zeros(n)
    elif tplgcl_dscrptr_type == "edgewise":
        if tplgcl_dscrptr == "l_cmpnts":
            tplgcl_dscrptr_result = np.zeros((m, dim))
        elif dtype_int: tplgcl_dscrptr_result = np.zeros(m, dtype=int)
        else: tplgcl_dscrptr_result = np.zeros(m)

    # Create nx.Graph or nx.MultiGraph by adding nodes before edges for
    # the as-provided network
    if multigraph: conn_graph = nx.MultiGraph()
    else: conn_graph = nx.Graph()
    conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
    conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

    # If called for, extract the elastically-effective end-linked
    # network. Otherwise, extract the largest connected component.
    if eeel_ntwrk == True:
        altrd_conn_graph = elastically_effective_end_linked_graph(conn_graph)
    else: altrd_conn_graph = largest_connected_component(conn_graph)
    
    # Extract edges from the altered network
    altrd_conn_edges = extract_edges_to_numpy_array(altrd_conn_graph)
    del altrd_conn_graph

    # Lexicographically sort the edges for the altered network
    altrd_conn_edges = lexsorted_edges(altrd_conn_edges)
    altrd_m = np.shape(altrd_conn_edges)[0]

    # Downselect edges type and edges contour length for the altered
    # network from the as-provided network
    altrd_conn_edges_type = np.empty(altrd_m, dtype=int)
    altrd_l_cntr_conn_edges = np.empty(altrd_m)
    altrd_edge = 0
    edge = 0
    while altrd_edge < altrd_m:
        if np.array_equal(altrd_conn_edges[altrd_edge], conn_edges[edge]):
            altrd_conn_edges_type[altrd_edge] = conn_edges_type[edge]
            altrd_l_cntr_conn_edges[altrd_edge] = l_cntr_conn_edges[edge]
            altrd_edge += 1
        edge += 1
    
    # Calculate the topological descriptor for the altered network
    if tplgcl_dscrptr in l_tplgcl_dscrptr_list:
        altrd_tplgcl_dscrptr_result = tplgcl_dscrptr_func(
            altrd_conn_edges, altrd_conn_edges_type, coords, L)
    elif tplgcl_dscrptr == "l_naive":
        altrd_tplgcl_dscrptr_result = tplgcl_dscrptr_func(
            altrd_conn_edges, coords)
    elif tplgcl_dscrptr == "l_cntr":
        altrd_tplgcl_dscrptr_result = altrd_l_cntr_conn_edges
    elif tplgcl_dscrptr in gamma_tplgcl_dscrptr_list:
        altrd_tplgcl_dscrptr_result = tplgcl_dscrptr_func(
            altrd_conn_edges, altrd_conn_edges_type, altrd_l_cntr_conn_edges,
            coords, L)
    else:
        # Create nx.Graph or nx.MultiGraph by adding nodes for the
        # altered network
        if multigraph: altrd_conn_graph = nx.MultiGraph()
        else: altrd_conn_graph = nx.Graph()
        altrd_conn_graph = add_nodes_from_numpy_array(
            altrd_conn_graph, core_nodes)
        
        # If called for, add edges and edge attributes for the altered
        # network
        if tplgcl_dscrptr in l_attr_tplgcl_dscrptr_list:
            altrd_conn_graph = add_edges_and_edge_attributes_from_numpy_arrays(
                altrd_conn_graph, altrd_conn_edges, ["l"],
                l_func(altrd_conn_edges, altrd_conn_edges_type, coords, L))
        elif tplgcl_dscrptr in l_inv_attr_tplgcl_dscrptr_list:
            altrd_conn_graph = add_edges_and_edge_attributes_from_numpy_arrays(
                altrd_conn_graph, altrd_conn_edges, ["l_inv"],
                l_inv_func(altrd_conn_edges, altrd_conn_edges_type, coords, L))
        elif tplgcl_dscrptr in gamma_attr_tplgcl_dscrptr_list:
            altrd_conn_graph = add_edges_and_edge_attributes_from_numpy_arrays(
                altrd_conn_graph, altrd_conn_edges, ["gamma"],
                gamma_func(altrd_conn_edges, altrd_conn_edges_type, altrd_l_cntr_conn_edges, coords, L))
        elif tplgcl_dscrptr in gamma_inv_attr_tplgcl_dscrptr_list:
            altrd_conn_graph = add_edges_and_edge_attributes_from_numpy_arrays(
                altrd_conn_graph, altrd_conn_edges, ["gamma_inv"],
                gamma_inv_func(altrd_conn_edges, altrd_conn_edges_type, altrd_l_cntr_conn_edges, coords, L))
        # Otherwise, add edges for the altered network
        else:
            altrd_conn_graph = add_edges_from_numpy_array(
                altrd_conn_graph, altrd_conn_edges)
        
        # Remove isolate nodes in order to maintain the extracted
        # elastically-effective end-linked network or the extracted
        # largest connected component
        altrd_conn_graph.remove_nodes_from(list(nx.isolates(altrd_conn_graph)))
        
        # Calculate graph-based topological descriptor
        altrd_tplgcl_dscrptr_result = tplgcl_dscrptr_func(altrd_conn_graph)

        # Extract nodes or edges from the altered network
        if tplgcl_dscrptr_type == "nodewise":
            altrd_core_nodes = extract_nodes_to_numpy_array(altrd_conn_graph)
        elif tplgcl_dscrptr_type == "edgewise":
            altrd_conn_edges = extract_edges_to_numpy_array(altrd_conn_graph)
    
    # Properly format and transfer the topological descriptor result
    # with respect to the as-provided network
    if tplgcl_dscrptr_type == "nodewise":
        # Sort the node numbers in ascending order, and correspondingly
        # sort the topological descriptor
        argsort_indcs = np.argsort(altrd_core_nodes)
        altrd_core_nodes = altrd_core_nodes[argsort_indcs]
        altrd_tplgcl_dscrptr_result = altrd_tplgcl_dscrptr_result[argsort_indcs]
        altrd_n = np.shape(altrd_core_nodes)[0]
        altrd_node = 0
        node = 0
        while altrd_node < altrd_n:
            if altrd_core_nodes[altrd_node] == core_nodes[node]:
                tplgcl_dscrptr_result[node] = (
                    altrd_tplgcl_dscrptr_result[altrd_node]
                )
                altrd_node += 1
            node += 1
    elif tplgcl_dscrptr_type == "edgewise":
        # Lexicographically sort the edges and the topological
        # descriptor for the altered network
        altrd_conn_edges, lexsort_indcs = lexsorted_edges(
            altrd_conn_edges, return_indcs=True)
        altrd_tplgcl_dscrptr_result = altrd_tplgcl_dscrptr_result[lexsort_indcs]
        altrd_m = np.shape(altrd_conn_edges)[0]
        altrd_edge = 0
        edge = 0
        while altrd_edge < altrd_m:
            if np.array_equal(altrd_conn_edges[altrd_edge], conn_edges[edge]):
                tplgcl_dscrptr_result[edge] = (
                    altrd_tplgcl_dscrptr_result[altrd_edge]
                )
                altrd_edge += 1
            edge += 1
    
    # Save topological descriptor result (if called for)
    if save_result:
        if dtype_int:
            np.savetxt(result_filename, tplgcl_dscrptr_result, fmt="%d")
        else: np.savetxt(result_filename, tplgcl_dscrptr_result)

    # Return topological descriptor result (if called for)
    if return_result: return tplgcl_dscrptr_result
    else: return None

def network_global_topological_descriptor(
        b: float,
        L: np.ndarray,
        coords: np.ndarray,
        core_nodes: np.ndarray,
        conn_edges: np.ndarray,
        conn_edges_type: np.ndarray,
        l_cntr_conn_edges: np.ndarray,
        multigraph: bool,
        result_filename: str,
        tplgcl_dscrptr: str,
        np_oprtn: str,
        eeel_ntwrk: bool,
        save_result: bool,
        return_result: bool) -> np.ndarray | float | int | None:
    """Global network topological descriptor.
    
    This function calculates the result of a global topological
    descriptor for a supplied network. Several options can be activated
    to this end: the elastically-effective end-linked network from the
    supplied network may be extracted, a numpy function may operate on
    the result of the topological descriptor calculation, and the
    resulting global topological descriptor can be saved and returned.

    Args:
        b (float): Chain segment and/or cross-linker diameter.
        L (np.ndarray): Simulation box side lengths.
        coords (np.ndarray): Coordinates of the core nodes.
        core_nodes (np.ndarray): Core nodes.
        conn_edges (np.ndarray): Edges from the graph capturing the periodic connections between the core nodes.
        conn_edges_type (np.ndarray): Type label for the edges from the graph capturing the periodic connections between the core nodes. Core edges are of type 1, and periodic boundary edges are of type 0.
        l_cntr_conn_edges (np.ndarray): Contour length of the edges from the graph capturing the periodic connections between the core nodes.
        multigraph (bool): Boolean indicating if a nx.Graph (False) or nx.MultiGraph (True) object is needed to represent the supplied network.
        result_filename (str): Filename for the global topological descriptor result.
        tplgcl_dscrptr (str): Topological descriptor name.
        np_oprtn (str): numpy function/operation name.
        eeel_ntwrk (bool): Boolean indicating if the elastically-effective end-linked network ought to be supplied for the global topological descriptor calculation.
        save_result (bool): Boolean indicating if the result ought to be saved.
        return_result (bool): Boolean indicating if the result ought to be returned.
    
    Returns:
        np.ndarray | float | int | None: Global topological descriptor
        result.
    
    """
    # Integer-valuewise topological descriptors
    dtype_int_val_tplgcl_dscrptr_list = ["n", "m", "r", "sigma"]
    # Float-valuewise topological descriptors
    dtype_float_val_tplgcl_dscrptr_list = [
        "prop_ee_n", "prop_eeel_n", "prop_ee_m", "prop_eeel_m", "rho_graph",
        "glbl_avrg_kappa", "lambda_1", "r_pearson", "l_attr_r", "l_inv_attr_r",
        "gamma_attr_r", "gamma_inv_attr_r", "l_attr_sigma", "l_inv_attr_sigma",
        "gamma_attr_sigma", "gamma_inv_attr_sigma",
    ]
    # Arraywise topological descriptors
    arr_tplgcl_dscrptr_list = [
        "l", "l_inv", "l_cmpnts", "l_naive", "l_cntr", "gamma", "gamma_inv",
        "k", "avrg_nn_k", "k_diff", "avrg_k_diff", "c", "kappa",
        "lcl_avrg_kappa", "epsilon", "l_attr_epsilon", "l_inv_attr_epsilon",
        "gamma_attr_epsilon", "gamma_inv_attr_epsilon", "d", "l_attr_d",
        "l_inv_attr_d", "gamma_attr_d", "gamma_inv_attr_d", "avrg_d",
        "avrg_l_attr_d", "avrg_l_inv_attr_d", "avrg_gamma_attr_d",
        "avrg_gamma_inv_attr_d", "e", "l_attr_e", "l_inv_attr_e",
        "gamma_attr_e", "gamma_inv_attr_e", "avrg_e", "avrg_l_attr_e",
        "avrg_l_inv_attr_e", "avrg_gamma_attr_e", "avrg_gamma_inv_attr_e",
        "bc", "l_attr_bc", "l_inv_attr_bc", "gamma_attr_bc",
        "gamma_inv_attr_bc", "ebc", "l_attr_ebc", "l_inv_attr_ebc",
        "gamma_attr_ebc", "gamma_inv_attr_ebc", "cc", "l_attr_cc",
        "l_inv_attr_cc", "gamma_attr_cc", "gamma_inv_attr_cc"
    ]
    # Edge length topological descriptors
    l_tplgcl_dscrptr_list = ["l", "l_inv", "l_cmpnts"]
    # Chain/edge stretch topological descriptors
    gamma_tplgcl_dscrptr_list = ["gamma", "gamma_inv"]
    # Edge length attributed topological descriptors
    l_attr_tplgcl_dscrptr_list = [
        "l_attr_r", "l_attr_sigma", "l_attr_epsilon", "l_attr_d",
        "avrg_l_attr_d", "l_attr_e", "avrg_l_attr_e", "l_attr_bc", "l_attr_ebc",
        "l_attr_cc"
    ]
    # Inverse edge length attributed topological descriptors
    l_inv_attr_tplgcl_dscrptr_list = [
        "l_inv_attr_r", "l_inv_attr_sigma", "l_inv_attr_epsilon",
        "l_inv_attr_d", "avrg_l_inv_attr_d", "l_inv_attr_e",
        "avrg_l_inv_attr_e", "l_inv_attr_bc", "l_inv_attr_ebc", "l_inv_attr_cc"
    ]
    # Edge stretch attributed topological descriptors
    gamma_attr_tplgcl_dscrptr_list = [
        "gamma_attr_r", "gamma_attr_sigma", "gamma_attr_epsilon",
        "gamma_attr_d", "avrg_gamma_attr_d", "gamma_attr_e",
        "avrg_gamma_attr_e", "gamma_attr_bc", "gamma_attr_ebc", "gamma_attr_cc"
    ]
    # Inverse edge stretch attributed topological descriptors
    gamma_inv_attr_tplgcl_dscrptr_list = [
        "gamma_inv_attr_r", "gamma_inv_attr_sigma", "gamma_inv_attr_epsilon",
        "gamma_inv_attr_d", "avrg_gamma_inv_attr_d", "gamma_inv_attr_e",
        "avrg_gamma_inv_attr_e", "gamma_inv_attr_bc", "gamma_inv_attr_ebc",
        "gamma_inv_attr_cc"
    ]
    
    # Exit if the topological descriptor is incompatible with the
    # possible integer-valuewise, float-valuewise, and arraywise
    # topological descriptors
    if (tplgcl_dscrptr not in dtype_int_val_tplgcl_dscrptr_list \
        and tplgcl_dscrptr not in dtype_float_val_tplgcl_dscrptr_list) \
        and tplgcl_dscrptr not in arr_tplgcl_dscrptr_list:
            error_str = (
                "The topological descriptor is incompatible with the "
                + "possible integer-valuewise, float-valuewise, and "
                + "arraywise topological descriptors. Please modify "
                + "the requested topological descriptor accordingly."
            )
            print(error_str)
            return None
    
    # Modify topological descriptor string to match function name
    # convention
    tplgcl_dscrptr_func_str = tplgcl_dscrptr + "_func"

    # Probe each topological descriptors module to identify the
    # topological descriptor calculation function
    if hasattr(descriptors.nodal_degree_topological_descriptors, tplgcl_dscrptr_func_str):
        tplgcl_dscrptr_func = getattr(
            descriptors.nodal_degree_topological_descriptors, tplgcl_dscrptr_func_str)
    elif hasattr(descriptors.shortest_path_topological_descriptors, tplgcl_dscrptr_func_str):
        tplgcl_dscrptr_func = getattr(
            descriptors.shortest_path_topological_descriptors, tplgcl_dscrptr_func_str)
    elif hasattr(descriptors.general_topological_descriptors, tplgcl_dscrptr_func_str):
        tplgcl_dscrptr_func = getattr(
            descriptors.general_topological_descriptors, tplgcl_dscrptr_func_str)
    else:
        error_str = (
            "The topological descriptor ``" + tplgcl_dscrptr + "'' is "
            + "not implemented!"
        )
        print(error_str)
        return None
    
    # For the as-provided network, sort the node numbers in ascending
    # order, and correspondingly sort both the core nodes type and the
    # coordinates
    argsort_indcs = np.argsort(core_nodes)
    core_nodes = core_nodes[argsort_indcs]
    coords = coords[argsort_indcs]

    # For the as-provided network, lexicographically sort the edges,
    # edges type, and edges contour length
    conn_edges, lexsort_indcs = lexsorted_edges(conn_edges, return_indcs=True)
    conn_edges_type = conn_edges_type[lexsort_indcs]
    l_cntr_conn_edges = l_cntr_conn_edges[lexsort_indcs]

    # Create nx.Graph or nx.MultiGraph by adding nodes before edges for
    # the as-provided network
    if multigraph: conn_graph = nx.MultiGraph()
    else: conn_graph = nx.Graph()
    conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
    conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

    # If called for, extract the elastically-effective end-linked
    # network. Otherwise, extract the largest connected component.
    if eeel_ntwrk == True:
        altrd_conn_graph = elastically_effective_end_linked_graph(conn_graph)
    else: altrd_conn_graph = largest_connected_component(conn_graph)
    
    # Extract edges from the altered network
    altrd_conn_edges = extract_edges_to_numpy_array(altrd_conn_graph)
    del altrd_conn_graph

    # Lexicographically sort the edges for the altered network
    altrd_conn_edges = lexsorted_edges(altrd_conn_edges)
    altrd_m = np.shape(altrd_conn_edges)[0]

    # Downselect edges type and edges contour length for the altered
    # network from the as-provided network
    altrd_conn_edges_type = np.empty(altrd_m, dtype=int)
    altrd_l_cntr_conn_edges = np.empty(altrd_m)
    altrd_edge = 0
    edge = 0
    while altrd_edge < altrd_m:
        if np.array_equal(altrd_conn_edges[altrd_edge], conn_edges[edge]):
            altrd_conn_edges_type[altrd_edge] = conn_edges_type[edge]
            altrd_l_cntr_conn_edges[altrd_edge] = l_cntr_conn_edges[edge]
            altrd_edge += 1
        edge += 1
    
    # Calculate topological descriptor
    if tplgcl_dscrptr in l_tplgcl_dscrptr_list:
        tplgcl_dscrptr_result = tplgcl_dscrptr_func(
            altrd_conn_edges, altrd_conn_edges_type, coords, L)
    elif tplgcl_dscrptr == "l_naive":
        tplgcl_dscrptr_result = tplgcl_dscrptr_func(altrd_conn_edges, coords)
    elif tplgcl_dscrptr == "l_cntr":
        tplgcl_dscrptr_result = altrd_l_cntr_conn_edges
    elif tplgcl_dscrptr in gamma_tplgcl_dscrptr_list:
        tplgcl_dscrptr_result = tplgcl_dscrptr_func(
            altrd_conn_edges, altrd_conn_edges_type, altrd_l_cntr_conn_edges,
            coords, L)
    else:
        # Create nx.Graph or nx.MultiGraph by adding nodes for the
        # altered network
        if multigraph: altrd_conn_graph = nx.MultiGraph()
        else: altrd_conn_graph = nx.Graph()
        altrd_conn_graph = add_nodes_from_numpy_array(
            altrd_conn_graph, core_nodes)
        
        # If called for, add edges and edge attributes for the altered
        # network
        if tplgcl_dscrptr in l_attr_tplgcl_dscrptr_list:
            altrd_conn_graph = add_edges_and_edge_attributes_from_numpy_arrays(
                altrd_conn_graph, altrd_conn_edges, ["l"],
                l_func(altrd_conn_edges, altrd_conn_edges_type, coords, L))
        elif tplgcl_dscrptr in l_inv_attr_tplgcl_dscrptr_list:
            altrd_conn_graph = add_edges_and_edge_attributes_from_numpy_arrays(
                altrd_conn_graph, altrd_conn_edges, ["l_inv"],
                l_inv_func(altrd_conn_edges, altrd_conn_edges_type, coords, L))
        elif tplgcl_dscrptr in gamma_attr_tplgcl_dscrptr_list:
            altrd_conn_graph = add_edges_and_edge_attributes_from_numpy_arrays(
                altrd_conn_graph, altrd_conn_edges, ["gamma"],
                gamma_func(altrd_conn_edges, altrd_conn_edges_type, altrd_l_cntr_conn_edges, coords, L))
        elif tplgcl_dscrptr in gamma_inv_attr_tplgcl_dscrptr_list:
            altrd_conn_graph = add_edges_and_edge_attributes_from_numpy_arrays(
                altrd_conn_graph, altrd_conn_edges, ["gamma_inv"],
                gamma_inv_func(altrd_conn_edges, altrd_conn_edges_type, altrd_l_cntr_conn_edges, coords, L))
        # Otherwise, add edges for the altered network
        else:
            altrd_conn_graph = add_edges_from_numpy_array(
                altrd_conn_graph, altrd_conn_edges)
        
        # Remove isolate nodes in order to maintain the extracted
        # elastically-effective end-linked network or the extracted
        # largest connected component
        altrd_conn_graph.remove_nodes_from(list(nx.isolates(altrd_conn_graph)))
        
        # Calculate graph-based topological descriptor
        tplgcl_dscrptr_result = tplgcl_dscrptr_func(altrd_conn_graph)
    
    # Probe the numpy module to identify the numpy function to send the
    # topological descriptor result to (if called for)
    if tplgcl_dscrptr in arr_tplgcl_dscrptr_list and np_oprtn == "":
        np_oprtn = "mean"
    if np_oprtn == "": pass
    elif hasattr(np, np_oprtn):
        # Deploy numpy function and carefully handle the input parameter
        # set in the process
        np_func = getattr(np, np_oprtn)
        if tplgcl_dscrptr == "l_cmpnts":
            tplgcl_dscrptr_result = np_func(tplgcl_dscrptr_result, axis=0)
        else: tplgcl_dscrptr_result = np_func(tplgcl_dscrptr_result)
    else:
        error_str = "The numpy function ``" + np_oprtn + "'' does not exist!"
        print(error_str)
        return None
    
    # Save topological descriptor result (if called for)
    if save_result:
        if tplgcl_dscrptr in dtype_int_val_tplgcl_dscrptr_list and np_oprtn == "":
            np.savetxt(
                result_filename, np.asarray([tplgcl_dscrptr_result], dtype=int),
                fmt="%d")
        else: np.savetxt(result_filename, np.asarray([tplgcl_dscrptr_result]))

    # Return topological descriptor result (if called for)
    if return_result: return tplgcl_dscrptr_result
    else: return None

def network_global_morphological_descriptor(
        b: float,
        L: np.ndarray,
        coords: np.ndarray,
        core_nodes: np.ndarray,
        conn_edges: np.ndarray,
        conn_edges_type: np.ndarray,
        l_cntr_conn_edges: np.ndarray,
        result_filename: str,
        mrphlgcl_dscrptr: str,
        save_result: bool,
        return_result: bool) -> np.ndarray | float | int | None:
    """Global network morphological descriptor.
    
    This function calculates the result of a global morphological
    descriptor for a supplied network. The resulting global
    morphological descriptor can be saved and returned.

    Args:
        b (float): Chain segment and/or cross-linker diameter.
        L (np.ndarray): Simulation box side lengths.
        coords (np.ndarray): Coordinates of the core nodes.
        core_nodes (np.ndarray): Core nodes.
        conn_edges (np.ndarray): Edges from the graph capturing the periodic connections between the core nodes.
        conn_edges_type (np.ndarray): Type label for the edges from the graph capturing the periodic connections between the core nodes. Core edges are of type 1, and periodic boundary edges are of type 0.
        l_cntr_conn_edges (np.ndarray): Contour length of the edges from the graph capturing the periodic connections between the core nodes.
        result_filename (str): Filename for the global morphological descriptor result.
        mrphlgcl_dscrptr (str): Morphological descriptor name.
        save_result (bool): Boolean indicating if the result ought to be saved.
        return_result (bool): Boolean indicating if the result ought to be returned.
    
    Returns:
        np.ndarray | float | int | None: Global morphological descriptor
        result.
    
    """
    # Morphological descriptors
    mrphlgcl_dscrptr_list = ["xi_corr", "n_fractal_dim"]
    
    # Exit if the morphological descriptor is incompatible with the
    # possible morphological descriptors
    if mrphlgcl_dscrptr not in mrphlgcl_dscrptr_list:
            error_str = (
                "The morphological descriptor is incompatible with the "
                + "possible morphological descriptors. Please modify "
                + "the requested morphological descriptor accordingly."
            )
            print(error_str)
            return None
    
    # Modify morphological descriptor string to match function name
    # convention
    mrphlgcl_dscrptr_func_str = mrphlgcl_dscrptr + "_func"

    # Probe the morphological descriptors module to identify the
    # morphological descriptor calculation function
    if hasattr(descriptors.morphological_descriptors, mrphlgcl_dscrptr_func_str):
        mrphlgcl_dscrptr_func = getattr(
            descriptors.morphological_descriptors, mrphlgcl_dscrptr_func_str)
    else:
        error_str = (
            "The morphological descriptor ``" + mrphlgcl_dscrptr + "'' "
            + "is not implemented!"
        )
        print(error_str)
        return None
    
    # For the as-provided network, sort the node numbers in ascending
    # order, and correspondingly sort both the core nodes type and the
    # coordinates
    argsort_indcs = np.argsort(core_nodes)
    core_nodes = core_nodes[argsort_indcs]
    coords = coords[argsort_indcs]

    # For the as-provided network, lexicographically sort the edges,
    # edges type, and edges contour length
    conn_edges, lexsort_indcs = lexsorted_edges(conn_edges, return_indcs=True)
    conn_edges_type = conn_edges_type[lexsort_indcs]
    l_cntr_conn_edges = l_cntr_conn_edges[lexsort_indcs]

    # Add nodes before edges for the as-provided network to a
    # nx.MultiGraph object, and then convert back down to a nx.Graph
    # object
    conn_graph = nx.MultiGraph()
    conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
    conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)
    conn_graph = nx.Graph(conn_graph)

    # Extract the largest connected component.
    altrd_conn_graph = largest_connected_component(conn_graph)

    # Extract nodes from the altered network
    altrd_core_nodes = extract_nodes_to_numpy_array(altrd_conn_graph)
    del altrd_conn_graph

    # Extract nodal coordinates and recalibrate node numbers
    coords = coords[np.sort(altrd_core_nodes)]
    core_nodes = np.arange(np.shape(coords)[0], dtype=int)
    
    # Calculate morphological descriptor
    mrphlgcl_dscrptr_result = mrphlgcl_dscrptr_func(b, L, coords, core_nodes)
    
    # Save morphological descriptor result (if called for)
    if save_result:
        np.savetxt(result_filename, np.asarray([mrphlgcl_dscrptr_result]))
    
    # Return topological descriptor result (if called for)
    if return_result: return mrphlgcl_dscrptr_result
    else: return None