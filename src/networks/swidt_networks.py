import numpy as np
import networkx as nx
from src.file_io.file_io import (
    L_filename_str,
    config_filename_str,
    config_pruning_filename_str
)
from src.helpers.simulation_box_utils import L_arg_eta_func
from src.networks.delaunay_networks import (
    delaunay_network_topology_initialization
)
from src.descriptors.nodal_degree_topological_descriptors import k_func
from src.helpers.graph_utils import (
    lexsorted_edges,
    add_nodes_and_node_attributes_from_numpy_arrays,
    add_edges_and_edge_attributes_from_numpy_arrays,
    extract_nodes_to_numpy_array,
    extract_edges_to_numpy_array,
    extract_node_attribute_to_numpy_array,
    extract_edge_attribute_to_numpy_array,
    largest_connected_component
)
from src.helpers.network_topology_initialization_utils import (
    core_node_values_update_in_edges_func
)
from src.descriptors.network_descriptors import (
    network_local_topological_descriptor,
    network_global_topological_descriptor,
    network_global_morphological_descriptor
)

def swidt_filename_str(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int,
        pruning: int) -> str:
    """Filename prefix associated with spider web-inspired
    Delaunay-triangulated network data files.

    This function returns the filename prefix associated with spider
    web-inspired Delaunay-triangulated network data files.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "swidt" is applicable (corresponding to spider web-inspired Delaunay-triangulated networks ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
        pruning (int): Edge pruning procedure number.
    
    Returns:
        str: The filename prefix associated with spider web-inspired
        Delaunay-triangulated network data files.
    
    """
    # This filename prefix convention is only applicable for data files
    # associated with spider web-inspired Delaunay-triangulated
    # networks. Exit if a different type of network is passed.
    if network != "swidt":
        error_str = (
            "This filename prefix convention is only applicable for "
            + "data files associated with spider web-inspired "
            + "Delaunay-triangulated networks. This filename prefix "
            + "will only be supplied if network = ``swidt''."
        )
        raise ValueError(error_str)
    return config_pruning_filename_str(
        network, date, batch, sample, config, pruning)

def swidt_L(
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        b: float,
        n: int,
        eta_n: float) -> None:
    """Simulation box side lengths for spider web-inspired
    Delaunay-triangulated networks.

    This function calculates the simulation box side lengths for spider
    web-inspired Delaunay-triangulated networks.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "swidt" is applicable (corresponding to spider web-inspired Delaunay-triangulated networks ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Node diameter.
        n (int): Intended number of core nodes.
        eta_n (float): Node packing density.
    
    """
    # This calculation for L is only applicable for spider web-inspired
    # Delaunay-triangulated networks. Exit if a different type of
    # network is passed.
    if network != "swidt":
        error_str = (
            "This calculation for L is only applicable for spider "
            + "web-inspired Delaunay-triangulated networks. This "
            + "calculation will only proceed if network = ``swidt''."
        )
        raise ValueError(error_str)
    
    # Calculate and save L
    np.savetxt(
        L_filename_str(network, date, batch, sample),
        L_arg_eta_func(dim, b, n, eta_n))

def swidt_network_topology(
        network: str,
        date: str,
        batch: str,
        sample: int,
        scheme: str,
        dim: int,
        n: int,
        config: int) -> None:
    """Spider web-inspired Delaunay-triangulated network topology.

    This function confirms that the network being called for is a spider
    web-inspired Delaunay-triangulated network. Then, the function calls
    the Delaunay-triangulated network initialization function to create
    the initial Delaunay-triangulated network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "swidt" is applicable (corresponding to spider web-inspired Delaunay-triangulated networks ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular scheme used to generate the positions of the core nodes; either "random", "prhd", "pdhu", or "lammps" (corresponding to the random node placement procedure ("random"), periodic random hard disk node placement procedure ("prhd"), periodic disordered hyperuniform node placement procedure ("pdhu"), or nodes randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        n (int): Number of core nodes.
        config (int): Configuration number.
    
    """
    # Network topology initialization procedure is only applicable for 
    # spider web-inspired Delaunay-triangulated networks. Exit if a
    # different type of network is passed.
    if network != "swidt":
        error_str = (
            "Network topology initialization procedure is only "
            + "applicable for spider web-inspired "
            + "Delaunay-triangulated networks. This procedure will "
            + "only proceed if network = ``swidt''."
        )
        raise ValueError(error_str)
    delaunay_network_topology_initialization(
        network, date, batch, sample, scheme, dim, n, config)

def swidt_network_edge_pruning_procedure(
        network: str,
        date: str,
        batch: str,
        sample: int,
        n: int,
        k: int,
        config: int,
        pruning: int) -> None:
    """Edge pruning procedure for the initialized topology of spider
    web-inspired Delaunay-triangulated networks.

    This function loads fundamental graph constituents along with core
    node coordinates, performs a random edge pruning procedure such that
    each node in the network is connected to, at most, k edges, and
    isolates the maximum connected component from the resulting network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "swidt" is applicable (corresponding to spider web-inspired Delaunay-triangulated networks ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        n (int): Number of core nodes.
        k (int): Maximum node degree/functionality; either 3, 4, 5, 6, 7, or 8.
        config (int): Configuration number.
        pruning (int): Edge pruning procedure number.
    
    """
    # Edge pruning procedure is only applicable for the initialized
    # topology of spider web-inspired Delaunay-triangulated networks.
    # Exit if a different type of network is passed.
    if network != "swidt":
        error_str = (
            "Edge pruning procedure is only applicable for the "
            + "initialized topology of spider web-inspired "
            + "Delaunay-triangulated networks. This procedure will "
            + "only proceed if network = ``swidt''."
        )
        raise ValueError(error_str)
    
    # Initialize random number generator
    rng = np.random.default_rng()

    # Initialize node number integer constants
    core_node_0 = 0
    core_node_1 = 0

    # Generate configuration filename prefix
    config_filename_prefix = config_filename_str(
        network, date, batch, sample, config)
    coords_filename = config_filename_prefix + ".coords"
    conn_edges_filename = config_filename_prefix + "-conn_edges" + ".dat"
    conn_edges_type_filename = (
        config_filename_prefix + "-conn_edges_type" + ".dat"
    )

    # Generate configuration and pruning filename prefix. This
    # establishes the configuration and pruning filename prefix as the
    # filename prefix associated with spider web-inspired
    # Delaunay-triangulated network data files, which is reflected in
    # the swidt_filename_str() function.
    config_pruning_filename_prefix = config_pruning_filename_str(
        network, date, batch, sample, config, pruning)
    mx_cmp_pruned_coords_filename = config_pruning_filename_prefix + ".coords"
    mx_cmp_pruned_core_nodes_type_filename = (
        config_pruning_filename_prefix + "-core_nodes_type" + ".dat"
    )
    mx_cmp_pruned_conn_edges_filename = (
        config_pruning_filename_prefix + "-conn_edges" + ".dat"
    )
    mx_cmp_pruned_conn_edges_type_filename = (
        config_pruning_filename_prefix + "-conn_edges_type" + ".dat"
    )
    
    # Load fundamental graph constituents
    core_nodes = np.arange(n, dtype=int)
    core_nodes_type = np.ones(n, dtype=int)
    conn_edges = np.loadtxt(conn_edges_filename, dtype=int)
    conn_edges_type = np.loadtxt(conn_edges_type_filename, dtype=int)

    # Load core node coordinates
    coords = np.loadtxt(coords_filename)
    
    # Create nx.Graph by adding nodes and node attributes before edges
    # and edge attributes
    conn_graph = nx.Graph()
    conn_graph = add_nodes_and_node_attributes_from_numpy_arrays(
        conn_graph, core_nodes, ["type"], core_nodes_type)
    conn_graph = add_edges_and_edge_attributes_from_numpy_arrays(
        conn_graph, conn_edges, ["type"], conn_edges_type)
    
    # Degree of nodes in the graph
    conn_graph_k = k_func(conn_graph)

    if np.any(conn_graph_k > k):
        # Explicit edge pruning procedure
        while np.any(conn_graph_k > k):
            # Identify the nodes connected to more than k edges in the
            # graph, i.e., hyperconnected nodes
            conn_graph_hyprconn_nodes = np.where(conn_graph_k > k)[0]
            # Identify the edges connected to the hyperconnected nodes
            conn_graph_hyprconn_edges = np.logical_or(
                np.isin(conn_edges[:, 0], conn_graph_hyprconn_nodes),
                np.isin(conn_edges[:, 1], conn_graph_hyprconn_nodes))
            conn_graph_hyprconn_edge_indcs = (
                np.where(conn_graph_hyprconn_edges)[0]
            )
            # Randomly select a hyperconnected edge to remove
            edge_indcs_indx_to_remove_indx = (
                rng.integers(
                    np.shape(conn_graph_hyprconn_edge_indcs)[0], dtype=int)
            )
            edge_indx_to_remove = (
                conn_graph_hyprconn_edge_indcs[edge_indcs_indx_to_remove_indx]
            )
            core_node_0 = int(conn_edges[edge_indx_to_remove, 0])
            core_node_1 = int(conn_edges[edge_indx_to_remove, 1])

            # Remove hyperconnected edge in the graphs
            if conn_graph.has_edge(core_node_0, core_node_1):
                conn_graph.remove_edge(core_node_0, core_node_1)
                conn_edges = np.delete(conn_edges, edge_indx_to_remove, axis=0)
                
                # Update degree of nodes in the graph
                conn_graph_k[core_node_0] -= 1
                conn_graph_k[core_node_1] -= 1
                
        # Isolate largest/maximum connected component, and extract nodes
        # and edges
        mx_cmp_pruned_conn_graph = largest_connected_component(conn_graph)
        mx_cmp_pruned_core_nodes = extract_nodes_to_numpy_array(
            mx_cmp_pruned_conn_graph)
        mx_cmp_pruned_conn_edges = extract_edges_to_numpy_array(
            mx_cmp_pruned_conn_graph)
        mx_cmp_pruned_conn_n = np.shape(mx_cmp_pruned_core_nodes)[0]
        
        # Degree of nodes in the largest/maximum connected component
        mx_cmp_pruned_conn_graph_k = k_func(mx_cmp_pruned_conn_graph)

        # Determine the type of each core node for the largest/maximum
        # connected component based upon the degree of each node.
        # Dangling nodes are of type 2, all other nodes are already of
        # type 1.
        for indx in range(mx_cmp_pruned_conn_n):
            core_node = mx_cmp_pruned_core_nodes[indx]
            core_node_k = mx_cmp_pruned_conn_graph_k[indx]

            if core_node_k == 1:
                mx_cmp_pruned_conn_graph.nodes[core_node]["type"] = 2
        
        # Extract core nodes type
        mx_cmp_pruned_core_nodes_type = extract_node_attribute_to_numpy_array(
            mx_cmp_pruned_conn_graph, "type", dtype_int=True)
        
        # Sort the node numbers in ascending order, and correspondingly
        # sort the core nodes type
        argsort_indcs = np.argsort(mx_cmp_pruned_core_nodes)
        mx_cmp_pruned_core_nodes = mx_cmp_pruned_core_nodes[argsort_indcs]
        mx_cmp_pruned_core_nodes_type = (
            mx_cmp_pruned_core_nodes_type[argsort_indcs]
        )

        # Isolate the core node coordinates for the largest/maximum
        # connected component
        mx_cmp_pruned_coords = coords[mx_cmp_pruned_core_nodes]
        
        # Update each node value in each edge with an updated node value
        # corresponding to an ascending array of core nodes
        mx_cmp_pruned_conn_edges = core_node_values_update_in_edges_func(
            mx_cmp_pruned_core_nodes, mx_cmp_pruned_conn_edges
        )
        
        # Extract edges type for the largest/maximum connected component
        mx_cmp_pruned_conn_edges_type = extract_edge_attribute_to_numpy_array(
            mx_cmp_pruned_conn_graph, "type", dtype_int=True)

        # Lexicographically sort the edges and edge types
        mx_cmp_pruned_conn_edges, lexsort_indcs = lexsorted_edges(
            mx_cmp_pruned_conn_edges, return_indcs=True)
        mx_cmp_pruned_conn_edges_type = (
            mx_cmp_pruned_conn_edges_type[lexsort_indcs]
        )

        # Save fundamental graph constituents from this topology
        np.savetxt(
            mx_cmp_pruned_core_nodes_type_filename, mx_cmp_pruned_core_nodes_type,
            fmt="%d")
        np.savetxt(
            mx_cmp_pruned_conn_edges_filename, mx_cmp_pruned_conn_edges,
            fmt="%d")
        np.savetxt(
            mx_cmp_pruned_conn_edges_type_filename, mx_cmp_pruned_conn_edges_type,
            fmt="%d")
        
        # Save the core node coordinates
        np.savetxt(mx_cmp_pruned_coords_filename, mx_cmp_pruned_coords)
    else:
        # Save fundamental graph constituents from this topology
        np.savetxt(
            mx_cmp_pruned_core_nodes_type_filename, core_nodes_type, fmt="%d")
        np.savetxt(mx_cmp_pruned_conn_edges_filename, conn_edges, fmt="%d")
        np.savetxt(
            mx_cmp_pruned_conn_edges_type_filename, conn_edges_type, fmt="%d")
        
        # Save the core node coordinates
        np.savetxt(mx_cmp_pruned_coords_filename, coords)

def swidt_network_local_topological_descriptor(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int,
        pruning: int,
        b: float,
        tplgcl_dscrptr: str,
        eeel_ntwrk: bool,
        save_result: bool,
        return_result: bool) -> np.ndarray | float | int | None:
    """Spider web-inspired Delaunay-triangulated network local
    topological descriptor.
    
    This function extracts a spider web-inspired Delaunay-triangulated
    network and sets a variety of input parameters corresponding to a
    particular local topological descriptor of interest. These are then
    passed to the master network_local_topological_descriptor()
    function, which calculates (and, if called for, saves) the result of
    the local topological descriptor for the spider web-inspired
    Delaunay-triangulated network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "swidt" is applicable (corresponding to spider-web inspired Delaunay-triangulated networks ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
        pruning (int): Edge pruning procedure number.
        b (float): Chain segment and/or cross-linker diameter.
        tplgcl_dscrptr (str): Topological descriptor name.
        eeel_ntwrk (bool): Boolean indicating if the elastically-effective end-linked network ought to be supplied for the local topological descriptor calculation.
        save_result (bool): Boolean indicating if the result ought to be saved.
        return_result (bool): Boolean indicating if the result ought to be returned.
    
    Returns:
        np.ndarray | float | int | None: Local topological descriptor
        result.
    
    """
    # This local topological descriptor calculation is only applicable
    # for data files associated with spider web-inspired
    # Delaunay-triangulated networks. Exit if a different type of
    # network is passed.
    if network != "swidt":
        error_str = (
            "This local topological descriptor calculation is only "
            + "applicable for data files associated with spider "
            + "web-inspired Delaunay-triangulated networks. This "
            + "calculation will proceed only if network = ``swidt''."
        )
        raise ValueError(error_str)
    
    # A spider web-inspired Delaunay-triangulated network is represented
    # via an nx.Graph object
    multigraph = False

    # Generate filenames
    L_filename = L_filename_str(network, date, batch, sample)
    swidt_filename = swidt_filename_str(
        network, date, batch, sample, config, pruning)
    coords_filename = swidt_filename + ".coords"
    conn_edges_filename = swidt_filename + "-conn_edges" + ".dat"
    conn_edges_type_filename = swidt_filename + "-conn_edges_type" + ".dat"
    
    if eeel_ntwrk:
        result_filename = swidt_filename + "-eeel-lcl-" + tplgcl_dscrptr + ".dat"
    else:
        result_filename = swidt_filename + "-lcl-" + tplgcl_dscrptr + ".dat"

    # Load simulation box side lengths and node coordinates
    L = np.loadtxt(L_filename)
    coords = np.loadtxt(coords_filename)

    # Load fundamental graph constituents
    core_nodes = np.arange(np.shape(coords)[0], dtype=int)
    conn_edges = np.loadtxt(conn_edges_filename, dtype=int)
    conn_edges_type = np.loadtxt(conn_edges_type_filename, dtype=int)
    l_cntr_conn_edges = np.ones(np.shape(conn_edges)[0], dtype=int)
    
    # Call the master network_local_topological_descriptor() function
    return network_local_topological_descriptor(
        b, L, coords, core_nodes, conn_edges, conn_edges_type,
        l_cntr_conn_edges, multigraph, result_filename, tplgcl_dscrptr,
        eeel_ntwrk, save_result, return_result)

def swidt_network_global_topological_descriptor(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int,
        pruning: int,
        b: float,
        tplgcl_dscrptr: str,
        np_oprtn: str,
        eeel_ntwrk: bool,
        save_result: bool,
        return_result: bool) -> np.ndarray | float | int | None:
    """Spider web-inspired Delaunay-triangulated network global
    topological descriptor.
    
    This function extracts a spider web-inspired Delaunay-triangulated
    network and sets a variety of input parameters corresponding to a
    particular global topological descriptor (and numpy function) of
    interest. These are then passed to the master
    network_global_topological_descriptor() function, which calculates
    (and, if called for, saves) the result of the global topological
    descriptor for the spider web-inspired Delaunay-triangulated
    network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "swidt" is applicable (corresponding to spider-web inspired Delaunay-triangulated networks ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
        pruning (int): Edge pruning procedure number.
        b (float): Chain segment and/or cross-linker diameter.
        tplgcl_dscrptr (str): Topological descriptor name.
        np_oprtn (str): numpy function/operation name.
        eeel_ntwrk (bool): Boolean indicating if the elastically-effective end-linked network ought to be supplied for the global topological descriptor calculation.
        save_result (bool): Boolean indicating if the result ought to be saved.
        return_result (bool): Boolean indicating if the result ought to be returned.
    
    Returns:
        np.ndarray | float | int | None: Global topological descriptor
        result.
    
    """
    # This global topological descriptor calculation is only applicable
    # for data files associated with spider web-inspired
    # Delaunay-triangulated networks. Exit if a different type of
    # network is passed.
    if network != "swidt":
        error_str = (
            "This global topological descriptor calculation is only "
            + "applicable for data files associated with spider "
            + "web-inspired Delaunay-triangulated networks. This "
            + "calculation will proceed only if network = ``swidt''."
        )
        raise ValueError(error_str)
    
    # A spider web-inspired Delaunay-triangulated network is represented
    # via an nx.Graph object
    multigraph = False

    # Generate filenames
    L_filename = L_filename_str(network, date, batch, sample)
    swidt_filename = swidt_filename_str(
        network, date, batch, sample, config, pruning)
    coords_filename = swidt_filename + ".coords"
    conn_edges_filename = swidt_filename + "-conn_edges" + ".dat"
    conn_edges_type_filename = swidt_filename + "-conn_edges_type" + ".dat"
    
    if eeel_ntwrk:
        if np_oprtn == "":
            result_filename = (
                swidt_filename + "-eeel-glbl-" + tplgcl_dscrptr + ".dat"
            )
        else:
            result_filename = (
                swidt_filename + "-eeel-glbl-" + np_oprtn + "-" + tplgcl_dscrptr
                + ".dat"
            )
    else:
        if np_oprtn == "":
            result_filename = (
                swidt_filename + "-glbl-" + tplgcl_dscrptr + ".dat"
            )
        else:
            result_filename = (
                swidt_filename + "-glbl-" + np_oprtn + "-" + tplgcl_dscrptr
                + ".dat"
            )

    # Load simulation box side lengths and node coordinates
    L = np.loadtxt(L_filename)
    coords = np.loadtxt(coords_filename)

    # Load fundamental graph constituents
    core_nodes = np.arange(np.shape(coords)[0], dtype=int)
    conn_edges = np.loadtxt(conn_edges_filename, dtype=int)
    conn_edges_type = np.loadtxt(conn_edges_type_filename, dtype=int)
    l_cntr_conn_edges = np.ones(np.shape(conn_edges)[0], dtype=int)
    
    # Call the master network_global_topological_descriptor() function
    return network_global_topological_descriptor(
        b, L, coords, core_nodes, conn_edges, conn_edges_type,
        l_cntr_conn_edges, multigraph, result_filename, tplgcl_dscrptr,
        np_oprtn, eeel_ntwrk, save_result, return_result)

def swidt_network_global_morphological_descriptor(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int,
        pruning: int,
        b: float,
        mrphlgcl_dscrptr: str,
        save_result: bool,
        return_result: bool) -> np.ndarray | float | int | None:
    """Spider web-inspired Delaunay-triangulated network global
    morphological descriptor.
    
    This function extracts a spider web-inspired Delaunay-triangulated
    network and sets a variety of input parameters corresponding to a
    particular global morphological descriptor of interest. These are
    then passed to the master network_global_morphological_descriptor()
    function, which calculates (and, if called for, saves) the result of
    the global morphological descriptor for the spider web-inspired
    Delaunay-triangulated network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "swidt" is applicable (corresponding to spider-web inspired Delaunay-triangulated networks ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
        pruning (int): Edge pruning procedure number.
        b (float): Chain segment and/or cross-linker diameter.
        mrphlgcl_dscrptr (str): Morphological descriptor name.
        save_result (bool): Boolean indicating if the result ought to be saved.
        return_result (bool): Boolean indicating if the result ought to be returned.
    
    Returns:
        np.ndarray | float | int | None: Global morphological descriptor
        result.
    
    """
    # This global morphological descriptor calculation is only
    # applicable for data files associated with spider web-inspired
    # Delaunay-triangulated networks. Exit if a different type of
    # network is passed.
    if network != "swidt":
        error_str = (
            "This global morphological descriptor calculation is only "
            + "applicable for data files associated with spider "
            + "web-inspired Delaunay-triangulated networks. This "
            + "calculation will proceed only if network = ``swidt''."
        )
        raise ValueError(error_str)
    
    # Generate filenames
    L_filename = L_filename_str(network, date, batch, sample)
    swidt_filename = swidt_filename_str(
        network, date, batch, sample, config, pruning)
    coords_filename = swidt_filename + ".coords"
    conn_edges_filename = swidt_filename + "-conn_edges" + ".dat"
    conn_edges_type_filename = swidt_filename + "-conn_edges_type" + ".dat"
    result_filename = swidt_filename + "-glbl-" + mrphlgcl_dscrptr + ".dat"

    # Load simulation box side lengths and node coordinates
    L = np.loadtxt(L_filename)
    coords = np.loadtxt(coords_filename)

    # Load fundamental graph constituents
    core_nodes = np.arange(np.shape(coords)[0], dtype=int)
    conn_edges = np.loadtxt(conn_edges_filename, dtype=int)
    conn_edges_type = np.loadtxt(conn_edges_type_filename, dtype=int)
    l_cntr_conn_edges = np.ones(np.shape(conn_edges)[0], dtype=int)
    
    # Call the master network_global_morphological_descriptor() function
    return network_global_morphological_descriptor(
        b, L, coords, core_nodes, conn_edges, conn_edges_type,
        l_cntr_conn_edges, result_filename, mrphlgcl_dscrptr, save_result,
        return_result)