import numpy as np
from scipy.spatial import Delaunay
from src.file_io.file_io import (
    L_filename_str,
    config_filename_str
)
from src.helpers.simulation_box_utils import L_arg_eta_func
from src.helpers.network_topology_initialization_utils import (
    core_node_tessellation
)
from src.helpers.graph_utils import (
    unique_lexsorted_edges,
    lexsorted_edges
)
from src.descriptors.network_descriptors import (
    network_local_topological_descriptor,
    network_global_topological_descriptor,
    network_global_morphological_descriptor
)

def delaunay_filename_str(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int) -> str:
    """Filename prefix associated with Delaunay-triangulated network
    data files.

    This function returns the filename prefix associated with
    Delaunay-triangulated network data files.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "delaunay" is applicable (corresponding to Delaunay-triangulated networks ("delaunay")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
    
    Returns:
        str: The filename prefix associated with Delaunay-triangulated
        network data files.
    
    """
    # This filename prefix convention is only applicable for data files
    # associated with Delaunay-triangulated networks. Exit if a
    # different type of network is passed.
    if network != "delaunay":
        error_str = (
            "This filename prefix convention is only applicable for "
            + "data files associated with Delaunay-triangulated "
            + "networks. This filename prefix will only be supplied if "
            + "network = ``delaunay''."
        )
        raise ValueError(error_str)
    return config_filename_str(network, date, batch, sample, config)

def delaunay_L(
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        b: float,
        n: int,
        eta_n: float) -> None:
    """Simulation box side lengths for Delaunay-triangulated networks.

    This function calculates and saves the simulation box side lengths
    for Delaunay-triangulated networks.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "delaunay" is applicable (corresponding to Delaunay-triangulated networks ("delaunay")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Node diameter.
        n (int): Intended number of core nodes.
        eta_n (float): Node packing density.
    
    """
    # This calculation for L is only applicable for
    # Delaunay-triangulated networks. Exit if a different type of
    # network is passed.
    if network != "delaunay":
        error_str = (
            "This calculation for L is only applicable for "
            + "Delaunay-triangulated networks. This calculation will "
            + "only proceed if network = ``delaunay''."
        )
        raise ValueError(error_str)
    
    # Calculate and save L
    np.savetxt(
        L_filename_str(network, date, batch, sample),
        L_arg_eta_func(dim, b, n, eta_n))

def delaunay_network_topology_initialization(
        network: str,
        date: str,
        batch: str,
        sample: int,
        scheme: str,
        dim: int,
        n: int,
        config: int) -> None:
    """Network topology initialization procedure for
    Delaunay-triangulated networks.

    This function loads the simulation box side lengths and the core
    node coordinates. Then, this function ``tessellates'' the core nodes
    about themselves, applies Delaunay triangulation to the resulting
    tessellated network via the scipy.spatial.Delaunay() function,
    acquires back the periodic network topology of the core nodes, and
    ascertains fundamental graph constituents (node and edge
    information) from this topology.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "delaunay" or "swidt" are applicable (corresponding to Delaunay-triangulated networks ("delaunay") and spider web-inspired Delaunay-triangulated networks ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular scheme used to generate the positions of the core nodes; either "random", "prhd", "pdhu", or "lammps" (corresponding to the random node placement procedure ("random"), periodic random hard disk node placement procedure ("prhd"), periodic disordered hyperuniform node placement procedure ("pdhu"), or nodes randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        n (int): Number of core nodes.
        config (int): Configuration number.
    
    """
    # Load simulation box side lengths
    L = np.loadtxt(L_filename_str(network, date, batch, sample))

    # Generate configuration filename prefix. This establishes the
    # configuration filename prefix as the filename prefix associated
    # with Delaunay-triangulated network data files, which is reflected
    # in the delaunay_filename_str() function.
    config_filename_prefix = config_filename_str(
        network, date, batch, sample, config)

    # Generate filenames
    coords_filename = config_filename_prefix + ".coords"
    conn_edges_filename = config_filename_prefix + "-conn_edges" + ".dat"
    conn_edges_type_filename = (
        config_filename_prefix + "-conn_edges_type" + ".dat"
    )

    # Call appropriate helper function to initialize network topology
    if (scheme == "random") or (scheme == "prhd") or (scheme == "pdhu"):
        # Load core node coordinates
        coords = np.loadtxt(coords_filename)
    elif scheme == "lammps":
        skiprows_num = 15
        # Load core node coordinates
        coords = np.loadtxt(coords_filename, skiprows=skiprows_num, max_rows=n)
    
    # Actual number of core nodes
    n = np.shape(coords)[0]

    # Core nodes
    core_nodes = np.arange(n, dtype=int)

    # Tessellate the core node coordinates and construct the
    # pb_to_core_nodes np.ndarray
    tsslltd_coords, pb_to_core_nodes = core_node_tessellation(
        dim, core_nodes, coords, L)
    
    del core_nodes

    # Shift the coordinate origin to the center of the simulation box
    # for improved Delaunay triangulation performance
    tsslltd_coords -= 0.5 * L

    # Apply Delaunay triangulation
    tsslltd_core_delaunay = Delaunay(tsslltd_coords)

    del tsslltd_coords

    # Extract the simplices from the Delaunay triangulation
    simplices = tsslltd_core_delaunay.simplices

    # List for edges of the core and periodic boundary nodes
    tsslltd_core_pb_edges = []

    for simplex in simplices:
        # In two dimensions, each simplex is a triangle
        if dim == 2:
            node_0 = int(simplex[0])
            node_1 = int(simplex[1])
            node_2 = int(simplex[2])

            # If any of the nodes involved in any simplex edge
            # correspond to the original core nodes, then add that edge
            # to the edge list. Duplicate entries will arise.
            if (node_0 < n) or (node_1 < n):
                tsslltd_core_pb_edges.append((node_0, node_1))
            if (node_1 < n) or (node_2 < n):
                tsslltd_core_pb_edges.append((node_1, node_2))
            if (node_2 < n) or (node_0 < n):
                tsslltd_core_pb_edges.append((node_2, node_0))
            else: pass
        # In three dimensions, each simplex is a tetrahedron
        elif dim == 3:
            node_0 = int(simplex[0])
            node_1 = int(simplex[1])
            node_2 = int(simplex[2])
            node_3 = int(simplex[3])

            # If any of the nodes involved in any simplex edge
            # correspond to the original core nodes, then add those
            # nodes and that edge to the appropriate lists. Duplicate
            # entries will arise.
            if (node_0 < n) or (node_1 < n):
                tsslltd_core_pb_edges.append((node_0, node_1))
            if (node_1 < n) or (node_2 < n):
                tsslltd_core_pb_edges.append((node_1, node_2))
            if (node_2 < n) or (node_0 < n):
                tsslltd_core_pb_edges.append((node_2, node_0))
            if (node_3 < n) or (node_0 < n):
                tsslltd_core_pb_edges.append((node_3, node_0))
            if (node_3 < n) or (node_1 < n):
                tsslltd_core_pb_edges.append((node_3, node_1))
            if (node_3 < n) or (node_2 < n):
                tsslltd_core_pb_edges.append((node_3, node_2))
            else: pass
    
    del simplex, simplices, tsslltd_core_delaunay

    # Convert edge list to np.ndarray, and retain the unique
    # lexicographically sorted edges from the core and periodic boundary
    # nodes
    tsslltd_core_pb_edges = unique_lexsorted_edges(tsslltd_core_pb_edges)

    # Lists for the edges of the graph capturing the periodic
    # connections between the core nodes
    conn_core_edges = []
    conn_pb_edges = []

    for edge in range(np.shape(tsslltd_core_pb_edges)[0]):
        node_0 = int(tsslltd_core_pb_edges[edge, 0])
        node_1 = int(tsslltd_core_pb_edges[edge, 1])

        # Edge is a core edge
        if (node_0 < n) and (node_1 < n):
            conn_core_edges.append((node_0, node_1))
        # Edge is a periodic boundary edge
        else:
            node_0 = int(pb_to_core_nodes[node_0])
            node_1 = int(pb_to_core_nodes[node_1])
            conn_pb_edges.append((node_0, node_1))
    
    # Convert edge lists to np.ndarrays, and retain unique and
    # lexicographically sorted edges
    conn_core_edges = unique_lexsorted_edges(conn_core_edges)
    conn_pb_edges = unique_lexsorted_edges(conn_pb_edges)

    # Explicitly denote edge type via np.ndarrays
    conn_core_edges_type = np.ones(np.shape(conn_core_edges)[0], dtype=int)
    conn_pb_edges_type = 2 * np.ones(np.shape(conn_pb_edges)[0], dtype=int)

    # Combine edge arrays and edge type arrays
    conn_edges = np.vstack((conn_core_edges, conn_pb_edges), dtype=int)
    conn_edges_type = np.concatenate(
        (conn_core_edges_type, conn_pb_edges_type), dtype=int)
    
    # Lexicographically sort the edges and edge types
    conn_edges, lexsort_indcs = lexsorted_edges(conn_edges, return_indcs=True)
    conn_edges_type = conn_edges_type[lexsort_indcs]

    # Save fundamental graph constituents from this topology
    np.savetxt(conn_edges_filename, conn_edges, fmt="%d")
    np.savetxt(conn_edges_type_filename, conn_edges_type, fmt="%d")
    
def delaunay_network_topology(
        network: str,
        date: str,
        batch: str,
        sample: int,
        scheme: str,
        dim: int,
        n: int,
        config: int) -> None:
    """Delaunay-triangulated network topology.

    This function confirms that the network being called for is a
    Delaunay-triangulated network. Then, the function calls the
    Delaunay-triangulated network initialization function to create the
    Delaunay-triangulated network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "delaunay" is applicable (corresponding to Delaunay-triangulated networks ("delaunay")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular scheme used to generate the positions of the core nodes; either "random", "prhd", "pdhu", or "lammps" (corresponding to the random node placement procedure ("random"), periodic random hard disk node placement procedure ("prhd"), periodic disordered hyperuniform node placement procedure ("pdhu"), or nodes randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        n (int): Number of core nodes.
        config (int): Configuration number.
    
    """
    # Network topology initialization procedure is only applicable for 
    # Delaunay-triangulated networks. Exit if a different type of
    # network is passed.
    if network != "delaunay":
        error_str = (
            "Network topology initialization procedure is only "
            + "applicable for Delaunay-triangulated networks. This "
            + "procedure will only proceed if network = ``delaunay''."
        )
        raise ValueError(error_str)
    delaunay_network_topology_initialization(
        network, date, batch, sample, scheme, dim, n, config)

def delaunay_network_local_topological_descriptor(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int,
        b: float,
        tplgcl_dscrptr: str,
        save_result: bool,
        return_result: bool) -> np.ndarray | float | int | None:
    """Delaunay-triangulated network local topological descriptor.
    
    This function extracts a Delaunay-triangulated network and sets a
    variety of input parameters corresponding to a particular local
    topological descriptor of interest. These are then passed to the
    master network_local_topological_descriptor() function, which
    calculates (and, if called for, saves) the result of the local
    topological descriptor for the Delaunay-triangulated network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "delaunay" is applicable (corresponding to Delaunay-triangulated networks ("delaunay")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
        b (float): Chain segment and/or cross-linker diameter.
        tplgcl_dscrptr (str): Topological descriptor name.
        save_result (bool): Boolean indicating if the result ought to be saved.
        return_result (bool): Boolean indicating if the result ought to be returned.
    
    Returns:
        np.ndarray | float | int | None: Local topological descriptor
        result.
    
    """
    # This local topological descriptor calculation is only applicable
    # for data files associated with Delaunay-triangulated networks.
    # Exit if a different type of network is passed.
    if network != "delaunay":
        error_str = (
            "This local topological descriptor calculation is only "
            + "applicable for data files associated with "
            + "Delaunay-triangulated networks. This calculation will "
            + "proceed only if network = ``delaunay'."
        )
        raise ValueError(error_str)
    
    # Delaunay-triangulated networks are completely
    # elastically-effective, and thus there is no need to specify if an
    # elastically-effective end-linked network is desired
    eeel_ntwrk = False

    # A Delaunay-triangulated network is represented via an nx.Graph
    # object
    multigraph = False

    # Generate filenames
    L_filename = L_filename_str(network, date, batch, sample)
    delaunay_filename = delaunay_filename_str(
        network, date, batch, sample, config)
    coords_filename = delaunay_filename + ".coords"
    conn_edges_filename = delaunay_filename + "-conn_edges" + ".dat"
    conn_edges_type_filename = delaunay_filename + "-conn_edges_type" + ".dat"
    result_filename = delaunay_filename + "-lcl-" + tplgcl_dscrptr + ".dat"

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

def delaunay_network_global_topological_descriptor(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int,
        b: float,
        tplgcl_dscrptr: str,
        np_oprtn: str,
        save_result: bool,
        return_result: bool) -> np.ndarray | float | int | None:
    """Delaunay-triangulated network global topological descriptor.
    
    This function extracts a Delaunay-triangulated network and sets a
    variety of input parameters corresponding to a particular global
    topological descriptor (and numpy function) of interest. These are
    then passed to the master network_global_topological_descriptor()
    function, which calculates (and, if called for, saves) the result of
    the global topological descriptor for the Delaunay-triangulated
    network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "delaunay" is applicable (corresponding to Delaunay-triangulated networks ("delaunay")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
        b (float): Chain segment and/or cross-linker diameter.
        tplgcl_dscrptr (str): Topological descriptor name.
        np_oprtn (str): numpy function/operation name.
        save_result (bool): Boolean indicating if the result ought to be saved.
        return_result (bool): Boolean indicating if the result ought to be returned.
    
    Returns:
        np.ndarray | float | int | None: Global topological descriptor
        result.
    
    """
    # This global topological descriptor calculation is only applicable
    # for data files associated with Delaunay-triangulated networks.
    # Exit if a different type of network is passed.
    if network != "delaunay":
        error_str = (
            "This global topological descriptor calculation is only "
            + "applicable for data files associated with "
            + "Delaunay-triangulated networks. This calculation will "
            + "proceed only if network = ``delaunay'."
        )
        raise ValueError(error_str)
    
    # Delaunay-triangulated networks are completely
    # elastically-effective, and thus there is no need to specify if an
    # elastically-effective end-linked network is desired
    eeel_ntwrk = False

    # A Delaunay-triangulated network is represented via an nx.Graph
    # object
    multigraph = False

    # Generate filenames
    L_filename = L_filename_str(network, date, batch, sample)
    delaunay_filename = delaunay_filename_str(
        network, date, batch, sample, config)
    coords_filename = delaunay_filename + ".coords"
    conn_edges_filename = delaunay_filename + "-conn_edges" + ".dat"
    conn_edges_type_filename = delaunay_filename + "-conn_edges_type" + ".dat"
    
    if np_oprtn == "":
        result_filename = delaunay_filename + "-glbl-" + tplgcl_dscrptr + ".dat"
    else:
        result_filename = (
            delaunay_filename + "-glbl-" + np_oprtn + "-" + tplgcl_dscrptr
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

def delaunay_network_global_morphological_descriptor(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int,
        b: float,
        mrphlgcl_dscrptr: str,
        save_result: bool,
        return_result: bool) -> np.ndarray | float | int | None:
    """Delaunay-triangulated network global morphological descriptor.
    
    This function extracts a Delaunay-triangulated network and sets a
    variety of input parameters corresponding to a particular global
    morphological descriptor of interest. These are then passed to the
    master network_global_morphological_descriptor() function, which
    calculates (and, if called for, saves) the result of the global
    morphological descriptor for the Delaunay-triangulated network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "delaunay" is applicable (corresponding to Delaunay-triangulated networks ("delaunay")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
        b (float): Chain segment and/or cross-linker diameter.
        mrphlgcl_dscrptr (str): Morphological descriptor name.
        save_result (bool): Boolean indicating if the result ought to be saved.
        return_result (bool): Boolean indicating if the result ought to be returned.
    
    Returns:
        np.ndarray | float | int | None: Global morphological descriptor
        result.
    
    """
    # This global morphological descriptor calculation is only
    # applicable for data files associated with Delaunay-triangulated
    # networks. Exit if a different type of network is passed.
    if network != "delaunay":
        error_str = (
            "This global morphological descriptor calculation is only "
            + "applicable for data files associated with "
            + "Delaunay-triangulated networks. This calculation will "
            + "proceed only if network = ``delaunay'."
        )
        raise ValueError(error_str)
    
    # Generate filenames
    L_filename = L_filename_str(network, date, batch, sample)
    delaunay_filename = delaunay_filename_str(
        network, date, batch, sample, config)
    coords_filename = delaunay_filename + ".coords"
    conn_edges_filename = delaunay_filename + "-conn_edges" + ".dat"
    conn_edges_type_filename = delaunay_filename + "-conn_edges_type" + ".dat"
    result_filename = delaunay_filename + "-glbl-" + mrphlgcl_dscrptr + ".dat"

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