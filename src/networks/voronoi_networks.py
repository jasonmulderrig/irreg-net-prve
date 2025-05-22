import numpy as np
from scipy.spatial import Voronoi
from src.file_io.file_io import (
    L_filename_str,
    config_filename_str
)
from src.helpers.simulation_box_utils import L_arg_eta_func
from src.helpers.network_topology_initialization_utils import (
    core_node_tessellation,
    box_neighborhood_id
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

def voronoi_filename_str(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int) -> str:
    """Filename prefix associated with Voronoi-tessellated network data
    files.

    This function returns the filename prefix associated with
    Voronoi-tessellated network data files.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "voronoi" is applicable (corresponding to Voronoi-tessellated networks ("voronoi")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
    
    Returns:
        str: The filename prefix associated with Voronoi-tessellated
        network data files.
    
    """
    # This filename prefix convention is only applicable for data files
    # associated with Voronoi-tessellated networks. Exit if a different
    # type of network is passed.
    if network != "voronoi":
        error_str = (
            "This filename prefix convention is only applicable for "
            + "data files associated with Voronoi-tessellated "
            + "networks. This filename prefix will only be supplied if "
            + "network = ``voronoi''."
        )
        raise ValueError(error_str)
    return config_filename_str(network, date, batch, sample, config)

def voronoi_L(
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        b: float,
        n: int,
        eta_n: float) -> None:
    """Simulation box side lengths for Voronoi-tessellated networks.

    This function calculates and saves the simulation box side lengths
    for Voronoi-tessellated networks.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "voronoi" is applicable (corresponding to Voronoi-tessellated networks ("voronoi")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Node diameter.
        n (int): Intended number of core nodes.
        eta_n (float): Node packing density.
    
    """
    # This calculation for L is only applicable for Voronoi-tessellated
    # networks. Exit if a different type of network is passed.
    if network != "voronoi":
        error_str = (
            "This calculation for L is only applicable for "
            + "Voronoi-tessellated networks. This calculation will "
            + "only proceed if network = ``voronoi''."
        )
        raise ValueError(error_str)
    
    # Calculate and save L
    np.savetxt(
        L_filename_str(network, date, batch, sample),
        L_arg_eta_func(dim, b, n, eta_n))

def voronoi_network_topology_initialization(
        network: str,
        date: str,
        batch: str,
        sample: int,
        scheme: str,
        dim: int,
        n: int,
        config: int) -> None:
    """Network topology initialization procedure for Voronoi-tessellated
    networks.

    This function loads the simulation box side lengths and the core
    node coordinates. Then, this function ``tessellates'' the core nodes
    about themselves, applies Voronoi tessellation to the resulting
    tessellated network via the scipy.spatial.Voronoi() function,
    acquires back the periodic network topology of the core nodes, and
    ascertains fundamental graph constituents (node and edge
    information) from this topology.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "voronoi" is applicable (corresponding to Voronoi-tessellated networks ("voronoi")).
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
    # with Voronoi-tessellated network data files, which is reflected in
    # the voronoi_filename_str() function.
    config_filename_prefix = config_filename_str(
        network, date, batch, sample, config)

    # Generate filenames
    coords_filename = config_filename_prefix + ".coords"
    germ_coords_filename = config_filename_prefix + "-germ" + ".coords"
    conn_edges_filename = config_filename_prefix + "-conn_edges" + ".dat"
    conn_edges_type_filename = (
        config_filename_prefix + "-conn_edges_type" + ".dat"
    )

    # Call appropriate helper function to initialize network topology
    if (scheme == "random") or (scheme == "prhd") or (scheme == "pdhu"):
        # Load germ core node coordinates
        germ_coords = np.loadtxt(coords_filename)
    elif scheme == "lammps":
        skiprows_num = 15
        # Load germ core node coordinates
        germ_coords = np.loadtxt(
            coords_filename, skiprows=skiprows_num, max_rows=n)
    
    # Save the germ core node coordinates
    np.savetxt(germ_coords_filename, germ_coords)

    # Tessellate the germ node coordinates
    tsslltd_germ_coords, _ = core_node_tessellation(
        dim, np.arange(np.shape(germ_coords)[0], dtype=int), germ_coords, L)
    
    # Shift the coordinate origin to the center of the simulation box
    # for improved Voronoi tessellation performance
    tsslltd_germ_coords -= 0.5 * L

    # Apply Voronoi tessellation about the tessellated germs
    tsslltd_core_voronoi = Voronoi(tsslltd_germ_coords)

    del tsslltd_germ_coords

    # Extract vertices from the Voronoi tessellation
    vertices = tsslltd_core_voronoi.vertices

    # Restore the coordinate origin to its original location
    vertices += 0.5 * L

    # Confirm that each vertex solely occupies a box neighborhood that
    # is \pm tol about itself
    tol = 1e-10
    for vertex in range(np.shape(vertices)[0]):
        # Determine if the vertex solely occupies a box neighborhood
        # that is \pm tol about itself
        _, box_nghbr_num = box_neighborhood_id(
            dim, vertices, vertices[vertex], tol, inclusive=True, indices=True)
        
        # If there is more than one neighbor, then the neighborhood is
        # overpopulated, and therefore this is an invalid set of input
        # node coordinates for Voronoi tessellation
        if box_nghbr_num > 1:
            error_str = (
                "A vertex neighborhood has more than one vertex! "
                + "Therefore, this is an invalid set of input node "
                + "coordinates for Voronoi tessellation."
            )
            raise ValueError(error_str)
    
    # Extract core vertices
    core_vertices = np.all(np.logical_and(vertices>=0, vertices<L), axis=1)
    core_vertices_indcs = np.where(core_vertices)[0]

    # Number of core vertices
    n = np.shape(core_vertices_indcs)[0]

    # Label core vertices as core nodes
    core_nodes = np.arange(n, dtype=int)

    # Gather core vertices
    core_vertices = vertices[core_vertices_indcs].copy()

    # Tessellate the core vertex coordinates and construct the
    # pb_to_core_nodes np.ndarray
    tsslltd_core_vertices, pb_to_core_nodes = core_node_tessellation(
        dim, core_nodes, core_vertices, L)
    
    del core_nodes
    
    # Extract the ridge vertices from the Voronoi tessellation
    ridge_vertices = tsslltd_core_voronoi.ridge_vertices

    # List for edges of the core and periodic boundary nodes
    tsslltd_core_pb_edges = []

    # In two dimensions, each ridge vertex is a line
    if dim == 2:
        for ridge_vertex in ridge_vertices:
            vertex_0 = int(ridge_vertex[0])
            vertex_1 = int(ridge_vertex[1])
            # Skip over ridge vertices that extend out to infinity
            if (vertex_0 == -1) or (vertex_1 == -1): continue
            else:
                # Determine the indices of the vertices in the
                # tessellated core node topology
                vertex_0_indcs, vertex_0_num = box_neighborhood_id(
                    dim, tsslltd_core_vertices, vertices[vertex_0], tol,
                    inclusive=True, indices=True)
                vertex_1_indcs, vertex_1_num = box_neighborhood_id(
                    dim, tsslltd_core_vertices, vertices[vertex_1], tol,
                    inclusive=True, indices=True)
                
                # Skip over situations where a vertex value is not able
                # to be solved for, which does not occur for the core
                # and periodic boundary vertices
                if (vertex_0_num != 1) or (vertex_1_num != 1):
                    continue
                else:
                    # Extract vertex node index
                    node_0 = int(vertex_0_indcs[0])
                    node_1 = int(vertex_1_indcs[0])
                    # If any of the nodes involved in the ridge vertex
                    # correspond to the original core nodes, then add
                    # that edge to the edge list. Duplicate entries will
                    # arise.
                    if (node_0 < n) or (node_1 < n):
                        tsslltd_core_pb_edges.append((node_0, node_1))
                    else: pass
    # In three dimensions, each ridge vertex is a facet
    elif dim == 3:
        for ridge_vertex in ridge_vertices:
            # Extract list of lines that altogether define the facet
            facet_vertices_0 = ridge_vertex[:-1] + [ridge_vertex[-1]]
            facet_vertices_1 = ridge_vertex[1:] + [ridge_vertex[0]]
            for facet_vertex in range(len(ridge_vertex)):
                vertex_0 = int(facet_vertices_0[facet_vertex])
                vertex_1 = int(facet_vertices_1[facet_vertex])
                # Skip over ridge vertices that extend out to infinity
                if (vertex_0 == -1) or (vertex_1 == -1): continue
                else:
                    # Determine the indices of the vertices in the
                    # tessellated core node topology
                    vertex_0_indcs, vertex_0_num = box_neighborhood_id(
                        dim, tsslltd_core_vertices, vertices[vertex_0], tol,
                        inclusive=True, indices=True)
                    vertex_1_indcs, vertex_1_num = box_neighborhood_id(
                        dim, tsslltd_core_vertices, vertices[vertex_1], tol,
                        inclusive=True, indices=True)
                    
                    # Skip over situations where a vertex value is not
                    # able to be solved for, which does not occur for
                    # the core and periodic boundary vertices
                    if (vertex_0_num != 1) or (vertex_1_num != 1):
                        continue
                    else:
                        # Extract vertex node index
                        node_0 = int(vertex_0_indcs[0])
                        node_1 = int(vertex_1_indcs[0])
                        # If any of the nodes involved in the ridge
                        # vertex correspond to the original core nodes,
                        # then add that edge to the edge list. Duplicate
                        # entries will arise.
                        if (node_0 < n) or (node_1 < n):
                            tsslltd_core_pb_edges.append((node_0, node_1))
                        else: pass

    del vertex, vertices, ridge_vertex, ridge_vertices, tsslltd_core_voronoi

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

    # Save the core node coordinates
    np.savetxt(coords_filename, core_vertices)

def voronoi_network_topology(
        network: str,
        date: str,
        batch: str,
        sample: int,
        scheme: str,
        dim: int,
        n: int,
        config: int) -> None:
    """Voronoi-tessellated network topology.

    This function confirms that the network being called for is a
    Voronoi-tessellated network. Then, the function calls the
    Voronoi-tessellated network initialization function to create the
    Voronoi-tessellated network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "voronoi" is applicable (corresponding to Voronoi-tessellated networks ("voronoi")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular scheme used to generate the positions of the core nodes; either "random", "prhd", "pdhu", or "lammps" (corresponding to the random node placement procedure ("random"), periodic random hard disk node placement procedure ("prhd"), periodic disordered hyperuniform node placement procedure ("pdhu"), or nodes randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        n (int): Number of core nodes.
        config (int): Configuration number.
    
    """
    # Network topology initialization procedure is only applicable for 
    # Voronoi-tessellated networks. Exit if a different type of network
    # is passed.
    if network != "voronoi":
        error_str = (
            "Network topology initialization procedure is only "
            + "applicable for Voronoi-tessellated networks. This "
            + "procedure will only proceed if network = ``voronoi''."
        )
        raise ValueError(error_str)
    voronoi_network_topology_initialization(
        network, date, batch, sample, scheme, dim, n, config)

def voronoi_network_local_topological_descriptor(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int,
        b: float,
        tplgcl_dscrptr: str,
        save_result: bool,
        return_result: bool) -> np.ndarray | float | int | None:
    """Voronoi-tessellated network local topological descriptor.
    
    This function extracts a Voronoi-tessellated network and sets a
    variety of input parameters corresponding to a particular local
    topological descriptor of interest. These are then passed to the
    master network_local_topological_descriptor() function, which
    calculates (and, if called for, saves) the result of the local
    topological descriptor for the Voronoi-tessellated network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "voronoi" is applicable (corresponding to Voronoi-tessellated networks ("voronoi")).
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
    # for data files associated with Voronoi-tessellated networks. Exit
    # if a different type of network is passed.
    if network != "voronoi":
        error_str = (
            "This local topological descriptor calculation is only "
            + "applicable for data files associated with "
            + "Voronoi-tessellated networks. This calculation will "
            + "proceed only if network = ``voronoi'."
        )
        raise ValueError(error_str)
    
    # Voronoi-tessellated networks are completely elastically-effective,
    # and thus there is no need to specify if an elastically-effective
    # end-linked network is desired
    eeel_ntwrk = False

    # A Voronoi-tessellated network is represented via an nx.Graph
    # object
    multigraph = False

    # Generate filenames
    L_filename = L_filename_str(network, date, batch, sample)
    voronoi_filename = voronoi_filename_str(
        network, date, batch, sample, config)
    coords_filename = voronoi_filename + ".coords"
    conn_edges_filename = voronoi_filename + "-conn_edges" + ".dat"
    conn_edges_type_filename = voronoi_filename + "-conn_edges_type" + ".dat"
    result_filename = voronoi_filename + "-lcl-" + tplgcl_dscrptr + ".dat"
    
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

def voronoi_network_global_topological_descriptor(
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
    """Voronoi-tessellated network global topological descriptor.
    
    This function extracts a Voronoi-tessellated network and sets a
    variety of input parameters corresponding to a particular global
    topological descriptor (and numpy function) of interest. These are
    then passed to the master network_global_topological_descriptor()
    function, which calculates (and, if called for, saves) the result of
    the global topological descriptor for the Voronoi-tessellated
    network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "voronoi" is applicable (corresponding to Voronoi-tessellated networks ("voronoi")).
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
    # for data files associated with Voronoi-tessellated networks. Exit
    # if a different type of network is passed.
    if network != "voronoi":
        error_str = (
            "This global topological descriptor calculation is only "
            + "applicable for data files associated with "
            + "Voronoi-tessellated networks. This calculation will "
            + "proceed only if network = ``voronoi'."
        )
        raise ValueError(error_str)
    
    # Voronoi-tessellated networks are completely elastically-effective,
    # and thus there is no need to specify if an elastically-effective
    # end-linked network is desired
    eeel_ntwrk = False

    # A Voronoi-tessellated network is represented via an nx.Graph
    # object
    multigraph = False

    # Generate filenames
    L_filename = L_filename_str(network, date, batch, sample)
    voronoi_filename = voronoi_filename_str(
        network, date, batch, sample, config)
    coords_filename = voronoi_filename + ".coords"
    conn_edges_filename = voronoi_filename + "-conn_edges" + ".dat"
    conn_edges_type_filename = voronoi_filename + "-conn_edges_type" + ".dat"
    
    if np_oprtn == "":
        result_filename = voronoi_filename + "-glbl-" + tplgcl_dscrptr + ".dat"
    else:
        result_filename = (
            voronoi_filename + "-glbl-" + np_oprtn + "-" + tplgcl_dscrptr
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

def voronoi_network_global_morphological_descriptor(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int,
        b: float,
        mrphlgcl_dscrptr: str,
        save_result: bool,
        return_result: bool) -> np.ndarray | float | int | None:
    """Voronoi-tessellated network global morphological descriptor.
    
    This function extracts a Voronoi-tessellated network and sets a
    variety of input parameters corresponding to a particular global
    morphological descriptor of interest. These are then passed to the
    master network_global_morphological_descriptor() function, which
    calculates (and, if called for, saves) the result of the global
    morphological descriptor for the Voronoi-tessellated network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "voronoi" is applicable (corresponding to Voronoi-tessellated networks ("voronoi")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
        b (float): Chain segment and/or cross-linker diameter.
        mrphlgcl_dscrptr (str): Morphological descriptor name.
        np_oprtn (str): numpy function/operation name.
        save_result (bool): Boolean indicating if the result ought to be saved.
        return_result (bool): Boolean indicating if the result ought to be returned.
    
    Returns:
        np.ndarray | float | int | None: Global morphological descriptor
        result.
    
    """
    # This global morphological descriptor calculation is only
    # applicable for data files associated with Voronoi-tessellated
    # networks. Exit if a different type of network is passed.
    if network != "voronoi":
        error_str = (
            "This global morphological descriptor calculation is only "
            + "applicable for data files associated with "
            + "Voronoi-tessellated networks. This calculation will "
            + "proceed only if network = ``voronoi'."
        )
        raise ValueError(error_str)
    
    # Generate filenames
    L_filename = L_filename_str(network, date, batch, sample)
    voronoi_filename = voronoi_filename_str(
        network, date, batch, sample, config)
    coords_filename = voronoi_filename + ".coords"
    conn_edges_filename = voronoi_filename + "-conn_edges" + ".dat"
    conn_edges_type_filename = voronoi_filename + "-conn_edges_type" + ".dat"
    result_filename = voronoi_filename + "-glbl-" + mrphlgcl_dscrptr + ".dat"
    
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