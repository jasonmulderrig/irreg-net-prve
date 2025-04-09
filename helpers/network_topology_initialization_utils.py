import numpy as np

def tessellation_protocol(dim: int) -> tuple[np.ndarray, int]:
    """Tessellation protocol.

    This function determines the tessellation protocol and the number of
    tessellations involved in that protocol. Each of these are sensitive
    to the physical dimensionality of the network.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
    
    Returns:
        tuple[np.ndarray, int]: Tessellation protocol and the number of
        tessellations involved in that protocol, respectively.
    
    """
    base_tsslltn = np.asarray([-1, 0, 1], dtype=int)
    if dim == 2:
        tsslltn = (
            np.asarray(np.meshgrid(base_tsslltn, base_tsslltn)).T.reshape(-1, 2)
        )
    elif dim == 3:
        tsslltn = (
            np.asarray(np.meshgrid(base_tsslltn, base_tsslltn, base_tsslltn)).T.reshape(-1, 3)
        )
    tsslltn_num = np.shape(tsslltn)[0]
    return tsslltn, tsslltn_num

def tessellation(
        coords: np.ndarray,
        tsslltn: np.ndarray,
        L: np.ndarray) -> np.ndarray:
    """Tessellation.
    
    This function fully tessellates (or translates) an arbitrary
    coordinate in each cardinal direction in the spatial plane via a
    scaling L.

    Args:
        coords (np.ndarray): Coordinates to be tessellated.
        tsslltn (np.ndarray): Tessellation protocol.
        L (np.ndarray): Tessellation scaling.
    
    Returns:
        np.ndarray: Tessellated coordinates.
    
    """
    return coords + tsslltn * L

def core_node_tessellation(
        dim: int,
        core_nodes: np.ndarray,
        core_coords: np.ndarray,
        L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Core node tessellation.
    
    This function fully tessellates (or translates) an arbitrary set of
    coordinates in each cardinal direction in the spatial plane via a
    scaling L. The initially provided coordinates are associated with
    the core nodes. The tessellated coordinates are stored after the
    initially provided coordinates of the core nodes. Additionally, an
    np.ndarray is created that identifies which core node it represents
    in the tessellated configuration.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        core_nodes (np.ndarray): Core nodes.
        coords (np.ndarray): Coordinates (associated with the core nodes) to be tessellated.
        L (np.ndarray): Tessellation scaling.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Tessellated coordinates and an
        np.ndarray that returns the core node that corresponds to each
        core and periodic boundary node, i.e.,
        pb2core_nodes[core_pb_node] = core_node.
    
    """
    if (np.shape(core_nodes)[0] != np.shape(core_coords)[0]) or (dim != np.shape(core_coords)[1]):
        error_str = (
            "Either the number of core nodes does not match the number "
            + "of core node coordinates or the specified network "
            + "dimension does not match the dimensionality of the core "
            + "node coordinates. This calculation will only proceed if "
            + "both of those conditions are satisfied. Please modify "
            + "accordingly."
        )
        print(error_str)
        return None
    else:
        # Tessellation protocol
        tsslltn, tsslltn_num = tessellation_protocol(dim)

        # Copy the coordinates as the first n entries in the tessellated
        # coordinate np.ndarray
        tsslltd_core_coords = core_coords.copy()

        # Tessellate the germ core nodes
        for tsslltn_actn in range(tsslltn_num):
            # Skip the zero tessellation call because the core nodes are
            # being tessellated about themselves
            if np.array_equal(tsslltn[tsslltn_actn], np.zeros(dim)) == True:
                continue
            else:
                tsslltd_core_coords = np.vstack(
                    (tsslltd_core_coords, tessellation(core_coords, tsslltn[tsslltn_actn], L)))

        # Construct the pb2core_nodes np.ndarray such that
        # pb2core_nodes[core_pb_node] = core_node
        pb2core_nodes = np.tile(core_nodes, tsslltn_num)

        return tsslltd_core_coords, pb2core_nodes

def core2pb_nodes_func(
        core_nodes: np.ndarray,
        pb2core_nodes: np.ndarray) -> list[np.ndarray]:
    """List of np.ndarrays corresponding to the periodic boundary nodes
    associated with a particular core node.

    This function creates a list of np.ndarrays corresponding to the
    periodic boundary nodes associated with a particular core node such
    that core2pb_nodes[core_node] = pb_nodes.

    Args:
        core_nodes (np.ndarray): Core nodes.
        pb2core_nodes (np.ndarray): np.ndarray that returns the core node that corresponds to each core and periodic boundary node, i.e., pb2core_nodes[core_pb_node] = core_node.
    
    Returns:
        list[np.ndarray]: list of np.ndarrays corresponding to the
        periodic boundary nodes associated with a particular core node.

    """
    core2pb_nodes = []
    for core_node in np.nditer(core_nodes):
        # Isolate core and periodic boundary nodes associated with the
        # core node, and delete the core node
        pb_nodes = np.delete(
            np.where(pb2core_nodes == int(core_node))[0], 0, axis=0)
        core2pb_nodes.append(pb_nodes)
    return core2pb_nodes

def core_node_values_update_in_edges_func(
        core_nodes: np.ndarray,
        edges: np.ndarray) -> np.ndarray:
    """Update each node value in each edge with an updated node value
    corresponding to an ascending array of core nodes.

    This function updates each node value in each edge with an updated
    node value corresponding to an ascending array of core nodes.

    Args:
        core_nodes (np.ndarray): Core nodes.
        edges (np.ndarray): Edges.
    
    Returns:
        np.ndarray: Edges with updated node values.

    """
    # Construct an np.ndarray that returns the index for each core node
    # number
    core_nodes_indcs = -1 * np.ones(np.max(core_nodes)+1, dtype=int)
    core_nodes_indcs[core_nodes] = np.arange(np.shape(core_nodes)[0], dtype=int)

    # Update all original core node values with updated core node values
    # for each edge
    for edge in range(np.shape(edges)[0]):
        edges[edge, 0] = int(core_nodes_indcs[edges[edge, 0]])
        edges[edge, 1] = int(core_nodes_indcs[edges[edge, 1]])
    
    return edges

def box_neighborhood_id(
        dim: int,
        coords: np.ndarray,
        coord: np.ndarray,
        l: float,
        inclusive: bool,
        indices: bool) -> tuple[np.ndarray, int]:
    """Box neighborhood identification.
    
    This function return the coordinates or the indices of the
    coordinates that lie within a box neighborhood that is \\pm
    half-side length l about a given coordinate.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        coords (np.ndarray): Coordinates that may or may not reside in the box neighborhood.
        coord (np.ndarray): Given coordinate which the box neighborhood is defined about.
        l (float): Half-side length defining the box neighborhood about the given coordinate.
        inclusive (bool): Boolean indicating if the box neighborhood is inclusive or exclusive of its boundary.
        indices (bool): Boolean indicating if the indices of the box neighbor coordinates are to be provided. If True, then the indices of the box neighbor coordinates are provided. If False, then the box neighbor coordinates are provided.
    
    Returns:
        tuple[np.ndarray, int]: Box neighbor coordinates or box neighbor
        indices, and the number of box neighbors.

    """
    # Extract the x- and y-coordinates of the box center point and the
    # candidate points
    x_coord = coord[0]
    y_coord = coord[1]
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    
    # Define the boundary of the box neighborhood
    box_nghbr_x_lb = x_coord - l
    box_nghbr_x_ub = x_coord + l
    box_nghbr_y_lb = y_coord - l
    box_nghbr_y_ub = y_coord + l

    # Determine which candidate points are box neighbors
    if inclusive:
        box_nghbrs = np.logical_and(
            np.logical_and(x_coords>=box_nghbr_x_lb, x_coords<=box_nghbr_x_ub),
            np.logical_and(y_coords>=box_nghbr_y_lb, y_coords<=box_nghbr_y_ub))
    else:
        box_nghbrs = np.logical_and(
            np.logical_and(x_coords>box_nghbr_x_lb, x_coords<box_nghbr_x_ub),
            np.logical_and(y_coords>box_nghbr_y_lb, y_coords<box_nghbr_y_ub))

    if dim == 3:
        # Extract the z-coordinates of the box center point and the
        # candidate points
        z_coord = coord[2]
        z_coords = coords[:, 2]

        # Define the boundary of the box neighborhood
        box_nghbr_z_lb = z_coord - l
        box_nghbr_z_ub = z_coord + l

        # Determine which candidate points are box neighbors
        if inclusive:
            box_nghbrs = np.logical_and(
                box_nghbrs,
                np.logical_and(z_coords>=box_nghbr_z_lb, z_coords<=box_nghbr_z_ub))
        else:
            box_nghbrs = np.logical_and(
                box_nghbrs,
                np.logical_and(z_coords>box_nghbr_z_lb, z_coords<box_nghbr_z_ub))
    
    # Determine the indices of the box neighbors, and calculate the
    # number of box neighbors
    box_nghbr_indcs = np.where(box_nghbrs)[0]
    box_nghbr_num = np.shape(box_nghbr_indcs)[0]

    # Box neighborhood is empty
    if box_nghbr_num == 0:
        return np.asarray([]), 0
    # Box neighborhood has at least one neighbor in it
    elif box_nghbr_num > 0:
        if indices:
            return box_nghbr_indcs, box_nghbr_num
        else:
            return coords[box_nghbr_indcs], box_nghbr_num

def orb_neighborhood_id(
        dim: int,
        coords: np.ndarray,
        coord: np.ndarray,
        r: float,
        inclusive: bool,
        indices: bool) -> tuple[np.ndarray, int]:
    """Orb neighborhood identification.

    This function identifies which coordinates lie within an orb
    neighborhood defined by radius r about a given coordinate.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        coords (np.ndarray): Coordinates that may or may not reside in the orb neighborhood.
        coord (np.ndarray): Given coordinate which the orb neighborhood is defined about.
        r (float): Radius defining the orb neighborhood about the given coordinate.
        inclusive (bool): Boolean indicating if the orb neighborhood is inclusive or exclusive of its boundary.
        indices (bool): Boolean indicating if the indices of the orb neighbor coordinates are to be provided. If True, then the indices of the orb neighbor coordinates are provided. If False, then the orb neighbor coordinates are provided.
    
    Returns:
        tuple[np.ndarray, int]: Orb neighbor coordinates or orb neighbor
        indices, and the number of orb neighbors.

    """
    # Gather the corresponding box neighborhood about the given
    # coordinate
    box_nghbr_indcs, box_nghbr_num = box_neighborhood_id(
        dim, coords, coord, r, inclusive, indices=True)
    
    # Corresponding box neighborhood is empty, which implies that the
    # orb neighborhood is also empty
    if box_nghbr_num == 0:
        return np.asarray([]), 0
    # Corresponding box neighborhood has at least one neighbor in it
    elif box_nghbr_num > 0:
        # Extract box neighbor coordinates
        box_nghbr_coords = coords[box_nghbr_indcs]

        # Calculate the distance between the given coordinate and its
        # box neighbors
        dist = np.asarray(
            [
                np.linalg.norm(coord-box_nghbr_coords[box_nghbr_indx])
                for box_nghbr_indx in range(box_nghbr_num)
            ])
        
        # Determine the naive indices of the orb neighbors, and
        # calculate the number of orb neighbors
        if inclusive:
            orb_nghbr_indcs = np.where(dist<=r)[0]
        else:
            orb_nghbr_indcs = np.where(dist<r)[0]
        orb_nghbr_num = np.shape(orb_nghbr_indcs)[0]

        # Orb neighborhood is empty
        if orb_nghbr_num == 0:
            return np.asarray([]), 0
        # Orb neighborhood has at least one neighbor in it
        elif orb_nghbr_num > 0:
            # Determine the indices of the orb neighbors
            orb_nghbr_indcs = box_nghbr_indcs[orb_nghbr_indcs]
            if indices:
                return orb_nghbr_indcs, orb_nghbr_num
            else:
                return coords[orb_nghbr_indcs], orb_nghbr_num