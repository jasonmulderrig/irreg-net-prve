import numpy as np
import networkx as nx
from file_io.file_io import (
    L_filename_str,
    config_filename_str
)
from helpers.network_utils import m_arg_stoich_func
from helpers.network_topology_initialization_utils import (
    core_node_tessellation,
    core2pb_nodes_func,
    core_node_values_update_in_edges_func,
    orb_neighborhood_id
)
from helpers.polymer_network_chain_statistics import (
    p_nu_bimodal_func,
    p_net_bimodal_gaussian_cnfrmtn_func,
    p_rel_net_bimodal_gaussian_cnfrmtn_func
)
from helpers.graph_utils import (
    lexsorted_edges,
    extract_nodes_to_numpy_array,
    extract_edges_to_numpy_array,
    add_nodes_and_node_attributes_from_numpy_arrays,
    add_edges_and_edge_attributes_from_numpy_arrays,
    extract_edge_attribute_to_numpy_array,
    largest_connected_component
)
from descriptors.general_topological_descriptors import l_func
from networks.aelp_networks import (
    core_node_update_func,
    dangling_chains_update_func
)

def abelp_network_nu_assignment(
        rng: np.random.Generator,
        b: float,
        L: np.ndarray,
        p: float,
        nu_min: int,
        nu_max: int,
        conn_edges: np.ndarray,
        conn_edges_type: np.ndarray,
        coords: np.ndarray) -> np.ndarray:
    """Chain segment number assignment procedure for artificial bimodal
    end-linked polymer networks.

    This function assigns a chain segment number to each chain in an
    artificial bimodal end-linked polymer network via a modified Hanson
    protocol.

    Args:
        rng (np.random.Generator): np.random.Generator object.
        b (float): Chain segment and/or cross-linker diameter.
        L (np.ndarray): Simulation box side lengths.
        p (float): Probability that a chain is composed of nu_min segments (and p-1 is the probability that a chain is composed of nu_max segments).
        nu_min (int): Minimum number of segments that chains in the bimodal distribution can adopt.
        nu_max (int): Maximum number of segments that chains in the bimodal distribution can adopt.
        conn_edges (np.ndarray): Edges from the graph capturing the periodic connections between the core nodes.
        conn_edges_type (np.ndarray): Type label for the edges from the graph capturing the periodic connections between the core nodes. Core edges are of type 1, and periodic boundary edges are of type 0.
        coords (np.ndarray): Coordinates of the core nodes.
    
    Returns:
        np.ndarray: Artificial bimodal end-linked polymer network chain
        segment numbers.
    
    """
    # Confirm that the specified minimum chain segment number is greater
    # than or equal to 3 in order to topologically support self-loop
    # chains (a self-loop chain must have at least 3 segments)
    if nu_min < 3:
        error_str = (
            "The specified minimum chain segment number is less than "
            + "3, which is the minimal number of segments a self-loop "
            + "chain can have. Please modify the specified minimum "
            + "chain segment number accordingly."
        )
        print(error_str)
        return None
    
    # Bimodal chain segment numbers
    nu_chn = np.asarray([nu_min, nu_max], dtype=int)
    
    # Calculate Euclidean end-to-end chain/edge lengths
    r_chns = l_func(conn_edges, conn_edges_type, coords, L)
    m = np.shape(conn_edges)[0]
    nu_chns = np.empty(m, dtype=int)

    # Randomly assign a segment number to each chain
    for chn in range(m):
        # Extract end-to-end chain length
        r_chn = r_chns[chn]
        # Determine the physically-constrained minimum chain segment
        # number
        physcl_cnstrnd_nu_min = int(np.ceil(r_chn/b))
        # Self-loop correction
        if physcl_cnstrnd_nu_min < 1: physcl_cnstrnd_nu_min = 1
        # Confirm that the physically-constrained minimum chain segment
        # number is less than or equal to the specified maximum chain
        # segment number
        if physcl_cnstrnd_nu_min > nu_max:
            error_str = (
                "The specified maximum chain segment number is "
                + "less than the physically-constrained minimum chain "
                + "segment number in the network. Please modify the "
                + "specified maximum chain segment number accordingly."
            )
            print(error_str)
            return None
        # If the physically-constrained minimum chain segment number for
        # the chain is greater than the specified minimum chain segment
        # number, then the chain adopts the specified maximum chain
        # segment number
        if physcl_cnstrnd_nu_min > nu_min: nu_chns[chn] = nu_max
        # Otherwise, use the polymer chain probability distribution to
        # randomly select a chain segment number
        else:
            # Calculate and normalize the polymer chain probability
            # distribution
            if r_chn < 1e-10:
                # For self-loop chains, revert back to the bimodal chain
                # segment number probability distribution 
                p_chn = p_nu_bimodal_func(
                    p, nu_min, nu_max, nu_chn.astype(float))
            else:
                p_chn = p_net_bimodal_gaussian_cnfrmtn_func(
                    b, p, nu_min, nu_max, nu_chn.astype(float), r_chn)
            Z_p_chn = np.sum(p_chn, dtype=float)
            p_chn /= Z_p_chn
            p_chn[np.isnan(p_chn)] = 0.
            Z_p_chn = np.sum(p_chn, dtype=float)
            if Z_p_chn == 0:
                p_chn = np.ones(np.shape(p_chn)[0]) / np.shape(p_chn)[0]
            else:
                p_chn /= Z_p_chn
            nu_chns[chn] = int(rng.choice(nu_chn, size=None, p=p_chn))

    # Return the chain segment numbers for the artificial bimodal
    # end-linked polymer network
    return nu_chns

def abelp_network_topology_initialization(
        network: str,
        date: str,
        batch: str,
        sample: int,
        scheme: str,
        dim: int,
        b: float,
        xi: float,
        k: int,
        n: int,
        p: float,
        en_min: int,
        en_max: int,
        config: int,
        max_try: int) -> None:
    """Network topology initialization procedure for artificial
    bimodal end-linked polymer networks.

    This function loads the simulation box side lengths and the core
    cross-linker coordinates. Then, this function initializes and saves
    the topology of an artificial bimodal end-linked polymer network via
    a modified Gusev-Hanson protocol (which is a Monte Carlo procedure).

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "abelp" is applicable (corresponding to artificial bimodal end-linked polymer networks ("abelp")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular scheme used to generate the positions of the core nodes; either "random", "prhd", "pdhu", or "lammps" (corresponding to the random node placement procedure ("random"), periodic random hard disk node placement procedure ("prhd"), periodic disordered hyperuniform node placement procedure ("pdhu"), or nodes randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Chain segment and/or cross-linker diameter.
        xi (float): Chain-to-cross-link connection probability.
        k (int): Maximum cross-linker degree/functionality; either 3, 4, 5, 6, 7, or 8.
        n (int): Number of core cross-linkers.
        p (float): Probability that a chain is composed of nu_min segments (and p-1 is the probability that a chain is composed of nu_max segments).
        en_min (int): Minimum number of segment particles that chains in the bimodal distribution can adopt.
        en_max (int): Maximum number of segment particles that chains in the bimodal distribution can adopt.
        config (int): Configuration number.
        max_try (int): Maximum number of dangling chain instantiation attempts.
    
    """
    # Load simulation box side lengths
    L = np.loadtxt(L_filename_str(network, date, batch, sample))

    # Generate configuration filename prefix. This establishes the
    # configuration filename prefix as the filename prefix associated
    # with artificial end-linked polymer network data files, which is
    # reflected in the aelp_filename_str() function.
    config_filename_prefix = config_filename_str(
        network, date, batch, sample, config)

    # Generate filenames
    coords_filename = config_filename_prefix + ".coords"
    mx_cmp_coords_filename = config_filename_prefix + ".coords"
    mx_cmp_core_nodes_type_filename = (
        config_filename_prefix + "-core_nodes_type" + ".dat"
    )
    mx_cmp_conn_edges_filename = (
        config_filename_prefix + "-conn_edges" + ".dat"
    )
    mx_cmp_conn_edges_type_filename = (
        config_filename_prefix + "-conn_edges_type" + ".dat"
    )
    mx_cmp_l_cntr_conn_edges_filename = (
        config_filename_prefix + "-l_cntr_conn_edges" + ".dat"
    )

    # Call appropriate helper function to initialize network topology
    if (scheme == "random") or (scheme == "prhd") or (scheme == "pdhu"):
        # Load core cross-linker coordinates
        coords = np.loadtxt(coords_filename)
    elif scheme == "lammps":
        skiprows_num = 15
        # Load core cross-linker coordinates
        coords = np.loadtxt(coords_filename, skiprows=skiprows_num, max_rows=n)
    
    # Actual number of core cross-linkers
    n = np.shape(coords)[0]
    
    # Calculate the stoichiometric number of chains
    m = m_arg_stoich_func(n, k)

    # As a fail-safe check, force int-valued parameters to be ints
    k = int(np.floor(k))
    n = int(np.floor(n))    
    en_min = int(np.floor(en_min))
    en_max = int(np.floor(en_max))
    max_try = int(np.floor(max_try))
    m = int(np.floor(m))

    # Calculate minimum and maximum chain segment numbers
    nu_min = en_min - 1
    nu_max = en_max - 1

    # Core cross-linker nodes
    core_nodes = np.arange(n, dtype=int)

    # Identify core nodes as cross-linkers
    core_nodes_type = np.ones(n, dtype=int)

    # Tessellate the core node coordinates and construct the
    # pb2core_nodes np.ndarray
    tsslltd_coords, pb2core_nodes = core_node_tessellation(
        dim, core_nodes, coords, L)

    # Maximal core cross-linker sampling neighborhood radius as per the
    # minimum image criterion 
    r_mic = np.min(L)
    # Maximal polymer chain contour length
    l_cntr_max = nu_max * b
    # Determine the core cross-linker sampling neighborhood radius
    r_nghbrhd = np.min(np.asarray([r_mic, l_cntr_max]))

    # Finely discretize the core cross-linker sampling neighborhood
    # radius
    r_nghbrhd_chns = np.linspace(0, r_nghbrhd, 10001)
    # Calculate and normalize the polymer chain probability distribution
    p_nghbrhd_chns = p_rel_net_bimodal_gaussian_cnfrmtn_func(
        b, p, nu_min, nu_max, r_nghbrhd_chns)
    p_nghbrhd_chns /= np.sum(p_nghbrhd_chns, dtype=float)

    # Initialize core cross-linker sampling neighborhood list
    core_nodes_nghbrhd_init = []
    r_core_nodes_nghbrhd_init = []

    # Determine core cross-linker sampling neighborhood
    for node in np.nditer(core_nodes):
        core_node = coords[int(node)]

        # Downselect the local orb neighborhood with radius r_nghbrhd
        # about the core node cross-linker
        orb_nghbr_indcs, orb_nghbr_num = orb_neighborhood_id(
            dim, tsslltd_coords, core_node, r_nghbrhd, inclusive=True,
            indices=True)
        
        if orb_nghbr_num > 0:
            # Calculate the distance between the core node cross-linker
            # and its local orb neighbors
            orb_nghbrs = tsslltd_coords[orb_nghbr_indcs]
            r_orb_nghbrs = np.asarray(
            [
                np.linalg.norm(core_node-orb_nghbrs[orb_nghbr_indx])
                for orb_nghbr_indx in range(orb_nghbr_num)
            ])

            # Add the cross-linker neighbor nodes array to the core
            # cross-linker sampling neighborhood list, and add the array
            # of the distance between the core node cross-linker
            # and its local orb neighbors to the core cross-linker
            # sampling neighborhood distance list
            core_nodes_nghbrhd_init.append(orb_nghbr_indcs)
            r_core_nodes_nghbrhd_init.append(r_orb_nghbrs)
        else:
            core_nodes_nghbrhd_init.append(np.asarray([]))
            r_core_nodes_nghbrhd_init.append(np.asarray([]))
    
    # Retain unique nodes from the core and periodic boundary
    # cross-linkers in the core cross-linker sampling neighborhood list
    core_pb_nodes = np.unique(
        np.concatenate(
            tuple(nghbrhd for nghbrhd in core_nodes_nghbrhd_init), dtype=int))
    
    # Construct an np.ndarray that returns the index for each node
    # number in the core_pb_nodes np.ndarray
    core_pb_nodes_indcs = -1 * np.ones(np.max(core_pb_nodes)+1, dtype=int)
    core_pb_nodes_indcs[core_pb_nodes] = np.arange(
        np.shape(core_pb_nodes)[0], dtype=int)

    # Extract the core and periodic boundary cross-linker coordinates
    # using the corresponding node numbers
    core_pb_coords = tsslltd_coords[core_pb_nodes].copy()

    del tsslltd_coords

    # Extract the core and periodic boundary cross-linker nodes in the
    # pb2core_nodes np.ndarray
    pb2core_nodes = pb2core_nodes[core_pb_nodes].copy()

    # Construct the core2pb_nodes list
    core2pb_nodes = core2pb_nodes_func(core_nodes, pb2core_nodes)

    # Refactor the node numbers in the core cross-linker sampling
    # neighborhood list
    for core_node in np.nditer(core_nodes):
        core_node = int(core_node)
        nghbr_nodes = core_nodes_nghbrhd_init[core_node]
        for nghbr_node_indx in range(np.shape(nghbr_nodes)[0]):
            nghbr_nodes[nghbr_node_indx] = int(
                core_pb_nodes_indcs[nghbr_nodes[nghbr_node_indx]])
        core_nodes_nghbrhd_init[core_node] = nghbr_nodes
    
    # Initialize tracker for failure of dangling chain creation
    dnglng_chn_fail = False

    # Initialize random number generator
    rng = np.random.default_rng()

    # Initialize dangling chain free end node coordinates list
    dnglng_chns_coords = []

    # Initialize edges and edges type lists
    conn_edges = []
    conn_edges_type = []
    conn_dnglng_edges = []
    conn_dnglng_edges_type = []

    # Initialize number of dangling chains
    dnglng_n = 0

    # Network topology initialization
    while True:
        # Initialize anti-degree of the core cross-linker nodes
        core_nodes_anti_k = k*np.ones(n, dtype=int)
        # Initialize core cross-linker node active sites
        core_nodes_active_sites = np.repeat(core_nodes, k)
        core_nodes_active_sites_num = np.shape(core_nodes_active_sites)[0]

        # Initialize core cross-linker sampling neighborhood list
        core_nodes_nghbrhd = core_nodes_nghbrhd_init.copy()
        r_core_nodes_nghbrhd = r_core_nodes_nghbrhd_init.copy()
        
        # Initialize cross-linker node coordinates for dangling chains
        core_pb_dnglng_chns_coords = core_pb_coords.copy()

        # Initialize dangling chain free end node coordinates list
        dnglng_chns_coords = []

        # Initialize edge array, edges list, and edges type lists
        edges = np.full((m, 2), np.inf)
        conn_edges = []
        conn_edges_type = []
        conn_dnglng_edges = []
        conn_dnglng_edges_type = []

        # Initialize number of dangling chains
        dnglng_n = 0

        # Initialize node number integer constants
        core_node = 0
        nghbr_node = 0

        # Initialize and randomly shuffle chain end numbers
        chn_ends = np.arange(2*m, dtype=int)
        rng.shuffle(chn_ends)

        # Initialize network topology on a chain end-by-chain end basis
        for chn_end in np.nditer(chn_ends):
            # Chain end is a free chain end
            if rng.random() > xi: continue
            else:
                # Identify the chain and its ends
                chn_end = int(chn_end)
                chn = int(np.floor(chn_end/2))
                chn_end_0 = int(chn_end%2)
                chn_end_1 = 1 - chn_end_0
                # If this chain has not yet been instantiated, then
                # randomly select an active site from the core nodes to
                # instantiate the chain
                if (edges[chn, 0] == np.inf) and (edges[chn, 1] == np.inf):
                    core_node_active_site_indx = rng.integers(
                        np.shape(core_nodes_active_sites)[0], dtype=int)
                    core_node = (
                        core_nodes_active_sites[core_node_active_site_indx]
                    )
                    # Instantiate the chain at one end
                    edges[chn, chn_end_0] = core_node
                    # Update inactive sites, anti-degree, active core
                    # cross-linker nodes, and the sampling neighborhood
                    # list for the parent core cross-linker node
                    (core_nodes_active_sites, core_nodes_active_sites_num, 
                     core_nodes_anti_k, core_nodes_nghbrhd,
                     r_core_nodes_nghbrhd) = core_node_update_func(
                         core_node, core_node_active_site_indx,
                         core_nodes_active_sites, core_nodes_active_sites_num,
                         core_nodes_anti_k, core_nodes, core2pb_nodes,
                         core_nodes_nghbrhd, r_core_nodes_nghbrhd)
                # Otherwise, the chain has previously been instantiated
                else:
                    # Extract sampling neighborhood about the
                    # previously-instantiated chain end
                    core_node = int(edges[chn, chn_end_1])
                    nghbr_nodes = core_nodes_nghbrhd[core_node]
                    nghbr_nodes_num = np.shape(nghbr_nodes)[0]
                    r_nghbr_nodes = r_core_nodes_nghbrhd[core_node]
                    # Check if the sampling neighborhood is empty. If
                    # so, then the chain is a dangling chain.
                    if nghbr_nodes_num == 0:
                        continue
                    else:
                        # Check if there is only one neighbor in the
                        # sampling neighborhood
                        if nghbr_nodes_num == 1:
                            nghbr_node = nghbr_nodes[0]
                        else:
                            # Calculate and normalize weighting factors
                            p_nghbrhd = np.empty(nghbr_nodes_num)
                            for nghbr_node_indx in range(nghbr_nodes_num):
                                r_nghbr_node = r_nghbr_nodes[nghbr_node_indx]
                                if r_nghbr_node == 0.0:
                                    p_nghbrhd[nghbr_node_indx] = 0.0
                                else:
                                    # Linear interpolation
                                    r_nghbrhd_chns_diff = (
                                        r_nghbrhd_chns - r_nghbr_node
                                    )
                                    r_nghbrhd_chns_diff[r_nghbrhd_chns_diff < 0] = (
                                        np.inf
                                    )
                                    indx_right = np.argmin(r_nghbrhd_chns_diff)
                                    r_right = r_nghbrhd_chns[indx_right]
                                    p_right = p_nghbrhd_chns[indx_right]
                                    indx_left = indx_right - 1
                                    r_left = r_nghbrhd_chns[indx_left]
                                    p_left = p_nghbrhd_chns[indx_left]
                                    p_nghbrhd[nghbr_node_indx] = (
                                        p_left + (r_nghbr_node-r_left)
                                        * (p_right-p_left) / (r_right-r_left)
                                    )
                            p_nghbrhd /= np.sum(p_nghbrhd, dtype=float)
                            # Randomly select a neighbor cross-linker
                            # node to host the other end of the chain
                            nghbr_node = int(
                                rng.choice(nghbr_nodes, size=None, p=p_nghbrhd))
                        # Instantiate the other end of the chain
                        edges[chn, chn_end_0] = nghbr_node
                        core_node = int(pb2core_nodes[nghbr_node])
                        core_node_active_site_indx = int(
                            np.where(core_nodes_active_sites == core_node)[0][0])
                        # Update inactive sites, anti-degree, active
                        # core cross-linker nodes, and the sampling
                        # neighborhood list for the parent core
                        # cross-linker node
                        (core_nodes_active_sites, core_nodes_active_sites_num,
                         core_nodes_anti_k, core_nodes_nghbrhd,
                         r_core_nodes_nghbrhd) = core_node_update_func(
                             core_node, core_node_active_site_indx,
                             core_nodes_active_sites, core_nodes_active_sites_num,
                             core_nodes_anti_k, core_nodes, core2pb_nodes,
                             core_nodes_nghbrhd, r_core_nodes_nghbrhd)
        
        # Post-process the edges array (that was populated during the
        # network topology initialization) on a chain-by-chain basis
        # into edges lists
        for chn in range(m):
            # Chain is a free chain
            if (edges[chn, 0] == np.inf) and (edges[chn, 1] == np.inf):
                continue
            # Chain is a core edge
            elif (edges[chn, 0] < n) and (edges[chn, 1] < n):
                conn_edges.append((int(edges[chn, 0]), int(edges[chn, 1])))
                conn_edges_type.append(1)
            # Chain is a periodic edge
            elif (edges[chn, 0] < n) and (n <= edges[chn, 1] < np.inf):
                conn_edges.append(
                    (int(edges[chn, 0]), int(pb2core_nodes[int(edges[chn, 1])])))
                conn_edges_type.append(0)
            elif (n <= edges[chn, 0] < np.inf) and (edges[chn, 1] < n):
                conn_edges.append(
                    (int(pb2core_nodes[int(edges[chn, 0])]), int(edges[chn, 1])))
                conn_edges_type.append(0)
            # Chain is a dangling chain
            elif (edges[chn, 0] < n) and (edges[chn, 1] == np.inf):
                (dnglng_chn_fail, dnglng_n, conn_dnglng_edges,
                 conn_dnglng_edges_type, dnglng_chns_coords,
                 core_pb_dnglng_chns_coords) = dangling_chains_update_func(
                     int(edges[chn, 0]), max_try, rng, r_nghbrhd_chns,
                     p_nghbrhd_chns, dim, b, n, L, dnglng_chn_fail, dnglng_n,
                     conn_dnglng_edges, conn_dnglng_edges_type,
                     dnglng_chns_coords, core_pb_dnglng_chns_coords)
                # Break if a dangling chain failed to be instantiated
                if dnglng_chn_fail == True: break
            elif (edges[chn, 0] == np.inf) and (edges[chn, 1] < n):
                (dnglng_chn_fail, dnglng_n, conn_dnglng_edges,
                 conn_dnglng_edges_type, dnglng_chns_coords,
                 core_pb_dnglng_chns_coords) = dangling_chains_update_func(
                     int(edges[chn, 1]), max_try, rng, r_nghbrhd_chns,
                     p_nghbrhd_chns, dim, b, n, L, dnglng_chn_fail, dnglng_n,
                     conn_dnglng_edges, conn_dnglng_edges_type,
                     dnglng_chns_coords, core_pb_dnglng_chns_coords)
                # Break if a dangling chain failed to be instantiated
                if dnglng_chn_fail == True: break

        # Restart the network topology initialization protocol if a
        # dangling chain failed to be instantiated
        if dnglng_chn_fail == True: continue
        
        del core_nodes_anti_k, core_nodes_active_sites
        del core_pb_dnglng_chns_coords
        del edges, chn_ends
        # Break out of acceptable initialized topology 
        break

    del core_pb_nodes, pb2core_nodes, core2pb_nodes
    del core_pb_coords
    del core_nodes_nghbrhd_init, core_nodes_nghbrhd
    del r_core_nodes_nghbrhd_init, r_core_nodes_nghbrhd
    del r_nghbrhd_chns, p_nghbrhd_chns

    # Convert edges and edges type lists to np.ndarrays
    conn_edges = np.asarray(conn_edges, dtype=int)
    conn_edges_type = np.asarray(conn_edges_type, dtype=int)
    conn_dnglng_edges = np.asarray(conn_dnglng_edges, dtype=int)
    conn_dnglng_edges_type = np.asarray(conn_dnglng_edges_type, dtype=int)

    # Update core nodes, core nodes type, conn_edges, conn_edges type,
    # and core node coordinate np.ndarrays if dangling chains were added
    if dnglng_n > 0:
        core_nodes = np.arange(n+dnglng_n, dtype=int)
        core_nodes_type = np.concatenate(
            (core_nodes_type, np.zeros(dnglng_n, dtype=int)), dtype=int)
        conn_edges = np.vstack((conn_edges, conn_dnglng_edges), dtype=int)
        conn_edges_type = np.concatenate(
            (conn_edges_type, conn_dnglng_edges_type), dtype=int)
        coords = np.vstack((coords, np.asarray(dnglng_chns_coords)))
    
    del dnglng_n, conn_dnglng_edges, conn_dnglng_edges_type, dnglng_chns_coords
    
    # Create nx.MultiGraph by adding nodes and node attributes before
    # edges and edge attributes
    conn_graph = nx.MultiGraph()
    conn_graph = add_nodes_and_node_attributes_from_numpy_arrays(
        conn_graph, core_nodes, ["type"], core_nodes_type)
    conn_graph = add_edges_and_edge_attributes_from_numpy_arrays(
        conn_graph, conn_edges, ["type"], conn_edges_type)
    
    # Isolate largest/maximum connected component, and extract nodes and
    # edges
    mx_cmp_conn_graph = largest_connected_component(conn_graph)
    mx_cmp_core_nodes = extract_nodes_to_numpy_array(mx_cmp_conn_graph)
    mx_cmp_conn_edges = extract_edges_to_numpy_array(mx_cmp_conn_graph)

    # Sort the node numbers in ascending order
    mx_cmp_core_nodes = np.sort(mx_cmp_core_nodes)

    # Isolate the core nodes type and the core node coordinates for the
    # largest/maximum connected component
    mx_cmp_core_nodes_type = core_nodes_type[mx_cmp_core_nodes]
    mx_cmp_coords = coords[mx_cmp_core_nodes]

    # Update each node value in each edge with an updated node value
    # corresponding to an ascending array of core nodes
    mx_cmp_conn_edges = core_node_values_update_in_edges_func(
        mx_cmp_core_nodes, mx_cmp_conn_edges
    )
    
    # Extract edges type for the largest/maximum connected component
    mx_cmp_conn_edges_type = extract_edge_attribute_to_numpy_array(
        mx_cmp_conn_graph, "type", dtype_int=True)
    
    # Lexicographically sort the edges and edge types
    mx_cmp_conn_edges, lexsort_indcs = lexsorted_edges(
        mx_cmp_conn_edges, return_indcs=True)
    mx_cmp_conn_edges_type = mx_cmp_conn_edges_type[lexsort_indcs]
    
    # Assign a contour length for each chain/edge in the largest/maximum
    # connected component
    mx_cmp_l_cntr_conn_edges = b * abelp_network_nu_assignment(
        rng, b, L, p, nu_min, nu_max, mx_cmp_conn_edges, mx_cmp_conn_edges_type,
        mx_cmp_coords)

    # Save fundamental graph constituents from this topology
    np.savetxt(
        mx_cmp_core_nodes_type_filename, mx_cmp_core_nodes_type, fmt="%d")
    np.savetxt(
        mx_cmp_conn_edges_filename, mx_cmp_conn_edges, fmt="%d")
    np.savetxt(
        mx_cmp_conn_edges_type_filename, mx_cmp_conn_edges_type, fmt="%d")
    np.savetxt(mx_cmp_l_cntr_conn_edges_filename, mx_cmp_l_cntr_conn_edges)
    
    # Save the core node coordinates
    np.savetxt(mx_cmp_coords_filename, mx_cmp_coords)

def abelp_network_topology(
        network: str,
        date: str,
        batch: str,
        sample: int,
        scheme: str,
        dim: int,
        b: float,
        xi: float,
        k: int,
        n: int,
        p: float,
        en_min: int,
        en_max: int,
        config: int,
        max_try: int) -> None:
    """Artificial bimodal end-linked polymer network topology.

    This function confirms that the network being called for is an
    artificial bimodal end-linked polymer network. Then, the function
    calls the artificial bimodal end-linked polymer network
    initialization function to create the artificial bimodal end-linked
    polymer network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "abelp" is applicable (corresponding to artificial bimodal end-linked polymer networks ("abelp")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular scheme used to generate the positions of the core nodes; either "random", "prhd", "pdhu", or "lammps" (corresponding to the random node placement procedure ("random"), periodic random hard disk node placement procedure ("prhd"), periodic disordered hyperuniform node placement procedure ("pdhu"), or nodes randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Chain segment and/or cross-linker diameter.
        xi (float): Chain-to-cross-link connection probability.
        k (int): Maximum cross-linker degree/functionality; either 3, 4, 5, 6, 7, or 8.
        n (int): Number of core cross-linkers.
        p (float): Probability that a chain is composed of nu_min segments (and p-1 is the probability that a chain is composed of nu_max segments).
        en_min (int): Minimum number of segment particles that chains in the bimodal distribution can adopt.
        en_max (int): Maximum number of segment particles that chains in the bimodal distribution can adopt.
        config (int): Configuration number.
        max_try (int): Maximum number of dangling chain instantiation attempts.
    
    """
    # Network topology initialization procedure is only applicable for
    # artificial bimodal end-linked polymer networks. Exit if a
    # different type of network is passed.
    if network != "abelp":
        error_str = (
            "Network topology initialization procedure is only "
            + "applicable for artificial bimodal end-linked polymer "
            + "networks. This calculation will only proceed if "
            + "network = ``abelp''."
        )
        print(error_str)
        return None
    abelp_network_topology_initialization(
        network, date, batch, sample, scheme, dim, b, xi, k, n, p,
        en_min, en_max, config, max_try)