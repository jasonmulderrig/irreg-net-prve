import numpy as np
import networkx as nx
from src.descriptors.general_topological_descriptors import n_func

def k_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Node degree.

    This function calculates the node degree in an (undirected) graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        np.ndarray: Node degree.
    
    """
    return np.asarray(list(dict(graph.degree()).values()), dtype=int)

def avrg_nn_k_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Average nearest neighbor degree.

    This function calculates the average nearest neighbor degree for
    each node in an (undirected) graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        np.ndarray: Nodewise average nearest neighbor degree.
    
    """
    return np.asarray(list(nx.average_neighbor_degree(graph).values()))

def k_diff_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Degree difference.

    This function calculates the degree difference for each edge in an
    (undirected) graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        np.ndarray: Edgewise degree difference.
    
    """
    # Calculate and store graph degree in a dictionary
    k_dict = dict(graph.degree())
    
    # Gather edges and initialize degree difference np.ndarray
    edges = list(graph.edges())
    k_diff = np.empty(len(edges), dtype=int)

    # Calculate and store the degree difference of each edge
    for edge_indx, edge in enumerate(edges):
        # Node numbers
        node_0 = int(edge[0])
        node_1 = int(edge[1])

        # Calculate and store the degree difference
        k_diff[edge_indx] = np.abs(k_dict[node_0]-k_dict[node_1], dtype=int)
    
    return k_diff

def avrg_k_diff_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Average degree difference.

    This function calculates the average degree difference for each node
    in an (undirected) graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        np.ndarray: Nodewise average degree difference.
    
    """
    # Initialize nodewise average degree difference np.ndarray
    n = n_func(graph)
    avrg_k_diff = np.empty(n)

    # Calculate and store graph degree in a dictionary
    k_dict = dict(graph.degree())
    
    # Initialize a dictionary to store nodewise degree difference
    # information
    k_diff_dict = k_dict.copy()
    for node in k_diff_dict: k_diff_dict[node] = 0
    
    # Gather edges
    edges = list(graph.edges())

    # Calculate the degree difference of each edge, and add that value
    # to a running sum of nodewise degree difference
    for edge in edges:
        # Node numbers
        node_0 = int(edge[0])
        node_1 = int(edge[1])

        # Calculate degree difference
        k_diff = np.abs(k_dict[node_0]-k_dict[node_1], dtype=int)

        # Add degree difference to each node involved in the edge
        k_diff_dict[node_0] += k_diff
        k_diff_dict[node_1] += k_diff
    
    # Calculate nodewise average degree difference
    indx = 0
    for node in k_diff_dict:
        avrg_k_diff[indx] = k_diff_dict[node] / k_dict[node]
        indx += 1
    
    return avrg_k_diff

def c_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Clustering coefficient.

    This function calculates the clustering coefficient for each node in
    an (undirected) graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        np.ndarray: Nodewise clustering coefficient.
    
    """
    # NetworkX clustering function only accepts graphs of type
    # nx.Graph()
    if graph.is_multigraph(): graph = nx.Graph(graph)
    
    return np.asarray(list(nx.clustering(graph).values()))

def kappa_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Nodal connectivity.

    This function calculates the nodal connectivity for all pairs of
    nodes in an (undirected) graph (excluding all self-loop node pairs).
    This function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Node-pairwise nodal connectivity, excluding all
        self-loop node pairs. Thus, for a network with n nodes labeled
        {0, 1, ..., n-1}, the first n-1 entries are associated with the
        nodal connectivities for all non-self-loop node pairs for the
        zeroth node (0-1, 0-2, ..., 0-(n-1)). The next n-1 entries are
        associated with that for the first node (1-0, 1-2, 1-3, ...,
        1-(n-1)), and so on.
    
    """
    kappa_dict = nx.all_pairs_node_connectivity(graph)
    node_list = list(graph.nodes())
    n = n_func(graph)
    n_pairs = n * (n-1)
    kappa = np.empty(n_pairs, dtype=int)

    indx = 0
    for node_0 in node_list:
        for node_1 in node_list:
            if node_0 == node_1: continue
            else:
                kappa[indx] = kappa_dict[node_0][node_1]
                indx += 1
    
    return kappa

def lcl_avrg_kappa_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Local average nodal connectivity.

    This function calculates the local average nodal connectivity for
    all nodes in an (undirected) graph. This function is best applied to
    fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Average nodewise nodal connectivity.
    
    """
    kappa_dict = nx.all_pairs_node_connectivity(graph)
    n = n_func(graph)
    lcl_avrg_kappa = np.empty(n)
    
    indx = 0
    for node in kappa_dict:
        lcl_avrg_kappa[indx] = sum(kappa_dict[node].values()) / (n-1)
        indx += 1

    return lcl_avrg_kappa

def glbl_avrg_kappa_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """(Global) Average nodal connectivity.

    This function calculates the average nodal connectivity for all
    pairs of nodes in an (undirected) graph. The nodal connectivity for
    each pair of nodes is averaged over all n choose 2 pairs of nodes.
    This function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        float: Average node-pairwise nodal connectivity.
    
    """
    return nx.average_node_connectivity(graph)

def lambda_1_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Algebraic connectivity.

    This function calculates the algebraic connectivity (the
    second-smallest eigenvalue of the Laplacian matrix) for an
    (undirected) graph. This function is best applied to fully connected
    graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        float: Algebraic connectivity.
    
    """
    return nx.algebraic_connectivity(graph)

def r_pearson_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Degree assortativity coefficient using the Pearson correlation
    coefficient.

    This function calculates the degree assortativity coefficient using
    the Pearson correlation coefficient for an (undirected) graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        float: Degree assortativity coefficient.
    
    """
    return nx.degree_pearson_correlation_coefficient(graph)