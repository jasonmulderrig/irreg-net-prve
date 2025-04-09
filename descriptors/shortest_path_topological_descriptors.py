import numpy as np
import networkx as nx
from descriptors.general_topological_descriptors import n_func

def attr_d_func(
        graph: nx.Graph | nx.MultiGraph,
        attr: str,
        dtype_int: bool) -> np.ndarray:
    """Edge-attributed shortest path length.

    This function calculates the shortest path length with respect to a
    specified edge attribute for all pairs of nodes in an (undirected)
    graph (excluding all self-loop node pairs). This function is best
    applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
        attr (str): Edge attribute with respect to which the shortest path length will be calculated.
        dtype_int (bool): Boolean indicating if the edge-attributed shortest path length values are integers (True) or not (False).
    
    Returns:
        np.ndarray: Node-pairwise edge-attributed shortest path length,
        excluding all self-loop node pairs. Thus, for a network with n
        nodes labeled {0, 1, ..., n-1}, the first n-1 entries are
        associated with the shortest path length for all non-self-loop
        node pairs for the zeroth node (0-1, 0-2, ..., 0-(n-1)). The
        next n-1 entries are associated with that for the first node
        (1-0, 1-2, 1-3, ..., 1-(n-1)), and so on.
    
    """
    if attr == "": d_dict = dict(nx.shortest_path_length(graph))
    else: d_dict = dict(nx.shortest_path_length(graph, weight=attr))
    node_list = list(graph.nodes())
    n = n_func(graph)
    n_pairs = n * (n-1)
    if dtype_int: d = np.empty(n_pairs, dtype=int)
    else: d = np.empty(n_pairs)

    indx = 0
    for node_0 in node_list:
        for node_1 in node_list:
            if node_0 == node_1: continue
            else:
                d[indx] = d_dict[node_0][node_1]
                indx += 1
    
    return d

def attr_avrg_d_func(graph: nx.Graph | nx.MultiGraph, attr: str) -> np.ndarray:
    """Edge-attributed average shortest path length.

    This function calculates the average shortest path length with
    respect to a specified edge attribute for each node in an
    (undirected) graph. This function is best applied to fully connected
    graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
        attr (str): Edge attribute with respect to which the average shortest path length will be calculated.
    
    Returns:
        np.ndarray: Nodewise edge-attributed average shortest path
        length.
    
    """
    if attr == "": d_dict = dict(nx.shortest_path_length(graph))
    else: d_dict = dict(nx.shortest_path_length(graph, weight=attr))
    n = n_func(graph)
    avrg_d = np.empty(n)

    indx = 0
    for node in d_dict:
        avrg_d[indx] = sum(d_dict[node].values()) / (n-1)
        indx += 1
    
    return avrg_d

def attr_e_func(
        graph: nx.Graph | nx.MultiGraph,
        attr: str,
        attr_d_dtype_int: bool) -> np.ndarray:
    """Edge-attributed graph efficiency.

    This function calculates the graph efficiency with respect to a
    specified edge attribute for all pairs of nodes in an (undirected)
    graph (excluding all self-loop node pairs). This function is best
    applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
        attr (str): Edge attribute with respect to which the graph efficiency will be calculated.
        attr_d_dtype_int (bool): Boolean indicating if the edge-attributed shortest path length values are integers (True) or not (False).
    
    Returns:
        np.ndarray: Node-pairwise edge-attributed graph efficiency,
        excluding all self-loop node pairs. Thus, for a network with n
        nodes labeled {0, 1, ..., n-1}, the first n-1 entries are
        associated with the graph efficiency for all non-self-loop
        node pairs for the zeroth node (0-1, 0-2, ..., 0-(n-1)). The
        next n-1 entries are associated with that for the first node
        (1-0, 1-2, 1-3, ..., 1-(n-1)), and so on.
    
    """
    d = attr_d_func(graph, attr, attr_d_dtype_int)
    if attr_d_dtype_int: d = d.astype(float)
    return np.reciprocal(d, where=d!=0.0)

def attr_avrg_e_func(graph: nx.Graph | nx.MultiGraph, attr: str) -> np.ndarray:
    """Edge-attributed average graph efficiency.

    This function calculates the average graph efficiency with respect
    to a specified edge attribute for each node in an (undirected)
    graph. This function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
        attr (str): Edge attribute with respect to which the average graph efficiency will be calculated.
    
    Returns:
        np.ndarray: Nodewise edge-attributed average graph efficiency.
    
    """
    if attr == "": d_dict = dict(nx.shortest_path_length(graph))
    else: d_dict = dict(nx.shortest_path_length(graph, weight=attr))
    node_list = list(graph.nodes())
    n = n_func(graph)
    avrg_e = np.empty(n)

    indx = 0
    for node_0 in node_list:
        avrg_e_sum = 0.
        for node_1 in node_list:
            d = d_dict[node_0][node_1] * 1.0
            avrg_e_sum += np.reciprocal(d, where=d!=0.0)
        avrg_e[indx] = avrg_e_sum / (n-1)
        indx += 1
    
    return avrg_e

def r_func(graph: nx.Graph | nx.MultiGraph) -> int:
    """Geodesic graph radius.

    This function calculates the geodesic radius (minimum eccentricity)
    of a given graph, where each edge has a unit weight. This function
    is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        int: Geodesic graph radius.
    
    """
    return nx.radius(graph)

def l_attr_r_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Euclidean edge length-weighted graph radius.

    This function calculates the Euclidean edge length-weighted radius
    (minimum eccentricity) of a given graph. This function is best
    applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        float: Euclidean edge length-weighted graph radius.
    
    """
    return nx.radius(graph, weight="l")

def l_inv_attr_r_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Inverse Euclidean edge length-weighted graph radius.

    This function calculates the inverse Euclidean edge length-weighted
    radius (minimum eccentricity) of a given graph. This function is
    best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        float: Inverse Euclidean edge length-weighted graph radius.
    
    """
    return nx.radius(graph, weight="l_inv")

def gamma_attr_r_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Chain/Edge stretch-weighted graph radius.

    This function calculates the chain/edge stretch-weighted radius
    (minimum eccentricity) of a given graph. This function is best
    applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        float: Chain/Edge stretch-weighted graph radius.
    
    """
    return nx.radius(graph, weight="gamma")

def gamma_inv_attr_r_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Inverse chain/edge stretch-weighted graph radius.

    This function calculates the inverse chain/edge stretch-weighted
    radius (minimum eccentricity) of a given graph. This function is
    best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        float: Inverse chain/edge stretch-weighted graph radius.
    
    """
    return nx.radius(graph, weight="gamma_inv")

def sigma_func(graph: nx.Graph | nx.MultiGraph) -> int:
    """Geodesic graph diameter.

    This function calculates the geodesic diameter (maximum
    eccentricity) of a given graph, where each edge has a unit weight.
    This function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        int: Geodesic graph diameter.
    
    """
    return nx.diameter(graph)

def l_attr_sigma_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Euclidean edge length-weighted graph diameter.

    This function calculates the Euclidean edge length-weighted diameter
    (maximum eccentricity) of a given graph. This function is best
    applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        float: Euclidean edge length-weighted graph diameter.
    
    """
    return nx.diameter(graph, weight="l")

def l_inv_attr_sigma_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Inverse Euclidean edge length-weighted graph diameter.

    This function calculates the inverse Euclidean edge length-weighted
    diameter (maximum eccentricity) of a given graph. This function is
    best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        float: Inverse Euclidean edge length-weighted graph diameter.
    
    """
    return nx.diameter(graph, weight="l_inv")

def gamma_attr_sigma_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Chain/Edge stretch-weighted graph diameter.

    This function calculates the chain/edge stretch-weighted diameter
    (maximum eccentricity) of a given graph. This function is best
    applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        float: Chain/Edge stretch-weighted graph diameter.
    
    """
    return nx.diameter(graph, weight="gamma")

def gamma_inv_attr_sigma_func(graph: nx.Graph | nx.MultiGraph) -> float:
    """Inverse chain/edge stretch-weighted graph diameter.

    This function calculates the inverse chain/edge stretch-weighted
    diameter (maximum eccentricity) of a given graph. This function is
    best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        float: Inverse chain/edge stretch-weighted graph diameter.
    
    """
    return nx.diameter(graph, weight="gamma_inv")

def epsilon_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Geodesic graph eccentricity.

    This function calculates the geodesic eccentricity (the maximum
    shortest path) for each node in an (undirected) graph, where each
    edge has a unit weight. This function is best applied to fully
    connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise geodesic graph eccentricity.
    
    """
    return np.asarray(list(nx.eccentricity(graph).values()), dtype=int)

def l_attr_epsilon_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Euclidean edge length-weighted graph eccentricity.

    This function calculates the Euclidean edge length-weighted
    eccentricity (the maximum shortest path) for each node in an
    (undirected) graph. This function is best applied to fully connected
    graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Euclidean edge length-weighted nodewise graph
        eccentricity.
    
    """
    return np.asarray(list(nx.eccentricity(graph, weight="l").values()))

def l_inv_attr_epsilon_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Inverse Euclidean edge length-weighted graph eccentricity.

    This function calculates the inverse Euclidean edge length-weighted
    eccentricity (the maximum shortest path) for each node in an
    (undirected) graph. This function is best applied to fully connected
    graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Inverse Euclidean edge length-weighted nodewise
        graph eccentricity.
    
    """
    return np.asarray(list(nx.eccentricity(graph, weight="l_inv").values()))

def gamma_attr_epsilon_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Chain/Edge stretch-weighted graph eccentricity.

    This function calculates the chain/edge stretch-weighted
    eccentricity (the maximum shortest path) for each node in an
    (undirected) graph. This function is best applied to fully connected
    graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Chain/Edge stretch-weighted nodewise graph
        eccentricity.
    
    """
    return np.asarray(list(nx.eccentricity(graph, weight="gamma").values()))

def gamma_inv_attr_epsilon_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Inverse chain/edge stretch-weighted graph eccentricity.

    This function calculates the inverse chain/edge stretch-weighted
    eccentricity (the maximum shortest path) for each node in an
    (undirected) graph. This function is best applied to fully connected
    graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Inverse chain/edge stretch-weighted nodewise graph
        eccentricity.
    
    """
    return np.asarray(list(nx.eccentricity(graph, weight="gamma_inv").values()))

def d_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Geodesic shortest path length.

    This function calculates the geodesic shortest path length for all
    pairs of nodes in an (undirected) graph (excluding all self-loop
    node pairs), where each edge has a unit weight. This function is
    best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Node-pairwise geodesic shortest path length,
        excluding all self-loop node pairs. Thus, for a network with n
        nodes labeled {0, 1, ..., n-1}, the first n-1 entries are
        associated with the shortest path length for all non-self-loop
        node pairs for the zeroth node (0-1, 0-2, ..., 0-(n-1)). The
        next n-1 entries are associated with that for the first node
        (1-0, 1-2, 1-3, ..., 1-(n-1)), and so on.
    
    """
    return attr_d_func(graph, "", True)

def l_attr_d_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Euclidean edge length-weighted shortest path length.

    This function calculates the Euclidean edge length-weighted shortest
    path length for all pairs of nodes in an (undirected) graph
    (excluding all self-loop node pairs). This function is best applied
    to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Node-pairwise Euclidean edge length-weighted
        shortest path length, excluding all self-loop node pairs. Thus,
        for a network with n nodes labeled {0, 1, ..., n-1}, the first
        n-1 entries are associated with the shortest path length for all
        non-self-loop node pairs for the zeroth node (0-1, 0-2, ...,
        0-(n-1)). The next n-1 entries are associated with that for the
        first node (1-0, 1-2, 1-3, ..., 1-(n-1)), and so on.
    
    """
    return attr_d_func(graph, "l", False)

def l_inv_attr_d_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Inverse Euclidean edge length-weighted shortest path length.

    This function calculates the inverse Euclidean edge length-weighted
    shortest path length for all pairs of nodes in an (undirected) graph
    (excluding all self-loop node pairs). This function is best applied
    to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Node-pairwise inverse Euclidean edge length-weighted
        shortest path length, excluding all self-loop node pairs. Thus,
        for a network with n nodes labeled {0, 1, ..., n-1}, the first
        n-1 entries are associated with the shortest path length for all
        non-self-loop node pairs for the zeroth node (0-1, 0-2, ...,
        0-(n-1)). The next n-1 entries are associated with that for the
        first node (1-0, 1-2, 1-3, ..., 1-(n-1)), and so on.
    
    """
    return attr_d_func(graph, "l_inv", False)

def gamma_attr_d_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Chain/Edge stretch-weighted shortest path length.

    This function calculates the chain/edge stretch-weighted shortest
    path length for all pairs of nodes in an (undirected) graph
    (excluding all self-loop node pairs). This function is best applied
    to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Node-pairwise chain/edge stretch-weighted shortest
        path length, excluding all self-loop node pairs. Thus, for a
        network with n nodes labeled {0, 1, ..., n-1}, the first n-1
        entries are associated with the shortest path length for all
        non-self-loop node pairs for the zeroth node
        (0-1, 0-2, ..., 0-(n-1)). The next n-1 entries are associated
        with that for the first node (1-0, 1-2, 1-3, ..., 1-(n-1)), and
        so on.
    
    """
    return attr_d_func(graph, "gamma", False)

def gamma_inv_attr_d_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Inverse chain/edge stretch-weighted shortest path length.

    This function calculates the inverse chain/edge stretch-weighted
    shortest path length for all pairs of nodes in an (undirected) graph
    (excluding all self-loop node pairs). This function is best applied
    to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Node-pairwise inverse chain/edge stretch-weighted
        shortest path length, excluding all self-loop node pairs. Thus,
        for a network with n nodes labeled {0, 1, ..., n-1}, the first
        n-1 entries are associated with the shortest path length for all
        non-self-loop node pairs for the zeroth node (0-1, 0-2, ...,
        0-(n-1)). The next n-1 entries are associated with that for the
        first node (1-0, 1-2, 1-3, ..., 1-(n-1)), and so on.
    
    """
    return attr_d_func(graph, "gamma_inv", False)

def avrg_d_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Average geodesic shortest path length.

    This function calculates the average geodesic shortest path length
    for each node in an (undirected) graph, where each edge has a unit
    weight. This function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise average geodesic shortest path length.
    
    """
    return attr_avrg_d_func(graph, "")

def avrg_l_attr_d_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Average Euclidean edge length-weighted shortest path length.

    This function calculates the average Euclidean edge length-weighted
    shortest path length for each node in an (undirected) graph. This
    function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise average Euclidean edge length-weighted
        shortest path length.
    
    """
    return attr_avrg_d_func(graph, "l")

def avrg_l_inv_attr_d_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Average inverse Euclidean edge length-weighted shortest path
    length.

    This function calculates the average inverse Euclidean edge
    length-weighted shortest path length for each node in an
    (undirected) graph. This function is best applied to fully connected
    graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise average inverse Euclidean edge
        length-weighted shortest path length.
    
    """
    return attr_avrg_d_func(graph, "l_inv")

def avrg_gamma_attr_d_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Chain/Edge stretch-weighted shortest path length.

    This function calculates the average chain/edge stretch-weighted
    shortest path length for each node in an (undirected) graph. This
    function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise average chain/edge stretch-weighted
        shortest path length.
    
    """
    return attr_avrg_d_func(graph, "gamma")

def avrg_gamma_inv_attr_d_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Average inverse chain/edge stretch-weighted shortest path
    length.

    This function calculates the average inverse chain/edge
    stretch-weighted shortest path length for each node in an
    (undirected) graph. This function is best applied to fully connected
    graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise average inverse chain/edge stretch-weighted
        shortest path length.
    
    """
    return attr_avrg_d_func(graph, "gamma_inv")

def e_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Geodesic graph efficiency.

    This function calculates the geodesic efficiency for all pairs of
    nodes in an (undirected) graph (excluding all self-loop node pairs),
    where each edge has a unit weight. This function is best applied to
    fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Node-pairwise geodesic graph efficiency, excluding
        all self-loop node pairs. Thus, for a network with n nodes
        labeled {0, 1, ..., n-1}, the first n-1 entries are associated
        with the efficiency for all non-self-loop node pairs for the
        zeroth node (0-1, 0-2, ..., 0-(n-1)). The next n-1 entries are
        associated with that for the first node (1-0, 1-2, 1-3, ...,
        1-(n-1)), and so on.
    
    """
    return attr_e_func(graph, "", True)

def l_attr_e_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Euclidean edge length-weighted graph efficiency.

    This function calculates the Euclidean edge length-weighted
    efficiency for all pairs of nodes in an (undirected) graph
    (excluding all self-loop node pairs). This function is best applied
    to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Node-pairwise Euclidean edge length-weighted
        efficiency, excluding all self-loop node pairs. Thus, for a
        network with n nodes labeled {0, 1, ..., n-1}, the first n-1
        entries are associated with the efficiency for all non-self-loop
        node pairs for the zeroth node (0-1, 0-2, ..., 0-(n-1)). The
        next n-1 entries are associated with that for the first node
        (1-0, 1-2, 1-3, ..., 1-(n-1)), and so on.
    
    """
    return attr_e_func(graph, "l", False)

def l_inv_attr_e_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Inverse Euclidean edge length-weighted graph efficiency.

    This function calculates the inverse Euclidean edge length-weighted
    efficiency for all pairs of nodes in an (undirected) graph
    (excluding all self-loop node pairs). This function is best applied
    to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Node-pairwise inverse Euclidean edge length-weighted
        efficiency, excluding all self-loop node pairs. Thus, for a
        network with n nodes labeled {0, 1, ..., n-1}, the first n-1
        entries are associated with the efficiency for all non-self-loop
        node pairs for the zeroth node (0-1, 0-2, ..., 0-(n-1)). The
        next n-1 entries are associated with that for the first node
        (1-0, 1-2, 1-3, ..., 1-(n-1)), and so on.
    
    """
    return attr_e_func(graph, "l_inv", False)

def gamma_attr_e_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Chain/Edge stretch-weighted graph efficiency.

    This function calculates the chain/edge stretch-weighted efficiency
    for all pairs of nodes in an (undirected) graph (excluding all
    self-loop node pairs). This function is best applied to fully
    connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Node-pairwise chain/edge stretch-weighted
        efficiency, excluding all self-loop node pairs. Thus, for a
        network with n nodes labeled {0, 1, ..., n-1}, the first n-1
        entries are associated with the efficiency for all non-self-loop
        node pairs for the zeroth node (0-1, 0-2, ..., 0-(n-1)). The
        next n-1 entries are associated with that for the first node
        (1-0, 1-2, 1-3, ..., 1-(n-1)), and so on.
    
    """
    return attr_e_func(graph, "gamma", False)

def gamma_inv_attr_e_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Inverse chain/edge stretch-weighted graph efficiency.

    This function calculates the inverse chain/edge stretch-weighted
    efficiency for all pairs of nodes in an (undirected) graph
    (excluding all self-loop node pairs). This function is best applied
    to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Node-pairwise inverse chain/edge stretch-weighted
        efficiency, excluding all self-loop node pairs. Thus, for a
        network with n nodes labeled {0, 1, ..., n-1}, the first n-1
        entries are associated with the efficiency for all non-self-loop
        node pairs for the zeroth node (0-1, 0-2, ..., 0-(n-1)). The
        next n-1 entries are associated with that for the first node
        (1-0, 1-2, 1-3, ..., 1-(n-1)), and so on.
    
    """
    return attr_e_func(graph, "gamma_inv", False)

def avrg_e_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Average geodesic graph efficiency.

    This function calculates the average geodesic efficiency for each
    node in an (undirected) graph, where each edge has a unit weight.
    This function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise average geodesic graph efficiency.
    
    """
    return attr_avrg_e_func(graph, "")

def avrg_l_attr_e_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Average Euclidean edge length-weighted graph efficiency.

    This function calculates the average Euclidean edge length-weighted
    efficiency for each node in an (undirected) graph, where each edge
    has a unit weight. This function is best applied to fully connected
    graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise average Euclidean edge length-weighted
        graph efficiency.
    
    """
    return attr_avrg_e_func(graph, "l")

def avrg_l_inv_attr_e_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Average inverse Euclidean edge length-weighted graph efficiency.

    This function calculates the average inverse Euclidean edge
    length-weighted efficiency for each node in an (undirected) graph,
    where each edge has a unit weight. This function is best applied to
    fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise average inverse Euclidean edge
        length-weighted graph efficiency.
    
    """
    return attr_avrg_e_func(graph, "l_inv")

def avrg_gamma_attr_e_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Average chain/edge stretch-weighted graph efficiency.

    This function calculates the average chain/edge stretch-weighted
    efficiency for each node in an (undirected) graph, where each edge
    has a unit weight. This function is best applied to fully connected
    graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise average chain/edge stretch-weighted graph
        efficiency.
    
    """
    return attr_avrg_e_func(graph, "gamma")

def avrg_gamma_inv_attr_e_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Average inverse chain/edge stretch-weighted graph efficiency.

    This function calculates the average inverse chain/edge
    stretch-weighted efficiency for each node in an (undirected) graph,
    where each edge has a unit weight. This function is best applied to
    fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise average inverse chain/edge stretch-weighted
        graph efficiency.
    
    """
    return attr_avrg_e_func(graph, "gamma_inv")

def bc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Geodesic node betweenness centrality.

    This function calculates the geodesic betweenness centrality for
    each node in an (undirected) graph, where each edge has a unit
    weight. This function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise geodesic betweenness centrality.
    
    """
    return np.asarray(list(nx.betweenness_centrality(graph).values()))

def l_attr_bc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Euclidean edge length-weighted node betweenness centrality.

    This function calculates the Euclidean edge length-weighted
    betweenness centrality for each node in an (undirected) graph. This
    function is best applied to fully connected graphs.

    Args:
        
    
    Returns:
        np.ndarray: Nodewise Euclidean edge length-weighted betweenness
        centrality.
    
    """
    return (
        np.asarray(list(nx.betweenness_centrality(graph, weight="l").values()))
    )

def l_inv_attr_bc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Inverse Euclidean edge length-weighted node betweenness
    centrality.

    This function calculates the inverse Euclidean edge length-weighted
    betweenness centrality for each node in an (undirected) graph. This
    function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise inverse Euclidean edge length-weighted
        betweenness centrality.
    
    """
    return (
        np.asarray(
            list(nx.betweenness_centrality(graph, weight="l_inv").values()))
    )

def gamma_attr_bc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Chain/Edge stretch-weighted node betweenness centrality.

    This function calculates the chain/edge stretch-weighted
    betweenness centrality for each node in an (undirected) graph. This
    function is best applied to fully connected graphs.

    Args:
        
    
    Returns:
        np.ndarray: Nodewise chain/edge stretch-weighted betweenness
        centrality.
    
    """
    return (
        np.asarray(list(nx.betweenness_centrality(graph, weight="gamma").values()))
    )

def gamma_inv_attr_bc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Inverse chain/edge stretch-weighted node betweenness
    centrality.

    This function calculates the inverse chain/edge stretch-weighted
    betweenness centrality for each node in an (undirected) graph. This
    function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise inverse chain/edge stretch-weighted
        betweenness centrality.
    
    """
    return (
        np.asarray(
            list(nx.betweenness_centrality(graph, weight="gamma_inv").values()))
    )

def ebc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Geodesic edge betweenness centrality.

    This function calculates the geodesic edge betweenness
    centrality for each edge in an (undirected) graph, where each edge
    has a unit weight. This function is best applied to fully connected
    graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Geodesic edgewise edge betweenness centrality.
    
    """
    return np.asarray(list(nx.edge_betweenness_centrality(graph).values()))

def l_attr_ebc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Euclidean edge length-weighted edge betweenness centrality.

    This function calculates the Euclidean edge length-weighted edge
    betweenness centrality for each edge in an (undirected) graph. This
    function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Edgewise Euclidean edge length-weighted edge
        betweenness centrality.
    
    """
    return (
        np.asarray(
            list(nx.edge_betweenness_centrality(graph, weight="l").values()))
    )

def l_inv_attr_ebc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Inverse Euclidean edge length-weighted edge betweenness
    centrality.

    This function calculates the inverse Euclidean edge length-weighted
    edge betweenness centrality for each edge in an (undirected) graph.
    This function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Edgewise inverse Euclidean edge length-weighted edge
        betweenness centrality.
    
    """
    return (
        np.asarray(
            list(nx.edge_betweenness_centrality(graph, weight="l_inv").values()))
    )

def gamma_attr_ebc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Chain/Edge stretch-weighted edge betweenness centrality.

    This function calculates the chain/edge stretch-weighted edge
    betweenness centrality for each edge in an (undirected) graph. This
    function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Edgewise chain/edge stretch-weighted edge
        betweenness centrality.
    
    """
    return (
        np.asarray(
            list(nx.edge_betweenness_centrality(graph, weight="gamma").values()))
    )

def gamma_inv_attr_ebc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Inverse chain/edge stretch-weighted edge betweenness
    centrality.

    This function calculates the inverse chain/edge stretch-weighted
    edge betweenness centrality for each edge in an (undirected) graph.
    This function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Edgewise inverse chain/edge stretch-weighted edge
        betweenness centrality.
    
    """
    return (
        np.asarray(
            list(nx.edge_betweenness_centrality(graph, weight="gamma_inv").values()))
    )

def cc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Geodesic closeness centrality.

    This function calculates the geodesic closeness centrality for each
    node in an (undirected) graph, where each edge has a unit weight.
    This function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise geodesic closeness centrality.
    
    """
    return np.asarray(list(nx.closeness_centrality(graph).values()))

def l_attr_cc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Euclidean edge length-weighted closeness centrality.

    This function calculates the Euclidean edge length-weighted
    closeness centrality for each node in an (undirected) graph. This
    function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise Euclidean edge length-weighted closeness
        centrality.
    
    """
    return (
        np.asarray(list(nx.closeness_centrality(graph, distance="l").values()))
    )

def l_inv_attr_cc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Inverse Euclidean edge length-weighted closeness centrality.

    This function calculates the inverse Euclidean edge length-weighted
    closeness centrality for each node in an (undirected) graph. This
    function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise inverse Euclidean edge length-weighted
        closeness centrality.
    
    """
    return (
        np.asarray(
            list(nx.closeness_centrality(graph, distance="l_inv").values()))
    )

def gamma_attr_cc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Chain/Edge stretch-weighted closeness centrality.

    This function calculates the chain/edge stretch-weighted closeness
    centrality for each node in an (undirected) graph. This function is
    best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise chain/edge stretch-weighted closeness
        centrality.
    
    """
    return (
        np.asarray(list(nx.closeness_centrality(graph, distance="gamma").values()))
    )

def gamma_inv_attr_cc_func(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Inverse chain/edge stretch-weighted closeness centrality.

    This function calculates the inverse chain/edge stretch-weighted
    closeness centrality for each node in an (undirected) graph. This
    function is best applied to fully connected graphs.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph. Ideally, this graph ought to be fully connected as-is.
    
    Returns:
        np.ndarray: Nodewise inverse chain/edge stretch-weighted
        closeness centrality.
    
    """
    return (
        np.asarray(
            list(nx.closeness_centrality(graph, distance="gamma_inv").values()))
    )