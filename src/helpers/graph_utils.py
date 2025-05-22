import numpy as np
import networkx as nx

def lexsorted_edges(
        edges: list[tuple[int, int]] | np.ndarray,
        return_indcs: bool=False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Lexicographically sorted edges.

    This function takes a list (or an np.ndarray) of (A, B) nodes
    specifying edges, converts this to an np.ndarray, and
    lexicographically sorts the edge entries.

    Args:
        edges (list[tuple[int, int]] | np.ndarray): Edges.
        return_indcs (bool): Flag indicating if the indices of the lexicographic edge sort should be returned, or not. Default is False.
    
    Returns:
        np.ndarray | tuple[np.ndarray, np.ndarray]: Lexicographically
        sorted edges, or the lexicographically sorted edges and the
        indices of the lexicographic edge sort.
    
    """
    edges = np.sort(np.asarray(edges, dtype=int), axis=1)
    lexsort_indcs = np.lexsort((edges[:, 1], edges[:, 0]))
    if return_indcs: return edges[lexsort_indcs], lexsort_indcs
    else: return edges[lexsort_indcs]

def unique_lexsorted_edges(edges: list[tuple[int, int]]) -> np.ndarray:
    """Unique lexicographically sorted edges.

    This function takes a list (or an np.ndarray) of (A, B) nodes
    specifying edges, converts this to an np.ndarray, lexicographically
    sorts the edge entries, and extracts the resulting unique edges.

    Args:
        edges (list[tuple[int, int]] | np.ndarray): Edges.
    
    Returns:
        np.ndarray: Unique lexicographically sorted edges.
    
    """
    return np.unique(lexsorted_edges(edges), axis=0)

def add_nodes_from_numpy_array(
        graph: nx.Graph | nx.MultiGraph,
        nodes: np.ndarray) -> nx.Graph | nx.MultiGraph:
    """Add node numbers from a np.ndarray array to an undirected
    NetworkX graph.

    This function adds node numbers from a np.ndarray array to an
    undirected NetworkX graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
        nodes (np.ndarray): Node numbers.
    
    Returns:
        nx.Graph | nx.MultiGraph: (Undirected) NetworkX graph.
    
    """
    graph.add_nodes_from(nodes.tolist())
    return graph

def add_edges_from_numpy_array(
        graph: nx.Graph | nx.MultiGraph,
        edges: np.ndarray) -> nx.Graph | nx.MultiGraph:
    """Add edges from a two-dimensional np.ndarray to an undirected
    NetworkX graph.

    This function adds edges from a two-dimensional np.ndarray to an
    undirected NetworkX graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
        edges (np.ndarray): Edges.
    
    Returns:
        nx.Graph | nx.MultiGraph: (Undirected) NetworkX graph.
    
    """
    graph.add_edges_from(list(tuple(edge) for edge in edges.tolist()))
    return graph

def add_nodes_and_node_attributes_from_numpy_arrays(
        graph: nx.Graph | nx.MultiGraph,
        nodes: np.ndarray,
        attr_names: list[str],
        *attr_vals: tuple[np.ndarray]) -> nx.Graph | nx.MultiGraph:
    """Add node numbers and node attributes from np.ndarray arrays to an
    undirected NetworkX graph.

    This function adds node numbers and associated node attributes from
    np.ndarray arrays to an undirected NetworkX graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
        nodes (np.ndarray): Node numbers.
        attr_names (list[str]): Node attribute names.
        *attr_vals (tuple[np.ndarray]): Node attribute values.
    
    Returns:
        nx.Graph | nx.MultiGraph: (Undirected) NetworkX graph.
    
    """
    # Ensure that the number of node attributes matches the number of
    # provided node attribute arrays
    if len(attr_names) != len(attr_vals):
        error_str = (
            "The number of node attribute names must match the number "
            + "of node attribute value arrays."
        )
        raise ValueError(error_str)
    # Ensure that each node value attribute array has an attribute
    # specified for each node
    if any(np.shape(attr)[0] != np.shape(nodes)[0] for attr in attr_vals):
        error_str = (
            "Each node attribute value array must have the same number "
            + "of entries as the number of nodes."
        )
        raise ValueError(error_str)
    
    # Convert np.ndarrays to lists
    nodes = nodes.tolist()
    attr_vals = tuple(attr.tolist() for attr in attr_vals)

    # Add nodes and node attributes to graph
    graph.add_nodes_from(
        (node, {attr_names[attr_indx]: attr_vals[attr_indx][node_indx] for attr_indx in range(len(attr_names))})
        for node_indx, node in enumerate(nodes)
    )
    return graph

def add_edges_and_edge_attributes_from_numpy_arrays(
        graph: nx.Graph | nx.MultiGraph,
        edges: np.ndarray,
        attr_names: list[str],
        *attr_vals: tuple[np.ndarray]) -> nx.Graph | nx.MultiGraph:
    """Add edges and edge attributes from np.ndarray arrays to an
    undirected NetworkX graph.

    This function adds edges and associated edge attributes from
    np.ndarray arrays to an undirected NetworkX graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
        edges (np.ndarray): Edges.
        attr_names (list[str]): Edge attribute names.
        *attr_vals (tuple[np.ndarray]): Edge attribute values.
    
    Returns:
        nx.Graph | nx.MultiGraph: (Undirected) NetworkX graph.
    
    """
    # Ensure that the number of edge attributes matches the number of
    # provided edge attribute arrays
    if len(attr_names) != len(attr_vals):
        error_str = (
            "The number of edge attribute names must match the number "
            + "of edge attribute value arrays."
        )
        raise ValueError(error_str)
    # Ensure that each edge value attribute array has an attribute
    # specified for each edge
    if any(np.shape(attr)[0] != np.shape(edges)[0] for attr in attr_vals):
        error_str = (
            "Each edge attribute value array must have the same number "
            + "of entries as the number of edges."
        )
        raise ValueError(error_str)
    
    # Convert np.ndarrays to lists
    edges = list(tuple(edge) for edge in edges.tolist())
    attr_vals = tuple(attr.tolist() for attr in attr_vals)

    # Add edges and edge attributes to graph
    graph.add_edges_from(
        (edge[0], edge[1], {attr_names[attr_indx]: attr_vals[attr_indx][edge_indx] for attr_indx in range(len(attr_names))})
        for edge_indx, edge in enumerate(edges)
    )
    return graph

def extract_nodes_to_numpy_array(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Extract nodes from an undirected NetworkX graph to a np.ndarray
    array.

    This function extract nodes from an undirected NetworkX graph to a
    np.ndarray array.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        np.ndarray: Node numbers.
    
    """
    return np.asarray(list(graph.nodes()), dtype=int)

def extract_edges_to_numpy_array(graph: nx.Graph | nx.MultiGraph) -> np.ndarray:
    """Extract edges from an undirected NetworkX graph to a np.ndarray
    array.

    This function extract edges from an undirected NetworkX graph to a
    np.ndarray array.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        np.ndarray: Edges.
    
    """
    return np.asarray(list(graph.edges()), dtype=int)

def extract_node_attribute_to_numpy_array(
        graph: nx.Graph | nx.MultiGraph,
        attr_name: str,
        dtype_int: bool=False) -> np.ndarray:
    """Extract node attribute from an undirected NetworkX graph.

    This function extracts a node attribute from an undirected NetworkX
    graph to a np.ndarray array.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
        attr_name (str): Node attribute name.
        dtype_int (bool): Boolean indicating if the node attribute is of type int. Default value is False.
    
    Returns:
        np.ndarray: Node attribute.
    
    """
    if dtype_int:
        return np.asarray(
            list(dict(nx.get_node_attributes(graph, attr_name)).values()),
            dtype=int)
    else:
        return np.asarray(
            list(dict(nx.get_node_attributes(graph, attr_name)).values()))

def extract_edge_attribute_to_numpy_array(
        graph: nx.Graph | nx.MultiGraph,
        attr_name: str,
        dtype_int: bool=False) -> np.ndarray:
    """Extract edge attribute from an undirected NetworkX graph.

    This function extracts an edge attribute from an undirected NetworkX
    graph to a np.ndarray array.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
        attr_name (str): Edge attribute name.
        dtype_int (bool): Boolean indicating if the edge attribute is of type int. Default value is False.
    
    Returns:
        np.ndarray: Edge attribute.
    
    """
    if dtype_int:
        return np.asarray(
            list(dict(nx.get_edge_attributes(graph, attr_name)).values()),
            dtype=int)
    else:
        return np.asarray(
            list(dict(nx.get_edge_attributes(graph, attr_name)).values()))

def largest_connected_component(
        graph: nx.Graph | nx.MultiGraph) -> nx.Graph | nx.MultiGraph:
    """Isolate and return the largest/maximum connected component.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        nx.Graph | nx.MultiGraph: (Undirected) NetworkX graph that is
        the largest/maximum connected component of the input graph.

    """
    return graph.subgraph(max(nx.connected_components(graph), key=len)).copy()

def edge_id(graph: nx.Graph | nx.MultiGraph) -> tuple[np.ndarray, np.ndarray]:
    """Edge identification.
    
    This function identifies and counts the edges in a graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Edges and number of times each
        edge occurs in the graph.
    
    """
    # Gather edges and edges counts
    graph_edges = np.asarray(list(graph.edges()), dtype=int)
    if graph.is_multigraph():
        graph_edges, graph_edges_counts = np.unique(
            np.sort(graph_edges, axis=1), return_counts=True, axis=0)
    else:
        graph_edges_counts = np.ones(np.shape(graph_edges)[0], dtype=int)
    
    return graph_edges, graph_edges_counts

def yasuda_morita_procedure(
        A: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, int]], list[tuple[int, int]], list[int]]:
    """Yasuda-Morita procedure.
    
    This function applies the Yasuda-Morita procedure to yield the
    elastically-effective network satisfying the Scanlan-Case criteria.

    Args:
        A (np.ndarray): Adjacency matrix (with no multiedges).
    
    Returns:
        tuple[np.ndarray, list[tuple[int, int, int]], list[tuple[int, int]], list[int]]:
        Adjacency matrix of the elastically-effective network (with no
        multiedges), list of nodes involved with removed bridge centers,
        list of nodes involved with removed dangling edges, and list of
        nodes involved with removed self-loops.
    """
    # Gather nodes
    n = np.shape(A)[0]
    nodes = np.arange(n, dtype=int)

    # Initialize lists
    bridge_center_node_list = []
    dangling_node_list = []
    loop_list = []

    # Yasuda-Morita procedure
    while True:
        # Initialize trackers
        bridge_center_node_elim = False
        dangling_node_elim = False
        loop_elim = False
        
        # Bridge center node elimination
        for center_node in range(n):
            center_node = int(center_node)
            # Edges excluding self-loops
            A_row = np.delete(A[center_node, :], center_node, axis=0)
            A_row_nodes = np.delete(nodes, center_node, axis=0)
            # Check if node is a bridge center node
            if np.sum(A_row) == 2:
                bridge_center_node_elim = True
                # Check if the bridge bridges the same two nodes, and
                # thus is actually a bridging loop
                if np.size(np.where(A_row == 2)[0]) > 0:
                    root_node = int(A_row_nodes[np.where(A_row == 2)[0][0]])
                    # Eliminate the bridge center node from the network
                    A[root_node, center_node] = 0
                    A[center_node, root_node] = 0
                    # Add loop to root node
                    A[root_node, root_node] += 2
                    # Add to bridge center node list
                    bridge_center_node_list.append(
                        (root_node, center_node, root_node))
                # Otherwise, the bridge bridges two distinct nodes
                else:
                    # Identify all nodes involved in the bridge
                    bridge_nodes = A_row_nodes[np.where(A_row == 1)[0]]
                    left_node = int(bridge_nodes[0])
                    right_node = int(bridge_nodes[1])
                    # Eliminate the bridge center node from the network
                    A[left_node, center_node] = 0
                    A[center_node, right_node] = 0
                    A[right_node, center_node] = 0
                    A[center_node, left_node] = 0
                    # Ensure bridge remains intact in the network
                    A[left_node, right_node] += 1
                    A[right_node, left_node] += 1
                    # Add to bridge center node list
                    bridge_center_node_list.append(
                        (left_node, center_node, right_node))
                break
        if bridge_center_node_elim == True:
            continue
        else:
            
            # Dangling node elimination
            for dangling_node in range(n):
                dangling_node = int(dangling_node)
                # Edges excluding self-loops
                A_row = np.delete(A[dangling_node, :], dangling_node, axis=0)
                A_row_nodes = np.delete(nodes, dangling_node, axis=0)
                # Check if node is a dangling node
                if np.sum(A_row) == 1:
                    dangling_node_elim = True
                    # Identify the root node for the dangling node
                    root_node = int(A_row_nodes[np.where(A_row == 1)[0][0]])
                    # Eliminate the dangling node from the network
                    A[root_node, dangling_node] = 0
                    A[dangling_node, root_node] = 0
                    # Add to dangling node list
                    dangling_node_list.append((root_node, dangling_node))
                    break
            if dangling_node_elim == True:
                continue
            else:
                
                # Loop elimination
                for node in range(n):
                    node = int(node)
                    if A[node, node] >= 2:
                        loop_elim = True
                        # Eliminate the loop from the network
                        A[node, node] -= 2
                        # Add to loop list
                        loop_list.append(node)
                        break
                if loop_elim == True:
                    continue
                else: break # Yasuda-Morita procedure has finished
    
    return A, bridge_center_node_list, dangling_node_list, loop_list

def surviving_bridge_restoration(
        A: np.ndarray,
        bridge_center_node_list: list[tuple[int, int, int]]) -> np.ndarray:
    """Surviving bridge restoration.

    This function adds back bridges that were once removed to yield the
    most fundamental elastically-effective network but yet still exist
    between surviving nodes in end-linked networks.

    Args:
        A (np.ndarray): Adjacency matrix (with no multiedges).
        bridge_center_node_list (list[tuple[int, int, int]]): List of nodes involved with bridge centers that were removed to yield the most fundamental elastically-effective network.
    
    Returns:
        np.ndarray: Adjacency matrix of the elastically-effective
        network (with no multiedges) containing bridge centers in
        between surviving nodes.
    
    """
    # Add back bridging centers between surviving nodes in reverse
    # elimination order
    bridge_center_node_list.reverse()

    for bridge_nodes in bridge_center_node_list:
        left_node = int(bridge_nodes[0])
        center_node = int(bridge_nodes[1])
        right_node = int(bridge_nodes[2])

        # Ignore bridging loops
        if left_node == right_node: pass
        else:
            # Add back bridging center node between surviving nodes
            if np.sum(A[left_node, :]) > 0 and np.sum(A[right_node, :]) > 0:
                A[left_node, right_node] -= 1
                A[right_node, left_node] -= 1
                A[left_node, center_node] += 1
                A[center_node, right_node] += 1
                A[right_node, center_node] += 1
                A[center_node, left_node] += 1
            # Ignore bridges that bridge eliminated nodes
            else: pass
    
    return A

def multiedge_restoration(
        graph: nx.Graph | nx.MultiGraph,
        graph_edges: np.ndarray,
        graph_edges_counts: np.ndarray) -> nx.Graph | nx.MultiGraph:
    """Multiedge restoration.
    
    This function restores multiedges in a graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
        np.ndarray: Edges.
        np.ndarray: Number of (multi)edges involved for each edge.
    
    Returns:
        nx.Graph | nx.MultiGraph: (Undirected) NetworkX graph.
    
    """
    # If the graph is of type nx.MultiGraph, then add back redundant
    # edges to all multiedges.
    if graph.is_multigraph():
        # Address multiedges by adding back redundant edges
        if np.any(graph_edges_counts > 1):
            # Extract multiedges
            multiedges = np.where(graph_edges_counts > 1)[0]
            for multiedge in np.nditer(multiedges):
                multiedge = int(multiedge)
                # Number of edges in the multiedge
                edge_num = graph_edges_counts[multiedge]
                # Multiedge nodes
                node_0 = int(graph_edges[multiedge, 0])
                node_1 = int(graph_edges[multiedge, 1])
                # Add back redundant edges
                if graph.has_edge(node_0, node_1):
                    graph.add_edges_from(
                        list((node_0, node_1) for _ in range(edge_num-1)))
    
    return graph

def elastically_effective_graph(
        graph: nx.Graph | nx.MultiGraph) -> nx.Graph | nx.MultiGraph:
    """Elastically-effective graph.

    This function returns the portion of a given graph that corresponds
    to the elastically-effective network in the graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        nx.Graph | nx.MultiGraph: (Undirected) NetworkX graph
        corresponding to the elastically-effective network of the given
        graph.
    
    """
    # Gather edges and edge counts
    graph_edges, graph_edges_counts = edge_id(graph)

    # Acquire adjacency matrix with no multiedges
    if graph.is_multigraph():
        A = nx.to_numpy_array(nx.Graph(graph), dtype=int)
    else:
        A = nx.to_numpy_array(graph, dtype=int)
    
    # Apply the Yasuda-Morita procedure to return the
    # elastically-effective network that satisfies the Scanlan-Case
    # criteria.
    A, _, _, _ = yasuda_morita_procedure(A)

    # Acquire NetworkX graph
    if graph.is_multigraph():
        graph = nx.from_numpy_array(A, create_using=nx.MultiGraph)
    else:
        graph = nx.from_numpy_array(A)

    # Restore multiedges
    graph = multiedge_restoration(graph, graph_edges, graph_edges_counts)

    # As a hard fail-safe, isolate and return the largest/maximum
    # connected component
    return largest_connected_component(graph)

def elastically_effective_end_linked_graph(
        graph: nx.Graph | nx.MultiGraph) -> nx.Graph | nx.MultiGraph:
    """Elastically-effective end-linked graph.

    This function returns the portion of a given graph that corresponds
    to the elastically-effective end-linked network in the graph.

    Args:
        graph (nx.Graph | nx.MultiGraph): (Undirected) NetworkX graph.
    
    Returns:
        nx.Graph | nx.MultiGraph: (Undirected) NetworkX graph
        corresponding to the elastically-effective end-linked network of
        the given graph.
    
    """
    # Multiedge identification
    graph_edges, graph_edges_counts = edge_id(graph)

    # Acquire adjacency matrix with no multiedges
    if graph.is_multigraph():
        A = nx.to_numpy_array(nx.Graph(graph), dtype=int)
    else:
        A = nx.to_numpy_array(graph, dtype=int)
    
    # Apply the Yasuda-Morita procedure to return the
    # elastically-effective network that satisfies the Scanlan-Case
    # criteria.
    A, bridge_center_node_list, _, _ = yasuda_morita_procedure(A)

    # Add back bridging centers between surviving nodes in reverse
    # elimination order
    A = surviving_bridge_restoration(A, bridge_center_node_list)

    # Acquire NetworkX graph
    if graph.is_multigraph():
        graph = nx.from_numpy_array(A, create_using=nx.MultiGraph)
    else:
        graph = nx.from_numpy_array(A)

    # Restore multiedges
    graph = multiedge_restoration(graph, graph_edges, graph_edges_counts)

    # As a hard fail-safe, isolate and return the largest/maximum
    # connected component
    return largest_connected_component(graph)