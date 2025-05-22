from src.networks.voronoi_networks import (
    voronoi_L,
    voronoi_network_topology,
    voronoi_network_local_topological_descriptor,
    voronoi_network_global_topological_descriptor,
    voronoi_network_global_morphological_descriptor
)
from src.networks.delaunay_networks import (
    delaunay_L,
    delaunay_network_topology,
    delaunay_network_local_topological_descriptor,
    delaunay_network_global_topological_descriptor,
    delaunay_network_global_morphological_descriptor
)
from src.networks.swidt_networks import (
    swidt_L,
    swidt_network_topology,
    swidt_network_edge_pruning_procedure,
    swidt_network_local_topological_descriptor,
    swidt_network_global_topological_descriptor,
    swidt_network_global_morphological_descriptor
)
from src.helpers.node_placement_utils import initial_node_seeding

def run_voronoi_L(args):
    voronoi_L(*args)

def run_delaunay_L(args):
    delaunay_L(*args)

def run_swidt_L(args):
    swidt_L(*args)

def run_initial_node_seeding(args):
    initial_node_seeding(*args)

def run_voronoi_network_topology(args):
    voronoi_network_topology(*args)

def run_delaunay_network_topology(args):
    delaunay_network_topology(*args)

def run_swidt_network_topology(args):
    swidt_network_topology(*args)

def run_swidt_network_edge_pruning_procedure(args):
    swidt_network_edge_pruning_procedure(*args)

def run_voronoi_network_local_topological_descriptor(args):
    voronoi_network_local_topological_descriptor(*args)

def run_delaunay_network_local_topological_descriptor(args):
    delaunay_network_local_topological_descriptor(*args)

def run_swidt_network_local_topological_descriptor(args):
    swidt_network_local_topological_descriptor(*args)

def run_voronoi_network_global_topological_descriptor(args):
    voronoi_network_global_topological_descriptor(*args)

def run_delaunay_network_global_topological_descriptor(args):
    delaunay_network_global_topological_descriptor(*args)

def run_swidt_network_global_topological_descriptor(args):
    swidt_network_global_topological_descriptor(*args)

def run_voronoi_network_global_morphological_descriptor(args):
    voronoi_network_global_morphological_descriptor(*args)

def run_delaunay_network_global_morphological_descriptor(args):
    delaunay_network_global_morphological_descriptor(*args)

def run_swidt_network_global_morphological_descriptor(args):
    swidt_network_global_morphological_descriptor(*args)