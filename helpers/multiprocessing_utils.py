from networks.voronoi_networks import (
    voronoi_L,
    voronoi_network_topology,
    voronoi_network_local_topological_descriptor,
    voronoi_network_global_topological_descriptor,
    voronoi_network_global_morphological_descriptor
)
from networks.delaunay_networks import (
    delaunay_L,
    delaunay_network_topology,
    delaunay_network_local_topological_descriptor,
    delaunay_network_global_topological_descriptor,
    delaunay_network_global_morphological_descriptor
)
from networks.swidt_networks import (
    swidt_L,
    swidt_network_topology,
    swidt_network_edge_pruning_procedure,
    swidt_network_local_topological_descriptor,
    swidt_network_global_topological_descriptor,
    swidt_network_global_morphological_descriptor
)
from networks.aelp_networks import (
    aelp_L,
    aelp_network_local_topological_descriptor,
    aelp_network_global_topological_descriptor,
    aelp_network_global_morphological_descriptor
)
from networks.auelp_networks import auelp_network_topology
from networks.abelp_networks import abelp_network_topology
from networks.apelp_networks import apelp_network_topology
from helpers.node_placement import initial_node_seeding

def run_voronoi_L(args):
    voronoi_L(*args)

def run_delaunay_L(args):
    delaunay_L(*args)

def run_swidt_L(args):
    swidt_L(*args)

def run_aelp_L(args):
    aelp_L(*args)

def run_initial_node_seeding(args):
    initial_node_seeding(*args)

def run_voronoi_network_topology(args):
    voronoi_network_topology(*args)

def run_delaunay_network_topology(args):
    delaunay_network_topology(*args)

def run_swidt_network_topology(args):
    swidt_network_topology(*args)

def run_auelp_network_topology(args):
    auelp_network_topology(*args)

def run_abelp_network_topology(args):
    abelp_network_topology(*args)

def run_apelp_network_topology(args):
    apelp_network_topology(*args)

def run_swidt_network_edge_pruning_procedure(args):
    swidt_network_edge_pruning_procedure(*args)

def run_voronoi_network_local_topological_descriptor(args):
    voronoi_network_local_topological_descriptor(*args)

def run_delaunay_network_local_topological_descriptor(args):
    delaunay_network_local_topological_descriptor(*args)

def run_swidt_network_local_topological_descriptor(args):
    swidt_network_local_topological_descriptor(*args)

def run_aelp_network_local_topological_descriptor(args):
    aelp_network_local_topological_descriptor(*args)

def run_voronoi_network_global_topological_descriptor(args):
    voronoi_network_global_topological_descriptor(*args)

def run_delaunay_network_global_topological_descriptor(args):
    delaunay_network_global_topological_descriptor(*args)

def run_swidt_network_global_topological_descriptor(args):
    swidt_network_global_topological_descriptor(*args)

def run_aelp_network_global_topological_descriptor(args):
    aelp_network_global_topological_descriptor(*args)

def run_voronoi_network_global_morphological_descriptor(args):
    voronoi_network_global_morphological_descriptor(*args)

def run_delaunay_network_global_morphological_descriptor(args):
    delaunay_network_global_morphological_descriptor(*args)

def run_swidt_network_global_morphological_descriptor(args):
    swidt_network_global_morphological_descriptor(*args)

def run_aelp_network_global_morphological_descriptor(args):
    aelp_network_global_morphological_descriptor(*args)