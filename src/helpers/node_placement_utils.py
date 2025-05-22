import numpy as np
from src.helpers.network_topology_initialization_utils import (
    tessellation_protocol,
    tessellation,
    orb_neighborhood_id
)
from src.file_io.file_io import (
    L_filename_str,
    config_filename_str
)

def initial_random_node_placement(
        dim: int,
        L: np.ndarray,
        n: int,
        config_filename_prefix: str) -> None:
    """Initial random node placement procedure.

    This function randomly places/seeds nodes within an empty simulation
    box.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        L (np.ndarray): Simulation box side lengths.
        n (int): Number of nodes.
        config_filename_prefix (str): Configuration filename prefix.
    
    """
    # Coordinates filename
    coords_filename = config_filename_prefix + ".coords"

    # Initialize random number generator
    rng = np.random.default_rng()

    # Save random node coordinates
    np.savetxt(coords_filename, L*rng.random((n, dim)))

def periodic_random_hard_disk_node_placement(
        dim: int,
        L: np.ndarray,
        b: float,
        max_try: int,
        rng: np.random.Generator,
        tsslltn: np.ndarray,
        coords: np.ndarray,
        tsslltd_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Periodic random hard disk node placement procedure.

    This function randomly places/seeds nodes within a simulation box
    where each node is treated as the center of a hard disk, and the
    simulation box is periodic.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        L (np.ndarray): Simulation box side lengths.
        b (float): Hard disk diameter.
        max_try (int): Maximum number of node placement attempts.
        rng (np.random.Generator): np.random.Generator object.
        tsslltn (np.ndarray): Tessellation protocol.
        coords (np.ndarray): Node coordinates.
        tsslltd_coords (np.ndarray): Tessellated node coordinates.
    
    Return:
        tuple[np.ndarray, np.ndarray]: Node coordinates and tessellated
        node coordinates.
    
    """
    # Begin periodic random hard disk node placement procedure
    num_try = 0
    
    while num_try < max_try:
        # Generate randomly placed node candidate
        seed_cnddt = L * rng.random(size=dim)

        # Downselect the previously-accepted tessellated nodes to those
        # that reside in a local orb neighborhood with radius b about
        # the node candidate
        _, orb_nghbr_num = orb_neighborhood_id(
            dim, tsslltd_coords, seed_cnddt, b, inclusive=False, indices=True)
        
        # Try again if the local orb neighborhood has at least one
        # neighbor in it
        if orb_nghbr_num > 0:
            num_try += 1
            continue
        
        # Accept and tessellate the node candidate if no local orb
        # neighborhood of tessellated nodes exists about the node
        # candidate
        coords = np.vstack((coords, seed_cnddt))
        tsslltd_coords = np.vstack(
            (tsslltd_coords, tessellation(seed_cnddt, tsslltn, L)))
        break

    return coords, tsslltd_coords

def initial_periodic_random_hard_disk_node_placement(
        dim: int,
        L: np.ndarray,
        b: float,
        n: int,
        max_try: int,
        config_filename_prefix: str) -> None:
    """Initial periodic random hard disk node placement procedure.

    This function randomly places/seeds nodes within an empty simulation
    box where each node is treated as the center of a hard disk, and the
    simulation box is periodic.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        L (np.ndarray): Simulation box side lengths.
        b (float): Hard disk diameter.
        n (int): Intended number of nodes.
        max_try (int): Maximum number of node placement attempts.
        config_filename_prefix (str): Configuration filename prefix.
    
    """
    # Coordinates filename
    coords_filename = config_filename_prefix + ".coords"

    # Initialize random number generator
    rng = np.random.default_rng()

    # Tessellation protocol
    tsslltn, _ = tessellation_protocol(dim)

    # Periodic random hard disk node placement procedure
    for seed_attmpt in range(n):
        # Accept and tessellate the first node
        if seed_attmpt == 0:
            # Accept the first node
            seed = L * rng.random(size=dim)
            coords = seed.copy()

            # Tessellate the first node
            seed_tsslltn = tessellation(seed, tsslltn, L)
            tsslltd_coords = seed_tsslltn.copy()
        else:
            # Begin periodic random hard disk node placement procedure
            coords, tsslltd_coords = periodic_random_hard_disk_node_placement(
                dim, L, b, max_try, rng, tsslltn, coords, tsslltd_coords)

    # Save coordinates
    np.savetxt(coords_filename, coords)

def initial_periodic_disordered_hyperuniform_node_placement(
        dim: int,
        L: np.ndarray,
        n: int,
        config_filename_prefix: str) -> None:
    """Initial periodic disordered hyperuniform node placement
    procedure.

    This function places/seeds nodes within an empty simulation box in a
    periodic disordered hyperuniform fashion.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        L (np.ndarray): Simulation box side lengths.
        n (int): Intended number of nodes.
        config_filename_prefix (str): Configuration filename prefix.
    
    """
    error_str = (
        "Periodic disordered hyperuniform node placement procedure has "
        + "not been defined yet!"
    )
    raise NotImplementedError(error_str)

def initial_lammps_input_file_generator(
        dim: int,
        L: np.ndarray,
        b: float,
        n: int,
        config_filename_prefix: str) -> None:
    """LAMMPS input file generator for soft pushoff and FIRE energy
    minimization of randomly placed nodes.

    This function generates a LAMMPS input file that randomly places
    nodes within an empty simulation box, followed by a soft pushoff
    procedure, and finally followed by a FIRE energy minimization
    procedure.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        L (np.ndarray): Simulation box side lengths.
        b (float): Node diameter.
        n (int): Number of core nodes.
        config_filename_prefix (str): Configuration filename prefix.
    
    """
    # This function is only applicable to three-dimensional networks
    if dim != 3:
        error_str = "The network must be three dimensional."
        raise ValueError(error_str)
    
    # Generate filenames
    lammps_input_filename = config_filename_prefix + ".in"
    coords_filename = config_filename_prefix + ".coords"
    
    # Create random integers
    rng = np.random.default_rng()
    create_atoms_random_int = rng.integers(100000, dtype=int) + 1
    velocity_random_int = rng.integers(10000000, dtype=int) + 1
    
    lammps_input_file = open(lammps_input_filename, "w")

    # Specify fundamental parameters and simulation box
    lammps_input_file.write("units         lj\n")
    lammps_input_file.write(f"dimension     {dim:d}\n")
    lammps_input_file.write("boundary      p p p\n")
    lammps_input_file.write(f"region        box block 0 {L[0]:0.4f} 0 {L[1]:0.4f} 0 {L[2]:0.4f}\n")
    lammps_input_file.write("create_box    1 box\n")
    # Randomly place nodes in simulation box
    lammps_input_file.write(f"create_atoms  1 random {n:d} {create_atoms_random_int:d} NULL\n")
    lammps_input_file.write("mass          1 1.0\n\n")

    # Initiate thermodynammics
    lammps_input_file.write(f"velocity all create 1.0 {velocity_random_int:d} mom yes rot yes dist gaussian\n\n")

    lammps_input_file.write("timestep 0.002\n\n")

    lammps_input_file.write("thermo_style custom step temp pe ke ebond pxx pyy pzz lx ly lz vol density\n")
    lammps_input_file.write("thermo 1\n\n")

    # Soft pushoff procedure
    lammps_input_file.write(f"pair_style soft {b:0.1f}\n")
    lammps_input_file.write("pair_coeff * * 10.0\n")
    lammps_input_file.write("pair_modify shift yes\n\n")

    # FIRE energy minimization procedure
    lammps_input_file.write("min_style fire\n")
    lammps_input_file.write("minimize 1.0e-10 1.0e-10 100000 100000\n\n")

    # Write coordinates to coordinates file
    lammps_input_file.write(f"write_data {coords_filename} nocoeff")

    lammps_input_file.close()

def initial_node_seeding(
        network: str,
        date: str,
        batch: str,
        sample: int,
        scheme: str,
        dim: int,
        b: float,
        n: int,
        config: int,
        max_try: int) -> None:
    """Initial node placement procedure for irregular spatial networks.

    This function loads the simulation box side lengths for the
    irregular spatial networks. Then, depending on the particular core
    node placement scheme, this function calls upon a corresponding
    helper function to calculate the initial core node positions.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; either "auelp", "apelp", "swidt", "delaunay", or "voronoi" (corresponding to artificial uniform end-linked polymer networks ("auelp"), artificial polydisperse end-linked polymer networks ("apelp"), spider web-inspired Delaunay-triangulated networks ("swidt"), Delaunay-triangulated networks ("delaunay"), or Voronoi-tessellated networks ("voronoi")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular scheme used to generate the positions of the core nodes; either "random", "prhd", "pdhu", or "lammps" (corresponding to the random node placement procedure ("random"), periodic random hard disk node placement procedure ("prhd"), periodic disordered hyperuniform node placement procedure ("pdhu"), or nodes randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Node diameter.
        n (int): Number of core nodes.
        config (int): Configuration number.
        max_try (int): Maximum number of node placement attempts for the periodic random hard disk node placement procedure ("prhd").
    
    """
    # Load L
    L = np.loadtxt(L_filename_str(network, date, batch, sample))

    # Configuration filename prefix
    config_filename_prefix = config_filename_str(
        network, date, batch, sample, config)

    # Call appropriate initial node placement helper function
    if scheme == "random":
        initial_random_node_placement(dim, L, n, config_filename_prefix)
    elif scheme == "prhd":
        initial_periodic_random_hard_disk_node_placement(
            dim, L, b, n, max_try, config_filename_prefix)
    elif scheme == "pdhu":
        initial_periodic_disordered_hyperuniform_node_placement(
            dim, L, n, config_filename_prefix)
    elif scheme == "lammps":
        initial_lammps_input_file_generator(dim, L, b, n, config_filename_prefix)