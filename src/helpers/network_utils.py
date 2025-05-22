import numpy as np

def a_or_v_func(dim: int, b: float) -> float:
    """Area or volume of a chain segment and/or cross-linker.

    This function calculates the area or volume of a chain segment
    and/or cross-linker, given its diameter.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Chain segment and/or cross-linker diameter.

    Returns:
        float: Chain segment and/or cross-linker area or volume.
    """
    return np.pi * b**dim / (2*dim)

def rho_func(n: float, A_or_V: float) -> float:
    """Particle number density.

    This function calculates the particle number density given the
    number of particles and the simulation box area or volume in two or
    three dimensions, respectively.

    Args:
        n (float): Number of particles.
        A_or_V (float): Simulation box area or volume.

    Returns:
        float: Particle number density.
    """
    return n / A_or_V

def eta_func(dim: int, b: float, rho: float) -> float:
    """Particle packing density.

    This function calculates the particle packing density given the
    particle number density.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Particle diameter.
        rho (float): Particle number density.

    Returns:
        float: Particle packing density.
    """
    return a_or_v_func(dim, b) * rho