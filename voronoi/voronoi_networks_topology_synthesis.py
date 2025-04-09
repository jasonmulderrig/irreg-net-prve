# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
import multiprocessing
import random
import numpy as np
from file_io.file_io import config_filename_str
from helpers.multiprocessing_utils import (
    run_voronoi_L,
    run_initial_node_seeding,
    run_voronoi_network_topology
)
from networks.voronoi_networks_config import (
    voronoiConfig,
    sample_params_arr_func,
    sample_config_params_arr_func,
)

# Hydra ConfigStore initialization
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(name="config", node=voronoiConfig)

def params_list_func(params_arr: np.ndarray) -> list[tuple]:
    if params_arr.ndim == 1: return [tuple(params_arr)]
    else: return list(map(tuple, params_arr))

@hydra.main(version_base=None, config_path=".", config_name="voronoi_networks_config")
def main(cfg: voronoiConfig) -> None:
    # Gather arrays of configuration parameters
    sample_params_arr, _ = sample_params_arr_func(cfg)
    sample_config_params_arr, sample_config_num = sample_config_params_arr_func(
        cfg)

    ##### Calculate and save L for each Voronoi-tessellated network
    ##### parameter sample
    print("Calculating simulation box side lengths", flush=True)

    voronoi_L_params_list = params_list_func(sample_params_arr)
    voronoi_L_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(dim), b, int(n), eta_n)
            for (sample, dim, b, n, eta_n) in voronoi_L_params_list
        ]
    )
    random.shuffle(voronoi_L_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(run_voronoi_L, voronoi_L_args)
    
    ##### Perform the initial node seeding procedure for each
    ##### Voronoi-tessellated network parameter sample
    print("Performing the initial node seeding", flush=True)

    initial_node_seeding_params_arr = (
        sample_config_params_arr[:, [0, 1, 2, 3, 5]]
    ) # sample, dim, b, n, config
    initial_node_seeding_params_list = params_list_func(
        initial_node_seeding_params_arr)
    initial_node_seeding_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), cfg.label.scheme, int(dim), b, int(n), int(config), int(cfg.synthesis.max_try))
            for (sample, dim, b, n, config) in initial_node_seeding_params_list
        ]
    )
    random.shuffle(initial_node_seeding_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(run_initial_node_seeding, initial_node_seeding_args)
    
    # Check to see if the number of seeded nodes, prhd_n, equals the
    # intended/specified number of nodes to be seeded, n. Continue to
    # the topology initialization procedure ONLY IF prhd_n = n. If
    # prhd_n != n for any specified network, then the code block
    # identifies which particular set(s) of network parameters
    # prhd_n != n occurred for.
    if cfg.label.scheme == "prhd":
        prhd_n_vs_n = np.zeros(sample_config_num)
        for indx in range(sample_config_num):
            sample = int(sample_config_params_arr[indx, 0])
            n = int(sample_config_params_arr[indx, 3])
            config = int(sample_config_params_arr[indx, 5])

            coords_filename = (
                config_filename_str(cfg.label.network, cfg.label.date, cfg.label.batch, sample, config)
                + ".coords"
            )
            coords = np.loadtxt(coords_filename)
            
            if np.shape(coords)[0] == n: prhd_n_vs_n[indx] = 1
            else: pass

        sample_config_params_prhd_n_neq_n = (
            sample_config_params_arr[np.where(prhd_n_vs_n == 0)]
        )
        
        if np.shape(sample_config_params_prhd_n_neq_n)[0] == 0:
            print_str = "Success! prhd_n = n for all network parameters!"
            print(print_str, flush=True)
        elif np.shape(sample_config_params_prhd_n_neq_n)[0] > 0:
            print_str = (
                "prhd_n != n for at least one set of network parameters. "
                + "Repeat the periodic random hard disk node placement "
                + "procedure for the applicable set of network parameters before "
                + "continuing on to the topology initialization procedure."
            )
            print(print_str, flush=True)
    
    ##### Perform the topology initialization procedure for each
    ##### Voronoi-tessellated network parameter sample
    print(
        "Performing the Voronoi network topology initialization procedure",
        flush=True)

    voronoi_network_topology_params_arr = (
        sample_config_params_arr[:, [0, 1, 3, 5]]
    ) # sample, dim, n, config
    voronoi_network_topology_params_list = params_list_func(
        voronoi_network_topology_params_arr)
    voronoi_network_topology_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), cfg.label.scheme, int(dim), int(n), int(config))
            for (sample, dim, n, config) in voronoi_network_topology_params_list
        ]
    )
    random.shuffle(voronoi_network_topology_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(run_voronoi_network_topology, voronoi_network_topology_args)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Voronoi network synthesis protocol took {execution_time} seconds to run")