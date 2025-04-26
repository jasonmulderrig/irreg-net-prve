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
from helpers.polymer_network_chain_statistics import nu_mean_p_nu_bimodal_func
from helpers.multiprocessing_utils import (
    run_aelp_L,
    run_initial_node_seeding,
    run_abelp_network_topology
)
from networks.abelp_networks_config import (
    abelpConfig,
    sample_params_arr_func,
    sample_config_params_arr_func
)

# Hydra ConfigStore initialization
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(name="config", node=abelpConfig)

def params_list_func(params_arr: np.ndarray) -> list[tuple]:
    if params_arr.ndim == 1: return [tuple(params_arr)]
    else: return list(map(tuple, params_arr))

@hydra.main(version_base=None, config_path=".", config_name="abelp_networks_config")
def main(cfg: abelpConfig) -> None:
    # Gather arrays of configuration parameters
    sample_params_arr, sample_num = sample_params_arr_func(cfg)
    sample_config_params_arr, sample_config_num = sample_config_params_arr_func(
        cfg)
    
    ##### Calculate and save L for each artificial polydisperse
    ##### end-linked polymer network parameter sample
    print("Calculating simulation box side lengths", flush=True)

    if sample_params_arr.ndim == 1:
        sample = sample_params_arr[0]
        dim = sample_params_arr[1]
        rho_en = sample_params_arr[4]
        k = sample_params_arr[5]
        n = sample_params_arr[6]
        p = sample_params_arr[7]
        en_min = int(sample_params_arr[8])
        en_max = int(sample_params_arr[9])
        nu_min = en_min - 1
        nu_max = en_max - 1
        nu = nu_mean_p_nu_bimodal_func(p, nu_min, nu_max)
        en = nu + 1
        L_params_arr = np.asarray([sample, dim, rho_en, k, n, en])
    else:
        L_params_arr = np.empty((sample_num, 6))
        for indx in range(sample_num):
            sample = sample_params_arr[indx, 0]
            dim = sample_params_arr[indx, 1]
            rho_en = sample_params_arr[indx, 4]
            k = sample_params_arr[indx, 5]
            n = sample_params_arr[indx, 6]
            p = sample_params_arr[indx, 7]
            en_min = int(sample_params_arr[indx, 8])
            en_max = int(sample_params_arr[indx, 9])
            nu_min = en_min - 1
            nu_max = en_max - 1
            nu = nu_mean_p_nu_bimodal_func(p, nu_min, nu_max)
            en = nu + 1
            L_params_arr[indx, :] = np.asarray([sample, dim, rho_en, k, n, en])
    L_params_list = params_list_func(L_params_arr)
    L_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(dim), rho_en, int(k), int(n), int(en))
            for (sample, dim, rho_en, k, n, en) in L_params_list
        ]
    )
    random.shuffle(L_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(run_aelp_L, L_args)
    
    ##### Perform the initial node seeding procedure for each artificial
    ##### bimodal end-linked polymer network parameter sample
    print("Performing the initial node seeding", flush=True)

    initial_node_seeding_params_arr = (
        sample_config_params_arr[:, [0, 1, 2, 6, 10]]
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
            n = int(sample_config_params_arr[indx, 6])
            config = int(sample_config_params_arr[indx, 10])
            
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
            print_str = (
                "Success! prhd_n = n for all abelp network parameters!"
            )
            print(print_str, flush=True)
        elif np.shape(sample_config_params_prhd_n_neq_n)[0] > 0:
            print_str = (
                "prhd_n != n for at least one set of abelp network "
                + "parameters. Repeat the periodic random hard disk node "
                + "placement procedure for the applicable set of abelp "
                + "network parameters before continuing on to the "
                + "topology initialization procedure."
            )
            print(print_str, flush=True)
    
    ##### Perform the network topology initialization procedure for each
    ##### bimodal artificial end-linked polymer network parameter sample
    print_str = (
        "Performing the artificial bimodal end-linked polymer network "
        + "topology initialization procedure"
    )
    print(print_str, flush=True)

    topology_params_arr = (
        np.delete(sample_config_params_arr, 4, axis=1)
    ) # sample, dim, b, xi, k, n, p, en_min, en_max, config
    topology_params_list = params_list_func(topology_params_arr)
    topology_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), cfg.label.scheme, int(dim), b, xi, int(k), int(n), p, int(en_min), int(en_max), int(config), int(cfg.synthesis.max_try))
            for (sample, dim, b, xi, k, n, p, en_min, en_max, config) in topology_params_list
        ]
    )
    random.shuffle(topology_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(run_abelp_network_topology, topology_args)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Artificial bimodal end-linked polymer network synthesis protocol took {execution_time} seconds to run")