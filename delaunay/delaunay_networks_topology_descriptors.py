# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
import multiprocessing
import random
from helpers.multiprocessing_utils import (
    run_delaunay_network_local_topological_descriptor,
    run_delaunay_network_global_topological_descriptor,
    run_delaunay_network_global_morphological_descriptor
)
from networks.delaunay_networks_config import (
    delaunayConfig,
    params_arr_func
)

# Hydra ConfigStore initialization
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(name="config", node=delaunayConfig)

@hydra.main(version_base=None, config_path=".", config_name="delaunay_networks_config")
def main(cfg: delaunayConfig) -> None:
    _, sample_num = params_arr_func(cfg)
    b = cfg.topology.b[0]

    local_topological_descriptor_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(config), b, *lcl_tplgcl_dscrptrs)
            for sample in range(sample_num)
            for config in range(cfg.topology.config)
            for lcl_tplgcl_dscrptrs in list(map(tuple, cfg.descriptors.local_topological_descriptors))
        ]
    )
    random.shuffle(local_topological_descriptor_args)

    global_topological_descriptor_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(config), b, *glbl_tplgcl_dscrptrs)
            for sample in range(sample_num)
            for config in range(cfg.topology.config)
            for glbl_tplgcl_dscrptrs in list(map(tuple, cfg.descriptors.global_topological_descriptors))
        ]
    )
    random.shuffle(global_topological_descriptor_args)

    global_morphological_descriptor_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(config), b, *glbl_mrphlgcl_dscrptrs)
            for sample in range(sample_num)
            for config in range(cfg.topology.config)
            for glbl_mrphlgcl_dscrptrs in list(map(tuple, cfg.descriptors.global_morphological_descriptors))
        ]
    )
    random.shuffle(global_morphological_descriptor_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(
            run_delaunay_network_local_topological_descriptor,
            local_topological_descriptor_args)
        pool.map(
            run_delaunay_network_global_topological_descriptor,
            global_topological_descriptor_args)
        pool.map(
            run_delaunay_network_global_morphological_descriptor,
            global_morphological_descriptor_args)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Delaunay network topology descriptors calculation took {execution_time} seconds to run")