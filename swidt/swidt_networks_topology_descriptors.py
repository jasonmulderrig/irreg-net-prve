# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
from omegaconf import DictConfig
import multiprocessing
import random
from src.helpers.multiprocessing_utils import (
    run_swidt_network_local_topological_descriptor,
    run_swidt_network_global_topological_descriptor,
    run_swidt_network_global_morphological_descriptor
)
from src.networks.swidt_networks_config import params_arr_func

@hydra.main(
        version_base=None,
        config_path="../configs/networks/swidt",
        config_name="swidt_networks")
def main(cfg: DictConfig) -> None:
    _, sample_num = params_arr_func(cfg)
    b = cfg.topology.b[0]

    ##### Calculate descriptors
    print("Calculating descriptors", flush=True)

    local_topological_descriptor_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(config), int(pruning), b, *lcl_tplgcl_dscrptrs)
            for sample in range(sample_num)
            for config in range(cfg.topology.config)
            for pruning in range(cfg.topology.pruning)
            for lcl_tplgcl_dscrptrs in list(map(tuple, cfg.descriptors.local_topological_descriptors))
        ]
    )
    random.shuffle(local_topological_descriptor_args)

    global_topological_descriptor_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(config), int(pruning), b, *glbl_tplgcl_dscrptrs)
            for sample in range(sample_num)
            for config in range(cfg.topology.config)
            for pruning in range(cfg.topology.pruning)
            for glbl_tplgcl_dscrptrs in list(map(tuple, cfg.descriptors.global_topological_descriptors))
        ]
    )
    random.shuffle(global_topological_descriptor_args)

    global_morphological_descriptor_args = (
        [
            (cfg.label.network, cfg.label.date, cfg.label.batch, int(sample), int(config), int(pruning), b, *glbl_mrphlgcl_dscrptrs)
            for sample in range(sample_num)
            for config in range(cfg.topology.config)
            for pruning in range(cfg.topology.pruning)
            for glbl_mrphlgcl_dscrptrs in list(map(tuple, cfg.descriptors.global_morphological_descriptors))
        ]
    )
    random.shuffle(global_morphological_descriptor_args)

    with multiprocessing.Pool(processes=cfg.multiprocessing.cpu_num) as pool:
        pool.map(
            run_swidt_network_local_topological_descriptor,
            local_topological_descriptor_args)
        pool.map(
            run_swidt_network_global_topological_descriptor,
            global_topological_descriptor_args)
        pool.map(
            run_swidt_network_global_morphological_descriptor,
            global_morphological_descriptor_args)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Spider web-inspired Delaunay network topology descriptors calculation took {execution_time} seconds to run")