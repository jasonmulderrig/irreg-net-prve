# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
import numpy as np
from networks.aelp_networks_plotting import aelp_network_topology_plotter
from networks.auelp_networks_config import (
    auelpConfig,
    params_arr_func
)

# Hydra ConfigStore initialization
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(name="config", node=auelpConfig)

@hydra.main(version_base=None, config_path=".", config_name="auelp_networks_config")
def main(cfg: auelpConfig) -> None:
    params_arr, _ = params_arr_func(cfg)
    
    # Artificial uniform end-linked polymer network plotting parameters
    plt_pad_prefactor = 0.30
    core_tick_inc_prefactor = 0.2

    # Network parameters
    dim_2 = 2
    dim_3 = 3
    b = 1.0
    xi = 0.98
    rho_en = 0.85
    k = 4
    n = 100
    en = 51
    config = 0

    # Identification of the sample value for the desired network
    dim_2_sample = int(
        np.where(np.all(params_arr == (dim_2, b, xi, rho_en, k, n, en), axis=1))[0][0])
    dim_3_sample = int(
        np.where(np.all(params_arr == (dim_3, b, xi, rho_en, k, n, en), axis=1))[0][0])

    # Artificial uniform end-linked polymer network plotting
    aelp_network_topology_plotter(
        plt_pad_prefactor, core_tick_inc_prefactor, cfg.label.network,
        cfg.label.date, cfg.label.batch, dim_2_sample, config)
    aelp_network_topology_plotter(
        plt_pad_prefactor, core_tick_inc_prefactor, cfg.label.network,
        cfg.label.date, cfg.label.batch, dim_3_sample, config)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Artificial uniform end-linked polymer network topology plotting took {execution_time} seconds to run")