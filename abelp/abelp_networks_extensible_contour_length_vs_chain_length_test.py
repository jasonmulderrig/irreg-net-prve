# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
import numpy as np
from file_io.file_io import L_filename_str
from descriptors.general_topological_descriptors import l_func
from networks.aelp_networks import aelp_filename_str
from networks.abelp_networks_config import (
    abelpConfig,
    params_arr_func
)

# Hydra ConfigStore initialization
from hydra.core.config_store import ConfigStore
cs = ConfigStore.instance()
cs.store(name="config", node=abelpConfig)

@hydra.main(version_base=None, config_path=".", config_name="abelp_networks_config")
def main(cfg: abelpConfig) -> None:
    _, sample_num = params_arr_func(cfg)
        
    mean_perc_errant_edges = 0.
    for sample in range(sample_num):
        for config in range(cfg.topology.config):
            # Generate filenames
            L_filename = L_filename_str(
                cfg.label.network, cfg.label.date, cfg.label.batch, sample)
            aelp_filename = aelp_filename_str(
                cfg.label.network, cfg.label.date, cfg.label.batch, sample, config)
            coords_filename = aelp_filename + ".coords"
            conn_edges_filename = aelp_filename + "-conn_edges" + ".dat"
            conn_edges_type_filename = (
                aelp_filename + "-conn_edges_type" + ".dat"
            )
            l_cntr_conn_edges_filename = (
                aelp_filename + "-l_cntr_conn_edges" + ".dat"
            )

            # Load simulation box side lengths and node coordinates
            L = np.loadtxt(L_filename)
            coords = np.loadtxt(coords_filename)

            # Load fundamental graph constituents
            conn_edges = np.loadtxt(conn_edges_filename, dtype=int)
            conn_edges_type = np.loadtxt(conn_edges_type_filename, dtype=int)
            l_cntr_conn_edges = np.loadtxt(l_cntr_conn_edges_filename)
            m = np.shape(conn_edges)[0]

            # Calculate Euclidean edge length
            l_conn_edges = l_func(conn_edges, conn_edges_type, coords, L)
            
            errant_edges = 0
            for edge in range(m):
                if l_cntr_conn_edges[edge] < l_conn_edges[edge]:
                    errant_edges += 1
                    print("abelp sample {} config {} edge {} nu = {} < l = {}".format(sample, config, edge, l_cntr_conn_edges[edge], l_conn_edges[edge]))
            perc_errant_edges = errant_edges / m * 100
            if perc_errant_edges > 0:
                print("abelp sample {} config {} has {} errant edges, {} percent of edges in the network".format(sample, config, errant_edges, perc_errant_edges))
            mean_perc_errant_edges += perc_errant_edges
    mean_perc_errant_edges /= cfg.topology.config

    print("{} percent of abelp edges are errant (on average)".format(mean_perc_errant_edges))

if __name__ == "__main__":
    main()