from dataclasses import dataclass
import numpy as np

@dataclass
class Labels:
    network: str
    date: str
    batch: str
    scheme: str

@dataclass
class Topology:
    dim: list[int]
    b: list[float]
    n: list[int]
    eta_n: list[float]
    config: int

@dataclass
class Synthesis:
    max_try: int

@dataclass
class Descriptors:
    local_topological_descriptors: list[list[str, bool, bool]]
    global_topological_descriptors: list[list[str, str, bool, bool]]
    global_morphological_descriptors: list[list[str, bool, bool]]

@dataclass
class Multiprocessing:
    cpu_num: int

@dataclass
class delaunayConfig:
    labels: Labels
    topology: Topology
    synthesis: Synthesis
    descriptors: Descriptors
    multiprocessing: Multiprocessing

def params_arr_func(cfg: delaunayConfig) -> tuple[np.ndarray, int]:
    dim_arr = np.asarray(cfg.topology.dim, dtype=int)
    b_arr = np.asarray(cfg.topology.b)
    n_arr = np.asarray(cfg.topology.n, dtype=int)
    eta_n_arr = np.asarray(cfg.topology.eta_n)

    dim_num = np.shape(dim_arr)[0]
    b_num = np.shape(b_arr)[0]
    n_num = np.shape(n_arr)[0]
    eta_n_num = np.shape(eta_n_arr)[0]
    sample_num = dim_num * b_num * n_num * eta_n_num

    params_arr = np.empty((sample_num, 4))
    sample = 0
    for dim in dim_arr:
        for b in b_arr:
            for n in n_arr:
                for eta_n in eta_n_arr:
                    params_arr[sample, :] = (
                        np.asarray(
                            [
                                dim,
                                b,
                                n,
                                eta_n
                            ]
                        )
                    )
                    sample += 1
    
    return params_arr, sample_num

def sample_params_arr_func(cfg: delaunayConfig) -> tuple[np.ndarray, int]:
    dim_arr = np.asarray(cfg.topology.dim, dtype=int)
    b_arr = np.asarray(cfg.topology.b)
    n_arr = np.asarray(cfg.topology.n, dtype=int)
    eta_n_arr = np.asarray(cfg.topology.eta_n)

    dim_num = np.shape(dim_arr)[0]
    b_num = np.shape(b_arr)[0]
    n_num = np.shape(n_arr)[0]
    eta_n_num = np.shape(eta_n_arr)[0]
    sample_num = dim_num * b_num * n_num * eta_n_num

    sample_params_arr = np.empty((sample_num, 5))
    sample = 0
    for dim in dim_arr:
        for b in b_arr:
            for n in n_arr:
                for eta_n in eta_n_arr:
                    sample_params_arr[sample, :] = (
                        np.asarray(
                            [
                                sample,
                                dim,
                                b,
                                n,
                                eta_n
                            ]
                        )
                    )
                    sample += 1
    
    return sample_params_arr, sample_num

def sample_config_params_arr_func(cfg: delaunayConfig) -> tuple[np.ndarray, int]:
    dim_arr = np.asarray(cfg.topology.dim, dtype=int)
    b_arr = np.asarray(cfg.topology.b)
    n_arr = np.asarray(cfg.topology.n, dtype=int)
    eta_n_arr = np.asarray(cfg.topology.eta_n)
    config_arr = np.arange(cfg.topology.config, dtype=int)

    dim_num = np.shape(dim_arr)[0]
    b_num = np.shape(b_arr)[0]
    n_num = np.shape(n_arr)[0]
    eta_n_num = np.shape(eta_n_arr)[0]
    config_num = np.shape(config_arr)[0]
    sample_config_num = dim_num * b_num * n_num * eta_n_num * config_num

    sample_config_params_arr = np.empty((sample_config_num, 6))
    sample = 0
    indx = 0
    for dim in dim_arr:
        for b in b_arr:
            for n in n_arr:
                for eta_n in eta_n_arr:
                    for config in config_arr:
                        sample_config_params_arr[indx, :] = (
                            np.asarray(
                                [
                                    sample,
                                    dim,
                                    b,
                                    n,
                                    eta_n,
                                    config
                                ]
                            )
                        )
                        indx += 1
                    sample += 1
    
    return sample_config_params_arr, sample_config_num