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
    xi: list[float]
    rho_en: list[float]
    k: list[int]
    n: list[int]
    en: list[int]
    config: int

@dataclass
class Synthesis:
    max_try: int

@dataclass
class Descriptors:
    local_topological_descriptors: list[list[str, bool, bool, bool]]
    global_topological_descriptors: list[list[str, str, bool, bool, bool]]
    global_morphological_descriptors: list[list[str, bool, bool]]

@dataclass
class Multiprocessing:
    cpu_num: int

@dataclass
class auelpConfig:
    labels: Labels
    topology: Topology
    synthesis: Synthesis
    descriptors: Descriptors
    multiprocessing: Multiprocessing

def params_arr_func(cfg: auelpConfig) -> tuple[np.ndarray, int]:
    dim_arr = np.asarray(cfg.topology.dim, dtype=int)
    b_arr = np.asarray(cfg.topology.b)
    xi_arr = np.asarray(cfg.topology.xi)
    rho_en_arr = np.asarray(cfg.topology.rho_en)
    k_arr = np.asarray(cfg.topology.k, dtype=int)
    n_arr = np.asarray(cfg.topology.n, dtype=int)
    en_arr = np.asarray(cfg.topology.en, dtype=int)

    dim_num = np.shape(dim_arr)[0]
    b_num = np.shape(b_arr)[0]
    xi_num = np.shape(xi_arr)[0]
    rho_en_num = np.shape(rho_en_arr)[0]
    k_num = np.shape(k_arr)[0]
    n_num = np.shape(n_arr)[0]
    en_num = np.shape(en_arr)[0]
    sample_num = dim_num * b_num * xi_num * rho_en_num * k_num * n_num * en_num

    params_arr = np.empty((sample_num, 7))
    sample = 0
    for dim in dim_arr:
        for b in b_arr:
            for xi in xi_arr:
                for rho_en in rho_en_arr:
                    for k in k_arr:
                        for n in n_arr:
                            for en in en_arr:
                                params_arr[sample, :] = (
                                    np.asarray(
                                        [
                                            dim,
                                            b,
                                            xi,
                                            rho_en,
                                            k,
                                            n,
                                            en
                                        ]
                                    )
                                )
                                sample += 1
    
    return params_arr, sample_num

def sample_params_arr_func(cfg: auelpConfig) -> tuple[np.ndarray, int]:
    dim_arr = np.asarray(cfg.topology.dim, dtype=int)
    b_arr = np.asarray(cfg.topology.b)
    xi_arr = np.asarray(cfg.topology.xi)
    rho_en_arr = np.asarray(cfg.topology.rho_en)
    k_arr = np.asarray(cfg.topology.k, dtype=int)
    n_arr = np.asarray(cfg.topology.n, dtype=int)
    en_arr = np.asarray(cfg.topology.en, dtype=int)

    dim_num = np.shape(dim_arr)[0]
    b_num = np.shape(b_arr)[0]
    xi_num = np.shape(xi_arr)[0]
    rho_en_num = np.shape(rho_en_arr)[0]
    k_num = np.shape(k_arr)[0]
    n_num = np.shape(n_arr)[0]
    en_num = np.shape(en_arr)[0]
    sample_num = dim_num * b_num * xi_num * rho_en_num * k_num * n_num * en_num

    sample_params_arr = np.empty((sample_num, 8))
    sample = 0
    for dim in dim_arr:
        for b in b_arr:
            for xi in xi_arr:
                for rho_en in rho_en_arr:
                    for k in k_arr:
                        for n in n_arr:
                            for en in en_arr:
                                sample_params_arr[sample, :] = (
                                    np.asarray(
                                        [
                                            sample,
                                            dim,
                                            b,
                                            xi,
                                            rho_en,
                                            k,
                                            n,
                                            en
                                        ]
                                    )
                                )
                                sample += 1
    
    return sample_params_arr, sample_num

def sample_config_params_arr_func(cfg: auelpConfig) -> tuple[np.ndarray, int]:
    dim_arr = np.asarray(cfg.topology.dim, dtype=int)
    b_arr = np.asarray(cfg.topology.b)
    xi_arr = np.asarray(cfg.topology.xi)
    rho_en_arr = np.asarray(cfg.topology.rho_en)
    k_arr = np.asarray(cfg.topology.k, dtype=int)
    n_arr = np.asarray(cfg.topology.n, dtype=int)
    en_arr = np.asarray(cfg.topology.en, dtype=int)
    config_arr = np.arange(cfg.topology.config, dtype=int)

    dim_num = np.shape(dim_arr)[0]
    b_num = np.shape(b_arr)[0]
    xi_num = np.shape(xi_arr)[0]
    rho_en_num = np.shape(rho_en_arr)[0]
    k_num = np.shape(k_arr)[0]
    n_num = np.shape(n_arr)[0]
    en_num = np.shape(en_arr)[0]
    config_num = np.shape(config_arr)[0]
    sample_config_num = (
        dim_num * b_num * xi_num * rho_en_num * k_num * n_num * en_num
        * config_num
    )

    sample_config_params_arr = np.empty((sample_config_num, 9))
    sample = 0
    indx = 0
    for dim in dim_arr:
        for b in b_arr:
            for xi in xi_arr:
                for rho_en in rho_en_arr:
                    for k in k_arr:
                        for n in n_arr:
                            for en in en_arr:
                                for config in config_arr:
                                    sample_config_params_arr[indx, :] = (
                                        np.asarray(
                                            [
                                                sample,
                                                dim,
                                                b,
                                                xi,
                                                rho_en,
                                                k,
                                                n,
                                                en,
                                                config
                                            ]
                                        )
                                    )
                                    indx += 1
                                sample += 1
    
    return sample_config_params_arr, sample_config_num