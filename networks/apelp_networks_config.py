import numpy as np
import omegaconf

def params_list_func(params_arr: np.ndarray) -> list[tuple]:
    if params_arr.ndim == 1: return [tuple(params_arr)]
    else: return list(map(tuple, params_arr))

def params_arr_func(cfg) -> tuple[np.ndarray, int]:
    try:
        topology = cfg.topology
    except omegaconf.errors.ConfigAttributeError:
        topology = cfg.networks.apelp.topology
    
    dim_arr = np.asarray(topology.dim, dtype=int)
    b_arr = np.asarray(topology.b)
    xi_arr = np.asarray(topology.xi)
    rho_en_arr = np.asarray(topology.rho_en)
    k_arr = np.asarray(topology.k, dtype=int)
    n_arr = np.asarray(topology.n, dtype=int)
    en_arr = np.asarray(topology.en, dtype=int)

    dim_num = np.shape(dim_arr)[0]
    b_num = np.shape(b_arr)[0]
    xi_num = np.shape(xi_arr)[0]
    rho_en_num = np.shape(rho_en_arr)[0]
    k_num = np.shape(k_arr)[0]
    n_num = np.shape(n_arr)[0]
    en_num = np.shape(en_arr)[0]
    sample_num = dim_num * b_num * xi_num * rho_en_num * k_num * n_num * en_num

    params_arr = np.empty((sample_num, 8))
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
                                            en[0],
                                            en[1]
                                        ]
                                    )
                                )
                                sample += 1
    
    return params_arr, sample_num

def sample_params_arr_func(cfg) -> tuple[np.ndarray, int]:
    try:
        topology = cfg.topology
    except omegaconf.errors.ConfigAttributeError:
        topology = cfg.networks.apelp.topology
    
    dim_arr = np.asarray(topology.dim, dtype=int)
    b_arr = np.asarray(topology.b)
    xi_arr = np.asarray(topology.xi)
    rho_en_arr = np.asarray(topology.rho_en)
    k_arr = np.asarray(topology.k, dtype=int)
    n_arr = np.asarray(topology.n, dtype=int)
    en_arr = np.asarray(topology.en, dtype=int)

    dim_num = np.shape(dim_arr)[0]
    b_num = np.shape(b_arr)[0]
    xi_num = np.shape(xi_arr)[0]
    rho_en_num = np.shape(rho_en_arr)[0]
    k_num = np.shape(k_arr)[0]
    n_num = np.shape(n_arr)[0]
    en_num = np.shape(en_arr)[0]
    sample_num = dim_num * b_num * xi_num * rho_en_num * k_num * n_num * en_num

    sample_params_arr = np.empty((sample_num, 9))
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
                                            en[0],
                                            en[1]
                                        ]
                                    )
                                )
                                sample += 1
    
    return sample_params_arr, sample_num

def sample_config_params_arr_func(cfg) -> tuple[np.ndarray, int]:
    try:
        topology = cfg.topology
    except omegaconf.errors.ConfigAttributeError:
        topology = cfg.networks.apelp.topology
    
    dim_arr = np.asarray(topology.dim, dtype=int)
    b_arr = np.asarray(topology.b)
    xi_arr = np.asarray(topology.xi)
    rho_en_arr = np.asarray(topology.rho_en)
    k_arr = np.asarray(topology.k, dtype=int)
    n_arr = np.asarray(topology.n, dtype=int)
    en_arr = np.asarray(topology.en, dtype=int)
    config_arr = np.arange(topology.config, dtype=int)

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

    sample_config_params_arr = np.empty((sample_config_num, 10))
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
                                                en[0],
                                                en[1],
                                                config
                                            ]
                                        )
                                    )
                                    indx += 1
                                sample += 1
    
    return sample_config_params_arr, sample_config_num