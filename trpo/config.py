import os
import warnings

def get_params():
    params = {
        "env_name": "PongNoFrameskip-v4",
        "num_worker": -1,
        "total_iterations": 800000,
        "interval": 1500,
        "do_test": False,
        "train_from_scratch": False,
        "seed": 132,
        "state_shape": (4, 84, 84),
        "damping": 1e-3,
        "k": 10,
        "trust_region": 0.001,
        "batch_size": 512,
        "line_search_num": 10,
        "value_opt_epoch": 3,
        "value_mini_batch_size": 64,
        "value_lr": 1e-4,
        "lambda": 1,
        "gamma": 0.98,
        "ent_coeff": 0.01,
        "n_workers": os.cpu_count(),
    }

    if params["num_worker"] == -1:
        params["n_workers"] = os.cpu_count()

    if params["n_workers"] > os.cpu_count():
        warnings.warn("You're using more workers than your machine's CPU cores")

    return params
