"""
Script to test random seed variation for coalescence models
"""

import json
import os
import random
import sys
import time
import uuid
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(project_root)

from src import data_utils as du
from src import model_factory, training_utils, recon_coalescence_losses

# Configuration: Select model type
# MODEL_TYPE = "autoregressive"
# MODEL_TYPE = "nn_dzdt"
MODEL_TYPE = "sindy"

# Base parameters (update as needed)
params = {
    "encoder_type": "ffnn",
    "decoder_type": "ffnn",
    "dynamics_type": MODEL_TYPE,
    "latent_dim": 3,
    "poly_order": 2,  # for SINDy
    "layer_size": (128, 128, 64),  # for NN dzdt
    "n_lag": 1,  # for AR
    "data_src": "erf",
    "batch_size": 100,
    "learning_rate": 0.001,
    "wd": 1e-5,
    "random_seed": 42,
    "tol": 1e-10,  # Small value for numerical stability in log
}


def train_model(args):
    # Unpack args
    random_seed, n_bins, train_loader, test_loader, params_copy = args

    # Set this once per worker process
    torch.set_num_threads(1)

    # Set seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Update params with this seed
    params_copy["random_seed"] = random_seed

    # Create model using the factory
    model = model_factory.create_model(
        encoder_type=params_copy["encoder_type"],
        decoder_type=params_copy["decoder_type"],
        dynamics_type=params_copy["dynamics_type"],
        params=params_copy,
        n_bins=n_bins,
    )

    # Setup device
    device = training_utils.setup_device(params_copy)
    model = model.to(device)

    # Setup optimization
    optimizer, scheduler, _ = training_utils.setup_optimization(model, params_copy)

    # Get loss function
    loss_fn = recon_coalescence_losses.get_loss_function(params_copy["dynamics_type"])

    # Training loop (short for seed variation)
    params_copy["num_epochs"] = 10
    _, losses = training_utils.train_and_eval(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        params=params_copy,
        device=device,
        early_stopping=None,
        optuna_trial=None,
    )

    # Extract minimum total loss
    best_train_loss = np.min(losses["total"])

    # Print
    print(f"Best train loss for seed {random_seed}: {best_train_loss:.6f}")

    return best_train_loss


if __name__ == "__main__":
    total_trials = 200  # On mac with 8 perf. cores, choose multiple of 8 total_trials
    parallel_flag = True

    # Setup dataloaders using unified utility
    train_loader, test_loader, metadata = training_utils.setup_dataloaders(
        params["data_src"], params
    )
    n_bins = metadata["n_bins"]
    dsd_time = metadata["dsd_time"]
    x_train = metadata["x_train"]
    m_train = metadata["m_train"]
    train_data = train_loader.dataset

    # Set loss weights if needed
    params = training_utils.setup_loss_weights(params, train_data)
    # if params["dynamics_type"] in ["sindy", "nn_dzdt"]:
    #     if (
    #         "loss_weight_dx" not in params
    #         or "loss_weight_dz" not in params
    #         or "loss_weight_recon" not in params
    #     ):
    #         lambda1, lambda2, lambda3 = du.champion_calculate_weights(train_data)
    #         params["loss_weight_recon"] = 1.0
    #         params["loss_weight_dx"] = lambda1
    #         params["loss_weight_dz"] = lambda2

    # Set up save folder
    base_output_directory = Path("results/Random Seeds/")
    id = str(uuid.uuid4().hex)
    output_directory = base_output_directory / (
        f"{params['encoder_type']}_{params['dynamics_type']}_"
        + datetime.now().isoformat().split(".")[0]
        + "_"
        + id
    )
    if not output_directory.exists():
        output_directory.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Folder '{output_directory}' already exists.")

    # Set up parallel info
    n_workers = 8  # 8 performance cores and 4 efficiency cores
    if parallel_flag:
        if total_trials % n_workers:
            raise RuntimeError("Ensure total trials is a multiple of n_workers")
    trials_per_worker = int(total_trials / n_workers)

    # Run study
    start_time = time.time()
    if parallel_flag:
        # Create independent copies of params for each worker
        import copy

        worker_args = [
            (i, n_bins, train_loader, test_loader, copy.deepcopy(params))
            for i in range(total_trials)
        ]
        with Pool(processes=n_workers) as pool:
            res = pool.map(train_model, worker_args)
        res = np.array(res)
    else:
        res = np.zeros(total_trials)
        for i in range(total_trials):
            import copy

            args = (i, n_bins, train_loader, test_loader, copy.deepcopy(params))
            res[i] = train_model(args)
    stop_time = time.time()
    print(f"Duration: {stop_time - start_time}")

    # Calculate
    best_seed = np.argmin(res)
    best_train_loss = res[best_seed]

    # Save best results
    best_seed_file = output_directory / "best_seed.json"
    best_seed_dict = {
        "loss": float(best_train_loss),
        "seed": int(best_seed),
        "runtime": stop_time - start_time,
    }
    best_seed_file.write_text(json.dumps(best_seed_dict, indent=4))

    # Save all results
    all_params_file = output_directory / "all_seeds.csv"
    sf = pd.Series(res, name="loss").sort_values()
    sf.index.name = "seed"
    sf.to_csv(all_params_file)

    # Plot results
    hist_file = output_directory / "hist.png"
    fig, axes = plt.subplots(1, 2, figsize=(10, 10), layout="constrained")
    # ---
    ax = axes[0]
    ax.hist(res, bins=20, label="Training Loss")
    ax.set_xlabel("Training Loss")
    ax.set_ylabel("Number of trials")
    ax.set_title("Total")
    # ---
    ax = axes[1]
    ax.hist(
        res, bins=20, range=(res.min(), np.quantile(res, 0.95)), label="Training Loss"
    )
    ax.set_xlabel("Training Loss")
    ax.set_ylabel("Number of trials")
    ax.set_title("95th Percentile Zoom")
    # ---
    fig.suptitle(f"Training Loss Over Random Seed ({total_trials} Trials)")
    fig.savefig(hist_file)

    # Print best values
    print(f"Best seed is {best_seed} with loss {best_train_loss}")

    # Finish up
    plt.close("all")
