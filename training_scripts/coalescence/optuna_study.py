"""
Script to conduct hyperparameter optimization with Optuna for coalescence models
"""

import copy
import csv
import json
import os
import random
import sys
import time
import uuid
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import optuna
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from src import data_utils as du
from src import diagnostics, model_factory, recon_coalescence_losses, training_utils

# Configuration: Select model type
# MODEL_TYPE = "autoregressive"
MODEL_TYPE = "nn_dzdt"
# MODEL_TYPE = "sindy"

# Base parameters (update as needed)
params = {
    "encoder_type": "ffnn",
    "decoder_type": "ffnn",
    "dynamics_type": MODEL_TYPE,
    "latent_dim": 3,
    "poly_order": 2,  # for SINDy
    "n_lag": 1,  # for AR
    "data_src": "erf",
    "batch_size": 8,  # Will be overridden by Optuna
    "wd": 1e-5,
    "random_seed": 42,
    "tol": 1e-10,  # Small value for numerical stability in log
}


def objective(trial, params, n_bins, train_data, test_data, dsd_time, x_train, m_train):
    # Set seed
    torch.manual_seed(params["random_seed"])
    np.random.seed(params["random_seed"])
    random.seed(params["random_seed"])

    # Make a copy of params for this trial
    trial_params = copy.deepcopy(params)

    # Hyperparameter options
    lr = trial.suggest_float("lr", 1e-6, 1e-1, log=True)
    batch_size = trial.suggest_int("batch_size", 4, 256)

    if MODEL_TYPE == "autoregressive":
        layer1_size = trial.suggest_int("layer1_size", 20, 180)
        layer2_size = trial.suggest_int("layer2_size", 20, 180)
        layer3_size = trial.suggest_int("layer3_size", 20, 180)
        trial_params["layer_size"] = (layer1_size, layer2_size, layer3_size)
        trial_params["w_recon"] = trial.suggest_float("w_recon", 0.1, 1.9)
        trial_params["w_dx"] = trial.suggest_float("w_dx", 0.1, 1.9)
        trial_params["w_dz"] = trial.suggest_float("w_dz", 0.1, 1.9)
    elif MODEL_TYPE == "nn_dzdt":
        layer1_size = trial.suggest_int("layer1_size", 20, 60)
        layer2_size = trial.suggest_int("layer2_size", 20, 60)
        layer3_size = trial.suggest_int("layer3_size", 20, 60)
        trial_params["layer_size"] = (layer1_size, layer2_size, layer3_size)
        lambda1_metaweight = trial.suggest_float("lambda1_metaweight", 0.50, 1.5)
        lambda1, lambda2, lambda3 = du.champion_calculate_weights(
            train_data, lambda1_metaweight=lambda1_metaweight, lambda3=1.0
        )
        trial_params["loss_weight_recon"] = lambda3
        trial_params["loss_weight_dx"] = lambda1
        trial_params["loss_weight_dz"] = lambda2
    elif MODEL_TYPE == "sindy":
        # latent_dim = trial.suggest_int("latent_dim", 1, 4)
        # poly_order = trial.suggest_int("poly_order", 2, 3)
        lambda1_metaweight = trial.suggest_float("lambda1_metaweight", 0.50, 1.5)
        lambda1, lambda2, lambda3 = du.champion_calculate_weights(
            train_data, lambda1_metaweight=lambda1_metaweight, lambda3=1.0
        )
        trial_params["loss_weight_recon"] = lambda3
        trial_params["loss_weight_dx"] = lambda1
        trial_params["loss_weight_dz"] = lambda2
    else:
        raise NotImplementedError(f"Model type {MODEL_TYPE} is not implemented")

    # Update learning rate and epochs
    trial_params["learning_rate"] = lr
    trial_params["num_epochs"] = 10  # Reduced for faster trials

    # Create model using the factory
    model = model_factory.create_model(
        encoder_type=trial_params["encoder_type"],
        decoder_type=trial_params["decoder_type"],
        dynamics_type=trial_params["dynamics_type"],
        params=trial_params,
        n_bins=n_bins,
    )

    # Setup device
    device = training_utils.setup_device(trial_params)
    model = model.to(device)

    # Setup optimization
    optimizer, scheduler, _ = training_utils.setup_optimization(model, trial_params)

    # Get loss function
    loss_fn = recon_coalescence_losses.get_loss_function(trial_params["dynamics_type"])

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=len(test_data), shuffle=True
    )

    # Training loop
    best_model, _ = training_utils.train_and_eval(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        params=trial_params,
        device=device,
        early_stopping=None,
        optuna_trial=trial,
    )

    # Calculate error (wass distance) over full dataset
    if MODEL_TYPE == "autoregressive":
        z_pred, z_data, x_pred = diagnostics.get_latent_trajectories_AR(
            trial_params["latent_dim"], best_model, dsd_time, x_train, m_train
        )
    else:  # sindy or nn_dzdt
        z_pred, z_data, x_pred = diagnostics.get_latent_trajectories_dzdt(
            trial_params["latent_dim"],
            best_model,
            dsd_time,
            x_train,
            m_train,
            x_train,
            m_train,
        )
    _, train_wass, _, _ = diagnostics.get_performance_metrics(
        x_train, m_train, z_pred, x_pred
    )
    mean_trainset_wass = np.mean(train_wass)

    return mean_trainset_wass


def optimize_worker(args):
    worker_id, n_trials, storage_url, study_name, objective_func = args

    # Set this once per worker process
    torch.set_num_threads(1)

    # Do optimization
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    study.optimize(objective_func, n_trials=n_trials)

    return None


if __name__ == "__main__":
    total_trials = 1024  # On mac with 8 perf. cores, choose multiple of 8 total_trials
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
    test_data = test_loader.dataset

    # Set loss weights if needed (will be overridden by Optuna trials if tuning them)
    params = training_utils.set_loss_weights(params, train_data)
    
    # Set up save folder
    base_output_directory = Path("results/Optuna_coalescence_studies/")
    id = str(uuid.uuid4().hex)
    output_directory = base_output_directory / (
        f"{params['encoder_type']}_{params['dynamics_type']}_"
        + datetime.now().isoformat().split(".")[0]  # + "_" + id
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

    # Set up SQLite storage in results folder
    db_path = output_directory / "study.db"
    storage_url = f"sqlite:///{db_path}"
    study_name = MODEL_TYPE

    # Set up study
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.HyperbandPruner()
    study = optuna.create_study(
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        direction="minimize",
        load_if_exists=True,
    )

    # Run study
    start_time = time.time()
    objective_with_args = partial(
        objective,
        params=params,
        n_bins=n_bins,
        train_data=train_data,
        test_data=test_data,
        dsd_time=dsd_time,
        x_train=x_train,
        m_train=m_train,
    )
    if parallel_flag:
        worker_args = [
            (i, trials_per_worker, storage_url, study_name, objective_with_args)
            for i in range(n_workers)
        ]
        with Pool(processes=n_workers) as pool:
            results = pool.map(optimize_worker, worker_args)
    else:
        study.optimize(objective_with_args, n_trials=total_trials)
    stop_time = time.time()

    # Save best hyperparameters
    best_params_file = output_directory / "best_params.json"
    best_params = {
        "value": study.best_trial.value,
        "params": study.best_trial.params,
        "optuna_runtime": stop_time - start_time,
        "random_seed": params["random_seed"],
    }
    best_params_file.write_text(json.dumps(best_params, indent=4))

    # Save full study object
    study_file = output_directory / "study.pkl"
    torch.save(study, study_file)

    # Save visualizations
    try:
        import optuna.visualization as vis

        vis.plot_optimization_history(study).write_html(
            str(output_directory / "opt_history.html")
        )
        vis.plot_param_importances(study).write_html(
            str(output_directory / "param_importance.html")
        )
        vis.plot_parallel_coordinate(study).write_html(
            str(output_directory / "parallel_coords.html")
        )
        vis.plot_contour(study).write_html(
            str(output_directory / "param_contours.html")
        )

    except Exception as e:
        print(f"Could not generate visualizations: {e}")

    # Save all trials to CSV
    csv_file = output_directory / "all_trials.csv"
    with csv_file.open("w", newline="") as f:
        fieldnames = ["trial_number", "value", "state", "duration"] + list(
            study.best_trial.params.keys()
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for trial in study.trials:
            row = {
                "trial_number": trial.number,
                "value": trial.value,
                "state": trial.state.name,
                "duration": str(trial.duration),
            }
            row.update(trial.params)
            writer.writerow(row)

    # Print best value
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print(f"Duration: {stop_time - start_time}")
