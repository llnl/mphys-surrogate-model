"""
Unified training utilities for all model types.
Provides generic training loop, device setup, data loading, and optimization setup.
"""

import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src import data_utils as du
from src import diagnostics


def setup_device(params=None, allow_mps=False, batch_size=None):
    """
    Set up the compute device (CUDA, MPS, or CPU).

    Args:
        params: Optional dict with 'device' (override) and 'dynamics_type' keys
        allow_mps: If True, allows MPS device when available (some ops don't work on MPS)
        batch_size: Optional batch size to check MPS suitability

    Returns:
        torch.device
    """
    # Check for explicit device override in params
    if params is not None and "device" in params:
        return torch.device(params["device"])

    # SINDy models have MPS issues with as_strided operations
    # Force CPU for SINDy to avoid PyTorch MPS bugs
    if params is not None and params.get("dynamics_type") == "sindy":
        return torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif allow_mps and torch.backends.mps.is_available():
        # Only use MPS for large batch sizes where it's beneficial
        if batch_size is None or batch_size > 1000:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device


def setup_dataloaders(data_src, params):
    """
    Open dataset and create data loaders based on parameters.

    Args:
        data_src: "box" or "erf"
        params: Dict with batch_size and model-specific parameters

    Returns:
        train_loader, test_loader, metadata dict with (r_bins_edges, n_bins, dsd_time, x_test, m_test, x_train, m_train)
    """
    # Open dataset
    if data_src == "box":
        (
            x_train,
            m_train,
            x_test,
            m_test,
            r_bins_edges,
            n_bins,
            dsd_time,
        ) = du.open_box_dataset()
    elif data_src == "erf":
        (
            x_train,
            m_train,
            x_test,
            m_test,
            r_bins_edges,
            n_bins,
            dsd_time,
        ) = du.open_erf_dataset(sample_time=np.arange(0, 61, 5))
    else:
        raise NotImplementedError("only erf and box data options exist")

    # Create datasets based on dynamics type
    dynamics_type = params.get("dynamics_type", "autoregressive")

    if dynamics_type == "autoregressive":
        # AR uses lag-based dataset
        train_data = du.NormedBinDatasetAR(x_train, m_train, lag=params["n_lag"])
        test_data = du.NormedBinDatasetAR(x_test, m_test, lag=params["n_lag"])
    elif dynamics_type in ["sindy", "nn_dzdt"]:
        # Derivative-based dynamics use DzDt dataset
        train_data = du.NormedBinDatasetDzDt(x_train, dsd_time, m_train)
        test_data = du.NormedBinDatasetDzDt(x_test, dsd_time, m_test)
    elif dynamics_type == "none":
        # Pure autoencoder
        train_data = du.NormedBinDatasetDzDt(x_train, dsd_time, m_train)
        test_data = du.NormedBinDatasetDzDt(x_test, dsd_time, m_test)
    else:
        raise NotImplementedError(f"Unknown dynamics_type: {dynamics_type}")

    # Create data loaders
    train_loader = DataLoader(
        train_data, batch_size=params["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=len(test_data), shuffle=False
    )

    metadata = {
        "r_bins_edges": r_bins_edges,
        "n_bins": n_bins,
        "dsd_time": dsd_time,
        "x_train": x_train,
        "m_train": m_train,
        "x_test": x_test,
        "m_test": m_test,
    }

    return train_loader, test_loader, metadata

def setup_dataloaders_sed(data_src, params):
    """
    Open dataset and create data loaders based on parameters.

    Args:
        data_src: "erf"
        params: Dict with batch_size and model-specific parameters

    Returns:
        train_loader, test_loader, metadata dict with (r_bins_edges, n_bins, dsd_time, x_test, flux_test, x_train, flux_train)
    """
    # Open dataset
    if data_src == "erf":
        data = du.open_sed_datasets()
    elif data_src == "erf_mini":
        path = Path(__file__).parent.parent / "data/erf_data/sed_congestus/mini"
        data = du.open_sed_datasets(path)
    else:
        raise NotImplementedError("only erf option exists")

    # Create datasets based on dynamics type
    dynamics_type = params.get("dynamics_type", "nn_dzdt")
    if dynamics_type in ["sindy", "nn_dzdt", "none"]:
        # Derivative-based dynamics use DzDt dataset
        train_data = du.BinDatasetSed(data["x_train"], data["flux_train"], data["m_train"])
        test_data = du.BinDatasetSed(data["x_test"], data["flux_test"], data["m_test"])
    else:
        raise NotImplementedError(f"Unknown dynamics_type: {dynamics_type}")

    # Create data loaders
    train_loader = DataLoader(
        train_data, batch_size=params["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=len(test_data), shuffle=False
    )

    metadata = {
        "r_bins_edges": data["r_bins_edges"],
        "n_bins": data["n_bins"],
        "x_train": data["x_train"],
        "flux_train": data["flux_train"],
        "x_test": data["x_test"],
        "flux_test": data["flux_test"],
        "m_scale": data["m_scale"],
        "flux_scale": data["flux_scale"],
    }

    return train_loader, test_loader, metadata


def setup_dataloaders_cond(params, data):
    """
    Create data loaders for condensation models.

    Args:
        params: Parameters dict with batch_size
        data: Dict from open_congestus_5400_dataset() containing train/test data

    Returns:
        train_loader, test_loader, metadata dict
    """
    # Create datasets
    train_data = du.CondensationDataset(
        data["x_train"],
        data["dgdt_train"],
        data["temp_train"],
        data["sat_ratio_train"],
        data["m_train"],
    )

    test_data = du.CondensationDataset(
        data["x_test"],
        data["dgdt_test"],
        data["temp_test"],
        data["sat_ratio_test"],
        data["m_test"],
    )

    # Create data loaders
    train_loader = DataLoader(
        train_data, batch_size=params["batch_size"], shuffle=True
    )
    test_loader = DataLoader(
        test_data, batch_size=len(test_data), shuffle=False
    )

    # Metadata for model creation and analysis
    metadata = {
        "n_bins": data["n_bins"],
        "r_bins_edges": data["r_bins_edges"],
        "rbin_median": data["rbin_median"],
        "m_scale": data["m_scale"],
        "cond_scale": data["cond_scale"],
        "x_train": data["x_train"],
        "m_train": data["m_train"],
        "dgdt_train": data["dgdt_train"],
        "temp_train": data["temp_train"],
        "sat_ratio_train": data["sat_ratio_train"],
        "x_test": data["x_test"],
        "m_test": data["m_test"],
        "dgdt_test": data["dgdt_test"],
        "temp_test": data["temp_test"],
        "sat_ratio_test": data["sat_ratio_test"],
    }

    return train_loader, test_loader, metadata


def setup_loss_weights(params, train_data):
    """
    Set up loss weights based on dynamics type.
    Only computes automatic weights if not manually specified.

    Args:
        params: Parameters dict
        train_data: Training dataset

    Returns:
        Updated params dict with loss weights
    """
    dynamics_type = params["dynamics_type"]

    if dynamics_type in ["sindy", "nn_dzdt"]:
        # Check if weights are manually specified
        has_manual_weights = (
            "loss_weight_recon" in params
            and "loss_weight_dx" in params
            and "loss_weight_dz" in params
        )

        if has_manual_weights:
            print(
                f"Using manual loss weights - recon: {params['loss_weight_recon']}, "
                f"dx: {params['loss_weight_dx']}, dz: {params['loss_weight_dz']}"
            )
        else:
            # Use Champion et al. recommendations
            lambda1, lambda2, lambda3 = du.champion_calculate_weights(
                train_data, lambda1_metaweight=params.get("lambda1_metaweight", 0.5)
            )
            params["loss_weight_recon"] = lambda3
            params["loss_weight_dx"] = lambda1
            params["loss_weight_dz"] = lambda2
            print(
                f"Using Champion et al. loss weights - recon: {lambda3}, "
                f"dx: {lambda1:.2f}, dz: {lambda2:.2f}"
            )

    elif dynamics_type == "autoregressive":
        # Set defaults if not specified
        if "w_dx" not in params:
            params["w_dx"] = 1.0
        if "w_recon" not in params:
            params["w_recon"] = 1.0
        if "w_dz" not in params:
            params["w_dz"] = 0.1
        print(
            f"Loss weights - w_dx: {params['w_dx']}, "
            f"w_recon: {params['w_recon']}, w_dz: {params['w_dz']}"
        )

    elif dynamics_type == "none":
        # Set defaults if not specified
        if "loss_weight_l2" not in params:
            params["loss_weight_l2"] = 0.01
        print(f"Loss weights - kl: 1.0, l2: {params['loss_weight_l2']}")

    return params

def setup_optimization(model, params):
    """
    Create optimizer, learning rate scheduler, and early stopping.

    Args:
        model: PyTorch model
        params: Dict with learning_rate, wd, lr_sched, patience

    Returns:
        optimizer, scheduler, early_stopping
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=params.get("wd", 1e-3),
    )

    scheduler = None
    if params.get("lr_sched", True):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

    early_stopping = None
    if "patience" in params:
        early_stopping = diagnostics.EarlyStopping(patience=params["patience"])

    return optimizer, scheduler, early_stopping


def train_and_eval(
    model,
    train_loader,
    test_loader,
    optimizer,
    scheduler,
    loss_fn,
    params,
    device="cpu",
    early_stopping=None,
    optuna_trial=None,
):
    """
    Generic training and evaluation loop for all model types.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        loss_fn: Function that computes loss given (model, batch, params, device)
                 and returns (loss, loss_dict) where loss_dict has named losses
        params: Training parameters dict with num_epochs, print_frequency
        device: Compute device
        early_stopping: EarlyStopping object (optional)
        optuna_trial: Optuna trial for hyperparameter optimization (optional)

    Returns:
        best_model, losses dict with arrays for each loss component
    """
    model.to(device)
    n_epochs = params["num_epochs"]
    print_frequency = params.get("print_frequency", 1)

    # Initialize loss storage
    losses = {}
    test_losses = {}
    best_test_loss = float("inf")
    best_model = None

    for epoch in range(n_epochs):
        # Training phase
        epoch_start_time = time.time()
        model.train()
        epoch_losses = {}

        for batch in train_loader:
            # Move batch to device
            batch = tuple(b.to(device) if isinstance(b, torch.Tensor) else b for b in batch)

            # Forward pass and loss computation
            loss, loss_dict = loss_fn(model, batch, params, device)

            # Track losses
            for key, value in loss_dict.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value.item() if isinstance(value, torch.Tensor) else value)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            optimizer.step()

        # Average training losses for this epoch
        for key, values in epoch_losses.items():
            if key not in losses:
                losses[key] = np.zeros(n_epochs) * np.nan
            losses[key][epoch] = np.mean(values)

        # Evaluation phase
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = tuple(b.to(device) if isinstance(b, torch.Tensor) else b for b in batch)
                test_loss, test_loss_dict = loss_fn(model, batch, params, device)

        # Store test losses
        for key, value in test_loss_dict.items():
            test_key = f"test_{key}"
            if test_key not in test_losses:
                test_losses[test_key] = np.zeros(n_epochs) * np.nan
            test_losses[test_key][epoch] = value.item() if isinstance(value, torch.Tensor) else value

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model = copy.deepcopy(model)

        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

        # Print progress
        epoch_end_time = time.time()
        if epoch % print_frequency == 0:
            print(
                f"Epoch [{epoch}/{n_epochs}], "
                f"Train Loss: {losses['total'][epoch]:.4f} | "
                f"Test Loss: {test_losses['test_total'][epoch]:.4f} | "
                f"LR: {scheduler.get_last_lr() if scheduler else [optimizer.param_groups[0]['lr']]} | "
                f"Epoch Time: {epoch_end_time - epoch_start_time:.2f} s"
            )

            # Print component losses if they exist
            loss_components = [k for k in losses.keys() if k != 'total']
            if loss_components:
                comp_str = " | ".join([f"{k}: {losses[k][epoch]:.2e}" for k in loss_components])
                print(f"  {comp_str}")

        # Optuna reporting
        if optuna_trial is not None:
            optuna_trial.report(losses['total'][epoch], epoch)
            if optuna_trial.should_prune():
                import optuna
                raise optuna.exceptions.TrialPruned()

        # Early stopping
        if early_stopping is not None:
            early_stopping(test_loss)
            if early_stopping.early_stop:
                print("Training stopped early.")
                break

    # Combine train and test losses into single dict
    all_losses = {**losses, **test_losses}

    return best_model, all_losses
