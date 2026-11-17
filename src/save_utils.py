"""
Unified save and plotting utilities for all model types.
Provides consistent output directory structure and model artifact saving.
"""

import copy
import json
import os
import warnings
import pickle as pkl
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src import diagnostics, plotting
from src.constants import LOG_TOLERANCE


def generate_case_name(params):
    """
    Generate a consistent case name for saving models.

    Args:
        params: Dict with model parameters

    Returns:
        Case name string with format: {datetime}_{data}_{hyperparams}
    """
    # Generate datetime ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dynamics_type = params.get("dynamics_type", "autoregressive")
    data_src = params.get("data_src", "erf")

    # Build case name starting with date
    base_name = f"{timestamp}_{data_src}_latent{params['latent_dim']}"

    # Add dynamics-specific parameters
    if dynamics_type == "sindy":
        case_name = base_name + "_order{}_ep{}_lr{:.0e}_bs{}_w{:.1f}-{:.1f}-{:.0f}".format(
            params.get("poly_order", 2),
            params["num_epochs"],
            params["learning_rate"],
            params["batch_size"],
            params.get("loss_weight_recon", 1.0),
            params.get("loss_weight_dz", 1.0),
            params.get("loss_weight_dx", 1.0),
        )
    elif dynamics_type == "nn_dzdt":
        case_name = base_name + "_ep{}_lr{:.0e}_bs{}_w{:.1f}-{:.1f}-{:.0f}".format(
            params["num_epochs"],
            params["learning_rate"],
            params["batch_size"],
            params.get("loss_weight_recon", 1.0),
            params.get("loss_weight_dz", 1.0),
            params.get("loss_weight_dx", 1.0),
        )
    elif dynamics_type == "autoregressive":
        case_name = base_name + "_lag{}_ep{}_lr{:.0e}_bs{}_w{:.1f}-{:.1f}-{:.1f}".format(
            params.get("n_lag", 1),
            params["num_epochs"],
            params["learning_rate"],
            params["batch_size"],
            params.get("w_dx", 1.0),
            params.get("w_recon", 1.0),
            params.get("w_dz", 1.0),
        )
    elif dynamics_type == "none":
        # Pure autoencoder
        case_name = base_name + "_ep{}_lr{:.0e}_bs{}_w{:.1f}-{:.2f}".format(
            params["num_epochs"],
            params["learning_rate"],
            params["batch_size"],
            1.0,  # KL weight (always 1)
            params.get("loss_weight_l2", 0.01),
        )
    else:
        # Generic fallback
        warnings.warn("Unknown dynamics type '{}', using generic case name format.".format(dynamics_type))
        case_name = base_name + "_ep{}_lr{:.0e}_bs{}".format(
            params["num_epochs"],
            params["learning_rate"],
            params["batch_size"],
        )

    return case_name


def setup_output_dir(params):
    """
    Create unified output directory structure.

    Args:
        params: Dict with model parameters including encoder_type, decoder_type, dynamics_type

    Returns:
        Tuple of (output_dir Path, case_name string, timestamp string)
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    process_root = params["process"] if "process" in params else ""
    case_name = generate_case_name(params)
    timestamp = case_name.split("_")[0]  # Extract timestamp from case_name

    encoder_type = params.get("encoder_type", "ffnn")
    decoder_type = params.get("decoder_type", "ffnn")
    dynamics_type = params.get("dynamics_type", "autoregressive")

    # Build directory name: {encoder}_{decoder}_{dynamics} (no "ae" prefix)
    if dynamics_type == "none":
        # Pure autoencoder (no dynamics)
        base_dir = f"{encoder_type}_{decoder_type}"
    else:
        base_dir = f"{encoder_type}_{decoder_type}_{dynamics_type}"

    output_dir = Path(project_root + "/trained_models") / process_root / base_dir / case_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir, case_name, timestamp


def save_model_artifacts(model, losses, params, output_dir, timestamp):
    """
    Save model weights, parameters, and losses.

    Args:
        model: Trained model
        losses: Dict of loss arrays
        params: Dict with model parameters
        output_dir: Path to save directory
        timestamp: Timestamp string for file naming (unused, kept for API compatibility)
    """
    # Move model to CPU for saving
    model.eval()
    model = model.to("cpu")

    # Save losses
    with open(output_dir / "losses.pkl", "wb") as pickle_file:
        pkl.dump(losses, pickle_file)

    # Save params (convert numpy types to Python types for JSON)
    params_save = copy.deepcopy(params)
    for key, value in params_save.items():
        if isinstance(value, (np.float32, np.float64)):
            params_save[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)):
            params_save[key] = int(value)
        elif isinstance(value, np.ndarray):
            params_save[key] = value.tolist()

    params_file = output_dir / "params.json"
    params_file.write_text(json.dumps(params_save, indent=4))

    # Save model
    model_file = output_dir / "model.pth"
    torch.save(model.state_dict(), model_file)

    print(f"Saved model artifacts to {output_dir}/")


def plot_training_losses(losses, params, output_dir):
    """
    Plot and save training losses based on dynamics type.

    Args:
        losses: Dict of loss arrays
        params: Dict with model parameters
        output_dir: Path to save directory
    """
    dynamics_type = params.get("dynamics_type", "autoregressive")
    save = params.get("save", True)
    show_plots = params.get("show_plots", False)

    # Prepare sub-losses and labels based on dynamics type
    sub_losses = []
    labels = []

    for key, value in losses.items():
        if key[0:5] != "total" and key[0:4] != "test":
            labels.append(key)
            sub_losses.append(value)

    # Plot
    fig = plotting.plot_losses(
        losses.get("total", []),
        test_losses=losses.get("test_total", []),
        sub_losses=sub_losses if sub_losses else None,
        labels=labels if labels else None,
        title="Training Loss",
    )

    if save:
        fig.savefig(output_dir / "losses.png")
    if show_plots:
        fig.show()

    return fig


def generate_plots(model, metadata, params, output_dir):
    """
    Generate and save plots based on model type.

    Args:
        model: Trained model
        metadata: Dict with data arrays (x_train, x_test, m_train, m_test, r_bins_edges, dsd_time, n_bins)
        params: Dict with model parameters
        output_dir: Path to save directory
    """
    dynamics_type = params.get("dynamics_type", "autoregressive")
    encoder_type = params.get("encoder_type", "ffnn")
    save = params.get("save", True)
    show_plots = params.get("show_plots", False)

    x_test = metadata["x_test"]
    x_train = metadata["x_train"]
    m_test = metadata["m_test"]
    m_train = metadata["m_train"]
    r_bins_edges = metadata["r_bins_edges"]
    dsd_time = metadata["dsd_time"]

    test_ids = [0, 10, 20, 30]
    tplt = [0, 5, -1]

    # Plot distributions: reconstruction
    fig = plotting.plot_reconstructions(
        model,
        test_ids,
        x_test,
        r_bins_edges,
    )
    if save:
        fig.savefig(output_dir / "reconstructions.png")
    if show_plots:
        fig.show()

    # Dynamics-specific plots
    if dynamics_type in ["sindy", "nn_dzdt"]:
        # Predictions: Multi time step
        fig = plotting.plot_predictions_dzdt(
            test_ids,
            tplt,
            params["latent_dim"],
            model,
            dsd_time,
            x_test,
            m_test,
            x_train,
            m_train,
            r_bins_edges,
        )
        if save:
            fig.savefig(output_dir / "predictions.png")
        if show_plots:
            fig.show()

        # Plot trajectories of latent variables
        from src.data_utils import NormedBinDatasetDzDt
        test_data = NormedBinDatasetDzDt(x_test, dsd_time, m_test)

        z_pred, z_data, x_pred = diagnostics.get_latent_trajectories_dzdt(
            params["latent_dim"],
            model,
            test_data.t,
            x_test,
            m_test,
            x_train,
            m_train,
        )
        fig = plotting.plot_latent_trajectories(
            params["latent_dim"], test_data.t, z_pred, z_data
        )
        if save:
            fig.savefig(output_dir / "trajectories.png")
        if show_plots:
            fig.show()

        # Full test set performance
        fig = plotting.plot_full_testset_performance_recon(
            model, x_test, params.get("tol", LOG_TOLERANCE)
        )
        if save:
            fig.savefig(output_dir / "full_test_recon.png")
        if show_plots:
            fig.show()

        test_kl, test_wass, test_wun, test_mass_diff = diagnostics.get_performance_metrics(
            x_test, m_test, z_pred, x_pred
        )
        fig = plotting.plot_full_testset_performance_pred(test_kl, test_wass, test_mass_diff, dsd_time)
        if save:
            fig.savefig(output_dir / "full_test_pred.png")
        if show_plots:
            fig.show()

        # Plot quantiles
        fig = plotting.plot_testset_quantiles_pred(
            x_test, x_pred, test_wass, tplt, dsd_time, r_bins_edges
        )
        if save:
            fig.savefig(output_dir / "quantiles_test_pred.png")
        if show_plots:
            fig.show()

    elif dynamics_type == "autoregressive":
        # AR-specific plots
        from src.data_utils import NormedBinDatasetAR
        test_data = NormedBinDatasetAR(x_test, m_test, lag=params["n_lag"])

        # Predictions
        fig = plotting.plot_predictions_AE_AR(
            model,
            test_ids,
            dsd_time,
            tplt,
            x_test,
            m_test,
            r_bins_edges,
            n_lag=params["n_lag"]
        )
        if save:
            fig.savefig(output_dir / "predictions.png")
        if show_plots:
            fig.show()

        # Latent trajectories
        z_pred, z_data, x_pred = diagnostics.get_latent_trajectories_AR(
            params["latent_dim"],
            model,
            dsd_time,
            x_test,
            m_test,
            n_lag=params["n_lag"]
        )
        fig = plotting.plot_latent_trajectories(
            params["latent_dim"], dsd_time, z_pred, z_data
        )
        if save:
            fig.savefig(output_dir / "trajectories.png")
        if show_plots:
            fig.show()

        # Full test set performance
        fig = plotting.plot_full_testset_performance_recon(
            model, x_test, params.get("tol", LOG_TOLERANCE)
        )
        if save:
            fig.savefig(output_dir / "full_test_recon.png")
        if show_plots:
            fig.show()

        test_kl, test_wass, test_wun, test_mass_diff = diagnostics.get_performance_metrics(
            x_test, m_test, z_pred, x_pred
        )
        fig = plotting.plot_full_testset_performance_pred(test_kl, test_wass, test_mass_diff, dsd_time)
        if save:
            fig.savefig(output_dir / "full_test_pred.png")
        if show_plots:
            fig.show()

        # Quantiles
        fig = plotting.plot_testset_quantiles_pred(
            x_test, x_pred, test_wass, tplt, dsd_time, r_bins_edges
        )
        if save:
            fig.savefig(output_dir / "quantiles_test_pred.png")
        if show_plots:
            fig.show()

    # Plot latent space (generic for all models)
    fig = plotting.viz_3d_latent_space(
        model,
        x_test,
        dsd_time,
    )
    if save:
        fig.write_html(output_dir / "latent_space.html")
    if show_plots:
        fig.show()

    # Plot NWI weights if applicable
    if encoder_type == "nwi":
        fig = plotting.plot_nnwi_weights(model, r_bins_edges)
        if save:
            fig.savefig(output_dir / "weights.png")
        if show_plots:
            fig.show()
