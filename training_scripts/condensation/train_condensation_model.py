"""
Unified training script for condensation surrogate models.
Supports: AE-SINDy, AE-NNdzdt, and pure autoencoder (NNWI).
Two loss variants: ERF (train against simulated condensation) and theory (train against theoretical condensation).
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import numpy as np
import torch

from src import model_factory, recon_condensation_losses, save_utils, training_utils

# Parameters - configure these for your desired model
params = {
    # Process type
    "process": "condensation",

    # Model architecture
    # NOTE: Condensation models train ONLY the autoencoder (encoder + decoder).
    # There is NO separate dynamics model. Condensation tendencies are either from
    # ERF data or computed theoretically, then encoded and projected through the decoder.
    "encoder_type": "nwi",  # "ffnn" or "nwi"
    "decoder_type": "nwi_simple",  # "ffnn", "nwi_simple", or "nwi_deep"
    "dynamics_type": "none",  # Must be "none" for condensation

    # NWI-specific (only used if encoder_type="nwi" or decoder_type contains "nwi")
    "num_blocks": 3,
    "hidden_size": 128,

    # Data
    # Training
    "random_seed": 1,
    "num_epochs": 20,
    "batch_size": 500,
    "learning_rate": 1e-3,
    "wd": 1e-3,
    "lr_sched": True,
    "patience": 50,
    "print_frequency": 1,

    # Model parameters
    "latent_dim": 3,
    "number_aware": True,

    # Loss parameters
    "tol": 1e-8,
    "cond_loss_type": "theory",  # "erf", "theory", or "none" (pure autoencoder)
    # Target the projected tendency is compared against:
    #   "eulerian"   bin tendency dg/dt = -d(vg)/dlnr + 3vg, consistent with compute_dhdt
    #   "lagrangian" 3vg / ERF dgdt_cond as-is (previous behaviour; inconsistent with compute_dhdt)
    "cond_target": "eulerian",
    "cond_theory_scheme": "central",  # FV scheme for the eulerian theory target

    # Loss weights for condensation-aware autoencoder
    # - loss_weight_recon: DSD reconstruction (KL divergence)
    # - loss_weight_cond_dx: Projected condensation in physical space
    "loss_weight_recon": 1.0,
    "loss_weight_mass_recon": 1.0,
    "loss_weight_number_recon": 1.0,
    "loss_weight_cond_dx": 1e5,

    # Output
    "save": True,
    "plot": True,
    "show_plots": True,
}


if __name__ == "__main__":
    # Setup
    torch.manual_seed(params["random_seed"])
    np.random.seed(params["random_seed"])

    device = training_utils.setup_device()
    print(f"Using {device} device")
    print(
        f"Training model: {params['encoder_type']}/{params['decoder_type']}/{params['dynamics_type']}"
    )
    print(f"Loss variant: {params['cond_loss_type']}")

    # Load data
    print("\nLoading condensation dataset...")
    loaders, metadata = training_utils.setup_all_processes_dataloaders(params)
    train_loader = loaders["train"]
    # Use validation for model selection; preserve the held-out test split.
    val_loader = loaders["validate"]
    print(f"Training samples: {metadata['split_sizes']['train']}")
    print(f"Validation samples: {metadata['split_sizes']['validate']}")
    print(f"Held-out test samples: {metadata['split_sizes']['test']}")
    params.update(metadata["normalization"])
    params["dlnr"] = metadata["dlnr"]
    params["rbin_median"] = metadata["rbin_median"]
    params["spectra_representation"] = metadata["spectra_representation"]

    # Number-aware models register the geometric-mean radii used to diagnose
    # number DSDs, so these must be available when the model is constructed.
    params["rbin_median"] = metadata["rbin_median"]

    # Create model using factory
    model = model_factory.create_model(
        params["encoder_type"],
        params["decoder_type"],
        params["dynamics_type"],
        params,
        metadata["n_bins"],
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params} (condensation-aware autoencoder, no separate dynamics model)")

    # Add physical parameters to params for loss functions
    params["rbin_eval"] = metadata.get("rbin_eval", "center")

    # Print loss configuration
    print(f"\nLoss configuration:")
    print(f"  Condensation loss type: {params['cond_loss_type']}")
    print(f"  loss_weight_recon:      {params['loss_weight_recon']:.6f}")
    print(f"  loss_weight_cond_dx:    {params['loss_weight_cond_dx']:.6f}")

    # Get loss function
    loss_fn = recon_condensation_losses.get_loss_function(
        params["cond_loss_type"]
    )

    # Setup optimization
    optimizer, scheduler, early_stopping = training_utils.setup_optimization(
        model, params
    )

    # Train
    print("\nStarting training...")
    best_model, losses = training_utils.train_and_eval(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        params,
        device,
        early_stopping=early_stopping,
    )

    # Save and plot
    if params["save"]:
        output_dir, case_name, timestamp = save_utils.setup_output_dir(params)
        print(f"\nOutput directory: {output_dir}")

        # Prepare params for saving (remove non-JSON-serializable objects)
        params_to_save = params.copy()
        # Convert numpy arrays to lists or remove runtime-only params
        if "rbin_median" in params_to_save:
            params_to_save["rbin_median"] = params_to_save["rbin_median"].tolist()

        # Save model
        save_utils.save_model_artifacts(
            best_model, losses, params_to_save, output_dir, timestamp
        )
        print(f"Model saved to: {output_dir}/{case_name}.pth")

        print(f"Losses saved to: {output_dir}/{case_name}_losses.npz")

        # Plot losses
        save_utils.plot_training_losses(losses, params, output_dir)
        print(f"Loss plot saved to: {output_dir}/losses.png")

        if params.get("plot", True):
            print(
                "Combined-data plotting is limited to training losses for now; "
                "additional condensation plots have not yet been updated."
            )

    print("\nTraining complete!")
