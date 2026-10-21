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
import matplotlib.pyplot as plt

from src import data_utils, model_factory, recon_condensation_losses, save_utils, training_utils, plotting

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
    "data_src": "congestus_5400",
    "test_size": 0.2,

    # Training
    "random_seed": 1,
    "num_epochs": 100,
    "batch_size": 500,
    "learning_rate": 1e-3,
    "wd": 1e-3,
    "lr_sched": True,
    "patience": 50,
    "print_frequency": 1,

    # Model parameters
    "latent_dim": 3,

    # Loss parameters
    "tol": 1e-8,
    "cond_loss_type": "erf",  # "erf", "theory", or "none" (pure autoencoder)

    # Loss weights for condensation-aware autoencoder
    # - loss_weight_recon: DSD reconstruction (KL divergence)
    # - loss_weight_cond_dx: Projected condensation in physical space
    "loss_weight_recon": 1.0,
    "loss_weight_cond_dx": 1e4,

    # Output
    "save": True,
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
    data = data_utils.open_congestus_5400_dataset(
        test_size=params["test_size"],
        random_state=params["random_seed"]
    )
    print(f"Training samples: {data['x_train'].shape[0]}")
    print(f"Test samples: {data['x_test'].shape[0]}")

    # Create dataloaders
    train_loader, test_loader, metadata = training_utils.setup_dataloaders_cond(
        params, data
    )

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
    params["rbin_median"] = metadata["rbin_median"]
    params["m_scale"] = metadata["m_scale"]
    params["cond_scale"] = metadata["cond_scale"]

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
        test_loader,
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

        # Additional plots if requested
        if params.get("plot", True):
            print("\nGenerating analysis plots...")

            # Select random test samples for visualization
            n_samples = min(5, len(metadata["x_test"]))
            test_ids = np.random.choice(len(metadata["x_test"]), n_samples, replace=False)

            # Plot 1: DSD reconstructions
            print("  - Plotting reconstructions...")
            fig = plotting.plot_reconstructions(
                best_model,
                test_ids,
                metadata["x_test"][:, np.newaxis, :],
                metadata["r_bins_edges"]
            )
            fig.savefig(output_dir / "reconstructions.png", dpi=150, bbox_inches='tight')
            if params["show_plots"]:
                fig.show()
            plt.close(fig)

            # Plot 2: NWI weights (if applicable)
            if params['encoder_type'] == "nwi":
                print("  - Plotting NWI weights...")
                fig = plotting.plot_nnwi_weights(best_model, metadata['r_bins_edges'])
                fig.savefig(output_dir / "weights.png", dpi=150, bbox_inches='tight')
                if params['show_plots']:
                    fig.show()
                plt.close(fig)

            # Plot 3: Condensation tendency projections (if condensation loss is used)
            if params["cond_loss_type"] != "none":
                print("  - Plotting condensation projections...")
                fig = plotting.plot_condensation_projections(
                    best_model,
                    test_ids,
                    metadata["x_test"],
                    metadata["dgdt_test"],
                    metadata["temp_test"],
                    metadata["sat_ratio_test"],
                    metadata["r_bins_edges"],
                    metadata["m_scale"],
                    metadata["cond_scale"]
                )
                fig.savefig(output_dir / "condensation_projections.png", dpi=150, bbox_inches='tight')
                if params["show_plots"]:
                    fig.show()
                plt.close(fig)

                # Plot 4: Theory comparison (regardless of training loss type)
                print("  - Plotting theory comparison...")
                fig = plotting.plot_condensation_theory_comparison(
                    best_model,
                    test_ids,
                    metadata["x_test"],
                    metadata["dgdt_test"],
                    metadata["temp_test"],
                    metadata["sat_ratio_test"],
                    metadata["r_bins_edges"],
                    metadata["rbin_median"],
                    metadata["m_scale"],
                    metadata["cond_scale"]
                )
                fig.savefig(output_dir / "theory_comparison.png", dpi=150, bbox_inches='tight')
                if params["show_plots"]:
                    fig.show()
                plt.close(fig)

                # Plot 5: Latent space condensation
                print("  - Plotting latent space...")
                fig = plotting.plot_condensation_latent_space(
                    best_model,
                    metadata["x_test"],
                    metadata["dgdt_test"],
                    metadata["temp_test"],
                    metadata["sat_ratio_test"],
                    metadata["rbin_median"],
                    metadata["m_scale"],
                    metadata["cond_scale"],
                    params["cond_loss_type"]
                )
                fig.savefig(output_dir / "latent_space.png", dpi=150, bbox_inches='tight')
                if params["show_plots"]:
                    fig.show()
                plt.close(fig)

            print(f"\nAll plots saved to: {output_dir}/")

    print("\nTraining complete!")
