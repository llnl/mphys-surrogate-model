"""
Export pytorch weights to fortran compatible matrices with ftorch
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import torch
from src import model_factory

if __name__ == "__main__":
    model_dir = "/Users/dejong5/Documents/mphys-surrogate-model/trained_models/sedimentation/nwi_nwi_simple_nn_dzdt/20260610_214622_erf_latent3_ep250_lr1e-03_bs1000_w1.0-0.1-100000/model.pth"
    model = model_factory.create_model(
        "nwi",
        "nwi_simple",
        "nn_dzdt",
        {"latent_dim": 3, "process": "sedimentation"},
        64,
    )
    model.load_state_dict(torch.load(model_dir, map_location=torch.device("cpu")))

    model.eval()

    enc = model.encoder
    wf = model.encoder.wf_mat()
    dec = model.decoder
    deriv = model.dzdt

    # example input: 64 bins
    example_x = torch.randn(1, 64)
    example_z = torch.randn(1, 4)
    example_l = torch.randn(1, 4)

    traced_encoder = torch.jit.trace(enc, example_x)
    traced_encoder.save("../../data/ftorch_weights/nwi_encoder.pt")
    print("saved encoder")
    traced_decoder = torch.jit.trace(dec, example_l)
    traced_decoder.save("../../data/ftorch_weights/nwi_decoder.pt")
    print("saved decoder")
    traced_dzdt = torch.jit.trace(deriv, example_z)
    traced_dzdt.save("../../data/ftorch_weights/sed_dzdt.pt")
    print("saved dzdt")
