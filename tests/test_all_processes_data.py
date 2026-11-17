"""Tests for the combined ERF all-processes data pipeline.

Run directly in the project environment with:

    conda run -n learning python tests/test_all_processes_data.py

The tests use a small synthetic dataset so they do not load the full ERF file.
"""

import unittest
from pathlib import Path
import sys

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_utils import (
    AllProcessesDataset,
    DEFAULT_ALL_PROCESSES_PATH,
    denormalize_all_processes,
    open_all_processes_dataset,
    prepare_all_processes_dataset,
)
from src.training_utils import setup_all_processes_dataloaders


M_SCALE = 2.0
N_SCALE = 10.0
COAL_RATE_SCALE = 0.25
V_SCALE = 4.0
REPORT_PERCENTILES = (0, 1, 25, 50, 75, 99, 100)


def print_distribution_table(title, fields):
    """Print min, selected percentiles, median, and max for named arrays."""
    headings = ("min", "p01", "p25", "median", "p75", "p99", "max")
    print(f"\n{title}")
    print(f"{'variable':<28}" + "".join(f"{name:>14}" for name in headings))
    print("-" * (28 + 14 * len(headings)))
    for name, values in fields.items():
        percentiles = np.percentile(np.asarray(values), REPORT_PERCENTILES)
        print(
            f"{name:<28}"
            + "".join(f"{value:>14.6e}" for value in percentiles)
        )


def make_dataset():
    """Create nonuniform log-radius bins to exercise the dlnr weighting."""
    split = np.array(["train", "val", "test", "train"])
    rbin_l = np.array([1.0, 2.0, 8.0])
    rbin_r = np.array([2.0, 8.0, 16.0])
    dmdlnr = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 1.0],
            [3.0, 1.0, 2.0],
            [4.0, 2.0, 4.0],
        ]
    )
    dndlnr = np.array(
        [
            [2.0, 1.0, 4.0],
            [1.0, 3.0, 2.0],
            [4.0, 2.0, 1.0],
            [8.0, 1.0, 2.0],
        ]
    )
    dgdt_cond = np.array(
        [
            [-0.10, 0.20, 0.30],
            [-0.20, 0.40, 0.10],
            [-0.30, 0.10, 0.20],
            [-0.40, 0.20, 0.40],
        ]
    )
    dgdt_coal = np.array(
        [
            [-0.05, 0.02, 0.03],
            [-0.10, 0.04, 0.06],
            [-0.15, 0.06, 0.09],
            [-0.20, 0.08, 0.12],
        ]
    )
    vt_mass_flux = np.array(
        [
            [0.5, 1.0, 1.5],
            [1.0, 2.0, 0.5],
            [1.5, 0.5, 1.0],
            [2.0, 1.0, 2.0],
        ]
    )
    return xr.Dataset(
        data_vars={
            "split": ("sample", split),
            "dmdlnr": (("sample", "bin"), dmdlnr),
            "dndlnr": (("sample", "bin"), dndlnr),
            "dgdt_cond": (("sample", "bin"), dgdt_cond),
            "vt_mass_flux": (("sample", "bin"), vt_mass_flux),
            "dgdt_coal": (("sample", "bin"), dgdt_coal),
            "temp": ("sample", np.array([280.0, 281.0, 282.0, 283.0])),
            "rh": ("sample", np.array([0.90, 1.00, 1.10, 1.20])),
            "rbin_l": ("bin", rbin_l),
            "rbin_r": ("bin", rbin_r),
        }
    )


def prepare(ds=None):
    if ds is None:
        ds = make_dataset()
    return prepare_all_processes_dataset(
        ds,
        M_scale=M_SCALE,
        N_scale=N_SCALE,
        coal_rate_scale=COAL_RATE_SCALE,
        v_scale=V_SCALE,
    )


class TestAllProcessesData(unittest.TestCase):
    def setUp(self):
        self.source = make_dataset()
        self.result = prepare(self.source)

    def test_split_names_sizes_and_source_indices(self):
        expected_indices = {
            "train": np.array([0, 3]),
            "validate": np.array([1]),
            "test": np.array([2]),
        }
        self.assertEqual(
            self.result["metadata"]["split_sizes"],
            {"train": 2, "validate": 1, "test": 1},
        )
        self.assertEqual(set(self.result["datasets"]), set(expected_indices))
        for split, indices in expected_indices.items():
            np.testing.assert_array_equal(
                self.result["arrays"][split]["sample_index"], indices
            )

    def test_total_mass_and_number_include_dlnr(self):
        dlnr = np.log(self.source["rbin_r"].values / self.source["rbin_l"].values)
        expected_M = (self.source["dmdlnr"].values * dlnr).sum(axis=1)
        expected_N = (self.source["dndlnr"].values * dlnr).sum(axis=1)
        train = self.result["arrays"]["train"]

        np.testing.assert_allclose(
            train["M"], expected_M[[0, 3]] / M_SCALE, rtol=1e-6
        )
        np.testing.assert_allclose(
            train["N"], expected_N[[0, 3]] / N_SCALE, rtol=1e-6
        )
        np.testing.assert_allclose(self.result["metadata"]["dlnr"], dlnr)

    def test_normalized_spectra_are_bin_fractions_that_sum_to_one(self):
        for arrays in self.result["arrays"].values():
            np.testing.assert_allclose(
                arrays["dmdlnr"].sum(axis=-1), 1.0, rtol=1e-6, atol=1e-7
            )
            np.testing.assert_allclose(
                arrays["dndlnr"].sum(axis=-1), 1.0, rtol=1e-6, atol=1e-7
            )
            self.assertTrue(np.all(arrays["dmdlnr"] >= 0.0))
            self.assertTrue(np.all(arrays["dndlnr"] >= 0.0))
        self.assertEqual(
            self.result["metadata"]["spectra_representation"],
            "normalized_bin_fraction",
        )

    def test_fixed_process_scales_and_physical_fields(self):
        arrays = self.result["arrays"]["validate"]
        source_index = 1
        physical_M = (
            self.source["dmdlnr"].values[source_index]
            * self.result["metadata"]["dlnr"]
        ).sum()

        np.testing.assert_allclose(
            arrays["dgdt_coal"],
            self.source["dgdt_coal"].values[[source_index]]
            / (physical_M * COAL_RATE_SCALE),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            arrays["vt_mass_flux"],
            self.source["vt_mass_flux"].values[[source_index]]
            / (physical_M * V_SCALE),
            rtol=1e-6,
        )
        np.testing.assert_array_equal(arrays["temp"], [281.0])
        np.testing.assert_array_equal(arrays["rh"], [1.0])
        np.testing.assert_allclose(
            arrays["dgdt_cond"],
            self.source["dgdt_cond"].values[[source_index]],
        )
        self.assertEqual(
            self.result["metadata"]["normalization"],
            {
                "M_scale": M_SCALE,
                "N_scale": N_SCALE,
                "coal_rate_scale": COAL_RATE_SCALE,
                "v_scale": V_SCALE,
            },
        )

    def test_numpy_denormalization_round_trip(self):
        split_indices = {
            "train": np.array([0, 3]),
            "validate": np.array([1]),
            "test": np.array([2]),
        }
        for split, indices in split_indices.items():
            normalized = self.result["arrays"][split]
            physical = denormalize_all_processes(normalized, self.result["metadata"])

            for field in (
                "dmdlnr",
                "dndlnr",
                "dgdt_cond",
                "dgdt_coal",
                "vt_mass_flux",
                "temp",
                "rh",
            ):
                np.testing.assert_allclose(
                    physical[field],
                    self.source[field].values[indices],
                    rtol=2e-6,
                    atol=1e-7,
                )

            dlnr = self.result["metadata"]["dlnr"]
            expected_M = (self.source["dmdlnr"].values[indices] * dlnr).sum(axis=1)
            expected_N = (self.source["dndlnr"].values[indices] * dlnr).sum(axis=1)
            np.testing.assert_allclose(physical["M"], expected_M, rtol=2e-6)
            np.testing.assert_allclose(physical["N"], expected_N, rtol=2e-6)

            # The helper returns a new mapping and does not replace normalized data.
            self.assertIsNot(physical, normalized)
            np.testing.assert_allclose(normalized["dmdlnr"].sum(axis=1), 1.0)

    def test_torch_dataset_batch_denormalization_round_trip(self):
        normalized_arrays = self.result["arrays"]["train"]
        dataset = AllProcessesDataset(normalized_arrays)
        batch = torch.utils.data.default_collate([dataset[0], dataset[1]])
        physical = denormalize_all_processes(batch, self.result["metadata"])

        self.assertTrue(all(torch.is_tensor(value) for value in physical.values()))
        np.testing.assert_allclose(
            physical["dmdlnr"].numpy(),
            self.source["dmdlnr"].values[[0, 3]],
            rtol=2e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            physical["dndlnr"].numpy(),
            self.source["dndlnr"].values[[0, 3]],
            rtol=2e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            physical["dgdt_coal"].numpy(),
            self.source["dgdt_coal"].values[[0, 3]],
            rtol=2e-6,
            atol=1e-7,
        )

    def test_rejects_invalid_scale(self):
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            prepare_all_processes_dataset(make_dataset(), M_scale=0.0)
        with self.assertRaisesRegex(ValueError, "positive and finite"):
            prepare_all_processes_dataset(make_dataset(), coal_rate_scale=np.nan)

    def test_rejects_nonpositive_total_spectrum(self):
        source = make_dataset()
        source["dndlnr"].values[0] = 0.0
        with self.assertRaisesRegex(ValueError, "positive total mass M and number N"):
            prepare(source)

    def test_denormalization_requires_all_normalized_fields(self):
        incomplete = dict(self.result["arrays"]["test"])
        del incomplete["dgdt_coal"]
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            denormalize_all_processes(incomplete, self.result["metadata"])

    def test_full_dataset_distribution_report(self):
        """Load the real ERF data and report physical and scaled distributions.

        This is intentionally an integration/reporting test rather than a test
        of exact percentile values. It catches non-finite preprocessing output,
        verifies spectrum normalization, and makes scale changes visible when
        the test is run directly.
        """
        if not DEFAULT_ALL_PROCESSES_PATH.exists():
            self.skipTest(f"Full dataset not found: {DEFAULT_ALL_PROCESSES_PATH}")

        result = open_all_processes_dataset()
        metadata = result["metadata"]
        split_order = ("train", "validate", "test")
        scaled = {
            field: np.concatenate(
                [result["arrays"][split][field] for split in split_order], axis=0
            )
            for field in (
                "dmdlnr",
                "dndlnr",
                "M",
                "N",
                "temp",
                "rh",
                "dgdt_cond",
                "dgdt_coal",
                "vt_mass_flux",
            )
        }

        # Recovering physical fields through the public inverse operation also
        # exercises the complete real-data normalization round trip.
        physical_splits = {
            split: denormalize_all_processes(result["arrays"][split], metadata)
            for split in split_order
        }
        physical = {
            field: np.concatenate(
                [physical_splits[split][field] for split in split_order], axis=0
            )
            for field in scaled
        }

        print(f"\nFull dataset: {DEFAULT_ALL_PROCESSES_PATH}")
        print(f"Split sizes: {metadata['split_sizes']}")
        print(f"Normalization constants: {metadata['normalization']}")
        print_distribution_table(
            "Original physical distributions",
            {
                "dmdlnr [kg m-3]": physical["dmdlnr"],
                "dndlnr [m-3]": physical["dndlnr"],
                "M [kg m-3]": physical["M"],
                "N [m-3]": physical["N"],
                "temperature [K]": physical["temp"],
                "RH": physical["rh"],
                "dgdt_cond": physical["dgdt_cond"],
                "dgdt_coal": physical["dgdt_coal"],
                "vt_mass_flux": physical["vt_mass_flux"],
            },
        )
        print_distribution_table(
            "Scaled distributions",
            {
                "mass bin fraction": scaled["dmdlnr"],
                "number bin fraction": scaled["dndlnr"],
                "M / M_scale": scaled["M"],
                "N / N_scale": scaled["N"],
                "temperature [K]": scaled["temp"],
                "RH": scaled["rh"],
                "dgdt_cond (physical)": scaled["dgdt_cond"],
                "scaled dgdt_coal": scaled["dgdt_coal"],
                "scaled vt_mass_flux": scaled["vt_mass_flux"],
            },
        )

        expected_samples = sum(metadata["split_sizes"].values())
        self.assertEqual(expected_samples, 89_497)
        for name, values in scaled.items():
            self.assertTrue(np.all(np.isfinite(values)), name)
        np.testing.assert_allclose(
            scaled["dmdlnr"].sum(axis=1), 1.0, rtol=1e-6, atol=1e-6
        )
        np.testing.assert_allclose(
            scaled["dndlnr"].sum(axis=1), 1.0, rtol=1e-6, atol=1e-6
        )

    def test_full_dataset_dataloaders(self):
        """Exercise the public DataLoader setup with the real ERF dataset."""
        if not DEFAULT_ALL_PROCESSES_PATH.exists():
            self.skipTest(f"Full dataset not found: {DEFAULT_ALL_PROCESSES_PATH}")

        batch_size = 128
        eval_batch_size = 256
        loaders, metadata = setup_all_processes_dataloaders(
            {
                "batch_size": batch_size,
                "eval_batch_size": eval_batch_size,
                "num_workers": 0,
            }
        )

        self.assertEqual(set(loaders), {"train", "validate", "test"})
        self.assertEqual(
            metadata["split_sizes"],
            {"train": 56_438, "validate": 14_109, "test": 18_950},
        )
        expected_fields = {
            "dmdlnr",
            "dndlnr",
            "M",
            "N",
            "temp",
            "rh",
            "dgdt_cond",
            "vt_mass_flux",
            "dgdt_coal",
            "sample_index",
        }

        for split, loader in loaders.items():
            expected_batch_size = batch_size if split == "train" else eval_batch_size
            self.assertEqual(loader.batch_size, expected_batch_size)
            self.assertEqual(len(loader.dataset), metadata["split_sizes"][split])

            batch = next(iter(loader))
            self.assertEqual(set(batch), expected_fields)
            self.assertEqual(batch["dmdlnr"].shape, (expected_batch_size, 64))
            self.assertEqual(batch["dndlnr"].shape, (expected_batch_size, 64))
            self.assertEqual(batch["dgdt_cond"].shape, (expected_batch_size, 64))
            self.assertEqual(batch["dgdt_coal"].shape, (expected_batch_size, 64))
            self.assertEqual(batch["vt_mass_flux"].shape, (expected_batch_size, 64))
            for scalar in ("M", "N", "temp", "rh", "sample_index"):
                self.assertEqual(batch[scalar].shape, (expected_batch_size,))

            for field in expected_fields - {"sample_index"}:
                self.assertEqual(batch[field].dtype, torch.float32)
                self.assertTrue(torch.isfinite(batch[field]).all().item(), field)
            self.assertEqual(batch["sample_index"].dtype, torch.int64)
            torch.testing.assert_close(
                batch["dmdlnr"].sum(dim=-1),
                torch.ones(expected_batch_size),
                rtol=1e-6,
                atol=1e-6,
            )
            torch.testing.assert_close(
                batch["dndlnr"].sum(dim=-1),
                torch.ones(expected_batch_size),
                rtol=1e-6,
                atol=1e-6,
            )

            physical = denormalize_all_processes(batch, metadata)
            self.assertEqual(physical["dmdlnr"].shape, (expected_batch_size, 64))
            self.assertEqual(physical["dndlnr"].shape, (expected_batch_size, 64))
            self.assertTrue(torch.isfinite(physical["dmdlnr"]).all().item())
            self.assertTrue(torch.isfinite(physical["dndlnr"]).all().item())

        print("\nDataLoader integration")
        print(
            f"train: batch_size={batch_size}, samples={len(loaders['train'].dataset)}"
        )
        print(
            "validate: "
            f"batch_size={eval_batch_size}, samples={len(loaders['validate'].dataset)}"
        )
        print(
            f"test: batch_size={eval_batch_size}, samples={len(loaders['test'].dataset)}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
