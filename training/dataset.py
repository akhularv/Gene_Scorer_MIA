# - Custom dataset and batch sampler for MIA gene scoring
# - Dataset loads expression + metadata + precomputed targets
# - Sampler ensures every batch has multiple timepoints AND both conditions
# - This is required for L_temporal to function (needs cross-timepoint variance)
# - Batch composition: 2-3 animals per timepoint-condition-region cell

import os
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

import yaml

from precompute.compute_priors import cpm_log1p


class MIADataset(Dataset):
    """Dataset for MIA gene scoring. One item = one animal's expression + metadata.

    Loads expression data, normalizes it (CPM + log1p), and pairs each animal
    with its metadata indices and precomputed targets.
    """

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        config: dict,
        animal_ids: list[str] | None = None,
    ) -> None:
        """
        Args:
            data_dir: path to data/ directory
            output_dir: path to outputs/ with precomputed .npy files
            config: full config dict with maps
            animal_ids: if provided, subset to these animals only (for train/val split)
        """
        self.config = config
        self.tp_map = config["timepoint_map"]
        self.cond_map = config["condition_map"]
        self.reg_map = config["region_map"]

        # load precomputed targets
        self.perturb_targets = np.load(
            os.path.join(output_dir, "perturbation_targets.npy")
        )
        self.transfer_targets = np.load(
            os.path.join(output_dir, "transferability_targets.npy")
        )
        self.priors = np.load(os.path.join(output_dir, "p_g.npy"))

        # load raw expression data
        meta_cols = ["animal_id", "timepoint", "condition"]
        ef_raw = pd.read_csv(os.path.join(data_dir, "expression_ef.csv"))
        wc_raw = pd.read_csv(os.path.join(data_dir, "expression_wc.csv"))

        gene_cols = [c for c in ef_raw.columns if c not in meta_cols]
        self.n_genes = len(gene_cols)

        # normalize expression
        ef_expr = cpm_log1p(ef_raw[gene_cols].values.astype(np.float64))
        wc_expr = cpm_log1p(wc_raw[gene_cols].values.astype(np.float64))

        # build unified sample list: each row is one animal-region observation
        self.samples = []  # list of dicts with expression + metadata

        for i in range(len(ef_raw)):
            aid = str(ef_raw.iloc[i]["animal_id"])
            if animal_ids is not None and aid not in animal_ids:
                continue
            self.samples.append({
                "expression": ef_expr[i].astype(np.float32),
                "animal_id": aid,
                "timepoint": str(ef_raw.iloc[i]["timepoint"]),
                "condition": str(ef_raw.iloc[i]["condition"]),
                "region": "excitatory_frontal",
            })

        for i in range(len(wc_raw)):
            aid = str(wc_raw.iloc[i]["animal_id"])
            if animal_ids is not None and aid not in animal_ids:
                continue
            self.samples.append({
                "expression": wc_expr[i].astype(np.float32),
                "animal_id": aid,
                "timepoint": str(wc_raw.iloc[i]["timepoint"]),
                "condition": str(wc_raw.iloc[i]["condition"]),
                "region": "whole_cortex",
            })

        # build index for stratified sampling
        self._build_group_index()

    def _build_group_index(self) -> None:
        """Index samples by (timepoint, condition, region) for stratified batching."""
        self.group_indices: dict[tuple[str, str, str], list[int]] = {}
        for idx, s in enumerate(self.samples):
            key = (s["timepoint"], s["condition"], s["region"])
            if key not in self.group_indices:
                self.group_indices[key] = []
            self.group_indices[key].append(idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return one animal's data as tensors."""
        s = self.samples[idx]
        tp_idx = self.tp_map[s["timepoint"]]

        return {
            "expression": torch.from_numpy(s["expression"]),
            "timepoint_id": torch.tensor(tp_idx, dtype=torch.long),
            "condition_id": torch.tensor(
                self.cond_map[s["condition"]], dtype=torch.long
            ),
            "region_id": torch.tensor(
                self.reg_map[s["region"]], dtype=torch.long
            ),
            "timepoint_name": s["timepoint"],
            "condition_name": s["condition"],
        }


class StratifiedBatchSampler(Sampler[list[int]]):
    """Samples batches with animals from multiple timepoints and both conditions.

    Each batch pulls 2-3 animals per available (timepoint, condition, region) group.
    This ensures L_temporal has cross-timepoint signal in every batch.
    """

    def __init__(
        self,
        dataset: MIADataset,
        animals_per_group: int = 2,
        drop_last: bool = False,
    ) -> None:
        self.group_indices = dataset.group_indices
        self.animals_per_group = animals_per_group
        self.drop_last = drop_last
        self.rng = np.random.RandomState(0)

        # count how many batches we can generate per epoch
        min_group_size = min(len(v) for v in self.group_indices.values())
        self._num_batches = max(1, min_group_size // animals_per_group)

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batch index lists."""
        # shuffle within each group
        shuffled = {
            k: self.rng.permutation(v).tolist()
            for k, v in self.group_indices.items()
        }

        for batch_idx in range(self._num_batches):
            batch = []
            for key, indices in shuffled.items():
                start = batch_idx * self.animals_per_group
                end = start + self.animals_per_group
                # wrap around if we run out
                selected = []
                for i in range(start, end):
                    selected.append(indices[i % len(indices)])
                batch.extend(selected)
            yield batch

    def __len__(self) -> int:
        return self._num_batches


def collate_batch(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Custom collate: stack tensors and collect string metadata."""
    return {
        "expression": torch.stack([b["expression"] for b in batch]),
        "timepoint_id": torch.stack([b["timepoint_id"] for b in batch]),
        "condition_id": torch.stack([b["condition_id"] for b in batch]),
        "region_id": torch.stack([b["region_id"] for b in batch]),
        "timepoint_name": [b["timepoint_name"] for b in batch],
        "condition_name": [b["condition_name"] for b in batch],
    }


def train_val_split(
    data_dir: str, config: dict
) -> tuple[list[str], list[str]]:
    """Split animal IDs into train/val within each timepoint-condition group.

    Ensures every group is represented in both sets. Split is by animal,
    not by sample — same animal's EF and WC data stay together.
    """
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    train_frac = config["train_split"]
    rng = np.random.RandomState(42)

    train_ids, val_ids = [], []

    # group animals by timepoint + condition (ignore region for splitting)
    groups = meta.groupby(["timepoint", "condition"])["animal_id"].unique()

    for (tp, cond), animals in groups.items():
        animals = list(set(str(a) for a in animals))
        rng.shuffle(animals)
        n_train = max(1, int(len(animals) * train_frac))
        train_ids.extend(animals[:n_train])
        val_ids.extend(animals[n_train:])

    return list(set(train_ids)), list(set(val_ids))
