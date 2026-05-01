"""Training data loaders for the PI-TNet NASA reproduction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class NasaDataBundle:
    """Container returned by `create_nasa_dataloaders`."""

    train: DataLoader
    test: DataLoader
    metadata: pd.DataFrame
    feature_names: tuple[str, ...]


class NasaPiTNetDataset(Dataset):
    """Fixed-length NASA discharge-cycle dataset.

    Samples follow the paper's input/target pairing:
    voltage/current/temperature/time -> measured capacity and SOH.
    """

    def __init__(
        self,
        feature_path: Path,
        metadata_path: Path,
        split: str,
        battery_id: str | None = None,
    ) -> None:
        feature_path = Path(feature_path)
        metadata_path = Path(metadata_path)
        if split not in {"train", "test"}:
            raise ValueError("split must be either 'train' or 'test'")

        archive = np.load(feature_path, allow_pickle=False)
        metadata = pd.read_csv(metadata_path)
        if len(metadata) != archive["x"].shape[0]:
            raise ValueError(
                "Feature tensor and metadata row counts do not match: "
                f"{archive['x'].shape[0]} vs {len(metadata)}"
            )

        mask = metadata["split"].eq(split)
        if battery_id is not None:
            mask &= metadata["battery_id"].eq(battery_id)
        indices = metadata.index[mask].to_numpy()
        if indices.size == 0:
            target = battery_id if battery_id is not None else "all cells"
            raise ValueError(f"No {split} samples found for {target}")

        self.x = torch.from_numpy(archive["x"][indices].astype(np.float32))
        self.capacity_ah = torch.from_numpy(
            metadata.loc[indices, "capacity_ah"].to_numpy(dtype=np.float32)
        ).unsqueeze(-1)
        self.soh = torch.from_numpy(
            metadata.loc[indices, "soh"].to_numpy(dtype=np.float32)
        ).unsqueeze(-1)
        self.metadata = metadata.loc[indices].reset_index(drop=True)
        self.feature_names = tuple(str(name) for name in archive["feature_names"])
        self.split = split
        self.battery_id = battery_id
        self.max_discharge_index_by_battery = (
            self.metadata.groupby("battery_id")["discharge_index"].max().to_dict()
        )

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> dict:
        row = self.metadata.iloc[index]
        return {
            "x": self.x[index],
            "capacity_ah": self.capacity_ah[index],
            "soh": self.soh[index],
            "battery_id": row["battery_id"],
            "discharge_index": int(row["discharge_index"]),
            "max_discharge_index": int(
                self.max_discharge_index_by_battery[row["battery_id"]]
            ),
            "cycle_index": int(row["cycle_index"]),
        }


def create_nasa_dataloaders(
    feature_path: Path = Path("data/processed/nasa_pi_tnet_features.npz"),
    metadata_path: Path = Path("data/processed/nasa_pi_tnet_metadata.csv"),
    battery_id: str | None = None,
    batch_size: int = 16,
    seed: int = 42,
    num_workers: int = 0,
) -> NasaDataBundle:
    """Create paper-consistent train/test loaders.

    The PI-TNet implementation details specify that the first 70% discharge
    cycles are used as training data, the training set is shuffled, and the
    batch size is 16. The split assignment itself is produced in M2.
    """

    train_dataset = NasaPiTNetDataset(
        feature_path=feature_path,
        metadata_path=metadata_path,
        split="train",
        battery_id=battery_id,
    )
    test_dataset = NasaPiTNetDataset(
        feature_path=feature_path,
        metadata_path=metadata_path,
        split="test",
        battery_id=battery_id,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    metadata = pd.concat(
        [train_dataset.metadata.assign(split="train"), test_dataset.metadata.assign(split="test")],
        ignore_index=True,
    )
    return NasaDataBundle(
        train=train_loader,
        test=test_loader,
        metadata=metadata,
        feature_names=train_dataset.feature_names,
    )
