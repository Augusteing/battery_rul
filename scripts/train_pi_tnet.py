"""Train data-only PI-TNet on NASA discharge cycles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from battery_rul.data.dataloaders import create_nasa_dataloaders
from battery_rul.models import build_pi_tnet_from_yaml
from battery_rul.training import save_train_result, set_reproducible_seed, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/model/pi_tnet.yaml"))
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/nasa_pi_tnet_features.npz"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/processed/nasa_pi_tnet_metadata.csv"),
    )
    parser.add_argument("--battery-id", type=str, default="B0005")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-standardize", action="store_true")
    return parser.parse_args()


def load_training_config(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    config = load_training_config(args.config)
    training_config = config["training"]
    seed = int(training_config.get("seed", 42))
    batch_size = int(training_config.get("batch_size", 16))
    max_epochs = int(args.max_epochs or training_config.get("max_epochs", 54))
    run_name = args.run_name or f"pi_tnet_data_only_{args.battery_id}_{max_epochs}epochs"
    set_reproducible_seed(seed)

    bundle = create_nasa_dataloaders(
        feature_path=args.features,
        metadata_path=args.metadata,
        battery_id=args.battery_id,
        batch_size=batch_size,
        seed=seed,
    )
    model = build_pi_tnet_from_yaml(args.config)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    device = resolve_device(args.device)
    result = train_model(
        model=model,
        train_loader=bundle.train,
        test_loader=bundle.test,
        optimizer=optimizer,
        device=device,
        max_epochs=max_epochs,
        standardize=not args.no_standardize,
    )
    paths = save_train_result(result, args.output_dir, run_name)

    final = result.history.iloc[-1].to_dict()
    summary = {
        "stage": "M5_data_only_training",
        "run_name": run_name,
        "battery_id": args.battery_id,
        "device": str(device),
        "epochs": max_epochs,
        "batch_size": batch_size,
        "standardize": not args.no_standardize,
        "final_metrics": final,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    summary_path = args.output_dir / "logs" / f"{run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"run: {run_name}")
    print(f"device: {device}")
    print(f"epochs: {max_epochs}")
    print(f"train loss: {final['train_loss']:.6f}")
    print(f"test SOH MAE: {final['test_soh_mae']:.6f}")
    print(f"test SOH RMSE: {final['test_soh_rmse']:.6f}")
    print(f"test SOH R2: {final['test_soh_r2']:.6f}")
    for label, path in paths.items():
        print(f"Saved {label}: {path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
