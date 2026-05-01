"""Train PI-TNet on NASA discharge cycles with optional Verhulst constraints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from battery_rul.data.dataloaders import create_nasa_dataloaders
from battery_rul.models import build_pi_tnet_from_yaml
from battery_rul.physics import build_physics_objective_from_mapping
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
    parser.add_argument("--no-physics", action="store_true")
    parser.add_argument("--physics-time-mode", type=str, default=None)
    parser.add_argument("--physics-data-loss-mode", type=str, default=None)
    parser.add_argument("--disable-structural-loss", action="store_true")
    parser.add_argument("--disable-temporal-loss", action="store_true")
    parser.add_argument("--monotonicity-weight", type=float, default=None)
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
    physics_section = config.setdefault("physics", {})
    if args.physics_time_mode is not None:
        physics_section["time_mode"] = args.physics_time_mode
    if args.physics_data_loss_mode is not None:
        physics_section["data_loss_mode"] = args.physics_data_loss_mode
    if args.disable_structural_loss:
        physics_section["use_structural_loss"] = False
    if args.disable_temporal_loss:
        physics_section["use_temporal_loss"] = False
    if args.monotonicity_weight is not None:
        physics_section["monotonicity_weight"] = float(args.monotonicity_weight)
    training_config = config["training"]
    physics_config = config.get("physics", {})
    seed = int(training_config.get("seed", 42))
    batch_size = int(training_config.get("batch_size", 16))
    max_epochs = int(args.max_epochs or training_config.get("max_epochs", 54))
    physics_enabled = bool(physics_config.get("enabled", False)) and not args.no_physics
    default_prefix = "pi_tnet_physics_informed" if physics_enabled else "pi_tnet_data_only"
    run_name = args.run_name or f"{default_prefix}_{args.battery_id}_{max_epochs}epochs"
    set_reproducible_seed(seed)

    bundle = create_nasa_dataloaders(
        feature_path=args.features,
        metadata_path=args.metadata,
        battery_id=args.battery_id,
        batch_size=batch_size,
        seed=seed,
    )
    model = build_pi_tnet_from_yaml(args.config)
    physics_objective = None
    if physics_enabled:
        physics_objective = build_physics_objective_from_mapping(config)

    optimizer_parameters = list(model.parameters())
    if physics_objective is not None:
        optimizer_parameters.extend(list(physics_objective.parameters()))

    optimizer = torch.optim.Adam(
        optimizer_parameters,
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
        physics_objective=physics_objective,
    )
    paths = save_train_result(result, args.output_dir, run_name)

    final = result.history.iloc[-1].to_dict()
    best = result.history.loc[result.history["epoch"] == result.best_epoch].iloc[0].to_dict()
    summary = {
        "stage": "M5_physics_informed_training" if physics_enabled else "M5_data_only_training",
        "run_name": run_name,
        "battery_id": args.battery_id,
        "device": str(device),
        "epochs": max_epochs,
        "batch_size": batch_size,
        "standardize": not args.no_standardize,
        "physics_enabled": physics_enabled,
        "best_epoch": result.best_epoch,
        "final_metrics": final,
        "best_epoch_metrics": best,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    if physics_objective is not None:
        summary["physics"] = {
            **physics_objective.diagnostics(),
            "time_mode": physics_objective.config.time_mode,
            "data_loss_mode": physics_objective.config.data_loss_mode,
            "use_structural_loss": physics_objective.config.use_structural_loss,
            "use_temporal_loss": physics_objective.config.use_temporal_loss,
            "monotonicity_weight": physics_objective.config.monotonicity_weight,
        }
    summary_path = args.output_dir / "logs" / f"{run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"run: {run_name}")
    print(f"device: {device}")
    print(f"epochs: {max_epochs}")
    print(f"train loss: {final['train_loss']:.6f}")
    if physics_enabled:
        print(f"train structural loss: {final['train_structural_loss']:.6f}")
        print(f"train temporal loss: {final['train_temporal_loss']:.6f}")
    print(f"test SOH MAE: {final['test_soh_mae']:.6f}")
    print(f"test SOH RMSE: {final['test_soh_rmse']:.6f}")
    print(f"test SOH R2: {final['test_soh_r2']:.6f}")
    print(f"best epoch: {result.best_epoch}")
    print(f"best test SOH RMSE: {best['test_soh_rmse']:.6f}")
    for label, path in paths.items():
        print(f"Saved {label}: {path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
