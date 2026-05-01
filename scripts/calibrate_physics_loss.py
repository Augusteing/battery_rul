"""Run a small calibration sweep for physics-informed PI-TNet."""

from __future__ import annotations

import argparse
import copy
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
    parser.add_argument("--battery-id", type=str, default="B0005")
    parser.add_argument("--max-epochs", type=int, default=54)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    training_config = base_config["training"]
    seed = int(training_config.get("seed", 42))
    batch_size = int(training_config.get("batch_size", 16))
    set_reproducible_seed(seed)
    device = resolve_device(args.device)

    bundle = create_nasa_dataloaders(
        battery_id=args.battery_id,
        batch_size=batch_size,
        seed=seed,
    )

    candidates = [
        {
            "name": "raw_capacity_loss_default",
            "time_mode": "raw",
            "data_loss_mode": "capacity_loss",
            "lambda_u": 1.0,
            "lambda_t": 1.0,
            "lambda_f": 1.0,
        },
        {
            "name": "normalized_capacity_loss_default",
            "time_mode": "normalized",
            "data_loss_mode": "capacity_loss",
            "lambda_u": 1.0,
            "lambda_t": 1.0,
            "lambda_f": 1.0,
        },
        {
            "name": "normalized_capacity_loss_data_heavy",
            "time_mode": "normalized",
            "data_loss_mode": "capacity_loss",
            "lambda_u": 5.0,
            "lambda_t": 0.5,
            "lambda_f": 0.5,
        },
        {
            "name": "normalized_capacity_ah_data_heavy",
            "time_mode": "normalized",
            "data_loss_mode": "capacity_ah",
            "lambda_u": 5.0,
            "lambda_t": 0.5,
            "lambda_f": 0.5,
        },
        {
            "name": "normalized_soh_data_heavy",
            "time_mode": "normalized",
            "data_loss_mode": "soh",
            "lambda_u": 5.0,
            "lambda_t": 0.5,
            "lambda_f": 0.5,
        },
    ]

    rows = []
    for candidate in candidates:
        config = copy.deepcopy(base_config)
        config["physics"]["time_mode"] = candidate["time_mode"]
        config["physics"]["data_loss_mode"] = candidate["data_loss_mode"]
        config["physics"]["adaptive_weights"] = {
            "lambda_u": candidate["lambda_u"],
            "lambda_t": candidate["lambda_t"],
            "lambda_f": candidate["lambda_f"],
        }

        model = build_pi_tnet_from_yaml(args.config)
        physics_objective = build_physics_objective_from_mapping(config)
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(physics_objective.parameters()),
            lr=float(training_config.get("learning_rate", 1e-3)),
            weight_decay=float(training_config.get("weight_decay", 1e-4)),
        )
        run_name = f"calib_{args.battery_id}_{candidate['name']}_{args.max_epochs}epochs"
        result = train_model(
            model=model,
            train_loader=bundle.train,
            test_loader=bundle.test,
            optimizer=optimizer,
            device=device,
            max_epochs=args.max_epochs,
            standardize=True,
            physics_objective=physics_objective,
        )
        paths = save_train_result(result, args.output_dir, run_name)
        final = result.history.iloc[-1].to_dict()
        best_idx = result.history["test_soh_rmse"].idxmin()
        best = result.history.loc[best_idx].to_dict()
        summary = {
            "candidate": candidate,
            "run_name": run_name,
            "final_metrics": final,
            "best_epoch_metrics": best,
            "outputs": {key: str(value) for key, value in paths.items()},
            "physics": physics_objective.diagnostics(),
        }
        summary_path = args.output_dir / "logs" / f"{run_name}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        rows.append(
            {
                "candidate": candidate["name"],
                "time_mode": candidate["time_mode"],
                "data_loss_mode": candidate["data_loss_mode"],
                "init_lambda_u": candidate["lambda_u"],
                "init_lambda_t": candidate["lambda_t"],
                "init_lambda_f": candidate["lambda_f"],
                "final_test_soh_mae": final["test_soh_mae"],
                "final_test_soh_rmse": final["test_soh_rmse"],
                "final_test_soh_r2": final["test_soh_r2"],
                "best_epoch": int(best["epoch"]),
                "best_test_soh_mae": best["test_soh_mae"],
                "best_test_soh_rmse": best["test_soh_rmse"],
                "best_test_soh_r2": best["test_soh_r2"],
            }
        )
        print(
            f"{candidate['name']}: final RMSE={final['test_soh_rmse']:.6f}, "
            f"best epoch={int(best['epoch'])}, best RMSE={best['test_soh_rmse']:.6f}"
        )

    import pandas as pd

    df = pd.DataFrame(rows).sort_values("best_test_soh_rmse").reset_index(drop=True)
    output_path = args.output_dir / "tables" / f"calibration_{args.battery_id}_physics_loss.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print("\nCalibration summary:")
    print(df.to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
