"""Verhulst-based physics constraints for PI-TNet training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


def _softplus_inverse(value: float, eps: float = 1e-6) -> float:
    """Map a positive value to the unconstrained softplus domain."""

    clipped = max(float(value), eps)
    return math.log(math.expm1(clipped))


@dataclass(frozen=True)
class PhysicsConfig:
    """Configuration for Verhulst-constrained PI-TNet training."""

    enabled: bool = True
    nominal_capacity_ah: float = 2.0
    use_adaptive_weights: bool = True
    data_weight: float = 1.0
    structural_weight: float = 1.0
    temporal_weight: float = 1.0
    monotonicity_weight: float = 0.0
    initial_r: float = 0.01
    initial_k: float = 0.2
    initial_u: float = 0.0
    initial_R: float = 0.0


class AdaptiveLossWeights(nn.Module):
    """Learn uncertainty-style weights jointly with the network."""

    def __init__(self, loss_names: Iterable[str]) -> None:
        super().__init__()
        self.log_vars = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(())) for name in loss_names}
        )

    def combine(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        total = losses[next(iter(losses))].new_zeros(())
        for name, loss in losses.items():
            log_var = self.log_vars[name]
            precision = torch.exp(-log_var)
            total = total + precision * loss + log_var
        return total

    def weight_snapshot(self) -> dict[str, float]:
        return {
            f"{name}_weight": float(torch.exp(-log_var).detach().cpu().item())
            for name, log_var in self.log_vars.items()
        }


class VerhulstCurve(nn.Module):
    """Learnable Verhulst capacity-loss trajectory."""

    def __init__(
        self,
        initial_r: float = 0.01,
        initial_k: float = 0.2,
        initial_u: float = 0.0,
        initial_R: float = 0.0,
    ) -> None:
        super().__init__()
        self._raw_r = nn.Parameter(torch.tensor(_softplus_inverse(initial_r)))
        self._raw_k = nn.Parameter(torch.tensor(_softplus_inverse(initial_k)))
        self._raw_u = nn.Parameter(torch.tensor(_softplus_inverse(initial_u)))
        self._raw_R = nn.Parameter(torch.tensor(_softplus_inverse(initial_R)))

    @property
    def r(self) -> torch.Tensor:
        return nn.functional.softplus(self._raw_r)

    @property
    def k(self) -> torch.Tensor:
        return nn.functional.softplus(self._raw_k)

    @property
    def u(self) -> torch.Tensor:
        return nn.functional.softplus(self._raw_u)

    @property
    def R(self) -> torch.Tensor:
        return nn.functional.softplus(self._raw_R)

    def forward(self, time_index: torch.Tensor) -> torch.Tensor:
        exp_term = torch.exp(-self.r * time_index)
        return self.u + self.R / (1.0 + self.k * exp_term)

    def derivative(self, time_index: torch.Tensor) -> torch.Tensor:
        exp_term = torch.exp(-self.r * time_index)
        denominator = (1.0 + self.k * exp_term).pow(2)
        return self.R * self.k * self.r * exp_term / denominator

    def parameter_snapshot(self) -> dict[str, float]:
        return {
            "verhulst_r": float(self.r.detach().cpu().item()),
            "verhulst_k": float(self.k.detach().cpu().item()),
            "verhulst_u": float(self.u.detach().cpu().item()),
            "verhulst_R": float(self.R.detach().cpu().item()),
        }


class PhysicsInformedObjective(nn.Module):
    """Data, structural, and temporal constraints for PI-TNet."""

    def __init__(self, config: PhysicsConfig) -> None:
        super().__init__()
        self.config = config
        self.verhulst = VerhulstCurve(
            initial_r=config.initial_r,
            initial_k=config.initial_k,
            initial_u=config.initial_u,
            initial_R=config.initial_R,
        )
        self.adaptive_weights = AdaptiveLossWeights(("data", "structural", "temporal"))
        self.mse = nn.MSELoss()

    def _time_tensor(self, batch: dict, device: torch.device) -> torch.Tensor:
        return batch["discharge_index"].to(device=device, dtype=torch.float32).unsqueeze(-1)

    def _capacity_loss_prediction(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        return 1.0 - output["soh"]

    def _groupwise_temporal_loss(
        self,
        predicted_capacity_loss: torch.Tensor,
        physics_capacity_loss: torch.Tensor,
        time_index: torch.Tensor,
        battery_ids: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        losses: list[torch.Tensor] = []
        monotonic_penalties: list[torch.Tensor] = []

        flat_pred = predicted_capacity_loss.reshape(-1)
        flat_phys = physics_capacity_loss.reshape(-1)
        flat_time = time_index.reshape(-1)

        for battery_id in dict.fromkeys(battery_ids):
            indices = [i for i, value in enumerate(battery_ids) if value == battery_id]
            if len(indices) < 2:
                continue

            index_tensor = torch.tensor(indices, device=flat_time.device, dtype=torch.long)
            group_time = flat_time.index_select(0, index_tensor)
            order = torch.argsort(group_time)
            sorted_time = group_time.index_select(0, order)
            sorted_pred = flat_pred.index_select(0, index_tensor).index_select(0, order)
            sorted_phys = flat_phys.index_select(0, index_tensor).index_select(0, order)

            delta_t = (sorted_time[1:] - sorted_time[:-1]).clamp_min(1e-6)
            pred_derivative = (sorted_pred[1:] - sorted_pred[:-1]) / delta_t
            phys_derivative = (sorted_phys[1:] - sorted_phys[:-1]) / delta_t
            losses.append(self.mse(pred_derivative, phys_derivative))
            monotonic_penalties.append(nn.functional.relu(-(sorted_pred[1:] - sorted_pred[:-1])).mean())

        if not losses:
            zero = predicted_capacity_loss.new_zeros(())
            return zero, zero

        temporal_loss = torch.stack(losses).mean()
        monotonicity_loss = torch.stack(monotonic_penalties).mean()
        return temporal_loss, monotonicity_loss

    def forward(
        self,
        output: dict[str, torch.Tensor],
        batch: dict,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        target_capacity = batch["capacity_ah"].to(device)
        time_index = self._time_tensor(batch, device)
        predicted_capacity_loss = self._capacity_loss_prediction(output)
        physics_capacity_loss = self.verhulst(time_index)

        data_loss = self.mse(output["capacity_ah"], target_capacity)
        structural_loss = self.mse(predicted_capacity_loss, physics_capacity_loss)
        temporal_loss, monotonicity_loss = self._groupwise_temporal_loss(
            predicted_capacity_loss=predicted_capacity_loss,
            physics_capacity_loss=physics_capacity_loss,
            time_index=time_index,
            battery_ids=list(batch["battery_id"]),
        )

        main_losses = {
            "data": data_loss,
            "structural": structural_loss,
            "temporal": temporal_loss,
        }
        if self.config.use_adaptive_weights:
            total_loss = self.adaptive_weights.combine(main_losses)
        else:
            total_loss = (
                self.config.data_weight * data_loss
                + self.config.structural_weight * structural_loss
                + self.config.temporal_weight * temporal_loss
            )

        if self.config.monotonicity_weight > 0:
            total_loss = total_loss + self.config.monotonicity_weight * monotonicity_loss

        return {
            "total_loss": total_loss,
            "data_loss": data_loss,
            "structural_loss": structural_loss,
            "temporal_loss": temporal_loss,
            "monotonicity_loss": monotonicity_loss,
        }

    def diagnostics(self) -> dict[str, float]:
        metrics = self.verhulst.parameter_snapshot()
        if self.config.use_adaptive_weights:
            metrics.update(self.adaptive_weights.weight_snapshot())
        return metrics


def physics_config_from_mapping(config: dict) -> PhysicsConfig:
    """Build `PhysicsConfig` from the project YAML structure."""

    physics_config = config.get("physics", {})
    task_config = config.get("task", {})
    initial_parameters = physics_config.get("initial_parameters", {})
    return PhysicsConfig(
        enabled=bool(physics_config.get("enabled", True)),
        nominal_capacity_ah=float(task_config.get("nominal_capacity", 2.0)),
        use_adaptive_weights=bool(physics_config.get("use_adaptive_weights", True)),
        data_weight=float(physics_config.get("data_weight", 1.0)),
        structural_weight=float(physics_config.get("structural_weight", 1.0)),
        temporal_weight=float(physics_config.get("temporal_weight", 1.0)),
        monotonicity_weight=float(physics_config.get("monotonicity_weight", 0.0)),
        initial_r=float(initial_parameters.get("r", 0.01)),
        initial_k=float(initial_parameters.get("k", 0.2)),
        initial_u=float(initial_parameters.get("u", 0.0)),
        initial_R=float(initial_parameters.get("R", 0.0)),
    )


def build_physics_objective_from_mapping(config: dict) -> PhysicsInformedObjective | None:
    """Create the physics objective if the YAML enables it."""

    physics_config = physics_config_from_mapping(config)
    if not physics_config.enabled:
        return None
    return PhysicsInformedObjective(physics_config)
