"""Verhulst-based physics constraints for PI-TNet training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


def _softplus_inverse(value: float, eps: float = 1e-6) -> float:
    """Map a positive scalar to the unconstrained softplus domain."""

    clipped = max(float(value), eps)
    return math.log(math.expm1(clipped))


@dataclass(frozen=True)
class PhysicsConfig:
    """Configuration for Verhulst-constrained PI-TNet training."""

    enabled: bool = True
    nominal_capacity_ah: float = 2.0
    use_structural_loss: bool = True
    use_temporal_loss: bool = True
    monotonicity_weight: float = 0.0
    initial_r: float = 0.01
    initial_R: float = 0.0
    initial_K: float = 0.2
    initial_lambda_u: float = 1.0
    initial_lambda_t: float = 1.0
    initial_lambda_f: float = 1.0
    time_mode: str = "raw"
    data_loss_mode: str = "capacity_loss"


class AdaptivePhysicsWeights(nn.Module):
    """Positive trainable weights for the paper's adaptive loss balancing."""

    def __init__(
        self,
        initial_lambda_u: float = 1.0,
        initial_lambda_t: float = 1.0,
        initial_lambda_f: float = 1.0,
    ) -> None:
        super().__init__()
        self._raw_lambda_u = nn.Parameter(torch.tensor(_softplus_inverse(initial_lambda_u)))
        self._raw_lambda_t = nn.Parameter(torch.tensor(_softplus_inverse(initial_lambda_t)))
        self._raw_lambda_f = nn.Parameter(torch.tensor(_softplus_inverse(initial_lambda_f)))

    @property
    def lambda_u(self) -> torch.Tensor:
        return nn.functional.softplus(self._raw_lambda_u)

    @property
    def lambda_t(self) -> torch.Tensor:
        return nn.functional.softplus(self._raw_lambda_t)

    @property
    def lambda_f(self) -> torch.Tensor:
        return nn.functional.softplus(self._raw_lambda_f)

    def combine(
        self,
        loss_u: torch.Tensor,
        loss_t: torch.Tensor,
        loss_f: torch.Tensor,
        use_structural_loss: bool = True,
        use_temporal_loss: bool = True,
    ) -> torch.Tensor:
        weighted_terms = [self.lambda_u * loss_u]
        active_weights = [self.lambda_u]
        if use_structural_loss:
            weighted_terms.append(self.lambda_f * loss_f)
            active_weights.append(self.lambda_f)
        if use_temporal_loss:
            weighted_terms.append(self.lambda_t * loss_t)
            active_weights.append(self.lambda_t)
        product = torch.stack(active_weights).prod().clamp_min(1e-12)
        return torch.stack(weighted_terms).sum() - torch.log(product)

    def snapshot(self) -> dict[str, float]:
        return {
            "lambda_u": float(self.lambda_u.detach().cpu().item()),
            "lambda_t": float(self.lambda_t.detach().cpu().item()),
            "lambda_f": float(self.lambda_f.detach().cpu().item()),
        }


class VerhulstDynamics(nn.Module):
    """Learnable parameters of the Verhulst degradation equation."""

    def __init__(
        self,
        initial_r: float = 0.01,
        initial_R: float = 0.0,
        initial_K: float = 0.2,
    ) -> None:
        super().__init__()
        self._raw_r = nn.Parameter(torch.tensor(_softplus_inverse(initial_r)))
        self._raw_R = nn.Parameter(torch.tensor(_softplus_inverse(initial_R)))
        self._raw_delta = nn.Parameter(torch.tensor(_softplus_inverse(max(initial_K - initial_R, 1e-3))))

    @property
    def r(self) -> torch.Tensor:
        return nn.functional.softplus(self._raw_r)

    @property
    def R(self) -> torch.Tensor:
        return nn.functional.softplus(self._raw_R)

    @property
    def K(self) -> torch.Tensor:
        return self.R + nn.functional.softplus(self._raw_delta)

    def rhs(self, capacity_loss: torch.Tensor) -> torch.Tensor:
        denominator = (self.K - self.R).clamp_min(1e-6)
        shifted = capacity_loss - self.R
        return self.r * shifted * (1.0 - shifted / denominator)

    def snapshot(self) -> dict[str, float]:
        return {
            "verhulst_r": float(self.r.detach().cpu().item()),
            "verhulst_R": float(self.R.detach().cpu().item()),
            "verhulst_K": float(self.K.detach().cpu().item()),
        }


class PhysicsInformedObjective(nn.Module):
    """Paper-aligned physics-informed losses for PI-TNet."""

    def __init__(self, config: PhysicsConfig) -> None:
        super().__init__()
        self.config = config
        self.dynamics = VerhulstDynamics(
            initial_r=config.initial_r,
            initial_R=config.initial_R,
            initial_K=config.initial_K,
        )
        self.weights = AdaptivePhysicsWeights(
            initial_lambda_u=config.initial_lambda_u,
            initial_lambda_t=config.initial_lambda_t,
            initial_lambda_f=config.initial_lambda_f,
        )
        self.mse = nn.MSELoss()

    def _time_tensor(self, batch: dict, device: torch.device) -> torch.Tensor:
        discharge_index = batch["discharge_index"].to(
            device=device,
            dtype=torch.float32,
        ).reshape(-1)
        if self.config.time_mode == "raw":
            return discharge_index
        if self.config.time_mode == "normalized":
            max_discharge_index = batch["max_discharge_index"].to(
                device=device,
                dtype=torch.float32,
            ).reshape(-1)
            return discharge_index / max_discharge_index.clamp_min(1.0)
        raise ValueError(f"Unsupported time_mode: {self.config.time_mode}")

    def _capacity_loss_prediction(self, output: dict[str, torch.Tensor]) -> torch.Tensor:
        return (1.0 - output["soh"]).reshape(-1)

    def _capacity_loss_target(self, batch: dict, device: torch.device) -> torch.Tensor:
        return (1.0 - batch["soh"].to(device=device, dtype=torch.float32)).reshape(-1)

    def _groupwise_residual_terms(
        self,
        predicted_capacity_loss: torch.Tensor,
        time_index: torch.Tensor,
        battery_ids: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        structural_losses: list[torch.Tensor] = []
        temporal_losses: list[torch.Tensor] = []
        monotonicity_losses: list[torch.Tensor] = []

        for battery_id in dict.fromkeys(battery_ids):
            indices = [i for i, value in enumerate(battery_ids) if value == battery_id]
            if len(indices) < 2:
                continue

            index_tensor = torch.tensor(indices, device=time_index.device, dtype=torch.long)
            group_time = time_index.index_select(0, index_tensor)
            group_loss = predicted_capacity_loss.index_select(0, index_tensor)
            order = torch.argsort(group_time)
            sorted_time = group_time.index_select(0, order)
            sorted_loss = group_loss.index_select(0, order)

            delta_t = (sorted_time[1:] - sorted_time[:-1]).clamp_min(1e-6)
            df_dt = (sorted_loss[1:] - sorted_loss[:-1]) / delta_t
            rhs = self.dynamics.rhs(sorted_loss[:-1])
            residual = df_dt - rhs

            structural_losses.append((residual.pow(2)).mean())
            monotonicity_losses.append(nn.functional.relu(-(sorted_loss[1:] - sorted_loss[:-1])).mean())

            if residual.numel() >= 2:
                residual_dt = (residual[1:] - residual[:-1]) / delta_t[1:].clamp_min(1e-6)
                temporal_losses.append((residual_dt.pow(2)).mean())

        zero = predicted_capacity_loss.new_zeros(())
        structural_loss = torch.stack(structural_losses).mean() if structural_losses else zero
        temporal_loss = torch.stack(temporal_losses).mean() if temporal_losses else zero
        monotonicity_loss = (
            torch.stack(monotonicity_losses).mean() if monotonicity_losses else zero
        )
        return structural_loss, temporal_loss, monotonicity_loss

    def forward(
        self,
        output: dict[str, torch.Tensor],
        batch: dict,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        predicted_capacity_loss = self._capacity_loss_prediction(output)
        target_capacity_loss = self._capacity_loss_target(batch, device)
        time_index = self._time_tensor(batch, device)

        if self.config.data_loss_mode == "capacity_loss":
            loss_u = self.mse(predicted_capacity_loss, target_capacity_loss)
        elif self.config.data_loss_mode == "capacity_ah":
            capacity_true = batch["capacity_ah"].to(device=device, dtype=torch.float32).reshape(-1)
            capacity_pred = output["capacity_ah"].reshape(-1)
            loss_u = self.mse(capacity_pred, capacity_true)
        elif self.config.data_loss_mode == "soh":
            soh_true = batch["soh"].to(device=device, dtype=torch.float32).reshape(-1)
            soh_pred = output["soh"].reshape(-1)
            loss_u = self.mse(soh_pred, soh_true)
        else:
            raise ValueError(f"Unsupported data_loss_mode: {self.config.data_loss_mode}")
        loss_f, loss_t, monotonicity_loss = self._groupwise_residual_terms(
            predicted_capacity_loss=predicted_capacity_loss,
            time_index=time_index,
            battery_ids=list(batch["battery_id"]),
        )
        if not self.config.use_structural_loss:
            loss_f = loss_f.new_zeros(())
        if not self.config.use_temporal_loss:
            loss_t = loss_t.new_zeros(())
        total_loss = self.weights.combine(
            loss_u=loss_u,
            loss_t=loss_t,
            loss_f=loss_f,
            use_structural_loss=self.config.use_structural_loss,
            use_temporal_loss=self.config.use_temporal_loss,
        )

        if self.config.monotonicity_weight > 0:
            total_loss = total_loss + self.config.monotonicity_weight * monotonicity_loss

        return {
            "total_loss": total_loss,
            "data_loss": loss_u,
            "structural_loss": loss_f,
            "temporal_loss": loss_t,
            "monotonicity_loss": monotonicity_loss,
        }

    def diagnostics(self) -> dict[str, float]:
        metrics = self.dynamics.snapshot()
        metrics.update(self.weights.snapshot())
        return metrics


def physics_config_from_mapping(config: dict) -> PhysicsConfig:
    """Build `PhysicsConfig` from the project YAML structure."""

    physics_config = config.get("physics", {})
    task_config = config.get("task", {})
    initial_parameters = physics_config.get("initial_parameters", {})
    adaptive_weights = physics_config.get("adaptive_weights", {})
    return PhysicsConfig(
        enabled=bool(physics_config.get("enabled", True)),
        nominal_capacity_ah=float(task_config.get("nominal_capacity", 2.0)),
        use_structural_loss=bool(physics_config.get("use_structural_loss", True)),
        use_temporal_loss=bool(physics_config.get("use_temporal_loss", True)),
        monotonicity_weight=float(physics_config.get("monotonicity_weight", 0.0)),
        initial_r=float(initial_parameters.get("r", 0.01)),
        initial_R=float(initial_parameters.get("R", 0.0)),
        initial_K=float(initial_parameters.get("K", 0.2)),
        initial_lambda_u=float(adaptive_weights.get("lambda_u", 1.0)),
        initial_lambda_t=float(adaptive_weights.get("lambda_t", 1.0)),
        initial_lambda_f=float(adaptive_weights.get("lambda_f", 1.0)),
        time_mode=str(physics_config.get("time_mode", "raw")),
        data_loss_mode=str(physics_config.get("data_loss_mode", "capacity_loss")),
    )


def build_physics_objective_from_mapping(config: dict) -> PhysicsInformedObjective | None:
    """Create the physics objective if the YAML enables it."""

    physics_config = physics_config_from_mapping(config)
    if not physics_config.enabled:
        return None
    return PhysicsInformedObjective(physics_config)
