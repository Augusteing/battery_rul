# PI-TNet Formula Notes

This file records the paper-aligned loss structure used in the reproducible implementation.

## Confirmed from the paper

- The NASA experiment uses the first `70%` of discharge cycles for training and the remaining `30%` for testing.
- The training set is shuffled during model training.
- The reported training setup uses `54` epochs and batch size `16`.
- PI-TNet combines:
  - a regression fitting term `L_u`,
  - a Verhulst structural term `L_f`,
  - a temporal term `L_t`,
  - adaptive weighting learned during training.

## Implementation mapping

Let `soh_pred` be the neural-network SOH output and let the capacity-loss prediction be:

```text
f(alpha, t) = 1 - SOH_hat
```

The paper's Verhulst dynamic equation is:

```text
df(alpha, t) / dt = r * [f(alpha, t) - R] * (1 - (f(alpha, t) - R) / (K - R))
```

where:

- `r` is the degradation-rate parameter,
- `R` is the initial-loss parameter,
- `K` is the upper-limit parameter.

The residual function is implemented in the paper-consistent PINN form:

```text
E(alpha, t; Theta, Omega) = df(alpha, t) / dt - r * [f(alpha, t) - R] * (1 - (f(alpha, t) - R) / (K - R))
```

The training objective is:

```text
L_u     = MSE(f_pred, f_true)
L_f     = mean(E^2)
L_t     = mean((dE / dt)^2)
L_total = lambda_u * L_u + lambda_t * L_t + lambda_f * L_f - log(lambda_u * lambda_t * lambda_f)
```

An optional monotonicity penalty can be added, but it is disabled by default because the paper's core description emphasizes the Verhulst structural and temporal constraints.

## Recorded assumptions

- The paper PDF text extraction does not preserve every equation symbol cleanly, so the implementation follows the loss structure explicitly described in Section 3.6 and the user-provided formula screenshots.
- Lifecycle time `t` is represented by NASA discharge-cycle index in the current codebase.
- Temporal derivatives are approximated with finite differences on cycle-sorted samples within each battery.
- `Eq. (6)` in the paper appears internally inconsistent because the surrounding text says the regression term compares predicted `f(alpha,t)` against measured `f_i`, while the printed equation uses `E(...) - f_i`. The implementation adopts the self-consistent interpretation `L_u = MSE(f_pred, f_true)` and keeps `E` for `L_f` and `L_t`.
