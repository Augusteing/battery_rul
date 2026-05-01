# PI-TNet Formula Notes

This file records the paper-aligned loss structure used in the reproducible implementation.

## Confirmed from the paper

- The NASA experiment uses the first `70%` of discharge cycles for training and the remaining `30%` for testing.
- The training set is shuffled during model training.
- The reported training setup uses `54` epochs and batch size `16`.
- PI-TNet combines:
  - a data regression term,
  - a Verhulst-based structural constraint,
  - a temporal constraint,
  - adaptive weighting learned during training.

## Implementation mapping

Let `soh_pred` be the neural-network SOH output and let the capacity-loss prediction be:

```text
f_pred(t) = 1 - soh_pred(t)
```

The learnable Verhulst trajectory is implemented as:

```text
f_phys(t) = u + R / (1 + k * exp(-r * t))
```

where:

- `r` is the degradation-rate parameter,
- `k` is the logistic-shape parameter,
- `u` is the initial capacity-loss offset,
- `R` is the learnable saturation scale.

The training objective is:

```text
L_data       = MSE(capacity_pred, capacity_true)
L_structural = MSE(f_pred(t), f_phys(t))
L_temporal   = MSE(df_pred / dt, df_phys / dt)
L_total      = adaptive_weighting(L_data, L_structural, L_temporal)
```

An optional monotonicity penalty can be added, but it is disabled by default because the paper's core description emphasizes the Verhulst structural and temporal constraints.

## Recorded assumptions

- The paper PDF text extraction does not preserve every equation symbol cleanly, so the implementation follows the loss structure that is explicitly described in Section 3.6.
- Lifecycle time `t` is represented by NASA discharge-cycle index in the current codebase.
- Temporal derivatives are approximated with finite differences on cycle-sorted samples within each battery.
