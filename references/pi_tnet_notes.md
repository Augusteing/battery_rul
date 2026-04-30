# PI-TNet Paper Notes

## NASA Dataset Requirements

The paper's NASA reproduction uses four cells: `B0005`, `B0006`, `B0007`, and `B0018`.

Operational settings reported by the paper:

| Cell | Ambient temperature | Rated capacity | Discharge current | End voltage | Reported cycles |
| --- | --- | --- | --- | --- | --- |
| B0005 | 24 C | 2 Ah | 2 A | 2.7 V | 168 |
| B0006 | 24 C | 2 Ah | 2 A | 2.5 V | 168 |
| B0007 | 24 C | 2 Ah | 2 A | 2.2 V | 168 |
| B0018 | 24 C | 2 Ah | 2 A | 2.5 V | 168 |

The model input is described as discharge-cycle voltage, current, temperature, and timestamp. The predicted output is capacity/SOH.

The implementation section states that the first 70% of discharge cycles are selected as the training dataset; the training set is shuffled; training uses 54 epochs and batch size 16.

Loader implementation:

- `train`: chronological first 70%, shuffled by PyTorch `DataLoader`.
- `test`: chronological last 30%, no shuffle.
- Each batch returns `x`, `capacity_ah`, `soh`, `battery_id`, `discharge_index`, and `cycle_index`.
- `x` has shape `(batch, 4, 128)` in the current reproducible feature assumption.

## Data Audit Discrepancy

Local raw NASA files show:

| Cell | Audited discharge cycles |
| --- | ---: |
| B0005 | 168 |
| B0006 | 168 |
| B0007 | 168 |
| B0018 | 132 |

Therefore, the reproduction will use raw-data-derived cycle counts and explicitly report the `B0018` discrepancy. No synthetic cycles are introduced.

## Feature Construction Assumption

The paper reports that voltage/current values are measured at identical sampling time points during each cycle, and that voltage, current, temperature, capacity, and time are extracted during discharge. The exact number of sampling points used by the authors is not disclosed.

Local NASA discharge curves have variable lengths. For reproducibility, the implementation interpolates each discharge curve onto a normalized time grid with 128 points and retains raw physical units. This parameter is configuration-controlled and should be revisited if the authors' code or supplementary material becomes available.

## Model Implementation Notes

Paper-defined structure:

- CDP: Central Difference Convolution (CDC), Horizontal Difference Convolution (HDC), and Vertical Difference Convolution (VDC), followed by vanilla convolution.
- ViT/Transformer processor: models spatiotemporal dependencies after CDP feature extraction.
- Output: SOH/capacity prediction.
- Physics term: Verhulst-constrained loss, implemented after the base neural model.

Reproducible engineering choices:

- Input tensor `(batch, 4, 128)` is treated as a physical signal image `(batch, 1, 4, 128)` for 3x3 CDP convolutions.
- CDC is implemented as standard 3x3 convolution minus a central-difference response.
- HDC and VDC are implemented as learnable horizontal and vertical directional-difference convolutions.
- CDP high-dimensional features are fused with a 1x1 low-dimensional projection using a learnable sigmoid weight.
- Transformer tokens are obtained by averaging CDP features over the variable axis and preserving the temporal axis.
- The model outputs unconstrained SOH because the raw NASA labels include an initial `B0006` SOH greater than 1.0.

These choices follow the paper text, but the exact unpublished implementation may differ.

## Training Core

The data-only training core uses the paper's reported optimizer protocol where available:

- Adam optimizer.
- Batch size 16.
- 54 training epochs.
- First 70% discharge cycles for training, remaining 30% for testing.
- Training target is measured capacity; SOH is derived from capacity / 2 Ah.

For numerical conditioning, input channels are standardized using only training-split statistics. This does not change raw data artifacts or labels, but it is an implementation choice because the paper does not disclose its normalization details.
