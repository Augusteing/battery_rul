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

## Data Audit Discrepancy

Local raw NASA files show:

| Cell | Audited discharge cycles |
| --- | ---: |
| B0005 | 168 |
| B0006 | 168 |
| B0007 | 168 |
| B0018 | 132 |

Therefore, the reproduction will use raw-data-derived cycle counts and explicitly report the `B0018` discrepancy. No synthetic cycles are introduced.
