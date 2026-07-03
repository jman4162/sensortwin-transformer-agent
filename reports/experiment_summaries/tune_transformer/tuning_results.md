# Deep-model tuning grid (validation macro-F1)

Mode `colab_standard`, seed 0, 30 epochs. Selection on the validation split only; the test split is never read here. Every model gets the same budget: 3 learning rates x 2 capacities.

| Model | LR | Capacity | Params | Val macro-F1 | Best epoch |
| --- | ---: | --- | ---: | ---: | ---: |
| transformer | 0.0003 | base | 815k | 0.8233 | 24 |
| transformer | 0.0003 | large | 1824k | 0.8564 | 28 |
| transformer | 0.001 | base **(selected)** | 815k | 0.8749 | 25 |
| transformer | 0.001 | large | 1824k | 0.8688 | 27 |
| transformer | 0.003 | base | 815k | 0.8218 | 29 |
| transformer | 0.003 | large | 1824k | 0.8047 | 28 |

Commit each model's selected cell to its `configs/models/*.yaml` so the committed recipes trace back to this grid.
