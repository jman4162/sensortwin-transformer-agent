# Deep-model tuning grid (validation macro-F1)

Mode `colab_standard`, seed 0, 30 epochs. Selection on the validation split only; the test split is never read here. Every model gets the same budget: 3 learning rates x 2 capacities.

| Model | LR | Capacity | Params | Val macro-F1 | Best epoch |
| --- | ---: | --- | ---: | ---: | ---: |
| lstm | 0.0003 | base | 39k | 0.4439 | 28 |
| lstm | 0.0003 | large | 539k | 0.6585 | 29 |
| lstm | 0.001 | base | 39k | 0.5193 | 27 |
| lstm | 0.001 | large **(selected)** | 539k | 0.7296 | 25 |
| lstm | 0.003 | base | 39k | 0.6211 | 29 |
| lstm | 0.003 | large | 539k | 0.6998 | 26 |

Commit each model's selected cell to its `configs/models/*.yaml` so the committed recipes trace back to this grid.
