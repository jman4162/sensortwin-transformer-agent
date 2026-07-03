# Deep-model tuning grid (validation macro-F1)

Mode `colab_standard`, seed 0, 30 epochs. Selection on the validation split only; the test split is never read here. Every model gets the same budget: 3 learning rates x 2 capacities.

| Model | LR | Capacity | Params | Val macro-F1 | Best epoch |
| --- | ---: | --- | ---: | ---: | ---: |
| cnn | 0.0003 | base | 54k | 0.7216 | 26 |
| cnn | 0.0003 | large | 211k | 0.7631 | 25 |
| cnn | 0.001 | base | 54k | 0.7812 | 22 |
| cnn | 0.001 | large | 211k | 0.8130 | 28 |
| cnn | 0.003 | base | 54k | 0.8181 | 28 |
| cnn | 0.003 | large **(selected)** | 211k | 0.8249 | 25 |

Commit each model's selected cell to its `configs/models/*.yaml` so the committed recipes trace back to this grid.
