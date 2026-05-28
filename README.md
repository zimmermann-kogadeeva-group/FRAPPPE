
# FRAPPPE

13C Metabolic Flux Analysis via Flux Sampling and Random Forest Prediction

A Python toolkit for predicting intracellular flux ratios from 13C isotope
labeling data. The package samples the feasible flux solution space of a
metabolic network, simulates the resulting mass distribution vectors (MDVs),
and uses these simulated MDV–flux pairs to train a random forest model —
enabling rapid flux-ratio prediction from experimental MDV measurements.

## Installation

Install with pip:
```
pip install frapppe
```
or with uv:
```
uv add frapppe
```

