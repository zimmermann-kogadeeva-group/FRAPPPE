# FRAPPPE

**13C Metabolic Flux Analysis via Flux Sampling and Random Forest Prediction**

FRAPPPE is a Python toolkit for predicting intracellular flux ratios from 13C
isotope labeling data. It samples the feasible flux solution space of a
metabolic network, simulates the resulting mass distribution vectors (MDVs),
and uses these simulated MDV–flux pairs to train a random forest model —
enabling rapid flux-ratio prediction from experimental MDV measurements.

## Installation

Install with pip:

```bash
pip install frapppe
```

or with uv:

```bash
uv add frapppe
```

## How it works

FRAPPPE turns a metabolic network model into a flux-ratio predictor in three
steps, all driven by the [`MFA`][frapppe.mfa.MFA] class:

1. **Sample fluxes.** [`MFA.sample_fluxes`][frapppe.mfa.MFA.sample_fluxes]
   draws samples from the feasible flux space of the network (defined by the
   stoichiometric matrix and user-supplied bounds/constraints) using Markov
   chain Monte Carlo sampling via [hopsy](https://modsim.github.io/hopsy/).
2. **Simulate MDVs.** [`MFA.simulate_mdvs`][frapppe.mfa.MFA.simulate_mdvs]
   forward-simulates the mass distribution vectors for a chosen labeling
   strategy and set of target EMUs for each sampled flux vector (via
   [freeflux](https://freeflux.readthedocs.io/)), optionally in parallel.
3. **Train a predictor.** [`train_rf`][frapppe.ml.train_rf] fits a random
   forest that maps simulated MDVs to a flux ratio of interest, so that
   experimental MDVs can be used to predict flux ratios.

The [plotting utilities][frapppe.utils] help visualise sampled fluxes, MDV
distributions, and prediction results at each stage.

## Next steps

- Follow the [Example](example.ipynb) for a full, runnable walkthrough on a
  *Bacteroides uniformis* model.
- Browse the [API Reference](api.md) for details on every public function.
