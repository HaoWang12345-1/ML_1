# ML_1
# Mechanism encoded inverse design of fatigue resistant hydrated polymer networks

This repository provides the core data and scripts for mechanism encoded inverse design of fatigue resistant hydrated polymer networks. The workflow represents PEG-NR networks with hydration capacity, thermal interpenetration, and network locking descriptors, then integrates Gaussian process surrogate modeling, Pareto set learning, and uncertainty aware LCB/UCB-HVI active learning to identify retained load transfer windows.

## Core scripts

- `scripts/train_gp_models.py` trains Gaussian process surrogate models for cumulative measured datasets.
- `scripts/run_psl_inverse_design.py` runs GP-guided Pareto set learning and exports candidate design windows.
- `scripts/select_al_round_lcb_hvi.py` selects active-learning Round candidates using hydration and performance constraints with LCB/UCB-HVI.

The repository is intended to document the core inverse design workflow. 
