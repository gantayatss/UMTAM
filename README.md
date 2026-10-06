# UMTAM

UMTAM is a research-oriented PyTorch project centered on a custom optimizer and associated training/diagnostic tooling. The codebase includes:

- a stabilized optimizer implementation for training with rank-aware updates
- memory and saliency tracking utilities
- dataset loading helpers for FineWeb
- experiment notebooks and analysis directories
- lightweight validation scripts and tests

This repository appears to be a research codebase rather than a polished production package, so setup and usage may require adapting scripts to your local environment.

## Overview

The core optimizer logic is implemented in:

- `src/umtam_optimizer_stable.py`

This file defines `UMTAMOptimizerStable`, a custom optimizer with:
- momentum-based updates
- low-rank factorization / rank-aware state
- gradient clipping and stability safeguards
- memory tracking
- saliency score estimation

The repository also includes:
- dataset utilities in `src/fineweb_loader.py`
- environment checks in `test_setup.py`
- tests in `tests/test_umtam.py`
- experiment notebooks and research folders under `experiments/`

## Project Structure

```text
UMTAM/
├── archive/                     # archived or older project files
├── examples/                   # example usage / reference materials
├── experiments/                # notebooks and experiment directories
│   ├── ablation_rank/
│   ├── additional-experiment/
│   ├── merging/
│   ├── pruning/
│   ├── spectral-analysis/
│   └── umtam-tulu3/
├── real/                       # real-data related assets
├── src/
│   ├── fineweb_loader.py
│   ├── umtam_diagnostics.py
│   └── umtam_optimizer_stable.py
├── synthetic/                  # synthetic data / experiments
├── tests/
│   └── test_umtam.py
├── check-samples.py
├── data-download.py
├── fix_imports.py
├── test_setup.py
├── README.md
└── ...