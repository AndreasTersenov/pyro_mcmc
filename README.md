# GPPyro Emulator

A package to build Gaussian Process emulators using GPyTorch and perform Bayesian inference using Pyro.

## Installation

```bash
pip install -r requirements.txt
python setup.py install
# or for development
python setup.py develop
```

## Usage

Place your data files (`parameters.npy`, `histograms.npy`, `covariance.npy`) in the `data/` directory.

Run the example inference script:

```bash
python scripts/run_inference.py
```

## Structure

- `gppyro_emulator/`: Contains the core package code.
  - `gp_emulator.py`: Defines the GPyTorch GP model and training logic.
  - `pyro_model.py`: Defines the Pyro model for Bayesian inference.
  - `utils.py`: Utility functions for data loading.
- `scripts/`: Contains example scripts.
  - `run_inference.py`: Demonstrates training and inference.
- `data/`: Placeholder for input data files.
- `requirements.txt`: Package dependencies.
- `setup.py`: Package setup script.
- `README.md`: This file. 