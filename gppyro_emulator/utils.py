import numpy as np
import torch
import os

def load_npy_data(file_path: str) -> np.ndarray:
    """Loads data from a .npy file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    return np.load(file_path)

def load_training_data(params_path: str, hist_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads training parameters and histograms, converting them to PyTorch tensors."""
    params = load_npy_data(params_path)
    histograms = load_npy_data(hist_path)

    # Validate shapes
    if params.shape[0] != histograms.shape[0]:
        raise ValueError("Number of samples in parameters and histograms must match.")
    # Add shape validation based on user input if needed, e.g.
    # if params.shape[1] != 4:
    #     raise ValueError(f"Expected 4 parameters, got {params.shape[1]}")
    # if histograms.shape[1] != 48:
    #     raise ValueError(f"Expected 48 histogram bins, got {histograms.shape[1]}")

    # Convert to PyTorch tensors (use float32 for compatibility with GPs)
    params_tensor = torch.tensor(params, dtype=torch.float32)
    histograms_tensor = torch.tensor(histograms, dtype=torch.float32)

    return params_tensor, histograms_tensor

def load_covariance_matrix(cov_path: str) -> torch.Tensor:
    """Loads the covariance matrix and converts it to a PyTorch tensor."""
    covariance_matrix = load_npy_data(cov_path)

    # Validate shape - should be square and match histogram bins
    # if covariance_matrix.ndim != 2 or covariance_matrix.shape[0] != covariance_matrix.shape[1]:
    #     raise ValueError("Covariance matrix must be 2D and square.")
    # Add specific shape validation if needed, e.g.
    # if covariance_matrix.shape[0] != 48:
    #     raise ValueError(f"Covariance matrix dimensions ({covariance_matrix.shape[0]}) "
    #                      f"must match the number of histogram bins (48).")

    # Convert to PyTorch tensor
    covariance_tensor = torch.tensor(covariance_matrix, dtype=torch.float32)

    return covariance_tensor

def load_observed_data(obs_path: str) -> torch.Tensor:
    """Loads the observed histogram data."""
    observed_hist = load_npy_data(obs_path)

    # Validate shape if necessary (e.g., should be 1D with 48 bins)
    # if observed_hist.ndim != 1 or observed_hist.shape[0] != 48:
    #     raise ValueError(f"Observed histogram must be 1D with 48 bins, got shape {observed_hist.shape}")

    observed_tensor = torch.tensor(observed_hist, dtype=torch.float32)
    return observed_tensor 