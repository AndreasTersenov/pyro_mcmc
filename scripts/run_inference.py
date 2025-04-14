import torch
import pyro
import pyro.distributions as dist
import os
import matplotlib.pyplot as plt
import sys

# Make sure the main package directory is in the Python path
# This might be needed if running directly from the scripts directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

try:
    from getdist import plots, MCSamples # Use getdist
    getdist_available = True
except ImportError:
    getdist_available = False
    print("Optional dependency 'getdist' not found. Posterior plot will not be generated.")
    print("Install using: pip install getdist")

from gppyro_emulator.utils import (
    load_training_data,
    load_covariance_matrix,
    load_observed_data
)
from gppyro_emulator.gp_emulator import IndependentGPEmulator
from gppyro_emulator.pyro_model import run_pyro_mcmc

# --- Configuration ---
# Define data paths (relative to the project root)
DATA_DIR = os.path.join(project_root, "data")
PARAMS_PATH = os.path.join(DATA_DIR, "parameters.npy")      # Shape: (25, 4)
HIST_PATH = os.path.join(DATA_DIR, "histograms.npy")      # Shape: (25, 48)
COV_PATH = os.path.join(DATA_DIR, "covariance.npy")       # Shape: (48, 48)
OBS_PATH = os.path.join(DATA_DIR, "observed_histogram.npy") # Shape: (48,)

# GP Training settings
GP_TRAINING_ITERATIONS = 150
GP_LEARNING_RATE = 0.1
USE_GPU_GP_TRAINING = True # Set to True to train GPs on GPU (if available)

# MCMC settings
NUM_SAMPLES = 1000
WARMUP_STEPS = 500
NUM_CHAINS = 2
USE_GPU_MCMC = True # Set to True to run MCMC on GPU (if available)
JIT_COMPILE = False

# Parameter Priors (Optional - uncomment and adjust if needed)
# Example: Normal priors centered around 0.5 with std dev 0.2
# param_priors = {
#     'param_0': dist.Normal(0.5, 0.2),
#     'param_1': dist.Normal(0.5, 0.2),
#     'param_2': dist.Normal(0.5, 0.2),
#     'param_3': dist.Normal(0.5, 0.2),
# }
# If commented out, defaults to Uniform(0, 1) for each parameter
param_priors = None # Use default Uniform(0, 1) priors

# Parameter names for plotting
PARAM_NAMES = ["param_0", "param_1", "param_2", "param_3"] # Use names getdist expects
PARAM_LABELS = ["Param 0", "Param 1", "Param 2", "Param 3"] # Labels for the plot

# --- 1. Load Data ---
print("Loading data...")
try:
    train_params, train_histograms = load_training_data(PARAMS_PATH, HIST_PATH)
    covariance_matrix = load_covariance_matrix(COV_PATH)
    observed_histogram = load_observed_data(OBS_PATH)
    print(f"  Training Params shape: {train_params.shape}")
    print(f"  Training Histograms shape: {train_histograms.shape}")
    print(f"  Covariance Matrix shape: {covariance_matrix.shape}")
    print(f"  Observed Histogram shape: {observed_histogram.shape}")

    # Basic shape validation
    num_outputs = train_histograms.shape[1]
    num_params_input = train_params.shape[1]
    if covariance_matrix.shape != (num_outputs, num_outputs):
        raise ValueError("Covariance matrix shape mismatch with histogram bins.")
    if observed_histogram.shape != (num_outputs,):
        raise ValueError("Observed histogram shape mismatch with histogram bins.")
    if num_params_input != 4:
         print(f"Warning: Expected 4 input parameters based on priors/names, but got {num_params_input}. Plotting might be affected.")

except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure the following files exist in the 'data/' directory relative to the project root:")
    print(f"  - parameters.npy (e.g., {PARAMS_PATH})")
    print(f"  - histograms.npy (e.g., {HIST_PATH})")
    print(f"  - covariance.npy (e.g., {COV_PATH})")
    print(f"  - observed_histogram.npy (e.g., {OBS_PATH})")
    # Create dummy data if files not found for demonstration
    print("\nCreating dummy data for demonstration purposes...")
    num_train_samples = 25
    num_params_input = 4
    num_outputs = 48
    train_params = torch.rand(num_train_samples, num_params_input, dtype=torch.float32)
    train_histograms = torch.randn(num_train_samples, num_outputs, dtype=torch.float32) * 5 + train_params @ torch.randn(num_params_input, num_outputs) # Make y depend somewhat on x
    # Ensure positive definite covariance
    A = torch.randn(num_outputs, num_outputs, dtype=torch.float32)
    covariance_matrix = torch.matmul(A, A.T) + torch.eye(num_outputs) * 1e-3 # Add jitter
    observed_histogram = torch.randn(num_outputs, dtype=torch.float32) + 2 # Dummy observed data
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    print("Dummy data created in memory. No files were saved.")
except ValueError as e:
     print(f"Data loading error: {e}")
     sys.exit(1)

# Note: Data is loaded to CPU initially by the utility functions

# --- 2. Initialize and Train GP Emulator ---
print("\nInitializing GP Emulator...")
emulator = IndependentGPEmulator(num_outputs=num_outputs)

print("Training GP Emulator...")
emulator.train_gps(
    train_params,
    train_histograms,
    training_iterations=GP_TRAINING_ITERATIONS,
    learning_rate=GP_LEARNING_RATE,
    use_gpu=USE_GPU_GP_TRAINING
)

# --- 3. Run Pyro MCMC Inference ---
print("\nRunning MCMC Inference...")
mcmc_result = run_pyro_mcmc(
    emulator=emulator,
    covariance_matrix=covariance_matrix,
    observed_histogram=observed_histogram,
    param_priors=param_priors,
    num_samples=NUM_SAMPLES,
    warmup_steps=WARMUP_STEPS,
    num_chains=NUM_CHAINS,
    use_gpu=USE_GPU_MCMC,
    jit_compile=JIT_COMPILE
)

# --- 4. Process and Visualize Results ---
print("\nProcessing MCMC results...")
mcmc_result.summary()

# Get posterior samples
posterior_samples = mcmc_result.get_samples()
# Adjust extraction for getdist: it expects samples array and names
param_samples_getdist = posterior_samples # Keep the dictionary for now
param_samples_numpy = torch.stack([posterior_samples[name] for name in PARAM_NAMES]).T.cpu().numpy()

print(f"\nPosterior samples shape (numpy): {param_samples_numpy.shape}")

# Check for divergences (important for NUTS diagnostics)
diagnostics = mcmc_result.diagnostics()
print("\nMCMC Diagnostics:")
# Use .get() for safer access in case keys are missing
num_divergences_info = diagnostics.get('divergences', {})
if isinstance(num_divergences_info, dict):
    num_divergences = num_divergences_info.get('param_0', 'N/A') # NUTS divergences often reported per parameter
    print(f"Number of divergences reported (example for param_0): {num_divergences}")
    # A more robust check might iterate through parameters or check a summary field if available
    # For simplicity, we check if any non-zero divergence count exists
    has_divergences = any(v > 0 for k, v in num_divergences_info.items() if isinstance(v, (int, float)))
else:
    print("Could not parse divergence information from diagnostics.")
    has_divergences = False

if has_divergences:
    print("Warning: Divergences detected! Check model, priors, or sampler settings (e.g., adapt_delta, step_size).")

# Create a getdist corner plot
if getdist_available:
    if len(PARAM_NAMES) == param_samples_numpy.shape[1]:
        print("\nGenerating getdist corner plot...")
        try:
            # Create MCSamples object
            # Assumes samples are in shape (num_samples, num_params)
            samples = MCSamples(
                samples=param_samples_numpy,
                names=PARAM_NAMES,
                labels=PARAM_LABELS,
                label='Posterior Samples'
            )

            # Generate plot
            g = plots.get_subplot_plotter()
            g.triangle_plot(samples, filled=True)

            # Save the plot
            plot_path = os.path.join(project_root, "posterior_getdist_corner_plot.png")
            g.export(plot_path)
            print(f"Getdist corner plot saved to {plot_path}")
            # plt.show() # Uncomment to display the plot interactively (if needed, getdist might show it)

        except Exception as e:
            print(f"Error generating getdist corner plot: {e}")
    else:
        print("\nSkipping getdist corner plot: Number of parameter names does not match extracted samples.")
else:
    print("\nSkipping getdist corner plot generation ('getdist' package not installed).")

print("\nScript finished.") 