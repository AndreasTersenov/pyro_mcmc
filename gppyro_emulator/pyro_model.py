import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

from .gp_emulator import IndependentGPEmulator # Use relative import

def pyro_inference_model(
    emulator: IndependentGPEmulator,
    covariance_matrix: torch.Tensor,
    observed_histogram: torch.Tensor | None = None,
    param_priors: dict | None = None,
):
    """
    Pyro model defining priors, GP prediction, and likelihood.

    Args:
        emulator: The trained IndependentGPEmulator instance.
        covariance_matrix: The covariance matrix for the likelihood (expected on correct device).
        observed_histogram: The observed data vector (histogram) (expected on correct device).
        param_priors: Optional dictionary specifying pyro.distributions for each parameter.
                      Defaults to Uniform(0, 1) for each of the 4 parameters.
    """
    num_params = 4 # As specified by the user

    # --- Define Priors for the 4 parameters --- 
    # Default priors: Uniform(0, 1) - Remove .to(device)
    default_priors = {
        f'param_{i}': dist.Uniform(0.0, 1.0) for i in range(num_params)
    }
    if param_priors:
        # Priors don't need explicit device placement
        priors = {**default_priors, **param_priors}
    else:
        priors = default_priors

    # Ensure we have priors for all parameters
    if len(priors) != num_params or not all(f'param_{i}' in priors for i in range(num_params)):
         raise ValueError(f"Must provide a prior distribution for all {num_params} parameters.")

    # Sample parameters - Pyro handles device placement implicitly
    params = torch.stack([
        pyro.sample(f'param_{i}', priors[f'param_{i}'])
        for i in range(num_params)
    ])
    
    # Ensure params tensor is 2D for the GP predictor [1, num_params]
    params_input = params.unsqueeze(0)

    # --- Get GP Prediction --- 
    # Emulator predict method handles its own device placement
    # Input params_input should be on the correct device from sampling
    predicted_histogram_mean = emulator.predict(params_input).squeeze(0)

    # --- Define Likelihood --- 
    # Ensure covariance and observation are on the correct device (handled in run_mcmc)
    # MultivariateNormal will use the device of loc and covariance_matrix
    pyro.sample(
        "obs",
        dist.MultivariateNormal(loc=predicted_histogram_mean, covariance_matrix=covariance_matrix),
        obs=observed_histogram
    )

def run_pyro_mcmc(
    emulator: IndependentGPEmulator,
    covariance_matrix: torch.Tensor,
    observed_histogram: torch.Tensor,
    param_priors: dict | None = None,
    num_samples: int = 1000,
    warmup_steps: int = 500,
    num_chains: int = 1,
    use_gpu: bool = False,
    jit_compile: bool = False
) -> MCMC:
    """
    Sets up and runs the NUTS MCMC sampler for the defined Pyro model.

    Args:
        emulator: The trained IndependentGPEmulator.
        covariance_matrix: The likelihood covariance matrix.
        observed_histogram: The observed data vector.
        param_priors: Optional dictionary specifying parameter priors for the model.
        num_samples: Number of MCMC samples to draw.
        warmup_steps: Number of warmup steps for the NUTS sampler.
        num_chains: Number of MCMC chains to run.
        use_gpu: Whether to run computations on GPU (if available).
        jit_compile: Whether to JIT compile the model for speed.

    Returns:
        An MCMC object containing the results.
    """
    # Determine target device: prioritize CUDA if available, fallback to CPU
    can_use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu and can_use_cuda else "cpu")
    print(f"Setting up MCMC on device: {device}")
    if use_gpu and not can_use_cuda:
        print("Warning: GPU (CUDA) requested for MCMC but not available, using CPU.")

    # Ensure the emulator's models are on the target device
    # This assumes emulator was trained potentially on a different device
    if emulator._trained and emulator.models[0] is not None:
        model_device = next(emulator.models[0].parameters()).device
        if model_device != device:
            print(f"Moving emulator models from {model_device} to {device} for MCMC.")
            for i in range(emulator.num_outputs):
                if emulator.models[i] is not None:
                    emulator.models[i] = emulator.models[i].to(device)
                if emulator.likelihoods[i] is not None:
                    emulator.likelihoods[i] = emulator.likelihoods[i].to(device)
        else:
             print(f"Emulator models already on target device: {device}")
    elif not emulator._trained:
         raise RuntimeError("Emulator must be trained before running MCMC.")
    
    # Move data to the target device
    covariance_matrix = covariance_matrix.to(device)
    observed_histogram = observed_histogram.to(device)

    print("Setting up NUTS kernel...")
    # The model no longer takes the device argument explicitly
    kernel = NUTS(
        pyro_inference_model, # Pass the model directly
        jit_compile=jit_compile,
        ignore_jit_warnings=True
    )

    mcmc = MCMC(
        kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        num_chains=num_chains,
        mp_context="spawn" if num_chains > 1 else None, 
        disable_progbar=False,
    )

    # Prepare model args (data tensors are already on the target device)
    model_args = {
        'emulator': emulator,
        'covariance_matrix': covariance_matrix,
        'observed_histogram': observed_histogram,
        'param_priors': param_priors
    }

    print(f"Running MCMC with {num_chains} chain(s), {num_samples} samples, and {warmup_steps} warmup steps on {device}...")
    mcmc.run(**model_args)

    print("MCMC run complete.")
    return mcmc 