import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.optim import Adam

# Define the GP model
class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        # Use Matern Kernel as a robust default, can be changed
        self.covar_module = ScaleKernel(MaternKernel(ard_num_dims=train_x.shape[-1]))
        # Alternative: RBF Kernel
        # self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[-1]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class IndependentGPEmulator:
    """Manages a collection of independent GP models, one for each output dimension."""
    def __init__(self, num_outputs: int):
        self.num_outputs = num_outputs
        self.models = [None] * num_outputs
        self.likelihoods = [None] * num_outputs
        self._trained = False
        # We'll determine the device from the models after training

    def train_gps(self, train_x: torch.Tensor, train_y: torch.Tensor, training_iterations: int = 100, learning_rate: float = 0.1, use_gpu: bool = False):
        """Trains an independent GP for each output dimension."""
        if train_y.shape[1] != self.num_outputs:
            raise ValueError(f"train_y has {train_y.shape[1]} outputs, but emulator expects {self.num_outputs}")

        # Determine target device: prioritize CUDA if available, fallback to CPU
        can_use_cuda = torch.cuda.is_available()
        target_device = torch.device("cuda" if use_gpu and can_use_cuda else "cpu")
        print(f"Training GPs on device: {target_device}")
        if use_gpu and not can_use_cuda:
            print("Warning: GPU (CUDA) requested but not available, using CPU.")
        
        # Move training data to target device *once* before the loop
        train_x = train_x.to(target_device)
        train_y = train_y.to(target_device)

        print(f"Training {self.num_outputs} independent GPs...")
        for i in range(self.num_outputs):
            # Select the target data for this GP (already on target_device)
            train_y_i = train_y[:, i]

            # Initialize likelihood and model directly on the target device
            self.likelihoods[i] = GaussianLikelihood().to(target_device)
            self.models[i] = ExactGPModel(train_x, train_y_i, self.likelihoods[i]).to(target_device)

            # Find optimal model hyperparameters
            self.models[i].train()
            self.likelihoods[i].train()

            # Use the Adam optimizer
            optimizer = Adam(self.models[i].parameters(), lr=learning_rate)

            # "Loss" for GPs - the marginal log likelihood
            mll = ExactMarginalLogLikelihood(self.likelihoods[i], self.models[i])

            for iter_num in range(training_iterations):
                optimizer.zero_grad()
                output = self.models[i](train_x) # train_x is already on target_device
                loss = -mll(output, train_y_i) # train_y_i is already on target_device
                loss.backward()
                optimizer.step()
            
            # Models and likelihoods remain on target_device
            # Optional: Clear cache periodically if memory is tight during long training
            # if target_device.type == 'cuda':
            #     torch.cuda.empty_cache()

        self._trained = True
        print("Training complete.")

    def predict(self, test_x: torch.Tensor) -> torch.Tensor:
        """Makes predictions using the trained GPs."""
        if not self._trained:
            raise RuntimeError("Emulator must be trained before making predictions.")
        if self.models[0] is None:
             raise RuntimeError("Models have not been trained yet.")
        if test_x.dim() == 1: # Handle single input point
            test_x = test_x.unsqueeze(0)
        if test_x.dim() != 2:
            raise ValueError("Input test_x must be a 2D tensor (n_samples, n_params).")

        n_samples = test_x.shape[0]
        
        # Determine device from the first model (where training placed it)
        model_device = next(self.models[0].parameters()).device
        print(f"Predicting on device: {model_device}")

        # Move test data to the model's device
        test_x = test_x.to(model_device)

        # Ensure predictions tensor is created on the model device initially
        predictions = torch.zeros(n_samples, self.num_outputs, device=model_device)

        print(f"Predicting for {n_samples} input points...")
        for i in range(self.num_outputs):
            if self.models[i] is None or self.likelihoods[i] is None:
                 raise RuntimeError(f"Model or likelihood for output {i} not trained/initialized.")

            self.models[i].eval()
            self.likelihoods[i].eval()
            # Ensure model/likelihood are on the correct device (should be already)
            self.models[i] = self.models[i].to(model_device)
            self.likelihoods[i] = self.likelihoods[i].to(model_device)

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihoods[i](self.models[i](test_x))
                predictions[:, i] = observed_pred.mean # Already on model_device

        # Return predictions on the device they were computed on (model_device)
        # The calling code (e.g., MCMC) should handle moving if necessary
        return predictions

    def get_models_and_likelihoods(self) -> tuple[list[ExactGPModel], list[GaussianLikelihood]]:
        """Returns the trained models and likelihoods."""
        if not self._trained:
            raise RuntimeError("Emulator must be trained first.")
        return self.models, self.likelihoods 