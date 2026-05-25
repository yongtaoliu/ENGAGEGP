"""
Gaussian Process Regression with Deep Kernel Learning.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import RBFKernel, MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import norm

from .models import get_feature_extractor
from .kernels import AttentionWeightedRBFKernel, InputMixingRBFKernel


# ============================================================================
# Confidence-Weighted MLL for Regression
# ============================================================================

class ConfidenceWeightedMLL(nn.Module):
    """
    Marginal log likelihood with confidence weighting for regression.
    
    FIXED: Ensures scalar output for backpropagation.
    
    Parameters
    ----------
    likelihood : gpytorch.likelihoods.Likelihood
        GP likelihood
    model : gpytorch.models.GP
        GP model
    confidence_weights : torch.Tensor
        Confidence weights for each data point, shape (n,)
    """
    
    def __init__(self, likelihood, model, confidence_weights):
        super().__init__()
        self.likelihood = likelihood
        self.model = model
        
        if confidence_weights.dtype != torch.float64:
            confidence_weights = confidence_weights.double()
        self.confidence_weights = confidence_weights
        
        # Normalize weights to maintain scale
        if confidence_weights.sum() > 0:
            self.normalized_weights = (
                confidence_weights / confidence_weights.sum() * len(confidence_weights)
            )
        else:
            self.normalized_weights = confidence_weights
    
    def forward(self, output, target):
        """
        Compute weighted marginal log likelihood.
        
        Parameters
        ----------
        output : gpytorch.distributions.MultivariateNormal
            GP posterior
        target : torch.Tensor
            Target values
            
        Returns
        -------
        torch.Tensor
            Weighted log likelihood (SCALAR)
        """
        mean = output.mean
        variance = output.variance
        
        # FIX: Ensure everything is 1D to avoid shape issues
        if target.dim() > 1:
            target = target.squeeze()
        if mean.dim() > 1:
            mean = mean.squeeze()
        if variance.dim() > 1:
            variance = variance.squeeze()
        
        # Compute residuals
        residuals = target - mean
        
        # Compute log probabilities
        log_probs = -0.5 * (
            torch.log(2 * torch.pi * variance) + 
            (residuals ** 2) / variance
        )
        
        # Weight by confidence
        weighted_log_probs = self.normalized_weights * log_probs
        
        # FIX: CRITICAL - Always return scalar by summing
        return weighted_log_probs.sum()


class SampleWeightedMLL(nn.Module):
    """
    Marginal log likelihood with learnable sample weights.
    
    Similar to ConfidenceWeightedMLL but weights are dynamically updated
    during training for sample-level attention.
    
    Parameters
    ----------
    likelihood : gpytorch.likelihoods.Likelihood
        GP likelihood
    model : gpytorch.models.GP
        GP model
    sample_weight_module : SampleWeightModule
        Module containing learnable sample weights
    """
    
    def __init__(self, likelihood, model, sample_weight_module):
        super().__init__()
        self.likelihood = likelihood
        self.model = model
        self.sample_weight_module = sample_weight_module
    
    def forward(self, output, target):
        """
        Compute weighted marginal log likelihood with dynamic sample weights.
        
        Parameters
        ----------
        output : gpytorch.distributions.MultivariateNormal
            GP posterior
        target : torch.Tensor
            Target values
            
        Returns
        -------
        torch.Tensor
            Weighted log likelihood (SCALAR)
        """
        # Get current sample weights (dynamically updated during training)
        sample_weights = self.sample_weight_module.get_weights()
        
        mean = output.mean
        variance = output.variance
        
        # Ensure everything is 1D to avoid shape issues
        if target.dim() > 1:
            target = target.squeeze()
        if mean.dim() > 1:
            mean = mean.squeeze()
        if variance.dim() > 1:
            variance = variance.squeeze()
        
        # Compute residuals
        residuals = target - mean
        
        # Compute log probabilities per sample
        log_probs = -0.5 * (
            torch.log(2 * torch.pi * variance) + 
            (residuals ** 2) / variance
        )
        
        # Weight by learnable sample weights
        weighted_log_probs = sample_weights * log_probs
        
        # Return scalar
        return weighted_log_probs.sum()
    
# ============================================================================
# Deep Kernel GP Regression Model
# ============================================================================

class DeepKernelGP(nn.Module):
    """
    Deep Kernel Learning for Gaussian Process Regression.
    
    Combines a neural network feature extractor with a Gaussian Process
    to handle high-dimensional inputs while maintaining GP benefits.
    
    Parameters
    ----------
    datapoints : torch.Tensor
        Training inputs, shape (n, input_dim)
    targets : torch.Tensor
        Training targets, shape (n,) or (n, 1)
    input_dim : int
        Dimensionality of input data
    feature_dim : int
        Dimensionality of learned feature space
    hidden_dims : list of int, optional
        Hidden layer dimensions for feature extractor
    extractor_type : str
        Type of feature extractor ('fc', 'fcbn', 'resnet', 'attention', 'wide_deep', 'custom')
    extractor_kwargs : dict, optional
        Additional arguments for feature extractor
    confidence_weights : torch.Tensor, optional
        Confidence weights for each data point
    noise_constraint : gpytorch.constraints.Constraint, optional
        Constraint on observation noise
    dropout : float
        Dropout rate for feature extractor
        
    Attributes
    ----------
    feature_extractor : nn.Module
        Neural network for dimensionality reduction
    gp_model : SingleTaskGP
        Gaussian Process model in feature space
    """
    
    def __init__(
        self,
        datapoints,
        targets,
        input_dim,
        feature_dim=16,
        hidden_dims=None,
        extractor_type='fcbn',
        extractor_kwargs=None,
        confidence_weights=None,
        noise_constraint=None,
        dropout=0.2,
        gp_kernel_type: str = "rbf",
        matern_nu: float = 2.5,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        if extractor_kwargs is None:
            extractor_kwargs = {}

        # extractor_type=None → standard GP: identity extractor, GP lives in input space
        if extractor_type is None:
            feature_dim = input_dim

        # Create feature extractor
        self.feature_extractor = get_feature_extractor(
            extractor_type=extractor_type,
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            **extractor_kwargs
        )

        self.feature_extractor = self.feature_extractor.to(
            device=datapoints.device,
            dtype=datapoints.dtype
        )

        # Extract initial features
        with torch.no_grad():
            train_features = self.feature_extractor(datapoints)

        # Set up GP kernel (default: RBF on feature space).
        # When extractor_type is None, feature_dim is forced to input_dim and the GP
        # operates directly on the original input space. In that case, we optionally
        # enable attention-like mechanisms inside the kernel while keeping X as the
        # kernel input (no learned feature extractor).
        if gp_kernel_type == "rbf":
            base_kernel = RBFKernel(ard_num_dims=feature_dim)
        elif gp_kernel_type == "matern":
            base_kernel = MaternKernel(nu=matern_nu, ard_num_dims=feature_dim)
        elif gp_kernel_type == "matern_12":
            base_kernel = MaternKernel(nu=0.5, ard_num_dims=feature_dim)
        elif gp_kernel_type == "matern_32":
            base_kernel = MaternKernel(nu=1.5, ard_num_dims=feature_dim)
        elif gp_kernel_type == "matern_52":
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=feature_dim)
        elif gp_kernel_type == "attention_weighted" and extractor_type is None:
            base_kernel = AttentionWeightedRBFKernel(input_dim=input_dim)
        elif gp_kernel_type == "direct_attention" and extractor_type is None:
            base_kernel = InputMixingRBFKernel(input_dim=input_dim)
        else:
            raise ValueError(
                "Unknown or incompatible gp_kernel_type="
                f"{gp_kernel_type!r} for extractor_type={extractor_type!r}. "
                "Use 'rbf' or 'matern*' always, or for extractor_type=None use "
                "'attention_weighted' or 'direct_attention'."
            )

        covar_module = ScaleKernel(base_kernel)
        
        # Initialize likelihood
        likelihood = GaussianLikelihood()
        if noise_constraint is not None:
            likelihood.noise_covar.register_constraint("raw_noise", noise_constraint)

        # Handle target shapes
        if targets.ndim == 1:
            targets_for_gp = targets.unsqueeze(-1)
        elif targets.ndim == 2 and targets.shape[-1] == 1:
            targets_for_gp = targets
        else:
            targets_for_gp = targets.squeeze().unsqueeze(-1)

        # Initialize GP model
        self.gp_model = SingleTaskGP(
            train_X=train_features,
            train_Y=targets_for_gp,
            covar_module=covar_module,
            likelihood=likelihood,
            input_transform=Normalize(d=feature_dim),
            outcome_transform=Standardize(m=1)
        )

        self.train_datapoints = datapoints
        self.train_targets = targets.squeeze()
        self.feature_dim = feature_dim
        self.input_dim = input_dim
        self.extractor_type = extractor_type

        # Store confidence weights
        if confidence_weights is not None:
            if confidence_weights.dtype != torch.float64:
                confidence_weights = confidence_weights.double()
            self.confidence_weights = confidence_weights.to(datapoints.device)
        else:
            self.confidence_weights = torch.ones(
                len(datapoints),
                dtype=torch.float64, 
                device=datapoints.device
            )
            
        self.register_buffer('_confidence_weights', self.confidence_weights)

    def forward(self, x):
        """
        Forward pass through the model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch_size, input_dim)
            
        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            GP posterior distribution
        """
        features = self.feature_extractor(x)
        return self.gp_model(features)

    def update_gp_data(self):
        """Update GP training data with current features."""
        features = self.feature_extractor(self.train_datapoints)
        
        # Ensure targets are 1D
        targets = self.train_targets
        if targets.dim() > 1:
            targets = targets.squeeze()
        
        # Convert to (n, 1) for set_train_data
        targets_2d = targets.unsqueeze(-1)
        
        # Update GP data
        self.gp_model.set_train_data(features, targets_2d, strict=False)
        
        # Ensure train_targets remains 1D (GPyTorch expects this)
        self.gp_model.train_targets = targets


# ============================================================================
# Training Functions
# ============================================================================

def train_dkgp(
    datapoints,
    targets,
    input_dim,
    feature_dim=16,
    hidden_dims=None,
    extractor_type='fcbn',
    extractor_kwargs=None,
    confidence_weights=None,
    use_custom_mll=None,
    num_epochs=1000,
    lr_features=1e-4,
    lr_gp=1e-2,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=True,
    patience=None,
    min_delta=1e-4,
    sample_weights_param=None,
    sample_weight_lr=None,
    gp_kernel_type: str = "rbf",
    matern_nu: float = 2.5,
):
    """
    Train Deep Kernel GP for regression.
    
    Parameters
    ----------
    datapoints : np.ndarray or torch.Tensor
        Training datapoints, shape (n, input_dim)
    targets : np.ndarray or torch.Tensor
        Target values, shape (n,) or (n, 1)
    input_dim : int
        Input dimensionality
    feature_dim : int
        Learned feature dimensionality
    hidden_dims : list of int, optional
        Hidden layer dimensions. Default: [256, 128, 64]
    extractor_type : str
        Feature extractor type ('fc', 'fcbn', 'resnet', 'attention', 
        'attention_weighted', 'wide_deep', 'custom')
    extractor_kwargs : dict, optional
        Additional arguments for feature extractor
    confidence_weights : np.ndarray or torch.Tensor, optional
        Confidence weights for data points, shape (n,)
    use_custom_mll : bool, optional
        If True, use ConfidenceWeightedMLL
        If False, use standard ExactMarginalLogLikelihood
        If None (default), auto-select based on confidence_weights
    num_epochs : int
        Number of training epochs
    lr_features : float
        Learning rate for feature extractor
    lr_gp : float
        Learning rate for GP parameters
    device : str
        Device to use ('cuda' or 'cpu')
    verbose : bool
        Print training progress
    patience : int, optional
        Early stopping patience (epochs without improvement)
    min_delta : float
        Minimum improvement to reset patience counter
    sample_weights_param : nn.Parameter, optional
        Learnable sample weights (for sample-level attention)
    sample_weight_lr : float, optional
        Learning rate for sample weights
    
    Returns
    -------
    model : DeepKernelGP
        Trained model
    losses : list
        Training losses per epoch
    """
    if hidden_dims is None:
        hidden_dims = [256, 128, 64]
    
    if extractor_kwargs is None:
        extractor_kwargs = {}
    
    # Convert to tensors
    if not isinstance(datapoints, torch.Tensor):
        datapoints = torch.from_numpy(datapoints).double()
    else:
        datapoints = datapoints.double()

    if not isinstance(targets, torch.Tensor):
        targets = torch.from_numpy(targets).double()
    else:
        targets = targets.double()

    # Handle confidence weights
    if confidence_weights is not None:
        if not isinstance(confidence_weights, torch.Tensor):
            confidence_weights = torch.from_numpy(confidence_weights).double()  
        else:
            confidence_weights = confidence_weights.double()  
        confidence_weights = confidence_weights.to(device)
    else:
        confidence_weights = torch.ones(
            len(datapoints), 
            dtype=torch.float64,  
            device=device
        )

    datapoints = datapoints.to(device)
    targets = targets.to(device)

    # MLL selection logic
    if use_custom_mll is None:
        has_varying_confidence = not torch.allclose(
            confidence_weights, 
            torch.ones_like(confidence_weights)
        )
        use_custom_mll = has_varying_confidence
        
        if verbose:
            if has_varying_confidence:
                print("\nAuto-selected: ConfidenceWeightedMLL")
            else:
                print("\nAuto-selected: ExactMarginalLogLikelihood")
    else:
        if verbose:
            if use_custom_mll:
                print("\nUser-selected: ConfidenceWeightedMLL")
            else:
                print("\nUser-selected: ExactMarginalLogLikelihood")

    # Create model
    model = DeepKernelGP(
        datapoints=datapoints,
        targets=targets,
        input_dim=input_dim,
        feature_dim=feature_dim,
        hidden_dims=hidden_dims,
        extractor_type=extractor_type,
        extractor_kwargs=extractor_kwargs,
        confidence_weights=confidence_weights,
        gp_kernel_type=gp_kernel_type,
        matern_nu=matern_nu,
    ).to(device)
    
    # Handle sample-level attention
    if sample_weights_param is not None:
        from .sample_weighting import SampleWeightModule
        
        # Create sample weight module
        model.sample_weight_module = SampleWeightModule(len(datapoints))
        model.sample_weight_module.log_weights = sample_weights_param
        model.sample_weight_module = model.sample_weight_module.to(device)
        
        # Register parameter with model  
        model.register_parameter('sample_weights_log', sample_weights_param)

    # Optimizer — skip feature extractor group when it has no parameters (extractor_type=None)
    has_extractor_params = len(list(model.feature_extractor.parameters())) > 0
    param_groups = []
    if has_extractor_params:
        param_groups.append({'params': model.feature_extractor.parameters(), 'lr': lr_features})
    param_groups.append({'params': model.gp_model.parameters(), 'lr': lr_gp})
    if sample_weights_param is not None:
        param_groups.append({'params': [sample_weights_param], 'lr': sample_weight_lr})
    optimizer = torch.optim.Adam(param_groups)

    # Select MLL
    if sample_weights_param is not None:
        # Use sample-weighted MLL (learnable weights)
        mll = SampleWeightedMLL(
            model.gp_model.likelihood,
            model.gp_model,
            model.sample_weight_module
        )
        mll_name = "SampleWeightedMLL"
    elif use_custom_mll:
        # Use confidence-weighted MLL (fixed weights)
        mll = ConfidenceWeightedMLL(
            model.gp_model.likelihood,
            model.gp_model,
            confidence_weights
        )
        mll_name = "ConfidenceWeightedMLL"
    else:
        # Standard MLL
        mll = ExactMarginalLogLikelihood(
            model.gp_model.likelihood,
            model.gp_model
        )
        mll_name = "ExactMarginalLogLikelihood"

    model.train()
    losses = []
    best_loss = float('inf')
    patience_counter = 0

    if verbose:
        if extractor_type is None:
            print(f"\nTraining Standard GP (no deep kernel)")
        else:
            print(f"\nTraining Deep Kernel GP")
        print("=" * 60)
        print(f"  Device: {device}")
        print(f"  Extractor type: {extractor_type if extractor_type is not None else 'None (standard GP)'}")
        print(f"  Input dim: {input_dim} → Feature dim: {model.feature_dim}")
        print(f"  Data points: {len(datapoints)}")
        if extractor_type is not None:
            print(f"  Hidden layers: {hidden_dims}")
        print(f"  MLL: {mll_name}")
        if patience:
            print(f"  Early stopping: patience={patience}, min_delta={min_delta}")
        print("=" * 60)

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        model.update_gp_data()
        output = model.gp_model(*model.gp_model.train_inputs)
        
        loss = -mll(output, model.gp_model.train_targets)
        
        # Ensure scalar
        if isinstance(loss, torch.Tensor):
            if loss.dim() > 0 or loss.numel() != 1:
                loss = loss.sum()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        current_loss = loss.item()
        losses.append(current_loss)

        # Early stopping
        if patience is not None:
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break

        if verbose and (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:4d}/{num_epochs}, Loss: {current_loss:.4f}")

    model.eval()
    
    # Store final sample weights if learned
    if sample_weights_param is not None:
        model.sample_weights = model.sample_weight_module.get_weights().detach()
    
    if verbose:
        print("=" * 60)
        print(f"Training complete! Final loss: {losses[-1]:.4f}")
        if sample_weights_param is not None:
            sw = model.sample_weights.cpu().numpy()
            print(f"Sample weights: min={sw.min():.3f}, max={sw.max():.3f}, "
                  f"mean={sw.mean():.3f}")
            n_low = (sw < 0.5).sum()
            print(f"Samples with low weight (<0.5): {n_low}/{len(sw)}")
        print("=" * 60)
    
    return model, losses


def fit_dkgp(
    X_train, 
    y_train, 
    confidence_weights=None, 
    use_custom_mll=None, 
    learn_sample_weights=False, 
    sample_weight_lr=0.01, 
    feature_dim=16, 
    hidden_dims=None,
    extractor_type='fcbn',
    extractor_kwargs=None,
    num_epochs=2000, 
    lr_features=1e-4,
    lr_gp=1e-2,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=True,
    patience=None,
    min_delta=1e-4,
    gp_kernel_type: str = "rbf",
    matern_nu: float = 2.5,
):
    """
    Fit Deep Kernel GP regression model.
    
    This is a high-level interface that handles training and returns all
    necessary objects for prediction and further analysis.

    Parameters
    ----------
    X_train : np.ndarray or torch.Tensor
        High-dimensional features (N, D)
    y_train : np.ndarray or torch.Tensor
        Target values (N,) or (N, 1)
    confidence_weights : np.ndarray or torch.Tensor, optional
        Confidence weights for each data point, shape (N,)
    use_custom_mll : bool, optional
        Explicitly choose MLL type (None = auto-select)
    learn_sample_weights : bool
        If True, learn per-sample weights to downweight noisy/outlier samples.
        This is sample-level attention: model learns which training samples
        are reliable vs noisy. Default: False
    sample_weight_lr : float
        Learning rate for sample weights (only used if learn_sample_weights=True).
        Default: 0.01
    feature_dim : int
        Dimensionality of learned feature space
    hidden_dims : list of int, optional
        Hidden layer dimensions. Default: [256, 128, 64]
    extractor_type : str or None
        Feature extractor type: 'fc', 'fcbn', 'resnet', 'attention',
        'attention_weighted', 'direct_attention', 'custom', or None.
        Pass ``None`` for a standard GP with no deep kernel — the GP kernel
        operates directly on the raw input features.
    extractor_kwargs : dict, optional
        Additional arguments for feature extractor
    num_epochs : int
        Number of training epochs
    lr_features : float
        Learning rate for feature extractor
    lr_gp : float
        Learning rate for GP parameters
    device : str
        Device to use
    verbose : bool
        Print training information
    patience : int, optional
        Early stopping patience
    min_delta : float
        Minimum improvement for early stopping

    Returns
    -------
    mll : MarginalLogLikelihood
        Marginal log likelihood object
    gp_model : SingleTaskGP
        The GP model (operates in feature space)
    dkl_model : DeepKernelGP
        Complete deep kernel model with feature extractor
    losses : list
        Training losses
    sample_weights : np.ndarray, optional
        Learned sample weights (only if learn_sample_weights=True)
        Shape: (N,), values in [0, 1], mean ≈ 1.0
        Low weights indicate noisy/outlier samples
        
    Examples
    --------
    >>> # Default extractor (FC + BatchNorm)
    >>> mll, gp, dkl, losses = fit_dkgp(X, y, feature_dim=16)
    
    >>> # With sample-level attention (detect noisy samples)
    >>> mll, gp, dkl, losses, sample_weights = fit_dkgp(
    ...     X, y, 
    ...     learn_sample_weights=True,  # Enable sample weighting
    ...     sample_weight_lr=0.01
    ... )
    >>> # Find noisy samples
    >>> noisy = np.where(sample_weights < 0.5)[0]
    >>> print(f"Detected {len(noisy)} noisy samples")
    
    >>> # Combine feature-level and sample-level attention
    >>> mll, gp, dkl, losses, sample_weights = fit_dkgp(
    ...     X, y,
    ...     extractor_type='attention_weighted',  # Feature attention
    ...     learn_sample_weights=True  # Sample attention
    ... )
    
    >>> # ResNet extractor
    >>> mll, gp, dkl, losses = fit_dkgp(X, y, extractor_type='resnet',
    ...                                 extractor_kwargs={'hidden_dim': 256, 'num_blocks': 3})
    
    >>> # Custom extractor
    >>> my_net = nn.Sequential(nn.Linear(100, 64), nn.ReLU(), nn.Linear(64, 16))
    >>> mll, gp, dkl, losses = fit_dkgp(X, y, extractor_type='custom',
    ...                                 extractor_kwargs={'custom_extractor': my_net})
    """
    if hidden_dims is None:
        hidden_dims = [256, 128, 64]
    
    if extractor_kwargs is None:
        extractor_kwargs = {}
    
    # Handle sample-level attention
    if learn_sample_weights:
        import numpy as np
        
        # Convert to numpy for easier handling
        if isinstance(X_train, torch.Tensor):
            X_np = X_train.cpu().numpy()
            y_np = y_train.cpu().numpy()
        else:
            X_np = X_train
            y_np = y_train
        
        n_samples = len(X_np)
        
        # Initialize sample weights as learnable parameters
        sample_weights_param = nn.Parameter(torch.zeros(n_samples).double())
        
        if verbose:
            print("=" * 60)
            print("Training Deep Kernel GP with Sample-Level Attention")
            print(f"Feature Extractor: {extractor_type}")
            print(f"Sample Weighting: ENABLED ({n_samples} learnable weights)")
            print("=" * 60)
    else:
        sample_weights_param = None

        if verbose:
            print("=" * 60)
            if extractor_type is None:
                print("Training Standard GP (no deep kernel)")
            else:
                print("Training Deep Kernel GP Regression Model")
                print(f"Feature Extractor: {extractor_type}")
            print("=" * 60)

    input_dim = X_train.shape[-1]

    dkl_model, losses = train_dkgp(
        datapoints=X_train,
        targets=y_train,
        input_dim=input_dim,
        feature_dim=feature_dim,
        hidden_dims=hidden_dims,
        extractor_type=extractor_type,
        extractor_kwargs=extractor_kwargs,
        confidence_weights=confidence_weights,
        use_custom_mll=use_custom_mll,
        num_epochs=num_epochs,
        lr_features=lr_features,
        lr_gp=lr_gp,
        device=device,
        verbose=verbose,
        patience=patience,
        min_delta=min_delta,
        sample_weights_param=sample_weights_param,  # Pass sample weights
        sample_weight_lr=sample_weight_lr if learn_sample_weights else None,
        gp_kernel_type=gp_kernel_type,
        matern_nu=matern_nu,
    )

    gp_model = dkl_model.gp_model
    
    if confidence_weights is not None or use_custom_mll:
        conf_weights = dkl_model.confidence_weights
        mll = ConfidenceWeightedMLL(
            gp_model.likelihood,
            gp_model,
            conf_weights
        )
    else:
        mll = ExactMarginalLogLikelihood(
            gp_model.likelihood,
            gp_model
        )
    
    # Return sample weights if they were learned
    if learn_sample_weights and hasattr(dkl_model, 'sample_weights'):
        sample_weights = dkl_model.sample_weights.detach().cpu().numpy()
        return mll, gp_model, dkl_model, losses, sample_weights
    else:
        return mll, gp_model, dkl_model, losses

# ============================================================================
# Acquisition Functions
# ============================================================================
"""
Acquisition functions for Bayesian optimization with Deep Kernel GP.
"""

def expected_improvement(
    model,
    candidates,
    best_f,
    xi=0.01,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    maximize=True
):
    """
    Expected Improvement acquisition function.
    
    EI measures the expected improvement over the current best observation.
    
    Parameters
    ----------
    model : DeepKernelGP
        Trained model
    candidates : np.ndarray or torch.Tensor
        Candidate points, shape (n_candidates, input_dim)
    best_f : float
        Best observed function value
    xi : float
        Exploration-exploitation trade-off parameter
    device : str
        Device to use
    maximize : bool
        If True, maximize the function (default)
        If False, minimize the function
    
    Returns
    -------
    ei_values : np.ndarray
        Expected improvement values for each candidate
    """
    if not isinstance(candidates, torch.Tensor):
        candidates = torch.from_numpy(candidates).double()
    else:
        candidates = candidates.double()
    
    candidates = candidates.to(device)
    model.eval()
    
    with torch.no_grad():
        posterior = model(candidates)
        mean = posterior.mean.cpu().numpy().squeeze()
        std = posterior.variance.cpu().numpy().squeeze() ** 0.5
    
    # Flip sign for minimization
    if not maximize:
        mean = -mean
        best_f = -best_f
    
    # Compute EI
    improvement = mean - best_f - xi
    Z = improvement / (std + 1e-9)
    
    ei = improvement * norm.cdf(Z) + std * norm.pdf(Z)
    ei[std == 0.0] = 0.0
    
    return ei

def upper_confidence_bound(
    model,
    candidates,
    beta=2.0,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    maximize=True
):
    """
    Upper Confidence Bound (UCB) acquisition function.
    
    UCB balances exploitation (high mean) with exploration (high uncertainty).
    
    Parameters
    ----------
    model : DeepKernelGP
        Trained model
    candidates : np.ndarray or torch.Tensor
        Candidate points, shape (n_candidates, input_dim)
    beta : float
        Exploration parameter (higher values favor exploration)
    device : str
        Device to use
    maximize : bool
        If True, maximize (UCB). If False, minimize (LCB)
    
    Returns
    -------
    ucb_values : np.ndarray
        UCB values for each candidate
    """
    if not isinstance(candidates, torch.Tensor):
        candidates = torch.from_numpy(candidates).double()
    else:
        candidates = candidates.double()
    
    candidates = candidates.to(device)
    model.eval()
    
    with torch.no_grad():
        posterior = model(candidates)
        mean = posterior.mean.cpu().numpy().squeeze()
        std = posterior.variance.cpu().numpy().squeeze() ** 0.5
    
    if maximize:
        ucb = mean + beta * std
    else:
        ucb = mean - beta * std
    
    return ucb


def probability_of_improvement(
    model,
    candidates,
    best_f,
    xi=0.01,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    maximize=True
):
    """
    Probability of Improvement acquisition function.
    
    PI measures the probability that a candidate will improve over the best.
    
    Parameters
    ----------
    model : DeepKernelGP
        Trained model
    candidates : np.ndarray or torch.Tensor
        Candidate points, shape (n_candidates, input_dim)
    best_f : float
        Best observed function value
    xi : float
        Exploration-exploitation trade-off parameter
    device : str
        Device to use
    maximize : bool
        If True, maximize the function
        If False, minimize the function
    
    Returns
    -------
    pi_values : np.ndarray
        Probability of improvement for each candidate
    """
    if not isinstance(candidates, torch.Tensor):
        candidates = torch.from_numpy(candidates).double()
    else:
        candidates = candidates.double()
    
    candidates = candidates.to(device)
    model.eval()
    
    with torch.no_grad():
        posterior = model(candidates)
        mean = posterior.mean.cpu().numpy().squeeze()
        std = posterior.variance.cpu().numpy().squeeze() ** 0.5
    
    # Flip sign for minimization
    if not maximize:
        mean = -mean
        best_f = -best_f
    
    # Compute PI
    improvement = mean - best_f - xi
    Z = improvement / (std + 1e-9)
    
    pi = norm.cdf(Z)
    pi[std == 0.0] = 0.0
    
    return pi

def thompson_sampling(
    model,
    candidates,
    n_samples=1,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    seed=None
):
    """
    Thompson Sampling acquisition function.
    
    Samples from the posterior and selects points with high sampled values.
    
    Parameters
    ----------
    model : DeepKernelGP
        Trained model
    candidates : np.ndarray or torch.Tensor
        Candidate points, shape (n_candidates, input_dim)
    n_samples : int
        Number of samples to draw
    device : str
        Device to use
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    samples : np.ndarray
        Samples from posterior, shape (n_samples, n_candidates)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    if not isinstance(candidates, torch.Tensor):
        candidates = torch.from_numpy(candidates).double()
    else:
        candidates = candidates.double()
    
    candidates = candidates.to(device)
    model.eval()
    
    with torch.no_grad():
        posterior = model(candidates)
        samples = posterior.sample(torch.Size([n_samples]))
        samples_np = samples.cpu().numpy().squeeze()
    
    return samples_np

def expected_improvement_with_constraints(
    model,
    candidates,
    best_f,
    constraint_models=None,
    constraint_thresholds=None,
    xi=0.01,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    maximize=True
):
    """
    Expected Improvement with constraints.
    
    Computes EI weighted by probability of satisfying constraints.
    
    Parameters
    ----------
    model : DeepKernelGP
        Trained model for objective
    candidates : np.ndarray or torch.Tensor
        Candidate points
    best_f : float
        Best observed function value
    constraint_models : list of DeepKernelGP, optional
        Models for constraint functions
    constraint_thresholds : list of float, optional
        Thresholds for constraints (constraint_i <= threshold_i)
    xi : float
        Exploration parameter
    device : str
        Device to use
    maximize : bool
        If True, maximize objective
    
    Returns
    -------
    constrained_ei : np.ndarray
        Constrained EI values
    """
    # Compute standard EI
    ei = expected_improvement(model, candidates, best_f, xi, device, maximize)
    
    if constraint_models is None or constraint_thresholds is None:
        return ei
    
    if not isinstance(candidates, torch.Tensor):
        candidates = torch.from_numpy(candidates).double()
    else:
        candidates = candidates.double()
    
    candidates = candidates.to(device)
    
    # Compute constraint satisfaction probability
    constraint_prob = np.ones(len(candidates))
    
    for c_model, threshold in zip(constraint_models, constraint_thresholds):
        c_model.eval()
        with torch.no_grad():
            posterior = c_model(candidates)
            mean = posterior.mean.cpu().numpy().squeeze()
            std = posterior.variance.cpu().numpy().squeeze() ** 0.5
        
        # P(constraint <= threshold)
        Z = (threshold - mean) / (std + 1e-9)
        prob_feasible = norm.cdf(Z)
        constraint_prob *= prob_feasible
    
    # Weight EI by constraint satisfaction probability
    constrained_ei = ei * constraint_prob
    
    return constrained_ei


# ============================================================================
# Prediction Functions
# ============================================================================
"""
Prediction utilities for Deep Kernel GP models.
"""

def predict_dkgpr(
    model, 
    test_data,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    return_std=False,
    batch_size=None
):
    """
    Predict outputs for test data.
    
    Parameters
    ----------
    model : DeepKernelGP
        Trained model
    test_data : np.ndarray or torch.Tensor
        Test features, shape (n_test, input_dim)
    device : str
        Device to use ('cuda' or 'cpu')
    return_std : bool
        If True, return standard deviation instead of variance
    batch_size : int, optional
        Process test data in batches to save memory
    
    Returns
    -------
    mean : np.ndarray
        Predicted means, shape (n_test,)
    uncertainty : np.ndarray
        Predicted variance (or std if return_std=True), shape (n_test,)
    """
    if not isinstance(test_data, torch.Tensor):
        test_data = torch.from_numpy(test_data).double()
    else:
        test_data = test_data.double()

    test_data = test_data.to(device)
    model.eval()
    
    if batch_size is None:
        # Process all at once
        with torch.no_grad():
            posterior = model(test_data)
            mean = posterior.mean.cpu().numpy().squeeze()
            variance = posterior.variance.cpu().numpy().squeeze()
    else:
        # Process in batches
        n_test = len(test_data)
        means = []
        variances = []
        
        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                batch = test_data[i:i+batch_size]
                posterior = model(batch)
                means.append(posterior.mean.cpu().numpy())
                variances.append(posterior.variance.cpu().numpy())
        
        mean = np.concatenate(means, axis=0).squeeze()
        variance = np.concatenate(variances, axis=0).squeeze()

    if return_std:
        return mean, np.sqrt(variance)
    return mean, variance
