"""
Deep Kernel GP - Deep Kernel Learning for Gaussian Process Regression and Classification
"""

# Feature extractors
from .models import (
    FCFeatureExtractor,
    FCBNFeatureExtractor,
    ResNetFeatureExtractor,
    AttentionFeatureExtractor,
    DirectAttentionExtractor,
    AttentionWeightedExtractor,
    IdentityExtractor,
    get_feature_extractor,
    ImageFeatureExtractor,
)

# Regression
from .gpr import (
    DeepKernelGP,
    ConfidenceWeightedMLL,
    SampleWeightedMLL,
    train_dkgp,
    fit_dkgp,
    predict_dkgpr,
    expected_improvement,
    upper_confidence_bound,
    probability_of_improvement,
    thompson_sampling,
    expected_improvement_with_constraints
)

# Sample weighting utilities
from .sample_weighting import (
    SampleWeightModule,
    analyze_sample_weights,
)

# Classification
from .gpc import (
    DeepKernelGPClassifier,
    BinaryGPClassificationModel,
    MultiClassGPClassificationModel,
    ConfidenceWeightedELBO,
    train_dkgp_classifier,
    fit_dkgp_classifier,
    predict_classifier,
)

# Pairwise GP
from .gppw import (
    DeepKernelPairwiseGP,
    fit_dkgppw,
    predict_utility,
    train_dkgppw,
    dkgppw_eubo,
    dkgppw_ucb,
    sample_comparison_pairs,
    get_user_preference,
    get_simulated_preference,
    acquire_preference,
    plot_option,
    plot_predictions,
)

# utilities
from .utils import (
    get_grid_coords,
    get_subimages,
    split_train_test,
    get_attention_scores,
    get_attention_for_sample,
    analyze_attention_locality,
    summarize_attention,
    save_model,
    load_model,
)

# Kernels
from .kernels import (
    AttentionWeightedRBFKernel,
    InputMixingRBFKernel,
)

# Submodules - for convenience imports
from . import gpr as dkgpr 
from . import gpc as dkgpc
from . import gppw as dkgppw 
from . import utils
from . import models
from . import kernels

__version__ = "0.2.0"

__all__ = [
    # Feature extractors
    "FCFeatureExtractor",
    "FCBNFeatureExtractor",
    "ResNetFeatureExtractor",
    "AttentionFeatureExtractor",
    "DirectAttentionExtractor",
    "AttentionWeightedExtractor",
    "IdentityExtractor",
    "get_feature_extractor",
    "ImageFeatureExtractor",
    # Regression
    "DeepKernelGP",
    "ConfidenceWeightedMLL",
    "SampleWeightedMLL",
    "train_dkgp",
    "fit_dkgp",
    "predict_dkgpr",
    "expected_improvement",
    "upper_confidence_bound",
    "probability_of_improvement",
    "thompson_sampling",
    "expected_improvement_with_constraints",
    # Sample weighting
    "SampleWeightModule",
    "analyze_sample_weights",
    # Classification
    "DeepKernelGPClassifier",
    "BinaryGPClassificationModel",
    "MultiClassGPClassificationModel",
    "ConfidenceWeightedELBO",
    "train_dkgp_classifier",
    "fit_dkgp_classifier",
    "predict_classifier",
    # Pairwise GP
    "DeepKernelPairwiseGP",
    "fit_dkgppw",
    "predict_utility",
    "train_dkgppw",
    "dkgppw_eubo",
    "dkgppw_ucb",
    "sample_comparison_pairs",
    "get_user_preference",
    "get_simulated_preference",
    "acquire_preference",
    "plot_option",
    "plot_predictions",
    # Utilities
    "get_grid_coords",
    "get_subimages",
    "split_train_test",
    "get_attention_scores",
    "get_attention_for_sample",
    "analyze_attention_locality",
    "summarize_attention",
    "save_model",
    "load_model",

    # Kernels
    "AttentionWeightedRBFKernel",
    "InputMixingRBFKernel",

    # Submodules
    "dkgpr",
    "dkgpc",
    "dkgppw",
    "utils",
    "models",
    "kernels",
]
