"""
Core model classes and feature extractors for Deep Kernel GP.
"""
import torch
import torch.nn as nn
import numpy as np

# ============================================================================
# Feature Extractors
# ============================================================================
class FCFeatureExtractor(nn.Module):
    """
    Simple fully-connected feature extractor.
    Lightweight, fast, good for prototyping.
    
    Parameters
    ----------
    input_dim : int
        Dimensionality of input data
    feature_dim : int
        Dimensionality of learned feature space
    """
    
    def __init__(self, input_dim, feature_dim=16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, feature_dim)
        )
        self.input_dim = input_dim
        self.feature_dim = feature_dim
    
    def forward(self, x):
        return self.network(x)


class FCBNFeatureExtractor(nn.Module):
    """
    Fully-connected feature extractor with BatchNorm and Dropout.
    More robust, prevents overfitting. Recommended for general use.
    
    Parameters
    ----------
    input_dim : int
        Dimensionality of input data
    feature_dim : int
        Dimensionality of learned feature space
    hidden_dims : list of int
        Hidden layer dimensions
    dropout : float
        Dropout rate
    """
    
    def __init__(self, input_dim, feature_dim=16, hidden_dims=None, dropout=0.2):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, feature_dim))
        self.network = nn.Sequential(*layers)
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.network(x)


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-style feature extractor with skip connections.
    Better gradient flow for deeper networks.
    
    Parameters
    ----------
    input_dim : int
        Input dimensionality
    feature_dim : int
        Output feature dimensionality
    hidden_dim : int
        Hidden layer dimension
    num_blocks : int
        Number of residual blocks
    dropout : float
        Dropout rate
    """
    
    class ResBlock(nn.Module):
        def __init__(self, dim, dropout=0.1):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)
            self.bn1 = nn.BatchNorm1d(dim)
            self.bn2 = nn.BatchNorm1d(dim)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            residual = x
            out = self.relu(self.bn1(self.fc1(x)))
            out = self.dropout(out)
            out = self.bn2(self.fc2(out))
            out += residual  # Skip connection
            return self.relu(out)
    
    def __init__(self, input_dim, feature_dim=16, hidden_dim=128, num_blocks=2, dropout=0.1):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        self.res_blocks = nn.Sequential(
            *[self.ResBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        
        self.output_projection = nn.Linear(hidden_dim, feature_dim)
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.res_blocks(x)
        x = self.output_projection(x)
        return x

class AttentionFeatureExtractor(nn.Module):
    """
    Self-attention based feature extractor.
    Good for learning feature interactions.
    
    Parameters
    ----------
    input_dim : int
        Input dimensionality
    feature_dim : int
        Output feature dimensionality
    hidden_dim : int
        Hidden dimension
    num_heads : int
        Number of attention heads
    """
    
    class AttentionBlock(nn.Module):
        def __init__(self, dim, num_heads=4):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            
            assert dim % num_heads == 0, "dim must be divisible by num_heads"
            
            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)
            self.out = nn.Linear(dim, dim)
            self.last_attention_weights = None # Store last attention weights
            
        def forward(self, x, return_attention=False):
            batch_size = x.shape[0]
            
            # Linear projections
            Q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
            K = self.key(x).view(batch_size, self.num_heads, self.head_dim)
            V = self.value(x).view(batch_size, self.num_heads, self.head_dim)
            
            # Attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
            attention = torch.softmax(scores, dim=-1)

            # Store for later retrieval
            self.last_attention_weights = attention.detach()

            # Apply attention
            out = torch.matmul(attention, V)
            out = out.view(batch_size, -1)
            out = self.out(out)
            # Optional return attention
            if return_attention:
                return out, attention
            return out
        
        def get_attention_weights(self):
            """Get the last computed attention weights."""
            return self.last_attention_weights
    
    def __init__(self, input_dim, feature_dim=16, hidden_dim=128, num_heads=4):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.attention = self.AttentionBlock(hidden_dim, num_heads)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, feature_dim)
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
    
    def forward(self, x):
        x = self.input_projection(x)
        
        # Attention block with residual
        attn_out = self.attention(x)
        x = self.layer_norm(x + attn_out)
        
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.layer_norm(x + ffn_out)
        
        x = self.output_projection(x)
        return x

    def get_attention_maps(self, x):
        """
        Get attention maps for input x.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (batch, input_dim)
            
        Returns
        -------
        attention_weights : torch.Tensor
            Attention weights, shape (batch, num_heads, num_heads)
        """
        self.eval()
        with torch.no_grad():
            x_proj = self.input_projection(x)
            _, attention = self.attention(x_proj, return_attention=True)
        return attention

class DirectAttentionExtractor(nn.Module):
    """
    Attention extractor that operates directly on input features (wavelengths)
    BEFORE compression.
    
    Perfect for spectroscopy where you want to see which wavelengths
    attend to each other directly!
    
    Key difference from AttentionFeatureExtractor:
    - AttentionFeatureExtractor: compress first (256→128), then attend on 128
    - DirectAttentionExtractor: attend on full 256, then compress
    
    This allows you to see direct wavelength-to-wavelength relationships!
    
    Parameters
    ----------
    input_dim : int
        Number of input features (e.g., 256 wavelengths)
    feature_dim : int
        Output feature dimension (e.g., 16)
    num_heads : int
        Number of attention heads (default: 4)
    intermediate_dim : int, optional
        Intermediate dimension after attention (default: same as input_dim)
    dropout : float
        Dropout rate (default: 0.1)
    
    Examples
    --------
    >>> # For spectroscopy with 256 wavelengths
    >>> extractor = DirectAttentionExtractor(
    ...     input_dim=256,
    ...     feature_dim=16,
    ...     num_heads=4
    ... )
    >>> 
    >>> # Get attention map showing wavelength relationships
    >>> attention_map = extractor.get_attention_map(x)
    >>> # Shape: (num_heads, 256, 256) - direct wavelength-to-wavelength!
    """
    
    def __init__(self, input_dim, feature_dim=16, num_heads=4, 
                 intermediate_dim=None, dropout=0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # If not specified, use input_dim
        if intermediate_dim is None:
            intermediate_dim = input_dim
        
        self.intermediate_dim = intermediate_dim
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Multi-head attention on FULL input dimension
        self.attention = DirectMultiHeadAttention(
            dim=input_dim,  # ← Works on full wavelength dimension!
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Post-attention norm
        self.norm = nn.LayerNorm(input_dim)
        
        # Optional intermediate projection
        if intermediate_dim != input_dim:
            self.intermediate_proj = nn.Sequential(
                nn.Linear(input_dim, intermediate_dim),
                nn.ReLU(),
                nn.LayerNorm(intermediate_dim),
                nn.Dropout(dropout)
            )
        else:
            self.intermediate_proj = None
        
        # Final compression to feature_dim
        final_input_dim = intermediate_dim if self.intermediate_proj else input_dim
        self.output_projection = nn.Sequential(
            nn.Linear(final_input_dim, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim)
        )
    
    def forward(self, x):
        """
        Forward pass with attention on raw input features.
        
        Parameters
        ----------
        x : torch.Tensor
            Input, shape (batch, input_dim)
            
        Returns
        -------
        features : torch.Tensor
            Extracted features, shape (batch, feature_dim)
        """
        # Normalize input
        x_norm = self.input_norm(x)
        
        # Apply attention directly on input features
        attn_out = self.attention(x_norm)
        
        # Residual connection
        x = self.norm(x + attn_out)
        
        # Optional intermediate projection
        if self.intermediate_proj is not None:
            x = self.intermediate_proj(x)
        
        # Final compression to feature_dim
        features = self.output_projection(x)
        
        return features
    
    def get_attention_map(self, x):
        """
        Get attention map showing direct feature-to-feature relationships.
        
        For spectroscopy, this shows which wavelengths attend to which!
        
        Parameters
        ----------
        x : torch.Tensor
            Input, shape (batch, input_dim) or (input_dim,)
            
        Returns
        -------
        attention_map : np.ndarray
            Attention weights showing feature relationships
            - If batch input: shape (batch, num_heads, input_dim, input_dim)
            - If single input: shape (num_heads, input_dim, input_dim)
            
        Examples
        --------
        >>> # Get attention for one spectrum
        >>> x = torch.randn(256)  # 256 wavelengths
        >>> attention_map = extractor.get_attention_map(x)
        >>> print(attention_map.shape)  # (4, 256, 256)
        >>> 
        >>> # See what wavelength 120 (e.g., 520nm) attends to
        >>> wavelength_120_attention = attention_map[0, 120, :]  # Head 0
        >>> top_5 = np.argsort(wavelength_120_attention)[-5:]
        >>> print(f"Wavelength 120 attends most to: {top_5}")
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)
        
        self.eval()
        with torch.no_grad():
            # Normalize
            x_norm = self.input_norm(x)
            
            # Get attention weights
            attn_weights = self.attention.get_attention_weights(x_norm)
        
        # Convert to numpy
        attn_np = attn_weights.cpu().numpy()
        
        # Remove batch dimension if single input
        if attn_np.shape[0] == 1:
            return attn_np[0]  # Shape: (num_heads, input_dim, input_dim)
        
        return attn_np  # Shape: (batch, num_heads, input_dim, input_dim)


class DirectMultiHeadAttention(nn.Module):
    """
    Multi-head attention for DirectAttentionExtractor.
    
    Computes attention on full input dimension (e.g., all 256 wavelengths).
    """
    
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q, K, V projections
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights
        self.last_attention_weights = None
    
    def forward(self, x):
        """
        Forward pass through multi-head attention.
        
        Parameters
        ----------
        x : torch.Tensor
            Input, shape (batch, dim)
            
        Returns
        -------
        out : torch.Tensor
            Output, shape (batch, dim)
        """
        batch_size = x.shape[0]
        
        # Generate Q, K, V
        Q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, self.num_heads, self.head_dim)
        
        # Compute attention scores: (batch, num_heads, num_heads)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Store for later retrieval (detach to save memory)
        self.last_attention_weights = attn.detach()
        
        # Apply attention to values
        out = torch.matmul(attn, V)  # (batch, num_heads, head_dim)
        out = out.view(batch_size, self.dim)
        out = self.proj(out)
        
        return out
    
    def get_attention_weights(self, x):
        """
        Get attention weights for input x.
        
        Parameters
        ----------
        x : torch.Tensor
            Input, shape (batch, dim)
            
        Returns
        -------
        attention : torch.Tensor
            Attention weights, shape (batch, num_heads, num_heads)
        """
        batch_size = x.shape[0]
        
        # Generate Q, K
        Q = self.query(x).view(batch_size, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        
        return attn

class AttentionWeightedExtractor(nn.Module):
    """
    Feature extractor with attention weights showing feature importance.
    
    Computes importance weight for each input feature (wavelength),
    then processes weighted features through a base extractor.
    
    Perfect for understanding which wavelengths matter for predictions!
    
    The attention mechanism learns which features are important and
    literally multiplies the input by these importance weights before
    further processing.
    
    Parameters
    ----------
    input_dim : int
        Number of input features (e.g., 256 wavelengths)
    feature_dim : int
        Output feature dimension (default: 16)
    base_extractor : str
        Base feature extractor to use after weighting
        Options: 'fc', 'fcbn', 'resnet'
        Default: 'fcbn'
    hidden_dims : list of int, optional
        Hidden dimensions for base extractor (for 'fcbn')
    dropout : float
        Dropout rate (default: 0.2)
    
    Examples
    --------
    >>> # For spectroscopy with 256 wavelengths
    >>> extractor = AttentionWeightedExtractor(
    ...     input_dim=256,
    ...     feature_dim=16,
    ...     base_extractor='fcbn'
    ... )
    >>> 
    >>> # Forward pass
    >>> features = extractor(x)
    >>> 
    >>> # Get attention weights showing feature importance
    >>> weights = extractor.get_attention_weights(x)
    >>> # Shape: (256,) - importance of each wavelength!
    >>> 
    >>> # Find most important features
    >>> top_10 = np.argsort(weights)[-10:][::-1]
    >>> print(f"Most important wavelengths: {top_10}")
    """
    
    def __init__(self, input_dim, feature_dim=16, base_extractor='fcbn',
                 hidden_dims=None, dropout=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.base_extractor_type = base_extractor
        
        # Attention mechanism to compute feature importance
        # Uses a small network to learn importance scores
        attention_hidden = max(input_dim // 4, 32)  # Adaptive hidden size
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_hidden),
            nn.Tanh(),
            nn.Dropout(dropout * 0.5),  # Less dropout in attention
            nn.Linear(attention_hidden, input_dim),
            nn.Softmax(dim=1)  # Normalize to get importance weights (sum to 1)
        )
        
        # Base feature extractor processes the weighted input
        if base_extractor == 'fc':
            self.base = FCFeatureExtractor(input_dim, feature_dim)
        elif base_extractor == 'fcbn':
            self.base = FCBNFeatureExtractor(input_dim, feature_dim, hidden_dims, dropout)
        elif base_extractor == 'resnet':
            hidden_dim = hidden_dims[0] if hidden_dims else 128
            self.base = ResNetFeatureExtractor(input_dim, feature_dim, hidden_dim, 2, dropout)
        else:
            raise ValueError(f"base_extractor must be 'fc', 'fcbn', or 'resnet', got '{base_extractor}'")
        
        # Store last attention weights for analysis
        self.last_attention_weights = None
    
    def forward(self, x):
        """
        Forward pass with attention weighting.
        
        Parameters
        ----------
        x : torch.Tensor
            Input, shape (batch, input_dim)
            
        Returns
        -------
        features : torch.Tensor
            Extracted features, shape (batch, feature_dim)
        """
        # Compute attention weights (importance of each feature)
        attention_weights = self.attention(x)  # (batch, input_dim)
        
        # Store for later analysis
        self.last_attention_weights = attention_weights.detach()
        
        # Apply attention weights to input
        # High-weight features get amplified, low-weight get suppressed
        weighted_input = x * attention_weights
        
        # Process through base extractor
        features = self.base(weighted_input)
        
        return features
    
    def get_attention_weights(self, x):
        """
        Get attention weights showing feature importance.
        
        Returns importance weight for each input feature.
        Higher values mean the feature is more important for predictions.
        
        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            Input, shape (batch, input_dim) or (input_dim,)
            
        Returns
        -------
        weights : np.ndarray
            Attention weights, shape (batch, input_dim) or (input_dim,)
            Values sum to 1.0 across features
            Higher values = more important features
            
        Examples
        --------
        >>> # Get importance for one spectrum
        >>> x = X_spectra[0]  # (256,)
        >>> weights = extractor.get_attention_weights(x)
        >>> print(weights.shape)  # (256,)
        >>> print(weights.sum())  # 1.0 (normalized)
        >>> 
        >>> # Find most important wavelengths
        >>> top_5 = np.argsort(weights)[-5:][::-1]
        >>> print(f"Top 5 wavelengths: {top_5}")
        >>> print(f"Their importance: {weights[top_5]}")
        >>> 
        >>> # Get importance for multiple samples
        >>> weights_batch = extractor.get_attention_weights(X_spectra[:10])
        >>> print(weights_batch.shape)  # (10, 256)
        >>> 
        >>> # Average importance across samples
        >>> avg_importance = weights_batch.mean(axis=0)
        >>> print(f"Average importance: {avg_importance.shape}")  # (256,)
        """
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).double()
        
        if x.ndim == 1:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Move to same device as model
        x = x.to(next(self.parameters()).device)
        
        self.eval()
        with torch.no_grad():
            weights = self.attention(x)
        
        weights_np = weights.cpu().numpy()
        
        if squeeze_output:
            return weights_np[0]  # (input_dim,)
        return weights_np  # (batch, input_dim)

# ============================================================================
# Feature Extractor Factory
# ============================================================================

class IdentityExtractor(nn.Module):
    """
    No-op feature extractor for standard (non-deep-kernel) GP.

    Passes the raw input directly to the GP without any transformation.
    Use by setting ``extractor_type=None`` in :func:`fit_dkgp`.
    The GP kernel then operates on the original input space.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input data (also the GP feature dimension).
    """

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = input_dim  # GP operates in input space

    def forward(self, x):
        return x


def get_feature_extractor(
    extractor_type='fcbn',
    input_dim=None,
    feature_dim=16,
    hidden_dims=None,
    dropout=0.2,
    **kwargs
):
    """
    Factory function to create feature extractors.
    
    Parameters
    ----------
    extractor_type : str
        Type of feature extractor:
        - 'fc': Simple fully-connected
        - 'fcbn': FC + BatchNorm + Dropout
        - 'resnet': ResNet with skip connections
        - 'attention': Self-attention based
        - 'direct_attention': Attention on full input BEFORE compression
        - 'attention_weighted': Attention weights showing feature importance 
        - 'custom': User-provided nn.Module
    input_dim : int
        Input dimensionality
    feature_dim : int
        Output feature dimensionality
    hidden_dims : list of int, optional
        Hidden layer dimensions (for 'fcbn' and 'wide_deep')
    dropout : float
        Dropout rate
    **kwargs : additional arguments
        - custom_extractor: nn.Module for 'custom' type
        - hidden_dim: for 'resnet' and 'attention'
        - num_blocks: for 'resnet'
        - num_heads: for 'attention'
        - deep_dims: for 'wide_deep'
    
    Returns
    -------
    feature_extractor : nn.Module
        Instantiated feature extractor
        
    Examples
    --------
    >>> # Simple FC
    >>> extractor = get_feature_extractor('fc', input_dim=100, feature_dim=16)
    
    >>> # FC + BatchNorm (recommended)
    >>> extractor = get_feature_extractor('fcbn', input_dim=100, feature_dim=16,
    ...                                   hidden_dims=[512, 256, 128])
    
    >>> # ResNet
    >>> extractor = get_feature_extractor('resnet', input_dim=100, feature_dim=16,
    ...                                   hidden_dim=128, num_blocks=3)
    
    >>> # Custom
    >>> my_net = nn.Sequential(nn.Linear(100, 64), nn.ReLU(), nn.Linear(64, 16))
    >>> extractor = get_feature_extractor('custom', custom_extractor=my_net)
    """
    if extractor_type == 'fc':
        return FCFeatureExtractor(input_dim, feature_dim)
    
    elif extractor_type == 'fcbn':
        return FCBNFeatureExtractor(input_dim, feature_dim, hidden_dims, dropout)
    
    elif extractor_type == 'resnet':
        hidden_dim = kwargs.get('hidden_dim', 128)
        num_blocks = kwargs.get('num_blocks', 2)
        return ResNetFeatureExtractor(input_dim, feature_dim, hidden_dim, num_blocks, dropout)
    
    elif extractor_type == 'attention':
        hidden_dim = kwargs.get('hidden_dim', 128)
        num_heads = kwargs.get('num_heads', 4)
        return AttentionFeatureExtractor(input_dim, feature_dim, hidden_dim, num_heads)
    
    elif extractor_type == 'direct_attention': 
        num_heads = kwargs.get('num_heads', 4)
        intermediate_dim = kwargs.get('intermediate_dim', None)
        return DirectAttentionExtractor(
            input_dim, feature_dim, num_heads, intermediate_dim, dropout) 

    elif extractor_type == 'direct_attention_gp':
        # Pure-GP variant: keep the GP operating in the original input space
        # by matching feature_dim to input_dim (no compression).
        num_heads = kwargs.get('num_heads', 4)
        intermediate_dim = kwargs.get('intermediate_dim', input_dim)
        return DirectAttentionExtractor(
            input_dim=input_dim,
            feature_dim=input_dim,
            num_heads=num_heads,
            intermediate_dim=intermediate_dim,
            dropout=dropout,
        )
    
    elif extractor_type == 'attention_weighted':
        base_extractor = kwargs.get('base_extractor', 'fcbn')
        return AttentionWeightedExtractor(
            input_dim, feature_dim, base_extractor, hidden_dims, dropout
        ) 
    
    elif extractor_type == 'attention_weighted_gp':
        # Pure-GP variant: keep the GP operating in the original input space
        # by matching feature_dim to input_dim (no compression).
        base_extractor = kwargs.get('base_extractor', 'fc')
        return AttentionWeightedExtractor(
            input_dim=input_dim,
            feature_dim=input_dim,
            base_extractor=base_extractor,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    elif extractor_type == 'custom':
        custom_extractor = kwargs.get('custom_extractor')
        if custom_extractor is None:
            raise ValueError("Must provide 'custom_extractor' for type='custom'")
        return custom_extractor

    elif extractor_type is None:
        return IdentityExtractor(input_dim)

    else:
        raise ValueError(f"Unknown extractor_type: {extractor_type}. "
                        f"Choose from: 'fc', 'fcbn', 'resnet', 'attention', "
                        f"'direct_attention', 'attention_weighted', "
                        f"'direct_attention_gp', 'attention_weighted_gp', "
                        f"'custom', or None")


# ============================================================================
# Backward Compatibility
# ============================================================================

ImageFeatureExtractor = FCBNFeatureExtractor
