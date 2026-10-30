"""Nonnegative Weighted Integral (NWI) models for droplet size distribution emulation.

This module implements advectable autoencoders for learning latent representations
of droplet size distributions.

.. rubric:: References

Huang, K.-E., Wang, M., Rosenfeld, D., & Zhu, Y (2025). Exploring Advectable Latent Representations
for Droplet Size Distributions With Physics-Informed Autoencoders. *Journal of Advances in Modeling Earth
Systems*, 17, e2024MS004821. https://doi.org/10.1029/2024MS004821
"""
import torch
from torch import nn
from torch.nn import Softmax
import torch.nn.functional as F

from src import data_utils as du
from src.constants import LOG_TOLERANCE

class ResBlock(nn.Module):
    """Residual block with layer normalization and LeakyReLU activation.

    :param num_features: Number of features in the hidden layers, defaults to 256
    :type num_features: int, optional
    """
    def __init__(self, num_features=256):
        super(ResBlock, self).__init__()

        self.stack = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features),
            nn.LayerNorm(num_features),
            nn.SiLU(),
            nn.Linear(num_features, num_features),
        )
    def forward(self, x):
        """Forward pass with residual connection.

        :param x: Input tensor
        :type x: torch.Tensor
        :return: Output tensor with residual connection applied
        :rtype: torch.Tensor
        """
        return self.stack(x) + x

    def init_weights(self, m):
        """Initialize weights using Xavier uniform initialization.

        :param m: Module to initialize
        :type m: nn.Module
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

class SimpleBlock(nn.Module):
    """Simple feed-forward block with ReLU activation.

    :param num_features: Number of features in the hidden layers, defaults to 256
    :type num_features: int, optional
    """
    def __init__(self, num_features=256):
        super(SimpleBlock, self).__init__()

        self.stack = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
        )

        self.apply(self.init_weights)

    def forward(self, x):
        """Forward pass through the simple block.

        :param x: Input tensor
        :type x: torch.Tensor
        :return: Output tensor
        :rtype: torch.Tensor
        """
        return self.stack(x)

    def init_weights(self, m):
        """Initialize weights using Xavier uniform initialization.

        :param m: Module to initialize
        :type m: nn.Module
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

# From Huang 2024 https://zenodo.org/records/12866868
def DNN(in_features, out_features, hidden_features=256, num_resblocks=5):
    """Deep neural network with residual blocks.

    :param in_features: Number of input features
    :type in_features: int
    :param out_features: Number of output features
    :type out_features: int
    :param hidden_features: Number of hidden features, defaults to 256
    :type hidden_features: int, optional
    :param num_resblocks: Number of residual blocks, defaults to 5
    :type num_resblocks: int, optional
    :return: Sequential model with residual blocks
    :rtype: nn.Sequential
    """
    return nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Linear(in_features, hidden_features),
        *[ResBlock(hidden_features) for _ in range(num_resblocks)],
        nn.Linear(hidden_features, out_features)
    )

def SNN(in_features, out_features, hidden_features=256, num_blocks=5):
    """Simple neural network with feed-forward blocks.

    :param in_features: Number of input features
    :type in_features: int
    :param out_features: Number of output features
    :type out_features: int
    :param hidden_features: Number of hidden features, defaults to 256
    :type hidden_features: int, optional
    :param num_blocks: Number of simple blocks, defaults to 5
    :type num_blocks: int, optional
    :return: Sequential model with simple blocks
    :rtype: nn.Sequential
    """
    return nn.Sequential(
        nn.Linear(in_features, hidden_features),
        *[SimpleBlock(hidden_features) for _ in range(num_blocks)],
        nn.Linear(hidden_features, out_features)
    )

class MonotoneLinear(nn.Module):
    """Linear layer with monotone constraints using softplus on weights.

    :param in_features: Number of input features
    :type in_features: int
    :param out_features: Number of output features
    :type out_features: int
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        """Forward pass with softplus-constrained weights.

        :param x: Input tensor
        :type x: torch.Tensor
        :return: Output tensor with monotone transformation applied
        :rtype: torch.Tensor
        """
        return F.linear(x, F.softplus(self.weight), self.bias)

class NNWF(nn.Module):
    """Neural network weight function for learning latent representations.

    :param in_features: Number of bins for the log bin mass, defaults to 64
    :type in_features: int, optional
    :param nodes: Number of nodes in the hidden layer, defaults to 16
    :type nodes: int, optional
    """
    def __init__(self, in_features=64, nodes=16):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Linear(1, nodes),
            ResBlock(num_features=nodes),
            nn.Linear(nodes, 1),
        )
        self.register_buffer(
            "log_bin_mass",
            torch.linspace(-1, 1, in_features)[:, None]
        )
        self.in_features = 1
        self.out_features = 1

    def forward(self):
        """Forward pass through the weight function network.

        :return: Weight function values for each bin
        :rtype: torch.Tensor
        """
        return self.stack(self.log_bin_mass) # note: exp moved to LinearEncoder softmax


class LinearEncoder(nn.Module):
    """Linear encoder for transforming distributions to latent space.

    Maps input distributions x of shape [B, T, N] to latent representations
    of shape [B, T, L] using learned weight functions.

    :param n_bins: Number of bins in the input distribution, defaults to 64
    :type n_bins: int, optional
    :param n_latent: Number of latent dimensions, defaults to 3
    :type n_latent: int, optional
    :param type: Type of encoder, either 'fnM' (neural network weight functions)
        or 'simple' (learnable parameter matrix), defaults to 'fnM'
    :type type: str, optional
    """
    # in this version, ln(W) = NNWF(in_features=n_bins)
    def __init__(self, n_bins=64, n_latent=3, type="fnM"):
        super().__init__()
        self.n_bins = n_bins
        if type == "fnM":
            self.wfs = nn.ModuleList([NNWF(in_features=n_bins) for i in range(n_latent)])
        elif type == "simple":
            self.wfs = nn.Parameter(torch.rand(n_bins, n_latent))
        else:
            raise ValueError(f"Unknown NWI encoder type: {type}")
        self.out_features = sum([wf.out_features for wf in self.wfs])

    def wf_mat(self):
        """Compute the weight function matrix with softmax normalization.

        :return: Weight function matrix of shape [n_bins, n_latent + 1] with
            additional column of ones for mass conservation
        :rtype: torch.Tensor
        """
        lnW = torch.cat([wf() for wf in self.wfs], dim=1) # learn WFs
        Wf = torch.softmax(lnW, dim=0) * self.n_bins
        Wf = torch.cat([Wf, torch.ones(self.n_bins, 1)], dim=-1)  # Add row of ones for mass
        return Wf

    def forward(self, x):
        """Encode input distributions to latent space.

        :param x: Input tensor of shape [B, T, N] representing distributions
        :type x: torch.Tensor
        :return: Latent representation of shape [B, T, L+1]
        :rtype: torch.Tensor
        """
        return x @ self.wf_mat()

# Note: Operates on dimensioned latent variables to produce a dimensioned DSD
class SimpleDecoder(nn.Module):
    """Simple decoder for reconstructing distributions from latent space.

    :param n_bins: Number of bins in the output distribution, defaults to 64
    :type n_bins: int, optional
    :param n_latent: Number of latent dimensions, defaults to 3
    :type n_latent: int, optional
    :param hidden_features: Number of hidden features, defaults to 256
    :type hidden_features: int, optional
    :param num_blocks: Number of simple blocks, defaults to 5
    :type num_blocks: int, optional
    """
    def __init__(self, n_bins=64, n_latent=3, hidden_features=256, num_blocks=5):
        super(SimpleDecoder, self).__init__()
        self.n_bins = n_bins
        self.n_latent = n_latent

        self.network = SNN(n_latent, n_bins, hidden_features=hidden_features, num_blocks=num_blocks)
        self.sm = Softmax(dim=-1)

    def forward(self, h):
        """Decode latent representation to distribution.

        :param h: Latent tensor of shape [B, T, L+1] including mass
        :type h: torch.Tensor
        :return: Reconstructed distribution of shape [B, T, N]
        :rtype: torch.Tensor
        """
        hhat, mass = h_to_hhat_M(h)  # Convert to dimensionless latent variables and mass
        xhat = self.network(hhat)
        return self.sm(xhat) * mass.unsqueeze(-1) # convert back to a normalized PSD, scale by mass

class DeepDecoder(nn.Module):
    """Deep decoder with residual blocks for distribution reconstruction.

    :param n_bins: Number of bins in the output distribution, defaults to 64
    :type n_bins: int, optional
    :param n_latent: Number of latent dimensions, defaults to 3
    :type n_latent: int, optional
    :param hidden_features: Number of hidden features, defaults to 256
    :type hidden_features: int, optional
    :param num_blocks: Number of residual blocks, defaults to 5
    :type num_blocks: int, optional
    :param eps: Small value to prevent log(0), defaults to LOG_TOLERANCE
    :type eps: float, optional
    """
    def __init__(self, n_bins=64, n_latent=3, hidden_features=256, num_blocks=5, eps=LOG_TOLERANCE):
        super().__init__()
        self.stack = DNN(n_latent, n_bins, hidden_features=hidden_features, num_resblocks=num_blocks)
        self.eps = eps
        self.sm = Softmax(dim=-1)
        self.n_bins = n_bins
        self.n_latent = n_latent

    def forward(self, h):
        """Decode latent representation to distribution using log transform.

        :param h: Latent tensor of shape [B, T, L+1] including mass
        :type h: torch.Tensor
        :return: Reconstructed distribution of shape [B, T, N]
        :rtype: torch.Tensor
        """
        B = h.shape[0]
        hhat, mass = h_to_hhat_M(h)
        logh = torch.log(hhat + self.eps)
        logh = logh.reshape(-1, self.n_latent)
        xhat = self.stack(logh)
        xhat = xhat.reshape(B, -1, self.n_bins)
        return self.sm(xhat) * mass # convert back to normalized PSD


class NNWIAutoencoder(nn.Module):
    """Neural network-based autoencoder combining encoder and decoder.

    :param encoder: Encoder module (e.g., LinearEncoder)
    :type encoder: nn.Module
    :param decoder: Decoder module (e.g., SimpleDecoder or DeepDecoder)
    :type decoder: nn.Module
    """
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        """Forward pass through encoder and decoder.

        :param x: Input tensor of shape [B, T, N]
        :type x: torch.Tensor
        :return: Reconstructed tensor of shape [B, T, N]
        :rtype: torch.Tensor
        """
        return self.decoder(self.encoder(x))
    
def hhat_M_to_h(h_hat, M):
    """Convert dimensionless latent variables to dimensioned latent variables.

    :param h_hat: Dimensionless latent variables of shape [..., L]
    :type h_hat: torch.Tensor
    :param M: Mass values of shape [...]
    :type M: torch.Tensor
    :return: Dimensioned latent variables of shape [..., L+1] with mass appended
    :rtype: torch.Tensor
    """
    h = torch.concat([h_hat*M.unsqueeze(-1), M.unsqueeze(-1)], dim=-1)  # concatenate h_hat*M and M along the feature dimension
    return h

def h_to_hhat_M(h, eps=LOG_TOLERANCE):
    """Convert dimensioned latent variables to dimensionless latent variables.

    Works with any number of leading dimensions: [..., latent+1].

    :param h: Dimensioned latent variables of shape [..., L+1] with mass as last feature
    :type h: torch.Tensor
    :param eps: Small value to prevent division by zero, defaults to LOG_TOLERANCE
    :type eps: float, optional
    :return: Tuple of (h_hat, M) where h_hat is dimensionless latent variables of shape [..., L]
        and M is mass values of shape [...]
    :rtype: tuple of torch.Tensor
    """
    M = h[..., -1]  # Extract M (last feature)
    h_hat = h[..., :-1] / (M.unsqueeze(-1) + eps)  # Normalize by M (add eps to prevent division by zero)
    return h_hat, M