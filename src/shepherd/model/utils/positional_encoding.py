import math
import torch

def positional_encoding(position, dim, device):
    """
    Positional encoding for uniform timesteps.

    Args:
        position: (B,) where B is the number of molecules in the batch
        dim: int, the dimension of the position embeddings
        device: device to store the position embeddings on
    Returns:
        position embeddings of shape (B, dim)
    """
    if len(position.shape) == 1:
        position = position[None,:].T
    assert len(position.shape) == 2, f"Position must have shape (B, 1), got {position.shape}"
    assert position.shape[1] == 1, f"Position must have shape (B, 1), got {position.shape}"
    assert dim % 2 == 0, f"Dimension must be even, got {dim}"

    pe = torch.zeros(position.shape[0], dim, device = device)
    div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float, device = device) * -(math.log(10000.0) / dim)))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def fourier_embedding(input: torch.Tensor):
    """
    Fourier embedding for sigma in EDM.

    Assumes ``input`` is:
    sigma * w + b where w, b ~ N(0, I_dim)

    Args:
        input: (B, dim) where B is the number of molecules in the batch and dim is the embedding dimension
    Returns:
        fourier embedding of shape (B, dim)
    """
    assert len(input.shape) == 2, f"Input must have shape (B, dim), got {input.shape}"

    fe = torch.cos(2*torch.pi * input)
    return fe
