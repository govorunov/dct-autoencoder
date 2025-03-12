from typing import NamedTuple

import torch


class DCTBasis(NamedTuple):
    basis_functions: torch.Tensor
    spatial_frequencies_components: torch.Tensor
    spatial_frequencies_magnitude: torch.Tensor
    multiplication_factor_matrix: torch.Tensor
    multiplication_factor_scalar: float
    block_size: int


def get_dct_basis(block_size: int = 8) -> DCTBasis:
    """Generate the DCT basis variables for a given block size.

    Args:
        block_size (int, optional): The block size. Defaults to 8.

    Returns:
        DCTBasis: The DCT basis variables.
    """
    frequencies = torch.arange(block_size)
    x = torch.arange(block_size)
    y = torch.arange(block_size)
    x, y = torch.meshgrid(x, y, indexing="xy")
    basis_functions = torch.zeros(
        (block_size, block_size, block_size, block_size), dtype=torch.float32
    )
    spatial_frequencies = torch.zeros((block_size, block_size, 2), dtype=torch.int64)
    multiplication_factor_matrix = torch.zeros((block_size, block_size), dtype=torch.float32)
    for v in frequencies:
        for u in frequencies:
            # spatial frequencies
            spatial_frequencies[v, u] = torch.tensor([v, u])
            # basis functions
            x_ref_patch = torch.cos(((2 * x + 1) * u * torch.pi) / (2 * block_size))
            y_ref_patch = torch.cos(((2 * y + 1) * v * torch.pi) / (2 * block_size))
            basis_functions[v, u] = x_ref_patch * y_ref_patch
            # constants
            c_v = 1 / torch.sqrt(torch.tensor(2.0)) if v == 0 else 1
            c_u = 1 / torch.sqrt(torch.tensor(2.0)) if u == 0 else 1
            multiplication_factor_matrix[v, u] = c_u * c_v

    spatial_frequencies_magnitude = torch.linalg.norm(spatial_frequencies.type(torch.float32), dim=2)
    multiplication_factor_scalar = 2 / block_size

    return DCTBasis(
        basis_functions=basis_functions,
        spatial_frequencies_components=spatial_frequencies,
        spatial_frequencies_magnitude=spatial_frequencies_magnitude,
        multiplication_factor_matrix=multiplication_factor_matrix,
        multiplication_factor_scalar=multiplication_factor_scalar,
        block_size=block_size,
    )
