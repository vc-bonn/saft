import torch

# energy function
def energy(
    positions     : torch.Tensor,
    forces        : torch.Tensor,
):
    return - torch.sum(forces * positions, dim=-1)

# force function
def force(
    forces : torch.Tensor,
):
    return forces

def computeContributions(
    forces    : torch.Tensor,
    time_step : float,
) -> torch.Tensor:
    return time_step * force(forces).flatten()