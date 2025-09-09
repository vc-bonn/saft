import torch

def computeContributions(
    vertex_masses       : torch.Tensor,
    velocities          : torch.Tensor,
    previous_velocities : torch.Tensor,
    method              : str,
) -> tuple:
    vertex_masses = vertex_masses.flatten()
    indices = torch.arange(vertex_masses.shape[0], device=vertex_masses.device)
    indices = torch.stack((indices, indices), dim=0)
    mass_matrix = torch.sparse_coo_tensor(indices=indices, values=vertex_masses, size=(vertex_masses.shape[0], vertex_masses.shape[0]))

    if method == "euler" or method == "BDF1":
        momentum_vector = vertex_masses * velocities.flatten()
    elif method == "BDF2":
        momentum_vector = vertex_masses * (4.0/3.0 * velocities.flatten() - 1.0/3.0 * previous_velocities.flatten())

    return mass_matrix, momentum_vector

def computeContributionsIPC(
    vertex_masses       : torch.Tensor,
    positions           : torch.Tensor,
    predicted_positions : torch.Tensor,
) -> tuple:
    vertex_masses = vertex_masses.flatten()
    indices = torch.arange(vertex_masses.shape[0], device=vertex_masses.device)
    indices = torch.stack((indices, indices), dim=0)
    mass_matrix = torch.sparse_coo_tensor(indices=indices, values=vertex_masses, size=(vertex_masses.shape[0], vertex_masses.shape[0]))

    inertia_vector = - vertex_masses * (positions - predicted_positions).flatten()

    return mass_matrix, inertia_vector

###################################################################################################

def inverseStiffnessMatrix(
    pulling_stiffnesses    : torch.Tensor,
    stretching_stiffnesses : torch.Tensor,
    bending_stiffnesses    : torch.Tensor,
    shearing_stiffnesses   : torch.Tensor,
) -> torch.Tensor:
    n_constraints = pulling_stiffnesses.shape[0] + stretching_stiffnesses.shape[0] + bending_stiffnesses.shape[0] + shearing_stiffnesses.shape[0]
    
    diagonal_indices = torch.arange(n_constraints, device=pulling_stiffnesses.device)
    indices = torch.stack((diagonal_indices, diagonal_indices), dim=0)
    values = torch.cat([pulling_stiffnesses, stretching_stiffnesses, bending_stiffnesses, shearing_stiffnesses], dim = 0)
    stiffness_matrix = torch.sparse_coo_tensor(indices=indices, values=1.0/values, size=(n_constraints, n_constraints))

    return stiffness_matrix

def inverseMassMatrix(
    vertex_masses : torch.Tensor,
) -> torch.Tensor:
    vertex_masses = vertex_masses.flatten()
    indices = torch.arange(vertex_masses.shape[0], device=vertex_masses.device)
    indices = torch.stack((indices, indices), dim=0)
    mass_matrix = torch.sparse_coo_tensor(indices=indices, values=1.0/vertex_masses, size=(vertex_masses.shape[0], vertex_masses.shape[0]))

    return mass_matrix