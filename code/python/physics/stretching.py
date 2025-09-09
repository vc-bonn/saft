import torch

# energy function
def energy(
    vertex_groups : torch.Tensor,
    rest_lengths  : torch.Tensor,
    stiffnesses   : torch.Tensor,
) -> torch.Tensor:
    distance_residuals = torch.linalg.norm(vertex_groups[..., 1, :] - vertex_groups[..., 0, :], dim=-1) - rest_lengths
    return 0.5 * stiffnesses * distance_residuals**2

# temporary single force function
single_force_temp = torch.func.jacrev(
    func=energy,
    argnums=(0,),
)
# temporary batched force function (returns tuple)
batched_force_temp = torch.vmap(
    func=single_force_temp,
    in_dims=(0, 0, 0),
    out_dims=(0,),
    chunk_size=None,
)

# force function (multiple inputs, tensor as output)
def force(
    positions          : torch.Tensor,
    stretching_indices : torch.Tensor,
    rest_lengths       : torch.Tensor,
    stiffnesses        : torch.Tensor,
) -> torch.Tensor:
    return -batched_force_temp(
        positions[stretching_indices],
        rest_lengths,
        stiffnesses,
    )[0]

# temporary single force derivative
single_dforce_dx_temp = torch.func.jacrev(
    func=single_force_temp,
    argnums=(0,),
)
# temporary batched force derivative (returns tuple)
batched_dforce_dx_temp = torch.vmap(
    func=single_dforce_dx_temp,
    in_dims=(0, 0, 0),
    out_dims=(0,),
    chunk_size=None,
)

# force function (multiple inputs, tensor as output)
def dforce_dx(
    positions          : torch.Tensor,
    stretching_indices : torch.Tensor,
    rest_lengths       : torch.Tensor,
    stiffnesses        : torch.Tensor,
) -> torch.Tensor:
    return -batched_dforce_dx_temp(
        positions[stretching_indices],
        rest_lengths,
        stiffnesses,
    )[0][0]


def computeContributions(
    positions          : torch.Tensor,
    stretching_indices : torch.Tensor,
    rest_lengths       : torch.Tensor,
    stiffnesses        : torch.Tensor,
    time_step          : float,
) -> tuple:
    forces = force(
        positions,
        stretching_indices,
        rest_lengths,
        stiffnesses,
    ).flatten()

    dforces_dx = dforce_dx(
        positions,
        stretching_indices,
        rest_lengths,
        stiffnesses,
    ).flatten()

    arange = torch.arange(positions.shape[1], device=positions.device).unsqueeze(0).unsqueeze(1)
    
    vector_indices = positions.shape[-1] * stretching_indices.unsqueeze(-1) + arange
    matrix_indices_i = torch.repeat_interleave(vector_indices.unsqueeze(3), vector_indices.shape[-1] * vector_indices.shape[-2], dim=3)
    matrix_indices_i = matrix_indices_i.flatten(-3, -2)
    matrix_indices_j = matrix_indices_i.transpose(-1, -2)

    vector_indices = vector_indices.flatten()
    matrix_indices_i = matrix_indices_i.flatten()
    matrix_indices_j = matrix_indices_j.flatten()
    matrix_indices = torch.stack((matrix_indices_i, matrix_indices_j), dim = 0)


    force_vector = torch.zeros_like(positions).flatten()
    force_vector = time_step * torch.index_add(input=force_vector, dim=0, index=vector_indices, source=forces)
    matrix_values = - (time_step*time_step) * dforces_dx
    mass_matrix = torch.sparse_coo_tensor(
                      indices=matrix_indices,
                      values=matrix_values,
                      size=(positions.shape[-1]*positions.shape[-2], positions.shape[-1]*positions.shape[-2]),
                      dtype=force_vector.dtype,
                      device=force_vector.device
                  )

    return mass_matrix, force_vector

def computeForces(
    positions          : torch.Tensor,
    stretching_indices : torch.Tensor,
    rest_lengths       : torch.Tensor,
    stiffnesses        : torch.Tensor,
    time_step          : float,
) -> tuple:
    forces = force(
        positions,
        stretching_indices,
        rest_lengths,
        stiffnesses,
    ).flatten()

    arange = torch.arange(positions.shape[1], device=positions.device).unsqueeze(0)
    
    vector_indices = positions.shape[-1] * stretching_indices.unsqueeze(-1) + arange
    vector_indices = vector_indices.flatten()

    force_vector = torch.zeros_like(positions).flatten()
    force_vector = time_step * torch.index_add(input=force_vector, dim=0, index=vector_indices, source=forces)

    return force_vector.view_as(positions)

###################################################################################################

def contraint(
    vertex_groups : torch.Tensor,
    rest_lengths  : torch.Tensor,
) -> torch.Tensor:
    distance_residuals = torch.linalg.norm(vertex_groups[..., 1, :] - vertex_groups[..., 0, :], dim=-1) - rest_lengths
    return distance_residuals

# temporary single jacobian
single_jacobian_temp = torch.func.jacrev(
    func=contraint,
    argnums=(0,),
)
# temporary batched jacobian (returns tuple)
batched_jacobian_temp = torch.vmap(
    func=single_jacobian_temp,
    in_dims=(0, 0),
    out_dims=(0,),
    chunk_size=None,
)

# jacobian (multiple inputs, tensor as output)
def jacobian(
    positions          : torch.Tensor,
    stretching_indices : torch.Tensor,
    rest_lengths       : torch.Tensor,
) -> torch.Tensor:
    return -batched_jacobian_temp(
        positions[stretching_indices],
        rest_lengths,
    )[0]


def jacobian_matrix(
    positions              : torch.Tensor,
    stretching_indices     : torch.Tensor,
    rest_lengths           : torch.Tensor,
    constraint_block_start : int,
) -> torch.Tensor:
    values = jacobian(
                 positions=positions,
                 stretching_indices=stretching_indices,
                 rest_lengths=rest_lengths,
             ).flatten(-3, -1)

    row_indices = torch.arange(stretching_indices.shape[-2], device=positions.device)
    row_indices = torch.repeat_interleave(row_indices, stretching_indices.shape[-1] * positions.shape[-1], dim=0) + constraint_block_start

    col_indices = torch.arange(positions.shape[-1], device=positions.device)
    col_indices = positions.shape[-1] * stretching_indices.unsqueeze(-1) + col_indices.unsqueeze(-2).unsqueeze(-2)
    col_indices = col_indices.flatten()

    matrix_indices = torch.stack((row_indices, col_indices), dim = 0)
    jacobian_matrix = torch.sparse_coo_tensor(
                          indices=matrix_indices,
                          values=values,
                          device=positions.device,
                      )
    
    return jacobian_matrix