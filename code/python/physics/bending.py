import torch

# energy function
def energy(
    vertex_groups : torch.Tensor,
    rest_angles   : torch.Tensor,
    stiffnesses   : torch.Tensor,
) -> torch.Tensor:
    edges_1 = vertex_groups[..., 1, :] - vertex_groups[..., 0, :]
    edges_2 = vertex_groups[..., 2, :] - vertex_groups[..., 1, :]
    edges_1 = torch.nn.functional.normalize(edges_1, dim=-1, eps=1e-12)
    edges_2 = torch.nn.functional.normalize(edges_2, dim=-1, eps=1e-12)
    # edges_1 = edges_1 / (torch.linalg.norm(edges_1, dim=-1, keepdim=True) + 1e-5)
    # edges_2 = edges_2 / (torch.linalg.norm(edges_2, dim=-1, keepdim=True) + 1e-5)

    # preparation for stable arccos
    epsilon = 1e-2
    dot_product = (1.0 - epsilon) * torch.sum(edges_1 * edges_2, dim=-1)
    # dot_product = torch.sum(edges_1 * edges_2, dim=-1)

    angle_residuals = torch.arccos(dot_product) - rest_angles
    # angle_residuals = dot_product - rest_angles
    return 0.5 * stiffnesses * angle_residuals**2

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
    positions       : torch.Tensor,
    bending_indices : torch.Tensor,
    rest_angles     : torch.Tensor,
    stiffnesses     : torch.Tensor,
) -> torch.Tensor:
    return -batched_force_temp(
        positions[bending_indices],
        rest_angles,
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
    positions       : torch.Tensor,
    bending_indices : torch.Tensor,
    rest_angles     : torch.Tensor,
    stiffnesses     : torch.Tensor,
) -> torch.Tensor:
    return -batched_dforce_dx_temp(
        positions[bending_indices],
        rest_angles,
        stiffnesses,
    )[0][0]


def computeContributions(
    positions       : torch.Tensor,
    bending_indices : torch.Tensor,
    rest_angles     : torch.Tensor,
    stiffnesses     : torch.Tensor,
    time_step       : float,
) -> tuple:
    forces = force(
        positions,
        bending_indices,
        rest_angles,
        stiffnesses,
    ).flatten()

    dforces_dx = dforce_dx(
        positions,
        bending_indices,
        rest_angles,
        stiffnesses,
    ).flatten()

    arange = torch.arange(positions.shape[1], device=positions.device).unsqueeze(0).unsqueeze(1)
    
    vector_indices = positions.shape[-1] * bending_indices.unsqueeze(-1) + arange
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
    positions       : torch.Tensor,
    bending_indices : torch.Tensor,
    rest_angles     : torch.Tensor,
    stiffnesses     : torch.Tensor,
    time_step       : float,
) -> tuple:
    forces = force(
        positions,
        bending_indices,
        rest_angles,
        stiffnesses,
    ).flatten()

    arange = torch.arange(positions.shape[1], device=positions.device).unsqueeze(0)
    
    vector_indices = positions.shape[-1] * bending_indices.unsqueeze(-1) + arange
    vector_indices = vector_indices.flatten()

    force_vector = torch.zeros_like(positions).flatten()
    force_vector = time_step * torch.index_add(input=force_vector, dim=0, index=vector_indices, source=forces)

    return force_vector.view_as(positions)

###################################################################################################

def contraint(
    vertex_groups : torch.Tensor,
    rest_angles   : torch.Tensor,
) -> torch.Tensor:
    edges_1 = vertex_groups[..., 1, :] - vertex_groups[..., 0, :]
    edges_2 = vertex_groups[..., 2, :] - vertex_groups[..., 1, :]
    edges_1 = torch.nn.functional.normalize(edges_1, dim=-1)
    edges_2 = torch.nn.functional.normalize(edges_2, dim=-1)

    # preparation for stable arccos
    epsilon = 1e-5
    dot_product = torch.clamp(torch.sum(edges_1 * edges_2, dim=-1), min=-1.0 + epsilon, max=1.0 - epsilon)
    # dot_product = 0.999 * torch.sum(edges_1 * edges_2, dim=-1)

    angle_residuals = torch.arccos(dot_product) - rest_angles
    return angle_residuals

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
    positions       : torch.Tensor,
    bending_indices : torch.Tensor,
    rest_angles    : torch.Tensor,
) -> torch.Tensor:
    return -batched_jacobian_temp(
        positions[bending_indices],
        rest_angles,
    )[0]


def jacobian_matrix(
    positions              : torch.Tensor,
    bending_indices        : torch.Tensor,
    rest_angles           : torch.Tensor,
    constraint_block_start : int,
) -> torch.Tensor:
    values = jacobian(
                 positions=positions,
                 bending_indices=bending_indices,
                 rest_angles=rest_angles,
             ).flatten(-3, -1)

    row_indices = torch.arange(bending_indices.shape[-2], device=positions.device)
    row_indices = torch.repeat_interleave(row_indices, bending_indices.shape[-1] * positions.shape[-1], dim=0) + constraint_block_start

    col_indices = torch.arange(positions.shape[-1], device=positions.device)
    col_indices = positions.shape[-1] * bending_indices.unsqueeze(-1) + col_indices.unsqueeze(-2).unsqueeze(-2)
    col_indices = col_indices.flatten()

    matrix_indices = torch.stack((row_indices, col_indices), dim = 0)
    jacobian_matrix = torch.sparse_coo_tensor(
                          indices=matrix_indices,
                          values=values,
                          device=positions.device,
                      )
    
    return jacobian_matrix