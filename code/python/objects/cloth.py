import torch
import itertools
import openmesh as om

import utils.mesh_utils
import physics.general
import physics.pushing
import physics.pulling
import physics.stretching
import physics.bending
import physics.solver

class PullingSegments():
    def __init__(
        self,
        indices          : torch.Tensor,
        stiffnesses      : torch.Tensor,
        target_positions : torch.Tensor,
    ) -> None:
        self.indices = indices
        self.stiffnesses = stiffnesses
        self.target_positions = target_positions

class StretchingSegments():
    def __init__(
        self,
        indices      : torch.Tensor,
        stiffnesses  : torch.Tensor,
        rest_lengths : torch.Tensor,
    ) -> None:
        self.indices = indices
        self.stiffnesses = stiffnesses
        self.rest_lengths = rest_lengths

class BendingSegments():
    def __init__(
        self,
        indices     : torch.Tensor,
        stiffnesses : torch.Tensor,
        rest_angles : torch.Tensor,
    ) -> None:
        self.indices = indices
        self.stiffnesses = stiffnesses
        self.rest_angles = rest_angles

class ShearingSegments():
    def __init__(
        self,
        indices     : torch.Tensor,
        stiffnesses : torch.Tensor,
        rest_angles : torch.Tensor,
    ) -> None:
        self.indices = indices
        self.stiffnesses = stiffnesses
        self.rest_angles = rest_angles


def deformationEnergy(
    positions               : torch.Tensor,
    stretching_indices      : torch.Tensor,
    stretching_rest_lengths : torch.Tensor,
    stretching_stiffnesses  : torch.Tensor,
    bending_indices         : torch.Tensor,
    bending_rest_angles     : torch.Tensor,
    bending_stiffnesses     : torch.Tensor,
    shearing_indices        : torch.Tensor,
    shearing_rest_angles    : torch.Tensor,
    shearing_stiffnesses    : torch.Tensor,
) -> torch.Tensor:
    stretching_energy = physics.stretching.energy(
        vertex_groups=positions[stretching_indices],
        rest_lengths=stretching_rest_lengths,
        stiffnesses=stretching_stiffnesses,
    )

    bending_energy = physics.bending.energy(
        vertex_groups=positions[bending_indices],
        rest_angles=bending_rest_angles,
        stiffnesses=bending_stiffnesses,
    )

    shearing_energy = physics.bending.energy(
        vertex_groups=positions[shearing_indices],
        rest_angles=shearing_rest_angles,
        stiffnesses=shearing_stiffnesses,
    )

    deformation_energy = torch.sum(stretching_energy, dim=-1) + torch.sum(bending_energy, dim=-1) + torch.sum(shearing_energy, dim=-1)
    return deformation_energy

def totalEnergy(
    time_step               : float,
    positions               : torch.Tensor,
    vertex_masses           : torch.Tensor,
    predicted_positions     : torch.Tensor,
    external_forces         : torch.Tensor,
    pulling_indices         : torch.Tensor,
    anchor_points           : torch.Tensor,
    pulling_stiffnesses     : torch.Tensor,
    stretching_indices      : torch.Tensor,
    stretching_rest_lengths : torch.Tensor,
    stretching_stiffnesses  : torch.Tensor,
    bending_indices         : torch.Tensor,
    bending_rest_angles     : torch.Tensor,
    bending_stiffnesses     : torch.Tensor,
    shearing_indices        : torch.Tensor,
    shearing_rest_angles    : torch.Tensor,
    shearing_stiffnesses    : torch.Tensor,
) -> torch.Tensor:
    total_energy = 0.0

    position_shift = predicted_positions - positions
    total_energy = 0.5 * torch.sum(vertex_masses * position_shift * position_shift, dim=(-2, -1))

    # energy = physics.pushing.energy(
    #     positions=positions,
    #     forces=external_forces,
    # )
    # total_energy = total_energy + time_step * time_step * torch.sum(energy, dim=-1)

    # energy = physics.pulling.energy(
    #     vertex_groups=positions[pulling_indices],
    #     target_positions=anchor_points,
    #     stiffnesses=pulling_stiffnesses
    # )
    # total_energy = total_energy + time_step * time_step * torch.sum(energy, dim=-1)

    deformation_energy = deformationEnergy(
        positions,
        stretching_indices,
        stretching_rest_lengths,
        stretching_stiffnesses,
        bending_indices,
        bending_rest_angles,
        bending_stiffnesses,
        shearing_indices,
        shearing_rest_angles,
        shearing_stiffnesses,
    )
    total_energy = total_energy + time_step * time_step * deformation_energy

    return total_energy

batched_energies = torch.vmap(
    func=totalEnergy,
    in_dims=(None, 0, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None,),
    out_dims=(0,),
    chunk_size=None,
)




class Cloth():
    def __init__(
        self,
        time_step           : float,
        edges               : torch.Tensor,
        faces               : torch.Tensor,
        initial_positions   : torch.Tensor,
        initial_velocities  : torch.Tensor,
        vertex_masses       : torch.Tensor,
        gravity             : torch.Tensor,
        pulling_segments    : PullingSegments,
        stretching_segments : StretchingSegments,
        bending_segments    : BendingSegments,
        shearing_segments   : ShearingSegments,
    ):
        self.counter = 0

        self.edges = edges
        self.faces = faces

        # simulation parameters
        self.passed_time = 0.0
        self.initial_positions   = initial_positions
        self.initial_velocities  = initial_velocities
        self.positions           = initial_positions.clone()
        self.previous_positions  = initial_positions - time_step * initial_velocities
        self.velocities          = initial_velocities.clone()
        self.previous_velocities = initial_velocities.clone()

        self.gravity             = gravity
        self.pulling_segments    = pulling_segments
        self.stretching_segments = stretching_segments
        self.shearing_segments   = shearing_segments
        self.bending_segments    = bending_segments

        self.vertex_masses = vertex_masses

        self.normals = utils.mesh_utils.computeNormals(positions=self.positions, faces=self.faces)

    def reset(
        self,
        time_step : float,
    ) -> None:
        self.positions  = self.initial_positions.clone()
        self.velocities = self.initial_velocities.clone()
        self.previous_positions  = self.initial_positions - time_step * self.initial_velocities
        self.previous_velocities = self.initial_velocities.clone()
        self.normals = utils.mesh_utils.computeNormals(positions=self.initial_positions, faces=self.faces)

        self.passed_time = 0.0

    def updateStartingState(
        self,
        new_initial_positions  : torch.Tensor,
        new_initial_velocities : torch.Tensor,
    ):
        self.initial_positions[:]  = new_initial_positions
        self.initial_velocities[:] = new_initial_velocities

    def simulate(
        self,
        method     : str,
        integrator : str,
        time_step  : float,
        damping_factor  : torch.Tensor,
        external_forces : torch.Tensor,
        do_pulling      : bool,
        anchor_points   : torch.Tensor,
    ) -> None:
        if method in ["force", "primal"]:
            self.velocities = damping_factor * self.velocities # damping
            self.simulatePrimal(
                integrator=integrator,
                time_step=time_step,
                external_forces=external_forces,
                do_pulling=do_pulling,
                anchor_points=anchor_points,
            )
            self.updatePrimal(
                integrator=integrator,
                time_step=time_step,
            )
        elif method in ["IPC"]:
            self.previous_positions = self.positions
            self.previous_velocities = self.velocities
            predicted_positions = self.positions + time_step * (self.velocities + time_step / self.vertex_masses * external_forces)
            self.positions = self.positions + time_step * self.velocities

            for i in range(2):
                finished = self.simulateIPC(
                    time_step=time_step,
                    predicted_positions=predicted_positions,
                    external_forces=external_forces,
                    do_pulling=do_pulling,
                    anchor_points=anchor_points,
                )

                if finished:
                    break

            self.velocities = (self.positions - self.previous_positions) / time_step
            self.velocities = damping_factor * self.velocities # damping
            # self.velocities = self.velocities + time_step / self.vertex_masses * external_forces
        else:
            print("Please use one of the following methods: \"force\"/\"primal\" or \"constraint\"/\"dual\"")
            return
        
        self.passed_time += time_step
        self.normals = utils.mesh_utils.computeNormals(positions=self.positions, faces=self.faces)



    def simulatePrimal(
        self,
        integrator : str,
        time_step            : float,
        external_forces      : torch.Tensor,
        do_pulling           : bool,
        anchor_points        : torch.Tensor,
    ) -> None:
        self.system_vector = torch.zeros_like(self.positions).flatten()
        self.system_matrix = torch.sparse_coo_tensor(indices=[[], []], values=[], size = (self.system_vector.shape[-1], self.system_vector.shape[-1]), device=self.system_vector.device, dtype=self.positions.dtype)

        weight = 1.0
        if integrator == "BDF2":
            weight = 2.0 / 3.0

        mass_matrix, momentum_vector = physics.general.computeContributions(
            vertex_masses=self.vertex_masses,
            velocities=self.velocities,
            previous_velocities=self.previous_velocities,
            method=integrator,
        )
        self.system_matrix = self.system_matrix + mass_matrix
        self.system_vector = self.system_vector + momentum_vector

        force_vector = physics.pushing.computeContributions(
            forces=external_forces,
            time_step=time_step,
        )
        self.system_vector = self.system_vector + weight * force_vector

        if do_pulling:
            mass_matrix, force_vector = physics.pulling.computeContributions(
                positions=self.positions,
                pulling_indices=self.pulling_segments.indices,
                target_positions=anchor_points,
                stiffnesses=self.pulling_segments.stiffnesses,
                time_step=time_step,
            )
            self.system_matrix = self.system_matrix + weight * mass_matrix
            self.system_vector = self.system_vector + weight * force_vector

        mass_matrix, force_vector = physics.stretching.computeContributions(
            positions=self.positions,
            stretching_indices=self.stretching_segments.indices,
            rest_lengths=self.stretching_segments.rest_lengths,
            stiffnesses=self.stretching_segments.stiffnesses,
            time_step=time_step,
        )
        self.system_matrix = self.system_matrix + weight * mass_matrix
        self.system_vector = self.system_vector + weight * force_vector

        mass_matrix, force_vector = physics.bending.computeContributions(
            positions=self.positions,
            bending_indices=self.bending_segments.indices,
            rest_angles=self.bending_segments.rest_angles,
            stiffnesses=self.bending_segments.stiffnesses,
            time_step=time_step,
        )
        self.system_matrix = self.system_matrix + weight * mass_matrix
        self.system_vector = self.system_vector + weight * force_vector

        mass_matrix, force_vector = physics.bending.computeContributions(
            positions=self.positions,
            bending_indices=self.shearing_segments.indices,
            rest_angles=self.shearing_segments.rest_angles,
            stiffnesses=self.shearing_segments.stiffnesses,
            time_step=time_step,
        )
        self.system_matrix = self.system_matrix + weight * mass_matrix
        self.system_vector = self.system_vector + weight * force_vector


    def updatePrimal(
        self,
        integrator : str,
        time_step  : float,
    ) -> None:
        self.system_matrix = self.system_matrix.coalesce()

        # new_velocities = torch.linalg.solve(self.system_matrix.to_dense(), self.system_vector)
        # new_velocities, _, _, _ = torch.linalg.lstsq(self.system_matrix.to_dense(), self.system_vector, driver="gels")
        if torch.norm(self.system_vector) < 1e-8:
            new_velocities = self.velocities + time_step * self.system_vector.view(-1, self.positions.shape[-1])
        else:
            new_velocities = physics.solver.COOSolve.apply(
                                 self.system_matrix,
                                 self.system_vector,
                                 self.velocities.flatten(),
                                 None,
                                 torch.tensor([1000], device = "cuda"),
                                 torch.tensor([1e-4], device = "cuda"),
                                 torch.tensor([0.0], device = "cuda")
                             )
            new_velocities = new_velocities.view(-1, self.positions.shape[-1])
        
        if integrator == "euler" or integrator == "BDF1":
            new_positions = self.positions + time_step * new_velocities
        elif integrator == "BDF2":
            new_positions = 4.0/3.0 * self.positions - 1.0/3.0 * self.previous_positions + time_step * new_velocities
            self.previous_velocities = self.velocities
            self.previous_positions  = self.positions

        self.velocities = new_velocities
        self.positions = new_positions
        self.passed_time += time_step



    def simulateIPC(
        self,
        time_step            : float,
        predicted_positions  : torch.Tensor,
        external_forces      : torch.Tensor,
        do_pulling           : bool,
        anchor_points        : torch.Tensor,
    ) -> bool:
        self.system_vector = torch.zeros_like(self.positions).flatten()
        self.system_matrix  = torch.sparse_coo_tensor(indices=[[], []], values=[], size = (self.system_vector.shape[-1], self.system_vector.shape[-1]), device=self.system_vector.device, dtype=self.positions.dtype)

        mass_matrix, momentum_vector = physics.general.computeContributionsIPC(
            vertex_masses=self.vertex_masses,
            positions=self.positions,
            predicted_positions=predicted_positions,
        )
        self.system_matrix = self.system_matrix + mass_matrix
        self.system_vector = self.system_vector + momentum_vector# * time_step

        # External forces only necessary when not included in predicted positions
        # force_vector = physics.pushing.computeContributions(
        #     forces=external_forces,
        #     time_step=time_step,
        # )
        # self.system_vector = self.system_vector + time_step * force_vector

        if do_pulling:
            mass_matrix, force_vector = physics.pulling.computeContributions(
                positions=self.positions,
                pulling_indices=self.pulling_segments.indices,
                target_positions=anchor_points,
                stiffnesses=self.pulling_segments.stiffnesses,
                time_step=time_step,
            )
            self.system_matrix = self.system_matrix + mass_matrix
            self.system_vector = self.system_vector + time_step * force_vector

        mass_matrix, force_vector = physics.stretching.computeContributions(
            positions=self.positions,
            stretching_indices=self.stretching_segments.indices,
            rest_lengths=self.stretching_segments.rest_lengths,
            stiffnesses=self.stretching_segments.stiffnesses,
            time_step=time_step,
        )
        self.system_matrix = self.system_matrix + mass_matrix
        self.system_vector = self.system_vector + time_step * force_vector

        mass_matrix, force_vector = physics.bending.computeContributions(
            positions=self.positions,
            bending_indices=self.bending_segments.indices,
            rest_angles=self.bending_segments.rest_angles,
            stiffnesses=self.bending_segments.stiffnesses,
            time_step=time_step,
        )
        self.system_matrix = self.system_matrix + mass_matrix
        self.system_vector = self.system_vector + time_step * force_vector

        mass_matrix, force_vector = physics.bending.computeContributions(
            positions=self.positions,
            bending_indices=self.shearing_segments.indices,
            rest_angles=self.shearing_segments.rest_angles,
            stiffnesses=self.shearing_segments.stiffnesses,
            time_step=time_step,
        )
        self.system_matrix = self.system_matrix + mass_matrix
        self.system_vector = self.system_vector + time_step * force_vector

        return self.updateIPC(
            time_step=time_step,
            predicted_positions=predicted_positions,
            external_forces=external_forces,
            anchor_points=anchor_points,
        )


    def updateIPC(
        self,
        time_step            : float,
        predicted_positions  : torch.Tensor,
        external_forces      : torch.Tensor,
        anchor_points        : torch.Tensor,
    ) -> bool:
        self.system_matrix = self.system_matrix.coalesce()

        # position_shift = torch.linalg.solve(self.system_matrix.to_dense(), self.system_vector)
        # position_shift, _, _, _ = torch.linalg.lstsq(self.system_matrix.to_dense(), self.system_vector, driver="gels")
        position_shift = physics.solver.COOSolve.apply(
                             self.system_matrix,
                             self.system_vector,
                             self.velocities.flatten(),
                             None,
                             torch.tensor([1000], device = "cuda"),
                             torch.tensor([1e-4], device = "cuda"),
                             torch.tensor([0.0], device = "cuda")
                         )
        position_shift = position_shift.view_as(self.positions)

        with torch.no_grad():
            sampling_distances = torch.logspace(-1, 1, 199, device=self.positions.device)
            sampling_distances = torch.cat((torch.tensor([0.0], device=sampling_distances.device), sampling_distances, -sampling_distances)).unsqueeze(-1).unsqueeze(-1)
            test_positions = self.positions.unsqueeze(0) + sampling_distances * position_shift.unsqueeze(0)

            energies = batched_energies(
                time_step,
                test_positions,
                self.vertex_masses,
                predicted_positions,
                external_forces,
                self.pulling_segments.indices,
                anchor_points,
                self.pulling_segments.stiffnesses,
                self.stretching_segments.indices,
                self.stretching_segments.rest_lengths,
                self.stretching_segments.stiffnesses,
                self.bending_segments.indices,
                self.bending_segments.rest_angles,
                self.bending_segments.stiffnesses,
                self.shearing_segments.indices,
                self.shearing_segments.rest_angles,
                self.shearing_segments.stiffnesses,
            )
            index = torch.argmin(energies)
            scaling_factor = sampling_distances[index]
            # print(f"index = {index:4d}   E_0 = {energies[0]: .3e}   E_min = {energies[index]: .3e}   ΔE_rel = {(energies[0] - energies[index]) / energies[index]: .2e}")

        if index == 0:
            # zero shift
            return True

        self.positions = self.positions + scaling_factor * position_shift

        if energies[0] - energies[index] < 1e-2 * torch.abs(energies[index]):
            return True

        return False



class DefaultCloth(Cloth):
    def __init__(
        self,
        polygon_mesh         : om.PolyMesh,
        time_step            : float,
        initial_positions    : torch.Tensor,
        initial_velocities   : torch.Tensor,
        edges                : torch.Tensor,
        faces                : torch.Tensor,
        gravity              : torch.Tensor,
        area_density         : torch.Tensor,
        pulling_stiffness    : torch.Tensor,
        stretching_stiffness : torch.Tensor,
        bending_stiffness    : torch.Tensor,
        shearing_stiffness   : torch.Tensor,
    ):
        device = initial_positions.device

        # vertex masses
        face_areas = utils.mesh_utils.computeFaceAreas(
            positions=initial_positions,
            faces=faces,
        )
        vertex_masses = torch.zeros((initial_positions.shape[0]), dtype=initial_positions.dtype, device=device)
        vertex_masses = torch.index_add(input=vertex_masses, dim=0, index=faces[:, 0], source=face_areas/3.0)
        vertex_masses = torch.index_add(input=vertex_masses, dim=0, index=faces[:, 1], source=face_areas/3.0)
        vertex_masses = torch.index_add(input=vertex_masses, dim=0, index=faces[:, 2], source=face_areas/3.0)
        vertex_masses = (vertex_masses * area_density)
        vertex_masses = torch.repeat_interleave(vertex_masses.unsqueeze(1), repeats=initial_positions.shape[-1], dim=1)
        # vertex_masses = torch.full_like(vertex_masses, 0.2 / initial_positions.shape[0])

        # pulling segments
        pulling_indices = torch.tensor([0, 1, 2, 3], device=device)
        pulling_stiffnesses = torch.repeat_interleave(pulling_stiffness, pulling_indices.shape[0])[:pulling_indices.shape[0]]
        pulling_positions = initial_positions[pulling_indices]
        pulling_segments = PullingSegments(
                               indices=pulling_indices,
                               stiffnesses=pulling_stiffnesses,
                               target_positions=pulling_positions,
                           )

        # stretching segments
        rest_lengths = torch.linalg.norm(initial_positions[edges[:, 0]] - initial_positions[edges[:, 1]], dim=1)
        stretching_stiffnesses = torch.repeat_interleave(stretching_stiffness, edges.shape[0])[:edges.shape[0]]
        stretching_segments = StretchingSegments(
                                  indices=edges.clone(),
                                  stiffnesses=stretching_stiffnesses,
                                  rest_lengths=rest_lengths,
                              )

        # bending and shearing segments
        hinge_indices = torch.zeros((15*initial_positions.shape[0], 3), dtype=stretching_segments.indices.dtype, device=device) # initialize with large buffer
        hinge_i = 0
        for vertex in polygon_mesh.vertices():
            index_vertex = vertex.idx()
            neighbors = itertools.combinations(polygon_mesh.vv(vertex), r=2)
            for neighbor_1, neighbor_2 in neighbors:
                index_neighbor_1 = neighbor_1.idx()
                index_neighbor_2 = neighbor_2.idx()
                hinge_indices[hinge_i] = torch.tensor([index_neighbor_1, index_vertex, index_neighbor_2], dtype=hinge_indices.dtype, device=device)
                hinge_i += 1
        hinge_indices = hinge_indices[:hinge_i].contiguous()

        neighbors_1 = initial_positions[hinge_indices[:, 0]]
        vertices    = initial_positions[hinge_indices[:, 1]]
        neighbors_2 = initial_positions[hinge_indices[:, 2]]
        edges_1 = vertices - neighbors_1
        edges_2 = neighbors_2 - vertices
        edges_1 = torch.nn.functional.normalize(edges_1, dim=-1, eps=1e-12)
        edges_2 = torch.nn.functional.normalize(edges_2, dim=-1, eps=1e-12)
        # edges_1 = edges_1 / (torch.linalg.norm(edges_1, dim=-1, keepdim=True) + 1e-5)
        # edges_2 = edges_2 / (torch.linalg.norm(edges_2, dim=-1, keepdim=True) + 1e-5)
        epsilon = 1e-2
        # rest_angles = torch.arccos(torch.clamp(torch.sum(edges_1 * edges_2, dim=-1), min=-1.0 + epsilon, max=1.0 - epsilon))
        rest_angles = torch.arccos((1.0 - epsilon) * torch.sum(edges_1 * edges_2, dim=-1))
        # rest_angles = torch.clamp(torch.sum(edges_1 * edges_2, dim=-1), min=-1.0 + epsilon, max=1.0 - epsilon)
        # rest_angles = torch.sum(edges_1 * edges_2, dim=-1)

        hinge_threshold = torch.deg2rad(torch.tensor([45.0], device=device))
        condition = rest_angles < hinge_threshold
        # hinge_threshold = torch.cos(torch.deg2rad(torch.tensor([45.0], device=device)))
        # condition = rest_angles > hinge_threshold
        bending_indices = hinge_indices[condition]
        shearing_indices = hinge_indices[~condition]

        bending_segments = BendingSegments(
                               indices=bending_indices,
                               stiffnesses=torch.repeat_interleave(bending_stiffness, bending_indices.shape[0])[:bending_indices.shape[0]],
                               rest_angles=rest_angles[condition],
                            #    rest_angles=torch.full_like(rest_angles[condition], torch.deg2rad(torch.tensor([0.0], device=device)).item()),
                            #    rest_angles=torch.full_like(rest_angles[condition], 1.0),
                           )
        shearing_segments = ShearingSegments(
                                indices=shearing_indices,
                                stiffnesses=torch.repeat_interleave(shearing_stiffness, shearing_indices.shape[0])[:shearing_indices.shape[0]],
                                rest_angles=rest_angles[~condition],
                                # rest_angles=torch.full_like(rest_angles[~condition], torch.deg2rad(torch.tensor([90.0], device=device)).item()),
                                # rest_angles=torch.full_like(rest_angles[~condition], 0.0),
                            )

        super().__init__(
            time_step=time_step,
            edges=edges,
            faces=faces,
            initial_positions=initial_positions,
            initial_velocities=initial_velocities,
            vertex_masses=vertex_masses,
            gravity=gravity,
            pulling_segments=pulling_segments,
            stretching_segments=stretching_segments,
            shearing_segments=shearing_segments,
            bending_segments=bending_segments,
        )