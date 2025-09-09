import torch

import phases.general
import scene_content
import objects.cloth
import rendering.rendering
import evaluation.evaluation
import utils.auxiliary


def optimize(
    scene_content    : scene_content.SceneContent,
    renderer         : rendering.rendering.Renderer,
    auxiliary_data   : utils.auxiliary.AuxiliaryData,
    target_time_step : float,
    repetitions      : int,
    gui_images       : dict,
    point_clouds     : list,
    optimize         : bool,
    evaluate         : bool,
    debug            : bool,
    headless         : bool,
):
    scene_content.frame_counter += 1
    if scene_content.frame_counter == 0:
        phases.general.resetCloth(target_time_step / repetitions, scene_content)
        phases.general.resetLoss(scene_content)
    else:
        simulate(target_time_step, repetitions, scene_content)

    render_output = phases.general.renderImages(scene_content, renderer, False)
    if debug and not headless:
        with torch.no_grad():
            renderer.setGuiImages(
                scene_content,
                render_output,
                gui_images,
                use_diffrec=False,
            )
    computeLoss(scene_content, renderer, render_output)

    if evaluate:
        with torch.no_grad():
            point_clouds["ours"][scene_content.frame_counter, :point_clouds["lengths"][scene_content.frame_counter]] = evaluation.evaluation.sampleMesh(point_clouds["lengths"][scene_content.frame_counter].item(), scene_content.cloth_1.positions, scene_content.cloth_1.faces)

    if scene_content.frame_counter >= scene_content.last_frame - 1:
        chamfer_distance = torch.tensor([0.0])
        if evaluate and scene_content.frame_counter >= 0:
            with torch.no_grad():
                scene_content.chamfer_distances = evaluation.evaluation.computeChamferDistance(point_clouds["ground_truth"], point_clouds["ours"], scene_content.last_frame, point_clouds["lengths"])
                chamfer_distance = torch.mean(scene_content.chamfer_distances)

        if scene_content.epoch_counter % 20 == 0:
            auxiliary_data.printQuantities(scene_content, chamfer_distance)
        if not headless and (scene_content.epoch_counter % 1 == 0):
            auxiliary_data.logQuantities(scene_content, chamfer_distance)

        if optimize:
            phases.general.backward(scene_content)
            phases.general.updateParameters(scene_content, optimizer_keys=["stretching", "shearing", "bending", "constant_forces", "vertex_forces"], scheduler_keys=[])
            phases.general.clampOptimization(scene_content)
            phases.general.updateClothParameters(scene_content)

        if (    scene_content.last_frame < scene_content.max_frames
            and scene_content.epoch_counter % scene_content.new_frame_period == 0):
            scene_content.last_frame += 1
            if scene_content.last_frame == scene_content.max_frames:
                print("Reached max frames")
        scene_content.frame_counter = -1
        scene_content.epoch_counter += 1



def simulate(
    target_time_step : float,
    repetitions      : int,
    scene_content    : scene_content.SceneContent,
):
    external_forces = scene_content.cloth_1.vertex_masses * (scene_content.constant_forces + scene_content.vertex_forces[scene_content.frame_counter - 1])
    
    for i in range(repetitions):
        scene_content.cloth_1.simulate(
            method     = "primal",
            integrator = "BDF1",
            time_step  = target_time_step / repetitions,
            damping_factor  = scene_content.damping_factor,
            external_forces = external_forces,
            do_pulling      = False,
            anchor_points   = scene_content.anchor_points + torch.sum(scene_content.anchor_movement[:scene_content.frame_counter], axis=0),
        )



def computeLoss(
    scene_content : scene_content.SceneContent,
    renderer      : rendering.rendering.Renderer,
    render_output : dict,
):
    # rgb loss
    image_diff = (render_output["color_diffrast"][0] - scene_content.ground_truth_collection[0][scene_content.frame_counter, ..., 0:3])
    scene_content.losses["rgb"] += 1.0 * torch.mean(torch.abs(image_diff)**2)

    # silhouette loss
    mask_diff = (render_output["mask_diffrast"][0, ..., 0] - scene_content.ground_truth_collection[0][scene_content.frame_counter, ..., 3])
    scene_content.losses["silhouette"] += 0.2 * torch.mean(torch.abs(mask_diff)**2)

    # deformation regularization
    scene_content.losses["deformation_energy"] += 2.0 * objects.cloth.deformationEnergy(
        scene_content.cloth_1.positions,
        scene_content.cloth_1.stretching_segments.indices,
        scene_content.cloth_1.stretching_segments.rest_lengths,
        torch.repeat_interleave(10**scene_content.log_stretching_stiffness.detach(), scene_content.cloth_1.stretching_segments.indices.shape[0], dim=0).detach(),
        scene_content.cloth_1.bending_segments.indices,
        scene_content.cloth_1.bending_segments.rest_angles,
        torch.repeat_interleave(10**scene_content.log_bending_stiffness.detach(), scene_content.cloth_1.bending_segments.indices.shape[0], dim=0).detach(),
        scene_content.cloth_1.shearing_segments.indices,
        scene_content.cloth_1.shearing_segments.rest_angles,
        torch.repeat_interleave(10**scene_content.log_shearing_stiffness.detach(), scene_content.cloth_1.shearing_segments.indices.shape[0], dim=0).detach(),
    )

    # single camera only
    if scene_content.frame_counter > 0:
        force_vectors = scene_content.constant_forces + scene_content.vertex_forces[scene_content.frame_counter - 1]
        view_directions = torch.nn.functional.normalize(scene_content.cloth_1.previous_positions - renderer.camera_positions[0].unsqueeze(0).detach(), dim=-1)
        dot_product = torch.sum(force_vectors * view_directions, dim=-1)
        in_plane_forces = force_vectors - dot_product.unsqueeze(-1) * view_directions
        scene_content.losses["vertex_forces"] += 2e-4 * torch.mean(torch.linalg.norm(in_plane_forces, dim=-1))


    if scene_content.frame_counter >= scene_content.last_frame - 1:
        # average over frames
        for key in scene_content.losses:
            scene_content.losses[key] /= scene_content.last_frame
            scene_content.loss += scene_content.losses[key]

