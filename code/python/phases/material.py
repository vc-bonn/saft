import torch
import nvdiffrecmc.render.util

import nvdiffrecmc.render.light as light

import phases.general
import scene_content
import rendering.rendering
import evaluation.evaluation
import utils.auxiliary
import utils.mesh_utils


def optimize(
    scene_content        : scene_content.SceneContent,
    renderer             : rendering.rendering.Renderer,
    auxiliary_data       : utils.auxiliary.AuxiliaryData,
    target_time_step     : float,
    repetitions          : int,
    gui_images           : dict,
    point_clouds         : list,
    optimize             : bool,
    evaluate             : bool,
    debug                : bool,
    headless             : bool,
    load_positions       : bool = True,
    use_material_decoder : bool = True,
):
    scene_content.frame_counter += 1

    if scene_content.frame_counter == scene_content.first_frame:
        if not load_positions:
            phases.general.resetCloth(target_time_step / repetitions, scene_content)
        phases.general.resetLoss(scene_content)
    if load_positions:
        with torch.no_grad():
            scene_content.cloth_1.positions = scene_content.vertex_positions[scene_content.frame_counter]
            scene_content.cloth_1.normals = utils.mesh_utils.computeNormals(positions=scene_content.cloth_1.positions, faces=scene_content.cloth_1.faces)

    render_output = phases.general.renderImages(scene_content, renderer, True)
    if debug and not headless:
        with torch.no_grad():
            renderer.setGuiImages(
                scene_content,
                render_output,
                gui_images,
                use_diffrec=True,
            )
    computeLoss(scene_content, render_output, renderer)

    if evaluate:
        with torch.no_grad():
            point_clouds["ours"][scene_content.frame_counter, :point_clouds["lengths"][scene_content.frame_counter]] = evaluation.evaluation.sampleMesh(point_clouds["lengths"][scene_content.frame_counter].item(), scene_content.cloth_1.positions, scene_content.cloth_1.faces)

    if scene_content.frame_counter >= scene_content.last_frame - 1:
        chamfer_distance = torch.tensor([0.0])
        if evaluate and scene_content.frame_counter >= 0:
            with torch.no_grad():
                scene_content.chamfer_distances = evaluation.evaluation.computeChamferDistance(point_clouds["ground_truth"], point_clouds["ours"], scene_content.last_frame, point_clouds["lengths"])
                chamfer_distance = torch.mean(scene_content.chamfer_distances)

        if scene_content.epoch_counter % 50 == 0:
            auxiliary_data.printQuantities(scene_content, chamfer_distance)
        if not headless and (scene_content.epoch_counter % 10 == 0):
            auxiliary_data.logQuantities(scene_content, chamfer_distance)

        if optimize:
            phases.general.backward(scene_content)
            if use_material_decoder:
                scene_content.material_decoder.step()
                scene_content.material_decoder.zero_grad()
                scene_content.diffuse, scene_content.roughness, scene_content.metallic, scene_content.normal_map = scene_content.material_decoder()
                phases.general.updateParameters(scene_content, optimizer_keys=["environment_map"], scheduler_keys=[])
            else: # use direct optimization of diffuse, roughness and metallic
                phases.general.updateParameters(scene_content, optimizer_keys=["diffuse", "roughness", "metallic", "environment_map"], scheduler_keys=[])
        phases.general.clampOptimization(scene_content)

        scene_content.lerp = max(scene_content.lerp - (1/400), 0.0)
        interpolation = min(scene_content.lerp, 1)
        new_environment_map_tensor = interpolation * torch.mean(scene_content.environment_map_tensor, dim=-1, keepdim=True) + (1.0-interpolation) * scene_content.environment_map_tensor
        scene_content.environment_map = light.EnvironmentLight(new_environment_map_tensor)
        
        scene_content.frame_counter = scene_content.first_frame - 1
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
    render_output : dict,
    renderer      : rendering.rendering.Renderer,
):
    # rgb loss
    image_diff = (nvdiffrecmc.render.util.rgb_to_srgb(render_output["color_diffrec"]) - scene_content.ground_truth_collection[0][scene_content.frame_counter, ..., 0:3])
    scene_content.losses["rgb"] += 1.0 * torch.mean(torch.abs(image_diff)**2)

    # # amplitude regularizations
    # scene_content.loss += 1e0 * torch.mean(torch.nn.functional.relu(scene_content.diffuse - 0.95)**2)
    # scene_content.loss += 1e-5 * torch.mean(torch.abs(scene_content.roughness - 1))
    # scene_content.loss += 1e-5 * torch.mean(torch.abs(scene_content.metallic))
    # scene_content.loss += 1e-5 * torch.mean(torch.abs(scene_content.environment_map_tensor))

    # # smoothness regularizations
    # scene_content.loss += 5e-5 * torch.mean(torch.abs(scene_content.diffuse[:, 1:]                - scene_content.diffuse[:, :-1]               )**1)
    # scene_content.loss += 5e-5 * torch.mean(torch.abs(scene_content.diffuse[:, :, 1:]             - scene_content.diffuse[:, :, :-1]            )**1)
    # scene_content.loss += 5e-5 * torch.mean(torch.abs(scene_content.roughness[:, 1:]              - scene_content.roughness[:, :-1]             )**1)
    # scene_content.loss += 5e-5 * torch.mean(torch.abs(scene_content.roughness[:, :, 1:]           - scene_content.roughness[:, :, :-1]          )**1)
    # scene_content.loss += 5e-5 * torch.mean(torch.abs(scene_content.metallic[:, 1:]               - scene_content.metallic[:, :-1]              )**1)
    # scene_content.loss += 5e-5 * torch.mean(torch.abs(scene_content.metallic[:, :, 1:]            - scene_content.metallic[:, :, :-1]           )**1)
    scene_content.loss += 5e-5 * torch.mean(torch.abs(scene_content.environment_map_tensor[1:]    - scene_content.environment_map_tensor[:-1]   )**1)
    scene_content.loss += 5e-5 * torch.mean(torch.abs(scene_content.environment_map_tensor[:, 1:] - scene_content.environment_map_tensor[:, :-1])**1)

    # # luminance regularization
    # luminance = torch.mean(nvdiffrecmc.render.util.rgb_to_srgb(render_output["diffuse_light_diffrec"][0] + render_output["specular_light_diffrec"][0]), dim=-1)
    # luminance = torch.mean(render_output["diffuse_light_diffrec"][0] + render_output["specular_light_diffrec"][0], dim=-1)
    # scene_content.losses["vertex_forces"] += 1e-2 * torch.mean(torch.abs(luminance - torch.amax(nvdiffrecmc.render.util.srgb_to_rgb(scene_content.ground_truth_collection[0][scene_content.frame_counter, ..., 0:3]), dim=-1)))


    if scene_content.frame_counter >= scene_content.last_frame - 1:
        # average over frames
        for key in scene_content.losses.keys():
            scene_content.losses[key] /= scene_content.last_frame

        for key in scene_content.losses:
            scene_content.loss += scene_content.losses[key]

