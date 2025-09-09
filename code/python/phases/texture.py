import torch

import phases.general
import scene_content
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

    phases.general.resetCloth(target_time_step / repetitions, scene_content)
    phases.general.resetLoss(scene_content)

    render_output = phases.general.renderImages(scene_content, renderer, use_diffrec=False)
    if debug and not headless:
        with torch.no_grad():
            renderer.setGuiImages(
                scene_content,
                render_output,
                gui_images,
                use_diffrec=False,
            )
    computeLoss(scene_content, render_output)

    chamfer_distance = torch.tensor([0.0])
    if evaluate:
        with torch.no_grad():
            point_clouds["ours"][scene_content.frame_counter, :point_clouds["lengths"][scene_content.frame_counter]] = evaluation.evaluation.sampleMesh(point_clouds["lengths"][scene_content.frame_counter].item(), scene_content.cloth_1.positions, scene_content.cloth_1.faces)
            scene_content.chamfer_distances = evaluation.evaluation.computeChamferDistance(point_clouds["ground_truth"], point_clouds["ours"], scene_content.frame_counter + 1, point_clouds["lengths"])
            chamfer_distance = torch.mean(scene_content.chamfer_distances)

    if scene_content.epoch_counter % 50 == 0:
        auxiliary_data.printQuantities(scene_content, chamfer_distance)
    if not headless and (scene_content.epoch_counter % 10 == 0):
        auxiliary_data.logQuantities(scene_content, chamfer_distance)

    if optimize:
        phases.general.backward(scene_content)
        phases.general.updateParameters(scene_content, optimizer_keys=["texture"], scheduler_keys=[])

    scene_content.frame_counter = -1
    scene_content.epoch_counter += 1



def computeLoss(
    scene_content : scene_content.SceneContent,
    render_output : dict,
):
    # rgb loss
    image_diff = (render_output["color_diffrast"][0] - scene_content.ground_truth_collection[0][scene_content.frame_counter, ..., 0:3])
    scene_content.losses["rgb"] += 1.0 * torch.mean(torch.abs(image_diff)**2)

    scene_content.loss += 1e-4 * torch.mean(torch.abs(scene_content.texture[:, 1:]    - scene_content.texture[:, :-1]   )**1)
    scene_content.loss += 1e-4 * torch.mean(torch.abs(scene_content.texture[:, :, 1:] - scene_content.texture[:, :, :-1])**1)

    for key in scene_content.losses:
        scene_content.loss += scene_content.losses[key]