import torch

import scene_content
import rendering.rendering


def resetCloth(
    time_step     : float,
    scene_content : scene_content.SceneContent,
):
    with torch.no_grad():
        scene_content.cloth_1.reset(time_step)

        # remove temporally and spacially constant part that could be modeled by wind
        scene_content.vertex_forces -= torch.mean(scene_content.vertex_forces, dim = [0, 1]).unsqueeze_(0).unsqueeze_(1)

def resetLoss(
    scene_content : scene_content.SceneContent,
):
    for key in scene_content.losses:
        scene_content.losses[key] = torch.tensor([0.], dtype=scene_content.losses[key].dtype, device = scene_content.losses[key].device)
    scene_content.loss = torch.tensor([0.], dtype=scene_content.loss.dtype, device = scene_content.loss.device)



def renderImages(
    scene_content : scene_content.SceneContent,
    renderer      : rendering.rendering.Renderer,
    use_diffrec   : bool,
) -> dict:
    if use_diffrec:
        render_output = renderer.renderDiffrecmc(
                            scene_content.cloth_1.positions,
                            scene_content.cloth_1.faces,
                            scene_content.uv,
                            scene_content.cloth_1.normals,
                            scene_content.texture,
                            scene_content.diffuse,
                            scene_content.metallic,
                            scene_content.roughness,
                            scene_content.normal_map,
                            scene_content.environment_map,
                            renderer.model_view_projection_matrix,
                            renderer.camera_positions,
                            renderer.crop_size,
                        )
    else:
        render_output = renderer.renderDiffrast(
                            scene_content.cloth_1.positions,
                            scene_content.cloth_1.faces,
                            scene_content.uv,
                            scene_content.cloth_1.normals,
                            scene_content.texture,
                            renderer.model_view_projection_matrix,
                            renderer.crop_size,
                        )

    return render_output



def applyGradient(parameter, optimizer, clipper):
    if parameter.grad is not None:
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameter, max_norm=1e3)
        if torch.isfinite(gradient_norm) and torch.all(torch.isfinite(parameter.grad)):
            # adaptive gradient clipping
            clipper.step()
            optimizer.step()

def backward(
    scene_content : scene_content.SceneContent,
    retain_graph  : bool = False,
):
    loss_norm = scene_content.loss.detach() + 1e-8
    loss = scene_content.loss / loss_norm
    loss.backward(retain_graph=retain_graph)

def updateParameters(
    scene_content  : scene_content.SceneContent,
    optimizer_keys : list,
    scheduler_keys : list,
):
    with torch.no_grad():
        for key in optimizer_keys:
            if type(scene_content.parameters[key]) is list:
                for i in range(len(scene_content.parameters[key])):
                    applyGradient(scene_content.parameters[key][i], scene_content.optimizer[key][i], scene_content.clipper[key][i])
            else:
                applyGradient(scene_content.parameters[key], scene_content.optimizer[key], scene_content.clipper[key])
            
        for key in scene_content.optimizer.keys():
            if type(scene_content.optimizer[key]) is list:
                for i in range(len(scene_content.optimizer[key])):
                    scene_content.optimizer[key][i].zero_grad()
            else:
                scene_content.optimizer[key].zero_grad()

        for key in scheduler_keys:
            scene_content.scheduler[key].step(scene_content.loss)


def clampOptimization(scene_content : scene_content.SceneContent):
    with torch.no_grad():
        scene_content.damping_factor[:]           = torch.clamp(scene_content.damping_factor          , min= 0.0, max= 1.0)
        scene_content.log_stretching_stiffness[:] = torch.clamp(scene_content.log_stretching_stiffness, min= 1.0, max= 3.0)
        scene_content.log_shearing_stiffness[:]   = torch.clamp(scene_content.log_shearing_stiffness  , min=-4.0, max=-2.0)
        scene_content.log_bending_stiffness[:]    = torch.clamp(scene_content.log_bending_stiffness   , min=-5.0, max=-2.0)

        scene_content.diffuse[:]   = torch.clamp(scene_content.diffuse  , min=0.0, max=1.0)
        scene_content.metallic[:]  = torch.clamp(scene_content.metallic , min=0.0, max=1.0)
        scene_content.roughness[:] = torch.clamp(scene_content.roughness, min=0.001, max=1.0)
        scene_content.environment_map_tensor[:] = torch.clamp(scene_content.environment_map_tensor, min=0.01, max=1e10)


def updateClothParameters(
    scene_content : scene_content.SceneContent,
):
    scene_content.cloth_1.stretching_segments.stiffnesses = torch.repeat_interleave(10**scene_content.log_stretching_stiffness, scene_content.cloth_1.stretching_segments.stiffnesses.shape[0], dim=0)
    scene_content.cloth_1.shearing_segments.stiffnesses   = torch.repeat_interleave(10**scene_content.log_shearing_stiffness  , scene_content.cloth_1.shearing_segments.stiffnesses.shape[0]  , dim=0)
    scene_content.cloth_1.bending_segments.stiffnesses    = torch.repeat_interleave(10**scene_content.log_bending_stiffness   , scene_content.cloth_1.bending_segments.stiffnesses.shape[0]   , dim=0)
