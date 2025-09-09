import numpy as np
import torch
import time
import pathlib
import PIL.Image
import matplotlib.pyplot as plt

import nvdiffrecmc.render.util

import scene_content
import utils.texture_utils

class AuxiliaryData():
    def __init__(self) -> None:
        pass

    def initialize(
        self,
        scene_content     : scene_content.SceneContent,
    ):
        self.loss_original                 = 1
        self.stretching_stiffness_original = scene_content.log_stretching_stiffness.clone()
        self.shearing_stiffness_original   = scene_content.log_shearing_stiffness.clone()
        self.bending_stiffness_original    = scene_content.log_bending_stiffness.clone()
        self.constant_forces_original      = scene_content.constant_forces.clone()
        self.vertex_shifts_original        = 1
        self.plots = {
            "loss"              : [],
            "stretching"        : [],
            "shearing"          : [],
            "bending"           : [],
            "constant_forces_x" : [],
            "constant_forces_y" : [],
            "constant_forces_z" : [],
            "vertex_shift_loss" : [],
            "chamfer_distance"  : [],
        }
        self.plots_relative = {
            "loss"              : [],
            "stretching"        : [],
            "shearing"          : [],
            "bending"           : [],
        }

        self.time_start = time.perf_counter()
        self.time = time.perf_counter()


    def printQuantities(
        self,
        scene_content     : scene_content.SceneContent,
        chamfer_distance,
    ):
        t = time.perf_counter()
        print(f"{scene_content.epoch_counter:5d} | "
            f"{t - self.time:6.2f} s {t - self.time_start:8.2f} s | "
            f"{scene_content.loss.item() * scene_content.last_frame:.2e} "
            f"{scene_content.loss.item():.3e} "
            f"{scene_content.losses['rgb'].item():.2e} "
            f" {scene_content.losses['silhouette'].item():.2e}  "
            f"{scene_content.losses['vertex_forces'].item():.2e} "
            f"{scene_content.losses['deformation_energy'].item():.2e} | "
            f"{10**scene_content.log_stretching_stiffness.item():.2e} "
            f"{10**scene_content.log_shearing_stiffness.item():.2e} "
            f"{10**scene_content.log_bending_stiffness.item():.2e} "
            f"{(scene_content.damping_factor.item()):.2e} "
            f"{(scene_content.constant_forces[0, 0].item()): .2e} "
            f"{(scene_content.constant_forces[0, 1].item()): .2e} "
            f"{(scene_content.constant_forces[0, 2].item()): .2e} | "
            f"  {chamfer_distance.item():.2e}   "
            )
        self.time = t

        
    def logQuantities(
        self,
        scene_content    : scene_content.SceneContent,
        chamfer_distance : torch.Tensor,
    ):
        self.logAbsoluteQuantities(scene_content, chamfer_distance)
        self.logRelativeQuantities(scene_content)

    def logAbsoluteQuantities(
        self,
        scene_content : scene_content.SceneContent,
        chamfer_distance,
    ):
        self.plots["loss"].append((scene_content.loss).detach().cpu().item())
        self.plots["stretching"].append(scene_content.log_stretching_stiffness.detach().cpu().item())
        self.plots["shearing"].append(scene_content.log_shearing_stiffness.detach().cpu().item())
        self.plots["bending"].append(scene_content.log_bending_stiffness.detach().cpu().item())
        self.plots["constant_forces_x"].append(scene_content.constant_forces[0, 0].detach().cpu().item())
        self.plots["constant_forces_y"].append(scene_content.constant_forces[0, 1].detach().cpu().item())
        self.plots["constant_forces_z"].append(scene_content.constant_forces[0, 2].detach().cpu().item())
        self.plots["vertex_shift_loss"].append(chamfer_distance.cpu().item())
        self.plots["chamfer_distance"].append(chamfer_distance.item())

    def logRelativeQuantities(
        self,
        scene_content : scene_content.SceneContent,
    ):
        stretching_stiffness_relative = 10**scene_content.log_stretching_stiffness / 10**self.stretching_stiffness_original
        shearing_stiffness_relative   = 10**scene_content.log_shearing_stiffness / 10**self.shearing_stiffness_original
        bending_stiffness_relative    = 10**scene_content.log_bending_stiffness / 10**self.bending_stiffness_original
        
        if len(self.plots_relative["loss"]) == 0 and scene_content.loss.item() > 1e-5:
            self.plots_relative["loss"].append(1)
            self.loss_original = scene_content.loss.item()
        elif len(self.plots_relative["loss"]) == 0 or self.loss_original < 1e-5:
            self.plots_relative["loss"].append(scene_content.loss.detach().cpu().item())
        else:
            self.plots_relative["loss"].append((scene_content.loss.detach().cpu().item()) / self.loss_original)
        self.plots_relative["stretching"].append(stretching_stiffness_relative.detach().cpu().item())
        self.plots_relative["shearing"].append(shearing_stiffness_relative.detach().cpu().item())
        self.plots_relative["bending"].append(bending_stiffness_relative.detach().cpu().item())


def saveScreeshots(
    scene                  : str,
    directory_name         : str,
    camera_numbers         : tuple,
    frame_counter          : int,
    save_material          : bool,
    render_output          : dict,
    vertex_positions       : torch.Tensor,
    faces                  : torch.Tensor,
    uv                     : torch.Tensor,
    point_clouds           : torch.Tensor,
    point_cloud_lengths    : torch.Tensor,
    chamfer_distances      : torch.Tensor,
    texture                : torch.Tensor,
    diffuse                : torch.Tensor,
    metallic               : torch.Tensor,
    roughness              : torch.Tensor,
    normal_map             : torch.Tensor,
    environment_map_tensor : torch.Tensor,
    background_images      : torch.Tensor,
    ground_truth_images    : list,
    ground_truth_depths    : list,
    depth_image_range      : list,
):
    z_far = 1000.0
    z_near = 0.01

    render_output_flipped = {}
    for key in render_output:
        render_output_flipped[key] = torch.flip(render_output[key], dims=[-3])

    for i_camera in range(render_output_flipped["color_diffrast"].shape[0]):
        pathlib.Path(f"{directory_name}/rgbs/camera_{str(camera_numbers[i_camera]).zfill(3)}").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{directory_name}/shaded/camera_{str(camera_numbers[i_camera]).zfill(3)}").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{directory_name}/shaded_raw/camera_{str(camera_numbers[i_camera]).zfill(3)}").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{directory_name}/masks/camera_{str(camera_numbers[i_camera]).zfill(3)}").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{directory_name}/depths/camera_{str(camera_numbers[i_camera]).zfill(3)}/images").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{directory_name}/depths/camera_{str(camera_numbers[i_camera]).zfill(3)}/tensors").mkdir(parents=True, exist_ok=True)
        if ground_truth_images is not None:
            pathlib.Path(f"{directory_name}/difference_rgbs/camera_{str(camera_numbers[i_camera]).zfill(3)}").mkdir(parents=True, exist_ok=True)
            pathlib.Path(f"{directory_name}/difference_shaded/camera_{str(camera_numbers[i_camera]).zfill(3)}").mkdir(parents=True, exist_ok=True)
            pathlib.Path(f"{directory_name}/difference_masks/camera_{str(camera_numbers[i_camera]).zfill(3)}").mkdir(parents=True, exist_ok=True)
        if ground_truth_depths is not None:
            pathlib.Path(f"{directory_name}/difference_depths/camera_{str(camera_numbers[i_camera]).zfill(3)}").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{directory_name}/diffuse_texture/camera_{str(camera_numbers[i_camera]).zfill(3)}").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{directory_name}/diffuse_light/camera_{str(camera_numbers[i_camera]).zfill(3)}").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"{directory_name}/specular_light/camera_{str(camera_numbers[i_camera]).zfill(3)}").mkdir(parents=True, exist_ok=True)

        # workaround for different number of ground truth and rendering cameras
        if ground_truth_images is not None:
            ground_truth_image = torch.flip(ground_truth_images[i_camera%len(ground_truth_images)][frame_counter], dims=[-3])
        if ground_truth_depths is not None:
            ground_truth_depth = torch.flip(ground_truth_depths[i_camera%len(ground_truth_depths)][frame_counter], dims=[-2])
        
        colormap_jet = plt.get_cmap("jet", 256)
        depth = render_output_flipped["depth_diffrast"][i_camera, ..., 0]
        normalized_depth = (depth - depth_image_range[0]) / (depth_image_range[1] - depth_image_range[0])
        depth_image = colormap_jet(normalized_depth.detach().cpu().numpy())

        colormap_rdbu_r = plt.get_cmap("RdBu_r", 256)
        # depth = (2 * z_far * z_near) / ((z_far + z_near) - render_output_flipped["depth_diffrast"][i_camera, ..., 0] * (z_far - z_near))
        
        if ground_truth_depths is not None:
            normalized_depth = (depth - ground_truth_depth) / 0.1 + 0.5
            normalized_depth[(render_output_flipped["mask_diffrast"][i_camera, ..., 0] < 1.0) & (ground_truth_image[..., 3] < 1.0)] = 0.5 # set depth of background to neutral value
            depth_difference_image = colormap_rdbu_r(normalized_depth.detach().cpu().numpy())

        save_image = utils.texture_utils.floatTensorToImage(render_output_flipped["color_diffrast"][i_camera]).detach().cpu().numpy()
        save_image = PIL.Image.fromarray(save_image)
        save_image.save(f"{directory_name}/rgbs/camera_{str(camera_numbers[i_camera]).zfill(3)}/rgb_{str(frame_counter).zfill(3)}.png")

        save_image = utils.texture_utils.floatTensorToImage(torch.repeat_interleave(render_output_flipped["mask_diffrast"][i_camera], 3, dim=-1)).detach().cpu().numpy()
        save_image = PIL.Image.fromarray(save_image)
        save_image.save(f"{directory_name}/masks/camera_{str(camera_numbers[i_camera]).zfill(3)}/mask_{str(frame_counter).zfill(3)}.png")

        save_image = depth_image[..., :3]*255
        save_image = PIL.Image.fromarray(np.uint8(save_image))
        save_image.save(f"{directory_name}/depths/camera_{str(camera_numbers[i_camera]).zfill(3)}/images/depth_{str(frame_counter).zfill(3)}.png")

        torch.save(depth, f"{directory_name}/depths/camera_{str(camera_numbers[i_camera]).zfill(3)}/tensors/depth_{str(frame_counter).zfill(3)}.pt")

        if ground_truth_images is not None:
            save_image = utils.texture_utils.floatTensorToImage(torch.abs(render_output_flipped["color_diffrast"][i_camera] - ground_truth_image[..., 0:3])).detach().cpu().numpy()
            save_image = PIL.Image.fromarray(save_image)
            save_image.save(f"{directory_name}/difference_rgbs/camera_{str(camera_numbers[i_camera]).zfill(3)}/rgb_{str(frame_counter).zfill(3)}.png")

            save_image = utils.texture_utils.floatTensorToImage(torch.repeat_interleave(torch.abs(render_output_flipped["mask_diffrast"][i_camera] - ground_truth_image[..., 3:4]), 3, dim=-1)).detach().cpu().numpy()
            save_image = PIL.Image.fromarray(save_image)
            save_image.save(f"{directory_name}/difference_masks/camera_{str(camera_numbers[i_camera]).zfill(3)}/mask_{str(frame_counter).zfill(3)}.png")

        if ground_truth_depths is not None:
            save_image = depth_difference_image[..., :3]*255
            save_image = PIL.Image.fromarray(np.uint8(save_image))
            save_image.save(f"{directory_name}/difference_depths/camera_{str(camera_numbers[i_camera]).zfill(3)}/depth_{str(frame_counter).zfill(3)}.png")


        if "color_diffrec" in render_output_flipped.keys():
            save_image = utils.texture_utils.floatTensorToImage(nvdiffrecmc.render.util.rgb_to_srgb(render_output_flipped["color_diffrec"][i_camera])).detach().cpu().numpy()
            save_image = PIL.Image.fromarray(save_image)
            save_image.save(f"{directory_name}/shaded/camera_{str(camera_numbers[i_camera]).zfill(3)}/shaded_{str(frame_counter).zfill(3)}.png")

            save_image = utils.texture_utils.floatTensorToImage(render_output_flipped["color_diffrec"][i_camera]).detach().cpu().numpy()
            save_image = PIL.Image.fromarray(save_image)
            save_image.save(f"{directory_name}/shaded_raw/camera_{str(camera_numbers[i_camera]).zfill(3)}/shaded_raw_{str(frame_counter).zfill(3)}.png")

            if ground_truth_images is not None:
                save_image = utils.texture_utils.floatTensorToImage(torch.abs(render_output_flipped["color_diffrec"][i_camera] - ground_truth_image[..., 0:3])).detach().cpu().numpy()
                save_image = PIL.Image.fromarray(save_image)
                save_image.save(f"{directory_name}/difference_shaded/camera_{str(camera_numbers[i_camera]).zfill(3)}/shaded_{str(frame_counter).zfill(3)}.png")

            save_image = utils.texture_utils.floatTensorToImage(render_output_flipped["diffuse_texture_diffrec"][i_camera]).detach().cpu().numpy()
            save_image = PIL.Image.fromarray(save_image)
            save_image.save(f"{directory_name}/diffuse_texture/camera_{str(camera_numbers[i_camera]).zfill(3)}/diffuse_texture_{str(frame_counter).zfill(3)}.png")

            save_image = utils.texture_utils.floatTensorToImage(render_output_flipped["diffuse_light_diffrec"][i_camera]).detach().cpu().numpy()
            save_image = PIL.Image.fromarray(save_image)
            save_image.save(f"{directory_name}/diffuse_light/camera_{str(camera_numbers[i_camera]).zfill(3)}/diffuse_light_{str(frame_counter).zfill(3)}.png")

            save_image = utils.texture_utils.floatTensorToImage(render_output_flipped["specular_light_diffrec"][i_camera]).detach().cpu().numpy()
            save_image = PIL.Image.fromarray(save_image)
            save_image.save(f"{directory_name}/specular_light/camera_{str(camera_numbers[i_camera]).zfill(3)}/specular_light_{str(frame_counter).zfill(3)}.png")

    if point_clouds is not None and point_cloud_lengths is not None:
        pathlib.Path(f"{directory_name}/point_clouds").mkdir(parents=True, exist_ok=True)
        points = point_clouds[frame_counter, 0:point_cloud_lengths[frame_counter]].clone()
        torch.save(points, f"{directory_name}/point_clouds/point_cloud_{str(frame_counter).zfill(3)}.pt")

    if chamfer_distances is not None:
            torch.save(chamfer_distances, f"{directory_name}/chamfer_distances.pt")
            np.savetxt(f"{directory_name}/chamfer_distances.txt", chamfer_distances.detach().cpu().numpy())

    if save_material:
        torch.save(vertex_positions, f"{directory_name}/motion.pt")
        torch.save(faces, f"{directory_name}/faces.pt")
        torch.save(uv, f"{directory_name}/uv.pt")

        save_image = utils.texture_utils.floatTensorToImage(torch.flip(texture[0, ..., :3], dims=[0])).detach().cpu().numpy()
        save_image = PIL.Image.fromarray(save_image)
        save_image.save(f"{directory_name}/texture.png")

        save_image = utils.texture_utils.floatTensorToImage(nvdiffrecmc.render.util.rgb_to_srgb(torch.flip(diffuse[0, ..., :3], dims=[0]))).detach().cpu().numpy()
        save_image = PIL.Image.fromarray(save_image)
        save_image.save(f"{directory_name}/diffuse.png")

        save_image = utils.texture_utils.floatTensorToImage(torch.repeat_interleave(torch.flip(metallic[0], dims=[0]), repeats=3, dim=-1)).detach().cpu().numpy()
        save_image = PIL.Image.fromarray(save_image)
        save_image.save(f"{directory_name}/metallic.png")

        save_image = utils.texture_utils.floatTensorToImage(torch.repeat_interleave(torch.flip(roughness[0], dims=[0]), repeats=3, dim=-1)).detach().cpu().numpy()
        save_image = PIL.Image.fromarray(save_image)
        save_image.save(f"{directory_name}/roughness.png")

        save_image = utils.texture_utils.floatTensorToImage(nvdiffrecmc.render.util.rgb_to_srgb(torch.flip(environment_map_tensor, dims=[0]))).detach().cpu().numpy()
        save_image = PIL.Image.fromarray(save_image)
        save_image.save(f"{directory_name}/environment_map_1.png")

        save_image = utils.texture_utils.floatTensorToImage(nvdiffrecmc.render.util.rgb_to_srgb(torch.flip(0.5 * environment_map_tensor, dims=[0]))).detach().cpu().numpy()
        save_image = PIL.Image.fromarray(save_image)
        save_image.save(f"{directory_name}/environment_map_0_5.png")

        save_image = utils.texture_utils.floatTensorToImage(nvdiffrecmc.render.util.rgb_to_srgb(torch.flip(0.25 * environment_map_tensor, dims=[0]))).detach().cpu().numpy()
        save_image = PIL.Image.fromarray(save_image)
        save_image.save(f"{directory_name}/environment_map_0_25.png")

        torch.save(environment_map_tensor, f"{directory_name}/environment_map.pt")

