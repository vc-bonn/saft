import numpy as np
import torch
from PIL import Image
import json

import rendering
import utils.mesh_utils

def loadJson(file_name):
    with open(file_name) as json_file:
        dictionary = json.load(json_file)
        for key in ["camera_positions", "camera_directions", "camera_ups", "optical_centers", "focal_lengths", "image_size", "lower_left_corners", "upper_right_corners"]:
            dictionary[key] = np.array(dictionary[key])
    return dictionary

def loadGroundTruthImages(
    image_files : np.ndarray,
    mask_files  : np.ndarray,
) -> torch.Tensor:
    assert len(image_files) == len(mask_files), f"number of cameras does not match for rgb ({len(image_files)}) and mask images ({len(mask_files)})"

    n_cameras = len(image_files)
    ground_truth_collection = []

    for i_camera in range(n_cameras):
        n_images = len(image_files[i_camera])
        dummy = np.array(Image.open(image_files[i_camera][0]), dtype = np.float32)
        ground_truth_images = torch.zeros((n_images, dummy.shape[0], dummy.shape[1], 4), dtype=torch.float32).to("cuda:0")

        for i_image in range(n_images):
            image = torch.from_numpy(np.array(Image.open(image_files[i_camera][i_image]), dtype=np.float32)).to("cuda:0")
            ground_truth_images[i_image,:,:,:3] = image[..., :3]

            mask = torch.from_numpy(np.array(Image.open(mask_files[i_camera][i_image]), dtype=np.float32)).to("cuda:0")
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            ground_truth_images[i_image,:,:,3] = torch.mean(mask[..., :3], dim=-1)

        ground_truth_images = torch.flip(ground_truth_images, dims=[-3]) / 255.0
        ground_truth_collection.append(ground_truth_images)

    return ground_truth_collection

def loadGroundTruthDepths(
    depth_files : np.ndarray,
) -> torch.Tensor:
    n_cameras = len(depth_files)
    ground_truth_depth_collection = []

    for i_camera in range(n_cameras):
        n_depths = len(depth_files[i_camera])
        dummy = torch.load(depth_files[i_camera][0])
        ground_truth_depths = torch.zeros((n_depths, dummy.shape[0], dummy.shape[1]), dtype=torch.float32, device="cuda:0")

        for i_depth in range(n_depths):
            depth = torch.load(depth_files[i_camera][i_depth])
            ground_truth_depths[i_depth] = depth

        ground_truth_depths = torch.flip(ground_truth_depths, dims=[-2])
        ground_truth_depth_collection.append(ground_truth_depths)

    return ground_truth_depth_collection

def loadBackgroundImage(
    background_file : str,
) -> torch.Tensor:
    background = torch.from_numpy(np.array(Image.open(background_file), dtype = np.float32)).to("cuda:0")
    background = torch.rot90(torch.flip(background[..., :3], dims = [1]), k=2, dims=(0, 1)) / 255.0
    
    return background


def saveImage(
    file_name   : str,
    image       : torch.Tensor,
    transparent : bool = True,
) -> None:
    save_image = image.detach().clone()
    if not transparent and save_image.shape[-1] == 4:
        save_image[..., -1] = 1.0
    save_image = rendering.torchImageToNumpy(save_image)
    Image.fromarray(save_image).save(file_name)

def saveMask(
    file_name   : str,
    mask        : torch.Tensor,
    transparent : bool = True,
) -> None:
    save_mask = torch.repeat_interleave(mask.detach()[..., None], 4, dim = -1)
    if not transparent:
        save_mask[..., -1] = 1.0
    save_mask = rendering.torchImageToNumpy(save_mask)
    Image.fromarray(save_mask).save(file_name)

def saveAllImages(
    folder             : str,
    file_prefix        : str,
    frame_number       : int,
    image              : torch.Tensor,
    blurred_image      : torch.Tensor,
    ground_truth_image : torch.Tensor,
    transparent        : bool = True,
) -> None:
    print("Save images")

    saveImage(folder + file_prefix + "_rgb_" + frame_number + ".png", image, transparent)
    saveImage(folder + file_prefix + "_rgb_difference_" + frame_number + ".png", image - ground_truth_image, transparent)
    saveMask(folder + file_prefix + "_mask_" + frame_number + ".png", image[..., 3], transparent)
    saveMask(folder + file_prefix + "_mask_difference_" + frame_number + ".png", blurred_image[..., 3] - ground_truth_image[..., 3], transparent)


def saveStates(
    file_name : str,
    n_points  : int,
    positions : torch.Tensor,
    faces     : torch.Tensor,
) -> None:
    points = utils.mesh_utils.sampleMesh(n_points, positions, faces)
    torch.save(points, file_name)

def saveStatesLike(
    save_name : str,
    load_name : str,
    positions : torch.Tensor,
    faces     : torch.Tensor,
) -> None:
    load_data = torch.load(load_name)
    saveStates(save_name, load_data.shape[0], positions, faces)




