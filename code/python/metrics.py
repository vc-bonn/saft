import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import torch
import pathlib
from PIL import Image
import openmesh as om
import time

import evaluation.points_to_triangles
import evaluation.chamfer_distance
import pytorch3d.loss


print(f"Scene        L1 Chamfer   squared Chamfer   L1 p2s     squared p2s")

mean_chamfer         = 0.0
mean_squared_chamfer = 0.0
mean_p2s             = 0.0
mean_squared_p2s     = 0.0

# scenes = ["R1/2460", "R2/2410", "R3/2400", "R4/2390", "R5/2460", "R6/2410", "R7/2405", "R8/2400", "R9/2390", "SR1/2460", "SR2/2410", "SR3/2400", "SR4/2390", "SR5/2460"]
scenes = ["R1/2460", "R2/2410", "R3/2400", "R4/2390", "R5/2460", "R6/2410", "R7/2405", "R8/2400", "R9/2390"]
for scene in scenes:
    point_cloud_gt_list = [str(file) for file in sorted(pathlib.Path(f"data/{scene[:2]}/point_clouds/").rglob("*.pt"))]

    motion = torch.load(f"results/{scene}/motion.pt", weights_only=True)
    faces  = torch.load(f"results/{scene}/faces.pt", weights_only=True)

    chamfer         = 0.0
    chamfer_squared = 0.0
    p2s             = 0.0
    p2s_squared     = 0.0

    n_frames = len(point_cloud_gt_list)
    for frame in range(n_frames):
        point_cloud_gt = torch.load(point_cloud_gt_list[frame], weights_only=True).to("cuda:0")
        if scene[:2] in ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"]:
            point_cloud_gt[..., [0, 1]] *= -1.0

        point_cloud = evaluation.chamfer_distance.sampleMesh(number=point_cloud_gt.shape[0], vertices=motion[frame], triangles=faces)

        chamfer         += evaluation.chamfer_distance.ChamferDistanceKD(point_cloud_1=point_cloud_gt, point_cloud_2=point_cloud, squared=False).item()
        chamfer_squared += evaluation.chamfer_distance.ChamferDistanceKD(point_cloud_1=point_cloud_gt, point_cloud_2=point_cloud, squared=True).item()
        p2s             += torch.mean(evaluation.points_to_triangles.pointsToSurfaceDistance(points=point_cloud_gt, face_coordinates=motion[frame], face_indices=faces, group_size=1e7, squared=False))
        p2s_squared     += torch.mean(evaluation.points_to_triangles.pointsToSurfaceDistance(points=point_cloud_gt, face_coordinates=motion[frame], face_indices=faces, group_size=1e7, squared=True))


    chamfer         /= n_frames
    chamfer_squared /= n_frames
    p2s             /= n_frames
    p2s_squared     /= n_frames

    mean_chamfer         += chamfer
    mean_squared_chamfer += chamfer_squared
    mean_p2s             += p2s
    mean_squared_p2s     += p2s_squared

    print(f"{scene:10s}    {chamfer:.3e}      {chamfer_squared:.3e}     {p2s:.3e}    {p2s_squared:.3e}")

mean_chamfer         /= len(scenes)
mean_squared_chamfer /= len(scenes)
mean_p2s             /= len(scenes)
mean_squared_p2s     /= len(scenes)

print(f"Mean          {mean_chamfer:.3e}      {mean_squared_chamfer:.3e}     {mean_p2s:.3e}    {mean_squared_p2s:.3e}")






thresholds = np.array([
    [1.10, 1.48],
    [0.85, 1.35],
    [0.80, 1.20],
    [1.00, 1.60],
    [1.00, 1.70],
    [1.10, 1.40],
    [1.10, 1.65],
    [1.10, 1.50],
    [1.10, 1.50]
])
borders = np.array([
    [100, 300,  800, 1100],
    [100, 250,  900, 1050],
    [ 50, 450, 1000, 1300],
    [ 50, 300,  950, 1350],
    [  0, 200,  720, 1000],
    [  0, 350, 1000, 1350],
    [100, 350, 1000, 1350],
    [100, 400, 1000, 1400],
    [100, 400, 1000, 1400]
])
sizes = np.array([
    [1080, 1920],
    [1080, 1920],
    [1080, 1920],
    [1080, 1920],
    [ 720, 1280],
    [1080, 1920],
    [1080, 1920],
    [1080, 1920],
    [1080, 1920]
])

depth_mean = 0.0

scenes = ["R1/2460", "R2/2410", "R3/2400", "R4/2390", "R5/2460", "R6/2410", "R7/2405", "R8/2400", "R9/2390"]
for scene in scenes:
    ground_truth_list = [str(file) for file in sorted(pathlib.Path(f"data/{scene.split('/')[0]}/depths/camera_000/").rglob("*.pt"))]
    saft_list         = [str(file) for file in sorted(pathlib.Path(f"results/{scene}/depths/camera_000/").rglob("*.pt"))]

    ground_truth_masks = [str(file) for file in sorted(pathlib.Path(f"data/{scene.split('/')[0]}/masks/camera_000/").rglob("*.png"))]
    saft_masks         = [str(file) for file in sorted(pathlib.Path(f"results/{scene}/masks/camera_000/").rglob("*.png"))]

    depth_metric_masked = 0

    pathlib.Path(f"results/shaded/{scene}/depths/camera_000/differences/").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"results/shaded/{scene}/depths/camera_000/masked_differences/").mkdir(parents=True, exist_ok=True)
    
    for i in range(0, len(ground_truth_list)):
        ground_truth_depth = torch.load(ground_truth_list[i]).detach().cpu().numpy()
        ground_truth_depth = ground_truth_depth[sizes[int(scene[1])-1, 0]-borders[int(scene[1])-1, 2]:sizes[int(scene[1])-1, 0]-borders[int(scene[1])-1, 0], borders[int(scene[1])-1, 1]:borders[int(scene[1])-1, 3]]
        saft_depth         = torch.load(saft_list[i]).detach().cpu().numpy()
        
        ground_truth_mask = np.array(Image.open(ground_truth_masks[i]))
        ground_truth_mask = ground_truth_mask[sizes[int(scene[1])-1, 0]-borders[int(scene[1])-1, 2]:sizes[int(scene[1])-1, 0]-borders[int(scene[1])-1, 0], borders[int(scene[1])-1, 1]:borders[int(scene[1])-1, 3]]
        ground_truth_mask = ground_truth_mask.astype(np.bool_)
        saft_mask         = np.array(Image.open(saft_masks[i]))
        saft_mask         = np.linalg.norm(saft_mask, axis=-1)
        saft_mask         = saft_mask.astype(np.bool_)

        difference_depth = saft_depth - ground_truth_depth
        mask_overlap = (ground_truth_depth > thresholds[int(scene[1])-1,0]) & (saft_depth > thresholds[int(scene[1])-1,0])
        
        difference_overlap = difference_depth * mask_overlap
        depth_metric_masked += np.sum(np.abs(difference_overlap)) / np.sum(mask_overlap)

    depth_metric_masked /= len(ground_truth_list)
    depth_mean += depth_metric_masked

    print(f"{scene:8s}   {depth_metric_masked: .2e}")

depth_mean /= len(scenes)

print(f"Mean       {depth_mean: .2e}")
