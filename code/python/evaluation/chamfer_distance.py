import numpy as np
import torch
import pytorch3d.loss
from scipy.spatial import KDTree


# def naiveChamferDistance(
#     point_cloud_1 : torch.Tensor,
#     point_cloud_2 : torch.Tensor,
#     squared = False
# ):
#     pairwise_differences = point_cloud_1[..., None, :, :] - point_cloud_2[..., None, :]
#     pairwise_distances = torch.norm(pairwise_differences, dim=-1)
#     if squared:
#         pairwise_distances = pairwise_distances * pairwise_distances

#     chamfer_distance_12, indices = torch.min(pairwise_distances, dim=-1)
#     chamfer_distance_21, indices = torch.min(pairwise_distances, dim=-2)

#     chamfer_distance = torch.mean(chamfer_distance_12) + torch.mean(chamfer_distance_21)
#     return chamfer_distance


def ChamferDistanceKD(
    point_cloud_1 : torch.Tensor,
    point_cloud_2 : torch.Tensor,
    squared = False
):
    kd_tree_1 = KDTree(point_cloud_1.detach().cpu().numpy())
    kd_tree_2 = KDTree(point_cloud_2.detach().cpu().numpy())

    chamfer_distance_12, _ = kd_tree_1.query(point_cloud_2.detach().cpu().numpy())
    chamfer_distance_21, _ = kd_tree_2.query(point_cloud_1.detach().cpu().numpy())

    if squared:
        chamfer_distance_12 = chamfer_distance_12 * chamfer_distance_12
        chamfer_distance_21 = chamfer_distance_21 * chamfer_distance_21

    chamfer_distance = np.mean(chamfer_distance_12) + np.mean(chamfer_distance_21)
    return chamfer_distance


def sampleMesh(number, vertices, triangles):
    triangle_vertices = vertices[triangles]
    edges = triangle_vertices[:, 1:] - triangle_vertices[:, 0].unsqueeze(1)
    areas = torch.linalg.norm(torch.linalg.cross(edges[:, 0], edges[:, 1], dim = 1), dim = 1) # double of areas
    areas = torch.clip(areas, min=1e-8)

    triangle_samples = torch.multinomial(input=areas, num_samples=number, replacement=True)

    uv = torch.rand([number, 2], device = "cuda")
    u = 1 - torch.sqrt(uv[:, 0])
    v = torch.sqrt(uv[:, 0]) * (1 - uv[:, 1])
    w = uv[:, 1] * torch.sqrt(uv[:, 0])
    uvw = torch.stack([u, v, w], dim=1).unsqueeze_(2)

    points = torch.sum(triangle_vertices[triangle_samples] * uvw, dim = 1)
    return points


def computeChamferDistance(point_cloud_1, point_cloud_2):
    chamfer_distances = pytorch3d.loss.chamfer_distance(
                            point_cloud_1[None],
                            point_cloud_2[None],
                            batch_reduction=None,
                        )[0]

    return chamfer_distances

