import numpy as np
import torch


def _pointToBarycentric(
    points: torch.Tensor,
    triangles: torch.Tensor,
    eps : float = 1e-5,
) -> torch.Tensor:
    """
    Computes the barycentric coordinates of points wrt triangles.
    Note that points needs to live in the space spanned by triangles, i.e. by taking the projection of points on the space spanned by triangles.

    Args:
        points: FloatTensor of shape (n_points, 3)
        triangles: FloatTensor of shape (n_triangles, 3, 3)
    Returns:
        barycentric_coordinates: FloatTensor of shape (n_points, n_triangles, 3)
    """
    v_0, v_1, v_2 = triangles.unbind(-2)

    e_01 = v_1 - v_0
    e_02 = v_2 - v_0
    e_0p = points - v_0

    d_11 = torch.sum(e_01 * e_01, dim=-1)
    d_12 = torch.sum(e_01 * e_02, dim=-1)
    d_1p = torch.sum(e_01 * e_0p, dim=-1)
    d_22 = torch.sum(e_02 * e_02, dim=-1)
    d_2p = torch.sum(e_02 * e_0p, dim=-1)

    denominator = d_11 * d_22 - d_12 * d_12 + eps
    v = (d_22 * d_1p - d_12 * d_2p) / denominator
    w = (d_11 * d_2p - d_12 * d_1p) / denominator
    u = 1.0 - v - w

    barycentric_coordinates = torch.stack([u, v, w], dim=-1)
    return barycentric_coordinates

def _isInsideTriangle(
    points: torch.Tensor,
    triangles: torch.Tensor,
    epsilon : float = 1e-5,
) -> torch.Tensor:
    """
    Computes whether points is inside triangles.
    Note that points needs to live in the space spanned by triangles, i.e. by taking the projection of points on the space spanned by triangles.

    Args:
        points: FloatTensor of shape (n_points, 3)
        triangles: FloatTensor of shape (n_triangles, 3, 3)
    Returns:
        is_inside: BoolTensor of shape (n_points, n_triangles)
    """
    v_0 = triangles[..., 1, :] - triangles[..., 0, :]
    v_1 = triangles[..., 2, :] - triangles[..., 0, :]
    areas = 0.5 * torch.norm(torch.cross(v_0, v_1, dim=-1), dim=-1)

    is_inside = torch.ones(points.shape[:-1], device=points.device, dtype=torch.bool) # (n_points, n_edges)

    # check if triangle is a line or a point. In that case, return False
    is_inside[:, areas < epsilon] = False

    barycentric_coordinates = _pointToBarycentric(points, triangles)
    is_inside = torch.all((barycentric_coordinates >= 0.0) & (barycentric_coordinates <= 1.0), dim=-1)
    return is_inside

def _pointsToEdgesDistance(
    points  : torch.Tensor,
    edges   : torch.Tensor,
    epsilon : float = 1e-5,
    squared : bool = False,
) -> torch.Tensor:
    """
    Computes the euclidean distance of points to edges
    Args:
        points: FloatTensor of shape (n_points, 3)
        edges: FloatTensor of shape (n_edges, 2, 3)
    Returns:
        distances: FloatTensor of shape (n_points, n_edges)

    If a, b are the start and end points of the segments, we parametrize a point p as
        x(t) = a + t * (b - a)     with t in [0, 1]
    To find t which describes p we minimize (x(t) - p)^2.
    Note that p does not need to live in the space spanned by (a, b).
    """
    v_0, v_1 = edges.unbind(-2)

    d_01 = v_1 - v_0
    norm_d_01 = torch.sum(d_01 * d_01, dim=-1)

    distances = torch.zeros((*points.shape[:-1], edges.shape[-3]), device=points.device, dtype=points.dtype) # (n_points, n_edges)
    singular_edges = norm_d_01 < epsilon
    distances[:, singular_edges] = ( 0.5 * torch.sum((points[..., None, :] - v_0[singular_edges]) * (points[..., None, :] - v_0[singular_edges]), dim=-1)
                                   + 0.5 * torch.sum((points[..., None, :] - v_1[singular_edges]) * (points[..., None, :] - v_1[singular_edges]), dim=-1))

    t = torch.sum(d_01[~singular_edges] * (points[..., None, :] - v_0[~singular_edges]), dim=-1) / norm_d_01[~singular_edges]
    t = torch.clamp(t, min=0.0, max=1.0)
    closest_points = v_0[~singular_edges] + t[..., None] * d_01[~singular_edges]
    distances[:, ~singular_edges] = torch.sum((closest_points - points[..., None, :]) * (closest_points - points[..., None, :]), dim=-1)

    if not squared:
        distances = torch.sqrt(distances)

    return distances

def _pointsToTrianglesDistance(
    points            : torch.Tensor,
    triangle_vertices : torch.Tensor,
    squared           : bool = False,
) -> torch.Tensor:
    """
    Computes the euclidean distance between all combinations of points to triangles.
    Args:
        points: FloatTensor of shape (n_points, 3)
        triangles: FloatTensor of shape (n_triangles, 3, 3)
    Returns:
        distances: FloatTensor of shape (n_points, n_triangles)
    """
    v_0, v_1, v_2 = triangle_vertices.unbind(-2)
    normals = torch.cross(v_1 - v_0, v_2 - v_0, dim=-1)
    normals = torch.nn.functional.normalize(normals, dim=-1)

    normal_distances = torch.abs(torch.sum(normals * (points[..., None, :] - v_0[..., None, :, :]), dim=-1)) # (n_points, n_triangles)
    if squared:
        normal_distances = normal_distances*normal_distances
    projected_points = points[..., None, :] + normal_distances[..., None] * normals

    with torch.no_grad():
        inside_triangle = _isInsideTriangle(projected_points, triangle_vertices)

    normal_distances[~inside_triangle] = torch.inf

    # Compute the distance to all edge segments
    e01_dist = _pointsToEdgesDistance(points, triangle_vertices[..., [0, 1], :], squared=squared)
    e02_dist = _pointsToEdgesDistance(points, triangle_vertices[..., [0, 2], :], squared=squared)
    e12_dist = _pointsToEdgesDistance(points, triangle_vertices[..., [1, 2], :], squared=squared)

    distances = torch.stack((normal_distances, e01_dist, e02_dist, e12_dist), dim=-1)
    distances, indices = torch.min(distances, dim=-1)

    return distances

def pointsToTrianglesDistance(
    points            : torch.Tensor,
    face_coordinates  : torch.Tensor = None,
    face_indices      : torch.Tensor = None,
    triangle_vertices : torch.Tensor = None,
    group_size        : int = -1,
    squared           : bool = False,
) -> torch.Tensor:
    if (face_coordinates is None or face_indices is None) and triangle_vertices is None:
        raise ValueError("Please use either 'face_coordinates' and 'face_indices' or 'triangle_vertices' as an input")
    elif face_coordinates is not None and face_indices is not None and triangle_vertices is None:
        triangle_vertices = face_coordinates[face_indices]

    distances = torch.zeros((points.shape[0], triangle_vertices.shape[0]), device=points.device)
    n_groups = int(np.ceil(distances.numel() / max(group_size, points.shape[0])))

    for i in range(n_groups):
        distances[:, i::n_groups] = _pointsToTrianglesDistance(
            points=points,
            triangle_vertices=triangle_vertices[i::n_groups],
            squared=squared,
        )

    return distances


def pointsToSurfaceDistance(
    points            : torch.Tensor,
    face_coordinates  : torch.Tensor = None,
    face_indices      : torch.Tensor = None,
    triangle_vertices : torch.Tensor = None,
    group_size        : int = -1,
    squared           : bool = False,
) -> torch.Tensor:
    pairwise_p2t = pointsToTrianglesDistance(
        points=points,
        face_coordinates=face_coordinates,
        face_indices=face_indices,
        triangle_vertices=triangle_vertices,
        group_size=group_size,
        squared=squared,
    )

    distances_1, indices = torch.min(pairwise_p2t, dim=-1)
    distances_2, indices = torch.min(pairwise_p2t, dim=-2)

    p2s_distance = torch.mean(distances_1, dim=-1) + torch.mean(distances_2, dim=-1)

    return p2s_distance
