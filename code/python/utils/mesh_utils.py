import torch
import openmesh as om

def loadTriMesh(
    triangle_mesh : om.TriMesh,
    device        : str,
) -> tuple:
    vertices = torch.zeros([len(triangle_mesh.vertices()), 3], device=device)
    edges = torch.zeros([len(triangle_mesh.edges()), 2], dtype = torch.int32, device=device)
    faces = torch.zeros([len(triangle_mesh.faces()), 3], dtype = torch.int32, device=device)
    uv = torch.zeros([len(triangle_mesh.vertices()), 2], device=device)

    vertex_i = 0
    for vertex in triangle_mesh.vertices():
        vertices[vertex_i] = torch.from_numpy(triangle_mesh.point(vertex))
        uv[vertex_i] = torch.from_numpy(triangle_mesh.texcoord2D(vertex))
        vertex_i += 1

    edge_i = 0
    for vertex in triangle_mesh.vertices():
        for neighbor in triangle_mesh.vv(vertex):
            if neighbor.idx() > vertex.idx():
                edges[edge_i] = torch.tensor([vertex.idx(), neighbor.idx()], dtype = torch.int32, device=device)
                edge_i += 1

    face_i = 0
    for face in triangle_mesh.faces():
        vertex_i = 0
        for vh in triangle_mesh.fv(face):
            faces[face_i, vertex_i] = vh.idx()
            vertex_i += 1
        face_i += 1

    return vertices, edges, faces, uv


def loadPolyMesh(
    triangle_mesh : om.TriMesh,
    polygon_mesh  : om.PolyMesh,
    device        : str,
    data_type,
) -> tuple:
    vertices = torch.zeros([len(polygon_mesh.vertices()), 3], device=device, dtype=data_type)
    edges = torch.zeros([len(polygon_mesh.edges()), 2], device=device, dtype = torch.int32)
    faces = torch.zeros([len(triangle_mesh.faces()), 3], device=device, dtype = torch.int32)
    uv = torch.zeros([len(polygon_mesh.vertices()), 2], device=device, dtype=data_type)

    vertex_i = 0
    for vertex in polygon_mesh.vertices():
        vertices[vertex_i] = torch.from_numpy(polygon_mesh.point(vertex))
        uv[vertex_i] = torch.from_numpy(polygon_mesh.texcoord2D(vertex))
        vertex_i += 1

    edge_i = 0
    for vertex in polygon_mesh.vertices():
        for neighbor in polygon_mesh.vv(vertex):
            if neighbor.idx() > vertex.idx():
                edges[edge_i] = torch.tensor([vertex.idx(), neighbor.idx()], dtype = torch.int32, device=device)
                edge_i += 1

    face_i = 0
    for face in triangle_mesh.faces():
        vertex_i = 0
        for vh in triangle_mesh.fv(face):
            faces[face_i, vertex_i] = vh.idx()
            vertex_i += 1
        face_i += 1

    return vertices, edges, faces, uv

def edges_from_faces(faces : torch.Tensor):
    face_edges = torch.concatenate((faces[..., [0, 1]], faces[..., [1, 2]], faces[..., [2, 0]]), dim=-2)
    edges = torch.sort(face_edges, dim=-1)[0]
    edges = torch.unique(edges, sorted=True, return_inverse=False, dim=-2)

    return edges

def computeNormals(
    positions : torch.Tensor,
    faces     : torch.Tensor,
) -> torch.Tensor:
    face_positions = positions[faces]

    edges_1 = face_positions[:, 1] - face_positions[:, 0]
    edges_2 = face_positions[:, 2] - face_positions[:, 0]

    face_normals = torch.linalg.cross(edges_1, edges_2, dim=1)
    face_normals = torch.nn.functional.normalize(face_normals, dim=1)
    face_normals = torch.repeat_interleave(face_normals, faces.shape[-1], dim=0)

    vertex_normals = torch.zeros_like(positions)
    vertex_normals = torch.index_add(vertex_normals, index=faces.flatten(), source=face_normals, dim=0)
    vertex_normals = torch.nn.functional.normalize(vertex_normals, dim=1)

    return vertex_normals

def computeFaceAreas(
    positions : torch.Tensor,
    faces     : torch.Tensor,
) -> torch.Tensor:
    face_positions = positions[faces]

    edges_1 = face_positions[:, 1] - face_positions[:, 0]
    edges_2 = face_positions[:, 2] - face_positions[:, 0]

    face_normals = torch.linalg.cross(edges_1, edges_2, dim=1)
    return 0.5 * torch.linalg.norm(face_normals, dim=-1)

def sampleMesh(
    n_points  : int,
    vertices  : torch.Tensor,
    triangles : torch.Tensor,
) -> torch.Tensor:
    triangle_vertices = vertices[triangles]
    edges = triangle_vertices[:, 1:] - triangle_vertices[:, 0].unsqueeze(1)
    areas = torch.linalg.norm(torch.linalg.cross(edges[:, 0], edges[:, 1], dim = 1), dim = 1) # double of areas
    triangle_samples = torch.multinomial(input = areas, num_samples = n_points, replacement = True)

    uv = torch.rand([n_points, 2], device = "cuda:0")
    u = 1 - torch.sqrt(uv[:, 0])
    v = torch.sqrt(uv[:, 0]) * (1 - uv[:, 1])
    w = uv[:, 1] * torch.sqrt(uv[:, 0])
    uvw = torch.stack([u, v, w], dim = 1).unsqueeze_(2)

    points =  torch.sum(triangle_vertices[triangle_samples] * uvw, dim = 1)
    return points