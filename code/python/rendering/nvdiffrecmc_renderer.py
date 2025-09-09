import torch
import sys
import os

# add import path for nvdiffrecmc.denoiser.denoiser.py
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../nvdiffrecmc'))
sys.path.insert(0, parent_dir)

import nvdiffrecmc.render.render
import nvdiffrecmc.render.light as light
import nvdiffrecmc.denoiser.denoiser
import nvdiffrecmc.render.mesh
import nvdiffrecmc.render.texture


def create_mesh(
    verts: torch.Tensor,
    faces: torch.Tensor,
    uvs: torch.Tensor,
    uv_idx: torch.Tensor,
):
    mesh = nvdiffrecmc.render.mesh.Mesh(v_pos=verts, v_tex=uvs, t_pos_idx=faces, t_tex_idx=uv_idx)
    mesh = nvdiffrecmc.render.mesh.auto_normals(mesh)
    mesh = nvdiffrecmc.render.mesh.compute_tangents(mesh)
    return mesh


def create_material(
    diffuse: torch.Tensor,
    metallic: torch.Tensor,
    roughness: torch.Tensor,
    normal: torch.Tensor,# | None,
) -> dict:
    if metallic.ndim == 2 and metallic.shape[-1] != 1:
        metallic = metallic[None]
    if roughness.ndim == 2 and roughness.shape[-1] != 1:
        roughness = metallic[None]

    material = {
        "name": "_default_mat",
        "bsdf": "pbr",
        "kd": nvdiffrecmc.render.texture.Texture2D(diffuse),
        "ks": nvdiffrecmc.render.texture.Texture2D(
            torch.cat((torch.zeros_like(metallic), roughness, metallic), dim=-1)
        ),
    }
    if normal is None:
        material["no_perturbed_nrm"] = 1
    return material


class Render(torch.nn.Module):
    def __init__(self, args: dict) -> None:
        super().__init__()
        self.args = args

        self.denoiser = (
            nvdiffrecmc.denoiser.denoiser.BilateralDenoiser().cuda() if self.args.denoiser else None
        )

    def forward(
        self,
        diffrast_context,
        optix_context,
        verts: torch.Tensor,  # v, 3
        faces: torch.Tensor,  # f,3
        uvs: torch.Tensor,  # u,3
        uv_idx: torch.Tensor,  # f,3
        diffuse: torch.Tensor,
        metallic: torch.Tensor,
        roughness: torch.Tensor,
        normal: torch.Tensor,# | None,
        light: light.EnvironmentLight,
        transforms: torch.Tensor,  # n,4,4
        cam_pos: torch.Tensor,  # n,3
    ) -> torch.Tensor:

        mesh = create_mesh(verts, faces, uvs, uv_idx)
        mesh.material = create_material(diffuse, metallic, roughness, normal)

        buffers = nvdiffrecmc.render.render.render_mesh(
            self.args,
            diffrast_context,
            mesh,
            transforms,
            cam_pos,
            light,
            self.args.resolution,
            spp=self.args.spp,
            num_layers=self.args.layers,
            background=self.args.background,
            optix_ctx=optix_context,
            denoiser=self.denoiser,
        )
        return buffers
