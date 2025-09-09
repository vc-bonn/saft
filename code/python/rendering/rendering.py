import torch
import nvdiffrast.torch as dr
import matplotlib.pyplot as plt

import rendering.nvdiffrecmc_renderer
import nvdiffrecmc.render.light
import nvdiffrecmc.render.optixutils
import nvdiffrecmc.render.util

import utils.texture_utils
import scene_content

class Renderer():
    def __init__(self) -> None:
        pass

    def initialize(
        self,
        scene_parameters : dict,
        device           : str,
        datatype,
        args,
        optimize         : bool,
    ) -> None:
        self.rasterizer_context = dr.RasterizeCudaContext()
        # self.rasterizer_context = dr.RasterizeGLContext()
        self.optix_context = nvdiffrecmc.render.optixutils.OptiXContext()
        self.diffrec_renderer = rendering.nvdiffrecmc_renderer.Render(args)

        self.near_plane = 0.01
        self.far_plane = 1000.0

        self.camera_positions  = torch.tensor(scene_parameters["camera_positions"], device=device, dtype=datatype)
        self.camera_directions = torch.tensor(scene_parameters["camera_directions"], device=device, dtype=datatype)

        self.image_size          = torch.tensor(scene_parameters["image_size"]         , device=device, dtype=torch.int32)
        self.optical_centers     = torch.tensor(scene_parameters["optical_centers"]    , device=device, dtype=datatype, requires_grad=optimize)
        self.focal_lengths       = torch.tensor(scene_parameters["focal_lengths"]      , device=device, dtype=datatype, requires_grad=optimize)
        self.lower_left_corners  = torch.tensor(scene_parameters["lower_left_corners"] , device=device, dtype=self.image_size.dtype)
        self.upper_right_corners = torch.tensor(scene_parameters["upper_right_corners"], device=device, dtype=self.image_size.dtype)

        self.optical_center_shifts = torch.cat([self.image_size[0:1] - self.upper_right_corners[:, 0:1], self.lower_left_corners[:, 1:2]], dim=1)
        self.crop_size = self.upper_right_corners[0] - self.lower_left_corners[0]
        self.model_view_projection_matrix = getModelViewProjectionMatrix(
                                                self.camera_positions,
                                                self.camera_directions,
                                                self.optical_centers - self.optical_center_shifts,
                                                self.focal_lengths,
                                                self.near_plane,
                                                self.far_plane,
                                                self.crop_size
                                            )

        # compute ray direction for each pixel
        x_pixel = torch.arange(0, self.crop_size[1], 1, device=self.optical_centers.device).unsqueeze(0)
        y_pixel = torch.arange(0, self.crop_size[0], 1, device=self.optical_centers.device).unsqueeze(0)
        x_components = (x_pixel - (self.optical_centers - self.optical_center_shifts)[:, 1]) / self.focal_lengths[:, 1]
        y_components = (y_pixel - (self.optical_centers - self.optical_center_shifts)[:, 0]) / self.focal_lengths[:, 0]
        x_components = torch.repeat_interleave(x_components.unsqueeze(-2), self.crop_size[0], dim=-2)
        y_components = torch.repeat_interleave(y_components.unsqueeze(-1), self.crop_size[1], dim=-1)
        z_components = -torch.ones_like(x_components)
        pixel_directions = torch.stack((x_components, y_components, z_components), dim=-1)
        view_matrix = getViewMatrix(self.camera_positions, self.camera_directions)
        pixel_directions = torch.einsum("ijk,imnk->imnj", torch.linalg.inv(view_matrix[:, :3, :3]), pixel_directions)
        self.pixel_directions = torch.nn.functional.normalize(pixel_directions, dim=-1)
        
        self.render_output = {}

    def renderDiffrast(
        self,
        positions                    : torch.Tensor,
        faces                        : torch.Tensor,
        uv                           : torch.Tensor,
        normals                      : torch.Tensor,
        texture                      : torch.Tensor,
        model_view_projection_matrix : torch.Tensor,
        crop_size                    : torch.Tensor,
    ):
        vertex_masks = torch.ones([uv.shape[0], 1], device = "cuda:0")
        diffrast_attributes = {
            "uv_diffrast" : uv,
            "mask_diffrast" : vertex_masks,
            "normals_diffrast" : normals,
        }

        render_output = renderImages(
            self.rasterizer_context,
            positions,
            faces,
            diffrast_attributes,
            texture,
            model_view_projection_matrix,
            crop_size,
        )
        self.computeDepth(render_output)

        for key in render_output:
            render_output[key] = torch.flip(render_output[key], dims = [1,2])

        self.render_output = render_output

        return render_output
    
    def renderDiffrecmc(
        self,
        positions                    : torch.Tensor,
        faces                        : torch.Tensor,
        uv                           : torch.Tensor,
        normals                      : torch.Tensor,
        texture                      : torch.Tensor,
        diffuse                      : torch.Tensor,
        metallic                     : torch.Tensor,
        roughness                    : torch.Tensor,
        normal_map                   : torch.Tensor,
        environment_map              : nvdiffrecmc.render.light.EnvironmentLight,
        model_view_projection_matrix : torch.Tensor,
        camera_positions             : torch.Tensor,
        crop_size                    : torch.Tensor,
    ):
        vertex_masks = torch.ones([uv.shape[0], 1], device = "cuda:0")
        diffrast_attributes = {
            "uv_diffrast" : uv,
            "mask_diffrast" : vertex_masks,
            "normals_diffrast" : normals,
        }

        render_output_diffrast = renderImages(
            self.rasterizer_context,
            positions,
            faces,
            diffrast_attributes,
            texture,
            model_view_projection_matrix,
            crop_size,
        )
        self.computeDepth(render_output_diffrast)

        render_output_diffrecmc = self.diffrec_renderer(
            self.rasterizer_context,
            self.optix_context,
            positions,
            faces.to(torch.int64),
            uv,
            faces,
            diffuse,
            metallic,
            roughness,
            normal_map,
            environment_map,
            model_view_projection_matrix,
            camera_positions,
        )

        # merge both dictionaries
        render_output = {
            **render_output_diffrast,
            "color_diffrec"             : render_output_diffrecmc["shaded"][..., 0:3],
            "mask_diffrec"              : render_output_diffrecmc["shaded"][..., 3:4],
            "diffuse_light_diffrec"     : render_output_diffrecmc["diffuse_light"][..., 0:3],
            "specular_light_diffrec"    : render_output_diffrecmc["specular_light"][..., 0:3],
            "diffuse_texture_diffrec"   : render_output_diffrecmc["kd"][..., 0:3],
            "roughness_texture_diffrec" : render_output_diffrecmc["ks"][..., 1:2],
            "metallic_texture_diffrec"  : render_output_diffrecmc["ks"][..., 2:3],
            "normal_diffrec"            : render_output_diffrecmc["normal"][..., 0:3],
        }

        for key in render_output:
            render_output[key] = torch.flip(render_output[key], dims = [1,2])

        self.render_output = render_output

        return render_output

    def computeDepth(
        self,
        render_output : torch.Tensor,
    ):
        render_output["depth_diffrast"] = (2 * self.far_plane * self.near_plane) / ((self.far_plane + self.near_plane) - render_output["depth_diffrast"] * (self.far_plane - self.near_plane))
        render_output["depth_diffrast"] = render_output["mask_diffrast"] * render_output["depth_diffrast"]


    def setGuiImages(
        self,
        scene_content : scene_content.SceneContent,
        render_output : torch.Tensor,
        gui_images    : dict,
        use_diffrec   : bool,
    ) -> None:
        with torch.no_grad():
            colormap_jet = plt.get_cmap("jet", 256)
            colormap_rdbu_r = plt.get_cmap("RdBu_r", 256)
            gui_image = torch.ones_like(scene_content.ground_truth_collection[0][scene_content.frame_counter])
            gui_image[..., -1] = 1

            color_image = render_output["color_diffrast"][0]
            if use_diffrec:
                color_image = nvdiffrecmc.render.util.rgb_to_srgb(render_output["color_diffrec"][0])
            gui_image[..., 0:3] = color_image
            gui_images["image"][:] = utils.texture_utils.floatTensorToImage(gui_image)
            
            gui_image[..., 0:3] = torch.where(render_output["mask_diffrast"][0] == 0, 0, color_image)
            gui_images["cloth"][:] = utils.texture_utils.floatTensorToImage(gui_image)

            gui_image[..., 0:3] = torch.repeat_interleave(render_output["mask_diffrast"][0], 3, dim=2)
            gui_images["mask"][:] = utils.texture_utils.floatTensorToImage(gui_image)

            normalized_depth = ((render_output["depth_diffrast"][0, ..., 0] - scene_content.scene_parameters["depth_image_range"][0])
                                / (scene_content.scene_parameters["depth_image_range"][1] - scene_content.scene_parameters["depth_image_range"][0]))
            depth_image = colormap_jet(normalized_depth.detach().cpu().numpy())
            gui_image[..., 0:3] = torch.from_numpy(depth_image)[..., 0:3]
            gui_images["depth"][:] = utils.texture_utils.floatTensorToImage(gui_image)

            gui_image[..., 0:3] = scene_content.ground_truth_collection[0][scene_content.frame_counter, ..., 0:3]
            gui_images["ground_truth_image"][:] = utils.texture_utils.floatTensorToImage(gui_image)

            gui_image[..., 0:3] = torch.repeat_interleave(scene_content.ground_truth_collection[0][scene_content.frame_counter, ..., 3:4], 3, dim=2)
            gui_images["ground_truth_mask"][:] = utils.texture_utils.floatTensorToImage(gui_image)

            normalized_depth = ((scene_content.ground_truth_depth_collection[0][scene_content.frame_counter] - scene_content.scene_parameters["depth_image_range"][0])
                                / (scene_content.scene_parameters["depth_image_range"][1] - scene_content.scene_parameters["depth_image_range"][0]))
            depth_image = colormap_jet(normalized_depth.detach().cpu().numpy())
            gui_image[..., 0:3] = torch.from_numpy(depth_image)[..., :3]
            gui_images["ground_truth_depth"][:] = utils.texture_utils.floatTensorToImage(gui_image)

            gui_image[..., 0:3] = (color_image - scene_content.ground_truth_collection[0][scene_content.frame_counter, ..., 0:3]) * 0.5 + 0.5
            gui_images["difference_image"][:] = utils.texture_utils.floatTensorToImage(gui_image)

            gui_image[..., 0:3] = torch.repeat_interleave((render_output["mask_diffrast"][0] - scene_content.ground_truth_collection[0][scene_content.frame_counter, ..., 3:4]), 3, dim=2) * 0.5 + 0.5
            gui_images["difference_mask"][:] = utils.texture_utils.floatTensorToImage(gui_image)

            normalized_depth = 10.0 * (render_output["depth_diffrast"][0, ..., 0] - scene_content.ground_truth_depth_collection[0][scene_content.frame_counter]) * 0.5 + 0.5
            normalized_depth[(render_output["depth_diffrast"][0, ..., 0] < 1.0) & (scene_content.ground_truth_collection[0][scene_content.frame_counter, ..., 3] < 1.0)] = 0.5 # set depth of background to neutral value
            depth_image = colormap_rdbu_r(normalized_depth.detach().cpu().numpy())
            gui_image[..., 0:3] = torch.from_numpy(depth_image)[..., :3]
            gui_images["difference_depth"][:] = utils.texture_utils.floatTensorToImage(gui_image)

            if "color_diffrec" in render_output.keys():
                gui_image[..., 0:3] = render_output["color_diffrec"][0]
                gui_images["raw_image"][:] = utils.texture_utils.floatTensorToImage(gui_image)

                gui_image[..., 0:3] = render_output["diffuse_texture_diffrec"][0]
                gui_images["diffuse_texture"][:] = utils.texture_utils.floatTensorToImage(gui_image)

                gui_image[..., 0:3] = render_output["diffuse_light_diffrec"][0]
                gui_images["diffuse_light"][:] = utils.texture_utils.floatTensorToImage(gui_image)

                gui_image[..., 0:3] = render_output["specular_light_diffrec"][0]
                gui_images["specular_light"][:] = utils.texture_utils.floatTensorToImage(gui_image)

            gui_image[..., 0:3] = scene_content.background_images[0]
            gui_images["background"][:] = utils.texture_utils.floatTensorToImage(gui_image)

            gui_image = torch.ones((scene_content.texture.shape[1], scene_content.texture.shape[2], 4), device=scene_content.texture.device)
            gui_image[..., 0:3] = scene_content.texture[0, ..., 0:3]
            gui_images["texture"][:] = utils.texture_utils.floatTensorToImage(gui_image)

            gui_image = torch.ones_like(scene_content.diffuse[0])
            gui_images["diffuse"][:] = utils.texture_utils.floatTensorToImage(nvdiffrecmc.render.util.rgb_to_srgb(scene_content.diffuse[0]))
            gui_image[..., 0:3] = torch.repeat_interleave(scene_content.metallic[0], repeats=3, dim=-1)
            gui_images["metallic"][:] = utils.texture_utils.floatTensorToImage(gui_image)
            gui_image[..., 0:3] = torch.repeat_interleave(scene_content.roughness[0], repeats=3, dim=-1)
            gui_images["roughness"][:] = utils.texture_utils.floatTensorToImage(gui_image)

            gui_image = torch.ones((*scene_content.environment_map_tensor.shape[0:2], 4), device=scene_content.environment_map_tensor.device)
            gui_image[..., 0:3] = nvdiffrecmc.render.util.rgb_to_srgb(scene_content.environment_map_tensor)
            gui_images["environment_map"][:] = utils.texture_utils.floatTensorToImage(gui_image)

            if "color_diffrec" in render_output.keys():
                gui_image = torch.ones_like(scene_content.ground_truth_collection[0][scene_content.frame_counter])
                luminance = torch.mean(nvdiffrecmc.render.util.rgb_to_srgb(render_output["diffuse_light_diffrec"][0] + render_output["specular_light_diffrec"][0]), dim=-1, keepdim=True)
                regularization = torch.abs(luminance - torch.amax(scene_content.ground_truth_collection[0][scene_content.frame_counter, ..., 0:3], dim=-1, keepdim=True))
                gui_image[..., 0:3] = torch.repeat_interleave(regularization, 3, dim=-1)
                # gui_image[..., 0:3] = render_output["color_diffrast"][0]
                gui_images["test"][:] = utils.texture_utils.floatTensorToImage(gui_image)





def getViewMatrix(
    camera_positions  : torch.Tensor,
    camera_directions : torch.Tensor,
) -> torch.Tensor:
    data_type = camera_positions.dtype
    device = camera_positions.device

    camera_directions = torch.nn.functional.normalize(camera_directions, dim=-1)

    up = torch.tensor([[0.0, 1.0, 0.0]], dtype = data_type, device = device)
    right = torch.cross(camera_directions, up, dim=-1)
    right = torch.nn.functional.normalize(right, dim=-1)
    camera_up = torch.cross(right, camera_directions, dim=-1)
    camera_up = torch.nn.functional.normalize(camera_up, dim=-1)
    
    direction_matrix = torch.zeros((*camera_positions.shape[:-1], 4, 4), dtype = data_type, device = device)
    direction_matrix[..., 3, 3] = 1.0
    direction_matrix[..., :3, 0] = right
    direction_matrix[..., :3, 1] = camera_up
    direction_matrix[..., :3, 2] = - camera_directions
    
    position_matrix = torch.zeros((camera_positions.shape[0], 4, 4), dtype = data_type, device = device)
    position_matrix[..., :, :] = torch.eye(n=4, dtype = data_type, device = device)
    position_matrix[..., 3, :3] = - camera_positions
    
    return torch.transpose(position_matrix @ direction_matrix, dim0=-2, dim1=-1)

def opencvProjection(
    image_size      : torch.Tensor, # height, width
    optical_centers : torch.Tensor, # c in pixel (height, width)
    focal_lengths   : torch.Tensor, # f in pixel (height, width)
    near_plane      : float,
    far_plane       : float,
) -> torch.Tensor:
    device = optical_centers.device

    #focal_length = 0.5 * image_size / np.tan(-0.5 * np.deg2rad(fov)) # fov [deg] to focal length [pixel]
    optical_shifts = (image_size.unsqueeze(0) - 2 * optical_centers) / image_size.unsqueeze(0)
    optical_shifts[..., 1] = - optical_shifts[..., 1]
    relative_focal_length = -2 * focal_lengths / image_size.unsqueeze(0)

    projection = torch.zeros((optical_centers.shape[0], 4, 4), dtype = optical_centers.dtype, device = device)
    projection[..., 0, 0] = relative_focal_length[..., 1]
    projection[..., 1, 1] = relative_focal_length[..., 0]
    projection[..., 0, 2] = optical_shifts[..., 1]
    projection[..., 1, 2] = optical_shifts[..., 0]
    projection[..., 2, 2] = -(far_plane + near_plane) / (far_plane - near_plane)
    projection[..., 2, 3] = -2 * far_plane * near_plane / (far_plane - near_plane)
    projection[..., 3, 2] = -1.0

    return projection

def getModelViewProjectionMatrix(
    camera_positions  : torch.Tensor,
    camera_directions : torch.Tensor,
    optical_centers   : torch.Tensor,
    focal_lengths     : torch.Tensor,
    near_plane        : float,
    far_plane         : float,
    image_size        : torch.Tensor,
) -> torch.Tensor:
    view_matrix = getViewMatrix(camera_positions, camera_directions)
    projection_matrix = opencvProjection(image_size = image_size,
                                         optical_centers = optical_centers,
                                         focal_lengths = focal_lengths,
                                         near_plane = near_plane,
                                         far_plane = far_plane)
    return projection_matrix @ view_matrix


def renderImages(
    context,
    vertices          : torch.Tensor,
    triangles         : torch.Tensor,
    vertex_attributes : dict,
    texture           : torch.Tensor,
    camera_transforms : torch.Tensor,
    resolution        : torch.Tensor,
    antialias         : bool = True,
    crop              : bool = True,
) -> dict:
    device = vertices.device
    n = camera_transforms.shape[0]
    v = vertices.shape[0]

    vertices_hom = torch.cat([vertices, torch.ones([v, 1], device = device, dtype=vertices.dtype)], axis=1)
    vertices_pixel = torch.matmul(vertices_hom.expand(n, v, -1), torch.transpose(camera_transforms, -2, -1)).to(torch.float32)

    rast, diff_rast = dr.rasterize(context, vertices_pixel, triangles, resolution = resolution)

    image_attributes = {}
    for key in vertex_attributes:
        image_attributes[key], _ = dr.interpolate(vertex_attributes[key].to(torch.float32), rast, triangles, rast_db = diff_rast, diff_attrs = None)

    image_attributes["color_diffrast"] = dr.texture(texture.to(torch.float32), uv=image_attributes["uv_diffrast"], filter_mode="linear")
    image_attributes["depth_diffrast"] = rast[..., 2:3].contiguous() # depth in [-1,1]
    image_attributes["triangle_id_diffrast"] = rast[..., 3:4].contiguous()

    render_output = {}
    if antialias:
        for key in image_attributes:
            render_output[key] = dr.antialias(image_attributes[key], rast, vertices_pixel, triangles)

    if crop:
        for key in render_output:
            render_output[key] = torch.where(rast[..., 3:4] > 0, render_output[key], torch.tensor([0.0], device = "cuda:0"))
    
    return render_output



