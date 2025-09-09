import torch
import pathlib
import argparse
import json
import openmesh as om
import autoclip.torch as autoclip

import sys
import os
# add import path for nvdiffrecmc
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

import nvdiffrecmc.render.light as light
import nvdiffrecmc.render.util

import material_encoder.decoder_optimization

import utils.file_manager
import utils.mesh_utils
import utils.texture_utils
import objects.cloth

class SceneContent():
    def __init__(self) -> None:
        pass

    def initializeHeadless(
        self,
        scene_parameters,
        time_step,
        optimize,
        device,
        data_type,
    ) -> None:
        self.initializeParameters(scene_parameters, device, data_type)
        self.cloth_1 = self.createCloth(time_step, optimize, device, data_type)
        self.initializeOptimizers()
        self.loadData()

    def initializeParameters(
        self,
        scene_parameters,
        device,
        data_type,
    ):
        self.scene_parameters = scene_parameters
        self.scene_name = scene_parameters["scene"]
        self.losses = {
            "rgb" : torch.tensor([0.], dtype=data_type, device=device),
            "silhouette" : torch.tensor([0.], dtype=data_type, device=device),
            "vertex_forces" : torch.tensor([0.], dtype=data_type, device=device),
            "deformation_energy" : torch.tensor([0.], dtype=data_type, device=device),
        }
        self.loss = torch.tensor([0.], dtype=data_type, device=device)
        self.chamfer_distances = torch.tensor([0.], dtype=data_type, device=device)
        self.first_frame = 0
        self.last_frame = min(2, scene_parameters["n_images"]) # >0
        self.max_frames = min(100, scene_parameters["n_images"])
        self.epoch_counter = 0
        self.frame_counter = self.first_frame - 1
        self.new_frame_period = 5
        self.expected_epochs = self.new_frame_period * (self.max_frames - self.last_frame + 1)
        self.lerp = 3.0

    def createCloth(
        self,
        time_step,
        optimize,
        device,
        data_type,
    ) -> objects.cloth.Cloth:
        device = "cuda:0"

        mesh_file = self.scene_parameters["mesh_file"]
        poly_mesh = om.read_polymesh(mesh_file, vertex_tex_coord = True)
        tri_mesh  = om.read_trimesh(mesh_file, vertex_tex_coord = True)

        self.texture = torch.rand(size = (1, 512, 512, 3), device=device, dtype=data_type) * 0.0 + 0.5
        self.texture.requires_grad_(optimize)
        
        material_texture_size = (472, 472)
        self.diffuse = torch.rand(size = (1, material_texture_size[0], material_texture_size[1], 4), device=device, dtype=data_type) * 1.0 + 0.0
        self.diffuse[..., -1] = 1
        self.diffuse.requires_grad_(optimize)
        self.roughness = (torch.rand(size = (1, material_texture_size[0], material_texture_size[1], 1), device=device, dtype=data_type) * 0.0 + 1.0).requires_grad_(optimize)
        self.metallic = (torch.rand(size = (1, material_texture_size[0], material_texture_size[1], 1), device=device, dtype=data_type) * 0.0 + 0.0).requires_grad_(optimize)
        self.normal_map = None

        decoder_args = argparse.ArgumentParser(description="material_decoder")
        with open("./code/python/material_encoder/fit.json") as json_file:
            decoder_args.method_args = json.load(json_file)
        self.material_decoder = material_encoder.decoder_optimization.Decoder(decoder_args)

        self.environment_map_tensor = (torch.rand((256, 512, 3), dtype=torch.float32, device="cuda:0") * 0.0 + 0.5).requires_grad_(optimize)
        self.environment_map = light.EnvironmentLight(self.environment_map_tensor)


        self.args = argparse.Namespace()
        self.args.device = device
        self.args.resolution = torch.tensor(self.scene_parameters["upper_right_corners"][0] - self.scene_parameters["lower_left_corners"][0], device=device, dtype=torch.int32)
        self.args.background = torch.zeros([1, self.args.resolution[0], self.args.resolution[1], 3], dtype=torch.float32).cuda()
        self.args.decorrelated = 0
        self.args.spp = 1
        self.args.layers = 1
        self.args.n_samples = 4
        self.args.denoiser = None#"bilateral"
        self.args.denoiser_demodulate = False

        initial_positions, self.edges, self.faces, self.uv = utils.mesh_utils.loadPolyMesh(
            triangle_mesh=tri_mesh,
            polygon_mesh=poly_mesh,
            device=device,
            data_type=data_type,
        )
        initial_positions *= (self.scene_parameters["mesh_scaling"])
        initial_velocities = torch.zeros_like(initial_positions)

        self.vertex_positions = torch.zeros((self.scene_parameters["n_images"], initial_positions.shape[0], initial_positions.shape[1]), device=device, dtype=data_type)

        self.gravity                  = torch.tensor(self.scene_parameters["gravity"], dtype=data_type, device=device)
        self.damping_factor           = torch.tensor([self.scene_parameters["damping_factor"   ]], dtype=data_type, device=device).requires_grad_(optimize)
        self.area_density             = torch.tensor([self.scene_parameters["area_density"     ]], dtype=initial_positions.dtype, device=initial_positions.device).requires_grad_(optimize)
        self.pulling_stiffness        = torch.tensor([self.scene_parameters["pulling_stiffness"]], dtype=initial_positions.dtype, device=initial_positions.device)
        self.log_stretching_stiffness = torch.log10(torch.tensor([self.scene_parameters["stretching_stiffness"]], dtype=initial_positions.dtype, device=initial_positions.device)).requires_grad_(optimize)
        self.log_shearing_stiffness   = torch.log10(torch.tensor([self.scene_parameters["shearing_stiffness"  ]], dtype=initial_positions.dtype, device=initial_positions.device)).requires_grad_(optimize)
        self.log_bending_stiffness    = torch.log10(torch.tensor([self.scene_parameters["bending_stiffness"   ]], dtype=initial_positions.dtype, device=initial_positions.device)).requires_grad_(optimize)

        self.constant_forces = (- self.gravity.clone()).unsqueeze(0).requires_grad_(optimize)
        self.vertex_forces   = torch.zeros((self.scene_parameters["n_images"] - 1, initial_positions.shape[0], initial_positions.shape[1]), device=device, dtype=data_type, requires_grad=optimize)

        self.background_images = []
        for i in range(len(self.scene_parameters["upper_right_corners"])):
            self.background_images.append(torch.zeros(size = (*(self.scene_parameters["upper_right_corners"][i] - self.scene_parameters["lower_left_corners"][i]), 3), device="cuda:0", dtype=torch.float32, requires_grad=optimize))
        

        if self.scene_parameters["override_data"]:
            self.override_data(
                optimize,
                material_texture_size,
                device,
            )


        cloth_1 = objects.cloth.DefaultCloth(
            polygon_mesh         = poly_mesh,
            time_step            = time_step,
            initial_positions    = initial_positions,
            initial_velocities   = initial_velocities,
            edges                = self.edges,
            faces                = self.faces,
            gravity              = self.gravity,
            area_density         = self.area_density,
            pulling_stiffness    = self.pulling_stiffness,
            stretching_stiffness = 10**self.log_stretching_stiffness,
            bending_stiffness    = 10**self.log_bending_stiffness,
            shearing_stiffness   = 10**self.log_shearing_stiffness,
        )

        # self.pulling_indices = cloth_1.pulling_segments.indices
        self.anchor_points = cloth_1.pulling_segments.target_positions
        self.anchor_points.requires_grad_(optimize)
        with torch.no_grad():
            self.anchor_distance = torch.linalg.norm(self.anchor_points[1] - self.anchor_points[0])
        self.anchor_movement = torch.zeros((self.vertex_forces.shape[0], *self.anchor_points.shape), device=device, dtype=data_type, requires_grad=optimize)

        self.stretching_rest_lengths = cloth_1.stretching_segments.rest_lengths
        self.stretching_rest_lengths.requires_grad_(optimize)

        return cloth_1
    
    def override_data(
        self,
        optimize,
        material_texture_size,
        device,
    ):
        self.epoch_counter = self.scene_parameters["override_epoch_counter"]
        self.last_frame = self.max_frames

        motion = torch.load(self.scene_parameters["override_motion_file"])
        self.vertex_positions = motion.clone()
        if self.scene_parameters["override_geometry"]:
            self.faces = torch.load(self.scene_parameters["override_faces_file"])
            self.edges = utils.mesh_utils.edges_from_faces(self.faces)
            self.uv = torch.load(self.scene_parameters["override_uv_file"])

        self.texture = utils.texture_utils.loadTexture(self.scene_parameters["override_texture_file"], texture_size=torch.tensor([-1, -1], device=device))[..., 0:3].contiguous()

        material_texture_size = torch.tensor(material_texture_size)
        self.diffuse   = utils.texture_utils.loadTexture(self.scene_parameters["override_diffuse_file"], texture_size=material_texture_size)
        self.diffuse   = nvdiffrecmc.render.util.srgb_to_rgb(self.diffuse).requires_grad_(optimize)
        self.roughness = utils.texture_utils.loadTexture(self.scene_parameters["override_roughness_file"], texture_size=material_texture_size)[..., 0:1].requires_grad_(optimize)
        self.metallic  = utils.texture_utils.loadTexture(self.scene_parameters["override_metallic_file"], texture_size=material_texture_size)[..., 0:1].requires_grad_(optimize)
        self.environment_map_tensor = torch.load(self.scene_parameters["override_environment_map_file"]).requires_grad_(optimize)
        self.environment_map = light.EnvironmentLight(self.environment_map_tensor)
    
    def initializeOptimizers(self):
        self.parameters = {"damping":         self.damping_factor,
                           "density":         self.area_density,
                           "stretching":      self.log_stretching_stiffness,
                           "shearing":        self.log_shearing_stiffness,
                           "bending":         self.log_bending_stiffness,
                           "constant_forces": self.constant_forces,
                           "vertex_forces":   self.vertex_forces,
                           "texture":         self.texture,
                           "diffuse":         self.diffuse,
                           "roughness":       self.roughness,
                           "metallic":        self.metallic,
                           "environment_map": self.environment_map_tensor,
                           "uv":              self.uv,
                           "background":      self.background_images,
                           "anchor_points":   self.anchor_points,
                           "anchor_movement": self.anchor_movement,
                           "edge_lengths":    self.stretching_rest_lengths,
        }
        self.initial_learning_rates = {"damping":         1e-3,
                                       "density":         2e-3,
                                       "stretching":      2e-2,
                                       "shearing":        2e-2,
                                       "bending":         2e-2,
                                       "constant_forces": 1e-1,
                                       "vertex_forces":   2e-1,
                                       "texture":         5e-2,
                                       "diffuse":         1e-2,
                                       "roughness":       1e-2,
                                       "metallic":        1e-2,
                                       "environment_map": 1e-2,
                                       "uv":              1e-4,
                                       "background":      4e-2,
                                       "anchor_points":   1e-3,
                                       "anchor_movement": 2e-3,
                                       "edge_lengths":    1e-5,
        }
        self.optimizer = dict(zip(self.parameters.keys(), [None] * len(self.parameters)))
        self.scheduler = dict(zip(self.parameters.keys(), [None] * len(self.parameters)))
        self.clipper   = dict(zip(self.parameters.keys(), [None] * len(self.parameters)))
        
        for key in self.parameters:
            if type(self.parameters[key]) is list:
                self.optimizer[key] = []
                for i in range(len(self.parameters[key])):
                    self.optimizer[key].append(torch.optim.Adam([self.parameters[key][i]], lr=self.initial_learning_rates[key]))
            else:
                self.optimizer[key] = torch.optim.Adam([self.parameters[key]], lr=self.initial_learning_rates[key])

        for key in self.parameters:
            if type(self.parameters[key]) is list:
                self.clipper[key] = []
                for i in range(len(self.parameters[key])):
                    self.clipper[key].append(autoclip.QuantileClip([self.parameters[key][i]], quantile=0.75, history_length=11, global_threshold=True))
            else:
                self.clipper[key] = autoclip.QuantileClip([self.parameters[key]], quantile=0.75, history_length=11, global_threshold=True)


    def loadData(self):
        image_lists = []
        for directory in pathlib.Path(self.scene_parameters["image_files"]).iterdir():
            if directory.is_dir():
                image_lists.append([str(file) for file in sorted(directory.rglob("*.png"))])
        mask_lists = []
        for directory in pathlib.Path(self.scene_parameters["mask_files"]).iterdir():
            if directory.is_dir():
                mask_lists.append([str(file) for file in sorted(directory.rglob("*.png"))])
        self.ground_truth_collection = utils.file_manager.loadGroundTruthImages(image_lists, mask_lists)

        depth_lists = []
        for directory in pathlib.Path(self.scene_parameters["depth_files"]).iterdir():
            if directory.is_dir():
                depth_lists.append([str(file) for file in sorted(directory.rglob("*.pt"))])
        self.ground_truth_depth_collection = utils.file_manager.loadGroundTruthDepths(depth_lists)
        

        # crop images
        for i in range(len(self.ground_truth_collection)):
            self.ground_truth_collection[i] = self.ground_truth_collection[i][...,
                                                                              self.scene_parameters["lower_left_corners"][i][0]:self.scene_parameters["upper_right_corners"][i][0],
                                                                              self.scene_parameters["lower_left_corners"][i][1]:self.scene_parameters["upper_right_corners"][i][1],
                                                                              :].contiguous()
            self.ground_truth_depth_collection[i] = self.ground_truth_depth_collection[i][...,
                                                                                          self.scene_parameters["lower_left_corners"][i][0]:self.scene_parameters["upper_right_corners"][i][0],
                                                                                          self.scene_parameters["lower_left_corners"][i][1]:self.scene_parameters["upper_right_corners"][i][1]].contiguous()
        
        # background = file_manager.loadBackgroundImage(self.scene_parameters["background_file"])
        # background = background[self.scene_parameters["lower_left_corners"][i][0]:self.scene_parameters["upper_right_corners"][i][0],
        #                         self.scene_parameters["lower_left_corners"][i][1]:self.scene_parameters["upper_right_corners"][i][1],
        #                         :].contiguous()
        
        # self.ground_truth_images[..., :3] = torch.where(self.ground_truth_images[..., 3, None] == 0, background, self.ground_truth_images[..., :3])
        
        # self.blurred_ground_truth_collection = []
        # for i in range(len(self.ground_truth_collection)):
        #     temp = self.ground_truth_collection[i].clone()
        #     temp = torch.permute(temp, (0, 3, 1, 2))
        #     temp = gaussian_blur(temp, 21, 7)
        #     self.blurred_ground_truth_collection.append(torch.permute(temp, (0, 2, 3, 1)))