import charonload
import pathlib

VSCODE_STUBS_DIRECTORY = pathlib.Path(__file__).parent / "./build_python/typings"
charonload.module_config["cpp_solver"] = charonload.Config(
    # All paths must be absolute
    project_directory=pathlib.Path(__file__).parents[1] / "cpp",
    build_directory=pathlib.Path(__file__).parent / "./build_python",
    stubs_directory=VSCODE_STUBS_DIRECTORY,
    build_type="Release",
    verbose=True,
    stubs_invalid_ok=True,
)

import sys
import numpy as np
import torch
import time
import gc

import utils.file_manager
import scene_content
import utils.auxiliary
import rendering.rendering
import phases.texture
import phases.parameters
import phases.material
import evaluation.evaluation


class Optimization():
    def __init__(self, scene) -> None:
        print("Start: Initialization")
        t_start = time.perf_counter()

        with torch.no_grad():
            self.initializeParameters(scene)
            self.scene_content = scene_content.SceneContent()
            self.scene_content.initializeHeadless(self.scene_parameters, self.target_time_step, not self.scene_parameters["override_data"], self.device, self.data_type)
            self.renderer = rendering.rendering.Renderer()
            self.renderer.initialize(self.scene_parameters, self.device, self.data_type, self.scene_content.args, not self.scene_parameters["override_data"])
            self.auxiliary_data = utils.auxiliary.AuxiliaryData()
            self.auxiliary_data.initialize(self.scene_content)

            self.phase_transitions = [200,
                                      200 + self.scene_content.expected_epochs + 15,
                                      215 + self.scene_content.expected_epochs + 2000]

            ground_truth_point_clouds, point_clouds_lengths = evaluation.evaluation.loadGroundTruth(self.scene_parameters)
            if self.scene_parameters["scene"] in ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"]:
                ground_truth_point_clouds[..., 0:2] *= -1
            our_point_clouds = torch.zeros_like(ground_truth_point_clouds)
            self.point_clouds = {"ground_truth" : ground_truth_point_clouds, "ours" : our_point_clouds, "lengths" : point_clouds_lengths}

        t_end = time.perf_counter()
        print(f"Done: Initialization in {t_end - t_start:.3f} s\n")
        print("Epoch |  Time      Total t  |   Loss   per Frame  Image   Silhouette Vertex F  Deform  | Stretch   Shear     Bend   Damping    Wind x    Wind y    Wind z  | Chamfer Dist ")
        print("------+---------------------+----------------------------------------------------------+----------------------------------------------------------------------------------")


    def initializeParameters(self, scene : str):
        torch.set_printoptions(threshold = 100, linewidth = 160)

        self.scene_parameters = utils.file_manager.loadJson(scene)

        self.device = "cuda:0"
        self.data_type = torch.float32
        self.target_time_step = 0.02
        self.repetitions = 4
        self.evaluate = True
        self.save_screenshots = True
        self.debug = True

        self.counter = 0
        self.headless = True
        self.phase_transitions = []


    def update(self):
        # switch between optimization phases
        if self.scene_content.epoch_counter < 0:
            self.scene_content.epoch_counter += 1
        elif self.scene_content.epoch_counter < self.phase_transitions[0]:
            phases.texture.optimize(
                self.scene_content,
                self.renderer,
                self.auxiliary_data,
                self.target_time_step,
                self.repetitions,
                {},
                self.point_clouds,
                not self.scene_parameters["override_data"],
                self.evaluate,
                self.debug,
                self.headless,
            )
        elif self.scene_content.epoch_counter < self.phase_transitions[1]:
            phases.parameters.optimize(
                self.scene_content,
                self.renderer,
                self.auxiliary_data,
                self.target_time_step,
                self.repetitions,
                {},
                self.point_clouds,
                not self.scene_parameters["override_data"],
                self.evaluate,
                self.debug,
                self.headless,
            )
        else:
            phases.material.optimize(
                self.scene_content,
                self.renderer,
                self.auxiliary_data,
                self.target_time_step,
                self.repetitions,
                {},
                self.point_clouds,
                not self.scene_parameters["override_data"],
                self.evaluate,
                self.debug,
                self.headless,
                load_positions = True,
            )

        if self.scene_content.epoch_counter == self.phase_transitions[1] and self.scene_content.frame_counter == -1:
            print("Finished: ", end="")
            self.auxiliary_data.printQuantities(self.scene_content, torch.mean(self.scene_content.chamfer_distances))

        # save positions instead of resimulating
        if self.save_screenshots:
            with torch.no_grad():
                if (   (self.scene_content.epoch_counter == self.phase_transitions[1] - 1) and self.scene_content.frame_counter != -1
                    or (self.scene_content.epoch_counter == self.phase_transitions[1] and self.scene_content.frame_counter == -1)):
                    self.scene_content.vertex_positions[self.scene_content.frame_counter] = self.scene_content.cloth_1.positions


                save_epochs = np.array([0, 1, 50, 200, 300, self.phase_transitions[1], self.phase_transitions[1] + 1000, self.phase_transitions[-1]])
                for save_epoch in save_epochs:
                    if (   (self.scene_content.epoch_counter == save_epoch     and self.scene_content.frame_counter != -1)
                        or (self.scene_content.epoch_counter == save_epoch + 1 and self.scene_content.frame_counter == -1)):
                        save_material = False
                        save_frame = self.scene_content.frame_counter
                        if save_frame == -1:
                            save_frame = self.scene_content.last_frame - 1
                            save_material = True

                        utils.auxiliary.saveScreeshots(
                            self.scene_parameters['scene'],
                            f"results/{self.scene_parameters['scene']}/{str(save_epoch).zfill(3)}",
                            (0,),
                            save_frame,
                            save_material,
                            self.renderer.render_output,
                            self.scene_content.vertex_positions,
                            self.scene_content.cloth_1.faces,
                            self.scene_content.uv,
                            self.point_clouds["ours"],
                            self.point_clouds["lengths"],
                            self.scene_content.chamfer_distances,
                            self.scene_content.texture,
                            self.scene_content.diffuse,
                            self.scene_content.metallic,
                            self.scene_content.roughness,
                            self.scene_content.normal_map,
                            self.scene_content.environment_map_tensor,
                            self.scene_content.background_images,
                            self.scene_content.ground_truth_collection,
                            self.scene_content.ground_truth_depth_collection,
                            self.scene_content.scene_parameters["depth_image_range"],
                        )



def main():
    #scenes = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "SR1", "SR2", "SR3", "SR4", "SR5"]
    scenes = sys.argv[1:]
    chamfer_distances = {}
    times = {}

    time_1_start = time.perf_counter()
    for scene in scenes:
        print(scene)
        time_2_start = time.perf_counter()
        optimization = Optimization("./data/scenes/" + scene + ".json")

        while optimization.scene_content.epoch_counter <= optimization.phase_transitions[-1]:
            optimization.update()

        time_2_end = time.perf_counter()
        times[scene] = time_2_end - time_2_start
        chamfer_distances[scene] = optimization.scene_content.chamfer_distances
        del optimization
        gc.collect()

    print("-"*100)
    mean = torch.tensor([0.0], device="cuda:0")
    for key in chamfer_distances:
        print(key)
        print(f"Time: {times[key]} s = {times[key]/60} min = {times[key]/3600} h")
        mean += torch.mean(chamfer_distances[key])
        print(torch.mean(chamfer_distances[key]))
        print(chamfer_distances[key])
        print("-"*100)
    print(mean.item() / len(scenes))

    time_1_end = time.perf_counter()
    print(f"Time: {times[key]} s = {times[key]/60} min = {times[key]/3600} h")
    print(f"Total time: {time_1_end - time_1_start} s = {(time_1_end - time_1_start)/60} min = {(time_1_end - time_1_start)/3600} h")

if __name__ == "__main__":
    main()