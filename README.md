<p align="center">
  <h1 align="center">🧃 SAFT: <br> Shape and Appearance of Fabrics from Template <br> via Differentiable Physical Simulations from Monocular Video</h1>
  <h3 align="center"> David Stotko · Reinhard Klein <br> University of Bonn </h3>
  <h3 align="center">ICCV 2025</h3>
  <h3 align="center"> <a href=https://cg.cs.uni-bonn.de/publication/stotko-2025-saft>Project Page</a> &nbsp; | &nbsp; <a href=https://www.youtube.com/watch?v=EvioNjBOARc>Video</a> </h3>
  <div align="center"></div>
</p>

![](readme_images/teaser.gif)

This repository contains our source code of the paper "SAFT: Shape and Appearance of Fabrics from Template via Differentiable Physical Simulations from Monocular Video".

## Abstract

The reconstruction of three-dimensional dynamic scenes is a well-established yet challenging task within the domain of computer vision.
In this paper, we propose a novel approach that combines the domains of 3D geometry reconstruction and appearance estimation for physically based rendering and present a system that is able to perform both tasks for fabrics, utilizing only a single monocular RGB video sequence as input.
In order to obtain realistic and high-quality deformations and renderings, a physical simulation of the cloth geometry and differentiable rendering are employed.
In this paper, we introduce two novel regularization terms for the 3D reconstruction task that improve the plausibility of the reconstruction by addressing the depth ambiguity problem in monocular video.
In comparison with the most recent methods in the field, we have reduced the error in the 3D reconstruction by a factor of $2.64$ while requiring a medium runtime of $30\,\mathrm{min}$ per scene.
Furthermore, the optimized motion achieves sufficient quality to perform an appearance estimation of the deforming object, recovering sharp details from this single monocular RGB video.

## Prerequisites

To install the necessary dependencies, you can use the provided ```requirements.sh```.
Note that this is a shell file because not all packages can be installed in parallel.
Moreover, the file is writen for CUDA 12.4.
Please change the ```pytorch``` installation according to your CUDA version or install CUDA 12.4.
``` bash
python3 -m venv saft
source saft/bin/activate
bash requirements.sh
```
This should work for Linux systems.
On Windows you may have to install ```pytorch3D``` manually.

You can clone ```nvdiffrecmc``` using the other script
``` bash
bash nvdiffrecmc.sh
```
This will download the repository and automatically apply a necessary patch to the mipmapping procedure.


## Usage

To run the code, you need to provide RGB and mask images of a video sequence together with a polygon mesh of the initial cloth geometry of the first video frame.
The mesh has to contain the uv-coordinates and polygon edges encodes the vertex connections that are used during the physical simulation.
Although we automatically triangulate the mesh for rendering, the resulting triangles may be really bad and we therefore recommend using quad meshes or triangle meshes directly.

We evaluate the method on the [ϕ-SfT](https://4dqv.mpi-inf.mpg.de/phi-SfT/) dataset.
You can get the already prepared data from [here](https://uni-bonn.sciebo.de/s/CPHXtLN5kre9zGD).
You have to extract all the files into the main directory and are then able to run the code.
The data information is provided by ```json``` files in ```./data/scenes/```.
You can run all real-world scenes at once using the command
```
python .\code\python\main.py R1 R2 R3 R4 R5 R6 R7 R8 R9
```
Please note, that the reconstruction quality of some scenes may vary significantly due to non-deterministic GPU computations.
The results of the paper are loaded by changing the ```override_data``` parameter in line 74 of the ```json``` files to ```true```.
The results are saved at multiple epochs in the ```results``` directory.

You can evaluate the reconstruction metrics by running the 
```
python .\code\python\metrics.py
```
file.
It is configured to load the saved data of the last epoch for each scene R1 to R9.
Please change the loaded scene and other options in ```metrics.py``` according to your needs.

## Citation

```
@inproceedings{stotko2025saft,
	title = {{SAFT: Shape and Appearance of Fabrics from Template via Differentiable Physical Simulations from Monocular Video}},
	author = {Stotko, David and Klein, Reinhard},
	booktitle = {{International Conference on Computer Vision (ICCV)}},
	year = {2025},
}
```

## License

This software is provided under MIT license. See [LICENSE](LICENSE) for more information.

## Acknowledgements

This work has been funded by the DFG project KL 1142/11-2 (DFG Research Unit FOR 2535 Anticipating Human Behavior), by the Federal Ministry of Education and Research of Germany and the state of North-Rhine Westphalia as part of the Lamarr-Institute for Machine Learning and Artificial Intelligence and the InVirtuo 4.0 project, and additionally by the Federal Ministry of Education and Research under grant no. 01IS22094A WEST-AI.