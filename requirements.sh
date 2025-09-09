pip install cmake
pip install matplotlib
pip install imageio
pip install autoclip
pip install charonload

# for Cuda 12.4
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# openmesh needs an additional command line argument
CMAKE_POLICY_VERSION_MINIMUM=3.5 pip install openmesh

pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
