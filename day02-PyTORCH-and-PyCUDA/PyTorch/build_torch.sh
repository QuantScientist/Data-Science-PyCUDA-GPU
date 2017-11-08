# PyTorch GPU and CPU
# If you dont have CUDA installed, run this first:
# https://github.com/QuantScientist/Deep-Learning-Boot-Camp/blob/master/docker/deps_nvidia_docker.sh

#GPU version
export PATH=/usr/local/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
export CUDA_BIN_PATH=/usr/local/cuda
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0


# Build PyTorch from source
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git submodule update --init
#git checkout 4eb448a051a1421de1dda9bd2ddfb34396eb7287 

export TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1+PTX" 
export TORCH_NVCC_FLAGS="-Xfatbin -compress-all" 

#pip uninstall torch
#python setup.py clean
#python setup.py build
python setup.py install


# Build torch-vision from source
git clone https://github.com/pytorch/vision.git 
cd vision
#git checkout 83263d8571c9cdd46f250a7986a5219ed29d19a1 
git submodule update --init
python setup.py install 


# CPU version
#pip install git+https://github.com/pytorch/tnt.git@master

