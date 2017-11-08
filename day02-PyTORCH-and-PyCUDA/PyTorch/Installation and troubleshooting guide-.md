# PyTorch installation and troubleshooting guide:

What is my PyThon version?

```
import sys
print('__Python VERSION:', sys.version)
```

What is my PyTorch version?

```
import torch
print('__pyTorch VERSION:', torch.__version__)

```

Well ... It seems I don't have PyTorch, what should I do?
- You can install PyTorch with GPU support from source, this works most of the time.

```

# PyTorch for the GPU and the CPU
# If you don't have CUDA installed, run this first:
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
export TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1+PTX" 
export TORCH_NVCC_FLAGS="-Xfatbin -compress-all" 

python setup.py install


# Build torch-vision from source
git clone https://github.com/pytorch/vision.git 
cd vision
git submodule update --init
python setup.py install 


# CPU version
#pip install git+https://github.com/pytorch/tnt.git@master
```


What is my CUDA version?

```
print('__CUDNN VERSION:', torch.backends.cudnn.version())

```

How many GPU's do I have (version one)?

```
print('__Number CUDA Devices:', torch.cuda.device_count())

```

How do I know that everything works ... ?

```
from subprocess import call
# call(["nvcc", "--version"]) does not work
! nvcc --version
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
print('Active CUDA Device: GPU', torch.cuda.current_device())

print ('Available devices ', torch.cuda.device_count())
print ('Current cuda device ', torch.cuda.current_device())

import torch
from torch.autograd import Variable
x=torch.Tensor(3,2)
print (type(x))
print (x)

# how variables work
x = Variable(x)
print ("x:" + str (x))
print ("requires grad:" + str(x.requires_grad))
print ("data:" + str(x.data))

```

I don't have CUDA installed, what should I do?

- You can either use the official docs or head straight to this script:

```
# https://github.com/QuantScientist/Deep-Learning-Boot-Camp/blob/master/docker/deps_nvidia_docker.sh

#!/usr/bin/env bash

apt-get install nvidia-modprobe

# curl -O -s https://raw.githubusercontent.com/minimaxir/keras-cntk-docker/master/deps_nvidia_docker.sh
if lspci | grep -i 'nvidia'
then
  echo "\nNVIDIA GPU is likely present."
else
  echo "\nNo NVIDIA GPU was detected. Exiting ...\n"
  exit 1
fi

echo "\nChecking for NVIDIA drivers ..."
if ! nvidia-smi ; then
  echo "Error: nvidia-smi is not installed, or not working."

  apt-get update
  apt-get install -y curl build-essential

  curl -O -s http://us.download.nvidia.com/XFree86/Linux-x86_64/375.39/NVIDIA-Linux-x86_64-375.39.run
  sh ./NVIDIA-Linux-x86_64-375.39.run -a --ui=none --no-x-check && rm NVIDIA-Linux-x86_64-375.39.run

  echo "\nInstalled NVIDIA drivers."
else
  echo "NVIDIA driver is already installed."
fi

echo "\nChecking docker ..."
if ! [ -x "$(command -v docker)" ]; then
  echo "docker is not installed."
  apt-get -y install apt-transport-https ca-certificates curl
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

  add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
  apt-get update

  apt-get -y install docker-ce
  echo "\nInstalled docker."
else
  echo "docker is already installed."
fi
echo "\nChecking nvidia-docker ..."
if ! [ -x "$(command -v nvidia-docker)" ]; then
  echo "nvidia-docker is not installled."

  wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
  dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb

  echo "\nInstalled nvidia-docker."
else
  echo "nvidia-docker is already installed."
fi
echo "\nAll dependencies are installed."

echo "\nTry running: \n\tsudo nvidia-docker run --rm nvidia/cuda nvidia-smi"
```  

Happy programming, 


