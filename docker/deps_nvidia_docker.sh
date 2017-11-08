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
# Check for CUDA and try to install.
if ! dpkg-query -W cuda; then
  # The 16.04 installer works with 16.10.
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  apt-get update
  apt-get install cuda -y
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
