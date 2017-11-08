#!/usr/bin/env bash
# cd /home/gpu/db/Dropbox/dev2/cpp/gpgpu/Data-Science-PyCUDA-GPU/docker
nvidia-docker run -it -p 5555:5555 -p 7842:7842 -p 8787:8787 -p 8786:8786 -p 8788:8788 -v /home/gpu/db/Dropbox/dev2/:/root/sharedfolder -v /home/gpu/dev/data/:/root/data quantscientist/pycuda bash
