FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04


ENV CUDA_ARCH_BIN "30 35 50 52 60"
ENV CUDA_ARCH_PTX "60"

RUN rm -rf /var/lib/apt/lists/*
RUN apt-get clean

RUN apt-get update && apt-get install --no-install-recommends  -y \
    git cmake build-essential libgoogle-glog-dev libgflags-dev libeigen3-dev libopencv-dev libcppnetlib-dev libboost-dev libboost-all-dev libboost-iostreams-dev libcurl4-openssl-dev protobuf-compiler libopenblas-dev libhdf5-dev libprotobuf-dev libleveldb-dev libsnappy-dev liblmdb-dev libutfcpp-dev wget unzip  \
    python \
    python-dev \
    python2.7-dev \
    python3-dev \
    python-virtualenv \
    python-wheel \
	python-tk \
    pkg-config \
    libopenblas-base \
    python-numpy \
    python-scipy \
    # requirements for keras
    python-h5py \
    python-yaml \
    python-pydot \
    python-nose \
	python-h5py \
	python-skimage \
	python-matplotlib \
	python-pandas \
	python-sklearn \
	python-sympy \
	python-joblib \
        build-essential \
        software-properties-common \
        g++ \
        git \
        wget \
        tar \
        git \
        imagemagick \
        curl \
		bc \
		htop\
		curl \
		g++ \
		gfortran \
		git \
		libffi-dev \
		libfreetype6-dev \
		libhdf5-dev \
		libjpeg-dev \
		liblcms2-dev \
		libopenblas-dev \
		liblapack-dev \
		libssl-dev \
		libtiff5-dev \
		libwebp-dev \
		libzmq3-dev \
		nano \
		unzip \
		vim \
		zlib1g-dev \
		qt5-default \
		libvtk6-dev \
		zlib1g-dev \
		libjpeg-dev \
		libwebp-dev \
		libpng-dev \
		libtiff5-dev \
		libjasper-dev \
		libopenexr-dev \
		libgdal-dev \
		libdc1394-22-dev \
		libavcodec-dev \
		libavformat-dev \
		libswscale-dev \
		libtheora-dev \
		libvorbis-dev \
		libxvidcore-dev \
		libx264-dev \
		yasm \
		libopencore-amrnb-dev \
		libopencore-amrwb-dev \
		libv4l-dev \
		libxine2-dev \
		libtbb-dev \
		libeigen3-dev \
		doxygen \
		less \
        htop \
        procps \
        vim-tiny \
        libgraphviz-dev \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/* && \
	update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3


RUN apt-get update && apt-get install -y software-properties-common && \
    apt-get install -y --no-install-recommends \
        build-essential \
        clinfo \
        cmake \
        git \
        libfftw3-dev \
        libfontconfig1-dev \
        libfreeimage-dev \
        liblapack-dev \
        liblapacke-dev \
        libopenblas-dev \
        ocl-icd-opencl-dev \
        opencl-headers \
        wget \
        xorg-dev && \
rm -rf /var/lib/apt/lists/*




# Install caffe dependencies
RUN chmod 777 /tmp && apt-get update && apt-get install -y \
  git \
  wget \
  cmake \
  curl \
  vim \
  libatlas-base-dev \
  libatlas-dev \
  libopencv-dev \
  libprotobuf-dev \
  libgoogle-glog-dev \
  libgflags-dev \
  protobuf-compiler \
  libhdf5-dev \
  libleveldb-dev \
  liblmdb-dev \
  libsnappy-dev \
  python-dev \
  python-pip \
  python-numpy \
  gfortran > /dev/null

WORKDIR /

RUN apt-get -qyy install python2.7 python-pip python-dev
RUN pip install --upgrade pip
RUN pip install numpy

##################################################

# Install dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        git \
        libgflags-dev \
        libgoogle-glog-dev \
        libprotobuf-dev \
        pkg-config \
        protobuf-compiler \
        python-yaml \
        wget && \
    rm -rf /var/lib/apt/lists/*


RUN pip --no-cache-dir install numpy pandas sklearn matplotlib seaborn ipython jupyter pyyaml h5py ipykernel


# Install OpenCV 3.2.0 with CUDA support
RUN git clone --depth 1 -b 3.2.0 https://github.com/Itseez/opencv.git /opencv && \
    cd /opencv && \
    cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
          -DWITH_CUDA=ON -DCUDA_ARCH_BIN="${CUDA_ARCH}" -DCUDA_ARCH_PTX="${CUDA_ARCH}" \
          -DWITH_JPEG=ON -DBUILD_JPEG=ON -DWITH_PNG=ON -DBUILD_PNG=ON \
          -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DWITH_FFMPEG=OFF -DWITH_GTK=OFF \
          -DWITH_OPENCL=OFF -DWITH_QT=OFF -DWITH_V4L=OFF -DWITH_JASPER=OFF \
          -DWITH_1394=OFF -DWITH_TIFF=OFF -DWITH_OPENEXR=OFF -DWITH_IPP=OFF -DWITH_WEBP=OFF \
          -DBUILD_opencv_superres=OFF -DBUILD_opencv_java=OFF -DBUILD_opencv_python2=OFF \
          -DBUILD_opencv_videostab=OFF -DBUILD_opencv_apps=OFF -DBUILD_opencv_flann=OFF \
          -DBUILD_opencv_ml=OFF -DBUILD_opencv_photo=OFF -DBUILD_opencv_shape=OFF \
          -DBUILD_opencv_cudabgsegm=OFF -DBUILD_opencv_cudaoptflow=OFF -DBUILD_opencv_cudalegacy=OFF \
          -DCUDA_NVCC_FLAGS="--default-stream per-thread -O3" -DCUDA_FAST_MATH=ON && \
    make -j"$(nproc)" install && ldconfig && \
    rm -rf /opencv

##################################################

# Install NCCL for multi-GPU communication
RUN wget https://github.com/NVIDIA/nccl/releases/download/v1.2.3-1%2Bcuda8.0/libnccl1_1.2.3-1.cuda8.0_amd64.deb && \
  dpkg -i libnccl1_1.2.3-1.cuda8.0_amd64.deb && \
  rm libnccl1_1.2.3-1.cuda8.0_amd64.deb && \
  wget https://github.com/NVIDIA/nccl/releases/download/v1.2.3-1%2Bcuda8.0/libnccl-dev_1.2.3-1.cuda8.0_amd64.deb && \
  dpkg -i libnccl-dev_1.2.3-1.cuda8.0_amd64.deb && \
  rm libnccl-dev_1.2.3-1.cuda8.0_amd64.deb



# Clone Caffe repo and move into it
RUN  cd /root && git clone https://github.com/BVLC/caffe.git && cd caffe && \
	cat python/requirements.txt | xargs -n1 pip install && pip install -v thrift==0.9.3

RUN cd /root/caffe && \
# Make and move into build directory
  mkdir build && cd build && \
# CMake
  cmake .. && \
# Make
  make -j"$(nproc)" all && \
  make install

# Add to Python path
ENV PYTHONPATH=/root/caffe/python:$PYTHONPATH
# Add caffe to path
ENV PATH=/root/caffe/build/tools:$PATH

RUN pip --no-cache-dir install tensorflow-gpu
# Keras
RUN pip install git+https://github.com/fchollet/keras.git


WORKDIR /root/

ENV PATH=/usr/local/cuda-8.0/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV CUDA_BIN_PATH=/usr/local/cuda
ENV CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0



RUN pip install -v scikit-learn==0.18  pycuda && pip install opencv-python pycrypto  astropy autograd

# Torch

# RUN pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl 
RUN pip2 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post2-cp27-cp27mu-manylinux1_x86_64.whl

# Build PyTorch from source
# RUN git clone https://github.com/pytorch/pytorch.git \
#  && cd pytorch \
#  && git checkout 4eb448a051a1421de1dda9bd2ddfb34396eb7287 \
#  && TORCH_CUDA_ARCH_LIST="3.5 5.2 6.0 6.1+PTX" \
#     TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
#     python setup.py install \
#  && rm -rf pytorch

# Build torch-vision from source
# RUN git clone https://github.com/pytorch/vision.git \
#  && cd vision \
#  && git checkout 83263d8571c9cdd46f250a7986a5219ed29d19a1 \
#  && python setup.py install \
#  && rm -rf vision

# Install Torchnet, a high-level framework for PyTorch
RUN pip install git+https://github.com/pytorch/tnt.git@master
RUN pip install torchvision psutil


RUN python -m ipykernel.kernelspec
RUN python2 -m ipykernel.kernelspec --user
RUN jupyter notebook --allow-root --generate-config -y

COPY jupyter_notebook_config.py /root/.jupyter/
# Jupyter has issues with being run directly: https://github.com/ipython/ipython/issues/7062
COPY run_jupyter.sh /root/
WORKDIR "/root/"
RUN chmod +x run_jupyter.sh
RUN jupyter nbextension enable --py --sys-prefix widgetsnbextension

#ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib/root

# Expose Ports
EXPOSE 6006 3838 8787 8888 8786 9786 8788 5432 8000 7842
