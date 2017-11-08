
apt-get update && apt-get install --no-install-recommends  -y \
    git cmake build-essential libgoogle-glog-dev libgflags-dev libeigen3-dev libopencv-dev libcppnetlib-dev libboost-dev libboost-iostreams-dev libcurl4-openssl-dev protobuf-compiler libopenblas-dev libhdf5-dev libprotobuf-dev libleveldb-dev libsnappy-dev liblmdb-dev libutfcpp-dev wget unzip supervisor \
    python \
    python-dev \
    python2.7-dev \
    python3-dev \
    python-virtualenv \
    python-wheel \
	python-tk \
    pkg-config \
    # requirements for numpy
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
        libboost-dev \
        libgraphviz-dev \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/* && \
# Link BLAS library to use OpenBLAS using the alternatives mechanism (https://www.scipy.org/scipylib/building/linux.html#debian-ubuntu)
update-alternatives --set libblas.so.3 /usr/lib/openblas-base/libblas.so.3

apt-get update && apt-get install -y clang-3.7 libclang-common-3.7-dev libclang-3.7-dev libclang1-3.7  libllvm-3.7-ocaml-dev libllvm3.7 lldb-3.7 llvm-3.7 llvm-3.7-dev llvm-3.7-runtime clang-modernize-3.7 clang-format-3.7 lldb-3.7-dev
apt-get clean
rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

echo "/usr/lib/llvm-3.7/lib/" >> /etc/ld.so.conf && ldconfig


curl --silent https://bootstrap.pypa.io/get-pip.py | python
pip --no-cache-dir install setuptools==33.1.1
# Install other useful Python packages using pip
pip --no-cache-dir install \
		Cython \
		werkzeug pillow psycogreen flask celery redis Boto FileChunkIO nltk fuzzywuzzy rotate-backups oauthlib requests pyOpenSSL ndg-httpsclient pyasn1 \
		path.py \
		Pillow \
		pygments \
		six \
		sphinx \
		wheel \
		zmq


export LD_LIBRARY_PATH=/usr/lib/llvm-3.7/lib/
export LLVM_CONFIG=/usr/lib/llvm-3.7/bin/llvm-config

apt-get -qyy install build-essential scons pkg-config libx11-dev libxcursor-dev libxinerama-dev libgl1-mesa-dev libglu-dev libasound2-dev libpulse-dev libfreetype6-dev libssl-dev libudev-dev libxrandr-dev

pip --no-cache-dir install  cython pytest pandas scikit-learn statsmodels  line-profiler psutil spectrum memory_profiler pandas joblib pyparsing pydot pydot-ng graphviz pandoc SQLAlchemy flask toolz cloudpickle python-snappy s3fs widgetsnbextension ipywidgets terminado cytoolz bcolz blosc partd backports.lzma mock cachey moto pandas_datareader
pip install -i https://pypi.anaconda.org/sklam/simple llvmlite
pip --no-cache-dir install fastparquet

# Install Theano and set up Theano config (.theanorc) OpenBLAS
pip --no-cache-dir install theano && \
	\
	echo "[global]\ndevice=cpu\nfloatX=float32\nmode=FAST_RUN \
		\n[lib]\ncnmem=0.95 \
		\n[nvcc]\nfastmath=True \
		\n[blas]\nldflag = -L/usr/lib/openblas-base -lopenblas \
		\n[DebugMode]\ncheck_finite=1" \
	> /root/.theanorc



# Install BAYESIAN FRAMEWORKS
pip --no-cache-dir install  --upgrade pymc3 pystan edward watermark xgboost bokeh seaborn mmh3 tensorflow theano

export KERAS_VERSION=1.2.2
export KERAS_BACKEND=tensorflow
pip --no-cache-dir install --no-dependencies git+https://github.com/fchollet/keras.git@${KERAS_VERSION}


apt-get -qyy install python2.7 python-pip python-dev ipython ipython-notebook
pip install --upgrade pip
pip install --upgrade ipython
pip --no-cache-dir install jupyter
python -m ipykernel.kernelspec
python2 -m ipykernel.kernelspec --user
jupyter notebook --allow-root --generate-config -y


apt-get update && apt-get install -y software-properties-common && \
    apt-get install -y --no-install-recommends \
        build-essential \
        clinfo \
        cmake \
        git \
        libboost-all-dev \
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


apt-get update && apt-get install -y software-properties-common && \
    apt-get install -qyy --no-install-recommends \
        build-essential \
        clinfo \
        cmake \
        git \
        libboost-all-dev \
        libfftw3-dev \
        libfontconfig1-dev \
        libfreeimage-dev \
        liblapack-dev \
        liblapacke-dev \
        libopenblas-dev \
        ocl-icd-opencl-dev \
        opencl-headers \
        wget \
        libboost-all-dev \
        libfreeimage3 libfreeimage-dev \
        libopenblas-dev \
        xorg-dev && \
    rm -rf /var/lib/apt/lists/*

apt-get install gsl-dev openblas
apt-get install -y build-essential git cmake libfreeimage-dev
apt-get install -y cmake-curses-gui
# Using OpenBLAS
apt-get install libopenblas-dev libfftw3-dev liblapacke-dev
 # Using ATLAS
#apt-get install libatlas3gf-base libatlas-dev libfftw3-dev liblapacke-dev


ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so.1 && \
    ln -s /usr/lib/libcuda.so.1 /usr/lib/libcuda.so && \
    mkdir -p /etc/OpenCL/vendors && \
    echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd && \
    echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
export  PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}

export CUDA_BIN_PATH=/usr/local/cuda

export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0

# torch
#pip install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl
pip2 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post2-cp27-cp27mu-manylinux1_x86_64.whl
#pip install https://s3.amazonaws.com/pytorch/whl/cu75/torch-0.1.6.post22-cp27-none-linux_x86_64.whl 
pip install torchvision
 
# Install nvidia-docker and nvidia-docker-plugin
# wget -P /tmp https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
#dpkg -i /tmp/nvidia-docker*.deb && rm /tmp/nvidia-docker*.deb
