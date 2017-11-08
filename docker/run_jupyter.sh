pwd
ls -la

export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
export PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
export CUDA_BIN_PATH=/usr/local/cuda
export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-8.0


ldconfig /usr/local/cuda/lib64

#export PATH=$PATH:/root/cling/bin
NCPUS=`python -c "import multiprocessing as mp; print(mp.cpu_count())"`
echo "Detected $NCPUS cpus"

#python -c "import sys; sys.path.append('/root/inst/bin/')"
export PATH=/root/inst/bin/:$PATH

echo $PATH


#dask-scheduler --host localhost &
#dask-worker localhost:8786 $* &
jupyter notebook --allow-root "$@" &


