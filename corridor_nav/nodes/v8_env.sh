# Source before running v8_policy.py / v8_drive_node.py:
#   source ~/v8_env.sh
# Why LD_PRELOAD: this Jetson's CUDA trees are sbsa (server ARM) builds whose libcublas
# has NO sm_87 kernels, so cublasCreate() fails on Orin. These are the Tegra (arm64)
# 12.6.4.1 builds, which carry sm_87. LD_PRELOAD rather than LD_LIBRARY_PATH because
# libtorch_cuda.so has RUNPATH /usr/local/cuda/lib64 and kept winning.
source /opt/ros/humble/setup.bash
source ~/v8env/bin/activate
export LD_PRELOAD=$HOME/cuda126_tegra_cublas/lib/libcublas.so.12:$HOME/cuda126_tegra_cublas/lib/libcublasLt.so.12
