# Test-only: source AFTER ~/v8_env.sh. Preloads the Tegra (sm_87) cuDNN 9.3 from the JetPack r36.4 repo,
# unpacked to ~/cudnn93_tegra (no system change). The apt cuDNN 9.25 is an sbsa build -> conv nets fail on Orin.
_L=$HOME/cudnn93_tegra/usr/lib/aarch64-linux-gnu
export LD_LIBRARY_PATH=$_L:$LD_LIBRARY_PATH
export LD_PRELOAD=$LD_PRELOAD:$(ls $_L/libcudnn*.so.9 | tr '\n' ':')
