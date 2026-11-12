"""Test-only launcher: run a script with cuDNN disabled (this Jetson's apt cuDNN 9.25 is an
sbsa build; real conv nets fail with CUDNN_STATUS_EXECUTION_FAILED_CUDART). Not part of the repo.
Usage: python3 run_nocudnn.py <script.py> [args...]"""
import runpy, sys, torch
torch.backends.cudnn.enabled = False
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
