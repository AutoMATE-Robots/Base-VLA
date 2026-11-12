"""One-shot Jev check (T4 with Jev): runs adaptive_reasoning.live's CLI over sign frames with
Jev swapped in for the OpenAI-compatible VLM. Makes at most one Jev call per sign.
  source ~/v8_env.sh && source ~/signway_handoff/cudnn_tegra_env.sh && source ~/.signway_secrets
  cd ~/weights && python3 ~/signway_viz/jev_test.py --goal Elevators --tau 0.45 --wait \
      --model jev-latest --reasoning none --log-dir ~/signway_logs/T4_jev <frames...>"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path("~/weights").expanduser()))
import adaptive_reasoning.reasoner as R
from jev_vlm import JevVLM
R.OpenAICompatVLM = JevVLM
from adaptive_reasoning.live import main
main()
