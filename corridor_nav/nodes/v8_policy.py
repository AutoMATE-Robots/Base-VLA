#!/usr/bin/env python3
"""
v8_policy.py -- on-device inference wrapper for the v8 trajectory VLA (OpenVLA-OFT).

    from v8_policy import load_policy
    policy = load_policy("/home/hunter/weights/v8_vla")
    wp = policy.predict(pil_rgb_image, "straight")      # -> np.ndarray (8,2) float32

CONVENTION (identical to eval_openloop.py, which this file mirrors)
    wp[i] = (x_forward_m, y_left_m), in the robot's CURRENT frame, CUMULATIVE
    positions -- NOT per-step deltas. wp[0] is ~0.1 m ahead, wp[7] is the end of the
    ~0.8 s horizon. Metres. +y is LEFT.

    The 16 numbers the action head emits are the interleaved flattening
    [x0,y0,x1,y1,...,x7,y7] (ACTION_DIM=16, NUM_ACTIONS_CHUNK=1 -- one shot, no
    chunking), so reshape(-1,2) recovers the 8 waypoints. dataset_statistics.json
    confirms the interleaving: its q99 alternates large (forward, 0.15 -> 1.18,
    monotonically growing with waypoint index) and small (lateral, 0.003 -> 0.21).

UN-NORMALIZATION
    Done inside vla.predict_action() via _unnormalize_actions(), which uses
    norm_stats[unnorm_key]["action"] loaded from the checkpoint's
    dataset_statistics.json with ACTION_PROPRIO_NORMALIZATION_TYPE == BOUNDS_Q99:
        a_metres = 0.5*(a_norm + 1)*(q99 - q01 + 1e-8) + q01      (mask all True)
    So predict() returns metres directly; there is no second un-normalization here.

NO TENSORFLOW
    The repo's own preprocessing helpers (resize_image_for_policy, center_crop_image,
    and therefore get_vla_action / robot_utils.get_action) call tensorflow, which is
    not installed and will not build on this aarch64 Jetson. This file reimplements
    exactly those two steps with PIL + torch (see _preprocess) and calls
    vla.predict_action() directly. Image in, array out -- no bag, dataset or TF code.
"""
from __future__ import annotations

import sys
import types

# ---------------------------------------------------------------------------------------
# CRITICAL, AND FIRST: prismatic/vla/constants.py picks the robot platform by scanning
# sys.argv for a keyword. Without "signway" it falls back to LIBERO (ACTION_DIM=7) and the
# 16-dim action head fails to load with a size mismatch. Must happen before ANY prismatic
# import, including the indirect ones below. Safe to import this module standalone.
# ---------------------------------------------------------------------------------------
if not any("signway" in a.lower() for a in sys.argv):
    sys.argv.append("--signway")

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

OPENVLA_IMAGE_SIZE = 224
CROP_SCALE = 0.9                       # center-crop area fraction, as in training
VALID_PROMPTS = ("straight", "turn_left", "turn_right", "stop")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class _TFUnavailable:
    """Placeholder standing in for any tensorflow attribute.

    Usable in a type annotation (that is all we need it for), but raises loudly if any
    code actually tries to CALL into tensorflow, so a silent wrong-preprocessing path
    can never happen -- it would be an immediate, named error instead.
    """

    def __getattr__(self, name):          # tf.image.resize -> still a placeholder
        return self

    def __call__(self, *a, **k):
        raise RuntimeError(
            "tensorflow is deliberately not installed on this Jetson. If you hit this, "
            "inference reached the repo's TF preprocessing instead of V8Policy._preprocess."
        )


def _preseed_tensorflow_stub() -> None:
    """Let openvla_utils import without tensorflow present.

    openvla_utils already guards the import (`try: import tensorflow as tf / except
    ImportError: tf = None`), but that guard is incomplete: line 549 reads

        def crop_and_resize(image: tf.Tensor, ...) -> tf.Tensor:

    and a bare annotation is EVALUATED at function-definition time, so `tf = None`
    raises AttributeError('NoneType' has no attribute 'Tensor') at import. It is the only
    tf reference evaluated at import time; the other 16 are inside function bodies.

    Rather than patch the repo, we pre-seed sys.modules["tensorflow"] with a stub so the
    annotation resolves. Inference-safe: we never call the repo's TF preprocessing --
    V8Policy._preprocess reimplements both steps with PIL + torch -- and if anything ever
    did call it, the stub raises instead of silently preprocessing differently.
    """
    if "tensorflow" in sys.modules:
        return
    import importlib.machinery

    mod = types.ModuleType("tensorflow")
    mod.__getattr__ = lambda name: _TFUnavailable()      # PEP 562 module-level __getattr__
    # A bare ModuleType has __spec__ = None, and importlib.util.find_spec() raises
    # ValueError on such a module. transformers probes exactly that, so give the stub a
    # real spec. This does NOT make transformers take its TF code paths: it decides via
    # importlib.metadata.version("tensorflow"), which still finds no installed
    # distribution, so _tf_available stays False. Verified on this box.
    mod.__spec__ = importlib.machinery.ModuleSpec("tensorflow", None)
    sys.modules["tensorflow"] = mod


def _preseed_rlds_stub() -> None:
    """Let experiments.robot.openvla_utils import without tensorflow/dlimp.

    openvla_utils line 37 does:
        from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
    That module is a *re-export*: NormalizationType is DEFINED in prismatic.vla.constants
    (data_utils itself does `from prismatic.vla.constants import NormalizationType`).
    Importing it, however, executes prismatic/vla/datasets/__init__.py, which eagerly
    pulls in the whole RLDS training pipeline -> `import dlimp as dl` + tensorflow.
    Both are training-only and unavailable on this platform.

    So we pre-seed sys.modules with empty stand-ins for those four package levels and
    hang the GENUINE enum object (same identity, from constants) off the leaf. Python
    then finds them already "imported" and never executes the real RLDS modules.

    Inference-safe: the only thing openvla_utils uses from that import is
    NormalizationType, for the BOUNDS vs BOUNDS_Q99 comparison in normalize_proprio.
    We get the identical enum, so every comparison behaves identically. Nothing on the
    image->action path touches dataset code. The repo is left unmodified.
    """
    from prismatic.vla.constants import NormalizationType  # TF-free, safe

    chain = (
        "prismatic.vla.datasets",
        "prismatic.vla.datasets.rlds",
        "prismatic.vla.datasets.rlds.utils",
        "prismatic.vla.datasets.rlds.utils.data_utils",
    )
    for name in chain:
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__path__ = []                      # declare it a package so submodules resolve
            sys.modules[name] = mod
            parent, _, leaf = name.rpartition(".")
            setattr(sys.modules[parent], leaf, mod)
    sys.modules[chain[-1]].NormalizationType = NormalizationType


class V8Policy:
    """Loads the backbone + action head ONCE; predict() is the per-frame hot path."""

    def __init__(self, checkpoint: str, unnorm_key: str = "signway_dataset",
                 center_crop: bool = True, jpeg_roundtrip: bool = True):
        # Order matters. _preseed_rlds_stub() imports prismatic.vla.constants, which drags
        # in transformers, which probes for tensorflow exactly once at import. Let that probe
        # run BEFORE the stub exists so transformers settles on "no TF" the honest way; only
        # then install the stub that openvla_utils' import-time annotation needs.
        _preseed_rlds_stub()
        _preseed_tensorflow_stub()
        from experiments.robot.openvla_utils import get_action_head, get_processor, get_vla
        from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK, PROPRIO_DIM

        assert NUM_ACTIONS_CHUNK == 1, f"expected one-shot trajectory, got chunk {NUM_ACTIONS_CHUNK}"
        assert ACTION_DIM == 16, f"expected ACTION_DIM=16, got {ACTION_DIM} (argv keyword lost?)"
        assert PROPRIO_DIM == 0, f"expected no proprio, got PROPRIO_DIM={PROPRIO_DIM}"

        self.horizon = ACTION_DIM // 2                      # 16 numbers -> 8 (x,y) waypoints
        self.unnorm_key = unnorm_key
        self.center_crop = center_crop
        self.jpeg_roundtrip = jpeg_roundtrip

        # cfg mirrors eval_openloop.load_policy() exactly (what the get_* helpers read).
        self.cfg = types.SimpleNamespace(
            model_family="openvla",
            pretrained_checkpoint=checkpoint,
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
            num_images_in_input=1,
            use_proprio=False,                              # PROPRIO_DIM=0: no projector
            load_in_8bit=False,
            load_in_4bit=False,
            center_crop=center_crop,
            num_open_loop_steps=1,
            unnorm_key=unnorm_key,
            lora_rank=32,
        )

        self.vla = get_vla(self.cfg)                        # bf16, .eval(), .to(cuda) inside
        self.processor = get_processor(self.cfg)
        self.action_head = get_action_head(self.cfg, llm_dim=self.vla.llm_dim)

        # Fail loudly at construction rather than on the first frame.
        assert hasattr(self.vla, "norm_stats") and unnorm_key in self.vla.norm_stats, (
            f"unnorm_key {unnorm_key!r} not in checkpoint dataset_statistics.json; "
            f"available: {list(getattr(self.vla, 'norm_stats', {}))}"
        )
        assert self.vla.get_action_dim(unnorm_key) == ACTION_DIM

        # Attention backend: sdpa. The config may already carry it; set it where the
        # loaded modules actually read it so we don't silently fall back to eager.
        try:
            self.vla.config._attn_implementation = "sdpa"
        except Exception:
            pass

        self._grid = self._build_crop_grid() if center_crop else None

    # ---------------- preprocessing: exact TF-free equivalents ----------------

    def _build_crop_grid(self) -> torch.Tensor:
        """Precompute the sampling grid for the center crop (built once, reused every frame).

        Replicates tf.image.crop_and_resize(bilinear) as used by crop_and_resize() with
        crop_scale=0.9. TF's box is in normalized coords with sqrt(crop_scale) side length:
            side = sqrt(0.9);  offset = (1 - side)/2
        and TF samples source pixel
            c_i = offset*(H-1) + i * side*(H-1)/(out-1)
        With H == out == 224 this is c_i = offset*223 + i*side. grid_sample(align_corners=
        True) maps g in [-1,1] to pixel (g+1)/2*(H-1), so g_i = 2*c_i/(H-1) - 1.
        """
        n = OPENVLA_IMAGE_SIZE
        side = float(np.sqrt(CROP_SCALE))
        offset = (1.0 - side) / 2.0
        c = offset * (n - 1) + torch.arange(n, dtype=torch.float32) * side
        g = 2.0 * c / (n - 1) - 1.0
        yy, xx = torch.meshgrid(g, g, indexing="ij")        # square crop: same coords both axes
        return torch.stack((xx, yy), dim=-1).unsqueeze(0).to(DEVICE)   # (1,224,224,2)

    def _preprocess(self, image: Image.Image) -> Image.Image:
        """full-res RGB -> 224x224 PIL, matching resize_image_for_policy + center_crop_image."""
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        if image.mode != "RGB":
            image = image.convert("RGB")

        # (1) The RLDS pipeline stored frames as JPEG, so resize_image_for_policy does a
        # JPEG encode/decode round-trip before resizing. Kept for distribution match; it
        # is the single most expensive preprocessing step, so it can be switched off.
        if self.jpeg_roundtrip:
            import io
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=95)
            buf.seek(0)
            image = Image.open(buf).convert("RGB")

        # (2) tf.image.resize(method="lanczos3", antialias=True) -> PIL LANCZOS is the
        # same 3-lobe windowed sinc with antialiased downsampling.
        if image.size != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE):
            image = image.resize((OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE), Image.LANCZOS)

        # (3) center crop 90% area and resize back to 224, bilinear, on the GPU.
        if not self.center_crop:
            return image
        # np.array (not asarray): PIL exposes a read-only buffer and torch warns on it.
        t = torch.from_numpy(np.array(image, dtype=np.uint8)).to(DEVICE)
        t = t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)         # (1,3,224,224) in [0,1]
        t = F.grid_sample(t, self._grid, mode="bilinear",
                          padding_mode="border", align_corners=True)
        # TF converts float->uint8 with round + saturate; match it so the processor sees
        # byte-identical input to what training produced.
        t = t.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
        arr = t[0].permute(1, 2, 0).contiguous().cpu().numpy()
        return Image.fromarray(arr)

    # ---------------- the hot path ----------------

    @torch.inference_mode()
    def predict(self, image: Image.Image, prompt: str = "straight") -> np.ndarray:
        """One forward pass -> (8,2) float32 cumulative waypoints (x_fwd, y_left) in metres."""
        if prompt not in VALID_PROMPTS:
            raise ValueError(f"prompt must be one of {VALID_PROMPTS}, got {prompt!r}")

        pil = self._preprocess(image)

        # Prompt text is built exactly as get_vla_action does; our label IS the prompt
        # string with underscores spaced out ("turn_left" -> "turn left").
        task_label = prompt.replace("_", " ")
        text = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

        inputs = self.processor(text, pil).to(DEVICE, dtype=torch.bfloat16)
        actions, _ = self.vla.predict_action(
            **inputs,
            unnorm_key=self.unnorm_key,
            do_sample=False,
            proprio=None,
            proprio_projector=None,
            noisy_action_projector=None,
            action_head=self.action_head,
            use_film=False,
        )
        # (16,) already in metres (BOUNDS_Q99 un-normalized inside predict_action).
        a = np.asarray(actions, dtype=np.float32).reshape(-1)
        return a.reshape(-1, 2)[: self.horizon]


def load_policy(checkpoint: str = "/home/hunter/weights/v8_vla", **kw) -> V8Policy:
    """Load the policy once. Returns an object with .predict(pil_image, prompt) -> (8,2)."""
    return V8Policy(checkpoint, **kw)


if __name__ == "__main__":
    import time

    ckpt = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else \
        "/home/hunter/weights/v8_vla"
    t0 = time.time()
    pol = load_policy(ckpt)
    print(f"policy loaded in {time.time() - t0:.1f}s, "
          f"mem {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    img = Image.fromarray(np.full((1280, 1920, 3), 128, dtype=np.uint8))
    for p in VALID_PROMPTS:
        t0 = time.time()
        wp = pol.predict(img, p)
        dt = time.time() - t0
        print(f"{p:11s} {dt * 1e3:7.1f} ms  shape={wp.shape} "
              f"wp0=({wp[0][0]:+.3f},{wp[0][1]:+.3f}) wp7=({wp[-1][0]:+.3f},{wp[-1][1]:+.3f})")
