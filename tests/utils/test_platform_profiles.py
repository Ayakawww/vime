"""CPU contracts for the tests/platform_profiles layer.

Everything here runs on plain stdlib (no vime/torch/hardware required), so the
invariants of the unified test framework are enforced in every CI environment:

  * discovery: one file per platform under tests/platform_profiles/profiles/
  * cuda neutrality: shared scripts render byte-identical args on CUDA
  * profile content: the validated NPU/ROCm recipes stay pinned
  * selection: VIME_TEST_DEVICE / VIME_PLATFORM / auto-detection / HIP refine
  * golden replay: tests/test_qwen3_4B_ckpt.py produces exactly the
    pre-migration GPU command line, and the expected NPU variant
  * lint: shared scripts never compare against platform names
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

TESTS_DIR = Path(__file__).resolve().parents[1]
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import platform_profiles as pp
from platform_profiles import LaunchSpec, PlatformProfile

CKPT_SCRIPT = TESTS_DIR / "test_qwen3_4B_ckpt.py"


@pytest.fixture(autouse=True)
def _fresh_profile_cache():
    pp.reset_profile_cache()
    yield
    pp.reset_profile_cache()


# ── Discovery ────────────────────────────────────────────────────────────────


def test_profiles_discovered_from_profiles_dir():
    names = pp.available_profiles()
    assert {"cuda", "npu", "rocm"} <= set(names)
    for name in names:
        assert (pp.PROFILE_DIR / f"{name}.py").exists(), name


def test_profile_names_match_filenames():
    for name in pp.available_profiles():
        assert pp.get_profile(name).name == name


# ── cuda neutrality (backward compatibility with every existing GPU test) ───


def test_cuda_profile_is_neutral():
    p = pp.get_profile("cuda")
    assert p.train_args_extra == ""
    assert p.tuning == {}
    assert p.topology == {}
    assert p.env == {}
    assert p.asset_root_env_vars == ()
    assert p.tmp_under_asset_root is False
    assert p.launch_platform == "cuda"
    assert p.torch_dist_convert is True
    # every accessor passes the GPU-blessed default through unchanged
    assert p.tune("any_knob", 42) == 42
    assert p.flag("any-flag", "any_knob", 0.8) == "--any-flag 0.8 "
    assert p.flag("any-flag", "any_knob") == ""
    assert p.topo("tp", 2) == 2
    assert p.supports("vllm_sleep_mode") is False


def test_cuda_asset_root_ignores_hf_home(monkeypatch):
    # GPU tests always used /root even though GPU CI exports HF_HOME.
    monkeypatch.setenv("HF_HOME", "/root/.cache/huggingface")
    monkeypatch.delenv("VIME_TEST_ROOT", raising=False)
    assert pp.get_profile("cuda").asset_root() == "/root"


# ── Profile content pinning ──────────────────────────────────────────────────


def test_npu_profile_pins_validated_configuration():
    p = pp.get_profile("npu")
    assert p.launch_platform == "npu"
    assert p.torch_dist_convert is False  # native HF load, no conversion step
    extra = p.train_args_extra
    assert "--use-flash-attn " in extra
    assert "--no-gradient-accumulation-fusion " in extra
    m = re.search(r"--vllm-additional-config '([^']+)'", extra)
    assert m and json.loads(m.group(1)) == {"weight_nz_mode": 0}
    assert p.tuning["max_tokens_per_gpu"] == 8192
    assert p.tuning["log_probs_chunk_size"] == 128
    assert p.tuning["vllm_gpu_memory_utilization"] == 0.5
    assert p.tuning["vllm_max_num_seqs"] == 16
    assert p.tuning["vllm_max_cudagraph_capture_size"] == 8
    assert p.tuning["vllm_max_model_len"] == 32768
    assert p.supports("vllm_sleep_mode")
    assert p.tmp_dir() == f"{p.asset_root()}/tmp"


def test_npu_asset_root_priority(monkeypatch):
    p = pp.get_profile("npu")
    monkeypatch.delenv("VIME_TEST_ROOT", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    assert p.asset_root() == "/root"
    monkeypatch.setenv("HF_HOME", "/hf-cache")
    assert p.asset_root() == "/hf-cache"
    monkeypatch.setenv("VIME_TEST_ROOT", "/explicit")
    assert p.asset_root() == "/explicit"  # VIME_TEST_ROOT wins


def test_rocm_profile_rides_the_cuda_launch_path():
    p = pp.get_profile("rocm")
    assert p.launch_platform == "cuda"
    assert p.train_args_extra.strip() == "--no-gradient-accumulation-fusion"
    assert p.tune("vllm_gpu_memory_utilization", 0.8) == 0.3
    assert p.tune("update_weight_transport", "nccl") == "disk"


def test_torch_dist_convert_matches_launch_registry():
    try:
        from vime.utils.external_utils.launch import PLATFORMS
    except Exception as exc:  # noqa: BLE001 - vime/torch not installed here
        pytest.skip(f"vime launch registry not importable here: {exc}")
    for name in pp.available_profiles():
        profile = pp.get_profile(name)
        launch = PLATFORMS.get(profile.launch_platform)
        if launch is None:
            assert profile.launch_def is not None, f"{name}: unknown launch platform without a launch_def"
            continue
        assert profile.torch_dist_convert == launch.torch_dist_convert, name


# ── Accessor semantics ───────────────────────────────────────────────────────


def test_flag_and_tune_semantics():
    p = PlatformProfile(name="synthetic", tuning={"a": 1, "b": True, "c": None, "d": 0.5})
    assert p.flag("flag-a", "a") == "--flag-a 1 "
    assert p.flag("flag-b", "b") == "--flag-b "  # boolean renders as a bare flag
    assert p.flag("flag-c", "c") == ""  # explicit None omits the flag
    assert p.flag("flag-c", "c", 0.9) == ""  # None overrides a supplied default
    assert p.flag("flag-d", "d", 0.9) == "--flag-d 0.5 "  # override wins
    assert p.flag("flag-e", "e") == ""  # absent without default omits the flag
    assert p.flag("flag-e", "e", 16) == "--flag-e 16 "
    with pytest.raises(KeyError):
        p.tune("e")
    assert p.tune("e", 7) == 7


def test_topo_and_supports():
    p = PlatformProfile(name="synthetic", topology={"tp": 4}, capabilities={"vllm_sleep_mode": True})
    assert p.topo("tp", 2) == 4
    assert p.topo("cp", 2) == 2
    with pytest.raises(KeyError):
        p.topo("ep")
    assert p.supports("vllm_sleep_mode") is True
    assert p.supports("unknown") is False


def test_launch_def_auto_registers_into_launch_registry(monkeypatch):
    fake = ModuleType("vime.utils.external_utils.launch")

    class FakeLaunchPlatform:
        def __init__(self, name, ray_args, env, torch_dist_convert):
            self.name, self.ray_args = name, ray_args
            self.env, self.torch_dist_convert = env, torch_dist_convert

    fake.PLATFORMS = {}
    fake.Platform = FakeLaunchPlatform
    monkeypatch.setitem(sys.modules, "vime.utils.external_utils.launch", fake)

    profile = PlatformProfile(
        name="gaudi-like",
        launch_platform="gaudi-like",
        launch_def=LaunchSpec(ray_args="--num-gpus 0 --resources '{\"GAUDI\": {n}}'", torch_dist_convert=False),
    )
    pp._register_launch_spec(profile)

    assert "gaudi-like" in fake.PLATFORMS
    registered = fake.PLATFORMS["gaudi-like"]
    assert registered.torch_dist_convert is False
    assert "{n}" in registered.ray_args


# ── Selection ────────────────────────────────────────────────────────────────


def test_selection_via_vime_test_device(monkeypatch):
    monkeypatch.setenv("VIME_TEST_DEVICE", "rocm")
    assert pp.current_profile().name == "rocm"


def test_selection_via_vime_platform(monkeypatch):
    monkeypatch.setattr(pp, "_resolve_core_platform_name", lambda: "npu")
    assert pp.current_profile().name == "npu"


def test_cuda_refines_to_rocm_on_hip(monkeypatch):
    monkeypatch.delenv("VIME_TEST_DEVICE", raising=False)
    monkeypatch.setattr(pp, "_resolve_core_platform_name", lambda: "cuda")
    monkeypatch.setattr(pp, "_is_hip", lambda: True)
    assert pp.current_profile().name == "rocm"


def test_cuda_stays_cuda_without_hip(monkeypatch):
    monkeypatch.delenv("VIME_TEST_DEVICE", raising=False)
    monkeypatch.setattr(pp, "_resolve_core_platform_name", lambda: "cuda")
    monkeypatch.setattr(pp, "_is_hip", lambda: False)
    assert pp.current_profile().name == "cuda"


def test_unknown_profile_has_clear_error(monkeypatch):
    monkeypatch.setenv("VIME_TEST_DEVICE", "does-not-exist")
    with pytest.raises(ValueError, match="available profiles"):
        pp.current_profile()


# ── Golden replay of tests/test_qwen3_4B_ckpt.py ────────────────────────────


@pytest.fixture
def command_utils_stub(monkeypatch):
    """Recording stub for the two vime modules the ckpt script touches; works
    whether or not the real vime package is importable in this environment."""
    recorded = SimpleNamespace(commands=[], downloads=[], converted=[], train_args=[])
    stub = ModuleType("vime.utils.external_utils.command_utils")

    def _convert_checkpoint(**kwargs):
        # mirrors the real guard: platforms without torch_dist conversion skip it
        if pp.current_profile().torch_dist_convert:
            recorded.converted.append(kwargs)

    stub.exec_command = lambda cmd, capture_output=False: (recorded.commands.append(cmd), "")[1]
    stub.hf_download_dataset = lambda full_name: recorded.downloads.append(full_name)
    stub.convert_checkpoint = _convert_checkpoint
    stub.get_default_wandb_args = lambda *args, **kwargs: ""
    stub.execute_train = lambda **kwargs: recorded.train_args.append(kwargs["train_args"])
    misc = ModuleType("vime.utils.misc")
    misc.exec_command = stub.exec_command
    for name, module in (
        ("vime", ModuleType("vime")),
        ("vime.utils", ModuleType("vime.utils")),
        ("vime.utils.external_utils", ModuleType("vime.utils.external_utils")),
    ):
        module.__path__ = []
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "vime.utils.external_utils.command_utils", stub)
    monkeypatch.setitem(sys.modules, "vime.utils.misc", misc)
    return recorded


def _load_ckpt_script():
    spec = importlib.util.spec_from_file_location("replay_test_qwen3_4B_ckpt", CKPT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replay(monkeypatch, profile_name, mode="save", optimizer="cpu", checkpoint_dir="/root/models/GOLDEN"):
    monkeypatch.setattr(pp, "_resolve_core_platform_name", lambda: profile_name)
    monkeypatch.setattr(pp, "_is_hip", lambda: False)
    for var in ("VIME_TEST_DEVICE", "VIME_PLATFORM", "VIME_TEST_ROOT", "HF_HOME"):
        monkeypatch.delenv(var, raising=False)
    if profile_name == "npu":
        monkeypatch.setenv("VIME_TEST_ROOT", "/npu-assets")
    pp.reset_profile_cache()
    module = _load_ckpt_script()
    module.execute(mode, optimizer=optimizer, checkpoint_dir=checkpoint_dir)
    return module


# The pre-migration GPU script's argument blocks, verbatim (git: ascend@5397545d).
_ROLLOUT_ARGS = (
    "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
    "--input-key prompt --label-key label --apply-chat-template --rollout-shuffle "
    "--rm-type deepscaler --num-rollout 2 --rollout-batch-size 4 --n-samples-per-prompt 4 "
    "--rollout-max-response-len 1024 --rollout-temperature 0.8 --global-batch-size 16 --balance-data "
)
_OPTIMIZER_ARGS = (
    "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 "
    "--adam-beta1 0.9 --adam-beta2 0.98 --use-precision-aware-optimizer "
)
_PPO_ARGS = "--advantage-estimator grpo --kl-loss-coef 0.00 --kl-loss-type k1 --kl-coef 0.00 --entropy-coef 0.00 --eps-clip 0.2 "
_PERF_ARGS = (
    "--tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 "
    "--context-parallel-size 2 --recompute-granularity full --recompute-method uniform "
    "--recompute-num-layers 1 --use-dynamic-batch-size --max-tokens-per-gpu 16384 "
)
_VLLM_ARGS = "--rollout-num-gpus-per-engine 2 --vllm-gpu-memory-utilization 0.8 --vllm-max-cudagraph-capture-size 16 "
_CI_ARGS = "--ci-test "
_MISC_ARGS = (
    "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
    "--attention-softmax-in-fp32 --attention-backend flash --actor-num-nodes 1 "
    "--actor-num-gpus-per-node 8 --colocate "
)


def _pre_migration_gpu_train_args(mode, optimizer, checkpoint_dir="/root/models/GOLDEN"):
    ckpt_tail = {
        "save": f"--save {checkpoint_dir} --save-interval 2 ",
        "async_save": f"--save {checkpoint_dir} --save-interval 2 --async-save ",
        "load": f"--load {checkpoint_dir} --ckpt-step 1 ",
    }[mode]
    ckpt_args = f"--hf-checkpoint /root/models/Qwen3-4B/ --ref-load /root/models/Qwen3-4B_torch_dist {ckpt_tail}"
    opt_args = _OPTIMIZER_ARGS + (
        "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d " if optimizer == "cpu" else ""
    )
    wandb_args = ""  # get_default_wandb_args returns "" without WANDB_API_KEY
    # mirrors the original f-string join: each block followed by one space,
    # including the empty wandb slot (hence the double spaces between blocks)
    return f"{ckpt_args} {_ROLLOUT_ARGS} {opt_args} {_PPO_ARGS} {wandb_args} {_PERF_ARGS} {_VLLM_ARGS} {_CI_ARGS} {_MISC_ARGS} "


@pytest.mark.parametrize("mode", ["save", "async_save", "load"])
@pytest.mark.parametrize("optimizer", ["cpu", "gpu"])
def test_ckpt_script_is_byte_identical_to_pre_migration_gpu(command_utils_stub, monkeypatch, mode, optimizer):
    _replay(monkeypatch, "cuda", mode=mode, optimizer=optimizer)
    assert command_utils_stub.train_args == [_pre_migration_gpu_train_args(mode, optimizer)]


def test_ckpt_script_prepare_is_byte_identical_on_gpu(command_utils_stub, monkeypatch):
    module = _replay(monkeypatch, "cuda")
    module.prepare("/root/models/GOLDEN")
    assert command_utils_stub.commands == [
        "mkdir -p /root/models /root/datasets",
        "hf download Qwen/Qwen3-4B --local-dir /root/models/Qwen3-4B",
        "rm -rf /root/models/GOLDEN",
        "hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k",
        "hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/datasets/aime-2024",
    ]
    assert command_utils_stub.converted == [
        {
            "model_name": "Qwen3-4B",
            "megatron_model_type": "qwen3-4B",
            "num_gpus_per_node": 8,
            "dir_dst": "/root/models",
        }
    ]


def test_ckpt_script_adapts_to_npu_profile(command_utils_stub, monkeypatch):
    module = _replay(monkeypatch, "npu", checkpoint_dir="/npu-assets/models/GOLDEN")
    module.prepare("/npu-assets/models/GOLDEN")
    args = command_utils_stub.train_args[0]

    # paths move under the NPU asset root; no /root leakage
    assert args.startswith("--hf-checkpoint /npu-assets/models/Qwen3-4B/ --ref-load /npu-assets/models/Qwen3-4B ")
    assert "/root/models" not in args
    assert "/root/datasets" not in args
    # native HF load: ref-load points at the HF checkpoint, no torch_dist
    assert "torch_dist" not in args
    assert command_utils_stub.converted == []
    assert command_utils_stub.commands[0] == "mkdir -p /npu-assets/models /npu-assets/datasets"
    # memory bounds and launch flags from the profile
    assert "--max-tokens-per-gpu 8192 " in args
    assert "--vllm-gpu-memory-utilization 0.5 " in args
    assert "--vllm-max-cudagraph-capture-size 8 " in args
    assert "--vllm-max-num-seqs 16 " in args
    assert "--vllm-max-model-len 32768 " in args
    assert "--use-flash-attn " in args
    assert "--no-gradient-accumulation-fusion " in args
    assert "--vllm-additional-config '{\"weight_nz_mode\": 0}' " in args
    # test intent is unchanged (algorithm, dataset, rollout, parallel layout)
    assert "--rm-type deepscaler --num-rollout 2 " in args
    assert _PPO_ARGS in args
    assert "--tensor-model-parallel-size 2 " in args
    assert "--context-parallel-size 2 " in args
    assert "--rollout-num-gpus-per-engine 2 " in args


def test_ckpt_script_default_dir_follows_profile(command_utils_stub, monkeypatch):
    monkeypatch.setattr(pp, "_resolve_core_platform_name", lambda: "npu")
    monkeypatch.setattr(pp, "_is_hip", lambda: False)
    monkeypatch.setenv("VIME_TEST_ROOT", "/npu-assets")
    pp.reset_profile_cache()
    module = _load_ckpt_script()
    args = SimpleNamespace(async_save=True, save_optimizer="cpu", load_optimizer="gpu")
    assert module.default_checkpoint_dir(args) == "/npu-assets/models/Qwen3-4B_vime_async_cpu_save_gpu_load"


# ── Lint: platform names never appear in shared script logic ────────────────

_FORBIDDEN_NAME_COMPARISON = re.compile(r"[=!]=\s*[\"'](npu|cuda|rocm|gaudi)[\"']")
_FORBIDDEN_PLATFORM_EXPORT = re.compile(r"os\.environ\s*\[\s*[\"']VIME_PLATFORM[\"']\s*\]\s*=")


def test_shared_scripts_do_not_reference_platform_names():
    shared = [path for path in TESTS_DIR.glob("test_*.py") if "platform_profiles" in path.read_text(encoding="utf-8")]
    assert CKPT_SCRIPT in shared, "expected the migrated ckpt script to consume platform_profiles"
    offenders = []
    for path in shared:
        src = path.read_text(encoding="utf-8")
        if _FORBIDDEN_NAME_COMPARISON.search(src):
            offenders.append(f"{path.name}: platform-name comparison")
        if _FORBIDDEN_PLATFORM_EXPORT.search(src):
            offenders.append(f"{path.name}: sets VIME_PLATFORM")
    assert offenders == []
