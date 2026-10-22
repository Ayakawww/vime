"""CPU contracts for the tests/platform_profiles layer.

Everything here runs on plain stdlib (no vime/torch/hardware required), so the
invariants of the unified test framework are enforced in every CI environment:

  * discovery: one file per platform under tests/platform_profiles/profiles/
  * cuda neutrality: empty topology/tuning/env, so shared scripts render
    byte-identical args on CUDA (backward compatibility for all GPU suites)
  * profile content: the validated NPU/ROCm recipes stay pinned (topology,
    tuning bounds, capabilities)
  * selection: VIME_TEST_DEVICE / VIME_PLATFORM / auto-detection / HIP refine
  * golden replay: tests/test_qwen3_4B_ckpt.py and
    tests/test_full_disk_weight_update.py produce exactly the pre-migration
    GPU command lines, and the expected NPU variants
  * observability: P.dump() and validation_report() render the resolved
    configuration and the override accounting
  * hardware guards: visible_device_count/require_devices read the platform's
    visibility env and fail fast with a clear message
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


def test_cuda_topology_is_neutral():
    # CUDA keeps upstream layouts byte-identical: no topology overrides exist,
    # and every topo query falls through to the test's GPU-blessed default.
    p = pp.get_profile("cuda")
    assert p.topology == {}
    assert p.topo("num_gpus", 8) == 8
    assert p.topo("tp", 2) == 2
    assert p.topo("cp", 2) == 2
    assert p.topo("min_gpus", 8) == 8
    assert p.dump()  # dump stays usable on the neutral profile


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
    assert p.tuning["rollout_num_gpus"] == 4
    assert p.tuning["full_disk_max_tokens_per_gpu"] == 2048
    assert p.tuning["full_disk_vllm_max_model_len"] == 8192
    assert p.supports("vllm_sleep_mode")
    assert not p.supports("health_monitor")  # /health_generate 404 on vllm-ascend today
    assert p.tmp_dir() == f"{p.asset_root()}/tmp"


def test_npu_topology_pins_validated_4_card_layout():
    # The layout validated on 910B3 (A2): 4-card colocate, TP4/CP1 — same
    # shape as the NPU smoke test. Scenario-scoped needs use prefixed keys.
    p = pp.get_profile("npu")
    assert p.topology == {"num_gpus": 4, "tp": 4, "cp": 1}
    assert p.visible_devices_env == "ASCEND_RT_VISIBLE_DEVICES"
    assert p.hardware_names["ascend910b3"] == "ascend910b3 / A2"
    assert p.hardware_names["ascend910c"] == "ascend910c / A3"


def test_health_monitor_capability_split():
    # cuda states the feature exists (GPU behavior unchanged); npu declares it
    # absent so tests gate health-monitored fault tolerance on supports().
    assert pp.get_profile("cuda").supports("health_monitor") is True
    assert pp.get_profile("npu").supports("health_monitor") is False


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
    """Recording stub for the two vime modules the shared scripts touch; works
    whether or not the real vime package is importable in this environment."""
    recorded = SimpleNamespace(commands=[], downloads=[], converted=[], train_args=[], trains=[])
    stub = ModuleType("vime.utils.external_utils.command_utils")

    def _convert_checkpoint(**kwargs):
        # mirrors the real guard: platforms without torch_dist conversion skip it
        if pp.current_profile().torch_dist_convert:
            recorded.converted.append(kwargs)

    stub.exec_command = lambda cmd, capture_output=False: (recorded.commands.append(cmd), "")[1]
    stub.hf_download_dataset = lambda full_name: recorded.downloads.append(full_name)
    stub.convert_checkpoint = _convert_checkpoint
    stub.get_default_wandb_args = lambda *args, **kwargs: ""

    def _execute_train(**kwargs):
        recorded.trains.append(kwargs)
        recorded.train_args.append(kwargs["train_args"])

    stub.execute_train = _execute_train
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


def _load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _select_profile(monkeypatch, profile_name, extra_env=None):
    monkeypatch.setattr(pp, "_resolve_core_platform_name", lambda: profile_name)
    monkeypatch.setattr(pp, "_is_hip", lambda: False)
    for var in ("VIME_TEST_DEVICE", "VIME_PLATFORM", "VIME_TEST_ROOT", "HF_HOME", "VIME_TEST_NUM_ROLLOUT"):
        monkeypatch.delenv(var, raising=False)
    if profile_name == "npu":
        monkeypatch.setenv("VIME_TEST_ROOT", "/npu-assets")
    for key, value in (extra_env or {}).items():
        monkeypatch.setenv(key, value)
    pp.reset_profile_cache()


def _load_ckpt_script():
    return _load_script(CKPT_SCRIPT, "replay_test_qwen3_4B_ckpt")


def _replay(monkeypatch, profile_name, mode="save", optimizer="cpu", checkpoint_dir="/root/models/GOLDEN"):
    _select_profile(monkeypatch, profile_name)
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
    # the launch seam also sees the upstream GPU shape: 8 cards, TP2/CP2
    assert command_utils_stub.trains[0]["num_gpus_per_node"] == 8


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
    # the validated 4-card colocate topology from the profile
    assert "--tensor-model-parallel-size 4 " in args
    assert "--context-parallel-size 1 " in args
    assert "--rollout-num-gpus-per-engine 4 " in args
    assert "--rollout-num-gpus 4 " in args
    assert "--actor-num-gpus-per-node 4 " in args
    assert command_utils_stub.trains[0]["num_gpus_per_node"] == 4
    # test intent is unchanged (algorithm, dataset, rollout)
    assert "--rm-type deepscaler --num-rollout 2 " in args
    assert _PPO_ARGS in args


def test_ckpt_script_default_dir_follows_profile(command_utils_stub, monkeypatch):
    monkeypatch.setattr(pp, "_resolve_core_platform_name", lambda: "npu")
    monkeypatch.setattr(pp, "_is_hip", lambda: False)
    monkeypatch.setenv("VIME_TEST_ROOT", "/npu-assets")
    pp.reset_profile_cache()
    module = _load_ckpt_script()
    args = SimpleNamespace(async_save=True, save_optimizer="cpu", load_optimizer="gpu")
    assert module.default_checkpoint_dir(args) == "/npu-assets/models/Qwen3-4B_vime_async_cpu_save_gpu_load"


# ── Golden replay of tests/test_full_disk_weight_update.py ──────────────────

FULL_DISK_SCRIPT = TESTS_DIR / "test_full_disk_weight_update.py"

# The pre-migration GPU script's argument blocks, verbatim (git: ascend@5397545d).
_FD_ROLLOUT_ARGS = (
    "--prompt-data /root/datasets/gsm8k/train.parquet --input-key messages --label-key label "
    "--apply-chat-template --rollout-shuffle --rm-type math --num-rollout 1 --rollout-batch-size 4 "
    "--n-samples-per-prompt 4 --rollout-max-response-len 1024 --rollout-temperature 0.8 "
    "--over-sampling-batch-size 8 "
    "--dynamic-sampling-filter-path vime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
    "--global-batch-size 16 "
)
_FD_OPTIMIZER_ARGS = (
    "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
)
_FD_GRPO_ARGS = (
    "--advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl "
    "--entropy-coef 0.01 --eps-clip 0.2 --eps-clip-high 0.28 "
)
_FD_PERF_ARGS = (
    "--tensor-model-parallel-size 1 --sequence-parallel --pipeline-model-parallel-size 1 "
    "--context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
    "--use-dynamic-batch-size --max-tokens-per-gpu 9216 "
)
_FD_VLLM_ARGS = "--rollout-num-gpus-per-engine 1 --rollout-num-gpus 3 --vllm-gpu-memory-utilization 0.7 --vllm-max-cudagraph-capture-size 32 "
_FD_CI_ARGS = "--ci-test "
_FD_MISC_ARGS = (
    "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
    "--attention-softmax-in-fp32 --attention-backend flash --loss-mask-type qwen3_5 "
    "--actor-num-nodes 1 --actor-num-gpus-per-node 1 "
)


def _full_disk_golden(disk_dir="/tmp/GOLDEN_DISK", num_rollout=1):
    ckpt_args = "--hf-checkpoint /root/models/Qwen3.5-0.8B/ --ref-load /dev/shm/Qwen3.5-0.8B_torch_dist "
    rollout_args = _FD_ROLLOUT_ARGS.replace("--num-rollout 1 ", f"--num-rollout {num_rollout} ")
    disk_args = f"--update-weight-mode full --update-weight-transport disk --update-weight-disk-dir {disk_dir} --update-weight-disk-keep-files "
    wandb_args = ""  # get_default_wandb_args returns "" without WANDB_API_KEY
    return (
        f"{ckpt_args} {_FD_ROLLOUT_ARGS if num_rollout == 1 else rollout_args} {_FD_OPTIMIZER_ARGS} "
        f"{_FD_GRPO_ARGS} {wandb_args} {_FD_PERF_ARGS} {_FD_VLLM_ARGS} {disk_args} {_FD_CI_ARGS} {_FD_MISC_ARGS} "
    )


def _replay_full_disk(monkeypatch, profile_name, extra_env=None):
    _select_profile(monkeypatch, profile_name, extra_env)
    module = _load_script(FULL_DISK_SCRIPT, "replay_test_full_disk_weight_update")
    return module.build_train_args("/tmp/GOLDEN_DISK")


def test_full_disk_script_is_byte_identical_to_pre_migration_gpu(command_utils_stub, monkeypatch):
    assert _replay_full_disk(monkeypatch, "cuda") == _full_disk_golden()


def test_full_disk_num_rollout_env_knob(command_utils_stub, monkeypatch):
    # VIME_TEST_NUM_ROLLOUT keeps the existing VIME_TEST_* convention: unset
    # renders the GPU-blessed value; NPU CI sets it to bound request load.
    args = _replay_full_disk(monkeypatch, "cuda", {"VIME_TEST_NUM_ROLLOUT": "2"})
    assert args == _full_disk_golden(num_rollout=2)
    assert "--num-rollout 2 " in args


def test_full_disk_script_adapts_to_npu_profile(command_utils_stub, monkeypatch):
    args = _replay_full_disk(monkeypatch, "npu")
    # paths under the NPU asset root; the /dev/shm torch_dist ref is swapped
    # for the native HF checkpoint (no conversion step on NPU)
    assert args.startswith(
        "--hf-checkpoint /npu-assets/models/Qwen3.5-0.8B/ --ref-load /npu-assets/models/Qwen3.5-0.8B "
    )
    assert "/dev/shm" not in args
    assert "/root/models" not in args
    assert "/root/datasets" not in args
    # the scenario-scoped NPU bounds (tighter than the 4B-class recipe)
    assert "--max-tokens-per-gpu 2048 " in args
    assert "--log-probs-chunk-size 128 " in args
    assert "--vllm-gpu-memory-utilization 0.5 " in args
    assert "--vllm-max-cudagraph-capture-size 8 " in args
    assert "--vllm-max-num-seqs 16 " in args
    assert "--vllm-max-model-len 8192 " in args
    assert "--use-flash-attn " in args
    # the non-colocate layout is identical on both platforms (TP1/CP1 stays;
    # the npu topology override only applies to tests that query it)
    assert "--tensor-model-parallel-size 1 " in args
    assert "--rollout-num-gpus-per-engine 1 " in args
    assert "--rollout-num-gpus 3 " in args
    assert "--actor-num-gpus-per-node 1 " in args
    # weight-sync intent unchanged
    assert "--update-weight-mode full --update-weight-transport disk " in args
    assert "--loss-mask-type qwen3_5 " in args


# ── Observability: P.dump() and validation_report() ─────────────────────────


def test_dump_shows_resolved_values_and_gpu_defaults(monkeypatch):
    monkeypatch.setenv("VIME_TEST_ROOT", "/npu-assets")
    p = pp.get_profile("npu")
    p.asset_root()
    assert p.tune("max_tokens_per_gpu", 16384) == 8192
    assert p.flag("vllm-max-num-seqs", "vllm_max_num_seqs") == "--vllm-max-num-seqs 16 "
    assert p.flag("vllm-max-model-len", "vllm_max_model_len", 16384) == "--vllm-max-model-len 32768 "
    assert p.topo("tp", 2) == 4
    assert p.topo("cp", 2) == 1
    assert p.supports("health_monitor") is False
    text = p.dump()
    assert "[NPU Profile] torch_dist_convert=False" in text
    assert "path: asset_root=/npu-assets (via VIME_TEST_ROOT)" in text
    assert "tune: max_tokens_per_gpu=8192 (GPU default: 16384)" in text
    assert "flag: --vllm-max-num-seqs=16 (GPU default: omit)" in text
    assert "flag: --vllm-max-model-len=32768 (GPU default: 16384)" in text
    assert "topology: tp=4 (GPU default: 2), cp=1 (GPU default: 2)" in text
    assert "capability: health_monitor=False" in text
    assert "train_args_extra:" in text and "--use-flash-attn" in text


def test_dump_shows_recipe_before_the_test_resolves_anything():
    text = pp.get_profile("npu").dump()
    assert "tune recipe (not yet resolved by this test): " in text
    assert "max_tokens_per_gpu=8192" in text
    assert "topology recipe (not yet resolved by this test): " in text
    assert "num_gpus=4" in text
    # the neutral profile dumps cleanly too
    cuda_text = pp.get_profile("cuda").dump()
    assert "[CUDA Profile] torch_dist_convert=True" in cuda_text


def test_validation_report_counts_overrides_and_skips(monkeypatch):
    monkeypatch.setenv("VIME_TEST_ROOT", "/npu-assets")
    monkeypatch.setenv("SOC_VERSION", "ascend910b3")
    p = pp.get_profile("npu")
    p.asset_root()
    p.tune("max_tokens_per_gpu", 16384)
    p.flag("vllm-max-num-seqs", "vllm_max_num_seqs")
    p.tune("log_probs_chunk_size", 9216)
    p.topo("num_gpus", 8)
    p.topo("tp", 2)
    p.topo("cp", 2)
    p.supports("health_monitor")  # declared False -> "not applicable"
    p.supports("vllm_sleep_mode")  # declared True -> not skipped

    text = pp.validation_report(model="Qwen3-4B (ckpt test)", notes="colocate", profile=p)
    assert "Platform: NPU (ascend910b3 / A2)" in text
    assert "Model: Qwen3-4B (ckpt test)" in text
    assert "Topology: num_gpus=4, tp=4, cp=1 (colocate)" in text
    assert "Profile overrides applied: 7 (paths:1, args:3, topology:3)" in text
    assert "Profile overrides skipped: 1 (not applicable on this platform: health_monitor)" in text
    assert "Result: PASSED" in text


def test_validation_report_hardware_fallback(monkeypatch):
    for var in ("SOC_VERSION", "VIME_TEST_ROOT", "HF_HOME"):
        monkeypatch.delenv(var, raising=False)
    text = pp.validation_report(model="m", profile=pp.get_profile("npu"))
    assert "Platform: NPU" in text and "A2" not in text
    cuda_text = pp.validation_report(model="m", profile=pp.get_profile("cuda"))
    assert "Platform: CUDA" in cuda_text
    assert "Profile overrides applied: 0" in cuda_text


# ── Hardware guards: visible_device_count / require_devices ─────────────────


def test_visible_device_count_reads_platform_env(monkeypatch):
    npu = pp.get_profile("npu")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
    assert pp.visible_device_count(npu) == 4
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0, 1 ,2")
    assert pp.visible_device_count(npu) == 3
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES")
    assert pp.visible_device_count(npu) is None


def test_require_devices_fails_fast_with_clear_message(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
    pp.require_devices(4, pp.get_profile("npu"))  # exactly enough -> passes
    with pytest.raises(SystemExit, match="exposes only 4"):
        pp.require_devices(8, pp.get_profile("npu"))
    # undeclared visibility (e.g. CUDA containers) cannot be verified -> no-op
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES")
    pp.require_devices(8, pp.get_profile("npu"))


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
