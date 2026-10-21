# vime unified test framework — platform profiles

One test script, every platform. This package gives vime's E2E test suite a
platform adaptation layer: test files express **intent** (model, algorithm,
dataset, parallel layout, assertions) and pull every platform-specific value
(路径 / 参数 / 环境变量 / 资源声明 / 拓扑) through declarative profiles.

```
tests/
├── platform_profiles/          ← this package (test infrastructure, stdlib-only)
│   ├── base.py                 ← PlatformProfile / LaunchSpec dataclasses
│   ├── __init__.py             ← discovery, selection, path/download helpers
│   └── profiles/               ← ONE FILE PER PLATFORM (auto-discovered)
│       ├── cuda.py             ← neutral reference (empty = upstream GPU behavior)
│       ├── npu.py              ← Ascend 910B3/910C, the primary validation target
│       └── rocm.py             ← AMD Instinct (initial support)
├── test_qwen3_4B_ckpt.py       ← first migrated feature test (ckpt save/load roundtrip)
└── utils/test_platform_profiles.py  ← CPU contract tests (no hardware needed)
```

## 1. Layered architecture

The framework rides on the platform seams that already exist; it does not
reimplement them:

```
┌────────────────────────────────────────────────────────────────────────┐
│ CI (Buildkite)                                                         │
│  gpu_suites.py (unchanged)      npu_suites.py "feature-*" suites       │
│  sets VIME_TEST_* env           sets VIME_PLATFORM=npu, resource_class │
└───────────────┬───────────────────────────┬────────────────────────────┘
                │                           │
                ▼                           ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Test script (e.g. test_qwen3_4B_ckpt.py)  — platform names NEVER here  │
│   P = current_profile()                                                │
│   paths → P.model_path()/P.dataset_file()     args → P.tune()/P.flag() │
│   always-on flags → P.train_args_extra        capabilities → P.supports()│
└───────────────┬────────────────────────────────────────────────────────┘
                │ import platform_profiles (tests/ on sys.path)
                ▼
┌────────────────────────────────────────────────────────────────────────┐
│ platform_profiles (this package)                                       │
│  selection: VIME_TEST_DEVICE > VIME_PLATFORM > vime.platforms > HIP    │
│  discovery: profiles/*.py auto-scan (name must equal filename)         │
│  new-platform launch entries auto-register into the launch registry    │
└───────────────┬────────────────────────────────────────────────────────┘
                │ U.execute_train() — the existing seam in command_utils
                ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Launch layer (vime/utils/external_utils/launch.py — NOT modified)      │
│  PLATFORMS registry: cuda / npu (+ anything a profile's launch_def     │
│  registers). cuda path stays byte-identical to upstream.               │
└───────────────┬────────────────────────────────────────────────────────┘
                ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Core runtime (vime/platforms/ — NOT modified, constraint of this work) │
│  Ray resources, weight-transfer backend, Megatron bootstrap, ...       │
└────────────────────────────────────────────────────────────────────────┘
```

## 2. Where each platform difference lives

| Category (需求) | Profile field | Example (npu.py) |
|---|---|---|
| 1. 路径 paths | `asset_root_env_vars`, `default_asset_root`, `tmp_under_asset_root` | `("VIME_TEST_ROOT", "HF_HOME")` — warm HF cache on Ascend CI |
| 2. 参数 args | `train_args_extra` (always-on flags), `tuning` (per-knob values) | `--use-flash-attn`, `weight_nz_mode=0`, `max_tokens_per_gpu=8192` |
| 3. 环境变量 env | `env` (test-process exports; launch env belongs to LaunchSpec) | (empty today; extension point) |
| 4. 资源声明 resources | `launch_platform`, `launch_def`, `torch_dist_convert` | `npu` uses NPU Ray bundles, native HF load |
| 5. 拓扑 topology | `topology` (tp/pp/cp/ep defaults) | (only when hardware forces a layout change) |
| capabilities | `capabilities` (boolean features tests branch on) | `vllm_sleep_mode: True` |

Test-side API (all accessors take the **GPU-blessed value as the default**):

```python
P = current_profile()
P.model_path("Qwen3-4B")                       # → "{root}/models/Qwen3-4B" (POSIX, always)
P.dataset_file("zhuzilin/dapo-math-17k", "dapo-math-17k.jsonl")
P.tune("max_tokens_per_gpu", 16384)            # → 16384 on cuda, 8192 on npu
P.flag("vllm-max-num-seqs", "vllm_max_num_seqs")  # → "" on cuda (omit), "--vllm-max-num-seqs 16 " on npu
P.flag("vllm-enable-sleep-mode", "vllm_sleep_mode")  # boolean tuning → bare flag when True
P.train_args_extra                             # appended verbatim at the end of train_args
P.topo("cp", 2)                                # parallel-layout default
P.supports("vllm_sleep_mode")                  # capability query — never a name comparison
P.torch_dist_convert                           # native HF load when False
apply_profile_env()                            # __main__ exports P.env
ensure_asset_dirs() / download_model() / download_dataset()
```

## 3. The worked example: test_qwen3_4B_ckpt.py (Checkpoint 恢复)

First feature-dimension migration (GPU-only → GPU+NPU). The diff is exactly
the platform surface:

| Before (GPU-only) | After (shared) |
|---|---|
| `--hf-checkpoint /root/models/Qwen3-4B/ ` | `f"--hf-checkpoint {P.model_path(MODEL_NAME)}/ "` |
| `--ref-load /root/models/Qwen3-4B_torch_dist` | `ref_load = P.model_path(...) if P.torch_dist_convert else P.model_path(MODEL_NAME)` |
| `--prompt-data /root/datasets/dapo-math-17k/...` | `P.dataset_file(...)` |
| `--max-tokens-per-gpu 16384` | `P.tune("max_tokens_per_gpu", 16384)` |
| `--vllm-gpu-memory-utilization 0.8 --vllm-max-cudagraph-capture-size 16` (literal) | `P.flag(...)` renders both; npu adds mem-util 0.5, num-seqs 16, cudagraph 8, model-len 32768 |
| (nothing) | `f"{P.train_args_extra}"` appended — npu: `--use-flash-attn --no-gradient-accumulation-fusion --vllm-additional-config '{"weight_nz_mode": 0}'` |

Everything else — grpo block, rollout counts, TP2/CP2 parallel layout,
optimizer placement matrix, `--ci-test` guards — is untouched **test intent**.

**GPU behavior is byte-identical.** `tests/utils/test_platform_profiles.py`
replays the script with a stubbed launcher and asserts the generated
`train_args` equal the pre-migration string *exactly* (all 6 mode×optimizer
combinations), and `prepare()` emits the identical command list. The npu
replay asserts the adapted variant (paths under the NPU root, no torch_dist,
memory bounds, no `/root` leakage, intent blocks unchanged).

### Migration recipe (apply to any remaining GPU test)

1. Add the sys.path shim + `from platform_profiles import ...` (copy from
   the ckpt script header).
2. Replace hardcoded `/root/...` with `P.model_path()/P.dataset_file()/
   P.dataset_dir()`; route downloads through `ensure_asset_dirs() /
   download_model() / download_dataset()`.
3. Replace platform-conditional literals with `P.tune()/P.flag()` — GPU
   values become the defaults; overrides move to `profiles/npu.py`.
4. Route `--ref-load` through `P.torch_dist_convert`.
5. Append `f"{P.train_args_extra}"` to `train_args`.
6. Register the NPU run in `.buildkite/npu_suites.py` under a `feature-*`
   suite; the GPU registration in `gpu_suites.py` stays as-is.
7. Add a replay case to `tests/utils/test_platform_profiles.py` (golden =
   the pre-migration string) — this is what makes the migration reviewable
   and keeps future regressions out.

## 4. CI integration

- **GPU**: `gpu_suites.py` unchanged — shared scripts run exactly as before.
- **NPU**: `npu_suites.py` gains `feature-*` suites. `feature-ckpt` runs
  `test_qwen3_4B_ckpt.py` on `npu-8` with `VIME_PLATFORM=npu` (explicit for
  reproducible logs; auto-detection would also work). Optimizer gpu-placement
  entries are deliberately not registered yet — NPU's established policy is
  cpu-offload; enable after hardware validation. Selectable at the
  `npu-gate` block (`run-npu-ci-feature-ckpt`).
- **Env conventions preserved**: `VIME_TEST_ROOT` (asset-root override,
  highest priority everywhere), `VIME_TEST_ENABLE_EVAL`,
  `VIME_TEST_ENABLE_INFINITE_RUN`, `VIME_TEST_NUM_ROLLOUT`, ... all keep
  their existing meaning; profiles only add resolution *below* them
  (e.g. `HF_HOME` fallback on NPU).

## 5. Invariants enforced on CPU (tests/utils/test_platform_profiles.py)

1. **Discovery** — one file per platform; `PROFILE.name == filename`; unique.
2. **cuda neutrality** — cuda's args/tuning/topology/env sections are empty,
   so every shared script renders upstream-identical arguments (this is the
   backward-compatibility guarantee for all 53 GPU scenarios).
3. **Profile pinning** — npu/rocm carry exactly the validated recipes
   (weight_nz_mode JSON, memory bounds, transport defaults).
4. **Golden replay** — the migrated script's cuda output equals the
   pre-migration string byte-for-byte; the npu output carries only the
   sanctioned deltas.
5. **Name lint** — any `tests/test_*.py` that imports `platform_profiles`
   must not contain platform-name comparisons (`== "npu"` …) nor set
   `VIME_PLATFORM`; platform names may appear **only** in `profiles/*.py`.

## 6. Adding a new platform (thought experiment: Intel Gaudi)

Test-layer cost is **one profile file + one CI entry**. No existing test,
no `platform_profiles` module code, no `launch.py` edits.

`tests/platform_profiles/profiles/gaudi.py`:

```python
from platform_profiles import LaunchSpec, PlatformProfile

PROFILE = PlatformProfile(
    name="gaudi",
    description="Intel Gaudi 3",
    asset_root_env_vars=("VIME_TEST_ROOT", "GAUDI_ASSET_ROOT"),
    train_args_extra="--no-fused-optimizer ",            # whatever the port requires
    tuning={"vllm_gpu_memory_utilization": 0.7, "max_tokens_per_gpu": 8192},
    # 4. resources: self-register into the launch registry — launch.py stays untouched
    launch_platform="gaudi",
    launch_def=LaunchSpec(
        ray_args='--num-gpus 0 --resources \'{"HPU": {n}}\'',
        env={"HABANA_VISIBLE_MODULES": "all", "PT_HPU_LAZY_MODE": "1"},
    ),
    capabilities={"vllm_sleep_mode": False},
)
```

CI entry in `npu_suites.py`-style generator (or a new `hpu_suites.py`):

```python
"feature-ckpt": [("test_qwen3_4B_ckpt.py", "hpu-8", "", {"VIME_PLATFORM": "gaudi"}, 360)],
```

What is *not* test-layer work (and is unavoidable framework territory,
pre-existing and unchanged by this design): `vime/platforms/gaudi.py` — the
core platform aggregate (Ray resource spec, weight-transfer backend, Megatron
bootstrap). The profile layer deliberately does not duplicate that; it links
to it via `launch_platform`/`launch_def`.

### Cost estimate

| Step | Files touched | Est. effort |
|---|---|---|
| Profile file | 1 new (`profiles/<name>.py`) | 0.5–1 day (parallel to porting) |
| CI registration | 1 suite entry (+ gate option) | minutes |
| Core platform (framework, pre-existing constraint) | `vime/platforms/<name>.py` | framework team, unchanged scope |
| Test changes | **0** | — |

## 7. Compatibility with the PR #481 common helpers (reference, not a dependency)

PR #481 (`test/npu-gpu-suite-s10`) takes the opposite shape: platform
knowledge is added to `command_utils` (`get_test_root`, `get_test_platform_args`,
`get_test_tmp_dir`) and consumed via `U.current_platform().name == "npu"`
comparisons inside each migrated test. This framework keeps everything under
`tests/` and removes name comparisons from test bodies:

| PR #481 helper | This framework | Notes |
|---|---|---|
| `U.get_test_root()` | `P.asset_root()` | Same precedence: `VIME_TEST_ROOT` > `HF_HOME` (npu) > `/root` |
| `U.get_test_tmp_dir()` | `P.tmp_dir()` | NPU scratch under `<root>/tmp` |
| `U.get_test_platform_args()` | `P.train_args_extra` | Same validated flag set (flash-attn, no-grad-accum-fusion, weight_nz_mode) |
| `U.current_platform().torch_dist_convert` | `P.torch_dist_convert` | Mirrors the same launch-registry flag |
| `U.current_platform().name == "npu"` (per-test branches) | `P.tune()/P.flag()/P.supports()` | Declarative; no name literals in tests |

Convergence: the two layers can coexist on the same branch — they touch
disjoint files except the migrated test scripts themselves. If #481 lands
first, migrating a test to profiles is a mechanical rewrite of its
`U.current_platform().name == "npu"` branches into `P.tune()/P.flag()` calls;
the golden-replay contract tests make each rewrite provable. The reverse
order is equally mechanical. Neither layer's helpers conflict at runtime
(both resolve through the same `vime.platforms` selection).

## 8. Roadmap: remaining feature dimensions (需求 3)

The ckpt migration is the template; each row is one PR-sized migration
following the recipe in §3:

| Dimension | GPU scenario(s) to migrate | Profile surface needed |
|---|---|---|
| 权重同步 delta/full | `test_full_disk_weight_update.py`, `test_delta_disk_weight_update.py` (NPU-only today) | `P.tmp_dir()`, `VIME_TEST_NUM_ROLLOUT` already flows via env |
| 权重同步 collective | `test_qwen3_4B_external_pd.py` | `update_weight_transport` tuning key (rocm already uses it) |
| 并行布局 PP/CP/EP | `test_qwen2_5_0_5B_non_colocate_pp.py`, `test_qwen3_5_0_8B_top_p_cp2.py`, `test_qwen3_0.6B_parallel_check.py` | `P.topo()` for EP fallbacks |
| Rollout 模式 streaming/partial/async | `test_qwen3_4B_streaming_partial_rollout.py`, `test_qwen2.5_0.5B_fully_async_short.py` | none expected |
| Checkpoint 恢复 | `test_qwen3_4B_ckpt.py` (**done**), `test_straw_checkpoint_fork.py` | done |
| 故障容错 | `test_qwen2.5_0.5B_rollout_health.py`, `test_qwen2.5_0.5B_training_recovery.py` | `VIME_TEST_RAY_*` passthrough |
| 多模型服务 | `test_glm4.7_30B_A3B_pd_mooncake.py` | capability gates for PD transport |
