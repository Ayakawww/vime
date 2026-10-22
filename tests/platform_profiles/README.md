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
| 4. 资源声明 resources | `launch_platform`, `launch_def`, `torch_dist_convert`, `visible_devices_env`, `hardware_env`/`hardware_names` | `npu` uses NPU Ray bundles, native HF load, `ASCEND_RT_VISIBLE_DEVICES`, `SOC_VERSION=ascend910b3 → "A2"` |
| 5. 拓扑 topology | `topology` (num_gpus/tp/cp/min_gpus defaults) | `{"num_gpus": 4, "tp": 4, "cp": 1}` — the 4-card layout validated on 910B3 |
| capabilities | `capabilities` (boolean features tests branch on) | `vllm_sleep_mode: True`, `health_monitor: False` |

Test-side API (all accessors take the **GPU-blessed value as the default**):

```python
P = current_profile()
P.model_path("Qwen3-4B")                       # → "{root}/models/Qwen3-4B" (POSIX, always)
P.dataset_file("zhuzilin/dapo-math-17k", "dapo-math-17k.jsonl")
P.tune("max_tokens_per_gpu", 16384)            # → 16384 on cuda, 8192 on npu
P.flag("vllm-max-num-seqs", "vllm_max_num_seqs")  # → "" on cuda (omit), "--vllm-max-num-seqs 16 " on npu
P.flag("vllm-enable-sleep-mode", "vllm_sleep_mode")  # boolean tuning → bare flag when True
P.topo("num_gpus", 8)                          # → 8 on cuda, 4 on npu (card budget)
P.topo("tp", 2)                                # → 2 on cuda, 4 on npu (validated 4B layout)
P.topo("cp", 2)                                # → 2 on cuda, 1 on npu (CP2 does not fit 4 cards)
P.topo("min_gpus", 8)                          # resource requirement (30B-class tests override to 8)
P.train_args_extra                             # appended verbatim at the end of train_args
P.supports("health_monitor")                   # capability query — never a name comparison
P.torch_dist_convert                           # native HF load when False
apply_profile_env()                            # __main__ exports P.env
ensure_asset_dirs() / download_model() / download_dataset()
require_devices(P.topo("min_gpus", P.topo("num_gpus", 8)))  # fail fast when too few cards are visible
print(P.dump())                                # startup: the resolved platform configuration
print(validation_report(model=..., notes="colocate"))        # end of run: override accounting
```

### Topology and tuning scoping

Keys are queried only where the value genuinely varies by platform, so
unprefixed `topology`/`tuning` entries are the platform's **default recipe**
(npu: the 4B-class colocated shape). When a scenario needs different bounds,
use a scenario-prefixed key instead of fighting over the global one:

* `test_qwen3_4B_ckpt.py` (4B, colocate) queries the unprefixed keys:
  `P.topo("tp", 2)`, `P.tune("max_tokens_per_gpu", 16384)`.
* `test_full_disk_weight_update.py` (0.8B, non-colocate, TP1 identical on both
  platforms) keeps TP1/CP1 literals and only queries what varies:
  `P.tune("full_disk_max_tokens_per_gpu", 9216)` → 2048 on npu,
  `P.flag("vllm-max-model-len", "full_disk_vllm_max_model_len")` → 8192 on npu.

Tests whose layout does not vary by platform never query `topology` at all —
that is why the npu `tp: 4` override cannot leak into them.

## 3. The worked example: test_qwen3_4B_ckpt.py (Checkpoint 恢复)

First feature-dimension migration (GPU-only → GPU+NPU). The diff is exactly
the platform surface:

| Before (GPU-only) | After (shared) |
|---|---|
| `--hf-checkpoint /root/models/Qwen3-4B/ ` | `f"--hf-checkpoint {P.model_path(MODEL_NAME)}/ "` |
| `--ref-load /root/models/Qwen3-4B_torch_dist` | `ref_load = P.model_path(...) if P.torch_dist_convert else P.model_path(MODEL_NAME)` |
| `--prompt-data /root/datasets/dapo-math-17k/...` | `P.dataset_file(...)` |
| `--max-tokens-per-gpu 16384` | `P.tune("max_tokens_per_gpu", 16384)` |
| `--tensor-model-parallel-size 2` | `P.topo("tp", 2)` — npu renders the validated TP4 |
| `--context-parallel-size 2` | `P.topo("cp", 2)` — CP2 does not fit 4 cards, npu renders 1 |
| `NUM_GPUS = 8` + `--actor-num-gpus-per-node 8` + `execute_train(num_gpus_per_node=8)` | `P.topo("num_gpus", 8)` everywhere — npu runs the validated 4-card shape |
| `--rollout-num-gpus-per-engine 2` | `P.topo("tp", 2)` (colocated engines mirror training TP) |
| `--vllm-gpu-memory-utilization 0.8 --vllm-max-cudagraph-capture-size 16` (literal) | `P.flag(...)` renders both; npu adds mem-util 0.5, num-seqs 16, cudagraph 8, model-len 32768, rollout-num-gpus 4 |
| (nothing) | `f"{P.train_args_extra}"` appended — npu: `--use-flash-attn --no-gradient-accumulation-fusion --vllm-additional-config '{"weight_nz_mode": 0}'` |

Everything else — grpo block, rollout counts, optimizer placement matrix,
`--ci-test` guards — is untouched **test intent**.

The second migration, `test_full_disk_weight_update.py` (权重同步 full-disk,
Qwen3.5-0.8B, non-colocate), follows the same recipe with two additions:
`build_train_args()` is split out of `execute()` so the golden replay can
capture the command line without hardware, and the scratch dir moves under
`P.tmp_dir()` (None on cuda → system default, byte-identical).

**GPU behavior is byte-identical.** `tests/utils/test_platform_profiles.py`
replays both scripts with a stubbed launcher and asserts the generated
`train_args` equal the pre-migration strings *exactly* (ckpt: all 6
mode×optimizer combinations; full-disk: default plus the
`VIME_TEST_NUM_ROLLOUT` knob), and that `execute_train` still receives
`num_gpus_per_node=8` (ckpt) / `=4` (full-disk) on cuda. The npu replays
assert the adapted variants (paths under the NPU root, no torch_dist, the
validated 4-card TP4/CP1 topology for ckpt, scenario-scoped bounds for
full-disk, no `/root` leakage, intent blocks unchanged).

### Migration recipe (apply to any remaining GPU test)

1. Add the sys.path shim + `from platform_profiles import ...` (copy from
   the ckpt script header).
2. Replace hardcoded `/root/...` with `P.model_path()/P.dataset_file()/
   P.dataset_dir()`; route downloads through `ensure_asset_dirs() /
   download_model() / download_dataset()`.
3. Replace platform-conditional literals with `P.tune()/P.flag()` — GPU
   values become the defaults; overrides move to `profiles/npu.py`
   (scenario-prefixed keys when the recipe differs per test, see scoping
   above).
4. Replace hardcoded card counts and layouts with `P.topo("num_gpus"/"tp"/"cp",
   gpu_default)`; colocated engines mirror `P.topo("tp", ...)`.
5. Route `--ref-load` through `P.torch_dist_convert`; keep scratch dirs under
   `P.tmp_dir()`.
6. Append `f"{P.train_args_extra}"` to `train_args`.
7. In `__main__`: `print(P.dump())` at startup,
   `require_devices(P.topo("min_gpus", P.topo("num_gpus", N)))` as the fast
   fail guard, and `validation_report(...)` at the end (PASSED/FAILED).
8. Register the NPU run in `.buildkite/npu_suites.py` under a `feature-*`
   suite; the GPU registration in `gpu_suites.py` stays as-is.
9. Add a replay case to `tests/utils/test_platform_profiles.py` (golden =
   the pre-migration string) — this is what makes the migration reviewable
   and keeps future regressions out.

## 4. CI integration

- **GPU**: `gpu_suites.py` unchanged — shared scripts run exactly as before.
- **NPU**: `npu_suites.py` gains `feature-*` suites. `feature-ckpt` runs
  `test_qwen3_4B_ckpt.py` on `npu-8` with `VIME_PLATFORM=npu` (explicit for
  reproducible logs; auto-detection would also work). Optimizer gpu-placement
  entries are deliberately not registered yet — NPU's established policy is
  cpu-offload; enable after hardware validation. `feature-weight-sync` runs
  `test_full_disk_weight_update.py` on `npu-4` with
  `VIME_TEST_NUM_ROLLOUT=2` (bounds NPU request load and exercises the
  reload-after-sync path — the same two-round shape the delta-disk NPU test
  uses). Both are selectable at the `npu-gate` block.
- **Env conventions preserved**: `VIME_TEST_ROOT` (asset-root override,
  highest priority everywhere), `VIME_TEST_ENABLE_EVAL`,
  `VIME_TEST_ENABLE_INFINITE_RUN`, `VIME_TEST_NUM_ROLLOUT`, ... all keep
  their existing meaning; profiles only add resolution *below* them
  (e.g. `HF_HOME` fallback on NPU).

## 5. Invariants enforced on CPU (tests/utils/test_platform_profiles.py)

1. **Discovery** — one file per platform; `PROFILE.name == filename`; unique.
2. **cuda neutrality** — cuda's args/tuning/topology/env sections are empty
   (including an explicit topology-neutrality check: every `P.topo` query
   falls through to the test's GPU-blessed default), so every shared script
   renders upstream-identical arguments. This is the backward-compatibility
   guarantee for all 53 GPU scenarios. `capabilities` is exempt: it states
   feature presence (e.g. `health_monitor=True` on cuda keeps GPU behavior)
   and never changes rendered arguments.
3. **Profile pinning** — npu/rocm carry exactly the validated recipes
   (weight_nz_mode JSON, memory bounds, the 4-card TP4/CP1 topology,
   transport defaults, hardware labels).
4. **Golden replay** — both migrated scripts' cuda output equals the
   pre-migration strings byte-for-byte (including the launch-seam kwargs like
   `num_gpus_per_node`); the npu outputs carry only the sanctioned deltas.
5. **Name lint** — any `tests/test_*.py` that imports `platform_profiles`
   must not contain platform-name comparisons (`== "npu"` …) nor set
   `VIME_PLATFORM`; platform names may appear **only** in `profiles/*.py`.

## 6. Observability: P.dump() and the platform validation report

Every accessor call is recorded on the profile, which powers two debugging
surfaces (no more jumping between the test file and the profile file):

```python
# at test startup (__main__):
print(P.dump())
# [NPU Profile] platform=npu — Huawei Ascend 910B3 (A2) / 910C (A3), ...
# [NPU Profile] torch_dist_convert=False (launch_platform=npu)
# [NPU Profile] path: asset_root=/root/.cache/huggingface (via HF_HOME)
# [NPU Profile] tune recipe (not yet resolved by this test): ...   ← before args are built
# ...after the test resolves its knobs (e.g. in a second dump):
# [NPU Profile] tune: max_tokens_per_gpu=8192 (GPU default: 16384)
# [NPU Profile] topology: num_gpus=4 (GPU default: 8), tp=4 (GPU default: 2)
# [NPU Profile] capability: health_monitor=False
# [NPU Profile] train_args_extra: --use-flash-attn --no-gradient-accumulation-fusion ...

# at the end of the run:
print(validation_report(model="Qwen3-4B (ckpt test)", notes="colocate"))
# === Platform Validation Report ===
# Platform: NPU (ascend910b3 / A2)        ← via SOC_VERSION + hardware_names
# Model: Qwen3-4B (ckpt test)
# Topology: num_gpus=4, tp=4, cp=1 (colocate)
# Profile overrides applied: 7 (paths:1, args:3, topology:3)
# Profile overrides skipped: 1 (not applicable on this platform: health_monitor)
# Result: PASSED
```

Hardware identification is best-effort and profile-declared
(`hardware_env="SOC_VERSION"`, `hardware_names={"ascend910b3": "ascend910b3 /
A2", ...}`) — the framework itself stays free of platform names and never
shells out to `npu-smi`.

## 7. Known compatibility issues (910B3/A2 validation findings)

| Finding | Handling | Mechanism |
|---|---|---|
| `/health_generate` 404 on vllm-ascend | Documented + capability-gated (option B) | `capabilities.health_monitor`: cuda `True` (GPU behavior unchanged), npu `False`. Tests add `--use-fault-tolerance`-style flags only `if P.supports("health_monitor")`. Fixing the endpoint (option A) means editing `vllm_engine.py` — outside the test layer; revisit when vllm-ascend ships the endpoint. |
| `straw-queue` not preinstalled | Documented (option B) | Needed by the straw-checkpoint / training-recovery scenarios when they migrate. NPU-only concern; install manually before those suites: `pip install straw-queue`. Not added to requirements.txt (would leak into GPU environments). |
| 30B-A3B needs 8 cards (Megatron init ≈59 GB/card > 61 GB capacity); only 7 healthy cards (card 6 faulty), CI pool is 4 cards | Documented + fail-fast mechanism | Tests declare `require_devices(P.topo("min_gpus", 8))`; the profile can override `min_gpus` per platform. With `ASCEND_RT_VISIBLE_DEVICES` exposing fewer cards, the test exits immediately with a clear message instead of OOM'ing inside Megatron. 30B scenarios stay unregistered on NPU until an 8-card pool exists. |
| fanout_short blocked on `/health_generate` 404 | Same as row 1 | Will unblock when the migration reaches it; the `health_monitor` gate is already in place. |

## 8. Adding a new platform (thought experiment: Intel Gaudi)

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

## 9. Compatibility with the PR #481 common helpers (reference, not a dependency)

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

## 10. Roadmap: remaining feature dimensions (需求 3)

The ckpt and full-disk migrations are the templates; each row is one
PR-sized migration following the recipe in §3:

| Dimension | GPU scenario(s) to migrate | Profile surface needed |
|---|---|---|
| 权重同步 delta | `test_delta_disk_weight_update.py` (NPU-only today; retires into the shared pattern after full-disk is hardware-validated) | same as full-disk |
| 权重同步 collective | `test_qwen3_4B_external_pd.py` | `update_weight_transport` tuning key (rocm already uses it) |
| 并行布局 PP/CP/EP | `test_qwen2_5_0_5B_non_colocate_pp.py`, `test_qwen3_5_0_8B_top_p_cp2.py`, `test_qwen3_0.6B_parallel_check.py` | `P.topo()` for EP fallbacks |
| Rollout 模式 streaming/partial/async | `test_qwen3_4B_streaming_partial_rollout.py`, `test_qwen2.5_0.5B_fully_async_short.py` | none expected |
| Checkpoint 恢复 | `test_qwen3_4B_ckpt.py` (**done**), `test_straw_checkpoint_fork.py` (needs straw-queue, see §7) | done |
| 故障容错 | `test_qwen2.5_0.5B_rollout_health.py`, `test_qwen2.5_0.5B_training_recovery.py` | `health_monitor` capability gate (§7) |
| 多模型服务 | `test_glm4.7_30B_A3B_pd_mooncake.py` | capability gates for PD transport |
