"""Platform-agnostic test harness: profile discovery, selection, and helpers.

Shared test scripts (tests/test_*.py) consume this package instead of
hardcoding platform differences:

    from platform_profiles import current_profile
    P = current_profile()
    ... f"--max-tokens-per-gpu {P.tune('max_tokens_per_gpu', 16384)} "
    ... f"{P.flag('vllm-max-num-seqs', 'vllm_max_num_seqs')}"
    ... f"{P.train_args_extra}"

Selection order (mirrors and extends vime.platforms):

  1. VIME_TEST_DEVICE  — test-layer variant selector (e.g. "rocm"; the core
                         registry only distinguishes cuda/npu, ROCm rides the
                         CUDA launch path with HIP-specific behavior)
  2. VIME_PLATFORM     — core platform override (npu/cuda/...)
  3. vime.platforms.current_platform() — auto-detection (VIME_ACCELERATOR,
                         /dev/davinci* probing, default cuda)
  4. cuda refined to rocm when torch.version.hip is set

Profiles are auto-discovered from tests/platform_profiles/profiles/*.py —
adding a platform means dropping one file there (see README.md), no edit to
this module or to any existing test.

Stdlib only, so contract tests run without vime/torch installed.
"""

from __future__ import annotations

import importlib.util
import os
import shlex
from functools import cache
from pathlib import Path

from platform_profiles.base import LaunchSpec, PlatformProfile

PROFILE_DIR = Path(__file__).resolve().parent / "profiles"


@cache
def _load_profiles() -> dict[str, PlatformProfile]:
    profiles: dict[str, PlatformProfile] = {}
    for path in sorted(PROFILE_DIR.glob("*.py")):
        if path.stem.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"platform_profiles.profiles.{path.stem}", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        profile = getattr(module, "PROFILE", None)
        if profile is None:
            raise ValueError(f"profile file {path} must define a module-level PROFILE")
        if not isinstance(profile, PlatformProfile):
            raise ValueError(f"PROFILE in {path} must be a PlatformProfile, got {type(profile)!r}")
        if profile.name != path.stem:
            raise ValueError(f"profile name {profile.name!r} must match its filename {path.stem!r}")
        if profile.name in profiles:
            raise ValueError(f"duplicate profile name {profile.name!r}")
        _register_launch_spec(profile)
        profiles[profile.name] = profile
    if not profiles:
        raise ValueError(f"no profiles found under {PROFILE_DIR}")
    return profiles


def _register_launch_spec(profile: PlatformProfile) -> None:
    """Register a profile's LaunchSpec into vime's launch registry so a new
    platform launches without editing vime/utils/external_utils/launch.py.

    No-op when the registry is not importable (e.g. pure-CPU contract-test
    environments) — launching only ever happens where vime is installed.
    """
    if profile.launch_def is None:
        return
    try:
        from vime.utils.external_utils.launch import PLATFORMS
        from vime.utils.external_utils.launch import Platform as LaunchPlatform
    except ImportError:
        return
    if profile.launch_platform in PLATFORMS:
        return
    launch = profile.launch_def
    PLATFORMS[profile.launch_platform] = LaunchPlatform(
        name=profile.launch_platform,
        ray_args=launch.ray_args,
        env=dict(launch.env),
        torch_dist_convert=launch.torch_dist_convert,
    )


def get_profile(name: str) -> PlatformProfile:
    profiles = _load_profiles()
    try:
        return profiles[name.strip().lower()]
    except KeyError as exc:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown platform profile {name!r}; available profiles: {available}") from exc


def available_profiles() -> list[str]:
    return sorted(_load_profiles())


def _validate_core_override(override: str) -> str:
    """Fail fast when VIME_PLATFORM names something the core registry does not
    know (e.g. rocm): the profile layer would resolve it, but the launch seam
    inside execute_train would reject it much later with a confusing error."""
    try:
        from vime.platforms import get_platform
    except ImportError:
        return override  # core unavailable here; get_profile validates the name
    try:
        get_platform(override)
    except ValueError as exc:
        raise ValueError(
            f"VIME_PLATFORM={override!r} is not a core platform; if it is a "
            f"test-layer device variant (e.g. rocm), set VIME_TEST_DEVICE instead"
        ) from exc
    return override


def _resolve_core_platform_name() -> str:
    """Core platform selection (VIME_PLATFORM / VIME_ACCELERATOR / device
    probing). Factored out so contract tests can stub the seam."""
    override = os.environ.get("VIME_PLATFORM", "").strip().lower()
    if override:
        return _validate_core_override(override)
    try:
        from vime.platforms import current_platform

        return current_platform().name
    except Exception:
        # vime not importable (e.g. lint/contract environments) — the CUDA
        # profile is the neutral default and changes nothing.
        return "cuda"


def _is_hip() -> bool:
    try:
        import torch

        return torch.version.hip is not None
    except Exception:
        return False


@cache
def _resolve_profile_name() -> str:
    # 1. test-layer device variant (e.g. rocm on the AMD CI pipeline)
    device = os.environ.get("VIME_TEST_DEVICE", "").strip().lower()
    if device:
        return device
    # 2/3. core selection: explicit override, then auto-detection
    name = _resolve_core_platform_name()
    # 4. the core treats CUDA and ROCm alike; refine for the test layer
    if name == "cuda" and _is_hip() and "rocm" in _load_profiles():
        return "rocm"
    return name


def current_profile() -> PlatformProfile:
    """The profile for the machine this test is running on."""
    return get_profile(_resolve_profile_name())


def reset_profile_cache() -> None:
    """Clear discovery/selection caches (for tests and env changes)."""
    _load_profiles.cache_clear()
    _resolve_profile_name.cache_clear()


# ── Imperative helpers (call exec_command lazily; module stays stdlib-only) ──


def apply_profile_env(profile: PlatformProfile | None = None) -> None:
    """Export the profile's test-process env (category 3). Call once from
    __main__ before prepare()."""
    os.environ.update((profile or current_profile()).env)


def _exec(cmd: str) -> None:
    from vime.utils.misc import exec_command

    exec_command(cmd)


def ensure_asset_dirs(profile: PlatformProfile | None = None) -> None:
    p = profile or current_profile()
    root = shlex.quote(p.asset_root())
    _exec(f"mkdir -p {root}/models {root}/datasets")


def download_model(repo_id: str, name: str | None = None, profile: PlatformProfile | None = None) -> str:
    """hf-download a model into the profile's asset root; returns its path."""
    p = profile or current_profile()
    target = shlex.quote(p.model_path(name or repo_id.split("/")[-1]))
    _exec(f"hf download {repo_id} --local-dir {target}")
    return target


def download_dataset(full_name: str, profile: PlatformProfile | None = None) -> str:
    """hf-download a dataset into the profile's asset root; returns its dir."""
    p = profile or current_profile()
    target = shlex.quote(p.dataset_dir(full_name))
    _exec(f"hf download --repo-type dataset {full_name} --local-dir {target}")
    return target


# ── Hardware validation helpers ──────────────────────────────────────────────


def visible_device_count(profile: PlatformProfile | None = None) -> int | None:
    """Accelerators visible to this process, or None when the platform's
    visibility env var is not set (nothing to count against)."""
    p = profile or current_profile()
    raw = os.environ.get(p.visible_devices_env, "").strip()
    if not raw:
        return None
    return len([x for x in raw.split(",") if x.strip()])


def require_devices(count: int, profile: PlatformProfile | None = None) -> None:
    """Fail fast with a clear message when fewer accelerators are visible than
    the test needs, instead of OOM'ing deep inside Megatron. Used with a
    topology key: `require_devices(P.topo("min_gpus", 8))`. No-op when device
    visibility is not declared in the environment."""
    p = profile or current_profile()
    raw = os.environ.get(p.visible_devices_env, "").strip()
    if not raw:
        return
    available = len([x for x in raw.split(",") if x.strip()])
    if available >= count:
        return
    raise SystemExit(
        f"This test needs {count} accelerators but {p.visible_devices_env}={raw!r} exposes only {available} "
        f"(profile={p.name}). If the requirement looks wrong, adjust the 'min_gpus' topology key in the profile."
    )


def hardware_label(profile: PlatformProfile | None = None) -> str:
    """Best-effort hardware identification from the profile's hardware env
    (e.g. SOC_VERSION on Ascend); falls back to the profile name."""
    p = profile or current_profile()
    if p.hardware_env:
        raw = os.environ.get(p.hardware_env, "").strip()
        if raw:
            return p.hardware_names.get(raw, raw)
    return p.name


def validation_report(
    model: str,
    result: str = "PASSED",
    notes: str = "",
    profile: PlatformProfile | None = None,
) -> str:
    """End-of-run summary of what the profile actually changed for this test.

    Reads the accessor-call recording on the profile (populated by
    tune/flag/topo/asset_root/supports calls the test made) — call it after
    the test body has built its arguments. `notes` appends free-form context
    to the topology line (e.g. "colocate")."""
    p = profile or current_profile()
    hardware = hardware_label(p)
    platform_line = f"{p.name.upper()} ({hardware})" if hardware != p.name else p.name.upper()
    topos = p._queries_of("topo")
    topology_line = ", ".join(f"{q.key}={q.value}" for q in topos) or "n/a"
    if notes:
        topology_line = f"{topology_line} ({notes})"

    applied = {"paths": 0, "args": 0, "topology": 0}
    for q in p.queries.values():
        if not q.overridden:
            continue
        if q.kind == "path":
            applied["paths"] += 1
        elif q.kind in ("tune", "flag"):
            applied["args"] += 1
        elif q.kind == "topo":
            applied["topology"] += 1
    total = sum(applied.values())
    detail = ", ".join(f"{kind}:{count}" for kind, count in applied.items() if count)

    skipped = [q.key for q in p._queries_of("capability") if q.source == "profile" and q.value is False]

    lines = [
        "=== Platform Validation Report ===",
        f"Platform: {platform_line}",
        f"Model: {model}",
        f"Topology: {topology_line}",
        f"Profile overrides applied: {total} ({detail})" if detail else "Profile overrides applied: 0",
    ]
    if skipped:
        lines.append(
            f"Profile overrides skipped: {len(skipped)} (not applicable on this platform: {', '.join(skipped)})"
        )
    lines.append(f"Result: {result}")
    return "\n".join(lines)


__all__ = [
    "LaunchSpec",
    "PlatformProfile",
    "apply_profile_env",
    "available_profiles",
    "current_profile",
    "download_dataset",
    "download_model",
    "ensure_asset_dirs",
    "get_profile",
    "hardware_label",
    "require_devices",
    "reset_profile_cache",
    "validation_report",
    "visible_device_count",
]
