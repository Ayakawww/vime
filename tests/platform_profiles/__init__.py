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
    "reset_profile_cache",
]
