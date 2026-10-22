"""Declarative platform profiles for the unified (platform-agnostic) test suite.

A `PlatformProfile` captures every difference a *test script* is allowed to
observe between hardware platforms, grouped into the five categories below.
Test files express intent (model, algorithm, dataset, assertions) and pull all
platform-specific values through profile accessors — they never compare
against platform names.

Difference categories and where each one lives:

  1. paths        `asset_root_env_vars` / `default_asset_root` / `tmp_under_asset_root`
  2. args         `train_args_extra` (always-appended flags) + `tuning` (semantic knobs)
  3. env          `env` (exports for the test process; launch env belongs to LaunchSpec)
  4. resources    `launch_platform` (name in vime's launch registry) + `launch_def`
                  (self-registering entry for platforms the registry does not know yet)
                  + `visible_devices_env` (device visibility) + `hardware_env`/`hardware_names`
  5. topology     `topology` (parallel-layout defaults, e.g. tp/cp/ep fallbacks)

Plus `capabilities` for boolean features tests may branch on ("does this
platform support X"), and `torch_dist_convert` mirroring the launch registry
(kept in sync by tests/utils/test_platform_profiles.py).

Every accessor call is recorded, so `dump()` can show exactly which values a
test resolved on this platform (and which were GPU-blessed defaults), and
`validation_report()` can summarize the overrides that took effect.

Stdlib only, so the whole layer (and its contract tests) runs on any machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any


@dataclass(frozen=True)
class LaunchSpec:
    """Launch-level description for a platform that has no entry in
    vime/utils/external_utils/launch.py yet.

    When a profile carries one, the loader registers it into the launch
    registry on first use (see platform_profiles.register_launch_spec), so a
    new platform needs no edit to vime framework files for launching.
    """

    ray_args: str  # ray-start resource flags, "{n}"-templated with the device count
    env: dict[str, str] = field(default_factory=dict)  # device runtime env
    torch_dist_convert: bool = True  # False -> native HF loading, no conversion step


_UNSET = object()  # sentinel: "no default supplied" for flag()/tune()/topo()


@dataclass(frozen=True)
class ProfileQuery:
    """One resolved accessor call, recorded for dump()/validation_report()."""

    kind: str  # "path" | "tune" | "flag" | "topo" | "capability"
    key: str
    value: Any
    default: Any = _UNSET  # the GPU-blessed value the test supplied (or _UNSET)
    cli_name: str = ""  # for flags: the CLI flag this rendered into
    source: str = ""  # "profile" | "gpu-default" | env-var name | "default"

    @property
    def overridden(self) -> bool:
        if self.kind == "path":
            return self.source not in ("", "default")
        if self.kind == "capability":
            return False  # capabilities state facts; they never change GPU args
        return self.value != self.default


def _fmt_default(default: Any) -> str:
    return "omit" if default is _UNSET else str(default)


@dataclass(frozen=True)
class PlatformProfile:
    """All platform knowledge a shared test is allowed to consume.

    Invariant (enforced by tests/utils/test_platform_profiles.py): the `cuda`
    profile stays *neutral* — empty args/tuning/topology/env — so any test
    written against profile accessors renders byte-identical arguments to the
    pre-migration GPU script on CUDA hardware. `capabilities` is exempt: it
    states feature presence (e.g. health_monitor=True) and never changes
    rendered arguments.
    """

    # identity
    name: str
    description: str = ""

    # 1. paths — where models/datasets/scratch live
    asset_root_env_vars: tuple[str, ...] = ()  # first *set* variable wins; VIME_TEST_* first
    default_asset_root: str = "/root"
    tmp_under_asset_root: bool = False  # scratch dir under the asset root (e.g. for checkpoint roundtrips)

    # 2. args — flags the platform always needs + per-knob tuning values
    train_args_extra: str = ""  # appended verbatim by shared tests ("" on cuda)
    tuning: dict[str, Any] = field(default_factory=dict)  # semantic key -> override

    # 3. env — exports applied to the test process itself
    env: dict[str, str] = field(default_factory=dict)

    # 4. resources — link to the launch registry (vime/utils/external_utils/launch.py)
    launch_platform: str = ""  # "" -> same as `name`
    launch_def: LaunchSpec | None = None  # auto-registered when the registry lacks the platform
    torch_dist_convert: bool = True  # mirrors launch registry; synced by contract test
    visible_devices_env: str = "CUDA_VISIBLE_DEVICES"  # where device visibility is declared
    hardware_env: str = ""  # env var carrying the SoC/board id (e.g. SOC_VERSION on Ascend)
    hardware_names: dict[str, str] = field(default_factory=dict)  # env value -> human label

    # 5. topology — parallel-layout defaults the platform requires
    topology: dict[str, Any] = field(default_factory=dict)

    # capabilities — boolean features tests may branch on (never platform names)
    capabilities: dict[str, bool] = field(default_factory=dict)

    # runtime bookkeeping of resolved accessor calls (not part of equality)
    queries: dict[str, ProfileQuery] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.launch_platform:
            object.__setattr__(self, "launch_platform", self.name)

    # -- paths ----------------------------------------------------------------
    #
    # All path accessors return POSIX-style strings: they end up in shell
    # commands executed inside Linux containers, so they must render
    # identically on any dev/CI host OS.

    def asset_root(self) -> str:
        """Asset root honoring the VIME_TEST_* convention: explicit override
        first (VIME_TEST_ROOT), then platform-specific fallbacks (e.g. HF_HOME
        on Ascend CI), then the platform default."""
        for var in self.asset_root_env_vars:
            if value := os.environ.get(var, "").strip():
                root = str(PurePosixPath(value))
                self.queries["asset_root"] = ProfileQuery(
                    "path", "asset_root", root, self.default_asset_root, source=var
                )
                return root
        self.queries["asset_root"] = ProfileQuery(
            "path", "asset_root", self.default_asset_root, self.default_asset_root, source="default"
        )
        return self.default_asset_root

    def model_path(self, name: str) -> str:
        return f"{self.asset_root()}/models/{name}"

    def dataset_dir(self, full_name: str) -> str:
        return f"{self.asset_root()}/datasets/{full_name.split('/')[-1]}"

    def dataset_file(self, full_name: str, filename: str) -> str:
        return f"{self.dataset_dir(full_name)}/{filename}"

    def tmp_dir(self) -> str | None:
        """Scratch dir under the asset root, or None to use the system default."""
        return f"{self.asset_root()}/tmp" if self.tmp_under_asset_root else None

    # -- args -----------------------------------------------------------------

    def tune(self, key: str, default: Any = _UNSET) -> Any:
        """Platform value for a semantic knob; `default` is the GPU-blessed
        value kept when the platform has no override."""
        if key in self.tuning:
            value = self.tuning[key]
            self.queries[key] = ProfileQuery("tune", key, value, default, source="profile")
            return value
        if default is _UNSET:
            raise KeyError(f"no tuning value for {key!r} on profile {self.name!r} and no default given")
        self.queries[key] = ProfileQuery("tune", key, default, default, source="gpu-default")
        return default

    def flag(self, cli_name: str, key: str, default: Any = _UNSET) -> str:
        """Render `--{cli_name} {value} ` for a tuning knob.

        - value True  -> bare flag `--{cli_name} `
        - value None or absent (and no default) -> "" (flag omitted)
        - explicit None in `tuning` -> "" even when a default is given
        """
        value = self.tuning.get(key, default)
        self.queries[key] = ProfileQuery(
            "flag", key, value, default, cli_name=cli_name, source="profile" if key in self.tuning else "gpu-default"
        )
        if value is None or value is _UNSET:
            return ""
        if value is True:
            return f"--{cli_name} "
        return f"--{cli_name} {value} "

    # -- topology / capabilities ----------------------------------------------

    def topo(self, key: str, default: Any = _UNSET) -> Any:
        """Parallel-layout value (tp/pp/cp/ep/num_gpus...); `default` is the
        GPU-blessed layout."""
        if key in self.topology:
            value = self.topology[key]
            self.queries[key] = ProfileQuery("topo", key, value, default, source="profile")
            return value
        if default is _UNSET:
            raise KeyError(f"no topology value for {key!r} on profile {self.name!r} and no default given")
        self.queries[key] = ProfileQuery("topo", key, default, default, source="gpu-default")
        return default

    def supports(self, capability: str) -> bool:
        """Whether the platform provides a boolean capability (e.g.
        "vllm_sleep_mode"). Tests branch on capabilities, never on names."""
        value = self.capabilities.get(capability, False)
        self.queries[f"capability:{capability}"] = ProfileQuery(
            "capability", capability, value, value, source="profile" if capability in self.capabilities else "absent"
        )
        return value

    # -- observability ----------------------------------------------------------

    def dump(self) -> str:
        """Human-readable summary of everything this test resolved through the
        profile: which values came from the platform recipe and which kept
        their GPU-blessed defaults. Print it at test startup, e.g.
        `print(P.dump())`, so NPU triage never requires jumping between the
        test file and the profile file."""
        tag = f"[{self.name.upper()} Profile]"
        lines = [
            f"{tag} platform={self.name} — {self.description}" if self.description else f"{tag} platform={self.name}",
            f"{tag} torch_dist_convert={self.torch_dist_convert} (launch_platform={self.launch_platform})",
        ]
        for q in self._queries_of("path"):
            via = f"via {q.source}" if q.source not in ("", "default") else "platform default"
            lines.append(f"{tag} path: {q.key}={q.value} ({via})")
        if self._queries_of("tune") or self._queries_of("flag"):
            for q in self._queries_of("tune"):
                marker = "" if q.overridden else ", no override"
                lines.append(f"{tag} tune: {q.key}={q.value} (GPU default: {_fmt_default(q.default)}{marker})")
            for q in self._queries_of("flag"):
                if q.value is None or q.value is _UNSET:
                    lines.append(f"{tag} flag: --{q.cli_name} omitted (GPU default: {_fmt_default(q.default)})")
                elif q.value is True:
                    gpu = f"GPU default: {_fmt_default(q.default)}"
                    lines.append(f"{tag} flag: --{q.cli_name} (bare flag; {gpu})")
                else:
                    lines.append(f"{tag} flag: --{q.cli_name}={q.value} (GPU default: {_fmt_default(q.default)})")
        elif self.tuning:
            # startup dump, before the test resolved any knob: show the recipe
            recipe = ", ".join(f"{k}={v}" for k, v in self.tuning.items())
            lines.append(f"{tag} tune recipe (not yet resolved by this test): {recipe}")
        topos = self._queries_of("topo")
        if topos:
            parts = [
                f"{q.key}={q.value}" + (f" (GPU default: {_fmt_default(q.default)})" if q.overridden else "")
                for q in topos
            ]
            lines.append(f"{tag} topology: {', '.join(parts)}")
        elif self.topology:
            recipe = ", ".join(f"{k}={v}" for k, v in self.topology.items())
            lines.append(f"{tag} topology recipe (not yet resolved by this test): {recipe}")
        for q in self._queries_of("capability"):
            lines.append(f"{tag} capability: {q.key}={q.value}")
        if self.train_args_extra:
            lines.append(f"{tag} train_args_extra: {self.train_args_extra.strip()}")
        return "\n".join(lines)

    def _queries_of(self, kind: str) -> list[ProfileQuery]:
        return [q for q in self.queries.values() if q.kind == kind]
