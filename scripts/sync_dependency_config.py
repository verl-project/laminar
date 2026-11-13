# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["tomlkit==0.13.3", "packaging==25.0"]
# ///
"""Synchronize public uv backend settings from the exact verl source pin.

Laminar owns project metadata and public source pins in pyproject.toml. The
selected verl commit owns GPU backend indexes, sources and resolver policy.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import subprocess
from pathlib import Path
from urllib.parse import urlparse

import tomlkit
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version

BACKENDS = {"vllm", "fsdp", "megatron"}
PROVIDER_REQUIREMENT_COMMENT = "synced from pinned verl"
SYNCED_BEGIN_COMMENT = f"{PROVIDER_REQUIREMENT_COMMENT} (begin)"
SYNCED_END_COMMENT = f"{PROVIDER_REQUIREMENT_COMMENT} (end)"
UPSTREAM_PUBLIC_REPOS = {
    "verl": "https://github.com/verl-project/verl.git",
    "uni-agent": "https://github.com/verl-project/uni-agent.git",
}


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def source_pin(document: dict, name: str) -> dict:
    source = document["tool"]["uv"].get("sources", {}).get(name)
    if not isinstance(source, dict) or set(source) != {"git", "rev"}:
        raise ValueError(f"{name} needs a public Git source with an exact rev")
    url = urlparse(source["git"])
    if url.scheme != "https" or url.hostname != "github.com" or url.username or url.password:
        raise ValueError(f"{name} source must be a public HTTPS GitHub URL")
    if not re.fullmatch(r"[0-9a-f]{40}", source["rev"]):
        raise ValueError(f"{name} source must pin a full commit SHA")
    return copy.deepcopy(source)


def public_git_store(source: dict, cache_dir: Path) -> Path:
    repo = cache_dir / source["rev"]
    if not repo.exists():
        repo.mkdir(parents=True)
        git(repo, "init", "--bare", "--quiet")
    found = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", source["rev"] + "^{commit}"],
        capture_output=True,
    )
    if found.returncode:
        git(repo, "fetch", "--depth=1", source["git"], source["rev"])
    return repo


def backend_requirements(upstream: dict, selected: set[str]) -> list[str]:
    """Expand provider helper extras so root-owned source routing takes effect."""
    extras = upstream["project"]["optional-dependencies"]
    missing = selected - extras.keys()
    if missing:
        raise ValueError(f"verl removed required backends: {sorted(missing)}")
    seen, requirements, pending = set(), [], sorted(selected, reverse=True)
    while pending:
        extra = pending.pop()
        if extra in seen:
            continue
        seen.add(extra)
        for value in extras[extra]:
            requirement = Requirement(value)
            name = canonicalize_name(requirement.name)
            if name == "verl":
                pending.extend(sorted(requirement.extras, reverse=True))
            else:
                requirements.append(value)
    return list(dict.fromkeys(requirements))


def synchronize_backend_extras(document, upstream: dict) -> None:
    extras = document["project"]["optional-dependencies"]
    for backend in sorted(BACKENDS & extras.keys()):
        # Synced entries are bracketed by begin/end comments. Keeping the other
        # lines preserves caller-owned requirements while removing retired pins.
        owned_lines = []
        in_synced_block = False
        for line in extras[backend].as_string().splitlines():
            if SYNCED_BEGIN_COMMENT in line:
                in_synced_block = True
                continue
            if SYNCED_END_COMMENT in line:
                in_synced_block = False
                continue
            if not in_synced_block and PROVIDER_REQUIREMENT_COMMENT not in line:
                owned_lines.append(line)
        owned = tomlkit.parse("requirements = " + "\n".join(owned_lines))["requirements"].unwrap()
        values = tomlkit.array()
        for value in owned:
            values.add_line(value)
        synced = [value for value in backend_requirements(upstream, {backend}) if value not in owned]
        if synced:
            values.add_line(comment=SYNCED_BEGIN_COMMENT)
            for value in synced:
                values.add_line(value)
            values.add_line(comment=SYNCED_END_COMMENT)
        values.add_line(indent="")
        extras[backend] = values


def generate_uv(project: dict, upstream: dict) -> dict:
    selected = BACKENDS & project["project"]["optional-dependencies"].keys()
    if "vllm" not in selected:
        raise ValueError("Laminar's rollout requires the vllm extra")
    packages = {canonicalize_name(Requirement(value).name) for value in backend_requirements(upstream, selected)}
    settings = copy.deepcopy(upstream["tool"]["uv"])
    if "workspace" in settings:
        raise ValueError("upstream workspace needs an explicit standalone adaptation")

    conflicts = []
    for group in settings.get("conflicts", []):
        if any(set(item) != {"extra"} for item in group):
            raise ValueError("new upstream conflict form needs an explicit adaptation")
        chosen = [item for item in group if item["extra"] in selected]
        if len(chosen) > 1:
            conflicts.append(copy.deepcopy(chosen))
            conflicts.append([dict(item, package="verl") for item in chosen])
    settings["conflicts"] = conflicts

    sources = {}
    for name, source in settings.get("sources", {}).items():
        if canonicalize_name(name) not in packages:
            continue
        entries = source if isinstance(source, list) else [source]
        entries = [copy.deepcopy(item) for item in entries if item.get("extra", "vllm") in selected]
        if not entries:
            continue
        if any("path" in item or "workspace" in item for item in entries):
            raise ValueError(f"backend source needs a public standalone adaptation: {name}")
        # Laminar train/eval activate vLLM indirectly. All exposed backends use
        # the same GPU index, so source routing must also cover those entry extras.
        if any("extra" in item for item in entries):
            unmarked = [{key: value for key, value in item.items() if key != "extra"} for item in entries]
            if any(item != unmarked[0] for item in unmarked):
                raise ValueError(f"backend-specific source routing needs adaptation: {name}")
            entries = [unmarked[0]]
        sources[name] = entries[0] if len(entries) == 1 else entries
    for name in ("verl", "uni-agent"):
        sources[name] = source_pin(project, name)
    settings["sources"] = sources
    used_indexes = {
        item["index"]
        for source in sources.values()
        for item in (source if isinstance(source, list) else [source])
        if "index" in item
    }
    settings["index"] = [item for item in settings.get("index", []) if item["name"] in used_indexes]
    settings["dependency-metadata"] = [
        item for item in settings.get("dependency-metadata", []) if canonicalize_name(item["name"]) in packages
    ]
    constraints = project["tool"]["uv"].get("constraint-dependencies", [])
    if constraints:
        settings["constraint-dependencies"] = list(
            dict.fromkeys(settings.get("constraint-dependencies", []) + constraints)
        )
    return settings


def synchronized_document(root: Path, verl_repo: Path) -> str:
    document = tomlkit.parse((root / "pyproject.toml").read_text())
    project = document.unwrap()
    pin = source_pin(project, "verl")["rev"]
    resolved = git(verl_repo, "rev-parse", pin + "^{commit}").decode().strip()
    if resolved != pin:
        raise ValueError("selected verl Git object does not match its source pin")
    upstream = tomlkit.parse(git(verl_repo, "show", f"{pin}:pyproject.toml").decode()).unwrap()
    version = Version(git(verl_repo, "show", f"{pin}:verl/version/version").decode().strip())
    requirements = [
        Requirement(value)
        for values in (
            project["project"].get("dependencies", []),
            *project["project"].get("optional-dependencies", {}).values(),
        )
        for value in values
        if canonicalize_name(Requirement(value).name) == "verl"
    ]
    if not requirements or any(not item.specifier.contains(version, prereleases=True) for item in requirements):
        raise ValueError(f"selected verl version {version} violates Laminar's requirements")
    synchronize_backend_extras(document, upstream)
    table = tomlkit.table()
    table.add(tomlkit.comment("Backend settings below are synchronized from the public verl source pin."))
    for key, value in generate_uv(project, upstream).items():
        if key == "sources":
            sources_table = tomlkit.table(is_super_table=True)
            for pkg_name, pkg_source in value.items():
                upstream_url = UPSTREAM_PUBLIC_REPOS.get(pkg_name)
                if isinstance(pkg_source, dict) and upstream_url and pkg_source.get("git") != upstream_url:
                    pkg_table = tomlkit.table()
                    pkg_table.add(tomlkit.comment(f"TODO: switch back to {upstream_url} once upstream PRs are merged."))
                    for src_key, src_value in pkg_source.items():
                        pkg_table[src_key] = src_value
                    sources_table[pkg_name] = pkg_table
                else:
                    sources_table[pkg_name] = pkg_source
            table[key] = sources_table
            continue
        table[key] = value
        if isinstance(table[key], tomlkit.items.Array) and all(isinstance(item, str) for item in value):
            table[key].multiline(True)
    document["tool"]["uv"] = table
    return tomlkit.dumps(document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--verl-repo", type=Path, help="Existing Git store containing the exact public verl commit")
    parser.add_argument("--check", action="store_true", help="Reject stale public backend configuration")
    default_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "laminar/dependency-config"
    parser.add_argument("--cache-dir", type=Path, default=default_cache)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        project = tomlkit.parse((root / "pyproject.toml").read_text()).unwrap()
        repo = args.verl_repo or public_git_store(source_pin(project, "verl"), args.cache_dir)
        rendered = synchronized_document(root, repo.resolve())
        target = root / "pyproject.toml"
        if args.check:
            if target.read_text() != rendered:
                raise ValueError("dependency configuration is stale; run this script and uv lock")
            print("Public dependency configuration is synchronized")
        else:
            target.write_text(rendered)
            print("Synchronized public backend settings; run uv lock before installing")
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
