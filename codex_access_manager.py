"""Bootstrap Codex data access across repositories and branches.

This utility creates a `.codex/training_data_snapshot.json` file in each target
repository and installs a `post-checkout` git hook so the snapshot is refreshed
whenever a branch changes.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

HOOK_BODY = """#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(git rev-parse --show-toplevel)"
python3 "$ROOT_DIR/codex_data_access.py" >/dev/null
mkdir -p "$ROOT_DIR/.codex"
cp "$ROOT_DIR/training_data_snapshot.json" "$ROOT_DIR/.codex/training_data_snapshot.json"
"""


def _is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()


def discover_repositories(root: Path) -> list[Path]:
    """Discover git repositories under *root* (including root itself)."""
    repos: list[Path] = []
    if _is_git_repo(root):
        repos.append(root)

    for git_dir in root.glob("**/.git"):
        repo = git_dir.parent
        if repo not in repos:
            repos.append(repo)

    return sorted(repos)


def install_post_checkout_hook(repo: Path) -> Path:
    hooks_dir = repo / ".git" / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_path = hooks_dir / "post-checkout"
    hook_path.write_text(HOOK_BODY, encoding="utf-8")
    hook_path.chmod(0o755)
    return hook_path


def bootstrap_repo(repo: Path) -> None:
    """Create snapshots and branch hook for one repository."""
    snapshot_path = repo / "training_data_snapshot.json"
    subprocess.run(["python3", str(repo / "codex_data_access.py")], cwd=repo, check=True)

    codex_dir = repo / ".codex"
    codex_dir.mkdir(exist_ok=True)
    shutil.copy2(snapshot_path, codex_dir / "training_data_snapshot.json")
    install_post_checkout_hook(repo)


def bootstrap_many(repositories: Iterable[Path]) -> list[Path]:
    updated: list[Path] = []
    for repo in repositories:
        if (repo / "codex_data_access.py").exists():
            bootstrap_repo(repo)
            updated.append(repo)
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enable Codex data access on repositories/branches")
    parser.add_argument(
        "--root",
        default=".",
        help="Root directory to scan for repositories (default: current directory)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    repos = discover_repositories(root)
    updated = bootstrap_many(repos)

    if not updated:
        print("No compatible repositories found (missing codex_data_access.py).")
        return 1

    print("Enabled Codex access for:")
    for repo in updated:
        print(f"- {repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
