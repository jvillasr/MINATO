"""Fail when built MINATO artefacts contain non-release material."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import tarfile
import zipfile


DEVELOPMENT_RECORDS = {
    "AGENTS.md",
    "CHANGELOG.md",
    "NOTES.md",
    "ROADMAP.md",
}
FORBIDDEN_TREES = {
    ("contrib",),
    ("minato", "contrib"),
    ("minato", "models"),
    ("minato", "tutorials"),
}


def archive_members(path: Path) -> list[str]:
    """Return regular-file names from a wheel or source archive."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return [item.filename for item in archive.infolist() if not item.is_dir()]
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, mode="r:gz") as archive:
            return [item.name for item in archive.getmembers() if item.isfile()]
    raise ValueError(f"Unsupported release artefact: {path}")


def contains_tree(parts: tuple[str, ...], tree: tuple[str, ...]) -> bool:
    """Return whether *tree* occurs as consecutive path components."""
    width = len(tree)
    return any(parts[index : index + width] == tree for index in range(len(parts) - width + 1))


def violations(member: str) -> list[str]:
    """Describe release-policy violations for one archive member."""
    parts = PurePosixPath(member).parts
    problems: list[str] = []

    if any(contains_tree(parts, tree) for tree in FORBIDDEN_TREES):
        problems.append("contains a development-only or data tree")
    if contains_tree(parts, ("minato", "spdis.py")):
        problems.append("contains the external disentangling adaptation")

    basename = parts[-1]
    if basename in DEVELOPMENT_RECORDS or basename.endswith(("_PLAN.md", "_PLANS.md")):
        problems.append("contains a development record")

    if "minato" in parts:
        package_index = parts.index("minato")
        package_parts = parts[package_index + 1 :]
        if package_parts and not basename.endswith(".py"):
            problems.append("contains non-Python package data")

    return problems


def find_artifacts(inputs: list[Path]) -> list[Path]:
    """Expand input directories into sorted wheel and source archives."""
    artifacts: list[Path] = []
    for path in inputs:
        if path.is_dir():
            artifacts.extend(path.glob("*.whl"))
            artifacts.extend(path.glob("*.tar.gz"))
        else:
            artifacts.append(path)
    return sorted(set(artifacts))


def check_artifact(path: Path) -> list[str]:
    """Return formatted policy violations for one release artefact."""
    problems: list[str] = []
    for member in archive_members(path):
        for problem in violations(member):
            problems.append(f"{path.name}: {member}: {problem}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Artefacts or directories to inspect")
    args = parser.parse_args()

    artifacts = find_artifacts(args.paths)
    if not artifacts:
        parser.error("no wheel or .tar.gz source archive found")

    problems = [problem for artifact in artifacts for problem in check_artifact(artifact)]
    if problems:
        print("Release artefact policy failed:")
        for problem in problems:
            print(f"- {problem}")
        return 1

    for artifact in artifacts:
        print(f"Release artefact policy passed: {artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
