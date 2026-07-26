"""Validate current repository documentation links and paragraph layout."""

from __future__ import annotations

import re
import sys
from collections import Counter
from html import unescape as html_unescape
from pathlib import Path
from urllib.parse import unquote

_EXCLUDED_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "blog",
    "build",
    "dist",
    "results",
}
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})(.*)$")
_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
_BARE_LINK_TARGET = r"(?:\\.|[^()\s\\]|\((?:\\.|[^()\s\\])*\))+"
_INLINE_LINK_RE = re.compile(
    rf"!?\[[^\]]*]\(\s*(<[^>\n]+>|{_BARE_LINK_TARGET})"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)"
)
_REFERENCE_LINK_RE = re.compile(r"^\s*\[[^\]]+]:\s*(<[^>]+>|\S+)")
_MARKDOWN_ESCAPE_RE = re.compile(r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])")
_LIST_RE = re.compile(r"^\s{0,3}(?:[-+*]|\d+[.)])\s+")
_STRUCTURAL_PREFIXES = ("#", ">", "|", "<", "[!", ":::", "---", "***", "___")


def _documentation_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.md"):
        relative = path.relative_to(root)
        if any(part in _EXCLUDED_PARTS for part in relative.parts):
            continue
        if relative.parts[:2] == ("tests", "fixtures"):
            continue
        if (
            relative.parts[:2] == ("data", "processed")
            and path.name != "README.md"
        ):
            continue
        files.append(path)
    return sorted(files)


def _heading_text(raw: str) -> str:
    text = re.sub(r"\s+#+\s*$", "", raw)
    text = re.sub(r"\[([^\]]+)]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_unescape(text)
    text = text.replace("`", "").replace("*", "")
    return re.sub(r"(?<!\w)_{1,3}|_{1,3}(?!\w)", "", text)


def _github_slug(raw: str) -> str:
    text = _heading_text(raw).strip().lower()
    text = re.sub(r"[^\w\-\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s", "-", text)


def _advance_fence(line: str, current: str | None) -> tuple[str | None, bool]:
    """Return the active fence and whether *line* is a fence boundary."""

    match = _FENCE_RE.match(line)
    if match is None:
        return current, False
    marker, remainder = match.groups()
    if current is None:
        if marker[0] == "`" and "`" in remainder:
            return None, False
        return marker, True
    if (
        marker[0] == current[0]
        and len(marker) >= len(current)
        and not remainder.strip()
    ):
        return None, True
    return current, False


def _anchors(path: Path) -> set[str]:
    anchors: set[str] = set()
    occurrences: Counter[str] = Counter()
    fence: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        fence, is_boundary = _advance_fence(line, fence)
        if is_boundary or fence is not None:
            continue
        match = _HEADING_RE.match(line)
        if match is None:
            continue
        base = _github_slug(match.group(2))
        candidate = base
        while candidate in occurrences:
            occurrences[base] += 1
            candidate = f"{base}-{occurrences[base]}"
        occurrences[candidate] = 0
        anchors.add(candidate)
    return anchors


def _split_target(raw: str) -> tuple[str, str | None]:
    target = raw[1:-1] if raw.startswith("<") and raw.endswith(">") else raw
    target = _MARKDOWN_ESCAPE_RE.sub(r"\1", target)
    path, separator, fragment = target.partition("#")
    return unquote(path), unquote(fragment) if separator else None


def _validate_target(
    *,
    root: Path,
    source: Path,
    line_number: int,
    raw_target: str,
    anchor_cache: dict[Path, set[str]],
) -> list[str]:
    target = raw_target.strip()
    if target.startswith(("http://", "https://", "mailto:", "tel:", "data:")):
        return []
    relative_target, fragment = _split_target(target)
    destination = (
        source
        if not relative_target
        else (source.parent / relative_target).resolve()
    )
    try:
        destination.relative_to(root)
    except ValueError:
        return [
            f"{source.relative_to(root)}:{line_number}: "
            f"link escapes repository: {target}"
        ]
    if not destination.exists():
        return [
            f"{source.relative_to(root)}:{line_number}: "
            f"missing local link target: {target}"
        ]
    if fragment and destination.is_file() and destination.suffix.lower() == ".md":
        anchors = anchor_cache.setdefault(destination, _anchors(destination))
        if fragment not in anchors:
            return [
                f"{source.relative_to(root)}:{line_number}: "
                f"missing heading anchor #{fragment} in "
                f"{destination.relative_to(root)}"
            ]
    return []


def _is_prose(line: str, *, in_fence: bool) -> bool:
    if in_fence or not line.strip():
        return False
    stripped = line.lstrip()
    if len(line) - len(stripped) >= 4:
        return False
    if stripped.startswith(_STRUCTURAL_PREFIXES):
        return False
    if _LIST_RE.match(line) or _REFERENCE_LINK_RE.match(line):
        return False
    return re.match(r"^\s*:\s+", line) is None


def _validate_file(
    root: Path,
    path: Path,
    anchor_cache: dict[Path, set[str]],
) -> list[str]:
    errors: list[str] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    fence: str | None = None
    previous_prose = False
    previous_line_number = 0
    for line_number, line in enumerate(lines, start=1):
        fence, is_boundary = _advance_fence(line, fence)
        if is_boundary or fence is not None:
            previous_prose = False
            continue
        for match in _INLINE_LINK_RE.finditer(line):
            errors.extend(
                _validate_target(
                    root=root,
                    source=path,
                    line_number=line_number,
                    raw_target=match.group(1),
                    anchor_cache=anchor_cache,
                )
            )
        reference = _REFERENCE_LINK_RE.match(line)
        if reference is not None:
            errors.extend(
                _validate_target(
                    root=root,
                    source=path,
                    line_number=line_number,
                    raw_target=reference.group(1),
                    anchor_cache=anchor_cache,
                )
            )
        prose = _is_prose(line, in_fence=False)
        if prose and previous_prose:
            errors.append(
                f"{path.relative_to(root)}:{line_number}: unintended hard wrap "
                f"after line {previous_line_number}; keep one logical paragraph "
                "on one source line"
            )
        previous_prose = prose
        previous_line_number = line_number
    if fence is not None:
        errors.append(f"{path.relative_to(root)}: unclosed fenced code block")
    return errors


def _linked_documentation_files(
    root: Path,
    source: Path,
    documentation_files: set[Path],
) -> set[Path]:
    """Return repository documentation pages linked from *source*."""

    destinations: set[Path] = set()
    fence: str | None = None
    for line in source.read_text(encoding="utf-8").splitlines():
        fence, is_boundary = _advance_fence(line, fence)
        if is_boundary or fence is not None:
            continue
        raw_targets = [match.group(1) for match in _INLINE_LINK_RE.finditer(line)]
        reference = _REFERENCE_LINK_RE.match(line)
        if reference is not None:
            raw_targets.append(reference.group(1))
        for raw_target in raw_targets:
            target = raw_target.strip()
            if target.startswith(("http://", "https://", "mailto:", "tel:", "data:")):
                continue
            relative_target, _ = _split_target(target)
            if not relative_target:
                continue
            destination = (source.parent / relative_target).resolve()
            if destination.is_dir():
                destination /= "README.md"
            if destination in documentation_files:
                destinations.add(destination)
    return destinations


def _unreachable_documentation_files(root: Path, files: list[Path]) -> list[Path]:
    """Find documentation pages that cannot be reached from the root README."""

    documentation_files = {path.resolve() for path in files}
    root_readme = (root / "README.md").resolve()
    if root_readme not in documentation_files:
        return sorted(documentation_files)
    graph = {
        path: _linked_documentation_files(root, path, documentation_files)
        for path in documentation_files
    }
    reachable: set[Path] = set()
    pending = [root_readme]
    while pending:
        current = pending.pop()
        if current in reachable:
            continue
        reachable.add(current)
        pending.extend(graph[current] - reachable)
    return sorted(documentation_files - reachable)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    files = _documentation_files(root)
    anchor_cache: dict[Path, set[str]] = {}
    errors = [
        error
        for path in files
        for error in _validate_file(root, path, anchor_cache)
    ]
    errors.extend(
        f"{path.relative_to(root)}: documentation page is not reachable from README.md"
        for path in _unreachable_documentation_files(root, files)
    )
    if errors:
        print("\n".join(errors))
        print(
            f"markdown-docs-check-failed: {len(errors)} issue(s) across "
            f"{len(files)} files",
            file=sys.stderr,
        )
        return 1
    print(
        f"markdown-docs-check-ok: {len(files)} files; "
        "navigation, local targets, heading anchors, fences, and paragraph layout valid"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
