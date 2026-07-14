"""Check public-repo notebook hygiene.

The notebooks are examples, not generated reports. They should stay small,
output-free, path-portable, and free of stale article artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_DIR = Path("notebooks")
FORBIDDEN_PATTERNS = (
    "/Users/",
    "\\\\Users\\\\",
    "part_1",
    "medium",
    "docs/images",
    "reports/experiment_datasets/",
    "data/external/experiment_datasets/",
)


def main() -> int:
    failures: list[str] = []
    notebooks = sorted(NOTEBOOK_DIR.glob("*.ipynb"))
    if not notebooks:
        failures.append("No notebooks found under notebooks/.")

    for path in notebooks:
        payload = json.loads(path.read_text(encoding="utf-8"))
        cells = payload.get("cells", [])
        if not cells:
            failures.append(f"{path}: notebook has no cells.")
            continue

        text = "".join("".join(cell.get("source", [])) for cell in cells)
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in text:
                failures.append(f"{path}: contains forbidden pattern {pattern!r}.")

        code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
        for index, cell in enumerate(code_cells, start=1):
            if cell.get("execution_count") is not None:
                failures.append(f"{path}: code cell {index} has execution_count set.")
            if cell.get("outputs"):
                failures.append(f"{path}: code cell {index} has stored outputs.")

        first_markdown = next((cell for cell in cells if cell.get("cell_type") == "markdown"), None)
        if first_markdown is None:
            failures.append(f"{path}: missing introductory markdown cell.")

        if path.name != "06_graph_ml_capabilities.ipynb" and "ROOT = Path.cwd()" not in text:
            failures.append(f"{path}: missing repo-root bootstrap cell.")
        if path.name == "06_graph_ml_capabilities.ipynb" and "ROOT = Path.cwd()" not in text:
            failures.append(f"{path}: missing repo-root bootstrap cell.")

    if failures:
        for failure in failures:
            print(failure)
        return 1

    print(f"notebook-hygiene-ok: {len(notebooks)} notebooks checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
