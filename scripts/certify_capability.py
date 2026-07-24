#!/usr/bin/env python3
"""Execute one manifest capability and emit nonce-bound semantic evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from kg_construction.certification.capabilities import certify_capability


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing capability_manifest.yaml.",
    )
    return parser


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required certification environment variable {name} is missing")
    return value


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        capability = _required_environment("KG_CERT_CAPABILITY")
        profile = _required_environment("KG_CERT_PROFILE")
        run_id = _required_environment("KG_CERT_RUN_ID")
        evidence_path = Path(_required_environment("KG_CERT_EVIDENCE_PATH"))
        evidence = certify_capability(
            capability,
            profile=profile,
            run_id=run_id,
            root=args.root.resolve(),
        )
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(
            json.dumps(evidence, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except Exception as exc:
        print(
            f"semantic certification failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    if evidence["observations"]["passed"] is not True:
        print(
            f"semantic certification threshold failed for {capability}",
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "capability": capability,
                "profile": profile,
                "run_id": run_id,
                "status": "passed",
                "metrics": evidence["observations"]["evaluated_evidence"]["metrics"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
