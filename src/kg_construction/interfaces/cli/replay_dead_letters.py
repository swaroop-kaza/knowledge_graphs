"""Replay dead-lettered graph ingestion batches."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from kg_construction.infrastructure.graph_stores import (
    LadybugIngestionAdapter,
    LadybugIngestionConfig,
    Neo4jIngestionAdapter,
    Neo4jIngestionConfig,
)
from kg_construction.infrastructure.production import DeadLetterQueue, DeadLetterReplayTool
from kg_construction.infrastructure.production.dead_letter import DeadLetterWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay pending graph ingestion dead letters.")
    parser.add_argument("--dead-letter-path", required=True, help="Path to dead-letter JSONL file.")
    parser.add_argument(
        "--store",
        required=True,
        choices=["neo4j_ingestion_adapter", "ladybug_ingestion_adapter"],
    )
    parser.add_argument("--limit", type=int, default=None, help="Maximum pending records to replay.")
    parser.add_argument("--dry-run", action="store_true", help="Inspect replay candidates without writing.")
    parser.add_argument("--neo4j-uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--neo4j-user", default=os.getenv("NEO4J_USER"))
    parser.add_argument("--neo4j-password", default=os.getenv("NEO4J_PASSWORD"))
    parser.add_argument("--neo4j-database", default=os.getenv("NEO4J_DATABASE"))
    parser.add_argument(
        "--ladybug-database",
        default="data/processed/ladybug_graph",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    queue = DeadLetterQueue(Path(args.dead_letter_path))
    writers: dict[str, DeadLetterWriter] = {}
    if not args.dry_run:
        if args.store == "neo4j_ingestion_adapter":
            writers[args.store] = Neo4jIngestionAdapter(
                Neo4jIngestionConfig(
                    uri=args.neo4j_uri,
                    user=args.neo4j_user,
                    password=args.neo4j_password,
                    database=args.neo4j_database,
                )
            )
        else:
            writers[args.store] = LadybugIngestionAdapter(
                LadybugIngestionConfig(database_path=args.ladybug_database)
            )

    report = DeadLetterReplayTool(queue).replay(
        writers,
        store_name=args.store,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
