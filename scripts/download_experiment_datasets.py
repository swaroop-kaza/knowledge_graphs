#!/usr/bin/env python3
"""Download bounded public datasets for Data-to-KG experiments.

The full upstream datasets behind PubMed, DBLP, and Enron are large. This
script downloads official or well-known public samples and records every
sampling decision in a manifest so experiments remain reproducible.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import sqlite3
import ssl
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "external" / "experiment_datasets"
REPORT_DIR = ROOT / "reports" / "experiment_datasets"
MANIFEST_PATH = REPORT_DIR / "dataset_download_manifest.json"

USER_AGENT = (
    "kg-construction-lab/0.1 "
    "(bounded public research dataset sampler; contact: swaroopkaza)"
)


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(
    *,
    dataset_id: str,
    source_name: str,
    source_url: str,
    path: Path,
    status: str,
    notes: str = "",
    rows: int | None = None,
    sampled_from: str | None = None,
) -> dict[str, Any]:
    path = path.resolve()
    record: dict[str, Any] = {
        "dataset_id": dataset_id,
        "source_name": source_name,
        "source_url": source_url,
        "path": str(path),
        "relative_path": str(path.relative_to(ROOT)),
        "status": status,
        "bytes": path.stat().st_size if path.exists() else 0,
        "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        "notes": notes,
    }
    if rows is not None:
        record["rows"] = rows
    if sampled_from is not None:
        record["sampled_from"] = sampled_from
    return record


def note_record(dataset_id: str, source_name: str, source_url: str, status: str, notes: str) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "source_name": source_name,
        "source_url": source_url,
        "path": None,
        "relative_path": None,
        "status": status,
        "bytes": 0,
        "sha256": None,
        "notes": notes,
    }


def open_url(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float = 60.0,
):
    request_headers = {"User-Agent": USER_AGENT, **dict(headers or {})}
    request = urllib.request.Request(url, headers=request_headers)
    try:
        return urllib.request.urlopen(request, timeout=timeout_seconds)
    except urllib.error.URLError as exc:
        if "CERTIFICATE_VERIFY_FAILED" not in str(exc) and "self-signed" not in str(exc):
            raise
        context = ssl._create_unverified_context()  # noqa: S323 - local dataset sampler fallback.
        return urllib.request.urlopen(request, timeout=timeout_seconds, context=context)


def download_file(
    *,
    dataset_id: str,
    source_name: str,
    source_url: str,
    destination: Path,
    records: list[dict[str, Any]],
    force: bool,
    headers: Mapping[str, str] | None = None,
    timeout_seconds: float = 60.0,
) -> Path | None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        records.append(
            file_record(
                dataset_id=dataset_id,
                source_name=source_name,
                source_url=source_url,
                path=destination,
                status="exists",
            )
        )
        return destination

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with open_url(source_url, headers=headers, timeout_seconds=timeout_seconds) as response, temporary.open(
            "wb"
        ) as handle:
            shutil.copyfileobj(response, handle)
        temporary.replace(destination)
        records.append(
            file_record(
                dataset_id=dataset_id,
                source_name=source_name,
                source_url=source_url,
                path=destination,
                status="downloaded",
            )
        )
        return destination
    except Exception as exc:  # pragma: no cover - network-dependent guard.
        temporary.unlink(missing_ok=True)
        records.append(
            note_record(
                dataset_id,
                source_name,
                source_url,
                "failed",
                f"{type(exc).__name__}: {exc}",
            )
        )
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def count_csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def download_chinook(records: list[dict[str, Any]], *, force: bool) -> None:
    download_file(
        dataset_id="chinook_sqlite",
        source_name="Chinook sample relational database",
        source_url=(
            "https://raw.githubusercontent.com/lerocha/chinook-database/master/"
            "ChinookDatabase/DataSources/Chinook_Sqlite.sqlite"
        ),
        destination=DATA_DIR / "chinook" / "Chinook_Sqlite.sqlite",
        records=records,
        force=force,
    )


def download_w3c_rdb2rdf(records: list[dict[str, Any]], *, force: bool) -> None:
    base = "https://dvcs.w3.org/hg/rdb2rdf-tests/raw-file/default/D001-1table1column1row"
    target_dir = DATA_DIR / "w3c_rdb2rdf" / "D001-1table1column1row"
    create_sql = download_file(
        dataset_id="w3c_rdb2rdf_d001",
        source_name="W3C RDB2RDF direct mapping test D001 create.sql",
        source_url=f"{base}/create.sql",
        destination=target_dir / "create.sql",
        records=records,
        force=force,
    )
    download_file(
        dataset_id="w3c_rdb2rdf_d001",
        source_name="W3C RDB2RDF direct mapping test D001 manifest.ttl",
        source_url=f"{base}/manifest.ttl",
        destination=target_dir / "manifest.ttl",
        records=records,
        force=force,
    )
    download_file(
        dataset_id="w3c_rdb2rdf_d001",
        source_name="W3C RDB2RDF direct mapping expected graph Turtle",
        source_url=f"{base}/directGraph.ttl",
        destination=target_dir / "directGraph.ttl",
        records=records,
        force=force,
    )
    if create_sql is None:
        return
    sqlite_path = target_dir / "D001.sqlite"
    if sqlite_path.exists() and not force:
        records.append(
            file_record(
                dataset_id="w3c_rdb2rdf_d001",
                source_name="SQLite database built from W3C RDB2RDF D001 create.sql",
                source_url=str(create_sql),
                path=sqlite_path,
                status="exists",
            )
        )
        return
    try:
        sqlite_path.unlink(missing_ok=True)
        with sqlite3.connect(sqlite_path) as connection:
            connection.executescript(create_sql.read_text(encoding="utf-8"))
        records.append(
            file_record(
                dataset_id="w3c_rdb2rdf_d001",
                source_name="SQLite database built from W3C RDB2RDF D001 create.sql",
                source_url=str(create_sql),
                path=sqlite_path,
                status="generated",
            )
        )
    except Exception as exc:  # pragma: no cover - fixture generation guard.
        records.append(
            note_record(
                "w3c_rdb2rdf_d001",
                "SQLite database built from W3C RDB2RDF D001 create.sql",
                str(create_sql),
                "failed",
                f"{type(exc).__name__}: {exc}",
            )
        )


def download_nyc_311(records: list[dict[str, Any]], *, force: bool) -> None:
    url = (
        "https://data.cityofnewyork.us/resource/erm2-nwe9.csv?"
        "$limit=1000&"
        "$select=unique_key,created_date,closed_date,agency,complaint_type,"
        "descriptor,incident_zip,borough,status,resolution_description&"
        "$order=created_date%20DESC"
    )
    path = download_file(
        dataset_id="nyc_311_service_requests",
        source_name="NYC 311 Service Requests from 2020 to present sample",
        source_url=url,
        destination=DATA_DIR / "nyc_311" / "nyc_311_service_requests_sample.csv",
        records=records,
        force=force,
        timeout_seconds=90,
    )
    if path is not None:
        records.append(
            file_record(
                dataset_id="nyc_311_service_requests",
                source_name="NYC 311 CSV row count marker",
                source_url=url,
                path=path,
                status="profiled",
                rows=count_csv_rows(path),
            )
        )


def download_sec_companyfacts(records: list[dict[str, Any]], *, force: bool) -> None:
    sec_headers = {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    full_path = download_file(
        dataset_id="sec_apple_companyfacts",
        source_name="SEC EDGAR Apple companyfacts JSON",
        source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json",
        destination=DATA_DIR / "sec" / "apple_companyfacts_full.json",
        records=records,
        force=force,
        headers=sec_headers,
        timeout_seconds=90,
    )
    if full_path is None:
        return
    sample_path = DATA_DIR / "sec" / "apple_companyfacts_sample.json"
    if sample_path.exists() and not force:
        records.append(
            file_record(
                dataset_id="sec_apple_companyfacts",
                source_name="SEC EDGAR Apple companyfacts bounded sample",
                source_url=str(full_path),
                path=sample_path,
                status="exists",
                sampled_from=str(full_path),
            )
        )
        return
    payload = json.loads(full_path.read_text(encoding="utf-8"))
    facts = payload.get("facts", {}).get("us-gaap", {})
    preferred_concepts = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "Assets",
        "Liabilities",
        "NetIncomeLoss",
        "OperatingIncomeLoss",
        "AccountsPayableCurrent",
    ]
    sampled_facts: dict[str, Any] = {}
    for concept in preferred_concepts:
        concept_payload = facts.get(concept)
        if not isinstance(concept_payload, dict):
            continue
        sampled_units: dict[str, Any] = {}
        for unit, values in concept_payload.get("units", {}).items():
            if not isinstance(values, list):
                continue
            sampled_units[unit] = sorted(
                values,
                key=lambda item: str(item.get("end", "")) if isinstance(item, dict) else "",
                reverse=True,
            )[:20]
        sampled_facts[concept] = {
            "label": concept_payload.get("label"),
            "description": concept_payload.get("description"),
            "units": sampled_units,
        }
    write_json(
        sample_path,
        {
            "cik": payload.get("cik"),
            "entityName": payload.get("entityName"),
            "facts": {"us-gaap": sampled_facts},
            "sampling": {
                "concepts": list(sampled_facts),
                "max_facts_per_unit": 20,
                "source": str(full_path),
            },
        },
    )
    records.append(
        file_record(
            dataset_id="sec_apple_companyfacts",
            source_name="SEC EDGAR Apple companyfacts bounded sample",
            source_url=str(full_path),
            path=sample_path,
            status="generated",
            sampled_from=str(full_path),
        )
    )


def sample_pubmed(records: list[dict[str, Any]], *, force: bool, max_articles: int) -> None:
    url = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed26n0001.xml.gz"
    output = DATA_DIR / "pubmed" / "pubmed26n0001_sample.xml"
    if output.exists() and not force:
        records.append(
            file_record(
                dataset_id="pubmed_baseline",
                source_name="PubMed baseline first-file XML sample",
                source_url=url,
                path=output,
                status="exists",
                rows=max_articles,
            )
        )
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with (
            open_url(url, timeout_seconds=120) as response,
            gzip.GzipFile(fileobj=response) as gzipped,
            output.open("w", encoding="utf-8") as handle,
        ):
            context = ET.iterparse(gzipped, events=("end",))
            handle.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            handle.write("<PubmedArticleSet>\n")
            for _event, element in context:
                if element.tag.endswith("PubmedArticle"):
                    handle.write(ET.tostring(element, encoding="unicode"))
                    handle.write("\n")
                    count += 1
                    element.clear()
                    if count >= max_articles:
                        break
            handle.write("</PubmedArticleSet>\n")
        records.append(
            file_record(
                dataset_id="pubmed_baseline",
                source_name="PubMed baseline first-file XML sample",
                source_url=url,
                path=output,
                status="sampled",
                rows=count,
                sampled_from=url,
            )
        )
    except Exception as exc:  # pragma: no cover - network-dependent guard.
        output.unlink(missing_ok=True)
        records.append(
            note_record(
                "pubmed_baseline",
                "PubMed baseline first-file XML sample",
                url,
                "failed",
                f"{type(exc).__name__}: {exc}",
            )
        )


def download_dblp(records: list[dict[str, Any]], *, force: bool) -> None:
    url = "https://dblp.uni-trier.de/search/publ/api?q=knowledge%20graph&h=100&format=json"
    download_file(
        dataset_id="dblp_publication_api",
        source_name="DBLP publication API search sample for knowledge graph",
        source_url=url,
        destination=DATA_DIR / "dblp" / "dblp_knowledge_graph_publications.json",
        records=records,
        force=force,
        timeout_seconds=90,
    )
    records.append(
        note_record(
            "dblp_publication_api",
            "DBLP full XML dump",
            "https://dblp.org/xml/dblp.xml.gz",
            "not_downloaded_full_corpus",
            "The full DBLP XML dump is intentionally not mirrored; the API sample is enough for JSON/API mapping tests.",
        )
    )


def download_loghub(records: list[dict[str, Any]], *, force: bool) -> None:
    base = "https://raw.githubusercontent.com/logpai/loghub/master/HDFS"
    download_file(
        dataset_id="loghub_hdfs",
        source_name="Loghub HDFS 2k raw log",
        source_url=f"{base}/HDFS_2k.log",
        destination=DATA_DIR / "loghub" / "HDFS_2k.log",
        records=records,
        force=force,
    )
    structured_path = download_file(
        dataset_id="loghub_hdfs",
        source_name="Loghub HDFS 2k structured CSV",
        source_url=f"{base}/HDFS_2k.log_structured.csv",
        destination=DATA_DIR / "loghub" / "HDFS_2k.log_structured.csv",
        records=records,
        force=force,
    )
    if structured_path is not None:
        records.append(
            file_record(
                dataset_id="loghub_hdfs",
                source_name="Loghub HDFS structured row count marker",
                source_url=f"{base}/HDFS_2k.log_structured.csv",
                path=structured_path,
                status="profiled",
                rows=count_csv_rows(structured_path),
            )
        )


WIKIANN_LABELS = {
    0: "O",
    1: "Person",
    2: "Person",
    3: "Organization",
    4: "Organization",
    5: "Location",
    6: "Location",
}


def spans_from_iob(tokens: list[str], tags: list[int]) -> list[dict[str, Any]]:
    text = " ".join(tokens)
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = cursor
        end = start + len(token)
        offsets.append((start, end))
        cursor = end + 1

    spans: list[dict[str, Any]] = []
    current_label: str | None = None
    current_start = 0
    current_end = 0
    for index, tag in enumerate(tags):
        label = WIKIANN_LABELS.get(int(tag), "O")
        raw_tag = int(tag)
        is_begin = raw_tag in {1, 3, 5}
        if label == "O":
            if current_label is not None:
                spans.append(
                    {
                        "label": current_label,
                        "text": text[current_start:current_end],
                        "start": current_start,
                        "end": current_end,
                    }
                )
            current_label = None
            continue
        start, end = offsets[index]
        if current_label != label or is_begin:
            if current_label is not None:
                spans.append(
                    {
                        "label": current_label,
                        "text": text[current_start:current_end],
                        "start": current_start,
                        "end": current_end,
                    }
                )
            current_label = label
            current_start = start
        current_end = end
    if current_label is not None:
        spans.append(
            {
                "label": current_label,
                "text": text[current_start:current_end],
                "start": current_start,
                "end": current_end,
            }
        )
    return spans


def download_wikiann(records: list[dict[str, Any]], *, force: bool, max_rows: int) -> None:
    parquet_path = download_file(
        dataset_id="wikiann_en_ner",
        source_name="WikiANN English train parquet",
        source_url=(
            "https://huggingface.co/datasets/unimelb-nlp/wikiann/resolve/main/"
            "en/train-00000-of-00001.parquet"
        ),
        destination=DATA_DIR / "wikiann" / "wikiann_en_train.parquet",
        records=records,
        force=force,
        timeout_seconds=120,
    )
    if parquet_path is None:
        return
    jsonl_path = DATA_DIR / "wikiann" / "wikiann_en_sample.jsonl"
    text_path = DATA_DIR / "wikiann" / "wikiann_en_sample.txt"
    if jsonl_path.exists() and text_path.exists() and not force:
        records.append(
            file_record(
                dataset_id="wikiann_en_ner",
                source_name="WikiANN English bounded JSONL sample",
                source_url=str(parquet_path),
                path=jsonl_path,
                status="exists",
                rows=max_rows,
                sampled_from=str(parquet_path),
            )
        )
        records.append(
            file_record(
                dataset_id="wikiann_en_ner",
                source_name="WikiANN English bounded text sample",
                source_url=str(parquet_path),
                path=text_path,
                status="exists",
                rows=max_rows,
                sampled_from=str(parquet_path),
            )
        )
        return
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dataset-tool dependency guard.
        records.append(
            note_record(
                "wikiann_en_ner",
                "WikiANN English sample conversion",
                str(parquet_path),
                "failed",
                f"pandas is required to convert parquet samples: {exc}",
            )
        )
        return
    frame = pd.read_parquet(parquet_path).head(max_rows)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for _, row in frame.iterrows():
            tokens = [str(token) for token in row["tokens"]]
            tags = [int(tag) for tag in row["ner_tags"]]
            text = " ".join(tokens)
            lines.append(text)
            handle.write(
                json.dumps(
                    {
                        "tokens": tokens,
                        "ner_tags": tags,
                        "text": text,
                        "spans": spans_from_iob(tokens, tags),
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    text_path.write_text("\n".join(lines), encoding="utf-8")
    records.append(
        file_record(
            dataset_id="wikiann_en_ner",
            source_name="WikiANN English bounded JSONL sample",
            source_url=str(parquet_path),
            path=jsonl_path,
            status="generated",
            rows=len(frame),
            sampled_from=str(parquet_path),
        )
    )
    records.append(
        file_record(
            dataset_id="wikiann_en_ner",
            source_name="WikiANN English bounded text sample",
            source_url=str(parquet_path),
            path=text_path,
            status="generated",
            rows=len(frame),
            sampled_from=str(parquet_path),
        )
    )


def category_from_zip_name(name: str) -> str:
    lowered = name.lower()
    for category in ("ham", "spam", "test"):
        if f"/{category}/" in lowered or lowered.startswith(f"{category}/"):
            return category
    return "other"


def safe_file_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "sample.txt"


def selected_zip_members(names: Iterable[str], per_category: int) -> list[str]:
    buckets: dict[str, list[str]] = {"ham": [], "spam": [], "test": [], "other": []}
    for name in names:
        if name.endswith("/") or not name.lower().endswith(".txt"):
            continue
        buckets[category_from_zip_name(name)].append(name)
    selected: list[str] = []
    for category in ("ham", "spam", "test", "other"):
        selected.extend(sorted(buckets[category])[:per_category])
    return selected


def download_enron_sample(records: list[dict[str, Any]], *, force: bool, per_category: int) -> None:
    zip_path = download_file(
        dataset_id="enron_email_subset",
        source_name="Enron-derived spam classification email text archive",
        source_url="https://github.com/crossedbanana/Enron-Email-Classification/raw/master/data.zip",
        destination=DATA_DIR / "enron" / "enron_email_classification_data.zip",
        records=records,
        force=force,
        timeout_seconds=120,
    )
    records.append(
        note_record(
            "enron_email_subset",
            "CMU Enron full maildir archive",
            "https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz",
            "not_downloaded_full_corpus",
            "The full CMU archive is about 1.7 GB compressed and is not mirrored by this bounded experiment downloader.",
        )
    )
    if zip_path is None:
        return
    extract_dir = DATA_DIR / "enron" / "sample_email_texts"
    corpus_path = DATA_DIR / "enron" / "enron_sample_corpus.txt"
    eml_path = DATA_DIR / "enron" / "enron_wrapped_sample.eml"
    if corpus_path.exists() and eml_path.exists() and not force:
        records.append(
            file_record(
                dataset_id="enron_email_subset",
                source_name="Enron-derived bounded text corpus",
                source_url=str(zip_path),
                path=corpus_path,
                status="exists",
                sampled_from=str(zip_path),
            )
        )
        records.append(
            file_record(
                dataset_id="enron_email_subset",
                source_name="Enron-derived wrapped EML sample",
                source_url=str(zip_path),
                path=eml_path,
                status="exists",
                sampled_from=str(zip_path),
            )
        )
        return
    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[tuple[str, Path, str]] = []
    with zipfile.ZipFile(zip_path) as archive:
        for member in selected_zip_members(archive.namelist(), per_category):
            category = category_from_zip_name(member)
            output = extract_dir / category / safe_file_name(Path(member).name)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(archive.read(member))
            extracted.append((member, output, category))
    corpus_blocks: list[str] = []
    for member, output, category in extracted:
        text = output.read_text(encoding="utf-8", errors="replace")
        corpus_blocks.append(f"SOURCE_FILE: {member}\nCATEGORY: {category}\n{text.strip()}")
    corpus_path.write_text("\n\n--- EMAIL SAMPLE BREAK ---\n\n".join(corpus_blocks), encoding="utf-8")
    body = extracted[0][1].read_text(encoding="utf-8", errors="replace") if extracted else ""
    eml_path.write_text(
        "\n".join(
            [
                "From: enron.sample@example.com",
                "To: kg-construction@example.com",
                "Subject: Enron-derived bounded sample",
                "Date: Mon, 01 Jan 2001 09:00:00 -0600",
                "Message-ID: <enron-derived-bounded-sample@example.com>",
                "",
                body.strip(),
            ]
        ),
        encoding="utf-8",
    )
    records.append(
        file_record(
            dataset_id="enron_email_subset",
            source_name="Enron-derived bounded text corpus",
            source_url=str(zip_path),
            path=corpus_path,
            status="generated",
            rows=len(extracted),
            sampled_from=str(zip_path),
        )
    )
    records.append(
        file_record(
            dataset_id="enron_email_subset",
            source_name="Enron-derived wrapped EML sample",
            source_url=str(zip_path),
            path=eml_path,
            status="generated",
            rows=1,
            sampled_from=str(zip_path),
            notes="Wrapped from text subset so the repository EmailLoader can exercise RFC 5322 parsing.",
        )
    )


def write_manifest(records: list[dict[str, Any]]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    dataset_ids = sorted({str(record["dataset_id"]) for record in records})
    manifest = {
        "generated_at": now_iso(),
        "data_dir": str(DATA_DIR.resolve()),
        "dataset_count": len(dataset_ids),
        "artifact_count": len(records),
        "datasets": dataset_ids,
        "records": records,
    }
    write_json(MANIFEST_PATH, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Refresh existing downloaded/generated artifacts.")
    parser.add_argument("--wikiann-rows", type=int, default=250, help="Rows to keep from WikiANN parquet.")
    parser.add_argument("--pubmed-articles", type=int, default=50, help="PubMed articles to sample from baseline XML.")
    parser.add_argument("--enron-per-category", type=int, default=8, help="Emails to keep per Enron text category.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    started = time.perf_counter()

    download_chinook(records, force=args.force)
    download_w3c_rdb2rdf(records, force=args.force)
    download_nyc_311(records, force=args.force)
    download_sec_companyfacts(records, force=args.force)
    sample_pubmed(records, force=args.force, max_articles=args.pubmed_articles)
    download_dblp(records, force=args.force)
    download_loghub(records, force=args.force)
    download_wikiann(records, force=args.force, max_rows=args.wikiann_rows)
    download_enron_sample(records, force=args.force, per_category=args.enron_per_category)

    records.append(
        {
            "dataset_id": "download_run",
            "source_name": "download runtime",
            "source_url": None,
            "path": None,
            "relative_path": None,
            "status": "completed",
            "bytes": 0,
            "sha256": None,
            "notes": f"Downloader completed in {time.perf_counter() - started:.2f} seconds.",
        }
    )
    write_manifest(records)
    print(f"Wrote dataset manifest: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
