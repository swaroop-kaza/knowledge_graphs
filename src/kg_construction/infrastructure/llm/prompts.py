"""Prompt templates for evidence-grounded graph extraction."""

from __future__ import annotations

import json
from typing import Any

GRAPH_EXTRACTION_PROMPT_VERSION = "ollama_graph_extraction_v1"
LABEL_PROPOSAL_PROMPT_VERSION = "ollama_label_proposal_v1"
RELATION_TYPING_PROMPT_VERSION = "ollama_relation_typing_v1"


def build_graph_extraction_prompt(
    *,
    chunk_text: str,
    allowed_node_types: list[str],
    allowed_relationship_types: list[str],
) -> str:
    """Build the required evidence-grounded graph extraction prompt."""

    return f"""You are extracting a knowledge graph from source text.

Return only valid JSON.

Use only facts explicitly supported by the text.
Every node and relationship must include evidence_text.
Do not infer facts that are not present.
If unsure, omit the fact.

Allowed node types:
{json.dumps(allowed_node_types)}

Allowed relationship types:
{json.dumps(allowed_relationship_types)}

Text:
{chunk_text}

Return:
{{
  "nodes": [
    {{
      "id_hint": "...",
      "type": "...",
      "name": "...",
      "properties": {{}},
      "evidence_text": "...",
      "confidence": 0.0
    }}
  ],
  "relationships": [
    {{
      "source": "...",
      "type": "...",
      "target": "...",
      "properties": {{}},
      "evidence_text": "...",
      "confidence": 0.0
    }}
  ]
}}
"""


def build_repair_prompt(*, original_prompt: str, invalid_output: str, error: str) -> str:
    """Ask the model to repair malformed JSON while preserving the original task."""

    return f"""Repair the previous response into valid JSON only.

Original task:
{original_prompt}

Invalid response:
{invalid_output}

Validation error:
{error}

Return only corrected JSON. Do not add markdown fences or commentary.
"""


DEFAULT_LABEL_PROPOSAL_TEMPLATE = """You are proposing entity labels for a GLiNER named entity recognition pass.

Return only valid JSON.
Do not extract entities yet.
Propose labels that are useful for knowledge graph construction from the text.
Prefer reusable graph labels such as Person, Organization, System, Incident, Date, Product, Location, Metric, Document, API, ErrorCode.
Avoid labels that are too broad, such as Thing, Object, Data, Text, Other, Unknown.
Return at most {allowed_label_count} labels.

Seed labels that may be useful:
{seed_labels}

Text:
{chunk_text}

Return:
{{
  "labels": [
    {{
      "name": "System",
      "description": "Software service, API, application, model, or platform component.",
      "rationale": "Why this label is useful for this text.",
      "confidence": 0.0
    }}
  ]
}}
"""

DEFAULT_RELATION_TYPING_TEMPLATE = """You are typing relationships between already extracted entity spans.

Return only valid JSON.
Use only relationships explicitly supported by the text.
Every relationship must include evidence_text copied exactly from the text.
Do not create new entities.
If the evidence is weak or implied, omit the relationship.

Allowed relationship types:
{allowed_relationship_types}

Extracted entities:
{entities}

Text:
{chunk_text}

Return:
{{
  "relations": [
    {{
      "source": "entity name or candidate id from Extracted entities",
      "type": "USES",
      "target": "entity name or candidate id from Extracted entities",
      "properties": {{}},
      "evidence_text": "exact text span supporting the relationship",
      "confidence": 0.0
    }}
  ]
}}
"""


def build_label_proposal_prompt(
    *,
    chunk_text: str,
    allowed_label_count: int,
    seed_labels: list[str],
    prompt_template: str = DEFAULT_LABEL_PROPOSAL_TEMPLATE,
) -> str:
    """Build the label-discovery prompt used before GLiNER extraction."""

    return prompt_template.format(
        chunk_text=chunk_text,
        allowed_label_count=allowed_label_count,
        seed_labels=json.dumps(seed_labels),
    )


def build_relation_typing_prompt(
    *,
    chunk_text: str,
    entities: list[dict[str, object]],
    allowed_relationship_types: list[str],
    prompt_template: str = DEFAULT_RELATION_TYPING_TEMPLATE,
) -> str:
    """Build the GLiNER-entities-to-LLM-relation-typing prompt."""

    return prompt_template.format(
        chunk_text=chunk_text,
        entities=json.dumps(entities, indent=2, sort_keys=True),
        allowed_relationship_types=json.dumps(allowed_relationship_types),
    )


def validate_graph_payload(payload: Any) -> bool:
    """Return true for the graph JSON shape expected from LLM extraction."""

    if not isinstance(payload, dict):
        return False
    nodes = payload.get("nodes")
    relationships = payload.get("relationships")
    if not isinstance(nodes, list) or not isinstance(relationships, list):
        return False
    for node in nodes:
        if not isinstance(node, dict):
            return False
        if not {"type", "name", "evidence_text"} <= set(node):
            return False
    for relationship in relationships:
        if not isinstance(relationship, dict):
            return False
        if not {"source", "type", "target", "evidence_text"} <= set(relationship):
            return False
    return True


def validate_label_payload(payload: Any) -> bool:
    """Return true for the label proposal JSON shape."""

    if not isinstance(payload, dict):
        return False
    labels = payload.get("labels")
    if not isinstance(labels, list):
        return False
    for label in labels:
        if not isinstance(label, dict):
            return False
        if not isinstance(label.get("name"), str) or not label["name"].strip():
            return False
    return True


def validate_relation_typing_payload(payload: Any) -> bool:
    """Return true for the relation typing JSON shape."""

    if not isinstance(payload, dict):
        return False
    relations = payload.get("relations", payload.get("relationships"))
    if not isinstance(relations, list):
        return False
    for relation in relations:
        if not isinstance(relation, dict):
            return False
        if not {"source", "type", "target", "evidence_text"} <= set(relation):
            return False
    return True


__all__ = [
    "DEFAULT_LABEL_PROPOSAL_TEMPLATE",
    "DEFAULT_RELATION_TYPING_TEMPLATE",
    "GRAPH_EXTRACTION_PROMPT_VERSION",
    "LABEL_PROPOSAL_PROMPT_VERSION",
    "RELATION_TYPING_PROMPT_VERSION",
    "build_graph_extraction_prompt",
    "build_label_proposal_prompt",
    "build_relation_typing_prompt",
    "build_repair_prompt",
    "validate_graph_payload",
    "validate_label_payload",
    "validate_relation_typing_payload",
]
