"""Hard contracts for mandatory offline model-backed unstructured providers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from kg_construction.domain.claim import DocumentRef
from kg_construction.domain.models import SourceRecord
from kg_construction.infrastructure.extractors.model_backed import (
    ExactSpanOutput,
    GlinerMentionDetector,
    GlinerRelExClaimExtractor,
    HydratedArtifact,
    HydratedModelSet,
    HydratedServiceModel,
    MandatoryClaimEnsemble,
    MandatoryMentionEnsemble,
    MandatoryProviderError,
    MentionInference,
    MentionProviderConfig,
    ModelBackendOverrides,
    ModelClaimExtractor,
    ModelMentionDetector,
    OfflineArtifactError,
    OllamaQwenClaimExtractor,
    ProviderOutputError,
    RelationInference,
    SpacyMentionDetector,
    TransformersMentionDetector,
    TransformersQwenClaimExtractor,
    build_mandatory_model_providers,
    build_mandatory_model_providers_from_lock,
    detect_mandatory_profile,
)
from kg_construction.infrastructure.extractors.model_backed.claims import (
    ClaimProviderConfig,
    LocalMlxQwenBackend,
    LocalOllamaQwenBackend,
    LocalTransformersQwenBackend,
    _align_quoted_claim_response,
    _strict_claim_json_schema,
)
from kg_construction.infrastructure.extractors.model_backed.mentions import (
    LocalGlinerBackend,
)
from kg_construction.infrastructure.extractors.model_backed.relations import (
    LocalGlinerRelExBackend,
)
from kg_construction.unstructured.pipeline import (
    ConservativeCoreferenceResolver,
    RuleSegmenter,
)


class FakeMentionBackend:
    def __init__(
        self,
        outputs: tuple[MentionInference, ...] = (),
        *,
        error: Exception | None = None,
        calls: list[str] | None = None,
        name: str = "provider",
    ) -> None:
        self.outputs = outputs
        self.error = error
        self.calls = calls
        self.name = name

    def infer(self, text: str) -> tuple[MentionInference, ...]:
        if self.calls is not None:
            self.calls.append(self.name)
        if self.error is not None:
            raise self.error
        return self.outputs


class FakeRelationBackend:
    def __init__(
        self,
        outputs: tuple[RelationInference, ...] = (),
        *,
        error: Exception | None = None,
        calls: list[str] | None = None,
        name: str = "provider",
    ) -> None:
        self.outputs = outputs
        self.error = error
        self.calls = calls
        self.name = name

    def infer(self, text: str) -> tuple[RelationInference, ...]:
        if self.calls is not None:
            self.calls.append(self.name)
        if self.error is not None:
            raise self.error
        return self.outputs


class FakeStructuredBackend:
    def __init__(
        self,
        payload: object,
        *,
        error: Exception | None = None,
        calls: list[str] | None = None,
        name: str = "provider",
    ) -> None:
        self.payload = payload
        self.error = error
        self.calls = calls
        self.name = name
        self.schemas: list[object] = []
        self.prompts: list[str] = []

    def generate_json(self, *, prompt: str, json_schema) -> object:
        if self.calls is not None:
            self.calls.append(self.name)
        self.prompts.append(prompt)
        self.schemas.append(json_schema)
        if self.error is not None:
            raise self.error
        return self.payload


class FakeOllamaTransport:
    def __init__(self, models: dict[str, str], response: object) -> None:
        self.models = models
        self.response = response
        self.chat_calls = 0
        self.last_payload = None

    def list_models(self, base_url: str) -> dict[str, str]:
        del base_url
        return self.models

    def chat(self, base_url: str, payload) -> object:
        del base_url
        self.chat_calls += 1
        self.last_payload = payload
        return self.response


def _artifact(
    directory: Path,
    name: str,
    *,
    repo_id: str,
    backend: str,
    task: str,
    device: str = "cpu",
) -> HydratedArtifact:
    path = directory / name
    path.mkdir()
    return HydratedArtifact(
        name=name,
        repo_id=repo_id,
        revision="a" * 40,
        backend=backend,
        task=task,
        device=device,
        local_path=path.resolve(),
    )


def _document(text: str = "Ada founded Acme in 2020."):
    source = SourceRecord.from_text(text, source_name="test.txt")
    document = DocumentRef.from_source(source)
    segments = RuleSegmenter().segment(document, text)
    return document, segments


def _claim_payload(text: str = "Ada founded Acme in 2020.") -> dict:
    subject_start = text.index("Ada")
    predicate_start = text.index("founded")
    object_start = text.index("Acme")
    temporal_start = text.index("2020")
    return {
        "claims": [
            {
                "kind": "relation",
                "subject_span": {
                    "start": subject_start,
                    "end": subject_start + 3,
                    "text": "Ada",
                },
                "predicate": "founded",
                "predicate_span": {
                    "start": predicate_start,
                    "end": predicate_start + 7,
                    "text": "founded",
                },
                "object_span": {
                    "start": object_start,
                    "end": object_start + 4,
                    "text": "Acme",
                },
                "evidence_spans": [{"start": 0, "end": len(text), "text": text}],
                "temporal_span": {
                    "start": temporal_start,
                    "end": temporal_start + 4,
                    "text": "2020",
                },
                "confidence": 0.9,
            }
        ]
    }


def _model_set(directory: Path, profile: str) -> HydratedModelSet:
    device = "mps" if profile == "darwin-arm64" else "cpu"
    artifacts = [
        _artifact(
            directory,
            "gliner_entity_extractor",
            repo_id="gliner-community/gliner_small-v2.5",
            backend="gliner",
            task="token-classification",
            device=device,
        ),
        _artifact(
            directory,
            "gliner_relex_extractor",
            repo_id="knowledgator/gliner-relex-large-v1.0",
            backend="gliner",
            task="relation-extraction",
            device=device,
        ),
        _artifact(
            directory,
            "transformer_ner",
            repo_id="dslim/bert-base-NER",
            backend="transformers",
            task="token-classification",
            device=device,
        ),
        _artifact(
            directory,
            "spacy_pipeline",
            repo_id="spacy/en_core_web_sm",
            backend="spacy",
            task="sentence-pos-ner",
        ),
        _artifact(
            directory,
            "structured_transformer_extractor",
            repo_id="Qwen/Qwen3-1.7B",
            backend="transformers",
            task="text-generation",
            device=device,
        ),
        _artifact(
            directory,
            "embedding_resolver",
            repo_id="sentence-transformers/all-MiniLM-L6-v2",
            backend="sentence-transformers",
            task="feature-extraction",
            device=device,
        ),
        _artifact(
            directory,
            "cross_encoder_resolver",
            repo_id="cross-encoder/stsb-roberta-base",
            backend="sentence-transformers",
            task="text-ranking",
            device=device,
        ),
    ]
    if profile == "darwin-arm64":
        artifacts.append(
            _artifact(
                directory,
                "mlx_structured_extractor",
                repo_id="mlx-community/Qwen3-1.7B-4bit",
                backend="mlx",
                task="text-generation",
                device="mlx",
            )
        )
    return HydratedModelSet(
        profile=profile,
        artifacts=tuple(artifacts),
        service_models=(
            HydratedServiceModel(
                name="ollama_structured_extractor",
                service="ollama",
                model="qwen3:1.7b-q4_K_M",
                digest=f"sha256:{'b' * 64}",
            ),
        ),
    )


class ExactMentionProviderTests(unittest.TestCase):
    def test_gliner_loads_the_absolute_snapshot_with_no_hub_fallback(self) -> None:
        class FakeGlinerModel:
            def __init__(self) -> None:
                self.device: str | None = None

            def to(self, device: str):
                self.device = device
                return self

            @staticmethod
            def predict_entities(
                _text: str,
                _labels: list[str],
                *,
                threshold: float,
                flat_ner: bool,
            ) -> list[dict[str, object]]:
                del threshold, flat_ner
                return [
                    {
                        "start": 0,
                        "end": 3,
                        "text": "Ada",
                        "label": "Person",
                        "score": 0.99,
                    }
                ]

        with tempfile.TemporaryDirectory() as temp:
            artifact = _artifact(
                Path(temp),
                "gliner",
                repo_id="gliner-community/gliner_small-v2.5",
                backend="gliner",
                task="token-classification",
            )
            (artifact.local_path / "gliner_config.json").write_text(
                '{"encoder_config": {"model_type": "deberta-v2"}}',
                encoding="utf-8",
            )
            model = FakeGlinerModel()
            backend = LocalGlinerBackend(
                artifact,
                labels=("Person",),
                threshold=0.5,
            )
            with patch(
                "gliner.GLiNER.from_pretrained",
                return_value=model,
            ) as loader:
                output = backend.infer("Ada founded Acme.")

        loader.assert_called_once_with(
            str(artifact.local_path),
            local_files_only=True,
        )
        self.assertTrue(artifact.local_path.is_absolute())
        self.assertNotEqual(loader.call_args.args[0], artifact.repo_id)
        self.assertEqual(model.device, "cpu")
        self.assertEqual(output[0].text, "Ada")

    def test_gliner_null_encoder_config_is_injected_without_a_repo_id(self) -> None:
        class FakeGlinerModel:
            @staticmethod
            def to(_device: str) -> FakeGlinerModel:
                return FakeGlinerModel()

            @staticmethod
            def predict_entities(
                _text: str,
                _labels: list[str],
                **_kwargs: object,
            ) -> list[dict[str, object]]:
                return []

        with tempfile.TemporaryDirectory() as temp:
            artifact = _artifact(
                Path(temp),
                "gliner",
                repo_id="gliner-community/gliner_small-v2.5",
                backend="gliner",
                task="token-classification",
            )
            (artifact.local_path / "gliner_config.json").write_text(
                json.dumps(
                    {
                        "model_type": "gliner",
                        "model_name": "microsoft/deberta-v3-small",
                        "encoder_config": None,
                        "span_mode": "markerV0",
                        "vocab_size": 128003,
                    }
                ),
                encoding="utf-8",
            )
            backend = LocalGlinerBackend(
                artifact,
                labels=("Person",),
                threshold=0.5,
            )
            model = FakeGlinerModel()
            with (
                patch(
                    "gliner.model.UniEncoderSpanGLiNER.from_pretrained",
                    return_value=model,
                ) as loader,
                patch(
                    "gliner.GLiNER.from_pretrained",
                    side_effect=AssertionError(
                        "the outer GLiNER loader would re-resolve the encoder repo"
                    ),
                ),
            ):
                self.assertEqual(backend.infer("Ada founded Acme."), ())

        loader.assert_called_once_with(
            str(artifact.local_path),
            local_files_only=True,
        )
        self.assertTrue(Path(loader.call_args.args[0]).is_absolute())
        self.assertNotEqual(loader.call_args.args[0], artifact.repo_id)

    def test_three_real_detector_adapters_preserve_offsets_and_provenance(self) -> None:
        text = "Ada founded Acme."
        document, segments = _document(text)
        raw = (
            MentionInference(
                start=0,
                end=3,
                text="Ada",
                entity_type="Person",
                confidence=0.93,
            ),
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            gliner_artifact = _artifact(
                root,
                "gliner",
                repo_id="gliner-community/gliner_small-v2.5",
                backend="gliner",
                task="token-classification",
            )
            transformer_artifact = _artifact(
                root,
                "transformer",
                repo_id="dslim/bert-base-NER",
                backend="transformers",
                task="token-classification",
            )
            spacy_artifact = _artifact(
                root,
                "spacy",
                repo_id="spacy/en_core_web_sm",
                backend="spacy",
                task="sentence-pos-ner",
            )
            providers = (
                GlinerMentionDetector(
                    gliner_artifact,
                    labels=("Person",),
                    backend=FakeMentionBackend(raw),
                ),
                TransformersMentionDetector(
                    transformer_artifact,
                    backend=FakeMentionBackend(raw),
                ),
                SpacyMentionDetector(
                    spacy_artifact,
                    backend=FakeMentionBackend(raw),
                ),
            )
            outputs = [provider.detect(document, text, segments)[0] for provider in providers]

        self.assertEqual(
            [item.provenance.provider for item in outputs],
            ["gliner", "transformers", "spacy"],
        )
        for mention in outputs:
            self.assertEqual(mention.surface, text[mention.span.start : mention.span.end])
            self.assertTrue(mention.evidence.is_exact_for(document, text))
            self.assertTrue(mention.provenance.complete)
            self.assertIn("@", mention.provenance.model_version)

    def test_inexact_backend_span_is_a_hard_failure(self) -> None:
        text = "Ada founded Acme."
        document, segments = _document(text)
        backend = FakeMentionBackend(
            (
                MentionInference(
                    start=0,
                    end=3,
                    text="Eve",
                    entity_type="Person",
                    confidence=0.9,
                ),
            )
        )
        detector = ModelMentionDetector(
            MentionProviderConfig(
                extractor="fake",
                provider="fake",
                model_version="fake@abc",
                configuration_id="fake-v1",
            ),
            backend,
        )
        with self.assertRaisesRegex(ProviderOutputError, "inexact mention"):
            detector.detect(document, text, segments)


class GlinerRelExProviderTests(unittest.TestCase):
    def test_relation_claim_preserves_exact_arguments_evidence_and_provenance(self) -> None:
        text = "Ada founded Acme in 2020."
        document, segments = _document(text)
        mention_detector = ModelMentionDetector(
            MentionProviderConfig(
                extractor="mentions",
                provider="mentions",
                model_version="mentions@1",
                configuration_id="mentions-v1",
            ),
            FakeMentionBackend(
                (
                    MentionInference(
                        start=0,
                        end=3,
                        text="Ada",
                        entity_type="Person",
                        confidence=1.0,
                    ),
                    MentionInference(
                        start=12,
                        end=16,
                        text="Acme",
                        entity_type="Organization",
                        confidence=1.0,
                    ),
                )
            ),
        )
        mentions = mention_detector.detect(document, text, segments)
        relation = RelationInference(
            subject_span=ExactSpanOutput(start=0, end=3, text="Ada"),
            predicate="founded",
            object_span=ExactSpanOutput(start=12, end=16, text="Acme"),
            confidence=0.92,
        )
        with tempfile.TemporaryDirectory() as temp:
            artifact = _artifact(
                Path(temp),
                "gliner-relex",
                repo_id="knowledgator/gliner-relex-large-v1.0",
                backend="gliner",
                task="relation-extraction",
            )
            extractor = GlinerRelExClaimExtractor(
                artifact,
                entity_labels=("Person", "Organization"),
                relation_labels=("founded",),
                backend=FakeRelationBackend((relation,)),
            )
            claim = extractor.extract(document, text, segments, mentions)[0]

        self.assertEqual((claim.subject, claim.predicate, claim.object), ("Ada", "founded", "Acme"))
        self.assertEqual(claim.subject_mention_id, mentions[0].mention_id)
        self.assertEqual(claim.object_mention_id, mentions[1].mention_id)
        self.assertEqual(claim.evidence, (segments[0],))
        self.assertTrue(claim.has_complete_provenance(text))
        self.assertEqual(claim.provenance.provider, "gliner_relex")
        self.assertIn(
            "knowledgator/gliner-relex-large-v1.0@",
            claim.provenance.model_version,
        )

    def test_inexact_relation_argument_is_a_hard_failure(self) -> None:
        text = "Ada founded Acme."
        document, segments = _document(text)
        relation = RelationInference(
            subject_span=ExactSpanOutput(start=0, end=3, text="Eve"),
            predicate="founded",
            object_span=ExactSpanOutput(start=12, end=16, text="Acme"),
            confidence=0.9,
        )
        with tempfile.TemporaryDirectory() as temp:
            artifact = _artifact(
                Path(temp),
                "gliner-relex",
                repo_id="knowledgator/gliner-relex-large-v1.0",
                backend="gliner",
                task="relation-extraction",
            )
            extractor = GlinerRelExClaimExtractor(
                artifact,
                entity_labels=("Person", "Organization"),
                relation_labels=("founded",),
                backend=FakeRelationBackend((relation,)),
            )
            with self.assertRaisesRegex(ProviderOutputError, "inexact subject"):
                extractor.extract(document, text, segments, ())

    def test_local_backend_uses_only_hydrated_path_and_native_joint_api(self) -> None:
        text = "Ada worked at Acme."

        class FakeGlinerRelExModel:
            def __init__(self) -> None:
                self.device: str | None = None
                self.kwargs: dict = {}

            def to(self, device: str):
                self.device = device
                return self

            def inference(self, **kwargs):
                self.kwargs = kwargs
                return (
                    [[]],
                    [
                        [
                            {
                                "head": {
                                    "start": 0,
                                    "end": 3,
                                    "text": "Ada",
                                },
                                "tail": {
                                    "start": 14,
                                    "end": 18,
                                    "text": "Acme",
                                },
                                "relation": "worked at",
                                "score": 0.88,
                            }
                        ]
                    ],
                )

        with tempfile.TemporaryDirectory() as temp:
            artifact = _artifact(
                Path(temp),
                "gliner-relex",
                repo_id="knowledgator/gliner-relex-large-v1.0",
                backend="gliner",
                task="relation-extraction",
            )
            model = FakeGlinerRelExModel()
            backend = LocalGlinerRelExBackend(
                artifact,
                entity_labels=("Person", "Organization", "other"),
                relation_labels=("worked at",),
            )
            with patch("gliner.GLiNER.from_pretrained", return_value=model) as loader:
                output = backend.infer(text)

        loader.assert_called_once_with(
            str(artifact.local_path),
            local_files_only=True,
        )
        self.assertEqual(model.device, "cpu")
        self.assertEqual(model.kwargs["texts"], [text])
        self.assertTrue(model.kwargs["return_relations"])
        self.assertEqual(output[0].predicate, "worked_at")
        self.assertTrue(output[0].subject_span.is_exact_in(text))
        self.assertTrue(output[0].object_span.is_exact_in(text))


class StructuredClaimProviderTests(unittest.TestCase):
    def test_structured_json_becomes_an_exact_provenanced_claim(self) -> None:
        text = "Ada founded Acme in 2020."
        document, segments = _document(text)
        mention_backend = FakeMentionBackend(
            (
                MentionInference(
                    start=0,
                    end=3,
                    text="Ada",
                    entity_type="Person",
                    confidence=1,
                ),
                MentionInference(
                    start=12,
                    end=16,
                    text="Acme",
                    entity_type="Organization",
                    confidence=1,
                ),
            )
        )
        mention_detector = ModelMentionDetector(
            MentionProviderConfig(
                extractor="mentions",
                provider="mentions",
                model_version="mentions@1",
                configuration_id="mentions-v1",
            ),
            mention_backend,
        )
        mentions = mention_detector.detect(document, text, segments)
        backend = FakeStructuredBackend(_claim_payload(text))
        extractor = ModelClaimExtractor(
            ClaimProviderConfig(
                extractor="claims",
                provider="qwen3",
                model_version="Qwen/Qwen3-1.7B@abc",
                configuration_id="schema-v1",
            ),
            backend,
        )

        claim = extractor.extract(document, text, segments, mentions)[0]

        self.assertEqual(claim.subject, "Ada")
        self.assertEqual(claim.object, "Acme")
        self.assertEqual(claim.temporal, "2020")
        self.assertEqual(claim.subject_mention_id, mentions[0].mention_id)
        self.assertEqual(claim.object_mention_id, mentions[1].mention_id)
        self.assertTrue(claim.has_complete_provenance(text))
        self.assertIn("absolute", backend.prompts[0])
        schema = backend.schemas[0]
        self.assertIsInstance(schema, dict)
        assert isinstance(schema, dict)
        properties = schema["properties"]
        self.assertIsInstance(properties, dict)
        assert isinstance(properties, dict)
        self.assertIn("claims", properties)

    def test_ungrounded_semantic_or_evidence_span_rejects_the_whole_response(self) -> None:
        text = "Ada founded Acme in 2020."
        document, segments = _document(text)
        payload = _claim_payload(text)
        payload["claims"][0]["object_span"]["text"] = "Fake"
        extractor = ModelClaimExtractor(
            ClaimProviderConfig(
                extractor="claims",
                provider="qwen3",
                model_version="qwen3@abc",
                configuration_id="schema-v1",
            ),
            FakeStructuredBackend(payload),
        )
        with self.assertRaises(ProviderOutputError):
            extractor.extract(document, text, segments, ())

    def test_negation_and_modality_require_exact_cue_spans(self) -> None:
        text = "Ada may not acquire Acme."
        document, segments = _document(text)
        payload = {
            "claims": [
                {
                    "kind": "relation",
                    "subject_span": {"start": 0, "end": 3, "text": "Ada"},
                    "predicate": "acquire",
                    "predicate_span": {"start": 12, "end": 19, "text": "acquire"},
                    "object_span": {"start": 20, "end": 24, "text": "Acme"},
                    "evidence_spans": [{"start": 0, "end": len(text), "text": text}],
                    "negated": True,
                    "modality": "possible",
                    "confidence": 0.8,
                }
            ]
        }
        extractor = ModelClaimExtractor(
            ClaimProviderConfig(
                extractor="claims",
                provider="qwen3",
                model_version="qwen3@abc",
                configuration_id="schema-v1",
            ),
            FakeStructuredBackend(payload),
        )
        with self.assertRaisesRegex(ProviderOutputError, "negation_span"):
            extractor.extract(document, text, segments, ())

        payload["claims"][0]["negation_span"] = {
            "start": 8,
            "end": 11,
            "text": "not",
        }
        payload["claims"][0]["modality_span"] = {
            "start": 4,
            "end": 7,
            "text": "may",
        }
        claim = extractor.extract(document, text, segments, ())[0]
        self.assertTrue(claim.negated)
        self.assertEqual(claim.modality.value, "possible")

    def test_grounded_argument_gap_completes_semantic_predicate(self) -> None:
        text = "Ada worked at Acme."
        document, segments = _document(text)
        payload = {
            "claims": [
                {
                    "kind": "relation",
                    "subject_span": {"start": 0, "end": 3, "text": "Ada"},
                    "predicate": "worked",
                    "predicate_span": {
                        "start": 4,
                        "end": 10,
                        "text": "worked",
                    },
                    "object_span": {
                        "start": 14,
                        "end": 18,
                        "text": "Acme",
                    },
                    "evidence_spans": [
                        {"start": 0, "end": len(text), "text": text}
                    ],
                    "confidence": 0.9,
                }
            ]
        }
        extractor = ModelClaimExtractor(
            ClaimProviderConfig(
                extractor="claims",
                provider="qwen3",
                model_version="qwen3@abc",
                configuration_id="schema-v1",
            ),
            FakeStructuredBackend(payload),
        )

        claim = extractor.extract(document, text, segments, ())[0]

        self.assertEqual(claim.predicate, "worked_at")

    def test_exact_argument_bridge_repairs_overwide_predicate_cue(self) -> None:
        text = "Alan Turing worked at Bletchley Park."
        document, segments = _document(text)
        payload = {
            "claims": [
                {
                    "kind": "relation",
                    "subject_span": {
                        "start": 0,
                        "end": 11,
                        "text": "Alan Turing",
                    },
                    "predicate": "worked",
                    "predicate_span": {
                        "start": 19,
                        "end": 36,
                        "text": "at Bletchley Park",
                    },
                    "object_span": {
                        "start": 22,
                        "end": 36,
                        "text": "Bletchley Park",
                    },
                    "evidence_spans": [
                        {"start": 0, "end": len(text), "text": text}
                    ],
                    "confidence": 1.0,
                }
            ]
        }
        extractor = ModelClaimExtractor(
            ClaimProviderConfig(
                extractor="claims",
                provider="qwen3",
                model_version="qwen3@abc",
                configuration_id="schema-v1",
            ),
            FakeStructuredBackend(payload),
        )

        claim = extractor.extract(document, text, segments, ())[0]

        self.assertEqual(claim.predicate, "worked_at")

    def test_ungrounded_model_predicate_fails_closed(self) -> None:
        text = "Ada worked at Acme."
        document, segments = _document(text)
        payload = {
            "claims": [
                {
                    "kind": "relation",
                    "subject_span": {"start": 0, "end": 3, "text": "Ada"},
                    "predicate": "founded",
                    "predicate_span": {
                        "start": 4,
                        "end": 10,
                        "text": "worked",
                    },
                    "object_span": {
                        "start": 14,
                        "end": 18,
                        "text": "Acme",
                    },
                    "evidence_spans": [
                        {"start": 0, "end": len(text), "text": text}
                    ],
                    "confidence": 0.9,
                }
            ]
        }
        extractor = ModelClaimExtractor(
            ClaimProviderConfig(
                extractor="claims",
                provider="qwen3",
                model_version="qwen3@abc",
                configuration_id="schema-v1",
            ),
            FakeStructuredBackend(payload),
        )

        with self.assertRaisesRegex(ProviderOutputError, "not lexically grounded"):
            extractor.extract(document, text, segments, ())

    def test_strict_json_schema_uses_kind_discriminated_semantics(self) -> None:
        schema = _strict_claim_json_schema()
        items = schema["properties"]["claims"]["items"]
        variants = {
            variant["properties"]["kind"]["const"]: variant
            for variant in items["oneOf"]
        }

        self.assertEqual(set(variants), {"relation", "fact", "intent", "event"})
        for kind in ("relation", "fact", "intent"):
            self.assertIn("object_span", variants[kind]["required"])
            self.assertEqual(
                variants[kind]["properties"]["object_span"]["$ref"],
                "#/$defs/ExactSpanOutput",
            )
        self.assertNotIn("object_span", variants["event"]["required"])
        self.assertIn("event_roles", variants["event"]["required"])
        self.assertEqual(
            variants["event"]["properties"]["event_roles"]["minItems"],
            1,
        )
        for kind, variant in variants.items():
            self.assertEqual(
                variant["properties"]["kind"],
                {"const": kind, "type": "string"},
            )
            self.assertFalse(variant["additionalProperties"])

    def test_full_document_sentinel_field_shift_uses_only_exact_prompt_anchors(
        self,
    ) -> None:
        text = (
            "On 2026-07-09, Platform Team reviewed INCIDENT-1042.\n"
            "Contact [REDACTED] and see https://example.com/runbook.\n"
            "The service version is v2.4.1.\n"
            "INCIDENT-1042 affected Recommendation API.\n"
        )
        self.assertEqual(len(text), 183)
        input_payload = {
            "document": {"text": text},
            "segments": [
                {"start": 0, "end": 52, "text": text[0:52]},
            ],
            "mentions": [
                {"start": 3, "end": 13, "text": "2026-07-09", "entity_type": "Date"},
                {
                    "start": 15,
                    "end": 28,
                    "text": "Platform Team",
                    "entity_type": "Organization",
                },
                {
                    "start": 38,
                    "end": 51,
                    "text": "INCIDENT-1042",
                    "entity_type": "Incident",
                },
                {
                    "start": 140,
                    "end": 153,
                    "text": "INCIDENT-1042",
                    "entity_type": "Incident",
                },
                {
                    "start": 163,
                    "end": 181,
                    "text": "Recommendation API",
                    "entity_type": "Service",
                },
            ],
        }
        prompt = "Extract grounded claims.\nINPUT=" + json.dumps(
            input_payload,
            sort_keys=True,
        )
        shifted = {
            "claims": [
                {
                    "kind": "relation",
                    "subject_span": {
                        "start": 0,
                        "end": 12,
                        "text": text[0:52],
                    },
                    "predicate": "reviewed",
                    "predicate_span": {
                        "start": 13,
                        "end": 28,
                        "text": "reviewed",
                    },
                    "object_span": {
                        "start": 29,
                        "end": 36,
                        "text": "INCIDENT-1042",
                    },
                    "evidence_spans": [
                        {"start": 0, "end": len(text), "text": "document"}
                    ],
                }
            ]
        }

        aligned, alignment_count = _align_quoted_claim_response(
            shifted,
            prompt=prompt,
        )
        claim = aligned.claims[0]

        self.assertEqual(
            (
                claim.subject_span.start,
                claim.subject_span.end,
                claim.subject_span.text,
            ),
            (15, 28, "Platform Team"),
        )
        self.assertEqual(
            (
                claim.predicate_span.start,
                claim.predicate_span.end,
                claim.predicate_span.text,
            ),
            (29, 37, "reviewed"),
        )
        self.assertEqual(
            (
                claim.object_span.start if claim.object_span else None,
                claim.object_span.end if claim.object_span else None,
                claim.object_span.text if claim.object_span else None,
            ),
            (38, 51, "INCIDENT-1042"),
        )
        self.assertEqual(
            (
                claim.evidence_spans[0].start,
                claim.evidence_spans[0].end,
                claim.evidence_spans[0].text,
            ),
            (0, 52, text[0:52]),
        )
        self.assertEqual(claim.confidence, 0.75)
        self.assertEqual(alignment_count, 4)

        shifted_claim = shifted["claims"][0]
        shifted_claim["predicate_span"] = {
            "start": 13,
            "end": 25,
            "text": "INCIDENT-1042",
        }
        shifted_claim["object_span"] = {
            "start": 51,
            "end": 60,
            "text": "INCIDENT-1042",
        }
        shifted_claim["evidence_spans"][0]["end"] = len(text.rstrip("\r\n"))
        second_variant, second_alignment_count = _align_quoted_claim_response(
            shifted,
            prompt=prompt,
        )
        self.assertEqual(second_variant.claims[0].subject_span.text, "Platform Team")
        self.assertEqual(second_variant.claims[0].predicate_span.text, "reviewed")
        self.assertEqual(second_variant.claims[0].object_span.text, "INCIDENT-1042")
        self.assertEqual(second_alignment_count, 4)

        shifted_claim["evidence_spans"][0]["end"] = len(text) - 2
        with self.assertRaisesRegex(
            ProviderOutputError,
            "sentinel must select the immutable document",
        ):
            _align_quoted_claim_response(shifted, prompt=prompt)

    def test_overlapping_claim_arguments_are_a_hard_failure(self) -> None:
        text = "Ada reviewed Acme."
        document, segments = _document(text)
        payload = {
            "claims": [
                {
                    "kind": "relation",
                    "subject_span": {
                        "start": 0,
                        "end": len(text),
                        "text": text,
                    },
                    "predicate": "reviewed",
                    "predicate_span": {
                        "start": 4,
                        "end": 12,
                        "text": "reviewed",
                    },
                    "object_span": {
                        "start": 13,
                        "end": 17,
                        "text": "Acme",
                    },
                    "evidence_spans": [
                        {"start": 0, "end": len(text), "text": text}
                    ],
                    "confidence": 0.9,
                }
            ]
        }
        extractor = ModelClaimExtractor(
            ClaimProviderConfig(
                extractor="claims",
                provider="qwen3",
                model_version="qwen3@abc",
                configuration_id="schema-v1",
            ),
            FakeStructuredBackend(payload),
        )

        with self.assertRaisesRegex(ProviderOutputError, "argument spans overlap"):
            extractor.extract(document, text, segments, ())

    def test_all_qwen_backends_align_only_uniquely_grounded_quotes(self) -> None:
        text = "Ada worked at Acme."
        prompt = (
            "Extract grounded claims.\nINPUT="
            + json.dumps({"document": {"text": text}}, sort_keys=True)
        )
        miscounted = json.dumps(
            {
                "claims": [
                    {
                        "kind": "relation",
                        "subject_span": {"start": 1, "end": 4, "text": "Ada"},
                        "predicate": "worked",
                        "predicate_span": {
                            "start": 0,
                            "end": 6,
                            "text": "worked",
                        },
                        "object_span": {
                            "start": 0,
                            "end": 4,
                            "text": "Acme",
                        },
                        "evidence_spans": [
                            {"start": 1, "end": 20, "text": text}
                        ],
                    }
                ]
            }
        )

        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                del messages, kwargs
                return "rendered"

        def generate(model, tokenizer, **kwargs):
            del model, tokenizer, kwargs
            return miscounted

        with tempfile.TemporaryDirectory() as temp:
            transformers_artifact = _artifact(
                Path(temp),
                "transformers",
                repo_id="Qwen/Qwen3-1.7B",
                backend="transformers",
                task="text-generation",
            )
            transformers = LocalTransformersQwenBackend(
                transformers_artifact,
                max_retries=0,
            )
            transformers._complete = lambda _messages: miscounted  # type: ignore[method-assign]
            transformers_output = transformers.generate_json(
                prompt=prompt,
                json_schema={"type": "object"},
            )
            artifact = _artifact(
                Path(temp),
                "mlx",
                repo_id="mlx-community/Qwen3-1.7B-4bit",
                backend="mlx",
                task="text-generation",
            )
            mlx = LocalMlxQwenBackend(artifact, max_retries=0)
            mlx._load = lambda: (object(), Tokenizer(), generate)  # type: ignore[method-assign]
            mlx_output = mlx.generate_json(
                prompt=prompt,
                json_schema={"type": "object"},
            )

        model = HydratedServiceModel(
            name="ollama_structured_extractor",
            service="ollama",
            model="qwen3:test",
            digest=f"sha256:{'a' * 64}",
        )
        transport = FakeOllamaTransport(
            {"qwen3:test": f"sha256:{'a' * 64}"},
            {"message": {"content": miscounted}},
        )
        ollama = LocalOllamaQwenBackend(
            model,
            transport=transport,
            max_retries=0,
        )
        ollama_output = ollama.generate_json(
            prompt=prompt,
            json_schema={"type": "object"},
        )

        for output in (transformers_output, mlx_output, ollama_output):
            claim = output.claims[0]
            self.assertEqual(
                (
                    claim.subject_span.start,
                    claim.predicate_span.start,
                    claim.object_span.start if claim.object_span else None,
                    claim.evidence_spans[0].start,
                ),
                (0, 4, 14, 0),
            )
            self.assertEqual(claim.confidence, 0.75)
        self.assertEqual(transformers.last_quote_alignment_count, 4)
        self.assertEqual(mlx.last_quote_alignment_count, 4)
        self.assertEqual(ollama.last_quote_alignment_count, 4)

    def test_repeated_quote_without_exact_occurrence_anchor_is_hard_failure(
        self,
    ) -> None:
        text = "Ada met Ada."
        prompt = (
            "Extract grounded claims.\nINPUT="
            + json.dumps({"document": {"text": text}}, sort_keys=True)
        )
        ambiguous = {
            "claims": [
                {
                    "kind": "relation",
                    "subject_span": {"start": 1, "end": 4, "text": "Ada"},
                    "predicate": "met",
                    "predicate_span": {"start": 4, "end": 7, "text": "met"},
                    "object_span": {"start": 8, "end": 11, "text": "Ada"},
                    "evidence_spans": [
                        {"start": 0, "end": len(text), "text": text}
                    ],
                    "confidence": 0.9,
                }
            ]
        }

        with self.assertRaisesRegex(
            ProviderOutputError,
            "ambiguous without an exact occurrence or unique evidence anchor",
        ):
            _align_quoted_claim_response(ambiguous, prompt=prompt)

    def test_repeated_quote_uses_one_unique_claim_evidence_anchor(self) -> None:
        text = "Ada met Bob. Bob met Ada."
        prompt = (
            "Extract grounded claims.\nINPUT="
            + json.dumps({"document": {"text": text}}, sort_keys=True)
        )
        payload = {
            "claims": [
                {
                    "kind": "relation",
                    "subject_span": {"start": 2, "end": 5, "text": "Ada"},
                    "predicate": "met",
                    "predicate_span": {"start": 0, "end": 3, "text": "met"},
                    "object_span": {"start": 0, "end": 3, "text": "Bob"},
                    "evidence_spans": [
                        {
                            "start": 1,
                            "end": 13,
                            "text": "Ada met Bob.",
                        }
                    ],
                    "confidence": 0.9,
                }
            ]
        }

        aligned, _ = _align_quoted_claim_response(payload, prompt=prompt)
        claim = aligned.claims[0]

        self.assertEqual(claim.subject_span.start, 0)
        self.assertEqual(claim.predicate_span.start, 4)
        self.assertEqual(claim.object_span.start if claim.object_span else None, 8)
        self.assertEqual(claim.evidence_spans[0].start, 0)

    def test_globally_valid_repeated_offset_cannot_escape_claim_evidence(
        self,
    ) -> None:
        text = "Alan met Charles. Charles met Ada."
        prompt = (
            "Extract grounded claims.\nINPUT="
            + json.dumps({"document": {"text": text}}, sort_keys=True)
        )
        payload = {
            "claims": [
                {
                    "kind": "relation",
                    # Offset 9 is a real Charles occurrence, but it is outside
                    # the model's exact evidence for this claim.
                    "subject_span": {
                        "start": 9,
                        "end": 16,
                        "text": "Charles",
                    },
                    "predicate": "met",
                    "predicate_span": {
                        "start": 26,
                        "end": 29,
                        "text": "met",
                    },
                    "object_span": {
                        "start": 30,
                        "end": 33,
                        "text": "Ada",
                    },
                    "evidence_spans": [
                        {
                            "start": 18,
                            "end": 34,
                            "text": "Charles met Ada.",
                        }
                    ],
                    "confidence": 0.9,
                }
            ]
        }

        aligned, _ = _align_quoted_claim_response(payload, prompt=prompt)

        self.assertEqual(aligned.claims[0].subject_span.start, 18)
        self.assertEqual(aligned.claims[0].evidence_spans[0].start, 18)

    def test_field_shift_repairs_only_exact_displaced_mention_and_negation(
        self,
    ) -> None:
        text = "Ada Lovelace never worked with Charles Babbage."
        prompt = (
            "Extract grounded claims.\nINPUT="
            + json.dumps(
                {
                    "document": {"text": text},
                    "mentions": [
                        {
                            "start": 0,
                            "end": 12,
                            "text": "Ada Lovelace",
                        },
                        {
                            "start": 31,
                            "end": 46,
                            "text": "Charles Babbage",
                        },
                        {"start": 31, "end": 32, "text": "C"},
                        {
                            "start": 32,
                            "end": 46,
                            "text": "harles Babbage",
                        },
                    ],
                },
                sort_keys=True,
            )
        )
        payload = {
            "claims": [
                {
                    "kind": "relation",
                    "subject_span": {
                        "start": 1,
                        "end": 4,
                        "text": "Ada Lovelace",
                    },
                    "predicate": "never worked with",
                    # The exact object quote was shifted into predicate_span.
                    "predicate_span": {
                        "start": 5,
                        "end": 8,
                        "text": "Charles Babbage",
                    },
                    # The subject was shifted into object_span.
                    "object_span": {
                        "start": 9,
                        "end": 12,
                        "text": "Ada Lovelace",
                    },
                    "evidence_spans": [
                        {"start": 1, "end": 10, "text": text}
                    ],
                    "confidence": 0.9,
                }
            ]
        }

        aligned, _ = _align_quoted_claim_response(payload, prompt=prompt)
        claim = aligned.claims[0]

        self.assertEqual(claim.predicate, "worked_with")
        self.assertEqual(claim.predicate_span.text, "worked with")
        self.assertEqual(claim.object_span.text if claim.object_span else None, "Charles Babbage")
        self.assertTrue(claim.negated)
        self.assertEqual(claim.negation_span.text if claim.negation_span else None, "never")

    def test_displaced_object_with_multiple_exact_mentions_is_rejected(
        self,
    ) -> None:
        text = "Ada worked with Bob and Carol."
        prompt = (
            "Extract grounded claims.\nINPUT="
            + json.dumps(
                {
                    "document": {"text": text},
                    "mentions": [
                        {"start": 0, "end": 3, "text": "Ada"},
                        {"start": 16, "end": 19, "text": "Bob"},
                        {"start": 24, "end": 29, "text": "Carol"},
                    ],
                },
                sort_keys=True,
            )
        )
        payload = {
            "claims": [
                {
                    "kind": "relation",
                    "subject_span": {"start": 0, "end": 3, "text": "Ada"},
                    "predicate": "worked_with",
                    "predicate_span": {
                        "start": 16,
                        "end": 29,
                        "text": "Bob and Carol",
                    },
                    "object_span": {
                        "start": 4,
                        "end": 10,
                        "text": "worked",
                    },
                    "evidence_spans": [
                        {"start": 0, "end": len(text), "text": text}
                    ],
                    "confidence": 0.9,
                }
            ]
        }

        with self.assertRaisesRegex(
            ProviderOutputError,
            "displaced ambiguously",
        ):
            _align_quoted_claim_response(payload, prompt=prompt)

    def test_normalized_predicate_quote_recovers_from_exact_argument_window(
        self,
    ) -> None:
        text = (
            "Alan Turing worked with Charles Babbage. "
            "Charles Babbage worked with Ada Lovelace."
        )
        prompt = (
            "Extract grounded claims.\nINPUT="
            + json.dumps({"document": {"text": text}}, sort_keys=True)
        )
        payload = {
            "claims": [
                {
                    "kind": "relation",
                    "subject_span": {
                        "start": 2,
                        "end": 13,
                        "text": "Alan Turing",
                    },
                    "predicate": "worked_with",
                    "predicate_span": {
                        "start": 14,
                        "end": 25,
                        "text": "worked_with",
                    },
                    "object_span": {
                        "start": 28,
                        "end": 43,
                        "text": "Charles Babbage",
                    },
                    "evidence_spans": [
                        {
                            "start": 1,
                            "end": 82,
                            "text": "Alan Turing worked with Charles Babbage.",
                        }
                    ],
                }
            ]
        }

        aligned, _ = _align_quoted_claim_response(payload, prompt=prompt)
        claim = aligned.claims[0]

        self.assertEqual(claim.predicate_span.text, "worked with")
        self.assertEqual(claim.predicate_span.start, 12)
        self.assertEqual(claim.object_span.start if claim.object_span else None, 24)
        self.assertEqual(claim.confidence, 0.75)

    def test_reviewed_predicate_aliases_ground_exact_source_cues(self) -> None:
        cases = (
            (
                "Alan worked at Acme.",
                {
                    "kind": "relation",
                    "subject_span": {"start": 0, "end": 4, "text": "Alan"},
                    "predicate": "works_at",
                    "predicate_span": {
                        "start": 5,
                        "end": 14,
                        "text": "worked at",
                    },
                    "object_span": {
                        "start": 15,
                        "end": 19,
                        "text": "Acme",
                    },
                    "evidence_spans": [
                        {
                            "start": 0,
                            "end": 20,
                            "text": "Alan worked at Acme.",
                        }
                    ],
                    "confidence": 0.9,
                },
                "works_at",
            ),
            (
                "Bombe was a Machine.",
                {
                    "kind": "fact",
                    "subject_span": {"start": 0, "end": 5, "text": "Bombe"},
                    "predicate": "is_a",
                    "predicate_span": {
                        "start": 6,
                        "end": 9,
                        "text": "was",
                    },
                    "object_span": {
                        "start": 12,
                        "end": 19,
                        "text": "Machine",
                    },
                    "evidence_spans": [
                        {
                            "start": 0,
                            "end": 20,
                            "text": "Bombe was a Machine.",
                        }
                    ],
                    "confidence": 0.9,
                },
                "is_a",
            ),
        )
        for text, output, expected in cases:
            with self.subTest(predicate=expected):
                document, segments = _document(text)
                extractor = ModelClaimExtractor(
                    ClaimProviderConfig(
                        extractor="qwen",
                        provider="qwen3",
                        model_version="qwen3@abc",
                        configuration_id="schema-v1",
                    ),
                    FakeStructuredBackend({"claims": [output]}),
                )

                claims = extractor.extract(document, text, segments, ())

                self.assertEqual(claims[0].predicate, expected)

    def test_transformers_and_ollama_extractors_keep_locked_model_identity(self) -> None:
        text = "Ada founded Acme in 2020."
        document, segments = _document(text)
        with tempfile.TemporaryDirectory() as temp:
            artifact = _artifact(
                Path(temp),
                "qwen",
                repo_id="Qwen/Qwen3-1.7B",
                backend="transformers",
                task="text-generation",
            )
            transformer = TransformersQwenClaimExtractor(
                artifact,
                backend=FakeStructuredBackend(_claim_payload(text)),
            )
            ollama_model = HydratedServiceModel(
                name="ollama_structured_extractor",
                service="ollama",
                model="qwen3:1.7b-q4_K_M",
                digest=f"sha256:{'d' * 64}",
            )
            ollama = OllamaQwenClaimExtractor(
                ollama_model,
                backend=FakeStructuredBackend(_claim_payload(text)),
            )
            claims = [
                transformer.extract(document, text, segments, ())[0],
                ollama.extract(document, text, segments, ())[0],
            ]
        self.assertEqual(
            [claim.provenance.provider for claim in claims],
            ["transformers_qwen3", "ollama_qwen3"],
        )
        self.assertIn("Qwen/Qwen3-1.7B@", claims[0].provenance.model_version)
        self.assertIn("qwen3:1.7b-q4_K_M@sha256:", claims[1].provenance.model_version)

    def test_ollama_refuses_absent_or_wrong_digest_without_chatting(self) -> None:
        model = HydratedServiceModel(
            name="ollama_structured_extractor",
            service="ollama",
            model="qwen3:1.7b-q4_K_M",
            digest=f"sha256:{'e' * 64}",
        )
        transport = FakeOllamaTransport(
            {"qwen3:1.7b-q4_K_M": f"sha256:{'f' * 64}"},
            {"message": {"content": '{"claims": []}'}},
        )
        backend = LocalOllamaQwenBackend(model, transport=transport)
        with self.assertRaisesRegex(ProviderOutputError, "digest mismatch"):
            backend.generate_json(prompt="extract", json_schema={"type": "object"})
        self.assertEqual(transport.chat_calls, 0)

    def test_mlx_retries_with_validator_feedback_and_never_accepts_bad_json(
        self,
    ) -> None:
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                del kwargs
                return messages[-1]["content"]

        outputs = iter(["not-json", '{"claims": []}'])
        prompts: list[str] = []

        def generate(model, tokenizer, *, prompt, **kwargs):
            del model, tokenizer, kwargs
            prompts.append(prompt)
            return next(outputs)

        with tempfile.TemporaryDirectory() as temp:
            artifact = _artifact(
                Path(temp),
                "mlx",
                repo_id="mlx-community/Qwen3-1.7B-4bit",
                backend="mlx",
                task="text-generation",
            )
            backend = LocalMlxQwenBackend(artifact, max_retries=2)
            backend._load = lambda: (object(), Tokenizer(), generate)  # type: ignore[method-assign]
            result = backend.generate_json(
                prompt="extract",
                json_schema={"type": "object"},
            )

        self.assertEqual(result.claims, ())
        self.assertEqual(backend.last_attempt_count, 2)
        self.assertEqual(len(prompts), 2)
        self.assertIn("REPAIR REQUIRED", prompts[1])
        self.assertNotIn("not-json", prompts[1])
        self.assertTrue(prompts[1].endswith("extract"))


class MandatoryEnsembleTests(unittest.TestCase):
    def _mention_provider(
        self,
        provider: str,
        backend: FakeMentionBackend,
    ) -> ModelMentionDetector:
        return ModelMentionDetector(
            MentionProviderConfig(
                extractor=f"{provider}_extractor",
                provider=provider,
                model_version=f"{provider}@abc",
                configuration_id=f"{provider}-v1",
            ),
            backend,
        )

    def _claim_provider(
        self,
        provider: str,
        backend: FakeStructuredBackend,
    ) -> ModelClaimExtractor:
        return ModelClaimExtractor(
            ClaimProviderConfig(
                extractor=f"{provider}_extractor",
                provider=provider,
                model_version=f"{provider}@abc",
                configuration_id=f"{provider}-v1",
            ),
            backend,
        )

    def test_mention_ensemble_unions_exact_items_and_records_agreement(self) -> None:
        text = "Ada founded Acme."
        document, segments = _document(text)
        ada = MentionInference(
            start=0,
            end=3,
            text="Ada",
            entity_type="Person",
            confidence=0.9,
        )
        acme = MentionInference(
            start=12,
            end=16,
            text="Acme",
            entity_type="Organization",
            confidence=0.8,
        )
        ensemble = MandatoryMentionEnsemble(
            (
                self._mention_provider("one", FakeMentionBackend((ada, acme))),
                self._mention_provider("two", FakeMentionBackend((ada,))),
                self._mention_provider("three", FakeMentionBackend((ada,))),
            )
        )

        mentions = ensemble.detect(document, text, segments)

        self.assertEqual([item.surface for item in mentions], ["Ada", "Acme"])
        by_surface = {
            mention.surface: agreement
            for mention, agreement in zip(
                mentions,
                ensemble.last_agreements,
                strict=True,
            )
        }
        self.assertEqual(by_surface["Ada"].ratio, 1.0)
        self.assertEqual(by_surface["Acme"].ratio, 1 / 3)
        self.assertEqual(
            next(item for item in mentions if item.surface == "Acme").provenance.confidence,
            0.8,
        )
        self.assertEqual(mentions[0].provenance.provider, "one+three+two")
        self.assertIn("agreement=3/3", mentions[0].provenance.configuration_id)

    def test_mention_ensemble_resolves_competing_types_before_coreference(self) -> None:
        text = "Ada founded Acme."
        document, segments = _document(text)
        generic = MentionInference(
            start=0,
            end=3,
            text="Ada",
            entity_type="Entity",
            confidence=0.99,
        )
        person = MentionInference(
            start=0,
            end=3,
            text="Ada",
            entity_type="Person",
            confidence=0.7,
        )
        ensemble = MandatoryMentionEnsemble(
            (
                self._mention_provider(
                    "generic-provider",
                    FakeMentionBackend((generic,)),
                ),
                self._mention_provider(
                    "person-provider",
                    FakeMentionBackend((person,)),
                ),
            )
        )

        mentions = ensemble.detect(document, text, segments)
        clustered = ConservativeCoreferenceResolver().cluster(mentions)

        self.assertEqual(len(mentions), 1)
        self.assertEqual(mentions[0].entity_type, "Person")
        self.assertEqual(mentions[0].provenance.confidence, 0.7)
        self.assertIn(
            "type_policy=provider_vote>specific_over_generic>",
            mentions[0].provenance.configuration_id,
        )
        self.assertEqual(len(clustered), 1)
        self.assertEqual(len({mention.mention_id for mention in clustered}), 1)
        agreement = ensemble.last_agreements[0]
        self.assertEqual(agreement.selected_type, "Person")
        self.assertEqual(
            agreement.participating_providers,
            ("generic-provider", "person-provider"),
        )
        self.assertEqual(
            agreement.agreeing_providers,
            ("person-provider",),
        )
        self.assertEqual(
            agreement.provider_types,
            {
                "generic-provider": "Entity",
                "person-provider": "Person",
            },
        )
        self.assertEqual(
            agreement.provider_confidences,
            {
                "generic-provider": 0.99,
                "person-provider": 0.7,
            },
        )
        self.assertEqual(agreement.ratio, 0.5)

    def test_mention_ensemble_suppresses_only_unique_maximal_fragments(self) -> None:
        text = "Alan Turing met Ada at Bletchley Park."
        document, segments = _document(text)

        def inference(
            surface: str,
            entity_type: str,
            confidence: float,
        ) -> MentionInference:
            start = text.index(surface)
            return MentionInference(
                start=start,
                end=start + len(surface),
                text=surface,
                entity_type=entity_type,
                confidence=confidence,
            )

        calls: list[str] = []
        ensemble = MandatoryMentionEnsemble(
            (
                self._mention_provider(
                    "one",
                    FakeMentionBackend(
                        (
                            inference("Alan Turing", "Person", 0.9),
                            inference("Ada", "Person", 0.9),
                            inference("Bletchley Park", "Place", 0.9),
                        ),
                        calls=calls,
                        name="one",
                    ),
                ),
                self._mention_provider(
                    "two",
                    FakeMentionBackend(
                        (
                            inference("Turing", "Person", 0.8),
                            inference("Ada", "Person", 0.8),
                            inference("Park", "Place", 0.8),
                        ),
                        calls=calls,
                        name="two",
                    ),
                ),
                self._mention_provider(
                    "three",
                    FakeMentionBackend(
                        (
                            inference("Alan Turing", "Person", 0.7),
                            inference("Bletchley Park", "Place", 0.7),
                        ),
                        calls=calls,
                        name="three",
                    ),
                ),
            )
        )

        mentions = ensemble.detect(document, text, segments)

        self.assertEqual(calls, ["one", "two", "three"])
        self.assertEqual(
            [mention.surface for mention in mentions],
            ["Alan Turing", "Ada", "Bletchley Park"],
        )
        self.assertEqual(len(ensemble.last_agreements), 5)
        agreement_keys = {
            agreement.item_key for agreement in ensemble.last_agreements
        }
        suppressions = {
            (item.fragment_surface, item.containing_surface)
            for item in ensemble.last_suppressions
        }
        self.assertEqual(
            suppressions,
            {
                ("Turing", "Alan Turing"),
                ("Park", "Bletchley Park"),
            },
        )
        for suppression in ensemble.last_suppressions:
            self.assertIn(suppression.fragment_item_key, agreement_keys)
            self.assertIn(suppression.containing_item_key, agreement_keys)

    def test_mention_ensemble_keeps_ambiguous_overlapping_alternatives(self) -> None:
        text = "New York City"
        document, segments = _document(text)
        new_york = MentionInference(
            start=0,
            end=8,
            text="New York",
            entity_type="Place",
            confidence=0.9,
        )
        york = MentionInference(
            start=4,
            end=8,
            text="York",
            entity_type="Place",
            confidence=0.8,
        )
        york_city = MentionInference(
            start=4,
            end=13,
            text="York City",
            entity_type="Place",
            confidence=0.9,
        )
        ensemble = MandatoryMentionEnsemble(
            (
                self._mention_provider(
                    "one",
                    FakeMentionBackend((new_york, york)),
                ),
                self._mention_provider(
                    "two",
                    FakeMentionBackend((york_city,)),
                ),
            )
        )

        mentions = ensemble.detect(document, text, segments)

        self.assertEqual(
            [mention.surface for mention in mentions],
            ["New York", "York", "York City"],
        )
        self.assertEqual(ensemble.last_suppressions, ())

    def test_mention_ensemble_uses_source_suffix_for_exact_span_type_ties(
        self,
    ) -> None:
        cases = (
            ("Bletchley Park", "Location", "Place"),
            ("Acme Corp", "Person", "Organization"),
            ("Recommendation API", "System", "Artifact"),
            ("Analytical Engine", "Organization", "Artifact"),
            ("Bombe", "Organization", "Artifact"),
            ("Machine", "Organization", "Artifact"),
        )
        for surface, high_confidence_type, expected_type in cases:
            with self.subTest(surface=surface):
                document, segments = _document(surface)
                high_confidence = MentionInference(
                    start=0,
                    end=len(surface),
                    text=surface,
                    entity_type=high_confidence_type,
                    confidence=0.99,
                )
                suffix_supported = MentionInference(
                    start=0,
                    end=len(surface),
                    text=surface,
                    entity_type=expected_type,
                    confidence=0.1,
                )
                ensemble = MandatoryMentionEnsemble(
                    (
                        self._mention_provider(
                            "high-confidence",
                            FakeMentionBackend((high_confidence,)),
                        ),
                        self._mention_provider(
                            "suffix-supported",
                            FakeMentionBackend((suffix_supported,)),
                        ),
                    )
                )

                mentions = ensemble.detect(document, surface, segments)

                self.assertEqual(len(mentions), 1)
                self.assertEqual(mentions[0].entity_type, expected_type)
                self.assertEqual(mentions[0].provenance.confidence, 0.1)
                self.assertIn(
                    "source_suffix>reject_unresolved_tie",
                    mentions[0].provenance.configuration_id,
                )
                agreement = ensemble.last_agreements[0]
                self.assertEqual(agreement.selected_type, expected_type)
                self.assertEqual(
                    agreement.agreeing_providers,
                    ("suffix-supported",),
                )
                self.assertEqual(
                    agreement.provider_confidences,
                    {
                        "high-confidence": 0.99,
                        "suffix-supported": 0.1,
                    },
                )

    def test_mention_ensemble_uses_observed_identifier_lexical_form_for_tie(
        self,
    ) -> None:
        text = "INCIDENT-1042"
        document, segments = _document(text)
        votes = (
            ("identifier-provider", "Identifier", 0.1),
            ("incident-provider", "Incident", 0.99),
            ("organization-provider", "Organization", 0.8),
        )
        ensemble = MandatoryMentionEnsemble(
            tuple(
                self._mention_provider(
                    provider,
                    FakeMentionBackend(
                        (
                            MentionInference(
                                start=0,
                                end=len(text),
                                text=text,
                                entity_type=entity_type,
                                confidence=confidence,
                            ),
                        )
                    ),
                )
                for provider, entity_type, confidence in votes
            )
        )

        mentions = ensemble.detect(document, text, segments)

        self.assertEqual(len(mentions), 1)
        self.assertEqual(mentions[0].entity_type, "Identifier")
        self.assertEqual(mentions[0].provenance.confidence, 0.1)
        self.assertIn(
            "source_lexical_form>source_suffix",
            mentions[0].provenance.configuration_id,
        )
        agreement = ensemble.last_agreements[0]
        self.assertEqual(
            agreement.agreeing_providers,
            ("identifier-provider",),
        )
        self.assertEqual(
            agreement.participating_providers,
            (
                "identifier-provider",
                "incident-provider",
                "organization-provider",
            ),
        )
        self.assertEqual(
            agreement.provider_types,
            {
                "identifier-provider": "Identifier",
                "incident-provider": "Incident",
                "organization-provider": "Organization",
            },
        )

    def test_mention_ensemble_uses_other_observed_exact_lexical_forms(
        self,
    ) -> None:
        cases = (
            ("2026-07-09", "Date", "Location"),
            ("ops@example.com", "EmailAddress", "Organization"),
            ("v2.4.1", "Version", "Location"),
        )
        for surface, expected_type, conflicting_type in cases:
            with self.subTest(surface=surface):
                document, segments = _document(surface)
                ensemble = MandatoryMentionEnsemble(
                    (
                        self._mention_provider(
                            "lexical-form-provider",
                            FakeMentionBackend(
                                (
                                    MentionInference(
                                        start=0,
                                        end=len(surface),
                                        text=surface,
                                        entity_type=expected_type,
                                        confidence=0.1,
                                    ),
                                )
                            ),
                        ),
                        self._mention_provider(
                            "conflicting-provider",
                            FakeMentionBackend(
                                (
                                    MentionInference(
                                        start=0,
                                        end=len(surface),
                                        text=surface,
                                        entity_type=conflicting_type,
                                        confidence=0.99,
                                    ),
                                )
                            ),
                        ),
                    )
                )

                mentions = ensemble.detect(document, surface, segments)

                self.assertEqual(len(mentions), 1)
                self.assertEqual(mentions[0].entity_type, expected_type)
                self.assertEqual(
                    ensemble.last_agreements[0].agreeing_providers,
                    ("lexical-form-provider",),
                )

    def test_identifier_form_cannot_invent_an_unobserved_tied_type(self) -> None:
        text = "INCIDENT-1042"
        document, segments = _document(text)
        ensemble = MandatoryMentionEnsemble(
            tuple(
                self._mention_provider(
                    provider,
                    FakeMentionBackend(
                        (
                            MentionInference(
                                start=0,
                                end=len(text),
                                text=text,
                                entity_type=entity_type,
                                confidence=confidence,
                            ),
                        )
                    ),
                )
                for provider, entity_type, confidence in (
                    ("incident-provider", "Incident", 0.99),
                    ("organization-provider", "Organization", 0.1),
                )
            )
        )

        with self.assertRaisesRegex(
            ProviderOutputError,
            "unresolved exact-span type vote tie",
        ):
            ensemble.detect(document, text, segments)

    def test_mention_ensemble_rejects_unresolved_exact_span_type_tie(self) -> None:
        text = "Ada"
        document, segments = _document(text)
        calls: list[str] = []
        organization = MentionInference(
            start=0,
            end=3,
            text="Ada",
            entity_type="Organization",
            confidence=0.99,
        )
        person = MentionInference(
            start=0,
            end=3,
            text="Ada",
            entity_type="Person",
            confidence=0.1,
        )
        ensemble = MandatoryMentionEnsemble(
            (
                self._mention_provider(
                    "high-confidence",
                    FakeMentionBackend(
                        (organization,),
                        calls=calls,
                        name="high-confidence",
                    ),
                ),
                self._mention_provider(
                    "low-confidence",
                    FakeMentionBackend(
                        (person,),
                        calls=calls,
                        name="low-confidence",
                    ),
                ),
            )
        )

        with self.assertRaisesRegex(
            ProviderOutputError,
            "unresolved exact-span type vote tie",
        ):
            ensemble.detect(document, text, segments)

        self.assertEqual(calls, ["high-confidence", "low-confidence"])

    def test_all_mention_providers_run_even_when_one_fails(self) -> None:
        text = "Ada founded Acme."
        document, segments = _document(text)
        calls: list[str] = []
        providers = tuple(
            self._mention_provider(
                name,
                FakeMentionBackend(
                    error=RuntimeError("boom") if name == "two" else None,
                    calls=calls,
                    name=name,
                ),
            )
            for name in ("one", "two", "three")
        )
        ensemble = MandatoryMentionEnsemble(providers)
        with self.assertRaises(MandatoryProviderError) as raised:
            ensemble.detect(document, text, segments)
        self.assertEqual(calls, ["one", "two", "three"])
        self.assertEqual(set(raised.exception.failures), {"two"})

    def test_claim_ensemble_runs_union_and_records_agreement(self) -> None:
        text = "Ada founded Acme in 2020."
        document, segments = _document(text)
        providers = tuple(
            self._claim_provider(
                name,
                FakeStructuredBackend(_claim_payload(text)),
            )
            for name in ("one", "two", "three")
        )
        ensemble = MandatoryClaimEnsemble(providers)

        claims = ensemble.extract(document, text, segments, ())

        self.assertEqual(len(claims), 1)
        self.assertEqual(ensemble.last_agreements[0].ratio, 1.0)
        self.assertEqual(claims[0].provenance.provider, "one+three+two")
        self.assertIn("agreement=3/3", claims[0].provenance.configuration_id)

    def test_all_claim_providers_run_even_when_one_fails(self) -> None:
        text = "Ada founded Acme in 2020."
        document, segments = _document(text)
        calls: list[str] = []
        providers = tuple(
            self._claim_provider(
                name,
                FakeStructuredBackend(
                    _claim_payload(text),
                    error=RuntimeError("boom") if name == "two" else None,
                    calls=calls,
                    name=name,
                ),
            )
            for name in ("one", "two", "three")
        )
        ensemble = MandatoryClaimEnsemble(providers)
        with self.assertRaises(MandatoryProviderError) as raised:
            ensemble.extract(document, text, segments, ())
        self.assertEqual(calls, ["one", "two", "three"])
        self.assertEqual(set(raised.exception.failures), {"two"})


class PlatformFactoryTests(unittest.TestCase):
    def _overrides(self) -> ModelBackendOverrides:
        empty_mentions = FakeMentionBackend(())
        empty_claims = FakeStructuredBackend({"claims": []})
        return ModelBackendOverrides(
            gliner=empty_mentions,
            gliner_relex=FakeRelationBackend(()),
            transformers_ner=empty_mentions,
            spacy=empty_mentions,
            transformers_claims=empty_claims,
            mlx_claims=empty_claims,
            ollama_claims=empty_claims,
        )

    def test_darwin_arm64_factory_requires_and_includes_all_backends(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            models = _model_set(Path(temp), "darwin-arm64")
            bundle = build_mandatory_model_providers(
                models,
                overrides=self._overrides(),
                system="Darwin",
                machine="arm64",
            )
        self.assertEqual(
            bundle.mandatory_provider_names,
            (
                "deterministic_rule_mentions",
                "gliner",
                "transformers_ner",
                "spacy",
                "deterministic_pattern_claims",
                "gliner_relex",
                "transformers_qwen3",
                "mlx_qwen3",
                "ollama_qwen3",
            ),
        )
        self.assertEqual(len(bundle.mention_detector.providers), 4)
        self.assertEqual(len(bundle.claim_extractor.providers), 5)
        openrefine = next(
            resolver
            for resolver in bundle.resolution_ensemble.resolvers
            if resolver.name == "openrefine_reconciler"
        )
        self.assertEqual(openrefine.config.type_map, {"Person": "Q5"})

    def test_linux_factory_includes_transformers_gliner_spacy_and_ollama(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            models = _model_set(Path(temp), "linux-cpu")
            bundle = build_mandatory_model_providers(
                models,
                overrides=self._overrides(),
                system="Linux",
                machine="x86_64",
            )
        self.assertEqual(
            bundle.mandatory_provider_names,
            (
                "deterministic_rule_mentions",
                "gliner",
                "transformers_ner",
                "spacy",
                "deterministic_pattern_claims",
                "gliner_relex",
                "transformers_qwen3",
                "ollama_qwen3",
            ),
        )
        self.assertEqual(len(bundle.mention_detector.providers), 4)
        self.assertEqual(len(bundle.claim_extractor.providers), 4)

    def test_factory_claim_ensemble_runs_relex_and_all_generators_before_failure(
        self,
    ) -> None:
        text = "Ada founded Acme in 2020."
        document, segments = _document(text)
        calls: list[str] = []
        empty_mentions = FakeMentionBackend(())
        overrides = ModelBackendOverrides(
            gliner=empty_mentions,
            gliner_relex=FakeRelationBackend(
                error=RuntimeError("relex boom"),
                calls=calls,
                name="gliner_relex",
            ),
            transformers_ner=empty_mentions,
            spacy=empty_mentions,
            transformers_claims=FakeStructuredBackend(
                {"claims": []},
                calls=calls,
                name="transformers_qwen3",
            ),
            ollama_claims=FakeStructuredBackend(
                {"claims": []},
                calls=calls,
                name="ollama_qwen3",
            ),
        )
        with tempfile.TemporaryDirectory() as temp:
            bundle = build_mandatory_model_providers(
                _model_set(Path(temp), "linux-cpu"),
                relation_labels=("founded",),
                overrides=overrides,
                system="Linux",
                machine="x86_64",
            )
            with self.assertRaises(MandatoryProviderError) as raised:
                bundle.claim_extractor.extract(document, text, segments, ())

        self.assertEqual(
            calls,
            ["gliner_relex", "transformers_qwen3", "ollama_qwen3"],
        )
        self.assertEqual(set(raised.exception.failures), {"gliner_relex"})

    def test_profile_selection_is_exact_and_unsupported_hosts_fail(self) -> None:
        self.assertEqual(
            detect_mandatory_profile(system="Darwin", machine="arm64"),
            "darwin-arm64",
        )
        self.assertEqual(
            detect_mandatory_profile(
                system="Linux",
                machine="x86_64",
                cuda_available=False,
            ),
            "linux-cpu",
        )
        self.assertEqual(
            detect_mandatory_profile(
                system="Linux",
                machine="x86_64",
                cuda_available=True,
            ),
            "linux-cuda",
        )
        self.assertEqual(
            detect_mandatory_profile(
                system="Linux",
                machine="AMD64",
                cuda_available=False,
            ),
            "linux-cpu",
        )
        with self.assertRaises(OfflineArtifactError):
            detect_mandatory_profile(system="Windows", machine="AMD64")

    def test_factory_resolves_only_existing_locked_snapshot_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cache = root / "hub"
            revision = "c" * 40
            models = [
                (
                    "gliner_entity_extractor",
                    "gliner-community/gliner_small-v2.5",
                    "gliner",
                    "token-classification",
                ),
                (
                    "transformer_ner",
                    "dslim/bert-base-NER",
                    "transformers",
                    "token-classification",
                ),
                (
                    "gliner_relex_extractor",
                    "knowledgator/gliner-relex-large-v1.0",
                    "gliner",
                    "relation-extraction",
                ),
                (
                    "spacy_pipeline",
                    "spacy/en_core_web_sm",
                    "spacy",
                    "sentence-pos-ner",
                ),
                (
                    "structured_transformer_extractor",
                    "Qwen/Qwen3-1.7B",
                    "transformers",
                    "text-generation",
                ),
                (
                    "embedding_resolver",
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers",
                    "feature-extraction",
                ),
                (
                    "cross_encoder_resolver",
                    "cross-encoder/stsb-roberta-base",
                    "sentence-transformers",
                    "text-ranking",
                ),
            ]
            rows = []
            for name, repo_id, backend, task in models:
                snapshot = cache / f"models--{repo_id.replace('/', '--')}" / "snapshots" / revision
                snapshot.mkdir(parents=True)
                rows.append(
                    {
                        "name": name,
                        "repo_id": repo_id,
                        "revision": revision,
                        "backend": backend,
                        "task": task,
                        "profiles": ["linux-cpu"],
                        "device": "cpu",
                    }
                )
            lock = root / "models.lock.yaml"
            lock.write_text(
                yaml.safe_dump(
                    {
                        "models": rows,
                        "service_models": [
                            {
                                "name": "ollama_structured_extractor",
                                "service": "ollama",
                                "model": "qwen3:1.7b-q4_K_M",
                                "digest": f"sha256:{'d' * 64}",
                                "profiles": ["linux-cpu"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            bundle = build_mandatory_model_providers_from_lock(
                lock,
                cache_root=cache,
                overrides=self._overrides(),
                system="Linux",
                machine="x86_64",
                cuda_available=False,
            )
            self.assertEqual(bundle.profile, "linux-cpu")
            missing = cache / "models--Qwen--Qwen3-1.7B" / "snapshots" / revision
            missing.rmdir()
            with self.assertRaisesRegex(ValueError, "not hydrated"):
                build_mandatory_model_providers_from_lock(
                    lock,
                    cache_root=cache,
                    overrides=self._overrides(),
                    system="Linux",
                    machine="x86_64",
                    cuda_available=False,
                )


if __name__ == "__main__":
    unittest.main()
