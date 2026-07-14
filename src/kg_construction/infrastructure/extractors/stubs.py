"""Optional NLP, ML, OpenIE, and supervised relation extraction adapters."""

from __future__ import annotations

from kg_construction.domain.models import ConstructionBatch, SourceRecord
from kg_construction.infrastructure.loaders.stubs import AdapterNotConfigured


class ExtensibleExtractorStub:
    """Base class for extractors represented in the framework registry."""

    name = "extractor_stub"
    technique = "extractor_stub"
    dependency_hint = "Install and configure the model-specific adapter."

    def extract(self, source: SourceRecord) -> ConstructionBatch:
        raise AdapterNotConfigured(
            f"{self.name} is represented as an extensible extractor. "
            f"{self.dependency_hint} Source: {source.source_name}"
        )


class SentenceSplitterAdapter(ExtensibleExtractorStub):
    name = "sentence_splitter_adapter"
    technique = "classical_nlp_sentence_splitting"
    dependency_hint = "Use spaCy, NLTK, syntok, or a domain sentence splitter."


class PosTaggingAdapter(ExtensibleExtractorStub):
    name = "pos_tagging_adapter"
    technique = "classical_nlp_pos_tagging"
    dependency_hint = "Use spaCy/Stanza models and preserve token offsets."


class CoreferenceAdapter(ExtensibleExtractorStub):
    name = "coreference_adapter"
    technique = "classical_nlp_coreference"
    dependency_hint = "Use a coreference resolver and write mention-chain candidates."


class SpacyEntityRulerExtractor(ExtensibleExtractorStub):
    name = "spacy_entity_ruler_extractor"
    technique = "rule_based_spacy_entity_ruler"
    dependency_hint = "Install spaCy and configure EntityRuler patterns."


class DependencyPatternRelationExtractor(ExtensibleExtractorStub):
    name = "dependency_pattern_relation_extractor"
    technique = "rule_based_dependency_patterns"
    dependency_hint = "Install spaCy/Stanza and configure dependency path patterns."


class SpacyNerExtractor(ExtensibleExtractorStub):
    name = "spacy_ner_extractor"
    technique = "classical_nlp_spacy_ner"
    dependency_hint = "Install spaCy and the selected language model."


class TransformerNerExtractor(ExtensibleExtractorStub):
    name = "transformer_ner_extractor"
    technique = "ml_transformer_token_classification"
    dependency_hint = "Install transformers/torch and provide model id and label map."


class OpenIERelationExtractor(ExtensibleExtractorStub):
    name = "openie_relation_extractor"
    technique = "open_information_extraction"
    dependency_hint = "Use OpenIE/Stanford/CoreNLP or a local OpenIE adapter."


class PairwiseRelationClassifierAdapter(ExtensibleExtractorStub):
    name = "pairwise_relation_classifier_adapter"
    technique = "supervised_pairwise_relation_classification"
    dependency_hint = "Provide pair candidates, model checkpoint, and schema labels."


class JointEntityRelationExtractor(ExtensibleExtractorStub):
    name = "joint_entity_relation_extractor"
    technique = "supervised_joint_entity_relation_extraction"
    dependency_hint = "Provide a joint IE model checkpoint and decoding constraints."


class RebelRelationExtractor(ExtensibleExtractorStub):
    name = "rebel_relation_extractor"
    technique = "generative_rebel_relation_extraction"
    dependency_hint = "Install transformers/torch and configure REBEL-style model."


class SupervisedRelationExtractor(ExtensibleExtractorStub):
    name = "supervised_relation_extractor"
    technique = "supervised_relation_extraction"
    dependency_hint = "Provide labeled relation schema and classifier model."


__all__ = [
    "CoreferenceAdapter",
    "DependencyPatternRelationExtractor",
    "ExtensibleExtractorStub",
    "JointEntityRelationExtractor",
    "OpenIERelationExtractor",
    "PairwiseRelationClassifierAdapter",
    "PosTaggingAdapter",
    "RebelRelationExtractor",
    "SentenceSplitterAdapter",
    "SpacyEntityRulerExtractor",
    "SpacyNerExtractor",
    "SupervisedRelationExtractor",
    "TransformerNerExtractor",
]
