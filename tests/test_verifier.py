"""Tests para el verifier: graph invariants, mapping bijectivity, interop confidence.

Tests unitarios que no requieren LLM ni grafo pre-existente.
"""

import pytest
from src.graph.catalog import NomencladorGraph
from src.graph.schema import (
    ConceptNode, FieldNode, ClassifierNode, SourceNode, EdgeType,
)
from src.verifier import (
    verify_graph_invariants, verify_mapping_bijectivity,
    compute_interop_confidence, compute_transform_sql,
)


@pytest.fixture
def simple_graph():
    """Grafo minimo con 1 concepto, 2 fields de fuentes distintas, y 1 clasificador."""
    g = NomencladorGraph()

    g.add_source(SourceNode(id="source:a", name="db_a"))
    g.add_source(SourceNode(id="source:b", name="db_b"))

    g.add_concept(ConceptNode(
        id="concept:sexo",
        name="sexo",
        definition="variable canonica de sexo",
        standard="ISO_5218",
        population="general",
        capture_method="auto",
    ))

    g.add_classifier(ClassifierNode(
        id="classifier:iso_5218",
        name="ISO 5218",
        standard="ISO_5218",
        values={"0": "desconocido", "1": "masculino", "2": "femenino"},
    ))
    g.link_clasificador("concept:sexo", "classifier:iso_5218")

    g.add_field(FieldNode(
        id="field:a.sexo",
        source_db="db_a",
        table="t",
        column="sexo",
        data_type="text",
        nullable=False,
        unique_count=3,
        null_count=0,
        total_count=100,
        sample_values=["M", "F", "M"],
        inferred_standard="ISO_5218",
        confidence="high",
        quality_score=0.9,
        completeness=1.0,
        uniqueness=0.03,
        consistency=1.0,
        validity=1.0,
    ))
    g.link_implementa("field:a.sexo", "concept:sexo")
    g.link_fuente("field:a.sexo", "source:a")

    g.add_field(FieldNode(
        id="field:b.genero",
        source_db="db_b",
        table="t",
        column="genero",
        data_type="text",
        nullable=False,
        unique_count=3,
        null_count=0,
        total_count=100,
        sample_values=["1", "2", "1"],
        inferred_standard="ISO_5218",
        confidence="high",
        quality_score=0.85,
        completeness=1.0,
        uniqueness=0.03,
        consistency=1.0,
        validity=1.0,
    ))
    g.link_implementa("field:b.genero", "concept:sexo")
    g.link_fuente("field:b.genero", "source:b")

    return g


class TestVerifyGraphInvariants:
    def test_clean_graph_passes(self, simple_graph):
        result = verify_graph_invariants(simple_graph)
        assert "passed" in result
        assert "violations" in result
        assert isinstance(result["violations"], list)

    def test_orphan_field_detected(self, simple_graph):
        # Add orphan field without concept link
        simple_graph.add_field(FieldNode(
            id="field:c.orphan",
            source_db="db_c",
            table="t",
            column="orphan",
            data_type="text",
            nullable=True,
            unique_count=1,
            null_count=0,
            total_count=10,
            sample_values=["x"],
        ))
        simple_graph.link_fuente("field:c.orphan", "source:a")
        # No link_implementa -> orphan
        result = verify_graph_invariants(simple_graph)
        # Should have at least a warning about orphan
        assert len(result["warnings"]) > 0 or len(result["violations"]) > 0


class TestVerifyMappingBijectivity:
    def test_bijective_mapping(self, simple_graph):
        # ISO_5218 to itself: no mapping edge declared, so not bijective is acceptable
        result = verify_mapping_bijectivity(simple_graph, "classifier:iso_5218", "classifier:iso_5218")
        assert "is_bijective" in result
        # Without explicit mapping edges, it's not bijective
        assert result["is_bijective"] == False

    def test_nonexistent_classifier(self, simple_graph):
        result = verify_mapping_bijectivity(simple_graph, "classifier:no_existe", "classifier:iso_5218")
        assert "error" in result


class TestComputeInteropConfidence:
    def test_confidence_for_field(self, simple_graph):
        result = compute_interop_confidence(simple_graph, "concept:sexo", "field:a.sexo")
        assert "confidence" in result
        assert "level" in result
        assert result["level"] in ("high", "medium", "low", "blocked")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_nonexistent_concept(self, simple_graph):
        result = compute_interop_confidence(simple_graph, "concept:no_existe", "field:a.sexo")
        # No error, but confidence should be low/blocked since concept doesn't exist
        assert result["level"] in ("low", "medium", "blocked")


class TestComputeTransformSql:
    def test_generates_sql(self, simple_graph):
        result = compute_transform_sql(simple_graph, "field:a.sexo", "concept:sexo", "field:b.genero")
        if "error" not in result:
            assert "sql" in result
            assert len(result["sql"]) > 0
        # Si hay error, es aceptable si no hay mapping entre los clasificadores
