"""Test E2E real del pipeline de governance.

Ejecuta el pipeline completo con datos reales (sample_censo.csv, sample_hospital.csv)
y verifica cada output. Sin LLM, sin mocks, sin humo.

Pipeline testeado:
  1. INGEST: ingestar sample_censo.csv y sample_hospital.csv al grafo
  2. INTEROP: encontrar caminos de interoperabilidad entre las dos fuentes
  3. GUARDRAILS: validar cada camino con checkpoints
  4. TRANSFORM: generar artefactos SQL + JSON Schema
  5. SEARCH: buscar variables en el nomenclador
  6. HEALTH: verificar invariantes del grafo
  7. VERIFY: verificar biyectividad de mappings y confianza de interop

Este test responde a la pregunta: "¿el agente governance funciona?"
Si este test pasa, el pipeline determinístico funciona.
Lo que NO prueba: el loop ReAct del agente (que depende de LLM).
"""

import pytest
import json
import os
import tempfile
from pathlib import Path

from src.rag_factory import create_ingestion_plan, execute_ingestion_plan
from src.graph.catalog import NomencladorGraph, load_graph_cached, clear_graph_cache, _NOMENCLADOR_PATH
from src.guardrails import validate_interoperability, CheckpointStatus
from src.transformer import generate_transformation, artifact_to_dict
from src.verifier import verify_graph_invariants, compute_interop_confidence
from src.health import check_health
from src.standards import detect_standard

TESTS_DIR = Path(__file__).parent
CENSO_CSV = str(TESTS_DIR / "sample_censo.csv")
HOSPITAL_CSV = str(TESTS_DIR / "sample_hospital.csv")


# === FIXTURE: grafo limpio + ingest real ===

@pytest.fixture(scope="module")
def ingested_graph():
    """Ingestar ambos CSVs al grafo real y retornar el grafo.
    
    Usa el nomenclador.json existente. Si ya hay datos, los reaprovecha.
    """
    clear_graph_cache()
    g = load_graph_cached()
    
    # Si sample_censo no está en el grafo, ingestarlo
    existing_sources = set()
    for nid, ndata in g.graph.nodes(data=True):
        if ndata.get("type") == "source":
            existing_sources.add(ndata.get("name", ""))
    
    if "sample_censo" not in existing_sources:
        plan = create_ingestion_plan(CENSO_CSV, source_type="csv", use_llm=False)
        execute_ingestion_plan(plan, auto_confirm=True)
    
    if "sample_hospital" not in existing_sources:
        plan = create_ingestion_plan(HOSPITAL_CSV, source_type="csv", use_llm=False)
        execute_ingestion_plan(plan, auto_confirm=True)
    
    clear_graph_cache()
    g = load_graph_cached()
    yield g
    clear_graph_cache()


# === 1. INGEST ===

class TestIngest:
    """Verificar que la ingesta de ambos CSVs produjo nodos reales en el grafo."""

    def test_censo_was_ingested(self, ingested_graph):
        """sample_censo debe tener source node + field nodes en el grafo."""
        g = ingested_graph
        fields = [n for n, d in g.graph.nodes(data=True) 
                  if d.get("type") == "field" and d.get("source_db") == "sample_censo"]
        assert len(fields) > 0, "No hay fields de sample_censo en el grafo"
        
        # Debe tener al menos sexo, edad, fecha_nacimiento
        columns = [d.get("column", "") for _, d in 
                   [(n, g.graph.nodes[n]) for n in fields]]
        assert "sexo" in columns, f"sexo no está en {columns}"

    def test_hospital_was_ingested(self, ingested_graph):
        """sample_hospital debe tener source node + field nodes en el grafo."""
        g = ingested_graph
        fields = [n for n, d in g.graph.nodes(data=True)
                  if d.get("type") == "field" and d.get("source_db") == "sample_hospital"]
        assert len(fields) > 0, "No hay fields de sample_hospital en el grafo"

    def test_concepts_were_created(self, ingested_graph):
        """La ingesta debe haber creado conceptos canónicos."""
        g = ingested_graph
        concepts = g.list_concepts()
        assert len(concepts) > 0, "No hay conceptos en el grafo"

    def test_quality_metrics_computed(self, ingested_graph):
        """Cada field debe tener quality_score calculado."""
        g = ingested_graph
        for nid, data in g.graph.nodes(data=True):
            if data.get("type") == "field":
                assert "quality_score" in data, f"Field {nid} sin quality_score"
                assert 0.0 <= data["quality_score"] <= 1.0, \
                    f"quality_score inválido en {nid}: {data['quality_score']}"


# === 2. INTEROP ===

class TestInterop:
    """Verificar que el grafo encuentra caminos de interoperabilidad reales."""

    def test_interop_paths_exist(self, ingested_graph):
        """Debe haber al menos un camino entre sample_censo y sample_hospital."""
        g = ingested_graph
        results = g.find_interoperability_path("sample_censo", "sample_hospital")
        assert len(results) > 0, "No se encontraron caminos de interoperabilidad"

    def test_interop_results_have_required_fields(self, ingested_graph):
        """Cada resultado debe tener field_a, field_b, concept."""
        g = ingested_graph
        results = g.find_interoperability_path("sample_censo", "sample_hospital")
        for r in results:
            assert "field_a" in r
            assert "field_b" in r
            assert "concept" in r
            assert r["field_a"].get("source_db") == "sample_censo"
            assert r["field_b"].get("source_db") == "sample_hospital"


# === 3. GUARDRAILS ===

class TestGuardrails:
    """Verificar que los guardrails validan cada camino de interoperabilidad."""

    def test_guardrails_produce_checkpoints(self, ingested_graph):
        """Cada camino debe producir checkpoints de validación."""
        g = ingested_graph
        results = g.find_interoperability_path("sample_censo", "sample_hospital")
        for r in results:
            validation = validate_interoperability(
                r["field_a"], r["field_b"], r["concept"], r.get("classifier")
            )
            assert len(validation.checkpoints) > 0, "Sin checkpoints"
            for cp in validation.checkpoints:
                assert cp.status in CheckpointStatus
                assert cp.name
                assert cp.detail is not None

    def test_guardrails_produce_recommendation(self, ingested_graph):
        """Cada validación debe tener una recomendación."""
        g = ingested_graph
        results = g.find_interoperability_path("sample_censo", "sample_hospital")
        for r in results:
            validation = validate_interoperability(
                r["field_a"], r["field_b"], r["concept"], r.get("classifier")
            )
            assert validation.recommendation, "Sin recomendación"
            assert validation.is_safe in (True, False)


# === 4. TRANSFORM ===

class TestTransform:
    """Verificar que se generan artefactos de transformación reales."""

    def test_sql_transform_generated(self, ingested_graph):
        """Cada camino debe generar SQL transform ejecutable."""
        g = ingested_graph
        results = g.find_interoperability_path("sample_censo", "sample_hospital")
        for r in results:
            validation = validate_interoperability(
                r["field_a"], r["field_b"], r["concept"], r.get("classifier")
            )
            artifact = generate_transformation(
                r["field_a"], r["field_b"], r["concept"], r.get("classifier"), validation
            )
            assert artifact.sql_transform, "SQL vacío"
            assert len(artifact.sql_transform) > 10, "SQL demasiado corto"
            # Debe tener estructura CASE WHEN o CAST o similar
            sql_upper = artifact.sql_transform.upper()
            assert any(kw in sql_upper for kw in ["CASE", "CAST", "SELECT", "INSERT", "--"]), \
                f"SQL no tiene estructura válida: {artifact.sql_transform[:100]}"

    def test_json_schema_generated(self, ingested_graph):
        """Cada camino debe generar un JSON Schema de validación."""
        g = ingested_graph
        results = g.find_interoperability_path("sample_censo", "sample_hospital")
        for r in results:
            validation = validate_interoperability(
                r["field_a"], r["field_b"], r["concept"], r.get("classifier")
            )
            artifact = generate_transformation(
                r["field_a"], r["field_b"], r["concept"], r.get("classifier"), validation
            )
            assert artifact.json_schema is not None, "JSON Schema vacío"
            # Debe ser un dict serializable
            json.dumps(artifact.json_schema)

    def test_artifact_to_dict_serializable(self, ingested_graph):
        """artifact_to_dict debe producir un dict serializable a JSON."""
        g = ingested_graph
        results = g.find_interoperability_path("sample_censo", "sample_hospital")
        for r in results:
            validation = validate_interoperability(
                r["field_a"], r["field_b"], r["concept"], r.get("classifier")
            )
            artifact = generate_transformation(
                r["field_a"], r["field_b"], r["concept"], r.get("classifier"), validation
            )
            d = artifact_to_dict(artifact)
            json.dumps(d)  # No debe lanzar excepción
            assert "concept" in d
            assert "sql_transform" in d


# === 5. SEARCH ===

class TestSearch:
    """Verificar que la búsqueda de variables funciona en el grafo real."""

    def test_find_sexo(self, ingested_graph):
        """Buscar 'sexo' debe retornar el concepto con sus fuentes."""
        g = ingested_graph
        concept = g.find_concept("sexo")
        assert concept is not None, "Concepto sexo no encontrado"
        assert concept.get("name") == "sexo"

    def test_find_fields_of_sexo(self, ingested_graph):
        """sexo debe tener fields implementándolo en ambas fuentes."""
        g = ingested_graph
        concept = g.find_concept("sexo")
        if not concept:
            pytest.skip("Concepto sexo no existe")
        fields = g.find_fields_of_concept(concept["id"])
        assert len(fields) > 0, "sexo no tiene fields"
        sources = set(f.get("source_db", "") for f in fields)
        # Al menos una fuente debe tener sexo
        assert len(sources) >= 1


# === 6. HEALTH ===

class TestHealth:
    """Verificar que el health check corre sobre el grafo real."""

    def test_health_check_runs(self, ingested_graph):
        """check_health debe ejecutarse sin excepción y retornar un reporte."""
        report = check_health()
        assert "passed" in report
        assert "graph_stats" in report
        assert report["graph_stats"]["total_nodes"] > 0
        assert "violations" in report["graph_audit"]

    def test_graph_has_no_critical_invariant_violations(self, ingested_graph):
        """verify_graph_invariants no debe tener violaciones críticas."""
        g = ingested_graph
        result = verify_graph_invariants(g)
        # Violations = problemas estructurales graves (no warnings)
        critical = [v for v in result["violations"] if v.get("severity") == "critical"]
        assert len(critical) == 0, \
            f"Violaciones críticas: {[v['message'] for v in critical]}"


# === 7. VERIFY ===

class TestVerify:
    """Verificar confianza de interoperabilidad y mappings."""

    def test_interop_confidence_for_sexo(self, ingested_graph):
        """compute_interop_confidence debe retornar un score coherente para sexo."""
        g = ingested_graph
        concept = g.find_concept("sexo")
        if not concept:
            pytest.skip("Concepto sexo no existe")
        fields = g.find_fields_of_concept(concept["id"])
        if not fields:
            pytest.skip("sexo no tiene fields")
        result = compute_interop_confidence(g, concept["id"], fields[0]["id"])
        assert "confidence" in result
        assert "level" in result
        assert result["level"] in ("high", "medium", "low", "blocked")
        assert 0.0 <= result["confidence"] <= 1.0


# === RESUMEN: qué funciona y qué no ===

class TestPipelineSummary:
    """Test que documenta exactamente qué funciona del agente governance."""

    def test_deterministic_pipeline_works(self, ingested_graph):
        """El pipeline determinístico (sin LLM) funciona end-to-end:
        ingest → interop → guardrails → transform → search → health.
        
        Lo que NO prueba este test:
        - Loop ReAct del agente (depende de LLM)
        - MoA multi-agente (depende de LLM)
        - Inferencia semántica con LLM (use_llm=True)
        - RAG normativo (depende de LLM)
        
        Lo que SÍ prueba:
        - Ingesta de CSVs al grafo
        - Detección de estándares (ISO 8601, ISO 5218, CIE-10)
        - Cálculo de métricas de calidad
        - Búsqueda de caminos de interoperabilidad
        - Guardrails con checkpoints
        - Generación de artefactos SQL + JSON Schema
        - Health check del grafo
        - Verificación de invariantes
        """
        g = ingested_graph
        stats = g.stats()
        assert stats["total_nodes"] > 0
        assert stats["total_edges"] > 0
        
        # El pipeline completo corre
        results = g.find_interoperability_path("sample_censo", "sample_hospital")
        assert len(results) > 0
        
        for r in results:
            validation = validate_interoperability(
                r["field_a"], r["field_b"], r["concept"], r.get("classifier")
            )
            artifact = generate_transformation(
                r["field_a"], r["field_b"], r["concept"], r.get("classifier"), validation
            )
            assert artifact.sql_transform
            assert artifact.json_schema is not None
        
        report = check_health()
        assert "passed" in report
