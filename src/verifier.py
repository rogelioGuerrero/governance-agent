"""
Computational Verification Layer — Wolfram-style para el Knowledge Graph.

Filosofia: "Compute, don't guess."

El agente ReAct usa el LLM (probabilistico) para razonar sobre QUE hacer.
Este modulo hace la EJECUCION deterministica:

1. verify_classifier_consistency: sample_values del field ⊆ valid_values del classifier
2. verify_mapping_bijectivity: mapping 1:1 entre clasificadores es biyectivo
3. compute_interop_confidence: score determinista (quality + review + staleness + classifier_match)
4. compute_transform_sql: genera SQL CASE WHEN desde el mapping del grafo, sin LLM
5. verify_graph_invariants: audit completo de integridad del grafo

Nada de esto usa el LLM. Todo es computo simbolico sobre la estructura del grafo.
"""

import logging
import re
from datetime import date, datetime
from typing import Optional

from .graph.catalog import NomencladorGraph
from .graph.schema import NodeType, EdgeType
from .standards import STANDARDS, get_standard_values

logger = logging.getLogger(__name__)


# === 1. CLASSIFIER CONSISTENCY ===

def verify_classifier_consistency(graph: NomencladorGraph, field_id: str) -> dict:
    """Verificar que los sample_values de un field estan dentro del clasificador.

    Determinista: compara los valores muestrales contra los valores validos
    del clasificador asociado al concepto que este field implementa.

    Returns:
        {
            "field_id": str,
            "concept_id": str,
            "classifier_id": str | None,
            "valid_count": int,
            "invalid_count": int,
            "invalid_values": list[str],
            "match_ratio": float,   # 0.0-1.0
            "passed": bool,
        }
    """
    result = {
        "field_id": field_id,
        "concept_id": None,
        "classifier_id": None,
        "valid_count": 0,
        "invalid_count": 0,
        "invalid_values": [],
        "match_ratio": 0.0,
        "passed": False,
    }

    field_node = graph.graph.nodes.get(field_id)
    if not field_node or field_node.get("type") != NodeType.FIELD.value:
        result["error"] = f"Field '{field_id}' no encontrado"
        return result

    sample_values = field_node.get("sample_values", [])
    if not sample_values:
        result["error"] = "Sin sample_values para verificar"
        return result

    # Encontrar el concepto que este field implementa
    concept_id = None
    for successor in graph.graph.successors(field_id):
        edge = graph.graph.get_edge_data(field_id, successor)
        if edge and edge.get("type") == EdgeType.IMPLEMENTA.value:
            concept_id = successor
            break

    if not concept_id:
        result["error"] = "Field no implementa ningun concepto"
        return result

    result["concept_id"] = concept_id

    # Encontrar el clasificador del concepto
    classifier = graph.find_classifier_of_concept(concept_id)
    if not classifier:
        result["error"] = "Concepto no tiene clasificador asociado"
        return result

    result["classifier_id"] = classifier["id"]

    # Obtener valores validos del clasificador
    standard_id = classifier.get("standard")
    if standard_id and standard_id in STANDARDS:
        valid_values = set(get_standard_values(standard_id).keys())
    else:
        # Usar valores del nodo clasificador si no hay estandar registrado
        valid_values = set(classifier.get("values", {}).keys())

    if not valid_values:
        result["error"] = "Clasificador sin valores validos cargados"
        return result

    # Comparar case-insensitive
    valid_lower = {v.lower().strip() for v in valid_values if v}
    invalid = []
    for sv in sample_values:
        if sv is None:
            continue
        sv_str = str(sv).lower().strip()
        if sv_str and sv_str not in valid_lower:
            invalid.append(str(sv))

    result["valid_count"] = len(sample_values) - len(invalid)
    result["invalid_count"] = len(invalid)
    result["invalid_values"] = invalid[:20]
    total = len([s for s in sample_values if s is not None])
    result["match_ratio"] = round((total - len(invalid)) / total, 3) if total > 0 else 0.0
    result["passed"] = len(invalid) == 0
    return result


# === 2. MAPPING BIJECTIVITY ===

def verify_mapping_bijectivity(graph: NomencladorGraph, classifier_a: str, classifier_b: str) -> dict:
    """Verificar que un mapping EQUIVALE_A entre dos clasificadores es biyectivo.

    Determinista: revisa que cada valor de A mapee a un unico valor de B
    y viceversa (para cardinalidad 1:1).

    Returns:
        {
            "classifier_a": str,
            "classifier_b": str,
            "declared_cardinality": str,
            "is_bijective": bool,
            "a_to_b": dict,       # mapping directo
            "b_to_a": dict,       # mapping inverso computado
            "conflicts": list[str],
        }
    """
    edge_data = graph.graph.get_edge_data(classifier_a, classifier_b)
    if not edge_data or edge_data.get("type") != EdgeType.EQUIVALE_A.value:
        # Buscar en direccion inversa
        edge_data = graph.graph.get_edge_data(classifier_b, classifier_a)
        if not edge_data or edge_data.get("type") != EdgeType.EQUIVALE_A.value:
            return {
                "error": f"No hay edge EQUIVALE_A entre {classifier_a} y {classifier_b}",
                "is_bijective": False,
            }
        # Swap para que A->B sea el mapping
        classifier_a, classifier_b = classifier_b, classifier_a

    mapping = edge_data.get("mapping", {})
    cardinality = edge_data.get("cardinality", "1:1")

    result = {
        "classifier_a": classifier_a,
        "classifier_b": classifier_b,
        "declared_cardinality": cardinality,
        "is_bijective": False,
        "a_to_b": mapping,
        "b_to_a": {},
        "conflicts": [],
    }

    if cardinality != "1:1":
        result["conflicts"].append(
            f"Cardinalidad declarada es {cardinality}, no 1:1 — biyectividad no aplica"
        )
        return result

    # Computar mapping inverso
    b_to_a: dict[str, list[str]] = {}
    for a_val, b_val in mapping.items():
        b_to_a.setdefault(str(b_val), []).append(str(a_val))

    result["b_to_a"] = {k: v[0] if len(v) == 1 else v for k, v in b_to_a.items()}

    # Verificar biyectividad: cada B recibe exactamente un A
    conflicts = []
    for b_val, a_vals in b_to_a.items():
        if len(a_vals) > 1:
            conflicts.append(
                f"B='{b_val}' recibe multiples valores de A: {a_vals}"
            )

    # Verificar que no hay valores de A sin mapear (si B tiene valores conocidos)
    node_b = graph.graph.nodes.get(classifier_b, {})
    b_valid_values = node_b.get("valid_values", {}) or node_b.get("values", {})
    if b_valid_values:
        mapped_b_values = set(str(v) for v in mapping.values())
        known_b_values = set(str(k) for k in b_valid_values.keys())
        unmapped = known_b_values - mapped_b_values
        if unmapped:
            conflicts.append(
                f"Valores de B sin mapeo desde A: {sorted(unmapped)[:10]}"
            )

    result["conflicts"] = conflicts
    result["is_bijective"] = len(conflicts) == 0
    return result


# === 3. INTEROP CONFIDENCE (formula determinista) ===

def _staleness_factor(last_verified: str) -> float:
    """Factor de frescura: 1.0 si verificado hoy, decae a 0.0 en 365 dias.

    Formula: max(0, 1 - days_since / 365)
    """
    if not last_verified:
        return 0.0  # Nunca verificado = frescura 0
    try:
        verified = datetime.fromisoformat(last_verified).date()
        days = (date.today() - verified).days
        return max(0.0, 1.0 - days / 365.0)
    except (ValueError, TypeError):
        return 0.0


def _review_factor(review_status: str) -> float:
    """Factor de revision: 1.0 si approved, 0.5 si under_review, 0.0 si proposed/rejected."""
    return {
        "approved": 1.0,
        "under_review": 0.5,
        "proposed": 0.0,
        "rejected": 0.0,
    }.get(review_status, 0.0)


def compute_interop_confidence(
    graph: NomencladorGraph,
    concept_id: str,
    field_id: str,
) -> dict:
    """Calcular confidence score determinista para interoperabilidad.

    Formula ponderada (sin LLM):
        confidence = quality_score * 0.35
                   + review_factor * 0.25
                   + staleness_factor * 0.15
                   + classifier_match * 0.25

    Returns:
        {
            "concept_id": str,
            "field_id": str,
            "quality_score": float,
            "review_factor": float,
            "staleness_factor": float,
            "classifier_match": float,
            "confidence": float,       # 0.0-1.0
            "level": str,              # "high" | "medium" | "low" | "blocked"
            "reasons": list[str],
        }
    """
    result = {
        "concept_id": concept_id,
        "field_id": field_id,
        "quality_score": 0.0,
        "review_factor": 0.0,
        "staleness_factor": 0.0,
        "classifier_match": 0.0,
        "confidence": 0.0,
        "level": "blocked",
        "reasons": [],
    }

    field_node = graph.graph.nodes.get(field_id)
    if not field_node:
        result["error"] = f"Field '{field_id}' no encontrado"
        return result

    # 1. Quality score (0.0-1.0)
    qs = field_node.get("quality_score", 0.0)
    if qs == 0.0:
        qs = graph.compute_quality_score(field_id)
    result["quality_score"] = round(qs, 3)

    # 2. Review factor
    rs = field_node.get("review_status", "approved")
    rf = _review_factor(rs)
    result["review_factor"] = rf
    if rf == 0.0:
        result["reasons"].append(f"review_status={rs} bloquea interoperabilidad")

    # 3. Staleness factor
    lv = field_node.get("last_verified", "")
    sf = _staleness_factor(lv)
    result["staleness_factor"] = round(sf, 3)
    if sf < 0.5:
        result["reasons"].append(
            f"last_verified={lv or 'nunca'} — dato posiblemente desactualizado"
        )

    # 4. Classifier match ratio
    cm = 0.0
    concept_node = graph.graph.nodes.get(concept_id, {})
    standard_id = concept_node.get("standard")
    if standard_id and standard_id in STANDARDS:
        valid_values = set(str(k).lower() for k in get_standard_values(standard_id).keys())
        samples = [str(s).lower().strip() for s in field_node.get("sample_values", []) if s]
        if samples and valid_values:
            matched = sum(1 for s in samples if s in valid_values)
            cm = matched / len(samples)
    result["classifier_match"] = round(cm, 3)
    if cm < 0.8:
        result["reasons"].append(
            f"classifier_match={cm:.0%} — valores muestrales fuera del estandar"
        )

    # Formula determinista
    confidence = qs * 0.35 + rf * 0.25 + sf * 0.15 + cm * 0.25
    result["confidence"] = round(confidence, 3)

    # Nivel de confianza
    if rf == 0.0:
        result["level"] = "blocked"
    elif confidence >= 0.7:
        result["level"] = "high"
    elif confidence >= 0.4:
        result["level"] = "medium"
    else:
        result["level"] = "low"

    return result


# === 4. COMPUTE TRANSFORM SQL (sin LLM) ===

def _escape_sql_literal(value: str) -> str:
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def _escape_sql_identifier(identifier: str) -> str:
    if not identifier or not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', identifier):
        logger.warning("SQL identifier invalido: %r — usando placeholder", identifier)
        return '"invalid_identifier"'
    return f'"{identifier}"'


def compute_transform_sql(
    graph: NomencladorGraph,
    field_a_id: str,
    concept_id: str,
    field_b_id: str,
) -> dict:
    """Generar SQL CASE WHEN determinista desde el mapping del grafo.

    A diferencia de generate_sql_transform() en transformer.py (que usa el
    estandar registrado), esta funcion lee directamente el edge EQUIVALE_A
    del grafo si existe, o el clasificador del concepto.

    Returns:
        {
            "concept_id": str,
            "field_a": str,
            "field_b": str,
            "sql": str,
            "mapping_used": dict,
            "source": str,   # "edge_equivalente_a" | "classifier" | "standard"
        }
    """
    result = {
        "concept_id": concept_id,
        "field_a": field_a_id,
        "field_b": field_b_id,
        "sql": "",
        "mapping_used": {},
        "source": "",
    }

    field_a = graph.graph.nodes.get(field_a_id, {})
    field_b = graph.graph.nodes.get(field_b_id, {})
    col_a = field_a.get("column", "source_col")
    col_b = field_b.get("column", "target_col")

    # Intentar 1: edge EQUIVALE_A entre clasificadores
    classifier = graph.find_classifier_of_concept(concept_id)
    mapping = None
    source = ""

    if classifier:
        # Buscar edges EQUIVALE_A desde/hacia este clasificador
        for _, target, data in graph.graph.out_edges(classifier.get("id", ""), data=True):
            if data.get("type") == EdgeType.EQUIVALE_A.value:
                mapping = data.get("mapping", {})
                source = "edge_equivalente_a"
                break
        if not mapping:
            for source_node, _, data in graph.graph.in_edges(classifier.get("id", ""), data=True):
                if data.get("type") == EdgeType.EQUIVALE_A.value:
                    mapping = data.get("mapping", {})
                    source = "edge_equivalente_a"
                    break

    # Intentar 2: valores del clasificador del concepto
    if not mapping and classifier:
        standard_id = classifier.get("standard")
        if standard_id and standard_id in STANDARDS:
            std_values = get_standard_values(standard_id)
            if std_values:
                # Mapping identidad: cada valor canonico mapea a si mismo
                mapping = {k: k for k in std_values}
                source = "standard"

    # Intentar 3: valores directos del nodo clasificador
    if not mapping and classifier:
        cv = classifier.get("values", {})
        if cv:
            mapping = {k: k for k in cv}
            source = "classifier"

    if not mapping:
        result["error"] = "No se encontro mapping en el grafo para generar SQL"
        return result

    result["mapping_used"] = mapping
    result["source"] = source

    # Generar SQL CASE WHEN determinista
    col_escaped = _escape_sql_identifier(col_a)
    target_escaped = _escape_sql_identifier(col_b)

    case_lines = [f"CASE"]
    for source_val, target_val in mapping.items():
        case_lines.append(
            f"  WHEN {col_escaped} = {_escape_sql_literal(source_val)}"
            f" THEN {_escape_sql_literal(target_val)}"
        )
    case_lines.append(f"  ELSE NULL")
    case_lines.append(f"END AS {target_escaped}")

    result["sql"] = "\n".join(case_lines)
    return result


# === 5. GRAPH INVARIANTS (audit completo) ===

def verify_graph_invariants(graph: NomencladorGraph) -> dict:
    """Audit determinista de integridad del grafo.

    Verifica invariantes estructurales sin LLM:
    - Todo Concept tiene al menos un Field que lo implementa (o esta marcado como propuesto)
    - Todo Field implementa exactamente un Concept
    - Todo edge IMPLEMENTA conecta Field -> Concept (no al reves)
    - Todo edge USA_CLASIFICADOR conecta Concept -> Classifier
    - Todo Concept approved tiene clasificador asociado
    - Todo Field con quality_score < 0.3 tiene al menos un QualityIssue
    - No hay nodos huerfanos (sin edges)

    Returns:
        {
            "total_nodes": int,
            "total_edges": int,
            "violations": list[dict],
            "warnings": list[dict],
            "passed": bool,
        }
    """
    violations = []
    warnings = []

    for node_id, data in graph.graph.nodes(data=True):
        node_type = data.get("type", "unknown")

        # Invariante: Concept approved sin clasificador
        if node_type == NodeType.CONCEPT.value:
            review = data.get("review_status", "approved")
            if review == "approved":
                classifier = graph.find_classifier_of_concept(node_id)
                if not classifier:
                    warnings.append({
                        "node": node_id,
                        "type": "concept_without_classifier",
                        "message": f"Concepto aprobado '{data.get('name', node_id)}' sin clasificador",
                    })

                # Concept approved sin fields que lo implementen
                fields = graph.find_fields_of_concept(node_id)
                if not fields:
                    warnings.append({
                        "node": node_id,
                        "type": "concept_without_fields",
                        "message": f"Concepto aprobado '{data.get('name', node_id)}' sin implementaciones fisicas",
                    })

        # Invariante: Field con quality_score bajo sin issues
        if node_type == NodeType.FIELD.value:
            qs = data.get("quality_score", 0.0)
            if qs < 0.3 and qs > 0.0:
                issues = graph.find_issues_of_field(node_id)
                if not issues:
                    warnings.append({
                        "node": node_id,
                        "type": "low_quality_without_issue",
                        "message": f"Field '{data.get('column', node_id)}' quality_score={qs} sin issues registrados",
                    })

            # Field sin concepto
            has_concept = False
            for successor in graph.graph.successors(node_id):
                edge = graph.graph.get_edge_data(node_id, successor)
                if edge and edge.get("type") == EdgeType.IMPLEMENTA.value:
                    has_concept = True
                    break
            if not has_concept:
                violations.append({
                    "node": node_id,
                    "type": "field_without_concept",
                    "message": f"Field '{data.get('column', node_id)}' no implementa ningun concepto",
                })

    # Verificar direccion de edges
    for source, target, data in graph.graph.edges(data=True):
        edge_type = data.get("type", "")
        source_type = graph.graph.nodes.get(source, {}).get("type", "")
        target_type = graph.graph.nodes.get(target, {}).get("type", "")

        if edge_type == EdgeType.IMPLEMENTA.value:
            if source_type != NodeType.FIELD.value or target_type != NodeType.CONCEPT.value:
                violations.append({
                    "edge": f"{source} -> {target}",
                    "type": "invalid_implements_direction",
                    "message": f"IMPLEMENTA debe ser Field->Concept, pero es {source_type}->{target_type}",
                })

        if edge_type == EdgeType.USA_CLASIFICADOR.value:
            if source_type != NodeType.CONCEPT.value or target_type != NodeType.CLASSIFIER.value:
                violations.append({
                    "edge": f"{source} -> {target}",
                    "type": "invalid_classifier_direction",
                    "message": f"USA_CLASIFICADOR debe ser Concept->Classifier, pero es {source_type}->{target_type}",
                })

        if edge_type == EdgeType.EQUIVALE_A.value:
            if source_type != NodeType.CLASSIFIER.value or target_type != NodeType.CLASSIFIER.value:
                violations.append({
                    "edge": f"{source} -> {target}",
                    "type": "invalid_equivalence_direction",
                    "message": f"EQUIVALE_A debe ser Classifier->Classifier, pero es {source_type}->{target_type}",
                })

        if edge_type == EdgeType.TIENE_CONTEXTO.value:
            if source_type != NodeType.CONCEPT.value or target_type != NodeType.CONTEXT.value:
                violations.append({
                    "edge": f"{source} -> {target}",
                    "type": "invalid_context_direction",
                    "message": f"TIENE_CONTEXTO debe ser Concept->Context, pero es {source_type}->{target_type}",
                })

    # Nodos huerfanos
    for node_id, data in graph.graph.nodes(data=True):
        if graph.graph.degree(node_id) == 0:
            warnings.append({
                "node": node_id,
                "type": "orphan_node",
                "message": f"Nodo huerfano: {data.get('name', node_id)} ({data.get('type', '?')})",
            })

    return {
        "total_nodes": graph.graph.number_of_nodes(),
        "total_edges": graph.graph.number_of_edges(),
        "violations": violations,
        "warnings": warnings,
        "passed": len(violations) == 0,
    }


# === BATCH VERIFICATION ===

def verify_all_fields(graph: NomencladorGraph) -> list[dict]:
    """Ejecutar verify_classifier_consistency sobre todos los fields del grafo."""
    results = []
    for node_id, data in graph.graph.nodes(data=True):
        if data.get("type") == NodeType.FIELD.value:
            results.append(verify_classifier_consistency(graph, node_id))
    return results


def compute_all_confidence(graph: NomencladorGraph) -> list[dict]:
    """Calcular interop_confidence para todos los pares field-concept del grafo."""
    results = []
    for node_id, data in graph.graph.nodes(data=True):
        if data.get("type") != NodeType.FIELD.value:
            continue
        # Encontrar concepto que implementa
        for successor in graph.graph.successors(node_id):
            edge = graph.graph.get_edge_data(node_id, successor)
            if edge and edge.get("type") == EdgeType.IMPLEMENTA.value:
                results.append(compute_interop_confidence(graph, successor, node_id))
                break
    return results
