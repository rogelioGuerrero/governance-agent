"""
Agente ReAct con LangGraph + Groq.

State machine que implementa el loop Reasoning + Acting:
1. THINK: Groq razona sobre la tarea y decide qué tool usar
2. ACT: ejecuta la tool seleccionada
3. OBSERVE: observa el resultado
4. Repite hasta que el agente decide FINAL

Diferencia con el CLI: el CLI ejecuta comandos fijos en secuencia.
El agente ReAct RAZONA sobre qué hacer siguiente, puede combinar
tools, pedir aclaración al humano, y adaptar su estrategia.

Ejemplo:
    Usuario: "¿Puedo cruzar el censo con el hospital?"
    
    THINK: "Necesito verificar interoperabilidad entre sample_censo y sample_hospital"
    ACT: validate_interop("sample_censo", "sample_hospital")
    OBSERVE: "3 caminos, todos con asimetría de población y metodología"
    THINK: "Hay asimetría pero el clasificador coincide. Debo generar las transformaciones SQL"
    ACT: generate_transform("sample_censo", "sample_hospital")
    OBSERVE: "SQL CASE WHEN generado para sexo, nivel_educativo, diagnostico"
    THINK: "Tengo toda la información para responder"
    FINAL: "No recomendado por asimetría de contexto, pero generé las transformaciones..."
"""

import csv
import json
import logging
import re
from typing import TypedDict, Optional
from pathlib import Path

from langgraph.graph import StateGraph, END

from .llm_client import call_groq, call_groq_with_tools
from .graph.catalog import NomencladorGraph, load_graph_cached, clear_graph_cache
from .graph.schema import EdgeType
from .standards import detect_standard, STANDARDS, get_standard_values, list_standards
from .guardrails import validate_interoperability, CheckpointStatus
from .transformer import generate_transformation
from .verifier import (
    verify_classifier_consistency,
    verify_mapping_bijectivity,
    compute_interop_confidence,
    compute_transform_sql,
    verify_graph_invariants,
)
from .health import check_health, fix_orphan_nodes, retry_stuck_proposals, format_health_report
from .lifecycle import recall_feedback as lifecycle_recall_feedback

from semantic_tools.similarity import (
    cosine_similarity as st_cosine,
    jaccard_similarity as st_jaccard,
    overlap_coefficient as st_overlap,
    tfidf_similarity as st_tfidf,
    tfidf_similarity_batch as st_tfidf_batch,
    composite_similarity as st_composite,
)
from semantic_tools.statistics import (
    welch_t_test as st_welch_t_test,
    chi_square_test as st_chi_square_test,
    pearson_correlation as st_pearson,
    spearman_correlation as st_spearman,
    distribution_summary as st_distribution_summary,
)
from semantic_tools.clustering import (
    cluster_columns as st_cluster_columns,
    optimal_k as st_optimal_k,
    column_profile as st_column_profile,
)
from semantic_tools.quality import (
    detect_numeric_anomalies as st_detect_numeric_anomalies,
    detect_categorical_anomalies as st_detect_categorical_anomalies,
    auto_clean as st_auto_clean,
    column_quality_score as st_column_quality_score,
)
from semantic_tools.profiling import (
    profile_csv as st_profile_csv,
    format_profile_summary as st_format_profile,
)
from semantic_tools.schema_match import (
    auto_match as st_auto_match,
    format_match_result as st_format_match,
)

logger = logging.getLogger(__name__)

NOMENCLADOR_PATH = Path(__file__).parent.parent / "nomenclador" / "nomenclador.json"

SYSTEM_PROMPT = """Eres un agente de governance para interoperabilidad semantica.
Tu trabajo es ayudar a verificar, mapear y transformar variables entre fuentes de datos.

Tienes acceso a tools nativas (function calling). Usa las tools cuando necesites informacion
o ejecutar acciones. Cuando ya tengas toda la informacion, responde directamente al usuario
sin usar tools.

REGLAS:
- Una sola tool por turno.
- Siempre razona brevemente antes de actuar (en el campo content).
- Si los guardrails detectan asimetría, mencionalo explicitamente.
- Si no estás seguro, usa ask_human.
- Responde en español.
- Sé conciso pero completo.
- CALIDAD DE DATOS: Cuando veas quality_score de un campo, considera su confiabilidad:
  * score >= 0.7 = alta confianza, usar normalmente
  * score 0.4-0.7 = confianza media, mencionar advertencia al usuario
  * score < 0.4 = baja confianza, NO recomendar usar este campo para interoperabilidad sin limpieza previa
- REVIEW STATUS: Si un campo esta en "proposed" o "rejected", no usarlo para transformaciones.
- STALENESS: Si last_verified tiene mas de 180 dias, mencionar que el dato podria estar desactualizado.
- COMPUTE, DON'T GUESS: Prefiere tools deterministas (verify_field, verify_mapping, compute_confidence, compute_transform, audit_graph) sobre razonamiento probabilistico. Usa generate_transform solo cuando compute_transform no encuentre mapping en el grafo. Usa validate_interop para guardrails cualitativos, pero SIEMPRE complementa con compute_confidence para el score cuantitativo.
- SELF-MONITORING: Usa graph_health para diagnosticar el estado del grafo. Si detectas nodos huerfanos, usa fix_orphans. Si hay proposals stale, usa retry_proposals. Auto-sana lo que puedas, flaggea lo que requiera humano.
- AUTO-HEALING: Despues de ingest o nomenclar, considera ejecutar graph_health para verificar que el grafo quedo consistente. Si hay violaciones, intenta fix_orphans antes de reportar al humano.
- VALUE-LEVEL EQUIVALENCE: Para descubrir equivalencias semanticas entre datasets de formato largo (donde el significado esta en los valores, no en los nombres de columnas), usa sample_column_values para extraer valores unicos de una columna, y luego compare_value_sets para que el LLM razone sobre equivalencias entre dos conjuntos de valores. Para casos donde la combinacion de 2+ columnas (ej: Item+Element) equivale a una sola columna en otro dataset, usa compare_composite_values. Despues de descubrir equivalencias, usa persist_equivalences para guardarlas en el grafo como aristas EQUIVALE_A. Este es el unico modo de descubrir equivalencias no obvias que validate_interop no puede detectar.
- DETERMINISTIC SIMILARITY: Usa semantic_similarity para comparar dos columnas cuantitativamente (cosine, jaccard, overlap) sin gastar tokens de LLM. Usa text_similarity para comparar definiciones de conceptos via TF-IDF. Usa composite_similarity para combinar ambos (valores + texto) en un solo score. Estas tools son deterministas y gratuitas — usalas como pre-filtro antes de compare_value_sets (LLM) para descartar pares obvios.
- STATISTICAL ANALYSIS: Usa compare_distributions para comparar dos columnas numericas con Welch t-test (diferencia significativa de medias) o chi_square_test (distribucion categorica). Usa correlation para verificar si dos columnas estan correlacionadas (Pearson o Spearman). Usa distribution_summary para inspeccionar la distribucion de una columna (media, mediana, std, cuartiles, cardinalidad). Todas deterministas, sin LLM.
- COLUMN CLUSTERING: Usa cluster_columns para agrupar columnas de un dataset por perfil estructural (tipo, cardinalidad, missing, etc.). Usa optimal_k para encontrar el numero optimo de clusters via elbow method. Util para descubrir columnas similares dentro de un mismo dataset antes de comparar entre datasets.
- DATA QUALITY: Usa column_quality para detectar anomalias (outliers IQR/zscore en numericas, valores raros o artefactos de encoding en categoricas) y obtener un score de calidad 0-100. Usa auto_clean para limpiar whitespace, encoding mojibake, y fill de valores faltantes. Ejecuta column_quality ANTES de recomendar interoperabilidad con un campo de baja calidad.
- AUTONOMOUS PIPELINE: Para mapear dos datasets completos, usa profile_dataset para perfilar todas las columnas en un solo paso, luego auto_match para ejecutar similarity + statistics + quality en batch y obtener mapeos con score de confianza. Los mapeos de alta confianza (>=0.7) se persisten en el grafo automaticamente. Los de confianza media (0.4-0.7) se escalan a ask_human. Los bajos se descartan. NO uses tool por tool manualmente cuando auto_match puede hacer todo en un paso.
- PROFILE FIRST: Antes de cualquier comparacion entre datasets, usa profile_dataset para entender que tipo de datos tiene cada columna (numerica, categorica, fecha, ID, booleano, libre). Esto evita comparar columnas incompatibles.
- LEARNING LOOP: Si ves "DECISIONES PREVIAS RELEVANTES" en el contexto, considera ese feedback antes de proponer mapeos o crear conceptos. Si un humano rechazo algo similar antes, no repitas el mismo error. Usa recall_feedback para buscar mas contexto si lo necesitas.
- COMMUNITY DETECTION: Usa detect_communities para descubrir grupos de variables relacionadas estructuralmente (Louvain, sin LLM). Usa community_reports para generar resumenes narrativos de cada grupo con LLM. Usa global_search para responder preguntas transversales sobre todo el nomenclador (ej: "que areas tematicas cubre?", "hay brechas de cobertura?"). Estas tools emulan GraphRAG Global Search.
"""


# === STATE ===

class AgentState(TypedDict):
    messages: list  # plain list of dicts (NOT add_messages — that converts to LangChain objects)
    user_query: str
    scratchpad: list[str]  # historial de thoughts + observations
    current_thought: str
    current_action: str
    current_action_input: str
    tool_call_id: str  # ID del tool_call para feed-back nativo
    tool_result: str
    iteration: int
    final_answer: str
    needs_human_input: str  # pregunta al humano si necesita aclaración
    max_iterations: int  # circuit breaker — limite de iteraciones


# === TOOLS ===



def tool_search_graph(query: str) -> str:
    g = load_graph_cached()
    concept = g.find_concept(query)
    if not concept:
        concepts = g.list_concepts()
        names = [c.get("name", "") for c in concepts]
        return f"Variable '{query}' no encontrada. Conceptos disponibles: {', '.join(names)}"
    if concept.get("review_status") in ("proposed", "rejected"):
        status_label = "propuesta" if concept["review_status"] == "proposed" else "rechazada"
        return f"Variable '{query}' existe pero esta {status_label}. No se usa para interoperabilidad hasta ser aprobada."
    
    fields = g.find_fields_of_concept(concept["id"])
    lines = [f"Concepto: {concept.get('name', '')} | Estandar: {concept.get('standard', '-') or '-'}"]
    if fields:
        lines.append("Fuentes:")
        for f in fields:
            qs = f.get("quality_score", 0.0)
            cs = f.get("completeness", 0.0)
            rs = f.get("review_status", "approved")
            staleness = f.get("last_verified", "")
            quality_label = "alta" if qs >= 0.7 else ("media" if qs >= 0.4 else "baja")
            staleness_label = f" | verificado: {staleness}" if staleness else ""
            lines.append(
                f"  - {f.get('source_db', '?')}.{f.get('column', '?')}"
                f" | calidad: {quality_label} (score={qs}, compl={cs:.0%})"
                f" | review: {rs}{staleness_label}"
                f" | valores: {', '.join(f.get('sample_values', [])[:5])}"
            )
    return "\n".join(lines)


def tool_detect_standard(column_name: str, sample_values: list[str]) -> str:
    candidates = detect_standard(column_name, sample_values)
    if not candidates:
        return f"No se detecto estandar para '{column_name}' con valores {sample_values}"
    
    lines = []
    for c in candidates:
        lines.append(f"Estandar: {c['standard']} | Confianza: {c['confidence']} | Razon: {c['reason']}")
    return "\n".join(lines)


def tool_validate_interop(source_db: str, target_db: str) -> str:
    g = load_graph_cached()
    results = g.find_interoperability_path(source_db, target_db)
    if not results:
        return f"No se encontraron caminos entre {source_db} y {target_db}"
    
    lines = [f"{len(results)} camino(s):"]
    for i, result in enumerate(results, 1):
        field_a = result["field_a"]
        field_b = result["field_b"]
        concept = result["concept"]
        classifier = result.get("classifier")
        
        validation = validate_interoperability(field_a, field_b, concept, classifier)
        
        lines.append(f"\nCamino {i}: {concept.get('name', '?')}")
        lines.append(f"  {field_a.get('source_db', '')}.{field_a.get('column', '')} <-> {field_b.get('source_db', '')}.{field_b.get('column', '')}")
        # PMBOK Quality: mostrar quality_score de cada campo
        qa = field_a.get("quality_score", 0.0)
        qb = field_b.get("quality_score", 0.0)
        ca = field_a.get("completeness", 0.0)
        cb = field_b.get("completeness", 0.0)
        if qa or qb:
            expected_loss = round(1.0 - min(ca, cb), 3)
            lines.append(f"  [Calidad] {field_a.get('source_db','')}: score={qa} (compl={ca:.0%}) | {field_b.get('source_db','')}: score={qb} (compl={cb:.0%}) | pérdida esperada: {expected_loss:.0%}")
        for cp in validation.checkpoints:
            icon = {"match": "OK", "mismatch": "!!", "unknown": "??"}.get(cp.status.value, "??")
            lines.append(f"  [{icon}] {cp.name}: {cp.detail}")
        lines.append(f"  => {validation.recommendation}")
    return "\n".join(lines)


def tool_generate_transform(source_db: str, target_db: str) -> str:
    g = load_graph_cached()
    results = g.find_interoperability_path(source_db, target_db)
    if not results:
        return f"No se encontraron caminos entre {source_db} y {target_db}"
    
    lines = []
    for result in results:
        field_a = result["field_a"]
        field_b = result["field_b"]
        concept = result["concept"]
        classifier = result.get("classifier")
        
        validation = validate_interoperability(field_a, field_b, concept, classifier)
        artifact = generate_transformation(field_a, field_b, concept, classifier, validation)
        
        lines.append(f"=== {artifact.concept_name} ({artifact.standard}) ===")
        lines.append(f"SQL: {artifact.sql_transform[:200]}")
        # PMBOK Quality: mostrar assessment de calidad del cruce
        if artifact.quality_assessment:
            qa = artifact.quality_assessment
            lines.append(f"  Calidad: {qa.get('recommendation', 'N/A')} | pérdida esperada: {qa.get('expected_record_loss', 0):.0%}")
        if validation.warnings:
            lines.append(f"Warnings: {'; '.join(validation.warnings[:2])}")
    return "\n".join(lines)


def tool_list_concepts(**kwargs) -> str:
    g = load_graph_cached()
    concepts = g.list_concepts()
    if not concepts:
        return "Nomenclador vacio"
    return "Conceptos: " + ", ".join(c.get("name", "?") for c in concepts)


def tool_get_classifier(standard_id: str) -> str:
    std = STANDARDS.get(standard_id)
    if not std:
        available = [s["id"] for s in list_standards()]
        return f"Estandar '{standard_id}' no encontrado. Disponibles: {', '.join(available) if available else '(ninguno registrado)'}"
    std_type = std.get("standard_type", "classifier")
    if std_type == "format":
        return f"{std['name']} (formato): estandar de validacion de formato, no tiene valores enumerados. Regex: {std.get('regex', 'N/A')}"
    values = get_standard_values(standard_id)
    if values:
        return f"{std['name']} (clasificador): " + ", ".join(f"{k}={v}" for k, v in list(values.items())[:20])
    return f"{std['name']} (clasificador): sin valores cargados. Usa import_catalog para cargar."


def tool_get_concept_context(query: str) -> str:
    """Context assembly: retorna contexto completo de un concepto en una sola llamada."""
    g = load_graph_cached()
    concept = g.find_concept(query)
    if not concept:
        return f"Concepto '{query}' no encontrado."
    ctx = g.build_concept_context(concept["id"])
    if "error" in ctx:
        return ctx["error"]

    c = ctx["concept"]
    lines = [
        f"CONCEPTO: {c['name']} | Estandar: {c.get('standard') or '-'}",
        f"  Definicion: {c['definition'][:120]}..." if len(c["definition"]) > 120 else f"  Definicion: {c['definition']}",
        f"  Poblacion: {c['population']} | Captura: {c['capture_method']}",
        f"  Custodio: {c['custodian']} | Review: {c['review_status']} | Verificado: {c.get('last_verified') or 'nunca'}",
    ]

    qs = ctx.get("quality_summary", {})
    lines.append(f"  Calidad: avg={qs.get('avg_score', 0)} | campos_baja_calidad={qs.get('low_quality_count', 0)}/{qs.get('total_fields', 0)}")

    if ctx["fields"]:
        lines.append("  CAMPOS:")
        for f in ctx["fields"]:
            ql = "alta" if f["quality_score"] >= 0.7 else ("media" if f["quality_score"] >= 0.4 else "baja")
            issues_str = f" | issues: {len(f.get('issues', []))}" if f.get("issues") else ""
            lines.append(f"    - {f['source_db']}.{f['column']} | calidad={ql}({f['quality_score']}) | review={f['review_status']}{issues_str}")

    if ctx["classifier"]:
        cl = ctx["classifier"]
        lines.append(f"  CLASIFICADOR: {cl['name']} | version={cl.get('version_label') or '-'} | vigente={cl.get('is_current')}")

    if ctx["normatives"]:
        lines.append("  NORMATIVAS:")
        for n in ctx["normatives"]:
            lines.append(f"    - {n['title']} | {n.get('source', '')} {n.get('article', '')}")

    if ctx["context_conflicts"]:
        lines.append("  CONFLICTOS DE CONTEXTO:")
        for cc in ctx["context_conflicts"]:
            lines.append(f"    - {cc['source_db']}: {cc['meaning'][:80]}")

    if ctx["composites"]:
        lines.append(f"  COMPONE: {', '.join(c.get('name', c.get('id', '')) for c in ctx['composites'])}")

    if ctx["derived_from"]:
        lines.append(f"  DERIVA DE: {', '.join(d['name'] for d in ctx['derived_from'])}")

    return "\n".join(lines)


def tool_verify_field(field_id: str) -> str:
    """Verificacion determinista: sample_values del field vs clasificador del concepto."""
    g = load_graph_cached()
    result = verify_classifier_consistency(g, field_id)
    if "error" in result:
        return result["error"]
    status = "OK" if result["passed"] else "FAIL"
    lines = [
        f"[{status}] Field '{field_id}' vs clasificador '{result.get('classifier_id', '?')}'",
        f"  Match ratio: {result['match_ratio']:.0%} ({result['valid_count']}/{result['valid_count'] + result['invalid_count']})",
    ]
    if result["invalid_values"]:
        lines.append(f"  Valores invalidos: {', '.join(result['invalid_values'][:10])}")
    return "\n".join(lines)


def tool_verify_mapping(classifier_a: str, classifier_b: str) -> str:
    """Verificacion determinista: biyectividad del mapping entre clasificadores."""
    g = load_graph_cached()
    result = verify_mapping_bijectivity(g, classifier_a, classifier_b)
    if "error" in result:
        return result["error"]
    status = "BIJECTIVE" if result["is_bijective"] else "NOT BIJECTIVE"
    lines = [
        f"[{status}] {classifier_a} <-> {classifier_b}",
        f"  Cardinalidad declarada: {result.get('declared_cardinality', '?')}",
    ]
    if result.get("conflicts"):
        for c in result["conflicts"]:
            lines.append(f"  Conflicto: {c}")
    return "\n".join(lines)


def tool_compute_confidence(concept_name: str) -> str:
    """Confidence score determinista para todos los fields de un concepto."""
    g = load_graph_cached()
    concept = g.find_concept(concept_name)
    if not concept:
        return f"Concepto '{concept_name}' no encontrado."
    fields = g.find_fields_of_concept(concept["id"])
    if not fields:
        return f"Concepto '{concept_name}' sin fields."
    lines = [f"Confidence para '{concept_name}':"]
    for f in fields:
        result = compute_interop_confidence(g, concept["id"], f["id"])
        level = result["level"]
        conf = result["confidence"]
        icon = {"high": "OK", "medium": "..", "low": "!!", "blocked": "XX"}.get(level, "??")
        lines.append(
            f"  [{icon}] {f.get('source_db', '?')}.{f.get('column', '?')}"
            f" | confidence={conf} ({level})"
            f" | q={result['quality_score']} r={result['review_factor']}"
            f" s={result['staleness_factor']} c={result['classifier_match']}"
        )
        if result.get("reasons"):
            for r in result["reasons"]:
                lines.append(f"       - {r}")
    return "\n".join(lines)


def tool_compute_transform(source_db: str, target_db: str) -> str:
    """Generar SQL CASE WHEN determinista desde el mapping del grafo (sin LLM)."""
    g = load_graph_cached()
    paths = g.find_interoperability_path(source_db, target_db)
    if not paths:
        return f"No se encontraron caminos entre {source_db} y {target_db}."
    lines = []
    for p in paths:
        field_a = p["field_a"]
        field_b = p["field_b"]
        concept = p["concept"]
        result = compute_transform_sql(g, field_a["id"], concept["id"], field_b["id"])
        if "error" in result:
            lines.append(f"=== {concept.get('name', '?')} === ERROR: {result['error']}")
            continue
        lines.append(f"=== {concept.get('name', '?')} (fuente: {result['source']}) ===")
        lines.append(result["sql"])
        lines.append("")
    return "\n".join(lines)


def tool_audit_graph(**kwargs) -> str:
    """Audit determinista de invariantes del grafo."""
    g = load_graph_cached()
    result = verify_graph_invariants(g)
    status = "PASS" if result["passed"] else "FAIL"
    lines = [
        f"[{status}] Graph audit: {result['total_nodes']} nodos, {result['total_edges']} edges",
        f"  Violations: {len(result['violations'])}",
        f"  Warnings: {len(result['warnings'])}",
    ]
    for v in result["violations"][:10]:
        lines.append(f"  VIOLATION: {v['message']}")
    for w in result["warnings"][:10]:
        lines.append(f"  WARNING: {w['message']}")
    return "\n".join(lines)


def tool_graph_health(**kwargs) -> str:
    """Diagnostico completo del estado del governance-agent."""
    report = check_health()
    return format_health_report(report)


def tool_fix_orphans(dry_run: bool = True) -> str:
    """Auto-healing: linkear nodos huerfanos a conceptos existentes."""
    result = fix_orphan_nodes(dry_run=dry_run)
    lines = [f"Fix orphans (dry_run={dry_run}):"]
    if dry_run:
        lines.append(f"  Suggested fixes: {len(result['fixes_suggested'])}")
        for f in result["fixes_suggested"][:5]:
            lines.append(f"    - {f['node_name']} -> {f['suggested_concept']} (score={f['match_score']})")
    else:
        lines.append(f"  Applied fixes: {len(result['fixes_applied'])}")
        for f in result["fixes_applied"][:5]:
            lines.append(f"    - {f['node_name']} -> {f['suggested_concept']} (score={f['match_score']})")
    lines.append(f"  Manual needed: {len(result['manual_needed'])}")
    for m in result["manual_needed"][:5]:
        lines.append(f"    - {m['node_name']}: {m['reason']}")
    return "\n".join(lines)


def tool_retry_proposals(dry_run: bool = True) -> str:
    """Auto-healing: re-intentar nodos proposed stale."""
    result = retry_stuck_proposals(dry_run=dry_run)
    lines = [f"Retry proposals (dry_run={dry_run}):"]
    lines.append(f"  Auto-approved: {len(result['auto_approved'])}")
    for a in result["auto_approved"][:5]:
        lines.append(f"    - {a['node_name']} (qs={a['quality_score']}, {a['days_stale']}d stale)")
    lines.append(f"  Flagged manual: {len(result['flagged_manual'])}")
    for f in result["flagged_manual"][:5]:
        lines.append(f"    - {f['node_name']} (qs={f['quality_score']}): {f['reason']}")
    lines.append(f"  Alerts: {len(result['alerts'])}")
    for a in result["alerts"][:5]:
        lines.append(f"    - {a['node_name']}: {a['reason']}")
    return "\n".join(lines)


def _find_csv_for_source_db(source_db: str) -> Optional[Path]:
    """Encontrar el archivo CSV correspondiente a un source_db."""
    search_dirs = [
        Path(__file__).parent.parent / "datasets" / "real",
        Path(__file__).parent.parent / "demo",
        Path(__file__).parent.parent / "tests",
    ]
    for d in search_dirs:
        candidate = d / f"{source_db}.csv"
        if candidate.exists():
            return candidate
    return None


def _extract_unique_values(csv_path: Path, column_name: str, max_values: int = 50) -> tuple[list[str], Optional[list[str]]]:
    """Extraer valores unicos de una columna de un CSV.

    Returns (values, None) on success, ([], headers) if column not found.
    """
    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        matched_col = column_name
        if column_name not in headers:
            col_norm = column_name.lower().replace(" ", "_")
            for h in headers:
                if h.lower().replace(" ", "_") == col_norm:
                    matched_col = h
                    break
            if matched_col == column_name:
                return [], headers
        unique = set()
        for row in reader:
            val = row.get(matched_col, "")
            if val and val.strip() and val.strip().lower() not in ("", "na", "n/a", "null", "none"):
                unique.add(val.strip())
            if len(unique) >= max_values:
                break
    return sorted(unique)[:max_values], None


def tool_sample_column_values(source_db: str, column_name: str) -> str:
    """Extraer valores unicos de una columna de un dataset CSV."""
    csv_path = _find_csv_for_source_db(source_db)
    if not csv_path:
        return f"No se encontro CSV para source_db='{source_db}'. Buscado en datasets/real, demo, tests."

    values, err_headers = _extract_unique_values(csv_path, column_name)
    if err_headers is not None:
        return f"Columna '{column_name}' no encontrada en {csv_path.name}. Columnas disponibles: {', '.join(err_headers)}"

    lines = [
        f"Dataset: {source_db} ({csv_path.name})",
        f"Columna: {column_name}",
        f"Valores unicos: {len(values)}",
        "",
    ]
    for v in values:
        lines.append(f"  - {v}")
    return "\n".join(lines)


def tool_compare_value_sets(source_db_a: str, column_a: str, source_db_b: str, column_b: str) -> str:
    """Comparar dos conjuntos de valores y descubrir equivalencias semanticas via LLM."""
    csv_a = _find_csv_for_source_db(source_db_a)
    csv_b = _find_csv_for_source_db(source_db_b)
    if not csv_a:
        return f"No se encontro CSV para '{source_db_a}'"
    if not csv_b:
        return f"No se encontro CSV para '{source_db_b}'"

    vals_a, err_a = _extract_unique_values(csv_a, column_a)
    if err_a is not None:
        return f"Columna '{column_a}' no encontrada en {source_db_a}. Columnas: {', '.join(err_a)}"
    vals_b, err_b = _extract_unique_values(csv_b, column_b)
    if err_b is not None:
        return f"Columna '{column_b}' no encontrada en {source_db_b}. Columnas: {', '.join(err_b)}"

    if not vals_a or not vals_b:
        return f"Uno o ambos conjuntos de valores estan vacios. A={len(vals_a)}, B={len(vals_b)}"

    values_a_text = "\n".join(f"- {v}" for v in vals_a)
    values_b_text = "\n".join(f"- {v}" for v in vals_b)

    prompt = (
        "Eres un experto en interoperabilidad semantica de datos. "
        "Dados dos conjuntos de valores de dos datasets distintos, encuentra equivalencias semanticas.\n\n"
        f"DATASET A ({source_db_a}, columna: {column_a}):\n{values_a_text}\n\n"
        f"DATASET B ({source_db_b}, columna: {column_b}):\n{values_b_text}\n\n"
        "Identifica:\n"
        "1. Equivalencias directas (mismo significado, diferente nombre)\n"
        "2. Equivalencias parciales (uno es subconjunto o superconjunto)\n"
        "3. Equivalencias compuestas (combinacion de valores de A equivale a un valor de B)\n"
        "4. Valores sin equivalencia\n\n"
        'Responde en JSON: {"equivalences": [{"a": "valor_a", "b": "valor_b", '
        '"type": "directa|parcial|compuesta", "confidence": "alta|media|baja", '
        '"reason": "explicacion"}], "no_match_a": ["..."], "no_match_b": ["..."]}'
    )

    try:
        response = call_groq(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=3000,
            json_mode=True,
        )
        result = json.loads(response)
        lines = [f"Equivalencias semanticas entre {source_db_a}.{column_a} y {source_db_b}.{column_b}:", ""]
        for eq in result.get("equivalences", []):
            icon = {"alta": "OK", "media": "..", "baja": "??"}.get(eq.get("confidence", ""), "??")
            lines.append(f"  [{icon}] {eq.get('a', '?')} <-> {eq.get('b', '?')} ({eq.get('type', '?')})")
            lines.append(f"       {eq.get('reason', '')}")
        if result.get("no_match_a"):
            lines.append(f"\n  Sin equivalencia en A: {', '.join(result['no_match_a'][:10])}")
        if result.get("no_match_b"):
            lines.append(f"  Sin equivalencia en B: {', '.join(result['no_match_b'][:10])}")
        return "\n".join(lines)
    except Exception as e:
        return (
            f"Error en LLM: {e}\n\n"
            f"Valores A ({source_db_a}.{column_a}): {', '.join(vals_a[:20])}\n"
            f"Valores B ({source_db_b}.{column_b}): {', '.join(vals_b[:20])}\n"
            "Puedes razonar sobre estas equivalencias tu mismo."
        )


def _extract_composite_values(csv_path: Path, column_names: list[str], max_values: int = 50) -> tuple[list[str], Optional[list[str]]]:
    """Extraer valores compuestos concatenando multiples columnas de un CSV.

    Returns (composite_values, None) on success, ([], headers) if any column not found.
    """
    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        matched_cols = []
        for col_name in column_names:
            matched = col_name
            if col_name not in headers:
                col_norm = col_name.lower().replace(" ", "_")
                for h in headers:
                    if h.lower().replace(" ", "_") == col_norm:
                        matched = h
                        break
                if matched == col_name:
                    return [], headers
            matched_cols.append(matched)
        unique = set()
        for row in reader:
            parts = []
            for mc in matched_cols:
                val = row.get(mc, "")
                if val and val.strip() and val.strip().lower() not in ("", "na", "n/a", "null", "none"):
                    parts.append(val.strip())
            if parts:
                unique.add("|".join(parts))
            if len(unique) >= max_values:
                break
    return sorted(unique)[:max_values], None


def tool_compare_composite_values(
    source_db_a: str, columns_a: str, source_db_b: str, column_b: str
) -> str:
    """Comparar valores compuestos (multi-columna) de un dataset vs una columna de otro.

    Usa para formato largo donde la combinacion de 2+ columnas (ej: Item+Element)
    equivale semanticamente a una sola columna en otro dataset (ej: Indicator Name).
    """
    csv_a = _find_csv_for_source_db(source_db_a)
    csv_b = _find_csv_for_source_db(source_db_b)
    if not csv_a:
        return f"No se encontro CSV para '{source_db_a}'"
    if not csv_b:
        return f"No se encontro CSV para '{source_db_b}'"

    col_list = [c.strip() for c in columns_a.split(",") if c.strip()]
    if len(col_list) < 2:
        return f"columns_a debe tener al menos 2 columnas separadas por coma. Recibido: '{columns_a}'"

    vals_a, err_a = _extract_composite_values(csv_a, col_list)
    if err_a is not None:
        return f"Columna no encontrada en {source_db_a}. Columnas disponibles: {', '.join(err_a)}"
    vals_b, err_b = _extract_unique_values(csv_b, column_b)
    if err_b is not None:
        return f"Columna '{column_b}' no encontrada en {source_db_b}. Columnas: {', '.join(err_b)}"

    if not vals_a or not vals_b:
        return f"Uno o ambos conjuntos de valores estan vacios. A={len(vals_a)}, B={len(vals_b)}"

    values_a_text = "\n".join(f"- {v}" for v in vals_a)
    values_b_text = "\n".join(f"- {v}" for v in vals_b)
    cols_a_label = " + ".join(col_list)

    prompt = (
        "Eres un experto en interoperabilidad semantica de datos. "
        "Dados dos conjuntos de valores de dos datasets distintos, encuentra equivalencias semanticas.\n\n"
        f"DATASET A ({source_db_a}, columnas compuestas: {cols_a_label}):\n"
        "Los valores estan concatenados con '|'. Cada valor es una combinacion de las columnas.\n"
        f"{values_a_text}\n\n"
        f"DATASET B ({source_db_b}, columna: {column_b}):\n{values_b_text}\n\n"
        "Identifica:\n"
        "1. Equivalencias directas (mismo significado, diferente nombre)\n"
        "2. Equivalencias parciales (uno es subconjunto o superconjunto)\n"
        "3. Equivalencias compuestas (combinacion de valores de A equivale a un valor de B)\n"
        "4. Valores sin equivalencia\n\n"
        'Responde en JSON: {"equivalences": [{"a": "valor_a", "b": "valor_b", '
        '"type": "directa|parcial|compuesta", "confidence": "alta|media|baja", '
        '"reason": "explicacion"}], "no_match_a": ["..."], "no_match_b": ["..."]}'
    )

    try:
        response = call_groq(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=3000,
            json_mode=True,
        )
        result = json.loads(response)
        lines = [f"Equivalencias semanticas entre {source_db_a}.{cols_a_label} y {source_db_b}.{column_b}:", ""]
        for eq in result.get("equivalences", []):
            icon = {"alta": "OK", "media": "..", "baja": "??"}.get(eq.get("confidence", ""), "??")
            lines.append(f"  [{icon}] {eq.get('a', '?')} <-> {eq.get('b', '?')} ({eq.get('type', '?')})")
            lines.append(f"       {eq.get('reason', '')}")
        if result.get("no_match_a"):
            lines.append(f"\n  Sin equivalencia en A: {', '.join(result['no_match_a'][:10])}")
        if result.get("no_match_b"):
            lines.append(f"  Sin equivalencia en B: {', '.join(result['no_match_b'][:10])}")
        return "\n".join(lines)
    except Exception as e:
        return (
            f"Error en LLM: {e}\n\n"
            f"Valores A ({source_db_a}.{cols_a_label}): {', '.join(vals_a[:20])}\n"
            f"Valores B ({source_db_b}.{column_b}): {', '.join(vals_b[:20])}\n"
            "Puedes razonar sobre estas equivalencias tu mismo."
        )


def tool_persist_equivalences(
    source_db_a: str, column_a: str, source_db_b: str, column_b: str, equivalences_json: str
) -> str:
    """Persistir equivalencias descubiertas en el grafo como aristas EQUIVALE_A."""
    g = load_graph_cached()

    field_a_id = None
    field_b_id = None
    for node_id, data in g.graph.nodes(data=True):
        if data.get("type") == "field":
            if data.get("source_db") == source_db_a and data.get("column") == column_a:
                field_a_id = node_id
            if data.get("source_db") == source_db_b and data.get("column") == column_b:
                field_b_id = node_id

    if not field_a_id:
        return f"No se encontro field para {source_db_a}.{column_a} en el grafo. Ingesta el dataset primero."
    if not field_b_id:
        return f"No se encontro field para {source_db_b}.{column_b} en el grafo. Ingesta el dataset primero."

    try:
        equivalences = json.loads(equivalences_json)
    except json.JSONDecodeError as e:
        return f"JSON invalido en equivalences_json: {e}"

    eq_list = equivalences if isinstance(equivalences, list) else equivalences.get("equivalences", [])
    if not eq_list:
        return "No hay equivalencias para persistir."

    mapping = {}
    confidence_map = {}
    for eq in eq_list:
        a_val = eq.get("a", "")
        b_val = eq.get("b", "")
        conf = eq.get("confidence", "baja")
        if a_val and b_val:
            mapping[a_val] = b_val
            confidence_map[a_val] = conf

    g.graph.add_edge(
        field_a_id, field_b_id,
        type=EdgeType.EQUIVALE_A.value,
        mapping=mapping,
        confidence_map=confidence_map,
        source="value_level_discovery",
    )
    g._db_upsert_edge(
        field_a_id, field_b_id,
        EdgeType.EQUIVALE_A.value,
        {"mapping": mapping, "confidence_map": confidence_map, "source": "value_level_discovery"},
    )

    from .graph.catalog import _NOMENCLADOR_PATH
    g.save(str(_NOMENCLADOR_PATH))
    clear_graph_cache()

    high = sum(1 for c in confidence_map.values() if c == "alta")
    med = sum(1 for c in confidence_map.values() if c == "media")
    low = sum(1 for c in confidence_map.values() if c == "baja")

    try:
        from .lifecycle import log_event
        log_event(field_a_id, "equivalence_created", actor="agent",
                  reason=f"Equivalencia descubierta con {field_b_id} via value-level discovery",
                  details=f"mapping_size={len(mapping)}, high={high}, med={med}, low={low}")
    except Exception:
        pass

    return (
        f"Equivalencias persistidas en el grafo: {field_a_id} -> {field_b_id}\n"
        f"  Total mapeos: {len(mapping)}\n"
        f"  Alta confianza: {high} | Media: {med} | Baja: {low}\n"
        f"  Arista EQUIVALE_A creada con source=value_level_discovery.\n"
        f"  Grafo guardado y cache invalidada."
    )


def tool_semantic_similarity(source_db_a: str, column_a: str, source_db_b: str, column_b: str) -> str:
    """Similitud determinista entre dos columnas (cosine, jaccard, overlap). Sin LLM."""
    csv_a = _find_csv_for_source_db(source_db_a)
    csv_b = _find_csv_for_source_db(source_db_b)
    if not csv_a:
        return f"No se encontro CSV para '{source_db_a}'"
    if not csv_b:
        return f"No se encontro CSV para '{source_db_b}'"

    vals_a, err_a = _extract_unique_values(csv_a, column_a)
    if err_a is not None:
        return f"Columna '{column_a}' no encontrada en {source_db_a}. Columnas: {', '.join(err_a)}"
    vals_b, err_b = _extract_unique_values(csv_b, column_b)
    if err_b is not None:
        return f"Columna '{column_b}' no encontrada en {source_db_b}. Columnas: {', '.join(err_b)}"

    if not vals_a or not vals_b:
        return f"Conjuntos vacios. A={len(vals_a)}, B={len(vals_b)}"

    cos = st_cosine(vals_a, vals_b)
    jac = st_jaccard(vals_a, vals_b)
    ovl = st_overlap(vals_a, vals_b)

    verdict = "alto" if cos >= 0.7 else ("medio" if cos >= 0.4 else "bajo")

    lines = [
        f"Similitud determinista: {source_db_a}.{column_a} vs {source_db_b}.{column_b}",
        f"  Valores A: {len(vals_a)} | Valores B: {len(vals_b)}",
        f"  Cosine: {cos:.4f} (distribucion)",
        f"  Jaccard: {jac:.4f} (overlap de conjuntos)",
        f"  Overlap coeff: {ovl:.4f} (subconjunto)",
        f"  Verdict: {verdict}",
    ]
    if cos >= 0.7:
        lines.append("  => Alta similitud. Probablemente equivalentes.")
    elif cos >= 0.4:
        lines.append("  => Similitud media. Revisar con compare_value_sets o LLM.")
    else:
        lines.append("  => Baja similitud. Probablemente no equivalentes.")
    return "\n".join(lines)


def tool_text_similarity(concept_a: str, concept_b: str) -> str:
    """Similitud TF-IDF entre definiciones de dos conceptos. Sin LLM."""
    g = load_graph_cached()
    c_a = g.find_concept(concept_a)
    c_b = g.find_concept(concept_b)
    if not c_a:
        return f"Concepto '{concept_a}' no encontrado."
    if not c_b:
        return f"Concepto '{concept_b}' no encontrado."

    text_a = c_a.get("definition", "") or c_a.get("name", "")
    text_b = c_b.get("definition", "") or c_b.get("name", "")

    if not text_a or not text_b:
        return f"Definiciones vacias. A='{text_a[:50]}', B='{text_b[:50]}'"

    score = st_tfidf(text_a, text_b)
    verdict = "alto" if score >= 0.6 else ("medio" if score >= 0.3 else "bajo")

    lines = [
        f"Similitud TF-IDF: '{c_a.get('name', concept_a)}' vs '{c_b.get('name', concept_b)}'",
        f"  Score: {score:.4f}",
        f"  Verdict: {verdict}",
    ]
    if score >= 0.6:
        lines.append("  => Definiciones muy similares. Probablemente el mismo concepto.")
    elif score >= 0.3:
        lines.append("  => Similitud parcial. Podrian estar relacionados.")
    else:
        lines.append("  => Definiciones distintas. Probablemente conceptos diferentes.")
    return "\n".join(lines)


def tool_composite_similarity(source_db_a: str, column_a: str, source_db_b: str, column_b: str) -> str:
    """Similitud compuesta (valores + definiciones) entre dos campos. Sin LLM."""
    csv_a = _find_csv_for_source_db(source_db_a)
    csv_b = _find_csv_for_source_db(source_db_b)
    if not csv_a:
        return f"No se encontro CSV para '{source_db_a}'"
    if not csv_b:
        return f"No se encontro CSV para '{source_db_b}'"

    vals_a, err_a = _extract_unique_values(csv_a, column_a)
    if err_a is not None:
        return f"Columna '{column_a}' no encontrada en {source_db_a}. Columnas: {', '.join(err_a)}"
    vals_b, err_b = _extract_unique_values(csv_b, column_b)
    if err_b is not None:
        return f"Columna '{column_b}' no encontrada en {source_db_b}. Columnas: {', '.join(err_b)}"

    g = load_graph_cached()
    text_a = ""
    text_b = ""
    for node_id, data in g.graph.nodes(data=True):
        if data.get("type") == "field":
            if data.get("source_db") == source_db_a and data.get("column") == column_a:
                for succ in g.graph.successors(node_id):
                    edge = g.graph.get_edge_data(node_id, succ)
                    if edge and edge.get("type") == "implementa":
                        concept = g.graph.nodes.get(succ, {})
                        text_a = concept.get("definition", "") or concept.get("name", "")
            if data.get("source_db") == source_db_b and data.get("column") == column_b:
                for succ in g.graph.successors(node_id):
                    edge = g.graph.get_edge_data(node_id, succ)
                    if edge and edge.get("type") == "implementa":
                        concept = g.graph.nodes.get(succ, {})
                        text_b = concept.get("definition", "") or concept.get("name", "")

    result = st_composite(
        values_a=vals_a,
        values_b=vals_b,
        text_a=text_a,
        text_b=text_b,
    )

    lines = [
        f"Similitud compuesta: {source_db_a}.{column_a} vs {source_db_b}.{column_b}",
        f"  Cosine: {result['cosine']:.4f}",
        f"  Jaccard: {result['jaccard']:.4f}",
        f"  Overlap: {result['overlap']:.4f}",
        f"  TF-IDF: {result['tfidf']:.4f}" + (" (sin definicion disponible)" if result['tfidf'] == 0.0 else ""),
        f"  Composite: {result['composite']:.4f}",
        f"  Verdict: {result['verdict']}",
    ]
    if result['verdict'] in ('high', 'medium'):
        lines.append(f"  => Confirmar con compare_value_sets (LLM) para equivalencias a nivel de valor.")
    else:
        lines.append(f"  => Baja similitud. Probablemente no equivalentes.")
    return "\n".join(lines)


def _extract_all_values(csv_path: Path, column_name: str, max_values: int = 500) -> tuple[list, Optional[list[str]]]:
    """Extraer todos los valores (no solo unicos) de una columna de un CSV.

    Returns (values, None) on success, ([], headers) if column not found.
    """
    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        matched_col = column_name
        if column_name not in headers:
            col_norm = column_name.lower().replace(" ", "_")
            for h in headers:
                if h.lower().replace(" ", "_") == col_norm:
                    matched_col = h
                    break
            if matched_col == column_name:
                return [], headers
        values = []
        for row in reader:
            val = row.get(matched_col, "")
            if val and val.strip() and val.strip().lower() not in ("", "na", "n/a", "null", "none"):
                values.append(val.strip())
            if len(values) >= max_values:
                break
    return values, None


def tool_compare_distributions(source_db_a: str, column_a: str, source_db_b: str, column_b: str, test_type: str = "auto") -> str:
    """Compara dos columnas con pruebas estadisticas (Welch t-test o chi-square). Sin LLM."""
    csv_a = _find_csv_for_source_db(source_db_a)
    csv_b = _find_csv_for_source_db(source_db_b)
    if not csv_a:
        return f"No se encontro CSV para '{source_db_a}'"
    if not csv_b:
        return f"No se encontro CSV para '{source_db_b}'"

    vals_a, err_a = _extract_all_values(csv_a, column_a)
    if err_a is not None:
        return f"Columna '{column_a}' no encontrada en {source_db_a}. Columnas: {', '.join(err_a)}"
    vals_b, err_b = _extract_all_values(csv_b, column_b)
    if err_b is not None:
        return f"Columna '{column_b}' no encontrada en {source_db_b}. Columnas: {', '.join(err_b)}"

    if not vals_a or not vals_b:
        return f"Conjuntos vacios. A={len(vals_a)}, B={len(vals_b)}"

    # Auto-detect: if both columns are mostly numeric, use t-test; otherwise chi-square
    from semantic_tools.statistics import _to_float_list
    num_a = _to_float_list(vals_a)
    num_b = _to_float_list(vals_b)
    is_numeric = len(num_a) > len(vals_a) * 0.8 and len(num_b) > len(vals_b) * 0.8

    if test_type == "auto":
        test_type = "ttest" if is_numeric else "chi2"

    lines = [
        f"Comparacion estadistica: {source_db_a}.{column_a} vs {source_db_b}.{column_b}",
        f"  Valores A: {len(vals_a)} | Valores B: {len(vals_b)}",
        f"  Tipo detectado: {'numerico' if is_numeric else 'categorico'}",
        f"  Test: {test_type}",
    ]

    if test_type == "ttest":
        if not is_numeric:
            return "Welch t-test requiere columnas numericas. Usa test_type='chi2' para datos categoricos."
        result = st_welch_t_test(num_a, num_b)
        lines.extend([
            f"  Media A: {result['mean_a']:.4f} | Media B: {result['mean_b']:.4f}",
            f"  Std A: {result['std_a']:.4f} | Std B: {result['std_b']:.4f}",
            f"  t-statistic: {result['t_stat']:.4f}",
            f"  p-value: {result['p_value']:.6f}",
            f"  Diferencia significativa: {'SI' if result['significant'] else 'NO'} (alpha=0.05)",
        ])
        if result['significant']:
            lines.append("  => Las medias son estadisticamente diferentes. Las columnas NO son equivalentes.")
        else:
            lines.append("  => No hay diferencia significativa. Las distribuciones podrian ser equivalentes.")

    elif test_type == "chi2":
        result = st_chi_square_test(vals_a, vals_b)
        lines.extend([
            f"  Chi-square: {result['chi2']:.4f}",
            f"  p-value: {result['p_value']:.6f}",
            f"  Diferencia significativa: {'SI' if result['significant'] else 'NO'} (alpha=0.05)",
        ])
        if result['significant']:
            lines.append("  => Las distribuciones categoricas son diferentes. Las columnas NO son equivalentes.")
        else:
            lines.append("  => No hay diferencia significativa. Las distribuciones podrian ser equivalentes.")

    else:
        return f"test_type invalido: '{test_type}'. Usa 'auto', 'ttest' o 'chi2'."

    return "\n".join(lines)


def tool_correlation(source_db: str, column_a: str, column_b: str, method: str = "pearson") -> str:
    """Calcula correlacion entre dos columnas del mismo dataset. Sin LLM."""
    csv_path = _find_csv_for_source_db(source_db)
    if not csv_path:
        return f"No se encontro CSV para '{source_db}'"

    vals_a, err_a = _extract_all_values(csv_path, column_a)
    if err_a is not None:
        return f"Columna '{column_a}' no encontrada en {source_db}. Columnas: {', '.join(err_a)}"
    vals_b, err_b = _extract_all_values(csv_path, column_b)
    if err_b is not None:
        return f"Columna '{column_b}' no encontrada en {source_db}. Columnas: {', '.join(err_b)}"

    # Truncate to same length
    n = min(len(vals_a), len(vals_b))
    if n < 3:
        return f"Insuficientes valores para correlacion (n={n}). Necesario >= 3."

    from semantic_tools.statistics import _to_float_list
    num_a = _to_float_list(vals_a[:n])
    num_b = _to_float_list(vals_b[:n])

    if len(num_a) < n * 0.8 or len(num_b) < n * 0.8:
        return f"Una o ambas columnas no son suficientemente numericas (A: {len(num_a)}/{n}, B: {len(num_b)}/{n})."

    # Align by index (only pairs where both are numeric)
    pairs = [(a, b) for a, b in zip(num_a, num_b) if a is not None and b is not None]
    if len(pairs) < 3:
        return f"Insuficientes pares numericos validos ({len(pairs)}). Necesario >= 3."

    clean_a = [p[0] for p in pairs]
    clean_b = [p[1] for p in pairs]

    if method == "pearson":
        result = st_pearson(clean_a, clean_b)
    elif method == "spearman":
        result = st_spearman(clean_a, clean_b)
    else:
        return f"method invalido: '{method}'. Usa 'pearson' o 'spearman'."

    lines = [
        f"Correlacion {method}: {source_db}.{column_a} vs {source_db}.{column_b}",
        f"  Pares validos: {len(pairs)}/{n}",
        f"  Coeficiente r: {result['r']:.4f}",
        f"  Interpretacion: {result['verdict']}",
    ]
    abs_r = abs(result['r'])
    if abs_r >= 0.7:
        lines.append("  => Correlacion fuerte. Las variables estan estrechamente relacionadas.")
    elif abs_r >= 0.4:
        lines.append("  => Correlacion moderada. Relacion parcial.")
    else:
        lines.append("  => Correlacion debil o nula. Las variables son independientes.")

    return "\n".join(lines)


def tool_distribution_summary(source_db: str, column_name: str) -> str:
    """Resume la distribucion de una columna (media, mediana, std, cuartiles, cardinalidad). Sin LLM."""
    csv_path = _find_csv_for_source_db(source_db)
    if not csv_path:
        return f"No se encontro CSV para '{source_db}'"

    vals, err = _extract_all_values(csv_path, column_name)
    if err is not None:
        return f"Columna '{column_name}' no encontrada en {source_db}. Columnas: {', '.join(err)}"

    if not vals:
        return f"Columna '{column_name}' esta vacia en {source_db}."

    result = st_distribution_summary(vals)
    lines = [
        f"Distribucion: {source_db}.{column_name}",
        f"  Total: {result['count']} | Unicos: {result['unique']} | Missing: {result['missing']}",
    ]

    if result["type"] == "numeric":
        lines.extend([
            f"  Tipo: numerico",
            f"  Media: {result['mean']:.4f} | Mediana: {result['median']:.4f}",
            f"  Std: {result['std']:.4f} | Min: {result['min']:.4f} | Max: {result['max']:.4f}",
            f"  Q1: {result['q1']:.4f} | Q3: {result['q3']:.4f}",
        ])
    else:
        top_vals = ', '.join(f'{v}({c})' for v, c in list(result.get('top_5', {}).items())[:5])
        lines.extend([
            f"  Tipo: categorico",
            f"  Cardinalidad: {result['unique']}",
            f"  Top valores: {top_vals}",
        ])

    return "\n".join(lines)


def tool_cluster_columns(source_db: str, k: str = "auto") -> str:
    """Agrupa columnas de un dataset por perfil estructural (k-means). Sin LLM."""
    csv_path = _find_csv_for_source_db(source_db)
    if not csv_path:
        return f"No se encontro CSV para '{source_db}'"

    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        if not headers:
            return f"CSV '{source_db}' no tiene columnas."
        rows = list(reader)

    # Build column -> values mapping
    columns = {}
    for h in headers:
        columns[h] = [row.get(h, "") for row in rows]

    if not columns:
        return f"No se pudieron extraer columnas de {source_db}."

    # Determine k
    n_cols = len(columns)
    if k == "auto":
        # Build feature vectors and find optimal k
        from semantic_tools.clustering import column_profile as _cp
        feature_vectors = [_cp(columns[h])["features"] for h in columns]
        max_k = min(n_cols, 6)
        if max_k <= 2:
            k_num = 1
        else:
            ok_result = st_optimal_k(feature_vectors, k_max=max_k)
            k_num = ok_result["optimal_k"]
        k_str = f"auto (optimal_k={k_num})"
    else:
        try:
            k_num = int(k)
        except ValueError:
            return f"k invalido: '{k}'. Usa 'auto' o un numero entero."
        k_str = str(k_num)

    result = st_cluster_columns(columns, k=k_num)

    lines = [
        f"Clustering de columnas: {source_db}",
        f"  Columnas: {n_cols} | k={k_str} | iteraciones: {result['iterations']}",
        f"  Inertia: {result['inertia']:.4f}",
        "",
    ]

    for cluster in result["clusters"]:
        lines.append(f"  Cluster {cluster['cluster_id']} ({cluster['size']} cols): {', '.join(cluster['members'])}")
        centroid = cluster['centroid']
        notable = []
        if centroid.get('numeric_ratio', 0) > 0.7:
            notable.append("numerico")
        if centroid.get('cardinality_ratio', 0) > 0.8:
            notable.append("alta cardinalidad")
        if centroid.get('missing_ratio', 0) > 0.3:
            notable.append(f"missing {centroid['missing_ratio']:.0%}")
        if centroid.get('has_dates', 0) > 0.5:
            notable.append("fechas")
        if notable:
            lines.append(f"    Perfil: {', '.join(notable)}")

    return "\n".join(lines)


def tool_column_quality(source_db: str, column_name: str) -> str:
    """Detecta anomalias y calcula score de calidad (0-100) de una columna. Sin LLM."""
    csv_path = _find_csv_for_source_db(source_db)
    if not csv_path:
        return f"No se encontro CSV para '{source_db}'"

    vals, err = _extract_all_values(csv_path, column_name)
    if err is not None:
        return f"Columna '{column_name}' no encontrada en {source_db}. Columnas: {', '.join(err)}"

    if not vals:
        return f"Columna '{column_name}' esta vacia en {source_db}."

    # Quality score
    score_result = st_column_quality_score(vals)

    # Anomaly detection
    from semantic_tools.statistics import _to_float_list
    num_vals = _to_float_list(vals)
    is_numeric = len(num_vals) > len(vals) * 0.8

    lines = [
        f"Calidad de datos: {source_db}.{column_name}",
        f"  Score: {score_result['score']}/100 (Grade: {score_result['grade']})",
        f"  Completitud: {score_result['completeness']:.0%} | Consistencia: {score_result['consistency']:.0%} | Validez: {score_result['validity']:.0%}",
    ]

    if score_result["issues"]:
        lines.append(f"  Issues: {'; '.join(score_result['issues'])}")

    # Anomaly details
    if is_numeric and len(num_vals) >= 4:
        anomaly_result = st_detect_numeric_anomalies(vals, method="iqr")
        if anomaly_result["anomaly_count"] > 0:
            lines.append(f"  Anomalias (IQR): {anomaly_result['anomaly_count']} ({anomaly_result['anomaly_ratio']:.1%})")
            lines.append(f"  Bounds: [{anomaly_result['bounds']['lower']:.4f}, {anomaly_result['bounds']['upper']:.4f}]")
            for a in anomaly_result["anomalies"][:5]:
                lines.append(f"    - idx={a['index']}: {a['value']} ({a['reason']})")
            if anomaly_result["anomaly_count"] > 5:
                lines.append(f"    ... y {anomaly_result['anomaly_count'] - 5} mas")
    else:
        cat_result = st_detect_categorical_anomalies(vals)
        if cat_result["rare_values"]:
            lines.append(f"  Valores raros: {len(cat_result['rare_values'])}")
            for r in cat_result["rare_values"][:3]:
                lines.append(f"    - '{r['value']}' (freq={r['freq']})")
        if cat_result["high_cardinality"]:
            lines.append(f"  Alta cardinalidad: {cat_result['cardinality']} valores unicos")
        if cat_result["suspicious_patterns"]:
            lines.append(f"  Patrones sospechosos: {len(cat_result['suspicious_patterns'])}")
            for p in cat_result["suspicious_patterns"][:3]:
                lines.append(f"    - idx={p['index']}: {p['issues']}")

    if score_result["grade"] in ("D", "F"):
        lines.append("  => Calidad baja. Recomendado usar auto_clean antes de interoperabilidad.")
    elif score_result["grade"] == "C":
        lines.append("  => Calidad media. Revisar issues antes de usar.")
    else:
        lines.append("  => Calidad aceptable para interoperabilidad.")

    return "\n".join(lines)


def tool_auto_clean(source_db: str, column_name: str) -> str:
    """Limpia una columna (whitespace, encoding, fill missing) y muestra los cambios. Sin LLM."""
    csv_path = _find_csv_for_source_db(source_db)
    if not csv_path:
        return f"No se encontro CSV para '{source_db}'"

    vals, err = _extract_all_values(csv_path, column_name)
    if err is not None:
        return f"Columna '{column_name}' no encontrada en {source_db}. Columnas: {', '.join(err)}"

    if not vals:
        return f"Columna '{column_name}' esta vacia en {source_db}."

    result = st_auto_clean(vals)

    lines = [
        f"Auto-clean: {source_db}.{column_name}",
        f"  Valores procesados: {len(vals)} | Cambios: {result['total_changes']}",
        f"  Fixes aplicados: {', '.join(result['fixes_applied']) if result['fixes_applied'] else 'ninguno'}",
    ]

    if result["changes"]:
        lines.append("  Detalle de cambios (primeros 10):")
        for c in result["changes"][:10]:
            lines.append(f"    - idx={c['index']}: '{c['original']}' -> '{c['cleaned']}' ({', '.join(c['fixes'])})")
        if result["total_changes"] > 10:
            lines.append(f"    ... y {result['total_changes'] - 10} mas")
    else:
        lines.append("  No se requirieron cambios. Columna limpia.")

    return "\n".join(lines)


def tool_profile_dataset(source_db: str) -> str:
    """Perfila todas las columnas de un dataset en un solo paso. Sin LLM."""
    csv_path = _find_csv_for_source_db(source_db)
    if not csv_path:
        return f"No se encontro CSV para '{source_db}'"
    result = st_profile_csv(csv_path)
    return st_format_profile(result)


def tool_auto_match(source_db_a: str, source_db_b: str) -> str:
    """Matchea automaticamente columnas entre dos datasets en batch. Sin LLM.

    Ejecuta similarity + statistics + quality en todos los pares de columnas.
    Retorna mapeos con score de confianza: high (>=0.7), medium (0.4-0.7), low (<0.4).
    Los de alta confianza se persisten en el grafo automaticamente.
    Los de confianza media se escalan a ask_human.
    """
    csv_a = _find_csv_for_source_db(source_db_a)
    csv_b = _find_csv_for_source_db(source_db_b)
    if not csv_a:
        return f"No se encontro CSV para '{source_db_a}'"
    if not csv_b:
        return f"No se encontro CSV para '{source_db_b}'"

    result = st_auto_match(csv_a, csv_b)
    output = st_format_match(result)

    # Persist high-confidence mappings to the graph
    high = result.get("high_confidence", [])
    if high:
        output += "\n\n=== PERSISTENCIA AUTOMATICA ===\n"
        g = load_graph_cached()
        all_fields = g.list_fields()
        persisted = 0
        for m in high:
            col_a = m["column_a"]
            col_b = m["column_b"]
            # Find field nodes for both columns
            fields_a = [f for f in all_fields if f.get("source_db", "") == source_db_a and f.get("column", "").lower().replace(" ", "_") == col_a.lower().replace(" ", "_")]
            fields_b = [f for f in all_fields if f.get("source_db", "") == source_db_b and f.get("column", "").lower().replace(" ", "_") == col_b.lower().replace(" ", "_")]
            if fields_a and fields_b:
                for fa in fields_a:
                    for fb in fields_b:
                        try:
                            g.graph.add_edge(fa["id"], fb["id"], type=EdgeType.EQUIVALE_A.value, source="auto_match")
                            g._db_upsert_edge(fa["id"], fb["id"], EdgeType.EQUIVALE_A.value, {"source": "auto_match"})
                            persisted += 1
                        except Exception:
                            pass
        if persisted > 0:
            from .graph.catalog import _NOMENCLADOR_PATH
            g.save(str(_NOMENCLADOR_PATH))
            clear_graph_cache()
            try:
                from .lifecycle import log_event
                log_event(f"{source_db_a}::{source_db_b}", "equivalence_created", actor="agent",
                          reason=f"auto_match persistió {persisted} aristas EQUIVALE_A",
                          details=f"source_db_a={source_db_a}, source_db_b={source_db_b}")
            except Exception:
                pass
            output += f"  {persisted} aristas EQUIVALE_A persistidas en el grafo.\n"
        else:
            output += "  No se encontraron field nodes en el grafo para persistir. Ingesta los datasets primero.\n"

    # Escalate medium-confidence to human
    medium = result.get("medium_confidence", [])
    if medium:
        output += f"\n=== REVISION HUMANA REQUERIDA ({len(medium)} mapeos ambiguos) ===\n"
        for m in medium[:5]:
            output += f"  {m['column_a']} <-> {m['column_b']} (conf={m['confidence']:.2f})\n"
        if len(medium) > 5:
            output += f"  ... y {len(medium) - 5} mas. Usa ask_human para resolver.\n"

    return output


# === LEARNING LOOP: recall de feedback ===

def tool_recall_feedback(query: str) -> str:
    """Recupera decisiones pasadas del decision log relevantes para la consulta.

    Busca rechazos, aprobaciones y cambios de estado que coincidan con
    palabras clave de la consulta. Prioriza feedback humano (rejected/approved).
    """
    events = lifecycle_recall_feedback(query, limit=5)
    if not events:
        return "No hay decisiones pasadas relevantes para esta consulta."

    lines = [f"Decisiones pasadas relevantes ({len(events)}):"]
    for e in events:
        actor_tag = "humano" if e["actor"] == "human" else "auto"
        concept_short = e["concept_id"].replace("concept:", "")
        reason = e["reason"] or "(sin razon)"
        lines.append(f"  [{e['timestamp'][:10]}] {e['action']} ({actor_tag}) — {concept_short}: {reason}")

    return "\n".join(lines)


# === COMMUNITY DETECTION + GLOBAL SEARCH (inspirado en GraphRAG) ===

def tool_detect_communities(resolution: str = "1.0") -> str:
    """Detecta comunidades de variables relacionadas usando Louvain. Sin LLM."""
    try:
        res = float(resolution)
    except (ValueError, TypeError):
        res = 1.0
    g = load_graph_cached()
    communities = g.detect_communities(resolution=res)
    if not communities:
        return "No hay suficientes nodos en el grafo para detectar comunidades. Ingesta datasets primero."
    lines = [f"Comunidades detectadas: {len(communities)} (Louvain, resolution={res})"]
    for c in communities[:10]:
        names = ", ".join(c["member_names"][:8])
        if len(c["member_names"]) > 8:
            names += f" ... (+{len(c['member_names']) - 8})"
        lines.append(f"  Comunidad {c['community_id']}: {c['size']} nodos ({c['dominant_type']}) — {names}")
    if len(communities) > 10:
        lines.append(f"  ... y {len(communities) - 10} comunidades mas.")
    lines.append("\nUsa global_search para hacer preguntas globales sobre el nomenclador usando estas comunidades.")
    return "\n".join(lines)


def tool_global_search(query: str) -> str:
    """Busqueda global sobre el nomenclador usando community reports + LLM (emula GraphRAG).

    Detecta comunidades con Louvain, genera un resumen de cada una con LLM,
    y sintetiza una respuesta global a la pregunta del usuario.
    Usa para preguntas globales como 'que areas tematicas cubre el nomenclador?'
    o 'hay brechas de cobertura en datos de salud?'.
    """
    g = load_graph_cached()
    return g.global_search(query)


def tool_community_reports(resolution: str = "1.0") -> str:
    """Genera reportes narrativos de cada comunidad usando LLM. Emula GraphRAG community summarization."""
    try:
        res = float(resolution)
    except (ValueError, TypeError):
        res = 1.0
    g = load_graph_cached()
    reports = g.get_community_reports(resolution=res)
    if not reports:
        return "No hay suficientes nodos para generar community reports."
    lines = [f"Community reports ({len(reports)} comunidades):"]
    for r in reports:
        lines.append(f"\n--- Comunidad {r['community_id']} ({r['size']} nodos, {r['dominant_type']}) ---")
        lines.append(r["report"])
    return "\n".join(lines)


# === TOOL DISPATCHER ===

TOOLS = {
    "recall_feedback": tool_recall_feedback,
    "search_graph": tool_search_graph,
    "detect_standard": tool_detect_standard,
    "validate_interop": tool_validate_interop,
    "generate_transform": tool_generate_transform,
    "list_concepts": tool_list_concepts,
    "get_classifier": tool_get_classifier,
    "get_concept_context": tool_get_concept_context,
    "verify_field": tool_verify_field,
    "verify_mapping": tool_verify_mapping,
    "compute_confidence": tool_compute_confidence,
    "compute_transform": tool_compute_transform,
    "audit_graph": tool_audit_graph,
    "graph_health": tool_graph_health,
    "fix_orphans": tool_fix_orphans,
    "retry_proposals": tool_retry_proposals,
    "sample_column_values": tool_sample_column_values,
    "compare_value_sets": tool_compare_value_sets,
    "compare_composite_values": tool_compare_composite_values,
    "persist_equivalences": tool_persist_equivalences,
    "semantic_similarity": tool_semantic_similarity,
    "text_similarity": tool_text_similarity,
    "composite_similarity": tool_composite_similarity,
    "compare_distributions": tool_compare_distributions,
    "correlation": tool_correlation,
    "distribution_summary": tool_distribution_summary,
    "cluster_columns": tool_cluster_columns,
    "column_quality": tool_column_quality,
    "auto_clean": tool_auto_clean,
    "profile_dataset": tool_profile_dataset,
    "auto_match": tool_auto_match,
    "detect_communities": tool_detect_communities,
    "global_search": tool_global_search,
    "community_reports": tool_community_reports,
}

TOOLS_SCHEMA = {
    "recall_feedback": {"query": "str"},
    "search_graph": {"query": "str"},
    "detect_standard": {"column_name": "str", "sample_values": "list[str]"},
    "validate_interop": {"source_db": "str", "target_db": "str"},
    "generate_transform": {"source_db": "str", "target_db": "str"},
    "list_concepts": {},
    "get_classifier": {"standard_id": "str"},
    "get_concept_context": {"query": "str"},
    "verify_field": {"field_id": "str"},
    "verify_mapping": {"classifier_a": "str", "classifier_b": "str"},
    "compute_confidence": {"concept_name": "str"},
    "compute_transform": {"source_db": "str", "target_db": "str"},
    "audit_graph": {},
    "ask_human": {"question": "str"},
    "graph_health": {},
    "fix_orphans": {"dry_run": "bool"},
    "retry_proposals": {"dry_run": "bool"},
    "sample_column_values": {"source_db": "str", "column_name": "str"},
    "compare_value_sets": {"source_db_a": "str", "column_a": "str", "source_db_b": "str", "column_b": "str"},
    "compare_composite_values": {"source_db_a": "str", "columns_a": "str", "source_db_b": "str", "column_b": "str"},
    "persist_equivalences": {"source_db_a": "str", "column_a": "str", "source_db_b": "str", "column_b": "str", "equivalences_json": "str"},
    "semantic_similarity": {"source_db_a": "str", "column_a": "str", "source_db_b": "str", "column_b": "str"},
    "text_similarity": {"concept_a": "str", "concept_b": "str"},
    "composite_similarity": {"source_db_a": "str", "column_a": "str", "source_db_b": "str", "column_b": "str"},
    "compare_distributions": {"source_db_a": "str", "column_a": "str", "source_db_b": "str", "column_b": "str", "test_type": "str"},
    "correlation": {"source_db": "str", "column_a": "str", "column_b": "str", "method": "str"},
    "distribution_summary": {"source_db": "str", "column_name": "str"},
    "cluster_columns": {"source_db": "str", "k": "str"},
    "column_quality": {"source_db": "str", "column_name": "str"},
    "auto_clean": {"source_db": "str", "column_name": "str"},
    "profile_dataset": {"source_db": "str"},
    "auto_match": {"source_db_a": "str", "source_db_b": "str"},
    "detect_communities": {"resolution": "str"},
    "global_search": {"query": "str"},
    "community_reports": {"resolution": "str"},
}

# Descripciones de tools para el schema OpenAI
_TOOL_DESCRIPTIONS = {
    "recall_feedback": "Recupera decisiones pasadas del decision log relevantes para una consulta. Retorna rechazos, aprobaciones y cambios de estado con razon. Usa ANTES de proponer un mapeo o crear un concepto para verificar si ya hubo decisiones humanas sobre el tema.",
    "search_graph": "Busca una variable en el nomenclador. Retorna concepto canonico + fuentes con quality_score y review_status.",
    "detect_standard": "Detecta el estandar para una columna dado su nombre y valores muestra.",
    "validate_interop": "Verifica interoperabilidad entre dos fuentes con guardrails (poblacion, metodologia, clasificador, distribucion).",
    "generate_transform": "Genera transformaciones SQL CASE WHEN + JSON Schema para conectar dos fuentes.",
    "list_concepts": "Lista todos los conceptos canonicos del nomenclador.",
    "get_classifier": "Obtiene los valores validos de un estandar.",
    "get_concept_context": "Retorna contexto completo de un concepto (campos, calidad, normativas, conflictos, clasificador).",
    "verify_field": "Verificacion determinista de que los sample_values de un campo estan dentro del clasificador.",
    "verify_mapping": "Verificacion determinista de biyectividad del mapping entre dos clasificadores.",
    "compute_confidence": "Score determinista de confidence para interoperabilidad de cada field del concepto.",
    "compute_transform": "Genera SQL CASE WHEN determinista desde el mapping del grafo, sin LLM.",
    "audit_graph": "Audit determinista de invariantes del grafo (edges validos, nodos huerfanos, fields sin concepto).",
    "ask_human": "Pregunta al humano cuando necesitas aclaracion.",
    "graph_health": "Diagnostico completo del estado del governance-agent: connectivity, cobertura, calidad, nodos huerfanos, proposals stale.",
    "fix_orphans": "Auto-healing de nodos huerfanos. Busca conceptos similares y linkea fields sin concepto. dry_run=true solo sugiere.",
    "retry_proposals": "Auto-healing de nodos proposed stale. Auto-aprueba alta calidad (>=0.7) con 7+ dias, flaggea baja calidad.",
    "sample_column_values": "Extrae valores unicos de una columna de un dataset CSV. Usa para inspeccionar el contenido de columnas en formato largo donde el significado semantico esta en los valores.",
    "compare_value_sets": "Compara dos conjuntos de valores de dos datasets y descubre equivalencias semanticas via LLM. Usa para encontrar equivalencias no obvias entre columnas de formato largo que validate_interop no puede detectar.",
    "compare_composite_values": "Compara valores compuestos (2+ columnas concatenadas con coma) de un dataset vs una columna de otro. Usa para formato largo donde Item+Element equivale a Indicator Name.",
    "persist_equivalences": "Persiste equivalencias descubiertas en el grafo como aristas EQUIVALE_A entre fields. Acepta JSON de equivalencias de compare_value_sets o compare_composite_values. Requiere que ambos datasets esten ingestados en el grafo.",
    "semantic_similarity": "Compara dos columnas cuantitativamente (cosine, jaccard, overlap) sin LLM. Determinista y gratuito. Usa como pre-filtro antes de compare_value_sets.",
    "text_similarity": "Compara definiciones de dos conceptos via TF-IDF sin LLM. Determinista. Usa para verificar si dos conceptos son el mismo.",
    "composite_similarity": "Combina similitud de valores (cosine, jaccard, overlap) + similitud de texto (TF-IDF) en un solo score. Sin LLM. Usa para evaluacion completa de equivalencia.",
    "compare_distributions": "Compara dos columnas con pruebas estadisticas (Welch t-test para numericas, chi-square para categoricas). Auto-detecta el tipo. Sin LLM. Usa para verificar si dos columnas tienen la misma distribucion.",
    "correlation": "Calcula correlacion (Pearson o Spearman) entre dos columnas del mismo dataset. Sin LLM. Usa para detectar relaciones entre variables.",
    "distribution_summary": "Resume la distribucion de una columna: media, mediana, std, cuartiles (numericas) o cardinalidad y top valores (categoricas). Sin LLM.",
    "cluster_columns": "Agrupa columnas de un dataset por perfil estructural usando k-means. Auto-detecta k optimo via elbow method. Sin LLM. Usa para descubrir columnas similares dentro de un dataset.",
    "column_quality": "Detecta anomalias (outliers IQR, valores raros, encoding artifacts) y calcula score de calidad 0-100 con grade A-F. Sin LLM. Usa ANTES de recomendar interoperabilidad.",
    "auto_clean": "Limpia una columna: trim whitespace, fix encoding mojibake, fill valores faltantes (median/mean). Sin LLM. Usa despues de column_quality si hay issues.",
    "profile_dataset": "Perfila todas las columnas de un dataset en un solo paso: tipo semantico, cardinalidad, missing, patrones, calidad, estadisticas. Sin LLM. Usa ANTES de comparar dos datasets.",
    "auto_match": "Matchea automaticamente todas las columnas entre dos datasets en batch: similarity + statistics + quality + confidence score. Alta confianza (>=0.7) se persiste en el grafo, media (0.4-0.7) escala a humano, baja se descarta. Sin LLM. Usa despues de profile_dataset.",
    "detect_communities": "Detecta comunidades de variables relacionadas usando Louvain (algoritmo de community detection). Agrupa conceptos, fields y clasificadores densamente conectados. Sin LLM. Usa para entender la estructura tematica del nomenclador.",
    "global_search": "Busqueda global sobre el nomenclador: detecta comunidades, genera reportes con LLM, y sintetiza una respuesta global. Emula GraphRAG Global Search. Usa para preguntas transversales como 'que areas tematicas cubre el nomenclador?' o 'hay brechas de cobertura?'.",
    "community_reports": "Genera un resumen narrativo de cada comunidad del nomenclador usando LLM. Emula GraphRAG community summarization. Usa para obtener una vision general de las areas tematicas y como se relacionan.",
}

# Mapeo de tipos Python a tipos OpenAI
_PY_TO_OPENAI_TYPE = {
    "str": "string",
    "bool": "boolean",
    "int": "integer",
    "float": "number",
    "list[str]": "array",
}


def _build_openai_tools_schema() -> list[dict]:
    """Generar schema de tools en formato OpenAI function calling desde TOOLS_SCHEMA."""
    _OPTIONAL_PARAMS = {"dry_run", "test_type", "k", "method", "resolution"}
    tools = []
    for name, params in TOOLS_SCHEMA.items():
        properties = {}
        required = []
        for param_name, param_type in params.items():
            openai_type = _PY_TO_OPENAI_TYPE.get(param_type, "string")
            prop = {"type": openai_type}
            if openai_type == "array":
                prop["items"] = {"type": "string"}
            properties[param_name] = prop
            if param_name not in _OPTIONAL_PARAMS:
                required.append(param_name)

        tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": _TOOL_DESCRIPTIONS.get(name, name),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        tools.append(tool)
    return tools


OPENAI_TOOLS_SCHEMA = _build_openai_tools_schema()


# === NODES (native tool calling) ===

def think_node(state: AgentState) -> AgentState:
    """El LLM razona y decide qué tool usar (o responder directamente) via function calling nativo.

    Context engineering: en lugar de enviar todo el historial (scratchpad + mensajes),
    se envian solo los ultimos 3 turnos con detalle completo y los turnos anteriores
    como resumen condensado. Esto reduce el consumo de tokens y mejora la calidad del
    razonamiento en iteraciones tardias.
    """
    scratchpad = state.get("scratchpad", [])
    all_messages = state.get("messages", [])

    RECENT_TURNS = 3
    RECENT_MSG_COUNT = RECENT_TURNS * 2  # assistant + tool por turno

    # Dividir: turnos recientes (detalle) vs antiguos (resumen)
    if len(all_messages) > RECENT_MSG_COUNT:
        recent_messages = all_messages[-RECENT_MSG_COUNT:]
        old_scratchpad = scratchpad[:-RECENT_TURNS] if len(scratchpad) > RECENT_TURNS else []
    else:
        recent_messages = all_messages
        old_scratchpad = []

    # Resumen condensado de turnos antiguos (solo nombres de tools usados)
    summary_text = ""
    if old_scratchpad:
        summary_lines = []
        for entry in old_scratchpad:
            for line in entry.split("\n"):
                if line.startswith("ACTION:"):
                    summary_lines.append(f"  - {line[7:].strip()}")
        summary_text = f"\n\nAcciones anteriores (resumen):\n" + "\n".join(summary_lines) + "\n"

    # LEARNING LOOP: inyectar feedback pasado relevante (solo en primera iteracion)
    feedback_text = ""
    if state.get("iteration", 0) == 0:
        past_events = lifecycle_recall_feedback(state["user_query"], limit=3)
        if past_events:
            fb_lines = []
            for e in past_events:
                actor_tag = "humano" if e["actor"] == "human" else "auto"
                concept_short = e["concept_id"].replace("concept:", "")
                reason = e["reason"] or "(sin razon)"
                fb_lines.append(f"  [{e['timestamp'][:10]}] {e['action']} ({actor_tag}) — {concept_short}: {reason}")
            feedback_text = f"\n\nDECISIONES PREVIAS RELEVANTES (learning loop):\n" + "\n".join(fb_lines) + "\nConsidera este feedback antes de actuar.\n"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Pregunta del usuario: {state['user_query']}{feedback_text}{summary_text}\n\n¿Qué haces ahora?"},
    ]

    # Solo incluir mensajes de los ultimos 3 turnos (detalle completo)
    for msg in recent_messages:
        if isinstance(msg, dict):
            messages.append(msg)
        elif hasattr(msg, "model_dump"):
            messages.append(msg.model_dump())
        elif hasattr(msg, "dict"):
            messages.append(msg.dict())
        else:
            messages.append({"role": "assistant", "content": str(msg)})

    try:
        result = call_groq_with_tools(
            messages,
            tools=OPENAI_TOOLS_SCHEMA,
            temperature=0.2,
            max_tokens=2000,
        )
    except Exception as e:
        logger.warning("think_node: call_groq_with_tools fallo: %s", e)
        return {
            **state,
            "current_thought": f"Error: LLM no disponible ({e})",
            "current_action": "",
            "current_action_input": "",
            "final_answer": f"No pude procesar la consulta: el servicio LLM fallo ({e}). Intenta de nuevo.",
            "iteration": state.get("iteration", 0) + 1,
        }

    content = result.get("content", "")
    tool_calls = result.get("tool_calls")

    # Si no hay tool_calls, el LLM responde directamente → final
    if not tool_calls:
        return {
            **state,
            "current_thought": content,
            "current_action": "",
            "current_action_input": "",
            "final_answer": content,
            "iteration": state.get("iteration", 0) + 1,
        }

    # Hay tool_calls — tomar la primera (regla: una tool por turno)
    first_call = tool_calls[0]
    tool_name = first_call["function"]["name"]
    tool_args_raw = first_call["function"]["arguments"]
    tool_call_id = first_call.get("id", "")

    try:
        tool_args = json.loads(tool_args_raw) if tool_args_raw else {}
    except json.JSONDecodeError:
        tool_args = {"query": tool_args_raw} if tool_args_raw else {}

    # Guardar mensaje assistant con tool_calls para que el mensaje tool tenga contexto
    assistant_msg = {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    }
    messages = state.get("messages", [])
    messages.append(assistant_msg)

    return {
        **state,
        "current_thought": content,
        "current_action": tool_name,
        "current_action_input": json.dumps(tool_args),
        "tool_call_id": tool_call_id,
        "messages": messages,
        "iteration": state.get("iteration", 0) + 1,
    }


def act_node(state: AgentState) -> AgentState:
    """Ejecuta la tool seleccionada por el LLM via function calling."""
    action = state.get("current_action", "").strip()
    action_input_raw = state.get("current_action_input", "").strip()
    tool_call_id = state.get("tool_call_id", "")

    if action == "ask_human":
        return {
            **state,
            "tool_result": f"NEEDS_HUMAN_INPUT: {action_input_raw}",
            "needs_human_input": action_input_raw,
        }

    if action not in TOOLS:
        return {
            **state,
            "tool_result": f"Error: tool '{action}' no existe. Tools disponibles: {', '.join(TOOLS.keys())}",
        }

    try:
        args = json.loads(action_input_raw) if action_input_raw else {}
    except json.JSONDecodeError:
        args = {"query": action_input_raw} if action_input_raw else {}

    try:
        result = TOOLS[action](**args) if args else TOOLS[action]()
    except Exception as e:
        logger.warning("act_node: tool %r fallo: %s", action, e)
        result = f"Error ejecutando {action}: {e}"

    # Construir mensaje tool para feed-back al LLM en el proximo think
    result_str = str(result)
    if len(result_str) > 2000:
        cut = result_str.rfind('}', 0, 2000)
        if cut > 100:
            result_str = result_str[:cut + 1] + "\n... (truncado)"
        else:
            cut = result_str.rfind('\n', 0, 2000)
            if cut > 100:
                result_str = result_str[:cut] + "\n... (truncado)"
            else:
                result_str = result_str[:2000] + "... (truncado)"
    tool_msg = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": result_str,
    }

    messages = state.get("messages", [])
    messages.append(tool_msg)

    return {
        **state,
        "tool_result": result,
        "messages": messages,
    }


def observe_node(state: AgentState) -> AgentState:
    """Registra la observación en el scratchpad y activa circuit breaker si es necesario."""
    thought = state.get("current_thought", "")
    action = state.get("current_action", "")
    result = state.get("tool_result", "")

    entry = f"THOUGHT: {thought}\nACTION: {action}\nOBSERVATION: {result[:500]}"

    scratchpad = state.get("scratchpad", [])
    scratchpad.append(entry)

    # Circuit breaker: si se alcanzo el limite, generar respuesta parcial
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 8)
    final_answer = state.get("final_answer", "")

    if iteration >= max_iter and not final_answer:
        tools_used = []
        for e in scratchpad:
            for line in e.split("\n"):
                if line.startswith("ACTION:"):
                    tool_name = line[7:].strip()
                    if tool_name:
                        tools_used.append(tool_name)
        final_answer = (
            f"Alcance el limite de {max_iter} iteraciones sin llegar a una respuesta final. "
            f"Tools usadas: {', '.join(tools_used)}. "
            f"Ultima observacion: {result[:300]}. "
            f"Reformula la consulta o continua manualmente con las tools del CLI."
        )
        logger.warning("observe_node: circuit breaker activado en iteracion %d", iteration)

    return {
        **state,
        "scratchpad": scratchpad,
        "final_answer": final_answer,
    }


# === ROUTING ===

def should_act(state: AgentState) -> str:
    """Decidir si ejecutar una tool o terminar después de pensar."""
    if state.get("final_answer"):
        return "end"
    if state.get("needs_human_input"):
        return "end"
    if not state.get("current_action"):
        return "end"
    return "act"


def should_continue(state: AgentState) -> str:
    """Decidir si continuar el loop o terminar después de observar."""
    if state.get("final_answer"):
        return "end"
    if state.get("needs_human_input"):
        return "end"
    if state.get("iteration", 0) >= state.get("max_iterations", 8):
        return "end"
    if not state.get("current_action"):
        return "end"
    return "continue"


# === BUILD GRAPH ===

def build_agent_graph():
    """Construir el state machine del agente ReAct."""
    graph = StateGraph(AgentState)
    
    # Nodos
    graph.add_node("think", think_node)
    graph.add_node("act", act_node)
    graph.add_node("observe", observe_node)
    
    # Edges
    graph.set_entry_point("think")
    
    # Conditional: después de think, ejecutar tool o terminar
    graph.add_conditional_edges(
        "think",
        should_act,
        {
            "act": "act",
            "end": END,
        },
    )
    
    graph.add_edge("act", "observe")
    
    # Conditional: después de observe, volver a think o terminar
    graph.add_conditional_edges(
        "observe",
        should_continue,
        {
            "continue": "think",
            "end": END,
        },
    )
    
    return graph.compile()


# === RUN ===

def run_agent(query: str, max_iterations: int = 8) -> dict:
    """
    Ejecutar el agente ReAct con una consulta del usuario.
    Retorna dict con: final_answer, scratchpad, needs_human_input, iterations, tools_used, health_verified.
    """
    app = build_agent_graph()
    
    initial_state = AgentState(
        messages=[],
        user_query=query,
        scratchpad=[],
        current_thought="",
        current_action="",
        current_action_input="",
        tool_call_id="",
        tool_result="",
        iteration=0,
        final_answer="",
        needs_human_input="",
        max_iterations=max_iterations,
    )
    
    result = app.invoke(initial_state, {"recursion_limit": max_iterations * 3})

    # === POST-EXECUTION VERIFICATION ===
    # Extraer qué tools fueron usadas del scratchpad
    scratchpad = result.get("scratchpad", [])
    tools_used = []
    for entry in scratchpad:
        # Cada entry tiene formato "THOUGHT: ...\nACTION: <tool_name>\nOBSERVATION: ..."
        for line in entry.split("\n"):
            if line.startswith("ACTION:"):
                tool_name = line[len("ACTION:"):].strip()
                if tool_name:
                    tools_used.append(tool_name)

    # Verificar si el agente usó graph_health cuando era relevante
    health_keywords = ["health", "salud", "orphan", "huérfano", "stale", "proposal", "auto-heal", "monitoreo"]
    query_lower = query.lower()
    health_relevant = any(kw in query_lower for kw in health_keywords)
    health_verified = "graph_health" in tools_used

    if health_relevant and not health_verified and not result.get("needs_human_input"):
        logger.warning(
            "run_agent: query parece requerir graph_health pero no fue usada. "
            "Tools usadas: %s", tools_used
        )

    return {
        "final_answer": result.get("final_answer", ""),
        "scratchpad": scratchpad,
        "needs_human_input": result.get("needs_human_input", ""),
        "iterations": result.get("iteration", 0),
        "tools_used": tools_used,
        "health_verified": health_verified,
    }


# === Compatibilidad para moa_agent.py (patron ReAct texto) ===

def _load_graph() -> NomencladorGraph:
    """Alias para load_graph_cached (compatibilidad moa_agent)."""
    return load_graph_cached()


def _clear_graph_cache() -> None:
    """Alias para clear_graph_cache (compatibilidad moa_agent)."""
    clear_graph_cache()


def _parse_response(text: str) -> dict:
    """Parsear respuesta ReAct en formato texto (Thought/Action/Action Input/Final).

    Usado por moa_agent.py que aun usa el patron ReAct textual.
    Case-insensitive: acepta THOUGHT/thought/Thought, ACTION/action/Action, etc.
    Acepta tanto 'Final Answer:' como 'FINAL:' como marca de respuesta final.
    """
    result = {"thought": "", "action": "", "action_input": "", "final": ""}

    # Final Answer o FINAL (case-insensitive)
    final_match = re.search(r"(?:Final Answer|FINAL):\s*(.*?)(?:\n[A-Z]|\Z)", text, re.DOTALL | re.IGNORECASE)
    if final_match:
        result["final"] = final_match.group(1).strip()
        return result

    # Thought (case-insensitive)
    thought_match = re.search(r"THOUGHT:\s*(.*?)(?:\nACTION:|\Z)", text, re.DOTALL | re.IGNORECASE)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    # Action (case-insensitive, terminador acepta espacio o underscore)
    action_match = re.search(r"ACTION:\s*(.*?)(?:\nACTION[\s_]+INPUT:|\Z)", text, re.DOTALL | re.IGNORECASE)
    if action_match:
        result["action"] = action_match.group(1).strip()

    # Action Input o ACTION_INPUT (case-insensitive, acepta espacio o underscore)
    input_match = re.search(r"ACTION[\s_]+INPUT:\s*(.*?)(?:\nObservation:|\nThought:|\nTHOUGHT:|\Z)", text, re.DOTALL | re.IGNORECASE)
    if input_match:
        result["action_input"] = input_match.group(1).strip()

    return result
