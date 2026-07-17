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

from .groq_client import call_groq, call_groq_with_tools
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

    return (
        f"Equivalencias persistidas en el grafo: {field_a_id} -> {field_b_id}\n"
        f"  Total mapeos: {len(mapping)}\n"
        f"  Alta confianza: {high} | Media: {med} | Baja: {low}\n"
        f"  Arista EQUIVALE_A creada con source=value_level_discovery.\n"
        f"  Grafo guardado y cache invalidada."
    )


# === TOOL DISPATCHER ===

TOOLS = {
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
}

TOOLS_SCHEMA = {
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
}

# Descripciones de tools para el schema OpenAI
_TOOL_DESCRIPTIONS = {
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

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Pregunta del usuario: {state['user_query']}{summary_text}\n\n¿Qué haces ahora?"},
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
    tool_msg = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": str(result)[:2000],
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
    """
    result = {"thought": "", "action": "", "action_input": "", "final": ""}

    # Final Answer
    final_match = re.search(r"Final Answer:\s*(.*?)(?:\n[A-Z]|\Z)", text, re.DOTALL)
    if final_match:
        result["final"] = final_match.group(1).strip()
        return result

    # Thought
    thought_match = re.search(r"Thought:\s*(.*?)(?:\nAction:|\Z)", text, re.DOTALL)
    if thought_match:
        result["thought"] = thought_match.group(1).strip()

    # Action
    action_match = re.search(r"Action:\s*(.*?)(?:\nAction Input:|\Z)", text, re.DOTALL)
    if action_match:
        result["action"] = action_match.group(1).strip()

    # Action Input
    input_match = re.search(r"Action Input:\s*(.*?)(?:\nObservation:|\nThought:|\Z)", text, re.DOTALL)
    if input_match:
        result["action_input"] = input_match.group(1).strip()

    return result
