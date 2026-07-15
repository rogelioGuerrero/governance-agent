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

import json
import logging
import re
from typing import TypedDict, Annotated, Optional
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from .groq_client import call_groq, call_groq_with_tools
from .graph.catalog import NomencladorGraph, load_graph_cached, clear_graph_cache
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
"""


# === STATE ===

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
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


def tool_list_concepts() -> str:
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


def tool_audit_graph() -> str:
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


def tool_graph_health() -> str:
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
    """El LLM razona y decide qué tool usar (o responder directamente) via function calling nativo."""
    scratchpad_text = "\n".join(state.get("scratchpad", []))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Pregunta del usuario: {state['user_query']}\n\nHistorial de acciones:\n{scratchpad_text}\n\n¿Qué haces ahora?"},
    ]

    # Si hay resultados de tools previos, incluirlos como mensajes tool
    tool_messages = state.get("messages", [])
    if tool_messages:
        messages.extend(tool_messages)

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

    return {
        **state,
        "current_thought": content,
        "current_action": tool_name,
        "current_action_input": json.dumps(tool_args),
        "tool_call_id": tool_call_id,
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
    """Registra la observación en el scratchpad."""
    thought = state.get("current_thought", "")
    action = state.get("current_action", "")
    result = state.get("tool_result", "")
    
    entry = f"THOUGHT: {thought}\nACTION: {action}\nOBSERVATION: {result[:500]}"
    
    scratchpad = state.get("scratchpad", [])
    scratchpad.append(entry)
    
    return {
        **state,
        "scratchpad": scratchpad,
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
    if state.get("iteration", 0) >= 8:
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
