"""
MoA (Mixture of Agents) multi-agente para governance.

3 agentes especializados analizan una consulta desde sus perspectivas:
1. Agente JURIDICO — normativo, legal, proteccion de datos, respaldo normativo
2. Agente TECNICO — estandares, interoperabilidad, transformaciones, tipos de datos
3. Agente ESTADISTICO — poblacion, muestreo, calidad de datos, sesgos

Un SINTETIZADOR combina las 3 perspectivas en una respuesta final.

Flujo:
    Usuario consulta
        |
    +---+---+
    |   |   |
  JUR  TEC  EST   (paralelo, cada uno con sus tools)
    |   |   |
    +---+---+
        |
   SINTETIZADOR  (Groq combina las 3 perspectivas)
        |
   Respuesta final

Cada agente usa el mismo loop ReAct del agent.py pero con system prompt
especializado y subconjunto de tools relevante a su especialidad.
"""

import json
import re
from pathlib import Path
from typing import Optional

from .groq_client import call_groq
from .log_config import get_logger
from .agent import (
    _load_graph, _clear_graph_cache, _parse_response,
    tool_search_graph, tool_detect_standard, tool_validate_interop,
    tool_generate_transform, tool_list_concepts, tool_get_classifier,
    NOMENCLADOR_PATH,
)
from .graph.catalog import NomencladorGraph
from .lifecycle import get_history, get_explanatory_note, find_deprecated

log = get_logger("moa")


# === TOOLS ESPECIFICAS POR AGENTE ===

def tool_get_normative(concept_name: str) -> str:
    """Buscar respaldo normativo de un concepto."""
    g = _load_graph()
    concept = g.find_concept(concept_name)
    if not concept:
        return f"Concepto '{concept_name}' no encontrado"
    normatives = g.find_normative_of_concept(concept["id"])
    if not normatives:
        return f"'{concept_name}' no tiene respaldo normativo"
    lines = [f"Respaldos normativos de '{concept_name}':"]
    for n in normatives:
        lines.append(f"  - {n.get('source', '?')}: {n.get('citation', '')[:100]} (score: {n.get('similarity_score', 0):.3f})")
    return "\n".join(lines)


def tool_get_lifecycle(concept_name: str) -> str:
    """Ver historial y estado de una variable."""
    g = _load_graph()
    concept = g.find_concept(concept_name)
    if not concept:
        return f"Concepto '{concept_name}' no encontrado"
    history = get_history(concept["id"])
    status = concept.get("status", "activo")
    if not history:
        return f"'{concept_name}' - Estado: {status}. Sin historial."
    lines = [f"'{concept_name}' - Estado: {status} ({len(history)} eventos):"]
    for e in history:
        lines.append(f"  [{e['timestamp'][:10]}] {e['action']} ({e['actor']}): {e.get('reason', '')}")
    return "\n".join(lines)


def tool_get_custodian(concept_name: str) -> str:
    """Ver custodio y departamento de una variable."""
    g = _load_graph()
    concept = g.find_concept(concept_name)
    if not concept:
        return f"Concepto '{concept_name}' no encontrado"
    custodian = concept.get("custodian", "") or "-"
    dept = concept.get("custodian_department", "") or "-"
    why = concept.get("why", "") or "-"
    what_for = concept.get("what_for", "") or "-"
    return f"'{concept_name}': Custodio: {custodian} | Depto: {dept} | Por que: {why} | Para que: {what_for}"


def tool_list_deprecated() -> str:
    """Listar variables deprecadas o retiradas."""
    deprecated = find_deprecated()
    if not deprecated:
        return "No hay variables deprecadas"
    lines = ["Variables deprecadas/retiradas:"]
    for d in deprecated:
        lines.append(f"  - {d['concept_id']}: {d['last_status']}")
    return "\n".join(lines)


def tool_get_contexts(concept_name: str) -> str:
    """Ver significados contextuales de una variable."""
    g = _load_graph()
    concept = g.find_concept(concept_name)
    if not concept:
        return f"Concepto '{concept_name}' no encontrado"
    meanings = g.get_context_meanings(concept["id"])
    if not meanings:
        return f"'{concept_name}' sin contextos registrados"
    lines = [f"Contextos de '{concept_name}':"]
    for m in meanings:
        lines.append(f"  - {m.get('source_db', '?')}: {m.get('description', '')} (contexto: {m.get('name', '-')})")
    return "\n".join(lines)


def tool_find_conflicts() -> str:
    """Detectar conflictos de contexto."""
    g = _load_graph()
    conflicts = g.find_context_conflicts()
    if not conflicts:
        return "No hay conflictos de contexto"
    lines = [f"Conflictos de contexto ({len(conflicts)}):"]
    for c in conflicts:
        lines.append(f"  - {c['concept_name']}:")
        for m in c["meanings"]:
            lines.append(f"    {m.get('source_db', '?')}: {m.get('description', '')}")
    return "\n".join(lines)


def tool_get_composites(concept_name: str) -> str:
    """Ver componentes de una variable compuesta."""
    g = _load_graph()
    concept = g.find_concept(concept_name)
    if not concept:
        return f"Concepto '{concept_name}' no encontrado"
    components = g.find_components(concept["id"])
    if not components:
        return f"'{concept_name}' no es una variable compuesta"
    lines = [f"'{concept_name}' se compone de:"]
    for c in components:
        lines.append(f"  - {c.get('name', c.get('id', '?'))} (operacion: {c.get('operation', '?')})")
    return "\n".join(lines)


def tool_version_info() -> str:
    """Ver version del nomenclador."""
    g = _load_graph()
    info = g.version_info()
    return f"Version: {info['version']} | Cambios: {info['total_changes']}"


# === PROMPTS POR AGENTE ===

JURIDICO_PROMPT = """Eres un agente JURIDICO especializado en governance de datos institucionales.
Tu perspectiva es NORMATIVA y LEGAL. Analizas:

- Respaldo normativo: la variable tiene documento legal que la respalda?
- Proteccion de datos: hay consideraciones de datos personales?
- Cumplimiento: la variable cumple con normativas vigentes?
- Custodio: quien es el responsable legal de la variable?
- Estado: la variable esta activa, deprecada, retirada?

Tools disponibles:
1. search_graph(query): Buscar variable en el nomenclador
2. get_normative(concept_name): Ver respaldo normativo de una variable
3. get_lifecycle(concept_name): Ver historial y estado de una variable
4. get_custodian(concept_name): Ver custodio y departamento
5. list_deprecated(): Listar variables deprecadas
6. list_concepts(): Listar todos los conceptos

FORMATO ReAct:
THOUGHT: <razonamiento>
ACTION: <tool>
ACTION_INPUT: <json>

Para finalizar:
THOUGHT: <razonamiento final>
FINAL: <analisis juridico de la consulta>
VEREDICTO: {"can_proceed": true/false, "objection_type": "legal|none", "reason": "..."}

Responde en español. Sé especifico sobre normativas, custodios y estado legal.
El VEREDICTO es OBLIGATORIO: can_proceed=false si hay objecion legal, true si no la hay.
"""

TECNICO_PROMPT = """Eres un agente TECNICO especializado en interoperabilidad semantica.
Tu perspectiva es TECNICA y de ESTANDARES. Analizas:

- Estandares: la variable usa estandares internacionales? Cuales?
- Interoperabilidad: se puede cruzar entre fuentes? Hay asimetrias?
- Transformaciones: que transformaciones SQL se necesitan?
- Tipos de datos: hay problemas de formato?
- Clasificadores: los valores validos coinciden entre fuentes?
- Variables compuestas: la variable se compone de otras?

Tools disponibles:
1. search_graph(query): Buscar variable en el nomenclador
2. detect_standard(column_name, sample_values): Detectar estandar
3. validate_interop(source_db, target_db): Verificar interoperabilidad
4. generate_transform(source_db, target_db): Generar transformaciones SQL
5. get_classifier(standard_id): Ver valores validos de un estandar
6. list_concepts(): Listar conceptos
7. get_composites(concept_name): Ver componentes de variable compuesta
8. get_contexts(concept_name): Ver contextos de una variable
9. find_conflicts(): Detectar conflictos de contexto

FORMATO ReAct:
THOUGHT: <razonamiento>
ACTION: <tool>
ACTION_INPUT: <json>

Para finalizar:
THOUGHT: <razonamiento final>
FINAL: <analisis tecnico de la consulta>
VEREDICTO: {"can_proceed": true/false, "objection_type": "technical|none", "reason": "..."}

Responde en español. Sé especifico sobre estandares, transformaciones y compatibilidad.
El VEREDICTO es OBLIGATORIO: can_proceed=false si hay objecion tecnica insalvable, true si es viable.
"""

ESTADISTICO_PROMPT = """Eres un agente ESTADISTICO especializado en calidad de datos.
Tu perspectiva es ESTADISTICA y de CALIDAD. Analizas:

- Poblacion: la variable aplica a toda la poblacion o un subconjunto?
- Metodologia: como se captura el dato? Auto-reporte? Observacion clinica?
- Calidad: hay valores faltantes, outliers, inconsistencias?
- Sesgos: la variable puede tener sesgo de seleccion o medicion?
- Contextos: la variable significa lo mismo en todas las fuentes?
- Conflictos: hay conflictos de contexto entre fuentes?

Tools disponibles:
1. search_graph(query): Buscar variable en el nomenclador
2. list_concepts(): Listar todos los conceptos
3. get_contexts(concept_name): Ver significados contextuales
4. find_conflicts(): Detectar conflictos de contexto
5. get_lifecycle(concept_name): Ver historial de cambios
6. version_info(): Ver version del nomenclador

FORMATO ReAct:
THOUGHT: <razonamiento>
ACTION: <tool>
ACTION_INPUT: <json>

Para finalizar:
THOUGHT: <razonamiento final>
FINAL: <analisis estadistico de la consulta>

Responde en español. Sé especifico sobre poblacion, metodologia, sesgos y calidad.
"""

SINTETIZADOR_PROMPT = """Eres un SINTETIZADOR de un sistema MoA (Mixture of Agents).
Recibes los analisis de 3 agentes especializados sobre la misma consulta:

1. AGENTE JURIDICO: perspectiva normativa y legal
2. AGENTE TECNICO: perspectiva de estandares e interoperabilidad
3. AGENTE ESTADISTICO: perspectiva de calidad de datos y sesgos

Tu trabajo es COMBINAR las 3 perspectivas en una respuesta unificada que:
- Identifique puntos de acuerdo entre los agentes
- Destaque tensiones o conflictos entre perspectivas
- Priorice recomendaciones accionables
- Estructure la respuesta claramente

GUARDRAIL DE ARBITRAJE (CRITICO):
Cuando el Agente JURIDICO y el Agente TECNICO entren en conflicto sobre una variable:
- El JURIDICO tiene prioridad ABSOLUTA en temas de proteccion de datos, PII y cumplimiento legal.
- Si el JURIDICO veta una variable por violar normativa de proteccion de datos, la recomendacion final DEBE ser NO usar la variable, independientemente de la viabilidad tecnica.
- Si el JURIDICO no encuentra objecion legal, el TECNICO prevalece en temas de estandares e interoperabilidad.
- El ESTADISTICO tiene voz consultiva: sus observaciones sobre calidad y sesgos se incluyen como advertencias pero no pueden vetar una variable.
- En caso de conflicto no resuelto, recomendar NO proceder hasta que un custodio humano decida.

Formato de salida:
## Analisis Juridico
<resumen del analisis juridico>

## Analisis Tecnico
<resumen del analisis tecnico>

## Analisis Estadistico
<resumen del analisis estadistico>

## Sintesis
<puntos de acuerdo, tensiones, y recomendacion final>

Responde en español. Sé conciso pero completo.
"""


# === TOOLS POR AGENTE ===

JURIDICO_TOOLS = {
    "search_graph": tool_search_graph,
    "get_normative": tool_get_normative,
    "get_lifecycle": tool_get_lifecycle,
    "get_custodian": tool_get_custodian,
    "list_deprecated": tool_list_deprecated,
    "list_concepts": tool_list_concepts,
}

TECNICO_TOOLS = {
    "search_graph": tool_search_graph,
    "detect_standard": tool_detect_standard,
    "validate_interop": tool_validate_interop,
    "generate_transform": tool_generate_transform,
    "get_classifier": tool_get_classifier,
    "list_concepts": tool_list_concepts,
    "get_composites": tool_get_composites,
    "get_contexts": tool_get_contexts,
    "find_conflicts": tool_find_conflicts,
}

ESTADISTICO_TOOLS = {
    "search_graph": tool_search_graph,
    "list_concepts": tool_list_concepts,
    "get_contexts": tool_get_contexts,
    "find_conflicts": tool_find_conflicts,
    "get_lifecycle": tool_get_lifecycle,
    "version_info": tool_version_info,
}


# === EJECUCION DE AGENTES ===

def _run_single_agent(
    name: str,
    system_prompt: str,
    tools: dict,
    query: str,
    max_iterations: int = 5,
) -> str:
    """Ejecutar un agente especializado con loop ReAct simplificado."""
    scratchpad = []

    for iteration in range(max_iterations):
        scratchpad_text = "\n".join(scratchpad) if scratchpad else "(sin acciones previas)"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Consulta: {query}\n\nHistorial:\n{scratchpad_text}\n\nQue haces ahora?"},
        ]

        try:
            response = call_groq(messages, temperature=0.2, max_tokens=4000)
        except Exception as e:
            return f"[{name}] Error: {e}"

        parsed = _parse_response(response)

        if parsed.get("final"):
            return parsed["final"]

        action = parsed.get("action", "").strip()
        action_input_raw = parsed.get("action_input", "").strip()

        if not action:
            return f"[{name}] No pudo determinar accion. {parsed.get('thought', '')}"

        # Ejecutar tool
        if action not in tools:
            result = f"Error: tool '{action}' no disponible. Tools: {', '.join(tools.keys())}"
        else:
            try:
                args = json.loads(action_input_raw) if action_input_raw else {}
                result = tools[action](**args) if args else tools[action]()
            except json.JSONDecodeError:
                args = {"query": action_input_raw} if action_input_raw else {}
                try:
                    result = tools[action](**args) if args else tools[action]()
                except Exception as e:
                    log.warning("[%s] tool %r fallo: %s", name, action, e)
                    result = f"Error: {e}"
            except Exception as e:
                log.warning("[%s] tool %r fallo: %s", name, action, e)
                result = f"Error: {e}"

        scratchpad.append(f"THOUGHT: {parsed.get('thought', '')}\nACTION: {action}\nOBSERVATION: {result[:400]}")

    return f"[{name}] Alcanzo el limite de iteraciones. Ultimo pensamiento: {parsed.get('thought', 'N/A')}"


# === SINTETIZADOR ===

def _extract_verdict(agent_output: str) -> dict:
    """Extraer veredicto estructurado del output de un agente.
    
    Busca un bloque JSON con formato:
        VEREDICTO: {"can_proceed": true/false, "objection_type": "legal|technical|statistical", "reason": "..."}
    
    Si no encuentra JSON, hace fallback a deteccion por patrones.
    
    Returns: {"can_proceed": bool, "objection_type": str, "reason": str}
    """
    # Buscar bloque VEREDICTO: {...}
    pattern = r'VEREDICTO:\s*(\{[^}]+\})'
    match = re.search(pattern, agent_output, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            data = json.loads(match.group(1))
            return {
                "can_proceed": data.get("can_proceed", True),
                "objection_type": data.get("objection_type", ""),
                "reason": data.get("reason", ""),
            }
        except json.JSONDecodeError:
            pass
    
    # Fallback: deteccion por patrones estructurada (no keywords sueltos)
    output_lower = agent_output.lower()
    
    # Patrones de veto (mas estructurados que keywords sueltos)
    veto_patterns = [
        r"no\s+(?:se\s+)?puede\s+(?:usar|proceder|intercambiar)",
        r"viola\s+(?:la\s+)?(?:normativa|ley|reglamento)",
        r"prohib(?:e|ido)\s+(?:el\s+)?(?:uso|intercambio)",
        r"veta(?:do)?\s+(?:el\s+)?(?:uso|intercambio)",
        r"no\s+cumple\s+(?:con\s+)?(?:la\s+)?(?:normativa|ley)",
        r"requiere\s+(?:consentimiento|autorizacion)\s+(?:previo|explicito)",
    ]
    # Patrones de aprobacion
    approve_patterns = [
        r"(?:es\s+)?(?:vi|fa)ctible",
        r"se\s+puede\s+(?:usar|proceder|intercambiar)",
        r"(?:es\s+)?compatible\s+con",
        r"(?:es\s+)?interoperable",
        r"no\s+hay\s+(?:objecion|problema)\s+(?:tecnico|legal)",
        r"recomenda(?:do|ble)\s+(?:usar|proceder)",
    ]
    
    has_veto = any(re.search(p, output_lower) for p in veto_patterns)
    has_approve = any(re.search(p, output_lower) for p in approve_patterns)
    
    if has_veto:
        return {"can_proceed": False, "objection_type": "unknown", "reason": "Veto detectado por patron"}
    elif has_approve:
        return {"can_proceed": True, "objection_type": "", "reason": "Aprobacion detectada por patron"}
    return {"can_proceed": True, "objection_type": "", "reason": "Sin veredicto explicito"}


def _detect_juridico_tecnico_conflict(juridico: str, tecnico: str) -> str:
    """Detectar conflicto entre agente juridico y tecnico (Gap M).

    Usa extraccion estructurada de veredictos en vez de keywords sueltos.
    Returns descripcion del conflicto o cadena vacia si no hay conflicto.
    """
    jur_verdict = _extract_verdict(juridico)
    tec_verdict = _extract_verdict(tecnico)
    
    # Conflicto: juridico veta + tecnico aprueba
    if not jur_verdict["can_proceed"] and tec_verdict["can_proceed"]:
        objection = jur_verdict.get("objection_type", "legal")
        reason = jur_verdict.get("reason", "sin razon especificada")
        return (f"CONFLICTO DETECTADO: El agente JURIDICO veta la operacion "
                f"(tipo: {objection}, razon: {reason}) "
                f"mientras el agente TECNICO la considera viable. "
                f"El guardrail de arbitraje da prioridad ABSOLUTA al JURIDICO.")
    
    # Conflicto inverso: juridico aprueba + tecnico veta (menos critico)
    if jur_verdict["can_proceed"] and not tec_verdict["can_proceed"]:
        reason = tec_verdict.get("reason", "sin razon especificada")
        return (f"TENSION DETECTADA: El agente TECNICO identifica objeciones "
                f"(razon: {reason}) pero el JURIDICO no encuentra impedimento legal. "
                f"El TECNICO prevalece en temas de interoperabilidad.")
    
    return ""


def _synthesize(query: str, juridico: str, tecnico: str, estadistico: str) -> str:
    """Combinar las 3 perspectivas en una respuesta unificada."""
    # Gap M: Detectar conflicto juridico vs tecnico antes de sintetizar
    conflict = _detect_juridico_tecnico_conflict(juridico, tecnico)
    conflict_notice = ""
    if conflict:
        conflict_notice = f"\n\n*** {conflict} ***\nEl sintetizador DEBE aplicar el guardrail de arbitraje: el JURIDICO tiene prioridad absoluta en temas legales.\n"

    user_content = f"""Consulta original: {query}

=== ANALISIS JURIDICO ===
{juridico}

=== ANALISIS TECNICO ===
{tecnico}

=== ANALISIS ESTADISTICO ===
{estadistico}{conflict_notice}

Ahora sintetiza las 3 perspectivas en una respuesta unificada."""

    messages = [
        {"role": "system", "content": SINTETIZADOR_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        return call_groq(messages, temperature=0.3, max_tokens=2000)
    except Exception as e:
        return f"Error en sintetizador: {e}\n\nJuridico: {juridico}\n\nTecnico: {tecnico}\n\nEstadistico: {estadistico}"


# === PUNTO DE ENTRADA ===

def run_moa(query: str, max_iterations: int = 5) -> dict:
    """
    Ejecutar MoA: 3 agentes especializados en paralelo + sintetizador.

    Los 3 agentes (Juridico, Tecnico, Estadistico) son independientes
    y se ejecutan en paralelo con ThreadPoolExecutor. El sintetizador
    espera a que todos terminen antes de combinar las perspectivas.

    Returns:
        dict con: final_answer, juridico, tecnico, estadistico
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    log.info("Iniciando 3 agentes especializados en paralelo...")

    agents = {
        "JURIDICO": (JURIDICO_PROMPT, JURIDICO_TOOLS),
        "TECNICO": (TECNICO_PROMPT, TECNICO_TOOLS),
        "ESTADISTICO": (ESTADISTICO_PROMPT, ESTADISTICO_TOOLS),
    }

    results = {}

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            name: pool.submit(
                _run_single_agent,
                name, prompt, tools, query, max_iterations,
            )
            for name, (prompt, tools) in agents.items()
        }

        for future in as_completed(futures):
            name_str = {f: n for n, f in futures.items()}[future]
            try:
                output = future.result()
                results[name_str] = output
                log.info("%s listo (%d chars)", name_str, len(output))
            except Exception as e:
                results[name_str] = f"[{name_str}] Error: {e}"
                log.error("%s fallo: %s", name_str, e)

    juridico = results.get("JURIDICO", "")
    tecnico = results.get("TECNICO", "")
    estadistico = results.get("ESTADISTICO", "")

    # Sintetizador (secuencial — depende de los 3 resultados)
    log.info("Sintetizando perspectivas...")
    final = _synthesize(query, juridico, tecnico, estadistico)
    log.info("Sintesis completa")

    return {
        "final_answer": final,
        "juridico": juridico,
        "tecnico": tecnico,
        "estadistico": estadistico,
    }
