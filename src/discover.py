"""
Discover: generacion de hipotesis de politica publica desde los datos del grafo.

Flujo:
1. Recolectar todos los insights acumulados del grafo
2. Recolectar conceptos y quality scores por fuente
3. LLM genera hipotesis de problemas abordables con los datos disponibles
4. Para cada hipotesis, ejecuta analyze_policy_problem automaticamente
5. Retorna descubrimientos con factibilidad evaluada
"""

import json
import logging
from datetime import datetime, timezone

from .groq_client import call_groq
from .graph.catalog import NomencladorGraph
from .graph.schema import InsightNode
from .policy import analyze_policy_problem

log = logging.getLogger(__name__)

DISCOVER_PROMPT = """Eres un analista de datos institucional experto en politica publica.

Tu tarea es generar HIPOTESIS de problemas de politica publica que podrian abordarse con los datos disponibles en el nomenclador.

DATOS DISPONIBLES (insights acumulados por fuente):

{insights_summary}

CONCEPTOS EN EL GRAFO:

{concepts_summary}

CAMPOS DISPONIBLES (nombres reales en el grafo, usar EXACTAMENTE estos nombres en variables_used):

{fields_summary}

INSTRUCCIONES:
1. Analiza que variables estan disponibles y con que calidad
2. Identifica combinaciones de variables que permiten formular problemas reales
3. Piensa en problemas que un ministerio o agencia publica enfrentaria
4. Considera que variables de baja calidad limitan el analisis pero no lo imposibilitan
5. Genera entre 3 y 5 hipotesis, priorizando las que usen variables de mayor calidad
6. Cada hipotesis debe ser una narrativa de problema accionable (no una pregunta vaga)
7. En variables_used usa EXACTAMENTE los nombres de campos listados arriba cuando existan

Devuelve SOLO un JSON con esta estructura:
{{
  "hypotheses": [
    {{
      "title": "titulo corto del problema",
      "narrative": "descripcion del problema en 2-3 frases, como lo plantearia un tomador de decisiones",
      "variables_used": ["edad", "sexo", "diagnostico"],
      "feasibility_hint": "ALTA | MEDIA | BAJA",
      "rationale": "por que estos datos permiten abordar este problema"
    }}
  ]
}}
"""


def _strip_reasoning(raw: str) -> str:
    """Quitar tokens de reasoning si el LLM los incluye."""
    raw = raw.strip()
    if raw.startswith("<think>"):
        end = raw.find("</think>")
        if end != -1:
            raw = raw[end + 8:].strip()
    return raw


def _parse_json(raw: str) -> dict:
    """Parsear JSON con fallback a regex."""
    raw = _strip_reasoning(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        raise ValueError(f"LLM no devolvio JSON valido. Primeros 500 chars: {raw[:500]}")


def _build_insights_summary(graph: NomencladorGraph) -> str:
    """Construir resumen de insights acumulados para el prompt."""
    insights = graph.find_insights()
    if not insights:
        return "(No hay insights acumulados todavia. Usando solo conceptos del grafo.)"

    lines = []
    for ins in insights:
        source = ins.get("source_id", "?")
        domain = ins.get("domain", "?")
        obs = ins.get("observation", "")
        variables = ins.get("variables_covered", [])
        potential = ins.get("cross_source_potential", "")
        qs = ins.get("quality_snapshot", {})

        lines.append(
            f"- [{source}] (dominio: {domain})\n"
            f"  Observacion: {obs}\n"
            f"  Variables: {', '.join(variables) if variables else 'N/A'}\n"
            f"  Calidad promedio: {qs.get('avg_qs', 'N/A')}\n"
            f"  Potencial cross-source: {potential}"
        )
    return "\n".join(lines)


def _build_fields_summary(graph: NomencladorGraph) -> str:
    """Construir lista de fields disponibles con sus nombres reales y fuentes.

    Esto permite al LLM usar los nombres exactos de los campos en variables_used,
    evitando que proponga variables en otro idioma o convencion que no matchean.
    """
    concepts = graph.find_all_concepts()
    lines = []
    for c in concepts:
        context = graph.build_concept_context(c.get("id", ""))
        fields = context.get("fields", [])
        if not fields:
            continue
        concept_name = c.get("name", "?")
        field_names = [f.get("name", "?") for f in fields]
        sources = list(set(f.get("source_db", "") for f in fields if f.get("source_db")))
        lines.append(
            f"- {concept_name}: {', '.join(field_names)} "
            f"(fuentes: {', '.join(sources) if sources else 'ninguna'})"
        )
    return "\n".join(lines) if lines else "(sin fields disponibles)"


def _build_concepts_summary(graph: NomencladorGraph) -> str:
    """Construir resumen de conceptos con calidad por fuente."""
    concepts = graph.find_all_concepts()
    lines = []
    for c in concepts:
        name = c.get("name", "?")
        standard = c.get("standard") or "sin estandar"
        definition = c.get("definition", "")[:100]
        context = graph.build_concept_context(c.get("id", ""))
        fields = context.get("fields", [])
        quality = context.get("quality_summary", {})
        sources = list(set(f.get("source_db", "") for f in fields if f.get("source_db")))

        lines.append(
            f"- {name} (estandar: {standard}): {definition}\n"
            f"  Fuentes: {', '.join(sources) if sources else 'ninguna'}. "
            f"Fields: {quality.get('total_fields', 0)}. "
            f"Calidad promedio: {quality.get('avg_score', 0.0):.2f}."
        )
    return "\n".join(lines)


def generate_insights_for_source(
    graph: NomencladorGraph,
    source_id: str,
    domain: str = "",
    profile_data: list[dict] | None = None,
) -> list[dict]:
    """Generar y guardar insights para una fuente especifica del grafo.

    Usa el LLM para analizar los fields de una fuente y generar observaciones
    acumulables. Los insights se guardan como nodos en el grafo.

    Args:
        graph: Knowledge Graph
        source_id: ID del source (ej: "source:hospital")
        domain: dominio de la fuente (ej: "salud")
        profile_data: profiling real de CSVs (lista de dicts con column, data_type,
                      total_count, null_count, unique_count, sample_values, min_value,
                      max_value). Si se provee, los insights se basan en datos reales.

    Returns:
        Lista de insights generados y guardados
    """
    if source_id not in graph.graph:
        log.warning("generate_insights: source %s no existe en el grafo", source_id)
        return []

    # Recolectar fields de esta fuente
    source_name = graph.graph.nodes[source_id].get("name", source_id)
    all_fields = graph.list_fields()

    # Los fields se vinculan al source via arista PROVIENE_DE
    source_fields = []
    for f in all_fields:
        field_id = f.get("id", "")
        if field_id in graph.graph:
            for successor in graph.graph.successors(field_id):
                edge = graph.graph.get_edge_data(field_id, successor)
                if edge and edge.get("type") == "proviene_de" and successor == source_id:
                    # Buscar el concepto que implementa
                    concept_name = ""
                    for concept_target in graph.graph.successors(field_id):
                        concept_edge = graph.graph.get_edge_data(field_id, concept_target)
                        if concept_edge and concept_edge.get("type") == "implementa":
                            concept_name = graph.graph.nodes[concept_target].get("name", "")
                            break
                    source_fields.append({
                        "column": f.get("column", ""),
                        "concept": concept_name,
                        "quality_score": f.get("quality_score", 0.0),
                        "completeness": f.get("completeness", 0.0),
                    })
                    break

    if not source_fields:
        log.warning("generate_insights: source %s no tiene fields", source_id)
        return []

    # Construir resumen para el LLM
    # Si hay profiling real, incluirlo; si no, usar solo quality scores
    profile_map = {}
    if profile_data:
        for p in profile_data:
            profile_map[p.get("column", "")] = p

    fields_lines = []
    for f in source_fields:
        col = f["column"]
        base = f"- {col} -> concepto: {f['concept']}, qs={f['quality_score']:.2f}, completeness={f['completeness']:.2f}"
        p = profile_map.get(col)
        if p:
            null_pct = (p["null_count"] / p["total_count"] * 100) if p["total_count"] else 0
            base += f"\n    DATOS REALES: {p['total_count']} filas, {p['unique_count']} unicos, {null_pct:.1f}% nulos"
            base += f", tipo: {p['data_type']}"
            if p.get("min_value") and p.get("max_value"):
                base += f", rango: [{p['min_value']} .. {p['max_value']}]"
            if p.get("sample_values"):
                samples = p["sample_values"][:8]
                base += f", muestras: {samples}"
        fields_lines.append(base)
    fields_text = "\n".join(fields_lines)

    if profile_data:
        INSIGHT_GEN_PROMPT = """Eres un analista de datos institucional.

Analiza los campos de la fuente "{source_name}" (dominio: {domain}) y genera insights acumulables.

IMPORTANTE: Los datos a continuacion incluyen PROFILING REAL de los CSVs. Usa las estadisticas reales
(total de filas, nulos, unicos, rangos, muestras) para generar observaciones basadas en evidencia, no suposiciones.

CAMPOS DE LA FUENTE (con profiling real):
{fields_text}

Para cada insight, describe:
- observation: hallazgo basado en los DATOS REALES (ej: "X% de nulos en columna Y", "distribucion de Z muestra...", "rango de valores en W es...")
- variables_covered: que conceptos cubre esta fuente
- cross_source_potential: que variables de esta fuente podrian combinarse con otras fuentes para generar nuevos analisis

Devuelve SOLO un JSON:
{{
  "insights": [
    {{
      "observation": "descripcion del hallazgo basada en datos reales",
      "variables_covered": ["var1", "var2"],
      "cross_source_potential": "que combinaciones permite"
    }}
  ]
}}
"""
    else:
        INSIGHT_GEN_PROMPT = """Eres un analista de datos institucional.

Analiza los campos de la fuente "{source_name}" (dominio: {domain}) y genera insights acumulables.

CAMPOS DE LA FUENTE:
{fields_text}

Para cada insight, describe:
- observation: que patrones o caracteristicas notables tienen los datos
- variables_covered: que conceptos cubre esta fuente
- cross_source_potential: que variables de esta fuente podrian combinarse con otras fuentes para generar nuevos analisis

Devuelve SOLO un JSON:
{{
  "insights": [
    {{
      "observation": "descripcion del hallazgo",
      "variables_covered": ["var1", "var2"],
      "cross_source_potential": "que combinaciones permite"
    }}
  ]
}}
"""
    log.info("generate_insights: analizando fuente %s con %d fields", source_id, len(source_fields))
    response = call_groq(
        messages=[
            {"role": "system", "content": "Eres un analista de datos. Respondes SOLO en JSON."},
            {"role": "user", "content": INSIGHT_GEN_PROMPT.format(
                source_name=source_name,
                domain=domain or "general",
                fields_text=fields_text,
            )},
        ],
        temperature=0.3,
        max_tokens=2000,
        json_mode=True,
    )

    parsed = _parse_json(response)
    raw_insights = parsed.get("insights", [])

    # Calcular quality snapshot
    scores = [f["quality_score"] for f in source_fields]
    avg_qs = round(sum(scores) / len(scores), 3) if scores else 0.0
    low_q = sum(1 for s in scores if s < 0.4)
    quality_snapshot = {"avg_qs": avg_qs, "field_count": len(source_fields), "low_quality_count": low_q}

    now = datetime.now(timezone.utc).isoformat()
    saved = []
    for i, ri in enumerate(raw_insights):
        insight_id = f"insight:{source_id.split(':')[-1]}:{i+1:03d}"
        node = InsightNode(
            id=insight_id,
            source_id=source_id,
            domain=domain,
            observation=ri.get("observation", ""),
            variables_covered=ri.get("variables_covered", []),
            quality_snapshot=quality_snapshot,
            cross_source_potential=ri.get("cross_source_potential", ""),
            created_at=now,
        )
        graph.add_insight(node)
        saved.append({"id": insight_id, **node.model_dump()})

    log.info("generate_insights: %d insights guardados para %s", len(saved), source_id)
    return saved


def discover(graph: NomencladorGraph, domain: str = "", auto_analyze: bool = True) -> dict:
    """Descubrir problemas de politica publica abordables con los datos del grafo.

    1. Recolecta insights acumulados + conceptos del grafo
    2. LLM genera hipotesis de problemas
    3. Para cada hipotesis, evalua factibilidad con analyze_policy_problem

    Args:
        graph: Knowledge Graph
        domain: filtrar por dominio (ej: "salud"). Vacio = todos.
        auto_analyze: si True, ejecuta analyze_policy_problem para cada hipotesis

    Returns:
        dict con:
        - hypotheses: lista de hipotesis generadas
        - discoveries: lista de resultados de analyze_policy_problem (si auto_analyze)
        - insights_count: numero de insights acumulados usados
    """
    insights = graph.find_insights(domain=domain if domain else None)
    log.info("discover: %d insights acumulados, dominio=%s", len(insights), domain or "todos")

    insights_summary = _build_insights_summary(graph)
    concepts_summary = _build_concepts_summary(graph)
    fields_summary = _build_fields_summary(graph)

    log.info("discover: generando hipotesis con LLM...")
    response = call_groq(
        messages=[
            {"role": "system", "content": "Eres un analista de datos institucional experto en politica publica. Respondes SOLO en JSON."},
            {"role": "user", "content": DISCOVER_PROMPT.format(
                insights_summary=insights_summary,
                concepts_summary=concepts_summary,
                fields_summary=fields_summary,
            )},
        ],
        temperature=0.4,
        max_tokens=3000,
        json_mode=True,
    )

    parsed = _parse_json(response)
    hypotheses = parsed.get("hypotheses", [])

    log.info("discover: %d hipotesis generadas", len(hypotheses))

    discoveries = []
    if auto_analyze:
        for i, hyp in enumerate(hypotheses):
            narrative = hyp.get("narrative", "")
            if not narrative:
                continue
            log.info("discover: analizando hipotesis %d/%d: %s", i + 1, len(hypotheses), hyp.get("title", "?"))
            try:
                result = analyze_policy_problem(narrative, graph)
                # Calcular score de priorizacion
                cov = result.get("coverage", {})
                cov_pct = cov.get("percentage", 0) / 100.0
                variables = result.get("variables", [])
                avail_vars = [v for v in variables if v.get("status") == "available"]
                avg_qs = (
                    sum(v.get("quality_score", 0.0) for v in avail_vars) / len(avail_vars)
                    if avail_vars else 0.0
                )
                # Detectar cross-domain: variables provienen de fuentes de dominios distintos
                all_source_domains = set()
                for v in avail_vars:
                    for src in v.get("sources", []):
                        all_source_domains.add(src)
                cross_domain = len(all_source_domains) > 1
                impact_factor = 1.5 if cross_domain else 1.0
                # Bonus por numero de variables (mas complejo = mas impacto potencial)
                var_bonus = min(len(variables) / 10.0, 0.3)
                score = round(cov_pct * avg_qs * impact_factor + var_bonus, 3)

                discoveries.append({
                    "title": hyp.get("title", ""),
                    "feasibility_hint": hyp.get("feasibility_hint", ""),
                    "rationale": hyp.get("rationale", ""),
                    "analysis": result,
                    "score": score,
                    "cross_domain": cross_domain,
                    "avg_quality": round(avg_qs, 2),
                    "coverage_pct": cov.get("percentage", 0),
                })
            except Exception as e:
                log.error("discover: error analizando hipotesis %d: %s", i + 1, e)
                discoveries.append({
                    "title": hyp.get("title", ""),
                    "feasibility_hint": hyp.get("feasibility_hint", ""),
                    "rationale": hyp.get("rationale", ""),
                    "error": str(e),
                    "score": 0.0,
                    "cross_domain": False,
                    "avg_quality": 0.0,
                    "coverage_pct": 0,
                })

    # Ordenar discoveries por score descendente (priorizacion)
    discoveries.sort(key=lambda d: d.get("score", 0.0), reverse=True)

    return {
        "hypotheses": hypotheses,
        "discoveries": discoveries,
        "insights_count": len(insights),
    }


DEEP_DIVE_PROMPT = """Eres un analista de datos senior especializado en politica publica.

Se te da una HIPOTESIS de problema de politica publica y el analisis de factibilidad de datos.

Tu tarea es generar un PLAN DE ANALISIS accionable: pasos concretos que un equipo de datos ejecutaria para responder a la hipotesis.

HIPOTESIS:
- Titulo: {title}
- Narrativa: {narrative}

ANALISIS DE FACTIBILIDAD:
- Cobertura: {coverage_pct}% ({coverage_detail})
- Variables disponibles: {variables_detail}

CONCEPTOS Y FUENTES DEL GRAFO:
{concepts_summary}

INSTRUCCIONES:
1. Genera entre 4 y 7 pasos secuenciales y concretos
2. Cada paso debe tener: titulo, descripcion, fuentes involucradas, tipo de operacion (join, agregacion, filtro, visualizacion, modelo)
3. Cuando sea relevante, sugiere el JOIN concreto entre fuentes (campo de enlace)
4. Estima el esfuerzo de cada paso (bajo/medio/alto) en funcion de la calidad de los datos
5. Identifica riesgos o limitaciones (datos faltantes, calidad baja, sesgo potencial)
6. Sugiere que visualizacion o metrica final responderia a la hipotesis

Devuelve SOLO un JSON con esta estructura:
{{
  "plan_title": "titulo del plan",
  "summary": "resumen ejecutivo en 2-3 frases",
  "steps": [
    {{
      "step": 1,
      "title": "titulo del paso",
      "description": "que hacer y como",
      "sources": ["hospital", "censo"],
      "operation": "join | agregacion | filtro | visualizacion | modelo",
      "join_hint": "campo de enlace entre fuentes (si aplica)",
      "effort": "bajo | medio | alto",
      "effort_reason": "por que este nivel de esfuerzo"
    }}
  ],
  "risks": ["riesgo 1", "riesgo 2"],
  "final_output": "que metrica/visualizacion/grafico responderia a la hipotesis",
  "estimated_total_effort": "bajo | medio | alto",
  "estimated_time": "estimacion en dias-persona (numero)"
}}
"""


def deep_dive(
    graph: NomencladorGraph,
    title: str,
    narrative: str = "",
    analysis: dict | None = None,
) -> dict:
    """Generar un plan de analisis accionable para una hipotesis.

    Args:
        graph: Knowledge Graph con conceptos y fuentes
        title: titulo de la hipotesis
        narrative: narrativa del problema (opcional si ya esta en analysis)
        analysis: resultado de analyze_policy_problem (opcional, se re-analiza si no se provee)

    Returns:
        Plan de analisis con steps, risks, final_output, effort
    """
    if analysis is None:
        analysis = analyze_policy_problem(narrative or title, graph)

    cov = analysis.get("coverage", {})
    variables = analysis.get("variables", [])
    avail_vars = [v for v in variables if v.get("status") == "available"]

    variables_detail = "\n".join(
        f"  - {v['name']}: fuentes={v.get('sources', [])}, calidad={v.get('quality_score', 0):.2f}"
        for v in avail_vars
    ) or "  (ninguna variable disponible)"

    coverage_detail = f"{cov.get('available', 0)}/{cov.get('total', 0)} variables"
    concepts_summary = _build_concepts_summary(graph)

    prompt = DEEP_DIVE_PROMPT.format(
        title=title,
        narrative=narrative or "(ver titulo)",
        coverage_pct=cov.get("percentage", 0),
        coverage_detail=coverage_detail,
        variables_detail=variables_detail,
        concepts_summary=concepts_summary[:3000],
    )

    log.info("Deep-dive: generando plan para '%s'", title)
    response = call_groq(
        messages=[
            {"role": "system", "content": "Eres un analista de datos senior especializado en politica publica. Respondes SOLO en JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        max_tokens=2000,
        json_mode=True,
    )
    result = _parse_json(response)

    result["hypothesis_title"] = title
    result["coverage_pct"] = cov.get("percentage", 0)
    result["variables_count"] = len(avail_vars)
    return result
