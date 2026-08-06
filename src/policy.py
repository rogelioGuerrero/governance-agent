"""
Policy Problem Analyzer.

Dado una narrativa de problema de politica publica:
1. LLM extrae variables requeridas (con razonamiento, no regex)
2. Para cada variable, busca en el Knowledge Graph
3. Clasifica: existe con datos / existe sin datos / gap
4. Sintetiza reporte de factibilidad basado en datos
"""

import json
import logging
from .llm_client import call_groq
from .graph.catalog import NomencladorGraph, load_graph_cached

log = logging.getLogger(__name__)

EXTRACT_PROMPT = """Eres un experto en politica publica y gestion de datos institucional.

Dado un problema de politica publica, tu tarea es identificar las VARIABLES DE DATOS requeridas para abordarlo.

Una variable de datos es una pieza de informacion que se captura en un sistema (ej: "edad", "municipio de residencia", "estado de vacunacion").

Para cada variable, debes inferir:
- name: nombre corto de la variable (snake_case, en espanol)
- description: que mide o representa
- why_needed: por que es necesaria para este problema especifico
- data_type_hint: tipo de dato probable (texto, numero, fecha, categorico)
- possible_standards: estandares conocidos que podrian aplicar (ej: ISO 8601, ICD-10, ONU, DINAP)

Reglas:
- Identifica entre 3 y 8 variables (no mas, no menos)
- Solo variables que realmente se necesitan para el problema, no genericas
- Si el problema menciona una poblacion, incluye variables para identificarla
- Si el problema menciona una decision, incluye variables para soportarla
- Piensa en variables de contexto (ubicacion, tiempo) y variables sustantivas

Devuelve SOLO un JSON con esta estructura:
{{
  "problem_summary": "reformulacion del problema en 1-2 frases",
  "variables": [
    {{
      "name": "edad",
      "description": "Edad del individuo en anos",
      "why_needed": "Permite segmentar la poblacion objetivo por grupos etarios",
      "data_type_hint": "numero",
      "possible_standards": []
    }}
  ]
}}

Problema:
\"\"\"
{narrative}
\"\"\"
"""


FEASIBILITY_PROMPT = """Eres un analista de datos institucional. Tu tarea es sintetizar un reporte de factibilidad.

PROBLEMA:
{problem_summary}

VARIABLES REQUERIDAS Y SU ESTADO EN EL NOMENCLADOR:

{variables_status}

INSTRUCCIONES:
1. Escribe un reporte en lenguaje claro (no tecnico) para un tomador de decisiones
2. Para cada variable, indica si existe, donde se captura, y si es utilizable
3. Identifica que porcentaje del problema se puede abordar con los datos actuales
4. Para los gaps, sugiere que instrumento o sistema podria capturar esa variable
5. Termina con una recomendacion: ES POSIBLE / ES POSIBLE CON AJUSTES / NO ES POSIBLE todavia

Formato del reporte:
## Factibilidad: [titulo del problema]

### Variables disponibles
- [variable]: [donde se captura, calidad, utilizable?]

### Gaps de datos
- [variable faltante]: [que se necesitaria para capturarla]

### Cobertura de datos
X de N variables disponibles ({{porcentaje}}%)

### Recomendacion
[ES POSIBLE / ES POSIBLE CON AJUSTES / NO ES POSIBLE todavia]
[Explicacion en 2-3 frases]
"""


def analyze_policy_problem(narrative: str, graph: NomencladorGraph) -> dict:
    """Analizar un problema de politica publica y evaluar factibilidad de datos.

    Args:
        narrative: Descripcion del problema en lenguaje natural
        graph: Knowledge Graph del nomenclador

    Returns:
        dict con:
        - problem_summary: reformulacion del problema
        - variables: lista de variables requeridas con estado
        - report: reporte de factibilidad en texto
        - coverage: {total, available, gaps, percentage}
    """
    # Fase 1: Extraer variables del problema
    log.info("Policy analyzer: extrayendo variables del problema...")
    extract_response = call_groq(
        messages=[
            {"role": "system", "content": "Eres un experto en politica publica y gestion de datos. Respondes SOLO en JSON."},
            {"role": "user", "content": EXTRACT_PROMPT.format(narrative=narrative)},
        ],
        temperature=0.2,
        max_tokens=3000,
        json_mode=True,
    )

    # gpt-oss-120b puede incluir reasoning tokens antes del JSON
    raw = extract_response.strip()
    # Quitar prefijo de reasoning si existe
    if raw.startswith("<think>"):
        end_think = raw.find("</think>")
        if end_think != -1:
            raw = raw[end_think + 8:].strip()

    try:
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        # Intentar extraer JSON de la respuesta
        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            try:
                extracted = json.loads(match.group())
            except json.JSONDecodeError:
                raise ValueError(f"LLM no devolvio JSON valido. Respuesta (primeros 500 chars): {extract_response[:500]}")
        else:
            raise ValueError(f"LLM no devolvio JSON valido. Respuesta (primeros 500 chars): {extract_response[:500]}")

    problem_summary = extracted.get("problem_summary", narrative)
    required_vars = extracted.get("variables", [])

    log.info("Policy analyzer: %d variables extraidas", len(required_vars))

    # Fase 2: Buscar cada variable en el grafo
    all_concepts = graph.find_all_concepts()
    concept_names = {c.get("name", "").lower(): c for c in all_concepts}

    # Build field name index for semantic matching (Capa 2)
    # Maps field_name -> concept that owns it
    field_to_concept = {}
    for c in all_concepts:
        ctx = graph.build_concept_context(c.get("id", ""))
        for f in ctx.get("fields", []):
            fname = f.get("name", "").lower()
            if fname:
                field_to_concept[fname] = c

    # Construir indice de palabras clave por concepto para matching flexible
    concept_keywords = {}
    for c in all_concepts:
        cname = c.get("name", "").lower()
        words = set(cname.replace("_", " ").split())
        # Agregar sinonimos simples
        synonyms = {
            "edad": {"nino", "anos", "age"},
            "sexo": {"genero", "gender", "masculino", "femenino"},
            "diagnostico": {"cie", "cie10", "enfermedad", "morbilidad", "diag"},
            "fecha_nacimiento": {"fecha_nac", "nacimiento", "birth"},
            "fecha_ingreso": {"f_ingreso", "ingreso", "admission"},
            "municipio": {"municipalidad", "distrito", "ciudad", "residencia"},
            "nivel_educativo": {"escolaridad", "educacion", "instruccion", "education"},
            "nombre_completo": {"nombre", "name"},
            "ocupacion": {"trabajo", "profesion", "occupation"},
            "pais": {"country", "nacionalidad"},
            "year": {"ano", "fecha", "periodo"},
            "valor": {"value", "magnitud", "magnitude"},
            "indicador": {"indicator", "metrica", "metric"},
        }
        for key, syns in synonyms.items():
            if key in cname:
                words.update(syns)
        # Agregar palabras de la definicion
        defn = c.get("definition", "").lower()
        if defn:
            words.update(defn.replace("_", " ").split())
        concept_keywords[c.get("id", "")] = words

    variables_status = []
    available_count = 0

    for var in required_vars:
        var_name = var.get("name", "").lower()
        var_desc = var.get("description", "").lower()

        # Buscar por nombre exacto
        matched_concept = concept_names.get(var_name)

        # Buscar por nombre parcial si no hay match exacto
        if not matched_concept:
            for cname, cdata in concept_names.items():
                if var_name in cname or cname in var_name:
                    matched_concept = cdata
                    break

        # Buscar por field name exacto (Capa 2: matching por campo, no por concepto)
        if not matched_concept:
            for fname, fconcept in field_to_concept.items():
                if var_name == fname:
                    matched_concept = fconcept
                    break

        # Buscar por field name con overlap de palabras (no substring suelto)
        if not matched_concept:
            var_words = set(var_name.replace("_", " ").split())
            for fname, fconcept in field_to_concept.items():
                fname_words = set(fname.replace("_", " ").split())
                if var_words & fname_words and len(var_words & fname_words) >= min(len(var_words), len(fname_words)):
                    matched_concept = fconcept
                    break

        # Buscar por overlap de palabras clave
        if not matched_concept:
            var_words = set(var_name.replace("_", " ").split())
            var_words.update(var_desc.replace("_", " ").split())
            best_score = 0
            best_concept = None
            for cid, cwords in concept_keywords.items():
                if not cwords or not var_words:
                    continue
                overlap = len(var_words & cwords)
                score = overlap / max(len(var_words), 1)
                if score > best_score and overlap >= 1:
                    best_score = score
                    best_concept = next((c for c in all_concepts if c.get("id") == cid), None)
            if best_concept and best_score >= 0.15:
                matched_concept = best_concept

        if matched_concept:
            # Construir contexto del concepto
            concept_id = matched_concept.get("id", "")
            context = graph.build_concept_context(concept_id)

            fields = context.get("fields", [])
            quality = context.get("quality_summary", {})
            sources = list(set(f.get("source_db", "") for f in fields if f.get("source_db")))

            status = {
                "name": var.get("name"),
                "description": var_desc,
                "why_needed": var.get("why_needed", ""),
                "status": "available",
                "concept_id": concept_id,
                "concept_name": matched_concept.get("name", ""),
                "definition": matched_concept.get("definition", ""),
                "standard": matched_concept.get("standard"),
                "sources": sources,
                "field_count": len(fields),
                "quality_score": quality.get("avg_score", 0.0),
                "low_quality": quality.get("low_quality_count", 0),
                "review_status": matched_concept.get("review_status", "approved"),
                "usable": (
                    len(fields) > 0
                    and quality.get("avg_score", 0.0) >= 0.4
                    and matched_concept.get("review_status", "approved") not in ("proposed", "rejected")
                ),
            }
            available_count += 1
        else:
            status = {
                "name": var.get("name"),
                "description": var_desc,
                "why_needed": var.get("why_needed", ""),
                "status": "gap",
                "data_type_hint": var.get("data_type_hint", ""),
                "possible_standards": var.get("possible_standards", []),
            }

        variables_status.append(status)

    log.info("Policy analyzer: %d/%d variables disponibles", available_count, len(required_vars))

    # Fase 3: Sintetizar reporte de factibilidad
    coverage = {
        "total": len(required_vars),
        "available": available_count,
        "gaps": len(required_vars) - available_count,
        "percentage": round(available_count / len(required_vars) * 100) if required_vars else 0,
    }

    # Formatear variables para el prompt de sintesis
    status_lines = []
    for v in variables_status:
        if v["status"] == "available":
            usable_label = "UTILIZABLE" if v["usable"] else "NO UTILIZABLE (calidad baja o en revision)"
            sources_str = ", ".join(v["sources"]) if v["sources"] else "sin fuentes"
            status_lines.append(
                f"- {v['name']}: EXISTE. Concepto: {v['concept_name']}. "
                f"Fuentes: {sources_str}. Fields: {v['field_count']}. "
                f"Quality: {v['quality_score']:.2f}. {usable_label}."
            )
        else:
            standards_str = ", ".join(v.get("possible_standards", [])) or "ninguno"
            status_lines.append(
                f"- {v['name']}: GAP (no existe en el nomenclador). "
                f"Tipo: {v.get('data_type_hint', '?')}. "
                f"Estandares sugeridos: {standards_str}."
            )

    variables_text = "\n".join(status_lines)

    log.info("Policy analyzer: sintetizando reporte...")
    report = call_groq(
        messages=[
            {"role": "system", "content": "Eres un analista de datos institucional. Escribes reportes claros para tomadores de decisiones."},
            {"role": "user", "content": FEASIBILITY_PROMPT.format(
                problem_summary=problem_summary,
                variables_status=variables_text,
            )},
        ],
        temperature=0.3,
        max_tokens=2000,
    )

    return {
        "problem_summary": problem_summary,
        "variables": variables_status,
        "coverage": coverage,
        "report": report,
    }
