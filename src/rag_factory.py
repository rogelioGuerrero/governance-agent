"""
RAG Factory: Sub-agente de curación sintáctica para ingesta masiva.

Problema: Los diccionarios de datos de 20-30 sistemas públicos vienen en
Exceles caóticos, PDFs de políticas, SQLs sin comentar. Ingestarlos
manualmente al grafo es un cuello de botella masivo.

Solución: Este sub-agente toma esquemas sucios, deduce tipos, limpia
basura conceptual ("Garbage In") y propone conexiones al grafo canónico.
El humano solo da "Un clic para autorizar" la inserción.

Pipeline:
1. EXTRACT: Lee el archivo (CSV, Excel, SQL DDL, JSON)
2. PROFILE: Deduce tipos, detecta nulos, valores imposibles
3. CLEAN: Normaliza nombres de columnas, elimina basura
4. MATCH: Propone mapeo a conceptos canónicos del nomenclador
5. PROPOSE: Genera un "plan de ingesta" para aprobación humana
6. INGEST: Ejecuta el plan (solo después de aprobación)

Usa Groq LLM para razonamiento semántico en las fases CLEAN y MATCH.
"""

import csv
import json
import logging
import re
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from .standards import detect_standard, STANDARDS, get_standard_values
from .graph.catalog import NomencladorGraph, load_graph_cached, clear_graph_cache
from .graph.schema import (
    ConceptNode, FieldNode, ClassifierNode, SourceNode, EdgeType,
)
from .llm_client import call_groq
from .inference import infer_semantic_type

logger = logging.getLogger(__name__)


@dataclass
class RawColumn:
    """Columna cruda extraída de un archivo sucio."""
    raw_name: str
    clean_name: str = ""
    data_type: str = ""
    sample_values: list[str] = field(default_factory=list)
    null_count: int = 0
    total_count: int = 0
    unique_count: int = 0
    notes: str = ""


@dataclass
class IngestionPlan:
    """Plan de ingesta propuesto por el RAG Factory."""
    source_name: str
    source_type: str  # csv, excel, sql, json
    columns: list[dict] = field(default_factory=list)
    proposed_mappings: list[dict] = field(default_factory=list)
    issues_found: list[str] = field(default_factory=list)
    cleanup_actions: list[str] = field(default_factory=list)
    confidence: str = "medium"  # low, medium, high
    requires_human_review: bool = True


# === FASE 1: EXTRACT ===

def extract_from_csv(file_path: str, max_rows: int = 100) -> list[RawColumn]:
    """Extraer columnas de un CSV sucio."""
    columns = []
    with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration:
            return []

        # Inicializar columnas
        for h in headers:
            columns.append(RawColumn(raw_name=h.strip()))

        # Leer muestras
        rows = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)

        # Perfilar cada columna
        for col_idx, col in enumerate(columns):
            values = []
            null_count = 0
            unique_vals = set()
            for row in rows:
                if col_idx < len(row):
                    val = row[col_idx].strip()
                    if not val or val.lower() in ("", "nan", "null", "none", "na", "n/a", "-"):
                        null_count += 1
                    else:
                        values.append(val)
                        unique_vals.add(val.lower())

            col.sample_values = values[:10]
            col.null_count = null_count
            col.total_count = len(rows)
            col.unique_count = len(unique_vals)
            col.data_type = _infer_type(values)

    return columns


def extract_from_sql_ddl(file_path: str) -> list[RawColumn]:
    """Extraer columnas de un DDL SQL (CREATE TABLE)."""
    with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
        content = f.read()

    columns = []
    # Buscar CREATE TABLE ... ( ... )
    pattern = r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\w\.\"`]+\s*\((.*?)\);"
    matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)

    for match in matches:
        lines = match.strip().split("\n")
        for line in lines:
            line = line.strip().rstrip(",")
            # Saltar constraints
            if re.match(r"(PRIMARY|FOREIGN|UNIQUE|CHECK|CONSTRAINT|KEY|INDEX)", line, re.IGNORECASE):
                continue
            # Parsear: column_name TYPE ...
            # Soportar tipos compuestos: VARCHAR(255), DECIMAL(10,2),
            # TIMESTAMP WITHOUT TIME ZONE, DOUBLE PRECISION, etc.
            col_match = re.match(
                r'([\w"`\[\]]+)\s+(\w+(?:\s+\w+)*)\s*[\(\,]',
                line + ',',
            )
            if col_match:
                col_name = col_match.group(1).strip('"`[]')
                col_type = col_match.group(2).upper()
                col = RawColumn(raw_name=col_name, data_type=col_type)
                columns.append(col)

    return columns


def extract_from_json_schema(file_path: str) -> list[RawColumn]:
    """Extraer columnas de un JSON Schema."""
    with open(file_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    columns = []
    properties = schema.get("properties", {})
    for prop_name, prop_def in properties.items():
        col = RawColumn(
            raw_name=prop_name,
            data_type=prop_def.get("type", "unknown"),
            notes=prop_def.get("description", ""),
        )
        if "enum" in prop_def:
            col.sample_values = [str(v) for v in prop_def["enum"][:10]]
        columns.append(col)

    return columns


# === FASE 2: TYPE INFERENCE ===

def _infer_type(values: list[str]) -> str:
    """Inferir tipo de dato de una lista de valores."""
    if not values:
        return "unknown"

    int_count = 0
    float_count = 0
    date_count = 0
    bool_count = 0

    for v in values[:50]:
        v = v.strip()
        try:
            int(v)
            int_count += 1
            continue
        except ValueError:
            pass
        try:
            float(v)
            float_count += 1
            continue
        except ValueError:
            pass
        if re.match(r"\d{4}-\d{2}-\d{2}", v) or re.match(r"\d{2}/\d{2}/\d{4}", v):
            date_count += 1
            continue
        if v.lower() in ("true", "false", "si", "no", "yes", "1", "0"):
            bool_count += 1
            continue

    total = len(values[:50])
    if int_count / total > 0.8:
        return "integer"
    if (int_count + float_count) / total > 0.8:
        return "float"
    if date_count / total > 0.7:
        return "date"
    if bool_count / total > 0.8:
        return "boolean"
    return "text"


# === FASE 3: CLEAN ===

# Palabras basura comunes en diccionarios institucionales
GARBAGE_PREFIXES = ["col_", "campo_", "fld_", "f_", "c_", "var_"]
GARBAGE_SUFFIXES = ["_id", "_cod", "_cod2", "_txt", "_str"]
ENCODING_FIXES = {
    "a~o": "ano",
    "Ã±": "n",
    "Ã¡": "a",
    "Ã©": "e",
    "Ã­": "i",
    "Ã³": "o",
    "Ãº": "u",
}


def clean_column_name(raw_name: str) -> str:
    """Limpiar un nombre de columna sucio."""
    name = raw_name.strip()

    # Fix encoding
    for bad, good in ENCODING_FIXES.items():
        name = name.replace(bad, good)

    # Remover prefijos basura
    for prefix in GARBAGE_PREFIXES:
        if name.lower().startswith(prefix):
            name = name[len(prefix):]
            break

    # Normalizar a lowercase + underscore
    name = re.sub(r"([A-Z])", r"_\1", name).lower().lstrip("_")
    name = re.sub(r"[^a-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")

    return name


def detect_issues(col: RawColumn) -> list[str]:
    """Detectar problemas en una columna cruda."""
    issues = []
    if col.null_count / max(col.total_count, 1) > 0.5:
        issues.append(f"Alta nulidad: {col.null_count}/{col.total_count} ({col.null_count/max(col.total_count,1)*100:.0f}%)")
    if col.unique_count == 1 and col.total_count > 5:
        issues.append(f"Columna constante: solo un valor '{col.sample_values[0] if col.sample_values else '?'}'")
    if col.unique_count == col.total_count and col.total_count > 0:
        issues.append("Posible clave primaria (todos valores unicos)")
    if col.data_type == "unknown":
        issues.append("Tipo de dato no detectado")
    if not col.raw_name.strip():
        issues.append("Columna sin nombre")
    if re.search(r"[^\x00-\x7F]", col.raw_name):
        issues.append(f"Encoding roto en nombre: '{col.raw_name}'")
    return issues


def _compute_quality_metrics(col_data: dict, mapping: dict) -> dict:
    """Calcular métricas de calidad estructuradas (PMBOK Quality Management).

    Retorna dict con: completeness, uniqueness, consistency, validity, quality_score.
    """
    total = max(col_data.get("total_count", 0), 1)
    null_count = col_data.get("null_count", 0)
    unique_count = col_data.get("unique_count", 0)
    sample_values = col_data.get("sample_values", [])
    standard_id = mapping.get("standard")

    # Completeness: % no nulos
    completeness = round((total - null_count) / total, 3)

    # Uniqueness: ratio unique/total (1.0 = todos únicos, 0.0 = todos duplicados)
    uniqueness = round(unique_count / total, 3)

    # Consistency: % valores que matchean el estándar (si hay estándar)
    consistency = 0.0
    if standard_id and sample_values:
        std_values = get_standard_values(standard_id)
        if std_values:
            std_set = set(str(v).strip().upper() for v in std_values)
            matching = sum(1 for v in sample_values if str(v).strip().upper() in std_set)
            consistency = round(matching / max(len(sample_values), 1), 3)
        else:
            consistency = 1.0  # sin estándar para validar, asumir neutro
    else:
        consistency = 1.0  # sin estándar, no se puede evaluar

    # Validity: % valores que pasan validación básica de formato
    validity = 1.0
    data_type = col_data.get("data_type", "")
    if data_type == "date" and sample_values:
        valid = sum(1 for v in sample_values if re.match(r"\d{4}-\d{2}-\d{2}", str(v)) or re.match(r"\d{2}/\d{2}/\d{4}", str(v)))
        validity = round(valid / max(len(sample_values), 1), 3)
    elif data_type == "integer" and sample_values:
        valid = sum(1 for v in sample_values if str(v).strip().lstrip("-").isdigit())
        validity = round(valid / max(len(sample_values), 1), 3)
    elif data_type == "float" and sample_values:
        valid = sum(1 for v in sample_values if _is_float(str(v)))
        validity = round(valid / max(len(sample_values), 1), 3)

    # Quality score: promedio ponderado
    quality_score = round(completeness * 0.4 + consistency * 0.3 + validity * 0.2 + uniqueness * 0.1, 3)

    return {
        "completeness": completeness,
        "uniqueness": uniqueness,
        "consistency": consistency,
        "validity": validity,
        "quality_score": quality_score,
    }


def _is_float(val: str) -> bool:
    try:
        float(val)
        return True
    except ValueError:
        return False


def _issues_to_structured(col_data: dict, issues: list[str]) -> list[dict]:
    """Convertir issues de texto a estructurados para persistir en el grafo."""
    structured = []
    for issue_text in issues:
        if "Alta nulidad" in issue_text:
            pct = col_data.get("null_count", 0) / max(col_data.get("total_count", 1), 1)
            structured.append({
                "issue_type": "alta_nulidad",
                "severity": "error" if pct > 0.7 else "warning",
                "detail": issue_text,
                "metric_value": round(pct, 3),
            })
        elif "Columna constante" in issue_text:
            structured.append({
                "issue_type": "columna_constante",
                "severity": "warning",
                "detail": issue_text,
                "metric_value": 1.0,
            })
        elif "clave primaria" in issue_text:
            structured.append({
                "issue_type": "clave_primaria",
                "severity": "info",
                "detail": issue_text,
                "metric_value": 1.0,
            })
        elif "Tipo de dato no detectado" in issue_text:
            structured.append({
                "issue_type": "tipo_no_detectado",
                "severity": "warning",
                "detail": issue_text,
                "metric_value": 0.0,
            })
        elif "Encoding roto" in issue_text:
            structured.append({
                "issue_type": "encoding_roto",
                "severity": "warning",
                "detail": issue_text,
                "metric_value": 0.0,
            })
        elif "sin nombre" in issue_text:
            structured.append({
                "issue_type": "sin_nombre",
                "severity": "error",
                "detail": issue_text,
                "metric_value": 0.0,
            })
        else:
            structured.append({
                "issue_type": "otro",
                "severity": "warning",
                "detail": issue_text,
                "metric_value": 0.0,
            })
    return structured


# === FASE 4: MATCH (con Groq LLM) ===

def _value_overlap_score(source_values: list[str], concept_values: list[str]) -> float:
    """Calcular overlap entre valores de origen y valores del concepto/clasificador.
    
    Retorna score 0.0-1.0 basado en:
    - Interseccion exacta (case-insensitive)
    - Ratio de valores origen que estan en el concepto
    """
    if not source_values or not concept_values:
        return 0.0
    src_set = set(str(v).strip().upper() for v in source_values if v)
    con_set = set(str(v).strip().upper() for v in concept_values if v)
    if not src_set or not con_set:
        return 0.0
    overlap = src_set & con_set
    return len(overlap) / len(src_set)


def _cardinality_compatible(col: RawColumn, concept: dict) -> bool:
    """Verificar si la cardinalidad de la columna es compatible con el concepto."""
    col_unique = col.unique_count
    concept_std = concept.get("standard", "")
    
    # Conceptos con clasificador (estandar) suelen tener baja cardinalidad
    if concept_std:
        # Si la columna tiene >1000 valores unicos y el concepto tiene estandar, probablemente no es match
        if col_unique > 1000:
            return False
        return True
    
    # Conceptos sin estandar: aceptar cualquier cardinalidad
    return True


def match_to_canonical(col: RawColumn, existing_concepts: list[dict], graph: 'NomencladorGraph' = None) -> dict:
    """
    Proponer mapeo de una columna a un concepto canonico del nomenclador.
    
    Estrategia en cascada:
    1. Deteccion por reglas (estandares registrados)
    2. Match exacto por nombre + compatibilidad de cardinalidad
    3. Match por distribucion de valores (overlap contra clasificador del concepto)
    4. Match parcial por nombre + score de valores
    5. Fallback: LLM Groq para razonamiento semantico
    """
    # 1. Deteccion por reglas (estandares conocidos)
    candidates = detect_standard(col.clean_name, col.sample_values)
    if candidates:
        best = candidates[0]
        return {
            "column": col.clean_name,
            "proposed_concept": _standard_to_concept(best["standard"]),
            "standard": best["standard"],
            "confidence": best["confidence"],
            "method": "rule_based",
            "reason": best["reason"],
        }

    # 2. Match exacto por nombre + cardinalidad compatible
    for concept in existing_concepts:
        if col.clean_name == concept.get("name", ""):
            if _cardinality_compatible(col, concept):
                return {
                    "column": col.clean_name,
                    "proposed_concept": concept["name"],
                    "standard": concept.get("standard"),
                    "confidence": "high",
                    "method": "exact_name_match",
                    "reason": f"Nombre coincide + cardinalidad compatible ({col.unique_count} unicos)",
                }
            else:
                return {
                    "column": col.clean_name,
                    "proposed_concept": concept["name"],
                    "standard": concept.get("standard"),
                    "confidence": "medium",
                    "method": "exact_name_match_cardinality_mismatch",
                    "reason": f"Nombre coincide pero cardinalidad dudosa ({col.unique_count} unicos vs estandar {concept.get('standard', '?')})",
                }

    # 3. Match por distribucion de valores contra clasificadores de conceptos
    if graph and col.sample_values:
        best_concept = None
        best_score = 0.0
        for concept in existing_concepts:
            concept_id = concept.get("id", f"concept:{concept['name']}")
            classifier = graph.find_classifier_of_concept(concept_id)
            if classifier:
                concept_values = list(classifier.get("values", {}).keys())
                score = _value_overlap_score(col.sample_values, concept_values)
                if score > best_score:
                    best_score = score
                    best_concept = concept
        
        if best_concept and best_score >= 0.5:
            return {
                "column": col.clean_name,
                "proposed_concept": best_concept["name"],
                "standard": best_concept.get("standard"),
                "confidence": "high" if best_score >= 0.8 else "medium",
                "method": "value_distribution_match",
                "reason": f"Overlap de valores {best_score:.0%} con clasificador del concepto '{best_concept['name']}'",
            }

    # 4. Match parcial por nombre + score de valores
    partial_matches = []
    for concept in existing_concepts:
        name_score = 0.0
        if col.clean_name in concept.get("name", "") or concept.get("name", "") in col.clean_name:
            name_score = 0.5
            # Bonus si los nombres comparten tokens
            src_tokens = set(col.clean_name.split("_"))
            con_tokens = set(concept.get("name", "").split("_"))
            if src_tokens & con_tokens:
                name_score = 0.7
        
        if name_score > 0:
            partial_matches.append((concept, name_score))
    
    # Ordenar por score de nombre y evaluar con valores si hay grafo
    if partial_matches:
        partial_matches.sort(key=lambda x: x[1], reverse=True)
        best_concept, name_score = partial_matches[0]
        
        # Si hay grafo, refinar con overlap de valores
        value_score = 0.0
        if graph and col.sample_values:
            concept_id = best_concept.get("id", f"concept:{best_concept['name']}")
            classifier = graph.find_classifier_of_concept(concept_id)
            if classifier:
                concept_values = list(classifier.get("values", {}).keys())
                value_score = _value_overlap_score(col.sample_values, concept_values)
        
        combined = name_score * 0.6 + value_score * 0.4
        if combined >= 0.5:
            return {
                "column": col.clean_name,
                "proposed_concept": best_concept["name"],
                "standard": best_concept.get("standard"),
                "confidence": "medium" if combined >= 0.6 else "low",
                "method": "partial_name_value_match",
                "reason": f"Match parcial nombre ({name_score:.0%}) + valores ({value_score:.0%}) = {combined:.0%}",
            }

    # 4.5 Inferencia semántica (sin LLM): patrones, listas de referencia, huella de valores
    if col.sample_values:
        existing_with_fields = []
        for concept in existing_concepts:
            cid = concept.get("id", f"concept:{concept['name']}")
            fields_data = []
            if graph:
                for f in graph.find_fields_of_concept(cid):
                    fields_data.append({"sample_values": f.get("sample_values", [])})
            existing_with_fields.append({**concept, "id": cid, "fields": fields_data})

        inf_result = infer_semantic_type(
            col.clean_name, col.sample_values, col.unique_count, existing_with_fields,
        )

        if inf_result.confidence in ("high", "medium"):
            concept_name = inf_result.matched_concept or inf_result.suggested_concept_name
            if concept_name:
                return {
                    "column": col.clean_name,
                    "proposed_concept": concept_name,
                    "standard": inf_result.suggested_standard_id,
                    "confidence": inf_result.confidence,
                    "method": f"inference_{inf_result.semantic_type or inf_result.reference_match or 'fingerprint'}",
                    "reason": inf_result.reason,
                }

    # 5. Fallback: sin mapeo
    return {
        "column": col.clean_name,
        "proposed_concept": None,
        "standard": None,
        "confidence": "low",
        "method": "no_match",
        "reason": "No se encontro concepto canonico. Requiere revision humana.",
    }


def _standard_to_concept(standard_id: str) -> Optional[str]:
    """Mapear estandar a nombre de concepto canonico.
    
    Usa los name_hints del estandar registrado dinamicamente.
    Solo retorna concepto para estandares de tipo "classifier".
    Estandares de tipo "format" (ej: ISO 8601) no representan conceptos semanticos.
    """
    std = STANDARDS.get(standard_id, {})
    if std.get("standard_type", "classifier") == "format":
        return None
    hints = std.get("name_hints", [])
    if hints:
        return hints[0].lower().replace(" ", "_")
    return None


def match_with_llm(col: RawColumn, existing_concepts: list[str]) -> dict:
    """
    Usar Groq LLM para razonamiento semantico cuando las reglas no encuentran match.
    """
    prompt = (
        f"Columna: {col.clean_name}\n"
        f"Tipo: {col.data_type}\n"
        f"Valores: {col.sample_values[:5]}\n"
        f"Conceptos existentes: {', '.join(existing_concepts[:20])}\n\n"
        f"Responde solo JSON: "
        f'{{"concept": "nombre_concepto_o_null", "standard": "id_estandar_o_null", "confidence": "low|medium|high", "reason": "breve"}}'
    )

    try:
        response = call_groq(
            [{"role": "user", "content": prompt}],
            max_tokens=200,
            json_mode=True,
        )
        # Parsear respuesta JSON — usar regex greedy para capturar JSON anidado
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                result = json.loads(json_match.group())
            except json.JSONDecodeError:
                # Fallback: intentar extraer el primer objeto JSON valido
                result = {"concept": None, "standard": None, "confidence": "low", "reason": "JSON no parseable"}
            result["method"] = "llm_groq"
            result["column"] = col.clean_name
            return result
    except Exception as e:
        logger.warning("match_with_llm fallo para columna %r: %s", col.clean_name, e)

    return {
        "column": col.clean_name,
        "proposed_concept": None,
        "standard": None,
        "confidence": "low",
        "method": "llm_failed",
        "reason": "LLM no disponible o respuesta no parseable",
    }


# === FASE 5: PROPOSE ===

def create_ingestion_plan(
    file_path: str,
    source_type: str = "csv",
    use_llm: bool = False,
) -> IngestionPlan:
    """
    Crear un plan de ingesta completo a partir de un archivo sucio.
    Este es el output principal del RAG Factory: el humano revisa y aprueba.
    """
    source_name = Path(file_path).stem

    # Fase 1: Extract
    if source_type == "csv":
        raw_columns = extract_from_csv(file_path)
    elif source_type == "sql":
        raw_columns = extract_from_sql_ddl(file_path)
    elif source_type == "json":
        raw_columns = extract_from_json_schema(file_path)
    else:
        raw_columns = extract_from_csv(file_path)

    if not raw_columns:
        return IngestionPlan(
            source_name=source_name,
            source_type=source_type,
            issues_found=["No se pudieron extraer columnas del archivo"],
        )

    # Cargar conceptos existentes del nomenclador (usar cache para reuso de conexion)
    g = load_graph_cached()
    existing_concepts = g.list_concepts()

    # Fase 3: Clean + detect issues
    all_issues = []
    cleanup_actions = []
    columns_data = []

    for col in raw_columns:
        col.clean_name = clean_column_name(col.raw_name)
        if col.clean_name != col.raw_name:
            cleanup_actions.append(f"Renombrado: '{col.raw_name}' -> '{col.clean_name}'")

        issues = detect_issues(col)
        all_issues.extend(issues)

        columns_data.append({
            "raw_name": col.raw_name,
            "clean_name": col.clean_name,
            "data_type": col.data_type,
            "sample_values": col.sample_values[:5],
            "null_count": col.null_count,
            "total_count": col.total_count,
            "unique_count": col.unique_count,
            "issues": issues,
        })

    # Fase 4: Match
    proposed_mappings = []
    for col in raw_columns:
        mapping = match_to_canonical(col, existing_concepts, graph=g)

        # Si no hay match y LLM está activado, intentar con Groq
        if mapping["proposed_concept"] is None and use_llm and col.sample_values:
            concept_names = [c["name"] for c in existing_concepts]
            llm_mapping = match_with_llm(col, concept_names)
            if llm_mapping.get("proposed_concept") or llm_mapping.get("concept"):
                mapping = llm_mapping
                mapping["proposed_concept"] = mapping.get("concept") or mapping.get("proposed_concept")

        proposed_mappings.append(mapping)

    # Determinar confianza general
    high_count = sum(1 for m in proposed_mappings if m.get("confidence") == "high")
    medium_count = sum(1 for m in proposed_mappings if m.get("confidence") == "medium")
    low_count = sum(1 for m in proposed_mappings if m.get("confidence") == "low")

    if high_count > len(proposed_mappings) * 0.6:
        confidence = "high"
    elif (high_count + medium_count) > len(proposed_mappings) * 0.5:
        confidence = "medium"
    else:
        confidence = "low"

    return IngestionPlan(
        source_name=source_name,
        source_type=source_type,
        columns=columns_data,
        proposed_mappings=proposed_mappings,
        issues_found=all_issues,
        cleanup_actions=cleanup_actions,
        confidence=confidence,
        requires_human_review=confidence != "high",
    )


# === FASE 6: INGEST (ejecutar plan aprobado) ===

def execute_ingestion_plan(plan: IngestionPlan, auto_confirm: bool = False) -> str:
    """
    Ejecutar un plan de ingesta aprobado.
    Solo se ejecuta despues de que el humano aprueba (o si auto_confirm=True).
    """
    nomenclador_path = Path(__file__).parent.parent / "nomenclador" / "nomenclador.json"
    g = load_graph_cached()

    # Registrar fuente
    source_id = f"source:{plan.source_name}"
    g.add_source(SourceNode(
        id=source_id,
        name=plan.source_name,
        description=f"Ingerido via RAG Factory ({plan.source_type})",
    ))

    # Inferir contexto
    from .cli import _infer_context
    ctx = _infer_context(plan.source_name)

    # Registrar cada columna con su mapeo
    for col_data, mapping in zip(plan.columns, plan.proposed_mappings):
        col_name = col_data["clean_name"]
        field_id = f"field:{plan.source_name}.{col_name}"

        # Calcular métricas de calidad
        quality = _compute_quality_metrics(col_data, mapping)

        g.add_field(FieldNode(
            id=field_id,
            source_db=plan.source_name,
            table=plan.source_name,
            column=col_name,
            data_type=col_data["data_type"],
            nullable=col_data["null_count"] > 0,
            unique_count=col_data["unique_count"],
            null_count=col_data["null_count"],
            total_count=col_data["total_count"],
            sample_values=col_data["sample_values"],
            inferred_standard=mapping.get("standard"),
            confidence=mapping.get("confidence", "low"),
            population=ctx["population"],
            capture_method=ctx["capture_method"],
            context_label=ctx["context_label"],
            completeness=quality["completeness"],
            uniqueness=quality["uniqueness"],
            consistency=quality["consistency"],
            validity=quality["validity"],
            quality_score=quality["quality_score"],
        ))
        g.link_fuente(field_id, source_id)

        # Persistir issues de calidad como nodos estructurados
        structured_issues = _issues_to_structured(col_data, col_data.get("issues", []))
        for si in structured_issues:
            g.add_quality_issue(
                field_id=field_id,
                issue_type=si["issue_type"],
                severity=si["severity"],
                detail=si["detail"],
                metric_value=si["metric_value"],
                detected_by="rag_factory",
            )

        # Si hay concepto canónico, vincular
        concept_name = mapping.get("proposed_concept")
        standard_id = mapping.get("standard")

        if concept_name:
            concept_id = f"concept:{concept_name}"
            if concept_id not in g.graph:
                std = STANDARDS.get(standard_id, {}) if standard_id else {}
                g.add_concept(ConceptNode(
                    id=concept_id,
                    name=concept_name,
                    definition=f"Variable canonica con estandar {std.get('name', standard_id or 'sin estandar')}",
                    standard=standard_id or "",
                    population=ctx["population"],
                    capture_method=ctx["capture_method"],
                ))
                if standard_id and std.get("values"):
                    classifier_id = f"classifier:{standard_id.lower()}"
                    if classifier_id not in g.graph:
                        g.add_classifier(ClassifierNode(
                            id=classifier_id,
                            name=std["name"],
                            standard=standard_id,
                            values=std["values"],
                        ))
                    g.link_clasificador(concept_id, classifier_id)

                # Auto-log: variable creada
                try:
                    from .lifecycle import log_event
                    log_event(concept_id, "created", actor="agent",
                              reason=f"Creada desde {plan.source_name} ({mapping.get('column', '')})",
                              details=f"standard={standard_id}, method={mapping.get('method', '')}")
                except Exception as e:
                    logger.warning("Lifecycle log_event fallo: %s", e)

            g.link_implementa(field_id, concept_id)

    # Guardar
    mapped = sum(1 for m in plan.proposed_mappings if m.get("proposed_concept"))
    unmapped = len(plan.proposed_mappings) - mapped
    g.bump_version("minor", f"Ingesta: {plan.source_name} ({mapped} conceptos)")
    g.save(str(nomenclador_path))

    # === FASE 7: NORMATIVE_BACKING ===
    backing_report = _normative_backing(g, plan)

    # Reporte de calidad
    quality_summary = g.get_quality_summary(plan.source_name)
    quality_report = (
        f"\n[bold]Calidad de datos:[/bold] "
        f"{quality_summary['total_fields']} campos | "
        f"score promedio: {quality_summary['avg_quality']} | "
        f"campos críticos (<0.5): {len(quality_summary['low_quality_fields'])} | "
        f"issues: {quality_summary['issues_by_severity']}"
    )

    parts = [
        f"Ingestion completa: {plan.source_name} | {mapped} mapeadas, {unmapped} sin mapear | {g.stats()['total_nodes']} nodos totales | v{g.version}",
    ]
    parts.append(quality_report)
    if backing_report:
        parts.append(backing_report)
    return "\n".join(parts)


def _normative_backing(graph: NomencladorGraph, plan: IngestionPlan) -> str:
    """
    Fase 7: Buscar respaldo normativo automatico para cada concepto mapeado.
    Busca en el corpus normativo (RAG documental) y vincula al grafo.
    """
    try:
        from .normative_rag import NormativeRAG
    except ImportError:
        return ""

    rag = NormativeRAG()
    if rag.stats()["total_chunks"] == 0:
        return ""

    concepts_to_check = []
    seen = set()
    for mapping in plan.proposed_mappings:
        name = mapping.get("proposed_concept")
        std = mapping.get("standard")
        if name and name not in seen:
            seen.add(name)
            concepts_to_check.append({"name": name, "standard": std})

    if not concepts_to_check:
        return ""

    results = rag.batch_backing(concepts_to_check)

    from .graph.schema import NormativeNode

    with_backing = 0
    without_backing = 0
    lines = ["\n[bold]Respaldo normativo:[/bold]"]

    for r in results:
        concept_id = f"concept:{r['concept']}"
        if r["found"] and r["references"]:
            best = r["references"][0]
            norm_id = f"normative:{r['concept']}:{best['source']}"
            if norm_id not in graph.graph:
                graph.add_normative(NormativeNode(
                    id=norm_id,
                    title=best["source"],
                    source=best["source"],
                    citation=best["cite"],
                    similarity_score=best["score"],
                    chunk_id=r["references"][0].get("id", ""),
                ))
            graph.link_normative(concept_id, norm_id)
            with_backing += 1
            lines.append(f"  [green]OK[/green] {r['concept']} <- {best['source']} (score: {best['score']})")

            # Auto-log: respaldo normativo adjuntado
            try:
                from .lifecycle import log_event
                log_event(concept_id, "normative_attached", actor="agent",
                          reason=f"Respaldo encontrado: {best['source']} (score: {best['score']:.3f})",
                          details=f"citation={best['cite'][:100]}")
            except Exception as e:
                logger.warning("Lifecycle log_event fallo: %s", e)
        else:
            without_backing += 1
            lines.append(f"  [yellow]--[/yellow] {r['concept']} sin respaldo normativo")

    nom_path = Path(__file__).parent.parent / "nomenclador" / "nomenclador.json"
    graph.save(str(nom_path))

    lines.append(f"  [dim]{with_backing} con respaldo | {without_backing} sin respaldo[/dim]")
    return "\n".join(lines)


def plan_to_dict(plan: IngestionPlan) -> dict:
    """Serializar plan a dict para mostrar al humano."""
    return {
        "source_name": plan.source_name,
        "source_type": plan.source_type,
        "confidence": plan.confidence,
        "requires_human_review": plan.requires_human_review,
        "columns": plan.columns,
        "proposed_mappings": plan.proposed_mappings,
        "issues_found": plan.issues_found,
        "cleanup_actions": plan.cleanup_actions,
        "summary": {
            "total_columns": len(plan.columns),
            "mapped": sum(1 for m in plan.proposed_mappings if m.get("proposed_concept")),
            "unmapped": sum(1 for m in plan.proposed_mappings if not m.get("proposed_concept")),
            "issues": len(plan.issues_found),
            "cleanups": len(plan.cleanup_actions),
        },
    }
