"""
Generador de Esquemas de Transformación.

El output final del agente cuando encuentra un camino válido en el grafo
para conectar DB1 con DB2 es un artefacto ejecutable:

1. SQL transform (CASE WHEN) para alinear valores al estándar canónico
2. JSON Schema de validación por variable canónica
3. Mapeo declarativo (campo origen -> campo canónico + transformación)
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from .standards import STANDARDS, get_standard_values
from .guardrails import InteropValidation, CheckpointStatus

logger = logging.getLogger(__name__)


def _escape_identifier(identifier: str) -> str:
    """Escapar un identificador SQL (nombre de columna/tabla) para evitar inyeccion.
    
    Permite solo alphanumeric + underscore, envuelve en comillas dobles.
    """
    if not identifier or not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', identifier):
        logger.warning("SQL identifier invalido: %r — usando placeholder", identifier)
        return '"invalid_identifier"'
    return f'"{identifier}"'


def _escape_literal(value: str) -> str:
    """Escapar un literal SQL (valor de string) para evitar inyeccion.
    
    Escapa comillas simples duplicandolas.
    """
    if value is None:
        return 'NULL'
    return "'" + str(value).replace("'", "''") + "'"


@dataclass
class TransformationArtifact:
    """Artefacto de transformación entre dos campos."""
    concept_name: str
    source_db: str
    source_table: str
    source_column: str
    target_db: str
    target_table: str
    target_column: str
    standard: str
    sql_transform: str
    json_schema: dict
    mapping: dict
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    quality_assessment: dict = field(default_factory=dict)


def generate_sql_transform(
    source_column: str,
    source_values: list[str],
    standard_id: str,
    target_column: str = None,
) -> str:
    """
    Generar SQL CASE WHEN para transformar valores de origen al estándar canónico.

    Ejemplo para ISO 5218:
        CASE
            WHEN sexo IN ('M', 'm') THEN '1'
            WHEN sexo IN ('F', 'f') THEN '2'
            WHEN sexo IN ('H', 'h') THEN '1'  -- H = Masculino en algunos países
            ELSE NULL
        END AS sexo_canonico
    """
    std = STANDARDS.get(standard_id, {})
    canonical_values = get_standard_values(standard_id)

    if not canonical_values:
        return f"-- No hay transformación automática para {standard_id}\n-- Catálogo vacío. Use import_catalog('{standard_id}', 'ruta.csv') para cargar valores."

    target = _escape_identifier(target_column or f"{source_column}_canonico")
    col = _escape_identifier(source_column)

    # Construir mapeo de valores origen -> valor canónico
    # Basándose en los sample_values y el estándar
    lines = []
    lines.append(f"-- Transformación de {source_column} a estándar {standard_id}")
    lines.append(f"-- Estándar: {std.get('name', standard_id)}")
    lines.append(f"-- Catálogo: {len(canonical_values)} valores canónicos disponibles")
    lines.append("CASE")

    # Mapeo conocido de valores comunes a códigos canónicos
    value_mappings = _build_value_mappings(standard_id, source_values, canonical_values)

    for source_val, canonical_code in value_mappings.items():
        label = canonical_values.get(canonical_code, "")
        variants = list(set([source_val, source_val.upper(), source_val.lower()]))
        variants_str = ", ".join(_escape_literal(v) for v in variants)
        lines.append(f"    WHEN {col} IN ({variants_str}) THEN {_escape_literal(canonical_code)}  -- {label}")

    # Detectar valores muestra no mapeados
    unmapped = [v for v in source_values if v and v.strip() not in value_mappings 
                and v.strip().upper() not in value_mappings
                and v.strip().lower() not in value_mappings]
    if unmapped:
        lines.append(f"    -- WARNING: {len(unmapped)} valores muestra sin mapeo: {', '.join(unmapped[:5])}")

    lines.append(f"    ELSE NULL  -- Valor no reconocido, requiere revisión manual")
    lines.append(f"END AS {target}")

    return "\n".join(lines)


def _build_value_mappings(
    standard_id: str,
    source_values: list[str],
    canonical_values: dict[str, str],
) -> dict[str, str]:
    """Construir mapeo de valores de origen a codigos canonicos.
    
    Estrategia agnostica al dominio — no asume ningun estandar especifico:
    1. Match exacto (case-insensitive) contra el catalogo canonico
    2. Match por etiqueta (si el valor origen coincide con la etiqueta del codigo canonico)
    3. Match por prefijo (si el valor origen es prefijo o sufijo del codigo canonico)
    """
    mappings = {}
    
    # Construir indice inverso: etiqueta -> codigo (case-insensitive)
    label_to_code = {}
    for code, label in canonical_values.items():
        label_to_code[label.upper()] = code
        label_to_code[label.lower()] = code

    for v in source_values:
        v_clean = v.strip()
        if not v_clean:
            continue
        v_upper = v_clean.upper()
        v_lower = v_clean.lower()

        # 1. Match exacto contra codigo canonico
        if v_clean in canonical_values:
            mappings[v_clean] = v_clean
            continue
        if v_upper in canonical_values:
            mappings[v_clean] = v_upper
            continue

        # 2. Match por etiqueta del catalogo
        if v_upper in label_to_code:
            mappings[v_clean] = label_to_code[v_upper]
            continue
        if v_lower in label_to_code:
            mappings[v_clean] = label_to_code[v_lower]
            continue

        # 3. Match por etiqueta parcial (valor origen contiene la etiqueta canonica o viceversa)
        # Solo para valores de al menos 3 chars para evitar ambiguedad (ej: "M" matchea "MASCULINO" y "MUJER")
        matched = False
        if len(v_upper) >= 3:
            for code, label in canonical_values.items():
                label_upper = label.upper()
                if len(label_upper) >= 3 and (v_upper == label_upper or v_upper in label_upper or label_upper in v_upper):
                    mappings[v_clean] = code
                    matched = True
                    break
        if matched:
            continue

        # 4. Si el estandar tiene regex, validar formato y hacer pass-through
        std = STANDARDS.get(standard_id, {})
        regex = std.get("regex")
        if regex:
            try:
                if re.match(regex, v_upper):
                    mappings[v_clean] = v_upper
            except re.error:
                pass

    return mappings


def generate_json_schema(concept: dict, classifier: Optional[dict] = None) -> dict:
    """
    Generar JSON Schema de validación para una variable canónica.
    """
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": concept.get("name", "variable"),
        "description": concept.get("definition", ""),
        "type": "string",
    }

    if classifier and classifier.get("values"):
        # Si hay clasificador, los valores válidos son los códigos
        valid_codes = list(classifier["values"].keys())
        schema["enum"] = valid_codes
        schema["enumDescriptions"] = classifier["values"]
    elif concept.get("standard"):
        std_values = get_standard_values(concept["standard"])
        if std_values:
            valid_codes = list(std_values.keys())
            schema["enum"] = valid_codes
            schema["enumDescriptions"] = std_values

    # Metadata adicional
    schema["x-nomenclador"] = {
        "standard": concept.get("standard"),
        "version": concept.get("version", "1.0"),
        "population": concept.get("population", ""),
        "capture_method": concept.get("capture_method", ""),
        "why": concept.get("why", ""),
        "what_for": concept.get("what_for", ""),
    }

    return schema


def generate_transformation(
    field_a: dict,
    field_b: dict,
    concept: dict,
    classifier: Optional[dict] = None,
    validation: Optional[InteropValidation] = None,
) -> TransformationArtifact:
    """
    Generar el artefacto completo de transformación entre dos campos.

    Esto es el entregable: el desarrollador solo revisa y aprueba.
    """
    standard_id = concept.get("standard", "")

    # Determinar cuál campo necesita transformación
    # Si field_a usa el estándar pero field_b no, transformar field_b
    # Si ambos usan el estándar, no hay transformación necesaria
    # Si ninguno usa el estándar, transformar ambos

    std_a = field_a.get("inferred_standard", "")
    std_b = field_b.get("inferred_standard", "")

    warnings = []
    notes = []

    if validation:
        warnings = validation.warnings
        for cp in validation.checkpoints:
            if cp.status == CheckpointStatus.UNKNOWN:
                notes.append(f"INFO: {cp.name} sin información — completar metadata para validar")

    # Gap D: Warning si el clasificador tiene cardinalidad 1:N o N:1
    if classifier and classifier.get("cardinality") and classifier["cardinality"] != "1:1":
        warnings.append(f"EQUIVALE_A cardinalidad {classifier['cardinality']}: transformacion no directa, requiere revision manual")

    # Determinar qué campo necesita transformación
    # Comparar valores reales contra los códigos canónicos del estándar
    std = STANDARDS.get(standard_id, {})
    canonical_codes = set(k.upper() for k in get_standard_values(standard_id).keys())

    def _needs_transform(field: dict) -> bool:
        """Un campo necesita transformación si sus valores no son códigos canónicos."""
        if not canonical_codes:
            return False
        samples = set(str(v).strip().upper() for v in field.get("sample_values", []) if v)
        if not samples:
            return False
        # Si todos los valores ya son códigos canónicos, no necesita transform
        non_canonical = samples - canonical_codes
        return len(non_canonical) > 0

    a_needs = _needs_transform(field_a)
    b_needs = _needs_transform(field_b)

    if b_needs and not a_needs:
        sql = generate_sql_transform(
            source_column=field_b.get("column", ""),
            source_values=field_b.get("sample_values", []),
            standard_id=standard_id,
            target_column=concept.get("name", "canonico"),
        )
        notes.append(f"Transformacion aplicada a {field_b.get('source_db', '')}.{field_b.get('column', '')} para alinear a {standard_id}")
    elif a_needs and not b_needs:
        sql = generate_sql_transform(
            source_column=field_a.get("column", ""),
            source_values=field_a.get("sample_values", []),
            standard_id=standard_id,
            target_column=concept.get("name", "canonico"),
        )
        notes.append(f"Transformacion aplicada a {field_a.get('source_db', '')}.{field_a.get('column', '')} para alinear a {standard_id}")
    elif a_needs and b_needs:
        # Ambos necesitan transformación, generar para ambos
        sql_a = generate_sql_transform(
            source_column=field_a.get("column", ""),
            source_values=field_a.get("sample_values", []),
            standard_id=standard_id,
            target_column=concept.get("name", "canonico"),
        )
        sql_b = generate_sql_transform(
            source_column=field_b.get("column", ""),
            source_values=field_b.get("sample_values", []),
            standard_id=standard_id,
            target_column=concept.get("name", "canonico"),
        )
        sql = f"-- {field_a.get('source_db', '')}.{field_a.get('column', '')}\n{sql_a}\n\n-- {field_b.get('source_db', '')}.{field_b.get('column', '')}\n{sql_b}"
        notes.append(f"Ambas fuentes necesitan transformacion a {standard_id}")
    else:
        sql = f"-- Ambas fuentes ya usan codigos canonicos de {standard_id}. Mapeo directo: {field_a.get('column', '')} = {field_b.get('column', '')}"
        notes.append("Ambas fuentes ya usan codigos canonicos. Mapeo 1:1.")

    # Gap A: Anonimizacion automatica para PII / datos sensibles
    data_cls = concept.get("data_classification", "publico")
    anon_sql = ""
    if data_cls in ("pii", "sensible"):
        anon_sql = _generate_anonymization_sql(
            column=concept.get("name", "variable"),
            data_classification=data_cls,
            data_type=field_a.get("data_type", ""),
        )
        if anon_sql:
            notes.append(f"ANONIMIZACION: {data_cls.upper()} - {anon_sql.split(chr(10))[0]}")

    # Generar JSON Schema
    json_sch = generate_json_schema(concept, classifier)

    # Gap A: Incluir data_classification en JSON schema
    json_sch["x-nomenclador"]["data_classification"] = data_cls
    if anon_sql:
        json_sch["x-nomenclador"]["anonymization"] = anon_sql

    # Mapeo declarativo
    mapping = {
        "concept": concept.get("name", ""),
        "standard": standard_id,
        "data_classification": data_cls,
        "sources": [
            {
                "db": field_a.get("source_db", ""),
                "table": field_a.get("table", ""),
                "column": field_a.get("column", ""),
                "needs_transform": a_needs,
            },
            {
                "db": field_b.get("source_db", ""),
                "table": field_b.get("table", ""),
                "column": field_b.get("column", ""),
                "needs_transform": b_needs,
            },
        ],
    }

    # PMBOK Quality Management: assessment de calidad del cruce
    qa_a = field_a.get("quality_score", 0.0)
    qa_b = field_b.get("quality_score", 0.0)
    comp_a = field_a.get("completeness", 0.0)
    comp_b = field_b.get("completeness", 0.0)
    expected_completeness = round(min(comp_a, comp_b), 3)
    expected_loss = round(1.0 - expected_completeness, 3)
    quality_assessment = {
        "source_quality": round(qa_a, 3),
        "target_quality": round(qa_b, 3),
        "expected_completeness": expected_completeness,
        "expected_record_loss": expected_loss,
        "recommendation": (
            "cruce viable" if expected_loss < 0.1
            else f"cruce con pérdida del {expected_loss:.0%} de registros"
            if expected_loss < 0.3
            else "cruce no recomendado: pérdida significativa de registros"
        ),
    }
    if expected_loss >= 0.1:
        warnings.append(
            f"Pérdida esperada del {expected_loss:.0%} al cruzar "
            f"{field_a.get('source_db', '')} ({comp_a:.0%} completitud) con "
            f"{field_b.get('source_db', '')} ({comp_b:.0%} completitud)"
        )

    # Gap A: Concatenar SQL de anonimizacion si aplica
    full_sql = sql
    if anon_sql:
        full_sql = sql + "\n\n-- Anonimizacion (Gap A): dato " + data_cls.upper() + "\n" + anon_sql

    return TransformationArtifact(
        concept_name=concept.get("name", ""),
        source_db=field_a.get("source_db", ""),
        source_table=field_a.get("table", ""),
        source_column=field_a.get("column", ""),
        target_db=field_b.get("source_db", ""),
        target_table=field_b.get("table", ""),
        target_column=field_b.get("column", ""),
        standard=standard_id,
        sql_transform=full_sql,
        json_schema=json_sch,
        mapping=mapping,
        warnings=warnings,
        notes=notes,
        quality_assessment=quality_assessment,
    )


def _get_anon_salt() -> str:
    """Obtener salt para anonimizacion.

    Prioridad:
    1. Env var ANON_SALT (configurada por el operador)
    2. Salt persistente generado automaticamente en .anon_salt

    El salt persistente se genera una vez con secrets.token_hex(16)
    y se reutiliza en ejecuciones posteriores para consistencia.
    """
    salt = os.environ.get("ANON_SALT", "")
    if salt:
        return salt

    salt_path = Path(__file__).parent.parent / ".anon_salt"
    if salt_path.exists():
        return salt_path.read_text(encoding="utf-8").strip()

    import secrets
    generated = secrets.token_hex(16)
    salt_path.write_text(generated, encoding="utf-8")
    try:
        gitignore = Path(__file__).parent.parent / ".gitignore"
        if gitignore.exists():
            content = gitignore.read_text(encoding="utf-8")
            if ".anon_salt" not in content:
                gitignore.write_text(content.rstrip() + "\n.anon_salt\n", encoding="utf-8")
    except Exception:
        pass
    return generated


def _generate_anonymization_sql(column: str, data_classification: str, data_type: str = "") -> str:
    """Generar SQL de anonimizacion segun el nivel de clasificacion (Gap A).

    PII: seudonimizacion (hash SHA-256 con salt obligatorio)
    Sensible: generalizacion (reduce granularidad)

    El salt se obtiene via _get_anon_salt() para evitar ataques de
    fuerza bruta sobre universos pequenos (ej. DUI de un pais).
    """
    col = _escape_identifier(column)
    col_anon = _escape_identifier(f"{column}_anon")
    if data_classification == "pii":
        if "date" in data_type.lower() or "fecha" in column.lower():
            return f"-- PII: Generalizar fecha a año\nEXTRACT(YEAR FROM {col}) AS {col_anon}"
        salt = _get_anon_salt()
        salt_lit = _escape_literal(salt)
        return f"-- PII: Seudonimizacion via hash SHA-256 con salt\nSUBSTRING(ENCODE(DIGEST({col}::text || {salt_lit}, 'sha256'), 'hex'), 1, 16) AS {col_anon}"
    elif data_classification == "sensible":
        return f"-- SENSIBLE: Generalizacion - agrupar categorias\nCASE WHEN {col} IS NOT NULL THEN 'registrado' ELSE NULL END AS {col_anon}"
    return ""


def artifact_to_dict(artifact: TransformationArtifact) -> dict:
    """Convertir artefacto a diccionario para serialización."""
    return {
        "concept": artifact.concept_name,
        "standard": artifact.standard,
        "source": {
            "db": artifact.source_db,
            "table": artifact.source_table,
            "column": artifact.source_column,
        },
        "target": {
            "db": artifact.target_db,
            "table": artifact.target_table,
            "column": artifact.target_column,
        },
        "sql_transform": artifact.sql_transform,
        "json_schema": artifact.json_schema,
        "mapping": artifact.mapping,
        "warnings": artifact.warnings,
        "notes": artifact.notes,
        "quality_assessment": artifact.quality_assessment,
    }
