"""
Framework agnostico de estandares y clasificadores.

El sistema NO incluye estandares de ningun dominio especifico (salud, educacion, laboral).
Los estandares se registran dinamicamente cuando se entra a un dominio, via:
1. register_standard() — registra un estandar con sus valores y metadatos
2. import_catalog() — carga valores desde archivo CSV/JSON externo
3. RAG documental — referencia normativa de dominio

Esto garantiza que el nomenclador sea agnostico al dominio.
"""

import csv
import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

CATALOG_DIR = Path(__file__).parent.parent / "nomenclador" / "catalogs"

# Registro dinamico de estandares. Inicia vacio.
# Los estandares se agregan via register_standard() o import_catalog()
# cuando el sistema entra a un dominio especifico.
STANDARDS: dict[str, dict] = {}


def register_standard(
    standard_id: str,
    name: str,
    domain: str = "transversal",
    standard_type: str = "classifier",
    values: dict = None,
    detect_patterns: list = None,
    regex: str = None,
    name_hints: list = None,
    catalog_file: str = None,
) -> None:
    """Registrar un estandar en el catalogo dinamico.
    
    Args:
        standard_id: identificador unico (ej: "ISO_5218", "CINE_2011")
        name: nombre descriptivo
        domain: dominio al que pertenece (transversal, salud, educacion, etc.)
        standard_type: "classifier" (valores enumerados con significado) o "format" (validacion de formato sin significado semantico)
        values: dict {codigo: etiqueta} de valores validos (solo para classifier)
        detect_patterns: patrones de deteccion por valores
        regex: expresion regular para validar formato de codigo
        name_hints: nombres de columna que sugieren este estandar
        catalog_file: archivo CSV/JSON con valores (carga diferida)
    """
    STANDARDS[standard_id] = {
        "name": name,
        "domain": domain,
        "standard_type": standard_type,
        "values": values or {},
        "detect_patterns": detect_patterns or [],
        "regex": regex,
        "_compiled_regex": re.compile(regex) if regex else None,
        "name_hints": name_hints or [],
        "_name_hints_lower": {h.lower() for h in (name_hints or [])},
        "importable": catalog_file is not None,
        "catalog_file": catalog_file,
    }
    logger.info("Estandar registrado: %s (%s) — tipo: %s, dominio: %s, valores: %d",
                standard_id, name, standard_type, domain, len(values or {}))


def unregister_standard(standard_id: str) -> bool:
    """Eliminar un estandar del catalogo dinamico."""
    if standard_id in STANDARDS:
        del STANDARDS[standard_id]
        return True
    return False


def list_standards(domain: str = None) -> list[dict]:
    """Listar estandares registrados, opcionalmente filtrados por dominio."""
    result = []
    for sid, std in STANDARDS.items():
        if domain and std.get("domain") != domain:
            continue
        result.append({
            "id": sid,
            "name": std["name"],
            "domain": std["domain"],
            "standard_type": std.get("standard_type", "classifier"),
            "values_count": len(std.get("values", {})),
            "importable": std.get("importable", False),
        })
    return result


def import_catalog(standard_id: str, file_path: str = None) -> int:
    """Importar un catalogo de valores desde un archivo CSV o JSON.
    
    El archivo debe tener columnas 'code' y 'label' (CSV) o ser un
    dict {code: label} (JSON). Si no se especifica file_path, busca
    en CATALOG_DIR/{catalog_file} definido en el estandar.
    
    Si el estandar no existe, lo crea con metadatos minimos.
    
    Returns: numero de valores cargados, 0 si falla.
    """
    std = STANDARDS.get(standard_id)
    if not std:
        std = {"name": standard_id, "domain": "unknown", "values": {}}
        STANDARDS[standard_id] = std
    
    path = file_path
    if not path:
        catalog_file = std.get("catalog_file")
        if not catalog_file:
            return 0
        path = str(CATALOG_DIR / catalog_file)
    
    if not os.path.exists(path):
        logger.warning("Catalogo no encontrado: %s", path)
        return 0
    
    loaded = 0
    try:
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                std["values"] = data
                loaded = len(data)
            elif isinstance(data, list):
                values = {}
                for item in data:
                    code = item.get("code", "")
                    label = item.get("label", item.get("name", ""))
                    if code:
                        values[str(code)] = label
                std["values"] = values
                loaded = len(values)
        elif path.endswith(".csv"):
            values = {}
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    code = row.get("code", "").strip()
                    label = row.get("label", row.get("name", "")).strip()
                    if code:
                        values[code] = label
            std["values"] = values
            loaded = len(values)
        logger.info("Catalogo cargado: %s — %d valores desde %s", standard_id, loaded, path)
    except Exception as e:
        logger.warning("Error cargando catalogo %s: %s", standard_id, e)
    
    return loaded


def get_standard_values(standard_id: str) -> dict:
    """Obtener los valores de un estandar, cargando catalogo externo si es necesario."""
    std = STANDARDS.get(standard_id, {})
    values = std.get("values", {})
    if not values and std.get("importable"):
        import_catalog(standard_id)
        values = std.get("values", {})
    return values


def detect_standard(column_name: str, sample_values: list[str]) -> list[dict]:
    """
    Detectar posibles estandares para una columna basandose en
    su nombre y valores de muestra.
    
    Usa name_hints y regex de cada estandar registrado dinamicamente.
    No asume ningun dominio especifico.
    
    Retorna lista de estandares candidatos con nivel de confianza.
    """
    candidates = []
    col_lower = column_name.lower().strip()

    # 1. Detectar por name_hints de cada estandar registrado
    for std_id, std in STANDARDS.items():
        hints_lower = std.get("_name_hints_lower", set())
        if col_lower in hints_lower:
            candidates.append({
                "standard": std_id,
                "name": std["name"],
                "standard_type": std.get("standard_type", "classifier"),
                "confidence": "high",
                "reason": f"Nombre de columna '{column_name}' coincide con hint del estandar",
            })

    # 2. Detectar por regex de cada estandar registrado
    if sample_values:
        for std_id, std in STANDARDS.items():
            pattern = std.get("_compiled_regex")
            if not pattern:
                continue
            try:
                matches = sum(1 for v in sample_values if v and pattern.match(str(v).strip().upper()))
                if matches > len(sample_values) * 0.5:
                    candidates.append({
                        "standard": std_id,
                        "name": std["name"],
                        "standard_type": std.get("standard_type", "classifier"),
                        "confidence": "high",
                        "reason": f"{matches}/{len(sample_values)} valores coinciden con patron de {std_id}",
                    })
            except re.error:
                continue

    # 3. Detectar por valores exactos contra catalogos cargados
    if sample_values:
        val_set = set(str(v).strip().upper() for v in sample_values if v)
        for std_id, std in STANDARDS.items():
            values = get_standard_values(std_id)
            if not values:
                continue
            canonical_upper = set(k.upper() for k in values.keys())
            overlap = val_set & canonical_upper
            if overlap and len(overlap) >= len(val_set) * 0.5:
                candidates.append({
                    "standard": std_id,
                    "name": std["name"],
                    "standard_type": std.get("standard_type", "classifier"),
                    "confidence": "medium",
                    "reason": f"{len(overlap)}/{len(val_set)} valores coinciden con catalogo de {std_id}",
                })

    # Deduplicar: si un estandar aparece multiples veces, quedarse con el de mayor confianza
    seen: dict[str, dict] = {}
    confidence_order = {"high": 3, "medium": 2, "low": 1}
    for c in candidates:
        sid = c["standard"]
        if sid not in seen or confidence_order.get(c["confidence"], 0) > confidence_order.get(seen[sid]["confidence"], 0):
            seen[sid] = c
    return list(seen.values())
