"""
Guardrails de Validacion de Contexto.

Antes de declarar que dos variables son interoperables, el agente debe
validar explicitamente cuatro Checkpoints en el grafo:

1. Poblacion objetivo: ambas fuentes miden la misma poblacion?
2. Metodologia de captura: el dato se obtuvo de la misma manera?
3. Clasificador activo: ambas usan el mismo estandar de codificacion?
4. Distribucion de datos: cardinalidad, null rate y overlap de valores son compatibles?

Si uno no coincide, el agente emite un "Warning de Asimetria Semantica".
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class CheckpointStatus(str, Enum):
    MATCH = "match"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"


@dataclass
class CheckpointResult:
    name: str
    status: CheckpointStatus
    detail: str = ""
    field_a_value: str = ""
    field_b_value: str = ""


@dataclass
class InteropValidation:
    """Resultado de validación de interoperabilidad entre dos campos."""
    field_a_id: str
    field_b_id: str
    concept_id: str
    checkpoints: list[CheckpointResult]
    is_safe: bool
    warnings: list[str]
    recommendation: str = ""


def validate_interoperability(
    field_a: dict,
    field_b: dict,
    concept: dict,
    classifier: Optional[dict] = None,
) -> InteropValidation:
    """
    Validar si dos campos físicos son realmente interoperables
    a través de un concepto canónico.

    Ejecuta los 4 checkpoints y retorna el resultado.
    """
    checkpoints = []

    # === Checkpoint 1: Población objetivo ===
    pop_a = field_a.get("population", "").strip().lower()
    pop_b = field_b.get("population", "").strip().lower()
    pop_canonical = concept.get("population", "").strip().lower()

    if not pop_a and not pop_b:
        cp = CheckpointResult(
            name="Población objetivo",
            status=CheckpointStatus.UNKNOWN,
            detail="Ninguna fuente tiene población definida",
            field_a_value="(no definida)",
            field_b_value="(no definida)",
        )
    elif pop_a == pop_b:
        cp = CheckpointResult(
            name="Población objetivo",
            status=CheckpointStatus.MATCH,
            detail=f"Ambas fuentes miden: {pop_a}",
            field_a_value=pop_a,
            field_b_value=pop_b,
        )
    else:
        cp = CheckpointResult(
            name="Población objetivo",
            status=CheckpointStatus.MISMATCH,
            detail=f"Asimetría: '{pop_a}' vs '{pop_b}'",
            field_a_value=pop_a or "(no definida)",
            field_b_value=pop_b or "(no definida)",
        )
    checkpoints.append(cp)

    # === Checkpoint 2: Metodología de captura ===
    cap_a = field_a.get("capture_method", "").strip().lower()
    cap_b = field_b.get("capture_method", "").strip().lower()
    cap_canonical = concept.get("capture_method", "").strip().lower()

    if not cap_a and not cap_b:
        cp = CheckpointResult(
            name="Metodología de captura",
            status=CheckpointStatus.UNKNOWN,
            detail="Ninguna fuente tiene metodología definida",
            field_a_value="(no definida)",
            field_b_value="(no definida)",
        )
    elif cap_a == cap_b:
        cp = CheckpointResult(
            name="Metodología de captura",
            status=CheckpointStatus.MATCH,
            detail=f"Ambas fuentes capturan: {cap_a}",
            field_a_value=cap_a,
            field_b_value=cap_b,
        )
    else:
        cp = CheckpointResult(
            name="Metodología de captura",
            status=CheckpointStatus.MISMATCH,
            detail=f"Asimetría: '{cap_a}' vs '{cap_b}' — el dato se obtuvo diferente",
            field_a_value=cap_a or "(no definida)",
            field_b_value=cap_b or "(no definida)",
        )
    checkpoints.append(cp)

    # === Checkpoint 3: Clasificador activo ===
    std_a = field_a.get("inferred_standard", "").strip()
    std_b = field_b.get("inferred_standard", "").strip()
    std_canonical = concept.get("standard", "").strip()

    if not std_a and not std_b:
        cp = CheckpointResult(
            name="Clasificador activo",
            status=CheckpointStatus.UNKNOWN,
            detail="Ninguna fuente tiene clasificador definido",
            field_a_value="(no definido)",
            field_b_value="(no definido)",
        )
    elif std_a == std_b:
        cp = CheckpointResult(
            name="Clasificador activo",
            status=CheckpointStatus.MATCH,
            detail=f"Ambas usan: {std_a}",
            field_a_value=std_a,
            field_b_value=std_b,
        )
    else:
        cp = CheckpointResult(
            name="Clasificador activo",
            status=CheckpointStatus.MISMATCH,
            detail=f"Asimetría: '{std_a}' vs '{std_b}' — diferentes sistemas de codificación",
            field_a_value=std_a or "(no definido)",
            field_b_value=std_b or "(no definido)",
        )
    checkpoints.append(cp)

    # === Checkpoint 4: Distribucion de datos reales ===
    # Comparar cardinalidad, null rate y overlap de valores
    total_a = field_a.get("total_count", 0) or 0
    total_b = field_b.get("total_count", 0) or 0
    unique_a = field_a.get("unique_count", 0) or 0
    unique_b = field_b.get("unique_count", 0) or 0
    null_a = field_a.get("null_count", 0) or 0
    null_b = field_b.get("null_count", 0) or 0
    samples_a = field_a.get("sample_values", []) or []
    samples_b = field_b.get("sample_values", []) or []

    data_issues = []
    data_match = True

    # 4a. Comparar cardinalidad (ratio de valores unicos)
    if unique_a > 0 and unique_b > 0:
        card_ratio = min(unique_a, unique_b) / max(unique_a, unique_b)
        if card_ratio < 0.3:
            data_issues.append(f"Cardinalidad muy diferente: {unique_a} vs {unique_b} unicos (ratio {card_ratio:.0%})")
            data_match = False
        elif card_ratio < 0.7:
            data_issues.append(f"Cardinalidad discrepante: {unique_a} vs {unique_b} unicos (ratio {card_ratio:.0%})")

    # 4b. Comparar null rate
    if total_a > 0 and total_b > 0:
        null_rate_a = null_a / total_a
        null_rate_b = null_b / total_b
        null_diff = abs(null_rate_a - null_rate_b)
        if null_diff > 0.3:
            data_issues.append(f"Null rate muy diferente: {null_rate_a:.0%} vs {null_rate_b:.0%}")
            data_match = False
        elif null_diff > 0.15:
            data_issues.append(f"Null rate discrepante: {null_rate_a:.0%} vs {null_rate_b:.0%}")

    # 4c. Comparar overlap de valores (para datos categoricos de baja cardinalidad)
    if samples_a and samples_b and unique_a < 100 and unique_b < 100:
        set_a = set(str(v).strip().upper() for v in samples_a if v)
        set_b = set(str(v).strip().upper() for v in samples_b if v)
        if set_a and set_b:
            overlap = set_a & set_b
            overlap_ratio = len(overlap) / len(set_a | set_b) if (set_a | set_b) else 0
            if overlap_ratio < 0.2 and len(set_a) > 2 and len(set_b) > 2:
                data_issues.append(f"Overlap de valores bajo: {len(overlap)}/{len(set_a | set_b)} valores compartidos ({overlap_ratio:.0%})")
                data_match = False

    if not samples_a and not samples_b:
        cp = CheckpointResult(
            name="Distribucion de datos",
            status=CheckpointStatus.UNKNOWN,
            detail="Sin datos de muestra para comparar",
            field_a_value="(sin muestras)",
            field_b_value="(sin muestras)",
        )
    elif data_match and not data_issues:
        cp = CheckpointResult(
            name="Distribucion de datos",
            status=CheckpointStatus.MATCH,
            detail=f"Distribuciones compatibles (cardinalidad: {unique_a}/{unique_b}, null: {null_a}/{null_b})",
            field_a_value=f"uniq={unique_a}, null={null_a}/{total_a}",
            field_b_value=f"uniq={unique_b}, null={null_b}/{total_b}",
        )
    elif data_issues:
        cp = CheckpointResult(
            name="Distribucion de datos",
            status=CheckpointStatus.MISMATCH if not data_match else CheckpointStatus.UNKNOWN,
            detail="; ".join(data_issues),
            field_a_value=f"uniq={unique_a}, null={null_a}/{total_a}",
            field_b_value=f"uniq={unique_b}, null={null_b}/{total_b}",
        )
    else:
        cp = CheckpointResult(
            name="Distribucion de datos",
            status=CheckpointStatus.UNKNOWN,
            detail="Datos insuficientes para comparar",
            field_a_value=f"uniq={unique_a}",
            field_b_value=f"uniq={unique_b}",
        )
    checkpoints.append(cp)

    # === Determinar si es seguro ===
    mismatches = [c for c in checkpoints if c.status == CheckpointStatus.MISMATCH]
    unknowns = [c for c in checkpoints if c.status == CheckpointStatus.UNKNOWN]

    warnings = []
    for c in mismatches:
        warnings.append(f"WARNING de Asimetría Semántica — {c.name}: {c.detail}")

    for c in unknowns:
        warnings.append(f"WARNING de Información Incompleta — {c.name}: {c.detail}")

    # Gap C: conceptos proposed/rejected no son seguros para interoperabilidad
    review_status = concept.get("review_status", "approved")
    if review_status in ("proposed", "rejected"):
        warnings.append(f"WARNING de Estado de Revisión — concepto en estado '{review_status}', no aprobado")
        is_safe = False
    else:
        is_safe = len(mismatches) == 0

    if is_safe and len(unknowns) == 0:
        recommendation = "INTEROPERABILIDAD SEGURA — todos los checkpoints coinciden"
    elif is_safe and len(unknowns) > 0:
        recommendation = f"INTEROPERABILIDAD PROBABLE — {len(unknowns)} checkpoint(s) sin información, verificar manualmente"
    elif len(mismatches) == 1:
        recommendation = "INTEROPERABILIDAD CONDICIONAL — requiere transformación o normalización"
    else:
        recommendation = "INTEROPERABILIDAD NO RECOMENDADA — múltiples asimetrías semánticas"

    return InteropValidation(
        field_a_id=field_a.get("id", ""),
        field_b_id=field_b.get("id", ""),
        concept_id=concept.get("id", ""),
        checkpoints=checkpoints,
        is_safe=is_safe,
        warnings=warnings,
        recommendation=recommendation,
    )
