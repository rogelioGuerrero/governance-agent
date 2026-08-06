"""
Lifecycle + Decision Log: registro del ciclo de vida de cada variable canonica.

Cada variable es "viva": nace, cambia, se deprecia, se retira.
Cada transicion queda registrada con quien, que, por que y cuando.

Usos:
1. Decision log — auditoria: quien decidio que y cuando
2. Nota explicativa — acumulada: el "por que" en lenguaje humano
3. Historico — lifecycle: nacimiento, cambios, deprecacion, retiro

Estados:
- activo: en uso, aparece en fuentes
- deprecado: sigue en el grafo pero no recomendado para nuevas fuentes
- retirado: fuera del grafo activo, historial conservado

Almacenamiento: nomenclador/decision_log.json
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

LOG_PATH = Path(__file__).parent.parent / "nomenclador" / "decision_log.json"

VALID_STATUSES = {"activo", "deprecado", "retirado"}

_DATABASE_URL = os.environ.get("DATABASE_URL", "")


def _get_pg_conn():
    """Obtener conexion PostgreSQL si DATABASE_URL esta configurada."""
    if not _DATABASE_URL:
        return None
    try:
        import psycopg
        return psycopg.connect(_DATABASE_URL, autocommit=True)
    except Exception:
        return None


def _pg_log_event(conn, concept_id: str, action: str, actor: str, reason: str, details: str):
    """Insertar evento en governance.lifecycle_log (PostgreSQL)."""
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO governance.lifecycle_log (concept_id, action, actor, reason, details) "
                "VALUES (%s, %s, %s, %s, %s)",
                (concept_id, action, actor, reason, details),
            )
    except Exception as e:
        logger.warning("PostgreSQL log_event fallo para %s: %s", concept_id, e)

# Gap C: Estados de revision para nodos propuestos por IA
REVIEW_STATUSES = {"proposed", "under_review", "approved", "rejected"}
REVIEW_ACTIONS = {
    "proposed": "proposed",
    "under_review": "review_started",
    "approved": "approved",
    "rejected": "rejected",
}
# Reverse mapping: action -> review_status
ACTION_TO_REVIEW_STATUS = {v: k for k, v in REVIEW_ACTIONS.items()}


def _load_log() -> dict:
    """Cargar el decision log desde JSON."""
    if not LOG_PATH.exists():
        return {"entries": {}}
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_log(data: dict):
    """Guardar el decision log."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def log_event(
    concept_id: str,
    action: str,
    actor: str = "agent",
    reason: str = "",
    details: str = "",
):
    """Registrar un evento en el decision log de una variable.

    Dual-write: PostgreSQL (governance.lifecycle_log) + JSON local (fallback).

    Args:
        concept_id: ID del concepto (ej: "concept:sexo")
        action: Tipo de accion (created, updated, normative_attached, custodian_assigned,
                 deprecated, reactivated, retired, definition_changed, standard_changed)
        actor: Quien ejecuta ("agent" o "human")
        reason: Por que se hizo (opcional para auto-events)
        details: Detalles adicionales del evento
    """
    # PostgreSQL (primario)
    conn = _get_pg_conn()
    if conn:
        _pg_log_event(conn, concept_id, action, actor, reason, details)
        conn.close()

    # JSON local (fallback / mirror)
    data = _load_log()
    entries = data["entries"].setdefault(concept_id, [])

    entry = {
        "timestamp": _now(),
        "action": action,
        "actor": actor,
        "reason": reason,
        "details": details,
    }
    entries.append(entry)
    _save_log(data)


def get_history(concept_id: str) -> list[dict]:
    """Obtener el historial completo de una variable."""
    data = _load_log()
    return data.get("entries", {}).get(concept_id, [])


def get_all_history() -> dict:
    """Obtener el decision log completo."""
    return _load_log().get("entries", {})


def get_explanatory_note(concept_id: str) -> str:
    """Construir nota explicativa acumulada desde el historial."""
    history = get_history(concept_id)
    if not history:
        return ""
    lines = []
    for e in history:
        if e.get("reason"):
            prefix = f"[{e['timestamp'][:10]}] "
            if e["actor"] == "human":
                prefix += "(humano) "
            else:
                prefix += "(auto) "
            lines.append(f"{prefix}{e['action']}: {e['reason']}")
    return "\n".join(lines)


def change_status(
    concept_id: str,
    new_status: str,
    actor: str = "human",
    reason: str = "",
    replacement: str = "",
) -> str:
    """Cambiar el estado de una variable y registrar el evento.

    Args:
        concept_id: ID del concepto
        new_status: activo | deprecado | retirado
        actor: quien ejecuta
        reason: por que
        replacement: si se deprecia, concepto que lo reemplaza (opcional)

    Returns:
        Mensaje de confirmacion
    """
    if new_status not in VALID_STATUSES:
        return f"Estado invalido: {new_status}. Validos: {VALID_STATUSES}"

    action_map = {
        "activo": "reactivated",
        "deprecado": "deprecated",
        "retirado": "retired",
    }
    action = action_map.get(new_status, "status_changed")

    details = f"reemplazado_por={replacement}" if replacement else ""

    log_event(
        concept_id=concept_id,
        action=action,
        actor=actor,
        reason=reason,
        details=details,
    )

    return f"Estado cambiado a '{new_status}' para {concept_id}"


def format_history(concept_id: str, concept_name: str = "") -> str:
    """Formatear el historial para mostrar en consola."""
    history = get_history(concept_id)
    if not history:
        return f"Sin historial para '{concept_name or concept_id}'"

    lines = [f"Historial de '{concept_name or concept_id}' ({len(history)} eventos):"]
    lines.append("=" * 60)

    for e in history:
        ts = e["timestamp"][:16].replace("T", " ")
        actor_tag = "humano" if e["actor"] == "human" else "auto"
        lines.append(f"\n  [{ts}] {e['action']} ({actor_tag})")
        if e.get("reason"):
            lines.append(f"    Razon: {e['reason']}")
        if e.get("details"):
            lines.append(f"    Detalles: {e['details']}")

    note = get_explanatory_note(concept_id)
    if note:
        lines.append("\n" + "-" * 60)
        lines.append("Nota explicativa acumulada:")
        lines.append(note)

    return "\n".join(lines)


def find_deprecated() -> list[dict]:
    """Listar todas las variables deprecadas o retiradas."""
    data = _load_log()
    result = []
    for concept_id, entries in data.get("entries", {}).items():
        last_status = None
        for e in reversed(entries):
            if e["action"] in ("deprecated", "retired", "reactivated", "status_changed"):
                last_status = e["action"]
                break
        if last_status in ("deprecated", "retired"):
            result.append({"concept_id": concept_id, "last_status": last_status})
    return result


# === GAP C: WORKFLOW DE REVISION ===

def log_review_event(concept_id: str, review_status: str, actor: str = "human", reason: str = ""):
    """Registrar un evento de revision en el decision log.

    Args:
        concept_id: ID del concepto
        review_status: proposed | under_review | approved | rejected
        actor: quien ejecuta (human o agent)
        reason: razon de la decision
    """
    action = REVIEW_ACTIONS.get(review_status, "review_changed")
    log_event(
        concept_id=concept_id,
        action=action,
        actor=actor,
        reason=reason,
        details=f"review_status={review_status}",
    )


def find_pending_reviews() -> list[dict]:
    """Listar todos los conceptos con eventos de revision pendientes."""
    data = _load_log()
    result = []
    for concept_id, entries in data.get("entries", {}).items():
        last_review = None
        for e in reversed(entries):
            if e["action"] in ("proposed", "review_started", "approved", "rejected"):
                last_review = e["action"]
                break
        if last_review in ("proposed", "review_started"):
            result.append({"concept_id": concept_id, "review_status": ACTION_TO_REVIEW_STATUS.get(last_review, last_review)})
    return result


# === LEARNING LOOP: recall de feedback pasado ===

def recall_feedback(query: str, limit: int = 5) -> list[dict]:
    """Recuperar decisiones pasadas relevantes para una consulta.

    Busca en el decision_log eventos donde el concept_id o la razon
    coincidan con palabras clave de la consulta. Prioriza:
    1. Rechazos humanos (feedback negativo — el mas valioso para aprender)
    2. Aprobaciones humanas (feedback positivo)
    3. Cambios de estado (deprecation, reactivacion)

    Args:
        query: Consulta del usuario o nombre de concepto
        limit: Maximo de eventos a retornar

    Returns:
        Lista de dicts: concept_id, action, actor, reason, timestamp
    """
    data = _load_log()
    all_entries = data.get("entries", {})
    if not all_entries:
        return []

    query_lower = query.lower().strip()
    keywords = {w for w in query_lower.replace("_", " ").split() if len(w) > 2}
    if not keywords:
        return []

    results = []
    for concept_id, entries in all_entries.items():
        concept_lower = concept_id.lower().replace("concept:", "")
        concept_words = set(concept_lower.replace("_", " ").split())
        concept_match = keywords & concept_words

        for e in entries:
            reason_lower = (e.get("reason") or "").lower()
            reason_match = any(kw in reason_lower for kw in keywords)

            if not concept_match and not reason_match:
                continue

            results.append({
                "concept_id": concept_id,
                "action": e.get("action", ""),
                "actor": e.get("actor", ""),
                "reason": e.get("reason", ""),
                "timestamp": e.get("timestamp", ""),
            })

    priority = {"rejected": 0, "approved": 1, "deprecated": 2, "reactivated": 3}
    results.sort(key=lambda r: (
        priority.get(r["action"], 9),
        r["timestamp"],
    ))
    return results[:limit]
