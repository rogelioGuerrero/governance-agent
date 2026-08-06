"""
PackMemory: memoria persistente de correcciones por Domain Pack.

Cada vez que el governance agent corrige un error y el usuario lo acepta o rechaza,
se registra aquí. La próxima vez que aparezca el mismo error:
- Si fue aceptado antes → se aplica automáticamente sin preguntar
- Si fue rechazado → no se intenta esa corrección de nuevo
- Si se aplicó 5+ veces sin rechazo → se convierte en regla automática

Persistencia dual: PostgreSQL (Supabase) + JSON local (fallback).
"""

from __future__ import annotations

import json
import logging
import os
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_DIR = Path(__file__).parent.parent.parent / "data" / "pack_memory"


@dataclass
class CorrectionRecord:
    """Registro de una corrección aplicada a un error."""
    pack_name: str
    error_signature: str  # hash del tipo de error + campo + contexto
    error_type: str       # ej: "missing_field", "type_mismatch", "out_of_range"
    field_name: str
    original_value: str
    corrected_value: str
    correction_method: str  # inference, llm_suggestion, user_guided
    user_accepted: Optional[bool] = None  # None = pendiente, True = aceptado, False = rechazado
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    count: int = 1  # veces que se ha aplicado

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> "CorrectionRecord":
        return CorrectionRecord(**data)


def compute_error_signature(error_type: str, field_name: str, context: str = "") -> str:
    """Calcular hash único para un tipo de error en un contexto."""
    raw = f"{error_type}|{field_name}|{context}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class PackMemory:
    """
    Memoria de correcciones por Domain Pack.

    Estrategia:
    1. Cargar correcciones persistidas al iniciar
    2. Consultar antes de preguntar al usuario
    3. Registrar resultado después de cada interacción
    4. Promover a regla automática tras N aceptaciones
    """

    AUTO_RULE_THRESHOLD = 5  # veces aceptadas para auto-promover

    def __init__(self, pack_name: str, memory_dir: Path = None, supabase_url: str = None,
                 supabase_key: str = None):
        self.pack_name = pack_name
        self.memory_dir = memory_dir or DEFAULT_MEMORY_DIR
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / f"{pack_name}_memory.json"
        self.supabase_url = supabase_url or os.getenv("SUPABASE_URL")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self._records: dict[str, CorrectionRecord] = {}
        self._load()

    def _load(self):
        """Cargar memoria desde JSON local."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for record_data in data.get("records", []):
                    record = CorrectionRecord.from_dict(record_data)
                    self._records[record.error_signature] = record
                logger.debug("PackMemory cargada: %s (%d registros)", self.pack_name, len(self._records))
            except Exception as e:
                logger.warning("Error cargando PackMemory: %s", e)

    def _save(self):
        """Persistir memoria a JSON local."""
        try:
            data = {"pack_name": self.pack_name, "records": [r.to_dict() for r in self._records.values()]}
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Error guardando PackMemory: %s", e)

    def lookup(self, error_type: str, field_name: str, context: str = "") -> Optional[CorrectionRecord]:
        """
        Buscar si ya tenemos una corrección registrada para este error.

        Returns:
            CorrectionRecord si existe, None si no.
            Revisar record.user_accepted y record.count para decidir acción.
        """
        sig = compute_error_signature(error_type, field_name, context)
        return self._records.get(sig)

    def should_auto_apply(self, error_type: str, field_name: str, context: str = "") -> Optional[CorrectionRecord]:
        """
        Verificar si una corrección debe aplicarse automáticamente sin preguntar.

        Returns CorrectionRecord si debe auto-aplicarse, None si no.
        """
        record = self.lookup(error_type, field_name, context)
        if record and record.user_accepted is True and record.count >= self.AUTO_RULE_THRESHOLD:
            logger.info("Auto-aplicando corrección para %s.%s (aceptada %d veces)",
                       self.pack_name, field_name, record.count)
            return record
        return None

    def should_skip(self, error_type: str, field_name: str, context: str = "") -> bool:
        """Verificar si una corrección fue rechazada y no debe intentarse de nuevo."""
        record = self.lookup(error_type, field_name, context)
        return record is not None and record.user_accepted is False

    def record_correction(self, error_type: str, field_name: str, original_value: str,
                          corrected_value: str, correction_method: str,
                          user_accepted: Optional[bool] = None,
                          context: str = "") -> CorrectionRecord:
        """
        Registrar o actualizar una corrección.

        Si ya existe, incrementa el contador y actualiza user_accepted.
        Si es nueva, la crea.
        """
        sig = compute_error_signature(error_type, field_name, context)

        if sig in self._records:
            record = self._records[sig]
            record.count += 1
            if user_accepted is not None:
                record.user_accepted = user_accepted
            record.timestamp = datetime.now(timezone.utc).isoformat()
        else:
            record = CorrectionRecord(
                pack_name=self.pack_name,
                error_signature=sig,
                error_type=error_type,
                field_name=field_name,
                original_value=original_value,
                corrected_value=corrected_value,
                correction_method=correction_method,
                user_accepted=user_accepted,
                context=context,
            )
            self._records[sig] = record

        self._save()
        return record

    def get_stats(self) -> dict:
        """Estadísticas de la memoria."""
        total = len(self._records)
        accepted = sum(1 for r in self._records.values() if r.user_accepted is True)
        rejected = sum(1 for r in self._records.values() if r.user_accepted is False)
        pending = sum(1 for r in self._records.values() if r.user_accepted is None)
        auto_rules = sum(1 for r in self._records.values()
                        if r.user_accepted is True and r.count >= self.AUTO_RULE_THRESHOLD)
        return {
            "pack_name": self.pack_name,
            "total_corrections": total,
            "accepted": accepted,
            "rejected": rejected,
            "pending": pending,
            "auto_rules": auto_rules,
        }

    def get_auto_rules(self) -> list[CorrectionRecord]:
        """Obtener todas las correcciones que se han convertido en reglas automáticas."""
        return [r for r in self._records.values()
                if r.user_accepted is True and r.count >= self.AUTO_RULE_THRESHOLD]
