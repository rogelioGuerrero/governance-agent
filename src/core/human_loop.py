"""
HumanInTheLoop: interacción humana no intrusiva.

Principios:
- No detener el pipeline por cada duda
- No abrumar al usuario con preguntas
- Autocorrección primero: si se puede inferir, hacerlo y loguear
- Solo preguntar si es CRITICAL y no hay autocorrección posible
- Acumular dudas WARNING y preguntar en batch al final
- Confianza progresiva: si el usuario confirma, aprender (PackMemory)

Niveles:
- INFO: no preguntar, solo loguear
- WARNING: acumular, preguntar en batch al final
- CRITICAL: detener y preguntar inmediatamente
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class QuestionLevel(str, Enum):
    INFO = "info"        # no preguntar, solo loguear
    WARNING = "warning"  # acumular, preguntar en batch
    CRITICAL = "critical"  # detener y preguntar inmediatamente


@dataclass
class Question:
    """Pregunta para el usuario."""
    level: QuestionLevel
    field_name: str
    message: str
    suggested_value: str = ""
    options: list[str] = field(default_factory=list)
    context: str = ""
    # Respuesta del usuario (se llena después)
    answer: Optional[str] = None
    accepted: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "field_name": self.field_name,
            "message": self.message,
            "suggested_value": self.suggested_value,
            "options": self.options,
            "context": self.context,
            "answer": self.answer,
            "accepted": self.accepted,
        }


class HumanInTheLoop:
    """
    Gestiona la interacción humana durante la validación.

    Uso:
        hitl = HumanInTheLoop(pack_memory)
        # durante validación:
        hitl.ask(Question(level=WARNING, field_name="lat", message="¿Coordenada correcta?", suggested_value="4.71"))
        # al final:
        questions = hitl.get_pending_questions()
        if questions:
            # presentar al usuario en batch
            for q in questions:
                q.answer = "yes"  # o "no"
                q.accepted = True
            hitl.process_answers()
    """

    def __init__(self, pack_memory=None, auto_apply: bool = True):
        """
        Args:
            pack_memory: instancia de PackMemory para consulta/registro
            auto_apply: si True, aplica correcciones de memoria automáticamente
        """
        self.pack_memory = pack_memory
        self.auto_apply = auto_apply
        self._questions: list[Question] = []
        self._auto_corrections: list[dict] = []
        self._critical_blocker: Optional[Question] = None

    def should_ask(self, error_type: str, field_name: str, context: str = "") -> bool:
        """
        Determinar si se debe preguntar al usuario o si hay una corrección automática.

        Returns True si se debe preguntar, False si se puede auto-aplicar o skip.
        """
        if not self.pack_memory:
            return True

        # ¿Fue rechazado antes? No preguntar
        if self.pack_memory.should_skip(error_type, field_name, context):
            logger.debug("Skip corrección para %s.%s (rechazada antes)", error_type, field_name)
            return False

        # ¿Es regla automática? Aplicar sin preguntar
        if self.auto_apply:
            auto_record = self.pack_memory.should_auto_apply(error_type, field_name, context)
            if auto_record:
                self._auto_corrections.append({
                    "field_name": field_name,
                    "original": auto_record.original_value,
                    "corrected": auto_record.corrected_value,
                    "method": "pack_memory_auto",
                })
                return False

        return True

    def ask(self, question: Question) -> None:
        """
        Registrar una pregunta. Si es CRITICAL, marca como blocker.

        Antes de preguntar, verifica PackMemory para evitar preguntar lo ya respondido.
        """
        if question.level == QuestionLevel.CRITICAL:
            # Verificar memoria primero
            if self.pack_memory and not self.should_ask("critical", question.field_name, question.context):
                return
            self._critical_blocker = question
        elif question.level == QuestionLevel.WARNING:
            if self.pack_memory and not self.should_ask("warning", question.field_name, question.context):
                return
            self._questions.append(question)
        else:  # INFO
            logger.info("[HITL] %s: %s", question.field_name, question.message)

    def has_critical_blocker(self) -> bool:
        """¿Hay un blocker crítico que requiera atención inmediata?"""
        return self._critical_blocker is not None

    def get_critical_blocker(self) -> Optional[Question]:
        """Obtener el blocker crítico actual."""
        return self._critical_blocker

    def get_pending_questions(self) -> list[Question]:
        """Obtener todas las preguntas WARNING pendientes (para batch)."""
        return [q for q in self._questions if q.answer is None]

    def get_auto_corrections(self) -> list[dict]:
        """Obtener correcciones aplicadas automáticamente desde memoria."""
        return self._auto_corrections

    def process_answers(self) -> None:
        """
        Procesar respuestas del usuario y registrar en PackMemory.

        Llamar después de que el usuario haya respondido todas las preguntas.
        """
        if not self.pack_memory:
            return

        # Procesar preguntas batch (WARNING)
        for q in self._questions:
            if q.answer is None:
                continue
            accepted = q.answer.lower() in ("yes", "si", "s", "y", "true", "1", "accept")
            self.pack_memory.record_correction(
                error_type="warning",
                field_name=q.field_name,
                original_value="",
                corrected_value=q.suggested_value,
                correction_method="user_guided",
                user_accepted=accepted,
                context=q.context,
            )

        # Procesar blocker crítico
        if self._critical_blocker and self._critical_blocker.answer is not None:
            accepted = self._critical_blocker.answer.lower() in ("yes", "si", "s", "y", "true", "1", "accept")
            self.pack_memory.record_correction(
                error_type="critical",
                field_name=self._critical_blocker.field_name,
                original_value="",
                corrected_value=self._critical_blocker.suggested_value,
                correction_method="user_guided",
                user_accepted=accepted,
                context=self._critical_blocker.context,
            )

    def get_summary(self) -> dict:
        """Resumen del estado de interacción humana."""
        return {
            "total_questions": len(self._questions),
            "pending": len(self.get_pending_questions()),
            "auto_corrections": len(self._auto_corrections),
            "has_critical_blocker": self.has_critical_blocker(),
            "critical_field": self._critical_blocker.field_name if self._critical_blocker else None,
        }
