"""
ValidationEngine: motor de validación por capas abstracto.

Capa 1: Validación estructural (código, sin LLM)
  - Tipos, formatos, required, rangos, enum
  - 100% reutilizable, 0% dominio

Capa 2: Validación semántica (LLM + Domain Pack)
  - Se invoca solo si la Capa 1 pasa
  - Usa reglas semánticas del Domain Pack
  - Detecta inconsistencias lógicas
  - Sugiere correcciones

Capa 3: Validadores custom (plugins del Domain Pack)
  - Funciones Python específicas del dominio
  - Ej: check_coords_in_bounds, check_time_window_overlap

Integración con PackMemory y HumanInTheLoop:
  - Antes de reportar un error, consulta memoria para auto-aplicar
  - Antes de preguntar al usuario, consulta memoria para evitar repetir
  - Después de resolver, registra en memoria para futuro
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .domain_pack import DomainPack, FieldSchema
from .pack_memory import PackMemory
from .human_loop import HumanInTheLoop, Question, QuestionLevel

logger = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    """Issue encontrado durante validación."""
    layer: str  # "structural", "semantic", "custom"
    severity: str  # "info", "warning", "error"
    field_name: str
    issue_type: str  # missing_field, type_mismatch, out_of_range, pattern_mismatch, enum_invalid, semantic_violation
    message: str
    original_value: Any = None
    suggested_value: Any = None
    context: str = ""

    def to_dict(self) -> dict:
        return {
            "layer": self.layer, "severity": self.severity, "field_name": self.field_name,
            "issue_type": self.issue_type, "message": self.message,
            "original_value": self.original_value, "suggested_value": self.suggested_value,
            "context": self.context,
        }


@dataclass
class ValidationResult:
    """Resultado completo de la validación."""
    is_valid: bool = False
    payload: Optional[dict] = None  # datos limpios listos para el solver
    issues: list[ValidationIssue] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)  # correcciones aplicadas
    warnings: list[str] = field(default_factory=list)
    auto_corrections: list[dict] = field(default_factory=list)
    human_questions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "payload": self.payload,
            "issues": [i.to_dict() for i in self.issues],
            "actions": self.actions,
            "warnings": self.warnings,
            "auto_corrections": self.auto_corrections,
            "human_questions": self.human_questions,
        }


class ValidationEngine:
    """
    Motor de validación abstracto por capas.

    Uso:
        engine = ValidationEngine(pack=vrp_pack, pack_memory=memory, hitl=hitl)
        result = engine.validate(raw_data)
        if result.is_valid:
            send_to_solver(result.payload)
        elif hitl.has_critical_blocker():
            ask_user(hitl.get_critical_blocker())
    """

    def __init__(self, pack: DomainPack, pack_memory: PackMemory = None,
                 hitl: HumanInTheLoop = None, llm_client=None):
        """
        Args:
            pack: Domain Pack con schema, reglas y validadores
            pack_memory: memoria de correcciones (opcional)
            hitl: human-in-the-loop (opcional)
            llm_client: cliente LLM para Capa 2 (opcional, si no hay se omite)
        """
        self.pack = pack
        self.pack_memory = pack_memory
        self.hitl = hitl
        self.llm_client = llm_client

    def validate(self, data: dict) -> ValidationResult:
        """
        Ejecutar las 3 capas de validación en secuencia.

        Capa 1 (estructural) → si hay errores críticos, parar
        Capa 2 (semántica con LLM) → solo si Capa 1 pasa
        Capa 3 (validadores custom) → solo si Capa 1 pasa
        """
        result = ValidationResult()
        working_data = dict(data) if isinstance(data, dict) else {}

        # ── Capa 1: Validación estructural ──
        structural_issues = self._validate_structural(working_data, result)
        has_critical = any(i.severity == "error" for i in structural_issues)

        if has_critical:
            result.is_valid = False
            return result

        # ── Capa 3: Validadores custom (antes de LLM para ser deterministas) ──
        custom_issues = self._validate_custom(working_data, result)
        has_critical = any(i.severity == "error" for i in custom_issues)

        if has_critical:
            result.is_valid = False
            return result

        # ── Capa 2: Validación semántica (LLM) ──
        if self.llm_client:
            semantic_issues = self._validate_semantic(working_data, result)
            has_critical = any(i.severity == "error" for i in semantic_issues)
            if has_critical:
                result.is_valid = False
                return result

        # ── Consolidar resultado ──
        result.payload = working_data
        result.is_valid = not any(i.severity == "error" for i in result.issues)

        # Agregar auto-correcciones y preguntas humanas al resultado
        if self.hitl:
            result.auto_corrections = self.hitl.get_auto_corrections()
            result.human_questions = [q.to_dict() for q in self.hitl.get_pending_questions()]

        return result

    def _validate_structural(self, data: dict, result: ValidationResult) -> list[ValidationIssue]:
        """Capa 1: validación estructural contra el schema del Domain Pack."""
        issues = []

        for field_name, fs in self.pack.schema_fields.items():
            value = self._find_value(data, field_name, fs.aliases)

            if value is None:
                if fs.required:
                    issue = ValidationIssue(
                        layer="structural", severity="error", field_name=field_name,
                        issue_type="missing_field",
                        message=f"Campo requerido '{field_name}' no encontrado",
                    )
                    issues.append(issue)
                    result.issues.append(issue)
                continue

            # Validar tipo
            type_ok, coerced = self._check_type(value, fs.type)
            if not type_ok:
                # Intentar corrección desde memoria
                if self.pack_memory and not self.pack_memory.should_skip("type_mismatch", field_name):
                    auto = self.pack_memory.should_auto_apply("type_mismatch", field_name)
                    if auto:
                        value = auto.corrected_value
                        result.actions.append({
                            "field": field_name, "original": str(value),
                            "fixed": auto.corrected_value, "method": "pack_memory_auto",
                        })
                        type_ok, coerced = self._check_type(value, fs.type)

                if not type_ok:
                    issue = ValidationIssue(
                        layer="structural", severity="error", field_name=field_name,
                        issue_type="type_mismatch",
                        message=f"Campo '{field_name}' esperaba tipo {fs.type}, obtuvo {type(value).__name__}",
                        original_value=value,
                        suggested_value=str(coerced) if coerced is not None else None,
                    )
                    issues.append(issue)
                    result.issues.append(issue)
                    continue
            else:
                if coerced is not None and coerced != value:
                    result.actions.append({
                        "field": field_name, "original": str(value),
                        "fixed": str(coerced), "method": "type_coercion",
                    })
                    data[field_name] = coerced
                    value = coerced

            # Validar rango
            if fs.min is not None or fs.max is not None:
                try:
                    num_val = float(value)
                    if fs.min is not None and num_val < fs.min:
                        issue = ValidationIssue(
                            layer="structural", severity="warning", field_name=field_name,
                            issue_type="out_of_range",
                            message=f"Campo '{field_name}' valor {num_val} < mínimo {fs.min}",
                            original_value=value, suggested_value=fs.min,
                        )
                        issues.append(issue)
                        result.issues.append(issue)
                    if fs.max is not None and num_val > fs.max:
                        issue = ValidationIssue(
                            layer="structural", severity="warning", field_name=field_name,
                            issue_type="out_of_range",
                            message=f"Campo '{field_name}' valor {num_val} > máximo {fs.max}",
                            original_value=value, suggested_value=fs.max,
                        )
                        issues.append(issue)
                        result.issues.append(issue)
                except (ValueError, TypeError):
                    pass

            # Validar enum
            if fs.enum and str(value) not in fs.enum:
                issue = ValidationIssue(
                    layer="structural", severity="error", field_name=field_name,
                    issue_type="enum_invalid",
                    message=f"Campo '{field_name}' valor '{value}' no está en enum {fs.enum}",
                    original_value=value,
                )
                issues.append(issue)
                result.issues.append(issue)

            # Validar pattern (regex)
            if fs.pattern:
                import re
                if not re.match(fs.pattern, str(value)):
                    issue = ValidationIssue(
                        layer="structural", severity="warning", field_name=field_name,
                        issue_type="pattern_mismatch",
                        message=f"Campo '{field_name}' valor '{value}' no coincide con patrón {fs.pattern}",
                        original_value=value,
                    )
                    issues.append(issue)
                    result.issues.append(issue)

        return issues

    def _validate_custom(self, data: dict, result: ValidationResult) -> list[ValidationIssue]:
        """Capa 3: ejecutar validadores custom del Domain Pack."""
        issues = []
        for validator_ref in self.pack.custom_validators:
            try:
                module_name, func_name = validator_ref.rsplit(":", 1)
                # Intentar import directo, si falla anteponer src.
                try:
                    module = importlib.import_module(module_name)
                except ModuleNotFoundError:
                    module = importlib.import_module(f"src.{module_name}")
                func = getattr(module, func_name)
                validator_issues = func(data, self.pack)
                for vi in validator_issues:
                    issues.append(vi)
                    result.issues.append(vi)
            except Exception as e:
                logger.warning("Error en validador custom %s: %s", validator_ref, e)
        return issues

    def _validate_semantic(self, data: dict, result: ValidationResult) -> list[ValidationIssue]:
        """Capa 2: validación semántica usando LLM + reglas del Domain Pack."""
        issues = []
        if not self.llm_client:
            return issues

        rules_text = self.pack.get_system_prompt_rules()
        system_prompt = f"""You are a data validation agent. Analyze the data against the domain rules.
Report ONLY actual issues. Do not report things that are working correctly.

{rules_text}

For each issue found, respond in JSON:
{{
  "issues": [
    {{
      "field_name": "...",
      "severity": "warning|error",
      "issue_type": "semantic_violation",
      "message": "...",
      "suggested_value": "..."
    }}
  ]
}}

If no issues, respond: {{"issues": []}}"""

        data_str = str(data)[:4000]  # truncar para no exceder tokens
        try:
            llm_result = self.llm_client.call(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": data_str},
                ],
                temperature=0.1,
            )
            if llm_result.ok and llm_result.content:
                import json
                parsed = json.loads(llm_result.content)
                for issue_data in parsed.get("issues", []):
                    issue = ValidationIssue(
                        layer="semantic",
                        severity=issue_data.get("severity", "warning"),
                        field_name=issue_data.get("field_name", ""),
                        issue_type="semantic_violation",
                        message=issue_data.get("message", ""),
                        suggested_value=issue_data.get("suggested_value"),
                    )
                    issues.append(issue)
                    result.issues.append(issue)

                    # Si hay sugerencia y no es error, preguntar al usuario (vía HITL)
                    if issue.suggested_value and self.hitl and issue.severity != "error":
                        if self.hitl.should_ask("semantic_violation", issue.field_name):
                            self.hitl.ask(Question(
                                level=QuestionLevel.WARNING,
                                field_name=issue.field_name,
                                message=issue.message,
                                suggested_value=issue.suggested_value,
                            ))
        except Exception as e:
            logger.warning("Error en validación semántica LLM: %s", e)

        return issues

    def _find_value(self, data: dict, field_name: str, aliases: list[str] = None) -> Any:
        """Buscar valor por nombre o alias."""
        if field_name in data:
            return data[field_name]
        if aliases:
            for alias in aliases:
                if alias in data:
                    return data[alias]
                # Case-insensitive
                for key in data:
                    if key.lower() == alias.lower():
                        return data[key]
        # Case-insensitive del nombre mismo
        for key in data:
            if key.lower() == field_name.lower():
                return data[key]
        return None

    def _check_type(self, value: Any, expected_type: str) -> tuple[bool, Any]:
        """
        Verificar tipo y intentar coerción.

        Returns (is_valid, coerced_value). Si coercion fue aplicada, coerced_value != value.
        """
        type_checkers = {
            "string": lambda v: (isinstance(v, str), str(v) if v is not None else None),
            "integer": lambda v: self._check_int(v),
            "float": lambda v: self._check_float(v),
            "boolean": lambda v: self._check_bool(v),
            "array": lambda v: (isinstance(v, list), v),
            "object": lambda v: (isinstance(v, dict), v),
        }
        checker = type_checkers.get(expected_type)
        if checker:
            return checker(value)
        return (True, value)

    def _check_int(self, value: Any) -> tuple[bool, Any]:
        if isinstance(value, int) and not isinstance(value, bool):
            return (True, value)
        if isinstance(value, float) and value.is_integer():
            return (True, int(value))
        if isinstance(value, str):
            try:
                return (True, int(value))
            except ValueError:
                return (False, None)
        return (False, None)

    def _check_float(self, value: Any) -> tuple[bool, Any]:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return (True, float(value))
        if isinstance(value, str):
            try:
                return (True, float(value))
            except ValueError:
                return (False, None)
        return (False, None)

    def _check_bool(self, value: Any) -> tuple[bool, Any]:
        if isinstance(value, bool):
            return (True, value)
        if isinstance(value, str):
            if value.lower() in ("true", "1", "yes", "si", "s", "y"):
                return (True, True)
            if value.lower() in ("false", "0", "no", "n"):
                return (True, False)
        if isinstance(value, int):
            return (True, bool(value))
        return (False, None)
