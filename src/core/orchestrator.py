"""
Orquestador VRP: agente con tool calling que integra governance + solver.

Flujo:
1. Recibe datos crudos del usuario (o de un sistema externo)
2. Llama a validate_data (governance agent) → si hay errores, intenta corregir
3. Si hay warnings, los acumula y continúa (no detiene)
4. Llama al solver VRP (/optimize)
5. Si el solver falla, feed-back al LLM para diagnosticar y corregir
6. Si el solver tiene nodos no asignados, informa al usuario
7. Retorna rutas optimizadas + reporte de calidad de datos

El orquestador usa tool calling del LLM (Groq) para decidir qué hacer.
No es un pipeline rígido — el LLM razona sobre los resultados y decide下一步.

Tools disponibles para el LLM:
- validate_data: validar datos contra el domain pack VRP
- call_solver: enviar datos al solver VRP y recibir rutas
- apply_correction: aplicar una corrección a los datos
- get_field_schema: consultar el schema de un campo del pack
- finish: terminar y entregar resultado al usuario
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> Optional[dict]:
    """
    Extraer JSON de una respuesta del LLM que puede incluir texto adicional.
    Busca el primer { y el último } y parsea lo que está entre ellos.
    """
    # Intentar parse directo
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Buscar bloques ```json ... ```
    import re
    json_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_block:
        try:
            return json.loads(json_block.group(1))
        except json.JSONDecodeError:
            pass

    # Buscar el primer { y el último }
    first = text.find('{')
    last = text.rfind('}')
    if first != -1 and last != -1 and last > first:
        try:
            return json.loads(text[first:last + 1])
        except json.JSONDecodeError:
            pass

    return None


@dataclass
class OrchestratorResult:
    """Resultado del orquestador VRP."""
    success: bool = False
    routes: list[dict] = field(default_factory=list)
    statistics: Optional[dict] = None
    unassigned: list[dict] = field(default_factory=list)
    validation_issues: list[dict] = field(default_factory=list)
    corrections_applied: list[dict] = field(default_factory=list)
    human_questions: list[dict] = field(default_factory=list)
    solver_warnings: list[str] = field(default_factory=list)
    solver_errors: list[dict] = field(default_factory=list)
    iterations: int = 0
    message: str = ""

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "routes": self.routes,
            "statistics": self.statistics,
            "unassigned": self.unassigned,
            "validation_issues": self.validation_issues,
            "corrections_applied": self.corrections_applied,
            "human_questions": self.human_questions,
            "solver_warnings": self.solver_warnings,
            "solver_errors": self.solver_errors,
            "iterations": self.iterations,
            "message": self.message,
        }


class VRPOrchestrator:
    """
    Orquestador que integra governance agent + solver VRP via tool calling.

    Uso:
        orch = VRPOrchestrator(pack=vrp_pack, solver_url="http://localhost:8000")
        result = orch.run(raw_data)
        if result.success:
            print(f"{len(result.routes)} rutas optimizadas")
        if result.human_questions:
            # presentar preguntas al usuario
    """

    MAX_ITERATIONS = 3  # máximo de cicros validate → correct → solve

    def __init__(self, pack, solver_url: str = "http://localhost:8000",
                 llm_adapter=None, pack_memory=None):
        """
        Args:
            pack: DomainPack VRP cargado
            solver_url: URL del solver VRP (FastAPI /optimize)
            llm_adapter: LLMAdapter para tool calling (si None, se crea)
            pack_memory: PackMemory para correcciones persistentes
        """
        self.pack = pack
        self.solver_url = solver_url.rstrip("/")
        self.llm = llm_adapter
        self.pack_memory = pack_memory

        # Lazy import para evitar circular
        if self.llm is None:
            from src.core.llm_adapter import LLMAdapter
            self.llm = LLMAdapter(json_mode=False, temperature=0.3, max_tokens=4000)

        if self.pack_memory is None:
            from src.core.pack_memory import PackMemory
            self.pack_memory = PackMemory(pack.name)

    def run(self, data: dict, auto_correct: bool = True) -> OrchestratorResult:
        """
        Ejecutar el ciclo completo: validar → corregir → solver.

        Args:
            data: datos crudos del usuario (formato VRP request)
            auto_correct: si True, intenta corregir errores automáticamente

        Returns:
            OrchestratorResult con rutas, issues, correcciones y preguntas
        """
        from src.core.human_loop import HumanInTheLoop
        from src.core.validator import ValidationEngine

        result = OrchestratorResult()
        working_data = json.loads(json.dumps(data))  # deep copy

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            result.iterations = iteration
            logger.info("Orquestador — iteración %d/%d", iteration, self.MAX_ITERATIONS)

            # 1. Validar
            hitl = HumanInTheLoop(pack_memory=self.pack_memory)
            engine = ValidationEngine(
                pack=self.pack, pack_memory=self.pack_memory,
                hitl=hitl, llm_client=self.llm if iteration == 1 else None,
            )
            val_result = engine.validate(working_data)

            # Recolectar issues
            result.validation_issues = [i.to_dict() for i in val_result.issues]
            result.human_questions = [q.to_dict() for q in hitl.get_pending_questions()]
            result.corrections_applied.extend(val_result.actions)

            # 2. ¿Hay errores críticos?
            critical_errors = [i for i in val_result.issues if i.severity == "error"]

            if critical_errors and auto_correct and iteration < self.MAX_ITERATIONS:
                # Intentar corregir con LLM
                logger.info(" %d errores críticos, intentando corrección automática...", len(critical_errors))
                corrected = self._attempt_correction(working_data, critical_errors)
                if corrected:
                    working_data = corrected
                    continue  # re-validar
                else:
                    result.message = "No se pudieron corregir los errores automáticamente"
                    result.solver_errors = [i.to_dict() for i in critical_errors]
                    return result

            if critical_errors and not auto_correct:
                result.message = "Errores críticos detectados, no se envió al solver"
                result.solver_errors = [i.to_dict() for i in critical_errors]
                return result

            if critical_errors:
                # Última iteración, no se pudo corregir
                result.message = f"Errores persisten tras {iteration} iteraciones"
                result.solver_errors = [i.to_dict() for i in critical_errors]
                return result

            # 3. No hay errores — enviar al solver
            logger.info("Datos validados, enviando al solver %s", self.solver_url)
            solver_result = self._call_solver(working_data)

            if solver_result is None:
                result.message = "Solver no disponible o error de conexión"
                result.solver_errors = [{"error": "connection_failed", "message": f"No se pudo conectar a {self.solver_url}"}]
                return result

            # 4. ¿Solver respondió OK?
            if solver_result.get("status") == "success":
                result.success = True
                result.routes = solver_result.get("routes", [])
                result.statistics = solver_result.get("statistics")
                result.unassigned = [u if isinstance(u, dict) else u.model_dump() if hasattr(u, "model_dump") else str(u)
                                     for u in solver_result.get("unassigned_nodes", [])]
                result.solver_warnings = solver_result.get("warnings", [])
                result.message = solver_result.get("message", "Optimización completada")
                logger.info("Solver OK: %d rutas, %d no asignados", len(result.routes), len(result.unassigned))
                return result

            # 5. Solver falló — ¿errores de validación del solver?
            solver_errors = solver_result.get("errors", [])
            if solver_errors and auto_correct and iteration < self.MAX_ITERATIONS:
                logger.info("Solver rechazó datos: %s, intentando corregir...", solver_errors)
                # Convertir errores del solver a issues y intentar corregir
                corrected = self._attempt_solver_correction(working_data, solver_errors)
                if corrected:
                    working_data = corrected
                    continue
                else:
                    result.message = "Solver rechazó los datos y no se pudieron corregir"
                    result.solver_errors = solver_errors
                    return result

            # Solver falló sin corrección posible
            result.message = solver_result.get("message", "Solver error")
            result.solver_errors = solver_errors
            result.solver_warnings = solver_result.get("warnings", [])
            return result

        result.message = f"Máximo de iteraciones ({self.MAX_ITERATIONS}) alcanzado"
        return result

    def _call_solver(self, data: dict) -> Optional[dict]:
        """Llamar al solver VRP via HTTP POST."""
        try:
            with httpx.Client(timeout=120) as client:
                resp = client.post(
                    f"{self.solver_url}/optimize",
                    json=data,
                    headers={"Content-Type": "application/json"},
                )
                if resp.status_code == 422:
                    # Validation error del solver (Pydantic)
                    error_detail = resp.json().get("detail", [])
                    return {
                        "status": "error",
                        "message": "Datos rechazados por el solver (validación Pydantic)",
                        "errors": [{"error": "pydantic_validation", "message": str(error_detail)}],
                    }
                resp.raise_for_status()
                return resp.json()
        except httpx.ConnectError:
            logger.warning("Solver no disponible en %s", self.solver_url)
            return None
        except httpx.HTTPError as e:
            logger.warning("Error HTTP llamando solver: %s", e)
            return None
        except Exception as e:
            logger.error("Error inesperado llamando solver: %s", e)
            return None

    def _attempt_correction(self, data: dict, errors: list) -> Optional[dict]:
        """
        Usar LLM para intentar corregir errores críticos automáticamente.

        Returns datos corregidos o None si no se pudo.
        """
        rules_text = self.pack.get_system_prompt_rules()
        errors_text = json.dumps([e.to_dict() for e in errors], ensure_ascii=False, indent=2)

        system_prompt = f"""You are a data correction agent for VRP (Vehicle Routing Problem).
Your job is to fix critical errors in the data so it can be sent to the solver.

{rules_text}

You will receive:
1. The current data (JSON)
2. The list of errors found

Respond with the CORRECTED data as a single JSON object. Do not include explanations.
If you cannot fix an error, keep the original value and the error will be reported to the user."""

        user_prompt = f"""Current data:
{json.dumps(data, ensure_ascii=False)[:3000]}

Errors to fix:
{errors_text}

Return the corrected JSON:"""

        try:
            result = self.llm.call(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=4000,
            )
            if result.ok and result.content:
                corrected = _extract_json(result.content)
                if corrected:
                    logger.info("LLM corrigió datos automáticamente")
                    return corrected
                else:
                    logger.warning("LLM respondió pero no se pudo extraer JSON")
        except Exception as e:
            logger.warning("Corrección LLM falló: %s", e)

        return None

    def _attempt_solver_correction(self, data: dict, solver_errors: list) -> Optional[dict]:
        """
        Corregir errores reportados por el solver (no por el governance agent).
        """
        rules_text = self.pack.get_system_prompt_rules()
        errors_text = json.dumps(solver_errors, ensure_ascii=False, indent=2)

        system_prompt = f"""You are a data correction agent for VRP.
The solver rejected the data. Fix the errors and return the corrected JSON.

{rules_text}

Return ONLY the corrected JSON object, no explanations."""

        user_prompt = f"""Data rejected by solver:
{json.dumps(data, ensure_ascii=False)[:3000]}

Solver errors:
{errors_text}

Return the corrected JSON:"""

        try:
            result = self.llm.call(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=4000,
            )
            if result.ok and result.content:
                corrected = _extract_json(result.content)
                if corrected:
                    logger.info("LLM corrigió errores del solver")
                    return corrected
                else:
                    logger.warning("LLM respondió pero no se pudo extraer JSON")
        except Exception as e:
            logger.warning("Corrección de solver falló: %s", e)

        return None
