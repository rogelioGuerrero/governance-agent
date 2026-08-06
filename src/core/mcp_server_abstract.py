"""
MCP Server abstracto del Governance Agent.

A diferencia del mcp_server.py original (que está atado al dominio salud/nomenclador),
este servidor es agnóstico y acepta cualquier Domain Pack.

Tools expuestas:
- validate_data: validar datos contra el domain pack cargado
- get_pack_info: info del domain pack activo
- get_pack_rules: reglas semánticas del pack
- get_field_schema: schema de un campo específico
- get_auto_corrections: correcciones automáticas aprendidas
- get_memory_stats: estadísticas de memoria del pack

Uso:
    # Con pack VRP
    python -m src.core.mcp_server_abstract --pack vrp

    # Con pack salud
    python -m src.core.mcp_server_abstract --pack salud
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Asegurar root del proyecto en path (para que 'src' sea paquete importable)
_root = str(Path(__file__).parent.parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from mcp.server.fastmcp import FastMCP

from src.core.domain_pack import DomainPack, PackLoader
from src.core.pack_memory import PackMemory
from src.core.human_loop import HumanInTheLoop
from src.core.validator import ValidationEngine
from src.core.standards import register_pack_standards


def create_mcp_server(pack: DomainPack, pack_memory: PackMemory = None) -> FastMCP:
    """
    Crear servidor MCP configurado con un Domain Pack específico.

    Args:
        pack: Domain Pack cargado
        pack_memory: memoria del pack (opcional, se crea si no se pasa)

    Returns:
        Instancia FastMCP lista para run()
    """
    if pack_memory is None:
        pack_memory = PackMemory(pack.name)

    # Registrar estándares del pack si los tiene
    register_pack_standards(pack)

    mcp = FastMCP(
        f"governance-{pack.name}",
        instructions=(
            f"Governance Agent MCP Server para dominio '{pack.name}'. "
            f"Valida datos contra el schema y reglas semánticas del dominio. "
            f"Usa validate_data para validar payloads antes de enviar al solver."
        ),
    )

    @mcp.tool()
    def validate_data(data: str) -> str:
        """
        Validar datos contra el domain pack activo.

        Args:
            data: JSON string con los datos a validar

        Returns:
            Reporte de validación: issues, correcciones auto, preguntas humanas
        """
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            return f"Error: JSON inválido — {e}"

        hitl = HumanInTheLoop(pack_memory=pack_memory)
        engine = ValidationEngine(pack=pack, pack_memory=pack_memory, hitl=hitl)
        result = engine.validate(parsed)

        return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)

    @mcp.tool()
    def get_pack_info() -> str:
        """Obtener información del domain pack activo."""
        info = {
            "name": pack.name,
            "version": pack.version,
            "description": pack.description,
            "fields": len(pack.schema_fields),
            "semantic_rules": len(pack.semantic_rules),
            "inference_mappings": len(pack.inference_mappings),
            "custom_validators": len(pack.custom_validators),
            "has_solver_contract": pack.solver_contract is not None,
        }
        return json.dumps(info, ensure_ascii=False, indent=2)

    @mcp.tool()
    def get_pack_rules() -> str:
        """Obtener las reglas semánticas del domain pack activo."""
        return pack.get_system_prompt_rules()

    @mcp.tool()
    def get_field_schema(field_name: str) -> str:
        """
        Obtener el schema de un campo específico del pack.

        Args:
            field_name: nombre del campo (o alias)

        Returns:
            Schema del campo: tipo, required, min/max, pattern, enum, aliases
        """
        fs = pack.get_field(field_name)
        if not fs:
            available = ", ".join(pack.schema_fields.keys())
            return f"Campo '{field_name}' no encontrado. Disponibles: {available}"

        return json.dumps({
            "name": fs.name,
            "type": fs.type,
            "required": fs.required,
            "min": fs.min,
            "max": fs.max,
            "pattern": fs.pattern,
            "enum": fs.enum,
            "description": fs.description,
            "aliases": fs.aliases,
        }, ensure_ascii=False, indent=2)

    @mcp.tool()
    def get_auto_corrections() -> str:
        """Obtener las correcciones automáticas aprendidas por la memoria del pack."""
        auto_rules = pack_memory.get_auto_rules()
        if not auto_rules:
            return "No hay correcciones automáticas aprendidas todavía."

        lines = [f"Correcciones automáticas para '{pack.name}' ({len(auto_rules)} reglas):\n"]
        for r in auto_rules:
            lines.append(
                f"- {r.field_name} ({r.error_type}): "
                f"'{r.original_value}' → '{r.corrected_value}' "
                f"[aceptada {r.count} veces, método: {r.correction_method}]"
            )
        return "\n".join(lines)

    @mcp.tool()
    def get_memory_stats() -> str:
        """Obtener estadísticas de la memoria del pack."""
        return json.dumps(pack_memory.get_stats(), ensure_ascii=False, indent=2)

    @mcp.tool()
    def record_correction(field_name: str, original_value: str, corrected_value: str,
                          accepted: bool, error_type: str = "manual") -> str:
        """
        Registrar una corrección manualmente en la memoria del pack.

        Útil cuando el usuario corrige algo fuera del flujo automático
        y quiere que el agente aprenda para futuras veces.

        Args:
            field_name: campo corregido
            original_value: valor original
            corrected_value: valor corregido
            accepted: si el usuario aceptó la corrección
            error_type: tipo de error (manual, type_mismatch, etc.)
        """
        record = pack_memory.record_correction(
            error_type=error_type,
            field_name=field_name,
            original_value=original_value,
            corrected_value=corrected_value,
            correction_method="manual",
            user_accepted=accepted,
        )
        return f"Corrección registrada. Total para {field_name}: {record.count} veces."

    return mcp


def load_pack_by_name(pack_name: str, packs_dir: Path = None) -> DomainPack:
    """Cargar un domain pack por nombre desde el directorio de packs."""
    if packs_dir is None:
        packs_dir = Path(__file__).parent.parent / "domain_packs"

    yaml_path = packs_dir / pack_name / "pack.yaml"
    json_path = packs_dir / pack_name / "pack_generated.json"

    # Preferir el generado (tiene schema auto-generado) si existe
    if json_path.exists():
        return PackLoader.from_json(str(json_path))
    elif yaml_path.exists():
        return PackLoader.from_yaml(str(yaml_path))
    else:
        raise FileNotFoundError(
            f"No se encontró pack '{pack_name}' en {packs_dir}. "
            f"Buscado: {yaml_path} o {json_path}"
        )


def main():
    """Entry point. Uso: python -m src.core.mcp_server_abstract --pack vrp"""
    import argparse
    parser = argparse.ArgumentParser(description="MCP Server del Governance Agent")
    parser.add_argument("--pack", required=True, help="Nombre del domain pack (ej: vrp, salud)")
    args = parser.parse_args()

    pack = load_pack_by_name(args.pack)
    mcp = create_mcp_server(pack)
    mcp.run()


if __name__ == "__main__":
    main()
