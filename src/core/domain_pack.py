"""
Domain Pack: plugin de dominio para el Governance Agent.

Un Domain Pack contiene:
- schema: definición de campos esperados (tipos, required, rangos)
- semantic_rules: reglas en lenguaje natural para el LLM
- inference_mappings: sinónimos de campos (ej: lat = latitude = latitud)
- custom_validators: funciones Python específicas del dominio
- solver_contract: referencia al contrato del solver (para auto-generación)

El núcleo del governance agent es agnóstico al dominio.
El Domain Pack es lo único que conoce el dominio.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class FieldSchema:
    """Esquema de un campo esperado por el dominio."""
    name: str
    type: str  # string, integer, float, boolean, datetime, array, object
    required: bool = True
    min: Optional[float] = None
    max: Optional[float] = None
    pattern: Optional[str] = None  # regex
    enum: Optional[list[str]] = None
    description: str = ""
    aliases: list[str] = field(default_factory=list)


@dataclass
class SolverContract:
    """Referencia al contrato del solver para auto-generación."""
    pydantic_module: str = ""  # ej: "vrp_solver.models"
    pydantic_class: str = ""  # ej: "OptimizeRequest"
    json_schema_path: Optional[str] = None  # path alternativo al JSON schema


@dataclass
class DomainPack:
    """
    Plugin de dominio cargable por el governance agent.

    Auto-generable 80% desde el solver (Pydantic → JSON Schema → Pack).
    20% manual: reglas semánticas, validadores custom, mapeos.
    """
    name: str
    version: str = "1.0.0"
    description: str = ""
    schema_fields: dict[str, FieldSchema] = field(default_factory=dict)
    semantic_rules: list[str] = field(default_factory=list)
    inference_mappings: dict[str, list[str]] = field(default_factory=dict)
    custom_validators: list[str] = field(default_factory=list)  # module:function names
    solver_contract: Optional[SolverContract] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_field(self, name: str) -> Optional[FieldSchema]:
        """Buscar campo por nombre o alias."""
        if name in self.schema_fields:
            return self.schema_fields[name]
        for field_name, fs in self.schema_fields.items():
            if name.lower() in [a.lower() for a in fs.aliases] or name.lower() == field_name.lower():
                return fs
        return None

    def get_system_prompt_rules(self) -> str:
        """Generar texto de reglas semánticas para el system prompt del LLM."""
        lines = [f"Domain: {self.name}", f"Description: {self.description}", ""]
        if self.semantic_rules:
            lines.append("Semantic rules:")
            for i, rule in enumerate(self.semantic_rules, 1):
                lines.append(f"  {i}. {rule}")
        if self.schema_fields:
            lines.append("")
            lines.append("Expected fields:")
            for name, fs in self.schema_fields.items():
                req = "required" if fs.required else "optional"
                lines.append(f"  - {name} ({fs.type}, {req}): {fs.description}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serializar a dict para persistencia."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "schema_fields": {
                name: {
                    "name": fs.name, "type": fs.type, "required": fs.required,
                    "min": fs.min, "max": fs.max, "pattern": fs.pattern,
                    "enum": fs.enum, "description": fs.description, "aliases": fs.aliases,
                }
                for name, fs in self.schema_fields.items()
            },
            "semantic_rules": self.semantic_rules,
            "inference_mappings": self.inference_mappings,
            "custom_validators": self.custom_validators,
            "solver_contract": {
                "pydantic_module": self.solver_contract.pydantic_module,
                "pydantic_class": self.solver_contract.pydantic_class,
                "json_schema_path": self.solver_contract.json_schema_path,
            } if self.solver_contract else None,
            "metadata": self.metadata,
        }


class PackLoader:
    """Carga Domain Packs desde archivos YAML/JSON o auto-generados desde Pydantic."""

    @staticmethod
    def from_yaml(path: str) -> DomainPack:
        """Cargar pack desde archivo YAML."""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return PackLoader._from_dict(data)

    @staticmethod
    def from_json(path: str) -> DomainPack:
        """Cargar pack desde archivo JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return PackLoader._from_dict(data)

    @staticmethod
    def from_pydantic(module_path: str, class_name: str, pack_name: str = "",
                      semantic_rules: list[str] = None,
                      inference_mappings: dict[str, list[str]] = None,
                      custom_validators: list[str] = None) -> DomainPack:
        """
        Auto-generar pack desde modelos Pydantic del solver.

        Extrae el JSON Schema de los modelos Pydantic y lo convierte a FieldSchema.
        El 80% (schema, tipos, required, rangos) se genera automáticamente.
        El 20% (reglas semánticas, mapeos, validadores) se pasa manualmente.
        """
        import importlib
        module = importlib.import_module(module_path)
        model_cls = getattr(module, class_name)

        json_schema = model_cls.model_json_schema()
        pack = PackLoader._from_json_schema(json_schema, pack_name or class_name.lower())
        pack.solver_contract = SolverContract(
            pydantic_module=module_path,
            pydantic_class=class_name,
        )
        if semantic_rules:
            pack.semantic_rules = semantic_rules
        if inference_mappings:
            pack.inference_mappings = inference_mappings
        if custom_validators:
            pack.custom_validators = custom_validators
        return pack

    @staticmethod
    def _from_dict(data: dict) -> DomainPack:
        """Construir DomainPack desde dict (YAML/JSON cargado)."""
        schema_fields = {}
        for name, fs_data in data.get("schema_fields", {}).items():
            schema_fields[name] = FieldSchema(
                name=name,
                type=fs_data.get("type", "string"),
                required=fs_data.get("required", True),
                min=fs_data.get("min"),
                max=fs_data.get("max"),
                pattern=fs_data.get("pattern"),
                enum=fs_data.get("enum"),
                description=fs_data.get("description", ""),
                aliases=fs_data.get("aliases", []),
            )

        contract_data = data.get("solver_contract")
        solver_contract = None
        if contract_data:
            solver_contract = SolverContract(
                pydantic_module=contract_data.get("pydantic_module", ""),
                pydantic_class=contract_data.get("pydantic_class", ""),
                json_schema_path=contract_data.get("json_schema_path"),
            )

        return DomainPack(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            schema_fields=schema_fields,
            semantic_rules=data.get("semantic_rules", []),
            inference_mappings=data.get("inference_mappings", {}),
            custom_validators=data.get("custom_validators", []),
            solver_contract=solver_contract,
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def _from_json_schema(json_schema: dict, pack_name: str) -> DomainPack:
        """Convertir JSON Schema (Pydantic) a DomainPack con FieldSchemas.

        Recursivamente extrae campos de sub-modelos referenciados via $ref
        dentro de arrays (ej: locations → Location.id, Location.coords, etc.).
        """
        schema_fields = {}
        defs = json_schema.get("$defs", json_schema.get("definitions", {}))

        def resolve_ref(ref: str) -> dict:
            if ref.startswith("#/$defs/"):
                key = ref.split("/")[-1]
                return defs.get(key, {})
            return {}

        def map_type(prop: dict) -> str:
            if "$ref" in prop:
                ref_data = resolve_ref(prop["$ref"])
                prop = {**ref_data, **prop}
                prop.pop("$ref", None)

            json_type = prop.get("type", "")
            any_of = prop.get("anyOf", [])
            if any_of:
                json_type = next((t.get("type") for t in any_of if t.get("type") != "null"), "")

            type_map = {
                "string": "string", "integer": "integer", "number": "float",
                "boolean": "boolean", "array": "array", "object": "object",
            }
            return type_map.get(json_type, "string")

        def parse_field(name: str, prop: dict, required: bool) -> FieldSchema:
            if "$ref" in prop:
                ref_data = resolve_ref(prop["$ref"])
                prop = {**ref_data, **prop}
                prop.pop("$ref", None)

            any_of = prop.get("anyOf", [])
            if any_of:
                # Tomar el primer tipo no-null
                for t in any_of:
                    if t.get("type") != "null":
                        prop = {**prop, **t}
                        break

            return FieldSchema(
                name=name,
                type=map_type(prop),
                required=required,
                min=prop.get("minimum") or prop.get("exclusiveMinimum"),
                max=prop.get("maximum") or prop.get("exclusiveMaximum"),
                pattern=prop.get("pattern"),
                enum=prop.get("enum"),
                description=prop.get("description", ""),
                aliases=[],
            )

        def parse_array_item_fields(prefix: str, prop: dict) -> None:
            """Extraer campos de sub-modelos dentro de un array.

            Ej: locations (array of Location) → locations.id, locations.coords, etc.
            """
            items = prop.get("items", {})
            if "$ref" in items:
                ref_data = resolve_ref(items["$ref"])
                sub_props = ref_data.get("properties", {})
                sub_required = set(ref_data.get("required", []))
                for sub_name, sub_prop in sub_props.items():
                    full_name = f"{prefix}.{sub_name}"
                    if full_name not in schema_fields:
                        schema_fields[full_name] = parse_field(full_name, sub_prop, sub_name in sub_required)

        properties = json_schema.get("properties", {})
        required_list = set(json_schema.get("required", []))

        for name, prop in properties.items():
            if name in ("config",):
                continue
            schema_fields[name] = parse_field(name, prop, name in required_list)

            # Si es array con $ref en items, extraer sub-campos
            if prop.get("type") == "array" or (
                prop.get("anyOf") and any(t.get("type") == "array" for t in prop.get("anyOf", []))
            ):
                parse_array_item_fields(name, prop)

        return DomainPack(
            name=pack_name,
            schema_fields=schema_fields,
        )

    @staticmethod
    def save_pack(pack: DomainPack, path: str) -> None:
        """Persistir pack como JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(pack.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("Pack guardado: %s → %s", pack.name, path)
