"""
Script para regenerar el VRP Domain Pack desde los modelos Pydantic del solver.

Uso:
    python -m scripts.generate_vrp_pack

Esto extrae el JSON Schema de vrp_solver.models.OptimizeRequest,
lo convierte a FieldSchema, y combina con las reglas semánticas
del pack.yaml manual.

El 80% (schema, tipos, required, rangos) se auto-genera.
El 20% (reglas semánticas, mapeos, validadores) se preserva del YAML.
"""

import sys
from pathlib import Path

# Asegurar que root está en el path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.core.domain_pack import DomainPack, PackLoader, FieldSchema


def generate_vrp_pack() -> DomainPack:
    """Generar pack VRP combinando auto-generación + reglas manuales."""

    # Cargar pack manual (reglas semánticas, mapeos, validadores)
    pack_yaml_path = root_dir / "src" / "domain_packs" / "vrp" / "pack.yaml"
    manual_pack = PackLoader.from_yaml(str(pack_yaml_path))

    # Auto-generar schema desde Pydantic
    # El solver está en D:\codebase\vrp-solver
    solver_src = Path("D:/codebase/vrp-solver/src")
    if solver_src.exists():
        sys.path.insert(0, str(solver_src))
        try:
            auto_pack = PackLoader.from_pydantic(
                module_path="vrp_solver.models",
                class_name="OptimizeRequest",
                pack_name="vrp",
                semantic_rules=manual_pack.semantic_rules,
                inference_mappings=manual_pack.inference_mappings,
                custom_validators=manual_pack.custom_validators,
            )
            # Combinar: schema auto-generado + metadata manual
            auto_pack.metadata = manual_pack.metadata
            auto_pack.solver_contract = manual_pack.solver_contract
            auto_pack.description = manual_pack.description
            return auto_pack
        except ImportError as e:
            print(f"[WARN] No se pudo importar vrp_solver.models: {e}")
            print("[INFO] Usando pack manual solamente")
            return manual_pack
    else:
        print(f"[WARN] Solver no encontrado en {solver_src}")
        print("[INFO] Usando pack manual solamente")
        return manual_pack


if __name__ == "__main__":
    pack = generate_vrp_pack()
    output_path = root_dir / "src" / "domain_packs" / "vrp" / "pack_generated.json"
    PackLoader.save_pack(pack, str(output_path))
    print(f"Pack generado: {pack.name} v{pack.version}")
    print(f"  Campos: {len(pack.schema_fields)}")
    print(f"  Reglas semánticas: {len(pack.semantic_rules)}")
    print(f"  Mapeos: {len(pack.inference_mappings)}")
    print(f"  Validadores custom: {len(pack.custom_validators)}")
    print(f"  Guardado en: {output_path}")
