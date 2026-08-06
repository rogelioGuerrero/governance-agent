#!/usr/bin/env python3
"""Quickstart demo — Governance Agent con datos sintéticos.

Ejecuta una validación completa sin necesidad de LLM ni solver externo.
Demuestra las 3 capas: estructural, reglas de dominio, y reporte de issues.

Uso:
    uv run python scripts/quickstart.py
"""

import json
import sys
from pathlib import Path

# Asegurar que 'src' sea importable
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from src.core.domain_pack import DomainPack, FieldSchema, PackLoader
from src.core.validator import ValidationEngine
from src.core.pack_memory import PackMemory
from src.core.human_loop import HumanInTheLoop


def build_demo_pack() -> DomainPack:
    """Construir un pack de demostración para un ministerio genérico."""
    return DomainPack(
        name="demo_ministerio",
        schema_fields={
            "productor_id": FieldSchema(
                name="productor_id",
                type="string",
                required=True,
                description="Identificador único del productor",
            ),
            "cultivo": FieldSchema(
                name="cultivo",
                type="string",
                required=True,
                enum=["arroz", "maiz", "papa", "cafe", "cacao", "frijol"],
                description="Cultivo principal registrado",
            ),
            "hectareas": FieldSchema(
                name="hectareas",
                type="float",
                required=True,
                min=0.1,
                max=10000,
                description="Superficie sembrada en hectáreas",
            ),
            "rendimiento": FieldSchema(
                name="rendimiento",
                type="float",
                required=True,
                min=0,
                max=100,
                description="Rendimiento en toneladas por hectárea",
            ),
            "departamento": FieldSchema(
                name="departamento",
                type="string",
                required=True,
                description="Departamento de ubicación",
            ),
            "latitud": FieldSchema(
                name="latitud",
                type="float",
                required=True,
                min=-90,
                max=90,
                description="Latitud de la finca",
            ),
            "longitud": FieldSchema(
                name="longitud",
                type="float",
                required=True,
                min=-180,
                max=180,
                description="Longitud de la finca",
            ),
        },
        semantic_rules=[
            "El rendimiento por hectárea debe ser plausible para el cultivo registrado (arroz: 4-8, maiz: 3-12, papa: 15-25, cafe: 0.8-2.5, cacao: 0.5-1.5, frijol: 0.8-2.0)",
            "La latitud y longitud deben corresponder al departamento declarado",
            "Las hectáreas sembradas no pueden exceder la superficie típica de un pequeño productor (max 500 ha)",
        ],
    )


def build_clean_data() -> dict:
    """Datos correctos que deben pasar validación sin issues."""
    return {
        "productor_id": "P-001",
        "cultivo": "arroz",
        "hectareas": 5.0,
        "rendimiento": 6.5,
        "departamento": "bolivar",
        "latitud": 8.88,
        "longitud": -74.78,
    }


def build_dirty_data() -> dict:
    """Datos con errores que deben ser detectados por las capas de validación."""
    return {
        "productor_id": "P-002",
        "cultivo": "papa",
        "hectareas": 0.05,
        "rendimiento": 50.0,
        "departamento": "bolivar",
        "latitud": 45.0,
        "longitud": -74.78,
    }


def run_demo():
    print("=" * 60)
    print("  Governance Agent — Quickstart Demo")
    print("  Calidad de datos para políticas públicas")
    print("=" * 60)

    # 1. Construir el pack del ministerio
    print("\n[1] Cargando Domain Pack: demo_ministerio")
    pack = build_demo_pack()
    print(f"    Schema fields: {len(pack.schema_fields)}")
    print(f"    Semantic rules: {len(pack.semantic_rules)}")

    # 2. Configurar el motor de validación (sin LLM para demo rápida)
    print("\n[2] Configurando motor de validación")
    memory = PackMemory("demo_ministerio")
    hitl = HumanInTheLoop(pack_memory=memory)
    engine = ValidationEngine(pack=pack, pack_memory=memory, hitl=hitl)
    print("    Capas activas: estructural + reglas de dominio")
    print("    (Capa semántica IA: requiere API key — ver .env)")

    # 3. Validar datos limpios
    print("\n[3] Validando datos CORRECTOS")
    clean = build_clean_data()
    print(f"    Input: {json.dumps(clean, ensure_ascii=False, indent=2)}")
    result_clean = engine.validate(clean)
    print(f"    Válido: {result_clean.is_valid}")
    print(f"    Issues: {len(result_clean.issues)}")
    if result_clean.issues:
        for issue in result_clean.issues:
            print(f"      [{issue.severity}] {issue.field_name}: {issue.message}")

    # 4. Validar datos con errores
    print("\n[4] Validando datos CON ERRORES")
    dirty = build_dirty_data()
    print(f"    Input: {json.dumps(dirty, ensure_ascii=False, indent=2)}")
    result_dirty = engine.validate(dirty)
    print(f"    Válido: {result_dirty.is_valid}")
    print(f"    Issues: {len(result_dirty.issues)}")
    for issue in result_dirty.issues:
        print(f"      [{issue.severity}] {issue.field_name}: {issue.message}")
        if issue.suggested_value:
            print(f"        Sugerencia: {issue.suggested_value}")

    # 5. Resumen
    print("\n" + "=" * 60)
    print("  Resumen")
    print("=" * 60)
    print(f"  Datos limpios: {'PASS' if result_clean.is_valid else 'FAIL'}")
    print(f"  Datos con errores: {'PASS' if result_dirty.is_valid else 'FAIL'}")
    print(f"  Issues detectados en datos sucios: {len(result_dirty.issues)}")
    print()
    print("  Para activar validación semántica con IA:")
    print("    1. Configurar GROQ_API_KEY en .env")
    print("    2. Ejecutar: uv run python scripts/test_llm_semantic.py")
    print()
    print("  Para probar con datos reales del solver VRP:")
    print("    uv run python scripts/test_real_data.py")
    print()
    print("  Para probar el orquestador completo (validar + corregir + solver):")
    print("    uv run python scripts/test_orchestrator.py")


if __name__ == "__main__":
    run_demo()
