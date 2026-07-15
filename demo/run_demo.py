"""
Demo: Interoperabilidad MAG <-> MARN
====================================

Demuestra como el governance-agent integra dos ministerios con datos
que cubren las mismas parcelas pero con codificaciones, unidades y
formatos incompatibles.

Ejecutar:
    python -m src.cli demo-agri-env

O paso a paso con los comandos CLI individuales (ver mas abajo).
"""
import sys
import os

# Asegurar que el paquete src es importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_demo():
    from src.cli import (
        console, load_graph, cmd_profile, cmd_nomenclar,
        cmd_catalog, cmd_interop, cmd_transform, cmd_impact,
    )
    from src.standards import register_standard

    base = os.path.dirname(os.path.abspath(__file__))
    csv_mag = os.path.join(base, "mag_produccion_agricola.csv")
    csv_marn = os.path.join(base, "marn_cobertura_forestal.csv")

    console.print("\n[bold cyan]========================================[/bold cyan]")
    console.print("[bold cyan]DEMO: Interoperabilidad MAG <-> MARN[/bold cyan]")
    console.print("[bold cyan]========================================[/bold cyan]\n")

    console.print("[dim]Ministerio de Agricultura (MAG): Anuario de Estadisticas Agropecuarias[/dim]")
    console.print("[dim]Ministerio de Medio Ambiente (MARN): Inventario de Cobertura Forestal[/dim]")
    console.print("[dim]Pregunta: La expansion agricola correlaciona con perdida forestal?[/dim]\n")

    # === PASO 1: Registrar estandares del dominio ===
    console.print("[bold yellow]PASO 1: Registrar estandares del dominio agroambiental[/bold yellow]\n")

    register_standard(
        standard_id="ISO_3166_2_SV",
        name="Codigos de departamento de El Salvador (14 departamentos)",
        domain="geografia",
        standard_type="classifier",
        values={
            "01": "San Salvador",
            "02": "Santa Ana",
            "03": "La Union",
            "04": "San Miguel",
            "05": "Usulutan",
            "06": "Sonsonate",
            "07": "Ahuachapan",
            "08": "La Libertad",
            "09": "Chalatenango",
            "10": "Cabanas",
            "11": "La Paz",
            "12": "San Vicente",
            "13": "Cuscatlan",
            "14": "Morazan",
        },
        name_hints=["depto", "departamento", "department", "cod_depto", "nombre_depto"],
    )
    console.print("  [green]OK[/green] ISO_3166_2_SV — 14 departamentos (01=San Salvador ... 14=Morazan)")

    register_standard(
        standard_id="ISO_8601",
        name="Formato de fecha ISO 8601",
        domain="transversal",
        standard_type="format",
        regex=r"\d{4}",
        name_hints=["fecha", "fecha_monitoreo", "fecha_siembra", "dob", "fecha_ingreso", "anio", "year"],
    )
    console.print("  [green]OK[/green] ISO_8601 — formato de fecha/anio (YYYY o YYYY-MM-DD)")

    register_standard(
        standard_id="CORINE_LAND_COVER",
        name="Corine Land Cover (cobertura del suelo)",
        domain="ambiental",
        standard_type="classifier",
        values={
            "211": "Tierras agricolas no irrigadas",
            "212": "Tierras agricolas irrigadas permanentemente",
            "221": "Pastizales",
            "243": "Mosaico agricola-forestal",
        },
        name_hints=["cobertura", "land_cover", "uso_suelo", "cover"],
    )
    console.print("  [green]OK[/green] CORINE_LAND_COVER — cobertura del suelo (211=agricola, 221=pastizal, ...)")

    register_standard(
        standard_id="CULTIVO_SV",
        name="Catalogo de cultivos de El Salvador",
        domain="agricultura",
        standard_type="classifier",
        values={
            "maiz": "Maiz",
            "frijol": "Frijol",
            "sorgo": "Sorgo (maicillo)",
            "cafe": "Cafe",
            "arroz": "Arroz",
        },
        name_hints=["cultivo", "crop", "especie", "producto"],
    )
    console.print("  [green]OK[/green] CULTIVO_SV — catalogo de cultivos (maiz, frijol, sorgo, cafe, arroz)")

    register_standard(
        standard_id="UNIDADES_SV",
        name="Unidades de medida agropecuarias y ambientales",
        domain="transversal",
        standard_type="classifier",
        values={
            "mz": "Manzanas (7,000 m2)",
            "ha": "Hectareas (10,000 m2)",
            "qq": "Quintales (45.36 kg)",
            "kg": "Kilogramos",
            "km2": "Kilometros cuadrados",
        },
        name_hints=["area_mz", "cobertura_ha", "perdida_ha", "ganancia_ha", "neta_ha", "produccion_qq", "superficie_km2", "area_ha"],
    )
    console.print("  [green]OK[/green] UNIDADES_SV — unidades de medida (mz, ha, qq, kg, km2)")

    console.print()

    # === PASO 2: Perfilar ambos CSVs ===
    console.print("[bold yellow]PASO 2: Perfilar fuentes de datos[/bold yellow]\n")
    console.print("[dim]Profileando MAG...[/dim]")
    cmd_profile(csv_mag, auto=True)
    console.print()
    console.print("[dim]Profileando MARN...[/dim]")
    cmd_profile(csv_marn, auto=True)
    console.print()

    # === PASO 3: Descubrir y mapear variables ===
    console.print("[bold yellow]PASO 3: Descubrir y mapear variables (nomenclar)[/bold yellow]\n")
    console.print("[dim]Nomenclar MAG...[/dim]")
    cmd_nomenclar(csv_mag, auto=True)
    console.print()
    console.print("[dim]Nomenclar MARN...[/dim]")
    cmd_nomenclar(csv_marn, auto=True)
    console.print()

    # === PASO 4: Mostrar catalogo unificado ===
    console.print("[bold yellow]PASO 4: Catalogo unificado[/bold yellow]\n")
    cmd_catalog()
    console.print()

    # === PASO 5: Verificar interoperabilidad ===
    console.print("[bold yellow]PASO 5: Verificar interoperabilidad MAG <-> MARN[/bold yellow]\n")
    cmd_interop("mag_produccion_agricola", "marn_cobertura_forestal")
    console.print()

    # === PASO 6: Generar transformacion SQL ===
    console.print("[bold yellow]PASO 6: Generar transformacion SQL (MARN -> MAG)[/bold yellow]\n")
    cmd_transform("marn_cobertura_forestal", "mag_produccion_agricola")
    console.print()

    # === PASO 7: Analisis de impacto ===
    console.print("[bold yellow]PASO 7: Analisis de impacto[/bold yellow]\n")
    cmd_impact("depto")
    console.print()

    console.print("[bold green]========================================[/bold green]")
    console.print("[bold green]DEMO COMPLETADA[/bold green]")
    console.print("[bold green]========================================[/bold green]\n")
    console.print("[dim]Pregunta de politica publica: expansion agricola vs perdida forestal.[/dim]")
    console.print("[dim]Datos basados en Anuario MAG 2022-2023 e Inventario Forestal MARN 2018.[/dim]")
    console.print("[dim]Transformaciones SQL generadas, validacion de guardrails, trazabilidad completa.[/dim]")


if __name__ == "__main__":
    run_demo()
