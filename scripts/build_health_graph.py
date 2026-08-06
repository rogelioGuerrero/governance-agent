"""Construir un grafo limpio focalizado en salud con quality scores realistas.

Dominio: salud
Fuentes: hospital, censo, seguro, siv (sistema de vacunacion)
~10 conceptos con quality scores asignados
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import os
from dotenv import load_dotenv
load_dotenv()
from src.graph.catalog import NomencladorGraph
from src.graph.schema import (
    ConceptNode, FieldNode, ClassifierNode, SourceNode, EdgeType,
)

NOMENCLADOR_PATH = Path(__file__).parent.parent / "nomenclador" / "nomenclador.json"

# Limpiar grafo anterior (JSON + PostgreSQL)
if NOMENCLADOR_PATH.exists():
    NOMENCLADOR_PATH.unlink()
    print(f"Grafo anterior (JSON) eliminado: {NOMENCLADOR_PATH}")

# Limpiar PostgreSQL si esta configurado
import os
db_url = os.environ.get("DATABASE_URL", "")
if db_url:
    try:
        import psycopg
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM governance.graph_edges")
                cur.execute("DELETE FROM governance.graph_nodes")
                cur.execute("DELETE FROM governance.nomenclador_version")
            conn.commit()
        print("PostgreSQL limpiado (graph_nodes, graph_edges, nomenclador_version)")
    except Exception as e:
        print(f"PostgreSQL cleanup omitido: {e}")

g = NomencladorGraph()

# === SOURCES ===
sources = [
    SourceNode(id="source:hospital", name="Hospital", description="Sistema de informacion hospitalaria"),
    SourceNode(id="source:censo", name="Censo", description="Censo nacional de poblacion"),
    SourceNode(id="source:seguro", name="Seguro", description="Sistema del seguro de salud"),
    SourceNode(id="source:siv", name="SIV", description="Sistema de Informacion de Vacunacion"),
]
for s in sources:
    g.add_source(s)

# === CONCEPTS ===
concepts = [
    ConceptNode(id="concept:edad", name="edad", definition="Edad del individuo en anos",
                standard=None, population="poblacion general", capture_method="calculada desde fecha_nacimiento",
                custodian="Ministerio de Salud", review_status="approved"),
    ConceptNode(id="concept:sexo", name="sexo", definition="Sexo biologico del individuo",
                standard="ISO 5218", population="poblacion general", capture_method="autorreportado",
                custodian="Ministerio de Salud", review_status="approved"),
    ConceptNode(id="concept:diagnostico", name="diagnostico", definition="Diagnostico medico principal",
                standard="ICD-10", population="pacientes atendidos", capture_method="registro clinico",
                custodian="Ministerio de Salud", review_status="approved"),
    ConceptNode(id="concept:municipio", name="municipio", definition="Municipio de residencia",
                standard=None, population="poblacion general", capture_method="autorreportado",
                custodian="DIGESTYC", review_status="approved"),
    ConceptNode(id="concept:fecha_nacimiento", name="fecha_nacimiento", definition="Fecha de nacimiento del individuo",
                standard="ISO 8601", population="poblacion general", capture_method="registro civil",
                custodian="Registro Nacional", review_status="approved"),
    ConceptNode(id="concept:nivel_educativo", name="nivel_educativo", definition="Maximo nivel educativo alcanzado",
                standard="ISCED 2011", population="poblacion 15+", capture_method="autorreportado",
                custodian="Ministerio de Educacion", review_status="approved"),
    ConceptNode(id="concept:fecha_ingreso", name="fecha_ingreso", definition="Fecha de ingreso hospitalario",
                standard="ISO 8601", population="pacientes hospitalizados", capture_method="registro administrativo",
                custodian="Ministerio de Salud", review_status="approved"),
    ConceptNode(id="concept:fecha_vacunacion", name="fecha_vacunacion", definition="Fecha de aplicacion de vacuna",
                standard="ISO 8601", population="poblacion vacunada", capture_method="registro de vacunacion",
                custodian="Ministerio de Salud", review_status="approved"),
    ConceptNode(id="concept:estado_vacunacion", name="estado_vacunacion", definition="Estado del esquema de vacunacion",
                standard=None, population="poblacion objetivo", capture_method="calculado desde dosis aplicadas",
                custodian="Ministerio de Salud", review_status="approved"),
    ConceptNode(id="concept:residencia_urbana", name="residencia_urbana", definition="Indicador urbano/rural de residencia",
                standard=None, population="poblacion general", capture_method="clasificacion territorial",
                custodian="DIGESTYC", review_status="approved"),
]
for c in concepts:
    g.add_concept(c)

# === CLASSIFIERS ===
classifiers = [
    ClassifierNode(id="classifier:sexo", name="Valores de sexo", standard="ISO 5218",
                   values={"M": "Masculino", "F": "Femenino", "0": "No especificado", "9": "Desconocido"},
                   version_label="ISO 5218:2022"),
    ClassifierNode(id="classifier:icd10", name="Codigos ICD-10", standard="ICD-10",
                   values={"A00-B99": "Enfermedades infecciosas", "C00-D49": "Neoplasias",
                           "E00-E90": "Endocrinas", "F01-F99": "Mentales",
                           "I00-I99": "Circulatorio", "J00-J99": "Respiratorio"},
                   version_label="ICD-10 2019"),
    ClassifierNode(id="classifier:isced", name="Niveles ISCED", standard="ISCED 2011",
                   values={"0": "Educacion inicial", "1": "Primaria", "2": "Secundaria baja",
                           "3": "Secundaria alta", "4": "Post-secundaria", "5": "Corto",
                           "6": "Grado", "7": "Maestria", "8": "Doctorado"},
                   version_label="ISCED 2011"),
]
for c in classifiers:
    g.add_classifier(c)

g.link_clasificador("concept:sexo", "classifier:sexo")
g.link_clasificador("concept:diagnostico", "classifier:icd10")
g.link_clasificador("concept:nivel_educativo", "classifier:isced")

# === FIELDS con quality scores realistas ===
# (source_db, column, concept_id, quality_score, completeness, review_status)
fields = [
    # hospital - buena calidad
    ("hospital", "edad", "concept:edad", 0.85, 0.92, "approved"),
    ("hospital", "sexo", "concept:sexo", 0.90, 0.98, "approved"),
    ("hospital", "cie10", "concept:diagnostico", 0.78, 0.85, "approved"),
    ("hospital", "fecha_nac", "concept:fecha_nacimiento", 0.88, 0.95, "approved"),
    ("hospital", "f_ingreso", "concept:fecha_ingreso", 0.82, 0.90, "approved"),
    ("hospital", "escolaridad", "concept:nivel_educativo", 0.45, 0.60, "approved"),

    # censo - calidad media-alta
    ("censo", "edad", "concept:edad", 0.80, 0.99, "approved"),
    ("censo", "sexo", "concept:sexo", 0.92, 0.99, "approved"),
    ("censo", "municipio", "concept:municipio", 0.65, 0.95, "approved"),
    ("censo", "fecha_nacimiento", "concept:fecha_nacimiento", 0.85, 0.88, "approved"),
    ("censo", "nivel_educativo", "concept:nivel_educativo", 0.55, 0.82, "approved"),
    ("censo", "residencia_urbana", "concept:residencia_urbana", 0.58, 0.90, "approved"),

    # seguro - calidad media
    ("seguro", "sexo_paciente", "concept:sexo", 0.88, 0.95, "approved"),
    ("seguro", "diag_cie", "concept:diagnostico", 0.72, 0.80, "approved"),
    ("seguro", "fecha_nacimiento", "concept:fecha_nacimiento", 0.82, 0.90, "approved"),
    ("seguro", "fecha_alta", "concept:fecha_ingreso", 0.75, 0.85, "approved"),

    # SIV - calidad baja (sistema nuevo, datos incompletos)
    ("siv", "edad_nino", "concept:edad", 0.40, 0.70, "approved"),
    ("siv", "fecha_vacunacion", "concept:fecha_vacunacion", 0.35, 0.65, "approved"),
    ("siv", "estado_esquema", "concept:estado_vacunacion", 0.30, 0.55, "approved"),
]

for source_db, column, concept_id, qs, completeness, status in fields:
    field_id = f"field:{source_db}:{column}"
    f = FieldNode(
        id=field_id,
        source_db=source_db,
        column=column,
        quality_score=qs,
        completeness=completeness,
        review_status=status,
    )
    g.add_field(f)
    g.link_implementa(field_id, concept_id)
    g.link_fuente(field_id, f"source:{source_db}")

# === Guardar ===
g.bump_version("major", "Grafo focalizado en salud con quality scores realistas")
g.save(str(NOMENCLADOR_PATH))

print(f"Grafo creado: {g.graph.number_of_nodes()} nodos, {g.graph.number_of_edges()} aristas")
print(f"Conceptos: {len(g.find_all_concepts())}")
print(f"Version: {g.version}")
print(f"Guardado en: {NOMENCLADOR_PATH}")

# === Generar insights acumulados por fuente ===
print("\nGenerando insights por fuente...")
from src.discover import generate_insights_for_source

for s in sources:
    try:
        saved = generate_insights_for_source(g, s.id, domain="salud")
        print(f"  {s.name}: {len(saved)} insights generados")
    except Exception as e:
        print(f"  {s.name}: Error - {e}")

g.save(str(NOMENCLADOR_PATH))
print(f"\nGrafo final: {g.graph.number_of_nodes()} nodos, {g.graph.number_of_edges()} aristas")
print(f"Insights acumulados: {len(g.find_insights())}")
