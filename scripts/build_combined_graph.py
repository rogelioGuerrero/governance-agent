"""Construir un grafo combinado salud + educacion para demostrar discover cross-domain.

Dominios: salud, educacion
Fuentes salud: hospital, censo, seguro, siv
Fuentes educacion: mined, censo (compartido), pnud

Conceptos compartidos (puentes cross-domain):
- edad, sexo, municipio, nivel_educativo, resididencia_urbana

Esto permite que discover genere hipotesis que cruzan ambos dominios.
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

# === Limpiar grafo anterior (JSON + PostgreSQL) ===
if NOMENCLADOR_PATH.exists():
    NOMENCLADOR_PATH.unlink()
    print(f"Grafo anterior (JSON) eliminado: {NOMENCLADOR_PATH}")

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
        print("PostgreSQL limpiado")
    except Exception as e:
        print(f"PostgreSQL cleanup omitido: {e}")

g = NomencladorGraph()

# === SOURCES ===
sources = [
    # Salud
    SourceNode(id="source:hospital", name="Hospital", description="Sistema de informacion hospitalaria"),
    SourceNode(id="source:censo", name="Censo", description="Censo nacional de poblacion"),
    SourceNode(id="source:seguro", name="Seguro", description="Sistema del seguro de salud"),
    SourceNode(id="source:siv", name="SIV", description="Sistema de Informacion de Vacunacion"),
    # Educacion
    SourceNode(id="source:mined", name="MINED", description="Ministerio de Educacion - sistema escolar"),
    SourceNode(id="source:pnud", name="PNUD", description="Datos socioeconomios PNUD/ONU"),
]
for s in sources:
    g.add_source(s)

# === CONCEPTOS COMPARTIDOS (puentes cross-domain) ===
shared_concepts = [
    ConceptNode(id="concept:edad", name="edad", definition="Edad del individuo en anos",
                standard=None, population="poblacion general", capture_method="calculada desde fecha_nacimiento",
                custodian="Ministerio de Salud", review_status="approved"),
    ConceptNode(id="concept:sexo", name="sexo", definition="Sexo biologico del individuo",
                standard="ISO 5218", population="poblacion general", capture_method="autorreportado",
                custodian="Ministerio de Salud", review_status="approved"),
    ConceptNode(id="concept:municipio", name="municipio", definition="Municipio de residencia",
                standard=None, population="poblacion general", capture_method="autorreportado",
                custodian="DIGESTYC", review_status="approved"),
    ConceptNode(id="concept:nivel_educativo", name="nivel_educativo", definition="Maximo nivel educativo alcanzado",
                standard="ISCED 2011", population="poblacion 15+", capture_method="autorreportado",
                custodian="Ministerio de Educacion", review_status="approved"),
    ConceptNode(id="concept:residencia_urbana", name="residencia_urbana", definition="Indicador urbano/rural de residencia",
                standard=None, population="poblacion general", capture_method="clasificacion territorial",
                custodian="DIGESTYC", review_status="approved"),
]

# === CONCEPTOS SALUD ===
health_concepts = [
    ConceptNode(id="concept:diagnostico", name="diagnostico", definition="Diagnostico medico principal",
                standard="ICD-10", population="pacientes atendidos", capture_method="registro clinico",
                custodian="Ministerio de Salud", review_status="approved"),
    ConceptNode(id="concept:fecha_nacimiento", name="fecha_nacimiento", definition="Fecha de nacimiento del individuo",
                standard="ISO 8601", population="poblacion general", capture_method="registro civil",
                custodian="Registro Nacional", review_status="approved"),
    ConceptNode(id="concept:fecha_ingreso", name="fecha_ingreso", definition="Fecha de ingreso hospitalario",
                standard="ISO 8601", population="pacientes hospitalizados", capture_method="registro administrativo",
                custodian="Ministerio de Salud", review_status="approved"),
    ConceptNode(id="concept:fecha_vacunacion", name="fecha_vacunacion", definition="Fecha de aplicacion de vacuna",
                standard="ISO 8601", population="poblacion vacunada", capture_method="registro de vacunacion",
                custodian="Ministerio de Salud", review_status="approved"),
    ConceptNode(id="concept:estado_vacunacion", name="estado_vacunacion", definition="Estado del esquema de vacunacion",
                standard=None, population="poblacion objetivo", capture_method="calculado desde dosis aplicadas",
                custodian="Ministerio de Salud", review_status="approved"),
    ConceptNode(id="concept:peso_nacer", name="peso_nacer", definition="Peso al nacer en gramos",
                standard=None, population="recien nacidos", capture_method="registro clinico",
                custodian="Ministerio de Salud", review_status="approved"),
]

# === CONCEPTOS EDUCACION ===
education_concepts = [
    ConceptNode(id="concept:asistencia_escolar", name="asistencia_escolar", definition="Indicador de asistencia regular a centro educativo",
                standard=None, population="poblacion 4-18 anos", capture_method="autorreportado / registro escolar",
                custodian="Ministerio de Educacion", review_status="approved"),
    ConceptNode(id="concept:rendimiento_academico", name="rendimiento_academico", definition="Promedio de notas del estudiante",
                standard=None, population="estudiantes inscritos", capture_method="evaluacion docente",
                custodian="Ministerio de Educacion", review_status="approved"),
    ConceptNode(id="concept:grado_escolar", name="grado_escolar", definition="Grado o ano escolar cursado",
                standard=None, population="estudiantes inscritos", capture_method="registro administrativo",
                custodian="Ministerio de Educacion", review_status="approved"),
    ConceptNode(id="concept:escuela_id", name="escuela_id", definition="Identificador del centro educativo",
                standard="Codigo MINED", population="centros educativos", capture_method="registro administrativo",
                custodian="Ministerio de Educacion", review_status="approved"),
    ConceptNode(id="concept:ratio_docente", name="ratio_docente", definition="Ratio alumnos por docente en el aula",
                standard=None, population="centros educativos", capture_method="calculo administrativo",
                custodian="Ministerio de Educacion", review_status="approved"),
    ConceptNode(id="concept:acceso_internet", name="acceso_internet", definition="Disponibilidad de internet en el hogar",
                standard=None, population="poblacion general", capture_method="autorreportado",
                custodian="DIGESTYC", review_status="approved"),
    ConceptNode(id="concept:desercion_escolar", name="desercion_escolar", definition="Abandono del sistema educativo antes de completar",
                standard=None, population="estudiantes inscritos", capture_method="calculo desde matricula vs asistencia",
                custodian="Ministerio de Educacion", review_status="approved"),
]

for c in shared_concepts + health_concepts + education_concepts:
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
    ClassifierNode(id="classifier:urbano_rural", name="Urbano/Rural", standard=None,
                   values={"U": "Urbano", "R": "Rural"},
                   version_label="DIGESTYC 2024"),
]
for c in classifiers:
    g.add_classifier(c)

g.link_clasificador("concept:sexo", "classifier:sexo")
g.link_clasificador("concept:diagnostico", "classifier:icd10")
g.link_clasificador("concept:nivel_educativo", "classifier:isced")
g.link_clasificador("concept:residencia_urbana", "classifier:urbano_rural")

# === FIELDS SALUD ===
health_fields = [
    # hospital - buena calidad
    ("hospital", "edad", "concept:edad", 0.85, 0.92, "approved"),
    ("hospital", "sexo", "concept:sexo", 0.90, 0.98, "approved"),
    ("hospital", "cie10", "concept:diagnostico", 0.78, 0.85, "approved"),
    ("hospital", "fecha_nac", "concept:fecha_nacimiento", 0.88, 0.95, "approved"),
    ("hospital", "f_ingreso", "concept:fecha_ingreso", 0.82, 0.90, "approved"),
    ("hospital", "escolaridad", "concept:nivel_educativo", 0.45, 0.60, "approved"),
    ("hospital", "peso_nacer", "concept:peso_nacer", 0.72, 0.80, "approved"),
    ("hospital", "municipio", "concept:municipio", 0.68, 0.85, "approved"),

    # censo - calidad media-alta (compartido entre dominios)
    ("censo", "edad", "concept:edad", 0.80, 0.99, "approved"),
    ("censo", "sexo", "concept:sexo", 0.92, 0.99, "approved"),
    ("censo", "municipio", "concept:municipio", 0.65, 0.95, "approved"),
    ("censo", "fecha_nacimiento", "concept:fecha_nacimiento", 0.85, 0.88, "approved"),
    ("censo", "nivel_educativo", "concept:nivel_educativo", 0.55, 0.82, "approved"),
    ("censo", "residencia_urbana", "concept:residencia_urbana", 0.58, 0.90, "approved"),
    ("censo", "asiste_escuela", "concept:asistencia_escolar", 0.70, 0.88, "approved"),
    ("censo", "internet_hogar", "concept:acceso_internet", 0.50, 0.75, "approved"),

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

# === FIELDS EDUCACION ===
education_fields = [
    # MINED - calidad media-alta
    ("mined", "grado", "concept:grado_escolar", 0.82, 0.95, "approved"),
    ("mined", "promedio_notas", "concept:rendimiento_academico", 0.65, 0.78, "approved"),
    ("mined", "cod_escuela", "concept:escuela_id", 0.90, 0.99, "approved"),
    ("mined", "alumnos_por_docente", "concept:ratio_docente", 0.75, 0.85, "approved"),
    ("mined", "abandono", "concept:desercion_escolar", 0.45, 0.60, "approved"),
    ("mined", "municipio_escuela", "concept:municipio", 0.88, 0.98, "approved"),
    ("mined", "nivel_educativo", "concept:nivel_educativo", 0.78, 0.92, "approved"),
    ("mined", "edad_alumno", "concept:edad", 0.72, 0.90, "approved"),
    ("mined", "sexo_alumno", "concept:sexo", 0.85, 0.95, "approved"),

    # PNUD - datos socioeconomios, calidad variable
    ("pnud", "idh_municipio", "concept:municipio", 0.70, 0.92, "approved"),
    ("pnud", "pobreza_multidim", "concept:residencia_urbana", 0.62, 0.80, "approved"),
    ("pnud", "acceso_internet", "concept:acceso_internet", 0.55, 0.70, "approved"),
    ("pnud", "escolaridad_promedio", "concept:nivel_educativo", 0.60, 0.85, "approved"),
]

all_fields = health_fields + education_fields

for source_db, column, concept_id, qs, completeness, status in all_fields:
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

# === Guardar grafo base ===
g.bump_version("major", "Grafo combinado salud + educacion para discover cross-domain")
g.save(str(NOMENCLADOR_PATH))

print(f"Grafo base creado: {g.graph.number_of_nodes()} nodos, {g.graph.number_of_edges()} aristas")
print(f"Conceptos: {len(g.find_all_concepts())}")
print(f"Fuentes: {len([n for n,d in g.graph.nodes(data=True) if d.get('type')=='source'])}")
print(f"Fields: {len(g.list_fields())}")
print(f"Version: {g.version}")
print(f"Guardado en: {NOMENCLADOR_PATH}")

# === Generar insights acumulados por fuente (con profiling real) ===
print("\nGenerando insights por fuente (con profiling real de CSVs)...")
from src.discover import generate_insights_for_source
from src.profiler import profile_csv
from dataclasses import asdict

DATA_DIR = Path(__file__).parent.parent / "data"

# Mapear source_id -> archivo CSV
CSV_MAP = {
    "source:hospital": "hospital.csv",
    "source:censo": "censo.csv",
    "source:seguro": "seguro.csv",
    "source:siv": "siv.csv",
    "source:mined": "mined.csv",
    "source:pnud": "pnud.csv",
}

for s in sources:
    domain = "salud" if s.id in ("source:hospital", "source:seguro", "source:siv") else "educacion"
    if s.id == "source:censo":
        domain = "salud"

    # Perfilar el CSV de esta fuente
    csv_file = CSV_MAP.get(s.id)
    profile_data = None
    if csv_file:
        csv_path = DATA_DIR / csv_file
        if csv_path.exists():
            tables = profile_csv(str(csv_path))
            if tables and tables[0].columns:
                profile_data = []
                for col in tables[0].columns:
                    profile_data.append({
                        "column": col.column,
                        "data_type": col.data_type,
                        "total_count": col.total_count,
                        "null_count": col.null_count,
                        "unique_count": col.unique_count,
                        "sample_values": col.sample_values[:10],
                        "min_value": col.min_value,
                        "max_value": col.max_value,
                    })
                print(f"  Profiling {s.name}: {tables[0].row_count} filas, {len(profile_data)} columnas")

    try:
        saved = generate_insights_for_source(g, s.id, domain=domain, profile_data=profile_data)
        print(f"  {s.name} ({domain}): {len(saved)} insights generados")
    except Exception as e:
        print(f"  {s.name}: Error - {e}")

g.save(str(NOMENCLADOR_PATH))
print(f"\nGrafo final: {g.graph.number_of_nodes()} nodos, {g.graph.number_of_edges()} aristas")
print(f"Insights acumulados: {len(g.find_insights())}")
print("\nListo para: uv run python -m src.cli discover")
