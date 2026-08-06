"""
Esquema del Knowledge Graph del Nomenclador.

3 tipos de nodos principales:
- Concept: Variable canónica (el "deber ser")
- Field: Implementación física en una DB real
- Classifier: Catálogo de valores válidos

Nodos secundarios:
- Operation: Transformación entre campos
- Context: Proceso de negocio
- Source: Base de datos / instrumento
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class NodeType(str, Enum):
    CONCEPT = "concept"
    FIELD = "field"
    CLASSIFIER = "classifier"
    OPERATION = "operation"
    CONTEXT = "context"
    SOURCE = "source"
    NORMATIVE = "normative"
    ANONYMIZATION = "anonymization"
    QUALITY_ISSUE = "quality_issue"
    INSIGHT = "insight"


class EdgeType(str, Enum):
    IMPLEMENTA = "implementa"           # Field -> Concept
    USA_CLASIFICADOR = "usa_clasificador"  # Concept -> Classifier
    TRANSFORMA_A = "transforma_a"       # Field -> Field (via Operation)
    PERTENECE_A = "pertenece_a"         # Field -> Context
    PROVIENE_DE = "proviene_de"         # Field -> Source
    COMPONE = "compone"                 # Concept -> Concept
    DERIVA_DE = "deriva_de"             # Concept -> Concept
    RESPALDADO_POR = "respaldado_por"   # Concept -> NormativeDocument
    APLICA_ANONIMIZACION = "aplica_anonimizacion"  # Concept/Field -> AnonymizationRule
    EQUIVALE_A = "equivalente_a"        # Classifier -> Classifier (mapeo entre versiones) o Field -> Field (equivalencia descubierta)
    SUBCONCEPTO_DE = "subconcepto_de"   # Classifier -> Classifier (jerarquia)
    TIENE_ISSUE = "tiene_issue"         # Field -> QualityIssue
    TIENE_CONTEXTO = "tiene_contexto"   # Concept -> Context
    GENERATES_INSIGHT = "generates_insight"  # Source -> Insight


class DataClassification(str, Enum):
    """Nivel de sensibilidad del dato (Gap A)."""
    PUBLICO = "publico"
    INTERNO = "interno"
    PII = "pii"              # Personally Identifiable Information
    SENSIBLE = "sensible"    # Dato sensible (salud, genetica, etc)


class ReviewStatus(str, Enum):
    """Estado de revision para nodos propuestos por IA (Gap C)."""
    PROPOSED = "proposed"          # Propuesto por agente IA
    UNDER_REVIEW = "under_review"  # En revision por custodio
    APPROVED = "approved"          # Aprobado por humano
    REJECTED = "rejected"          # Rechazado


class ConceptNode(BaseModel):
    """Variable canónica: el 'deber ser'."""
    id: str
    type: str = NodeType.CONCEPT.value
    name: str
    definition: str = ""
    context_production: str = ""
    granularity: str = ""
    standard: Optional[str] = None
    version: str = "1.0"
    why: str = ""       # Por qué existe
    what_for: str = ""  # Para qué sirve
    normative: str = "" # Resolución / acuerdo que la respalda
    # Guardrails: definición canónica
    population: str = ""        # Población objetivo (ej: "todos los pacientes")
    capture_method: str = ""   # Metodología de captura (ej: "auto-reporte", "observación clínica")
    # Gobernanza institucional: custodio de la variable
    custodian: str = ""             # Persona/rol responsable (ej: "Dr. Perez, Jefe Epidemiologia")
    custodian_department: str = ""  # Dirección/Departamento que usa la variable (ej: "Direccion de Vigilancia")
    custodian_contact: str = ""     # Contacto (email/teléfono) - opcional
    # Lifecycle: ciclo de vida de la variable
    status: str = "activo"          # activo | deprecado | retirado
    explanatory_note: str = ""      # Nota explicativa acumulada (lenguaje humano)
    # Gap A: Clasificacion de datos y privacidad
    data_classification: str = "publico"  # publico | interno | pii | sensible
    # Gap C: Workflow de aprobacion humana
    review_status: str = "approved"   # proposed | under_review | approved | rejected
    proposed_by: str = ""             # quien propuso (agent, agent:nomenclar, human)
    # Staleness tracking: cuando se verifico por ultima vez contra la fuente
    last_verified: str = ""           # ISO date (ej: "2026-07-09")


class FieldNode(BaseModel):
    """Implementación física en una DB real."""
    id: str
    type: str = NodeType.FIELD.value
    source_db: str = ""
    table: str = ""
    column: str = ""
    data_type: str = ""
    nullable: bool = True
    unique_count: int = 0
    null_count: int = 0
    total_count: int = 0
    sample_values: list[str] = Field(default_factory=list)
    inferred_standard: Optional[str] = None
    confidence: str = "low"  # low, medium, high
    # Guardrails: contexto de captura
    population: str = ""        # Población objetivo real de esta fuente
    capture_method: str = ""   # Cómo se capturó el dato
    context_label: str = ""    # Contexto de negocio (ej: "censo", "hospital", "seguro")
    # Gap A: Clasificacion de datos a nivel de campo fisico
    data_classification: str = "publico"  # publico | interno | pii | sensible
    # Gap C: Workflow de aprobacion
    review_status: str = "approved"   # proposed | under_review | approved | rejected
    # PMBOK Quality Management: métricas de calidad estructuradas
    completeness: float = 0.0      # % no nulos (0.0-1.0)
    uniqueness: float = 0.0        # ratio unique/total (0.0-1.0)
    consistency: float = 0.0       # % valores que matchean el estándar (0.0-1.0)
    validity: float = 0.0          # % valores que pasan regex/formato (0.0-1.0)
    quality_score: float = 0.0     # promedio ponderado de lo anterior (0.0-1.0)
    # Staleness tracking: cuando se verifico por ultima vez la calidad/estructura de este campo
    last_verified: str = ""        # ISO date (ej: "2026-07-09")


class ClassifierNode(BaseModel):
    """Catálogo de valores válidos."""
    id: str
    type: str = NodeType.CLASSIFIER.value
    name: str
    standard: Optional[str] = None
    values: dict[str, str] = Field(default_factory=dict)  # code -> label
    version: str = "1.0"
    # Gap D: Clasificadores jerarquicos y dinamicos
    version_label: str = ""        # ej: "ICD-10 2019", "ICD-10 2024"
    parent_id: Optional[str] = None  # referencia a clasificador padre (jerarquia)
    is_current: bool = True        # es la version vigente?
    # Gap C: Workflow de aprobacion
    review_status: str = "approved"   # proposed | under_review | approved | rejected


class OperationNode(BaseModel):
    """Transformación entre campos."""
    id: str
    type: str = NodeType.OPERATION.value
    name: str
    sql: str = ""
    description: str = ""


class ContextNode(BaseModel):
    """Proceso de negocio."""
    id: str
    type: str = NodeType.CONTEXT.value
    name: str
    description: str = ""


class SourceNode(BaseModel):
    """Base de datos o instrumento."""
    id: str
    type: str = NodeType.SOURCE.value
    name: str
    description: str = ""
    connection: str = ""  # connection string o path
    # Staleness tracking: cuando se verifico por ultima vez la estructura de esta fuente
    last_verified: str = ""  # ISO date (ej: "2026-07-09")
    # Gap C: Workflow de aprobacion
    review_status: str = "approved"   # proposed | under_review | approved | rejected


class NormativeNode(BaseModel):
    """Documento normativo que respalda un concepto canonico."""
    id: str
    type: str = NodeType.NORMATIVE.value
    title: str
    source: str = ""          # ej: "ley_general_salud"
    article: str = ""          # ej: "Art. 47"
    citation: str = ""         # cita breve para mostrar
    similarity_score: float = 0.0
    chunk_id: str = ""        # referencia al chunk en normative_corpus.json


class AnonymizationRuleNode(BaseModel):
    """Regla de anonimizacion aplicable a datos sensibles (Gap A).

    Tipos de anonimizacion:
    - enmascaramiento: reemplaza parcialmente (ej: 1234****5678)
    - hash: hash irreversible (SHA-256)
    - k_anonimato: agrega ruido para asegurar k-indistinguibilidad
    - generalizacion: reduce granularidad (ej: fecha exacta -> año)
    - supresion: elimina el dato completamente
    - seudonimizacion: reemplaza ID real por seudonimo reversible
    """
    id: str
    type: str = NodeType.ANONYMIZATION.value
    name: str
    technique: str = ""        # enmascaramiento | hash | k_anonimato | generalizacion | supresion | seudonimizacion
    sql_expression: str = ""   # ej: "SUBSTRING(sha256(column), 1, 16)"
    description: str = ""
    required_for: list[str] = Field(default_factory=list)  # que data_classification la requieren


class IssueSeverity(str, Enum):
    """Severidad de un issue de calidad de datos (PMBOK Quality Management)."""
    INFO = "info"        # informativo, no afecta calidad
    WARNING = "warning"  # calidad degradada pero utilizable
    ERROR = "error"      # calidad comprometida, requiere intervencion


class QualityIssueNode(BaseModel):
    """Issue de calidad de datos persistido en el grafo (PMBOK Quality Management).

    Permite que el agente consulte issues estructurados en vez de parsear texto,
    y que el MoA Estadistico cuantifique calidad de manera objetiva.
    """
    id: str
    type: str = NodeType.QUALITY_ISSUE.value
    issue_type: str = ""       # alta_nulidad | columna_constante | clave_primaria | encoding_roto | tipo_no_detectado | fuera_de_estandar
    severity: str = IssueSeverity.WARNING.value  # info | warning | error
    detail: str = ""           # descripcion humana del issue
    metric_value: float = 0.0  # valor cuantitativo (ej: 0.78 para 78% nulos)
    detected_at: str = ""     # timestamp de deteccion
    detected_by: str = ""     # rag_factory | profiler | agent | manual


class InsightNode(BaseModel):
    """Insight acumulado de una fuente de datos.

    Observaciones analiticas sobre los datos de una fuente que se guardan
    en el grafo para combinarse con insights de otras fuentes en discover.
    Se eliminan en cascada cuando la fuente se elimina.
    """
    id: str
    type: str = NodeType.INSIGHT.value
    source_id: str = ""           # source:hospital, source:unicef, etc.
    domain: str = ""              # salud, educacion, agricultura
    observation: str = ""         # descripcion del hallazgo
    variables_covered: list[str] = Field(default_factory=list)  # nombres de conceptos cubiertos
    quality_snapshot: dict = Field(default_factory=dict)  # {avg_qs, field_count, low_quality_count}
    cross_source_potential: str = ""  # que variables permiten correlacionar con otras fuentes
    created_at: str = ""          # ISO timestamp
