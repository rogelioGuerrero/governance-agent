# Governance Agent — Nomenclador Institucional

Agente de governance para interoperabilidad semantica entre sistemas de informacion.
Construye un nomenclador institucional usando un knowledge graph (NetworkX + PostgreSQL),
con agentes ReAct y MoA (Mixture of Agents) para razonamiento semantico, legal y estadistico,
y un motor de inferencia semantica que resuelve ~33% de columnas sin usar LLM.

## Arquitectura

```
Usuario consulta
    |
    +-- Agente ReAct (agent.py) — loop think/act/observe con LangGraph
    |       Tools: search_graph, detect_standard, validate_interop,
    |              generate_transform, list_concepts, get_classifier
    |
    +-- MoA (moa_agent.py) — 3 agentes especializados en paralelo
    |       +-- JURIDICO — normativo, legal, respaldo normativo
    |       +-- TECNICO — estandares, interoperabilidad, transformaciones
    |       +-- ESTADISTICO — calidad de datos, sesgos, metodologia
    |       +-- SINTETIZADOR — combina las 3 perspectivas (guardrail: juridico prioritario)
    |
    +-- Inference Engine (inference.py) — resolucion semantica sin LLM
    |       1. Patrones regex/heuristicas (fechas, años, DUI, NIT, email, booleanos, edad, %)
    |       2. Listas de referencia (departamentos_sv, meses_es, genero, estado_civil, etc.)
    |       3. Huella de valores (overlap coefficient contra conceptos existentes)
    |
    +-- RAG Factory (rag_factory.py) — ingesta masiva de diccionarios de datos
    |       Pipeline: Extract -> Profile -> Clean -> Match -> Inference -> Propose -> Ingest
    |
    +-- Nomenclador (nomenclar.py) — flujo de 2 rondas
    |       Ronda 1: descubrir variables, detectar gaps
    |       Ronda 2: completar gaps (inference high -> LLM -> normative RAG)
    |
    +-- RAG Documental (normative_rag.py) — respaldo normativo con Cohere embeddings
    |
    +-- MCP Server (mcp_server.py) — expone el nomenclador a IDEs (Cursor, Windsurf, VS Code)
    |
    +-- CLI (cli.py) — interfaz interactiva con rich console
```

## Knowledge Graph

Nodos: `Concept`, `Field`, `Classifier`, `Context`, `Source`, `Normative`, `AnonymizationRule`
Edges: `IMPLEMENTA`, `USA_CLASIFICADOR`, `RESPALDADO_POR`, `EQUIVALE_A`, `COMPUESTO_DE`, `TIENE_CONTEXTO`

**Persistencia dual-write:** PostgreSQL (Supabase, schema `governance`) + JSON local (fallback).

### Schema PostgreSQL

| Tabla | Proposito |
|---|---|
| `governance.graph_nodes` | Nodos del knowledge graph (id, type, data JSONB) |
| `governance.graph_edges` | Edges del grafo (source_id, target_id, type, data JSONB) |
| `governance.nomenclador_version` | Versionado del nomenclador |
| `governance.lifecycle_log` | Decision log del ciclo de vida de variables (dual-write) |

## Lifecycle y Decision Log

Cada variable canonica es "viva": nace, cambia, se deprecia, se retira.
Cada transicion se registra con **quien, que, por que y cuando**.

- **Estados:** `activo`, `deprecado`, `retirado`
- **Review workflow (Gap C):** `proposed` -> `under_review` -> `approved`/`rejected`
- **Dual-write:** PostgreSQL `governance.lifecycle_log` + JSON local `nomenclador/decision_log.json`

## Inference Engine

Motor de inferencia semantica — capa intermedia entre deteccion de estandares y LLM.
Resuelve columnas sin usar LLM, reduciendo costo y carga humana.

### 3 mecanismos en orden de confianza

1. **Patrones de tipo semantico** (regex/heuristicas) -> high confidence
   - Fechas (ISO 8601, latino), años (1900-2099)
   - DUI, NIT (El Salvador), email, booleanos, edad, porcentaje

2. **Listas de referencia** (soft standards) -> high/medium confidence
   - `departamentos_sv`, `meses_es`, `dias_semana_es`, `genero_binario`, `estado_civil`, `nivel_educativo`, `tipo_sangre`
   - Archivos CSV en `src/reference_lists/`

3. **Huella de valores** (overlap coefficient) -> high/medium confidence
   - Compara valores normalizados contra conceptos existentes en el grafo

### Flujo sin friccion

- **HIGH confidence** -> auto-aprobar (igual que estandares ISO)
- **MEDIUM confidence** -> marcar como proposed, seguir, NO detener pipeline
- **LOW confidence** -> ir al LLM si `--llm`, sino tambien proposed
- Humano revisa en lote post-hoc: `batch-approve --confidence medium`

## Stack

- **Python 3.12+** con `uv`
- **NetworkX** — knowledge graph en memoria
- **PostgreSQL** (Supabase) — persistencia dual-write
- **LangGraph** — state machine del agente ReAct
- **Groq** — `gpt-oss-120b` (primario) + `gpt-oss-20b` (fallback)
- **Cohere** — `embed-multilingual-v3.0` para RAG normativo
- **MCP** — servidor para integracion con IDEs
- **Rich** — CLI con consola enriquecida
- **Sin pandas/numpy** — usa libreria estandar `csv` (DLL load failed en Windows)

## Setup

```bash
# Instalar dependencias
uv sync

# Configurar entorno
cp .env.example .env
# Editar .env con tus API keys:
#   GROQ_API_KEY=gsk_...
#   GROQ_MODEL_PRIMARY=openai/gpt-oss-120b
#   GROQ_MODEL_FALLBACK=openai/gpt-oss-20b
#   COHERE_API_KEY=cohere_...
#   DATABASE_URL=postgresql://...  (opcional, sin esto usa JSON local)
#   ANON_SALT=...  (para seudonimizacion)
```

## Uso

### CLI interactivo

```bash
uv run python -m src.cli
```

### Comandos principales

#### Perfilar un CSV

```bash
python -m src.cli profile demo/mag_produccion_agricola.csv --auto

> **Nota:** Sin --auto, el comando entra en modo interactivo (Confirm.ask) y se queda esperando input. Usar --auto para pipelines automatizados.
```

Muestra:
- Tabla con tipo CSV, tipo inferido, nulos, unicos, estandar detectado
- Seccion de alertas (null ratio alto, columna constante, encoding, posible PK)
- Detalle por columna con razon de inferencia

#### Nomenclador — flujo de 2 rondas

```bash
# Interactivo (pide confirmacion humana)
python -m src.cli nomenclar demo/mag_produccion_agricola.csv

# Auto (sin confirmacion humana, conceptos quedan en proposed)
python -m src.cli nomenclar demo/mag_produccion_agricola.csv --auto
```

Ronda 1: descubre variables, detecta estandares, identifica gaps.
Ronda 2: completa gaps usando inference engine (high) -> LLM -> normative RAG.

Al final muestra **resumen de cobertura**:
```
Resumen de cobertura:
  Estandares ISO detectados:    0  (0%)
  Inferencia semantica high:    2  (33%)
  Inferencia semantica med:     0  (0%)
  LLM (Ronda 2):                4  (67%)
  Sin mapear:                   4  (67%)
  Auto-resueltos (sin LLM): 33%
```

#### Ingesta masiva

```bash
python -m src.cli ingest demo/mag_produccion_agricola.csv --auto
```

Pipeline completo de RAG Factory con resumen de cobertura por metodo.

#### Batch approve con filtro de confianza

```bash
# Aprobar solo conceptos de alta confianza
python -m src.cli batch-approve --confidence high

# Aprobar conceptos de confianza media o alta
python -m src.cli batch-approve --confidence medium
```

Muestra tabla con concepto, confianza (coloreada), razon de inferencia y estandar.

#### Otros comandos

| Comando | Descripcion |
|---|---|
| `catalog` | Listar conceptos del nomenclador |
| `search <variable>` | Buscar variable en el nomenclador |
| `interop <var1> <var2>` | Verificar interoperabilidad entre variables |
| `transform <var1> <var2>` | Generar SQL + JSON Schema de transformacion |
| `review` | Revisar conceptos propuestos (human-in-the-loop) |
| `classify <concept>` | Asignar clasificador a concepto |
| `sensitive <concept>` | Marcar concepto como sensible (PII) |
| `agent <pregunta>` | Consultar agente ReAct |
| `moa <pregunta>` | Consultar MoA (3 agentes + sintetizador) |
| `version` | Version actual del nomenclador |
| `history <concept>` | Historial de cambios de una variable |
| `register-standard` | Registrar nuevo estandar |
| `list-standards` | Listar estandares registrados |
| `demo-agri-env` | Demo completo con datos agricolas y ambientales |

### Agente ReAct

```python
from src.agent import run_agent

result = run_agent("¿La variable sexo es interoperable entre SISA y RAAG?")
print(result["final_answer"])
```

### MoA (Mixture of Agents)

```python
from src.moa_agent import run_moa

result = run_moa("¿Puedo usar la variable etnia del censo para cross-tab con registro civil?")
print(result["final_answer"])
# result["juridico"], result["tecnico"], result["estadistico"] tambien disponibles
```

### RAG Factory — ingesta masiva

```python
from src.rag_factory import create_ingestion_plan, execute_ingestion_plan

# 1. Crear plan (el humano revisa)
plan = create_ingestion_plan("diccionario_sisa.csv", source_type="csv", use_llm=True)

# 2. Ejecutar tras aprobacion
resultado = execute_ingestion_plan(plan)
```

### MCP Server

```bash
# Ejecutar servidor MCP
uv run nomenclador-mcp

# O desde un IDE (ej. Cursor), agregar al settings:
# {
#   "mcpServers": {
#     "nomenclador": { "command": "uv", "args": ["run", "nomenclador-mcp"] }
#   }
# }
```

Tools expuestas: `list_concepts`, `search_variable`, `get_concept`,
`check_interoperability`, `get_transform`, `validate_field`, `get_classifier`.

## Estructura del proyecto

```
src/
+-- inference.py        # Motor de inferencia semantica (patrones, listas, huella)
+-- agent.py            # Agente ReAct con LangGraph
+-- moa_agent.py        # Mixture of Agents (3 agentes paralelos + sintetizador)
+-- cli.py              # CLI interactivo (23 comandos)
+-- groq_client.py      # Cliente Groq con retry y fallback
+-- log_config.py       # Configuracion centralizada de logging
+-- mcp_server.py       # Servidor MCP para IDEs
+-- rag_factory.py      # Pipeline de ingesta masiva (7 fases + inference engine)
+-- nomenclar.py        # Flujo 2 rondas: descubrimiento + completado
+-- normative_rag.py    # RAG documental con Cohere embeddings
+-- profiler.py         # Profiling de CSV
+-- standards.py        # Estandares pre-registrados (ISO 3166, ISO 5218, etc.)
+-- guardrails.py       # Guardrails de interoperabilidad semantica
+-- transformer.py      # Generador de transformaciones SQL + JSON Schema
+-- verifier.py         # Verificacion determinista de interoperabilidad
+-- lifecycle.py        # Lifecycle + decision log (dual-write PG + JSON)
+-- graph/
|   +-- catalog.py      # NomencladorGraph (NetworkX + PostgreSQL dual-write)
|   +-- schema.py       # Definicion de nodos y edges (Pydantic)
+-- reference_lists/    # 7 CSVs con listas de referencia (soft standards)
|   +-- departamentos_sv.csv
|   +-- meses_es.csv
|   +-- dias_semana_es.csv
|   +-- genero_binario.csv
|   +-- estado_civil.csv
|   +-- nivel_educativo.csv
|   +-- tipo_sangre.csv
+-- nomenclador/        # Almacenamiento local (JSON)
    +-- nomenclador.json
    +-- decision_log.json
    +-- normative_corpus.json
demo/                   # Datos de demostracion
    +-- mag_produccion_agricola.csv
    +-- marn_cobertura_forestal.csv
tests/                  # CSVs de prueba
    +-- sample_censo.csv
    +-- sample_hospital.csv
    +-- sample_ministerio_sucio.csv
    +-- sample_seguro.csv
```

## Governance Gaps abordados

- **Gap A — Anonimizacion:** SQL de seudonimizacion automatica para PII/sensible
- **Gap B — Graph DB:** Dual-write NetworkX <-> PostgreSQL
- **Gap C — Human-in-the-Loop:** Review workflow para conceptos propuestos por IA
- **Gap D — Clasificadores dinamicos:** Importacion y deteccion automatica de estandares
- **Gap M — MoA Arbitraje:** Guardrail que da prioridad absoluta al agente juridico en conflictos legales

## Datos de demostracion

```bash
# Perfilar
python -m src.cli profile demo/mag_produccion_agricola.csv --auto

> **Nota:** Sin --auto, el comando entra en modo interactivo (Confirm.ask) y se queda esperando input. Usar --auto para pipelines automatizados.

# Nomenclador completo (2 rondas)
python -m src.cli nomenclar demo/mag_produccion_agricola.csv --auto
python -m src.cli nomenclar demo/marn_cobertura_forestal.csv --auto

# Ver catalogo resultante
python -m src.cli catalog

# Demo integrado (agricola + ambiental)
python -m src.cli demo-agri-env
```
