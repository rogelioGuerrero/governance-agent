# Governance-Agent — Documento de Evaluacion de Metodologia

## 1. PROBLEMA QUE RESUELVE

En instituciones publicas y de salud, multiples fuentes de datos (hospitales, censos, seguros, registros administrativos) usan nombres distintos para las mismas variables, con estandares inconsistentes y sin un catalogo central que garantice interoperabilidad semantica.

**Ejemplo real del sistema:** La variable "sexo" aparece como `col_sexo` en el ministerio, `genero_paciente` en el seguro, y `sexo` en el censo. Todas implementan el mismo concepto canonico con el estandar ISO 5218, pero sin un nomenclador institucional no hay forma de garantizar que el cruce entre fuentes sea semanticamente correcto.

## 2. METODOLOGIA GENERAL

El sistema implementa un **flujo de dos rondas** inspirado en el ciclo de vida de datos institucionales:

### Ronda 1 — DESCUBRIMIENTO (automatizado, sin intervencion humana)
1. **Perfila la fuente** (CSV, SQL DDL, JSON Schema) y extrae columnas raw
2. **Limpia nombres** (normalizacion: `col_sexo` -> `sexo`, `Campo_Escolaridad` -> `escolaridad`)
3. **Detecta estandares** para cada columna usando reglas deterministas (ISO 5218, ISCED 2011, ICD-10, ISCO-08, ISO 8601)
4. **Mapea contra conceptos existentes** en el nomenclador (match por nombre limpio + valores muestra)
5. **Identifica gaps**: columnas sin mapeo = conceptos potenciales nuevos
6. **Detecta issues de calidad** (claves primarias sospechosas, valores faltantes)
7. **Clasifica automaticamente** cada variable como: `publico`, `pii`, o `sensible` (Gap A)

### Ronda 2 — COMPLETADO (IA asistida, con revision humana)
1. **Toma los gaps** de la Ronda 1
2. **Usa LLM (Groq gpt-oss-120b)** para proponer definiciones y estandares en batch (JSON mode)
3. **Fallback individual** si el batch falla (un gap a la vez, prompt mas corto)
4. **Crea conceptos nuevos** con `review_status=proposed` (Gap C: no se confian ciegamente)
5. **Registra fields** fisicos vinculados a conceptos via arista IMPLEMENTA
6. **Busca respaldo normativo** via RAG documental (Ley General de Salud, normativas OMS/OPS)
7. **Bump de version semantica** (patch si sin conceptos nuevos, minor si hay nuevos)
8. **Reporta que requiere atencion humana** (conceptos sin definicion, sin estandar, sin normativa)

### Posterior — REVISION HUMANA (human-in-the-loop, Gap C)
- El humano revisa conceptos `proposed` via CLI: `review <variable> approve|reject|start`
- Solo conceptos `approved` se consideran activos en el nomenclador
- El ciclo de vida registra cada evento (created, approved, rejected, deprecated, reactivated)

## 3. AGENTES IA QUE INTERVIENEN

### 3.1 Agente Nomenclar (`src/nomenclar.py`)
- **Rol:** Descubrimiento y completado de variables
- **Modelo:** Groq gpt-oss-120b (via `call_groq` con retry + fallback a gpt-oss-20b)
- **Cuando actua:** Ronda 2, cuando hay gaps que completar
- **Que hace:** Propone definiciones (1-2 lineas), sugiere estandares internacionales, detecta PII/sensible
- **Limitaciones:** No crea conceptos approved directamente — todo lo propuesto queda en `review_status=proposed`
- **Evaluacion de calidad:** Si el LLM no responde o falla, el concepto se crea con definicion por defecto ("Variable descubierta de {fuente}") y se marca como gap pendiente

### 3.2 Agente ReAct (`src/agent.py`)
- **Rol:** Asistente conversacional para consultas de governance
- **Modelo:** Groq gpt-oss-120b con loop ReAct (Reasoning + Acting)
- **Arquitectura:** State machine LangGraph con 7 tools:
  1. `search_graph(query)` — buscar variable en nomenclador
  2. `detect_standard(column_name, sample_values)` — detectar estandar
  3. `validate_interop(source_db, target_db)` — verificar interoperabilidad con 3 guardrails
  4. `generate_transform(source_db, target_db)` — generar SQL CASE WHEN + JSON Schema
  5. `list_concepts()` — listar conceptos canonicos
  6. `get_classifier(standard_id)` — valores validos de un estandar
  7. `ask_human(question)` — pedir aclaracion al humano
- **Cuando actua:** On-demand, cuando el usuario hace consultas complejas
- **Limitaciones:** Maximo 5 iteraciones ReAct; si no puede resolver, pide ayuda al humano

### 3.3 MoA — Mixture of Agents (`src/moa_agent.py`)
- **Rol:** Analisis multi-perspectiva para decisiones de governance
- **Modelo:** Groq gpt-oss-120b para los 3 agentes + sintetizador
- **Arquitectura:** 3 agentes especializados en paralelo + 1 sintetizador

#### Agente JURIDICO
- **Perspectiva:** Normativa, legal, proteccion de datos
- **Tools (6):** search_graph, get_normative, get_lifecycle, get_custodian, list_deprecated, list_concepts
- **Analiza:** Respaldo normativo, proteccion de datos PII, cumplimiento legal, custodio responsable

#### Agente TECNICO
- **Perspectiva:** Estandares, interoperabilidad, transformaciones
- **Tools (9):** search_graph, detect_standard, validate_interop, generate_transform, get_classifier, list_concepts, get_composites, get_contexts, find_conflicts
- **Analiza:** Estandares internacionales, compatibilidad entre fuentes, transformaciones SQL, tipos de datos

#### Agente ESTADISTICO
- **Perspectiva:** Calidad de datos, poblacion, sesgos
- **Tools (6):** search_graph, list_concepts, get_contexts, find_conflicts, get_lifecycle, version_info
- **Analiza:** Poblacion objetivo, metodologia de captura, sesgos, conflictos de contexto

#### SINTETIZADOR
- **Rol:** Combina las 3 perspectivas en una respuesta unificada
- **Guardrail de Arbitraje (Gap M):**
  - **JURIDICO tiene prioridad ABSOLUTA** en temas de proteccion de datos, PII y cumplimiento legal
  - Si JURIDICO veta una variable, la recomendacion final DEBE ser NO usarla, independientemente de viabilidad tecnica
  - Si JURIDICO no encuentra objecion, TECNICO prevalece en estandares e interoperabilidad
  - ESTADISTICO tiene voz consultiva: sus observaciones se incluyen como advertencias pero no pueden vetar
  - En conflicto no resuelto, recomendar NO proceder hasta que un custodio humano decida
- **Deteccion automatica de conflictos:** `_detect_juridico_tecnico_conflict()` analiza los outputs de ambos agentes para identificar vetos juridicos vs aprobaciones tecnicas

## 4. KNOWLEDGE GRAPH — ESTRUCTURA

### Nodos (6 tipos + 2 de governance)
| Tipo | Funcion | Ejemplo |
|------|---------|---------|
| **Concept** | Variable canonica | `concept:sexo` (ISO 5218) |
| **Field** | Implementacion fisica | `field:ministerio.col_sexo` |
| **Classifier** | Valores validos | `classifier:iso_5218` |
| **Operation** | Transformacion compuesta | `operation:concat_nombre` |
| **Context** | Significado contextual | `context:sexo:ministerio` |
| **Source** | Fuente de datos | `source:ministerio` |
| **Normative** | Documento normativo (RAG) | `normative:ley_general_salud_art7` |
| **AnonymizationRule** | Regla de anonimizacion (Gap A) | `anon:hash_pii` |

### Aristas (11 tipos)
| Arista | Origen -> Destino | Significado |
|--------|-------------------|-------------|
| IMPLEMENTA | Field -> Concept | Campo fisico implementa concepto canonico |
| USA_CLASIFICADOR | Concept -> Classifier | Concepto usa valores validos de un estandar |
| TRANSFORMA_A | Field -> Operation -> Field | Transformacion entre campos |
| PERTENECE_A | Field -> Context | Campo pertenece a un contexto |
| PROVIENE_DE | Field -> Source | Campo proviene de una fuente |
| COMPONE | Concept -> Concept | Concepto compone otro (nombre_completo) |
| DERIVA_DE | Concept -> Concept | Concepto deriva de otro (año de fecha) |
| RESPALDADO_POR | Concept -> Normative | Concepto respaldado por documento normativo |
| APLICA_ANONIMIZACION | Concept/Field -> AnonymizationRule | Regla de anonimizacion aplicable (Gap A) |
| EQUIVALE_A | Classifier -> Classifier | Clasificadores equivalentes (Gap D) |
| SUBCONCEPTO_DE | Classifier -> Classifier | Jerarquia de clasificadores (Gap D) |

### Persistencia Dual (Gap B)
- **Capa 1 — NetworkX (in-memory):** Cache para consultas rapidas O(1)
- **Capa 2 — PostgreSQL (Supabase):** Persistencia ACID con write-through en cada operacion
- **Tablas:** `governance.graph_nodes(id, type, data JSONB)`, `governance.graph_edges(source_id, target_id, type, data JSONB)`, `governance.nomenclador_version(version, total_nodes, total_edges, changed_at)`
- **Fallback:** Si no hay conexion PostgreSQL, usa JSON local como respaldo
- **Sync inicial:** Si la BD esta vacia pero hay JSON local, sincroniza todo el JSON a PostgreSQL automaticamente
- **RLS habilitado:** Solo `service_role` puede acceder al schema `governance`

## 5. GAPS DE GOBERNANZA CUBIERTOS

### Gap A — Anonimato y Privacidad
- **Clasificacion automatica:** `publico` | `interno` | `pii` | `sensible` via deteccion por palabras clave
- **SQL de anonimizacion automatico:**
  - PII texto: Hash SHA-256 con salt configurable (`ANON_SALT` env var) → seudonimizacion 16 chars
  - PII fecha: Generalizacion a año (`EXTRACT(YEAR FROM fecha_nacimiento)`)
  - Sensible: Generalizacion categorica (`CASE WHEN x IS NOT NULL THEN 'registrado' ELSE NULL END`)
- **CLI:** `classify <variable> pii|sensible|publico|interno` + `sensitive` (lista todos)

### Gap B — Graph DB Persistence
- Dual-write NetworkX + PostgreSQL (detallado arriba)
- Demo verificado: 48 nodos, 51 aristas, version 1.3.3 persistida en Supabase

### Gap C — Human-in-the-Loop
- **Estados:** `proposed` → `under_review` → `approved` | `rejected`
- **Proposed_by:** Todo concepto creado por IA marca `proposed_by="agent:nomenclar"`
- **Lifecycle log:** Cada cambio se registra con timestamp, actor, razon
- **CLI:** `review <variable> approve|reject|start`
- **Principio:** La IA propone, el humano dispone. Ningun concepto creado por IA entra al nomenclador como approved automaticamente.

### Gap D — Clasificadores Dinamicos
- **Version_label + parent_id + is_current** en ClassifierNode
- **EQUIVALE_A:** Clasificadores equivalentes entre versiones (ICD-10 v2019 ↔ ICD-10 v2024)
- **SUBCONCEPTO_DE:** Jerarquia dentro de un clasificador (capitulo → codigo especifico)
- **Metodos:** `link_classifier_equivalent`, `link_classifier_subconcept`, `find_classifier_equivalents`, `find_classifier_hierarchy`

### Gap M — MoA Arbitraje
- Guardrail de arbitraje juridico vs tecnico (detallado en seccion 3.3)
- Deteccion automatica de conflictos via keywords en los outputs de los agentes
- Prioridad absoluta al juridico en temas legales; voz consultiva al estadistico

## 6. GUARDRAILS DE INTEROPERABILIDAD

Antes de generar transformaciones entre dos fuentes, el sistema valida 3 checkpoints:

1. **Checkpoint de Poblacion:** ¿Ambas fuentes cubren la misma poblacion objetivo?
   - Ejemplo: ministerio (poblacion general) vs seguro (afiliados) → WARNING de asimetria

2. **Checkpoint de Metodologia:** ¿El dato se captura de la misma forma?
   - Ejemplo: censo (auto-reporte) vs hospital (observacion clinica) → WARNING de asimetria

3. **Checkpoint de Clasificador:** ¿Los valores validos coinciden?
   - Ejemplo: sexo M/F (ISO 5218) coincide entre fuentes → OK

Los warnings no bloquean la transformacion, pero se reportan explicitamente para que el humano decida.

## 7. RAG DOCUMENTAL

- **Vector store local** con embeddings de Cohere (`embed-multilingual-v3.0`, 1024 dimensiones)
- **Documentos normativos:** Ley General de Salud, normativas OMS/OPS
- **Pipeline de ingesta (7 fases):** Extract → Clean → Detect Issues → Match → LLM Enrich → Register → Report
- **Busqueda semantica:** Cosine similarity sobre embeddings
- **Integracion:** Conceptos pueden tener arista RESPALDADO_POR hacia documentos normativos
- **CLI:** `normative <file>` (ingestar) + `normative-search <query>` (buscar)

## 8. MCP SERVER

El sistema expone 7 tools via FastMCP (Model Context Protocol) para integracion con otros agentes:
1. `search_variable(query)` — buscar en nomenclador
2. `list_all_concepts()` — listar conceptos
3. `detect_standard_for_column(column_name, sample_values)` — detectar estandar
4. `validate_interoperability(source_db, target_db)` — validar interoperabilidad
5. `generate_transformation_artifacts(source_db, target_db)` — generar SQL + Schema
6. `get_classifier_values(standard_id)` — valores validos
7. `get_graph_stats()` — estadisticas del grafo

## 9. DEMO VERIFICADA (Jul 2026)

```
$ uv run python -m src.cli nomenclar tests/sample_ministerio_sucio.csv --auto

=== RONDA 1: DESCUBRIMIENTO ===
Fuente: sample_ministerio_sucio
Columnas: 6 | Mapeadas: 5 | Sin mapear: 1

Limpieza (3):
  'col_sexo' -> 'sexo'
  'Campo_Escolaridad' -> 'escolaridad'
  'col_ocupacion' -> 'ocupacion'

Mapeos existentes (5):
  OK sexo -> sexo (ISO_5218)
  OK escolaridad -> nivel_educativo (ISCED_2011)
  OK cie10_diag -> diagnostico (ICD_10)
  OK fecha_ingreso -> fecha_ingreso (-)
  OK ocupacion -> ocupacion (ISCO_08)

Gaps - conceptos nuevos/potenciales (1):
  ?? fecha_nacimiento (tipo: date, std: ISO_8601)

=== RONDA 2: COMPLETADO ===
Version: 1.3.2 -> 1.3.3
Aun requiere atencion humana (1):
  ?? fecha_nacimiento: falta normativa
```

**PostgreSQL verificado:**
- 48 nodos persistidos (30 fields, 7 concepts, 4 sources, 3 classifiers, 2 contexts, 2 normatives)
- 51 aristas persistidas
- `review fecha_nacimiento approve` → `review_status=approved` en BD
- `classify fecha_nacimiento pii` → `data_classification=pii` en BD
- Transform SQL generado con anonimizacion automatica (generalizacion a año por ser PII tipo date)

## 10. STACK TECNOLOGICO

| Componente | Tecnologia | Version |
|------------|-----------|---------|
| LLM | Groq gpt-oss-120b | via API |
| LLM fallback | Groq gpt-oss-20b | via API |
| Framework agentes | LangGraph | 1.2.8 |
| Knowledge graph | NetworkX | 3.6.1 |
| Persistencia | PostgreSQL (Supabase) | via psycopg 3.3.4 |
| Embeddings RAG | Cohere embed-multilingual-v3.0 | 1024d |
| CLI | Rich | 15.0.0 |
| MCP | FastMCP | 1.28.1 |
| Python | 3.12+ | uv managed |

## 11. PRINCIPIOS DE DISENO

1. **La IA propone, el humano dispone** — Ningun concepto creado por IA entra como approved
2. **Write-through persistente** — Cada operacion en el grafo se persiste inmediatamente en PostgreSQL
3. **Guardrails antes de transformaciones** — 3 checkpoints validan antes de generar SQL
4. **Arbitraje juridico prioritario** — En conflicto legal vs tecnico, lo legal prevalece
5. **Fallback graceful** — Si no hay BD, usa JSON local; si LLM falla, usa definicion por defecto
6. **Trazabilidad total** — Lifecycle log registra cada evento con actor, timestamp y razon
7. **Versionado semantico** — Cada cambio bumpa version (patch/minor/major) con historial
8. **Aislamiento de datos** — Schema `governance` separado de otras apps en mismo Supabase

---

**Pregunta para el evaluador:** ¿Es esta metodologia correcta para un sistema de governance de datos institucionales? ¿Los agentes IA estan bien delimitados en sus roles? ¿Hay gaps o riesgos no cubiertos?
