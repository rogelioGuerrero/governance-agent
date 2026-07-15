# Governance Agent — Reporte para Evaluador (Ronda 2)

## Estado del Proyecto: 17 archivos Python, ~5,800 líneas

### Stack Técnico
- **Python + uv** | NetworkX | Rich | Groq gpt-oss-120b | MCP (FastMCP) | LangGraph
- **Sin pandas/numpy** (DLL load failed en Windows — librería estándar csv)
- **Cohere** embed-multilingual-v3.0 (1024d) para RAG documental
- **.env** con GROQ_API_KEY, GROQ_MODEL_PRIMARY, GROQ_MODEL_FALLBACK, COHERE_API_KEY

---

## Arquitectura: Knowledge Graph (NetworkX)

### Nodos (8 tipos)
| Nodo | Descripción | Campos nuevos (Gaps) |
|------|-------------|---------------------|
| **Concept** | Variable canónica (el "deber ser") | `data_classification`, `review_status`, `proposed_by` |
| **Field** | Implementación física en DB real | `data_classification`, `review_status` |
| **Classifier** | Catálogo de valores válidos | `version_label`, `parent_id`, `is_current` |
| **Operation** | Transformación entre campos | — |
| **Context** | Proceso de negocio | — |
| **Source** | Base de datos / instrumento | — |
| **Normative** | Documento normativo | — |
| **AnonymizationRule** 🆕 | Regla de anonimización (Gap A) | `technique`, `sql_expression`, `required_for` |

### Aristas (11 tipos)
| Arista | Origen → Destino | Gap |
|--------|-----------------|-----|
| IMPLEMENTA | Field → Concept | — |
| USA_CLASIFICADOR | Concept → Classifier | — |
| TRANSFORMA_A | Field → Field | — |
| PERTENECE_A | Field → Context | — |
| PROVIENE_DE | Field → Source | — |
| COMPONE | Concept → Concept | — |
| DERIVA_DE | Concept → Concept | — |
| RESPALDADO_POR | Concept → Normative | — |
| tiene_contexto | Concept → Context | — |
| **APLICA_ANONIMIZACION** 🆕 | Concept/Field → AnonymizationRule | Gap A |
| **EQUIVALE_A** 🆕 | Classifier → Classifier | Gap D |
| **SUBCONCEPTO_DE** 🆕 | Classifier → Classifier | Gap D |

### Grafo Actual
- **48 nodos** | **51 aristas** | **Versión 1.3.0** | **5 versiones en historial**
- Nodos por tipo: 30 fields, 7 concepts, 4 sources, 3 classifiers, 2 normatives, 2 contexts

---

## 8 Recomendaciones de Diseño — Estado

| # | Recomendación | Archivo | Estado |
|---|--------------|---------|--------|
| 1 | MCP Server | `src/mcp_server.py` (298 líneas) | ✅ FastMCP, 7 tools via stdio |
| 2 | Guardrails | `src/guardrails.py` (207 líneas) | ✅ 3 checkpoints (Población, Metodología, Clasificador) |
| 3 | Transformaciones | `src/transformer.py` (369 líneas) | ✅ SQL CASE WHEN + JSON Schema + **anonimización automática** (Gap A) |
| 4 | RAG Factory | `src/rag_factory.py` (657 líneas) | ✅ Pipeline 7 fases + curación sintáctica |
| 5 | LangGraph ReAct | `src/agent.py` (432 líneas) | ✅ Loop ReAct con Groq + 7 tools |
| 6 | RAG Documental | `src/normative_rag.py` (337 líneas) | ✅ Vector store local + Cohere + cosine similarity |
| 7 | Lifecycle + Decision Log | `src/lifecycle.py` (234 líneas) | ✅ Ciclo de vida + auditoría + **review workflow** (Gap C) |
| 8 | MoA Multi-Agente | `src/moa_agent.py` (472 líneas) | ✅ 3 especialistas + sintetizador + **guardrail arbitraje** (Gap M) |

---

## Gaps Críticos — Implementación

### Gap A: Anonimato y Privacidad (Gobernanza de Seguridad) ✅

**Problema identificado por el evaluador:**
> El modelo de nodos asocia campos de salud con leyes de protección de datos, pero el grafo no califica el nivel de sensibilidad del dato.

**Solución implementada:**

1. **`DataClassification` enum** en `src/graph/schema.py`:
   - `PUBLICO` | `INTERNO` | `PII` | `SENSIBLE`

2. **`AnonymizationRuleNode`** en `src/graph/schema.py`:
   - Técnicas: enmascaramiento, hash, k-anonimato, generalización, supresión, seudonimización
   - Campo `sql_expression` para inyectar SQL específico
   - Campo `required_for` que indica qué clasificaciones lo requieren

3. **Auto-detección en `src/nomenclar.py`**:
   - `_detect_data_classification()` analiza el nombre de la columna contra keyword sets
   - `_SENSIBLE_KEYWORDS`: diagnostico, enfermedad, salud_mental, vih, discapacidad, genetico, etc.
   - `_PII_KEYWORDS`: nombre, apellido, identificacion, documento, direccion, telefono, email, etc.
   - Los conceptos creados por el nomenclador reciben `data_classification` automáticamente

4. **Anonimización automática en `src/transformer.py`**:
   - `_generate_anonymization_sql()` genera SQL según clasificación:
     - **PII**: Seudonimización via hash SHA-256 (`SUBSTRING(ENCODE(DIGEST(..., 'sha256'), 'hex'), 1, 16)`)
     - **PII (fechas)**: Generalización a año (`EXTRACT(YEAR FROM ...)`)
     - **SENSIBLE**: Generalización de categorías (`CASE WHEN ... IS NOT NULL THEN 'registrado'`)
   - El SQL de anonimización se concatena al artefacto de transformación
   - `data_classification` se incluye en el JSON Schema (`x-nomenclador.data_classification`)
   - Se incluye en el mapeo declarativo

5. **Métodos en `src/graph/catalog.py`**:
   - `add_anonymization_rule()`, `link_anonymization()`, `find_anonymization_rules()`
   - `set_data_classification()`, `get_data_classification()`, `find_sensitive_data()`

6. **CLI**:
   - `classify <variable> [publico|interno|pii|sensible]` — clasificar manualmente
   - `sensitive` — listar todos los datos PII/sensibles del nomenclador (con tabla Rich)

---

### Gap B: Cuello de Botella de NetworkX (Escalabilidad) ⏳ PENDIENTE

**Problema identificado:**
> NetworkX persistido en JSON local generará race conditions cuando múltiples agentes o desarrolladores actualicen simultáneamente.

**Estado:** Pendiente decisión arquitectónica. Opciones evaluadas:
- **Apache Age** (extensión PostgreSQL para grafos) — natural con Supabase
- **Neo4j** — grafo nativo con ACID
- Migración requiere rediseñar `catalog.py` (persistencia) manteniendo la API

---

### Gap C: Paradoja del "Human-in-the-Loop" en Ronda 2 ✅

**Problema identificado:**
> Un LLM nunca puede auto-aprobar la inserción de un concepto canónico sin validación humana.

**Solución implementada:**

1. **`ReviewStatus` enum** en `src/graph/schema.py`:
   - `PROPOSED` → `UNDER_REVIEW` → `APPROVED` | `REJECTED`

2. **Campos nuevos en `ConceptNode` y `FieldNode`**:
   - `review_status: str = "approved"` — los nodos preexistentes quedan como aprobados
   - `proposed_by: str = ""` — registra quien propuso (ej: `agent:nomenclar`)

3. **Nomenclar (`src/nomenclar.py`)**:
   - Los conceptos creados por el LLM en Ronda 2 se crean con `review_status="proposed"` y `proposed_by="agent:nomenclar"`
   - Los fields asociados también quedan en `review_status="proposed"`
   - Esto significa que **ningún concepto propuesto por IA es activo por defecto**

4. **Lifecycle (`src/lifecycle.py`)**:
   - `REVIEW_ACTIONS` mapea estados a eventos del decision log
   - `log_review_event()` registra cada transición en el decision log con actor y razón
   - `find_pending_reviews()` lista conceptos con eventos de revisión pendientes

5. **Métodos en `src/graph/catalog.py`**:
   - `set_review_status()`, `get_review_status()`
   - `find_proposed_nodes()` — lista todos los nodos en `proposed` o `under_review`
   - `approve_node()`, `reject_node()`

6. **CLI**:
   - `review` (sin args) — lista nodos pendientes con su status, proposed_by y data_classification
   - `review <variable> approve` — aprueba un nodo (queda `approved`)
   - `review <variable> reject` — rechaza un nodo (queda `rejected`)
   - `review <variable> start` — marca como `under_review` (en revisión por custodio)
   - Cada acción registra evento en el decision log

**Flujo completo:**
```
Ronda 2 (LLM) → crea concepto con review_status="proposed"
                    ↓
Custodio ve: review (lista pendientes)
                    ↓
Custodio revisa: review <variable> start → "under_review"
                    ↓
Custodio decide: review <variable> approve → "approved" (activo)
                 review <variable> reject  → "rejected" (descartado)
```

---

### Gap D: Clasificadores Dinámicos e Históricos ✅

**Problema identificado:**
> ¿Qué pasa cuando el clasificador cambia o es una ontología jerárquica compleja (como ICD-10)?

**Solución implementada:**

1. **Campos nuevos en `ClassifierNode`** (`src/graph/schema.py`):
   - `version_label: str = ""` — ej: "ICD-10 2019", "ICD-10 2024"
   - `parent_id: Optional[str] = None` — referencia a clasificador padre (jerarquía)
   - `is_current: bool = True` — indica si es la versión vigente

2. **Aristas nuevas**:
   - `EQUIVALE_A` — mapeo entre versiones del mismo clasificador (ej: ICD-10 v2019 ↔ ICD-10 v2024)
   - `SUBCONCEPTO_DE` — jerarquía padre-hijo (ej: código específico → capítulo)

3. **Métodos en `src/graph/catalog.py`**:
   - `link_classifier_equivalent(a, b, mapping)` — vincula dos clasificadores como equivalentes con diccionario de mapeo opcional `{codigo_a: codigo_b}`
   - `link_classifier_subconcept(child, parent)` — vincula un clasificador como subconcepto
   - `find_classifier_equivalents(id)` — busca equivalentes en ambas direcciones (out + in edges)
   - `find_classifier_hierarchy(id)` — retorna `{parents: [...], children: [...]}`

**Caso de uso ejemplo:**
```
Hospital usa ICD-10 v2019 (classifier:icd10_2019)
Ministerio usa ICD-10 v2024 (classifier:icd10_2024)

link_classifier_equivalent("icd10_2019", "icd10_2024", 
    mapping={"A00": "A00", "A01.0": "A01.0", ...})

→ El transformer puede mapear códigos entre versiones
→ find_classifier_equivalents("icd10_2019") retorna la v2024 + mapping
```

---

### Gap M (Observación): MoA Guardrail de Arbitraje ✅

**Problema identificado:**
> Asegúrate de que el Sintetizador tenga un guardrail específico de arbitraje cuando el Jurídico y el Técnico entren en conflicto.

**Solución implementada:**

1. **`SINTETIZADOR_PROMPT` actualizado** (`src/moa_agent.py`):
   - Reglas de arbitraje explícitas:
     - **JURÍDICO tiene prioridad ABSOLUTA** en temas de protección de datos, PII y cumplimiento legal
     - Si JURÍDICO veta → recomendación final DEBE ser NO usar la variable
     - Si JURÍDICO no objeta → TÉCNICO prevalece en estandares/interoperabilidad
     - ESTADÍSTICO tiene voz consultiva (advertencias, no veto)
     - Conflicto no resuelto → recomendar NO proceder hasta decisión humana

2. **`_detect_juridico_tecnico_conflict()`** (`src/moa_agent.py`):
   - Detecta veto jurídico (keywords: "no cumple", "viola", "veta", "prohibido", "ilegal", "datos personales", "PII", "sensible")
   - Detecta aprobación técnica (keywords: "es viable", "se puede", "compatible", "interoperable", "factible")
   - Si hay conflicto, inyecta un notice en el input del sintetizador:
     ```
     *** CONFLICTO DETECTADO: El agente JURIDICO identifica objeciones legales 
     mientras el agente TECNICO considera viable la operacion. 
     El guardrail de arbitraje da prioridad al JURIDICO. ***
     El sintetizador DEBE aplicar el guardrail de arbitraje.
     ```

---

## CLI — 22 Comandos

| Comando | Descripción | Gap |
|---------|-------------|-----|
| `profile <csv>` | Perfilar CSV y construir nomenclador | — |
| `ingest <file>` | Ingerir archivo sucio via RAG Factory [--auto] [--llm] | — |
| `nomenclar <file>` | Descubrir + completar variables en 2 rondas [--auto] | A, C |
| `catalog` | Mostrar catálogo completo | — |
| `search <var>` | Buscar variable en el nomenclador | — |
| `interop <db1> <db2>` | Verificar interoperabilidad con guardrails | — |
| `transform <db1> <db2>` | Generar artefactos SQL + JSON Schema | A |
| `normative <file>` | Ingerir documento normativo [--tag] | — |
| `normative-search "q"` | Buscar en corpus normativo | — |
| `assign <variable>` | Asignar custodio/departamento | — |
| `history <variable>` | Ver decision log y ciclo de vida | — |
| `deprecate <variable>` | Marcar como deprecada [--reason] [--replacement] | — |
| `reactivate <variable>` | Reactivar variable | — |
| `version [info\|major\|minor\|patch]` | Versionado semántico | — |
| `compose <nombre>` | Crear variable compuesta | — |
| `context <variable>` | Registrar significado contextual | — |
| `conflicts` | Detectar conflictos de contexto | — |
| **`review [variable] [approve\|reject\|start]`** 🆕 | Gestionar conceptos propuestos por IA | C |
| **`classify <variable> [nivel]`** 🆕 | Clasificar sensibilidad del dato | A |
| **`sensitive`** 🆕 | Listar datos PII/sensibles | A |
| `agent "consulta"` | Ejecutar agente ReAct con Groq | — |
| `moa "consulta"` | MoA: 3 agentes + sintetizador con arbitraje | M |

---

## Archivos Principales (17 archivos, ~5,800 líneas)

| Archivo | Líneas | Cambios |
|---------|--------|---------|
| `src/cli.py` | 1,155 | +3 comandos nuevos (review, classify, sensitive) |
| `src/rag_factory.py` | 657 | — |
| `src/nomenclar.py` | 488 | +auto-detección PII (Gap A), +review_status=proposed (Gap C) |
| `src/graph/catalog.py` | 481 | +métodos anonimización, review, classifier hierarchy |
| `src/moa_agent.py` | 472 | +guardrail arbitraje jurídico vs técnico (Gap M) |
| `src/agent.py` | 432 | — |
| `src/transformer.py` | 369 | +anonimización SQL automática (Gap A) |
| `src/normative_rag.py` | 337 | — |
| `src/mcp_server.py` | 298 | — |
| `src/lifecycle.py` | 234 | +review workflow events (Gap C) |
| `src/guardrails.py` | 207 | — |
| `src/standards.py` | 190 | — |
| `src/profiler.py` | 187 | — |
| `src/graph/schema.py` | 185 | +DataClassification, ReviewStatus, AnonymizationRuleNode, classifier fields |
| `src/groq_client.py` | 95 | — |

---

## Pendiente

1. **Gap B**: Migración de persistencia NetworkX → graph database (Apache Age o Neo4j)
   - Requiere rediseñar capa de persistencia de `catalog.py`
   - Mantener API compatible para no romper los 8 componentes
   - Considerar Supabase + Apache Age como path natural

2. **Observaciones técnicas pendientes**:
   - Empaquetar en Docker container para aislar entorno (servidores gubernamentales)
   - Migrar vector store a pgvector (Supabase) cuando el corpus normativo crezca
