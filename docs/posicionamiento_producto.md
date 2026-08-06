# Governance Agent — Posicionamiento y Alcance del Producto

## Analogia

El governance agent es la **capa de presentacion y sesion** del mundo de datos (modelo OSI adaptado):
- No es la aplicacion (BI, analytics)
- No es el transporte (ETL, pipelines)
- Es la capa que establece **que significa cada cosa** antes de que se transmita

## Dos Modos de Valor

### Escenario A: Equivalencia Semantica (mismo dominio, sistemas distintos)

**Cuando aplica:**
- Empresa con 3 sistemas de RRHH fusionados que miden "empleado" con schemas distintos
- MINED y UNESCO miden "completion rate" con codigos distintos
- Institucion con 15 sistemitas legacy que necesitan unificarse

**Que entrega:**
- "esta variable de sistema 1 = esta variable de sistema 2"
- Equivalencias EQUIVALE_A entre columnas de datasets distintos
- Grafo semantico que persiste y crece con cada nuevo sistema

**Que tenemos construido:**
- `profile` — perfila cualquier CSV (tipos, nulos, unicos, estandares)
- `ingest` — ingiere con inference engine (33% auto-resuelto sin LLM)
- `nomenclar` — 2 rondas de descubrimiento + LLM para equivalencias semanticas
- `catalog` — construye el grafo (NetworkX + PostgreSQL dual-write)
- `interop` — verifica interoperabilidad entre datasets
- `transform` — genera SQL + JSON Schema para convertir entre sistemas
- Inference engine — patrones, listas de referencia, huella de valores
- Agente ReAct con tools (search, audit, health, fix_orphans, etc.)
- Datasets demo (sample_censo, sample_hospital, etc.)
- **Datasets reales listos:** UNESCO education + World Bank education (CR.1 = Primary completion rate, codigos totalmente distintos)

**Estado: FUNCIONAL.** El pipeline end-to-end funciona. Faltan pulir detalles y hacer una corrida limpia con los datasets reales de educacion.

---

### Escenario B: Compatibilidad Dimensional (dominios distintos, se complementan)

**Cuando aplica:**
- Economia + agricultura + salud + medioambiente
- Ninguna variable de contenido coincide entre fuentes
- Se cruzan por dimensiones compartidas (pais, ano, region)

**Que entrega:**
- "estos datasets comparten las dimensiones pais y ano, puedes cruzarlos por ahi"
- Certificacion de que el cruce dimensional es valido
- Deteccion de trampas metodologicas (ano fiscal vs ano escolar, codigos territoriales distintos)
- Documentacion del alcance (que se puede y que no se puede cruzar)

**Que tenemos construido:**
- El mismo pipeline profile/nomenclar/interop funciona
- Pero el output es solo compatibilidad dimensional (pais, ano)
- No hay equivalencias EQUIVALE_A de contenido que descubrir
- El agente certifica que el JOIN es viable pero no hace el analisis

**Que NO tenemos (y no es trabajo del agente):**
- Analisis cruzado de los datos (hallazgos, correlaciones, indicadores)
- Generacion de insights de politica publica
- Recomendaciones estrategicas

**Estado: PARCIAL.** El agente puede perfilar y certificar el cruce, pero el valor es menor que el Escenario A porque las equivalencias son obvias (pais = pais, ano = ano). El analisis downstream es trabajo del analista, no del agente.

---

## Productos que el agente entrega

| # | Producto | Descripcion | Escenario |
|---|----------|-------------|-----------|
| 1 | Inventario Semantico | Dado N fuentes, perfilar y decir cuantos conceptos compartidos/unicos hay | A y B |
| 2 | Nomenclador (grafo vivo) | Mapa de relaciones semanticas entre sistemas. Activo institucional | A |
| 3 | Certificado de Interoperabilidad | Sistema A y B pueden/n pueden intercambiar datos, por que dimensiones | A y B |
| 4 | Capa de Transformacion | SQL/JSON Schema que convierte datos de sistema A al formato de B | A |
| 5 | Auditoria de Calidad | Conceptos sin custodio, campos sin verificar, conflictos de definicion | A y B |
| 6 | Analisis de Impacto | Si cambias X, estos N sistemas y M campos se ven afectados | A |

## Mercado natural

- Instituciones publicas con sistemas siloados (MINED, MINSAL, Hacienda)
- Empresas que crecieron por adquisicion y heredaron sistemas distintos
- Empresas medianas sin SAP/Oracle que viven con "sistemitas"
- Consultoras que hacen integracion de datos y necesitan un primer diagnostico rapido

## Lo que el agente NO hace (valido en otro contexto)

- Analisis de politica publica sobre datos integrados → trabajo de analista/LLM
- Hallazgos, correlaciones, indicadores compuestos → capa de analytics
- Recomendaciones estrategicas → consultoria sobre datos limpios
- BI, dashboards, visualizacion → herramientas de BI

## Limitacion fundamental — Equivalencias a nivel de valor (formato largo)

**Hallazgo verificado con PoC FAOSTAT + World Bank Agriculture (Jul 2026):**

El pipeline funciona correctamente para equivalencias a nivel de **columna** (mismo nombre, diferente schema). Pero cuando los datasets usan **formato largo** (long format), el significado semantico esta en los **valores** de las columnas, no en los nombres.

Ejemplo real:
- FAOSTAT: `Item=Maize` + `Element=Area harvested` → mide hectareas de maiz
- World Bank: `Indicator Name=Land under cereal production` → mide hectareas de cereales

La equivalencia `Maize + Area harvested ≈ Land under cereal production` es **no obvia** y requiere razonamiento semantico sobre los valores, no sobre los metadatos.

**Lo que el governance agent hace solo:**
- Perfila, infiere tipos, detecta estandares
- Construye el grafo de conocimiento
- Detecta equivalencias a nivel de columna (year↔year, value↔value)
- Valida interoperabilidad con guardrails
- Persiste, versiona, audita
- **Descubre equivalencias a nivel de valor** via las tools `sample_column_values` + `compare_value_sets` (implementado Jul 2026)

**Evolucion de la capacidad (Jul 2026):**

Inicialmente el agente no podia descubrir equivalencias a nivel de valor. Se agregaron dos tools al ReAct agent:

| Tool | Funcion |
|------|---------|
| `sample_column_values` | Extrae valores unicos de una columna de cualquier CSV del dataset |
| `compare_value_sets` | Compara dos conjuntos de valores via Groq LLM y propone equivalencias semanticas |

Con estas tools, el ReAct agent puede autonomamente:
1. Extraer valores de columnas en formato largo (`Item`, `Element`, `Indicator Name`)
2. Razonar sobre equivalencias semanticas entre los valores usando Groq
3. Retornar tabla de equivalencias con confidence (alta/media/baja) y razon

**Lo que el governance agent NO hace solo (limitacion residual):**
- Decidir si una equivalencia descubierta debe persistirse en el grafo (requiere validacion humana)
- Razonar sobre contexto de dominio no presente en los datos (ej: conocimiento experto de politica publica)
- Debuggear su propio codigo o tomar decisiones arquitectonicas

**Conclusion: el governance agent es autonomo para descubrimiento semantico.**
Con las tools `sample_column_values` + `compare_value_sets`, el agente descubre equivalencias no obvias sin necesidad de un agente externo (Cascade u otro). El producto vendible es el governance agent **solo**, con Groq como motor de razonamiento integrado.

**Donde si se necesita un agente externo:** para desarrollo, debugging y evolucion del propio governance agent ( Cascade u otro IDE agentic). Pero para el caso de uso de interoperabilidad, el agente es autónomo.

## Datasets disponibles para PoC

| Dataset | Fuente | Registros | Escenario |
|---------|--------|-----------|-----------|
| unesco_sdg4_education_slv | UNESCO UIS | 18,518 | A (con WB education) |
| worldbank_education_slv | World Bank | 17,572 | A (con UNESCO) |
| worldbank_health_slv | World Bank | 8,550 | B (con UNESCO education) |
| unesco_demographic_slv | UNESCO | ~1,700 | B (con cualquier otro) |
| sample_censo, sample_hospital, etc. | Demo | ~100 c/u | A (demo interno) |

> **Nota:** FAOSTAT crops + World Bank Agriculture se usaron como PoC temporal (Jul 2026) para validar limitaciones del agente con formatos largos. Los datasets se eliminaron despues de la prueba. Ver seccion "Limitacion fundamental" arriba.

## Monetizacion

### Escenario A: MONETIZABLE — producto core

El dolor que resuelve es real, frecuente y caro:
- Una empresa que se fusiona gasta meses en consultoria para mapear sistemas. El agente lo hace en minutos.
- Una institucion con 15 sistemitas no sabe ni que tiene. El agente le da el inventario semantico.
- Cada vez que llega un sistema nuevo, alguien tiene que sentarse a ver que columnas equivalen a que. El agente lo automatiza.

| Producto | Modelo | Frecuencia |
|----------|--------|------------|
| Inventario Semantico | Consultoria / SaaS por dataset | Por proyecto |
| Nomenclador vivo (grafo) | SaaS mensual | Recurrente |
| Certificado de Interoperabilidad | Por par de sistemas | Por integracion |
| Capa de Transformacion (SQL/Schema) | Por transformacion | Por integracion |
| Auditoria continua | Suscripcion | Recurrente |

El cliente paga por **ahorrar meses de trabajo manual** y por **tener un activo institucional** (el grafo) que persiste y crece.

### Escenario B: NO vale la pena meterle mas esfuerzo al agente

El valor que el agente agrega es marginal:
- "Pais = pais y ano = ano" → lo ve cualquiera en 30 segundos
- "No puedes cruzar ano fiscal con ano escolar" → util, pero es un caveat, no un producto
- El analisis real (correlaciones, hallazgos, policy) → no es trabajo del agente

**Pero el Escenario B si tiene valor en otro producto:** si manana se construye un producto de analytics multi-dominio (ej: "donde instalo mi fabrica"), ese producto necesitara al governance agent como **capa base invisible**. El producto vendible seria el analytics, no el governance. El governance seria el motor que hace que el analytics sea confiable.

Es decir: el Escenario B no es un producto del governance agent. Es un **caso de uso de un producto downstream** que consume al governance agent.

## Decision estrategica

- **Foco 100% en Escenario A** como producto del governance agent
- El Escenario B queda como **capacidad documentada** — el agente puede hacerlo, pero no es el pitch de ventas
- Si en el futuro se construye un producto de analytics multi-dominio, el governance agent ya esta como capa base

## Conclusion

El Escenario A es donde el agente tiene mayor valor, es monetizable, y donde ya tenemos el pipeline funcional.
El Escenario B es util pero el valor del agente se reduce a certificar compatibilidad dimensional — no justifica inversion adicional.
El producto se posiciona como **interoperabilidad semantica entre sistemas del mismo dominio** como caso core y monetizable.
