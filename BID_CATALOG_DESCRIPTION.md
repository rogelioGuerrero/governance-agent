# Descripción para el catálogo Código para el Desarrollo del BID

## Datos de la herramienta

| Campo | Valor |
|-------|-------|
| **Nombre** | Governance Agent |
| **Tipo de herramienta** | API, Algoritmo, Plugin |
| **Licencia** | Apache License 2.0 |
| **Lenguaje** | Python |
| **Versión** | 1.0.0 |
| **Categorías** | Inteligencia artificial, Interoperabilidad de datos, Gestión de bases de datos |
| **País de origen** | El Salvador |
| **Estado** | Activo |

---

## Descripción general

Governance Agent es un framework de código abierto que asegura la calidad de los datos como insumo para la gestión pública. Cada ministerio o institución encapsula su nomenclador, reglas de validación y clasificadores de variables en un *Domain Pack* intercambiable. Cuando la institución consolida datos de sus sistemas transaccionales para planificar o evaluar, Governance Agent valida que los datos sean estructuralmente correctos, semánticamente consistentes mediante inteligencia artificial, y coherentes con el clasificador de variables del dominio.

Si la institución aún no cuenta con un clasificador de variables o nomenclador formal, Governance Agent puede apoyar su construcción: el framework es capaz de inferir automáticamente la estructura de los datos a partir de los modelos existentes en los sistemas transaccionales, y generar un Domain Pack inicial que el equipo técnico puede luego refinar. Adicionalmente, el agente puede incorporar documentación normativa y técnica del dominio como respaldo para la validación semántica, sin requerir configuración manual compleja.

Si detecta errores, el agente los corrige automáticamente usando IA. Si no puede corregirlos, formula preguntas al planificador en lote, sin detener el proceso. Y aprende de cada corrección aceptada o rechazada para no repetir los mismos errores en el futuro.

El valor de Governance Agent trasciende la validación previa a la formulación de una política. Entre otros usos, permite monitorear programas en ejecución, responder preguntas sobre los datos con confianza, detectar hallazgos que ningún sistema tradicional identifica, o gestionar de forma propositiva con datos integrados mediante el nomenclador.

El resultado: gestión pública apoyada en datos confiables, sin importar la fuente de origen.

---

## El problema que resuelve

La formulación de políticas públicas en América Latina y el Caribe se basa en datos extraídos de sistemas transaccionales que presentan problemas endémicos de calidad. Estos problemas afectan a todos los países de la región, independientemente de su nivel de desarrollo digital:

1. **Inconsistencias lógicas no detectables por validación tradicional**: Un registro puede tener "edad: 25" y "fecha de nacimiento: 2010". Ambos campos son válidos individualmente, pero imposibles en conjunto. Los sistemas de validación tradicionales no detectan este tipo de errores porque solo verifican estructura, no semántica.

2. **Nomencladores inconsistentes entre sistemas**: El Ministerio de Salud usa "cod_dx" para diagnóstico, el hospital usa "diagnostico", y el censo usa "CIE10". Ningún sistema reconcilia estas diferencias, lo que impide consolidar información para planificación.

3. **Datos geográficamente inválidos**: Coordenadas fuera del área de operación, depósitos ubicados a cientos de kilómetros de las rutas de entrega, puntos duplicados con identificadores distintos.

4. **Errores de captura masivos**: En consolidados de datos gubernamentales, hasta el 30% de los registros pueden contener campos erróneos que pasan validación estructural pero son lógicamente imposibles. Estos errores propagan decisiones equivocadas a la política pública.

Las herramientas existentes en el catálogo de Código para el Desarrollo abordan parte del problema. Data Cleaner aplica reglas sintácticas a archivos CSV. OpenRefine permite limpieza manual desde el navegador. Atypical Data Classifier detecta anomalías en encuestas de hogares. Sin embargo, ninguna combina validación semántica con IA, corrección automática, y la capacidad de adaptarse a cualquier dominio de política pública mediante packs intercambiables.

---

## Cómo funciona

Governance Agent aplica tres capas de validación secuencial sobre el consolidado de datos:

**Capa 1 — Estructural**: Verifica tipos de datos, campos obligatorios, enumeraciones y rangos mínimo/máximo. Esta capa usa el schema declarativo definido en el Domain Pack del ministerio.

**Capa 2 — Reglas de dominio**: Ejecuta validadores específicos del dominio escritos en Python. Por ejemplo: verificar que el monto de un subsidio no exceda el máximo legal permitido, que los códigos existan en el nomenclador, o que un beneficiario no esté duplicado entre programas.

**Capa 3 — Semántica con IA**: Utiliza modelos de lenguaje (LLM) para razonar sobre los datos y detectar inconsistencias lógicas que las capas anteriores no pueden predecir. Por ejemplo: un registro con edad 25 y fecha de nacimiento 2010 es contradictorio, o un beneficiario con ingresos superiores al umbral del programa. El agente puede incorporar documentación normativa del dominio como respaldo para enriquecer el razonamiento semántico.

Cuando se detectan errores críticos, el agente usa IA para corregir automáticamente los datos y re-intenta la validación, hasta tres iteraciones. Los warnings no críticos se acumulan como preguntas en lote para el planificador, sin detener el proceso. Cada corrección aceptada o rechazada se almacena en memoria, y tras cinco aceptaciones de la misma corrección, se auto-promueve a regla automática. El agente aprende del dominio.

---

## Casos de uso

### Ministerio de Agricultura

Un ministerio de agricultura necesita apoyar una política de subsidios a pequeños productores. Consolida los datos anuales de su registro de productores. Governance Agent valida que todos los registros tengan los campos obligatorios, que los cultivos existan en el nomenclador, que las coordenadas estén en área agrícola, y que los rendimientos sean plausibles. Si un registro reporta 50 toneladas por hectárea en papa, el agente sugiere corregir a 18 (rango normal: 15-25).

### Ministerio de Salud

Un ministerio de salud consolida datos de RIPS, SISS y el censo para evaluar cobertura. Governance Agent valida que los códigos CIE-10 existan en el nomenclador, que los municipios sean válidos, y que no haya contradicciones lógicas entre campos como edad y fecha de nacimiento. Si encuentra "edad: 25" con "fecha de nacimiento: 2010", sugiere corregir la fecha a 1999.

### Programa de Beneficencia Social

Un programa de transferencias condicionadas necesita verificar la elegibilidad de beneficiarios antes de emitir pagos. Governance Agent valida que los ingresos declarados sean consistentes con la actividad económica registrada, que no haya beneficiarios duplicados entre programas, y que las composiciones familiares sean coherentes (ej: no registrar hijos mayores de edad como dependientes sin justificación). Si un beneficiario reporta ingresos superiores al umbral del programa, el agente marca el registro para revisión.

### Alcaldía Municipal

Una alcaldía consolida datos de catastro, registro civil y servicios públicos para planificar inversiones locales. Governance Agent valida que los predios existan en el catastro, que las direcciones correspondan al municipio, y que los datos sean coherentes. Si un predio de 5 m² está declarado como residencia familiar, el agente marca el registro como implausible para revisión.

### Consulta de viabilidad de política pública

Una vez que el nomenclador está construido, la institución puede consultar el grafo para responder preguntas de política pública antes de invertir en nuevos sistemas:

- **¿Podemos monitorear esta política?** — El gobierno consulta qué variables necesita el indicador, en qué sistemas están, y con qué calidad. Si la variable existe en dos sistemas pero con 60% de completitud, sabe que necesita mejorar la captura antes de usar el dato.
- **¿Podemos actualizar el indicador sin crear un sistema nuevo?** — El gobierno consulta los caminos de interoperabilidad entre sistemas. Si el censo y el hospital comparten la variable "sexo" con el mismo clasificador ISO 5218, puede cruzarlos sin construir un nuevo pipeline.
- **¿Podemos crear una política nueva con variables que ya existen?** — El gobierno descubre que tiene variables dispersas en tres ministerios que, combinadas, permiten diseñar un programa de transferencias condicionadas. El nomenclador identifica qué variables están disponibles, en qué fuentes, y qué calidad tienen — sin necesidad de nuevas encuestas.

### Gobernanza de variables y respaldo normativo

Cada variable del nomenclador tiene un **custodio institucional** (la persona o dirección responsable de mantenerla) y **respaldo normativo trazable** (la ley, resolución o norma que la respalda). Esto permite:

- **Saber quién responde por cada variable** — si el indicador de cobertura de vacunación tiene problemas, el nomenclador dice que el custodio es la Dirección de Inmunizaciones, no hay que adivinar.
- **Trazar el respaldo legal** — cada variable está vinculada al artículo específico de la norma que la respalda (ej: Art. 47 de la Ley General de Salud para vacunación obligatoria).
- **Auditar el cumplimiento normativo** — el gobierno puede listar qué variables tienen respaldo normativo y cuáles no, identificando brechas de gobernanza antes de que se conviertan en problemas legales.

### Construcción del nomenclador

¿Cómo se construye el nomenclador? Los ministerios tienen sus datos en sistemas distintos, con estructuras de datos que nadie ha revisado en años. Governance Agent lee esas estructuras tal como están, identifica qué variable corresponde a cada elemento, y arma el nomenclador automáticamente. El equipo del ministerio solo revisa y aprueba — no tiene que construirlo desde cero.

### Descubrimiento de oportunidades de política pública

A partir de las variables disponibles en el nomenclador, el agente genera insights sobre las fuentes de datos: qué variables tienen mejor calidad, qué variables permiten cruzar información entre fuentes, y qué combinaciones de variables habilitan nuevos indicadores. El agente puede sugerir políticas o programas que la institución podría implementar con los datos que ya tiene — sin necesidad de recolectar información nueva.

### ¿Podemos implementar esta política?

Cuando una institución quiere diseñar o monitorear una política, necesita saber si tiene los datos para hacerlo. El agente lee la descripción de la política, identifica qué variables se necesitan, busca cada una en el nomenclador, y responde: **¿podemos hacerlo con los datos que tenemos hoy, necesitamos ajustes, o todavía no es posible?** El reporte dice qué variables existen y en qué sistemas, cuáles faltan, y qué porcentaje de la política se puede sustentar con datos actuales.

### Trazabilidad y auditoría de variables

Cada variable del nomenclador tiene un historial: quién la creó, quién la modificó, por qué, y cuándo. Si una variable cambia o se retira, el cambio queda registrado. Esto permite a la institución — y a los órganos de control — reconstruir el recorrido de cualquier dato que sustente una decisión de política pública.

### Registros administrativos con fines estadísticos

Un ministerio quiere usar los registros de vacunación para estimar cobertura nacional. Pero "vacunado" en el registro del hospital significa "dosis aplicada", mientras que en el censo significa "esquema completo". Governance Agent detecta que ambas fuentes tienen la variable "estado de vacunación" pero con distinta metodología de captura y distinta población objetivo, y advierte que no son directamente comparables — antes de que el ministerio publique un indicador equivocado.

### Contraloría social

Una organización de la sociedad civil descarga los datos abiertos publicados por tres ministerios y los carga en Governance Agent. El agente construye el nomenclador, evalúa la calidad de cada variable, detecta inconsistencias entre fuentes, y genera un informe de factibilidad: qué políticas públicas se pueden sustentar con los datos disponibles, qué variables faltan, y dónde hay brechas de calidad. La organización publica el informe como evidencia para el debate público — sin necesidad de acceso a sistemas internos del gobierno.

---

## Nivel de esfuerzo de implementación

**Alto** — Governance Agent es un framework que requiere equipo técnico para implementar. Cada ministerio necesita:

1. **Crear su Domain Pack**: definir el nomenclador, las reglas semánticas y los validadores específicos de su dominio.
2. **Integrar con sus sistemas**: configurar la extracción de consolidados desde sus sistemas transaccionales.
3. **Configurar las API keys de IA**: obtener acceso a al menos un proveedor de LLM compatible con la API de OpenAI (varios ofrecen free tier).
4. **Capacitar al equipo**: en el mantenimiento y evolución de los packs.

El framework está diseñado para ser flexible: cada gobierno tiene sistemas distintos, nomencladores distintos, y reglas de dominio distintas. Por ello, el núcleo es abstracto y la configuración específica de cada ministerio se realiza mediante Domain Packs, que pueden crearse manualmente o generarse automáticamente desde los modelos de datos existentes en los sistemas de la institución.

---

## Requisitos técnicos

- **Python 3.11+**
- **Gestor de paquetes**: uv
- **API LLM**: Cualquier proveedor compatible con la API de OpenIA (varios ofrecen free tier)
- **Sin dependencias pesadas**: no requiere pandas, numpy ni bases de datos externas para funcionamiento básico
- **Despliegue**: local, on-premise o cloud

---

## Roadmap

**Completado:**
- Núcleo abstracto con validación multi-capa
- Integración LLM agnóstica (cualquier proveedor compatible con OpenAI) para validación semántica
- Auto-corrección de errores con IA
- Memoria acumulativa con auto-promoción de reglas
- Human-in-the-loop batch no intrusivo
- Orquestador completo: validar → corregir → ejecutar
- Auto-generación de packs desde modelos Pydantic
- MCP server para integración con asistentes IA

**Futuras versiones:**
- Conectores para sistemas gubernamentales (API, DB, CSV, SFTP)
- Nomenclador canónico como puente entre sistemas
- UI web para planificadores no técnicos
- Dashboard de calidad de datos por dominio

*Estas funcionalidades están en desarrollo y se incorporarán en futuras versiones.*

---

## Información de contacto

- **Repositorio**: [https://github.com/rogelioGuerrero/governance-agent](https://github.com/rogelioGuerrero/governance-agent)
- **Issues**: [https://github.com/rogelioGuerrero/governance-agent/issues](https://github.com/rogelioGuerrero/governance-agent/issues)
- **Email**: [info@agtisa.com]

---

## Declaración de Bien Público Digital

Governance Agent aspira a ser reconocido como Bien Público Digital por su contribución a la mejora de la calidad de los datos gubernamentales en América Latina y el Caribe. El framework es de código abierto bajo licencia Apache 2.0, permite uso comercial, y está diseñado para ser reutilizable por cualquier gobierno de la región sin dependencias de proveedores específicos.
