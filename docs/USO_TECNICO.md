# Uso técnico

Documentación para desarrolladores que implementan o integran la herramienta.

---

## Instalación

### Requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (gestor de paquetes)
- Clave de acceso de al menos un proveedor de IA:
  - Cualquier proveedor compatible con la API de OpenAI (varios ofrecen nivel gratuito)

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/rogelioGuerrero/governance-agent.git
cd governance-agent

# 2. Instalar dependencias
uv sync

# 3. Configurar claves de acceso
cp .env.example .env
# Editar .env con tu clave de cualquier proveedor compatible con OpenAI
#   Ej: GROQ_API_KEY=gsk_...  |  GEMINI_API_KEY=...  |  SAMBANOVA_API_KEY=...

# 4. Verificar instalación
uv run python scripts/quickstart.py
```

---

## Uso rápido

```python
from src.core.domain_pack import PackLoader
from src.core.validator import ValidationEngine
from src.core.llm_adapter import LLMAdapter
from src.core.pack_memory import PackMemory
from src.core.human_loop import HumanInTheLoop

# 1. Cargar la configuración del ministerio
pack = PackLoader.from_yaml("src/domain_packs/salud/pack.yaml")

# 2. Configurar el motor de validación
llm = LLMAdapter(json_mode=True, temperature=0.1)
memory = PackMemory("salud")
hitl = HumanInTheLoop(pack_memory=memory)
engine = ValidationEngine(pack=pack, pack_memory=memory, hitl=hitl, llm_client=llm)

# 3. Validar el consolidado de datos
resultado = engine.validate(consolidado_de_datos)

# 4. Revisar resultados
print(f"Válido: {resultado.is_valid}")
print(f"Problemas: {len(resultado.issues)}")
for issue in resultado.issues:
    print(f"  [{issue.severity}] {issue.field_name}: {issue.message}")
    if issue.suggested_value:
        print(f"    Sugerencia: {issue.suggested_value}")

# 5. Preguntas para el planificador (no bloqueantes)
for q in hitl.get_pending_questions():
    print(f"  [{q.level}] {q.field_name}: {q.message}")
```

---

## Módulos de dominio

Un módulo de dominio encapsula todo el conocimiento de un área de política pública:

| Componente | Descripción | Ejemplo |
|-----------|-------------|---------|
| **Variables esperadas** | Estructuras de datos, tipos, obligatoriedad | `locations.id` (texto, requerido), `locations.coords` (coordenadas, requerido) |
| **Reglas semánticas** | Reglas lógicas en lenguaje natural para la IA | "end_time del vehículo ≥ time_window_end más tardío" |
| **Sinónimos de variables** | Equivalencias entre sistemas para interoperabilidad | `lat` ↔ `latitude` ↔ `latitud` ↔ `y` |
| **Validadores específicos** | Validadores en Python | Coordenadas en área de operación, balance recogida-entrega |
| **Clasificadores** | Nomencladores del dominio | CIE-10, CUOC, cultivos permitidos |
| **Configuración** | Parámetros del dominio | Área geográfica, horas típicas de servicio |

### Módulos disponibles

| Módulo | Dominio | Estado |
|--------|---------|--------|
| `vrp` | Planificación de despliegue territorial | Funcional con datos reales |
| `salud` | Nomenclador de salud | Funcional |

### Crear un nuevo módulo

```bash
# Auto-generar desde un modelo existente
uv run python scripts/generate_vrp_pack.py

# O crear manualmente un pack.yaml
# Ver src/domain_packs/salud/pack.yaml como ejemplo
```

---

## Arquitectura

```
governance-agent/
├── src/
│   ├── core/                          # Núcleo (independiente del dominio)
│   │   ├── domain_pack.py             # Configuración de módulos + loader
│   │   ├── validator.py               # Motor de validación multi-capa
│   │   ├── llm_adapter.py             # Adaptador de IA multi-proveedor
│   │   ├── orchestrator.py            # Orquestador: validar → corregir
│   │   ├── pack_memory.py             # Memoria de correcciones con auto-promoción
│   │   ├── human_loop.py              # Human-in-the-loop batch
│   │   ├── profiler.py                # Análisis de datos
│   │   ├── inference.py               # Inferencia de equivalencias entre variables
│   │   ├── standards.py               # Registro dinámico de clasificadores
│   │   └── mcp_server_abstract.py     # Servidor de integración
│   ├── domain_packs/                  # Módulos de dominio (intercambiables)
│   │   ├── vrp/                       # Planificación de despliegue territorial
│   │   │   ├── pack.yaml              # Configuración + reglas + mapeos
│   │   │   └── vrp_validators.py      # Validadores específicos
│   │   └── salud/                     # Nomenclador de salud
│   │       └── pack.yaml              # Configuración + reglas
│   ├── llm_client.py                  # Cliente de IA multi-proveedor
│   └── mcp_server.py                  # Servidor de integración
├── scripts/                           # Scripts de prueba y demostración
│   ├── quickstart.py                  # Demostración rápida
│   ├── test_real_data.py              # Pruebas con datos reales
│   ├── test_llm_semantic.py           # Pruebas de validación semántica
│   ├── test_orchestrator.py           # Pruebas del orquestador completo
│   └── generate_vrp_pack.py           # Generación automática de configuración
├── pyproject.toml
├── LICENSE
└── README.md
```

**Principio clave**: el núcleo (`core/`) no conoce ningún dominio. Todo el conocimiento de dominio vive en los módulos (`domain_packs/`). Un nuevo ministerio = un nuevo módulo. El código no se modifica.
