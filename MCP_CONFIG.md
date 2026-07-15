# Configuración MCP del Nomenclador

## Cómo conectar el Nomenclador a tu IDE

El servidor MCP del Nomenclador expone 7 tools que permiten a cualquier IDE
con MCP nativo (Cursor, Windsurf, VS Code con extensión MCP) consultar el
knowledge graph en tiempo real mientras escribes código.

### Windsurf / Cascade

Archivo: `~/.codeium/windsurf/mcp_config.json`

```json
{
  "mcpServers": {
    "nomenclador": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.mcp_server"],
      "cwd": "D:\\proyectoBolt\\governance-agent"
    }
  }
}
```

### Cursor

Archivo: `~/.cursor/mcp.json`

```json
{
  "mcpServers": {
    "nomenclador": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.mcp_server"],
      "cwd": "D:\\proyectoBolt\\governance-agent"
    }
  }
}
```

### VS Code (con extensión MCP)

Archivo: `.vscode/mcp.json` en el workspace

```json
{
  "servers": {
    "nomenclador": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "python", "-m", "src.mcp_server"],
      "cwd": "D:\\proyectoBolt\\governance-agent",
      "env": {}
    }
  }
}
```

## Tools disponibles

| Tool | Descripción |
|---|---|
| `list_concepts` | Lista todos los conceptos canónicos del nomenclador |
| `search_variable` | Busca una variable por nombre y muestra en qué fuentes está |
| `get_concept` | Obtiene detalle completo de un concepto + clasificador |
| `check_interoperability` | Valida interoperabilidad entre dos fuentes con guardrails (3 checkpoints) |
| `get_transform` | Genera SQL CASE WHEN + JSON Schema para conectar dos fuentes |
| `validate_field` | Valida si un campo cumple con el estándar canónico |
| `get_classifier` | Obtiene los valores válidos de un estándar |

## Ejemplos de uso en el IDE

### Caso 1: Desarrollador crea un nuevo campo

> **Tú en el IDE**: "Voy a crear un campo `sexo` en la tabla pacientes, ¿qué valores debo usar?"

Cascade consulta `validate_field("sexo", ["M", "F"])` y responde:
> "Los valores M/F no son canónicos. El estándar ISO 5218 requiere: 0=desconocido, 1=masculino, 2=femenino, 9=no_aplicable. Necesitas transformación."

### Caso 2: Desarrollador pregunta por interoperabilidad

> **Tú en el IDE**: "¿Puedo cruzar los datos del censo con los del hospital?"

Cascade consulta `check_interoperability("sample_censo", "sample_hospital")` y responde:
> "Hay 3 caminos posibles pero INTEROPERABILIDAD NO RECOMENDADA. La población es diferente (población general vs pacientes hospitalizados) y la metodología de captura también (auto-reporte vs registro clínico). El clasificador coincide (ISO 5218)."

### Caso 3: Generar transformación SQL

> **Tú en el IDE**: "Genera el SQL para alinear sexo del censo al estándar"

Cascade consulta `get_transform("sample_censo", "sample_hospital")` y obtiene el CASE WHEN listo.
