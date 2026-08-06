# Contribuir a Governance Agent

¡Gracias por tu interés en contribuir! Governance Agent es un proyecto de código abierto que busca mejorar la calidad de los datos para la formulación de políticas públicas en América Latina y el Caribe.

## Cómo contribuir

### Reportar problemas

Si encuentras un bug o tienes una sugerencia:

1. Busca si ya existe un [issue](https://github.com/rogelioGuerrero/governance-agent/issues) similar
2. Si no existe, crea uno nuevo con:
   - Descripción clara del problema
   - Pasos para reproducirlo
   - Comportamiento esperado vs. actual
   - Versión de Python y sistema operativo

### Proponer mejoras

1. Haz fork del repositorio
2. Crea una rama: `git checkout -b feature/mi-mejora`
3. Haz tus cambios
4. Verifica que los tests pasen: `uv run python scripts/test_core.py`
5. Haz commit con mensaje descriptivo
6. Abre un Pull Request

### Áreas de contribución

#### Domain Packs nuevos

Cada ministerio o dominio de política pública necesita su propio pack. Puedes contribuir creando packs para:

- **Agricultura**: nomenclador de cultivos, rendimientos, superficies
- **Educación**: matrículas, establecimientos, docentes
- **Vivienda**: catastro, subsidios, construcciones
- **Beneficencia social**: beneficiarios, elegibilidad, transferencias

Para crear un pack nuevo, copia `src/domain_packs/salud/pack.yaml` como plantilla y adapta:

- `schema_fields`: campos del dominio
- `semantic_rules`: reglas lógicas en lenguaje natural
- `inference_mappings`: sinónimos de campos
- `custom_validators`: validadores Python específicos

#### Conectores

Adaptadores para leer datos de sistemas gubernamentales específicos:

- APIs REST de ministerios
- Bases de datos PostgreSQL/MySQL
- Archivos CSV/Excel/SFTP
- Sistemas como DHIS2, OpenMRS, CommCare

#### Validadores semánticos

Reglas de dominio que el LLM usa para detectar inconsistencias lógicas. Cuantas más reglas, mejor la validación.

## Estilo de código

- Python 3.11+
- Type hints en todas las funciones públicas
- Docstrings en español (público objetivo: gobiernos de ALC)
- Sin dependencias innecesarias — preferir librería estándar
- Tests para toda funcionalidad nueva

## Estructura de commits

```
tipo(alcance): descripción breve

tipo: feat, fix, docs, refactor, test, chore
alcance: core, vrp, salud, docs, etc.
```

Ejemplos:
```
feat(salud): añadir validador de códigos CIE-10
fix(core): corregir parsing de arrays anidados en domain_pack
docs(readme): actualizar sección de instalación
```

## Licencia

Al contribuir, aceptas que tus cambios se publiquen bajo la [Apache License 2.0](LICENSE).
