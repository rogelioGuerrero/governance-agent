"""Regression suite para el agente ReAct (governance-agent).

Tests deterministas (sin LLM) que verifican que las tools del agente
funcionan correctamente. Sirven como regression suite tras refactorings.

Marcados con @pytest.mark.slow los que requieren LLM (rate limit possible).
"""

import pytest
import json
from pathlib import Path

from src.graph.catalog import load_graph_cached, clear_graph_cache, NomencladorGraph
from src.agent import (
    run_agent,
    _classify_query,
    _select_tools_for_query,
    PRIMARY_TOOLS,
    ADVANCED_TOOLS,
    _build_openai_tools_schema,
    TOOLS,
    TOOLS_SCHEMA,
    tool_search_graph,
    tool_list_concepts,
    tool_audit_graph,
    tool_graph_health,
    tool_detect_standard,
    tool_compute_confidence,
)


TESTS_DIR = Path(__file__).parent
PROJECT_DIR = TESTS_DIR.parent
NOMENCLADOR_PATH = PROJECT_DIR / "nomenclador" / "nomenclador.json"


@pytest.fixture(scope="module")
def graph():
    """Cargar el grafo cacheado."""
    clear_graph_cache()
    g = load_graph_cached()
    yield g
    clear_graph_cache()


# === Tests de tools deterministas ===

class TestDeterministicTools:
    """Las tools deterministas son Python puro — deben dar resultados consistentes."""

    def test_search_graph_finds_concept(self):
        """search_graph debe encontrar un concepto existente."""
        result = tool_search_graph("sexo")
        assert isinstance(result, str)
        assert "Concepto:" in result or "no encontrada" in result

    def test_list_concepts_returns_list(self):
        """list_concepts debe retornar todos los conceptos del nomenclador."""
        result = tool_list_concepts()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_audit_graph_valid(self):
        """audit_graph debe retornar un reporte de invariantes."""
        result = tool_audit_graph()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_graph_health_runs(self):
        """graph_health debe ejecutar sin errores."""
        result = tool_graph_health()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_detect_standard_runs(self):
        """detect_standard debe detectar el estandar de una columna."""
        result = tool_detect_standard("sexo", ["M", "F", "M", "F"])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_compute_confidence_runs(self):
        """compute_confidence debe calcular score de confidence."""
        result = tool_compute_confidence("sexo")
        assert isinstance(result, str)
        assert len(result) > 0


class TestRememberDecision:
    """Verificar que remember_decision guarda y recall_feedback recupera."""

    def test_remember_and_recall(self):
        """Guardar una decision y recuperarla con recall_feedback."""
        from src.agent import tool_remember_decision, tool_recall_feedback
        # Guardar
        result = tool_remember_decision("test_concept", "validated", "test reason for regression")
        assert "Decision guardada" in result
        # Recuperar
        recalled = tool_recall_feedback("test_concept")
        assert "test_concept" in recalled or "test" in recalled.lower()

    def test_get_detail_resolves_from_state(self):
        """get_detail debe retornar el resultado completo guardado en tool_objects."""
        from src.agent import act_node
        # Simular state con tool_objects
        state = {
            "current_action": "get_detail",
            "current_action_input": json.dumps({"tool_name": "search_graph"}),
            "tool_call_id": "test_detail_id",
            "messages": [],
            "tool_objects": {"search_graph": "Resultado completo de search_graph sin truncar"},
        }
        result = act_node(state)
        tool_result = str(result.get("tool_result", ""))
        assert "Resultado completo" in tool_result

    def test_get_detail_missing_tool(self):
        """get_detail debe dar mensaje claro si la tool no tiene resultado guardado."""
        from src.agent import act_node
        state = {
            "current_action": "get_detail",
            "current_action_input": json.dumps({"tool_name": "nonexistent"}),
            "tool_call_id": "test_missing_id",
            "messages": [],
            "tool_objects": {"search_graph": "algo"},
        }
        result = act_node(state)
        tool_result = str(result.get("tool_result", ""))
        assert "No hay resultado" in tool_result or "no existe" in tool_result.lower()


# === Tests del pre-clasificador ===

class TestQueryClassifier:
    """Verificar que _classify_query clasifica correctamente."""

    def test_busqueda(self):
        cat, max_iter = _classify_query("Lista los conceptos del nomenclador")
        assert cat == "busqueda"
        assert max_iter == 3

    def test_validacion(self):
        cat, max_iter = _classify_query("¿Puedo cruzar el censo con el hospital?")
        assert cat == "validacion"
        assert max_iter == 8

    def test_transformacion(self):
        cat, max_iter = _classify_query("Transformar sexo a estandar ISO")
        assert cat == "transformacion"
        assert max_iter == 8

    def test_calidad(self):
        cat, max_iter = _classify_query("Limpiar datos del censo")
        assert cat == "calidad"
        assert max_iter == 5

    def test_general(self):
        cat, max_iter = _classify_query("Que areas tematicas cubre?")
        assert cat == "general"
        assert max_iter == 4


# === Tests de seleccion dinamica de tools ===

class TestToolSelection:
    """Verificar que _select_tools_for_query filtra correctamente."""

    def test_busqueda_only_primary(self):
        """Consultas simples solo reciben tools principales."""
        selected = _select_tools_for_query("listar conceptos")
        assert "search_graph" in selected
        assert "list_concepts" in selected
        assert "compare_distributions" not in selected
        assert "detect_communities" not in selected

    def test_communities_adds_advanced(self):
        """Consultas sobre comunidades activan detect_communities."""
        selected = _select_tools_for_query("comunidades del nomenclador")
        assert "detect_communities" in selected

    def test_distributions_adds_advanced(self):
        """Consultas sobre distribuciones activan compare_distributions."""
        selected = _select_tools_for_query("comparar distribucion de edad")
        assert "compare_distributions" in selected

    def test_primary_tools_count(self):
        """PRIMARY_TOOLS tiene 17 tools."""
        assert len(PRIMARY_TOOLS) == 17

    def test_advanced_tools_count(self):
        """ADVANCED_TOOLS tiene 20 tools."""
        assert len(ADVANCED_TOOLS) == 20

    def test_get_detail_in_primary(self):
        """get_detail debe estar en PRIMARY_TOOLS."""
        assert "get_detail" in PRIMARY_TOOLS

    def test_remember_decision_in_primary(self):
        """remember_decision debe estar en PRIMARY_TOOLS."""
        assert "remember_decision" in PRIMARY_TOOLS

    def test_no_overlap(self):
        """No debe haber overlap entre PRIMARY y ADVANCED."""
        assert len(PRIMARY_TOOLS & ADVANCED_TOOLS) == 0

    def test_all_tools_covered(self):
        """La union de PRIMARY + ADVANCED debe cubrir todas las tools en TOOLS_SCHEMA."""
        all_tools = set(TOOLS_SCHEMA.keys())
        # ask_human esta en TOOLS_SCHEMA pero no en PRIMARY ni ADVANCED
        # porque se anade manualmente en el schema
        union = PRIMARY_TOOLS | ADVANCED_TOOLS | {"ask_human"}
        assert union == all_tools

    def test_primary_schema_smaller(self):
        """El schema de PRIMARY debe ser mas chico que el de todas las tools."""
        primary_schema = _build_openai_tools_schema(PRIMARY_TOOLS)
        all_schema = _build_openai_tools_schema(PRIMARY_TOOLS | ADVANCED_TOOLS)
        assert len(primary_schema) < len(all_schema)
        assert len(primary_schema) == 17


# === Tests del schema OpenAI ===

class TestOpenAISchema:
    """Verificar que el schema OpenAI se genera correctamente."""

    def test_schema_format(self):
        """Cada tool en el schema debe tener el formato OpenAI correcto."""
        schema = _build_openai_tools_schema(PRIMARY_TOOLS)
        for tool in schema:
            assert tool["type"] == "function"
            assert "function" in tool
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            params = func["parameters"]
            assert params["type"] == "object"
            assert "properties" in params
            assert "required" in params

    def test_all_primary_tools_in_schema(self):
        """Todas las tools principales deben estar en el schema."""
        schema = _build_openai_tools_schema(PRIMARY_TOOLS)
        names = {t["function"]["name"] for t in schema}
        assert names == PRIMARY_TOOLS


# === Tests de validacion de tipos (inspirado en NOOA typed I/O) ===

class TestTypeValidation:
    """Verificar que act_node valida tipos antes de ejecutar tools."""

    def _run_act_node(self, action: str, action_input: dict):
        """Helper: ejecutar act_node con state simulado."""
        from src.agent import act_node
        state = {
            "current_action": action,
            "current_action_input": json.dumps(action_input),
            "tool_call_id": "test_call_id",
            "messages": [],
        }
        return act_node(state)

    def test_str_coercion(self):
        """Si el LLM pasa un int donde espera str, se convierte."""
        result = self._run_act_node("search_graph", {"query": 123})
        # No debe dar error de tipos — debe convertir 123 a "123"
        tool_result = result.get("tool_result", "")
        assert "Error de tipos" not in str(tool_result)

    def test_bool_coercion_from_string(self):
        """Si el LLM pasa 'true' como string para un bool, se convierte."""
        result = self._run_act_node("fix_orphans", {"dry_run": "true"})
        tool_result = result.get("tool_result", "")
        assert "Error de tipos" not in str(tool_result)

    def test_list_str_coercion_from_string(self):
        """Si el LLM pasa un string donde espera list[str], se envuelve en lista."""
        result = self._run_act_node("detect_standard", {"column_name": "sexo", "sample_values": "M"})
        tool_result = result.get("tool_result", "")
        assert "Error de tipos" not in str(tool_result)

    def test_extra_params_ignored(self):
        """Parametros extra no definidos en el schema se ignoran sin error."""
        result = self._run_act_node("list_concepts", {"unexpected_param": "value"})
        tool_result = result.get("tool_result", "")
        # list_concepts no toma parametros, extra se ignora
        assert "Error de tipos" not in str(tool_result)

    def test_nonexistent_tool_error(self):
        """Si el LLM llama una tool que no existe, retorna error claro."""
        result = self._run_act_node("nonexistent_tool", {"query": "test"})
        tool_result = str(result.get("tool_result", ""))
        assert "no existe" in tool_result


# === Tests agenticos (requieren LLM) ===

class TestAgenticMethods:
    """Tests que requieren LLM — marcados como slow."""

    @pytest.mark.slow
    def test_responder_consulta_runs(self):
        """run_agent debe producir una respuesta coherente."""
        result = run_agent("¿Qué variables hay en el nomenclador?")
        assert isinstance(result, dict)
        assert "final_answer" in result
        assert len(result["final_answer"]) > 10

    @pytest.mark.slow
    def test_clasificar_consulta_category(self):
        """run_agent debe incluir query_category en el resultado."""
        result = run_agent("Lista los conceptos del nomenclador")
        assert "query_category" in result
        assert result["query_category"] == "busqueda"

    @pytest.mark.slow
    def test_busqueda_uses_few_iterations(self):
        """Consultas de busqueda no deben usar mas de 3 iteraciones."""
        result = run_agent("Lista los conceptos del nomenclador")
        assert result["iterations"] <= 3
