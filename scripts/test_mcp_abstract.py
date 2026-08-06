"""Test del MCP server abstracto con ambos packs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.mcp_server_abstract import load_pack_by_name, create_mcp_server
from src.core.standards import STANDARDS, register_pack_standards

# Test 1: Pack VRP
print("=== Pack VRP ===")
vrp_pack = load_pack_by_name("vrp")
print(f"Name: {vrp_pack.name}")
print(f"Fields: {len(vrp_pack.schema_fields)}")
print(f"Rules: {len(vrp_pack.semantic_rules)}")
print(f"Validators: {len(vrp_pack.custom_validators)}")
print(f"Solver contract: {vrp_pack.solver_contract}")

# Crear MCP server
vrp_mcp = create_mcp_server(vrp_pack)
print(f"MCP server: {vrp_mcp.name}")

# Test 2: Pack Salud
print("\n=== Pack Salud ===")
salud_pack = load_pack_by_name("salud")
print(f"Name: {salud_pack.name}")
print(f"Rules: {len(salud_pack.semantic_rules)}")
print(f"Mappings: {len(salud_pack.inference_mappings)}")
print(f"Metadata MOA: {salud_pack.metadata.get('moa_agents')}")

# Registrar estándares del pack
register_pack_standards(salud_pack)
print(f"Standards registered: {list(STANDARDS.keys())}")

# Crear MCP server
salud_mcp = create_mcp_server(salud_pack)
print(f"MCP server: {salud_mcp.name}")

print("\n✓ Ambos packs cargan correctamente en MCP server abstracto")
