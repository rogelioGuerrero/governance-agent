"""Test client for the MCP nomenclador server."""
import asyncio
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


async def test():
    params = StdioServerParameters(
        command="uv",
        args=["run", "python", "-m", "src.mcp_server"],
        cwd=r"D:\proyectoBolt\governance-agent",
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List tools
            tools = await session.list_tools()
            print("=== Tools disponibles ===")
            for t in tools.tools:
                print(f"  {t.name}: {t.description[:80] if t.description else ''}")

            # Test list_concepts
            print("\n=== list_concepts ===")
            result = await session.call_tool("list_concepts", {})
            if result.content:
                print(result.content[0].text[:600])

            # Test search_variable
            print("\n=== search_variable('sexo') ===")
            result = await session.call_tool("search_variable", {"name": "sexo"})
            if result.content:
                print(result.content[0].text[:600])

            # Test check_interoperability
            print("\n=== check_interoperability('sample_censo', 'sample_hospital') ===")
            result = await session.call_tool("check_interoperability", {
                "source_db": "sample_censo",
                "target_db": "sample_hospital",
            })
            if result.content:
                print(result.content[0].text[:800])

            # Test validate_field
            print("\n=== validate_field('sexo', ['M', 'F', 'H']) ===")
            result = await session.call_tool("validate_field", {
                "column_name": "sexo",
                "sample_values": ["M", "F", "H"],
            })
            if result.content:
                print(result.content[0].text[:600])

            # Test get_classifier
            print("\n=== get_classifier('ISO_5218') ===")
            result = await session.call_tool("get_classifier", {
                "standard_id": "ISO_5218",
            })
            if result.content:
                print(result.content[0].text[:400])


if __name__ == "__main__":
    asyncio.run(test())
