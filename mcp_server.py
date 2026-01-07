#!/usr/bin/env python3
"""
MCP Server for Bat Jester Model Training Project

Provides tools for accessing data preprocessing scripts and model training information.
"""

from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


# Initialize the MCP server
app = Server("bat-jester-model-training")

PROJECT_ROOT = Path(__file__).parent


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="list_data_preprocessing_scripts",
            description="List all data preprocessing scripts available in the project",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="get_project_structure",
            description="Get an overview of the project structure and key directories",
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""

    if name == "list_data_preprocessing_scripts":
        preprocessing_dir = PROJECT_ROOT / "data_preprocessing"
        scripts = []

        if preprocessing_dir.exists():
            for file in sorted(preprocessing_dir.glob("*.py")):
                if not file.name.startswith("__"):
                    scripts.append(f"- {file.name}")

        result = "Data Preprocessing Scripts:\n" + "\n".join(scripts)
        return [TextContent(type="text", text=result)]

    elif name == "get_project_structure":
        structure = f"""
Bat Jester Model Training Project Structure:

Root Directory: {PROJECT_ROOT}

Key Directories:
- data_preprocessing/: Scripts for data preprocessing pipeline
- docs/: Project documentation
- previous_approach/: Previous model training notebooks
- data/: Data directory (if exists)

Key Files:
- pyproject.toml: Python project configuration
- flake.nix: Nix development environment
- generate_production_test_data.py: Production test data generation
"""
        return [TextContent(type="text", text=structure)]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
