# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM-driven agent that converts natural language building descriptions into 3D zone geometry (JSON for Rhino/Grasshopper) with optional EnergyPlus energy simulation. Also includes a parametric L-shaped building generator with interactive tuning UI. Built with LangChain/LangGraph, Pydantic v2, Python >=3.12. Uses `uv` for dependency management.

## Common Commands

```bash
uv sync                                    # Install dependencies
cp .env.example .env                       # Configure LLM_PROVIDER, LLM_API_KEY, LLM_MODEL_NAME
python main.py -d "A house with 3 rooms"   # CLI batch mode (LLM pipeline)
python main.py -d "..." --simulate         # CLI with EnergyPlus simulation
streamlit run app.py                       # Interactive chat UI with 3D preview
streamlit run parametric_l_app.py          # Parametric L-shape tuning UI with energy analysis
python scripts/generate_l_gradient.py      # Generate parametric L-shape model (standalone)
bash scripts/setup_plugin.sh               # First-time EnergyPlus plugin setup
```

## Architecture

### LangGraph Pipeline (`src/agent/graph.py`)

```
START → intake → zone_agent → export → [energyplus] → END
```

State flows through `AgentState` (TypedDict in `src/agent/state.py`): `messages`, `building_name`, `zones`, `building_description`, `output_path`, optional `idf_output_path`/`simulation_result`.

### Nodes (`src/agent/nodes/`)

- **intake.py** — LLM parses natural language into a structured building summary
- **zone.py** — ReAct agent invokes zone tools to create/update/delete zones
- **export.py** — Assembles `BuildingModel` and writes JSON
- **energyplus.py** — (Optional) MCP client that converts zones to IDF and runs EnergyPlus simulation

### Tools (`src/agent/tools/`)

`zone_tools.py` defines 5 LangChain tools: `create_zone`, `list_zones`, `update_zone`, `delete_zone`, `export_json`. State is held in a module-level store managed via `reset_store`/`set_building_name`/`get_zones`.

### Data Models (`src/models/zone.py`)

Pydantic models: `Point3D`, `Dimensions`, `Zone`, `BuildingModel`.

### LLM Configuration (`src/config.py` + `src/agent/llm.py`)

`LLMConfig` reads from env vars. `init_chat_model()` uses `provider:model_name` format (e.g., `anthropic:claude-sonnet-4-6`).

## EnergyPlus Plugin

Git submodule at `plugins/energyplus_agent/`. Communicates via MCP protocol (langchain-mcp-adapters). Requires EnergyPlus 25.1.0+ installed locally. Setup: `bash scripts/setup_plugin.sh`.

## Two Generation Paths

The project supports two complementary workflows for creating building zone geometry, both producing the same `BuildingModel` JSON format:

1. **LLM Pipeline** — Natural language → `intake` → `zone_agent` (ReAct) → `export` → JSON. Used by `main.py` and `app.py`.
2. **Parametric Generation** — Mathematical parameters → `generate_l_gradient()` → JSON. Used by `scripts/generate_l_gradient.py` and `parametric_l_app.py`.

Both paths can feed into the EnergyPlus simulation node.

## Entry Points

- **`main.py`** — CLI with argparse; calls `run_agent()` async
- **`app.py`** — Streamlit multi-turn chat UI with Plotly 3D zone preview
- **`parametric_l_app.py`** — Streamlit parametric tuning UI; two tabs: geometric preview (sliders for L-shape params) and energy analysis (reads `eplustbl.csv`, maps energy data to zones)
- **`scripts/generate_l_gradient.py`** — Standalone script for parametric L-shaped building generation

## Parametric L-Shape Generator (`scripts/generate_l_gradient.py`)

Generates a gradient L-shaped office building that transitions from fragmented lower floors to a solid L-form at the top. Core parameters: `floors`, `arm_width`, `horizontal_length`, `vertical_length`, `scatter_gap`, `merge_power`, `bridge_start_floor`, `top_solid_floors`. Output is a `BuildingModel` JSON with ~76 zones for default settings.

## Output

JSON files written to `output/`: `{ building_name, zones: [{ name, origin: {x,y,z}, dimensions: {length,width,height} }] }`. IDF files generated when `--simulate` is used.
