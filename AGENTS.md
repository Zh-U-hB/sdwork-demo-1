# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

LLM-driven agent that converts natural language building descriptions into 3D zone geometry (JSON for Rhino/Grasshopper) with optional EnergyPlus energy simulation. Also includes a parametric L-shaped building generator with interactive tuning UI and a genetic algorithm optimizer. Built with LangChain/LangGraph, Pydantic v2, Python >=3.12. Uses `uv` for dependency management.

## Common Commands

```bash
uv sync                                    # Install dependencies
cp .env.example .env                       # Configure LLM_PROVIDER, LLM_API_KEY, LLM_MODEL_NAME
python main.py -d "A house with 3 rooms"   # CLI batch mode (LLM pipeline)
python main.py -d "..." --simulate         # CLI with EnergyPlus simulation
streamlit run app.py                       # Interactive chat UI with 3D preview
streamlit run parametric_l_app.py          # Parametric L-shape tuning UI with energy analysis
streamlit run ga_optimizer_app.py          # GA energy optimizer UI
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

`LLMConfig` reads from env vars. `init_chat_model()` uses `provider:model_name` format (e.g., `anthropic:Codex-sonnet-4-6`).

## Two Simulation Paths

Both produce the same end result (eplustbl.csv with energy data), but differ in speed and reliability:

### Direct Path (preferred — fast, deterministic, no LLM)

```
model_dict → scripts/idf_converter.convert_and_run() → IDF file → energyplus subprocess → eplustbl.csv
```

Uses **idfpy** (type-safe Pydantic models for IDF objects, no IDD file needed). Entry point: `scripts/idf_converter.py`. Default parameters in `scripts/idf_defaults.py`. Called by GA optimizer and `run_ep_simulation_direct()` in `ep_sim_utils.py`.

### MCP Path (original — slow, LLM-driven)

```
model_dict → MCP Agent (LLM) → 13 IDF construction steps → energyplus → eplustbl.csv
```

Uses the EnergyPlus Agent plugin at `plugins/energyplus_agent/`. Each simulation takes 2-5 minutes due to LLM step-by-step IDF construction. Kept for backward compatibility via `run_ep_simulation()` in `ep_sim_utils.py`.

### When to use which

- GA optimizer (`ga_core.py`) and any batch evaluation: **direct path only** (speed critical)
- `main.py --simulate` and `app.py`: still uses **MCP path** (can be migrated later)
- `parametric_l_app.py`: reads pre-existing `eplustbl.csv`, no simulation path preference

## Direct IDF Converter (`scripts/idf_converter.py`)

Converts BuildingModel JSON → EnergyPlus IDF using idfpy. Process:

1. Filter zones (skip height < 1.0m markers)
2. Sanitize zone names to ASCII (`Zone_01`, `Zone_02`, ...)
3. Detect shared walls between adjacent zones (X/Y axis, tolerance 0.01m)
4. Build IDF: Version → SimulationControl → Timestep → RunPeriod → GlobalGeometryRules → Building → Location → Materials → Constructions → Schedules → Thermostat → Zones + 6 surfaces each → Patch shared walls → People + Lights + IdealLoadsAirSystem per zone
5. Save IDF → run `energyplus` subprocess

Vertex ordering: UpperLeftCorner + CounterClockwise + World coordinate system (matches GlobalGeometryRules).

Defaults are fully configurable via `scripts/idf_defaults.py` dataclasses. Override example:
```python
defaults = make_default_settings()
defaults.location.latitude = 39.93
defaults.window.wwr = 0.4
result = convert_and_run(model, defaults=defaults)
```

**Note**: idfpy is generated from EnergyPlus 26.1 schema but the system runs EP 25.1. Some EP 26.1 new fields (e.g., `OutputControlTableStyle.format_numeric_values`) must be skipped in the converter.

## Three Generation Paths

1. **LLM Pipeline** — Natural language → `intake` → `zone_agent` (ReAct) → `export` → JSON. Used by `main.py` and `app.py`.
2. **Parametric Generation** — Mathematical parameters → `generate_l_gradient()` → JSON. Used by `scripts/generate_l_gradient.py` and `parametric_l_app.py`.
3. **GA Optimization** — Genetic algorithm searches parameter space → `generate_l_gradient()` → direct IDF → EP simulation → minimize EUI. Used by `ga_optimizer_app.py`.

All three produce the same `BuildingModel` JSON format and can feed into either simulation path.

## GA Optimizer (`scripts/ga_core.py` + `ga_optimizer_app.py`)

Minimizes EUI (MJ/m²) by searching 12 parameters of `generate_l_gradient()`: floors, lobby_height, floor_height, base_x/y, arm_width, horizontal/vertical_length, scatter_gap, min_fragment_scale, merge_power, top_solid_floors. Uses tournament selection + BLX-α crossover + Gaussian mutation with elitism. Results cached to `output/ga_cache.json`, checkpoints to `output/ga_checkpoint.json`. Constraint: building height < 50m.

## Entry Points

- **`main.py`** — CLI with argparse; calls `run_agent()` async
- **`app.py`** — Streamlit multi-turn chat UI with Plotly 3D zone preview
- **`parametric_l_app.py`** — Streamlit parametric tuning UI; geometric preview tab + energy analysis tab
- **`ga_optimizer_app.py`** — Streamlit GA optimization UI; evolution progress + best solution + population comparison
- **`scripts/generate_l_gradient.py`** — Standalone parametric L-shaped building generator

## EnergyPlus Plugin

Git submodule at `plugins/energyplus_agent/`. Communicates via MCP protocol (langchain-mcp-adapters). Requires EnergyPlus 25.1.0+ installed locally (`/usr/local/EnergyPlus-25-1-0/`). Setup: `bash scripts/setup_plugin.sh`. The plugin's converter modules (`src/converters/`) serve as reference for the direct converter's field names and patterns.

## Output

JSON files written to `output/`: `{ building_name, zones: [{ name, origin: {x,y,z}, dimensions: {length,width,height} }] }`. Simulation results: `output/direct_energyplus/` or `output/ga_*` containing IDF files and `eplustbl.csv`.

## Key Dependencies

- **idfpy** — Pydantic v2 models for 859 EnergyPlus IDF object types, used by direct converter
- **eppy** — Legacy IDF manipulation, used by EP Agent plugin (not the direct converter)
- **langchain/langgraph** — LLM orchestration for the natural language pipeline
- **pydantic >=2** — Data validation throughout
- **streamlit + plotly** — All UI pages

## IDF Library Preference

Use **idfpy** (not eppy) for any new IDF manipulation code. idfpy provides type-safe Pydantic models, requires no IDD file, and has built-in simulation execution. See `scripts/idf_converter.py` for usage patterns.
