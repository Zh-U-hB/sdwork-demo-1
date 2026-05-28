"""Build HybridConfig from manual_test_app sidebar state."""

from __future__ import annotations

from typing import Any

from ga_llm_hybrid.config import (
    EnergyPlusConfig,
    GAConfig,
    HybridConfig,
    LLMConfigSection,
    ObjectiveDef,
    ParamDef,
    ProjectConfig,
)


def build_config_from_manual_params(
    current_params: dict[str, Any],
    *,
    population_size: int = 10,
    max_generations: int = 8,
    max_rounds: int = 2,
    debug_mode: bool = False,
    llm_enabled: bool = True,
    morris_enabled: bool = False,
    partition_enabled: bool = True,
    perimeter_depth: float = 4.0,
    seed: int = 42,
    include_ep_tunables: bool = True,
) -> HybridConfig:
    """Create a :class:`HybridConfig` using sidebar geometry as fixed + tunable subset."""
    tunable_names = [
        "boundary_shift",
        "lobby_height",
        "floor_height",
        "low_aspect_ratio",
    ]
    if include_ep_tunables:
        tunable_names.extend(["window_wwr", "lights_watts_per_floor_area"])

    param_defs: list[ParamDef] = [
        ParamDef(
            name="boundary_shift",
            type="continuous",
            range=[
                max(0.0, float(current_params.get("boundary_shift", 40.0)) - 20.0),
                min(200.0, float(current_params.get("boundary_shift", 40.0)) + 20.0),
            ],
        ),
        ParamDef(name="lobby_height", type="continuous", range=[3.0, 9.0]),
        ParamDef(name="floor_height", type="continuous", range=[3.0, 5.0]),
        ParamDef(name="low_aspect_ratio", type="continuous", range=[0.5, 2.0]),
    ]
    if include_ep_tunables:
        param_defs.extend([
            ParamDef(name="window_wwr", type="continuous", range=[0.0, 0.6]),
            ParamDef(name="lights_watts_per_floor_area", type="continuous", range=[6.0, 18.0]),
        ])

    fixed = {k: v for k, v in current_params.items() if k not in tunable_names}
    fixed.setdefault("building_name", "GA_LLM_Hybrid_UI")

    if debug_mode:
        population_size = min(population_size, 5)
        max_generations = min(max_generations, 3)
        max_rounds = min(max_rounds, 1)

    return HybridConfig(
        project=ProjectConfig(
            name="manual_test_hybrid",
            seed=seed,
            max_rounds=max_rounds,
            min_rounds=1,
            debug_mode=debug_mode,
            morris_enabled=morris_enabled and not debug_mode,
            morris_trajectories=6,
        ),
        ga=GAConfig(
            population_size=population_size,
            max_generations=max_generations,
            init_mode="llm_guided" if llm_enabled else "random",
            llm_seed_fraction=0.2,
            global_explore_fraction=0.2,
        ),
        energyplus=EnergyPlusConfig(
            backend="arch_model",
            partition_enabled=partition_enabled,
            perimeter_depth=perimeter_depth,
            fixed_geometry=fixed,
        ),
        llm=LLMConfigSection(enabled=llm_enabled, analysis_frequency=1),
        parameters=param_defs,
        objectives=[
            ObjectiveDef(name="eui_mj_m2", weight=1.0, direction="minimize", output_key="eui_mj_m2"),
        ],
        building_context={
            "climate": "夏热冬冷（深圳 EPW）",
            "building_type": "办公建筑",
            "objectives": "最小化 EUI (MJ/m²)",
        },
    )
