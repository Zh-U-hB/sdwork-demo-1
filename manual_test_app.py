"""Manual test dashboard for geometry + EnergyPlus + optimizers.

Includes:
- Manual parameter editing + 3D preview
- LLM optimization (process + best result + charts + 3D zone energy)
- Genetic algorithm optimization (process + best result)
- GA + LLM hybrid optimization (one-click)
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import plotly.graph_objects as go
import streamlit as st

from scripts.ep_sim_utils import (
    MASS_HEIGHT_THRESHOLD,
    available_weather_cities,
    model_energy_map,
    read_eplustbl,
    run_ep_simulation_direct,
)
from scripts.generate_20260528 import generate_20260528, gross_area
from scripts.ga_core_20260528 import (
    CheckpointState,
    DEFAULT_GENES_20260528,
    GAConfig,
    load_checkpoint,
    run_ga,
    save_checkpoint,
)
from scripts.ga_defaults_20260528 import build_ga_fixed_params
from scripts.facade_params import FACADE_TUNABLE, default_facade_params
from scripts.llm_optimizer import (
    ALL_TUNABLE,
    IterationRecord,
    best_record,
    run_llm_optimization,
)
from scripts.zone_partition import partition_model_by_floor
from scripts.vis_utils import (
    model_metrics,
    render_end_use_chart,
    render_end_use_pie,
    render_comfort_weather_chart,
    render_monthly_eui_chart,
    render_model,
    render_zone_energy_chart,
    save_json,
)
from ga_llm_hybrid.ui_config import build_config_from_manual_params
from ga_llm_hybrid.orchestrator import HybridOrchestrator


st.set_page_config(page_title="Manual Test Dashboard", layout="wide")
st.title("手动测试页面 — 建模 / 模拟 / LLM / GA / GA+LLM 混合优化")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "mt_last_result_dir" not in st.session_state:
    st.session_state.mt_last_result_dir = None
if "mt_llm_history" not in st.session_state:
    st.session_state.mt_llm_history: list[IterationRecord] = []
if "mt_ga_history" not in st.session_state:
    st.session_state.mt_ga_history = []
if "mt_ga_best_params" not in st.session_state:
    st.session_state.mt_ga_best_params = None
if "mt_ga_best_fitness" not in st.session_state:
    st.session_state.mt_ga_best_fitness = None
if "mt_ga_best_model" not in st.session_state:
    st.session_state.mt_ga_best_model = None
if "mt_ga_best_result_dir" not in st.session_state:
    st.session_state.mt_ga_best_result_dir = None
if "mt_ga_total_evals" not in st.session_state:
    st.session_state.mt_ga_total_evals = 0
if "mt_ga_running" not in st.session_state:
    st.session_state.mt_ga_running = False
if "mt_ga_run_dir" not in st.session_state:
    st.session_state.mt_ga_run_dir = None
if "mt_hybrid_running" not in st.session_state:
    st.session_state.mt_hybrid_running = False
if "mt_hybrid_report" not in st.session_state:
    st.session_state.mt_hybrid_report = None
if "mt_hybrid_output_dir" not in st.session_state:
    st.session_state.mt_hybrid_output_dir = None


# ---------------------------------------------------------------------------
# Sidebar: global controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("参数设置")

    st.divider()
    st.subheader("3D 显示")
    site_size = st.number_input("site_size (m)", value=100.0, min_value=40.0, max_value=300.0, step=10.0)
    show_edges = st.checkbox("显示边线", value=True)
    opacity = st.slider("体量透明度", 0.15, 1.0, 0.60, 0.05)
    energy_metric = st.selectbox(
        "三维着色指标",
        options=[
            ("total_gj", "总能耗"),
            ("heating_gj", "采暖"),
            ("cooling_gj", "制冷"),
            ("lighting_gj", "照明"),
        ],
        format_func=lambda x: x[1],
        index=0,
    )[0]

    st.divider()
    st.subheader("输出")
    output_path = st.text_input("保存 JSON 路径", value="output/manual_test_model.json")

    st.divider()
    st.subheader("几何参数")
    bname = st.text_input("building_name", value="Manual_20260528", key="sb_20260528_name")
    total_area = st.slider("total_area (m²)", 6000.0, 16000.0, 10000.0, 100.0)
    lobby_height = st.slider("lobby_height (m)", 3.0, 9.0, 6.0, 0.1)
    floor_height = st.slider("floor_height (m)", 3.0, 5.0, 4.0, 0.1)

    st.caption("Setbacks (m)")
    setback_south = st.slider("setback_south", 0.0, 30.0, 15.0, 0.5)
    setback_west = st.slider("setback_west", 0.0, 30.0, 15.0, 0.5)
    setback_north = st.slider("setback_north", 0.0, 30.0, 10.0, 0.5)
    setback_east = st.slider("setback_east", 0.0, 30.0, 10.0, 0.5)

    st.caption("Shared building controls")
    shared_geometry_controls = st.checkbox("统一调节三栋建筑 aspect / angle / distance", value=False)
    if shared_geometry_controls:
        st.caption("统一模式下：低/中建筑使用该长宽比，高层建筑使用倒数，形成相反 x/y 比例。")
        shared_aspect_ratio = st.slider("shared_aspect_ratio", 0.5, 2.0, 1.0, 0.05)
        shared_offset_angle = st.slider("shared_offset_angle", 0.0, 360.0, 180.0, 1.0)
        shared_offset_distance = st.slider("shared_offset_distance", 0.0, 10.0, 2.0, 0.1)
        low_aspect_ratio = mid_aspect_ratio = shared_aspect_ratio
        high_aspect_ratio = 1.0 / shared_aspect_ratio
        low_offset_angle = mid_offset_angle = high_offset_angle = shared_offset_angle
        low_offset_distance = mid_offset_distance = high_offset_distance = shared_offset_distance
    else:
        shared_aspect_ratio = None
        shared_offset_angle = None
        shared_offset_distance = None

        st.caption("Footprint aspect ratios")
        low_aspect_ratio = st.slider("low_aspect_ratio", 0.5, 2.0, 1.0, 0.05)
        mid_aspect_ratio = st.slider("mid_aspect_ratio", 0.5, 2.0, 1.0, 0.05)
        high_aspect_ratio = st.slider("high_aspect_ratio", 0.5, 2.0, 1.0, 0.05)

    st.caption("Offset grouping")
    boundary_shift = st.slider("boundary_shift", 0.0, 200.0, 40.0, 1.0)
    group_size = st.slider("group_size", 1, 4, 2)

    if not shared_geometry_controls:
        st.caption("Offset angles (deg)")
        low_offset_angle = st.slider("low_offset_angle", 0.0, 360.0, 45.0, 1.0)
        mid_offset_angle = st.slider("mid_offset_angle", 0.0, 360.0, 180.0, 1.0)
        high_offset_angle = st.slider("high_offset_angle", 0.0, 360.0, 315.0, 1.0)

        st.caption("Offset distances (m)")
        low_offset_distance = st.slider("low_offset_distance", 0.0, 10.0, 2.0, 0.1)
        mid_offset_distance = st.slider("mid_offset_distance", 0.0, 10.0, 2.0, 0.1)
        high_offset_distance = st.slider("high_offset_distance", 0.0, 10.0, 2.0, 0.1)

    min_support_overlap_ratio = st.slider("min_support_overlap_ratio", 0.1, 1.0, 0.5, 0.05)
    add_aerial_platforms = st.checkbox("add_aerial_platforms", value=True)
    platform_edge_walk_distance = st.slider("platform_edge_walk_distance", 1.0, 12.0, 5.0, 0.5)
    add_open_space_markers = st.checkbox("add_open_space_markers", value=True)

    st.divider()
    st.subheader("立面窗户与遮阳")
    _facade_defaults = default_facade_params()
    window_enabled = st.checkbox("启用立面窗户 (几何模块窗)", value=_facade_defaults["window_enabled"])
    window_wwr = st.slider("window_wwr (几何窗墙比)", 0.0, 0.8, float(_facade_defaults["window_wwr"]), 0.05)
    window_module = st.slider("window_module (m)", 0.5, 3.0, float(_facade_defaults["window_module"]), 0.1)
    shading_enabled = st.checkbox("启用水平遮阳 overhang", value=_facade_defaults["shading_enabled"])
    shading_depth = st.slider("shading_depth (m)", 0.0, 2.0, float(_facade_defaults["shading_depth"]), 0.05)
    st.caption("运行 EnergyPlus 时，全局 IDF 窗墙比会与几何 window_wwr 同步。")

    st.divider()
    st.subheader("模拟设置")
    weather_city_paths = available_weather_cities()
    weather_city = st.selectbox(
        "Weather city / EPW",
        options=list(weather_city_paths.keys()),
        index=0,
        format_func=lambda city: city,
    )
    weather_file = weather_city_paths[weather_city]
    st.caption(f"EPW: `{Path(weather_file).name}`")
    show_extra_result_charts = st.checkbox("模拟结果包含 EUI 月度图和气象/舒适图", value=True)

    st.divider()
    st.subheader("分区（Partition）")
    partition_enabled = st.checkbox("启用分区后再模拟（推荐）", value=True)
    perimeter_depth = st.slider("perimeter_depth (m)", 1.0, 8.0, 4.0, 0.5)


def _render_sim_section(model: dict, building_name: str, *, key_prefix: str) -> tuple[str | None, dict]:
    """Render simulation runner + charts. Returns (result_dir, sim_data)."""
    # The simulation model should match the EnergyPlus run geometry.
    sim_model = model
    if partition_enabled:
        try:
            sim_model = partition_model_by_floor(
                model,
                perimeter_depth=float(perimeter_depth),
                lobby_height=float(current_params.get("lobby_height", 6.0)),
                floor_height=float(current_params.get("floor_height", 4.0)),
            )
        except Exception as e:
            st.warning(f"分区失败，将使用原始模型进行模拟与可视化：{e}")
            sim_model = model

    cols = st.columns([1, 2])
    with cols[0]:
        st.subheader("模拟")
        st.caption("走 direct 路径：JSON → IDF → EnergyPlus（无 MCP/LLM）。")
        st.caption(f"当前城市：{weather_city} | `{Path(weather_file).name}`")
        if st.button("▶ 运行 EnergyPlus（Direct）", type="primary", use_container_width=True, key=f"{key_prefix}_run_ep"):
            from scripts.facade_params import make_ep_defaults_for_geometry

            ep_defaults = make_ep_defaults_for_geometry(current_params)
            with st.spinner("正在运行 EnergyPlus…"):
                result_dir = run_ep_simulation_direct(
                    sim_model,
                    building_name=building_name,
                    defaults=ep_defaults,
                    weather_file=weather_file,
                )
            if result_dir:
                st.session_state.mt_last_result_dir = result_dir
                st.success(f"完成：{result_dir}")
                st.rerun()
            else:
                st.error("运行失败：未找到结果目录（eplustbl.csv）。")

        last = st.session_state.mt_last_result_dir
        manual_dir = st.text_input("结果目录（可手动覆盖）", value=last or "", key=f"{key_prefix}_result_dir")
        result_dir = manual_dir.strip() or None
        if result_dir:
            st.caption(f"使用结果：`{result_dir}`")

    sim_data = read_eplustbl(result_dir) if result_dir else {"exists": False}
    mapped_energy = model_energy_map(sim_model, sim_data) if sim_data.get("exists") else {}

    with cols[1]:
        st.subheader("结果概览")
        if not sim_data.get("exists"):
            st.info("未加载到 eplustbl.csv。你可以先运行模拟，或在左侧输入一个结果目录。")
        else:
            total_site = sim_data["site_energy"].get("Total Site Energy", 0.0)
            cond_area = sim_data["building_area"].get("Net Conditioned Building Area", 0.0)
            eui = total_site * 1000 / cond_area if cond_area > 0 else 0.0
            mcols = st.columns(4)
            mcols[0].metric("EUI", f"{eui:.1f} MJ/m²")
            mcols[1].metric("Total Site", f"{total_site:.2f} GJ")
            mcols[2].metric("Conditioned Area", f"{cond_area:.0f} m²")
            mcols[3].metric("解析到的 Zone 能耗", len(sim_data.get("zone_energy", {})))

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    render_end_use_chart(sim_data.get("end_uses", [])),
                    use_container_width=True,
                    key=f"{key_prefix}_end_use_bar",
                )
            with c2:
                st.plotly_chart(
                    render_end_use_pie(sim_data.get("end_uses", [])),
                    use_container_width=True,
                    key=f"{key_prefix}_end_use_pie",
                )

            if show_extra_result_charts:
                charts = sim_data.get("charts", {})
                st.subheader("附加结果图表")
                ec1, ec2 = st.columns(2)
                with ec1:
                    st.plotly_chart(
                        render_monthly_eui_chart(charts.get("monthly_eui", [])),
                        use_container_width=True,
                        key=f"{key_prefix}_monthly_eui",
                    )
                with ec2:
                    st.plotly_chart(
                        render_comfort_weather_chart(charts.get("comfort", {})),
                        use_container_width=True,
                        key=f"{key_prefix}_comfort_weather",
                    )

            if mapped_energy:
                st.subheader("模型体块分区能耗（stacked）")
                mass_zones = [
                    z for z in sim_model["zones"]
                    if z["dimensions"]["height"] > MASS_HEIGHT_THRESHOLD and z.get("category") != "open_space_reference"
                ]
                st.plotly_chart(
                    render_zone_energy_chart(
                        [{
                            "model_zone": z["name"],
                            "heating_gj": mapped_energy.get(z["name"], {}).get("heating_gj", 0.0),
                            "cooling_gj": mapped_energy.get(z["name"], {}).get("cooling_gj", 0.0),
                            "lighting_gj": mapped_energy.get(z["name"], {}).get("lighting_gj", 0.0),
                            "equipment_gj": mapped_energy.get(z["name"], {}).get("equipment_gj", 0.0),
                        } for z in mass_zones[:60]]
                    ),
                    use_container_width=True,
                    key=f"{key_prefix}_zone_energy_stack",
                )

    st.subheader("3D 分区能耗")
    st.plotly_chart(
        render_model(
            sim_model,
            site_size=site_size,
            show_edges=show_edges,
            opacity=opacity,
            zone_energy=(mapped_energy or None),
            energy_metric=energy_metric,
        ),
        use_container_width=True,
        key=f"{key_prefix}_model_3d_energy",
    )

    return result_dir, sim_data


# ---------------------------------------------------------------------------
# Build current model from sidebar parameters (generate_20260528)
# ---------------------------------------------------------------------------

current_params = dict(
    building_name=bname,
    site_size=float(site_size),
    total_area=float(total_area),
    lobby_height=float(lobby_height),
    floor_height=float(floor_height),
    setback_south=float(setback_south),
    setback_west=float(setback_west),
    setback_north=float(setback_north),
    setback_east=float(setback_east),
    shared_aspect_ratio=float(shared_aspect_ratio) if shared_aspect_ratio is not None else None,
    low_aspect_ratio=float(low_aspect_ratio),
    mid_aspect_ratio=float(mid_aspect_ratio),
    high_aspect_ratio=float(high_aspect_ratio),
    boundary_shift=float(boundary_shift),
    group_size=int(group_size),
    shared_offset_angle=float(shared_offset_angle) if shared_offset_angle is not None else None,
    low_offset_angle=float(low_offset_angle),
    mid_offset_angle=float(mid_offset_angle),
    high_offset_angle=float(high_offset_angle),
    shared_offset_distance=float(shared_offset_distance) if shared_offset_distance is not None else None,
    low_offset_distance=float(low_offset_distance),
    mid_offset_distance=float(mid_offset_distance),
    high_offset_distance=float(high_offset_distance),
    min_support_overlap_ratio=float(min_support_overlap_ratio),
    add_aerial_platforms=bool(add_aerial_platforms),
    platform_edge_walk_distance=float(platform_edge_walk_distance),
    add_open_space_markers=bool(add_open_space_markers),
    window_enabled=bool(window_enabled),
    window_wwr=float(window_wwr),
    window_module=float(window_module),
    shading_enabled=bool(shading_enabled),
    shading_depth=float(shading_depth),
)
current_model = generate_20260528(**current_params)
current_metrics = model_metrics(current_model, gross_area_fn=gross_area)

# For frontend visualization: optionally display the partitioned model
display_model = current_model
if partition_enabled:
    try:
        display_model = partition_model_by_floor(
            current_model,
            perimeter_depth=float(perimeter_depth),
            lobby_height=float(current_params.get("lobby_height", 6.0)),
            floor_height=float(current_params.get("floor_height", 4.0)),
        )
    except Exception as e:
        st.warning(f"分区失败，已回退到原始模型显示：{e}")
        display_model = current_model


# ---------------------------------------------------------------------------
# Main area: always show 3D model
# ---------------------------------------------------------------------------

st.caption(
    f"面积≈ {current_metrics['area']:.1f} m² | 高度≈ {current_metrics['height']:.1f} m | "
    f"zones={current_metrics['zone_count']}（mass={current_metrics['mass_zone_count']}）"
)
st.plotly_chart(
    render_model(display_model, site_size, show_edges, opacity),
    use_container_width=True,
    key="mt_main_model_3d",
)

col_exp, col_params = st.columns([1, 1])
with col_exp:
    if st.button("保存 JSON", use_container_width=True, key="mt_save_main"):
        path = save_json(display_model, output_path)
        st.success(f"已保存到 {path}")
with col_params:
    st.download_button(
        "下载 JSON",
        data=json.dumps(display_model, indent=2, ensure_ascii=False),
        file_name=Path(output_path).name or "model.json",
        mime="application/json",
        use_container_width=True,
    )


st.divider()


# ---------------------------------------------------------------------------
# Tabs: Simulation / LLM / GA
# ---------------------------------------------------------------------------

tab_sim, tab_llm, tab_ga, tab_hybrid = st.tabs(
    ["模拟与结果", "LLM 优化", "遗传算法优化 (GA)", "GA+LLM 混合优化"]
)

with tab_sim:
    # simulation runner will re-apply partition when enabled; keep input as raw model
    _render_sim_section(current_model, current_params["building_name"], key_prefix="mt_current")


# ---------------------------------------------------------------------------
# Tab 2: LLM optimization
# ---------------------------------------------------------------------------

with tab_llm:
    st.subheader("LLM 优化（generate_20260528）")
    st.caption("LLM 根据 eplustbl.csv 的 end-use 结果迭代调整参数，目标降低 EUI。")

    col_cfg, col_run = st.columns([2, 1])
    with col_cfg:
        llm_max_iter = st.slider("最大迭代次数", 1, 10, 5)
        llm_conv_thr = st.slider("收敛阈值 (MJ/m²)", 0.5, 10.0, 2.0, 0.5)
        include_ep_defaults = st.checkbox("允许优化 EnergyPlus 默认参数（照明/人员/窗墙比/设定温度）", value=True)
    with col_run:
        if st.button("▶ 开始 LLM 优化", type="primary", use_container_width=True):
            # Start from current sidebar params
            initial = dict(current_params)

            TUNABLE_20260528: dict[str, dict] = {
                "total_area": {"min": 6000.0, "max": 16000.0, "type": "float", "description": "Total gross floor area (m²)"},
                "lobby_height": {"min": 3.0, "max": 9.0, "type": "float", "description": "Lobby height (m)"},
                "floor_height": {"min": 3.0, "max": 5.0, "type": "float", "description": "Typical floor height (m)"},
                "setback_south": {"min": 0.0, "max": 30.0, "type": "float", "description": "South setback (m)"},
                "setback_west": {"min": 0.0, "max": 30.0, "type": "float", "description": "West setback (m)"},
                "setback_north": {"min": 0.0, "max": 30.0, "type": "float", "description": "North setback (m)"},
                "setback_east": {"min": 0.0, "max": 30.0, "type": "float", "description": "East setback (m)"},
                "low_aspect_ratio": {"min": 0.5, "max": 2.0, "type": "float", "description": "Low block footprint aspect ratio"},
                "mid_aspect_ratio": {"min": 0.5, "max": 2.0, "type": "float", "description": "Mid block footprint aspect ratio"},
                "high_aspect_ratio": {"min": 0.5, "max": 2.0, "type": "float", "description": "High block footprint aspect ratio"},
                "boundary_shift": {"min": 0.0, "max": 200.0, "type": "float", "description": "Shift along buildable boundary for placement"},
                "group_size": {"min": 1, "max": 4, "type": "int", "description": "Vertical grouping size (floors per group)"},
                "low_offset_angle": {"min": 0.0, "max": 360.0, "type": "float", "description": "Low block group offset angle (deg)"},
                "mid_offset_angle": {"min": 0.0, "max": 360.0, "type": "float", "description": "Mid block group offset angle (deg)"},
                "high_offset_angle": {"min": 0.0, "max": 360.0, "type": "float", "description": "High block group offset angle (deg)"},
                "low_offset_distance": {"min": 0.0, "max": 10.0, "type": "float", "description": "Low block group offset distance (m)"},
                "mid_offset_distance": {"min": 0.0, "max": 10.0, "type": "float", "description": "Mid block group offset distance (m)"},
                "high_offset_distance": {"min": 0.0, "max": 10.0, "type": "float", "description": "High block group offset distance (m)"},
                "min_support_overlap_ratio": {"min": 0.1, "max": 1.0, "type": "float", "description": "Minimum plan support overlap ratio"},
                "platform_edge_walk_distance": {"min": 1.0, "max": 12.0, "type": "float", "description": "Platform edge walk distance (m)"},
                "add_aerial_platforms": {"min": 0, "max": 1, "type": "int", "description": "Whether to add aerial platforms (0/1)"},
            }

            tunable = (
                (ALL_TUNABLE | TUNABLE_20260528 | FACADE_TUNABLE)
                if include_ep_defaults
                else (TUNABLE_20260528 | FACADE_TUNABLE)
            )
            with st.spinner("正在运行 LLM 优化（会多次触发 EnergyPlus 模拟）…"):
                st.session_state.mt_llm_history = run_llm_optimization(
                    initial_params=initial,
                    generator_fn=generate_20260528,
                    tunable_spec=tunable,
                    max_iterations=int(llm_max_iter),
                    convergence_threshold=float(llm_conv_thr),
                    partition_enabled=bool(partition_enabled),
                    perimeter_depth=float(perimeter_depth),
                    weather_file=weather_file,
                )
            st.success("LLM 优化完成。")
            st.rerun()

    history: list[IterationRecord] = st.session_state.mt_llm_history
    if not history:
        st.info("点击上方按钮开始 LLM 优化。")
    else:
        best = best_record(history)
        st.divider()
        mcols = st.columns(4)
        mcols[0].metric("最优 EUI", f"{best.eui:.1f} MJ/m²" if best else "N/A")
        mcols[1].metric("迭代次数", len(history))
        mcols[2].metric("最优 Total Site", f"{best.total_site_gj:.2f} GJ" if best else "N/A")
        mcols[3].metric("最优 Area", f"{best.conditioned_area_m2:.0f} m²" if best else "N/A")

        # EUI convergence plot
        fig = go.Figure(go.Scatter(
            x=[r.iteration for r in history],
            y=[r.eui for r in history],
            mode="lines+markers",
        ))
        fig.update_layout(height=280, xaxis_title="Iteration", yaxis_title="EUI (MJ/m²)")
        st.plotly_chart(fig, use_container_width=True, key="mt_llm_eui_curve")

        # Show best model with energy
        st.subheader("最优方案 3D 分区能耗")
        try:
            raw_model = generate_20260528(**best.params)
            model = raw_model
            if partition_enabled:
                model = partition_model_by_floor(
                    raw_model,
                    perimeter_depth=float(perimeter_depth),
                    lobby_height=float(best.params.get("lobby_height", 6.0)),
                    floor_height=float(best.params.get("floor_height", 4.0)),
                )
            sim_data = read_eplustbl(best.result_dir) if best.result_dir else {"exists": False}
            mapped = model_energy_map(model, sim_data) if sim_data.get("exists") else {}
            st.plotly_chart(
                render_model(
                    model,
                    site_size=site_size,
                    show_edges=show_edges,
                    opacity=opacity,
                    zone_energy=(mapped or None),
                    energy_metric=energy_metric,
                ),
                use_container_width=True,
                key="mt_llm_best_model_3d",
            )
            if sim_data.get("exists"):
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(
                        render_end_use_chart(sim_data.get("end_uses", [])),
                        use_container_width=True,
                        key="mt_llm_best_end_use_bar",
                    )
                with c2:
                    st.plotly_chart(
                        render_end_use_pie(sim_data.get("end_uses", [])),
                        use_container_width=True,
                        key="mt_llm_best_end_use_pie",
                    )
        except Exception as e:
            st.error(f"生成/展示最优方案失败：{e}")

        st.subheader("每次迭代摘要")
        st.dataframe(
            [{
                "iter": r.iteration,
                "eui_mj_m2": round(r.eui, 2),
                "total_site_gj": round(r.total_site_gj, 3),
                "area_m2": round(r.conditioned_area_m2, 1),
                "result_dir": r.result_dir or "",
                "error": r.error or "",
            } for r in history],
            hide_index=True,
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Tab 3: GA optimization
# ---------------------------------------------------------------------------

with tab_ga:
    st.subheader("遗传算法优化（GA）— generate_20260528")
    st.caption(
        "每次评估需运行完整 EnergyPlus（direct）。"
        "可调基因（统一三栋 aspect/angle/distance）："
        "`shared_aspect_ratio`、`shared_offset_angle`、`shared_offset_distance`、"
        "`boundary_shift`、`window_wwr`；"
        "固定 `shading_depth=1 m`、`platform_edge_walk_distance=10 m`，其余为生成器默认值。"
    )

    cfg_cols = st.columns(5)
    pop_size = cfg_cols[0].number_input("pop_size", value=10, min_value=5, max_value=30, step=1)
    n_gen = cfg_cols[1].number_input("n_gen", value=8, min_value=3, max_value=50, step=1)
    mutation_rate = cfg_cols[2].number_input("mutation_rate", value=0.15, min_value=0.01, max_value=0.30, step=0.01)
    elite_count = cfg_cols[3].number_input("elite_count", value=1, min_value=1, max_value=3, step=1)
    seed = cfg_cols[4].number_input("seed", value=42, step=1)

    run_cols = st.columns([1, 1, 2])
    if run_cols[0].button("▶ 开始 GA", type="primary", use_container_width=True):
        st.session_state.mt_ga_history = []
        st.session_state.mt_ga_best_params = None
        st.session_state.mt_ga_best_fitness = None
        st.session_state.mt_ga_best_model = None
        st.session_state.mt_ga_best_result_dir = None
        st.session_state.mt_ga_total_evals = 0
        st.session_state.mt_ga_running = True
        ts = time.strftime("%Y%m%d_%H%M%S")
        st.session_state.mt_ga_run_dir = f"output/ga_runs/ga_{ts}_{time.time_ns() % 1_000_000_000:09d}"
        st.rerun()

    if run_cols[1].button("📂 恢复 checkpoint", use_container_width=True):
        ckpt_path = st.session_state.mt_ga_run_dir
        ckpt = load_checkpoint(str(Path(ckpt_path) / "checkpoint.json")) if ckpt_path else None
        if ckpt and ckpt.history:
            st.session_state.mt_ga_history = ckpt.history
            best_h = min(ckpt.history, key=lambda h: h.get("best_fitness", 1e9))
            st.session_state.mt_ga_best_params = best_h.get("best_params")
            st.session_state.mt_ga_best_fitness = best_h.get("best_fitness")
            st.success(f"已恢复到第 {ckpt.generation} 代")
            st.rerun()
        else:
            st.warning("未找到 checkpoint。")

    # Trigger GA run
    if st.session_state.mt_ga_running and not st.session_state.mt_ga_history:
        run_dir = st.session_state.mt_ga_run_dir or "output/ga_runs/ga_unknown"
        st.caption(f"本次 GA 输出目录：`{run_dir}`")
        config = GAConfig(
            pop_size=int(pop_size),
            n_gen=int(n_gen),
            mutation_rate=float(mutation_rate),
            elite_count=int(elite_count),
            run_dir=run_dir,
            use_cache=False,  # 不去重：每次评估都跑模拟并落盘一个目录
            partition_enabled=bool(partition_enabled),
            perimeter_depth=float(perimeter_depth),
        )

        progress = st.progress(0.0, text="准备开始 GA…")
        status = st.empty()
        history_run: list[dict] = []
        total_evals = 0
        overall_best_fitness = float("inf")
        overall_best_params = None
        overall_best_model = None
        overall_best_result_dir = None

        fixed_params = build_ga_fixed_params(current_params)
        fixed_params["_weather_file"] = weather_file
        for result in run_ga(
            config,
            seed=int(seed),
            fixed_params=fixed_params,
            genes=DEFAULT_GENES_20260528,
        ):
            total_evals += len(result.pop_fitness)
            history_run.append({
                "gen": result.gen,
                "best_fitness": result.best_fitness,
                "avg_fitness": result.avg_fitness,
                "worst_fitness": result.worst_fitness,
                "best_params": result.best_params,
                "pop_fitness": result.pop_fitness,
            })

            if result.best_fitness < overall_best_fitness:
                overall_best_fitness = result.best_fitness
                overall_best_params = result.best_params
                overall_best_model = result.best_model
                overall_best_result_dir = result.best_result_dir

            pct = (result.gen + 1) / (config.n_gen + 1)
            progress.progress(
                pct,
                text=f"第 {result.gen}/{config.n_gen} 代 | 最优 EUI: {result.best_fitness:.1f} MJ/m² | 已评估 {total_evals} 个",
            )
            status.info(
                f"当前代最优: {result.best_fitness:.1f} MJ/m² | 平均: {result.avg_fitness:.1f} | 最差: {result.worst_fitness:.1f}"
            )

            # Save checkpoint each generation (history only)
            ckpt = CheckpointState(
                generation=result.gen,
                population=[],
                fitness=[],
                config={"pop_size": config.pop_size, "n_gen": config.n_gen},
                history=history_run,
            )
            save_checkpoint(ckpt, config.checkpoint_path)

        st.session_state.mt_ga_history = history_run
        st.session_state.mt_ga_best_params = overall_best_params
        st.session_state.mt_ga_best_fitness = overall_best_fitness
        st.session_state.mt_ga_best_model = overall_best_model
        st.session_state.mt_ga_best_result_dir = overall_best_result_dir
        st.session_state.mt_ga_total_evals = total_evals
        st.session_state.mt_ga_running = False
        st.rerun()

    history = st.session_state.mt_ga_history
    if not history:
        st.info("点击「开始 GA」运行遗传算法，或恢复 checkpoint。")
    else:
        gens = [h["gen"] for h in history]
        bests = [h["best_fitness"] for h in history]
        avgs = [h["avg_fitness"] for h in history]
        worsts = [h["worst_fitness"] for h in history]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=gens, y=bests, mode="lines+markers", name="best", line=dict(color="#059669")))
        fig.add_trace(go.Scatter(x=gens, y=avgs, mode="lines+markers", name="avg", line=dict(color="#2563EB")))
        fig.add_trace(go.Scatter(x=gens, y=worsts, mode="lines+markers", name="worst", line=dict(color="#DC2626")))
        fig.update_layout(height=300, xaxis_title="generation", yaxis_title="EUI (MJ/m²)")
        st.plotly_chart(fig, use_container_width=True, key="mt_ga_fitness_curve")

        best_params = st.session_state.mt_ga_best_params
        best_fit = st.session_state.mt_ga_best_fitness
        best_model = st.session_state.mt_ga_best_model
        best_result_dir = st.session_state.mt_ga_best_result_dir
        if best_params:
            st.subheader("最优方案")
            st.metric("Best EUI", f"{best_fit:.1f} MJ/m²" if best_fit else "N/A")
            st.json(best_params)
            try:
                # Prefer using the model produced during GA evaluation (guaranteed valid).
                model = best_model
                if model is None and best_result_dir:
                    try:
                        model_path = Path(best_result_dir).parent / "model.json"
                        if model_path.exists():
                            model = json.loads(model_path.read_text(encoding="utf-8"))
                    except Exception:
                        model = None
                if model is None:
                    st.warning("本次 GA 未产生可用的最优模型（可能全部个体被约束惩罚）。请查看曲线/日志或调整参数后重试。")
                    st.stop()
                st.plotly_chart(
                    render_model(model, site_size, show_edges, opacity),
                    use_container_width=True,
                    key="mt_ga_best_model_3d",
                )
            except Exception as e:
                st.error(f"生成最优模型失败：{e}")


# ---------------------------------------------------------------------------
# Tab 4: GA+LLM hybrid optimization
# ---------------------------------------------------------------------------

with tab_hybrid:
    st.subheader("GA+LLM 混合优化")
    st.caption("结合 GA 搜索与 LLM 范围收缩；结果保存在 `output/ga_llm_hybrid/ui_*`。")

    preset_cols = st.columns(4)
    hybrid_preset = preset_cols[0].selectbox(
        "运行预设",
        ["快速冒烟", "标准", "自定义"],
        index=0,
        key="hybrid_preset",
    )
    hybrid_seed = int(preset_cols[1].number_input("随机种子", value=42, step=1, key="hybrid_seed"))
    hybrid_llm = preset_cols[2].checkbox("启用 LLM", value=True, key="hybrid_llm")
    hybrid_morris = preset_cols[3].checkbox("Morris 初筛", value=False, key="hybrid_morris")

    if hybrid_preset == "快速冒烟":
        hy_pop, hy_gen, hy_rounds = 2, 1, 1
        hy_debug, hy_llm, hy_morris = True, False, False
    elif hybrid_preset == "标准":
        hy_pop, hy_gen, hy_rounds = 10, 6, 2
        hy_debug, hy_llm, hy_morris = False, bool(hybrid_llm), bool(hybrid_morris)
    else:
        c1, c2, c3 = st.columns(3)
        hy_pop = int(c1.number_input("population_size", 5, 30, 10, key="hyb_pop"))
        hy_gen = int(c2.number_input("max_generations", 1, 20, 8, key="hyb_gen"))
        hy_rounds = int(c3.number_input("max_rounds", 1, 5, 2, key="hyb_rounds"))
        hy_debug = st.checkbox("debug_mode（缩小规模）", value=False, key="hyb_debug")
        hy_llm = bool(hybrid_llm)
        hy_morris = bool(hybrid_morris)

    est_ep = hy_pop * (hy_gen + 1) * hy_rounds
    st.info(f"预计约 {est_ep} 次 EnergyPlus 评估（Morris/LLM 可能增加额外次数）。")

    btn_cols = st.columns([1, 1, 2])
    start_hybrid = btn_cols[0].button(
        "▶ 一键启动 GA+LLM 混合优化",
        type="primary",
        use_container_width=True,
        key="hybrid_start",
    )
    if btn_cols[1].button("清除结果", use_container_width=True, key="hybrid_clear"):
        st.session_state.mt_hybrid_running = False
        st.session_state.mt_hybrid_report = None
        st.session_state.mt_hybrid_output_dir = None
        st.rerun()

    if start_hybrid:
        st.session_state.mt_hybrid_running = True
        st.session_state.mt_hybrid_report = None
        st.rerun()

    if st.session_state.mt_hybrid_running and st.session_state.mt_hybrid_report is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        ns = time.time_ns() % 1_000_000_000
        out_dir = Path(f"output/ga_llm_hybrid/ui_{ts}_{ns:09d}")
        hybrid_cfg = build_config_from_manual_params(
            current_params,
            population_size=hy_pop,
            max_generations=hy_gen,
            max_rounds=hy_rounds,
            debug_mode=hy_debug,
            llm_enabled=hy_llm,
            morris_enabled=hy_morris,
            partition_enabled=bool(partition_enabled),
            perimeter_depth=float(perimeter_depth),
            seed=hybrid_seed,
        )
        with st.spinner("正在运行 GA+LLM 混合优化（耗时较长）…"):
            try:
                orch = HybridOrchestrator(hybrid_cfg, out_dir)
                report = orch.run()
                st.session_state.mt_hybrid_report = report
                st.session_state.mt_hybrid_output_dir = str(out_dir)
            except Exception as exc:
                st.error(f"混合优化失败：{exc}")
            finally:
                st.session_state.mt_hybrid_running = False
        st.rerun()

    report = st.session_state.mt_hybrid_report
    if report:
        best = report.get("best") or {}
        st.metric("最优 EUI (MJ/m²)", f"{(best.get('objectives') or {}).get('eui_mj_m2', best.get('fitness', 'N/A'))}")
        if st.session_state.mt_hybrid_output_dir:
            st.caption(f"输出目录：`{st.session_state.mt_hybrid_output_dir}`")
        st.json(report)
    elif not st.session_state.mt_hybrid_running:
        st.info("选择预设后点击「一键启动 GA+LLM 混合优化」。")
