"""Interactive Streamlit page for the platform-connected office massing generator."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from scripts.ep_sim_utils import (
    MASS_HEIGHT_THRESHOLD,
    model_energy_map,
    read_eplustbl,
    run_ep_simulation,
)
from scripts.generate_bridge_cluster import generate_bridge_cluster, gross_area
from scripts.llm_optimizer import (
    ALL_TUNABLE,
    BRIDGE_CLUSTER_TUNABLE,
    best_record,
    run_llm_optimization,
)
from scripts.vis_utils import (
    box_vertices,
    box_edges,
    model_metrics,
    render_model,
    render_end_use_chart,
    render_end_use_pie,
    render_zone_energy_chart,
    save_json,
)


DEFAULTS = {
    "building_name": "Platform Cluster Office",
    "output_path": "output/platform_cluster_office.json",
    "site_size": 100.0,
    "max_floors": 9,
    "lobby_height": 6.0,
    "floor_height": 4.0,
    "floor_plate_efficiency": 1.0,
    "west_x": 12.0,
    "west_y": 18.0,
    "west_length": 30.0,
    "west_width": 24.0,
    "west_floors": 6,
    "east_x": 60.0,
    "east_y": 26.0,
    "east_length": 28.0,
    "east_width": 24.0,
    "east_floors": 7,
    "north_x": 34.0,
    "north_y": 64.0,
    "north_length": 28.0,
    "north_width": 22.0,
    "north_floors": 5,
    "terrace_depth": 3.0,
    "platform_depth": 22.0,
    "platform_width": 16.0,
    "skip_west_ground": False,
    "skip_east_ground": False,
    "skip_north_ground": False,
    "add_open_space_markers": True,
    "simulation_dir": "output/direct_energyplus_real_run",
}

# ---------------------------------------------------------------------------
# Session state for simulation results and LLM optimizer
# ---------------------------------------------------------------------------
if "ep_result_dir" not in st.session_state:
    st.session_state.ep_result_dir = None
if "llm_opt_history" not in st.session_state:
    st.session_state.llm_opt_history = []
if "llm_opt_running" not in st.session_state:
    st.session_state.llm_opt_running = False


st.set_page_config(page_title="立体街区办公体量", layout="wide")

st.title("三体块一层平台立体街区")

with st.sidebar:
    st.header("参数")
    building_name = st.text_input("building_name", DEFAULTS["building_name"])
    output_path = st.text_input("output", DEFAULTS["output_path"])

    st.divider()
    site_size = st.number_input("site_size", 60.0, 200.0, DEFAULTS["site_size"], 1.0)
    max_floors = st.slider("max_floors", 4, 10, DEFAULTS["max_floors"], key="sb_max_floors")
    lobby_height = st.slider("lobby_height", 3.0, 9.0, DEFAULTS["lobby_height"], 0.1, key="sb_lobby_height")
    floor_height = st.slider("floor_height", 3.0, 5.0, DEFAULTS["floor_height"], 0.1, key="sb_floor_height")
    floor_plate_efficiency = st.slider("floor_plate_efficiency", 0.5, 1.0, DEFAULTS["floor_plate_efficiency"], 0.01)

    st.divider()
    st.subheader("西南体块")
    west_x = st.slider("west_x", 0.0, float(site_size), DEFAULTS["west_x"], 0.5)
    west_y = st.slider("west_y", 0.0, float(site_size), DEFAULTS["west_y"], 0.5)
    west_length = st.slider("west_length", 18.0, 42.0, DEFAULTS["west_length"], 0.5, key="sb_west_length")
    west_width = st.slider("west_width", 16.0, 36.0, DEFAULTS["west_width"], 0.5, key="sb_west_width")
    west_floors = st.slider("west_floors", 1, max_floors, min(DEFAULTS["west_floors"], max_floors), key="sb_west_floors")

    st.divider()
    st.subheader("东侧体块")
    east_x = st.slider("east_x", 0.0, float(site_size), DEFAULTS["east_x"], 0.5)
    east_y = st.slider("east_y", 0.0, float(site_size), DEFAULTS["east_y"], 0.5)
    east_length = st.slider("east_length", 18.0, 42.0, DEFAULTS["east_length"], 0.5, key="sb_east_length")
    east_width = st.slider("east_width", 16.0, 36.0, DEFAULTS["east_width"], 0.5, key="sb_east_width")
    east_floors = st.slider("east_floors", 1, max_floors, min(DEFAULTS["east_floors"], max_floors), key="sb_east_floors")

    st.divider()
    st.subheader("北侧体块")
    north_x = st.slider("north_x", 0.0, float(site_size), DEFAULTS["north_x"], 0.5)
    north_y = st.slider("north_y", 0.0, float(site_size), DEFAULTS["north_y"], 0.5)
    north_length = st.slider("north_length", 18.0, 42.0, DEFAULTS["north_length"], 0.5, key="sb_north_length")
    north_width = st.slider("north_width", 16.0, 36.0, DEFAULTS["north_width"], 0.5, key="sb_north_width")
    north_floors = st.slider("north_floors", 1, max_floors, min(DEFAULTS["north_floors"], max_floors), key="sb_north_floors")

    st.divider()
    st.subheader("一层平台与退台")
    terrace_depth = st.slider("terrace_depth", 0.0, 6.0, DEFAULTS["terrace_depth"], 0.5, key="sb_terrace_depth")
    platform_depth = st.slider("platform_depth", 12.0, 34.0, DEFAULTS["platform_depth"], 0.5, key="sb_platform_depth")
    platform_width = st.slider("platform_width", 10.0, 28.0, DEFAULTS["platform_width"], 0.5, key="sb_platform_width")

    st.divider()
    skip_west_ground = st.checkbox("skip_west_ground", DEFAULTS["skip_west_ground"])
    skip_east_ground = st.checkbox("skip_east_ground", DEFAULTS["skip_east_ground"])
    skip_north_ground = st.checkbox("skip_north_ground", DEFAULTS["skip_north_ground"])
    add_open_space_markers = st.checkbox("add_open_space_markers", DEFAULTS["add_open_space_markers"])

    st.divider()
    show_edges = st.checkbox("显示边线", True)
    opacity = st.slider("体量透明度", 0.15, 1.0, 0.58, 0.05)

    st.divider()
    st.header("模拟结果")
    simulation_dir = st.text_input("EnergyPlus 结果目录", DEFAULTS["simulation_dir"])
    energy_metric = st.selectbox(
        "三维着色指标",
        [
            ("total_gj", "总能耗"),
            ("heating_gj", "采暖"),
            ("cooling_gj", "制冷"),
            ("lighting_gj", "照明"),
        ],
        format_func=lambda item: item[1],
    )[0]


params = {
    "building_name": building_name,
    "site_size": site_size,
    "max_floors": max_floors,
    "lobby_height": lobby_height,
    "floor_height": floor_height,
    "floor_plate_efficiency": floor_plate_efficiency,
    "west_x": west_x,
    "west_y": west_y,
    "west_length": west_length,
    "west_width": west_width,
    "west_floors": west_floors,
    "east_x": east_x,
    "east_y": east_y,
    "east_length": east_length,
    "east_width": east_width,
    "east_floors": east_floors,
    "north_x": north_x,
    "north_y": north_y,
    "north_length": north_length,
    "north_width": north_width,
    "north_floors": north_floors,
    "terrace_depth": terrace_depth,
    "platform_depth": platform_depth,
    "platform_width": platform_width,
    "skip_west_ground": skip_west_ground,
    "skip_east_ground": skip_east_ground,
    "skip_north_ground": skip_north_ground,
    "add_open_space_markers": add_open_space_markers,
}

try:
    model = generate_bridge_cluster(**params)
    metrics = model_metrics(model, gross_area_fn=gross_area)

    top_cols = st.columns(4)
    top_cols[0].metric("建筑面积", f"{metrics['area']:.1f} m²")
    top_cols[1].metric("建筑高度", f"{metrics['height']:.1f} m")
    top_cols[2].metric("体块数量", metrics["mass_zone_count"])
    top_cols[3].metric("JSON zones", metrics["zone_count"])

    if not 9000 <= metrics["area"] <= 11000:
        st.warning("当前面积不在 9000-11000 m² 范围内。")
    if metrics["height"] >= 50:
        st.warning("当前高度达到或超过 50m。")

    mass_zones = [
        z for z in model["zones"]
        if z["dimensions"]["height"] > MASS_HEIGHT_THRESHOLD and z.get("category") != "open_space_reference"
    ]
    st.session_state["_last_model_zones"] = model["zones"]
    effective_sim_dir = st.session_state.ep_result_dir or simulation_dir
    sim_data = read_eplustbl(effective_sim_dir)
    mapped_energy = model_energy_map(model, sim_data) if sim_data.get("exists") else {}

    preview_tab, simulation_tab, llm_tab = st.tabs(["形体预览", "模拟结果", "LLM 优化"])

    with preview_tab:
        left, right = st.columns([2, 1])
        with left:
            st.plotly_chart(render_model(model, site_size, show_edges, opacity), use_container_width=True)

        with right:
            st.subheader("导出")
            if st.button("保存 JSON", use_container_width=True):
                path = save_json(model, output_path)
                st.success(f"已保存到 {path}")

            st.download_button(
                "下载当前 JSON",
                data=json.dumps(model, indent=2, ensure_ascii=False),
                file_name=Path(output_path).name or "platform_cluster_office.json",
                mime="application/json",
                use_container_width=True,
            )

            st.subheader("当前参数")
            st.json(params)

            st.subheader("前 12 个体块")
            st.dataframe(
                [
                    {
                        "name": z["name"],
                        "x": z["origin"]["x"],
                        "y": z["origin"]["y"],
                        "z": z["origin"]["z"],
                        "l": z["dimensions"]["length"],
                        "w": z["dimensions"]["width"],
                        "h": z["dimensions"]["height"],
                    }
                    for z in model["zones"][:12]
                ],
                hide_index=True,
                use_container_width=True,
            )

    with simulation_tab:
        # --- Simulation trigger button ---
        _mass_count = len(mass_zones)
        if _mass_count > 30:
            st.caption(f"⚠ 当前模型有 {_mass_count} 个体块，模拟将消耗较多 API 额度与时间。")
        if st.button("▶ 执行 EnergyPlus 模拟", type="primary", use_container_width=True):
            with st.spinner("正在执行 EnergyPlus 模拟，请耐心等待（可能需要数分钟）…"):
                try:
                    result_dir = run_ep_simulation(model, building_name)
                    if result_dir:
                        st.session_state.ep_result_dir = result_dir
                        st.success(f"模拟完成！结果目录：{result_dir}")
                        st.rerun()
                    else:
                        st.error("模拟完成但未找到结果文件 eplustbl.csv。")
                except Exception as e:
                    st.error(f"模拟失败：{e}")

        st.divider()

        if not sim_data.get("exists"):
            st.warning(f"没有找到模拟结果文件：{sim_data['path']}")
            st.info("点击上方「▶ 执行 EnergyPlus 模拟」按钮开始模拟，或在左侧修改结果目录路径。")
        else:
            st.caption(f"结果文件：{sim_data['path']}")
            total_site = sim_data["site_energy"].get("Total Site Energy", 0.0)
            total_source = sim_data["site_energy"].get("Total Source Energy", 0.0)
            conditioned_area = sim_data["building_area"].get("Net Conditioned Building Area", 0.0)
            eui = total_site * 1000 / conditioned_area if conditioned_area > 0.1 else 0.0
            mapped_total = sum(value.get("total_gj", 0.0) for value in mapped_energy.values())
            energy_source = "area_estimate"
            if mapped_energy and all(value.get("source") == "meter" for value in mapped_energy.values()):
                energy_source = "meter"

            sim_cols = st.columns(4)
            sim_cols[0].metric("总场地能耗", f"{total_site:.2f} GJ")
            sim_cols[1].metric("总源能耗", f"{total_source:.2f} GJ")
            sim_cols[2].metric("EUI", f"{eui:.1f} MJ/m²")
            sim_cols[3].metric("显示体块", f"{len(mapped_energy)} / {len(mass_zones)}")
            if energy_source == "area_estimate":
                st.info("当前结果没有覆盖所有模型 zone 的逐区 meter，页面已按体块面积把总采暖、制冷、照明能耗分摊到所有 zone。")

            chart_left, chart_right = st.columns(2)
            with chart_left:
                st.subheader("分项年能耗")
                st.plotly_chart(render_end_use_chart(sim_data["end_uses"]), use_container_width=True)
            with chart_right:
                st.subheader("分项占比")
                st.plotly_chart(render_end_use_pie(sim_data["end_uses"]), use_container_width=True)

            st.subheader("按 Zone 着色的三维能耗图")
            if not mapped_energy:
                st.warning("当前 eplustbl.csv 中没有可映射到模型体块的 zone 级能耗。")
            st.plotly_chart(
                render_model(
                    model,
                    site_size,
                    show_edges,
                    opacity,
                    zone_energy=mapped_energy,
                    energy_metric=energy_metric,
                ),
                use_container_width=True,
            )

            zone_rows = []
            for i, zone in enumerate(mass_zones, start=1):
                energy = mapped_energy.get(zone["name"], {})
                area = zone["dimensions"]["length"] * zone["dimensions"]["width"]
                total = energy.get("total_gj", 0.0)
                zone_rows.append({
                    "model_zone": zone["name"],
                    "energyplus_zone": f"ZONE_{i:02d}",
                    "area_m2": round(area, 2),
                    "heating_gj": round(energy.get("heating_gj", 0.0), 3),
                    "cooling_gj": round(energy.get("cooling_gj", 0.0), 3),
                    "lighting_gj": round(energy.get("lighting_gj", 0.0), 3),
                    "total_gj": round(total, 3),
                    "eui_mj_m2": round(total * 1000 / area, 2) if area else 0.0,
                    "source": energy.get("source", ""),
                })

            table_left, table_right = st.columns([3, 2])
            with table_left:
                st.subheader("Zone 能耗")
                if zone_rows:
                    st.plotly_chart(render_zone_energy_chart(zone_rows[:40]), use_container_width=True)
                    st.dataframe(zone_rows, hide_index=True, use_container_width=True)
            with table_right:
                st.subheader("原始 EnergyPlus Zone 数据")
                energyplus_rows = [
                    {"energyplus_zone": zone, **{k: round(v, 3) for k, v in values.items()}}
                    for zone, values in sorted(sim_data["zone_energy"].items())
                ]
                st.dataframe(energyplus_rows, hide_index=True, use_container_width=True)

                st.subheader("模拟文件")
                result_dir = Path(effective_sim_dir)
                files = [
                    {"file": file.name, "size_kb": round(file.stat().st_size / 1024, 1)}
                    for file in sorted(result_dir.glob("*"))
                    if file.is_file()
                ]
                st.dataframe(files, hide_index=True, use_container_width=True)

    # ── LLM Optimization Tab ─────────────────────────────────────────────
    with llm_tab:
        st.subheader("LLM 驱动的能耗优化")
        st.caption(
            "每轮自动运行 EnergyPlus → LLM 读取能耗结果并给出参数调整建议 → 重新模拟。"
            "迭代直到收敛或达到最大轮次。"
        )

        if not sim_data.get("exists"):
            st.warning("请先在「模拟结果」Tab 中完成一次 EnergyPlus 模拟，再启动 LLM 优化。")
        else:
            # --- Config controls ---
            opt_col1, opt_col2, opt_col3 = st.columns(3)
            with opt_col1:
                llm_max_iter = st.slider("最大迭代轮数", 2, 10, 5, key="llm_max_iter")
            with opt_col2:
                llm_conv_thr = st.number_input(
                    "收敛阈值 (MJ/m²)", value=2.0, min_value=0.5, max_value=10.0,
                    step=0.5, key="llm_conv_thr",
                    help="连续 3 轮 EUI 累计改善 < 此值时停止",
                )
            with opt_col3:
                include_ep_defaults = st.checkbox(
                    "同时优化 EnergyPlus 默认参数",
                    value=True, key="llm_ep_defaults",
                    help="允许 LLM 调整照明功率密度、窗墙比、供暖/制冷设定点",
                )

            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                start_llm = st.button(
                    "▶ 开始 LLM 优化", type="primary",
                    use_container_width=True,
                    disabled=st.session_state.llm_opt_running,
                )
            with btn_col2:
                if st.button("重置历史", use_container_width=True):
                    st.session_state.llm_opt_history = []
                    st.session_state.llm_opt_running = False
                    st.rerun()

            # --- Run optimization ---
            if start_llm and not st.session_state.llm_opt_running:
                st.session_state.llm_opt_running = True
                st.session_state.llm_opt_history = []

                tunable = ALL_TUNABLE if include_ep_defaults else BRIDGE_CLUSTER_TUNABLE

                # Streamlit runs synchronously; progress_bar.progress() calls inside a
                # blocking loop will NOT render mid-run — they only take effect on rerun.
                # We use a spinner + a live log container that accumulates text instead.
                progress_placeholder = st.empty()
                progress_placeholder.info(
                    f"LLM 优化运行中，最多 {llm_max_iter} 轮，请等待…"
                )
                iter_records = []

                def _on_iteration(record):
                    iter_records.append(record)
                    eui_str = f"{record.eui:.1f}" if record.eui < 1e5 else "FAIL"
                    progress_placeholder.info(
                        f"已完成 {len(iter_records)} / {llm_max_iter} 轮 | "
                        f"最新 EUI: {eui_str} MJ/m²"
                    )

                try:
                    with st.spinner(f"LLM 优化中（最多 {llm_max_iter} 轮）…"):
                        history = run_llm_optimization(
                            initial_params=dict(params),
                            generator_fn=generate_bridge_cluster,
                            tunable_spec=tunable,
                            max_iterations=llm_max_iter,
                            convergence_threshold=float(llm_conv_thr),
                            output_base="output/llm_opt",
                            progress_callback=_on_iteration,
                        )
                    st.session_state.llm_opt_history = [
                        {
                            "iteration": r.iteration,
                            "eui": r.eui,
                            "params": r.params,
                            "ep_defaults_overrides": r.ep_defaults_overrides,
                            "end_uses": r.end_uses,
                            "llm_analysis": r.llm_analysis,
                            "result_dir": r.result_dir,
                            "error": r.error,
                        }
                        for r in history
                    ]
                except Exception as exc:
                    st.error(f"LLM 优化出错：{exc}")
                finally:
                    st.session_state.llm_opt_running = False
                    st.rerun()

            # --- Display history ---
            opt_history = st.session_state.llm_opt_history
            if opt_history:
                valid_recs = [r for r in opt_history if r["eui"] < 1e5]

                if valid_recs:
                    # EUI convergence chart
                    st.subheader("EUI 收敛曲线")
                    fig_eui = go.Figure(go.Scatter(
                        x=[r["iteration"] for r in valid_recs],
                        y=[r["eui"] for r in valid_recs],
                        mode="lines+markers",
                        marker=dict(size=8),
                        line=dict(color="#2563EB"),
                        hovertemplate="Iter %{x}: %{y:.1f} MJ/m²<extra></extra>",
                    ))
                    fig_eui.update_layout(
                        height=280,
                        margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title="迭代轮次",
                        yaxis_title="EUI (MJ/m²)",
                    )
                    st.plotly_chart(fig_eui, use_container_width=True)

                    # Best result summary
                    best = min(valid_recs, key=lambda r: r["eui"])
                    st.subheader(f"最优结果 — 第 {best['iteration']} 轮  EUI: {best['eui']:.1f} MJ/m²")

                    diff_rows = []
                    for key, new_val in best["params"].items():
                        old_val = params.get(key)
                        if old_val is not None and old_val != new_val:
                            diff_rows.append({
                                "参数": key,
                                "初始值": old_val,
                                "最优值": new_val,
                                "变化": f"{new_val - old_val:+.3g}" if isinstance(new_val, (int, float)) else "—",
                            })
                    for key, new_val in best["ep_defaults_overrides"].items():
                        diff_rows.append({
                            "参数": f"[EP] {key}",
                            "初始值": "default",
                            "最优值": new_val,
                            "变化": "—",
                        })
                    if diff_rows:
                        st.dataframe(diff_rows, hide_index=True, use_container_width=True)

                    # Apply best params: write to sidebar widget session_state keys
                    # so sliders reflect new values immediately on next render.
                    _PARAM_TO_SB_KEY = {
                        "max_floors": "sb_max_floors",
                        "lobby_height": "sb_lobby_height",
                        "floor_height": "sb_floor_height",
                        "west_floors": "sb_west_floors",
                        "west_length": "sb_west_length",
                        "west_width": "sb_west_width",
                        "east_floors": "sb_east_floors",
                        "east_length": "sb_east_length",
                        "east_width": "sb_east_width",
                        "north_floors": "sb_north_floors",
                        "north_length": "sb_north_length",
                        "north_width": "sb_north_width",
                        "terrace_depth": "sb_terrace_depth",
                        "platform_depth": "sb_platform_depth",
                        "platform_width": "sb_platform_width",
                    }
                    if st.button("应用最优参数到侧边栏", use_container_width=True):
                        applied = []
                        for param_key, val in best["params"].items():
                            sb_key = _PARAM_TO_SB_KEY.get(param_key)
                            if sb_key:
                                st.session_state[sb_key] = val
                                applied.append(param_key)
                        if applied:
                            st.success(f"已应用 {len(applied)} 个参数：{', '.join(applied)}")
                            st.rerun()
                        else:
                            st.info("没有可应用的参数（可能与当前值相同）。")

                # Per-iteration breakdown table
                st.subheader("逐轮详情")
                rows_display = []
                for r in opt_history:
                    eui_str = f"{r['eui']:.1f}" if r["eui"] < 1e5 else "FAIL"
                    rows_display.append({
                        "轮次": r["iteration"],
                        "EUI (MJ/m²)": eui_str,
                        "LLM 分析": r["llm_analysis"][:120] + "…" if r["llm_analysis"] and len(r["llm_analysis"]) > 120 else r["llm_analysis"] or r.get("error", ""),
                        "结果目录": r["result_dir"] or "—",
                    })
                st.dataframe(rows_display, hide_index=True, use_container_width=True)

                # Full analysis per iteration in expanders
                with st.expander("查看各轮 LLM 完整分析"):
                    for r in opt_history:
                        eui_str = f"{r['eui']:.1f}" if r["eui"] < 1e5 else "FAIL"
                        st.markdown(f"**Iter {r['iteration']} — EUI: {eui_str} MJ/m²**")
                        if r.get("error"):
                            st.error(r["error"])
                        elif r["llm_analysis"]:
                            st.write(r["llm_analysis"])
                        if r["ep_defaults_overrides"]:
                            st.json(r["ep_defaults_overrides"])
                        st.divider()
            else:
                st.info("点击「▶ 开始 LLM 优化」启动优化循环。每轮约 3-10 秒。")


except ValueError as exc:
    st.error(str(exc))
    st.info("请调整体块位置、尺寸、层数或平台参数。")
