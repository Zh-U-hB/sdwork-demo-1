"""Interactive Streamlit page for the gradient L office massing generator."""

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
from scripts.generate_l_gradient import generate_l_gradient


DEFAULTS = {
    "building_name": "Gradient L Office",
    "output_path": "output/gradient_l_office.json",
    "site_size": 100.0,
    "floors": 11,
    "lobby_height": 5.5,
    "floor_height": 4.0,
    "base_x": 18.0,
    "base_y": 16.0,
    "arm_width": 13.5,
    "horizontal_length": 58.0,
    "vertical_length": 54.0,
    "scatter_gap": 8.0,
    "min_fragment_scale": 0.62,
    "merge_power": 1.35,
    "bridge_start_floor": 4,
    "top_solid_floors": 3,
    "add_courtyard_marker": True,
    "simulation_dir": "output/direct_energyplus_real_run",
}

# ---------------------------------------------------------------------------
# Session state for simulation results
# ---------------------------------------------------------------------------
if "ep_result_dir" not in st.session_state:
    st.session_state.ep_result_dir = None


def box_vertices(zone: dict) -> tuple[list[float], list[float], list[float]]:
    if "points" in zone:
        points = zone["points"]
        return (
            [point["x"] for point in points],
            [point["y"] for point in points],
            [point["z"] for point in points],
        )

    origin = zone["origin"]
    dims = zone["dimensions"]
    ox, oy, oz = origin["x"], origin["y"], origin["z"]
    length, width, height = dims["length"], dims["width"], dims["height"]
    vertices = [
        (ox, oy, oz),
        (ox + length, oy, oz),
        (ox + length, oy + width, oz),
        (ox, oy + width, oz),
        (ox, oy, oz + height),
        (ox + length, oy, oz + height),
        (ox + length, oy + width, oz + height),
        (ox, oy + width, oz + height),
    ]
    x, y, z = zip(*vertices)
    return list(x), list(y), list(z)


def box_edges(zone: dict) -> tuple[list[float], list[float], list[float]]:
    x, y, z = box_vertices(zone)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    ex, ey, ez = [], [], []
    for start, end in edges:
        ex.extend([x[start], x[end], None])
        ey.extend([y[start], y[end], None])
        ez.extend([z[start], z[end], None])
    return ex, ey, ez


def zone_floor_index(name: str) -> int:
    if len(name) >= 3 and name[0] == "F" and name[1:3].isdigit():
        return int(name[1:3])
    return 0


def model_metrics(model: dict) -> dict:
    zones = model["zones"]
    mass_zones = [z for z in zones if z["dimensions"]["height"] > 1.0]
    area = sum(z["dimensions"]["length"] * z["dimensions"]["width"] for z in mass_zones)
    max_height = max(
        z["origin"]["z"] + z["dimensions"]["height"]
        for z in mass_zones
    ) if mass_zones else 0.0
    return {
        "area": area,
        "height": max_height,
        "zone_count": len(zones),
        "mass_zone_count": len(mass_zones),
    }


def render_model(
    model: dict,
    site_size: float,
    show_edges: bool,
    opacity: float,
    zone_energy: dict[str, dict[str, float]] | None = None,
    energy_metric: str = "total_gj",
) -> go.Figure:
    fig = go.Figure()
    zones = model["zones"]
    values = [
        metrics.get(energy_metric, 0.0)
        for metrics in (zone_energy or {}).values()
    ]
    cmax = max(values) if values else 0.0
    cmin = 0.0
    has_energy = bool(zone_energy) and cmax > 0
    colorbar_shown = False

    for zone in zones:
        is_courtyard = zone["name"] == "site_inner_courtyard_reference"
        x, y, z = box_vertices(zone)
        floor_index = zone_floor_index(zone["name"])
        color = "#9CA3AF" if is_courtyard else "#2563EB"
        if floor_index >= 9:
            color = "#D97706"
        elif floor_index >= 5:
            color = "#059669"

        energy = (zone_energy or {}).get(zone["name"], {})
        value = energy.get(energy_metric, 0.0)
        hover = (
            f"<b>{zone['name']}</b><br>"
            f"origin: ({zone['origin']['x']}, {zone['origin']['y']}, {zone['origin']['z']})<br>"
            f"{zone['dimensions']['length']} x {zone['dimensions']['width']} x {zone['dimensions']['height']} m"
        )
        if energy:
            hover += (
                f"<br>total: {energy.get('total_gj', 0):.2f} GJ"
                f"<br>heating: {energy.get('heating_gj', 0):.2f} GJ"
                f"<br>cooling: {energy.get('cooling_gj', 0):.2f} GJ"
                f"<br>lighting: {energy.get('lighting_gj', 0):.2f} GJ"
                f"<br>source: {energy.get('source', 'meter')}"
            )

        mesh_kwargs = dict(
            x=x,
            y=y,
            z=z,
            i=[0, 0, 0, 1, 2, 4, 5, 6, 4, 7, 3, 0],
            j=[1, 2, 4, 5, 3, 5, 6, 7, 7, 6, 7, 3],
            k=[2, 3, 5, 4, 7, 6, 1, 2, 0, 2, 0, 4],
            opacity=0.22 if is_courtyard else opacity,
            name=zone["name"],
            hovertemplate=hover + "<extra></extra>",
        )
        if has_energy and not is_courtyard:
            mesh_kwargs.update(
                intensity=[value] * 8,
                colorscale="Turbo",
                cmin=cmin,
                cmax=cmax,
                showscale=not colorbar_shown,
                colorbar=dict(title=f"{energy_metric} GJ"),
            )
            colorbar_shown = True
        else:
            mesh_kwargs.update(color=color, showscale=False)

        fig.add_trace(go.Mesh3d(**mesh_kwargs))

        if show_edges:
            ex, ey, ez = box_edges(zone)
            fig.add_trace(go.Scatter3d(
                x=ex,
                y=ey,
                z=ez,
                mode="lines",
                line=dict(color="#111827" if not is_courtyard else "#6B7280", width=2),
                hoverinfo="skip",
                showlegend=False,
            ))

    fig.add_trace(go.Scatter3d(
        x=[0, site_size, site_size, 0, 0],
        y=[0, 0, site_size, site_size, 0],
        z=[0, 0, 0, 0, 0],
        mode="lines",
        line=dict(color="#DC2626", width=5),
        name="100m site boundary",
        hoverinfo="skip",
    ))

    fig.update_layout(
        height=720,
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis=dict(title="X (m)", range=[0, site_size], backgroundcolor="#F8FAFC"),
            yaxis=dict(title="Y (m)", range=[0, site_size], backgroundcolor="#F8FAFC"),
            zaxis=dict(title="Z (m)", range=[0, 50], backgroundcolor="#F8FAFC"),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.55),
            camera=dict(eye=dict(x=1.45, y=-1.6, z=1.05)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def render_end_use_chart(end_uses: list[dict]) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=[item["end_use"] for item in end_uses],
        y=[item["total_gj"] for item in end_uses],
        marker_color=["#DC2626", "#2563EB", "#F59E0B", "#6B7280", "#7C3AED", "#059669", "#0F766E"][:len(end_uses)],
        hovertemplate="<b>%{x}</b><br>%{y:.2f} GJ<extra></extra>",
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Annual Energy (GJ)",
        xaxis_title="",
    )
    return fig


def render_end_use_pie(end_uses: list[dict]) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=[item["end_use"] for item in end_uses],
        values=[item["total_gj"] for item in end_uses],
        hole=0.45,
        hovertemplate="<b>%{label}</b><br>%{value:.2f} GJ<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    return fig


def render_zone_energy_chart(zone_rows: list[dict]) -> go.Figure:
    fig = go.Figure()
    for key, label, color in [
        ("heating_gj", "Heating", "#DC2626"),
        ("cooling_gj", "Cooling", "#2563EB"),
        ("lighting_gj", "Lighting", "#F59E0B"),
    ]:
        fig.add_trace(go.Bar(
            x=[row["model_zone"] for row in zone_rows],
            y=[row[key] for row in zone_rows],
            name=label,
            marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.2f}} GJ<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack",
        height=360,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis_title="Annual Energy (GJ)",
        xaxis_title="",
    )
    return fig


def save_json(model: dict, output_path: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


st.set_page_config(page_title="参数化 L 形办公体量", layout="wide")

st.title("参数化 L 形办公体量")

with st.sidebar:
    st.header("参数")
    building_name = st.text_input("building_name", DEFAULTS["building_name"])
    output_path = st.text_input("output", DEFAULTS["output_path"])

    st.divider()
    site_size = st.number_input("site_size", 60.0, 200.0, DEFAULTS["site_size"], 1.0)
    floors = st.slider("floors", 2, 14, DEFAULTS["floors"])
    lobby_height = st.slider("lobby_height", 3.0, 9.0, DEFAULTS["lobby_height"], 0.1)
    floor_height = st.slider("floor_height", 3.0, 5.0, DEFAULTS["floor_height"], 0.1)

    st.divider()
    base_x = st.slider("base_x", 0.0, float(site_size), DEFAULTS["base_x"], 0.5)
    base_y = st.slider("base_y", 0.0, float(site_size), DEFAULTS["base_y"], 0.5)
    arm_width = st.slider("arm_width", 6.0, 30.0, DEFAULTS["arm_width"], 0.5)
    horizontal_length = st.slider("horizontal_length", 20.0, float(site_size), DEFAULTS["horizontal_length"], 0.5)
    vertical_length = st.slider("vertical_length", 20.0, float(site_size), DEFAULTS["vertical_length"], 0.5)

    st.divider()
    scatter_gap = st.slider("scatter_gap", 0.0, 18.0, DEFAULTS["scatter_gap"], 0.5)
    min_fragment_scale = st.slider("min_fragment_scale", 0.3, 1.0, DEFAULTS["min_fragment_scale"], 0.01)
    merge_power = st.slider("merge_power", 0.4, 3.0, DEFAULTS["merge_power"], 0.05)
    bridge_start_floor = st.slider("bridge_start_floor", 1, floors, min(DEFAULTS["bridge_start_floor"], floors))
    top_solid_floors = st.slider("top_solid_floors", 1, floors - 1, min(DEFAULTS["top_solid_floors"], floors - 1))
    add_courtyard_marker = st.checkbox("add_courtyard_marker", DEFAULTS["add_courtyard_marker"])

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
    "floors": floors,
    "lobby_height": lobby_height,
    "floor_height": floor_height,
    "base_x": base_x,
    "base_y": base_y,
    "arm_width": arm_width,
    "horizontal_length": horizontal_length,
    "vertical_length": vertical_length,
    "scatter_gap": scatter_gap,
    "min_fragment_scale": min_fragment_scale,
    "merge_power": merge_power,
    "bridge_start_floor": bridge_start_floor,
    "top_solid_floors": top_solid_floors,
    "add_courtyard_marker": add_courtyard_marker,
}

try:
    model = generate_l_gradient(**params)
    metrics = model_metrics(model)

    top_cols = st.columns(4)
    top_cols[0].metric("建筑面积", f"{metrics['area']:.1f} m²")
    top_cols[1].metric("建筑高度", f"{metrics['height']:.1f} m")
    top_cols[2].metric("体块数量", metrics["mass_zone_count"])
    top_cols[3].metric("JSON zones", metrics["zone_count"])

    if not 9000 <= metrics["area"] <= 11000:
        st.warning("当前面积不在 9000-11000 m² 范围内。")
    if metrics["height"] >= 50:
        st.warning("当前高度达到或超过 50m。")

    mass_zones = [z for z in model["zones"] if z["dimensions"]["height"] > 1.0]
    st.session_state["_last_model_zones"] = model["zones"]
    effective_sim_dir = st.session_state.ep_result_dir or simulation_dir
    sim_data = read_eplustbl(effective_sim_dir)
    mapped_energy = model_energy_map(model, sim_data) if sim_data.get("exists") else {}

    preview_tab, simulation_tab = st.tabs(["形体预览", "模拟结果"])

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
                file_name=Path(output_path).name or "gradient_l_office.json",
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
            eui = total_site * 1000 / conditioned_area if conditioned_area else 0.0
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

except ValueError as exc:
    st.error(str(exc))
    st.info("请调整场地尺寸、L 形长度、层数、高度或顶部完整层数。")
