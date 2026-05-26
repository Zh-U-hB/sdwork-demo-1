"""Interactive Streamlit page for the gradient L office massing generator."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

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
}


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


def render_model(model: dict, site_size: float, show_edges: bool, opacity: float) -> go.Figure:
    fig = go.Figure()
    zones = model["zones"]

    for zone in zones:
        is_courtyard = zone["name"] == "site_inner_courtyard_reference"
        x, y, z = box_vertices(zone)
        floor_index = zone_floor_index(zone["name"])
        color = "#9CA3AF" if is_courtyard else "#2563EB"
        if floor_index >= 9:
            color = "#D97706"
        elif floor_index >= 5:
            color = "#059669"

        fig.add_trace(go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=[0, 0, 0, 1, 2, 4, 5, 6, 4, 7, 3, 0],
            j=[1, 2, 4, 5, 3, 5, 6, 7, 7, 6, 7, 3],
            k=[2, 3, 5, 4, 7, 6, 1, 2, 0, 2, 0, 4],
            color=color,
            opacity=0.22 if is_courtyard else opacity,
            name=zone["name"],
            hovertemplate=(
                f"<b>{zone['name']}</b><br>"
                f"origin: ({zone['origin']['x']}, {zone['origin']['y']}, {zone['origin']['z']})<br>"
                f"{zone['dimensions']['length']} x {zone['dimensions']['width']} x {zone['dimensions']['height']} m"
                "<extra></extra>"
            ),
            showscale=False,
        ))

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

except ValueError as exc:
    st.error(str(exc))
    st.info("请调整场地尺寸、L 形长度、层数、高度或顶部完整层数。")
