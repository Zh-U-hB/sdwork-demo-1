"""Interactive Streamlit viewer for the 2026-05-28 boundary-offset model."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from scripts.generate_20260528 import generate_20260528, gross_area, model_height


DEFAULTS = {
    "building_name": "Boundary Offset Three Block Office",
    "output_path": "output/generate_20260528.json",
    "site_size": 100.0,
    "total_area": 10000.0,
    "lobby_height": 6.0,
    "floor_height": 4.0,
    "setback_south": 15.0,
    "setback_west": 15.0,
    "setback_north": 10.0,
    "setback_east": 10.0,
    "low_aspect_ratio": 1.0,
    "mid_aspect_ratio": 1.0,
    "high_aspect_ratio": 1.0,
    "boundary_shift": 40.0,
    "group_size": 2,
    "low_offset_angle": 45.0,
    "mid_offset_angle": 180.0,
    "high_offset_angle": 315.0,
    "low_offset_distance": 2.0,
    "mid_offset_distance": 2.0,
    "high_offset_distance": 2.0,
    "min_support_overlap_ratio": 0.5,
    "add_aerial_platforms": True,
    "platform_edge_walk_distance": 5.0,
    "add_open_space_markers": True,
}


def box_vertices(zone: dict) -> tuple[list[float], list[float], list[float]]:
    points = zone.get("points")
    if points:
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


def block_color(name: str, category: str) -> str:
    if category == "open_space_reference":
        return "#94A3B8"
    if category == "aerial_platform":
        return "#E11D48"
    if name.startswith("low_block"):
        return "#2563EB"
    if name.startswith("mid_block"):
        return "#059669"
    if name.startswith("high_block"):
        return "#D97706"
    return "#475569"


def add_boundary_trace(fig: go.Figure, x0: float, y0: float, x1: float, y1: float, name: str, color: str, width: int) -> None:
    fig.add_trace(go.Scatter3d(
        x=[x0, x1, x1, x0, x0],
        y=[y0, y0, y1, y1, y0],
        z=[0, 0, 0, 0, 0],
        mode="lines",
        line=dict(color=color, width=width),
        name=name,
        hoverinfo="skip",
    ))


def render_model(model: dict, show_edges: bool, opacity: float) -> go.Figure:
    fig = go.Figure()

    for zone in model["zones"]:
        category = zone.get("category", "")
        is_reference = category == "open_space_reference"
        x, y, z = box_vertices(zone)
        dims = zone["dimensions"]
        hover = (
            f"<b>{zone['name']}</b><br>"
            f"origin: ({zone['origin']['x']}, {zone['origin']['y']}, {zone['origin']['z']})<br>"
            f"{dims['length']} x {dims['width']} x {dims['height']} m"
        )
        fig.add_trace(go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=[0, 0, 0, 1, 2, 4, 5, 6, 4, 7, 3, 0],
            j=[1, 2, 4, 5, 3, 5, 6, 7, 7, 6, 7, 3],
            k=[2, 3, 5, 4, 7, 6, 1, 2, 0, 2, 0, 4],
            color=block_color(zone["name"], category),
            opacity=0.16 if is_reference else opacity,
            name=zone["name"],
            hovertemplate=hover + "<extra></extra>",
            showscale=False,
        ))
        if show_edges:
            ex, ey, ez = box_edges(zone)
            fig.add_trace(go.Scatter3d(
                x=ex,
                y=ey,
                z=ez,
                mode="lines",
                line=dict(color="#111827" if not is_reference else "#64748B", width=2),
                hoverinfo="skip",
                showlegend=False,
            ))

    metadata = model.get("metadata", {})
    site_size = metadata.get("site_size", 100.0)
    bounds = metadata.get("buildable_bounds", {})
    add_boundary_trace(fig, 0, 0, site_size, site_size, "site boundary", "#DC2626", 5)
    if bounds:
        add_boundary_trace(
            fig,
            bounds["x_min"],
            bounds["y_min"],
            bounds["x_max"],
            bounds["y_max"],
            "buildable boundary",
            "#16A34A",
            4,
        )

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


st.set_page_config(page_title="Boundary Offset Model", layout="wide")
st.title("三栋边界贴线错位体块参数化模型")

with st.sidebar:
    st.header("基础参数")
    building_name = st.text_input("建筑名称", DEFAULTS["building_name"])
    output_path = st.text_input("输出 JSON 路径", DEFAULTS["output_path"])
    total_area = st.slider("总建筑面积 m²", 6000.0, 16000.0, DEFAULTS["total_area"], 100.0)
    boundary_shift = st.slider("沿边界移动距离 m", 0.0, 320.0, DEFAULTS["boundary_shift"], 1.0)

    st.header("层高与退线")
    lobby_height = st.slider("首层高度 m", 3.0, 8.0, DEFAULTS["lobby_height"], 0.5)
    floor_height = st.slider("标准层高 m", 3.0, 5.5, DEFAULTS["floor_height"], 0.25)
    setback_south = st.slider("南侧退线 m", 0.0, 25.0, DEFAULTS["setback_south"], 1.0)
    setback_west = st.slider("西侧退线 m", 0.0, 25.0, DEFAULTS["setback_west"], 1.0)
    setback_north = st.slider("北侧退线 m", 0.0, 25.0, DEFAULTS["setback_north"], 1.0)
    setback_east = st.slider("东侧退线 m", 0.0, 25.0, DEFAULTS["setback_east"], 1.0)

    st.header("平面长宽比")
    low_aspect_ratio = st.slider("低层楼长宽比", 0.5, 2.5, DEFAULTS["low_aspect_ratio"], 0.05)
    mid_aspect_ratio = st.slider("中层楼长宽比", 0.5, 2.5, DEFAULTS["mid_aspect_ratio"], 0.05)
    high_aspect_ratio = st.slider("高层楼长宽比", 0.5, 2.5, DEFAULTS["high_aspect_ratio"], 0.05)

    st.header("错位参数")
    low_offset_angle = st.slider("低层楼错位角度", 0.0, 360.0, DEFAULTS["low_offset_angle"], 5.0)
    low_offset_distance = st.slider("低层楼错位距离 m", 0.0, 8.0, DEFAULTS["low_offset_distance"], 0.25)
    mid_offset_angle = st.slider("中层楼错位角度", 0.0, 360.0, DEFAULTS["mid_offset_angle"], 5.0)
    mid_offset_distance = st.slider("中层楼错位距离 m", 0.0, 8.0, DEFAULTS["mid_offset_distance"], 0.25)
    high_offset_angle = st.slider("高层楼错位角度", 0.0, 360.0, DEFAULTS["high_offset_angle"], 5.0)
    high_offset_distance = st.slider("高层楼错位距离 m", 0.0, 8.0, DEFAULTS["high_offset_distance"], 0.25)

    st.header("空中平台")
    add_aerial_platforms = st.checkbox("生成空中平台", DEFAULTS["add_aerial_platforms"])
    platform_edge_walk_distance = st.slider(
        "平台沿目标边移动距离 m",
        1.0,
        12.0,
        DEFAULTS["platform_edge_walk_distance"],
        0.5,
    )

    st.header("显示")
    min_support_overlap_ratio = st.slider("最小支撑重叠比例", 0.1, 1.0, DEFAULTS["min_support_overlap_ratio"], 0.05)
    opacity = st.slider("体块透明度", 0.2, 1.0, 0.72, 0.05)
    show_edges = st.checkbox("显示边线", True)
    add_open_space_markers = st.checkbox("显示可建范围参考面", True)

params = {
    "building_name": building_name,
    "site_size": DEFAULTS["site_size"],
    "total_area": total_area,
    "lobby_height": lobby_height,
    "floor_height": floor_height,
    "setback_south": setback_south,
    "setback_west": setback_west,
    "setback_north": setback_north,
    "setback_east": setback_east,
    "low_aspect_ratio": low_aspect_ratio,
    "mid_aspect_ratio": mid_aspect_ratio,
    "high_aspect_ratio": high_aspect_ratio,
    "boundary_shift": boundary_shift,
    "group_size": DEFAULTS["group_size"],
    "low_offset_angle": low_offset_angle,
    "mid_offset_angle": mid_offset_angle,
    "high_offset_angle": high_offset_angle,
    "low_offset_distance": low_offset_distance,
    "mid_offset_distance": mid_offset_distance,
    "high_offset_distance": high_offset_distance,
    "min_support_overlap_ratio": min_support_overlap_ratio,
    "add_aerial_platforms": add_aerial_platforms,
    "platform_edge_walk_distance": platform_edge_walk_distance,
    "add_open_space_markers": add_open_space_markers,
}

try:
    model = generate_20260528(**params)
    mass_zones = [zone for zone in model["zones"] if zone.get("category") == "mass_block"]
    platform_zones = [zone for zone in model["zones"] if zone.get("category") == "aerial_platform"]

    cols = st.columns(4)
    cols[0].metric("建筑面积", f"{gross_area(model):.1f} m²")
    cols[1].metric("建筑高度", f"{model_height(model):.1f} m")
    cols[2].metric("主体/平台", f"{len(mass_zones)} / {len(platform_zones)}")
    cols[3].metric("JSON zones", len(model["zones"]))

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(render_model(model, show_edges, opacity), use_container_width=True)

    with right:
        st.subheader("导出")
        if st.button("保存 JSON", use_container_width=True):
            path = save_json(model, output_path)
            st.success(f"已保存到 {path}")

        st.download_button(
            "下载当前 JSON",
            data=json.dumps(model, indent=2, ensure_ascii=False),
            file_name=Path(output_path).name or "generate_20260528.json",
            mime="application/json",
            use_container_width=True,
        )

        st.subheader("三栋楼参数")
        st.dataframe(model["metadata"]["buildings"], use_container_width=True)

        if platform_zones:
            st.subheader("空中平台")
            st.dataframe(model["metadata"].get("aerial_platforms", []), use_container_width=True)

        st.subheader("体块列表")
        st.dataframe(
            [
                {
                    "name": zone["name"],
                    "x": zone["origin"]["x"],
                    "y": zone["origin"]["y"],
                    "z": zone["origin"]["z"],
                    "l": zone["dimensions"]["length"],
                    "w": zone["dimensions"]["width"],
                    "h": zone["dimensions"]["height"],
                }
                for zone in mass_zones
            ],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("当前生成参数"):
        st.json(params)

except ValueError as exc:
    st.error(f"当前参数组合无效：{exc}")
    st.json(params)
