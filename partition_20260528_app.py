"""Streamlit viewer for partitioned 2026-05-28 zones."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from scripts.generate_20260528 import generate_20260528
from scripts.zone_partition import partition_model_by_floor


DEFAULT_OUTPUT = "output/generate_20260528_partitioned.json"


def prism_vertices(zone: dict) -> tuple[list[float], list[float], list[float]]:
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


def prism_mesh_indices(point_count: int) -> tuple[list[int], list[int], list[int]]:
    n = point_count // 2
    i: list[int] = []
    j: list[int] = []
    k: list[int] = []
    for idx in range(1, n - 1):
        i.append(0)
        j.append(idx)
        k.append(idx + 1)
        i.append(n)
        j.append(n + idx + 1)
        k.append(n + idx)
    for idx in range(n):
        nxt = (idx + 1) % n
        i.extend([idx, idx])
        j.extend([nxt, n + nxt])
        k.extend([n + nxt, n + idx])
    return i, j, k


def prism_edges(zone: dict) -> tuple[list[float], list[float], list[float]]:
    x, y, z = prism_vertices(zone)
    n = len(x) // 2
    edges = []
    for idx in range(n):
        nxt = (idx + 1) % n
        edges.append((idx, nxt))
        edges.append((n + idx, n + nxt))
        edges.append((idx, n + idx))
    ex, ey, ez = [], [], []
    for start, end in edges:
        ex.extend([x[start], x[end], None])
        ey.extend([y[start], y[end], None])
        ez.extend([z[start], z[end], None])
    return ex, ey, ez


def zone_color(category: str) -> str:
    return {
        "interior_zone": "#F59E0B",
        "perimeter_zone": "#2563EB",
        "horizontal_exposed_zone": "#DC2626",
        "aerial_platform_zone": "#E11D48",
    }.get(category, "#64748B")


def render_model(model: dict, opacity: float, show_edges: bool, visible_categories: set[str]) -> go.Figure:
    fig = go.Figure()
    for zone in model["zones"]:
        category = zone.get("category", "")
        if category not in visible_categories:
            continue
        x, y, z = prism_vertices(zone)
        i, j, k = prism_mesh_indices(len(x))
        dims = zone["dimensions"]
        metadata = zone.get("metadata", {})
        exposure = ", ".join(metadata.get("exposure", [])) or "none"
        hover = (
            f"<b>{zone['name']}</b><br>"
            f"category: {category}<br>"
            f"floor: {metadata.get('floor', '-')}<br>"
            f"exposure: {exposure}<br>"
            f"origin: ({zone['origin']['x']}, {zone['origin']['y']}, {zone['origin']['z']})<br>"
            f"{dims['length']} x {dims['width']} x {dims['height']} m"
        )
        fig.add_trace(go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=i,
            j=j,
            k=k,
            color=zone_color(category),
            opacity=opacity,
            name=category,
            hovertemplate=hover + "<extra></extra>",
            showscale=False,
            showlegend=False,
        ))
        if show_edges:
            ex, ey, ez = prism_edges(zone)
            fig.add_trace(go.Scatter3d(
                x=ex,
                y=ey,
                z=ez,
                mode="lines",
                line=dict(color="#111827", width=1.5),
                hoverinfo="skip",
                showlegend=False,
            ))

    site_size = model.get("metadata", {}).get("site_size", 100.0)
    fig.add_trace(go.Scatter3d(
        x=[0, site_size, site_size, 0, 0],
        y=[0, 0, site_size, site_size, 0],
        z=[0, 0, 0, 0, 0],
        mode="lines",
        line=dict(color="#111827", width=4),
        name="site boundary",
        hoverinfo="skip",
    ))
    fig.update_layout(
        height=760,
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis=dict(title="X (m)", range=[0, site_size], backgroundcolor="#F8FAFC"),
            yaxis=dict(title="Y (m)", range=[0, site_size], backgroundcolor="#F8FAFC"),
            zaxis=dict(title="Z (m)", range=[0, 50], backgroundcolor="#F8FAFC"),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.55),
            camera=dict(eye=dict(x=1.45, y=-1.6, z=1.05)),
        ),
    )
    return fig


def save_json(model: dict, path: str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


st.set_page_config(page_title="Partitioned Zones", layout="wide")
st.title("分区后热区模型")

with st.sidebar:
    perimeter_depth = st.slider("外围分区深度 m", 1.0, 8.0, 4.0, 0.5)
    opacity = st.slider("透明度", 0.15, 1.0, 0.62, 0.05)
    show_edges = st.checkbox("显示边线", True)
    output_path = st.text_input("输出 JSON", DEFAULT_OUTPUT)
    visible_categories = set(st.multiselect(
        "显示类别",
        ["interior_zone", "perimeter_zone", "horizontal_exposed_zone", "aerial_platform_zone"],
        default=["interior_zone", "perimeter_zone", "horizontal_exposed_zone", "aerial_platform_zone"],
    ))

raw_model = generate_20260528()
partitioned = partition_model_by_floor(raw_model, perimeter_depth=perimeter_depth)
partition = partitioned["metadata"]["partition"]
counts = partition["counts"]

cols = st.columns(4)
cols[0].metric("分区 zones", partition["partition_zone_count"])
cols[1].metric("楼层板", partition["floor_plate_count"])
cols[2].metric("外围 zones", counts.get("perimeter_zone", 0))
cols[3].metric("内部 zones", counts.get("interior_zone", 0))

left, right = st.columns([2, 1])
with left:
    st.plotly_chart(render_model(partitioned, opacity, show_edges, visible_categories), use_container_width=True)

with right:
    st.subheader("类别统计")
    st.dataframe(
        [{"category": key, "count": value} for key, value in sorted(counts.items())],
        hide_index=True,
        use_container_width=True,
    )
    if st.button("保存分区 JSON", use_container_width=True):
        path = save_json(partitioned, output_path)
        st.success(f"已保存到 {path}")
    st.download_button(
        "下载分区 JSON",
        data=json.dumps(partitioned, indent=2, ensure_ascii=False),
        file_name=Path(output_path).name,
        mime="application/json",
        use_container_width=True,
    )
    st.subheader("前 80 个 zone")
    st.dataframe(
        [
            {
                "name": zone["name"],
                "category": zone["category"],
                "floor": zone.get("metadata", {}).get("floor"),
                "exposure": ",".join(zone.get("metadata", {}).get("exposure", [])),
                "x": zone["origin"]["x"],
                "y": zone["origin"]["y"],
                "z": zone["origin"]["z"],
                "l": zone["dimensions"]["length"],
                "w": zone["dimensions"]["width"],
                "h": zone["dimensions"]["height"],
            }
            for zone in partitioned["zones"][:80]
        ],
        hide_index=True,
        use_container_width=True,
    )
