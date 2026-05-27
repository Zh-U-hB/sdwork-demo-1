"""Shared 3D visualization and utility functions for Streamlit apps.

Used by parametric_l_app.py and ga_optimizer_app.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go

from scripts.ep_sim_utils import MASS_HEIGHT_THRESHOLD


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


def model_metrics(model: dict, gross_area_fn=None) -> dict:
    """Compute model metrics. gross_area_fn is optional for backward compat."""
    zones = model["zones"]
    mass_zones = [
        z for z in zones
        if z["dimensions"]["height"] > MASS_HEIGHT_THRESHOLD and z.get("category") != "open_space_reference"
    ]
    if gross_area_fn:
        area = gross_area_fn(model)
    else:
        area = sum(
            z["dimensions"]["length"] * z["dimensions"]["width"]
            for z in mass_zones
        )
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
        is_reference = zone.get("category") == "open_space_reference" or zone["name"] == "site_inner_courtyard_reference"
        x, y, z = box_vertices(zone)
        floor_index = zone_floor_index(zone["name"])
        category = zone.get("category", "")
        color = "#94A3B8" if is_reference else "#2563EB"
        if category == "platform":
            color = "#F97316"
        elif category in ("mass_block", "support_mass"):
            pass  # keep default blue
        elif floor_index >= 9:
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
            opacity=0.28 if is_reference else opacity,
            name=zone["name"],
            hovertemplate=hover + "<extra></extra>",
        )
        if has_energy and not is_reference:
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
                line=dict(color="#111827" if not is_reference else "#64748B", width=2),
                hoverinfo="skip",
                showlegend=False,
            ))

    fig.add_trace(go.Scatter3d(
        x=[0, site_size, site_size, 0, 0],
        y=[0, 0, site_size, site_size, 0],
        z=[0, 0, 0, 0, 0],
        mode="lines",
        line=dict(color="#DC2626", width=5),
        name="site boundary",
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
