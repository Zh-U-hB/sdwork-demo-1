"""Shared 3D visualization and utility functions for Streamlit apps.

Used by parametric_l_app.py and ga_optimizer_app.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.ep_sim_utils import MASS_HEIGHT_THRESHOLD


def box_vertices(zone: dict) -> tuple[list[float], list[float], list[float]]:
    if "points" in zone:
        points = zone["points"]
        xs = [point["x"] for point in points]
        ys = [point["y"] for point in points]
        zs = [point["z"] for point in points]
        # Some generators may store extruded polygons as 2n points (bottom+top),
        # which are not compatible with the cube Mesh3d indices below. In that
        # case, fall back to rendering the zone's bounding box.
        if len(points) != 8:
            x0, x1 = min(xs), max(xs)
            y0, y1 = min(ys), max(ys)
            z0, z1 = min(zs), max(zs)
            vertices = [
                (x0, y0, z0),
                (x1, y0, z0),
                (x1, y1, z0),
                (x0, y1, z0),
                (x0, y0, z1),
                (x1, y0, z1),
                (x1, y1, z1),
                (x0, y1, z1),
            ]
            x, y, z = zip(*vertices)
            return list(x), list(y), list(z)
        return (xs, ys, zs)

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


def window_vertices(window: dict) -> tuple[list[float], list[float], list[float]] | None:
    points = window.get("vertices") or []
    if len(points) != 4:
        return None
    return (
        [point["x"] for point in points],
        [point["y"] for point in points],
        [point["z"] for point in points],
    )


def window_edges(window: dict) -> tuple[list[float], list[float], list[float]] | None:
    vertices = window_vertices(window)
    if vertices is None:
        return None
    x, y, z = vertices
    ex, ey, ez = [], [], []
    for start, end in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        ex.extend([x[start], x[end], None])
        ey.extend([y[start], y[end], None])
        ez.extend([z[start], z[end], None])
    return ex, ey, ez


def quad_vertices(item: dict) -> tuple[list[float], list[float], list[float]] | None:
    points = item.get("vertices") or []
    if len(points) != 4:
        return None
    return (
        [point["x"] for point in points],
        [point["y"] for point in points],
        [point["z"] for point in points],
    )


def quad_edges(item: dict) -> tuple[list[float], list[float], list[float]] | None:
    vertices = quad_vertices(item)
    if vertices is None:
        return None
    x, y, z = vertices
    ex, ey, ez = [], [], []
    for start, end in [(0, 1), (1, 2), (2, 3), (3, 0)]:
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
    window_x: list[float] = []
    window_y: list[float] = []
    window_z: list[float] = []
    window_i: list[int] = []
    window_j: list[int] = []
    window_k: list[int] = []
    window_edge_x: list[float] = []
    window_edge_y: list[float] = []
    window_edge_z: list[float] = []
    shading_x: list[float] = []
    shading_y: list[float] = []
    shading_z: list[float] = []
    shading_i: list[int] = []
    shading_j: list[int] = []
    shading_k: list[int] = []
    shading_edge_x: list[float] = []
    shading_edge_y: list[float] = []
    shading_edge_z: list[float] = []

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
                f"<br>equipment: {energy.get('equipment_gj', 0):.2f} GJ"
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

        for window in zone.get("windows", []):
            vertices = quad_vertices(window)
            if vertices is None:
                continue
            wx, wy, wz = vertices
            base = len(window_x)
            window_x.extend(wx)
            window_y.extend(wy)
            window_z.extend(wz)
            window_i.extend([base, base])
            window_j.extend([base + 1, base + 2])
            window_k.extend([base + 2, base + 3])
            if show_edges:
                edges = quad_edges(window)
                if edges is not None:
                    ex, ey, ez = edges
                    window_edge_x.extend(ex)
                    window_edge_y.extend(ey)
                    window_edge_z.extend(ez)

        for shading in zone.get("shading_surfaces", []):
            vertices = quad_vertices(shading)
            if vertices is None:
                continue
            sx, sy, sz = vertices
            base = len(shading_x)
            shading_x.extend(sx)
            shading_y.extend(sy)
            shading_z.extend(sz)
            shading_i.extend([base, base])
            shading_j.extend([base + 1, base + 2])
            shading_k.extend([base + 2, base + 3])
            if show_edges:
                edges = quad_edges(shading)
                if edges is not None:
                    ex, ey, ez = edges
                    shading_edge_x.extend(ex)
                    shading_edge_y.extend(ey)
                    shading_edge_z.extend(ez)

    if window_x:
        fig.add_trace(go.Mesh3d(
            x=window_x,
            y=window_y,
            z=window_z,
            i=window_i,
            j=window_j,
            k=window_k,
            color="#22D3EE",
            opacity=0.92,
            name="windows",
            hoverinfo="skip",
            showscale=False,
            showlegend=True,
        ))
        if show_edges and window_edge_x:
            fig.add_trace(go.Scatter3d(
                x=window_edge_x,
                y=window_edge_y,
                z=window_edge_z,
                mode="lines",
                line=dict(color="#075985", width=1.2),
                name="window edges",
                hoverinfo="skip",
                showlegend=False,
            ))
    if shading_x:
        fig.add_trace(go.Mesh3d(
            x=shading_x,
            y=shading_y,
            z=shading_z,
            i=shading_i,
            j=shading_j,
            k=shading_k,
            color="#FBBF24",
            opacity=0.72,
            name="overhang shading",
            hoverinfo="skip",
            showscale=False,
            showlegend=True,
        ))
        if show_edges and shading_edge_x:
            fig.add_trace(go.Scatter3d(
                x=shading_edge_x,
                y=shading_edge_y,
                z=shading_edge_z,
                mode="lines",
                line=dict(color="#92400E", width=1.2),
                name="shading edges",
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


def render_monthly_eui_chart(monthly_eui: list[dict]) -> go.Figure:
    fig = go.Figure()
    months = [row.get("month", "") for row in monthly_eui]
    for key, label, color in [
        ("equipment", "Equip", "#6B7280"),
        ("light", "Light", "#FDE047"),
        ("heat", "Heat", "#EF4444"),
        ("cool", "Cool", "#38A3D8"),
    ]:
        fig.add_trace(go.Bar(
            x=months,
            y=[row.get(key, 0.0) for row in monthly_eui],
            name=label,
            marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y:.2f}} kWh/m²<extra></extra>",
        ))
    fig.update_layout(
        title="Energy Use Intensity",
        barmode="stack",
        height=420,
        margin=dict(l=10, r=10, t=48, b=10),
        yaxis_title="[kWh/m²]",
        xaxis_title="",
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
    )
    return fig


def render_comfort_weather_chart(comfort: dict) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.12,
        row_heights=[0.62, 0.38],
        specs=[[{"secondary_y": True}], [{"secondary_y": False}]],
    )
    weather_week = comfort.get("first_week_hourly", [])
    zone_week = comfort.get("zone_first_week_hourly", [])
    x_week = [row.get("label", "") for row in weather_week] or [row.get("label", "") for row in zone_week]

    if weather_week:
        fig.add_trace(go.Scatter(
            x=x_week,
            y=[row.get("dry_bulb_c", 0.0) for row in weather_week],
            name="T-Ext",
            mode="lines",
            line=dict(color="#EF4444", width=1.5),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=x_week,
            y=[row.get("relative_humidity", 0.0) for row in weather_week],
            name="Relative Humidity Ext",
            mode="lines",
            line=dict(color="#93C5FD", width=1.2),
            opacity=0.62,
        ), row=1, col=1, secondary_y=True)

    if zone_week:
        x_zone = [row.get("label", "") for row in zone_week]
        for key, label, color in [
            ("operative", "T-Operative", "#DC2626"),
            ("air", "T-Air", "#F97316"),
            ("radiant", "T-MRT", "#F59E0B"),
        ]:
            vals = [row.get(key) for row in zone_week]
            if any(v is not None for v in vals):
                fig.add_trace(go.Scatter(
                    x=x_zone,
                    y=vals,
                    name=label,
                    mode="lines",
                    line=dict(color=color, width=1.4),
                ), row=1, col=1)

    annual = comfort.get("annual_daily", [])
    if annual:
        fig.add_trace(go.Scatter(
            x=[row.get("day_index", idx + 1) for idx, row in enumerate(annual)],
            y=[row.get("dry_bulb_c", 0.0) for row in annual],
            name="Annual T-Ext",
            mode="lines",
            line=dict(color="#F87171", width=1),
            showlegend=False,
        ), row=2, col=1)

    fig.update_layout(
        title="Outdoor / Zone Temperature and Humidity",
        height=560,
        margin=dict(l=10, r=10, t=48, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="Temperature [°C]", row=1, col=1)
    fig.update_yaxes(title_text="Relative Humidity [%]", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="T [°C]", row=2, col=1)
    fig.update_xaxes(title_text="First week", row=1, col=1)
    fig.update_xaxes(title_text="Day of year", row=2, col=1)
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
        ("equipment_gj", "Equipment", "#6B7280"),
    ]:
        fig.add_trace(go.Bar(
            x=[row["model_zone"] for row in zone_rows],
            y=[row.get(key, 0.0) for row in zone_rows],
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
