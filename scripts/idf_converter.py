"""Programmatic JSON → IDF converter and EnergyPlus runner.

Uses **idfpy** (https://github.com/ITOTI-Y/idfpy) — type-safe Pydantic models
for all EnergyPlus IDF objects.  No IDD file required at runtime.

Converts a BuildingModel zone JSON dict directly to an EnergyPlus IDF file
and optionally runs the simulation — no LLM, no MCP, pure Python.

Supports both:
- Rectangular box zones (origin + dimensions)
- Extruded polygon zones via `floor_polygon` (trapezoid / L-shape / etc.)
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

from scripts.idf_defaults import ConverterDefaults, ScheduleDef, make_default_settings

# ---------------------------------------------------------------------------
# idfpy imports
# ---------------------------------------------------------------------------
from idfpy import IDF
from idfpy.models import (
    Building,
    BuildingSurfaceDetailed,
    Construction,
    FenestrationSurfaceDetailed,
    GlobalGeometryRules,
    HVACTemplateThermostat,
    HVACTemplateZoneIdealLoadsAirSystem,
    Lights,
    Material,
    OutputDiagnostics,
    OutputTableSummaryReports,
    OutputVariable,
    OutputVariableDictionary,
    People,
    RunPeriod,
    ScheduleCompact,
    ScheduleTypeLimits,
    SimulationControl,
    SiteLocation,
    Timestep,
    WindowMaterialSimpleGlazingSystem,
    Zone,
)
from idfpy.models.outputs import (
    OutputDiagnosticsDiagnosticsItem,
    OutputTableSummaryReportsReportsItem,
)
from idfpy.models.schedules import ScheduleCompactDataItem
from idfpy.models.thermal_zones import BuildingSurfaceDetailedVerticesItem
from idfpy.sim import simulate

# ---------------------------------------------------------------------------
# ASCII name sanitisation
# ---------------------------------------------------------------------------

_NON_ASCII = re.compile(r"[^\x00-\x7f]")
_UNSAFE = re.compile(r"[^A-Za-z0-9_\-]")


def _ascii_name(name: str, index: int, prefix: str = "Zone") -> str:
    """Return a short ASCII-safe IDF name for *name*."""
    if _NON_ASCII.search(name):
        return f"{prefix}_{index:02d}"
    # EnergyPlus object names must be unique. Many long names can collide after truncation,
    # especially for partitioned zones. Always append an index suffix and truncate the base
    # to keep the total length stable.
    suffix = f"_{index:03d}"
    max_len = 48
    base_len = max(1, max_len - len(suffix))
    safe = _UNSAFE.sub("_", name)
    safe = safe[:base_len]
    safe = safe.rstrip("_")
    return (safe or prefix) + suffix


# ---------------------------------------------------------------------------
# Surface vertex helpers
# ---------------------------------------------------------------------------
# EnergyPlus GlobalGeometryRules: UpperLeftCorner + CounterClockWise + World.
# Vertices are ordered CCW when viewed from OUTSIDE.

Vec3 = tuple[float, float, float]


def _verts(pts: list[Vec3]) -> list[BuildingSurfaceDetailedVerticesItem]:
    return [
        BuildingSurfaceDetailedVerticesItem(
            vertex_x_coordinate=x,
            vertex_y_coordinate=y,
            vertex_z_coordinate=z,
        )
        for x, y, z in pts
    ]


def _floor_pts(ox: float, oy: float, oz: float, L: float, W: float) -> list[Vec3]:
    """Floor at z=oz, outward normal = −Z (CCW viewed from below)."""
    return [(ox, oy + W, oz), (ox + L, oy + W, oz), (ox + L, oy, oz), (ox, oy, oz)]


def _ceiling_pts(ox: float, oy: float, oz: float, L: float, W: float, H: float) -> list[Vec3]:
    """Ceiling at z=oz+H, outward normal = +Z (CCW viewed from above)."""
    return [(ox, oy, oz + H), (ox + L, oy, oz + H), (ox + L, oy + W, oz + H), (ox, oy + W, oz + H)]


def _south_pts(ox: float, oy: float, oz: float, L: float, H: float) -> list[Vec3]:
    """South wall at y=oy, outward normal = −Y."""
    return [(ox, oy, oz), (ox + L, oy, oz), (ox + L, oy, oz + H), (ox, oy, oz + H)]


def _north_pts(ox: float, oy: float, oz: float, L: float, W: float, H: float) -> list[Vec3]:
    """North wall at y=oy+W, outward normal = +Y."""
    return [(ox + L, oy + W, oz), (ox, oy + W, oz), (ox, oy + W, oz + H), (ox + L, oy + W, oz + H)]


def _west_pts(ox: float, oy: float, oz: float, W: float, H: float) -> list[Vec3]:
    """West wall at x=ox, outward normal = −X."""
    return [(ox, oy + W, oz), (ox, oy, oz), (ox, oy, oz + H), (ox, oy + W, oz + H)]


def _east_pts(ox: float, oy: float, oz: float, L: float, W: float, H: float) -> list[Vec3]:
    """East wall at x=ox+L, outward normal = +X."""
    return [(ox + L, oy, oz), (ox + L, oy + W, oz), (ox + L, oy + W, oz + H), (ox + L, oy, oz + H)]


def _window_pts(wall_pts: list[Vec3], wwr: float) -> list[Vec3] | None:
    """Return inset vertices for a centred rectangular window on an exterior wall.

    Assumes the wall is axis-aligned and rectangular.
    """
    if wwr <= 0.0:
        return None
    xs = [v[0] for v in wall_pts]
    ys = [v[1] for v in wall_pts]
    zs = [v[2] for v in wall_pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_min, z_max = min(zs), max(zs)
    wall_w = max(x_max - x_min, y_max - y_min)
    wall_h = z_max - z_min
    if wall_w < 0.5 or wall_h < 0.5:
        return None
    win_w = wall_w * wwr
    win_h = wall_h * wwr
    mw = (wall_w - win_w) / 2
    mh = (wall_h - win_h) / 2
    z0 = z_min + mh
    z1 = z0 + win_h
    if x_max - x_min > y_max - y_min:  # wall spans X (south or north)
        x0, x1 = x_min + mw, x_min + mw + win_w
        y_val = (y_min + y_max) / 2
        if wall_pts[0][0] < wall_pts[1][0]:
            return [(x0, y_val, z0), (x1, y_val, z0), (x1, y_val, z1), (x0, y_val, z1)]
        return [(x1, y_val, z0), (x0, y_val, z0), (x0, y_val, z1), (x1, y_val, z1)]
    # wall spans Y (west or east)
    y0, y1 = y_min + mw, y_min + mw + win_w
    x_val = (x_min + x_max) / 2
    if wall_pts[0][1] > wall_pts[1][1]:
        return [(x_val, y1, z0), (x_val, y0, z0), (x_val, y0, z1), (x_val, y1, z1)]
    return [(x_val, y0, z0), (x_val, y1, z0), (x_val, y1, z1), (x_val, y0, z1)]


# ---------------------------------------------------------------------------
# Sub-surface vertex helpers (for split surfaces)
# ---------------------------------------------------------------------------

def _sub_floor_pts(x0: float, y0: float, x1: float, y1: float, z: float) -> list[Vec3]:
    """Floor at z, outward normal = −Z (CCW viewed from below)."""
    return [(x0, y1, z), (x1, y1, z), (x1, y0, z), (x0, y0, z)]


def _sub_top_pts(x0: float, y0: float, x1: float, y1: float, z: float) -> list[Vec3]:
    """Top surface at z, outward normal = +Z (CCW viewed from above)."""
    return [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]


def _sub_wall_pts(direction: str, fixed: float, p0: float, p1: float, z0: float, z1: float) -> list[Vec3]:
    """Wall quad for a sub-rectangle on a wall plane."""
    if direction == "South":
        return [(p0, fixed, z0), (p1, fixed, z0), (p1, fixed, z1), (p0, fixed, z1)]
    if direction == "North":
        return [(p1, fixed, z0), (p0, fixed, z0), (p0, fixed, z1), (p1, fixed, z1)]
    if direction == "West":
        return [(fixed, p1, z0), (fixed, p0, z0), (fixed, p0, z1), (fixed, p1, z1)]
    # East
    return [(fixed, p0, z0), (fixed, p1, z0), (fixed, p1, z1), (fixed, p0, z1)]


def _wall_fixed_coord(box: "_ZoneBox", direction: str) -> float:
    if direction == "South":
        return box.oy
    if direction == "North":
        return box.y_max
    if direction == "West":
        return box.ox
    return box.x_max  # East


# ---------------------------------------------------------------------------
# Zone helpers
# ---------------------------------------------------------------------------


class _ZoneBox(NamedTuple):
    ascii_name: str
    ox: float
    oy: float
    oz: float
    L: float
    W: float
    H: float

    @property
    def x_max(self) -> float:
        return self.ox + self.L

    @property
    def y_max(self) -> float:
        return self.oy + self.W

    @property
    def z_max(self) -> float:
        return self.oz + self.H


class _ZonePoly(NamedTuple):
    ascii_name: str
    oz: float
    H: float
    floor_pts: list[tuple[float, float]]  # CCW from above

    @property
    def z_max(self) -> float:
        return self.oz + self.H


def _as_poly(zone_dict: dict, ascii_name: str) -> _ZonePoly | None:
    poly = zone_dict.get("floor_polygon")
    if not poly:
        return None
    pts: list[tuple[float, float]] = [(float(p["x"]), float(p["y"])) for p in poly]
    if len(pts) < 3:
        return None
    o = zone_dict["origin"]
    d = zone_dict["dimensions"]
    return _ZonePoly(ascii_name=ascii_name, oz=float(o["z"]), H=float(d["height"]), floor_pts=pts)


# ---------------------------------------------------------------------------
# Rectangle subtraction utilities (axis-aligned)
# ---------------------------------------------------------------------------

Rect2D = tuple[float, float, float, float]  # (u0, v0, u1, v1)


def _rect_subtract(rect: Rect2D, clip: Rect2D, tol: float = 0.01) -> list[Rect2D]:
    """Subtract *clip* from *rect* (both axis-aligned)."""
    r0, r1, r2, r3 = rect
    c0, c1, c2, c3 = clip
    if c0 >= r2 - tol or c2 <= r0 + tol or c1 >= r3 - tol or c3 <= r1 + tol:
        return [rect]
    result: list[Rect2D] = []
    if c0 > r0 + tol:
        result.append((r0, r1, c0, r3))
    if c2 < r2 - tol:
        result.append((c2, r1, r2, r3))
    if c1 > r1 + tol:
        result.append((max(r0, c0), r1, min(r2, c2), c1))
    if c3 < r3 - tol:
        result.append((max(r0, c0), c3, min(r2, c2), r3))
    return [r for r in result if (r[2] - r[0] > tol and r[3] - r[1] > tol)]


def _rect_subtract_multi(rect: Rect2D, clips: list[Rect2D], tol: float = 0.01) -> list[Rect2D]:
    remaining = [rect]
    for c in clips:
        nxt: list[Rect2D] = []
        for r in remaining:
            nxt.extend(_rect_subtract(r, c, tol))
        remaining = nxt
        if not remaining:
            break
    return remaining


# ---------------------------------------------------------------------------
# Adjacency detection and split-surface planning (box zones)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _VertPair:
    lower_idx: int
    upper_idx: int
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class _HorizPair:
    zone_a_idx: int
    dir_a: str
    zone_b_idx: int
    dir_b: str
    p0: float
    p1: float
    z0: float
    z1: float


def _detect_vertical_pairs(boxes: list[_ZoneBox], tol: float = 0.01) -> list[_VertPair]:
    pairs: list[_VertPair] = []
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes):
            if i == j:
                continue
            if abs(b.oz - a.z_max) > tol:
                continue
            x0 = max(a.ox, b.ox)
            y0 = max(a.oy, b.oy)
            x1 = min(a.x_max, b.x_max)
            y1 = min(a.y_max, b.y_max)
            if x1 - x0 > tol and y1 - y0 > tol:
                pairs.append(_VertPair(i, j, round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)))
    return pairs


def _detect_horizontal_pairs(boxes: list[_ZoneBox], tol: float = 0.01) -> list[_HorizPair]:
    pairs: list[_HorizPair] = []
    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]

            def _append(ai: int, ad: str, bi: int, bd: str, p0: float, p1: float, z0: float, z1: float) -> None:
                if p1 - p0 > tol and z1 - z0 > tol:
                    pairs.append(_HorizPair(ai, ad, bi, bd, round(p0, 3), round(p1, 3), round(z0, 3), round(z1, 3)))

            # East/West contact
            if abs(a.x_max - b.ox) < tol:
                _append(i, "East", j, "West", max(a.oy, b.oy), min(a.y_max, b.y_max), max(a.oz, b.oz), min(a.z_max, b.z_max))
            if abs(b.x_max - a.ox) < tol:
                _append(j, "East", i, "West", max(a.oy, b.oy), min(a.y_max, b.y_max), max(a.oz, b.oz), min(a.z_max, b.z_max))
            # North/South contact
            if abs(a.y_max - b.oy) < tol:
                _append(i, "North", j, "South", max(a.ox, b.ox), min(a.x_max, b.x_max), max(a.oz, b.oz), min(a.z_max, b.z_max))
            if abs(b.y_max - a.oy) < tol:
                _append(j, "North", i, "South", max(a.ox, b.ox), min(a.x_max, b.x_max), max(a.oz, b.oz), min(a.z_max, b.z_max))
    return pairs


@dataclass
class _SurfaceSpec:
    name: str
    surface_type: str
    construction: str
    zone: str
    boundary: str
    boundary_object: str
    sun: str
    wind: str
    pts: list[Vec3]
    is_wall: bool


def _build_split_surfaces(boxes: list[_ZoneBox], tol: float = 0.01) -> list[_SurfaceSpec]:
    """Compute split surfaces for all box zones.

    Requirement mapping
    -------------------
    - Covered portions of a lower zone's top: `Ceiling` + `Surface` boundary.
    - Uncovered portions of a lower zone's top: `Roof` + `Outdoors`.
    - Covered portions of an upper zone's bottom: `Floor` + `Surface`.
    - Uncovered portions of an upper zone's bottom: `Floor` + `Outdoors` (when oz>0).
    - Partially overlapped wall areas: `Wall` + `Surface` for overlap, `Wall` + `Outdoors` for remainder.
    """
    specs: list[_SurfaceSpec] = []
    vert_pairs = _detect_vertical_pairs(boxes, tol=tol)
    horiz_pairs = _detect_horizontal_pairs(boxes, tol=tol)

    int_wall = "InteriorWallConstruction"
    int_ceil = "InteriorCeilingConstruction"
    ext_wall = "ExteriorWallConstruction"
    ext_roof = "ExteriorRoofConstruction"
    floor_con = "FloorConstruction"

    top_clips: dict[int, list[Rect2D]] = {}
    bottom_clips: dict[int, list[Rect2D]] = {}
    wall_clips: dict[tuple[int, str], list[Rect2D]] = {}

    # Vertical interior pairs
    for k, vp in enumerate(vert_pairs):
        lower = boxes[vp.lower_idx]
        upper = boxes[vp.upper_idx]
        rect_xy = (vp.x0, vp.y0, vp.x1, vp.y1)
        top_clips.setdefault(vp.lower_idx, []).append(rect_xy)
        bottom_clips.setdefault(vp.upper_idx, []).append(rect_xy)

        ceil_name = f"{lower.ascii_name}_Ceiling_V{k:03d}"
        floor_name = f"{upper.ascii_name}_Floor_V{k:03d}"

        specs.append(_SurfaceSpec(
            name=ceil_name,
            surface_type="Ceiling",
            construction=int_ceil,
            zone=lower.ascii_name,
            boundary="Surface",
            boundary_object=floor_name,
            sun="NoSun",
            wind="NoWind",
            pts=_sub_top_pts(vp.x0, vp.y0, vp.x1, vp.y1, lower.z_max),
            is_wall=False,
        ))
        specs.append(_SurfaceSpec(
            name=floor_name,
            surface_type="Floor",
            construction=floor_con,
            zone=upper.ascii_name,
            boundary="Surface" if upper.oz >= tol else "Ground",
            boundary_object=ceil_name if upper.oz >= tol else "",
            sun="NoSun",
            wind="NoWind",
            pts=_sub_floor_pts(vp.x0, vp.y0, vp.x1, vp.y1, upper.oz),
            is_wall=False,
        ))

    # Horizontal interior wall pairs
    for k, hp in enumerate(horiz_pairs):
        za = boxes[hp.zone_a_idx]
        zb = boxes[hp.zone_b_idx]
        rect_pz = (hp.p0, hp.z0, hp.p1, hp.z1)
        wall_clips.setdefault((hp.zone_a_idx, hp.dir_a), []).append(rect_pz)
        wall_clips.setdefault((hp.zone_b_idx, hp.dir_b), []).append(rect_pz)

        name_a = f"{za.ascii_name}_Wall_{hp.dir_a}_H{k:03d}"
        name_b = f"{zb.ascii_name}_Wall_{hp.dir_b}_H{k:03d}"
        fixed_a = _wall_fixed_coord(za, hp.dir_a)
        fixed_b = _wall_fixed_coord(zb, hp.dir_b)

        specs.append(_SurfaceSpec(
            name=name_a,
            surface_type="Wall",
            construction=int_wall,
            zone=za.ascii_name,
            boundary="Surface",
            boundary_object=name_b,
            sun="NoSun",
            wind="NoWind",
            pts=_sub_wall_pts(hp.dir_a, fixed_a, hp.p0, hp.p1, hp.z0, hp.z1),
            is_wall=True,
        ))
        specs.append(_SurfaceSpec(
            name=name_b,
            surface_type="Wall",
            construction=int_wall,
            zone=zb.ascii_name,
            boundary="Surface",
            boundary_object=name_a,
            sun="NoSun",
            wind="NoWind",
            pts=_sub_wall_pts(hp.dir_b, fixed_b, hp.p0, hp.p1, hp.z0, hp.z1),
            is_wall=True,
        ))

    # Exterior remainder per zone
    for idx, box in enumerate(boxes):
        zn = box.ascii_name
        ox, oy, oz = box.ox, box.oy, box.oz
        L, W, H = box.L, box.W, box.H
        x0, y0, x1, y1 = ox, oy, box.x_max, box.y_max

        # Bottom floor
        full = (x0, y0, x1, y1)
        clips = bottom_clips.get(idx, [])
        if abs(oz) < tol:
            if not clips:
                specs.append(_SurfaceSpec(
                    name=f"{zn}_Floor",
                    surface_type="Floor",
                    construction=floor_con,
                    zone=zn,
                    boundary="Ground",
                    boundary_object="",
                    sun="NoSun",
                    wind="NoWind",
                    pts=_floor_pts(ox, oy, oz, L, W),
                    is_wall=False,
                ))
        else:
            exposed = _rect_subtract_multi(full, clips, tol=tol)
            for ri, r in enumerate(exposed):
                rx0, ry0, rx1, ry1 = r
                specs.append(_SurfaceSpec(
                    name=f"{zn}_Floor_E{ri:03d}" if clips else f"{zn}_Floor",
                    surface_type="Floor",
                    construction=floor_con,
                    zone=zn,
                    boundary="Outdoors",
                    boundary_object="",
                    sun="NoSun",
                    wind="NoWind",
                    pts=_sub_floor_pts(rx0, ry0, rx1, ry1, oz) if clips else _floor_pts(ox, oy, oz, L, W),
                    is_wall=False,
                ))

        # Top roof remainder (covered parts handled by interior ceilings above)
        clips_top = top_clips.get(idx, [])
        exposed_top = _rect_subtract_multi(full, clips_top, tol=tol)
        for ri, r in enumerate(exposed_top):
            rx0, ry0, rx1, ry1 = r
            specs.append(_SurfaceSpec(
                name=f"{zn}_Roof_{ri:03d}" if clips_top else f"{zn}_Roof",
                surface_type="Roof",
                construction=ext_roof,
                zone=zn,
                boundary="Outdoors",
                boundary_object="",
                sun="SunExposed",
                wind="WindExposed",
                pts=_sub_top_pts(rx0, ry0, rx1, ry1, box.z_max),
                is_wall=False,
            ))

        # Walls
        wall_defs: list[tuple[str, float, Rect2D]] = [
            ("South", box.oy, (x0, oz, x1, box.z_max)),
            ("North", box.y_max, (x0, oz, x1, box.z_max)),
            ("West", box.ox, (y0, oz, y1, box.z_max)),
            ("East", box.x_max, (y0, oz, y1, box.z_max)),
        ]
        for direction, fixed, full_w in wall_defs:
            clips_w = wall_clips.get((idx, direction), [])
            exposed_w = _rect_subtract_multi(full_w, clips_w, tol=tol)
            for ri, r in enumerate(exposed_w):
                p0, z0, p1, z1 = r
                specs.append(_SurfaceSpec(
                    name=f"{zn}_Wall_{direction}_{ri:03d}" if clips_w else f"{zn}_Wall_{direction}",
                    surface_type="Wall",
                    construction=ext_wall,
                    zone=zn,
                    boundary="Outdoors",
                    boundary_object="",
                    sun="SunExposed",
                    wind="WindExposed",
                    pts=_sub_wall_pts(direction, fixed, p0, p1, z0, z1) if clips_w else (
                        _south_pts(ox, oy, oz, L, H) if direction == "South"
                        else _north_pts(ox, oy, oz, L, W, H) if direction == "North"
                        else _west_pts(ox, oy, oz, W, H) if direction == "West"
                        else _east_pts(ox, oy, oz, L, W, H)
                    ),
                    is_wall=True,
                ))

    return specs


# ---------------------------------------------------------------------------
# IDF builder helpers
# ---------------------------------------------------------------------------


def _make_surface(
    name: str,
    surface_type: str,
    construction: str,
    zone: str,
    boundary: str,
    sun: str,
    wind: str,
    pts: list[Vec3],
    boundary_object: str = "",
) -> BuildingSurfaceDetailed:
    return BuildingSurfaceDetailed(
        name=name,
        surface_type=surface_type,
        construction_name=construction,
        zone_name=zone,
        outside_boundary_condition=boundary,
        outside_boundary_condition_object=boundary_object or None,
        sun_exposure=sun,
        wind_exposure=wind,
        view_factor_to_ground="autocalculate",
        vertices=_verts(pts),
    )


def _make_window(name: str, construction: str, wall_name: str, pts: list[Vec3]) -> FenestrationSurfaceDetailed:
    if len(pts) != 4:
        raise ValueError("Window must have exactly 4 vertices")
    return FenestrationSurfaceDetailed(
        name=name,
        surface_type="Window",
        construction_name=construction,
        building_surface_name=wall_name,
        multiplier=1,
        number_of_vertices=4,
        vertex_1_x_coordinate=pts[0][0],
        vertex_1_y_coordinate=pts[0][1],
        vertex_1_z_coordinate=pts[0][2],
        vertex_2_x_coordinate=pts[1][0],
        vertex_2_y_coordinate=pts[1][1],
        vertex_2_z_coordinate=pts[1][2],
        vertex_3_x_coordinate=pts[2][0],
        vertex_3_y_coordinate=pts[2][1],
        vertex_3_z_coordinate=pts[2][2],
        vertex_4_x_coordinate=pts[3][0],
        vertex_4_y_coordinate=pts[3][1],
        vertex_4_z_coordinate=pts[3][2],
    )


def _make_schedule_compact(sched: ScheduleDef) -> ScheduleCompact:
    return ScheduleCompact(
        name=sched.name,
        schedule_type_limits_name=sched.type_limits_name,
        data=[ScheduleCompactDataItem(field=entry) for entry in sched.data],
    )


def _poly_floor(poly: _ZonePoly) -> list[Vec3]:
    # input is CCW from above (+Z); floor outward is -Z, so reverse
    z = poly.oz
    return [(x, y, z) for (x, y) in reversed(poly.floor_pts)]


def _poly_roof(poly: _ZonePoly) -> list[Vec3]:
    z = poly.z_max
    return [(x, y, z) for (x, y) in poly.floor_pts]


def _poly_wall(poly: _ZonePoly, i0: int) -> list[Vec3]:
    n = len(poly.floor_pts)
    i1 = (i0 + 1) % n
    (x0, y0) = poly.floor_pts[i0]
    (x1, y1) = poly.floor_pts[i1]
    z0 = poly.oz
    z1 = poly.z_max
    return [(x0, y0, z0), (x1, y1, z0), (x1, y1, z1), (x0, y0, z1)]


def _build_zone_poly(idf: IDF, poly: _ZonePoly, defaults: ConverterDefaults) -> None:
    """Add Zone + exterior-only surfaces for one extruded polygon zone."""
    zn = poly.ascii_name
    idf.add(Zone(name=zn))

    ext_wall = "ExteriorWallConstruction"
    ext_roof = "ExteriorRoofConstruction"
    floor_con = "FloorConstruction"
    win_con = defaults.window.construction_name
    wwr = defaults.window.wwr

    # Floor boundary
    floor_boundary = "Ground" if abs(poly.oz) < 0.01 else "Outdoors"
    idf.add(_make_surface(
        name=f"{zn}_Floor",
        surface_type="Floor",
        construction=floor_con,
        zone=zn,
        boundary=floor_boundary,
        sun="NoSun",
        wind="NoWind",
        pts=_poly_floor(poly),
    ))

    # Roof
    idf.add(_make_surface(
        name=f"{zn}_Roof",
        surface_type="Roof",
        construction=ext_roof,
        zone=zn,
        boundary="Outdoors",
        sun="SunExposed",
        wind="WindExposed",
        pts=_poly_roof(poly),
    ))

    # Walls per edge
    for i in range(len(poly.floor_pts)):
        pts = _poly_wall(poly, i)
        sname = f"{zn}_Wall_E{i:02d}"
        idf.add(_make_surface(
            name=sname,
            surface_type="Wall",
            construction=ext_wall,
            zone=zn,
            boundary="Outdoors",
            sun="SunExposed",
            wind="WindExposed",
            pts=pts,
        ))

        # Only attempt windows on axis-aligned edges (rect helper assumption)
        dx = abs(pts[1][0] - pts[0][0])
        dy = abs(pts[1][1] - pts[0][1])
        axis_aligned = (dx < 1e-6) or (dy < 1e-6)
        if axis_aligned and wwr > 0.0:
            win_pts = _window_pts(pts, wwr)
            if win_pts:
                idf.add(_make_window(name=f"{sname}_Window", construction=win_con, wall_name=sname, pts=win_pts))


def _add_zone_loads_and_hvac(idf: IDF, zn: str, defaults: ConverterDefaults) -> None:
    ppl = defaults.people
    idf.add(People(
        name=f"{zn}_People",
        zone_or_zonelist_or_space_or_spacelist_name=zn,
        number_of_people_schedule_name=ppl.number_of_people_schedule_name,
        number_of_people_calculation_method="People/Area",
        people_per_floor_area=ppl.people_per_floor_area,
        fraction_radiant=ppl.fraction_radiant,
        sensible_heat_fraction="Autocalculate",
        activity_level_schedule_name=ppl.activity_level_schedule_name,
    ))
    lgt = defaults.lights
    idf.add(Lights(
        name=f"{zn}_Lights",
        zone_or_zonelist_or_space_or_spacelist_name=zn,
        schedule_name=lgt.schedule_name,
        design_level_calculation_method="Watts/Area",
        watts_per_floor_area=lgt.watts_per_floor_area,
        return_air_fraction=lgt.return_air_fraction,
        fraction_radiant=lgt.fraction_radiant,
        fraction_visible=lgt.fraction_visible,
    ))
    idf.add(HVACTemplateZoneIdealLoadsAirSystem(
        zone_name=zn,
        template_thermostat_name=defaults.hvac.thermostat_name,
    ))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

MASS_HEIGHT_THRESHOLD = 1.0  # metres — skip courtyard markers etc.


def convert_and_run(
    model_dict: dict,
    output_dir: str | Path = "output/direct_sim",
    weather_file: str | Path | None = None,
    defaults: ConverterDefaults | None = None,
    run_simulation: bool = True,
    idd_path: str | Path | None = None,  # kept for API compatibility, unused by idfpy
) -> str | None:
    defaults = defaults or make_default_settings()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if weather_file is None:
        from scripts.ep_sim_utils import resolve_weather_path

        weather_file = resolve_weather_path()
    epw_path = Path(weather_file)

    building_name = _ascii_name(model_dict.get("building_name", "Building"), 0, prefix="Building")

    raw_zones = [z for z in model_dict.get("zones", []) if z["dimensions"]["height"] >= MASS_HEIGHT_THRESHOLD]
    if not raw_zones:
        raise ValueError("No valid zones (all below height threshold).")

    boxes: list[_ZoneBox] = []
    polys: list[_ZonePoly] = []
    for idx, z in enumerate(raw_zones, 1):
        aname = _ascii_name(z["name"], idx, prefix="Zone")
        maybe_poly = _as_poly(z, aname)
        if maybe_poly is not None:
            polys.append(maybe_poly)
            continue
        o, d = z["origin"], z["dimensions"]
        boxes.append(_ZoneBox(
            ascii_name=aname,
            ox=float(o["x"]),
            oy=float(o["y"]),
            oz=float(o["z"]),
            L=float(d["length"]),
            W=float(d["width"]),
            H=float(d["height"]),
        ))

    # NOTE: adjacency splitting (shared walls / stacked floors) is currently only implemented
    # for axis-aligned box zones. Polygon zones are exported as exterior-only surfaces.
    # To avoid silently producing physically-wrong boundary conditions, we fail fast when
    # polygon zones coexist with any other zone.
    if polys and (boxes or len(polys) > 1):
        raise ValueError(
            "Polygon zones (floor_polygon) currently do not support shared-surface adjacency "
            "in IDF export. Please avoid mixing polygon zones with adjacent zones, or convert "
            "adjacent zones to box geometry before simulation."
        )

    idf = IDF()

    # Settings & envelope
    idf.add(SimulationControl(
        do_zone_sizing_calculation="No",
        do_system_sizing_calculation="No",
        do_plant_sizing_calculation="No",
        run_simulation_for_sizing_periods="Yes",
        run_simulation_for_weather_file_run_periods="Yes",
        do_hvac_sizing_simulation_for_sizing_periods="Yes",
        maximum_number_of_hvac_sizing_simulation_passes=1,
    ))
    idf.add(Timestep(number_of_timesteps_per_hour=4))
    idf.add(RunPeriod(
        name="FullYear",
        begin_month=1,
        begin_day_of_month=1,
        end_month=12,
        end_day_of_month=31,
    ))
    idf.add(GlobalGeometryRules(
        starting_vertex_position="UpperLeftCorner",
        vertex_entry_direction="CounterClockWise",
        coordinate_system="World",
    ))

    # Output requests
    idf.add(OutputVariableDictionary(key_field="regular"))
    idf.add(OutputDiagnostics(diagnostics=[OutputDiagnosticsDiagnosticsItem(key="DisplayExtraWarnings")]))
    idf.add(OutputTableSummaryReports(reports=[OutputTableSummaryReportsReportsItem(report_name="AllSummary")]))
    for var in [
        "Zone Ideal Loads Heat Recovery Total Heating Energy",
        "Zone Ideal Loads Supply Air Total Cooling Energy",
        "Zone Lights Electricity Energy",
    ]:
        idf.add(OutputVariable(key_value="*", variable_name=var, reporting_frequency="Annual"))

    # Building & location
    idf.add(Building(
        name=building_name,
        north_axis=0.0,
        terrain="Suburbs",
        solar_distribution="FullInteriorAndExterior",
        maximum_number_of_warmup_days=25,
        minimum_number_of_warmup_days=1,
    ))
    loc = defaults.location
    idf.add(SiteLocation(
        name=loc.name,
        latitude=loc.latitude,
        longitude=loc.longitude,
        time_zone=loc.time_zone,
        elevation=loc.elevation,
    ))

    # Materials
    for mat in defaults.opaque_materials:
        idf.add(Material(
            name=mat.name,
            roughness=mat.roughness,
            thickness=mat.thickness,
            conductivity=mat.conductivity,
            density=mat.density,
            specific_heat=mat.specific_heat,
        ))
    for gla in defaults.glazing_materials:
        idf.add(WindowMaterialSimpleGlazingSystem(
            name=gla.name,
            u_factor=gla.u_factor,
            solar_heat_gain_coefficient=gla.solar_heat_gain_coefficient,
            visible_transmittance=gla.visible_transmittance,
        ))

    # Constructions
    for con in defaults.constructions:
        layers = {
            "outside_layer": con.layers[0] if len(con.layers) > 0 else None,
            **{f"layer_{i+2}": con.layers[i + 1] for i in range(len(con.layers) - 1)},
        }
        idf.add(Construction(name=con.name, **layers))

    # Schedule type limits
    for st in defaults.schedule_types:
        idf.add(ScheduleTypeLimits(
            name=st.name,
            lower_limit_value=st.lower_limit,
            upper_limit_value=st.upper_limit,
            numeric_type=st.numeric_type,
            unit_type=st.unit_type,
        ))

    # Schedules
    for sched in defaults.schedules:
        idf.add(_make_schedule_compact(sched))

    # HVAC thermostat
    idf.add(HVACTemplateThermostat(
        name=defaults.hvac.thermostat_name,
        heating_setpoint_schedule_name=defaults.hvac.heating_setpoint_schedule_name,
        cooling_setpoint_schedule_name=defaults.hvac.cooling_setpoint_schedule_name,
    ))

    # ── Geometry ──────────────────────────────────────────────────────────
    # Box zones: add Zone objects first, then split surfaces (ceiling/roof + walls)
    for box in boxes:
        idf.add(Zone(name=box.ascii_name))
    if boxes:
        surface_specs = _build_split_surfaces(boxes)
        wwr = defaults.window.wwr
        win_con = defaults.window.construction_name
        for spec in surface_specs:
            idf.add(_make_surface(
                name=spec.name,
                surface_type=spec.surface_type,
                construction=spec.construction,
                zone=spec.zone,
                boundary=spec.boundary,
                sun=spec.sun,
                wind=spec.wind,
                pts=spec.pts,
                boundary_object=spec.boundary_object,
            ))
            if spec.is_wall and spec.boundary == "Outdoors" and wwr > 0.0:
                win_pts = _window_pts(spec.pts, wwr)
                if win_pts:
                    idf.add(_make_window(
                        name=f"{spec.name}_Window",
                        construction=win_con,
                        wall_name=spec.name,
                        pts=win_pts,
                    ))

    # Polygon zones: exterior-only for now (no splitting)
    for poly in polys:
        _build_zone_poly(idf, poly, defaults)

    # ── Loads & HVAC ───────────────────────────────────────────────────────
    for box in boxes:
        _add_zone_loads_and_hvac(idf, box.ascii_name, defaults)
    for poly in polys:
        _add_zone_loads_and_hvac(idf, poly.ascii_name, defaults)

    # ── Save IDF + model JSON (for viewer) ─────────────────────────────────
    import json as _json

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    idf_path = output_dir / f"building_{timestamp}.idf"
    idf.save(idf_path)

    model_json_path = output_dir / "model.json"
    model_json_path.write_text(_json.dumps(model_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[idf_converter] IDF saved → {idf_path}  "
        f"({len(raw_zones)} zones: {len(boxes)} box, {len(polys)} polygon)"
    )

    if not run_simulation:
        return str(output_dir)

    # ── Run EnergyPlus ─────────────────────────────────────────────────────
    if not epw_path.exists():
        raise FileNotFoundError(f"EPW file not found: {epw_path}")

    results_dir = output_dir / f"results_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    result = simulate(
        idf_path,
        weather=epw_path,
        output_dir=results_dir,
        expand_objects=True,
        echo=False,
    )

    if result.success:
        eplustbl = results_dir / "eplustbl.csv"
        if eplustbl.exists():
            print(f"[idf_converter] Simulation complete → {results_dir}")
            return str(results_dir)
        candidates = list(results_dir.rglob("eplustbl.csv"))
        if candidates:
            return str(candidates[0].parent)

    print(f"[idf_converter] Simulation failed. err: {result.err}")
    return None

