"""EnergyPlus Simulation Result Viewer.

Browses output directories produced by idf_converter / run_ep_simulation_direct,
loads eplustbl.csv + the paired model.json, and renders:

  - Key metrics banner (EUI, total energy, conditioned area, zone count)
  - End-use bar chart & donut chart
  - Zone-level energy stacked bar
  - 3D model with per-zone energy colour mapping
  - Raw data tables (end-uses, zone energy, simulation files)

Run:
    streamlit run sim_viewer_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from scripts.ep_sim_utils import model_energy_map, read_eplustbl
from scripts.vis_utils import (
    model_metrics,
    render_end_use_chart,
    render_end_use_pie,
    render_model,
    render_zone_energy_chart,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="模拟结果查看器",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OUTPUT_ROOT = Path("output")


def _resolve_run_dir(path: Path) -> tuple[Path | None, Path | None]:
    """Resolve a user-supplied path to (result_dir, idf_path).

    Accepted inputs
    ---------------
    * ``run_*`` directory  — e.g. ``output/direct_energyplus/run_20260527_231411``
      Auto-finds the first ``results_*/eplustbl.csv`` inside it and the first ``*.idf``.
    * ``results_*`` directory — e.g. ``…/results_20260527_231411``
      Looks for ``eplustbl.csv`` directly inside; IDF found in parent run dir.
    * Any directory containing ``eplustbl.csv`` directly — used as-is.

    Returns
    -------
    (result_dir, idf_path)  — either can be None when not found.
    """
    if not path.exists():
        return None, None

    # Case 1: path itself contains eplustbl.csv
    if (path / "eplustbl.csv").exists():
        result_dir = path
    else:
        # Case 2: path is a run_* dir — find results subdirectory
        csv_files = sorted(path.rglob("eplustbl.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not csv_files:
            return None, None
        result_dir = csv_files[0].parent

    # Find IDF: look in the run dir (parent of results_*) or the path itself
    idf_path: Path | None = None
    search_dirs = [result_dir.parent, path, result_dir]
    for d in search_dirs:
        idfs = sorted(d.glob("*.idf"), key=lambda f: f.stat().st_mtime, reverse=True)
        if idfs:
            idf_path = idfs[0]
            break

    return result_dir, idf_path


def _find_run_dirs(base: Path) -> list[Path]:
    """Return all run_* directories (and any dir containing eplustbl.csv) under base,
    sorted newest first."""
    seen: set[Path] = set()
    dirs: list[Path] = []
    for csv in sorted(base.rglob("eplustbl.csv"), key=lambda f: f.stat().st_mtime, reverse=True):
        # Prefer the run_* level over the results_* level for display
        run_candidate = csv.parent.parent
        if run_candidate.name.startswith("run_") and run_candidate not in seen:
            seen.add(run_candidate)
            dirs.append(run_candidate)
        elif csv.parent not in seen:
            seen.add(csv.parent)
            dirs.append(csv.parent)
    return dirs


def _find_model_json(result_dir: Path, run_dir: Path | None = None) -> Path | None:
    """Look for model.json near the result directory."""
    candidates = [
        result_dir.parent / "model.json",          # run_*/model.json  (typical)
        result_dir / "model.json",
        (run_dir / "model.json") if run_dir else None,
        result_dir.parent.parent / "model.json",
    ]
    for c in candidates:
        if c is not None and c.exists():
            return c
    return None


def _model_from_idf(idf_path: Path) -> dict | None:
    """Parse an EnergyPlus IDF file and return a model_dict compatible with render_model.

    Each zone's bounding box is computed from the bounding extent of all
    BuildingSurface:Detailed vertices that belong to it.  The resulting
    model_dict has the same shape as the JSON produced by generate_bridge_cluster
    etc., so all existing rendering code works unchanged.
    """
    try:
        from idfpy import IDF
        from collections import defaultdict

        idf = IDF.load(idf_path)

        surfs = idf.all_of_type("BuildingSurface:Detailed")
        if not surfs:
            return None

        zone_verts: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for surf in surfs.values():
            for v in surf.vertices or []:
                zone_verts[surf.zone_name].append((
                    float(v.vertex_x_coordinate),
                    float(v.vertex_y_coordinate),
                    float(v.vertex_z_coordinate),
                ))

        if not zone_verts:
            return None

        zones = []
        for zone_name, verts in zone_verts.items():
            xs = [v[0] for v in verts]
            ys = [v[1] for v in verts]
            zs = [v[2] for v in verts]
            ox, oy, oz = min(xs), min(ys), min(zs)
            L = round(max(xs) - ox, 3)
            W = round(max(ys) - oy, 3)
            H = round(max(zs) - oz, 3)
            zones.append({
                "name": zone_name,
                "origin": {"x": round(ox, 3), "y": round(oy, 3), "z": round(oz, 3)},
                "dimensions": {"length": L, "width": W, "height": H},
            })

        # Infer building name from IDF Building object if available
        # all_of_type returns {name: obj} dict; keys are the EnergyPlus object names
        buildings = idf.all_of_type("Building")
        building_name = next(iter(buildings.keys())) if buildings else idf_path.stem

        return {"building_name": building_name, "zones": zones}

    except Exception as exc:
        return None


def _eui(sim_data: dict) -> float:
    total = sim_data["site_energy"].get("Total Site Energy", 0.0)
    area = sim_data["building_area"].get("Net Conditioned Building Area", 0.0)
    return total * 1000 / area if area > 0 else 0.0


def _end_use_pct(end_uses: list[dict], name: str, total_gj: float) -> float:
    for item in end_uses:
        if item["end_use"] == name:
            return item["total_gj"] / total_gj * 100 if total_gj else 0.0
    return 0.0


# ---------------------------------------------------------------------------
# Sidebar — directory browser
# ---------------------------------------------------------------------------

st.sidebar.header("模拟结果目录")

# Manual path input — accepts run_* or results_* level
manual_path = st.sidebar.text_input(
    "直接输入目录路径",
    placeholder="output/direct_energyplus/run_20260527_231411",
    help="run_* 目录 或 results_* 目录均可，自动查找 IDF 和 eplustbl.csv",
)

# Auto-discover from output/
st.sidebar.divider()
st.sidebar.caption("或从 output/ 自动发现：")
discovered = _find_run_dirs(_OUTPUT_ROOT)
dir_labels = [str(p.relative_to(_OUTPUT_ROOT)) for p in discovered]
selected_label = st.sidebar.selectbox(
    "选择运行目录",
    options=["（请选择）"] + dir_labels,
    index=0,
    key="selected_result_dir",
)

# Resolve final path
result_dir: Path | None = None
idf_path_resolved: Path | None = None

if manual_path.strip():
    p = Path(manual_path.strip())
    result_dir, idf_path_resolved = _resolve_run_dir(p)
    if result_dir is None:
        st.sidebar.error(f"未找到 eplustbl.csv：{manual_path}")
elif selected_label != "（请选择）":
    result_dir, idf_path_resolved = _resolve_run_dir(_OUTPUT_ROOT / selected_label)

# Display options
st.sidebar.divider()
show_edges = st.sidebar.checkbox("显示体块边线", value=True)
opacity = st.sidebar.slider("体块透明度", 0.15, 1.0, 0.60, 0.05)
energy_metric = st.sidebar.selectbox(
    "3D 着色指标",
    options=[
        ("total_gj", "总能耗"),
        ("heating_gj", "采暖"),
        ("cooling_gj", "制冷"),
        ("lighting_gj", "照明"),
    ],
    format_func=lambda x: x[1],
    index=0,
)[0]

# Optional model JSON override
st.sidebar.divider()
model_json_override = st.sidebar.text_input(
    "体量 JSON 路径（可选）",
    placeholder="output/platform_cluster_office.json",
    help="不填则自动查找同目录的 model.json",
)
site_size_override = st.sidebar.number_input(
    "场地尺寸 (m)", value=100.0, min_value=40.0, max_value=300.0, step=10.0,
)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("EnergyPlus 模拟结果查看器")

if result_dir is None:
    st.info("请在左侧选择或输入模拟结果目录（含 eplustbl.csv）。")
    st.stop()

# ── Load simulation data ──────────────────────────────────────────────────
sim_data = read_eplustbl(str(result_dir))
if not sim_data.get("exists"):
    st.error(f"未找到 eplustbl.csv：{result_dir}")
    st.stop()

# ── Load model: IDF first, JSON as fallback ───────────────────────────────
model_path: Path | None = None
if model_json_override.strip():
    p = Path(model_json_override.strip())
    model_path = p if p.exists() else None
    if not model_path:
        st.warning(f"指定的 JSON 不存在：{model_json_override}")
else:
    run_dir_for_model = idf_path_resolved.parent if idf_path_resolved else None
    model_path = _find_model_json(result_dir, run_dir_for_model)

model: dict | None = None
model_source: str = ""

# Priority 1: parse geometry directly from the IDF (no JSON needed)
if idf_path_resolved and idf_path_resolved.exists():
    model = _model_from_idf(idf_path_resolved)
    if model:
        model_source = f"IDF ({len(model['zones'])} zones)"

# Priority 2: fall back to model.json
if model is None and model_path and model_path.exists():
    try:
        model = json.loads(model_path.read_text(encoding="utf-8"))
        model_source = f"model.json ({len(model['zones'])} zones)"
    except Exception as e:
        st.warning(f"加载 model.json 失败：{e}")

# ── Caption ───────────────────────────────────────────────────────────────
info_parts = [f"**结果目录** `{result_dir}`"]
if idf_path_resolved:
    info_parts.append(f"**IDF** `{idf_path_resolved.name}`")
if model_source:
    info_parts.append(f"**几何来源** {model_source}")
elif not idf_path_resolved:
    info_parts.append("未找到 IDF 或 model.json（3D 分区着色不可用）")
st.caption("  |  ".join(info_parts))

# ── Key metrics banner ────────────────────────────────────────────────────
total_site = sim_data["site_energy"].get("Total Site Energy", 0.0)
total_source = sim_data["site_energy"].get("Total Source Energy", 0.0)
cond_area = sim_data["building_area"].get("Net Conditioned Building Area", 0.0)
eui = _eui(sim_data)
end_uses = sim_data.get("end_uses", [])
lighting_pct = _end_use_pct(end_uses, "Interior Lighting", total_site)
cooling_pct = _end_use_pct(end_uses, "Cooling", total_site)
heating_pct = _end_use_pct(end_uses, "Heating", total_site)

m_cols = st.columns(5)
m_cols[0].metric("EUI", f"{eui:.1f} MJ/m²")
m_cols[1].metric("总场地能耗", f"{total_site:.2f} GJ")
m_cols[2].metric("总源能耗", f"{total_source:.2f} GJ")
m_cols[3].metric("调适面积", f"{cond_area:.0f} m²")
if model:
    metrics = model_metrics(model)
    m_cols[4].metric("体块数量", metrics["mass_zone_count"])
else:
    m_cols[4].metric("采暖/制冷/照明占比",
                     f"{heating_pct:.0f}% / {cooling_pct:.0f}% / {lighting_pct:.0f}%")

# Visual alert if any single end-use dominates
if lighting_pct > 60:
    st.warning(f"照明占总能耗 **{lighting_pct:.0f}%**，可降低照明功率密度（W/m²）来大幅减少 EUI。")
elif cooling_pct > 60:
    st.warning(f"制冷占总能耗 **{cooling_pct:.0f}%**，考虑调整窗墙比或制冷设定点。")
elif heating_pct > 50:
    st.info(f"采暖占总能耗 **{heating_pct:.0f}%**，可考虑改善外墙保温或调整供暖设定点。")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────
if model:
    tab_3d, tab_charts, tab_zones, tab_raw = st.tabs(
        ["3D 分区能耗", "分项图表", "Zone 详情", "原始数据"]
    )
else:
    tab_charts, tab_zones, tab_raw = st.tabs(["分项图表", "Zone 详情", "原始数据"])
    tab_3d = None

# ── Tab: 3D zone energy ───────────────────────────────────────────────────
if tab_3d and model:
    with tab_3d:
        mapped_energy = model_energy_map(model, sim_data)
        energy_source = "area_estimate"
        if mapped_energy and all(v.get("source") == "meter" for v in mapped_energy.values()):
            energy_source = "meter"
        if energy_source == "area_estimate":
            st.info(
                "EnergyPlus 逐区 meter 未能完全匹配模型体块，已按楼板面积比例分摊总能耗。"
            )

        col_3d, col_ctrl = st.columns([3, 1])
        with col_3d:
            if not mapped_energy:
                st.warning("未找到可映射到模型体块的 Zone 能耗数据。")
            fig3d = render_model(
                model,
                site_size_override,
                show_edges,
                opacity,
                zone_energy=mapped_energy or None,
                energy_metric=energy_metric,
            )
            st.plotly_chart(fig3d, use_container_width=True)

        with col_ctrl:
            st.subheader("统计")
            mass_zones = [
                z for z in model["zones"]
                if z["dimensions"]["height"] > 1.0 and z.get("category") != "open_space_reference"
            ]
            st.metric("模型体块", len(mass_zones))
            st.metric("有能耗数据", len(mapped_energy))
            st.metric("数据来源", "逐区 meter" if energy_source == "meter" else "面积估算")

            if mapped_energy:
                best_zone = min(mapped_energy.items(), key=lambda kv: kv[1].get("total_gj", 0))
                worst_zone = max(mapped_energy.items(), key=lambda kv: kv[1].get("total_gj", 0))
                st.metric("最低能耗 Zone", best_zone[0], f"{best_zone[1].get('total_gj', 0):.2f} GJ")
                st.metric("最高能耗 Zone", worst_zone[0], f"{worst_zone[1].get('total_gj', 0):.2f} GJ")

# ── Tab: end-use charts ───────────────────────────────────────────────────
with tab_charts:
    if not end_uses:
        st.warning("eplustbl.csv 中未解析到分项能耗数据。")
    else:
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.subheader("分项年能耗（柱状）")
            st.plotly_chart(render_end_use_chart(end_uses), use_container_width=True)
        with chart_col2:
            st.subheader("分项占比（环形）")
            st.plotly_chart(render_end_use_pie(end_uses), use_container_width=True)

        # Building area breakdown
        st.divider()
        area_cols = st.columns(3)
        area_cols[0].metric("总建筑面积", f"{sim_data['building_area'].get('Total Building Area', 0):.0f} m²")
        area_cols[1].metric("空调面积", f"{cond_area:.0f} m²")
        area_cols[2].metric("非空调面积", f"{sim_data['building_area'].get('Unconditioned Building Area', 0):.0f} m²")

        # Site vs source energy
        st.divider()
        st.subheader("场地能耗 vs 源能耗")
        fig_sv = go.Figure(go.Bar(
            x=["Total Site Energy", "Net Site Energy", "Total Source Energy", "Net Source Energy"],
            y=[
                sim_data["site_energy"].get("Total Site Energy", 0),
                sim_data["site_energy"].get("Net Site Energy", 0),
                sim_data["site_energy"].get("Total Source Energy", 0),
                sim_data["site_energy"].get("Net Source Energy", 0),
            ],
            marker_color=["#2563EB", "#3B82F6", "#DC2626", "#EF4444"],
            hovertemplate="<b>%{x}</b><br>%{y:.2f} GJ<extra></extra>",
        ))
        fig_sv.update_layout(
            height=280, margin=dict(l=0, r=0, t=10, b=0),
            yaxis_title="GJ",
        )
        st.plotly_chart(fig_sv, use_container_width=True)

# ── Tab: zone details ─────────────────────────────────────────────────────
with tab_zones:
    zone_energy_raw = sim_data.get("zone_energy", {})

    if not zone_energy_raw:
        st.info("eplustbl.csv 中未找到逐区能耗数据（需要 Output:Variable Zone 开启）。")
    else:
        # Build table rows
        zone_rows_ep = [
            {
                "EnergyPlus Zone": zname,
                "采暖 (GJ)": round(vals.get("heating_gj", 0), 3),
                "制冷 (GJ)": round(vals.get("cooling_gj", 0), 3),
                "照明 (GJ)": round(vals.get("lighting_gj", 0), 3),
                "合计 (GJ)": round(vals.get("total_gj", 0), 3),
            }
            for zname, vals in sorted(zone_energy_raw.items())
        ]

        # If model is available, also show model-zone-mapped rows
        if model:
            mapped_energy = model_energy_map(model, sim_data)
            mass_zones = [
                z for z in model["zones"]
                if z["dimensions"]["height"] > 1.0 and z.get("category") != "open_space_reference"
            ]
            zone_rows_mapped = []
            for i, zone in enumerate(mass_zones, start=1):
                energy = mapped_energy.get(zone["name"], {})
                area = zone["dimensions"]["length"] * zone["dimensions"]["width"]
                total = energy.get("total_gj", 0.0)
                zone_rows_mapped.append({
                    "模型 Zone": zone["name"],
                    "EP Zone": f"ZONE_{i:02d}",
                    "面积 (m²)": round(area, 1),
                    "采暖 (GJ)": round(energy.get("heating_gj", 0), 3),
                    "制冷 (GJ)": round(energy.get("cooling_gj", 0), 3),
                    "照明 (GJ)": round(energy.get("lighting_gj", 0), 3),
                    "合计 (GJ)": round(total, 3),
                    "EUI (MJ/m²)": round(total * 1000 / area, 1) if area else 0.0,
                    "数据来源": energy.get("source", "—"),
                })

            st.subheader("模型体块 → EnergyPlus Zone 映射能耗")
            if zone_rows_mapped:
                st.plotly_chart(
                    render_zone_energy_chart(
                        [{
                            "model_zone": r["模型 Zone"],
                            "heating_gj": r["采暖 (GJ)"],
                            "cooling_gj": r["制冷 (GJ)"],
                            "lighting_gj": r["照明 (GJ)"],
                        } for r in zone_rows_mapped[:40]]
                    ),
                    use_container_width=True,
                )
                st.dataframe(zone_rows_mapped, hide_index=True, use_container_width=True)

            st.divider()

        st.subheader("EnergyPlus 原始 Zone 能耗")
        st.dataframe(zone_rows_ep, hide_index=True, use_container_width=True)

        # Zone summary from eplustbl (if present)
        zone_summary = sim_data.get("zone_summary", [])
        if zone_summary:
            st.subheader("Zone 摘要（面积/空调状态/体积）")
            st.dataframe(zone_summary, hide_index=True, use_container_width=True)

# ── Tab: raw data ─────────────────────────────────────────────────────────
with tab_raw:
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.subheader("分项能耗（原始）")
        st.dataframe(end_uses, hide_index=True, use_container_width=True)

        st.subheader("场地与源能耗")
        st.dataframe(
            [{"指标": k, "值 (GJ)": v} for k, v in sim_data["site_energy"].items()],
            hide_index=True, use_container_width=True,
        )

        st.subheader("建筑面积")
        st.dataframe(
            [{"指标": k, "值 (m²)": v} for k, v in sim_data["building_area"].items()],
            hide_index=True, use_container_width=True,
        )

    with col_r2:
        st.subheader("结果目录文件")
        files = sorted(result_dir.iterdir()) if result_dir.is_dir() else []
        file_rows = [
            {
                "文件名": f.name,
                "大小 (KB)": round(f.stat().st_size / 1024, 1),
                "后缀": f.suffix,
            }
            for f in files if f.is_file()
        ]
        st.dataframe(file_rows, hide_index=True, use_container_width=True)

        # Show eplusout.end (EnergyPlus exit status)
        end_file = result_dir / "eplusout.end"
        if end_file.exists():
            st.subheader("模拟状态 (eplusout.end)")
            st.code(end_file.read_text(encoding="utf-8", errors="replace"), language="text")

        # Show last 20 lines of .err
        err_file = result_dir / "eplusout.err"
        if err_file.exists():
            st.subheader("错误日志摘要 (eplusout.err，最后 20 行)")
            lines = err_file.read_text(encoding="utf-8", errors="replace").splitlines()
            st.code("\n".join(lines[-20:]), language="text")

        # Download buttons
        st.divider()
        eplustbl = result_dir / "eplustbl.csv"
        if eplustbl.exists():
            st.download_button(
                "下载 eplustbl.csv",
                data=eplustbl.read_bytes(),
                file_name="eplustbl.csv",
                mime="text/csv",
                use_container_width=True,
            )
        if idf_path_resolved and idf_path_resolved.exists():
            st.download_button(
                "下载 IDF 文件",
                data=idf_path_resolved.read_bytes(),
                file_name=idf_path_resolved.name,
                mime="text/plain",
                use_container_width=True,
            )
        if model_path and model_path.exists():
            st.download_button(
                "下载 model.json",
                data=model_path.read_bytes(),
                file_name="model.json",
                mime="application/json",
                use_container_width=True,
            )
