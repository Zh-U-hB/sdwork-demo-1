"""Genetic algorithm optimizer for parametric L-shape building energy performance."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from scripts.ep_sim_utils import MASS_HEIGHT_THRESHOLD, model_energy_map, read_eplustbl
from scripts.ga_core import (
    DEFAULT_GENES,
    GAConfig,
    GenerationResult,
    load_checkpoint,
    run_ga,
    save_checkpoint,
)
from scripts.generate_l_gradient import generate_l_gradient

# Reuse visualization from parametric_l_app
from parametric_l_app import (
    box_vertices,
    box_edges,
    model_metrics,
    render_model,
    save_json,
)


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="GA 能耗优化", layout="wide")
st.title("GA 能耗优化 — 参数化 L 形办公楼")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "ga_history" not in st.session_state:
    st.session_state.ga_history = []          # list of GenerationResult dicts
if "ga_running" not in st.session_state:
    st.session_state.ga_running = False
if "ga_best" not in st.session_state:
    st.session_state.ga_best = None           # best individual params
if "ga_best_fitness" not in st.session_state:
    st.session_state.ga_best_fitness = None
if "ga_best_model" not in st.session_state:
    st.session_state.ga_best_model = None
if "ga_total_evals" not in st.session_state:
    st.session_state.ga_total_evals = 0


def _reset_state():
    for key in ("ga_history", "ga_best", "ga_best_fitness", "ga_best_model", "ga_total_evals"):
        st.session_state[key] = [] if key == "ga_history" else None if key != "ga_total_evals" else 0


# ---------------------------------------------------------------------------
# Sidebar config
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("GA 参数")
    pop_size = st.slider("种群大小", 5, 30, 10)
    n_gen = st.slider("代数", 3, 50, 10)
    mutation_rate = st.slider("变异率", 0.01, 0.30, 0.15, 0.01)
    elite_count = st.slider("精英保留数", 1, 3, 1)
    seed = st.number_input("随机种子", value=42, step=1)

    st.divider()
    st.caption("每次评估需运行完整 EnergyPlus 模拟（约 2-5 分钟/个）。")
    est_minutes = pop_size * (n_gen + 1) * 3
    st.caption(f"预计总耗时 ≈ {est_minutes} 分钟（{pop_size * (n_gen + 1)} 次评估）")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶ 开始优化", type="primary", use_container_width=True):
            _reset_state()
            st.session_state.ga_running = True
            st.rerun()
    with col2:
        if st.button("↻ 重置", use_container_width=True):
            _reset_state()
            st.session_state.ga_running = False
            st.rerun()

    st.divider()
    if st.button("📂 恢复上次运行", use_container_width=True):
        ckpt = load_checkpoint("output/ga_checkpoint.json")
        if ckpt:
            st.session_state.ga_history = ckpt.history
            st.session_state.ga_best_fitness = min(
                (h.get("best_fitness", 1e6) for h in ckpt.history), default=None
            )
            best_h = min(ckpt.history, key=lambda h: h.get("best_fitness", 1e6)) if ckpt.history else None
            if best_h:
                st.session_state.ga_best = best_h.get("best_params")
            st.success(f"已恢复到第 {ckpt.generation} 代")
            st.rerun()
        else:
            st.warning("未找到 checkpoint 文件。")

# ---------------------------------------------------------------------------
# Run GA (triggered by ga_running state)
# ---------------------------------------------------------------------------

if st.session_state.ga_running and not st.session_state.ga_history:
    config = GAConfig(
        pop_size=pop_size,
        n_gen=n_gen,
        mutation_rate=mutation_rate,
        elite_count=elite_count,
        checkpoint_path="output/ga_checkpoint.json",
    )

    progress = st.progress(0, text="准备开始 GA 优化...")
    status = st.empty()

    total_evals = 0
    history = []
    overall_best_fitness = float("inf")
    overall_best_params = None
    overall_best_model = None

    for result in run_ga(config, seed=int(seed)):
        total_evals += len(result.pop_fitness)
        history.append({
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

        progress_pct = (result.gen + 1) / (config.n_gen + 1)
        progress.progress(
            progress_pct,
            text=f"第 {result.gen}/{config.n_gen} 代 | 最优 EUI: {result.best_fitness:.1f} MJ/m² | 已评估 {total_evals} 个",
        )
        status.info(f"当前代最优: {result.best_fitness:.1f} MJ/m²  |  平均: {result.avg_fitness:.1f}  |  最差: {result.worst_fitness:.1f}")

        # Save checkpoint each generation
        from scripts.ga_core import CheckpointState
        ckpt = CheckpointState(
            generation=result.gen,
            population=[],  # Don't store full pop in checkpoint (too large)
            fitness=[],
            config={"pop_size": config.pop_size, "n_gen": config.n_gen},
            history=history,
        )
        save_checkpoint(ckpt, config.checkpoint_path)

    st.session_state.ga_history = history
    st.session_state.ga_best = overall_best_params
    st.session_state.ga_best_fitness = overall_best_fitness
    st.session_state.ga_best_model = overall_best_model
    st.session_state.ga_total_evals = total_evals
    st.session_state.ga_running = False
    st.rerun()

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------

history = st.session_state.ga_history
best_params = st.session_state.ga_best
best_fitness = st.session_state.ga_best_fitness

if not history:
    st.info("请在左侧点击「▶ 开始优化」按钮运行遗传算法。")
    st.stop()

progress_tab, best_tab, pop_tab = st.tabs(["进化过程", "最优方案", "种群对比"])

# --- Tab 1: Evolution progress ---
with progress_tab:
    st.subheader("适应度曲线")
    gens = [h["gen"] for h in history]
    bests = [h["best_fitness"] for h in history]
    avgs = [h["avg_fitness"] for h in history]
    worsts = [h["worst_fitness"] for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gens, y=bests, mode="lines+markers", name="最优", line=dict(color="#059669")))
    fig.add_trace(go.Scatter(x=gens, y=avgs, mode="lines+markers", name="平均", line=dict(color="#2563EB")))
    fig.add_trace(go.Scatter(x=gens, y=worsts, mode="lines+markers", name="最差", line=dict(color="#DC2626")))
    fig.update_layout(
        height=400,
        xaxis_title="代数",
        yaxis_title="EUI (MJ/m²)",
        yaxis_rangemode="tozero",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Parameter convergence
    if best_params:
        st.subheader("参数收敛趋势")
        gene_names = [g.name for g in DEFAULT_GENES]
        param_traces = {name: [] for name in gene_names}
        for h in history:
            bp = h.get("best_params", {})
            for name in gene_names:
                param_traces[name].append(bp.get(name))

        cols = st.columns(4)
        for idx, name in enumerate(gene_names):
            with cols[idx % 4]:
                fig_p = go.Figure(go.Scatter(
                    x=gens, y=param_traces[name], mode="lines+markers",
                    marker=dict(size=4),
                ))
                fig_p.update_layout(
                    height=180, margin=dict(l=30, r=10, t=25, b=25),
                    title=dict(text=name, font=dict(size=12)),
                )
                st.plotly_chart(fig_p, use_container_width=True)

# --- Tab 2: Best solution ---
with best_tab:
    if not best_params:
        st.warning("尚无有效最优解。")
    else:
        col_metrics = st.columns(4)
        col_metrics[0].metric("最优 EUI", f"{best_fitness:.1f} MJ/m²" if best_fitness else "N/A")
        col_metrics[1].metric("总代数", f"{len(history)}")
        col_metrics[2].metric("总评估数", f"{st.session_state.ga_total_evals}")

        # Generate best model for preview
        from scripts.ga_core import FIXED_PARAMS
        full = {**FIXED_PARAMS, **best_params}
        try:
            best_model = generate_l_gradient(**full)
            metrics = model_metrics(best_model)
            col_metrics[3].metric("建筑面积", f"{metrics['area']:.1f} m²")

            st.subheader("最优方案 3D 预览")
            fig3d = render_model(best_model, 100.0, True, 0.58)
            st.plotly_chart(fig3d, use_container_width=True)
        except Exception as e:
            st.error(f"模型生成失败：{e}")
            best_model = None

        st.subheader("最优参数")
        st.json(best_params)

        # Export
        st.divider()
        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            if st.button("保存最优方案 JSON", use_container_width=True):
                if best_model:
                    path = save_json(best_model, "output/ga_best.json")
                    st.success(f"已保存到 {path}")
                else:
                    st.warning("无模型可保存。")
        with col_exp2:
            if best_model:
                st.download_button(
                    "下载最优方案",
                    data=json.dumps(best_model, indent=2, ensure_ascii=False),
                    file_name="ga_best.json",
                    mime="application/json",
                    use_container_width=True,
                )

# --- Tab 3: Population comparison ---
with pop_tab:
    latest = history[-1] if history else None
    if not latest:
        st.info("暂无种群数据。")
    else:
        pop_fit = latest.get("pop_fitness", [])
        st.subheader(f"最终种群适应度分布（第 {latest['gen']} 代）")
        if pop_fit:
            valid_fit = [f for f in pop_fit if f < 1e6]
            if valid_fit:
                fig_hist = go.Figure(go.Histogram(
                    x=valid_fit,
                    nbinsx=max(5, len(valid_fit) // 2),
                    marker_color="#2563EB",
                ))
                fig_hist.update_layout(
                    height=300,
                    xaxis_title="EUI (MJ/m²)",
                    yaxis_title="个体数",
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.warning("所有个体均未通过评估。")

        # History table
        st.subheader("每代摘要")
        rows = []
        for h in history:
            rows.append({
                "代数": h["gen"],
                "最优 EUI": round(h["best_fitness"], 1) if h["best_fitness"] < 1e6 else "FAIL",
                "平均 EUI": round(h["avg_fitness"], 1) if h["avg_fitness"] < 1e6 else "FAIL",
                "最差 EUI": round(h["worst_fitness"], 1) if h["worst_fitness"] < 1e6 else "FAIL",
            })
        st.dataframe(rows, hide_index=True, use_container_width=True)
