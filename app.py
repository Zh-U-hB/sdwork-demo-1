"""Streamlit UI for the 3D Architectural Modeling Agent — multi-turn chat mode."""

import asyncio

import plotly.graph_objects as go
import streamlit as st

from src.agent.llm import create_llm
from src.agent.chat_agent import build_chat_agent, chat_turn
from src.agent.tools import reset_store

st.set_page_config(page_title="建筑3D建模 Agent", page_icon="🏗️", layout="wide")

# ── Session state init ──────────────────────────────────────────────
if "agent" not in st.session_state:
    llm = create_llm()
    st.session_state.agent = build_chat_agent(llm)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "zones" not in st.session_state:
    st.session_state.zones = []

if "output_path" not in st.session_state:
    st.session_state.output_path = "output/building.json"


def _box_edges(ox, oy, oz, l, w, h):
    corners = [
        (ox, oy, oz), (ox + l, oy, oz), (ox + l, oy + w, oz), (ox, oy + w, oz),
        (ox, oy, oz + h), (ox + l, oy, oz + h), (ox + l, oy + w, oz + h), (ox, oy + w, oz + h),
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    ex, ey, ez = [], [], []
    for i, j in edges:
        ex.extend([corners[i][0], corners[j][0], None])
        ey.extend([corners[i][1], corners[j][1], None])
        ez.extend([corners[i][2], corners[j][2], None])
    return ex, ey, ez


def render_preview(zones):
    if not zones:
        return None
    fig = go.Figure()
    colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
        "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
    ]
    for i, z in enumerate(zones):
        d = z if isinstance(z, dict) else z.model_dump()
        ox, oy, oz = d["origin"]["x"], d["origin"]["y"], d["origin"]["z"]
        l, w, h = d["dimensions"]["length"], d["dimensions"]["width"], d["dimensions"]["height"]
        ex, ey, ez = _box_edges(ox, oy, oz, l, w, h)
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez, mode="lines",
            line=dict(color=color, width=3), name=d["name"],
            hovertemplate=f"<b>{d['name']}</b><br>origin: ({ox}, {oy}, {oz})<br>{l}x{w}x{h}m<extra></extra>",
        ))
    fig.update_layout(
        scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)", aspectmode="data"),
        margin=dict(l=0, r=0, t=10, b=0), height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def run_agent_sync(user_input: str, history: list[dict], output_path: str):
    """Synchronous wrapper for async chat_turn with conversation history."""
    return asyncio.run(chat_turn(
        st.session_state.agent, user_input,
        history=history, output_path=output_path,
    ))


def reset_all():
    reset_store()
    st.session_state.messages = []
    st.session_state.zones = []


# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 设置")
    output_path = st.text_input("JSON 输出路径", value=st.session_state.output_path)
    st.session_state.output_path = output_path

    if st.button("🗑️ 清空对话 & 模型", use_container_width=True):
        reset_all()
        st.rerun()

    st.divider()
    st.subheader("🦏 Rhino 使用")
    st.markdown("""
    1. **File Path** → `output/building.json`
    2. **JSON Deserialize** 解析
    3. **List Item** → `zones`
    4. 遍历读取 `origin.x/y/z` 和 `dimensions.length/width/height`
    5. **Box** 电池生成几何体
    """)

    st.divider()
    st.caption(f"当前 {len(st.session_state.zones)} 个空间区域")

# ── Main: 3-column layout ───────────────────────────────────────────
st.title("🏗️ 建筑3D建模 Agent")

left, right = st.columns([1, 1])

with left:
    st.subheader("💬 对话")

    # Chat messages
    chat_container = st.container(height=520)
    with chat_container:
        for msg in st.session_state.messages:
            role = "user" if msg["role"] == "user" else "assistant"
            with st.chat_message(role):
                st.write(msg["content"])

    # Chat input
    if user_input := st.chat_input("描述建筑或输入修改指令…"):
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("Agent 思考中…"):
            result = run_agent_sync(
                user_input,
                history=st.session_state.messages[:-1],
                output_path=st.session_state.output_path,
            )

        response = result.get("response", "处理完成")
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.zones = result.get("zones", [])
        st.rerun()

with right:
    st.subheader("📊 3D 预览 & 空间列表")
    fig = render_preview(st.session_state.zones)
    if fig:
        st.plotly_chart(fig, use_container_width=True, key="preview3d")

    # Zone list
    zones_str = st.session_state.zones
    if zones_str:
        st.caption(f"共 {len(zones_str)} 个空间")
        for i, z in enumerate(zones_str):
            d = z if isinstance(z, dict) else z.model_dump()
            st.text(
                f"{i+1}. {d['name']}  "
                f"原点({d['origin']['x']}, {d['origin']['y']}, {d['origin']['z']})  "
                f"{d['dimensions']['length']}x{d['dimensions']['width']}x{d['dimensions']['height']}m"
            )
    else:
        st.info("左侧输入建筑描述开始建模。例如:\n\n"
                "> 设计一个100平米的住宅，包含客厅(6x5m)、主卧(4x4m)、次卧(3x3m)、厨房(3x3m)，层高3m")
