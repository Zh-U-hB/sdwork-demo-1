# arch-model-agent

LLM 驱动的建筑空间建模与能耗模拟 Agent。支持两种工作方式：

1. **自然语言建模** — 用一句话描述建筑，自动生成 3D 区域 JSON（Rhino/Grasshopper），并可串联 EnergyPlus 完成 IDF 生成与全年模拟。
2. **参数化 L 形办公** — 在 100m×100m 场地上，用算法生成「低层分散 → 高层完整 L」的渐变体量 JSON，并在 Streamlit 中调参、预览，叠加 EnergyPlus 模拟结果可视化。

---

## 功能概览

### 路径 A：LLM 流水线（LangGraph）

```
自然语言描述
  └─► intake_node      NL → 结构化空间摘要（LLM）
  └─► zone_agent       ReAct Agent → 创建各房间 Zone
  └─► export_node      写 output/*.json
  └─► energyplus_node  （可选，--simulate）
          └─► EnergyPlus-Agent MCP（plugins/energyplus_agent/）
                  └─► 材料 / 构造 / 表面 / 日程 / HVAC / 负荷
                  └─► validate_config → run_simulation
                  └─► .idf + eplustbl.csv
```

### 路径 B：参数化 L 形办公（无 LLM）

```
滑块参数
  └─► generate_l_gradient()     算法生成多层 L 形体块
  └─► output/gradient_l_office.json
  └─► parametric_l_app.py       3D 预览 + 导出
  └─► （单独跑 EnergyPlus 后）读取 eplustbl.csv → 分项/分区能耗图表与 3D 着色
```

两条路径**共享 JSON 结构**（`building_name` + `zones`），但**未在 CLI 中自动串联**；参数化模型需自行对接模拟或后续扩展 `--simulate`。

---

## 运行方式

| 模式 | 命令 | 输出 |
|------|------|------|
| LLM 批处理（仅 JSON） | `python main.py -d "..."` | `output/building.json` |
| LLM + EnergyPlus | `python main.py -d "..." --simulate` | JSON + IDF + 能耗报表 |
| LLM 对话 UI | `streamlit run app.py` | 多轮对话 + 3D 线框预览 |
| 参数化 L 形（CLI） | `python scripts/generate_l_gradient.py` | `output/gradient_l_office.json` |
| 参数化 L 形（UI） | `streamlit run parametric_l_app.py` | 实时调参 + 模拟结果 Tab |

---

## 技术栈

| 层级 | 技术 |
|------|------|
| LLM 编排 | LangChain + LangGraph |
| 默认模型 | Anthropic Claude / DeepSeek（可配置） |
| 数据校验 | Pydantic v2 |
| MCP 通信 | langchain-mcp-adapters |
| EnergyPlus 插件 | [EnergyPlus-Agent](https://github.com/ITOTI-Y/EnergyPlus-Agent)（git submodule） |
| UI / 可视化 | Streamlit + Plotly |
| 包管理 | uv（推荐） |
| Python | ≥ 3.12 |

---

## 快速开始

### 1. 克隆（含 submodule）

```bash
git clone --recursive https://github.com/<your-org>/arch-model-agent.git
cd arch-model-agent
```

已克隆但未拉取 submodule 时：

```bash
git submodule update --init --recursive
```

### 2. 安装依赖

```bash
uv sync          # 推荐（含 streamlit、plotly）
# 或
pip install -e .
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填写 LLM API Key 等
```

最小配置（仅 LLM 路径需要）：

```dotenv
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-...
LLM_MODEL_NAME=claude-sonnet-4-6
```

### 4. LLM 建模（仅 Zone JSON）

```bash
python main.py \
  --description "一栋100平米住宅，客厅6x5m，卧室4x4m，厨房3x3m，层高3m" \
  --name "我的住宅" \
  --output output/villa.json
```

### 5. 参数化 L 形办公（无需 API Key）

```bash
# 命令行生成 JSON
python scripts/generate_l_gradient.py \
  --output output/gradient_l_office.json \
  --floors 11

# 或启动交互调参页面
streamlit run parametric_l_app.py
```

默认目标：约 **9000–11000 m²** 建筑面积、总高 **&lt; 50 m**（可在 UI 侧边栏调整 `site_size`、臂长、层数等）。

---

## EnergyPlus 模拟（可选）

### 前置要求

- [EnergyPlus 25.1.0+](https://energyplus.net/downloads) 已安装
- `uv` 包管理器

### 初始化插件

```bash
bash scripts/setup_plugin.sh
```

脚本会：初始化 submodule → `uv sync` 插件依赖 → 检查 `Energy+.idd` 与 `.epw`。

手动放置文件：

```bash
cp /usr/local/EnergyPlus-25-1-0/Energy+.idd \
   plugins/energyplus_agent/data/dependencies/

cp Shenzhen.epw plugins/energyplus_agent/data/weather/
```

### 通过 LLM 流水线触发模拟

```bash
python main.py \
  --description "一栋小型办公楼，大开间办公室(10x8m)、会议室(6x5m)、茶水间(3x3m)，层高3.5m" \
  --name "示例办公楼" \
  --output output/office.json \
  --simulate \
  --idf-output output/office.idf
```

典型输出：

```
output/office.json
plugins/energyplus_agent/output/
  temp_YYYYMMDD_HHMMSS.idf
  results/energyplus_runs_*/
    eplustbl.csv
    eplusout.err
```

插件默认从 `plugins/energyplus_agent/` 自动启动 MCP（stdio）；无需设置 `ENERGYPLUS_AGENT_PATH`，除非使用自定义安装路径。

### 在参数化 UI 中查看模拟结果

完成 EnergyPlus 运行后，在 `parametric_l_app.py` 侧边栏将 **EnergyPlus 结果目录** 指向包含 `eplustbl.csv` 的文件夹（例如 `plugins/energyplus_agent/output/results/energyplus_runs_*` 或 `output/direct_energyplus_real_run`），在「模拟结果」Tab 查看分项能耗、Zone 分摊与 3D 能耗着色。

### 修改模拟地点

```dotenv
ENERGYPLUS_LOCATION_NAME=Beijing
ENERGYPLUS_LATITUDE=39.93
ENERGYPLUS_LONGITUDE=116.28
ENERGYPLUS_TIMEZONE=8
ENERGYPLUS_ELEVATION=44
ENERGYPLUS_WEATHER_FILE=data/weather/Beijing.epw
```

---

## Streamlit 应用

### 对话建模 — `app.py`

```bash
streamlit run app.py
```

- 多轮对话创建 / 修改 Zone
- Plotly 3D 线框预览
- 导出 JSON（Rhino 用）

### 参数化 L 形 — `parametric_l_app.py`

```bash
streamlit run parametric_l_app.py
```

- **形体预览**：层数、臂宽、分散度、顶部完整层数等滑块；Mesh3D + 100m 场地边界
- **模拟结果**：解析 `eplustbl.csv`，分项柱状图/饼图、按 Zone 堆叠能耗、三维着色（总能耗 / 采暖 / 制冷 / 照明）
- 保存 / 下载 `gradient_l_office.json`

---

## 目录结构

```
arch-model-agent/
├── main.py                      # LLM 批处理 CLI
├── app.py                       # LLM 对话 Streamlit
├── parametric_l_app.py          # 参数化 L 形 + 模拟可视化 Streamlit
├── pyproject.toml
├── .env.example
├── CLAUDE.md                    # 开发说明（Agent 用）
├── scripts/
│   ├── setup_plugin.sh          # EnergyPlus 插件初始化
│   └── generate_l_gradient.py   # 参数化 L 形 JSON 生成器
├── plugins/
│   └── energyplus_agent/        # git submodule（EnergyPlus-Agent）
├── output/                      # JSON / 本地模拟结果（gitignore）
└── src/
    ├── config.py
    ├── models/zone.py           # Pydantic：Zone / BuildingModel
    └── agent/
        ├── graph.py             # LangGraph：intake → zone → export → [energyplus]
        ├── state.py
        ├── llm.py
        ├── chat_agent.py
        ├── nodes/
        │   ├── intake.py
        │   ├── zone.py
        │   ├── export.py
        │   └── energyplus.py    # MCP 持久会话 + EnergyPlus 建模提示词
        └── tools/
            └── zone_tools.py
```

---

## CLI 参考

### `main.py`

```
python main.py [OPTIONS]

必填:
  -d, --description TEXT    建筑自然语言描述

可选:
  -n, --name TEXT           建筑名称（默认: Unnamed Building）
  -o, --output PATH         Zone JSON 路径（默认: output/building.json）
  -s, --simulate            启用 EnergyPlus 节点
      --idf-output PATH     IDF 路径（默认: 与 --output 同主名 .idf）
```

### `scripts/generate_l_gradient.py`

```
python scripts/generate_l_gradient.py [OPTIONS]

常用:
  --output PATH             输出 JSON（默认: output/gradient_l_office.json）
  --floors N                层数（默认 11）
  --site-size M             场地边长 m（默认 100）
  --horizontal-length / --vertical-length / --arm-width
  --scatter-gap / --merge-power / --top-solid-floors
  --no-courtyard-marker     不生成内院参考体块
```

---

## 输出 JSON 格式

### 基础格式（LLM / 简单模型）

```json
{
  "building_name": "示例办公楼",
  "zones": [
    {
      "name": "大开间办公室",
      "origin": { "x": 0, "y": 0, "z": 0 },
      "dimensions": { "length": 10, "width": 8, "height": 3.5 }
    }
  ]
}
```

- `origin`：房间左下角（米）
- `dimensions.length` / `width` / `height`：沿 X / Y / Z 的尺寸（米）

### 扩展格式（参数化生成器）

参数化脚本可为每个 zone 附加 **`points`**（8 个角点），供 `parametric_l_app.py` 做 Mesh3D 渲染；Rhino 流程仍可使用 `origin` + `dimensions` 生成 Box。

```json
{
  "name": "F01_corner_core",
  "origin": { "x": 19.688, "y": 17.688, "z": 0 },
  "dimensions": { "length": 10.125, "width": 10.125, "height": 5.5 },
  "points": [
    { "x": 19.688, "y": 17.688, "z": 0 },
    ...
  ]
}
```

`src/models/zone.py` 中的 Pydantic `Zone` 目前仅包含 `origin` 与 `dimensions`；带 `points` 的 JSON 在导出/预览时由 dict 直接处理。

---

## Rhino / Grasshopper

1. `File Path` → `output/building.json`（或 `gradient_l_office.json`）
2. `JSON Deserialize`
3. `List Item` → `zones`
4. 对每个 zone：`origin` + `dimensions` → `Box`
5. 若需精确角点，可改用 `points` 列表（参数化 JSON）

---

## 环境变量参考

| 变量 | 必填 | 说明 |
|------|------|------|
| `LLM_PROVIDER` | LLM 路径 | 如 `anthropic` |
| `LLM_API_KEY` | LLM 路径 | API Key |
| `LLM_BASE_URL` | | 自定义 API 地址 |
| `LLM_MODEL_NAME` | | 默认 `claude-sonnet-4-6` |
| `LLM_TEMPERATURE` | | 默认 0.7 |
| `LLM_MAX_TOKENS` | | 默认 64000 |
| `ENERGYPLUS_AGENT_PATH` | | 覆盖插件路径（默认 `plugins/energyplus_agent/`） |
| `ENERGYPLUS_AGENT_TRANSPORT` | | `stdio`（默认）/ `http` / `sse` / `streamable_http` |
| `ENERGYPLUS_AGENT_URL` | | HTTP 模式服务器地址 |
| `ENERGYPLUS_WEATHER_FILE` | | EPW 路径（相对插件根目录） |
| `ENERGYPLUS_LOCATION_NAME` | | 地点名（默认 Shenzhen） |
| `ENERGYPLUS_LATITUDE` / `LONGITUDE` | | 经纬度 |
| `ENERGYPLUS_TIMEZONE` | | UTC 偏移（默认 8） |
| `ENERGYPLUS_ELEVATION` | | 海拔 m（默认 5） |

---

## 开发说明

更详细的 Agent 架构与命令见仓库内 [`CLAUDE.md`](CLAUDE.md)。
