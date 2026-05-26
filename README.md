# arch-model-agent

LLM 驱动的建筑空间建模与能耗模拟 Agent。用一句自然语言描述建筑，自动生成 3D 区域 JSON（供 Rhino/Grasshopper 使用），并可一键触发 EnergyPlus 建模与全年能耗模拟。

---

## 功能概览

```
自然语言描述
  └─► intake_node    NL → 结构化空间摘要（LLM）
  └─► zone_agent     ReAct Agent → 创建各房间 Zone（LangGraph + LangChain）
  └─► export_node    写 output/building.json（供 Rhino/Grasshopper 消费）
  └─► energyplus_node（可选）
          └─► EnergyPlus-Agent MCP 服务器
                  └─► 建筑建模（材料、构造、表面、日程、HVAC、负荷）
                  └─► validate_config → run_simulation
                  └─► 输出 .idf 文件 + 全年能耗报表（eplustbl.csv）
```

### 两种运行模式

| 模式 | 命令 | 输出 |
|------|------|------|
| 仅生成 Zone JSON（Rhino 用） | `python main.py -d "..."` | `output/building.json` |
| 完整 EnergyPlus 模拟 | `python main.py -d "..." --simulate` | JSON + IDF + 能耗报表 |
| Streamlit 对话 UI | `streamlit run app.py` | 实时 3D 预览 + 导出 JSON |

---

## 技术栈

| 层级 | 技术 |
|------|------|
| LLM 编排 | LangChain + LangGraph |
| 默认模型 | Anthropic Claude / DeepSeek（可配置） |
| 数据校验 | Pydantic v2 |
| MCP 通信 | langchain-mcp-adapters |
| EnergyPlus 插件 | [EnergyPlus-Agent](https://github.com/ITOTI-Y/EnergyPlus-Agent)（git submodule） |
| 对话 UI | Streamlit + Plotly |
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

### 2. 安装主项目依赖

```bash
uv sync          # 推荐
# 或
pip install -e .
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填写 LLM API Key 等
```

最小配置：

```dotenv
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-...
LLM_MODEL_NAME=claude-sonnet-4-6
```

### 4. 运行（仅 Zone JSON）

```bash
python main.py \
  --description "一栋100平米住宅，客厅6x5m，卧室4x4m，厨房3x3m，层高3m" \
  --name "我的住宅" \
  --output output/villa.json
```

---

## EnergyPlus 模拟（可选）

### 前置要求

- [EnergyPlus 25.1.0+](https://energyplus.net/downloads) 已安装
- `uv` 包管理器

### 初始化插件

```bash
bash scripts/setup_plugin.sh
```

脚本会自动：
1. 执行 `git submodule update --init`
2. 在插件目录运行 `uv sync` 安装插件依赖
3. 检查 `Energy+.idd` 和 `.epw` 天气文件是否就位

手动放置必要文件：

```bash
# EnergyPlus IDD 文件（随 EnergyPlus 安装程序附带）
cp /usr/local/EnergyPlus-25-1-0/Energy+.idd \
   plugins/energyplus_agent/data/dependencies/

# 天气数据（从 https://energyplus.net/weather 下载）
cp Shenzhen.epw plugins/energyplus_agent/data/weather/
```

### 运行完整流水线

```bash
python main.py \
  --description "一栋小型办公楼，大开间办公室(10x8m)、会议室(6x5m)、茶水间(3x3m)，层高3.5m" \
  --name "示例办公楼" \
  --output output/office.json \
  --simulate \
  --idf-output output/office.idf
```

输出文件：

```
output/office.json                          ← Zone 几何（Rhino 用）
plugins/energyplus_agent/output/
  temp_YYYYMMDD_HHMMSS.idf                  ← EnergyPlus IDF 文件
  results/energyplus_runs_*/
    eplustbl.csv                            ← 全年能耗报表
    eplusout.err                            ← 运行日志
    eplusout.eio / .rdd / .mdd / ...        ← 其他输出
```

### 修改模拟地点

在 `.env` 中设置：

```dotenv
ENERGYPLUS_LOCATION_NAME=Beijing
ENERGYPLUS_LATITUDE=39.93
ENERGYPLUS_LONGITUDE=116.28
ENERGYPLUS_TIMEZONE=8
ENERGYPLUS_ELEVATION=44
ENERGYPLUS_WEATHER_FILE=data/weather/Beijing.epw
```

---

## Streamlit 对话 UI

```bash
# 先安装 UI 依赖
pip install streamlit plotly

streamlit run app.py
```

功能：
- 多轮对话逐步建模
- 实时 Plotly 3D 线框预览
- 侧边栏显示 Rhino 使用说明
- 一键清空重建

---

## 目录结构

```
arch-model-agent/
├── main.py                      # CLI 入口
├── app.py                       # Streamlit UI 入口
├── pyproject.toml
├── .env.example
├── scripts/
│   └── setup_plugin.sh          # 插件初始化脚本
├── plugins/
│   └── energyplus_agent/        # git submodule（EnergyPlus-Agent）
├── output/                      # Zone JSON 输出目录
└── src/
    ├── config.py                # LLM 配置
    ├── models/zone.py           # Pydantic 数据模型
    └── agent/
        ├── graph.py             # LangGraph 流水线
        ├── state.py             # AgentState
        ├── llm.py               # LLM 工厂
        ├── chat_agent.py        # 对话 ReAct Agent（UI 用）
        ├── nodes/
        │   ├── intake.py        # NL → 结构化摘要
        │   ├── zone.py          # ReAct Zone 创建 Agent
        │   ├── export.py        # 写 JSON
        │   └── energyplus.py    # EnergyPlus MCP 集成节点
        └── tools/
            └── zone_tools.py    # Zone CRUD LangChain 工具
```

---

## CLI 参数

```
python main.py [OPTIONS]

必填:
  -d, --description TEXT    建筑自然语言描述

可选:
  -n, --name TEXT           建筑名称（默认: "Unnamed Building"）
  -o, --output PATH         Zone JSON 输出路径（默认: output/building.json）
  -s, --simulate            启用 EnergyPlus 完整模拟流程
      --idf-output PATH     IDF 输出路径（默认: 与 --output 同名，后缀 .idf）
```

---

## 输出 JSON 格式

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

`origin` 为房间左下角坐标，`length`/`width`/`height` 单位均为米。

---

## Rhino/Grasshopper 使用

1. `File Path` 节点 → 选择 `output/building.json`
2. `JSON Deserialize` 解析
3. `List Item` 取 `zones` 列表
4. 遍历读取每个 zone 的 `origin.x/y/z` 和 `dimensions.length/width/height`
5. `Box` 电池生成几何体

---

## 环境变量参考

| 变量 | 必填 | 说明 |
|------|------|------|
| `LLM_PROVIDER` | ✓ | LLM 提供商（`anthropic` / `openai` 等） |
| `LLM_API_KEY` | ✓ | API Key |
| `LLM_BASE_URL` | | 自定义 API 地址（兼容 OpenAI 格式） |
| `LLM_MODEL_NAME` | | 模型名称（默认 `claude-sonnet-4-6`） |
| `LLM_TEMPERATURE` | | 温度（默认 0.7） |
| `LLM_MAX_TOKENS` | | 最大 token 数（默认 64000） |
| `ENERGYPLUS_AGENT_PATH` | | 插件路径覆盖（默认自动检测 `plugins/energyplus_agent/`） |
| `ENERGYPLUS_AGENT_TRANSPORT` | | `stdio`（默认）/ `http` / `sse` |
| `ENERGYPLUS_AGENT_URL` | | HTTP 模式下的服务器地址 |
| `ENERGYPLUS_WEATHER_FILE` | | EPW 天气文件路径（相对插件根目录） |
| `ENERGYPLUS_LOCATION_NAME` | | 地点名称（默认 Shenzhen） |
| `ENERGYPLUS_LATITUDE/LONGITUDE` | | 经纬度（默认深圳） |
| `ENERGYPLUS_TIMEZONE` | | UTC 时区偏移（默认 8） |
| `ENERGYPLUS_ELEVATION` | | 海拔高度 m（默认 5） |
