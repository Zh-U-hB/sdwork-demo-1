# 项目 Agent 框架图

## 1. 整体架构

```mermaid
graph TB
    User([用户]) --> CLI[CLI入口 main.py]
    User --> UI[Streamlit UI app.py]

    CLI --> AgentPipeline
    UI --> AgentPipeline

    subgraph AgentPipeline [Agent 核心管线]
        direction TB
        START --> Intake[Intake Node<br/>自然语言解析]
        Intake --> ZoneAgent[Zone Agent<br/>ReAct 智能体]
        ZoneAgent --> Export[Export Node<br/>JSON 导出]
        Export --> END
    end

    ZoneAgent --> Tools[工具集]
    Tools --> InMemory[内存存储 _zones]
    Export --> OutputDir[(output/ JSON)]
```

## 2. LangGraph 工作流

```mermaid
stateDiagram-v2
    state "START" as S
    state "Intake Node<br/>解析建筑描述 → 结构化摘要" as I
    state "Zone Agent (ReAct)<br/>循环调用工具创建区域" as Z
    state "Export Node<br/>组装 BuildingModel → 写 JSON" as E
    state "END" as En

    S --> I
    I --> Z
    Z --> Z : create_zone / update_zone / delete_zone
    Z --> E : export_json
    E --> En
```

## 3. Zone Agent 内部机制 (ReAct)

```mermaid
sequenceDiagram
    participant LLM as Claude (claude-sonnet-4-6)
    participant ReAct as ReAct Agent Loop
    participant Tools as 工具集
    participant Store as 内存存储 _zones

    LLM->>ReAct: 思考: 需要创建哪些区域
    ReAct->>Tools: 调用 create_zone("客厅", 0,0,0, 8,6,3)
    Tools->>Store: 写入区域数据
    Store-->>Tools: 成功
    Tools-->>ReAct: 返回结果
    ReAct->>LLM: 将结果交给LLM

    LLM->>ReAct: 思考: 继续创建下一个区域
    ReAct->>Tools: 调用 create_zone("卧室", 0,6,0, 5,4,3)
    Tools->>Store: 写入
    Store-->>Tools: 成功
    Tools-->>ReAct: 返回

    Note over LLM,ReAct: ... 重复直到所有区域创建完毕 ...

    LLM->>ReAct: 思考: 已完成,导出 JSON
    ReAct->>Tools: 调用 export_json("output/building.json")
    Store-->>Tools: 读取全部区域
    Tools-->>ReAct: 导出成功
```

## 4. 模块依赖关系

```mermaid
graph LR
    subgraph Config [配置层]
        config.py
    end

    subgraph Models [数据模型层]
        zone.py[zone.py<br/>Point3D / Dimensions / Zone / BuildingModel]
    end

    subgraph Agent [Agent 逻辑层]
        graph.py[graph.py<br/>LangGraph 编排]
        state.py[state.py<br/>AgentState 类型定义]
        llm.py[llm.py<br/>LLM 工厂]
        chat_agent.py[chat_agent.py<br/>多轮对话 ReAct]
        nodes/intake.py[intake.py<br/>NL → 结构化摘要]
        nodes/zone.py[zone.py<br/>ReAct 智能体节点]
        nodes/export.py[export.py<br/>JSON 输出节点]
    end

    subgraph Tools [工具层]
        zone_tools.py[zone_tools.py<br/>create_zone / list_zones<br/>update_zone / delete_zone<br/>export_json]
    end

    Agent --> Models
    Agent --> Config
    Tools --> Models
    nodes/zone.py --> Tools
    graph.py --> nodes/intake.py
    graph.py --> nodes/zone.py
    graph.py --> nodes/export.py
```

## 5. 文件调用流程

```mermaid
flowchart LR
    main.py -->|run_pipeline| src/agent/graph.py
    app.py -->|chat_interface| src/agent/chat_agent.py

    src/agent/graph.py -->|编译图| src/agent/nodes/intake.py
    src/agent/graph.py -->|编译图| src/agent/nodes/zone.py
    src/agent/graph.py -->|编译图| src/agent/nodes/export.py

    src/agent/nodes/zone.py -->|create_react_agent| src/agent/tools/zone_tools.py
    src/agent/chat_agent.py -->|create_react_agent| src/agent/tools/zone_tools.py

    src/agent/nodes/intake.py -->|init_chat_model| src/agent/llm.py
    src/agent/nodes/zone.py -->|init_chat_model| src/agent/llm.py
    src/agent/nodes/export.py -->|BuildingModel| src/models/zone.py

    src/agent/llm.py -->|LLMConfig| src/config.py
    src/config.py -->|dotenv| .env
```

## 6. 核心数据流

```mermaid
flowchart TB
    Input[/"用户输入 (自然语言)"/] --> Intake

    subgraph Intake [Intake Node]
        NL["'100平米的住宅, 一个客厅...'"]
        Summary["→ 结构化摘要 (文本)"]
        NL --> Summary
    end

    Summary --> ZoneLoop

    subgraph ZoneLoop [Zone Agent ReAct Loop]
        direction TB
        Think["思考: 需要创建哪些区域"] --> Call["调用 create_zone 工具"]
        Call --> Result["LLM 处理工具返回结果"]
        Result --> Decide{"还有区域<br/>未创建?"}
        Decide -->|是| Think
        Decide -->|否| ExportCall["调用 export_json 工具"]
    end

    ExportCall --> ExportNode

    subgraph ExportNode [Export Node]
        Assemble["组装 BuildingModel"]
        Write["写入 JSON 文件"]
        Assemble --> Write
    end

    Write --> Output[(output/building.json)]
    Output --> Rhino[Rhino 3D / Grasshopper]
```

## 7. 层次总览

| 层次 | 组件 | 职责 |
|------|------|------|
| **入口层** | `main.py`, `app.py` | CLI 和 Streamlit UI |
| **编排层** | `graph.py` | LangGraph 三节点工作流编排 |
| **Agent 层** | `zone.py`, `chat_agent.py` | ReAct 智能体, 通过思考-行动循环创建建筑区域 |
| **工具层** | `zone_tools.py` | 5 个 LangChain 工具 (create_zone / list_zones / update_zone / delete_zone / export_json) |
| **数据模型** | `zone.py` | Pydantic 模型: Point3D, Dimensions, Zone, BuildingModel |
| **配置层** | `config.py`, `.env` | LLM 提供商 / 模型 / 参数配置 |
| **输出层** | `output/*.json` | Rhino/Grasshopper 可直接读取的结构化 JSON |
