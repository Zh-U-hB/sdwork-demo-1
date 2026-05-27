# GA vs LLM vs 混合优化 — 三种方法对比实验计划

> 本文档描述如何构建三种建筑能耗优化方法并进行系统对比。
> 前置条件：GA 优化器和直接 IDF 转换器已完成。
> 最后更新：2026-05-27

---

## 1. 背景与目标

项目已有完整的遗传算法（GA）优化流程（`scripts/ga_core.py` + `ga_optimizer_app.py`），能搜索 `generate_l_gradient()` 的 12 个参数空间来最小化 EUI（MJ/m²）。本计划需要：

1. **构建纯 LLM 优化流程** — LLM 作为黑盒优化器，基于历史反馈迭代提出参数方案
2. **构建 LLM+GA 混合流程** — LLM 提供高质量初始种群，GA 精调
3. **三种方法系统对比** — 固定评估预算下对比解质量、效率、成本、失败率

---

## 2. 公平对比前提

- **相同搜索空间**：12 个参数，相同范围和步长（`DEFAULT_GENES`）
- **相同评估函数**：`generate_l_gradient()` + `convert_and_run()` + `read_eplustbl()` → EUI
- **相同缓存机制**：避免重复模拟，只统计唯一评估次数
- **相同约束**：建筑高度 < 50m，无效方案返回惩罚值 `PENALTY = 1e6`
- **相同计算预算**：按**模拟评估次数**（非时间）设定上限

---

## 3. 对比维度

| 维度 | 含义 | 测量方式 |
|---|---|---|
| 最终 EUI | 优化质量 | 全局最优 EUI |
| 收敛曲线 | 找到好解的速度 | 每次评估后的历史最优 EUI |
| 总评估次数 | 达到目标所需的模拟量 | 固定预算下的最终结果 |
| 总耗时 | 端到端时间 | 含 LLM 推理时间 vs GA 计算时间 |
| API 开销 | Token 消耗 | 每轮 input/output tokens |
| 失败率 | 无效方案比例 | 返回 PENALTY 的次数占比 |

---

## 4. 三种优化方法

### 方法 A：遗传算法（已有 ✅）

- **文件**：`scripts/ga_core.py`
- **方法**：随机初始化种群 → BLX-α 交叉 + 高斯变异 → 锦标赛选择 → 精英保留 → 逐代迭代
- **评估函数**：`evaluate_fitness(individual, cache)` → `(eui, model_dict)`

### 方法 B：纯 LLM 优化器（新建）

- **文件**：`scripts/llm_optimizer_core.py`
- **方法**：LLM 每轮提出 1~N 组参数 → 模拟评估 → EUI 反馈 → LLM 基于历史调整策略
- **核心循环**：

```
for i in range(max_iterations):
    params_list = ask_llm(history, gene_specs)  # LLM 提出参数方案
    for params in params_list:
        eui = evaluate_fitness(params, cache)     # 复用 GA 的评估函数
        history.append(params, eui)
    yield LLMIterationResult(...)
```

**LLM Prompt 设计**：

System prompt：
1. 你是一个建筑能耗优化器
2. 12 个可调参数的名称、范围、步长、物理含义
3. 约束：总高度 < 50m（lobby_height + (floors-1)*floor_height）
4. 输出 JSON 格式：`{"reasoning": "...", "proposals": [{param: value, ...}, ...]}`

每轮 user message：
1. 历史评估表格（参数 → EUI），按 EUI 排序（最多展示 top-20）
2. 当前最优方案及 EUI
3. 要求提出 N 组新参数并解释推理

**LLM 输出解析**：
- 解析 JSON，提取 `proposals` 数组
- 校验参数范围，越界则 clamp
- 对每组参数调用 `evaluate_fitness()`（复用 GA 缓存机制）
- 将结果追加到历史

### 方法 C：LLM 初始化 + GA 精调（新建）

- **文件**：`scripts/hybrid_optimizer_core.py`
- **方法**：LLM 先提出 K 个高质量初始方案，GA 从这些种子出发精调
- **流程**：
  1. 调用 LLM 生成 K 个初始方案（K = GA 种群大小）
  2. 用这些方案替换 GA 的随机初始化种群
  3. 正常运行 GA 的交叉/变异/选择循环
  4. 记录"LLM 初始化 vs 随机初始化"对 GA 收敛速度的影响

关键对比点：LLM 提供的初始种群质量 vs 随机种群 → GA 能否从更好的起点更快收敛。

---

## 5. 新增文件

### 5.1 `scripts/llm_optimizer_core.py` — 纯 LLM 优化核心

```python
@dataclass
class LLMOptConfig:
    max_iterations: int = 30
    proposals_per_iteration: int = 1  # 每轮提出的方案数
    temperature: float = 0.7
    cache_path: str = "output/llm_opt_cache.json"

@dataclass
class LLMIterationResult:
    iteration: int
    proposals: list[dict]              # LLM 提出的参数方案列表
    euis: list[float]                  # 对应 EUI 列表
    best_eui: float                    # 截至本轮的全局最优
    llm_reasoning: str                 # LLM 的推理文本
    input_tokens: int
    output_tokens: int
    wall_seconds: float

def run_llm_optimization(
    config: LLMOptConfig,
    budget: int | None = None,        # 评估预算上限
) -> Generator[LLMIterationResult, None, None]:
    """每轮 yield LLMIterationResult"""
```

**复用 GA 的模块**：
```python
from scripts.ga_core import evaluate_fitness, DEFAULT_GENES, FIXED_PARAMS, PENALTY
```

**LLM 调用**：
```python
from src.agent.llm import create_llm
from src.config import LLMConfig
```

### 5.2 `scripts/hybrid_optimizer_core.py` — 混合优化

```python
@dataclass
class HybridConfig:
    ga_config: GAConfig
    llm_temperature: float = 0.7

def run_hybrid_optimization(
    config: HybridConfig,
    seed: int | None = None,
) -> Generator[GenerationResult, None, None]:
    """先用 LLM 生成初始种群，再运行 GA"""
```

核心：调用 LLM 生成 `pop_size` 个方案 → 替换 `run_ga()` 中的 `random_individual()` → 正常运行 GA。

### 5.3 `scripts/compare_optimizers.py` — 三种方法自动化对比

```python
@dataclass
class MethodResult:
    name: str
    best_eui: float
    best_params: dict
    total_evals: int
    valid_evals: int
    fail_evals: int
    total_seconds: float
    llm_tokens: tuple[int, int]        # GA 为 (0, 0)
    history: list

@dataclass
class ComparisonReport:
    budget: int
    ga_result: MethodResult
    llm_result: MethodResult
    hybrid_result: MethodResult

def run_comparison(
    budget: int = 30,
    seed: int = 42,
) -> ComparisonReport:
    """运行三种优化，各限制在 budget 次评估内"""
```

输出：
- JSON 报告 `output/comparison_report_{timestamp}.json`
- 收敛曲线对比图 `output/comparison_convergence.png`

### 5.4 `llm_optimizer_app.py` — LLM 优化 Streamlit UI

参照 `ga_optimizer_app.py` 布局：
- 侧边栏：迭代数、每轮方案数、模型选择、开始/重置
- Tab 1：优化过程（EUI 收敛曲线 + LLM 推理日志）
- Tab 2：最优方案（3D 预览 + 参数 JSON）
- Tab 3：与 GA 结果对比面板

### 5.5 `comparison_app.py` — 对比实验 UI（可选）

统一入口同时运行三种优化并展示对比。

---

## 6. 修改文件

### `scripts/ga_core.py` — 无需修改

`evaluate_fitness()` 是模块级函数，接受外部 cache dict，天然支持多优化器共享：

```python
from scripts.ga_core import evaluate_fitness, DEFAULT_GENES, FIXED_PARAMS, PENALTY
```

---

## 7. 实现顺序

| 步骤 | 任务 | 依赖 |
|---|---|---|
| 1 | `scripts/llm_optimizer_core.py` — LLM 优化核心 | GA 的 evaluate_fitness |
| 2 | 验证 LLM 优化器（5 轮迭代） | LLM API + EP 模拟 |
| 3 | `scripts/hybrid_optimizer_core.py` — 混合优化 | LLM 优化器 + GA |
| 4 | `scripts/compare_optimizers.py` — 自动化对比 | 三种优化器 |
| 5 | `llm_optimizer_app.py` — LLM 优化 UI | LLM 优化核心 |
| 6 | `comparison_app.py` — 对比 UI（可选） | 对比脚本 |

---

## 8. 验证方法

1. **LLM 优化器单独测试**：5 轮迭代，确认 API 调用成功、参数解析正确、模拟能跑出 EUI
2. **混合优化器测试**：LLM 生成 pop_size=5 初始种群 → GA 3 代，确认种群替换正确
3. **对比脚本测试**：固定预算 30 次评估，运行三种方法，确认 JSON 报告和收敛曲线
4. **Streamlit UI**：分别启动 LLM 优化和对比页面，确认交互正常

---

## 9. 12 个可优化参数（共享搜索空间）

| 参数 | 类型 | 范围 | 步长 | 物理含义 |
|---|---|---|---|---|
| `floors` | int | 2 ~ 14 | 1 | 楼层数 |
| `lobby_height` | float | 3.0 ~ 9.0 | 0.1 | 首层层高 (m) |
| `floor_height` | float | 3.0 ~ 5.0 | 0.1 | 标准层层高 (m) |
| `base_x` | float | 0.0 ~ 30.0 | 0.5 | L 形角点 X 坐标 (m) |
| `base_y` | float | 0.0 ~ 30.0 | 0.5 | L 形角点 Y 坐标 (m) |
| `arm_width` | float | 6.0 ~ 30.0 | 0.5 | L 形臂宽 (m) |
| `horizontal_length` | float | 20.0 ~ 80.0 | 0.5 | 水平臂长 (m) |
| `vertical_length` | float | 20.0 ~ 80.0 | 0.5 | 垂直臂长 (m) |
| `scatter_gap` | float | 0.0 ~ 18.0 | 0.5 | 低层体块散布间距 (m) |
| `min_fragment_scale` | float | 0.3 ~ 1.0 | 0.01 | 最小碎片缩放比 |
| `merge_power` | float | 0.4 ~ 3.0 | 0.05 | 合并指数（控制渐变速度） |
| `top_solid_floors` | int | 1 ~ 5 | 1 | 顶部完整楼层数 |

固定参数：`building_name="GA_Candidate"`, `site_size=100.0`, `add_courtyard_marker=False`

约束：`lobby_height + (floors - 1) * floor_height < 50.0`（总高度 < 50m）
