# JSON→IDF 直接转换器 — 完整实施指南

> 本文档面向不熟悉本项目的新 agent，提供完整的上下文信息和操作指引。
> 最后更新：2026-05-27

---

## 1. 项目背景

### 1.1 项目是什么

这是一个 LLM 驱动的建筑几何生成工具。用户输入自然语言描述（如"一栋3层L形办公楼"），系统生成 3D 区域几何 JSON（供 Rhino/Grasshopper 使用），并可运行 EnergyPlus 能耗模拟。

项目还包含一个参数化 L 形建筑生成器 `scripts/generate_l_gradient.py`，可通过调节数学参数（楼层数、臂宽、散布间距等）生成不同的 L 形建筑变体。

### 1.2 为什么要做这个任务

原始的 EnergyPlus 模拟流程依赖 LLM Agent 通过 MCP 工具链逐步创建 IDF 对象（zone、surface、material 等），每次模拟需要 LLM 逐步调用 13 个步骤，耗时 2-5 分钟且结果不可预测。

**目标**：将 JSON→IDF 的转换过程改为纯程序化实现，直接使用 eppy 库创建 IDF 对象，彻底绕过 LLM。转换后用 subprocess 调用 `energyplus` 命令运行模拟。整个过程从 2-5 分钟缩短到几秒钟。

### 1.3 当前状态

代码已经实现完毕（Step 1-4），但 **集成测试尚未执行**（Step 5）。你需要完成集成测试，确保整个流程可以正常运行。

---

## 2. 代码架构

### 2.1 数据流

```
generate_l_gradient() → model_dict (JSON)
                              ↓
                  convert_and_run()
                              ↓
              eppy IDF 对象 (zone/surface/material/schedule/HVAC)
                              ↓
                   building.idf 文件
                              ↓
              subprocess 调用 energyplus 命令
                              ↓
                    eplustbl.csv 结果文件
                              ↓
                  read_eplustbl() → EUI (MJ/m²)
```

### 2.2 关键文件

| 文件 | 状态 | 作用 |
|---|---|---|
| `scripts/idf_defaults.py` | ✅ 已实现 | 所有默认参数定义（材料、构造、日程表、人员、照明、HVAC） |
| `scripts/idf_converter.py` | ✅ 已实现 | 核心转换器，`convert_and_run()` 入口函数 |
| `scripts/ep_sim_utils.py` | ✅ 已更新 | 新增 `run_ep_simulation_direct()` 封装函数 |
| `scripts/ga_core.py` | ✅ 已更新 | GA 优化器已切换到直接模拟路径 |
| `scripts/generate_l_gradient.py` | 无需修改 | 参数化 L 形建筑生成器，产生输入数据 |
| `ga_optimizer_app.py` | 无需修改 | Streamlit UI，调用 ga_core |
| `parametric_l_app.py` | 无需修改 | 参数化调参 UI |

### 2.3 模型数据格式

`generate_l_gradient()` 返回的 `model_dict` 结构如下：

```python
{
    "building_name": "Gradient L Office",
    "zones": [
        {
            "name": "F01_corner_core",       # 区域名（可能含中文）
            "origin": {"x": 18.0, "y": 16.0, "z": 0.0},  # 左下前角
            "dimensions": {
                "length": 13.5,   # X 方向
                "width": 13.5,    # Y 方向
                "height": 5.5     # Z 方向
            },
            "points": [...]       # 8 个顶点坐标（本转换器不用此字段）
        },
        # ... 更多区域，典型 50-80 个
    ]
}
```

转换器只使用 `name`、`origin`、`dimensions` 三个字段，每个 zone 被视为一个矩形盒子（box）。

---

## 3. 已实现代码详解

### 3.1 `scripts/idf_defaults.py` — 默认参数

这个文件定义了所有 EnergyPlus 模拟需要的默认参数，以 Python dataclass 组织：

**核心 dataclass 类型：**

| Dataclass | 作用 | 关键字段 |
|---|---|---|
| `MaterialDef` | 不透明材料（混凝土、保温层、石膏板） | name, roughness, thickness, conductivity, density, specific_heat |
| `GlazingDef` | 玻璃材料 | name, u_factor, solar_heat_gain_coefficient, visible_transmittance |
| `ConstructionDef` | 构造组合 | name, layers (材料名列表，外→内) |
| `ScheduleTypeDef` | 日程值范围限制 | name, lower_limit, upper_limit, numeric_type, unit_type |
| `ScheduleDef` | 紧凑日程表 | name, type_limits_name, data (字符串列表) |
| `LocationDef` | 地理位置信息 | name, latitude, longitude, time_zone, elevation |
| `PeopleDef` | 人员荷载 | people_per_floor_area=0.05, fraction_radiant=0.3 |
| `LightDef` | 照明荷载 | watts_per_floor_area=10.0, fraction_radiant=0.32 |
| `HvacDef` | HVAC 参数 | thermostat_name, heating/cooling_setpoint_schedule_name |
| `WindowDef` | 窗户参数 | wwr=0.0 (默认无窗), construction_name |
| `ConverterDefaults` | 顶层组合容器 | 包含以上所有 |

**工厂函数 `make_default_settings()` 返回完整默认配置：**

- **位置**：深圳 (22.55°N, 114.10°E, UTC+8)
- **材料**：Concrete200（混凝土）、Insulation50（保温）、Gypsum13（石膏板）、SimpleGlazing（玻璃）
- **构造**：6 种（外墙、内墙、屋顶、楼板、天花板、窗户）
- **日程表**：5 种（常开、人员占用、活动水平、供暖设定点 21°C、供冷设定点 26°C）

**覆盖方式**（用户可只改需要改的）：
```python
defaults = make_default_settings()
defaults.location.latitude = 39.93      # 换城市
defaults.window.wwr = 0.4              # 40% 窗墙比
defaults.people.people_per_floor_area = 0.08
result_dir = convert_and_run(model_dict, defaults=defaults)
```

### 3.2 `scripts/idf_converter.py` — 核心转换器

**入口函数 `convert_and_run()`**：

```python
def convert_and_run(
    model_dict: dict,            # BuildingModel JSON
    output_dir: str | Path,      # 输出目录
    weather_file: str | None,    # EPW 天气文件路径（None=自动查找）
    defaults: ConverterDefaults | None,  # 默认参数（None=使用内置默认）
    run_simulation: bool = True, # False=只生成 IDF 不运行
    idd_path: str | None,        # IDD 文件路径（None=自动查找）
) -> str | None:                 # 返回 eplustbl.csv 所在目录，或 None
```

**处理流程：**

1. **过滤区域**：跳过高度 < 1.0m 的区域（如庭院标记）
2. **名称处理**：中文名称自动映射为 `Zone_01`、`Zone_02` 等 ASCII 名称
3. **共享墙检测**：在构建 IDF 之前，先检测所有相邻 zone 的接触面
4. **初始化 IDF**：使用 eppy + IDD 文件创建空 IDF
5. **添加非几何对象**：Version、SimulationControl、Timestep、RunPeriod、GlobalGeometryRules、Building、Location、Material、Construction、Schedule、HVAC Thermostat
6. **添加几何对象**：每个 zone → Zone 对象 + 6 个 BuildingSurface:Detailed（地板、天花板、东南西北墙）
7. **修补共享墙**：将检测到的共享墙对互相设 boundary=Surface
8. **添加区域荷载**：每个 zone → People + Lights + IdealLoadsAirSystem
9. **保存 IDF**：写入文件
10. **运行模拟**：subprocess 调用 `energyplus` 命令

**几何计算核心**：

每个 box zone (origin=(ox,oy,oz), dimensions=(L,W,H)) 生成 6 个面，顶点按 EnergyPlus 的 UpperLeftCorner + CounterClockwise + World 坐标系规则排列：

```
Floor   (z=oz,   boundary=Ground):     CCW viewed from below
Ceiling (z=oz+H, boundary=Outdoors):   CCW viewed from above
Wall_S  (y=oy,   outward normal=-Y):   CCW viewed from outside
Wall_N  (y=oy+W, outward normal=+Y):   CCW viewed from outside
Wall_W  (x=ox,   outward normal=-X):   CCW viewed from outside
Wall_E  (x=ox+L, outward normal=+X):   CCW viewed from outside
```

**共享墙检测算法**：

遍历所有 zone 对，检查 X 或 Y 方向是否有接触面（容差 0.01m）：
- Zone A 的东墙 (x_max) ≈ Zone B 的西墙 (x_min)，且 Y/Z 范围重叠 → 共享
- Zone A 的北墙 (y_max) ≈ Zone B 的南墙 (y_min)，且 X/Z 范围重叠 → 共享

匹配后双方设 `Outside_Boundary_Condition=Surface`、`Outside_Boundary_Condition_Object=对方名称`、`Sun_Exposure=NoSun`、`Wind_Exposure=NoWind`、`Construction_Name=InteriorWallConstruction`。

**窗户生成**（当 `wwr > 0` 时）：

在外墙上按窗墙比 (WWR) 居中生成矩形 FenestrationSurface:Detailed。宽度和高度 = WWR × 墙面尺寸。

**EnergyPlus 运行**：

```python
cmd = ["energyplus", "-x", "-w", epw_path, "-d", output_dir, "-r", idf_path]
subprocess.run(cmd, capture_output=True, text=True)
```

### 3.3 `scripts/ep_sim_utils.py` — 模拟工具

此文件有两条模拟路径：

| 函数 | 方式 | 状态 |
|---|---|---|
| `run_ep_simulation()` | 通过 MCP Agent（LLM 驱动），保留向后兼容 | 旧路径 |
| `run_ep_simulation_direct()` | 直接调用 `idf_converter.convert_and_run()` | 新路径（快） |

新路径 `run_ep_simulation_direct()` 已实现，它会：
1. 调用 `convert_and_run()` 生成 IDF 并运行模拟
2. 返回包含 `eplustbl.csv` 的结果目录路径

其他重要函数：
- `resolve_weather_path()` → 返回 EPW 天气文件绝对路径
- `read_eplustbl(result_dir)` → 解析 eplustbl.csv，提取 EUI、分区能耗等
- `model_energy_map(model, sim_data)` → 将能耗数据映射回原始 zone

### 3.4 `scripts/ga_core.py` — GA 优化器

**已完成的修改**：`evaluate_fitness()` 中的 import 从 `run_ep_simulation` 改为 `run_ep_simulation_direct as run_ep_simulation`，接口完全兼容，无需其他改动。

GA 优化器的完整流程：
1. 随机初始化种群（12 个参数）
2. 评估适应度：生成模型 → 检查约束（高度 < 50m）→ 运行 EP 模拟 → 计算 EUI
3. 锦标赛选择 + BLX-α 交叉 + 高斯变异
4. 精英保留
5. 逐代迭代，记录最优解

---

## 4. 你需要做的事情（Step 5：集成测试）

### 4.1 前置条件

**安装 eppy**（主项目 venv 中尚未安装）：

```bash
cd /root/ls/sdwork-demo-1
uv add eppy
```

验证安装：
```bash
uv run python -c "from eppy.modeleditor import IDF; print('eppy OK')"
```

**确认 EnergyPlus 可用**：
```bash
which energyplus
energyplus --version
```

预期：EnergyPlus 25.1.0，安装在 `/usr/local/EnergyPlus-25-1-0/`。

**确认天气文件存在**：
```bash
ls -la plugins/energyplus_agent/data/weather/Shenzhen.epw
```

**确认 IDD 文件存在**：
```bash
ls -la plugins/energyplus_agent/data/dependencies/Energy+.idd
```

### 4.2 测试 1：基本转换器功能

创建测试脚本 `scripts/test_converter.py`：

```python
"""Quick smoke test for idf_converter."""
from scripts.generate_l_gradient import generate_l_gradient
from scripts.idf_converter import convert_and_run
from scripts.idf_defaults import make_default_settings

# 生成一个小型测试模型
model = generate_l_gradient(
    floors=3,
    lobby_height=5.5,
    floor_height=4.0,
    base_x=18.0,
    base_y=16.0,
    arm_width=13.5,
    horizontal_length=40.0,
    vertical_length=35.0,
    scatter_gap=6.0,
    min_fragment_scale=0.5,
    merge_power=1.0,
    top_solid_floors=2,
    building_name="TestBuilding",
    site_size=100.0,
    add_courtyard_marker=False,
)

print(f"Model has {len(model['zones'])} zones")

# 只生成 IDF，不运行模拟
result = convert_and_run(
    model,
    output_dir="output/test_converter",
    run_simulation=False,
)
print(f"IDF generated at: {result}")

# 检查 IDF 文件存在
from pathlib import Path
idf_files = list(Path("output/test_converter").glob("*.idf"))
assert idf_files, "No IDF file generated!"
print(f"IDF file: {idf_files[0]}")
print(f"IDF size: {idf_files[0].stat().st_size} bytes")
```

运行：
```bash
cd /root/ls/sdwork-demo-1
uv run python scripts/test_converter.py
```

**预期**：成功生成 IDF 文件，大小约 50-200 KB（取决于区域数量）。

如果出错，检查：
- eppy 是否安装成功
- IDD 文件路径是否正确
- zone 名称是否包含非 ASCII 字符（应被自动映射为 `Zone_01` 等）

### 4.3 测试 2：完整模拟运行

```python
"""Test full simulation pipeline."""
from scripts.generate_l_gradient import generate_l_gradient
from scripts.idf_converter import convert_and_run
from scripts.ep_sim_utils import read_eplustbl

model = generate_l_gradient(
    floors=3,
    lobby_height=5.5,
    floor_height=4.0,
    base_x=18.0,
    base_y=16.0,
    arm_width=13.5,
    horizontal_length=40.0,
    vertical_length=35.0,
    scatter_gap=6.0,
    min_fragment_scale=0.5,
    merge_power=1.0,
    top_solid_floors=2,
    building_name="TestSim",
    site_size=100.0,
    add_courtyard_marker=False,
)

result_dir = convert_and_run(
    model,
    output_dir="output/test_full_sim",
    run_simulation=True,
)

if result_dir:
    print(f"Simulation succeeded: {result_dir}")
    sim = read_eplustbl(result_dir)
    print(f"Results exist: {sim['exists']}")
    print(f"Site energy: {sim['site_energy']}")
    print(f"Building area: {sim['building_area']}")
    if sim['building_area'].get('Net Conditioned Building Area', 0) > 0:
        total_gj = sim['site_energy'].get('Total Site Energy', 0)
        area = sim['building_area']['Net Conditioned Building Area']
        eui = total_gj * 1000 / area
        print(f"EUI: {eui:.1f} MJ/m²")
else:
    print("Simulation FAILED")
    # 检查 EnergyPlus 输出的错误日志
    import glob
    for err in glob.glob("output/test_full_sim/**/*.err", recursive=True):
        with open(err) as f:
            lines = f.readlines()
            print(f"\n--- {err} (last 30 lines) ---")
            for line in lines[-30:]:
                print(line.rstrip())
```

运行：
```bash
uv run python -c "exec(open('scripts/test_full_sim.py').read())"  # 或直接运行测试脚本
```

**预期**：
- 模拟成功完成（约 30 秒 - 2 分钟）
- 生成 `eplustbl.csv`
- EUI 在 100-500 MJ/m² 范围内（深圳办公建筑合理范围）

如果模拟失败：
1. 检查 `*.err` 文件中的错误信息
2. 常见错误：
   - `**FATAL ERROR**` → IDF 对象引用错误，检查 construction/material 名称一致性
   - `Zone <name> not found` → zone 名称映射问题
   - `Surface vertex errors` → 顶点顺序或坐标问题
   - `Schedule not found` → schedule 名称引用问题

### 4.4 测试 3：带窗户模拟

```python
from scripts.idf_defaults import make_default_settings

defaults = make_default_settings()
defaults.window.wwr = 0.3  # 30% 窗墙比

result_dir = convert_and_run(model, output_dir="output/test_windows",
                              defaults=defaults, run_simulation=True)
# 检查 EUI 应该比无窗版本更高（因为玻璃 U 值更高）
```

### 4.5 测试 4：GA 优化器集成

用小参数测试 GA 优化器是否能正常工作：

```bash
streamlit run ga_optimizer_app.py
```

在 UI 中设置：
- 种群大小 = 3
- 代数 = 2
- 点击"开始优化"

**预期**：GA 运行，每代评估 3 个个体，2 代后完成。每个个体约 30 秒-2 分钟。总耗时约 5-15 分钟。

或者用 Python 直接测试：

```python
from scripts.ga_core import run_ga, GAConfig

config = GAConfig(
    pop_size=3,
    n_gen=2,
    mutation_rate=0.15,
    elite_count=1,
    cache_path="output/ga_test_cache.json",
    checkpoint_path="output/ga_test_checkpoint.json",
)

for result in run_ga(config, seed=42):
    print(f"Gen {result.gen}: best={result.best_fitness:.1f}, "
          f"avg={result.avg_fitness:.1f}, worst={result.worst_fitness:.1f}")
```

### 4.6 常见问题排查

| 问题 | 原因 | 解决方案 |
|---|---|---|
| `ModuleNotFoundError: No module named 'eppy'` | 主项目未安装 eppy | `uv add eppy` |
| `FileNotFoundError: Energy+.idd not found` | IDD 文件路径错误 | 检查 `plugins/energyplus_agent/data/dependencies/Energy+.idd` 是否存在 |
| `EnergyPlus executable not found` | energyplus 不在 PATH | 确认 `which energyplus` 有输出 |
| IDF 生成成功但模拟失败 | IDF 内容错误 | 检查 `*.err` 日志，对照 EP Agent 的成功运行结果 |
| zone 名称错误 | 中文名称映射 | `_ascii_name()` 自动处理中文 → `Zone_01` |
| 共享墙检测遗漏 | 容差太小 | 默认 `tol=0.01m`，通常足够 |
| `Site:Location` 字段数错误 | EP 25.1 的 IDD/schema 不匹配 | `_add_location()` 中已有 `obj.obj = obj.obj[:6]` 截断处理 |

---

## 5. 依赖项参考

### 5.1 已安装的依赖

```
langchain, langchain-anthropic, langchain-core, langgraph
langchain-mcp-adapters, pydantic>=2.0, plotly, python-dotenv
pyyaml, streamlit>=1.30, httpx[socks]
```

### 5.2 需要安装的依赖

```bash
uv add eppy    # IDF 文件操作库
```

### 5.3 系统依赖

- **EnergyPlus 25.1.0**：`/usr/local/EnergyPlus-25-1-0/`
- **Python**：>=3.12
- **uv**：包管理器

---

## 6. EP Agent 转换器参考（仅供参考，不需要修改）

原始 LLM 路径使用的转换器在 `plugins/energyplus_agent/src/converters/` 下。以下是 eppy 创建 IDF 对象时使用的字段名参考（新转换器中已使用相同的字段名）：

| IDF 对象类型 | eppy 字段名示例 |
|---|---|
| `Material` | `Name`, `Roughness`, `Thickness`, `Conductivity`, `Density`, `Specific_Heat` |
| `WindowMaterial:SimpleGlazingSystem` | `Name`, `UFactor`, `Solar_Heat_Gain_Coefficient`, `Visible_Transmittance` |
| `Construction` | `Name`, `Layer_1`, `Layer_2`, ... |
| `Zone` | `Name` |
| `BuildingSurface:Detailed` | `Name`, `Surface_Type`, `Construction_Name`, `Zone_Name`, `Outside_Boundary_Condition`, `Sun_Exposure`, `Wind_Exposure`, `Vertex_N_Xcoordinate` 等 |
| `FenestrationSurface:Detailed` | `Name`, `Surface_Type`, `Construction_Name`, `Building_Surface_Name`, `Vertex_N_Xcoordinate` 等 |
| `ScheduleTypeLimits` | `Name`, `Lower_Limit_Value`, `Upper_Limit_Value`, `Numeric_Type`, `Unit_Type` |
| `Schedule:Compact` | `Name`, `Schedule_Type_Limits_Name`, `Field_1`, `Field_2`, ... |
| `HVACTemplate:Thermostat` | `Name`, `Heating_Setpoint_Schedule_Name`, `Cooling_Setpoint_Schedule_Name` |
| `HVACTemplate:Zone:IdealLoadsAirSystem` | `Zone_Name`, `Template_Thermostat_Name` |
| `People` | `Name`, `Zone_or_ZoneList_or_Space_or_SpaceList_Name`, `Number_of_People_Calculation_Method`, `People_per_Floor_Area`, `Fraction_Radiant`, `Activity_Level_Schedule_Name` |
| `Lights` | `Name`, `Zone_or_ZoneList_or_Space_or_SpaceList_Name`, `Schedule_Name`, `Design_Level_Calculation_Method`, `Watts_per_Floor_Area`, `Fraction_Radiant`, `Fraction_Visible` |
| `SimulationControl` | `Do_Zone_Sizing_Calculation`, `Run_Simulation_for_Weather_File_Run_Periods` 等 |
| `Timestep` | `Number_of_Timesteps_per_Hour` |
| `RunPeriod` | `Name`, `Begin_Month`, `Begin_Day_of_Month`, `End_Month`, `End_Day_of_Month` |
| `GlobalGeometryRules` | `Starting_Vertex_Position`, `Vertex_Entry_Direction`, `Coordinate_System` |
| `Site:Location` | `Name`, `Latitude`, `Longitude`, `Time_Zone`, `Elevation` |
| `Building` | `Name`, `North_Axis`, `Terrain`, `Solar_Distribution` 等 |

---

## 7. 成功标准

集成测试通过的标准：

1. `uv add eppy` 安装成功
2. `convert_and_run(run_simulation=False)` 能生成合法的 IDF 文件
3. `convert_and_run(run_simulation=True)` 模拟成功，生成 `eplustbl.csv`
4. `read_eplustbl()` 能正确解析结果，EUI 在合理范围（100-500 MJ/m²）
5. GA 优化器能用直接路径完成至少 1 代的优化
6. 带窗户（wwr=0.3）的模拟也能正常运行

全部通过后，更新 `TASKS_idf_converter.md` 中 Step 5 的 checkbox。
