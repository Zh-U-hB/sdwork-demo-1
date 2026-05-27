# Task: 直接 JSON→IDF 转换器

> 纯程序化 `model_dict → IDF → EnergyPlus simulation → results`，绕过 LLM Agent

## 前置准备

- [x] ~~**安装 eppy**~~ → **改用 idfpy**：`uv add idfpy`（无需 IDD 文件，类型安全 Pydantic 模型）
- [x] **验证 idfpy 基本功能**：`from idfpy import IDF; from idfpy.models import Building, Zone`

## Step 1: `scripts/idf_defaults.py` — 默认参数配置 ✅

- [x] 定义 dataclass：`MaterialDef`、`GlazingDef`、`ConstructionDef`、`ScheduleTypeDef`、`ScheduleDef`、`LocationDef`、`PeopleDef`、`LightDef`、`HvacDef`、`WindowDef`、`ConverterDefaults`
- [x] 实现 `make_default_settings()` 工厂函数
- [x] 默认材料：Concrete200、Insulation50、Gypsum13（Standard）；SimpleGlazing（Glazing）
- [x] 默认构造：ExteriorWallConstruction、InteriorWallConstruction、ExteriorRoofConstruction、FloorConstruction、InteriorCeilingConstruction、WindowConstruction
- [x] 默认日程：AlwaysOnSchedule、OccupancySchedule（工作日 8-18）、ActivityLevelSchedule (120 W/person)、HeatingSetpointSchedule (21°C)、CoolingSetpointSchedule (26°C)
- [x] 默认位置：深圳（22.55°N, 114.10°E）
- [x] WindowDef（wwr=0.0，默认无窗，可设 0.0-1.0）

**扩展接口示例：**
```python
defaults = make_default_settings()
defaults.location.latitude  = 39.93        # 换城市
defaults.location.name      = "Beijing"
defaults.window.wwr         = 0.4          # 40% 开窗率
defaults.people.people_per_floor_area = 0.08
result_dir = convert_and_run(model_dict, defaults=defaults)
```

## Step 2: `scripts/idf_converter.py` — 核心转换器 ✅

### 基础设施
- [x] `_init_idf(idd_path)` — 初始化 eppy IDF，自动查找 IDD（优先 `plugins/energyplus_agent/data/dependencies/Energy+.idd`）
- [x] `_add_version` / `_add_simulation_control` / `_add_timestep(4/h)` / `_add_run_period(全年)` / `_add_global_geometry_rules` / `_add_output_requests`
- [x] `_add_building(idf, name)` — terrain=Suburbs, Solar_Distribution=FullInteriorAndExterior
- [x] `_add_location(idf, loc)` — Site:Location，eppy 25.1 第6字段截断兼容

### 材料 & 构造
- [x] `_add_opaque_material` → `Material`
- [x] `_add_glazing_material` → `WindowMaterial:SimpleGlazingSystem`
- [x] `_add_construction` → `Construction`（动态 Layer_1..N）

### 几何（核心）
- [x] `_ascii_name(name, idx)` — 非 ASCII 名称自动映射为 `Zone_01` 等
- [x] `_floor_vertices` / `_ceiling_vertices` / `_south/north/west/east_wall_vertices` — 正确的 EnergyPlus 右手定则顶点顺序（UpperLeftCorner + CCW + World）
- [x] `_build_zone_geometry(idf, box, defaults, shared_wall_names)` — 每个 zone 创建 Zone 对象 + 6个 BuildingSurface:Detailed，共享墙使用 InteriorWallConstruction
- [x] `_shared_walls(boxes)` — 检测 X/Y 方向接触面（带容差），返回需要互设 boundary=Surface 的墙名对
- [x] `_patch_shared_walls(idf, pairs)` — 批量更新 boundary condition
- [x] `_window_on_wall(wall_verts, wwr)` — 按 WWR 在外墙上居中生成 FenestrationSurface:Detailed

### 荷载 & HVAC
- [x] `_add_schedule_type_limits` / `_add_schedule_compact` — Schedule:Compact 用 Field_n 填充
- [x] `_add_hvac_thermostat` / `_add_hvac_ideal_loads`
- [x] `_add_people_for_zone`（People/Area 法）
- [x] `_add_lights_for_zone`（Watts/Area 法）

### 运行
- [x] `_run_energyplus(idf_path, epw_path, output_dir)` — subprocess 调用，失败时打印 stderr tail
- [x] `convert_and_run(model_dict, output_dir, weather_file, defaults, run_simulation, idd_path)` — 串联所有步骤，返回 eplustbl.csv 所在目录

## Step 3: `scripts/ep_sim_utils.py` — 新增直接模拟路径 ✅

- [x] 添加 `run_ep_simulation_direct(model_dict, building_name, defaults, output_base, weather_file)` 函数
- [x] 调用 `idf_converter.convert_and_run()`，返回结果目录路径
- [x] 保留原 `run_ep_simulation()` 不变（向后兼容）
- [x] 更新模块文档说明两条路径的区别

## Step 4: `scripts/ga_core.py` — 切换到直接模拟路径 ✅

- [x] 将 import 从 `run_ep_simulation` 改为 `run_ep_simulation_direct as run_ep_simulation`（接口兼容，零改动）

## Step 5: 集成测试 ✅

- [x] ~~安装 eppy~~ → 已安装 idfpy：`uv add idfpy`（v26.1.0.post5）
- [x] 用 `generate_l_gradient(floors=4, top_solid_floors=2)` 生成 20-zone 测试模型，`convert_and_run()` 成功生成 IDF 并运行模拟（**2.9 秒**）
- [x] `eplustbl.csv` 正确输出，EUI = 328.2 MJ/m²（20 个 zone 能耗数据全覆盖），5 个警告 0 个严重错误
- [ ] 在 GA 模块中用 pop=3, gen=2 跑完整优化流程（待测试）
- [ ] 对比直接路径 vs MCP 路径的 EUI 结果（参考值：MCP 路径 ~328 MJ/m²）

## 关键参考文件

| 文件 | 用途 |
|---|---|
| `scripts/idf_defaults.py` | 所有默认参数定义，替换入口 |
| `scripts/idf_converter.py` | 转换器主逻辑，`convert_and_run()` 入口 |
| `scripts/ep_sim_utils.py` | `run_ep_simulation_direct()` 封装，`read_eplustbl()` 解析 |
| `scripts/ga_core.py` | GA 优化器，已切换到直接路径 |
| `plugins/energyplus_agent/src/converters/*.py` | eppy 字段名参考 |
| `plugins/energyplus_agent/data/dependencies/Energy+.idd` | IDD 文件（自动查找） |

## 依赖安装

```bash
# 主项目 venv 使用 idfpy（无需 IDD 文件，比 eppy 更现代）
uv add idfpy     # v26.1.0.post5 已安装
```

注意：idfpy 是按 EnergyPlus 26.1 schema 生成的。本项目安装 EnergyPlus 25.1，以下对象有 26.1 新增字段会导致解析失败，`idf_converter.py` 中已跳过：
- `OutputControlTableStyle`（26.1 新增 `format_numeric_values` 字段）
