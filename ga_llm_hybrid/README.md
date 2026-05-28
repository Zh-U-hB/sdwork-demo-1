# GA + LLM Hybrid Optimizer

遗传算法负责数值搜索，LLM 根据中间结果建议缩窄参数范围与种子方案，多轮迭代直至收敛。

## 快速开始

```bash
# 调试模式（5 个体 × 3 代 × 1 轮，见 configs/ga_llm_hybrid_arch.yaml）
PYTHONPATH=. python ga_llm_hybrid/main.py \
  -c configs/ga_llm_hybrid_arch.yaml \
  -o output/ga_llm_hybrid \
  -v
```

需要已配置 `.env`（`LLM_*`）且本机可运行 EnergyPlus direct 路径。

## 后端

| `energyplus.backend` | 说明 |
|---------------------|------|
| `arch_model` | `generate_20260528` → 可选分区 → `idf_converter`（与 `manual_test_app` 一致） |
| `template_idf` | 模板 IDF + `mappings` 规则 + 并行 `energyplus` 子进程 |

## 输出目录

```
output/ga_llm_hybrid/
├── round_01/
│   ├── population.csv
│   ├── pareto_front.csv
│   ├── morris_sensitivity.json
│   ├── llm_prompt.txt / llm_response.txt / llm_analysis.json
│   ├── sims/gen_XX/ind_YYYY/
│   └── summary.json
├── config.yaml
├── parameter_space.json
└── final_report.json
```

## 测试

```bash
PYTHONPATH=. pytest tests/test_ga_llm_hybrid_*.py -q
```

## 近期修复

- `arch_model` 后端对齐 `run_ep_simulation_direct(output_base, run_id)` + `read_eplustbl`
- 从 `eplustbl` 解析 EUI、峰值冷热负荷、舒适度代理指标
- `GAEngine.run(seeds=...)` 正式传参，不再 monkey-patch
- 连续/离散分支交叉与变异；Morris 独立轨迹 + `sims/morris/` 目录
- IDF 映射正则修复
