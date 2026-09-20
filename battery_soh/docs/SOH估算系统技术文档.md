# 换电电池 SOH 估算系统 技术文档

> 版本：v2.0 ｜ 更新日期：2026-09-20 ｜ 项目路径：`battery_soh/`
> 依据：《换电电池SOH估算工程方案_v2修正版》（docs/换电电池SOH估算工程方案_v2修正版.docx）

---

## 1. 项目概况

### 1.1 目标

从换电柜 BMS 上报的原始时序数据（电压 / 电流 / SOC / 剩余容量 / 温度）中，**不依赖库仑积分**（电流永不累加），用"锚点法"估算每块电池的 SOH（健康状态 = 当前满充容量 / 标称容量）月度序列，进而得到衰退速率与退役预测，并支持按电池型号分类对比。

### 1.2 当前生产数据集

| 项目 | 内容 |
|---|---|
| 源数据 | `data/` 下 5 个 parquet 文件，约 **4.37 亿行**，时间跨度 2024-03-10 → 2026-09-14 |
| 设备清单 | `device_model/device_model.csv`，共 **10,815 台**（全部命中） |
| 型号 | 四美7237（3,816 台）、海池7237（6,999 台），标称容量均为 **37Ah** |
| 体系规格 | 据 `device_model/电池参数.csv`：**铁锂 LFP 24S**（单体 3.2/3.65V，合计 76.8/87.6V）；Step 0 平台斜率自动判别曾误判为 NMC 20S，已按规格书覆写 `config.generated.yaml` 重跑 |
| 运行路线 | R_MODE=DYNAMIC，ABSOLUTE（绝对容量法） |
| 有效产出 | 设备×月聚合 145,756 行（10,797 台有月点），可预测退役 10,529 台 |

### 1.3 核心结论（2026-09 快照，LFP 24S 体系）

| 型号 | 设备数 | 衰退速率中位 | P25 | P75 | 起点 SOH | 最新 SOH |
|---|---|---|---|---|---|---|
| 四美7237 | 3,800 | **-4.87%/年** | -7.36 | -2.94 | 1.019 | 0.942 |
| 海池7237 | 6,893 | **-5.64%/年** | -9.11 | -4.20 | 1.023 | 0.977 |

### 1.4 技术栈

- Python 3.13、polars 1.44.2（LazyFrame 流式引擎）、numpy、pyarrow（ParquetWriter 追加写）、PyYAML、matplotlib（趋势图）、pytest（39 个测试）
- 断点续传：SQLite（`output/progress.db`）

---

## 2. 总体架构

### 2.1 数据流

```
data/*.parquet (4.37亿行, 5文件)
      │  prep.py：过滤设备清单 + 电池id%32 分桶（流式，逐文件）
      ▼
data/filtered/part_00..31.parquet        ← Map 的处理与续传单元
      │
      │  Step 0 体检（抽样+streaming）→ 体系/路线/C_nom_BMS → config.generated.yaml
      │
      │  Step 1~5（Map）：每分片 × 每 64 台设备一批
      │    清洗 → 锚点打标 → 容量+温度补偿 → 质量评分分级 → 月聚合
      │    锚点行落盘 output/observation/parts/anchor_bXXXX.parquet（不进内存）
      │    月聚合落盘 output/monthly/parts/monthly_bXXXX.parquet
      │    进度写 progress.db（按设备），断点续传
      ▼
output/monthly/monthly_capacity.parquet（设备×月，万行级）
      │
      │  Step 6~8（Reduce，月聚合层）
      │    6: MAD去毛刺 + PAVA单调回归 + SOH
      │    7: 衰退速率/分析集/绘图集/告警/渠道看板
      │    8: 退役预测（线性外推+95%CI）
      ▼
output/soh | retirement | reports
      │
      │  report_model.py：join device_model.csv 的 电池型号 列
      ▼
output/reports/model_trend.csv + model_decay_stats.csv + model_soh_trend.png
```

### 2.2 关键工程设计

| 问题 | 方案 |
|---|---|
| 4.4 亿行无法进内存 | 全链路 polars `scan_parquet` + `sink_batches` 流式；锚点行只落盘不驻留；聚合层（设备×月）才 collect |
| 每批全表扫描代价高 | prep 按 `电池id % 32` 分桶，同一设备必落同一桶；Map 以桶为单元 |
| 长跑中断 | `progress.db` 记录 (step, channel, device, ym) 状态；重跑自动跳过已完成设备（标记在落盘**之后**写入，保证幂等） |
| 流式 join 大表 OOM | 型号列不在 prep 附加，改在报告层 join（1 万行映射表） |
| 亿级行 Python 循环物化 | Step 0 设备级检查用哈希抽样（`电池id % 25 == 0`，约 4%）；全量聚合走 streaming |
| 跨文件/文件内重复行 | `iter_device_batches` 内按 (电池id, 更新时间) `unique(keep="first")` 去重 |
| 阈值硬编码 | 铁律 P05：全部阈值集中在 `config/thresholds.yaml`，代码零硬编码 |
| 配置漂移 | 三段合并：`config.yaml` < `thresholds.yaml` < `config.generated.yaml`（Step 0 实测），启动 fail-fast 校验 |

### 2.3 目录结构

```
battery_soh/
├── run.py                  CLI 入口（--step / --from-month / --dry-run / --prep / --force）
├── config/
│   ├── config.yaml         人工基线（数据路径、设备清单、c_nom_spec）
│   ├── thresholds.yaml     全部阈值（体系/锚点/质量/SOH/退役/监控/自检）
│   ├── config.generated.yaml  Step 0 实测体系参数（自动覆写）
│   └── ocv_tables/         nmc_20s / nmc_21s / lfp_24s / lfp_25s（soc, ocv_v，严格单调）
├── src/
│   ├── constants.py        枚举：AnchorType / Tier / Flag / RMode / Route + 中文列名
│   ├── schema.py           原始/锚点 schema、OCV 表校验、dtype 强制
│   ├── validation.py       行级检查 + 汇总统计
│   ├── config.py           三段配置合并、Config 视图、路径解析
│   ├── io.py               流式扫描 / 设备分批 / 分区写 / ProgressDB
│   ├── prep.py             源数据过滤 + 32 桶分片
│   ├── step0_profile.py    体检（7 项）→ 路线决策
│   ├── step1_clean.py      清洗 + 电流零点校准
│   ├── step2_anchor.py     四类锚点打标 + soc_anchor
│   ├── step3_capacity.py   容量读取 + 温度补偿
│   ├── step4_quality.py    异常标记 + 评分 + Tier 分级
│   ├── step5_monthly.py    加权中位数月聚合
│   ├── step6_soh.py        MAD 去毛刺 + PAVA + SOH
│   ├── step7_trend.py      衰退速率 / 三集输出 / 渠道看板
│   ├── step8_retirement.py 退役预测（线性外推 + CI）
│   ├── analysis.py         三道自检 + 渠道看板指标（纯函数）
│   ├── pipeline.py         Map/Reduce 编排 + 断点续传
│   └── report_model.py     分型号 SOH 趋势报告
├── tests/                  conftest + 6 个测试文件（39 用例）
├── data/                   源 parquet + filtered/ 分片
├── device_model/           device_model.csv（电池id, 电池型号）
├── output/                 profile / observation / monthly / soh / retirement / reports / progress.db
├── logs/pipeline.log
└── docs/                   工程方案 + 本文档
```

---

## 3. 核心算法：锚点法

### 3.1 铁律（方案第二章）

- **禁止** `(剩余容量_end − 剩余容量_start) / (SOC_end − SOC_start)` —— 这是 BMS 自身算法的恒等式回声，无信息量。
- **允许** `剩余容量 / SOC_anchor`，且仅当该点的 SOC 被**独立物理事实**锚定（电压或业务事件）。
- 电流只参与符号与静置判据，**永不累加**。

### 3.2 四类锚点（Step 2）

| 代码 | 名称 | 判据 | soc_anchor 取值 |
|---|---|---|---|
| B | SWAP_FULL 站内满充完成 | I 由正(>5A)转静置(\|I\|<0.02C) + SOC≥0.98 + V < V_full×0.995 | **1.0**（业务事实） |
| C | EMPTY_END 放空末端 | SOC<0.10 + V < V_cutoff×1.05 + I<0 | max(SOC, 0.02)（防除零） |
| A | FULL_END 满充末端 | SOC>0.95 + V > V_full×0.99 + 0<I<5A（涓流） | max(SOC, 0.95) |
| D | OCV 静置开路电压 | \|I\|<0.02C 持续≥2h，取**静置段末行** | OCV 表反查值，clip [0.02, 0.98] |

- **互斥优先级 B > C > A > D**；同一行多命中打 `anchor_conflict` 标记。
- 静置持续时长用向量化累计（连续段内累加 dt，断档 >6h 或段首归零），OCV 锚点仅取每段末行，避免千万级冗余锚点。
- 体系参数（V_full=87.6V、V_cutoff=62.4V，**LFP 24S**，据电池参数.csv 规格书）来自配置，非硬编码。

### 3.3 容量与温度补偿（Step 3）

- `capacity_raw = 剩余容量 / soc_anchor`
- `capacity_corrected = capacity_raw × (1 + α × (T_ref − T))`，NMC α=0.0025/°C、LFP α=0.005/°C、T_ref=25°C。不补偿则季节性温差造成 ~15% 伪波动。
- 超界不删行，只打标记：`out_of_range`（容量比 [0.65,1.10] 外）、`temp_out`（温度 [5,45] 外）、`v_inconsistent`（放空端电压仍在平台 / 满充端电压偏低，用单体电压粗校验）。
- `C_nom_BMS`：Step 0 设备级 R 众数（0.5Ah 分箱取最密箱中值），**禁止用规格书**（P04）；缺失时回退全量 R 中位数。

### 3.4 质量评分与分级（Step 4）

评分 = 锚点类型基础分 + 温度/容量/电压/矛盾/漂移加扣分（见 §4.3）。分级：

| Tier | 条件 | 用途 |
|---|---|---|
| 1 | 评分≥30 且无严重异常 且 **7 天内有另一锚点支撑**（闭环判据） | 进 SOH 统计与退役判定 |
| 2 | 评分 10~29，或 Tier-1 条件但缺闭环支撑 | 保留观察，权重 ×0.3 |
| 3 | 其余 | 严禁入统计，仅绘图回溯 |

### 3.5 月聚合（Step 5）

- 分组键 [渠道号, 电池id, 年, 月]；聚合 = **加权中位数**（权重 = 锚点类型权重 × Tier 系数）。
- 同时输出：锚点数、Tier-1/2 计数、温度均值、评分均值、soh_bms 代理（剩余容量/c_nom 中位数）。
- 几十亿行 → 设备×月量级（本数据集 145,933 行）。

### 3.6 SOH 时序处理（Step 6）

1. **滚动 MAD 去毛刺**：窗口 7（|x−窗口中位| > 3×1.4826×MAD 判离群，零尺度退化保护），离群点用窗口内非离群中位数替换。自适应，禁固定阈值（P07）。
2. **PAVA 单调回归**：Pool Adjacent Violators 加权非增回归（电池容量只降不升），O(n)，保趋势不过度平滑。
3. **SOH = C_smoothed / denom**，denom = 该设备**首年 capacity_clean 中位数**（设备级 C_nom 口径，禁止规格书）。

### 3.7 衰退与退役（Step 7/8）

- 衰退速率：设备 SOH 月序列最小二乘斜率 → %/月 ×12 = %/年，附 R²。
- 退役预测：线性外推 SOH 跌破 0.80 的月份；95% 置信区间 = 反解 y = 0.80 ± 1.96×残差标准误；月数 <3 标"数据不足"，R²<0.6 标"仅供参考"，斜率≥0 不外推。
- 健康分级：≥0.85 健康 / ≥0.75 关注 / ≥0.65 预警 / <0.65 退役。

### 3.8 三道自检（analysis.py）

| 自检 | 内容 | 判据 |
|---|---|---|
| ① 满充 vs 放空交叉 | 同设备 A/B 与 C 两类锚点容量中位数对比 | \|bias\|<5% ok；否则标记可疑端整体降权 |
| ② R 漂移方向 | 设备级 capacity_raw(t) 拟合斜率 | 非负占比 >70% → 绝对值不可信，只信排序 |
| ③ 群体一致性 | 设备 SOH 与渠道均值曲线相关系数 | corr<0.3 → lagging（落后单体） |

---

## 4. 参数指标总表

### 4.1 体系参数（thresholds.yaml，Step 0 实测覆写）

| 参数 | LFP | NMC | 说明 |
|---|---|---|---|
| v_full | 91.25V (25S×3.65) | 88.2V (21S×4.2) | 满充总压；本批规格书 87.6V (24S LFP) |
| v_cutoff | 65.0V (25S×2.6) | 63.0V | 截止总压；本批规格书 62.4V (24S×2.6V) |
| α 温度系数 | 0.005/°C | 0.0025/°C | T_ref=25°C |

### 4.2 锚点判据

| 参数 | 值 |
|---|---|
| soc_full / soc_swap_full / soc_empty | 0.95 / 0.98 / 0.10 |
| v_full_ratio / v_cutoff_ratio | 0.99 / 1.05 |
| full_current_max_a（涓流上限） | 5.0A |
| rest_c_rate（静置） | 0.02C |
| duration_hours / gap_hours | 2h / 6h |
| deadband_a / sensor_drift_a | 0.2A / 2.0A |
| current_zero_frac_limit | \|I\|<1A 占比 >90% → 电流实质缺失 |

### 4.3 质量评分与阈值

| 项 | 分值/阈值 |
|---|---|
| 锚点基础分 | SWAP_FULL 20 / EMPTY_END 20 / FULL_END 15 / OCV 10 |
| 温度 | 正常区间 [15,35]°C +5；越界 [5,45] −10 |
| 容量 | 正常比 [0.75,1.00] +10；异常比 [0.50,1.20] 外 −20 |
| 电压 | 自洽 +10 / 不自洽 −15 |
| 前后矛盾（相邻锚点差>30%） | −15 |
| 传感器漂移 | −10 |
| tier1_min / tier2_min | 30 / 10 |
| 严重异常 | out_of_range, voltage_inconsistent, anchor_conflict, contradiction |
| 锚点权重（加权中位数） | B 1.0 / C 1.0 / A 0.8 / D 0.5；Tier-2 ×0.3 |

### 4.4 SOH / 退役 / 监控

| 参数 | 值 |
|---|---|
| mad_window / mad_threshold | 7 / 3.0×MAD |
| monthly_drop_alert | 单设备 SOH 月跌 >5% 告警 |
| soh_retire | 0.80 |
| health_levels | [0.85, 0.75, 0.65] |
| min_months / r2_min / ci_z | 3 / 0.6 / 1.96 |
| device_anchor_rate_min | 设备 Tier-1 锚点率 <20% → "数据质量差" |
| channel_anchor_rate_min / drop_mom | 30% / 环比跌 30% 告警 |
| selfcheck: r_cv_dynamic / r_slope_eps / neg_loopnum_frac / min_anchor_per_month | 0.02 / 0.0005 / 0.01 / 3 |

### 4.5 数据工程参数

| 参数 | 值 | 位置 |
|---|---|---|
| N_PARTS 分桶数 | 32 | prep.py |
| sink chunk_size | 1,000,000 行 | prep.py |
| batch_devices | 64 台/批 | pipeline.py |
| SAMPLE_MOD（Step 0 抽样） | 25（≈4% 设备） | step0_profile.py |
| c_nom_spec（报告基准） | 37.0Ah | config.yaml |

---

## 5. 各模块功能说明

### 5.1 prep.py — 数据准备（一次性）

逐源文件 `scan_parquet → filter(电池id ∈ 清单) → 加 _part 列 → sink_batches(100万/块)`，回调内 `partition_by("_part")` 后用 `pyarrow.parquet.ParquetWriter` 追加写对应桶。`is_ready()` 检查 32 桶齐全；`--force` 重建。产出 437,290,468 行 / 32 分片。

### 5.2 step0_profile.py — 体检（决定路线）

7 项检查：① R=剩余容量/SOC 的 CV 与斜率（DYNAMIC/CONSTANT）；② 体系判别（平台区 dV/dSOC 斜率 + 串数候选 {LFP:24,25 / NMC:20,21} 电压对齐）；③ 采样间隔分布；④ 循环次数单调性；⑤ 电流符号抽检（SOC 上升段 I 应正）；⑥ 电流零点占比（实质缺失检测）；⑦ 锚点候选密度。
输出：`diagnosis_report.json`、`config.generated.yaml`（覆写 config/ 下同名文件）、`device_list_valid.csv`（设备级 C_nom_BMS + 有效性：n_rows≥50 且跨度≥14 天）。
路线决策：DYNAMIC→ABSOLUTE；CONSTANT 且锚点密度≥3/月→ABSOLUTE；否则降级 L2_TREND（跳过 Step 1~8）。

### 5.3 step1_clean.py — 清洗

时间空删行；非数值打标记不删行；Step 0 结论的 reversed_devices 电流整体取反；每设备低电流段均值作零点 offset 校准（\|offset\|>2A 标 sensor_drift）；断档 >6h 标 time_gap。

### 5.4 step2~5 — Map 链

见 §3.2~3.5。step2 逐设备 numpy 向量化（静置累计、段末行、互斥优先级）；step5 纯函数 `aggregate()` 供测试复用。

### 5.5 step6~8 — Reduce 链

见 §3.6~3.7。step7 输出四件套：分析集（Tier-1，`soh_analysis.parquet`）、绘图集（Tier≤2 锚点+soh_point，`soh_plot.parquet` 流式 sink）、告警集（月跌幅+数据不足，`soh_alert.parquet`）、渠道看板（`dashboard.csv/html`、`group_consistency.csv`）。

### 5.6 pipeline.py — 编排

`steps_1_5()`：遍历分片 → 查 progress 跳过已完成设备 → 64 台/批跑 Step1→5 → 锚点/月聚合落盘 → 逐设备 mark done → glob 汇总月聚合。
`run(from_step, to_step, from_month)`：Step 0 后重载配置（拿到实测体系）；支持任意步骤区间与 `--from-month` 增量；Step 8 后若配置了 model_map 自动调 report_model。

### 5.7 report_model.py — 分型号报告

- `model_trend`：型号×月 的 SOH 中位/P10/P90/设备数/容量（中位 SOH × 37Ah）。
- `model_decay`：设备级衰退速率 %/年 的中位/P25/P75；**soh_start/soh_latest = 每设备首月/末月 SOH 的型号中位数**（先设备级、再型号级，避免个别离群设备污染）。
- `plot_trend`：matplotlib 双型号中位线 + P10~P90 区间 + 80% 退役线 → `model_soh_trend.png`。

### 5.8 支撑模块

- `config.py`：三段合并 + `_validate` fail-fast（体系自洽、锚点阈值序、tier 序、评分项齐全）；`model_map` 属性读 id→型号。
- `io.py`：`scan_raw`（schema 校验+dtype 强制+设备过滤）、`iter_device_batches`（批内去重排序）、`write_partitioned`（hive 风格分区）、`ProgressDB`（SQLite）。
- `schema.py`：`validate_ocv_table`（soc 严格单调递增、ocv_v 单调不减、覆盖 [0,1]）——**OCV 表非单调曾导致校准错误，为硬性约束**。
- `validation.py`：行级检查（空值率、SOC/容量/电压越界、时间单调性）。

---

## 6. 运行手册

```bash
python run.py --prep                    # 源数据变更/设备清单变更后重建分片
python run.py --step 0                  # 仅体检
python run.py --step 0-8                # 完整运行（分片缺失时自动 prep）
python run.py --step 5-8 --from-month 2026-10   # 月度增量
python run.py --step 0-8 --dry-run      # 干跑不写盘
python -m pytest tests -q               # 39 个测试
```

- 中断重跑：progress.db 自动跳过已完成设备（标记在落盘后写入，幂等）。
- 日志：`logs/pipeline.log`（带时间戳）。
- 全量参考耗时：Step 0 ≈ 80s；Step 1~8 ≈ 1,840s（192 批）。
- 本批锚点分布（988.4 万锚点行，锚点率 ≈2.3%）：OCV 5,567,324 / SWAP_FULL 3,311,399 / EMPTY_END 1,768 / FULL_END 274。

## 7. 测试与验证

| 层 | 覆盖 |
|---|---|
| 单元 | 锚点 A/B/C/D 触发与互斥优先级、soc_anchor 取值、温度补偿、MAD/PAVA、加权中位数、Tier 分级、退役外推 CI |
| 分支 | Step 0 路线分支（CONSTANT 降级）、电流取反、断点续传 |
| 集成 | `test_full_pipeline`：3 设备合成数据端到端（parts 读写、型号报告隔离输出） |
| 真实冒烟 | 4.37 亿行全量跑通；39/39 通过 |

注意：测试 `cfg` 夹具固定注入 21S generated 配置（与合成数据一致），不受真实数据 Step 0 覆写 `config.generated.yaml` 影响。

## 8. 已知限制与注意事项

1. **FULL_END 锚点样本极少**（本批仅 274 个）：LFP 24S 下"SOC>0.95 且 V>87.6×0.99 且涓流"判据偏严；自检① 满/放交叉验证以 SWAP_FULL vs EMPTY_END（1,768 个）为主。
2. **OCV 锚点占比最高**（557 万，约 56%）：soc_anchor 精度依赖 OCV 表质量，部署前建议人工抽查 dashboard 上的锚点散点。
3. **海池7237 数据窗口短**（2025-07 起 ≈14 个月）：其衰退速率与退役预测置信度低于四美（2024-04 起 ≈30 个月）。
4. **SOH 起点 ≈1.02**：首年容量中位数作分母时，BMS 剩余容量在满充后钳位略高，属已知系统性偏差（±2%），型号级对比不受影响。
5. **Step 0 体系自动判别在 LFP/NMC 边界可能误判**：本批曾被判为 NMC 20S，已按规格书（LFP 24S）人工覆写 `config/config.generated.yaml`。换数据集后须核对判别结果与 `device_model/电池参数.csv` 是否一致。
6. 退役预测为线性外推，对拐点（如电解液干涸加速期）不敏感，R²<0.6 的名单仅供参考。
