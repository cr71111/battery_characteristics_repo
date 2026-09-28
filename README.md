# 换电电池 SOH 估算工程管线

依据《换电电池SOH估算工程方案 v2 修正版》实现。核心思想：**只用被电压/业务事实锚定的离散锚点**估计绝对容量，
电流永不参与累加（避免库仑积分漂移），全部阈值集中在 `config/thresholds.yaml`（禁止硬编码）。

## 数据前提

原始 parquet 需含 10 列：`表名 / 电池id / 渠道号 / 循环次数 / 剩余容量 / SOC / 电压 / 电流 / 温度 / 更新时间`。
路径在 `config/config.yaml` 的 `data_path` 指定（支持目录或单文件）。

## 管线（Step 0~8）

| 步骤 | 模块 | 作用 |
|---|---|---|
| 0 | `step0_profile.py` | 一次性体检：R 有效性、体系判别（LFP/NMC+串数）、采样、循环单调性、电流符号/零点、C_nom 众数、锚点密度 → 决定 `R_MODE` 与路线（ABSOLUTE / L2 降级）；输出 `diagnosis_report.json`、`config.generated.yaml`、`device_list_valid.csv` |
| 1 | `step1_clean.py` | 单位统一、电流零点校准（每设备 offset）、电流符号修正、断档标记 |
| 2 | `step2_anchor.py` | 锚点打标 A/B/C/D（互斥优先级 B>C>A>D）+ soc_anchor 真值 |
| 3 | `step3_capacity.py` | C_raw = 剩余容量/SOC_anchor；温度补偿；合理性标记（不删行） |
| 4 | `step4_quality.py` | 异常检测 → 质量评分 → Tier-1/2/3 分级（Tier-1 需 7 天内闭环支撑） |
| 5 | `step5_monthly.py` | 加权中位数月聚合（Tier-3 禁入），按渠道分区落盘，支持 `--from-month` 增量 |
| 6 | `step6_soh.py` | 滚动 MAD 去毛刺 + PAVA 非增回归 + SOH（C_nom 取首年实测中位数） |
| 7 | `step7_trend.py` | 衰退速率（%/月、%/年）、分析集/绘图集/告警集、渠道看板、自检③群体一致性 |
| 8 | `step8_retirement.py` | 线性外推退役月 + R² + 95% 置信区间 + 健康分级 |

Map 阶段（Step 1~5）按设备分批流式处理，进度记入 `output/progress.db`（SQLite），断点续传不重复计算；
Reduce 阶段（Step 6~8）在月聚合层（万行级）运行。

## 运行

```bash
pip install -e ".[dev]"

python run.py --step 0            # 仅体检（先跑这个，生成 config.generated.yaml）
python run.py --step 0-8          # 完整运行
python run.py --step 5-8 --from-month 2024-09   # 增量更新（月度重聚合及以后）
python run.py --step 0-8 --dry-run              # 干跑，不写盘
```

## 配置

- `config/config.yaml` — 人工基线（数据路径、可覆盖体系参数），优先级最低
- `config/thresholds.yaml` — 全部阈值（铁律：代码禁止硬编码）
- `config/config.generated.yaml` — Step 0 实测参数（gitignore，机器生成）
- `config/ocv_tables/` — 单体 OCV-SOC 表（lfp_24s/25s、nmc_20s/21s）

三段按 config.yaml < thresholds.yaml < config.generated.yaml 合并，启动时 fail-fast 校验。

## 输出

- `output/profile/` — 体检报告与设备清单
- `output/observation/anchor_points.parquet` — 逐行锚点打标 + 容量 + 分级
- `output/monthly/monthly_capacity.parquet` — 设备×月容量点
- `output/soh/` — `soh_analysis`（分析集）/ `soh_plot`（绘图集）/ `soh_alert`（告警集）
- `output/retirement/retirement_forecast.parquet` — 退役预测
- `output/reports/` — `dashboard.csv/html`、`group_consistency.csv`

## 测试

```bash
python -m pytest tests/
```

合成数据夹具在 `tests/conftest.py`（NMC 21S 体系，可触发 A/B/C/D 四类锚点与 L2 降级分支）。
