# 多型号扩展：6 电池型号 + 渠道过滤 + 分目录输出

## Context
用户重新上传了基础参数（`battery_soh/data` 6 个半年度 parquet，~12.5GB），`device_model/device_model.csv` 现有 30,809 台设备、6 个型号：四美7237、海池7237、半固态6054、海池4840、海池4846、鼎山7240。`电池参数.csv` 给出每型号的电芯类型/标称容量/串数/电压：

| 型号 | 体系 | 串数 | 标称 | V_full | 台数 |
|---|---|---|---|---|---|
| 四美7237 / 海池7237 | LFP | 24S | 37Ah | 87.6 | 3816/6999 |
| 鼎山7240 | LFP | 24S | 40Ah | 87.6 | 5147 |
| 海池4840 | LFP | 16S | 40Ah | 58.4 | 5024 |
| 海池4846 | LFP | 16S | 46Ah | 58.4 | 7987 |
| 半固态6054 | NMC | 17S | 54Ah | 71.4 | 1836 |

现有管线假定**全局单一体系/串数/标称**（Step 0 判别、step2/3/4 读 cfg、step6 分母固定 37Ah）。需求：
1. 全部 6 型号进入模型计算（各型号用自己的体系参数与标称容量）
2. 渠道号只保留 GYDS / JXHC / SMKJ
3. 输出按 `output/reports/总体/` + `output/reports/<型号>/` 分文件夹

## 改动方案

### 1. 配置层（src/config.py + config.yaml + 新 OCV 表）
- 新增 `Config.model_params`：启动时读 `device_model/电池参数.csv`，解析 电池类型(铁锂→LFP/三元→NMC)、标称容量("37Ah"→37.0)、电芯数量→series、合计满充电压→v_full、cell_v_cutoff 按体系取 chemistry 段值 → 得 `{电池型号: {chemistry, series, c_nom_spec, v_full, v_cutoff, alpha, t_ref}}`；同时导出 `Config.id2model`（电池id→型号）与 `Config.id2params`
- 项目记忆规则：电池参数.csv 优先于 Step 0 平台斜率判别（Step 0 判别仅作缺参兜底/校验）
- 复制 OCV 表：`lfp_16s.csv`（内容同 lfp_24s，单体电压表与串数无关）、`nmc_17s.csv`（同 nmc_21s）
- config.yaml：删除全局 `c_nom_spec`（改为逐型号），新增 `channels: [GYDS, JXHC, SMKJ]`

### 2. prep（src/prep.py）
- 过滤条件加 `pl.col("渠道号").is_in(cfg.get("channels"))`；设备清单为 30,809 台
- 需 `--prep --force` 重建 data/filtered，并删除 `output/progress.db`、`output/observation/parts`、`output/monthly`（全量重算，锚点参数体系变了）

### 3. Step 0（src/step0_profile.py）
- `_chemistry` 改为逐型号校验：对每个型号抽样设备判别平台斜率/电压范围，与 电池参数.csv 比对打印告警（不一致以 CSV 为准）；不再产出全局唯一 chemistry/series
- `config.generated.yaml` 保留全局 `R_MODE`/`route`/`reversed_devices`/`current_effectively_missing`（这些与体系无关），删除全局 chemistry/series/v_full/v_cutoff
- `_c_nom`（设备级 R 众数）不变，仍供 step3 合理性过滤用

### 4. Map 阶段（src/pipeline.py steps_1_5）
- 每批数据先 join `id2model` 得 电池型号 列，**按型号分组**跑 step2/3/4（step1 体系无关不变）；step5 不变
- step2_anchor.run / step3_capacity.run / step4_quality.run 增加 `params: dict` 入参（series、v_full、v_cutoff、alpha、c_nom_default、ocv 表按 `{chemistry}_{series}s` 加载），缺省 None 时回退全局 cfg（保持测试兼容）
- 锚点 parts 与月聚合 parts 均带 电池型号 列

### 5. Step 6（src/step6_soh.py）
- SOH 分母改为**逐设备型号标称容量**：join id2model→model_params 得 c_nom_spec 列，`soh = capacity_smoothed / c_nom_spec`；前导异常月剔除（90% 阈值）逻辑不变（已是相对各自标称）
- soh_tbl 携带 c_nom_spec 列供下游

### 6. Step 7/8（step7_trend.py、step8_retirement.py）
- 逻辑不变（SOH 已是各型号绝对口径）；输出 analysis 带 电池型号 列，型号级统计自然扩展到 6 型号

### 7. 报告层重构（report_model.py、report_forecast.py）
- 输出目录：`output/reports/总体/`（跨型号对比图 + report_总体.pdf + 全局 CSV + excluded_devices.csv）与 `output/reports/<型号>/`（该型号 3 张趋势 PNG + 退役预测 PNG + report_<型号>.pdf + 型号级 CSV）
- 颜色：`_COLORS` 扩为 6 型号调色板（tab10 前 6 色）
- 合图（总体）：6 条线趋势/月龄/循环图；**横轴下多行分布标注仅在 ≤2 型号时显示**（6 型号 12 行不可读），单型号图保留 2 行分布
- caption 文案参数化：`caption_single_*(model, c_nom_spec)`，"标称 37Ah" 改为逐型号；总体 caption 说明各型号分母为各自标称容量
- report_forecast：退役预测合图 6 型号；低可信设备外推用**全体设备中位衰退率**（原"海池参照四美"推广）；每型号 PDF 章节沿用现有结构
- 预测窗口起点按最新数据月自动推导（不再硬编码 2026-09）

### 8. 测试（tests/）
- conftest 夹具补最小 model_params（NMC/21S/40Ah 单型号），cfg_env 注入；`c_nom_spec=40` 夹具逻辑改走 params 通道
- 全量跑 pytest 保证 39 项通过

## 执行顺序
1. 配置层 + OCV 表 + prep 渠道过滤
2. step0/2/3/4/pipeline 参数化改造
3. step6 分母逐型号
4. 报告层目录重构
5. 测试修复
6. `--prep --force` 重建分片 → 全量 `--step 0-8`（数据量大，后台跑）
7. 验证：6 型号各自 CSV/PNG/PDF 齐全、渠道仅 3 个、SOH 口径抽查

## 验证
- `python -m pytest tests -q` 全绿
- 重跑后检查 `output/reports/总体` + 6 个型号目录文件齐全
- 抽查每型号 device_count、soh 首月中位（应接近各自标称）、excluded_devices.csv 台数
- 确认 analysis 中渠道号 distinct == {GYDS, JXHC, SMKJ}
