"""管线编排（方案第十章 10.3）：--step 0-8 / --from-month / --dry-run。

Map 阶段（Step 1~5）按设备分批流式处理：锚点行只落盘（parts/），
月聚合（设备×月，万行级）在内存累积；进度按"分片"记入 progress.db，断点续传。
Reduce 阶段（Step 6~8）在月聚合层运行。
"""
from __future__ import annotations

import gc
import glob
import os
import time

import polars as pl

from . import (step0_profile, step1_clean, step2_anchor, step3_capacity,
               step4_quality, step5_monthly, step6_soh, step7_trend, step8_retirement)
from .config import LOGS_DIR, OUTPUT_DIR, Config, load_config
from .io import ProgressDB, iter_device_batches

OBS_DIR = os.path.join(OUTPUT_DIR, "observation")
MONTHLY_DIR = os.path.join(OUTPUT_DIR, "monthly")


def anchors_dir() -> str:
    return os.path.join(OBS_DIR, "parts")


def load_anchors() -> pl.LazyFrame:
    """读取锚点结果（parts 分片优先，兼容单文件），返回 LazyFrame（streaming 处理）。"""
    parts = sorted(glob.glob(os.path.join(anchors_dir(), "*.parquet")))
    if parts:
        return pl.concat([pl.scan_parquet(p) for p in parts])
    single = os.path.join(OBS_DIR, "anchor_points.parquet")
    return pl.scan_parquet(single)


def _log(msg: str) -> None:
    os.makedirs(LOGS_DIR, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(LOGS_DIR, "pipeline.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


class Pipeline:
    def __init__(self, cfg: Config | None = None, dry_run: bool = False):
        self.cfg = cfg or load_config()
        self.dry_run = dry_run
        self.progress = ProgressDB(os.path.join(OUTPUT_DIR, "progress.db"))

    # ------------------------------------------------ Step 0
    def step0(self):
        return step0_profile.run(self.cfg, dry_run=self.dry_run)

    # ------------------------------------------------ Step 1~5（Map，按分片×设备批）
    def steps_1_5(self, from_month: str | None = None) -> pl.DataFrame | None:
        """分片流式：清洗 → 锚点 → 容量 → 质量 → 月聚合。

        锚点行逐批写 output/observation/parts/anchor_XXXX.parquet（不进内存）；
        月聚合按分片写 output/monthly/parts/，返回全部月聚合的合并结果。
        """
        if self.cfg.route != "ABSOLUTE":
            _log(f"[pipeline] 路线={self.cfg.route}（降级），跳过 Step 1~8。"
                 f"输出：数据不足，建议补充累计充/放电量寄存器字段后重跑。")
            return None
        parts = sorted(glob.glob(os.path.join(self.cfg.data_path, "part_*.parquet")))
        if not parts:
            raise FileNotFoundError("先运行 prep（run.py --prep）生成分片数据")
        c_nom = step3_capacity.load_c_nom(self.cfg)
        os.makedirs(anchors_dir(), exist_ok=True)
        os.makedirs(os.path.join(MONTHLY_DIR, "parts"), exist_ok=True)

        # 多型号：电池id→型号→体系参数（无参数表时退化为单型号全局 cfg）
        model_map = self.cfg.model_map
        model_params = self.cfg.model_params
        id2model = (dict(zip(model_map["电池id"].to_list(), model_map["电池型号"].to_list()))
                    if model_map is not None else {})

        monthly_chunks = []
        batch_idx = -1
        for part_i, part in enumerate(parts):
            lf = pl.scan_parquet(part)
            devs_part = lf.select("电池id").unique().collect()["电池id"].to_list()
            pending = [d for d in devs_part
                       if not self.progress.done("step1_5", "ALL", d, "ALL")]
            if not pending:
                continue  # 该分片全部完成，月聚合走下方统一读取
            for batch in iter_device_batches(lf, batch_devices=64):
                batch = batch.filter(pl.col("电池id").is_in(pending))
                if batch.height == 0:
                    continue
                batch_idx += 1
                devs = batch["电池id"].unique().to_list()
                cleaned = step1_clean.run(batch, self.cfg)
                if id2model:
                    cleaned = cleaned.with_columns(
                        pl.col("电池id").replace_strict(id2model).alias("电池型号"))
                # 按型号分组跑体系相关步骤（step2/3/4）
                scored_chunks = []
                if model_params and "电池型号" in cleaned.columns:
                    groups = [(str(sub["电池型号"][0]), sub)
                              for sub in cleaned.partition_by("电池型号", maintain_order=True)]
                else:
                    groups = [("", cleaned)]
                for mname, sub in groups:
                    params = model_params.get(mname) if mname else None
                    anchored = step2_anchor.run(sub, self.cfg, params=params)
                    cap = step3_capacity.run(anchored, c_nom, self.cfg, params=params)
                    scored = step4_quality.run(cap, self.cfg)
                    if "电池型号" not in scored.columns:
                        scored = scored.with_columns(pl.lit(mname or "unknown").alias("电池型号"))
                    scored_chunks.append(scored)
                scored = pl.concat(scored_chunks, how="vertical_relaxed")
                monthly = step5_monthly.aggregate(
                    scored,
                    min_anchors=int(self.cfg.get("soh.min_anchors_per_month") or 3),
                    anchor_mad_k=float(self.cfg.get("soh.anchor_mad_k") or 3.0),
                )
                if from_month:
                    monthly = monthly.filter(pl.col("ym") >= from_month)
                _log(f"[pipeline] Map part{part_i} 批 {batch_idx}：设备 {len(devs)} 台，"
                     f"输入 {batch.height} 行 → 锚点 {scored.height} 行 → 月点 {monthly.height} 行")
                if self.dry_run:
                    monthly_chunks.append(monthly)
                    continue
                scored.write_parquet(os.path.join(anchors_dir(), f"anchor_b{batch_idx:04d}.parquet"))
                monthly.write_parquet(os.path.join(MONTHLY_DIR, "parts", f"monthly_b{batch_idx:04d}.parquet"))
                for d in devs:
                    self.progress.mark("step1_5", "ALL", d, "ALL", "done")
        # 汇总全部月聚合（新写的批 + 断点续传时已存在的历史批）
        all_m = sorted(glob.glob(os.path.join(MONTHLY_DIR, "parts", "monthly_b*.parquet")))
        if all_m:
            monthly_chunks = [pl.read_parquet(p) for p in all_m]
        elif os.path.exists(os.path.join(MONTHLY_DIR, "monthly_capacity.parquet")):
            # 无新批且分片已归档：直接复用全量月聚合结果
            monthly_chunks = [pl.read_parquet(
                os.path.join(MONTHLY_DIR, "monthly_capacity.parquet"))]

        if not monthly_chunks:
            _log("[pipeline] Step 1~5 无数据")
            return pl.DataFrame()
        res = pl.concat(monthly_chunks, how="vertical_relaxed").sort(["渠道号", "电池id", "ym"])
        single = os.path.join(MONTHLY_DIR, "monthly_capacity.parquet")
        if not self.dry_run:
            res.write_parquet(single)
            # 兼容旧路径：合并锚点单文件（数据量大时跳过，由 load_anchors 读 parts）
        _log(f"[pipeline] Map 完成：月聚合 {res.height} 行（设备×月）")
        return res

    # ------------------------------------------------ Step 6~8（Reduce）
    def step6(self, monthly: pl.DataFrame):
        return step6_soh.run(monthly, self.cfg)

    def step7(self, soh_tbl: pl.DataFrame, anchors: pl.DataFrame):
        return step7_trend.run(soh_tbl, anchors, self.cfg)

    def step8(self, analysis: pl.DataFrame):
        return step8_retirement.run(analysis, self.cfg)

    # ------------------------------------------------ 编排
    def run(self, from_step: int = 0, to_step: int = 8, from_month: str | None = None):
        t0 = time.time()
        anchors_p = os.path.join(OBS_DIR, "anchor_points.parquet")
        monthly_p = os.path.join(MONTHLY_DIR, "monthly_capacity.parquet")
        analysis_p = os.path.join(OUTPUT_DIR, "soh", "soh_analysis.parquet")
        anchors = analysis = monthly = soh_tbl = None

        if from_step <= 0 <= to_step:
            report = self.step0()
            if report.get("route") != "ABSOLUTE":
                _log(f"[pipeline] Step 0 判定路线={report['route']}：{report['recommendation']}")
                return
            self.cfg = load_config()  # 重新加载 Step 0 实测参数
        if from_step <= 5 <= to_step:
            monthly = self.steps_1_5(from_month)
            if monthly is None:
                return
            anchors = load_anchors()
        else:
            if os.path.exists(monthly_p):
                monthly = pl.read_parquet(monthly_p)
            if os.path.exists(anchors_p) or glob.glob(os.path.join(anchors_dir(), "*.parquet")):
                anchors = load_anchors()
        if from_step <= 6 <= to_step and monthly is not None and monthly.height:
            soh_tbl = self.step6(monthly)
        if from_step <= 7 <= to_step and soh_tbl is not None:
            if anchors is None:
                anchors = load_anchors()
            out = self.step7(soh_tbl, anchors)
            analysis = out["analysis"]
        elif os.path.exists(analysis_p):
            analysis = pl.read_parquet(analysis_p)
        if from_step <= 8 <= to_step and analysis is not None:
            self.step8(analysis)
            if self.cfg.model_map is not None:
                from . import report_model
                report_model.run(self.cfg)
                try:
                    from . import report_interactive
                    report_interactive.run(self.cfg)
                except Exception as e:
                    _log(f"[pipeline] 交互式 HTML 图表跳过: {e}")
                try:
                    from . import report_forecast
                    report_forecast.run(self.cfg)
                except Exception as e:
                    _log(f"[pipeline] 退役预测报告跳过: {e}")
        _log(f"[pipeline] 完成 Step {from_step}~{to_step}，用时 {time.time() - t0:.1f}s")
