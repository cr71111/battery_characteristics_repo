"""管线编排（方案第十章 10.3）：--step 0-8 / --from-month / --dry-run。

Map 阶段（Step 1~4）按设备分批流式处理，进度记入 progress.db，断点续传不重复计算；
Reduce 阶段（Step 6~8）在月聚合层（万行级）运行。
"""
from __future__ import annotations

import os
import time

import polars as pl

from . import (step0_profile, step1_clean, step2_anchor, step3_capacity,
               step4_quality, step5_monthly, step6_soh, step7_trend, step8_retirement)
from .config import LOGS_DIR, OUTPUT_DIR, Config, load_config
from .io import ProgressDB, iter_device_batches, scan_raw
from .validation import row_checks, summarize

OBS_DIR = os.path.join(OUTPUT_DIR, "observation")


def _log(msg: str) -> None:
    os.makedirs(LOGS_DIR, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
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

    # ------------------------------------------------ Step 1~4（Map，按设备分批）
    def steps_1_4(self) -> pl.DataFrame:
        """流式扫描 → 清洗 → 锚点 → 容量 → 质量分级，落盘 anchor_points.parquet。"""
        # R_MODE/CONSTANT 分支已在 Step 0 决定；L2 路线不进入本管线（P06）
        if self.cfg.route != "ABSOLUTE":
            _log(f"[pipeline] 路线={self.cfg.route}（降级），跳过 Step 1~8。"
                 f"输出：数据不足，建议补充累计充/放电量寄存器字段后重跑。")
            return None
        lf = scan_raw(self.cfg.data_path)
        c_nom = step3_capacity.load_c_nom(self.cfg)
        all_done = {int(d) for (_, d, _) in self._all_done_rows()}
        chunks, stats = [], []
        for batch in iter_device_batches(lf, batch_devices=20):
            pending = batch.filter(~pl.col("电池id").is_in(list(all_done)))
            if pending.height == 0:
                continue
            stats.append(row_checks(pending, "step1_in"))
            cleaned = step1_clean.run(pending, self.cfg)
            anchored = step2_anchor.run(cleaned, self.cfg)
            cap = step3_capacity.run(anchored, c_nom, self.cfg)
            scored = step4_quality.run(cap, self.cfg)
            stats.append(row_checks(scored, "step4_out"))
            chunks.append(scored)
            _log(f"[pipeline] Step1~4 批次完成：设备 {pending['电池id'].n_unique()} 台，"
                 f"输入 {pending.height} 行 → 锚点 {scored.height} 行")
        if not chunks:
            _log("[pipeline] Step 1~4 无新增设备（全部已完成）")
            return pl.read_parquet(os.path.join(OBS_DIR, "anchor_points.parquet"))
        res = pl.concat(chunks, how="vertical_relaxed")
        summary = summarize(stats)
        _log("[pipeline] Step1~4 行数变化:\n" + summary.__str__())
        if self.dry_run:
            return res
        os.makedirs(OBS_DIR, exist_ok=True)
        prev_p = os.path.join(OBS_DIR, "anchor_points.parquet")
        if os.path.exists(prev_p):
            res = pl.concat([pl.read_parquet(prev_p), res], how="vertical_relaxed")
        res.write_parquet(prev_p)
        # 落盘成功后才标记完成（避免中断导致进度与结果不一致）
        for df in chunks:
            for dev in df["电池id"].unique().to_list():
                self.progress.mark("step1_4", "ALL", dev, "ALL", "done")
        _log(f"[pipeline] 锚点结果落盘 {res.height} 行 → {prev_p}")
        return res

    def _all_done_rows(self):
        cur = self.progress._conn.execute(
            "SELECT channel, device, ym FROM progress WHERE step='step1_4' AND status='done'"
        )
        return cur.fetchall()

    # ------------------------------------------------ Step 5~8（Reduce）
    def step5(self, anchors: pl.DataFrame, from_month: str | None = None):
        return step5_monthly.run(anchors, self.cfg, from_month=from_month)

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
        monthly_p = os.path.join(OUTPUT_DIR, "monthly", "monthly_capacity.parquet")
        analysis_p = os.path.join(OUTPUT_DIR, "soh", "soh_analysis.parquet")
        anchors = analysis = monthly = soh_tbl = None

        if from_step <= 0 <= to_step:
            report = self.step0()
            if report.get("route") != "ABSOLUTE":
                _log(f"[pipeline] Step 0 判定路线={report['route']}：{report['recommendation']}")
                return
            self.cfg = load_config()  # 重新加载 Step 0 实测参数
        if from_step <= 4 <= to_step:
            anchors = self.steps_1_4()
            if anchors is None:
                return
        elif os.path.exists(anchors_p):
            anchors = pl.read_parquet(anchors_p)
        if from_step <= 5 <= to_step:
            if anchors is None and os.path.exists(anchors_p):
                anchors = pl.read_parquet(anchors_p)
            monthly = self.step5(anchors, from_month)
        elif os.path.exists(monthly_p):
            monthly = pl.read_parquet(monthly_p)
        if from_step <= 6 <= to_step and monthly is not None and monthly.height:
            soh_tbl = self.step6(monthly)
        if from_step <= 7 <= to_step and soh_tbl is not None:
            if anchors is None and os.path.exists(anchors_p):
                anchors = pl.read_parquet(anchors_p)
            out = self.step7(soh_tbl, anchors)
            analysis = out["analysis"]
        elif os.path.exists(analysis_p):
            analysis = pl.read_parquet(analysis_p)
        if from_step <= 8 <= to_step and analysis is not None:
            self.step8(analysis)
        _log(f"[pipeline] 完成 Step {from_step}~{to_step}，用时 {time.time() - t0:.1f}s")
