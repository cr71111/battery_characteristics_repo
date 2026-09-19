"""Step 5 月度聚合（Map 阶段，方案 5-Step5）。

分组键：[渠道号, 电池id, 年, 月]；聚合：加权中位数（权重=锚点质量分/Tier 权重）。
输出：月度容量点 + 点数 + tier 分布 + 温度均值。
几十亿条 → 设备×月量级；支持增量（--from-month 只重算指定月分区）。
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl

from .config import OUTPUT_DIR, Config
from .io import write_partitioned


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float(np.median(v))
    return float(v[np.searchsorted(cw, 0.5 * cw[-1])])


def aggregate(df: pl.DataFrame) -> pl.DataFrame:
    """对含 capacity_corrected/stat_weight 的锚点集做月聚合（纯函数，供测试）。"""
    df = df.filter((pl.col("cap_tier") <= 2) & pl.col("capacity_corrected").is_not_null())
    df = df.with_columns(
        pl.col("更新时间").dt.year().alias("year"),
        pl.col("更新时间").dt.month().alias("month"),
    )
    out = []
    for (ch, dev, y, m), sub in df.group_by(["渠道号", "电池id", "year", "month"], maintain_order=True):
        v = sub["capacity_corrected"].to_numpy()
        w = sub["stat_weight"].to_numpy()
        # soh_bms：由剩余容量直接推导的 SOH 代理（median(剩余容量)/c_nom）
        soh_bms = float(
            (sub["剩余容量"] / sub["c_nom_bms"]).median()
        ) if "c_nom_bms" in sub.columns and sub.height else None
        out.append(
            (
                ch, int(dev), int(y), int(m), f"{int(y)}-{int(m):02d}",
                _weighted_median(v, w),
                sub.height,
                int((sub["cap_tier"] == 1).sum()),
                int((sub["cap_tier"] == 2).sum()),
                float(sub["温度"].mean()),
                float(sub["quality_score"].mean()),
                soh_bms,
            )
        )
    return pl.DataFrame(
        out,
        schema={
            "渠道号": pl.Utf8, "电池id": pl.Int64, "year": pl.Int64, "month": pl.Int64,
            "ym": pl.Utf8, "capacity_median": pl.Float64, "capacity_count": pl.Int64,
            "tier1_count": pl.Int64, "tier2_count": pl.Int64, "temp_mean": pl.Float64,
            "score_mean": pl.Float64, "soh_bms": pl.Float64,
        },
        orient="row",
    ).sort(["渠道号", "电池id", "ym"])


def run(anchors: pl.DataFrame, cfg: Config, from_month: str | None = None) -> pl.DataFrame:
    monthly = aggregate(anchors)
    if from_month:
        monthly = monthly.filter(pl.col("ym") >= from_month)
    out_dir = os.path.join(OUTPUT_DIR, "monthly")
    write_partitioned(monthly, out_dir, by=["渠道号"], fmt="parquet")
    monthly.write_parquet(os.path.join(out_dir, "monthly_capacity.parquet"))
    print(f"[step5] 月度聚合 {monthly.height} 行（设备×月）")
    return monthly
