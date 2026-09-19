"""Step 1 数据清洗（方案 5-Step1，含 v2.0 电流零点校准）。

1a 基础清洗：更新时间为空删除；循环次数/传感器非数值打标记（不删行）。
1b 电流零点校准（每设备独立）：静置段均值 → offset；|offset|>2A 标记漂移。
1c 电流符号一致性：Step 0 抽检给出的 reversed_devices 整体取反。
1d 采样断档检测：相邻样本时间差 >6h 标记 time_gap（用于切段）。

电流铁律：本管线电流只取符号与静置判据，永不参与累加。
"""
from __future__ import annotations

import polars as pl

from .config import Config


def run(df: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    thr = cfg
    deadband = thr.get("anchor.deadband_a")
    drift_a = thr.get("anchor.sensor_drift_a")
    gap_h = thr.get("anchor.gap_hours")

    df = df.drop_nulls(["更新时间"])

    # 1a 非数值标记（cast strict=False 产生 null 即非数值）
    df = df.with_columns(
        pl.col("循环次数").cast(pl.Float64, strict=False).alias("循环次数_n"),
        pl.col("电压").cast(pl.Float64, strict=False).alias("电压_n"),
        pl.col("温度").cast(pl.Float64, strict=False).alias("温度_n"),
        pl.col("电流").cast(pl.Float64, strict=False).alias("电流_n"),
    ).with_columns(
        pl.col("循环次数_n").is_null().alias("invalid_loopnum"),
        (
            pl.col("电压_n").is_null() | pl.col("温度_n").is_null() | pl.col("电流_n").is_null()
        ).alias("invalid_sensor"),
    )

    # 1c 电流符号一致性（Step 0 抽检结论：整体取反）
    reversed_devices = set(thr.get("generated.reversed_devices", []) or [])
    if reversed_devices:
        df = df.with_columns(
            pl.when(pl.col("电池id").is_in(list(reversed_devices)))
            .then(-pl.col("电流_n"))
            .otherwise(pl.col("电流_n"))
            .alias("电流_c"),
            pl.col("电池id").is_in(list(reversed_devices)).alias("current_reversed"),
        )
    else:
        df = df.with_columns(pl.col("电流_n").alias("电流_c"), pl.lit(False).alias("current_reversed"))

    # 1b 电流零点校准：每设备取"低电流静置段"均值作 offset
    offsets = (
        df.filter(pl.col("电流_c").abs() <= max(deadband * 5, 1.0))  # 候选静置样本
        .group_by("电池id")
        .agg(pl.col("电流_c").mean().alias("offset"))
        .with_columns(
            (pl.col("offset").abs() > drift_a).alias("current_sensor_drift")
        )
    )
    df = df.join(offsets.select(["电池id", "offset", "current_sensor_drift"]), on="电池id", how="left")
    df = df.with_columns(
        (pl.col("电流_c") - pl.col("offset").fill_null(0.0)).alias("电流_0"),
        pl.col("current_sensor_drift").fill_null(False),
    )

    # 1d 断档检测
    df = df.sort(["电池id", "更新时间"]).with_columns(
        pl.col("更新时间").diff().over("电池id").dt.total_seconds().alias("gap_s")
    ).with_columns(
        (pl.col("gap_s") > gap_h * 3600).fill_null(True).alias("time_gap"),
    )

    # 统一列名（覆盖原列）
    df = df.with_columns(
        pl.col("循环次数_n").fill_null(0.0).alias("循环次数"),
        pl.col("电压_n").alias("电压"),
        pl.col("温度_n").alias("温度"),
        pl.col("电流_0").alias("电流"),
    ).drop(["电流_n", "电流_c", "offset", "电压_n", "温度_n", "循环次数_n", "gap_s"])
    return df
