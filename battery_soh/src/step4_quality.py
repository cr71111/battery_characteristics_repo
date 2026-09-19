"""Step 4 异常检测 + 质量评分 + 分级（方案 5-Step4，v2.0 合并修正顺序）。

4a 异常标记（先做）：容量越界 / 电压不自洽 / 温度异常 / 锚点冲突 / 传感器漂移 / 前后矛盾。
4b 质量评分（后做，依赖 4a）：锚点类型基础分 + 各加扣分项（全部来自 thresholds.yaml）。
4c 分级：Tier-1 评分>=30 且无严重异常；Tier-2 评分 10~29；Tier-3 其余。
    Tier-1 还要求 7 天内有另一锚点支撑（方案第四章闭环判据），否则降为 Tier-2。
"""
from __future__ import annotations

import polars as pl

from .config import Config


def run(anchor_pts: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    q = cfg.get("quality")
    score_cfg = q["score"]
    lo, hi = cfg.get("filter.cap_anomaly_ratio")
    nlo, nhi = cfg.get("filter.cap_normal_ratio")
    tlo, thi = cfg.get("filter.temp_range")
    tnlo, tnh = cfg.get("filter.temp_normal_range")
    ratio = q["contradiction_ratio"]

    df = anchor_pts.sort(["电池id", "更新时间"])

    # ---- 4a 异常标记 ----
    df = df.with_columns(
        (
            (pl.col("capacity_corrected") < pl.col("c_nom_bms") * lo)
            | (pl.col("capacity_corrected") > pl.col("c_nom_bms") * hi)
        ).fill_null(False).alias("f_out_of_range"),
        pl.col("v_inconsistent").fill_null(False).alias("f_voltage_inconsistent"),
        pl.col("temp_out").fill_null(False).alias("f_temp_out"),
        pl.col("flags").list.contains("anchor_conflict").fill_null(False).alias("f_anchor_conflict"),
        pl.col("flags").list.contains("sensor_drift").fill_null(False).alias("f_sensor_drift"),
    )
    # 前后矛盾：同设备同锚点类型，与相邻锚点容量差异 > ratio
    df = df.with_columns(
        pl.col("capacity_corrected").shift(1).over(["电池id", "anchor_type"]).alias("_p"),
        pl.col("capacity_corrected").shift(-1).over(["电池id", "anchor_type"]).alias("_n"),
    ).with_columns(
        (
            (((pl.col("capacity_corrected") - pl.col("_p")).abs()
              > pl.col("_p").abs() * ratio) & pl.col("_p").is_not_null())
            | (((pl.col("capacity_corrected") - pl.col("_n")).abs()
                > pl.col("_n").abs() * ratio) & pl.col("_n").is_not_null())
        ).fill_null(False).alias("f_contradiction")
    ).drop("_p", "_n")

    # ---- 4b 质量评分（依赖 4a 标记） ----
    sev_any = (
        pl.col("f_out_of_range") | pl.col("f_voltage_inconsistent")
        | pl.col("f_anchor_conflict") | pl.col("f_contradiction")
    ).alias("severe")
    df = df.with_columns(sev_any)

    df = df.with_columns(
        (
            pl.col("anchor_type").replace_strict(
                {k: float(score_cfg[k]) for k in ("SWAP_FULL", "EMPTY_END", "FULL_END", "OCV")},
                default=0.0,
            )
            + pl.when((pl.col("温度") >= tnlo) & (pl.col("温度") <= tnh))
            .then(float(score_cfg["temp_normal"]))
            .when((pl.col("温度") < tlo) | (pl.col("温度") > thi))
            .then(float(score_cfg["temp_out"]))
            .otherwise(0.0)
            + pl.when(
                (pl.col("capacity_corrected") >= pl.col("c_nom_bms") * nlo)
                & (pl.col("capacity_corrected") <= pl.col("c_nom_bms") * nhi)
            )
            .then(float(score_cfg["cap_normal"]))
            .when(pl.col("f_out_of_range"))
            .then(float(score_cfg["cap_anomaly"]))
            .otherwise(0.0)
            + pl.when(pl.col("f_voltage_inconsistent"))
            .then(float(score_cfg["v_anomaly"]))
            .otherwise(float(score_cfg["v_normal"]))
            + pl.when(pl.col("f_contradiction")).then(float(score_cfg["contradiction"])).otherwise(0.0)
            + pl.when(pl.col("f_sensor_drift")).then(float(score_cfg["sensor_drift"])).otherwise(0.0)
        ).cast(pl.Float64).alias("quality_score")
    )

    # ---- 4c 分级 ----
    df = df.with_columns(
        ((pl.col("quality_score") >= q["tier1_min"]) & ~pl.col("severe")).alias("_t1_raw"),
        (pl.col("quality_score") >= q["tier2_min"]).alias("_t2_raw"),
    )
    # Tier-1 闭环：7 天内同设备存在另一锚点支撑
    df = df.with_columns(pl.col("更新时间").dt.epoch("d").alias("_day"))
    df = df.with_columns(
        pl.col("_day").diff().over("电池id").abs().alias("_d1"),
        pl.col("_day").diff(-1).over("电池id").abs().alias("_d2"),
    ).with_columns(
        ((pl.col("_d1").fill_null(9999) <= 7) | (pl.col("_d2").fill_null(9999) <= 7)).alias("_closed"),
    ).with_columns(
        pl.when(pl.col("_t1_raw") & pl.col("_closed")).then(1)
        .when(pl.col("_t2_raw") | (pl.col("_t1_raw") & ~pl.col("_closed"))).then(2)
        .otherwise(3).cast(pl.Int32).alias("cap_tier"),
    ).drop("_t1_raw", "_t2_raw", "_d1", "_d2", "_closed", "_day")

    # 加权中位数权重：Tier-1 用锚点类型权重，Tier-2 乘 0.3，Tier-3 为 0（禁入统计）
    wmap = {k: float(q["weight"][k]) for k in ("SWAP_FULL", "EMPTY_END", "FULL_END", "OCV")}
    df = df.with_columns(
        pl.col("anchor_type").replace_strict(wmap, default=0.0).alias("_w_anchor")
    ).with_columns(
        (
            pl.when(pl.col("cap_tier") == 1).then(pl.col("_w_anchor"))
            .when(pl.col("cap_tier") == 2).then(pl.col("_w_anchor") * float(q["tier2_weight"]))
            .otherwise(0.0)
        ).alias("stat_weight")
    ).drop("_w_anchor")
    return df
