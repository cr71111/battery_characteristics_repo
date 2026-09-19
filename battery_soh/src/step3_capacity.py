"""Step 3 容量读取 + 温度补偿（方案 5-Step3，v2.0 拆分）。

3a 容量读取：C_raw = 剩余容量 / SOC_anchor（锚点处 SOC_anchor 已在 Step 2 取业务事实/OCV 值）。
3b 温度补偿：C_corrected = C_raw × (1 + α × (T_ref − T))。
    α：LFP 0.005/°C，NMC 0.0025/°C；T_ref=25°C。不补偿则季节性温差造成 ~15% 伪波动。
3c 合理性过滤：超界不删，打标记（out_of_range / temp_out / v_inconsistent）。
    C_nom_BMS 来自 Step 0 实测众数（设备级），禁止用规格书（P04）。
"""
from __future__ import annotations

import polars as pl

from .config import Config


def load_c_nom(cfg: Config) -> pl.DataFrame:
    """读取 Step 0 设备清单（渠道号/电池id/c_nom_bms）。"""
    import os
    from .config import OUTPUT_DIR

    path = os.path.join(OUTPUT_DIR, "profile", "device_list_valid.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("先运行 Step 0 生成 device_list_valid.csv")
    return pl.read_csv(path).select(["渠道号", "电池id", "c_nom_bms"])


def run(anchors: pl.DataFrame, c_nom: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    p = cfg.chem_params()
    alpha, t_ref = p["alpha"], p["t_ref"]
    lo, hi = cfg.get("filter.cap_range_ratio")
    tlo, thi = cfg.get("filter.temp_range")

    df = anchors.join(c_nom, on=["渠道号", "电池id"], how="left")
    # 设备清单缺失时回退：全量 R 中位数
    fallback = df.filter(pl.col("soc_anchor") > 0).with_columns(
        (pl.col("剩余容量") / pl.col("soc_anchor")).median()
    )["c_nom_bms"].drop_nulls().median()
    df = df.with_columns(pl.col("c_nom_bms").fill_null(float(fallback) if fallback else 40.0))

    anchor = df.filter(pl.col("anchor_type") != "NONE")
    anchor = anchor.with_columns(
        (pl.col("剩余容量") / pl.col("soc_anchor")).alias("capacity_raw"),
    ).with_columns(
        (pl.col("capacity_raw") * (1 + alpha * (t_ref - pl.col("温度")))).alias("capacity_corrected"),
    ).with_columns(
        pl.when((pl.col("capacity_corrected") < pl.col("c_nom_bms") * lo)
                | (pl.col("capacity_corrected") > pl.col("c_nom_bms") * hi))
        .then(pl.lit(True)).otherwise(pl.lit(False)).alias("out_of_range"),
        pl.when((pl.col("温度") < tlo) | (pl.col("温度") > thi))
        .then(pl.lit(True)).otherwise(pl.lit(False)).alias("temp_out"),
    )
    # v_inconsistent：放空端电压仍在平台区 / 满充端电压偏低（用单体电压粗校验）
    series = cfg.series
    anchor = anchor.with_columns(
        pl.when(
            ((pl.col("anchor_type") == "EMPTY_END")
             & (pl.col("电压") / series > p["cell_v_cutoff"] * 1.25))
            | (
                (pl.col("anchor_type") == "FULL_END")
                & (pl.col("电压") / series < p["cell_v_full"] * 0.9)
            )
        )
        .then(pl.lit(True)).otherwise(pl.lit(False)).alias("v_inconsistent"),
    )
    return anchor
