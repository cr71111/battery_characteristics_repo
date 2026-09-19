"""行级检查 + 汇总统计（贯穿各 Step 的数据质量守门员）。"""
from __future__ import annotations

import polars as pl


def row_checks(df: pl.DataFrame, name: str = "") -> dict:
    """行级检查：空值率、SOC/容量/电压越界、时间单调性。返回统计 dict。"""
    stats: dict = {"stage": name, "rows": df.height}
    for col in df.columns:
        null_frac = df.select(pl.col(col).null_count() / max(df.height, 1))[0, 0]
        if null_frac and null_frac > 0:
            stats[f"null_frac.{col}"] = round(float(null_frac), 5)
    if "SOC" in df.columns:
        stats["soc_out_of_01"] = int(
            df.filter((pl.col("SOC") < 0) | (pl.col("SOC") > 1)).height
        )
    if "剩余容量" in df.columns:
        stats["cap_nonpositive"] = int(df.filter(pl.col("剩余容量") <= 0).height)
    if "电压" in df.columns:
        stats["volt_out_of_20_120"] = int(
            df.filter((pl.col("电压") < 20) | (pl.col("电压") > 120)).height
        )
    if {"电池id", "更新时间"} <= set(df.columns):
        bad = (
            df.sort(["电池id", "更新时间"])
            .with_columns(pl.col("更新时间").diff().over("电池id"))
            .filter(pl.col("更新时间") < pl.duration())
            .height
        )
        stats["time_nonmonotonic"] = int(bad)
    return stats


def summarize(stats_list: list[dict]) -> pl.DataFrame:
    keys: list[str] = []
    for s in stats_list:
        for k in s:
            if k not in keys:
                keys.append(k)
    if not stats_list:
        return pl.DataFrame(schema=keys)
    return pl.DataFrame([[s.get(k) for k in keys] for s in stats_list], schema=keys)
