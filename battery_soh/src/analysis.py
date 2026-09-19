"""三道自检 + 渠道看板指标（方案第六章 / 第八章）——纯函数，输入输出均为 DataFrame。"""
from __future__ import annotations

import numpy as np
import polars as pl


# ---------------------------------------------------------------- 自检①
def selfcheck_full_vs_empty(anchors: pl.DataFrame) -> pl.DataFrame:
    """满充(A/B) vs 放空(C) 交叉验证：同一电池两类锚点算出的 C 中位数对比。

    系统性一高一低 → 某一端 SOC 标定有偏，该端整体降权。
    返回: 电池id, c_full_median, c_empty_median, bias_ratio, suspect_side
    """
    a = anchors.filter(pl.col("anchor_type").is_in(["FULL_END", "SWAP_FULL"]))
    c = anchors.filter(pl.col("anchor_type") == "EMPTY_END")
    if a.height == 0 or c.height == 0:
        return pl.DataFrame(
            schema={"电池id": pl.Int64, "c_full_median": pl.Float64,
                    "c_empty_median": pl.Float64, "bias_ratio": pl.Float64,
                    "suspect_side": pl.Utf8}
        )
    fa = a.group_by("电池id").agg(pl.col("capacity_corrected").median().alias("c_full_median"))
    fc = c.group_by("电池id").agg(pl.col("capacity_corrected").median().alias("c_empty_median"))
    j = fa.join(fc, on="电池id", how="inner").with_columns(
        ((pl.col("c_full_median") - pl.col("c_empty_median")) / pl.col("c_empty_median")).alias("bias_ratio"),
        pl.len().alias("_n"),
    ).filter(pl.col("_n") >= 3).drop("_n")
    return j.with_columns(
        pl.when(pl.col("bias_ratio").abs() < 0.05).then(pl.lit("ok"))
        .when(pl.col("bias_ratio") > 0).then(pl.lit("full_high"))
        .otherwise(pl.lit("empty_high")).alias("suspect_side")
    )


# ---------------------------------------------------------------- 自检②
def selfcheck_r_drift(anchors: pl.DataFrame) -> pl.DataFrame:
    """R 的长期漂移方向：每设备拟合 R(t) 斜率。斜率>=0 占比过高 → 绝对值不可信，只信排序。

    返回: 电池id, r_slope_per_month, r_months
    """
    df = anchors.with_columns(
        pl.col("更新时间").dt.datetime().alias("t"),
    ).sort(["电池id", "t"])
    out = []
    for (dev,), sub in df.group_by("电池id", maintain_order=True):
        t = (sub["t"] - sub["t"].min()).dt.total_days().to_numpy() / 30.44
        r = sub["capacity_raw"].to_numpy()
        m = np.isfinite(t) & np.isfinite(r)
        if m.sum() < 4:
            continue
        slope = float(np.polyfit(t[m], r[m], 1)[0])
        out.append((dev, slope, int(m.sum())))
    return pl.DataFrame(
        out, schema={"电池id": pl.Int64, "r_slope_per_month": pl.Float64, "r_months": pl.Int64},
        orient="row",
    )


def r_drift_summary(r_slopes: pl.DataFrame) -> dict:
    n = r_slopes.height
    if n == 0:
        return {"n_devices": 0, "nonneg_frac": None, "verdict": "no_data"}
    nonneg = float((r_slopes["r_slope_per_month"] >= 0).mean())
    return {
        "n_devices": n,
        "nonneg_frac": round(nonneg, 4),
        "verdict": "absolute_untrusted" if nonneg > 0.7 else "ok",
    }


# ---------------------------------------------------------------- 自检③
def selfcheck_group_consistency(soh_monthly: pl.DataFrame) -> pl.DataFrame:
    """渠道群体一致性：同渠道设备 SOH 曲线与渠道均值曲线的相关系数。

    返回: 渠道号, 电池id, corr_with_group, lagging(bool)
    """
    rows = []
    for (ch,), sub in soh_monthly.group_by(["渠道号"], maintain_order=True):
        # 以 ym 为轴手工对齐各设备 SOH 序列
        series = {}
        for (dev,), d in sub.group_by("电池id", maintain_order=True):
            series[int(dev)] = dict(zip(d["ym"], d["soh"]))
        all_ym = sorted({ym for s in series.values() for ym in s})
        if len(all_ym) < 4:
            continue
        mean_curve = np.array(
            [np.nanmean([s[ym] for s in series.values() if ym in s]) for ym in all_ym]
        )
        for dev, s in series.items():
            x = np.array([s.get(ym, np.nan) for ym in all_ym])
            m = np.isfinite(x) & np.isfinite(mean_curve)
            if m.sum() < 4 or np.nanstd(x[m]) < 1e-9:
                rows.append((ch, dev, None, False))
                continue
            corr = float(np.corrcoef(x[m], mean_curve[m])[0, 1])
            rows.append((ch, dev, round(corr, 4), corr < 0.3))
    return pl.DataFrame(
        rows,
        schema={"渠道号": pl.Utf8, "电池id": pl.Int64,
                "corr_with_group": pl.Float64, "lagging": pl.Boolean},
        orient="row",
    )


# ---------------------------------------------------------------- 渠道看板
def channel_dashboard(
    anchors: pl.DataFrame, soh_monthly: pl.DataFrame, decay: pl.DataFrame
) -> pl.DataFrame:
    """渠道级看板（方案 5-Step8）：有效锚点率、中位衰退速率、锚点率环比。"""
    total = anchors.group_by("渠道号").agg(pl.len().alias("anchor_total"))
    t1 = (
        anchors.filter(pl.col("cap_tier") == 1)
        .group_by("渠道号")
        .agg(pl.len().alias("tier1_count"))
    )
    dash = total.join(t1, on="渠道号", how="left").with_columns(
        (pl.col("tier1_count") / pl.col("anchor_total")).fill_null(0.0).alias("anchor_rate")
    )
    if decay.height:
        med = decay.group_by("渠道号").agg(
            pl.col("decay_rate_pct_year").median().alias("median_decay_pct_year"),
            pl.col("电池id").n_unique().alias("device_count"),
        )
        dash = dash.join(med, on="渠道号", how="left")
    # 锚点率环比（按月）
    monthly = (
        anchors.with_columns(pl.col("更新时间").dt.strftime("%Y-%m").alias("ym"))
        .group_by(["渠道号", "ym"])
        .agg(
            pl.len().alias("n"),
            (pl.col("cap_tier") == 1).sum().alias("n1"),
        )
        .with_columns((pl.col("n1") / pl.col("n")).alias("rate"))
        .sort(["渠道号", "ym"])
        .with_columns(
            pl.col("rate").diff().over("渠道号").alias("rate_mom_drop")
        )
        .filter(pl.col("rate_mom_drop") < 0)
        .group_by("渠道号")
        .agg(pl.col("rate_mom_drop").min().alias("worst_mom_drop"))
    )
    dash = dash.join(monthly, on="渠道号", how="left")
    return dash.sort("渠道号")
