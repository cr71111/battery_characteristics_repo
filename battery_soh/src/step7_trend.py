"""Step 7 双输出 + 衰退速率 + 渠道看板 + 自检③（方案 5-Step7/8、第六章）。

分析集（Tier-1 only）：月度 SOH 序列 + 衰退速率（%/月）+ 预测退役月。
绘图集（含 Tier-2）：原始点 + 清洗后曲线 + 锚点类型着色。
告警集：SOH 月跌幅 >5%、数据不足名单。
渠道看板：有效锚点率、中位衰退速率、锚点率环比、群体一致性（自检③）。

锚点层可能达千万行：全部经 LazyFrame streaming 处理，不落内存。
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl

from .analysis import channel_dashboard, selfcheck_group_consistency
from .config import OUTPUT_DIR, Config


def _ym_to_idx(ym: str) -> int:
    y, m = ym.split("-")
    return int(y) * 12 + int(m)


def decay_rate(sub: pl.DataFrame) -> dict:
    """月度衰退速率（%/月）：对 soh 序列做最小二乘线性拟合斜率。"""
    x = np.array([_ym_to_idx(v) for v in sub["ym"]], dtype=float)
    y = sub["soh"].to_numpy()
    m = np.isfinite(y)
    if m.sum() < 3:
        return {"decay_rate_pct_month": None, "r_squared": None}
    slope, intercept = np.polyfit(x[m], y[m], 1)
    pred = slope * x[m] + intercept
    ss_res = float(np.sum((y[m] - pred) ** 2))
    ss_tot = float(np.sum((y[m] - y[m].mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
    return {"decay_rate_pct_month": round(float(slope) * 100, 4), "r_squared": round(r2, 4) if r2 is not None else None}


def build_exclusions(soh_tbl: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """设备级剔除名单：仅"疑似旧电池换BMS"整台剔除。

    判定（基于 step5 可靠月 + step6 前导异常月已删后的序列）：
    - 全部可靠月（>=2 个）SOH 均 < soh_low_ratio → 容量一直上不来，
      疑似旧电池换 BMS 重新注册新 ID（循环归零）→ 整台剔除；
    - 其余设备一律保留（数据少 / 波动大不再整台剔除；
      前几月低、后面正常的已在 step6 平滑前删掉前导异常月）。
    """
    low = float(cfg.get("exclude.soh_low_ratio") or 0.90)
    d = (
        soh_tbl.sort(["渠道号", "电池id", "ym"])
        .group_by(["渠道号", "电池id"])
        .agg(
            pl.len().alias("reliable_months"),
            pl.col("soh").min().alias("soh_min"),
            pl.col("soh").max().alias("soh_max"),
            pl.col("soh").first().alias("soh_first_reliable"),
            pl.col("soh").head(2).alias("_h2"),
            pl.col("capacity_median").first().alias("_cap1"),
            pl.col("capacity_median").max().alias("_cmax"),
            pl.col("capacity_median").min().alias("_cmin"),
            pl.col("ym").first().alias("ym_first"),
            pl.col("ym").last().alias("ym_last"),
            pl.col("loop_median").first().alias("loop_first"),
        )
        .with_columns(
            pl.col("_h2").list.get(1, null_on_oob=True).alias("soh_second_reliable"),
            ((pl.col("_cmax") - pl.col("_cmin")) / pl.col("_cap1")).alias("spread_ratio"),
            pl.lit("bms_swap").alias("reasons"),
        )
        .filter((pl.col("reliable_months") >= 2) & (pl.col("soh_max") < low))
        .select(
            "渠道号", "电池id", "reliable_months", "soh_first_reliable",
            "soh_second_reliable", "spread_ratio", "loop_first",
            "ym_first", "ym_last", "reasons",
        )
        .sort(["渠道号", "电池id"])
    )
    return d


def run(soh_tbl: pl.DataFrame, anchors: pl.LazyFrame | pl.DataFrame, cfg: Config) -> dict:
    out = os.path.join(OUTPUT_DIR, "reports")
    os.makedirs(out, exist_ok=True)
    soh_out = os.path.join(OUTPUT_DIR, "soh")
    os.makedirs(soh_out, exist_ok=True)
    drop_alert = cfg.get("soh.monthly_drop_alert")
    al = anchors.lazy() if isinstance(anchors, pl.DataFrame) else anchors

    # 设备级剔除：仅疑似换BMS整台移除；其余保留（前导异常月已在 step6 处理）
    excl = build_exclusions(soh_tbl, cfg)
    excl.write_csv(os.path.join(out, "excluded_devices.csv"))
    print(f"[step7] 剔除疑似换BMS设备 {excl.height} 台 → reports/excluded_devices.csv")
    soh_tbl = soh_tbl.join(excl.select(["渠道号", "电池id"]), on=["渠道号", "电池id"],
                           how="anti", nulls_equal=True)

    # 分析集：月度点（soh_tbl 已按设备平滑）
    analysis = soh_tbl
    rows = []
    for (ch, dev), sub in analysis.group_by(["渠道号", "电池id"], maintain_order=True):
        d = decay_rate(sub)
        rows.append((ch, int(dev), d["decay_rate_pct_month"], d["r_squared"]))
    decay = pl.DataFrame(
        rows,
        schema={"渠道号": pl.Utf8, "电池id": pl.Int64,
                "decay_rate_pct_month": pl.Float64, "r_squared": pl.Float64},
        orient="row",
    ).with_columns(
        (pl.col("decay_rate_pct_month") * 12).alias("decay_rate_pct_year")
    )
    analysis = analysis.join(decay, on=["渠道号", "电池id"], how="left")
    analysis.write_parquet(os.path.join(soh_out, "soh_analysis.parquet"))

    # 绘图集：Tier-1/2 锚点 + soh_point（流式写出，不进内存）
    plot_cols = ["渠道号", "电池id", "更新时间", "anchor_type", "capacity_corrected",
                 "c_nom_bms", "cap_tier", "quality_score"]
    (
        al.filter(pl.col("cap_tier") <= 2)
        .select(plot_cols)
        .with_columns((pl.col("capacity_corrected") / pl.col("c_nom_bms")).alias("soh_point"))
        .sink_parquet(os.path.join(soh_out, "soh_plot.parquet"))
    )

    # 告警集：SOH 月跌幅 > 5%
    alerts = (
        analysis.sort(["电池id", "ym"])
        .with_columns(pl.col("soh").diff().over("电池id").alias("soh_drop"))
        .filter(pl.col("soh_drop") < -drop_alert)
        .select(
            pl.col("渠道号"), pl.col("电池id"),
            pl.lit("SOH月跌幅异常").alias("alert_type"),
            (pl.col("ym") + " 跌幅 " + (pl.col("soh_drop") * 100).round(2).cast(pl.Utf8) + "%/月").alias("alert_detail"),
            pl.lit(None, dtype=pl.Utf8).alias("detected_at"),
        )
    )
    # 数据不足名单（设备级 Tier-1 锚点率 < 20%）
    dev_anchor = (
        al.group_by(["渠道号", "电池id"])
        .agg(
            (pl.col("cap_tier") == 1).sum().cast(pl.Float64).alias("t1"),
            pl.len().cast(pl.Float64).alias("tot"),
        )
        .with_columns((pl.col("t1") / pl.col("tot")).alias("rate"))
        .collect(engine="streaming")
    )
    insufficient = dev_anchor.filter(pl.col("rate") < cfg.get("monitoring.device_anchor_rate_min")).select(
        pl.col("渠道号"), pl.col("电池id"),
        pl.lit("数据质量差").alias("alert_type"),
        ("有效锚点率 " + (pl.col("rate") * 100).round(1).cast(pl.Utf8) + "% < 20%").alias("alert_detail"),
        pl.lit(None, dtype=pl.Utf8).alias("detected_at"),
    )
    alerts = pl.concat([alerts, insufficient], how="vertical_relaxed")
    alerts.write_parquet(os.path.join(soh_out, "soh_alert.parquet"))

    # 渠道看板 + 自检③（锚点侧聚合走 streaming）
    dash = channel_dashboard(al, analysis, decay)
    consistency = selfcheck_group_consistency(analysis)
    dash.write_csv(os.path.join(out, "dashboard.csv"))
    consistency.write_csv(os.path.join(out, "group_consistency.csv"))
    _write_html(dash, consistency, os.path.join(out, "dashboard.html"))

    print(f"[step7] 分析集 {analysis.height} 行，告警 {alerts.height} 条，渠道 {dash.height} 个")
    return {"analysis": analysis, "alerts": alerts, "dashboard": dash}


def _write_html(dash: pl.DataFrame, consistency: pl.DataFrame, path: str) -> None:
    def tbl(df: pl.DataFrame) -> str:
        head = "".join(f"<th>{c}</th>" for c in df.columns)
        body = "".join(
            "<tr>" + "".join(f"<td>{'' if v is None else v}</td>" for v in row) + "</tr>"
            for row in df.iter_rows()
        )
        return f"<table border=1 cellpadding=4><tr>{head}</tr>{body}</table>"

    html = (
        "<html><meta charset='utf-8'><h2>渠道看板</h2>"
        + tbl(dash)
        + "<h2>自检③ 群体一致性（lagging=落后单体）</h2>"
        + tbl(consistency.filter(pl.col("lagging")) if consistency.height else consistency)
        + "</html>"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
