"""Step 7 双输出 + 衰退速率 + 渠道看板 + 自检③（方案 5-Step7/8、第六章）。

分析集（Tier-1 only）：月度 SOH 序列 + 衰退速率（%/月）+ 预测退役月。
绘图集（含 Tier-2）：原始点 + 清洗后曲线 + 锚点类型着色。
告警集：SOH 月跌幅 >5%、数据不足名单。
渠道看板：有效锚点率、中位衰退速率、锚点率环比、群体一致性（自检③）。
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


def run(soh_tbl: pl.DataFrame, anchors: pl.DataFrame, cfg: Config) -> dict:
    out = os.path.join(OUTPUT_DIR, "reports")
    os.makedirs(out, exist_ok=True)
    soh_out = os.path.join(OUTPUT_DIR, "soh")
    os.makedirs(soh_out, exist_ok=True)
    drop_alert = cfg.get("soh.monthly_drop_alert")

    # 分析集：Tier-1 月度点（soh_tbl 已按设备平滑）
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    analysis = soh_tbl
    # 衰退速率（每设备）
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
    analysis = analysis.join(decay, on=["渠道号", "电池id"], how="left").with_columns(
        (pl.col("decay_rate_pct_month") * 12).alias("decay_rate_pct_year")
    )
    analysis.write_parquet(os.path.join(soh_out, "soh_analysis.parquet"))

    # 绘图集：含 Tier-2 原始锚点 + 平滑曲线（锚点类型着色列）
    plot_pts = anchors.select(
        ["渠道号", "电池id", "更新时间", "anchor_type", "capacity_corrected",
         "c_nom_bms", "cap_tier", "quality_score"]
    ).with_columns(
        (pl.col("capacity_corrected") / pl.col("c_nom_bms")).alias("soh_point")
    )
    plot_pts.write_parquet(os.path.join(soh_out, "soh_plot.parquet"))

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
    # 数据不足名单（设备级有效锚点率 < 20%）
    dev_anchor = anchors.group_by(["渠道号", "电池id"]).agg(
        (pl.col("cap_tier") == 1).sum().cast(pl.Float64).alias("t1"),
        pl.len().cast(pl.Float64).alias("tot"),
    ).with_columns((pl.col("t1") / pl.col("tot")).alias("rate"))
    insufficient = dev_anchor.filter(pl.col("rate") < cfg.get("monitoring.device_anchor_rate_min")).select(
        pl.col("渠道号"), pl.col("电池id"),
        pl.lit("数据质量差").alias("alert_type"),
        ("有效锚点率 " + (pl.col("rate") * 100).round(1).cast(pl.Utf8) + "% < 20%").alias("alert_detail"),
        pl.lit(None, dtype=pl.Utf8).alias("detected_at"),
    )
    alerts = pl.concat([alerts, insufficient], how="vertical_relaxed")
    alerts.write_parquet(os.path.join(soh_out, "soh_alert.parquet"))

    # 渠道看板 + 自检③
    dash = channel_dashboard(anchors, analysis, decay)
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
