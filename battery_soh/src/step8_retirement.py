"""Step 8 退役预测（方案 5-Step7 分析集字段 + 第八章监控）。

对每设备 SOH 序列做线性回归外推：
  predicted_retire_month = SOH 跌破退役线（默认 0.80）的月份
  r_squared              回归拟合度，衡量预测可信度
  置信区间               95% CI（基于残差标准误外推），给出退役月上下界
月数不足 / R² 过低 → 标记"仅供参考"，不外推硬结论。
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl

from .config import OUTPUT_DIR, Config


def _idx(ym: str) -> int:
    y, m = ym.split("-")
    return int(y) * 12 + int(m)


def _ym(idx: int) -> str:
    idx = int(idx)
    return f"{(idx - 1) // 12}-{(idx - 1) % 12 + 1:02d}"


def forecast_device(ys: pl.Series, yms: list[str], retire: float,
                    min_months: int, r2_min: float, z: float) -> dict:
    x = np.array([_idx(v) for v in yms], dtype=float)
    y = ys.to_numpy().astype(float)
    m = np.isfinite(y)
    x, y = x[m], y[m]
    if len(y) < min_months:
        return {"predicted_retire_month": None, "r_squared": None,
                "retire_ci_low": None, "retire_ci_high": None, "note": "数据不足"}
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n = len(y)
    resid_std = float(np.sqrt(ss_res / max(n - 2, 1)))
    note = ""
    if slope >= -1e-12:  # 数值零容差：平坦或上升序列不外推
        return {"predicted_retire_month": None, "r_squared": round(r2, 4),
                "retire_ci_low": None, "retire_ci_high": None,
                "note": "无衰退趋势" if r2 >= r2_min else "无衰退趋势（R²低）"}
    if r2 < r2_min:
        note = f"R²={round(r2,3)}<{r2_min}，仅供参考"
    x_retire = (retire - intercept) / slope  # SOH 跌破 retire 的月索引
    # 置信区间：反解 y=retire ± z*resid_std
    x_lo = (retire + z * resid_std - intercept) / slope
    x_hi = (retire - z * resid_std - intercept) / slope
    lo, hi = sorted([x_lo, x_hi])
    return {
        "predicted_retire_month": _ym(int(round(x_retire))),
        "r_squared": round(r2, 4),
        "retire_ci_low": _ym(int(round(lo))),
        "retire_ci_high": _ym(int(round(hi))),
        "note": note,
    }


def run(analysis: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    r = cfg.get("retirement")
    retire, min_months, r2_min, z = r["soh_retire"], r["min_months"], r["r2_min"], r["ci_z"]
    levels = r["health_levels"]

    rows = []
    for (ch, dev), sub in analysis.group_by(["渠道号", "电池id"], maintain_order=True):
        sub = sub.sort("ym")
        f = forecast_device(sub["soh"], sub["ym"].to_list(), retire, min_months, r2_min, z)
        latest = sub["soh"].to_numpy()
        latest = latest[np.isfinite(latest)]
        soh_now = float(latest[-1]) if len(latest) else None
        hl = _health_level(soh_now, levels)
        rows.append((ch, int(dev), soh_now, hl, f["predicted_retire_month"],
                     f["r_squared"], f["retire_ci_low"], f["retire_ci_high"], f["note"]))
    out = pl.DataFrame(
        rows,
        schema={
            "渠道号": pl.Utf8, "电池id": pl.Int64, "soh_latest": pl.Float64,
            "health_level": pl.Utf8, "predicted_retire_month": pl.Utf8,
            "r_squared": pl.Float64, "retire_ci_low": pl.Utf8,
            "retire_ci_high": pl.Utf8, "note": pl.Utf8,
        },
        orient="row",
    )
    dest = os.path.join(OUTPUT_DIR, "retirement")
    os.makedirs(dest, exist_ok=True)
    out.write_parquet(os.path.join(dest, "retirement_forecast.parquet"))
    n_pred = out.filter(pl.col("predicted_retire_month").is_not_null()).height
    print(f"[step8] 退役预测：{out.height} 设备，可预测 {n_pred} 台")
    return out


def _health_level(soh, levels: list[float]) -> str:
    if soh is None:
        return "unknown"
    a, b, c = levels  # 0.85, 0.75, 0.65
    if soh >= a:
        return "健康"
    if soh >= b:
        return "关注"
    if soh >= c:
        return "预警"
    return "退役"
