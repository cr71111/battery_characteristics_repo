"""Step 6 设备级时序处理（Reduce 阶段，方案 5-Step6）。

6a 滚动 MAD 去毛刺：窗口 7（5~9），阈值 3×MAD，自适应（禁止固定阈值，P07）。
6b PAVA 单调回归：O(n)，强制非增（电池容量只降不升），保趋势不过度平滑。
6c SOH = C_smoothed / C_nom_BMS（C_nom 取设备级首年稳健中位数，禁止规格书，P04）。
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .config import Config


def mad_outlier_mask(values: np.ndarray, window: int, k: float) -> np.ndarray:
    """滚动 MAD：|x - 窗口中位| > k*1.4826*MAD → 毛刺（True=离群）。"""
    n = len(values)
    mask = np.zeros(n, dtype=bool)
    half = window // 2
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = values[lo:hi]
        seg = seg[np.isfinite(seg)]
        if len(seg) < 3:
            continue
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        scale = 1.4826 * mad
        if scale <= 1e-12:
            # 窗口内几乎全等：偏离中位数即毛刺
            if np.isfinite(values[i]) and abs(values[i] - med) > 1e-9:
                mask[i] = True
            continue
        if np.isfinite(values[i]) and abs(values[i] - med) > k * scale:
            mask[i] = True
    return mask


def pava_decreasing(y: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """PAVA（Pool Adjacent Violators）加权非增回归。"""
    n = len(y)
    if n == 0:
        return y.copy()
    w = np.ones(n) if w is None else np.maximum(w.astype(float), 1e-9)
    out = y.astype(float).copy()
    # 块结构：value, weight, start, end
    blocks: list[list] = []
    for i in range(n):
        blocks.append([out[i], w[i], i, i])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:  # 违反非增
            v1, w1, s1, e1 = blocks[-2]
            v2, w2, s2, e2 = blocks[-1]
            blocks[-2] = [(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, s1, e2]
            blocks.pop()
    res = np.empty(n)
    for v, _, s, e in blocks:
        res[s : e + 1] = v
    return res


def run(monthly: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    win = cfg.get("soh.mad_window")
    k = cfg.get("soh.mad_threshold")
    first_year = int(monthly["year"].min())

    out_chunks = []
    for (ch, dev), sub in monthly.group_by(["渠道号", "电池id"], maintain_order=True):
        sub = sub.sort("ym")
        cap = sub["capacity_median"].to_numpy()
        w = np.maximum(sub["capacity_count"].to_numpy().astype(float), 1.0)
        # 6a 去毛刺：离群点用窗口中位数替换
        mask = mad_outlier_mask(cap, win, k)
        cap_clean = cap.copy()
        if mask.any():
            rep = mad_replacements(cap, mask, win)
            cap_clean[mask] = rep[mask]
        # 6b PAVA 非增
        cap_smooth = pava_decreasing(cap_clean, w)
        # 6c SOH：首年中位数作 C_nom
        fy = sub["year"].to_numpy() == first_year
        denom = np.median(cap_clean[fy]) if fy.any() and np.isfinite(cap_clean[fy]).any() else np.median(cap_clean)
        soh = cap_smooth / denom if denom > 0 else np.full(len(cap_smooth), np.nan)
        soh_raw = cap / denom if denom > 0 else np.full(len(cap), np.nan)
        out_chunks.append(
            sub.with_columns(
                pl.Series("capacity_smoothed", cap_smooth),
                pl.Series("soh", soh),
                pl.Series("soh_raw", soh_raw),
                pl.Series("is_spike", mask.astype(bool).tolist()),
                pl.Series("c_nom_soh", [float(denom)] * len(cap)),
            )
        )
    return pl.concat(out_chunks).sort(["渠道号", "电池id", "ym"]) if out_chunks else monthly.clear()


def mad_replacements(values: np.ndarray, mask: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    out = values.copy()
    n = len(values)
    for i in np.where(mask)[0]:
        lo, hi = max(0, i - half), min(n, i + half + 1)
        seg = values[lo:hi]
        seg = seg[np.isfinite(seg) & ~mask[lo:hi]]
        if len(seg):
            out[i] = np.median(seg)
    return out
