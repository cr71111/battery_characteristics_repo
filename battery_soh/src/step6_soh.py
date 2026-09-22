"""Step 6 设备级时序处理（Reduce 阶段，方案 5-Step6）。

6a 滚动 MAD 去毛刺：窗口 7（5~9），阈值 3×MAD，自适应（禁止固定阈值，P07）。
   仅对月内锚点数 >= 2 的月份生效；单锚点月直接保留（稀疏数据不做抹平）。
6b PAVA 单调回归：O(n)，强制非增（电池容量只降不升），保趋势不过度平滑。
6c SOH = C_smoothed / C_nom_spec（分母=规格书标称容量 c_nom_spec，绝对口径，
   跨设备可比；用户 2026-09 决定改用规格书 37Ah，替代原"设备首年容量中位数"相对口径）。
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
    fallback = float(cfg.get("c_nom_spec") or 37.0)
    # 逐型号标称容量分母（绝对口径）；无型号列/缺参数时回退全局 c_nom_spec
    model_params = cfg.model_params
    low = float(cfg.get("exclude.soh_low_ratio") or 0.90)

    has_model = "电池型号" in monthly.columns
    out_chunks = []
    n_trimmed = 0
    for (ch, dev), sub in monthly.group_by(["渠道号", "电池id"], maintain_order=True):
        sub = sub.sort("ym")
        c_nom_spec = fallback
        if has_model:
            mp = model_params.get(str(sub["电池型号"][0]))
            if mp:
                c_nom_spec = float(mp["c_nom_spec"])
        cap = sub["capacity_median"].to_numpy()
        # 6-pre 前导异常月剔除：头 k 个月 SOH < low 而后续中位 >= low →
        # 初期校准噪声（换BMS新ID首月偏低、次月恢复），在平滑前删除，
        # 否则 PAVA 非增会把整条曲线拉低。整段都低的设备留给 step7 判换BMS。
        soh_pre = cap / c_nom_spec
        kk = 0
        while kk < len(cap) and np.isfinite(soh_pre[kk]) and soh_pre[kk] < low:
            kk += 1
        if 0 < kk < len(cap) and float(np.median(soh_pre[kk:])) >= low:
            n_trimmed += kk
            sub = sub.slice(kk)
            cap = cap[kk:]
        w = np.maximum(sub["capacity_count"].to_numpy().astype(float), 1.0)
        # 6a 去毛刺：仅月内锚点数>=2 的月份参与判定，单锚点月直接保留
        n_anchor = sub["capacity_count"].to_numpy()
        mask = mad_outlier_mask(cap, win, k) & (n_anchor >= 2)
        cap_clean = cap.copy()
        if mask.any():
            rep = mad_replacements(cap, mask, win)
            cap_clean[mask] = rep[mask]
        # 6b PAVA 非增
        cap_smooth = pava_decreasing(cap_clean, w)
        # 6c SOH = 平滑容量 / 规格书标称容量（绝对口径）
        denom = c_nom_spec
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
    result = pl.concat(out_chunks).sort(["渠道号", "电池id", "ym"]) if out_chunks else monthly.clear()
    if n_trimmed:
        print(f"[step6] 前导异常月剔除 {n_trimmed} 个月点（SOH<{low:.0%} 且后续恢复）")
    return result


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
