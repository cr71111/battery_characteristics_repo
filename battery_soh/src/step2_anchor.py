"""Step 2 锚点打标（方案 3.3 / 5-Step2）。

逐行打 tag：anchor_type ∈ {FULL_END, SWAP_FULL, EMPTY_END, OCV, NONE}，
并给出 soc_anchor（锚点处 SOC 真值：业务事实 1.0/0.0 或 OCV 查表值）。

电流在此步只参与符号与静置判据，不连续不影响（不累加）。
锚点判据使用 Step 0 输出的体系参数（V_full、V_cutoff）。

互斥优先级：SWAP_FULL(B) > EMPTY_END(C) > FULL_END(A) > OCV(D)；
同一行命中多个判据 → 取优先级最高者，并打 anchor_conflict 标记。
"""
from __future__ import annotations

import numpy as np
import polars as pl

from .config import Config


def _ocv_lookup(ocv_df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return ocv_df["soc"].to_numpy(), ocv_df["ocv_v"].to_numpy()


def _soc_from_ocv(v_cell: np.ndarray, soc_axis: np.ndarray, ocv_axis: np.ndarray) -> np.ndarray:
    """末段单体电压反查 OCV 表得 SOC_anchor（np.interp 要求 x 递增，OCV 表已校验单调）。"""
    return np.interp(v_cell, ocv_axis, soc_axis)


def run(df: pl.DataFrame, cfg: Config, params: dict | None = None) -> pl.DataFrame:
    a = cfg.get("anchor")
    p = params or {}
    v_full = p.get("v_full") or cfg.get("generated.v_full") or cfg.chem_params()["v_full"]
    v_cutoff = p.get("v_cutoff") or cfg.get("generated.v_cutoff") or cfg.chem_params()["v_cutoff"]
    series = p.get("series") or cfg.series
    chemistry = p.get("chemistry")
    c_nom_default = p.get("c_nom_spec") or cfg.get("generated.c_nom_default") or 40.0
    rest_i = cfg.get("anchor.rest_c_rate") * c_nom_default  # |I| < 0.02C
    dur_s = a["duration_hours"] * 3600.0
    gap_s = a["gap_hours"] * 3600.0
    soc_axis, ocv_axis = _ocv_lookup(cfg.ocv_table(chemistry, series))

    df = df.sort(["电池id", "更新时间"])
    out_chunks = []
    for (dev,), sub in df.group_by("电池id", maintain_order=True):
        n = sub.height
        t = sub["更新时间"].to_numpy().astype("datetime64[s]").astype(np.int64)
        soc = sub["SOC"].to_numpy()
        v = sub["电压"].to_numpy()
        i = sub["电流"].to_numpy()
        valid = np.isfinite(soc) & np.isfinite(v) & np.isfinite(i)

        dt = np.diff(t, prepend=t[0])  # 距上一行的秒差
        # 静置判据：|I| < 0.02C，且与上一行间隔 <= 断档容忍（否则持续性中断）
        rest = (np.abs(i) < rest_i) & valid & ((dt <= gap_s) | (np.arange(n) == 0))
        # 持续 >= 2h：向前累计静置时长（向量化：连续静置段内累加 dt，段首归零）
        cont = rest & np.concatenate(([False], rest[:-1])) & (dt <= gap_s)
        cadd = np.cumsum(np.where(cont, dt, 0.0))
        run_start = rest & ~cont
        base = np.where(run_start, cadd, -np.inf)
        rest_dur = np.where(rest, cadd - np.maximum.accumulate(base), 0.0)
        long_rest = rest_dur >= dur_s
        # 方案要求"取静置段末行"：仅每段最后一行标 OCV（下一行非静置/断档/数据末尾）
        next_rest = np.concatenate((rest[1:], [False]))
        seg_end = long_rest & ~next_rest

        # A. 满充末端：SOC>0.95 且 V>V_full*0.99 且 0<I<涓流上限 且持续>=2h（涓流段用静置近似）
        cond_a = (
            (soc > a["soc_full"])
            & (v > v_full * a["v_full_ratio"])
            & (i > 0) & (i < a["full_current_max_a"])
            & valid
        )
        # B. 站内满充完成：I 由正转静置 + SOC>=0.98 + 电压回落至静置态（V < 满充电压*0.995）
        i_prev = np.roll(i, 1)
        cond_b = (
            (i_prev > a["full_current_max_a"]) & (np.abs(i) < rest_i)
            & (soc >= a["soc_swap_full"])
            & (v < v_full * a["v_full_ratio"])
            & valid & (np.arange(n) > 0)
        )
        # C. 放空末端：SOC<0.10 且 V<V_cutoff*1.05 且 I<0
        cond_c = (soc < a["soc_empty"]) & (v < v_cutoff * a["v_cutoff_ratio"]) & (i < 0) & valid
        # D. 静置 OCV：|I|<0.02C 持续>=2h（取静置段末行），SOC_anchor 由电压查表
        cond_d = seg_end & valid

        conflict = ((cond_a.astype(int) + cond_b.astype(int) + cond_c.astype(int) + cond_d.astype(int)) > 1)
        atype = np.full(n, "NONE", dtype=object)
        atype[cond_d] = "OCV"
        atype[cond_a] = "FULL_END"
        atype[cond_c] = "EMPTY_END"
        atype[cond_b] = "SWAP_FULL"

        # soc_anchor：B=1.0（业务事实）；C=0.0 附近（低压保护强制归零，用上报值兜底防除零）；
        # A=SOC 上报值（已被钳位）；D=OCV 反查值（优先于上报值）
        soc_anchor = np.full(n, np.nan)
        soc_anchor[cond_b] = 1.0
        soc_anchor[cond_c] = np.maximum(soc[cond_c], 0.02)  # 防除零，最低 0.02
        soc_anchor[cond_a] = np.maximum(soc[cond_a], 0.95)
        v_cell = v / series
        soc_d = _soc_from_ocv(v_cell, soc_axis, ocv_axis)
        soc_anchor[cond_d] = np.clip(soc_d[cond_d], 0.02, 0.98)

        flags = [[] for _ in range(n)]
        for k in np.where(conflict)[0]:
            flags[k].append("anchor_conflict")
        for k in np.where(sub["current_sensor_drift"].to_numpy())[0]:
            flags[k].append("sensor_drift")
        for k in np.where(sub["invalid_loopnum"].to_numpy())[0]:
            flags[k].append("invalid_loopnum")
        for k in np.where(sub["invalid_sensor"].to_numpy())[0]:
            flags[k].append("invalid_sensor")

        base_cols = [c for c in sub.columns
                     if c not in ("invalid_loopnum", "invalid_sensor", "current_reversed",
                                  "current_sensor_drift", "time_gap")]
        chunk = sub.select(base_cols).with_columns(
            pl.Series("anchor_type", atype),
            pl.Series("soc_anchor", soc_anchor),
            pl.Series("flags", flags, dtype=pl.List(pl.Utf8)),
        )
        out_chunks.append(chunk)

    res = pl.concat(out_chunks) if out_chunks else df.clear()
    return res
