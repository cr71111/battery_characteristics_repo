"""Step 0 体检（方案第五章 Step 0）：一次性扫描，决定后续全部路线。

输出：
  output/profile/diagnosis_report.json   七张体检结论
  output/profile/config.generated.yaml   实测体系参数（供三段配置合并）
  output/profile/device_list_valid.csv   设备级 C_nom_BMS + 有效性

大数据量（亿级行）策略：设备级 Python 循环用哈希抽样（约 4%），
全量聚合（C_nom / 密度）走 polars streaming。
"""
from __future__ import annotations

import json
import os

import numpy as np
import polars as pl
import yaml

from .config import OUTPUT_DIR, PROJECT_ROOT, Config
from .io import scan_raw

SERIES_CANDIDATES = {"LFP": [24, 25], "NMC": [20, 21]}
PLATFORM_SLOPE_LFP_MAX = 0.03  # V/cell 每单位 SOC；平台区低于此 → LFP
SAMPLE_MOD = 25  # 设备抽样：电池id % 25 == 0（约 4%）


def _sampled(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter(pl.col("电池id") % SAMPLE_MOD == 0)


def _collect(lf: pl.LazyFrame) -> pl.DataFrame:
    return lf.collect(engine="streaming")


def _r_stats(lf: pl.LazyFrame) -> dict:
    """1. R 有效性：R=剩余容量/SOC 按设备的 CV 与斜率（抽样设备）。"""
    df = _collect(
        _sampled(lf)
        .filter((pl.col("SOC") > 0.05) & (pl.col("剩余容量") > 0))
        .with_columns((pl.col("剩余容量") / pl.col("SOC")).alias("R"))
        .sort(["电池id", "更新时间"])
        .select(["电池id", "更新时间", "R"])
    )
    rows = []
    for (dev,), sub in df.group_by("电池id", maintain_order=True):
        r = sub["R"].to_numpy()
        if len(r) < 30:
            continue
        cv = float(np.nanstd(r) / np.nanmean(r))
        t = (sub["更新时间"] - sub["更新时间"].min()).dt.total_days().to_numpy() / 30.44
        m = np.isfinite(t) & np.isfinite(r)
        slope = float(np.polyfit(t[m], r[m], 1)[0]) if m.sum() >= 12 else 0.0
        rows.append((dev, cv, slope, float(np.nanmedian(r))))
    st = pl.DataFrame(rows, schema={"电池id": pl.Int64, "cv": pl.Float64,
                                    "slope": pl.Float64, "r_med": pl.Float64},
                      orient="row")
    med_cv = float(st["cv"].median()) if st.height else 1.0
    med_slope = float(st["slope"].median()) if st.height else 0.0
    return {
        "n_devices": st.height,
        "median_cv": round(med_cv, 5),
        "median_slope_per_month": round(med_slope, 5),
        "cv_dynamic_frac": round(float((st["cv"] > 0.02).mean()), 4) if st.height else 0.0,
    }, st


def _chemistry(lf: pl.LazyFrame) -> dict:
    """2. 体系判别：平台区 dV/dSOC 斜率（长平台=LFP，单调斜线=NMC）+ 串数。"""
    d = _collect(
        _sampled(lf)
        .filter((pl.col("SOC") >= 0.2) & (pl.col("SOC") <= 0.8) & pl.col("电流").abs().lt(1.0))
        .select(["电池id", "SOC", "电压"])
    )
    slopes = []
    for (dev,), sub in d.group_by("电池id", maintain_order=True):
        if sub.height < 50:
            continue
        slopes.append(float(np.polyfit(sub["SOC"], sub["电压"], 1)[0]))
    med_slope = float(np.median(slopes)) if slopes else 0.0
    chem = "LFP" if med_slope < PLATFORM_SLOPE_LFP_MAX * 25 else "NMC"
    # 串数：满充电压/单体满充、放空电压/单体截止 取最接近的候选串数
    lim = _collect(
        _sampled(lf)
        .group_by("电池id")
        .agg(pl.col("电压").max().alias("vmax"), pl.col("电压").min().alias("vmin"))
    )
    cell = {"LFP": (3.65, 2.60), "NMC": (4.20, 3.00)}[chem]
    best, best_err = None, 1e9
    for s in SERIES_CANDIDATES[chem]:
        e_full = abs(float(lim["vmax"].median()) / s - cell[0])
        e_cut = abs(float(lim["vmin"].quantile(0.05)) / s - cell[1])
        err = e_full + e_cut
        if err < best_err:
            best, best_err = s, err
    return {
        "platform_slope_v_per_soc": round(med_slope, 3),
        "chemistry": chem,
        "series": best,
        "cell_v_full": cell[0],
        "cell_v_cutoff": cell[1],
        "pack_vmax_median": round(float(lim["vmax"].median()), 2),
        "pack_vmin_p05": round(float(lim["vmin"].quantile(0.05)), 2),
    }


def _sampling(lf: pl.LazyFrame) -> dict:
    """3. 真实采样间隔分布（抽样设备）。"""
    d = _collect(
        _sampled(lf)
        .sort(["电池id", "更新时间"])
        .with_columns(pl.col("更新时间").diff().over("电池id").dt.total_minutes().alias("gap"))
        .filter(pl.col("gap") > 0).select("gap")
    )
    q = list(d["gap"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]))
    return {
        "n": d.height,
        "frac_lt_30min": round(float((d["gap"] < 30).mean()), 4),
        "minutes_p05_p50_p95": [round(float(x), 1) for x in (q[0], q[2], q[4])],
    }


def _loopnum(lf: pl.LazyFrame, thr: float) -> dict:
    """4. 循环次数单调性（抽样设备）。"""
    d = _collect(
        _sampled(lf)
        .sort(["电池id", "更新时间"])
        .with_columns(pl.col("循环次数").diff().over("电池id").alias("dl"))
        .filter(pl.col("dl").is_not_null()).select("dl")
    )
    neg = float((d["dl"] < 0).mean()) if d.height else 0.0
    return {"neg_frac": round(neg, 5), "use_time_window": neg > thr}


def _current_sign(lf: pl.LazyFrame) -> dict:
    """5. 电流符号抽检：SOC 明显上升段 I 应恒正（抽样设备）。"""
    d = _collect(
        _sampled(lf)
        .sort(["电池id", "更新时间"])
        .with_columns(
            pl.col("SOC").diff().over("电池id").alias("dsoc"),
            pl.col("电流").rolling_mean(3).over("电池id").alias("i3"),
        )
        .filter(pl.col("dsoc") > 0.03)
        .group_by("电池id")
        .agg(
            pl.col("i3").median().alias("i_med"),
            (pl.col("i3") < 0).mean().alias("neg_frac"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= 20)
    )
    if d.height == 0:
        return {"checked_devices": 0, "reversed_devices": [], "note": "无 SOC 上升样本可检"}
    rev = d.filter(pl.col("neg_frac") > 0.8)["电池id"].to_list()
    return {
        "checked_devices": d.height,
        "reversed_devices": [int(x) for x in rev],
        "reversed_frac": round(len(rev) / d.height, 4),
    }


def _current_zero(lf: pl.LazyFrame, thr: float, near: float) -> dict:
    """6. 电流零点分布：|I|<1A 占比 >90% → 电流字段实质缺失（第一优先级）。"""
    frac = float(
        _collect(_sampled(lf).select(pl.col("电流").abs().lt(near).mean())).item()
    )
    return {"near_zero_frac": round(frac, 4), "current_effectively_missing": frac > thr}


def _c_nom(lf: pl.LazyFrame) -> pl.DataFrame:
    """4 章 P04：C_nom_BMS = 设备级 R 众数（0.5Ah 分箱取最密箱中值），非规格书猜测。

    全量设备（streaming 聚合，供 Step 3 join）。
    """
    df = (
        lf.filter((pl.col("SOC") > 0.05) & (pl.col("剩余容量") > 0))
        .with_columns(
            (pl.col("剩余容量") / pl.col("SOC")).alias("R"),
        )
        .with_columns((pl.col("R") / 0.5).floor().mul(0.5).alias("rb"))
        .group_by(["渠道号", "电池id", "rb"])
        .agg(pl.len().alias("cnt"))
        .sort(["渠道号", "电池id", "cnt"], descending=[False, False, True])
        .group_by(["渠道号", "电池id"], maintain_order=True)
        .agg(
            (pl.col("rb").first() + 0.25).alias("c_nom_bms"),
            pl.col("cnt").sum().alias("n_rows"),
        )
    )
    span = (
        lf.group_by(["渠道号", "电池id"])
        .agg(pl.col("更新时间").min().alias("t0"), pl.col("更新时间").max().alias("t1"))
    )
    return df.join(span, on=["渠道号", "电池id"], how="left").collect(engine="streaming")


def _anchor_density(lf: pl.LazyFrame, v_full: float, v_cutoff: float) -> dict:
    """CONSTANT 分支用：粗略锚点候选密度（每月每设备，抽样设备）。"""
    d = _collect(
        _sampled(lf)
        .filter(
            (pl.col("SOC") >= 0.98)
            | ((pl.col("SOC") < 0.10) & (pl.col("电压") < v_cutoff * 1.05))
            | ((pl.col("SOC") > 0.95) & (pl.col("电压") > v_full * 0.99))
        )
        .with_columns(pl.col("更新时间").dt.strftime("%Y-%m").alias("ym"))
        .group_by(["电池id", "ym"]).agg(pl.len().alias("n"))
    )
    months = d.height
    return {
        "anchor_candidate_rows": int(d["n"].sum()) if months else 0,
        "avg_anchor_per_device_month": round(float(d["n"].mean()), 3) if months else 0.0,
    }


def run(cfg: Config, dry_run: bool = False) -> dict:
    out_dir = os.path.join(OUTPUT_DIR, "profile")
    os.makedirs(out_dir, exist_ok=True)
    lf = scan_raw(cfg.data_path)

    r_stats, r_tbl = _r_stats(lf)
    chem = _chemistry(lf)
    sampling = _sampling(lf)
    loopnum = _loopnum(lf, cfg.get("selfcheck.neg_loopnum_frac"))
    sign = _current_sign(lf)
    czero = _current_zero(
        lf, cfg.get("selfcheck.current_zero_frac_limit", 0.9),
        cfg.get("anchor.current_near_zero_a", 1.0),
    )
    c_nom = _c_nom(lf)
    density = _anchor_density(lf, chem["series"] * chem["cell_v_full"], chem["series"] * chem["cell_v_cutoff"])

    # 7. R_MODE 分支（v2.0 必须写进主流程，P06）
    dynamic = r_stats["median_cv"] >= cfg.get("selfcheck.r_cv_dynamic") or \
        abs(r_stats["median_slope_per_month"]) > cfg.get("selfcheck.r_slope_eps")
    r_mode = "DYNAMIC" if dynamic else "CONSTANT"
    if r_mode == "DYNAMIC":
        route = "ABSOLUTE"
    elif density["avg_anchor_per_device_month"] >= cfg.get("selfcheck.min_anchor_per_month"):
        route = "ABSOLUTE"  # 锚点法仍可用，但数据稀疏
    else:
        route = "L2_TREND"  # 降级：放弃绝对容量，群体排序代理

    report = {
        "r_validity": r_stats,
        "chemistry": chem,
        "sampling": sampling,
        "loopnum": loopnum,
        "current_sign": sign,
        "current_zero": czero,
        "anchor_density": density,
        "R_MODE": r_mode,
        "route": route,
        "recommendation": (
            "正常走 Step 1~8（绝对容量法）" if route == "ABSOLUTE"
            else "数据不足，建议补充累计充/放电量寄存器字段后重跑（L2 群体排序代理）"
        ),
    }

    if dry_run:
        print("[step0][dry-run]", json.dumps(report, ensure_ascii=False, indent=2))
        return report

    with open(os.path.join(out_dir, "diagnosis_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    r_tbl.write_csv(os.path.join(out_dir, "r_stats.csv"))

    gen = {
        "chemistry": chem["chemistry"],
        "series": chem["series"],
        "R_MODE": r_mode,
        "route": route,
        "v_full": round(chem["series"] * chem["cell_v_full"], 2),
        "v_cutoff": round(chem["series"] * chem["cell_v_cutoff"], 2),
        "reversed_devices": sign["reversed_devices"],
        "current_effectively_missing": czero["current_effectively_missing"],
    }
    with open(os.path.join(out_dir, "config.generated.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(gen, f, allow_unicode=True, sort_keys=False)
    # 同步一份到 config/，供后续 Step 加载
    with open(os.path.join(PROJECT_ROOT, "config", "config.generated.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(gen, f, allow_unicode=True, sort_keys=False)

    valid = c_nom.with_columns(
        (pl.col("n_rows") >= 50).alias("valid"),
        ((pl.col("t1") - pl.col("t0")).dt.total_days() >= 14).alias("long_enough"),
    )
    valid.write_csv(os.path.join(out_dir, "device_list_valid.csv"))
    print(f"[step0] 完成: {chem['chemistry']}{chem['series']}S, R_MODE={r_mode}, route={route}, "
          f"设备 {valid.height}（有效 {int(valid['valid'].sum())}）")
    return report
