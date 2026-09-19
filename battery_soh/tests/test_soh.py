"""Step 5/6/7/8 测试：月聚合、MAD 去毛刺、PAVA、SOH、衰退速率、退役预测。"""
from __future__ import annotations

import numpy as np
import polars as pl

from src import step5_monthly, step6_soh, step7_trend, step8_retirement


# ---------------------------------------------------------- Step 5
def test_weighted_median_basic():
    v = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    w = np.array([1.0, 1.0, 1.0, 1.0, 0.01])
    med = step5_monthly._weighted_median(v, w)
    assert med <= 4.0  # 离群大权重极小，不应拉高中位数


def test_aggregate_partitions(scored_df):
    monthly = step5_monthly.aggregate(scored_df)
    assert monthly.height > 0
    assert {"渠道号", "电池id", "year", "month", "ym", "capacity_median",
            "capacity_count", "tier1_count", "tier2_count"} <= set(monthly.columns)
    # 每设备每月一行
    assert monthly.select(["渠道号", "电池id", "ym"]).is_unique().all()
    # Tier-3 禁入统计：计数只含 Tier-1/2
    assert (monthly["tier1_count"] + monthly["tier2_count"] == monthly["capacity_count"]).all()


# ---------------------------------------------------------- Step 6
def test_mad_outlier_mask():
    vals = np.array([10.0] * 20)
    vals[10] = 30.0
    mask = step6_soh.mad_outlier_mask(vals, window=7, k=3.0)
    assert mask[10]
    assert mask.sum() == 1


def test_mad_no_false_positive():
    rng = np.random.default_rng(1)
    vals = 40 + rng.normal(0, 0.1, 50)
    mask = step6_soh.mad_outlier_mask(vals, window=7, k=3.0)
    assert mask.sum() <= 1  # 小噪声基本不误伤


def test_pava_decreasing():
    y = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    out = step6_soh.pava_decreasing(y)
    assert (np.diff(out) <= 1e-12).all()          # 非增
    assert abs(out.mean() - y.mean()) < 1e-12     # 保均值


def test_pava_weighted():
    y = np.array([1.0, 2.0])
    w = np.array([3.0, 1.0])
    out = step6_soh.pava_decreasing(y, w)
    assert np.allclose(out, [1.25, 1.25])  # 加权池化 (1*3+2*1)/4


def test_soh_run_monotonic(scored_df):
    monthly = step5_monthly.aggregate(scored_df)
    soh_tbl = step6_soh.run(monthly, _cfg_soh())
    assert {"soh", "soh_raw", "capacity_smoothed", "is_spike", "c_nom_soh"} <= set(soh_tbl.columns)
    for (ch, dev), sub in soh_tbl.group_by(["渠道号", "电池id"], maintain_order=True):
        s = sub.sort("ym")["soh"].to_numpy()
        assert (np.diff(s) <= 1e-9).all(), f"设备 {dev} SOH 非单调"
        assert (sub["c_nom_soh"] > 0).all()


def _cfg_soh():
    from src.config import load_config

    return load_config()


# ---------------------------------------------------------- Step 7
def test_decay_rate_linear():
    ym = [f"2024-{m:02d}" for m in range(1, 7)]
    soh = pl.Series([1.0 - 0.01 * i for i in range(6)])
    df = pl.DataFrame({"ym": ym, "soh": soh})
    d = step7_trend.decay_rate(df)
    assert abs(d["decay_rate_pct_month"] + 1.0) < 0.01  # -1%/月
    assert d["r_squared"] > 0.99


def test_decay_rate_insufficient():
    df = pl.DataFrame({"ym": ["2024-01", "2024-02"], "soh": [1.0, 0.99]})
    d = step7_trend.decay_rate(df)
    assert d["decay_rate_pct_month"] is None


# ---------------------------------------------------------- Step 8
def test_forecast_retire_month():
    ym = [f"2024-{m:02d}" for m in range(1, 7)]
    # -1%/月，从 1.0 起：0.95 在 6 月，跌破 0.80 在 2024-01 + 20 个月 = 2025-09
    soh = pl.Series([1.0 - 0.01 * i for i in range(6)])
    f = step8_retirement.forecast_device(soh, ym, retire=0.80, min_months=3, r2_min=0.6, z=1.96)
    assert f["predicted_retire_month"] == "2025-09"
    assert f["r_squared"] > 0.99
    assert f["retire_ci_low"] <= f["predicted_retire_month"] <= f["retire_ci_high"]


def test_forecast_insufficient():
    f = step8_retirement.forecast_device(
        pl.Series([1.0, 0.99]), ["2024-01", "2024-02"], 0.80, 3, 0.6, 1.96
    )
    assert f["note"] == "数据不足"
    assert f["predicted_retire_month"] is None


def test_forecast_no_decay():
    f = step8_retirement.forecast_device(
        pl.Series([1.0, 1.0, 1.0, 1.0]), ["2024-01", "2024-02", "2024-03", "2024-04"],
        0.80, 3, 0.6, 1.96,
    )
    # 平坦序列 slope=0 → 不外推
    assert f["predicted_retire_month"] is None
