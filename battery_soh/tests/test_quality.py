"""Step 4 异常检测 + 质量评分 + 分级测试。"""
from __future__ import annotations

import polars as pl


def test_scored_columns(scored_df):
    for col in ["quality_score", "cap_tier", "stat_weight",
                "f_out_of_range", "f_voltage_inconsistent", "f_temp_out",
                "f_anchor_conflict", "f_sensor_drift", "f_contradiction"]:
        assert col in scored_df.columns, col


def test_tier_values(scored_df):
    assert set(scored_df["cap_tier"].unique().to_list()) <= {1, 2, 3}


def test_tier1_requires_no_severe(scored_df):
    sev = (
        scored_df["f_out_of_range"] | scored_df["f_voltage_inconsistent"]
        | scored_df["f_anchor_conflict"] | scored_df["f_contradiction"]
    )
    assert not (sev & (scored_df["cap_tier"] == 1)).any()


def test_tier3_zero_weight(scored_df):
    t3 = scored_df.filter(pl.col("cap_tier") == 3)
    assert (t3["stat_weight"] == 0.0).all()


def test_tier2_weight_discount(scored_df):
    """Tier-2 权重 = 锚点基础权重 × tier2_weight(0.3)。"""
    q = scored_df.filter(pl.col("cap_tier") == 2)
    import numpy as np

    wmap = {"SWAP_FULL": 1.0, "EMPTY_END": 1.0, "FULL_END": 0.8, "OCV": 0.5}
    expect = np.array([wmap[t] for t in q["anchor_type"]]) * 0.3
    assert np.allclose(q["stat_weight"].to_numpy(), expect)


def test_clean_swap_full_is_tier1(scored_df):
    """合成数据无异常：SWAP_FULL（基础分 20 + 温度正常 5 + 容量正常 10 + 电压正常 10 = 45）
    且相邻锚点 1h 内 → 闭环支撑 → Tier-1。"""
    b = scored_df.filter(pl.col("anchor_type") == "SWAP_FULL")
    assert b.height > 0
    assert (b["cap_tier"] == 1).all()
    assert (b["quality_score"] >= 30).all()


def test_injected_contradiction(cfg):
    """人为注入容量跳变 → f_contradiction 标记 + 降为 Tier-3。"""
    from conftest import chain_to_anchors, make_raw
    from src import step3_capacity, step4_quality

    anchors = chain_to_anchors(make_raw(n_devices=1, n_cycles=4), cfg)
    c_nom = pl.DataFrame(
        {"渠道号": ["TEST"], "电池id": [11000001], "c_nom_bms": [40.0]}
    )
    cap = step3_capacity.run(anchors, c_nom, cfg)
    # 把中间一个 SWAP_FULL 锚点容量翻倍
    idx = cap.select(pl.col("anchor_type") == "SWAP_FULL").to_series().to_numpy().nonzero()[0]
    assert len(idx) >= 2
    cap = cap.with_row_index("_i")
    cap = cap.with_columns(
        pl.when(pl.col("_i") == idx[1])
        .then(pl.col("capacity_corrected") * 2)
        .otherwise(pl.col("capacity_corrected"))
        .alias("capacity_corrected")
    ).drop("_i")
    scored = step4_quality.run(cap, cfg)
    assert scored["f_contradiction"].any()
    # 矛盾点被标记严重异常 → 不可能 Tier-1
    bad = scored.filter(pl.col("f_contradiction"))
    assert not (bad["cap_tier"] == 1).any()
