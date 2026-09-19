"""Step 2 锚点打标测试：A/B/C/D 四类锚点触发、互斥优先级、soc_anchor 取值。"""
from __future__ import annotations

import polars as pl
import pytest

from conftest import chain_to_anchors, make_raw


@pytest.fixture(scope="module")
def anchors(cfg):
    return chain_to_anchors(make_raw(n_devices=2, n_cycles=4), cfg)


def test_all_four_anchor_types_present(anchors):
    kinds = set(anchors["anchor_type"].unique().to_list())
    assert {"FULL_END", "SWAP_FULL", "EMPTY_END", "OCV"} <= kinds


def test_anchor_exclusive_priority(anchors):
    """B 行不应同时是 A/C/D；SWAP_FULL 的 soc_anchor 恒为 1.0。"""
    b = anchors.filter(pl.col("anchor_type") == "SWAP_FULL")
    assert b.height > 0
    assert (b["soc_anchor"] == 1.0).all()
    # 冲突标记与多命中一致
    conflict = anchors.filter(pl.col("flags").list.contains("anchor_conflict"))
    assert conflict.height == 0 or (
        conflict["anchor_type"] != "NONE"
    ).all()


def test_soc_anchor_range(anchors):
    a = anchors.filter(pl.col("anchor_type") != "NONE")
    assert (a["soc_anchor"] >= 0.02).all()
    assert (a["soc_anchor"] <= 1.0).all()


def test_empty_end_soc_anchor_low(anchors):
    c = anchors.filter(pl.col("anchor_type") == "EMPTY_END")
    assert c.height > 0
    assert (c["soc_anchor"] <= 0.10).all()


def test_ocv_anchor_uses_lookup(anchors, cfg):
    """OCV 锚点：soc_anchor 应接近电压查表值而非钳位值。"""
    d = anchors.filter(pl.col("anchor_type") == "OCV")
    assert d.height > 0
    import numpy as np

    soc_axis = cfg.ocv_table()["soc"].to_numpy()
    ocv_axis = cfg.ocv_table()["ocv_v"].to_numpy()
    v_cell = d["电压"].to_numpy() / cfg.series
    expect = np.interp(v_cell, ocv_axis, soc_axis)
    got = d["soc_anchor"].to_numpy()
    assert np.allclose(np.clip(expect, 0.02, 0.98), got, atol=1e-9)


def test_none_rows_have_no_soc_anchor(anchors):
    n = anchors.filter(pl.col("anchor_type") == "NONE")
    assert n["soc_anchor"].is_nan().all()


def test_output_columns(anchors):
    for col in ["anchor_type", "soc_anchor", "flags", "电池id", "更新时间"]:
        assert col in anchors.columns
