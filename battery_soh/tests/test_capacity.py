"""Step 3 容量读取 + 温度补偿测试。"""
from __future__ import annotations

import polars as pl

from conftest import chain_to_anchors, make_raw


def _c_nom():
    return pl.DataFrame(
        {
            "渠道号": ["TEST", "TEST"],
            "电池id": [11000001, 11000002],
            "c_nom_bms": [40.0, 39.5],
        }
    )


def test_only_anchor_rows_kept(cfg):
    from src import step3_capacity

    anchors = chain_to_anchors(make_raw(n_devices=2, n_cycles=3), cfg)
    cap = step3_capacity.run(anchors, _c_nom(), cfg)
    assert (cap["anchor_type"] != "NONE").all()


def test_capacity_raw_equals_cap_over_soc_anchor(cfg):
    from src import step3_capacity

    anchors = chain_to_anchors(make_raw(n_devices=2, n_cycles=3), cfg)
    cap = step3_capacity.run(anchors, _c_nom(), cfg)
    calc = (cap["剩余容量"] / cap["soc_anchor"]).to_numpy()
    got = cap["capacity_raw"].to_numpy()
    import numpy as np

    assert np.allclose(calc, got, rtol=1e-9)


def test_full_anchor_capacity_near_c_nom(cfg):
    """SWAP_FULL 锚点（soc_anchor=1.0）容量应接近设备真实容量（25°C 无补偿）。"""
    from src import step3_capacity

    anchors = chain_to_anchors(make_raw(n_devices=2, n_cycles=3), cfg)
    cap = step3_capacity.run(anchors, _c_nom(), cfg)
    b = cap.filter(pl.col("anchor_type") == "SWAP_FULL")
    import numpy as np

    # 首循环 c_nom=40，fade=0；后续 fade 递增 → 容量略降，但都应在 [35, 41]
    assert ((b["capacity_corrected"] > 35) & (b["capacity_corrected"] < 41)).all()


def test_temperature_compensation_applied(cfg):
    """温度补偿：C_corrected = C_raw * (1 + alpha*(t_ref - T))，25°C 时不改变。"""
    from src import step3_capacity

    anchors = chain_to_anchors(make_raw(n_devices=2, n_cycles=3), cfg)
    cap = step3_capacity.run(anchors, _c_nom(), cfg)
    import numpy as np

    # 合成数据温度恒 25°C = t_ref → corrected == raw
    assert np.allclose(cap["capacity_raw"], cap["capacity_corrected"], rtol=1e-12)


def test_out_of_range_flag(cfg):
    from src import step3_capacity

    anchors = chain_to_anchors(make_raw(n_devices=2, n_cycles=3), cfg)
    cap = step3_capacity.run(anchors, _c_nom(), cfg)
    assert "out_of_range" in cap.columns
    assert "temp_out" in cap.columns
    assert "v_inconsistent" in cap.columns
