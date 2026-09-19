"""分支测试：Step 0 R_MODE=CONSTANT → L2 降级路线；route!=ABSOLUTE 跳过 Step 1~8。"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest


def _constant_r_raw(n_devices: int = 2, n_rows: int = 300, seed: int = 2) -> pl.DataFrame:
    """R=剩余容量/SOC 恒定（纯共线）、SOC 只在 0.3~0.7 → 无锚点候选 → L2。"""
    rng = np.random.default_rng(seed)
    rows = []
    t0 = np.datetime64("2024-03-01T00:00:00")
    k = 0
    for d in range(n_devices):
        dev = 11000001 + d
        for _ in range(n_rows):
            t = t0 + np.timedelta64(k, "m")
            k += 60
            soc = round(float(0.3 + 0.4 * rng.random()), 4)
            rows.append(
                (
                    "bms_test", dev, "TEST", 0.0,
                    round(38.0 * soc, 3), soc, 70.0,
                    10.0 if k % 2 else -10.0, 25.0,
                    t.astype("datetime64[ns]").astype(object),
                )
            )
    return pl.DataFrame(
        rows,
        schema={
            "表名": pl.Utf8, "电池id": pl.Int64, "渠道号": pl.Utf8,
            "循环次数": pl.Float64, "剩余容量": pl.Float64, "SOC": pl.Float64,
            "电压": pl.Float64, "电流": pl.Float64, "温度": pl.Float64,
            "更新时间": pl.Datetime("ns"),
        },
        orient="row",
    )


def _dynamic_r_raw(n_devices: int = 1, n_rows: int = 300, seed: int = 3) -> pl.DataFrame:
    """R 随时间下降（BMS 容量重估）→ DYNAMIC → ABSOLUTE。"""
    rng = np.random.default_rng(seed)
    rows = []
    t0 = np.datetime64("2024-03-01T00:00:00")
    k = 0
    for d in range(n_devices):
        dev = 11000001 + d
        for j in range(n_rows):
            t = t0 + np.timedelta64(k, "m")
            k += 60
            soc = round(float(0.3 + 0.4 * rng.random()), 4)
            r = 40.0 - 0.01 * j  # 明显负斜率
            rows.append(
                (
                    "bms_test", dev, "TEST", 0.0,
                    round(r * soc, 3), soc, 70.0,
                    10.0 if k % 2 else -10.0, 25.0,
                    t.astype("datetime64[ns]").astype(object),
                )
            )
    return pl.DataFrame(
        rows,
        schema={
            "表名": pl.Utf8, "电池id": pl.Int64, "渠道号": pl.Utf8,
            "循环次数": pl.Float64, "剩余容量": pl.Float64, "SOC": pl.Float64,
            "电压": pl.Float64, "电流": pl.Float64, "温度": pl.Float64,
            "更新时间": pl.Datetime("ns"),
        },
        orient="row",
    )


def _write_parquet(df: pl.DataFrame, path) -> str:
    p = str(path / "raw.parquet")
    df.write_parquet(p)
    return p


def _branch_cfg(cfg, **gen):
    """基于会话 cfg 派生独立副本，避免污染其他测试。"""
    import copy

    from src.config import Config

    merged = copy.deepcopy(cfg._cfg)
    merged["generated"] = {**merged.get("generated", {}), **gen}
    return Config(merged)


def test_step0_constant_r_l2(cfg, tmp_path):
    from src import step0_profile

    c = _branch_cfg(cfg)
    c._cfg["data_path"] = _write_parquet(_constant_r_raw(), tmp_path)
    report = step0_profile.run(c, dry_run=True)
    assert report["R_MODE"] == "CONSTANT"
    assert report["route"] == "L2_TREND"


def test_step0_dynamic_r_absolute(cfg, tmp_path):
    from src import step0_profile

    c = _branch_cfg(cfg)
    c._cfg["data_path"] = _write_parquet(_dynamic_r_raw(), tmp_path)
    report = step0_profile.run(c, dry_run=True)
    assert report["R_MODE"] == "DYNAMIC"
    assert report["route"] == "ABSOLUTE"


def test_pipeline_skips_when_not_absolute(cfg):
    from src.pipeline import Pipeline

    p = Pipeline(_branch_cfg(cfg, route="L2_TREND"), dry_run=True)
    assert p.steps_1_4() is None


def test_config_fail_fast():
    from src.config import Config, _validate

    bad = {
        "chemistry": {
            "LFP": {"v_full": 60, "v_cutoff": 65, "alpha": 0.005},
            "NMC": {"v_full": 88, "v_cutoff": 63, "alpha": 0.0025},
        },
        "anchor": {"soc_empty": 0.1, "soc_full": 0.95, "soc_swap_full": 0.98,
                   "duration_hours": 2, "gap_hours": 6},
        "quality": {"tier1_min": 30, "tier2_min": 10,
                    "score": {"SWAP_FULL": 20, "EMPTY_END": 20,
                              "FULL_END": 15, "OCV": 10}},
    }
    with pytest.raises(ValueError, match="配置校验失败"):
        _validate(bad)
    assert Config(bad) is not None  # 构造本身不校验


def test_ocv_table_validation():
    from src.schema import SchemaError, validate_ocv_table

    ok = pl.DataFrame({"soc": [0.0, 0.25, 0.5, 0.75, 1.0], "ocv_v": [3.0, 3.3, 3.6, 3.9, 4.2]})
    validate_ocv_table(ok)
    bad = pl.DataFrame({"soc": [0.0, 0.25, 0.5, 0.75, 1.0], "ocv_v": [3.0, 3.3, 3.6, 3.5, 4.2]})
    with pytest.raises(SchemaError):
        validate_ocv_table(bad)
