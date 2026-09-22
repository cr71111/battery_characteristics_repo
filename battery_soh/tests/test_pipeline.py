"""端到端管线测试（小数据，输出重定向到 tmp_path）。"""
from __future__ import annotations

import glob
import os

import polars as pl
import pytest

from conftest import make_raw


@pytest.fixture()
def cfg_env(cfg, tmp_path, monkeypatch):
    """写合成 parquet 分片 + 重定向各模块 OUTPUT_DIR。"""
    raw = make_raw(n_devices=3, n_cycles=12)
    data_dir = tmp_path / "filtered"
    data_dir.mkdir()
    raw.write_parquet(str(data_dir / "part_00.parquet"))

    import src.pipeline as pipeline
    import src.step3_capacity as step3
    import src.step5_monthly as step5
    import src.step7_trend as step7
    import src.step8_retirement as step8

    out = str(tmp_path / "output")
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", out)
    monkeypatch.setattr(pipeline, "OBS_DIR", os.path.join(out, "observation"))
    monkeypatch.setattr(pipeline, "MONTHLY_DIR", os.path.join(out, "monthly"))
    monkeypatch.setattr(step5, "OUTPUT_DIR", out)
    monkeypatch.setattr(step7, "OUTPUT_DIR", out)
    monkeypatch.setattr(step8, "OUTPUT_DIR", out)
    import src.report_model as report_model
    monkeypatch.setattr(report_model, "OUTPUT_DIR", out)
    import src.report_forecast as report_forecast
    monkeypatch.setattr(report_forecast, "OUTPUT_DIR", out)
    monkeypatch.setattr(
        step3, "load_c_nom",
        lambda c: pl.DataFrame(
            {
                "渠道号": ["TEST"] * 3,
                "电池id": [11000001 + d for d in range(3)],
                "c_nom_bms": [40.0 - 0.5 * d for d in range(3)],
            }
        ),
    )
    c = cfg
    c._cfg["data_path"] = str(data_dir)
    # 合成数据标称 40Ah：对齐 SOH 分母、关闭低 SOH 剔除策略（由真实数据验证）；
    # 禁用真实型号参数表（合成设备 id 与 device_model.csv 撞号）
    c._cfg["c_nom_spec"] = 40.0
    c._cfg["model_params_path"] = None
    c._cfg.setdefault("exclude", {})["soh_low_ratio"] = 0.001
    return c, out


def _read_parts(d: str) -> pl.DataFrame:
    files = sorted(glob.glob(os.path.join(d, "*.parquet")))
    return pl.concat([pl.read_parquet(f) for f in files])


def test_full_pipeline(cfg_env):
    c, out = cfg_env
    from src.pipeline import Pipeline

    p = Pipeline(c)
    p.run(from_step=1, to_step=8)

    obs = _read_parts(os.path.join(out, "observation", "parts"))
    monthly = pl.read_parquet(os.path.join(out, "monthly", "monthly_capacity.parquet"))
    analysis = pl.read_parquet(os.path.join(out, "soh", "soh_analysis.parquet"))
    retire = pl.read_parquet(os.path.join(out, "retirement", "retirement_forecast.parquet"))

    assert obs.height > 0
    assert (obs["anchor_type"] != "NONE").all()
    assert monthly.height > 0
    assert {"soh", "decay_rate_pct_month"} <= set(analysis.columns)
    assert retire.height == 3
    assert os.path.exists(os.path.join(out, "reports", "总体", "dashboard.html"))
    # SOH 从 ~1.0 缓降（合成 fade 0.05/循环 → 12 循环约 -1.5%）
    last = analysis.sort(["电池id", "ym"]).group_by("电池id", maintain_order=True).last()
    assert (last["soh"] > 0.9).all()
    assert (last["soh"] <= 1.0001).all()


def test_resume_skips_done(cfg_env):
    """断点续传：第二次跑 Step1~5 不新增设备。"""
    c, out = cfg_env
    from src.pipeline import Pipeline

    p = Pipeline(c)
    p.run(from_step=1, to_step=5)
    first = _read_parts(os.path.join(out, "observation", "parts")).height
    p2 = Pipeline(c)
    p2.run(from_step=1, to_step=5)
    second = _read_parts(os.path.join(out, "observation", "parts")).height
    assert first == second


def test_dry_run_writes_nothing(cfg_env):
    c, out = cfg_env
    from src.pipeline import Pipeline

    Pipeline(c, dry_run=True).run(from_step=1, to_step=5)
    assert not os.path.exists(os.path.join(out, "observation", "parts", "anchor_b0000.parquet"))
