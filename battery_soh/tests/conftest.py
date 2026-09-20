"""测试夹具：合成小规模 BMS 数据 + 配置加载。

合成数据约定（NMC 21S 体系，与 config/config.generated.yaml 一致）：
  静置电压 = OCV(soc) * 21；充电过压 +3V；放电压降 -1V。
  每循环 = 30 步充电(1h/步) + 30 步放电 + 40 步静置，可触发 A/B/C/D 四类锚点。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import CONFIG_DIR, load_config  # noqa: E402

SERIES = 21
_C = np.genfromtxt(
    os.path.join(CONFIG_DIR, "ocv_tables", "nmc_21s.csv"), delimiter=",", skip_header=1
)


def ocv_v(soc: float) -> float:
    """单体 OCV 反查（测试端与 src 端同源查表）。"""
    return float(np.interp(soc, _C[:, 0], _C[:, 1]))


@pytest.fixture(scope="session")
def cfg(tmp_path_factory):
    """固定 21S 体系（与合成数据一致），避免真实数据 Step 0 覆写 generated 配置干扰测试。"""
    import yaml

    g = tmp_path_factory.mktemp("gen") / "config.generated.yaml"
    g.write_text(
        yaml.safe_dump(
            {"chemistry": "NMC", "series": 21, "R_MODE": "DYNAMIC", "route": "ABSOLUTE",
             "v_full": 88.2, "v_cutoff": 63.0,
             "reversed_devices": [], "current_effectively_missing": False},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return load_config(generated_path=str(g))


def make_raw(n_devices: int = 3, n_cycles: int = 12, seed: int = 0) -> pl.DataFrame:
    """合成 BMS 数据：每循环 = 50 步充电 + 20 步满充后静置 + 50 步放电 + 40 步放空后静置。

    可触发：A 满充末端（涓流段）、B 站内满充完成（充电→静置跳变）、
    C 放空末端（放到 0.02）、D 静置 OCV（>=2h 静置）。
    """
    rng = np.random.default_rng(seed)
    rows = []
    t0 = np.datetime64("2024-03-01T00:00:00")
    k = 0
    for d in range(n_devices):
        dev = 11000001 + d
        c_nom = 40.0 - 0.5 * d          # 设备间容量差异
        fade = 0.0                      # 逐循环衰减 0.05Ah
        soc = 0.02
        for cyc in range(n_cycles):
            for phase in range(160):
                t = t0 + np.timedelta64(k, "m")
                k += 60
                if phase < 50:          # 充电 0.02→1.0
                    soc = min(1.0, soc + 0.02)
                    # 满前涓流（触发 A）；soc=1.0 恢复大电流，
                    # 充电→静置跳变触发 B（要求 i_prev > 5A）
                    i = 2.0 if 0.95 < soc < 1.0 else 20.0
                    v = (ocv_v(soc) + 3.0 / SERIES) * SERIES   # 充电过压
                elif phase < 70:        # 满充后静置 20h（B + D）
                    i = 0.0
                    v = ocv_v(soc) * SERIES
                elif phase < 120:       # 放电 1.0→0.02（C）
                    soc = max(0.02, soc - 0.02)
                    i = -15.0
                    v = (ocv_v(soc) - 1.0 / SERIES) * SERIES    # 放电压降
                else:                   # 放空后静置 40h（D）
                    i = 0.0
                    v = ocv_v(soc) * SERIES
                cap = (c_nom - fade) * soc
                rows.append(
                    (
                        "bms_test", dev, "TEST", float(cyc),
                        round(cap, 3), round(soc, 2), round(float(v) + rng.normal(0, 0.02), 2),
                        float(i), 25.0, t.astype("datetime64[ns]").astype(object),
                    )
                )
            fade += 0.05
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


def chain_to_anchors(raw: pl.DataFrame, cfg) -> pl.DataFrame:
    """Step1 → Step2（锚点打标）。"""
    from src import step1_clean, step2_anchor

    return step2_anchor.run(step1_clean.run(raw, cfg), cfg)


def chain_to_scored(raw: pl.DataFrame, cfg) -> pl.DataFrame:
    """Step1 → Step4（含容量与质量分级）。"""
    from src import step3_capacity, step4_quality

    anchors = chain_to_anchors(raw, cfg)
    c_nom = pl.DataFrame(
        {
            "渠道号": ["TEST"] * 3,
            "电池id": [11000001 + d for d in range(3)],
            "c_nom_bms": [40.0 - 0.5 * d for d in range(3)],
        }
    )
    return step4_quality.run(step3_capacity.run(anchors, c_nom, cfg), cfg)


@pytest.fixture(scope="session")
def raw_df():
    return make_raw()


@pytest.fixture(scope="session")
def anchors_df(raw_df, cfg):
    return chain_to_anchors(raw_df, cfg)


@pytest.fixture(scope="session")
def scored_df(raw_df, cfg):
    return chain_to_scored(raw_df, cfg)
