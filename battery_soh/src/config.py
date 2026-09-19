"""配置加载：三段合并 + 启动校验（fail fast）。

优先级（后者覆盖前者）：
  1. config.yaml            人工基线（可空）
  2. thresholds.yaml        全部阈值（代码默认值不允许硬编码，P05）
  3. config.generated.yaml  Step 0 实测参数（体系/串数/V_full/V_cutoff/C_nom 等）
"""
from __future__ import annotations

import copy
import os
from typing import Any

import yaml

from .constants import AnchorType
from .schema import validate_ocv_table

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

import polars as pl

_OCV_CACHE: dict[str, pl.DataFrame] = {}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Config:
    """合并后的只读配置视图。cfg.get('anchor.soc_full') 取嵌套值。"""

    def __init__(self, cfg: dict, base_dir: str = PROJECT_ROOT):
        self._cfg = cfg
        self.base_dir = base_dir

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._cfg
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        return self._cfg[key]

    # ---------- 便捷访问 ----------

    @property
    def data_path(self) -> str:
        p = self.get("data_path") or os.path.join(self.base_dir, "data", "raw")
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(self.base_dir, p))
        return p

    @property
    def chemistry(self) -> str:
        return self.get("generated.chemistry") or self.get("chemistry") or "LFP"

    @property
    def series(self) -> int:
        return int(self.get("generated.series") or self.get("series") or 25)

    @property
    def r_mode(self) -> str:
        return self.get("generated.R_MODE", "DYNAMIC")

    @property
    def route(self) -> str:
        return self.get("generated.route", "ABSOLUTE")

    def chem_params(self, chemistry: str | None = None) -> dict:
        chem = chemistry or self.chemistry
        return self._cfg["chemistry"][chem]

    def ocv_table(self, chemistry: str | None = None, series: int | None = None) -> pl.DataFrame:
        """加载并校验 OCV 表（单体电压），按 chemistry_series 命名。"""
        chem = (chemistry or self.chemistry).lower()
        s = series or self.series
        name = f"{chem}_{s}s.csv"
        if name not in _OCV_CACHE:
            path = os.path.join(CONFIG_DIR, "ocv_tables", name)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"缺少 OCV 表: {path}。Step 0 判别体系为 {chem}/{s}S，"
                    f"请确认 config/ocv_tables/ 下有对应文件。"
                )
            df = pl.read_csv(path).sort("soc")
            validate_ocv_table(df)
            _OCV_CACHE[name] = df
        return _OCV_CACHE[name]

    def rest_current_threshold(self, c_nom: float) -> float:
        """静置判据 |I| < 0.02C（A）。"""
        return self.get("anchor.rest_c_rate") * c_nom

    def anchor_score(self, anchor_type: str) -> float:
        return float(self.get(f"quality.score.{anchor_type}", 0.0))

    def anchor_weight(self, anchor_type: str) -> float:
        w = self.get(f"quality.weight.{anchor_type}")
        if w is None:
            return 0.0
        return float(w)


def _validate(cfg: dict) -> None:
    """启动校验（fail fast）：关键阈值存在且自洽。"""
    problems: list[str] = []
    for chem in ("LFP", "NMC"):
        node = cfg.get("chemistry", {}).get(chem)
        if not node:
            problems.append(f"chemistry.{chem} 缺失")
            continue
        if node["v_full"] <= node["v_cutoff"]:
            problems.append(f"chemistry.{chem}: v_full 必须 > v_cutoff")
        if node["alpha"] <= 0:
            problems.append(f"chemistry.{chem}: alpha 必须 > 0")
    a = cfg.get("anchor", {})
    if not (0 < a.get("soc_empty", 0) < a.get("soc_full", 1) <= a.get("soc_swap_full", 1) <= 1):
        problems.append("anchor: 需满足 0 < soc_empty < soc_full <= soc_swap_full <= 1")
    if a.get("duration_hours", 0) <= 0 or a.get("gap_hours", 0) <= 0:
        problems.append("anchor: duration_hours / gap_hours 必须为正")
    q = cfg.get("quality", {})
    if q.get("tier1_min", 0) <= q.get("tier2_min", 0):
        problems.append("quality: tier1_min 必须 > tier2_min")
    scores = q.get("score", {})
    for at in AnchorType:
        if at == AnchorType.NONE:
            continue
        if at.value not in scores:
            problems.append(f"quality.score 缺少锚点 {at.value}")
    if problems:
        raise ValueError("配置校验失败:\n  - " + "\n  - ".join(problems))


def load_config(generated_path: str | None = None) -> Config:
    """三段合并加载。generated_path 默认 config/config.generated.yaml。"""
    manual = _load_yaml(os.path.join(CONFIG_DIR, "config.yaml"))
    thresholds = _load_yaml(os.path.join(CONFIG_DIR, "thresholds.yaml"))
    gpath = generated_path or os.path.join(CONFIG_DIR, "config.generated.yaml")
    generated = _load_yaml(gpath)

    merged = _deep_merge(thresholds, {k: v for k, v in manual.items() if k != "data_path"})
    merged["data_path"] = manual.get("data_path") or thresholds.get("data_path")
    merged["generated"] = generated
    _validate(merged)
    return Config(merged)
