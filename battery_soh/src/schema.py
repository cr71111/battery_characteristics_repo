"""字段名、dtype、OCV 表校验（方案第十章 工程规范）。"""
from __future__ import annotations

import polars as pl

from .constants import RAW_COLUMNS

# 原始数据 schema 约定（附录 A）
RAW_SCHEMA: dict[str, pl.DataType] = {
    "表名": pl.Utf8,
    "电池id": pl.Int64,
    "渠道号": pl.Utf8,
    "循环次数": pl.Float64,
    "剩余容量": pl.Float64,
    "SOC": pl.Float64,
    "电压": pl.Float64,
    "电流": pl.Float64,
    "温度": pl.Float64,
    "更新时间": pl.Datetime("ns"),
}

# 锚点打标输出 schema（Step 2）
ANCHOR_SCHEMA: dict[str, pl.DataType] = {
    "电池id": pl.Int64,
    "渠道号": pl.Utf8,
    "更新时间": pl.Datetime("ns"),
    "循环次数": pl.Float64,
    "剩余容量": pl.Float64,
    "SOC": pl.Float64,
    "电压": pl.Float64,
    "电流": pl.Float64,
    "温度": pl.Float64,
    "anchor_type": pl.Utf8,
    "soc_anchor": pl.Float64,
    "flags": pl.List(pl.Utf8),
}


class SchemaError(Exception):
    pass


def validate_raw_schema(schema: pl.Schema) -> None:
    """校验原始 parquet 列名与 dtype（fail fast）。"""
    missing = [c for c in RAW_COLUMNS if c not in schema]
    if missing:
        raise SchemaError(f"原始数据缺少列: {missing}")


def validate_ocv_table(df: pl.DataFrame) -> None:
    """OCV 表：列 [soc, ocv_v]，soc 单调递增且覆盖 [0,1]，ocv_v 单调不减。"""
    cols = set(df.columns)
    if not {"soc", "ocv_v"} <= cols:
        raise SchemaError(f"OCV 表必须包含 soc/ocv_v 列，实际: {sorted(cols)}")
    soc = df["soc"].to_numpy()
    ocv = df["ocv_v"].to_numpy()
    if len(soc) < 5:
        raise SchemaError("OCV 表样本过少")
    if soc[0] > 0.02 or soc[-1] < 0.98:
        raise SchemaError(f"OCV 表 SOC 范围须覆盖 [0,1]，实际 [{soc[0]}, {soc[-1]}]")
    if (soc[1:] <= soc[:-1]).any():
        raise SchemaError("OCV 表 soc 必须严格单调递增")
    if (ocv[1:] < ocv[:-1] - 1e-9).any():
        raise SchemaError("OCV 表 ocv_v 必须单调不减（OCV-SOC 物理约束）")


def enforce_schema(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """强制列 dtype（缺失列补 null），并保留 schema 之外的额外列（如 电池型号）。"""
    cols = df.collect_schema().names() if isinstance(df, pl.LazyFrame) else df.columns
    exprs = []
    for name, dtype in schema.items():
        if name in cols:
            exprs.append(pl.col(name).cast(dtype, strict=False))
        else:
            exprs.append(pl.lit(None, dtype=dtype).alias(name))
    extra = [c for c in cols if c not in schema]
    exprs = [pl.col(c) for c in extra] + exprs
    return df.select(exprs)
