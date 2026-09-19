"""IO 层：scan_parquet 流式读 / 分区写 / schema 强制 / progress.db。

部署建议（方案第八章）：Step 0~5 用 Polars 流式扫描；Step 6~8 在月聚合层运行。
"""
from __future__ import annotations

import os
import sqlite3
from typing import Iterator

import polars as pl

from .schema import RAW_SCHEMA, enforce_schema, validate_raw_schema


def scan_raw(path: str) -> pl.LazyFrame:
    """流式扫描原始 parquet（不物化），强制 schema。"""
    if os.path.isdir(path):
        lf = pl.scan_parquet(os.path.join(path, "*.parquet"))
    else:
        lf = pl.scan_parquet(path)
    validate_raw_schema(lf.collect_schema())
    return enforce_schema(lf, RAW_SCHEMA)


def iter_device_batches(
    lf: pl.LazyFrame, batch_devices: int = 50
) -> Iterator[pl.DataFrame]:
    """按设备分批物化（Map 阶段并行单元）。每批完整包含所选设备的所有行。"""
    devices = lf.select("电池id").unique().sort("电池id").collect()["电池id"].to_list()
    for i in range(0, len(devices), batch_devices):
        chunk = devices[i : i + batch_devices]
        yield lf.filter(pl.col("电池id").is_in(chunk)).collect()


def write_partitioned(
    df: pl.DataFrame, out_dir: str, by: list[str], fmt: str = "parquet"
) -> list[str]:
    """分区写：out_dir/col=val/.../part.NNN.fmt。返回写出的文件列表。"""
    os.makedirs(out_dir, exist_ok=True)
    paths: list[str] = []
    if not by:
        p = os.path.join(out_dir, "part.000." + fmt)
        _write(df, p, fmt)
        return [p]
    keys = df.select(by).unique().sort(by)
    for row in keys.iter_rows():
        sub = df
        for col, val in zip(by, row):
            sub = sub.filter(pl.col(col) == val)
        rel = os.path.join(*[f"{c}={_safe(v)}" for c, v in zip(by, row)])
        d = os.path.join(out_dir, rel)
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, "part.000." + fmt)
        _write(sub, p, fmt)
        paths.append(p)
    return paths


def _safe(v) -> str:
    s = str(v)
    for ch in '\\/:*?"<>| ':
        s = s.replace(ch, "_")
    return s


def _write(df: pl.DataFrame, path: str, fmt: str) -> None:
    if fmt == "parquet":
        df.write_parquet(path)
    elif fmt == "csv":
        df.write_csv(path)
    else:
        raise ValueError(f"未知格式 {fmt}")


def read_dir(dir_path: str, fmt: str = "parquet") -> pl.DataFrame:
    """读取分区目录（hive 风格自动推断分区列）。"""
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(dir_path)
    files = [
        os.path.join(dp, f)
        for dp, _, fs in os.walk(dir_path)
        for f in fs
        if f.endswith("." + fmt)
    ]
    if not files:
        raise FileNotFoundError(f"{dir_path} 下无 {fmt} 文件")
    if fmt == "parquet":
        return pl.scan_parquet(files).collect()
    return pl.concat([pl.read_csv(f) for f in sorted(files)])


# ------------------------------------------------------------------
# progress.db：SQLite 记录 [渠道号, 电池id, 年月] 处理进度（第八章）
# ------------------------------------------------------------------

class ProgressDB:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS progress ("
            " step TEXT, channel TEXT, device TEXT, ym TEXT,"
            " status TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP,"
            " PRIMARY KEY (step, channel, device, ym))"
        )
        self._conn.commit()

    def done(self, step: str, channel: str, device, ym: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM progress WHERE step=? AND channel=? AND device=? AND ym=? AND status='done'",
            (step, channel, str(device), ym),
        )
        return cur.fetchone() is not None

    def mark(self, step: str, channel: str, device, ym: str, status: str = "done") -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO progress (step, channel, device, ym, status)"
            " VALUES (?,?,?,?,?)",
            (step, channel, str(device), ym, status),
        )
        self._conn.commit()

    def pending(self, step: str) -> list[tuple]:
        cur = self._conn.execute(
            "SELECT DISTINCT channel, device, ym FROM progress WHERE step=? AND status!='done'",
            (step,),
        )
        return cur.fetchall()

    def close(self) -> None:
        self._conn.close()
