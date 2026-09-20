"""数据准备（一次性）：合并 data/ 下全部源 parquet → 按 device_filter 过滤设备
→ 按 电池id % N_PARTS 分桶（同一设备必落同一桶）→ 流式追加写 data/filtered/part_XX.parquet。

目的：Map 阶段以"桶"为处理与断点续传单元，避免每批全表扫描（源数据约 4.4 亿行）。
型号列（电池型号）不在 prep 中附加（避免流式 join OOM），由管线在批级 join（1 万行映射表，开销可忽略）。
跨文件重叠与同(设备,时间)重复行由 iter_device_batches 统一去重。
"""
from __future__ import annotations

import glob
import os

import polars as pl
import pyarrow.parquet as pq

from .config import PROJECT_ROOT, Config

N_PARTS = 32


def part_paths(out_dir: str) -> list[str]:
    return [os.path.join(out_dir, f"part_{i:02d}.parquet") for i in range(N_PARTS)]


def is_ready(cfg: Config) -> bool:
    out_dir = cfg.data_path
    return os.path.isdir(out_dir) and all(os.path.exists(p) for p in part_paths(out_dir))


def run(cfg: Config, force: bool = False) -> str:
    out_dir = cfg.data_path
    if is_ready(cfg) and not force:
        print(f"[prep] 已就绪 {N_PARTS} 个分片：{out_dir}（force=True 重建）")
        return out_dir

    src = cfg.get("source_path") or os.path.join(PROJECT_ROOT, "data")
    if not os.path.isabs(src):
        src = os.path.normpath(os.path.join(PROJECT_ROOT, src))
    files = sorted(
        f for f in glob.glob(os.path.join(src, "*.parquet"))
        if os.path.dirname(f) != os.path.normpath(out_dir)
    )
    if not files:
        raise FileNotFoundError(f"源数据目录无 parquet：{src}")
    dm_path = cfg.device_filter_path
    if dm_path is None:
        raise FileNotFoundError("prep 需要 config.device_filter（电池id+电池型号清单）")
    devs = pl.read_csv(dm_path)["电池id"].to_list()
    print(f"[prep] 源文件 {len(files)} 个，目标设备 {len(devs)} 台", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    for p in part_paths(out_dir):
        if os.path.exists(p):
            os.remove(p)

    writers: dict[int, pq.ParquetWriter] = {}
    counts: dict[int, int] = {}

    def _cb(batch: pl.DataFrame) -> None:
        for sub in batch.partition_by("_part", maintain_order=False):
            key = int(sub["_part"][0])
            sub = sub.drop("_part")
            tbl = sub.to_arrow()
            if key not in writers:
                writers[key] = pq.ParquetWriter(part_paths(out_dir)[key], tbl.schema)
            writers[key].write_table(tbl)
            counts[key] = counts.get(key, 0) + sub.height

    for fi, f in enumerate(files):
        (
            pl.scan_parquet(f)
            .filter(pl.col("电池id").is_in(devs))
            .with_columns((pl.col("电池id") % N_PARTS).alias("_part"))
            .sink_batches(_cb, chunk_size=1_000_000, maintain_order=False)
        )
        print(f"[prep] 源 {fi + 1}/{len(files)} 完成：{os.path.basename(f)}，"
              f"累计 {sum(counts.values())} 行", flush=True)

    for w in writers.values():
        w.close()
    total = sum(counts.values())
    print(f"[prep] 完成：{len(writers)} 个分片，共 {total} 行 → {out_dir}")
    return out_dir
