"""CLI 入口：python run.py --step 0-8 [--from-month YYYY-MM] [--dry-run] [--prep]

示例（方案 10.3）：
  python run.py --prep              # 预处理：过滤设备 + 分桶（首次运行前执行）
  python run.py --step 0            # 仅体检
  python run.py --step 0-8          # 完整运行
  python run.py --step 5-8 --from-month 2024-09   # 增量更新
  python run.py --step 0-8 --dry-run              # 干跑（不写盘）
"""
from __future__ import annotations

import argparse
import sys

sys.dont_write_bytecode = False

from src.config import load_config
from src.pipeline import Pipeline


def main() -> int:
    ap = argparse.ArgumentParser(description="换电电池 SOH 估算管线")
    ap.add_argument("--step", default="0-8", help="步骤范围，如 0-8 / 0 / 5-8")
    ap.add_argument("--from-month", default=None, help="增量起始月 YYYY-MM（仅作用于 Step 5）")
    ap.add_argument("--dry-run", action="store_true", help="仅输出行数变化，不写磁盘")
    ap.add_argument("--prep", action="store_true", help="预处理：过滤设备清单 + 分桶")
    ap.add_argument("--force", action="store_true", help="prep 强制重建分片")
    args = ap.parse_args()

    cfg = load_config()
    if args.prep:
        from src import prep
        prep.run(cfg, force=args.force)
        return 0

    if "-" in args.step:
        a, b = args.step.split("-")
    else:
        a = b = args.step
    from_step, to_step = int(a), int(b)
    if not (0 <= from_step <= to_step <= 8):
        ap.error("--step 需满足 0 <= from <= to <= 8")

    # 分片数据缺失时自动 prep（首次运行友好）
    from src import prep
    if not prep.is_ready(cfg):
        print("[run] 分片数据未就绪，先执行 prep ...")
        prep.run(cfg, force=False)

    Pipeline(cfg, dry_run=args.dry_run).run(from_step, to_step, args.from_month)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
