"""按电池型号分类的 SOH 趋势报告（device_model.csv 的 电池型号 维度）。

输入：output/soh/soh_analysis.parquet（设备×月 SOH）+ config.device_filter（id→型号映射）
输出（多型号目录结构）：
  output/reports/总体/model_trend.csv        型号×月 中位 SOH / P10 / P90 / 设备数
  output/reports/总体/model_decay_stats.csv  型号级衰退速率分布（中位/四分位）
  output/reports/总体/model_soh_*.png        跨型号对比图（日历月/月龄/循环）
  output/reports/<型号>/*.png|csv            单型号趋势图 + 型号级 CSV
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl

from .config import OUTPUT_DIR, Config

C_NOM_SPEC_FALLBACK = 37.0

# 型号配色（固定顺序，跨图一致；未知型号按调色板轮询）
_MODEL_ORDER = ["四美7237", "海池7237", "半固态6054", "海池4840", "海池4846", "鼎山7240"]
_PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b",
            "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def model_color(model: str) -> str:
    """型号 → 颜色。已知型号用固定色，未知型号轮询调色板。"""
    try:
        return _PALETTE[_MODEL_ORDER.index(str(model))]
    except ValueError:
        return _PALETTE[abs(hash(str(model))) % len(_PALETTE)]


def _cap_expr(df: pl.DataFrame, c_nom_spec: float):
    """capacity_ah 聚合表达式：优先逐设备分母列 c_nom_soh（逐型号标称），
    否则回退全局标称 c_nom_spec。"""
    if "c_nom_soh" in df.columns:
        return pl.col("soh").median() * pl.col("c_nom_soh").median()
    return pl.col("soh").median() * c_nom_spec


def model_trend(analysis: pl.DataFrame, model_map: pl.DataFrame, c_nom_spec: float) -> pl.DataFrame:
    """型号×月 聚合：SOH 分位数 + 设备数 + 容量（SOH×标称）+ 累计循环次数。"""
    df = analysis.join(model_map, on="电池id", how="inner")
    aggs = [
        pl.col("soh").median().alias("soh_median"),
        pl.col("soh").quantile(0.10).alias("soh_p10"),
        pl.col("soh").quantile(0.90).alias("soh_p90"),
        pl.col("电池id").n_unique().alias("device_count"),
        _cap_expr(df, c_nom_spec).alias("capacity_ah"),
    ]
    if "loop_median" in df.columns:  # 该月全部设备累计循环次数（BMS 值）
        aggs += [pl.col("loop_median").median().alias("loop_median"),
                 pl.col("loop_max").max().alias("loop_max")]
    return df.group_by(["电池型号", "ym"]).agg(aggs).sort(["电池型号", "ym"])


def model_decay(analysis: pl.DataFrame, model_map: pl.DataFrame) -> pl.DataFrame:
    """型号级衰退速率分布（设备级 %/年 的中位与四分位）。

    soh_start / soh_latest：先取每设备首月 / 末月 SOH，再对型号取中位数
    （避免个别离群设备或乱序 last() 污染型号级统计）。
    """
    df = analysis.join(model_map, on="电池id", how="inner").filter(
        pl.col("decay_rate_pct_year").is_not_null()
    )
    dev = (
        df.sort(["电池id", "ym"])
        .group_by(["电池型号", "电池id"])
        .agg(
            pl.col("soh").first().alias("soh_first_dev"),
            pl.col("soh").last().alias("soh_last_dev"),
        )
    )
    ends = dev.group_by("电池型号").agg(
        pl.col("soh_first_dev").median().alias("soh_start"),
        pl.col("soh_last_dev").median().alias("soh_latest"),
    )
    return (
        df.group_by("电池型号")
        .agg(
            pl.col("decay_rate_pct_year").median().alias("decay_pct_year_median"),
            pl.col("decay_rate_pct_year").quantile(0.25).alias("decay_pct_year_p25"),
            pl.col("decay_rate_pct_year").quantile(0.75).alias("decay_pct_year_p75"),
            pl.col("电池id").n_unique().alias("device_count"),
        )
        .join(ends, on="电池型号")
        .sort("电池型号")
    )


def model_age_trend(analysis: pl.DataFrame, model_map: pl.DataFrame, c_nom_spec: float) -> pl.DataFrame:
    """型号×月龄 聚合：每设备以自身首条数据月为 0 月龄，消除出厂/上线时间差。

    age_month = 该设备当前月 - 该设备最早月（月索引差）。
    """
    df = analysis.join(model_map, on="电池id", how="inner").with_columns(
        (pl.col("ym").str.slice(0, 4).cast(pl.Int64) * 12
         + pl.col("ym").str.slice(5, 2).cast(pl.Int64)).alias("_mi")
    )
    df = df.with_columns(
        (pl.col("_mi") - pl.col("_mi").min().over("电池id")).alias("age_month"),
        pl.col("_mi").min().over("电池id").alias("birth_mi"),
    )
    aggs = [
        pl.col("soh").median().alias("soh_median"),
        pl.col("soh").quantile(0.10).alias("soh_p10"),
        pl.col("soh").quantile(0.90).alias("soh_p90"),
        pl.col("电池id").n_unique().alias("device_count"),
        _cap_expr(df, c_nom_spec).alias("capacity_ah"),
        # 该月龄组设备的出厂年月（首条记录月）与循环次数分布（P25~P75），供横轴标注
        pl.col("birth_mi").quantile(0.25).alias("birth_p25"),
        pl.col("birth_mi").quantile(0.75).alias("birth_p75"),
    ]
    if "loop_median" in df.columns:  # 该月龄全部设备累计循环次数中位
        aggs += [pl.col("loop_median").median().alias("loop_median"),
                 pl.col("loop_median").quantile(0.25).alias("loop_p25"),
                 pl.col("loop_median").quantile(0.75).alias("loop_p75")]
    return df.group_by(["电池型号", "age_month"]).agg(aggs).sort(["电池型号", "age_month"])


def model_loop_trend(analysis: pl.DataFrame, model_map: pl.DataFrame,
                     c_nom_spec: float, bin_size: int = 25) -> pl.DataFrame:
    """型号×循环次数区间 聚合：SOH 随累计循环次数的衰退曲线（bin 宽默认 25 次）。"""
    df = analysis.join(model_map, on="电池id", how="inner").filter(
        pl.col("loop_median").is_not_null()
    ).with_columns(
        (pl.col("ym").str.slice(0, 4).cast(pl.Int64) * 12
         + pl.col("ym").str.slice(5, 2).cast(pl.Int64)).alias("_mi")
    ).with_columns(
        (pl.col("_mi") - pl.col("_mi").min().over("电池id")).alias("age_month"),
        pl.col("_mi").min().over("电池id").alias("birth_mi"),
    )
    df = df.with_columns(
        (pl.col("loop_median") / bin_size).floor().cast(pl.Int64).alias("loop_bin")
    )
    return (
        df.group_by(["电池型号", "loop_bin"])
        .agg(
            pl.col("soh").median().alias("soh_median"),
            pl.col("soh").quantile(0.10).alias("soh_p10"),
            pl.col("soh").quantile(0.90).alias("soh_p90"),
            pl.col("电池id").n_unique().alias("device_count"),
            _cap_expr(df, c_nom_spec).alias("capacity_ah"),
            # 该循环档设备的月龄与出厂年月（首条记录月）分布（P25~P75），供横轴标注
            pl.col("age_month").quantile(0.25).alias("age_p25"),
            pl.col("age_month").quantile(0.75).alias("age_p75"),
            pl.col("birth_mi").quantile(0.25).alias("birth_p25"),
            pl.col("birth_mi").quantile(0.75).alias("birth_p75"),
        )
        .with_columns(
            (pl.col("loop_bin") * bin_size).alias("loop_from"),
            ((pl.col("loop_bin") + 1) * bin_size).alias("loop_to"),
        )
        .sort(["电池型号", "loop_bin"])
    )


def model_stage_loops(analysis: pl.DataFrame, model_map: pl.DataFrame,
                      retirement: pl.DataFrame) -> pl.DataFrame:
    """型号×健康阶段 的循环次数统计：每设备按其最新 health_level 归组，
    取该设备末月累计循环次数（loop_max），给出各阶段的循环消耗画像。"""
    dev = (
        analysis.join(model_map, on="电池id", how="inner")
        .sort(["电池id", "ym"])
        .group_by(["渠道号", "电池型号", "电池id"])
        .agg(pl.col("loop_max").last().alias("loops_end"),
             pl.col("soh").last().alias("soh_end"))
    )
    dev = dev.join(retirement.select(["渠道号", "电池id", "health_level"]),
                   on=["渠道号", "电池id"], how="inner")
    order = {"健康": 0, "关注": 1, "预警": 2, "退役": 3, "unknown": 4}
    out = (
        dev.group_by(["电池型号", "health_level"])
        .agg(
            pl.len().alias("device_count"),
            pl.col("loops_end").median().alias("loops_median"),
            pl.col("loops_end").quantile(0.1).alias("loops_p10"),
            pl.col("loops_end").quantile(0.9).alias("loops_p90"),
            pl.col("soh_end").median().alias("soh_median"),
        )
        .with_columns(pl.col("health_level").replace_strict(order, default=9).alias("_o"))
        .sort(["电池型号", "_o"])
        .drop("_o")
    )
    return out


def _add_caption(fig, text: str, y: float = 0.225) -> None:
    """在图底部空白区写多行说明文字（配合 tight_layout(rect=...) 预留空间）。"""
    fig.text(0.012, y, text, ha="left", va="top", fontsize=8, linespacing=1.6,
             color="#333333")


# ---- 各图底部说明文字（PNG 内嵌 + PDF 文字块共用同一来源）----

def caption_trend() -> str:
    return (
        "【图说明】横轴为自然年月：每个点统计“该月在线且有锚点数据的全部设备”的 SOH 分布，是全部设备在某一时点的健康快照，\n"
        "反映“当前这批在役电池整体状态如何”，适合做运维监控与退役规划。\n"
        "实线=当月在线设备 SOH 中位数，阴影带=P10~P90 区间；虚线（右轴）=当月参与统计的设备数，每季度首月标注数值；\n"
        "顶部横轴=BMS 累计循环次数（每 100 次一刻度，颜色同型号）。解读注意：各电池出厂/上线时间不同，每月设备群体本身\n"
        "在变化（新电池上线会抬高快照），本图各型号的“最新 SOH 高低”混入了机龄构成差异，公平对比请看“SOH-月龄曲线”。"
    )


def caption_single_trend(model: str, desc: str = "", c_nom: float = 37.0) -> str:
    return (
        f"【图说明】{model}（{desc or '铁锂 24S'}，标称 {c_nom:g}Ah）。横轴为自然年月：每个点统计“该月在线且有锚点数据的全部该型号设备”\n"
        "的 SOH 分布，是全部设备在某一时点的健康快照，反映“当前这批在役电池整体状态如何”，适合做运维监控与退役规划。\n"
        "实线=当月在线设备 SOH 中位数，阴影带=P10~P90 区间；虚线（右轴）=当月参与统计的设备数，每季度首月标注数值；\n"
        "顶部横轴=BMS 累计循环次数（每 100 次一刻度）；点状竖线=中位 SOH 跌破 80% 处。解读注意：各电池出厂/上线时间不同，\n"
        "每月设备群体本身在变化（新电池上线会抬高快照），看自身衰退规律请配合“SOH-月龄曲线”同起点对齐口径。"
    )


def caption_age(c_nom: float = 37.0) -> str:
    return (
        "【图说明】横轴为“上线月龄”：每台电池以自身出现首条数据的月份为第 0 个月，所有设备对齐到同一起点后按月龄分组统计，\n"
        "消除了各电池出厂/上线时间不同带来的机龄差异，是各型号公平对比衰退快慢的口径。\n"
        "实线=该月龄所有设备 SOH 中位数，阴影带=P10~P90 区间（90% 设备落在带内）；虚线（右轴）=该月龄参与统计的设备数；\n"
        "点状竖线+标签=中位 SOH 跌破 80% 的月龄。循环次数维度请看独立的“SOH-循环次数曲线”图（两轴交点不同属正常：\n"
        "日历老化+分组构成差异）。横轴刻度下每型号两行：该组设备累计循环次数 P25~P75 ／ 出厂年月（首条记录月）P25~P75。\n"
        f"SOH 以各型号规格书标称容量为分母（绝对口径）；已剔除疑似旧电池换BMS设备（整段低容量），初期月份异常的设备仅剔除异常前导月（见 excluded_devices.csv）。\n"
        "解读注意：月龄越大剩余设备越少，尾部曲线由少数长寿命设备构成且区间变宽，代表性下降，建议以设备数较多的区段为准。"
    )


def caption_single_age(model: str, desc: str = "", c_nom: float = 37.0) -> str:
    return (
        f"【图说明】{model}（{desc or '铁锂 24S'}，标称 {c_nom:g}Ah）。横轴为“上线月龄”：每台电池以自身出现首条数据的月份为第 0 个月，\n"
        "所有设备对齐到同一起点后按月龄分组统计，消除了各电池出厂/上线时间不同带来的机龄差异，\n"
        "反映该型号自身随使用时间的衰退规律。实线=该月龄所有设备 SOH 中位数，阴影带=P10~P90 区间（90% 设备落在带内）；\n"
        "虚线（右轴）=该月龄参与统计的设备数；点状竖线=中位 SOH 跌破 80% 的月龄。循环次数维度请看独立的“SOH-循环次数曲线”图。\n"
        "横轴刻度下两行：该月龄组设备累计循环次数 P25~P75 ／ 出厂年月（首条记录月）P25~P75。\n"
        f"SOH 以规格书标称 {c_nom:g}Ah 为分母（绝对口径）；已剔除疑似换BMS设备，初期月份异常仅删异常前导月。月龄越大剩余设备越少，尾部代表性下降，建议以设备数较多区段为准。"
    )


def caption_loop() -> str:
    return (
        "【图说明】横轴为 BMS 上报的累计循环次数（每 25 次为一档，取档中值）：把每台设备每月的 SOH 按其当时的累计循环\n"
        "归档后取中位数。与“月龄曲线”按时间对齐不同，本图按“用了多少循环”对齐，反映同等循环消耗下的容量保持能力，\n"
        "可剥离使用强度（换电频次）差异。实线=SOH 中位数，阴影带=P10~P90；虚线（右轴）=落入该循环档的设备数；\n"
        "点状竖线+标签=中位 SOH 跌破 80% 的循环次数。时间维度请看独立的“SOH-月龄曲线”图（两轴 80% 交点不同属正常）。\n"
        "横轴刻度下每型号两行：该循环档设备月龄 P25~P75 ／ 出厂年月（首条记录月）P25~P75。\n"
        "解读注意：高循环档样本少（虚线低），且能达到高循环的设备本身使用强度高，尾部曲线代表性下降。"
    )


def caption_single_loop(model: str, desc: str = "", c_nom: float = 37.0) -> str:
    return (
        f"【图说明】{model}（{desc or '铁锂 24S'}，标称 {c_nom:g}Ah）。横轴为 BMS 上报的累计循环次数（每 25 次为一档，取档中值）：\n"
        "把每台设备每月的 SOH 按其当时的累计循环归档后取中位数。与“月龄曲线”按时间对齐不同，本图按“用了多少循环”\n"
        "对齐，反映同等循环消耗下的容量保持能力，可剥离使用强度（换电频次）差异。\n"
        "实线=SOH 中位数，阴影带=P10~P90；虚线（右轴）=落入该循环档的设备数；点状竖线=中位 SOH 跌破 80% 的循环次数。\n"
        "横轴刻度下两行：该循环档设备月龄 P25~P75 ／ 出厂年月（首条记录月）P25~P75。\n"
        "解读注意：高循环档样本少（虚线低），且能达到高循环的设备本身使用强度高，尾部曲线代表性下降。"
    )


_LOOP_COLORS = {m: model_color(m) for m in _MODEL_ORDER}


def _interp_at(x, y, target):
    """升序 x 数组中 y 首次穿越 target 处的插值 x（无穿越返回 None）。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    for i in range(1, len(x)):
        if (y[i - 1] - target) * (y[i] - target) <= 0 and y[i - 1] != y[i]:
            t = (target - y[i - 1]) / (y[i] - y[i - 1])
            return float(x[i - 1] + t * (x[i] - x[i - 1]))
    return None


def _add_loop_xaxis(ax, series: list[tuple], title: str) -> None:
    """顶部第二横轴：BMS 累计循环次数，整百（100 次）一个刻度。

    series: [(x_pts, loop_pts, color), ...]；刻度画在该型号循环曲线穿越
    100/200/... 次处的横坐标位置（与下方月龄/日历对齐），标签用同型号颜色。
    """
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    xt, xl, xcol = [], [], []
    for x_pts, loop_pts, color in series:
        if len(x_pts) == 0:
            continue
        hi = float(np.nanmax(loop_pts))
        for k in range(100, int(hi) + 1, 100):
            xc = _interp_at(x_pts, loop_pts, k)
            if xc is not None:
                xt.append(xc)
                xl.append(str(k))
                xcol.append(color)
    order = np.argsort(xt)
    xt = [xt[i] for i in order]
    xl = [xl[i] for i in order]
    xcol = [xcol[i] for i in order]
    ax_top.set_xticks(xt)
    ax_top.set_xticklabels(xl, fontsize=8)
    for lab, col in zip(ax_top.get_xticklabels(), xcol):
        lab.set_color(col)
    ax_top.set_xlabel(title, fontsize=9)
    ax_top.tick_params(axis="x", pad=2)


def _mark_soh80(ax, x, soh_pct, loops, color, fmt, below=None, above=None) -> None:
    """标注中位 SOH 首次跌破 80% 的位置：竖线 + 文字（支持两行，自动避让边界）。

    below=(x_axes, y_axes)：文字框放到图内底部指定位置，箭头指回 (xc, 80) 交点。
    above=dy_axes：文字框放到 80% 虚线上方（横坐标同交点），箭头向下指回交点；
    多型号合图用不同 dy 错开三档，防重叠。
    """
    xc = _interp_at(x, soh_pct, 80.0)
    if xc is None:
        return
    loop_at = float(np.interp(xc, np.asarray(x, dtype=float),
                              np.asarray(loops, dtype=float))) if loops is not None else None
    ax.axvline(xc, color=color, ls=":", lw=1.4, alpha=0.9)
    if above is not None:
        # 文字框放图表最上方（横坐标同交点，落在竖线顶端），颜色一致无需箭头
        from matplotlib.transforms import blended_transform_factory
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        ax.annotate(fmt(xc, loop_at), xy=(xc, above), xycoords=trans,
                    fontsize=8.5, color=color, weight="bold", ha="center", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.92))
        return
    if below is not None:
        ax.annotate(fmt(xc, loop_at), xy=(xc, 80), xycoords="data",
                    xytext=below, textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2,
                                    shrinkA=4, shrinkB=4,
                                    connectionstyle="arc3,rad=0.15"),
                    fontsize=8.5, color=color, weight="bold", ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.92))
        return
    xlim = ax.get_xlim()
    near_right = xc > xlim[0] + 0.72 * (xlim[1] - xlim[0])
    ax.annotate(fmt(xc, loop_at), xy=(xc, 80),
                xytext=(-8, 12) if near_right else (6, 12),
                textcoords="offset points", fontsize=8.5, color=color, weight="bold",
                ha="right" if near_right else "left",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.9))


def _fmt_mi(mi):
    """月索引（year*12+month）→ 'YY-MM'。"""
    if mi is None or (isinstance(mi, float) and np.isnan(mi)):
        return "?"
    mi = int(round(mi))
    return f"{mi // 12 % 100:02d}-{mi % 12:02d}"


def _set_age_dist_ticks(ax, dist: list) -> None:
    """月龄图横轴：每 2 个月一个刻度；主刻度（合图每 6、单型号每 3 个月）下加
    分布行——每型号两行（累计循环次数 P25~P75、出厂年月 P25~P75）。
    型号超过 2 个时分布行太挤，只标数值。dist: [(short, x, lp25, lp75, bp25, bp75)]"""
    if not dist:
        return
    lo = max(float(min(d[1].min() for d in dist)), 0)
    hi = float(max(d[1].max() for d in dist))
    major = 6 if len(dist) > 1 else 3
    show_rows = len(dist) <= 2
    dist = sorted(dist, key=lambda d: d[0])  # 四美行在前、海池行在后
    xt = [v for v in range(0, int(hi) + 1, 2) if v >= lo - 0.5]
    if not xt:
        return
    labs = []
    for v in xt:
        if (v % major != 0) or not show_rows:  # 次刻度只标数值
            labs.append(str(v))
            continue
        rows = []
        for short, x, lp25, lp75, bp25, bp75 in dist:
            if v < x.min() - 0.5 or v > x.max() + 0.5:
                rows.append(f"{short}—")
                rows.append(f"{short}—")
            else:
                i = int(np.argmin(np.abs(x - v)))
                rows.append(f"{short}{lp25[i]:.0f}~{lp75[i]:.0f}次")
                rows.append(f"{short}{_fmt_mi(bp25[i])}~{_fmt_mi(bp75[i])}")
        labs.append("\n".join([str(v)] + rows))
    ax.set_xticks(xt)
    ax.set_xticklabels(labs, fontsize=6)


def _set_loop_dist_ticks(ax, dist: list, cap: float | None = None) -> None:
    """循环图横轴：每 50 次一个刻度；主刻度（每 100 次）下加分布行——每型号两行
    （月龄 P25~P75、出厂年月 P25~P75），合图共四行：一二行四美、三四行海池。
    dist: [(short, x, a25, a75, bp25, bp75)]；cap：横轴显示上限（超出尾部截断）。"""
    if not dist:
        return
    lo = max(float(min(d[1].min() for d in dist)), 0)
    hi = float(max(d[1].max() for d in dist))
    if cap is not None:
        hi = min(hi, cap)
    show_rows = len(dist) <= 2
    dist = sorted(dist, key=lambda d: d[0])  # 四美行在前、海池行在后
    xt = [v for v in range(50, int(hi) + 51, 50) if v >= lo and (cap is None or v <= cap)]
    if not xt:
        return
    labs = []
    for v in xt:
        if (v % 100 != 0) or not show_rows:  # 次刻度只标数值
            labs.append(str(v))
            continue
        rows = []
        for short, x, a25, a75, bp25, bp75 in dist:
            if v < x.min() - 0.5 or v > x.max() + 0.5:
                rows.append(f"{short}—")
                rows.append(f"{short}—")
            else:
                i = int(np.argmin(np.abs(x - v)))
                rows.append(f"{short}{a25[i]:.0f}~{a75[i]:.0f}月")
                rows.append(f"{short}{_fmt_mi(bp25[i])}~{_fmt_mi(bp75[i])}")
        labs.append("\n".join([str(v)] + rows))
    ax.set_xticks(xt)
    ax.set_xticklabels(labs, fontsize=6)


def plot_age_trend(trend: pl.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(11, 8))
    ax2 = ax.twinx()  # 右轴：各月龄点参与统计的设备数
    mark_series, dist = [], []
    for (model,), sub in trend.group_by("电池型号", maintain_order=True):
        sub = sub.sort("age_month")
        c = model_color(model)
        x = sub["age_month"].to_numpy()
        ax.plot(x, sub["soh_median"] * 100, "-o", ms=3, color=c, label=f"{model} 中位")
        ax.fill_between(x, sub["soh_p10"] * 100, sub["soh_p90"] * 100, color=c, alpha=0.12,
                        label=f"{model} P10~P90")
        ax2.plot(x, sub["device_count"], "--s", ms=3, lw=1, color=c, alpha=0.55,
                 label=f"{model} 设备数")
        mark_series.append((str(model), x, sub["soh_median"].to_numpy() * 100, c))
        if "loop_p25" in sub.columns:
            dist.append((str(model)[:2], x, sub["loop_p25"].to_numpy(),
                         sub["loop_p75"].to_numpy(), sub["birth_p25"].to_numpy(),
                         sub["birth_p75"].to_numpy()))
        # 每 6 个月龄标注一次设备数，避免拥挤
        for xi, n in zip(x, sub["device_count"]):
            if int(xi) % 6 == 0:
                ax2.annotate(f"{int(n)}", (xi, n), textcoords="offset points",
                             xytext=(0, 6), fontsize=7, color=c, ha="center")
    ax.axhline(80, color="k", ls="--", lw=1, label="退役线 80%")
    cross80 = [(m, _interp_at(x, soh, 80.0), c, x, soh) for m, x, soh, c in mark_series]
    cross80 = [(m, xc, c, x, soh) for m, xc, c, x, soh in cross80 if xc is not None]
    cross80.sort(key=lambda t: t[1])
    for i, (m, xc, c, x_all, soh_all) in enumerate(cross80):  # 80% 交点：图顶部竖线上方三档
        _mark_soh80(ax, x_all, soh_all, None, c,
                    lambda xcv, la, m=m: f"{m} 80%@{xcv:.1f}月龄",
                    above=0.985 - 0.105 * (i % 3))
    _set_age_dist_ticks(ax, dist)
    ax.set_ylabel("SOH (%)")
    ax2.set_ylabel("设备数（台）")
    ax.set_xlabel("上线月龄（月）" + ("｜刻度下四行：一二行四美、三四行海池"
                  "（循环次数 P25~P75 ／ 出厂年月 P25~P75）" if len(dist) <= 2 else ""),
                  labelpad=48 if len(dist) <= 2 else 8)
    ax.set_title("分型号 SOH-月龄曲线（同起点时间轴对齐，消除出厂/上线时间差）")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.grid(alpha=0.3)
    fig.legend(h1 + h2, l1 + l2, fontsize=7.5, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 0.245), framealpha=0.9)
    _add_caption(fig, caption_age(), y=0.145)
    fig.tight_layout(rect=[0, 0.27, 1, 0.955])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_trend(trend: pl.DataFrame, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(11, 8))
    ax2 = ax.twinx()  # 右轴：各月参与统计的设备数
    loop_series = []
    for (model,), sub in trend.group_by("电池型号", maintain_order=True):
        sub = sub.sort("ym")
        x = mdates.date2num(sub["ym"].str.to_datetime(format="%Y-%m").to_numpy())
        c = model_color(model)
        ax.plot(x, sub["soh_median"] * 100, "-o", ms=3, color=c, label=f"{model} 中位")
        ax.fill_between(x, sub["soh_p10"] * 100, sub["soh_p90"] * 100, color=c, alpha=0.12,
                        label=f"{model} P10~P90")
        ax2.plot(x, sub["device_count"], "--s", ms=3, lw=1, color=c, alpha=0.55,
                 label=f"{model} 设备数")
        if "loop_median" in sub.columns:
            loop_series.append((x, sub["loop_median"].to_numpy(), c))
        yms = sub["ym"].to_list()
        for xi, n, ym in zip(x, sub["device_count"], yms):  # 每季度首月标注
            if ym[5:7] in ("01", "04", "07", "10"):
                ax2.annotate(f"{int(n)}", (xi, n), textcoords="offset points",
                             xytext=(0, 6), fontsize=7, color=c, ha="center")
    ax.axhline(80, color="k", ls="--", lw=1, label="退役线 80%")
    if loop_series:
        _add_loop_xaxis(ax, loop_series, "BMS 累计循环次数（次，颜色同型号曲线）")
    ax.set_ylabel("SOH (%)")
    ax2.set_ylabel("设备数（台）")
    ax.set_xlabel("年月")
    ax.set_title("分型号 SOH 月度趋势（锚点法，中位数与 P10~P90 区间）")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # 图例移到图表下方（避免遮挡曲线）
    ax.grid(alpha=0.3)
    fig.legend(h1 + h2, l1 + l2, fontsize=7.5, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 0.245), framealpha=0.9)
    # X 轴刻度显示为 年-月（否则显示 matplotlib 内部日期序列数）
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    _add_caption(fig, caption_trend(), y=0.145)
    fig.tight_layout(rect=[0, 0.27, 1, 0.955])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _model_color(model: str) -> str:
    return model_color(model)


def plot_single_trend(trend: pl.DataFrame, model: str, path: str,
                      desc: str = "", c_nom: float = 37.0) -> None:
    """单型号日历月趋势图（与 plot_trend 同标准，仅一个型号）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False

    c = _model_color(model)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax2 = ax.twinx()  # 右轴：各月参与统计的设备数
    sub = trend.filter(pl.col("电池型号") == model).sort("ym")
    x = mdates.date2num(sub["ym"].str.to_datetime(format="%Y-%m").to_numpy())
    ax.plot(x, sub["soh_median"] * 100, "-o", ms=3, color=c, label=f"{model} 中位")
    ax.fill_between(x, sub["soh_p10"] * 100, sub["soh_p90"] * 100, color=c, alpha=0.12,
                    label=f"{model} P10~P90")
    ax2.plot(x, sub["device_count"], "--s", ms=3, lw=1, color=c, alpha=0.55,
             label=f"{model} 设备数")
    for xi, n, ym in zip(x, sub["device_count"], sub["ym"].to_list()):  # 每季度首月标注
        if ym[5:7] in ("01", "04", "07", "10"):
            ax2.annotate(f"{int(n)}", (xi, n), textcoords="offset points",
                         xytext=(0, 6), fontsize=7, color=c, ha="center")
    ax.axhline(80, color="k", ls="--", lw=1, label="退役线 80%")
    if "loop_median" in sub.columns:
        lp = sub["loop_median"].to_numpy()
        _add_loop_xaxis(ax, [(x, lp, c)], "BMS 累计循环次数（次）")
        _mark_soh80(ax, x, sub["soh_median"].to_numpy() * 100, lp, c,
                    lambda xc, la: f"80%@{mdates.num2date(xc):%Y-%m}"
                    + (f"·{la:.0f}次" if la is not None else ""))
    ax.set_ylabel("SOH (%)")
    ax2.set_ylabel("设备数（台）")
    ax.set_xlabel("年月")
    ax.set_title(f"{model} SOH 月度趋势（锚点法，中位数与 P10~P90 区间）")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, ncol=2, loc="center left", framealpha=0.9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    _add_caption(fig, caption_single_trend(model, desc, c_nom), y=0.183)
    fig.tight_layout(rect=[0, 0.19, 1, 0.955])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_single_age(age_trend: pl.DataFrame, model: str, path: str,
                    desc: str = "", c_nom: float = 37.0) -> None:
    """单型号 SOH-月龄曲线（与 plot_age_trend 同标准，仅一个型号）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False

    c = _model_color(model)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax2 = ax.twinx()  # 右轴：各月龄点参与统计的设备数
    sub = age_trend.filter(pl.col("电池型号") == model).sort("age_month")
    x = sub["age_month"].to_numpy()
    ax.plot(x, sub["soh_median"] * 100, "-o", ms=3, color=c, label=f"{model} 中位")
    ax.fill_between(x, sub["soh_p10"] * 100, sub["soh_p90"] * 100, color=c, alpha=0.12,
                    label=f"{model} P10~P90")
    ax2.plot(x, sub["device_count"], "--s", ms=3, lw=1, color=c, alpha=0.55,
             label=f"{model} 设备数")
    for xi, n in zip(x, sub["device_count"]):  # 每 6 个月龄标注一次
        if int(xi) % 6 == 0:
            ax2.annotate(f"{int(n)}", (xi, n), textcoords="offset points",
                         xytext=(0, 6), fontsize=7, color=c, ha="center")
    ax.axhline(80, color="k", ls="--", lw=1, label="退役线 80%")
    _mark_soh80(ax, x, sub["soh_median"].to_numpy() * 100, None, c,
                lambda xc, la: f"80%@{xc:.1f}月龄")
    if "loop_p25" in sub.columns:
        _set_age_dist_ticks(ax, [(str(model)[:2], x, sub["loop_p25"].to_numpy(),
                                  sub["loop_p75"].to_numpy(), sub["birth_p25"].to_numpy(),
                                  sub["birth_p75"].to_numpy())])
    ax.set_ylabel("SOH (%)")
    ax2.set_ylabel("设备数（台）")
    ax.set_xlabel("上线月龄（月）｜刻度下两行：该组设备循环次数 P25~P75 ／ 出厂年月 P25~P75",
                  labelpad=34)
    ax.set_title(f"{model} SOH-月龄曲线（同起点时间轴对齐，消除出厂/上线时间差）")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, ncol=2, loc="lower left")
    ax.grid(alpha=0.3)
    _add_caption(fig, caption_single_age(model, desc, c_nom), y=0.183)
    fig.tight_layout(rect=[0, 0.19, 1, 0.955])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loop_trend(loop_trend: pl.DataFrame, path: str) -> None:
    """SOH-累计循环次数曲线（两型号对比，横轴=循环次数区间中值）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(11, 8))
    ax2 = ax.twinx()
    marks, dist = [], []
    for (model,), sub in loop_trend.group_by("电池型号", maintain_order=True):
        sub = sub.sort("loop_bin")
        c = model_color(model)
        x = (sub["loop_from"] + sub["loop_to"]) / 2.0
        ax.plot(x, sub["soh_median"] * 100, "-o", ms=3, color=c, label=f"{model} 中位")
        ax.fill_between(x, sub["soh_p10"] * 100, sub["soh_p90"] * 100, color=c, alpha=0.12,
                        label=f"{model} P10~P90")
        ax2.plot(x, sub["device_count"], "--s", ms=3, lw=1, color=c, alpha=0.55,
                 label=f"{model} 设备数")
        marks.append((str(model), x.to_numpy(), sub["soh_median"].to_numpy() * 100, c))
        if "age_p25" in sub.columns:
            dist.append((str(model)[:2], x.to_numpy(), sub["age_p25"].to_numpy(),
                         sub["age_p75"].to_numpy(), sub["birth_p25"].to_numpy(),
                         sub["birth_p75"].to_numpy()))
        for xi, n in zip(x, sub["device_count"]):
            if int(xi) % 100 < 12.5:  # 每 100 次循环标注一次
                ax2.annotate(f"{int(n)}", (xi, n), textcoords="offset points",
                             xytext=(0, 6), fontsize=7, color=c, ha="center")
    ax.axhline(80, color="k", ls="--", lw=1, label="退役线 80%")
    # 横轴截断：按设备数加权 98% 分位（不低于 800 次），防止个别高循环尾部压缩整体布局
    cap = None
    tot = int(loop_trend["device_count"].sum())
    if tot > 0:
        s = (loop_trend.sort("loop_to").with_columns(
            (pl.col("device_count").cum_sum() / tot).alias("_cc"))
            .filter(pl.col("_cc") <= 0.98))
        if s.height:
            cap = max(800.0, (float(s["loop_to"].max()) // 50 + 1) * 50)
            ax.set_xlim(0, cap)
    # 80% 交点说明：文字框放 80% 虚线上方（横坐标同交点），箭头向下指回交点，三档错开防重叠
    cross80 = [(m, _interp_at(x, soh, 80.0), c, x, soh) for m, x, soh, c in marks]
    cross80 = [(m, xc, c, x, soh) for m, xc, c, x, soh in cross80 if xc is not None]
    cross80.sort(key=lambda t: t[1])
    for i, (m, xc, c, x_all, soh_all) in enumerate(cross80):
        _mark_soh80(ax, x_all, soh_all, None, c,
                    lambda xcv, la, m=m: f"{m} 80%@{xcv:.0f}次",
                    above=0.985 - 0.105 * (i % 3))
    _set_loop_dist_ticks(ax, dist, cap)
    ax.set_ylabel("SOH (%)")
    ax2.set_ylabel("设备数（台）")
    ax.set_xlabel("BMS 累计循环次数（次，25 次一档）"
                  + (f"｜横轴截断至 {cap:.0f} 次（尾部少量设备未显示）" if cap else "")
                  + ("｜刻度下四行：一二行四美、三四行海池"
                     "（月龄 P25~P75 ／ 出厂年月 P25~P75）" if len(dist) <= 2 else ""),
                  labelpad=48 if len(dist) <= 2 else 8)
    ax.set_title("分型号 SOH-循环次数曲线（以累计循环为横轴，消除使用强度差异）")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.grid(alpha=0.3)
    # 图例移到图表下方（避免遮挡曲线）
    fig.legend(h1 + h2, l1 + l2, fontsize=7.5, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 0.245), framealpha=0.9)
    _add_caption(fig, caption_loop(), y=0.145)
    fig.tight_layout(rect=[0, 0.27, 1, 1])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_single_loop(loop_trend: pl.DataFrame, model: str, path: str,
                     desc: str = "", c_nom: float = 37.0) -> None:
    """单型号 SOH-循环次数曲线（与 plot_loop_trend 同标准，仅一个型号）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False

    c = _model_color(model)
    fig, ax = plt.subplots(figsize=(11, 8))
    ax2 = ax.twinx()  # 右轴：各循环档参与统计的设备数
    sub = loop_trend.filter(pl.col("电池型号") == model).sort("loop_bin")
    x = ((sub["loop_from"] + sub["loop_to"]) / 2.0).to_numpy()
    soh = sub["soh_median"].to_numpy() * 100
    ax.plot(x, soh, "-o", ms=3, color=c, label=f"{model} 中位")
    ax.fill_between(x, sub["soh_p10"] * 100, sub["soh_p90"] * 100, color=c, alpha=0.12,
                    label=f"{model} P10~P90")
    ax2.plot(x, sub["device_count"], "--s", ms=3, lw=1, color=c, alpha=0.55,
             label=f"{model} 设备数")
    for xi, n in zip(x, sub["device_count"]):  # 每 100 次循环标注一次
        if int(xi) % 100 < 12.5:
            ax2.annotate(f"{int(n)}", (xi, n), textcoords="offset points",
                         xytext=(0, 6), fontsize=7, color=c, ha="center")
    ax.axhline(80, color="k", ls="--", lw=1, label="退役线 80%")
    _mark_soh80(ax, x, soh, None, c, lambda xc, la: f"80%@{xc:.0f}次")
    if "age_p25" in sub.columns:
        _set_loop_dist_ticks(ax, [(str(model)[:2], x, sub["age_p25"].to_numpy(),
                                   sub["age_p75"].to_numpy(), sub["birth_p25"].to_numpy(),
                                   sub["birth_p75"].to_numpy())])
    ax.set_ylabel("SOH (%)")
    ax2.set_ylabel("设备数（台）")
    ax.set_xlabel("BMS 累计循环次数（次，25 次一档）｜刻度下两行：该档设备月龄 P25~P75 ／ 出厂年月 P25~P75",
                  labelpad=34)
    ax.set_title(f"{model} SOH-循环次数曲线（以累计循环为横轴，消除使用强度差异）")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, ncol=2, loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.3)
    _add_caption(fig, caption_single_loop(model, desc, c_nom), y=0.183)
    fig.tight_layout(rect=[0, 0.19, 1, 0.955])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def model_desc(model: str, cfg: Config) -> tuple[str, float]:
    """型号 → (体系描述如 '铁锂 24S', 标称容量 Ah)。缺参数回退 ('', 37)。"""
    mp = (cfg.model_params or {}).get(str(model))
    if not mp:
        return "", float(cfg.get("c_nom_spec") or C_NOM_SPEC_FALLBACK)
    chem = "铁锂" if mp.get("chemistry") == "LFP" else "三元"
    return f"{chem} {mp.get('series', '?')}S", float(mp.get("c_nom_spec", 37.0))


def run(cfg: Config) -> dict:
    root = os.path.join(OUTPUT_DIR, "reports")
    total_dir = os.path.join(root, "总体")
    os.makedirs(total_dir, exist_ok=True)
    model_map = cfg.model_map
    if model_map is None:
        print("[report_model] 未配置 device_filter 或缺少 电池型号 列，跳过")
        return {}
    c_nom_spec = float(cfg.get("c_nom_spec") or C_NOM_SPEC_FALLBACK)

    analysis = pl.read_parquet(os.path.join(OUTPUT_DIR, "soh", "soh_analysis.parquet"))
    trend = model_trend(analysis, model_map, c_nom_spec)
    decay = model_decay(analysis, model_map)
    age_trend = model_age_trend(analysis, model_map, c_nom_spec)
    trend.write_csv(os.path.join(total_dir, "model_trend.csv"))
    decay.write_csv(os.path.join(total_dir, "model_decay_stats.csv"))
    age_trend.write_csv(os.path.join(total_dir, "model_age_trend.csv"))
    # 循环次数维度（需月聚合含 loop 列）
    loop_trend = stage_loops = None
    if "loop_median" in analysis.columns:
        loop_trend = model_loop_trend(analysis, model_map, c_nom_spec)
        loop_trend.write_csv(os.path.join(total_dir, "model_loop_trend.csv"))
        ret_path = os.path.join(OUTPUT_DIR, "retirement", "retirement_forecast.parquet")
        if os.path.exists(ret_path):
            stage_loops = model_stage_loops(analysis, model_map, pl.read_parquet(ret_path))
            stage_loops.write_csv(os.path.join(total_dir, "model_stage_loops.csv"))

    models = sorted(str(m) for m in trend["电池型号"].unique().to_list())
    pngs: dict[str, dict] = {}  # model -> {trend/age/loop png 路径}
    try:
        png = os.path.join(total_dir, "model_soh_trend.png")
        plot_trend(trend, png)
        age_png = os.path.join(total_dir, "model_soh_age_trend.png")
        plot_age_trend(age_trend, age_png)
        loop_png = None
        if loop_trend is not None and loop_trend.height:
            loop_png = os.path.join(total_dir, "model_soh_loop_trend.png")
            plot_loop_trend(loop_trend, loop_png)
        for model in models:
            safe = model.replace("/", "_")
            mdir = os.path.join(root, safe)
            os.makedirs(mdir, exist_ok=True)
            desc, c_nom = model_desc(model, cfg)
            sub_trend = trend.filter(pl.col("电池型号") == model)
            sub_age = age_trend.filter(pl.col("电池型号") == model)
            sub_trend.write_csv(os.path.join(mdir, "model_trend.csv"))
            sub_age.write_csv(os.path.join(mdir, "model_age_trend.csv"))
            decay.filter(pl.col("电池型号") == model).write_csv(
                os.path.join(mdir, "model_decay_stats.csv"))
            if stage_loops is not None:
                stage_loops.filter(pl.col("电池型号") == model).write_csv(
                    os.path.join(mdir, "model_stage_loops.csv"))
            p1 = os.path.join(mdir, "model_soh_trend.png")
            p2 = os.path.join(mdir, "model_soh_age_trend.png")
            plot_single_trend(trend, model, p1, desc=desc, c_nom=c_nom)
            plot_single_age(age_trend, model, p2, desc=desc, c_nom=c_nom)
            entry = {"trend": p1, "age": p2}
            if loop_trend is not None and loop_trend.height:
                sub_loop = loop_trend.filter(pl.col("电池型号") == model)
                sub_loop.write_csv(os.path.join(mdir, "model_loop_trend.csv"))
                p4 = os.path.join(mdir, "model_soh_loop_trend.png")
                plot_single_loop(loop_trend, model, p4, desc=desc, c_nom=c_nom)
                entry["loop"] = p4
            pngs[model] = entry
    except Exception as e:  # 无显示环境等
        print(f"[report_model] 绘图跳过: {e}")
        png = age_png = loop_png = None
    print(f"[report_model] 型号趋势 {trend.height} 行（型号×月），"
          f"月龄趋势 {age_trend.height} 行（型号×月龄），型号 {len(models)} 个")
    return {"trend": trend, "decay": decay, "age_trend": age_trend,
            "loop_trend": loop_trend, "stage_loops": stage_loops,
            "png": png, "age_png": age_png, "loop_png": loop_png,
            "models": models, "model_pngs": pngs}
