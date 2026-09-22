"""退役预测：未来每个自然月预计有多少台电池到达退役标准（SOH<80%），并生成 PDF 报告。

预测口径：
  1) 设备级：step8 已按各设备自身衰退速率线性外推 predicted_retire_month（R² 加权可信）；
  2) 缺失/低可信设备（predicted_retire_month 为空或 R²<0.6）：用"型号中位衰退速率"从
     最新 SOH 外推补齐（数据短、拟合差的设备向本型号整体实测规律靠拢）。
输出（多型号目录结构）：
  output/reports/总体/retire_schedule.csv        型号×月 新增/累计退役台数
  output/reports/总体/model_retire_forecast.png  跨型号对比预测图
  output/reports/<型号>/model_retire_forecast.png  单型号预测图
  output/reports/<总体|型号>/report_<...>.pdf     PDF 报告
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl

from .config import OUTPUT_DIR, Config
from .report_model import (caption_age, caption_loop, caption_single_age,
                           caption_single_loop, caption_single_trend,
                           caption_trend, model_color, model_desc)

RETIRE_SOH = 0.80
C_NOM_SPEC_FALLBACK = 37.0


def _mi_from_str(s: str | None) -> int | None:
    if not s:
        return None
    try:
        y, m = s.split("-")
        return int(y) * 12 + int(m)
    except Exception:
        return None


def _mi_add(mi: int, k: int) -> int:
    """月索引 + k 个月。"""
    return mi + k


def _mi_str(mi: int) -> str:
    """月索引（year*12+month，month 1~12）→ 'YYYY-MM'。"""
    y = (mi - 1) // 12
    m = (mi - 1) % 12 + 1
    return f"{y}-{m:02d}"


def _latest_mi(analysis: pl.DataFrame) -> int:
    """数据集中最新的自然月（预测起点）。"""
    mx = analysis["ym"].max()
    return _mi_from_str(str(mx)[:7]) or (2026 * 12 + 9)


def retirement_schedule(analysis: pl.DataFrame, model_map: pl.DataFrame,
                        retirement: pl.DataFrame,
                        today_mi: int | None = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """型号×未来自然月：预计新增退役台数与累计退役台数。

    返回 (sched, beyond)：beyond 为超出预测窗口（6 年后才跌破 80%）的台数统计。
    每台设备取退役月份 = predicted_retire_month；缺失或 R²<0.6 的设备用
    （soh_latest-0.8）/（型号中位月衰退）外推补齐。
    today_mi：预测起点月索引（默认取数据最新月）。
    """
    if today_mi is None:
        today_mi = _latest_mi(analysis)
    cap_mi = today_mi + 71  # 预测窗口上限（6 年）
    # 各设备最新 SOH（用于补齐）
    last = (analysis.sort(["电池id", "ym"])
            .group_by(["渠道号", "电池id"])
            .agg(pl.col("soh").last().alias("soh_end"),
                 pl.col("ym").last().alias("ym_end")))
    dec = (analysis.filter(pl.col("decay_rate_pct_year").is_not_null())
           .join(model_map, on="电池id", how="inner")
           .group_by("电池型号")
           .agg(pl.col("decay_rate_pct_year").median().alias("dec_med")))
    # 全体中位速率（型号无足够样本时兜底）
    dec_all = float(analysis.filter(pl.col("decay_rate_pct_year").is_not_null())
                    ["decay_rate_pct_year"].median())

    df = (retirement.join(model_map, on="电池id", how="inner")
          .join(last, on=["渠道号", "电池id"], how="left")
          .join(dec.select("电池型号", "dec_med"), on="电池型号", how="left"))

    retire_mi, valid = [], []
    for row in df.iter_rows(named=True):
        pm = _mi_from_str(row.get("predicted_retire_month"))
        r2 = row.get("r_squared")
        ok = pm is not None and r2 is not None and r2 >= 0.6
        if ok:
            retire_mi.append(pm)
            valid.append(1)
            continue
        # 外推补齐：从末观测月往后推 (soh-0.8)/月衰退
        soh_end = row.get("soh_end")
        ym_end = _mi_from_str(row.get("ym_end"))
        if soh_end is None or ym_end is None or soh_end <= RETIRE_SOH:
            retire_mi.append(ym_end if ym_end is not None else 0)
            valid.append(0)
            continue
        rate = abs(row.get("dec_med") or dec_all) / 100.0 / 12.0  # 每月衰退比例
        months = (soh_end - RETIRE_SOH) / max(rate, 1e-6)
        retire_mi.append(int(round(ym_end + months)))
        valid.append(0)

    df = df.with_columns(pl.Series("retire_mi", retire_mi, dtype=pl.Int64),
                         pl.Series("pred_valid", valid, dtype=pl.Int32))
    # 超出预测窗口（6 年之后，多为衰退极慢或 SOH 异常高的设备）不进曲线，单独统计
    beyond = (df.filter(pl.col("retire_mi") > cap_mi)
              .group_by("电池型号").agg(pl.len().alias("n_outside_window")))
    # 汇总：型号×退役月
    sched = (df.filter((pl.col("retire_mi") >= today_mi)
                       & (pl.col("retire_mi") <= cap_mi))
             .group_by(["电池型号", "retire_mi"])
             .agg(pl.len().alias("new_retire"))
             .sort(["电池型号", "retire_mi"]))
    # 展开到连续月份（起点=预测起点，终点=窗口上限）
    out_rows = []
    for (model,), sub in sched.group_by("电池型号", maintain_order=True):
        sub = sub.filter(pl.col("retire_mi") >= today_mi)
        hi = int(sub["retire_mi"].max())
        cum_by_mi = {int(r["retire_mi"]): int(r["new_retire"]) for r in sub.to_dicts()}
        cum = 0
        for mi in range(today_mi, hi + 1):
            cum += cum_by_mi.get(mi, 0)
            out_rows.append((str(model), _mi_str(mi), mi, cum_by_mi.get(mi, 0), cum))
    return pl.DataFrame(out_rows,
                        schema={"电池型号": pl.Utf8, "ym": pl.Utf8, "_mi": pl.Int64,
                                "new_retire": pl.Int64, "cum_retire": pl.Int64},
                        orient="row").sort(["电池型号", "_mi"]), beyond


def plot_retire_forecast(sched: pl.DataFrame, path: str, model: str | None = None) -> None:
    """退役预测图：柱=每月新增退役台数，线=累计退役台数（右轴）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(11, 8))
    ax2 = ax.twinx()
    title = f"{model} 未来逐月退役台数预测" if model else "分型号未来逐月退役台数预测"
    models = [model] if model else sched["电池型号"].unique().to_list()
    n = len(models)
    for j, m in enumerate(sorted(models)):
        sub = sched.filter(pl.col("电池型号") == m)
        x = mdates.date2num(sub["ym"].str.to_datetime(format="%Y-%m").to_numpy()) + (j - (n - 1) / 2) * 11
        c = model_color(m)
        ax.bar(x, sub["new_retire"], width=10, color=c, alpha=0.55,
               label=f"{m} 月新增退役")
        ax2.plot(x, sub["cum_retire"], "-o", ms=2.5, lw=1.6, color=c,
                 label=f"{m} 累计退役（右轴）")
        # 50% 里程碑
        total = int(sub["cum_retire"].max())
        if total:
            half = sub.filter(pl.col("cum_retire") >= total * 0.5)
            if half.height:
                i = int(np.argmin((sub["cum_retire"].to_numpy() - total * 0.5) ** 2))
                ax2.annotate(f"50%退役 {sub['ym'][i]}（{total // 2}台）",
                             (float(x[i]), total * 0.5),
                             xytext=(6, -14), textcoords="offset points", fontsize=8,
                             color=c, weight="bold")
    ax.set_ylabel("每月新增退役台数（台）")
    ax2.set_ylabel("累计退役台数（台）")
    ax.set_xlabel("年月")
    ax.set_title(title)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3)
    _cap = (
        "【图说明】退役标准：SOH 跌破 80%。每台设备按其自身衰退轨迹（锚点法 SOH 序列线性拟合）外推预计跌破月份；\n"
        "拟合可信度低（R²<0.6）或数据不足的设备，用型号中位衰退速率从最新 SOH 外推补齐。\n"
        "柱=当月新增到达退役标准的台数，折线（右轴）=累计台数。\n"
        "注意：预测为趋势外推，未计入温度/使用强度变化与个体故障，建议每半年用最新数据滚动修正。"
    )
    fig.text(0.012, 0.16, _cap, ha="left", va="top", fontsize=8, linespacing=1.6,
             color="#333333")
    fig.tight_layout(rect=[0, 0.17, 1, 0.955])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- PDF 报告 --

def _fleet_stats(analysis: pl.DataFrame, model_map: pl.DataFrame,
                 retirement: pl.DataFrame) -> dict:
    """报告用的设备级统计汇总。"""
    df = analysis.join(model_map, on="电池id", how="inner")
    last = df.sort("ym").group_by(["电池型号", "电池id"]).agg(
        pl.col("soh").last().alias("soh_end"),
        pl.col("ym").last().alias("ym_end"),
        pl.col("loop_max").last().alias("loops"),
    )
    out = {}
    for (m,), sub in last.group_by("电池型号", maintain_order=True):
        soh = sub["soh_end"]
        out[str(m)] = {
            "devices": sub.height,
            "first_month": sub["ym_end"].min()[:7] if sub.height else "",
            "soh_median": float(soh.median()),
            "soh_p10": float(soh.quantile(0.1)),
            "soh_p90": float(soh.quantile(0.9)),
            "below80": int((soh < RETIRE_SOH).sum()),
            "loops_median": float(sub["loops"].median() or 0),
        }
    lv = (retirement.join(model_map, on="电池id", how="inner")
          .group_by(["电池型号", "health_level"]).agg(pl.len().alias("n")))
    for r in lv.to_dicts():
        out.setdefault(r["电池型号"], {})["levels"] = out.get(
            r["电池型号"], {}).get("levels", {}) | {r["health_level"]: int(r["n"])}
    return out


def _img_b64(path: str | None) -> str:
    if not path or not os.path.exists(path):
        return ""
    import base64

    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _split_fig(path: str | None) -> str:
    """裁剪 PNG 的图表部分（去掉底部内嵌说明文字），供 PDF 只嵌入图表块。

    PDF 中的图说明以 HTML 文字呈现，不再嵌图片，避免图过高产生留白。
    拆不出说明块时返回原图。
    """
    if not path or not os.path.exists(path):
        return ""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    a = np.asarray(im.convert("L"))
    h, w = a.shape
    blank = (a > 248).all(axis=1)
    # 从 55% 高度往下找第一条 ≥14 行的连续空白带（图表区与说明文字的间隔）
    start, run = None, 0
    for i in range(int(h * 0.55), h):
        if blank[i]:
            run += 1
            if run >= 14:
                start = i - run + 1
                break
        else:
            run = 0
    if start is None or start > h - 30:
        return path
    base = os.path.splitext(path)[0]
    cp = base + "_chart.png"
    im.crop((0, 0, w, start)).save(cp)
    return cp


def _fig_html(path: str | None, cap_text: str, caption: str | None = None) -> str:
    """一节图表：图表块 + 图注 + 说明文字块（HTML 文字，非图片）。"""
    if not path or not os.path.exists(path):
        return ""
    cp = _split_fig(path)
    out = [f"<div class='figc'><img src='{_img_b64(cp)}'></div>"]
    if cap_text:
        out.append(f"<div class='figcap'>{cap_text}</div>")
    if caption:
        lines = "<br>".join(caption.splitlines())
        out.append(f"<div class='figtext'>{lines}</div>")
    return "".join(out)


_CSS = """
@page { size: A4; margin: 0; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: "Microsoft YaHei", "PingFang SC", sans-serif; color: #26334d;
       font-size: 10.5pt; line-height: 1.65; }
.page { padding: 14mm 14mm 16mm; }
.banner { background: linear-gradient(135deg, #123a6b 0%, #1f5fa8 60%, #2f8fd8 100%);
          color: #fff; padding: 26px 30px 20px; }
.banner h1 { font-size: 21pt; font-weight: 700; letter-spacing: 1px; }
.banner .meta { margin-top: 8px; font-size: 9pt; opacity: .85; }
h2 { font-size: 13.5pt; color: #123a6b; margin: 22px 0 10px; padding-left: 10px;
     border-left: 5px solid #2f8fd8; }
p { margin: 6px 0; text-align: justify; }
.note { color: #5a6b85; font-size: 9pt; }
.kpis { display: flex; gap: 10px; margin: 14px 0 4px; }
.kpi { flex: 1; background: #f2f7fd; border: 1px solid #d8e6f5; border-radius: 8px;
       padding: 10px 12px; text-align: center; }
.kpi .v { font-size: 16pt; font-weight: 700; color: #1f5fa8; }
.kpi .k { font-size: 8.5pt; color: #5a6b85; margin-top: 2px; }
table { border-collapse: collapse; width: 100%; margin: 10px 0 4px; font-size: 9pt; }
th { background: #1f5fa8; color: #fff; padding: 6px 8px; text-align: center; font-weight: 600; }
td { border-bottom: 1px solid #e3ebf5; padding: 5px 8px; text-align: center; }
tr:nth-child(even) td { background: #f6f9fd; }
.figc { margin: 12px 0 4px; page-break-inside: avoid; }
.figc.sub { margin-top: 8px; page-break-inside: avoid; }
.figc img { width: 100%; border: 1px solid #dde6f0; border-radius: 6px; }
.figcap { font-size: 9pt; color: #5a6b85; margin: 4px 0 6px; text-align: center;
          page-break-before: avoid; }
.figtext { font-size: 8.5pt; color: #44506a; line-height: 1.7; background: #f6f9fd;
           border: 1px solid #e3ebf5; border-left: 4px solid #9db8d9; border-radius: 6px;
           padding: 8px 12px; margin: 6px 0 4px; text-align: justify; }
.footer { margin-top: 18px; padding-top: 8px; border-top: 1px solid #dde6f0;
          font-size: 8pt; color: #8a97ab; }
"""


def build_pdf(path: str, scope: str, stats: dict, sched: pl.DataFrame,
              decay: pl.DataFrame, stage_loops: pl.DataFrame | None,
              png: str, age_png: str | None, loop_png: str | None,
              beyond: pl.DataFrame | None = None,
              trend_png: str | None = None,
              cfg: Config | None = None,
              today_mi: int | None = None) -> None:
    """生成一份 PDF 报告（先渲染 HTML，再用 Chromium 转 PDF）。scope='总体' 或型号名。

    图表穿插在各章节内：日历月趋势→第二章、循环曲线→第四章、月龄趋势→第五章、
    退役预测→第六章。
    """
    is_all = scope == "总体"
    shown = stats if is_all else {scope: stats.get(scope, {})}
    tot_dev = sum(s.get("devices", 0) for s in shown.values())
    tot_below = sum(s.get("below80", 0) for s in shown.values())
    med_soh = (sum(s.get("soh_median", 0) * s.get("devices", 0) for s in shown.values())
               / tot_dev) if tot_dev else 0
    sc = sched if is_all else sched.filter(pl.col("电池型号") == scope)
    if today_mi is None:
        today_mi = int(sched["_mi"].min()) if sched.height else 2026 * 12 + 9
    win_lo, win_hi = _mi_str(today_mi), _mi_str(today_mi + 71)
    next12_end = _mi_str(today_mi + 11)
    next12 = int(sc.filter(pl.col("ym") <= next12_end)["new_retire"].sum())
    win_total = int(sc["new_retire"].sum())
    peak = sc.sort("new_retire", descending=True).head(1)
    peak_txt = f"{peak['ym'][0]}（{int(peak['new_retire'][0])} 台）" if peak.height else "—"
    # 逐型号体系描述
    desc, c_nom = ("", 37.0)
    if cfg is not None:
        desc, c_nom = model_desc(scope, cfg)
    import datetime as _dt
    _today = _dt.date.today().isoformat()
    _c_nom_txt = f"{c_nom:g}Ah" if not is_all else "各型号规格书标称容量"
    _desc_txt = (f"，电芯体系：{desc}，标称 {c_nom:g}Ah" if (not is_all and desc) else "")

    # 健康阶段分布
    _lv_order = ["健康", "关注", "预警", "退役"]
    lv_tot = {k: 0 for k in _lv_order}
    for s in shown.values():
        for k in _lv_order:
            lv_tot[k] += s.get("levels", {}).get(k, 0)

    h = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
         f"<style>{_CSS}</style></head><body>",
         "<div class='banner'><h1>换电电池 SOH 与退役预测报告"
         + ("" if is_all else f"｜{scope}") + "</h1>",
         f"<div class='meta'>报告日期：{_today}　·　数据来源：换电平台 BMS 上报（锚点法 SOH 估算管线 v2）"
         + _desc_txt
         + "　·　退役标准：SOH &lt; 80%</div></div>",
         "<div class='page'>"]

    # ---- 一、概况
    h.append("<h2>一、电池概况</h2>")
    if is_all:
        h.append(f"<p>本次分析覆盖电池共 <b>{tot_dev:,} 台</b>，分 "
                 f"{len(shown)} 个型号："
                 + "、".join(f"{m} {s['devices']:,} 台" for m, s in shown.items())
                 + "。各型号按自身化学体系与标称容量独立估算 SOH（绝对口径），"
                   "数据窗口较短的型号其低可信设备用本型号中位衰退速率外推预测。</p>")
    else:
        s = shown[scope]
        h.append(f"<p>{scope} 共 <b>{s.get('devices', 0):,} 台</b>电池，最早数据出现在 "
                 f"{s.get('first_month', '')}。该型号为 {desc or '锂电'} 电池包，"
                 f"标称容量 {c_nom:g}Ah，用于城市换电运营，"
                 "平均每台每天经历 1~2 次换电循环。</p>")
    h.append("<div class='kpis'>"
             f"<div class='kpi'><div class='v'>{tot_dev:,}</div><div class='k'>电池总数（台）</div></div>"
             f"<div class='kpi'><div class='v'>{med_soh*100:.1f}%</div><div class='k'>SOH 中位数</div></div>"
             f"<div class='kpi'><div class='v'>{tot_below}</div><div class='k'>已达退役标准（台）</div></div>"
             f"<div class='kpi'><div class='v'>{next12:,}</div><div class='k'>未来 12 个月预计退役（台）</div></div>"
             f"<div class='kpi'><div class='v'>{win_total:,}</div><div class='k'>6 年窗口内预计退役（台）</div></div>"
             "</div>")

    # ---- 二、健康水平统计
    h.append("<h2>二、当前健康水平</h2><table><tr><th>型号</th><th>设备数</th>"
             "<th>SOH 中位</th><th>SOH P10~P90</th><th>已&lt;80%</th>"
             "<th>健康/关注/预警/退役</th></tr>")
    for m, s in shown.items():
        lv = s.get("levels", {})
        lvs = "/".join(f"{lv.get(k, 0):,}" for k in _lv_order)
        h.append(f"<tr><td>{m}</td><td>{s.get('devices', 0):,}</td>"
                 f"<td>{s.get('soh_median', 0)*100:.1f}%</td>"
                 f"<td>{s.get('soh_p10', 0)*100:.1f}% ~ {s.get('soh_p90', 0)*100:.1f}%</td>"
                 f"<td>{s.get('below80', 0)}</td><td>{lvs}</td></tr>")
    h.append("</table>")
    h.append(f"<p class='note'>SOH（健康状态）= 当前实际可放出容量 ÷ 规格书标称容量（{_c_nom_txt}，绝对口径，跨电池可比）。"
             "100% 表示容量等于出厂标称；行业惯例跌破 80% 即达到退役标准。</p>")
    _cap_trend = caption_trend() if is_all else caption_single_trend(scope, desc, c_nom)
    h.append(_fig_html(trend_png, "图 1｜分型号 SOH 日历月趋势：实线=当月全部设备 SOH 中位数，"
                 "阴影带=P10~P90，虚线（右轴）=当月设备数。曲线整体下移即电池整体老化，"
                 "各型号差距含上线时间不同的构成因素。", _cap_trend))

    # ---- 三、算法逻辑
    h.append("<h2>三、估算与预测方法（简明）</h2>"
             "<p><b>1. 锚点法测容量</b>：从 BMS 上报的海量记录中挑选四类“可信时刻”——充满电末端、"
             "换电满放、放空末端、静置开路电压，每种时刻都能反推出一个“当前满容量”估计值，按质量分级。</p>"
             f"<p><b>2. 月度聚合</b>：每台电池每月把当月容量估计取加权中位数（先做月内锚点去毛刺，"
             "锚点不足 3 个的月份不纳入），得到逐月容量曲线；容量 ÷ 该型号规格书标称容量 = 该月 SOH。"
             "初期月份异常偏低、随后恢复正常的设备，只剔除异常的前导月份；"
             "全程容量都异常偏低（&lt;90%）的设备判定为疑似旧电池换 BMS（新 ID、循环归零），整台剔除出统计。</p>"
             "<p><b>3. 衰退速率</b>：对每台电池的逐月 SOH 做线性拟合，得到“每年衰减百分之几”，"
             "R² 表示拟合好坏（越高越可信）。</p>"
             "<p><b>4. 退役预测</b>：把每台电池的 SOH 直线延伸到 80% 水平线，穿越月份即预计退役时间；"
             "拟合不可信（R²&lt;0.6）或数据太短的设备，改用本型号中位衰退速率外推。</p>"
             "<p><b>5. 汇总</b>：按退役月份归堆，得到“未来每月预计有多少台到达退役标准”。</p>")

    # ---- 四、实际使用情况
    h.append("<h2>四、实际使用情况</h2><table><tr><th>型号</th><th>参与拟合设备数</th>"
             "<th>年衰退中位</th><th>年衰退 P25~P75</th><th>首月 SOH</th><th>最新 SOH</th></tr>")
    for r in decay.to_dicts():
        m = r["电池型号"]
        if not is_all and m != scope:
            continue
        h.append(f"<tr><td>{m}</td><td>{r['device_count']:,}</td>"
                 f"<td>{-r['decay_pct_year_median']:.2f}%</td>"
                 f"<td>{-r['decay_pct_year_p75']:.2f}% ~ {-r['decay_pct_year_p25']:.2f}%</td>"
                 f"<td>{r['soh_start']*100:.1f}%</td><td>{r['soh_latest']*100:.1f}%</td></tr>")
    h.append("</table>")
    loop_med = sum(s.get("loops_median", 0) * s.get("devices", 0) for s in shown.values())
    loop_med = loop_med / tot_dev if tot_dev else 0
    h.append(f"<p>全部设备中位累计换电循环约 <b>{loop_med:.0f} 次</b>。按换电频率估算，"
             "平均每台每月消耗约 15~25 个循环，即每年约 200~300 次。</p>")
    _cap_loop = caption_loop() if is_all else caption_single_loop(scope, desc, c_nom)
    h.append(_fig_html(loop_png, "图 2｜SOH-累计循环次数曲线：横轴为 BMS 累计循环次数（25 次一档），"
                 "反映同等循环消耗下的容量保持能力，可剥离使用强度差异。点状竖线=中位 SOH 跌破 80% "
                 "的循环次数；高循环档设备数少，尾部曲线代表性下降。", _cap_loop))

    # ---- 五、生命周期阶段
    h.append("<h2>五、生命周期阶段画像</h2>")
    if stage_loops is not None:
        h.append("<table><tr><th>型号</th><th>阶段</th><th>设备数</th>"
                 "<th>末月循环中位</th><th>循环 P10~P90</th><th>SOH 中位</th></tr>")
        for r in stage_loops.to_dicts():
            if not is_all and r["电池型号"] != scope:
                continue
            h.append(f"<tr><td>{r['电池型号']}</td><td>{r['health_level']}</td>"
                     f"<td>{r['device_count']:,}</td><td>{r['loops_median']:.0f}</td>"
                     f"<td>{r['loops_p10']:.0f} ~ {r['loops_p90']:.0f}</td>"
                     f"<td>{r['soh_median']*100:.1f}%</td></tr>")
        h.append("</table>")
    h.append("<p class='note'>阶段划分：健康（SOH≥95%）、关注（90~95%）、预警（80~90%）、退役（&lt;80%）。"
             "电池每降一档大约多消耗 50~100 个循环。</p>")
    _cap_age = caption_age() if is_all else caption_single_age(scope, desc, c_nom)
    h.append(_fig_html(age_png, "图 3｜SOH-上线月龄趋势：每台电池以自身首条数据月为 0 月龄对齐，"
                 "消除出厂/上线时间差，是同批电池衰退快慢的公平对比。点状竖线=中位 SOH 跌破 80% 的月龄；"
                 "横轴刻度下两行给出该组设备的循环次数与出厂年月分布。", _cap_age))

    # ---- 六、退役预测
    h.append(f"<h2>六、退役时间预测（未来 6 年：{win_lo} ~ {win_hi}）</h2>")
    h.append(f"<p>6 年窗口内预计共有 <b>{win_total:,} 台</b>到达退役标准，"
             f"未来 12 个月 <b>{next12:,} 台</b>；单月退役峰值预计出现在 <b>{peak_txt}</b>。</p>")
    if beyond is not None and beyond.height:
        b = beyond if is_all else beyond.filter(pl.col("电池型号") == scope)
        nb = int(b["n_outside_window"].sum()) if b.height else 0
        if nb:
            h.append(f"<p class='note'>另有 {nb:,} 台设备按当前衰退趋势要到 {win_hi} 之后才跌破 80%"
                     "（衰退较慢或当前 SOH 偏高），超出 6 年预测窗口，未计入下表与曲线。</p>")
    # 按季度汇总表（72 个月太长，季度归并更易读；月度明细在 CSV）
    h.append("<table><tr><th>季度</th><th>预计新增退役(台)</th><th>累计退役(台)</th></tr>")
    quarters: dict[str, int] = {}
    for r in sc.sort("_mi").to_dicts():
        y, mo = int(r["ym"][:4]), int(r["ym"][5:7])
        key = f"{y} Q{(mo - 1) // 3 + 1}"
        quarters[key] = quarters.get(key, 0) + int(r["new_retire"])
    cum = 0
    for q in sorted(quarters):
        cum += quarters[q]
        h.append(f"<tr><td>{q}</td><td>{quarters[q]:,}</td><td>{cum:,}</td></tr>")
    h.append("</table>")
    h.append("<p class='note'>月度明细见 retire_schedule.csv。预测为趋势外推，未计入温度、使用强度变化"
             "与个体故障；早期退役多为单体故障等异常样本。建议每半年用最新数据滚动修正。</p>")
    h.append(_fig_html(png, "图 4｜未来逐月退役台数预测：柱=当月新增到达退役标准（SOH&lt;80%）的台数，"
                 "折线（右轴）=累计台数，标注为 50% 退役里程碑。"))
    h.append(f"<div class='footer'>换电电池 SOH 估算系统 · 报告由分析管线自动生成 · {_today}</div>")
    h.append("</div></body></html>")

    html_path = os.path.splitext(path)[0] + ".html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("".join(h))
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("file:///" + html_path.replace("\\", "/"))
        page.wait_for_timeout(300)
        page.pdf(path=path, format="A4", print_background=True,
                 margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
        browser.close()


def run(cfg: Config) -> dict:
    root = os.path.join(OUTPUT_DIR, "reports")
    total_dir = os.path.join(root, "总体")
    os.makedirs(total_dir, exist_ok=True)
    model_map = cfg.model_map
    if model_map is None:
        print("[report_forecast] 无 device_filter，跳过")
        return {}
    analysis = pl.read_parquet(os.path.join(OUTPUT_DIR, "soh", "soh_analysis.parquet"))
    retirement = pl.read_parquet(os.path.join(OUTPUT_DIR, "retirement",
                                              "retirement_forecast.parquet"))
    today_mi = _latest_mi(analysis)
    sched, beyond = retirement_schedule(analysis, model_map, retirement, today_mi)
    sched.write_csv(os.path.join(total_dir, "retire_schedule.csv"))
    stats = _fleet_stats(analysis, model_map, retirement)
    decay = pl.read_csv(os.path.join(total_dir, "model_decay_stats.csv"))
    sl_path = os.path.join(total_dir, "model_stage_loops.csv")
    stage_loops = pl.read_csv(sl_path) if os.path.exists(sl_path) else None

    def _safe(fn, path):
        tmp = path + ".tmp.png"
        fn(tmp)
        os.replace(tmp, path)
        return path

    png_all = _safe(lambda p: plot_retire_forecast(sched, p),
                    os.path.join(total_dir, "model_retire_forecast.png"))
    models = sorted(str(m) for m in sched["电池型号"].unique().to_list())
    pdfs = []
    # 总体 PDF
    p_path = os.path.join(total_dir, "report_总体.pdf")
    build_pdf(p_path, "总体", stats, sched, decay, stage_loops, png_all,
              os.path.join(total_dir, "model_soh_age_trend.png"),
              os.path.join(total_dir, "model_soh_loop_trend.png"),
              beyond=beyond,
              trend_png=os.path.join(total_dir, "model_soh_trend.png"),
              cfg=cfg, today_mi=today_mi)
    pdfs.append(p_path)
    # 各型号 PDF（输出到各自目录）
    for m in models:
        safe = m.replace("/", "_")
        mdir = os.path.join(root, safe)
        os.makedirs(mdir, exist_ok=True)
        png_m = _safe(lambda p, m=m: plot_retire_forecast(sched, p, m),
                      os.path.join(mdir, "model_retire_forecast.png"))
        p_path = os.path.join(mdir, f"report_{safe}.pdf")
        build_pdf(p_path, m, stats, sched, decay, stage_loops, png_m,
                  os.path.join(mdir, "model_soh_age_trend.png"),
                  os.path.join(mdir, "model_soh_loop_trend.png"),
                  beyond=beyond,
                  trend_png=os.path.join(mdir, "model_soh_trend.png"),
                  cfg=cfg, today_mi=today_mi)
        pdfs.append(p_path)
    print(f"[report_forecast] 退役预测 {sched.height} 行（型号×月），"
          f"PDF×{len(pdfs)} 已生成（总体 + {len(models)} 型号）")
    return {"schedule": sched, "pdfs": pdfs}
