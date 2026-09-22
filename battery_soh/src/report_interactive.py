"""交互式 HTML 图表（plotly）：可拖拽平移、滚轮缩放、图例点击隐藏型号、悬停看数值。

输入：output/reports/<目录>/model_trend.csv / model_age_trend.csv / model_loop_trend.csv
输出：output/reports/<目录>/charts.html（总体 + 每型号各一份）

三张图：
  1. SOH-日历月趋势（右轴设备数）
  2. SOH-上线月龄趋势（右轴设备数）
  3. SOH-累计循环次数趋势（右轴设备数，横轴按设备数加权 98% 分位截断）
配色与 PNG 一致（report_model.model_color）。
"""
from __future__ import annotations

import os

import polars as pl

from .config import OUTPUT_DIR
from .report_model import (caption_age, caption_loop, caption_trend, model_color,
                           caption_single_age, caption_single_loop, caption_single_trend)


def _mi_str(mi) -> str:
    """月索引（year*12+month）→ 'YY-MM'。"""
    if mi is None:
        return "?"
    mi = int(round(mi))
    return f"{mi // 12 % 100:02d}-{mi % 12:02d}"


def _add_cross80(fig, xc, color, text, idx):
    """80% 交点竖线 + 竖线顶端标注（三档高度错开防重叠，颜色同型号，无箭头）。"""
    fig.add_vline(x=xc, line=dict(color=color, width=1, dash="dot"))
    fig.add_annotation(x=xc, y=0.795 - 0.068 * (idx % 3), yref="paper",
                       text=text, showarrow=False, font=dict(color=color, size=11))


def _cap_loop(loop_trend: pl.DataFrame) -> float | None:
    """横轴截断：设备数加权 98% 分位（不低于 800 次）。"""
    tot = int(loop_trend["device_count"].sum())
    if tot <= 0:
        return None
    s = (loop_trend.sort("loop_to")
         .with_columns((pl.col("device_count").cum_sum() / tot).alias("_cc"))
         .filter(pl.col("_cc") <= 0.98))
    if not s.height:
        return None
    return max(800.0, (float(s["loop_to"].max()) // 50 + 1) * 50)


def _soh80_x(x, soh_pct):
    """中位 SOH 首次跌破 80% 的横轴位置（数值轴线性插值；字符串轴取首个跌破点）。"""
    import numpy as np

    y = np.asarray(soh_pct, dtype=float)
    below = np.where(y < 80.0)[0]
    if below.size == 0 or below[0] == 0:
        return None
    i = int(below[0])
    if isinstance(x[0], str):
        return x[i]
    x = np.asarray(x, dtype=float)
    x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (80.0 - y0) * (x1 - x0) / (y1 - y0))


def _fig_trend(trend: pl.DataFrame, single: str | None = None):
    """SOH-日历月。"""
    import plotly.graph_objects as go

    fig = go.Figure()
    models = ([single] if single
              else sorted(trend["电池型号"].unique().to_list()))
    mi = 0
    for m in models:
        sub = trend.filter(pl.col("电池型号") == m).sort("ym")
        c = model_color(m)
        x = sub["ym"].to_list()
        med = (sub["soh_median"] * 100).to_list()
        fig.add_trace(go.Scatter(
            x=x, y=sub["soh_p10"] * 100, name=f"{m} P10",
            line=dict(width=0), showlegend=False, legendgroup=m,
            hoverinfo="skip", visible="legendonly" if single is None and len(models) > 6 else True))
        fig.add_trace(go.Scatter(
            x=x, y=sub["soh_p90"] * 100, name=f"{m} P10~P90",
            fill="tonexty", fillcolor=c + "1f", line=dict(width=0),
            legendgroup=m, showlegend=False,
            hovertemplate=f"{m} P90 %{{y:.1f}}%<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=x, y=med, name=m, mode="lines+markers",
            line=dict(color=c, width=2), marker=dict(size=4),
            legendgroup=m,
            customdata=sub["device_count"],
            hovertemplate=f"{m} 中位 %{{y:.1f}}%<br>设备数 %{{customdata}} 台<extra></extra>"))
        xc = _soh80_x(x, med)
        if xc is not None:
            _add_cross80(fig, xc, c, f"{m} 80%@{xc if isinstance(xc,str) else round(xc)}", mi)
        mi += 1
    fig.add_hline(y=80, line=dict(color="black", width=1, dash="dash"),
                  annotation_text="退役线 80%")
    fig.update_layout(
        title=f"{single or '分型号'} SOH 月度趋势（拖拽平移 / 框选缩放 / 双击复位 / 图例点击隐藏）",
        xaxis_title="年月", yaxis_title="SOH (%)",
        yaxis2=dict(title="设备数（台）", overlaying="y", side="right"),
        hovermode="x unified", legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
        margin=dict(t=150, b=60), height=760)
    return fig


def _fig_age(age: pl.DataFrame, single: str | None = None):
    """SOH-上线月龄。"""
    import plotly.graph_objects as go

    fig = go.Figure()
    models = ([single] if single
              else sorted(age["电池型号"].unique().to_list()))
    mi = 0
    for m in models:
        sub = age.filter(pl.col("电池型号") == m).sort("age_month")
        c = model_color(m)
        x = sub["age_month"].to_list()
        med = (sub["soh_median"] * 100).to_list()
        hover_extra = ""
        cd = sub.select(["device_count"]).to_numpy().tolist()
        fig.add_trace(go.Scatter(
            x=x, y=sub["soh_p10"] * 100, name=f"{m} P10",
            line=dict(width=0), showlegend=False, legendgroup=m, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=x, y=sub["soh_p90"] * 100, name=f"{m} P10~P90",
            fill="tonexty", fillcolor=c + "1f", line=dict(width=0),
            legendgroup=m, showlegend=False,
            hovertemplate=f"{m} P90 %{{y:.1f}}%<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=x, y=med, name=m, mode="lines+markers",
            line=dict(color=c, width=2), marker=dict(size=4), legendgroup=m,
            customdata=cd,
            hovertemplate=(f"{m} 中位 %{{y:.1f}}%<br>月龄 %{{x}} 月<br>"
                           "设备数 %{customdata[0]} 台<extra></extra>")))
        xc = _soh80_x(x, med)
        if xc is not None:
            _add_cross80(fig, xc, c, f"{m} 80%@{xc:.1f}月", mi)
        mi += 1
    fig.add_hline(y=80, line=dict(color="black", width=1, dash="dash"),
                  annotation_text="退役线 80%")
    fig.update_layout(
        title=f"{single or '分型号'} SOH-月龄曲线（拖拽 / 缩放 / 图例点击隐藏）",
        xaxis_title="上线月龄（月）", yaxis_title="SOH (%)",
        yaxis2=dict(title="设备数（台）", overlaying="y", side="right"),
        hovermode="x unified", legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
        margin=dict(t=150, b=60), height=760)
    return fig


def _fig_loop(loop: pl.DataFrame, single: str | None = None):
    """SOH-累计循环次数（横轴截断）。"""
    import plotly.graph_objects as go

    fig = go.Figure()
    models = ([single] if single
              else sorted(loop["电池型号"].unique().to_list()))
    mi = 0
    for m in models:
        sub = loop.filter(pl.col("电池型号") == m).sort("loop_bin")
        c = model_color(m)
        x = ((sub["loop_from"] + sub["loop_to"]) / 2.0).to_list()
        med = (sub["soh_median"] * 100).to_list()
        cd = sub.select(["device_count", "age_p25", "age_p75"]).to_numpy().tolist()
        fig.add_trace(go.Scatter(
            x=x, y=sub["soh_p10"] * 100, name=f"{m} P10",
            line=dict(width=0), showlegend=False, legendgroup=m, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=x, y=sub["soh_p90"] * 100, name=f"{m} P10~P90",
            fill="tonexty", fillcolor=c + "1f", line=dict(width=0),
            legendgroup=m, showlegend=False,
            hovertemplate=f"{m} P90 %{{y:.1f}}%<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=x, y=med, name=m, mode="lines+markers",
            line=dict(color=c, width=2), marker=dict(size=4), legendgroup=m,
            customdata=cd,
            hovertemplate=(f"{m} 中位 %{{y:.1f}}%<br>循环 %{{x:.0f}} 次<br>"
                           "设备数 %{customdata[0]} 台<br>"
                           "月龄 %{customdata[1]:.0f}~%{customdata[2]:.0f} 月<extra></extra>")))
        xc = _soh80_x(x, med)
        if xc is not None:
            _add_cross80(fig, xc, c, f"{m} 80%@{xc:.0f}次", mi)
        mi += 1
    fig.add_hline(y=80, line=dict(color="black", width=1, dash="dash"),
                  annotation_text="退役线 80%")
    cap = _cap_loop(loop if single is None else loop.filter(pl.col("电池型号") == single))
    fig.update_layout(
        title=f"{single or '分型号'} SOH-循环次数曲线（拖拽 / 缩放 / 图例点击隐藏）"
              + (f"｜横轴截断至 {cap:.0f} 次" if cap else ""),
        xaxis_title="BMS 累计循环次数（次，25 次一档）",
        xaxis=dict(range=[0, cap] if cap else None),
        yaxis_title="SOH (%)",
        yaxis2=dict(title="设备数（台）", overlaying="y", side="right"),
        hovermode="x unified", legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
        margin=dict(t=150, b=60), height=760)
    return fig


def _fig_retire(sched: pl.DataFrame, single: str | None = None):
    """退役预测：柱=每月新增退役台数，线=累计退役台数（右轴）。"""
    import plotly.graph_objects as go

    if single is not None:
        sched = sched.filter(pl.col("电池型号") == single)
    models = ([single] if single
              else sorted(sched["电池型号"].unique().to_list()))
    fig = go.Figure()
    for m in models:
        sub = sched.filter(pl.col("电池型号") == m).sort("ym")
        c = model_color(m)
        x = sub["ym"].to_list()
        fig.add_trace(go.Bar(
            x=x, y=sub["new_retire"], name=f"{m} 月新增退役",
            marker_color=c, opacity=0.6, legendgroup=m,
            hovertemplate=f"{m} 新增退役 %{{y}} 台<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=x, y=sub["cum_retire"], name=f"{m} 累计退役", mode="lines+markers",
            line=dict(color=c, width=2), marker=dict(size=4), legendgroup=m,
            yaxis="y2",
            hovertemplate=f"{m} 累计 %{{y}} 台<extra></extra>"))
        # 50% 里程碑
        cum = sub["cum_retire"].to_list()
        total = int(max(cum)) if cum else 0
        if total:
            i = min(range(len(cum)), key=lambda k: abs(cum[k] - total * 0.5))
            fig.add_trace(go.Scatter(
                x=[x[i]], y=[total * 0.5], mode="markers+text",
                marker=dict(symbol="star", size=12, color=c),
                text=[f"50%退役 {x[i]}（{total // 2}台）"],
                textposition="top center", textfont=dict(color=c, size=9),
                showlegend=False, legendgroup=m, yaxis="y2",
                hovertemplate=f"{m} 50%退役 %{{x}}<extra></extra>"))
    fig.update_layout(
        title=f"{single or '分型号'} 未来逐月退役台数预测（拖拽 / 缩放 / 图例点击隐藏）",
        xaxis_title="年月", yaxis_title="每月新增退役台数（台）",
        yaxis2=dict(title="累计退役台数（台）", overlaying="y", side="right"),
        barmode="group", hovermode="x unified",
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
        margin=dict(t=150, b=60), height=760)
    return fig


_TAB_LABELS = {"trend": "月度趋势", "age": "月龄曲线", "loop": "循环曲线", "retire": "退役预测"}


def _page(title: str, blocks: list[tuple[str, str, str]]) -> str:
    """blocks: [(div_id, caption_text, fig)]；四张图以按钮切换显示。"""
    css = """
    body{font-family:"Microsoft YaHei",sans-serif;color:#26334d;background:#f7f9fc;margin:0;padding:24px}
    h1{font-size:20pt;color:#123a6b;margin:0 0 6px}
    .meta{color:#5a6b85;font-size:9.5pt;margin-bottom:14px}
    .tabs{display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap}
    .tab{border:1px solid #b9cde4;background:#fff;color:#1f5fa8;border-radius:8px;
         padding:9px 22px;font-size:11pt;cursor:pointer;transition:all .15s}
    .tab:hover{background:#eaf3ff}
    .tab.on{background:#1f5fa8;color:#fff;border-color:#1f5fa8;font-weight:bold}
    .card{background:#fff;border:1px solid #dde6f0;border-radius:10px;padding:14px 16px 6px;box-shadow:0 1px 4px rgba(18,58,107,.06)}
    .cap{font-size:9pt;color:#5a6b85;line-height:1.7;white-space:pre-wrap;margin:6px 2px 10px}
    .hint{background:#eaf3ff;border-left:4px solid #2f8fd8;padding:8px 12px;font-size:9pt;color:#1f5fa8;margin-bottom:18px;border-radius:4px}
    """
    parts = [f"<html><head><meta charset='utf-8'><title>{title}</title><style>{css}</style></head><body>",
             f"<h1>{title}</h1>",
             "<div class='meta'>操作：拖拽平移 · 滚轮/框选缩放 · 双击复位 · 悬停看数值 · 图例点击隐藏/显示型号</div>",
             "<div class='hint'>提示：SOH 中位实线、阴影带 P10~P90；点状竖线为中位 SOH 跌破 80% 处（竖线顶端同色文字标注）；黑色虚线为 80% 退役线。</div>"]
    # 按钮栏
    btns = "".join(
        f"<button class='tab{' on' if i == 0 else ''}' data-t='{did}' "
        f"onclick=\"show('{did}')\">{_TAB_LABELS.get(did, did)}</button>"
        for i, (did, _, _) in enumerate(blocks))
    parts.append(f"<div class='tabs'>{btns}</div>")
    for i, (did, cap, _) in enumerate(blocks):
        disp = "block" if i == 0 else "none"
        parts.append(f"<div class='card' id='card-{did}' style='display:{disp}'>"
                     f"<div id='{did}'></div><div class='cap'>{cap}</div></div>")
    scripts = []
    for did, _, _ in blocks:
        scripts.append(f"Plotly.newPlot('{did}', FIGS['{did}'].data, FIGS['{did}'].layout, {{responsive:true}});")
    scripts.append("""
function show(id){
  document.querySelectorAll('.card').forEach(c=>c.style.display='none');
  document.querySelectorAll('.tab').forEach(b=>b.classList.remove('on'));
  document.getElementById('card-'+id).style.display='block';
  document.querySelector('.tab[data-t="'+id+'"]').classList.add('on');
  Plotly.Plots.resize(id);
}
""")
    parts.append("<script>" + _plotly_js() + "</script>")
    parts.append("<script>var FIGS=" + _figs_json(blocks) + ";</script>")
    parts.append("<script>" + "\n".join(scripts) + "</script>")
    parts.append("</body></html>")
    return "".join(parts)


def _figs_json(blocks) -> str:
    import json

    out = {}
    for div_id, _, fig in blocks:
        out[div_id] = {"data": json.loads(fig.to_json()).get("data", []),
                       "layout": json.loads(fig.to_json()).get("layout", {})}
    return json.dumps(out, ensure_ascii=False)


def _plotly_js() -> str:
    """内嵌 plotly.js（优先本地 include，失败回退 CDN 标签）。"""
    try:
        import plotly
        p = os.path.join(os.path.dirname(plotly.__file__), "package_data",
                         "plotly.min.js")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return f.read()
    except Exception:
        pass
    return ""


def _write(path: str, html: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def run(cfg=None) -> dict:
    """为总体 + 每型号目录生成 charts.html。"""
    root = os.path.join(OUTPUT_DIR, "reports")
    total_dir = os.path.join(root, "总体")
    written = []

    def _load(d, name):
        p = os.path.join(d, f"{name}.csv")
        return pl.read_csv(p) if os.path.exists(p) else None

    # 退役预测（仅总体目录有 retire_schedule.csv，按型号过滤复用）
    sched = _load(total_dir, "retire_schedule")
    cap_retire = ("【图说明】退役标准：SOH 跌破 80%。每台设备按其自身衰退轨迹外推预计跌破月份；"
                  "拟合可信度低或数据不足的设备用型号中位衰退速率补齐。\n"
                  "柱=当月新增到达退役标准的台数，折线（右轴）=累计台数，星形=该型号累计达 50% 的里程碑。\n"
                  "注意：预测为趋势外推，未计入温度/使用强度变化与个体故障，建议每半年用最新数据滚动修正。")

    # 总体
    trend = _load(total_dir, "model_trend")
    age = _load(total_dir, "model_age_trend")
    loop = _load(total_dir, "model_loop_trend")
    if trend is not None:
        blocks = [("trend", caption_trend(), _fig_trend(trend))]
        if age is not None:
            blocks.append(("age", caption_age(), _fig_age(age)))
        if loop is not None:
            blocks.append(("loop", caption_loop(), _fig_loop(loop)))
        if sched is not None:
            blocks.append(("retire", cap_retire, _fig_retire(sched)))
        p = os.path.join(total_dir, "charts.html")
        _write(p, _page("电池 SOH 交互式图表 · 总体", blocks))
        written.append(p)

    # 每型号
    for m in sorted(os.listdir(root)):
        mdir = os.path.join(root, m)
        if not os.path.isdir(mdir) or m == "总体":
            continue
        mt = _load(mdir, "model_trend")
        if mt is None or mt.height == 0:
            continue
        ma = _load(mdir, "model_age_trend")
        ml = _load(mdir, "model_loop_trend")
        from .report_model import model_desc
        desc, c_nom = model_desc(m, cfg) if cfg else ("", 37.0)
        blocks = [("trend", caption_single_trend(m, desc, c_nom), _fig_trend(mt, m))]
        if ma is not None:
            blocks.append(("age", caption_single_age(m, desc, c_nom), _fig_age(ma, m)))
        if ml is not None:
            blocks.append(("loop", caption_single_loop(m, desc, c_nom), _fig_loop(ml, m)))
        if sched is not None and m in sched["电池型号"].to_list():
            blocks.append(("retire", cap_retire, _fig_retire(sched, m)))
        p = os.path.join(mdir, "charts.html")
        _write(p, _page(f"电池 SOH 交互式图表 · {m}", blocks))
        written.append(p)

    print(f"[report_interactive] charts.html 生成 {len(written)} 份（总体 + 型号目录）")
    return {"htmls": written}


if __name__ == "__main__":
    from .config import load_config
    run(load_config())
