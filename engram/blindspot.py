"""
ENGRAM Blind Spot Visualization
Renders the "Forgetting Landscape" — a visual map of a student's
diagnostic strengths and weaknesses based on FSRS-6 state.
"""

from __future__ import annotations

import html

from .fsrs6 import BlindSpot


# Color scheme for mastery levels
MASTERY_COLORS = {
    "mastered": "#22c55e",   # Green
    "strong": "#84cc16",     # Lime
    "developing": "#eab308", # Yellow
    "weak": "#f97316",       # Orange
    "danger": "#ef4444",     # Red
}


def render_blindspot_html(spots: list[BlindSpot]) -> str:
    """
    Render blind spot analysis as styled HTML for Gradio display.
    Shows a bar chart with retention levels and mastery indicators.
    """
    if not spots:
        return "<div style='color: #888; padding: 20px;'>No data yet. Complete some reviews to see your diagnostic landscape.</div>"

    html_parts = [
        "<div style='font-family: system-ui, -apple-system, sans-serif; padding: 16px;'>",
        "<h3 style='margin: 0 0 16px 0; color: #e2e8f0;'>Your Diagnostic Landscape</h3>",
    ]

    # Summary stats
    mastered = sum(1 for s in spots if s.mastery_level == "mastered")
    danger = sum(1 for s in spots if s.mastery_level == "danger")
    avg_retention = sum(s.retention for s in spots) / len(spots) if spots else 0

    html_parts.append(
        f"<div style='display: flex; gap: 24px; margin-bottom: 20px; font-size: 14px;'>"
        f"<span style='color: #94a3b8;'>Avg Retention: <b style=\"color: #e2e8f0;\">{avg_retention:.0%}</b></span>"
        f"<span style='color: #22c55e;'>Mastered: <b>{mastered}</b></span>"
        f"<span style='color: #ef4444;'>Critical Gaps: <b>{danger}</b></span>"
        f"</div>"
    )

    # Bar chart
    for spot in spots:
        pct = spot.retention * 100
        color = MASTERY_COLORS.get(spot.mastery_level, "#888")
        bar_width = max(2, pct)

        level_badge = (
            f"<span style='font-size: 11px; padding: 2px 8px; border-radius: 4px; "
            f"background: {color}22; color: {color}; margin-left: 8px;'>"
            f"{spot.mastery_level}</span>"
        )

        html_parts.append(f"""
        <div style='margin-bottom: 12px;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;'>
                <span style='font-size: 13px; color: #cbd5e1; font-weight: 500;'>
                    {html.escape(spot.category)}{level_badge}
                </span>
                <span style='font-size: 13px; color: #94a3b8;'>
                    {pct:.0f}% · S={spot.stability:.1f}d · D={spot.difficulty:.1f}
                </span>
            </div>
            <div style='background: #1e293b; border-radius: 6px; height: 20px; overflow: hidden;'>
                <div style='width: {bar_width}%; height: 100%; background: {color};
                     border-radius: 6px; transition: width 0.5s ease;'></div>
            </div>
            <div style='font-size: 11px; color: #64748b; margin-top: 2px;'>
                {spot.total_reviews} reviews · {spot.total_lapses} lapses
            </div>
        </div>
        """)

    html_parts.append("</div>")
    return "".join(html_parts)


def render_calibration_chart_html(calibration_data: dict[str, dict]) -> str:
    """Render confidence-accuracy quadrant chart.
    Input: {category: {mean_confidence, mean_accuracy, calibration_gap, overconfident, n_reviews}}.
    Four quadrants: well-calibrated, OVERCONFIDENT (danger), underconfident, aware of weakness.
    """
    if not calibration_data:
        return "<div style='color:#64748b;padding:16px;font-size:13px;'>No confidence data yet. Rate your confidence on reviews to see calibration.</div>"

    parts = [
        "<div style='font-family:system-ui,-apple-system,sans-serif;padding:16px;'>",
        "<h3 style='margin:0 0 12px;color:#e2e8f0;'>Confidence Calibration</h3>",
        "<div style='font-size:11px;color:#64748b;margin-bottom:12px;'>Are you calibrated? Dots in the OVERCONFIDENT zone need more review.</div>",
    ]

    # Quadrant chart — CSS grid with positioned dots
    parts.append(
        "<div style='position:relative;width:100%;aspect-ratio:1;background:#0f172a;"
        "border-radius:8px;overflow:hidden;border:1px solid #334155;'>"
    )
    # Quadrant labels
    parts.append("<div style='position:absolute;top:8px;left:8px;font-size:10px;color:#64748b;'>Underconfident<br>+ Wrong</div>")
    parts.append("<div style='position:absolute;top:8px;right:8px;font-size:10px;color:#ef4444;font-weight:700;'>OVERCONFIDENT<br>+ Wrong</div>")
    parts.append("<div style='position:absolute;bottom:8px;left:8px;font-size:10px;color:#94a3b8;'>Aware of<br>Weakness</div>")
    parts.append("<div style='position:absolute;bottom:8px;right:8px;font-size:10px;color:#22c55e;font-weight:700;'>Well<br>Calibrated</div>")
    # Diagonal reference line
    parts.append(
        "<div style='position:absolute;top:0;left:0;width:141%;height:1px;"
        "background:#334155;transform:rotate(-45deg);transform-origin:top left;'></div>"
    )
    # Axis labels
    parts.append("<div style='position:absolute;bottom:2px;left:50%;transform:translateX(-50%);font-size:9px;color:#475569;'>Accuracy &rarr;</div>")
    parts.append("<div style='position:absolute;top:50%;left:2px;transform:rotate(-90deg) translateX(-50%);transform-origin:left;font-size:9px;color:#475569;'>Confidence &rarr;</div>")

    # Plot dots
    for cat, data in calibration_data.items():
        safe_cat = html.escape(cat)
        x_pct = data["mean_accuracy"] * 100
        y_pct = (1.0 - data["mean_confidence"]) * 100  # Invert Y
        color = "#ef4444" if data["overconfident"] else "#22c55e"
        size = min(16, 8 + data["n_reviews"])
        conf_val = data["mean_confidence"]
        acc_val = data["mean_accuracy"]
        parts.append(
            f"<div style='position:absolute;left:{x_pct}%;top:{y_pct}%;"
            f"width:{size}px;height:{size}px;border-radius:50%;background:{color};"
            f"transform:translate(-50%,-50%);' title='{safe_cat}: conf={conf_val:.0%} acc={acc_val:.0%}'></div>"
        )

    parts.append("</div>")

    # Legend
    parts.append("<div style='margin-top:10px;display:flex;flex-wrap:wrap;gap:8px;'>")
    for cat, data in sorted(calibration_data.items(), key=lambda x: -abs(x[1]["calibration_gap"])):
        safe_cat = html.escape(cat)
        color = "#ef4444" if data["overconfident"] else "#22c55e"
        gap_str = f"+{data['calibration_gap']:.0%}" if data["calibration_gap"] > 0 else f"{data['calibration_gap']:.0%}"
        parts.append(
            f"<span style='font-size:10px;padding:2px 6px;border-radius:4px;"
            f"background:{color}22;color:{color};'>{safe_cat} ({gap_str})</span>"
        )
    parts.append("</div></div>")
    return "".join(parts)


def render_session_stats_html(stats: dict) -> str:
    """Render session statistics as HTML."""
    return f"""
    <div style='font-family: system-ui, -apple-system, sans-serif; padding: 16px;'>
        <h3 style='margin: 0 0 16px 0; color: #e2e8f0;'>Session Progress</h3>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 12px;'>
            <div style='background: #1e293b; padding: 16px; border-radius: 8px;'>
                <div style='font-size: 24px; font-weight: bold; color: #e2e8f0;'>{stats.get('total_reviews', 0)}</div>
                <div style='font-size: 12px; color: #94a3b8;'>Total Reviews</div>
            </div>
            <div style='background: #1e293b; padding: 16px; border-radius: 8px;'>
                <div style='font-size: 24px; font-weight: bold; color: #e2e8f0;'>{stats.get('avg_score', 0):.0%}</div>
                <div style='font-size: 12px; color: #94a3b8;'>Avg Score</div>
            </div>
            <div style='background: #1e293b; padding: 16px; border-radius: 8px;'>
                <div style='font-size: 24px; font-weight: bold; color: #e2e8f0;'>{stats.get('avg_box_iou', 0):.0%}</div>
                <div style='font-size: 12px; color: #94a3b8;'>Spatial Accuracy</div>
            </div>
            <div style='background: #1e293b; padding: 16px; border-radius: 8px;'>
                <div style='font-size: 24px; font-weight: bold; color: #e2e8f0;'>{stats.get('categories_practiced', 0)}</div>
                <div style='font-size: 12px; color: #94a3b8;'>Categories</div>
            </div>
        </div>
    </div>
    """


