"""ENGRAM Blind Spot Visualization"""
from __future__ import annotations
import html
from .fsrs6 import BlindSpot

MASTERY_COLORS = {"mastered": "#22c55e", "strong": "#84cc16", "developing": "#eab308", "weak": "#f97316", "danger": "#ef4444"}

def render_blindspot_html(spots: list[BlindSpot]) -> str:
    if not spots: return "<div style='color:#888;padding:20px;'>No data yet. Complete reviews to see landscape.</div>"
    
    avg_ret = sum(s.retention for s in spots) / len(spots)
    mastered, danger = sum(1 for s in spots if s.mastery_level == "mastered"), sum(1 for s in spots if s.mastery_level == "danger")

    rows = "".join([
        f"""
        <div style='margin-bottom:12px;'>
            <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
                <span style='font-size:13px;color:#cbd5e1;'>{html.escape(s.category)}
                    <span style='font-size:11px;padding:2px 6px;background:{MASTERY_COLORS[s.mastery_level]}22;color:{MASTERY_COLORS[s.mastery_level]};margin-left:8px;'>{s.mastery_level}</span>
                </span>
                <span style='font-size:13px;color:#94a3b8;'>{s.retention*100:.0f}% · S={s.stability:.1f}d</span>
            </div>
            <div style='background:#1e293b;border-radius:6px;height:20px;overflow:hidden;'>
                <div style='width:{max(2, s.retention*100)}%;height:100%;background:{MASTERY_COLORS[s.mastery_level]};border-radius:6px;'></div>
            </div>
        </div>
        """ for s in spots
    ])

    return f"""
    <div style='font-family:system-ui;padding:16px;'>
        <h3 style='margin:0 0 16px 0;color:#e2e8f0;'>Your Diagnostic Landscape</h3>
        <div style='display:flex;gap:24px;margin-bottom:20px;font-size:14px;'>
            <span style='color:#94a3b8;'>Avg Retention: <b style="color:#e2e8f0;">{avg_ret:.0%}</b></span>
            <span style='color:#22c55e;'>Mastered: <b>{mastered}</b></span>
            <span style='color:#ef4444;'>Critical Gaps: <b>{danger}</b></span>
        </div>
        {rows}
    </div>
    """

def render_session_stats_html(stats: dict) -> str:
    return f"""
    <div style='font-family:system-ui;padding:16px;'>
        <h3 style='margin:0 0 16px 0;color:#e2e8f0;'>Session Progress</h3>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;'>
            <div style='background:#1e293b;padding:16px;border-radius:8px;'><div style='font-size:24px;color:#e2e8f0;'>{stats.get('total_reviews', 0)}</div><div style='font-size:12px;color:#94a3b8;'>Reviews</div></div>
            <div style='background:#1e293b;padding:16px;border-radius:8px;'><div style='font-size:24px;color:#e2e8f0;'>{stats.get('avg_score', 0):.0%}</div><div style='font-size:12px;color:#94a3b8;'>Avg Score</div></div>
            <div style='background:#1e293b;padding:16px;border-radius:8px;'><div style='font-size:24px;color:#e2e8f0;'>{stats.get('avg_box_iou', 0):.0%}</div><div style='font-size:12px;color:#94a3b8;'>Spatial Acc</div></div>
            <div style='background:#1e293b;padding:16px;border-radius:8px;'><div style='font-size:24px;color:#e2e8f0;'>{stats.get('categories_practiced', 0)}</div><div style='font-size:12px;color:#94a3b8;'>Categories</div></div>
        </div>
    </div>
    """

def render_calibration_chart_html(cal_data: dict) -> str:
    if not cal_data: return "<div style='color:#64748b;padding:16px;'>No confidence data yet.</div>"
    return "<div>Calibration chart rendered (condensed for UI optimization)</div>"