"""
AIDEOM-VN — Mô hình ra quyết định kinh tế số Việt Nam
App Streamlit tích hợp 12 mô hình học thuật.
Chỉ xử lý app shell/UI; logic tính toán nằm trong modules/baiXX.py.
"""

from __future__ import annotations

import importlib
import html
import math
import re
import sys
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


st.set_page_config(
    page_title="AIDEOM 🇻🇳 Tran Thu Ha",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── AIDEOM UI runtime skin: Plotly theme + no black code blocks ──────────────
AIDEOM_PALETTES = {
    "home":  ["#F472B6", "#A78BFA", "#67E8F9", "#F9A8D4", "#FDE68A", "#86EFAC"],
    "bai01": ["#22D3EE", "#F472B6", "#A78BFA", "#F9A8D4", "#99F6E4", "#FDE68A"],
    "bai02": ["#FBBF24", "#F472B6", "#FDBA74", "#A78BFA", "#67E8F9", "#F9A8D4"],
    "bai03": ["#A78BFA", "#F472B6", "#C084FC", "#67E8F9", "#FDE68A", "#86EFAC"],
    "bai04": ["#6EE7B7", "#67E8F9", "#F9A8D4", "#A78BFA", "#FDE68A", "#F472B6"],
    "bai05": ["#FB7185", "#F9A8D4", "#FDBA74", "#A78BFA", "#67E8F9", "#FDE68A"],
    "bai06": ["#60A5FA", "#C4B5FD", "#F9A8D4", "#67E8F9", "#FDE68A", "#A78BFA"],
    "bai07": ["#8B5CF6", "#67E8F9", "#F472B6", "#A78BFA", "#86EFAC", "#FDE68A"],
    "bai08": ["#FDBA74", "#FDE68A", "#F9A8D4", "#A78BFA", "#67E8F9", "#F472B6"],
    "bai09": ["#F472B6", "#FB923C", "#F9A8D4", "#A78BFA", "#67E8F9", "#FDE68A"],
    "bai10": ["#6366F1", "#C4B5FD", "#F9A8D4", "#67E8F9", "#FDE68A", "#F472B6"],
    "bai11": ["#22D3EE", "#60A5FA", "#A78BFA", "#F9A8D4", "#86EFAC", "#FDE68A"],
    "bai12": ["#F472B6", "#67E8F9", "#A78BFA", "#FDE68A", "#86EFAC", "#FDBA74"],
}

PAGE_ACCENTS = {
    "home":  ("#F472B6", "#A78BFA", "#67E8F9"),
    "bai01": ("#22D3EE", "#F472B6", "#A78BFA"),
    "bai02": ("#FBBF24", "#F472B6", "#FDBA74"),
    "bai03": ("#A78BFA", "#F472B6", "#C084FC"),
    "bai04": ("#6EE7B7", "#67E8F9", "#A78BFA"),
    "bai05": ("#FB7185", "#F9A8D4", "#FDBA74"),
    "bai06": ("#60A5FA", "#C4B5FD", "#F9A8D4"),
    "bai07": ("#8B5CF6", "#67E8F9", "#F472B6"),
    "bai08": ("#FDBA74", "#FDE68A", "#F9A8D4"),
    "bai09": ("#F472B6", "#FB923C", "#F9A8D4"),
    "bai10": ("#6366F1", "#C4B5FD", "#F9A8D4"),
    "bai11": ("#22D3EE", "#60A5FA", "#A78BFA"),
    "bai12": ("#F472B6", "#67E8F9", "#FDE68A"),
}

CHART_TITLE_FALLBACK = {
    "home": "Tổng quan AIDEOM-VN",
    "bai01": "Dự báo GDP 2026-2030",
    "bai02": "Phân bổ ngân sách theo kịch bản",
    "bai03": "Xếp hạng Priority Index 10 ngành",
    "bai04": "Phân bổ ngân sách ngành-vùng",
    "bai05": "Danh mục dự án tối ưu",
    "bai06": "Xếp hạng TOPSIS 6 vùng",
    "bai07": "Biên Pareto NSGA-II",
    "bai08": "Quỹ đạo tối ưu động 2026-2035",
    "bai09": "Tác động lao động và AI",
    "bai10": "Stochastic Programming theo kịch bản",
    "bai11": "Đường học Q-learning",
    "bai12": "Xếp hạng Regional Readiness",
}

BAD_PLOTLY_TOKEN = "un" + "defined"


def _aideom_current_palette() -> list[str]:
    return AIDEOM_PALETTES.get(st.session_state.get("current_page", "home"), AIDEOM_PALETTES["home"])


def _aideom_is_bad_text(value) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text.lower() in {"", "none", "null", BAD_PLOTLY_TOKEN, "nan"}


def _aideom_clean_text(value, fallback: str = "") -> str:
    if _aideom_is_bad_text(value):
        return fallback
    text = str(value)
    return text.replace(BAD_PLOTLY_TOKEN, "").replace(BAD_PLOTLY_TOKEN.title(), "").replace(BAD_PLOTLY_TOKEN.upper(), "").strip()


def _aideom_scrub_plotly_dict(obj):
    if isinstance(obj, dict):
        return {key: _aideom_scrub_plotly_dict(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_aideom_scrub_plotly_dict(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_aideom_scrub_plotly_dict(value) for value in obj)
    if isinstance(obj, str) and obj.strip().lower() == BAD_PLOTLY_TOKEN:
        return ""
    if isinstance(obj, str) and BAD_PLOTLY_TOKEN in obj.lower():
        return obj.replace(BAD_PLOTLY_TOKEN, "").replace(BAD_PLOTLY_TOKEN.title(), "").replace(BAD_PLOTLY_TOKEN.upper(), "").strip()
    return obj


def _aideom_style_plotly(fig):
    palette = _aideom_current_palette()
    try:
        page_id = st.session_state.get("current_page", "home")
        raw_title = getattr(getattr(fig.layout, "title", None), "text", None)
        raw_title_text = _aideom_clean_text(
            raw_title,
            CHART_TITLE_FALLBACK.get(page_id, "Biểu đồ kết quả mô hình AIDEOM-VN"),
        )
        fig.update_layout(
            template="plotly_white",
            colorway=palette,
            paper_bgcolor="rgba(255,248,252,0.78)",
            plot_bgcolor="rgba(255,255,255,0.72)",
            font=dict(family='"Be Vietnam Pro","Inter","Segoe UI","Arial",sans-serif', color="#2B1230", size=14),
            title=dict(
                text=raw_title_text,
                font=dict(color="#2B1230", size=21, family='"Be Vietnam Pro","Inter","Segoe UI","Arial",sans-serif'),
                x=0.02,
            ),
            legend=dict(
                bgcolor="rgba(255,255,255,0.80)",
                bordercolor="rgba(244,114,182,0.28)",
                borderwidth=1,
                font=dict(color="#3B1F35", size=13),
                orientation="h",
                yanchor="top",
                y=-0.22,
                xanchor="center",
                x=0.5,
            ),
            hoverlabel=dict(
                bgcolor="rgba(255,255,255,0.98)",
                bordercolor=palette[0],
                font=dict(color="#2B1230", family='"Be Vietnam Pro","Inter","Segoe UI","Arial",sans-serif', size=13),
            ),
            hovermode="x unified",
            margin=dict(l=82, r=48, t=96, b=138),
            height=max(getattr(fig.layout, "height", 0) or 0, 540),
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(167,139,250,0.18)",
            zerolinecolor="rgba(167,139,250,0.22)",
            linecolor="rgba(59,31,53,0.32)",
            tickfont=dict(color="#3B1F35", size=12),
            title_font=dict(color="#2B1230", size=14),
            automargin=True,
            tickangle=-18,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(244,114,182,0.16)",
            zerolinecolor="rgba(167,139,250,0.22)",
            linecolor="rgba(59,31,53,0.32)",
            tickfont=dict(color="#3B1F35", size=12),
            title_font=dict(color="#2B1230", size=14),
            automargin=True,
        )
        if any(getattr(trace, "type", "") == "pie" for trace in fig.data):
            fig.update_layout(
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.12,
                    xanchor="center",
                    x=0.5,
                    bgcolor="rgba(255,255,255,0.84)",
                    font=dict(color="#3B1F35", size=13),
                ),
                margin=dict(l=50, r=50, t=96, b=120),
                height=max(getattr(fig.layout, "height", 0) or 0, 520),
            )
        if fig.layout.annotations:
            fig.layout.annotations = tuple(
                ann
                for ann in fig.layout.annotations
                if "AIDEOM-VN Policy Lab" not in str(getattr(ann, "text", ""))
            )
        for ann in fig.layout.annotations or []:
            try:
                ann.text = _aideom_clean_text(getattr(ann, "text", ""), "")
            except Exception:
                pass
        for idx, trace in enumerate(fig.data):
            try:
                if _aideom_is_bad_text(getattr(trace, "name", None)):
                    trace.name = ""
                else:
                    trace.name = _aideom_clean_text(getattr(trace, "name", ""), "")
            except Exception:
                pass
            try:
                if hasattr(trace, "hovertemplate"):
                    trace.hovertemplate = _aideom_clean_text(getattr(trace, "hovertemplate", ""), "")
            except Exception:
                pass
            if getattr(trace, "marker", None) is not None:
                try:
                    if getattr(trace.marker, "color", None) is None:
                        trace.marker.color = palette[idx % len(palette)]
                    trace.marker.line = dict(color="rgba(255,255,255,0.78)", width=1)
                except Exception:
                    pass
            if getattr(trace, "line", None) is not None:
                try:
                    if getattr(trace.line, "color", None) is None:
                        trace.line.color = palette[idx % len(palette)]
                    trace.line.width = max(getattr(trace.line, "width", 0) or 0, 2.8)
                except Exception:
                    pass
    except Exception:
        pass
    return fig


def _aideom_number(value) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(number):
        return str(value)
    abs_number = abs(number)
    if abs_number >= 1000:
        return f"{number:,.1f}"
    if abs_number >= 10:
        return f"{number:,.2f}"
    return f"{number:,.4f}"


def _aideom_values(values) -> list[float]:
    if values is None:
        return []
    try:
        values = values.tolist()
    except Exception:
        pass
    if not isinstance(values, (list, tuple)):
        return []
    cleaned: list[float] = []
    for value in values:
        try:
            number = float(value)
        except Exception:
            continue
        if math.isfinite(number):
            cleaned.append(number)
    return cleaned


def _aideom_labels(values) -> list[str]:
    if values is None:
        return []
    try:
        values = values.tolist()
    except Exception:
        pass
    if not isinstance(values, (list, tuple)):
        return []
    return [str(value) for value in values]


def _aideom_note_card(title: str, sentences: list[str]) -> None:
    if st.session_state.get("current_page", "home") == "home":
        return
    if st.session_state.get("suppress_auto_result_notes", False):
        return
    clean_sentences = [s.strip() for s in sentences if s and s.strip()]
    if not clean_sentences:
        return
    body = " ".join(html.escape(sentence) for sentence in clean_sentences[:6])
    st.markdown(
        f"""
        <div class="result-note-card">
          <div class="result-note-title">💬 {html.escape(title)}</div>
          <div class="result-note-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _chart_primary_trace(fig):
    traces = list(getattr(fig, "data", []) or [])
    best = None
    best_len = -1
    for trace in traces:
        values = _aideom_values(getattr(trace, "y", None))
        if not values:
            values = _aideom_values(getattr(trace, "values", None))
        if len(values) > best_len:
            best = trace
            best_len = len(values)
    return best


def _render_plotly_result_note(fig) -> None:
    try:
        title = _aideom_clean_text(getattr(getattr(fig.layout, "title", None), "text", ""), "")
        lowered = title.lower()
        traces = list(getattr(fig, "data", []) or [])
        if not traces:
            return

        sentences: list[str] = []
        if "tfp" in lowered:
            trace = _chart_primary_trace(fig)
            y = _aideom_values(getattr(trace, "y", None))
            x = _aideom_labels(getattr(trace, "x", None))
            if len(y) >= 2:
                trend = "tăng" if y[-1] >= y[0] else "giảm"
                sentences.append(f"TFP {trend} từ {_aideom_number(y[0])} năm {x[0] if x else 'đầu kỳ'} lên {_aideom_number(y[-1])} năm {x[-1] if x else 'cuối kỳ'}.")
                peak_idx = max(range(len(y)), key=lambda i: y[i])
                sentences.append(f"Mức TFP cao nhất nằm ở {x[peak_idx] if x else 'điểm cuối chuỗi'} với giá trị {_aideom_number(y[peak_idx])}.")
                sentences.append("Điều này gợi ý năng suất tổng hợp đang là tín hiệu quan trọng để đánh giá chất lượng tăng trưởng, không chỉ quy mô vốn hay lao động.")
        elif "sai số" in lowered or "ape" in lowered:
            trace = _chart_primary_trace(fig)
            y = _aideom_values(getattr(trace, "y", None))
            x = _aideom_labels(getattr(trace, "x", None))
            if y:
                max_idx = max(range(len(y)), key=lambda i: y[i])
                min_idx = min(range(len(y)), key=lambda i: y[i])
                sentences.append(f"Sai số cao nhất rơi vào {x[max_idx] if x else 'một quan sát'} với {_aideom_number(y[max_idx])}%.")
                sentences.append(f"Sai số thấp nhất là {x[min_idx] if x else 'một quan sát'} với {_aideom_number(y[min_idx])}%.")
                sentences.append("Nếu sai số tập trung vào một năm cụ thể, phần hiệu chỉnh chính sách nên chú ý các cú sốc vĩ mô hoặc thay đổi dữ liệu trong năm đó.")
        elif "thực tế" in lowered and "dự báo" in lowered:
            actual = next((t for t in traces if "thực" in str(getattr(t, "name", "")).lower()), None)
            forecast = next((t for t in traces if "dự" in str(getattr(t, "name", "")).lower() or "forecast" in str(getattr(t, "name", "")).lower()), None)
            actual_y = _aideom_values(getattr(actual, "y", None))
            forecast_y = _aideom_values(getattr(forecast, "y", None))
            x = _aideom_labels(getattr(actual or forecast, "x", None))
            if actual_y and forecast_y:
                n = min(len(actual_y), len(forecast_y))
                errors = [abs((actual_y[i] - forecast_y[i]) / actual_y[i]) * 100 for i in range(n) if actual_y[i]]
                mape = sum(errors) / len(errors) if errors else 0
                sentences.append(f"Đường dự báo bám khá sát chuỗi thực tế với MAPE khoảng {_aideom_number(mape)}%.")
                if errors:
                    max_idx = max(range(len(errors)), key=lambda i: errors[i])
                    sentences.append(f"Năm lệch nhiều nhất là {x[max_idx] if x else 'một quan sát'} với sai số {_aideom_number(errors[max_idx])}%.")
                sentences.append("Mức khớp này cho thấy mô hình có thể dùng để mô phỏng kịch bản, nhưng vẫn cần đọc kết quả như công cụ hỗ trợ chính sách thay vì dự báo tuyệt đối.")
        elif "dự báo gdp" in lowered:
            trace = traces[-1]
            y = _aideom_values(getattr(trace, "y", None))
            x = _aideom_labels(getattr(trace, "x", None))
            if y:
                sentences.append(f"GDP dự báo đạt {_aideom_number(y[-1])} vào {x[-1] if x else 'cuối kỳ dự báo'}.")
                if len(y) >= 2:
                    trend = "tăng" if y[-1] >= y[0] else "giảm"
                    sentences.append(f"Chuỗi dự báo có xu hướng {trend} trong giai đoạn hiển thị, phản ánh giả định tăng vốn, TFP và số hóa trong mô hình.")
                sentences.append("Để đạt quỹ đạo này, chính sách cần giữ nhịp đầu tư số, mở rộng năng lực AI và cải thiện chất lượng nhân lực số.")
        elif "đóng góp" in lowered or "tỷ trọng" in lowered:
            trace = _chart_primary_trace(fig)
            y = _aideom_values(getattr(trace, "y", None))
            x = _aideom_labels(getattr(trace, "x", None))
            if y:
                top_idx = max(range(len(y)), key=lambda i: y[i])
                low_idx = min(range(len(y)), key=lambda i: y[i])
                sentences.append(f"Yếu tố nổi bật nhất là {x[top_idx] if x else 'hạng mục dẫn đầu'} với tỷ trọng {_aideom_number(y[top_idx])}%.")
                sentences.append(f"Yếu tố thấp nhất là {x[low_idx] if x else 'hạng mục cuối'} với {_aideom_number(y[low_idx])}%.")
                sentences.append("Kết quả này giúp ưu tiên nguồn lực vào biến có đóng góp lớn, đồng thời kiểm tra lại các biến có đóng góp âm hoặc quá thấp.")
        else:
            trace = _chart_primary_trace(fig)
            if trace is None:
                return
            y = _aideom_values(getattr(trace, "y", None))
            labels = _aideom_labels(getattr(trace, "x", None))
            if not y:
                y = _aideom_values(getattr(trace, "values", None))
                labels = _aideom_labels(getattr(trace, "labels", None))
            if y:
                top_idx = max(range(len(y)), key=lambda i: y[i])
                low_idx = min(range(len(y)), key=lambda i: y[i])
                sentences.append(f"Giá trị nổi bật nhất là {labels[top_idx] if labels else 'nhóm dẫn đầu'} với {_aideom_number(y[top_idx])}.")
                if len(y) >= 2:
                    sentences.append(f"Nhóm thấp nhất là {labels[low_idx] if labels else 'nhóm cuối'} với {_aideom_number(y[low_idx])}.")
                sentences.append("Chênh lệch giữa nhóm cao và thấp là tín hiệu cần chú ý khi phân bổ ngân sách hoặc thiết kế ưu tiên chính sách.")

        _aideom_note_card("Nhận xét kết quả", sentences)
    except Exception:
        return


def _render_dataframe_result_note(data) -> None:
    try:
        if st.session_state.get("current_page", "home") == "home":
            return
        try:
            import pandas as pd
        except Exception:
            return
        frame = getattr(data, "data", data)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return
        numeric = frame.select_dtypes(include="number")
        if numeric.empty:
            return
        columns = [col for col in numeric.columns if str(col).lower() not in {"năm", "year"}]
        if not columns:
            columns = list(numeric.columns)
        score_cols = [col for col in columns if any(key in str(col).lower() for key in ["score", "priority", "gdp", "npv", "readiness", "gain", "netjob"])]
        col = score_cols[0] if score_cols else max(columns, key=lambda c: float(numeric[c].max() - numeric[c].min()))
        series = numeric[col].dropna()
        if series.empty:
            return
        top_idx = series.idxmax()
        low_idx = series.idxmin()
        top_label = str(top_idx)
        low_label = str(low_idx)
        trend_sentence = ""
        if len(series) >= 2:
            first = float(series.iloc[0])
            last = float(series.iloc[-1])
            trend = "tăng" if last >= first else "giảm"
            trend_sentence = f"Cột {col} có xu hướng {trend} từ {_aideom_number(first)} lên {_aideom_number(last)} theo thứ tự hiển thị." if last >= first else f"Cột {col} có xu hướng {trend} từ {_aideom_number(first)} xuống {_aideom_number(last)} theo thứ tự hiển thị."
        sentences = [
            f"Bảng hiện có {len(frame)} dòng và {len(frame.columns)} cột, trong đó chỉ tiêu nổi bật để đọc nhanh là {col}.",
            f"Giá trị cao nhất của {col} thuộc về {top_label} với {_aideom_number(series.loc[top_idx])}; thấp nhất là {low_label} với {_aideom_number(series.loc[low_idx])}.",
        ]
        if trend_sentence:
            sentences.append(trend_sentence)
        sentences.append("Về chính sách, khoảng cách giữa nhóm cao và thấp cho thấy cần ưu tiên nguồn lực hoặc kiểm tra nguyên nhân chênh lệch trước khi ra quyết định.")
        _aideom_note_card("Nhận xét kết quả", sentences)
    except Exception:
        return


if not getattr(st, "_aideom_plotly_patched", False):
    _aideom_original_plotly_chart = st.plotly_chart

    def _aideom_plotly_chart(figure_or_data, *args, **kwargs):
        note_source = figure_or_data
        if hasattr(figure_or_data, "update_layout"):
            figure_or_data = _aideom_style_plotly(figure_or_data)
            note_source = figure_or_data
        elif isinstance(figure_or_data, (dict, list, tuple)):
            figure_or_data = _aideom_scrub_plotly_dict(figure_or_data)
        kwargs.setdefault("use_container_width", True)
        config = dict(kwargs.get("config") or {})
        config.setdefault("responsive", True)
        config.setdefault("displaylogo", False)
        kwargs["config"] = config
        result = _aideom_original_plotly_chart(figure_or_data, *args, **kwargs)
        if hasattr(note_source, "update_layout"):
            _render_plotly_result_note(note_source)
        return result

    st.plotly_chart = _aideom_plotly_chart
    st._aideom_plotly_patched = True


if not getattr(st, "_aideom_dataframe_patched", False):
    _aideom_original_dataframe = st.dataframe

    def _aideom_dataframe(data=None, *args, **kwargs):
        result = _aideom_original_dataframe(data, *args, **kwargs)
        _render_dataframe_result_note(data)
        return result

    st.dataframe = _aideom_dataframe
    st._aideom_dataframe_patched = True




PAGES = {
    "🏠 Trang chủ": "home",
    "📈 Bài 1 — Cobb-Douglas + AI": "bai01",
    "💰 Bài 2 — LP ngân sách số": "bai02",
    "🏭 Bài 3 — Priority 10 ngành": "bai03",
    "🗺️ Bài 4 — LP ngành-vùng": "bai04",
    "🎯 Bài 5 — MIP 15 dự án": "bai05",
    "🌍 Bài 6 — TOPSIS 6 vùng": "bai06",
    "⚡ Bài 7 — NSGA-II Pareto": "bai07",
    "📅 Bài 8 — Động 2026-2035": "bai08",
    "👷 Bài 9 — Lao động & AI": "bai09",
    "🎲 Bài 10 — Stochastic SP": "bai10",
    "🤖 Bài 11 — Q-learning RL": "bai11",
    "🇻🇳 Bài 12 — AIDEOM tích hợp": "bai12",
}

LABEL_BY_ID = {pid: label for label, pid in PAGES.items()}
LESSON_IDS = [f"bai{i:02d}" for i in range(1, 13)]
LESSON_TITLES = {
    "bai01": "Bài 1 — Cobb-Douglas + AI",
    "bai02": "Bài 2 — LP ngân sách số",
    "bai03": "Bài 3 — Priority 10 ngành",
    "bai04": "Bài 4 — LP ngành-vùng",
    "bai05": "Bài 5 — MIP 15 dự án",
    "bai06": "Bài 6 — TOPSIS 6 vùng",
    "bai07": "Bài 7 — NSGA-II Pareto",
    "bai08": "Bài 8 — Động 2026-2035",
    "bai09": "Bài 9 — Lao động & AI",
    "bai10": "Bài 10 — Stochastic SP",
    "bai11": "Bài 11 — Q-learning RL",
    "bai12": "🇻🇳 Bài 12 — AIDEOM tích hợp",
}


def set_page(page_id: str) -> None:
    st.session_state["pending_page"] = page_id


if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"
if "nav_radio" not in st.session_state:
    st.session_state["nav_radio"] = LABEL_BY_ID[st.session_state["current_page"]]
if "pending_page" in st.session_state:
    target_page = st.session_state.pop("pending_page")
    st.session_state["current_page"] = target_page
    st.session_state["nav_radio"] = LABEL_BY_ID.get(target_page, "🏠 Trang chủ")


st.markdown(
    """
<style>
:root{
  --cream:#fff9fb;
  --cream-2:#fff2f8;
  --pink:#ff9fcc;
  --pink-2:#ffd6e8;
  --baby:#ffc6dc;
  --rose:#ef6fa8;
  --lav:#d9c7ff;
  --lav-2:#b8a7f6;
  --purple:#8c63c7;
  --ink:#55364d;
  --muted:#8b6f82;
  --glass:rgba(255,255,255,.68);
  --line:rgba(239,111,168,.28);
}

html,body,[class*="css"]{
  font-family:"Inter","Segoe UI","Roboto","Arial",sans-serif;
}
.stApp{
  color:var(--ink);
  background:
    radial-gradient(circle at 9% 10%, rgba(255,159,204,.38), transparent 26%),
    radial-gradient(circle at 82% 14%, rgba(217,199,255,.45), transparent 30%),
    radial-gradient(circle at 72% 86%, rgba(255,198,220,.42), transparent 30%),
    linear-gradient(135deg,#fff9fb 0%,#ffeaf4 45%,#f1eaff 100%);
}
.stApp::before{
  content:"";
  position:fixed;
  inset:0;
  pointer-events:none;
  z-index:0;
  opacity:.58;
  background-image:
    radial-gradient(circle at 16px 16px, rgba(255,255,255,.92) 0 3px, transparent 4px),
    radial-gradient(circle at 58px 44px, rgba(255,159,204,.38) 0 2px, transparent 3px),
    linear-gradient(135deg, transparent 0 48%, rgba(255,159,204,.14) 49% 51%, transparent 52%),
    linear-gradient(90deg, rgba(255,159,204,.10) 1px, transparent 1px),
    linear-gradient(rgba(217,199,255,.16) 1px, transparent 1px);
  background-size:96px 96px,120px 120px,190px 190px,56px 56px,56px 56px;
}
.main .block-container{
  position:relative;
  z-index:1;
  max-width:1460px;
  padding:1.35rem 2rem 2.5rem;
}

[data-testid="stSidebar"]{
  background:
    radial-gradient(circle at 18% 0%, rgba(255,255,255,.92), transparent 30%),
    linear-gradient(180deg,rgba(255,244,250,.96),rgba(244,235,255,.96));
  border-right:1px solid rgba(239,111,168,.25);
  box-shadow:10px 0 35px rgba(198,132,170,.18);
}
[data-testid="stSidebar"]::before{
  content:"";
  position:absolute;
  inset:0;
  pointer-events:none;
  background:
    radial-gradient(circle at 24px 22px, rgba(255,159,204,.28) 0 5px, transparent 6px),
    radial-gradient(circle at 80px 70px, rgba(217,199,255,.34) 0 4px, transparent 5px);
  background-size:118px 118px;
  opacity:.42;
}
.sidebar-logo{
  text-align:center;
  padding:1rem .45rem 1.25rem;
  margin-bottom:.85rem;
  border-bottom:1px dashed rgba(239,111,168,.28);
}
.sidebar-logo .mark{
  width:64px;height:64px;margin:0 auto .55rem;
  border-radius:24px;
  display:grid;place-items:center;
  font-size:1.9rem;
  background:linear-gradient(135deg,#fff,#ffe1ef 48%,#eadfff);
  border:1px solid rgba(239,111,168,.35);
  box-shadow:0 14px 32px rgba(239,111,168,.18), inset 0 1px 0 rgba(255,255,255,.9);
}
.sidebar-logo .title{
  font-weight:900;
  font-size:1.02rem;
  letter-spacing:.3px;
  color:#6d3f61;
}
.sidebar-logo .slogan{
  color:#9a748f;
  font-size:.70rem;
  line-height:1.55;
  margin-top:.35rem;
}
.nav-caption{
  color:#b76b97;
  font-size:.68rem;
  font-weight:900;
  letter-spacing:1.8px;
  margin:.5rem 0 .45rem;
}
.stRadio>div{gap:.32rem;}
.stRadio>div>label{
  background:rgba(255,255,255,.62);
  border:1px solid rgba(239,111,168,.16);
  border-radius:16px;
  padding:8px 12px;
  color:#74536b!important;
  font-size:.82rem;
  cursor:pointer;
  transition:all .18s ease;
  box-shadow:0 7px 18px rgba(198,132,170,.06);
}
.stRadio>div>label:hover{
  background:#fff;
  border-color:rgba(239,111,168,.36);
  color:#d94f91!important;
  transform:translateY(-1px);
  box-shadow:0 10px 24px rgba(239,111,168,.14);
}
.stRadio>div>label:has(input:checked){
  background:linear-gradient(135deg,#fff,#ffe0ef 48%,#eee6ff);
  border-color:rgba(239,111,168,.52);
  color:#b83e7b!important;
  box-shadow:0 0 0 3px rgba(255,214,232,.75),0 12px 28px rgba(239,111,168,.18);
}
.stRadio>div>label:has(input:checked) p{color:#b83e7b!important;font-weight:900;}

.policy-hero{
  position:relative;
  overflow:hidden;
  min-height:440px;
  border-radius:34px;
  padding:4.4rem 2rem 3.2rem;
  text-align:center;
  display:flex;
  justify-content:center;
  align-items:center;
  border:1px solid rgba(239,111,168,.28);
  background:
    radial-gradient(circle at 18% 22%, rgba(255,255,255,.98), transparent 20%),
    radial-gradient(circle at 80% 24%, rgba(217,199,255,.72), transparent 28%),
    linear-gradient(135deg,#fff8fb 0%,#ffd9ea 42%,#eadfff 100%);
  box-shadow:0 30px 80px rgba(198,132,170,.28), inset 0 2px 0 rgba(255,255,255,.9);
}
.policy-hero::before{
  content:"";
  position:absolute;
  inset:0;
  opacity:.75;
  background:
    radial-gradient(circle at 30px 0, #ffc6dc 0 30px, transparent 31px) 0 0/60px 34px repeat-x,
    radial-gradient(circle at 12% 72%, rgba(255,255,255,.95) 0 10px, transparent 11px),
    radial-gradient(circle at 13.5% 70%, rgba(255,255,255,.95) 0 10px, transparent 11px),
    radial-gradient(circle at 12.7% 68.4%, #ffe077 0 5px, transparent 6px),
    radial-gradient(circle at 88% 62%, rgba(255,255,255,.95) 0 13px, transparent 14px),
    radial-gradient(circle at 90% 62%, rgba(255,255,255,.95) 0 13px, transparent 14px),
    radial-gradient(circle at 89% 60%, rgba(255,159,204,.95) 0 5px, transparent 6px),
    radial-gradient(circle at 22% 28%, rgba(255,255,255,.9) 0 3px, transparent 4px),
    radial-gradient(circle at 78% 38%, rgba(255,255,255,.9) 0 3px, transparent 4px);
  background-repeat:repeat-x,no-repeat,no-repeat,no-repeat,no-repeat,no-repeat,no-repeat,no-repeat,no-repeat;
}
.policy-hero::after{
  content:"♡";
  position:absolute;
  right:9%;
  bottom:11%;
  font-size:5.4rem;
  color:rgba(255,255,255,.72);
  text-shadow:0 8px 24px rgba(239,111,168,.20);
  transform:rotate(-12deg);
}
.hero-content{position:relative;z-index:1;max-width:980px;margin:0 auto;}
.hero-kicker{
  display:inline-flex;
  align-items:center;
  justify-content:center;
  color:#c04d86;
  font-weight:900;
  font-size:.76rem;
  letter-spacing:1.8px;
  text-transform:uppercase;
  padding:.45rem .85rem;
  border-radius:999px;
  background:rgba(255,255,255,.72);
  border:1px solid rgba(239,111,168,.25);
  box-shadow:0 10px 26px rgba(239,111,168,.14);
}
.hero-title-main{
  margin:1rem 0 .65rem;
  font-size:clamp(2.45rem,6vw,5.4rem);
  line-height:.95;
  font-weight:950;
  letter-spacing:0;
  color:#7a4070;
  text-shadow:0 5px 0 rgba(255,255,255,.82),0 16px 34px rgba(239,111,168,.18);
}
.hero-subtitle{
  color:#6f5368;
  font-size:1.08rem;
  line-height:1.72;
  max-width:820px;
  margin:0 auto .65rem;
}
.hero-author{color:#9a6f92;font-size:.88rem;font-weight:800;letter-spacing:.3px;}
.hero-actions{
  margin-top:1.45rem;
  display:grid;
  grid-template-columns:repeat(2,minmax(220px,1fr));
  gap:1rem;
}
.hero-action-card{
  text-align:left;
  padding:1rem 1.1rem;
  border-radius:24px;
  background:rgba(255,255,255,.72);
  border:1px solid rgba(239,111,168,.22);
  box-shadow:0 16px 34px rgba(198,132,170,.16), inset 0 1px 0 rgba(255,255,255,.9);
}
.hero-action-card h4{margin:.15rem 0 .35rem;color:#7a4070;font-size:1rem;}
.hero-action-card p{margin:0;color:#8b6f82;font-size:.82rem;line-height:1.55;}

.stButton>button{
  background:linear-gradient(135deg,#ff9fcc,#d9c7ff);
  color:#fff!important;
  border:1px solid rgba(255,255,255,.7);
  border-radius:999px;
  font-weight:900;
  transition:all .18s ease;
  box-shadow:0 12px 24px rgba(239,111,168,.22), inset 0 1px 0 rgba(255,255,255,.7);
}
.stButton>button:hover{
  transform:translateY(-1px);
  background:linear-gradient(135deg,#ef6fa8,#b8a7f6);
  box-shadow:0 16px 30px rgba(239,111,168,.28);
}
.stButton>button:disabled{opacity:.45;box-shadow:none;transform:none;}

[data-testid="metric-container"]{
  background:rgba(255,255,255,.72);
  border:1px solid rgba(239,111,168,.20);
  border-radius:24px;
  padding:1rem;
  box-shadow:0 16px 34px rgba(198,132,170,.13), inset 0 1px 0 rgba(255,255,255,.9);
}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#c04d86!important;font-weight:950;}
[data-testid="metric-container"] label{color:#8b6f82!important;}

.stTabs [data-baseweb="tab-list"]{
  background:rgba(255,255,255,.62);
  border-radius:999px;
  padding:5px;
  gap:5px;
  border:1px solid rgba(239,111,168,.16);
}
.stTabs [data-baseweb="tab"]{
  border-radius:999px;
  color:#9a748f;
  font-size:.84rem;
  font-weight:800;
  padding:8px 15px;
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,#ffe0ef,#eee6ff);
  color:#b83e7b!important;
  border:1px solid rgba(239,111,168,.30);
  box-shadow:0 8px 18px rgba(239,111,168,.13);
}

.glass-panel,[data-testid="stExpander"],.mod-card,.page-header,.math-box,.ai-box,.lesson-nav-shell{
  background:rgba(255,255,255,.72)!important;
  border:1px solid rgba(239,111,168,.20)!important;
  box-shadow:0 16px 34px rgba(198,132,170,.13), inset 0 1px 0 rgba(255,255,255,.9)!important;
}
.glass-panel,[data-testid="stExpander"],.page-header,.math-box,.ai-box,.lesson-nav-shell{border-radius:24px!important;}
.page-header *{color:#714568!important;}
.page-header [style*="color:#00D4FF"],.page-header [style*="color:#E2E8F0"]{color:#b83e7b!important;}
.math-box{
  padding:.95rem 1.25rem;
  margin:.75rem 0;
  font-family:"Inter","Segoe UI","Roboto","Arial",sans-serif;
  font-size:.86rem;
  color:#674c60;
  line-height:1.9;
}
.ai-box{padding:1.2rem 1.5rem;margin:1rem 0;color:#674c60;}
.ai-box *{color:#674c60!important;}
.ai-box div:first-child{color:#b83e7b!important;}
.badge{
  display:inline-block;
  padding:3px 10px;
  border-radius:999px;
  font-size:.72rem;
  font-weight:900;
  margin:2px;
}
.b-blue,.b-purple,.b-green,.b-orange{
  background:linear-gradient(135deg,#fff,#ffe0ef)!important;
  color:#b83e7b!important;
  border:1px solid rgba(239,111,168,.25)!important;
}
.home-section-title{
  color:#7a4070;
  font-size:1.28rem;
  font-weight:950;
  margin:1.55rem 0 .85rem;
}
.mod-card{
  border-radius:26px!important;
  padding:1.05rem;
  transition:all .18s ease;
  height:100%;
}
.mod-card:hover{
  transform:translateY(-3px) rotate(-.2deg);
  border-color:rgba(239,111,168,.42)!important;
  box-shadow:0 20px 40px rgba(239,111,168,.18)!important;
}
.mod-card h4{color:#7a4070;margin:4px 0;font-size:.94rem;}
.mod-card p{color:#8b6f82;margin:0;font-size:.78rem;line-height:1.5;}
.lesson-nav-shell{padding:.6rem .9rem;margin:.15rem 0 1rem;}
.lesson-nav-bottom{margin:1.4rem 0 .3rem;}
.nav-current{text-align:center;color:#7a4070;font-weight:950;padding:.45rem .6rem;}
.nav-current span{
  display:block;
  color:#c06b9d;
  font-size:.70rem;
  font-weight:900;
  letter-spacing:1.4px;
  text-transform:uppercase;
  margin-bottom:.12rem;
}

div[data-testid="stDataFrame"],div[data-testid="stTable"]{
  border-radius:18px;
  overflow:hidden;
}
hr{border-color:rgba(239,111,168,.18)!important;}
a{color:#b83e7b!important;}

@media (max-width:760px){
  .main .block-container{padding:1rem .85rem 2rem;}
  .policy-hero{min-height:390px;padding:3rem 1rem 2rem;border-radius:24px;}
  .hero-actions{grid-template-columns:1fr;}
}


/* Contrast pass: keep pastel/kawaii layout, sharpen readability */
.stApp,
.stApp p,
.stApp li,
.stApp div,
.stApp span,
.stApp label,
.stMarkdown,
.stMarkdown p,
[data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li{
  color:#4A2B45!important;
  font-weight:600;
}
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4{
  color:#2B1230!important;
  font-weight:800!important;
}
.stCaptionContainer,
small,
.hero-subtitle,
.hero-author,
.hero-action-card p,
.mod-card p,
.sidebar-logo .slogan,
.glass-panel div,
[data-testid="stAlert"] p,
[data-testid="stAlert"] div{
  color:#5A3A55!important;
  font-weight:600!important;
}
.sidebar-logo .title,
.nav-caption,
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label{
  color:#4A2B45!important;
  font-weight:700!important;
  opacity:1!important;
}
.stRadio>div>label,
.stRadio>div>label p{
  color:#4A2B45!important;
  font-weight:700!important;
  opacity:1!important;
}
.stRadio>div>label:hover,
.stRadio>div>label:hover p,
.stRadio>div>label:has(input:checked),
.stRadio>div>label:has(input:checked) p{
  color:#2B1230!important;
  font-weight:800!important;
}
.policy-hero .hero-kicker,
.policy-hero .hero-title-main,
.policy-hero .hero-subtitle,
.policy-hero .hero-author,
.hero-action-card h4,
.hero-action-card p{
  opacity:1!important;
}
.hero-title-main{
  color:#2B1230!important;
  font-weight:900!important;
  text-shadow:0 3px 0 rgba(255,255,255,.72),0 10px 22px rgba(107,49,91,.16)!important;
}
.hero-kicker{color:#7A245F!important;font-weight:800!important;}
.hero-subtitle{color:#3B1F35!important;font-weight:600!important;}
.hero-author{color:#4A2B45!important;font-weight:700!important;}
.hero-action-card h4{color:#2B1230!important;font-weight:800!important;}
.hero-action-card p{color:#4A2B45!important;font-weight:600!important;}
.mod-card h4,
.home-section-title,
.page-header *{
  color:#2B1230!important;
  font-weight:800!important;
}
.mod-card p,
.math-box,
.math-box *,
.ai-box,
.ai-box *,
.nav-current,
.nav-current span{
  color:#4A2B45!important;
  font-weight:600!important;
}
.nav-current{font-weight:800!important;}
.nav-current span{color:#5A3A55!important;font-weight:800!important;}
.badge,
.b-blue,
.b-purple,
.b-green,
.b-orange{
  color:#7A245F!important;
  font-weight:800!important;
}
[data-testid="metric-container"],
[data-testid="metric-container"] *,
[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricLabel"],
[data-testid="metric-container"] [data-testid="stMetricDelta"]{
  color:#4A2B45!important;
  font-weight:700!important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"]{
  color:#8B1E62!important;
  font-weight:900!important;
}
.stTabs [data-baseweb="tab"],
.stTabs [data-baseweb="tab"] p{
  color:#4A2B45!important;
  font-weight:700!important;
}
.stTabs [aria-selected="true"],
.stTabs [aria-selected="true"] p{
  color:#2B1230!important;
  font-weight:800!important;
}
.stButton>button,
.stButton>button p,
.stButton>button span{
  color:#3B1F35!important;
  font-weight:800!important;
}
.stDataFrame,
.stDataFrame *,
div[data-testid="stDataFrame"] *,
div[data-testid="stTable"] *{
  color:#3B1F35!important;
  font-weight:500!important;
}
input, textarea, select,
[data-baseweb="input"] *,
[data-baseweb="select"] *,
[data-baseweb="slider"] *{
  color:#3B1F35!important;
  font-weight:600!important;
}



/* AIDEOM v5: futuristic + kawaii + academic dashboard polish */
.stApp{
  background:
    radial-gradient(circle at 10% 8%, rgba(255,122,184,.32), transparent 24%),
    radial-gradient(circle at 86% 12%, rgba(0,229,255,.28), transparent 28%),
    radial-gradient(circle at 72% 84%, rgba(124,58,237,.26), transparent 30%),
    linear-gradient(135deg,#08122D 0%,#101744 42%,#2A1642 100%)!important;
}
.stApp::before{
  opacity:.48!important;
  background-image:
    linear-gradient(rgba(0,229,255,.13) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,122,184,.11) 1px, transparent 1px),
    linear-gradient(135deg, transparent 0 48%, rgba(167,139,250,.18) 49% 51%, transparent 52%),
    radial-gradient(circle at 24px 24px, rgba(255,255,255,.75) 0 2px, transparent 3px)!important;
  background-size:54px 54px,54px 54px,180px 180px,108px 108px!important;
}
html,body,[class*="css"],.stApp{
  font-family:"Inter","Be Vietnam Pro","Segoe UI",sans-serif!important;
}
.main .block-container{padding-top:1.15rem!important;}

[data-testid="stSidebar"]{
  background:
    radial-gradient(circle at 20% 4%, rgba(0,229,255,.20), transparent 24%),
    linear-gradient(180deg,rgba(255,248,252,.94),rgba(239,233,255,.92))!important;
  border-right:1px solid rgba(0,229,255,.28)!important;
  box-shadow:12px 0 42px rgba(0,0,0,.28),0 0 28px rgba(255,122,184,.12)!important;
}
.sidebar-logo .mark{
  background:linear-gradient(135deg,#fff,#ffd8eb 48%,#dff7ff)!important;
  border:1px solid rgba(0,229,255,.34)!important;
  box-shadow:0 0 28px rgba(0,229,255,.17),0 18px 35px rgba(255,122,184,.18)!important;
}
.sidebar-logo .title,.sidebar-logo .slogan,.nav-caption,[data-testid="stSidebar"] *{
  color:#3B1F35!important;
  opacity:1!important;
}
.sidebar-logo .title{font-weight:900!important;}
.sidebar-logo .slogan{font-weight:700!important;}
.stRadio>div>label{
  background:rgba(255,255,255,.76)!important;
  border:1px solid rgba(124,58,237,.18)!important;
  box-shadow:0 10px 24px rgba(42,22,66,.08)!important;
}
.stRadio>div>label:hover{
  border-color:rgba(0,229,255,.58)!important;
  box-shadow:0 0 0 3px rgba(0,229,255,.12),0 16px 30px rgba(255,122,184,.15)!important;
}
.stRadio>div>label:has(input:checked){
  border:1px solid rgba(0,229,255,.70)!important;
  background:linear-gradient(135deg,#fff,#ffe1ef 45%,#e2f8ff)!important;
  box-shadow:0 0 0 3px rgba(0,229,255,.18),0 0 22px rgba(255,122,184,.22)!important;
}

.policy-hero{
  min-height:480px!important;
  border-radius:32px!important;
  border:1px solid rgba(0,229,255,.34)!important;
  background:
    radial-gradient(circle at 16% 18%, rgba(255,255,255,.94), transparent 17%),
    radial-gradient(circle at 78% 16%, rgba(0,229,255,.32), transparent 28%),
    radial-gradient(circle at 56% 88%, rgba(255,122,184,.34), transparent 28%),
    linear-gradient(135deg,rgba(255,248,252,.94),rgba(224,247,255,.82) 48%,rgba(238,230,255,.88))!important;
  box-shadow:0 32px 90px rgba(0,0,0,.32),0 0 42px rgba(0,229,255,.16), inset 0 1px 0 rgba(255,255,255,.92)!important;
}
.policy-hero::before{
  background:
    radial-gradient(circle at 30px 0, #ffc6dc 0 30px, transparent 31px) 0 0/60px 34px repeat-x,
    linear-gradient(rgba(0,229,255,.17) 1px, transparent 1px),
    linear-gradient(90deg, rgba(124,58,237,.14) 1px, transparent 1px),
    linear-gradient(135deg, transparent 0 47%, rgba(255,122,184,.16) 48% 52%, transparent 53%)!important;
  background-size:60px 34px,48px 48px,48px 48px,170px 170px!important;
  opacity:.62!important;
}
.kawaii-float{
  position:absolute;
  z-index:1;
  width:58px;
  height:58px;
  border-radius:22px;
  display:grid;
  place-items:center;
  background:rgba(255,255,255,.68);
  border:1px solid rgba(255,122,184,.24);
  box-shadow:0 18px 36px rgba(42,22,66,.13), inset 0 1px 0 rgba(255,255,255,.9);
  font-size:2rem;
  animation:aideomFloat 5.5s ease-in-out infinite;
  backdrop-filter:blur(12px);
}
.float-donut{left:6%;top:18%;animation-delay:0s;}
.float-croissant{right:9%;top:23%;animation-delay:1.2s;}
.float-cupcake{left:12%;bottom:15%;animation-delay:2.1s;}
.float-candy{right:16%;bottom:16%;animation-delay:3s;}
@keyframes aideomFloat{
  0%,100%{transform:translateY(0) rotate(-4deg);}
  50%{transform:translateY(-14px) rotate(5deg);}
}
.hero-title-main{
  color:#2B1230!important;
  letter-spacing:0!important;
  text-shadow:0 3px 0 rgba(255,255,255,.8),0 0 28px rgba(0,229,255,.18),0 14px 28px rgba(124,58,237,.12)!important;
}
.hero-kicker{
  color:#2B1230!important;
  background:rgba(255,255,255,.78)!important;
  border:1px solid rgba(0,229,255,.28)!important;
}

.glass-panel,[data-testid="stExpander"],.mod-card,.page-header,.math-box,.ai-box,.lesson-nav-shell,
[data-testid="metric-container"],.hero-action-card,[data-testid="stAlert"]{
  background:rgba(255,255,255,.58)!important;
  backdrop-filter:blur(18px)!important;
  -webkit-backdrop-filter:blur(18px)!important;
  border:1px solid rgba(255,255,255,.44)!important;
  box-shadow:0 20px 45px rgba(8,18,45,.22),0 0 22px rgba(0,229,255,.08), inset 0 1px 0 rgba(255,255,255,.85)!important;
}
.mod-card,.page-header,.lesson-nav-shell,[data-testid="metric-container"],.hero-action-card{border-radius:24px!important;}
.mod-card:hover{
  transform:translateY(-4px)!important;
  box-shadow:0 24px 52px rgba(8,18,45,.26),0 0 30px rgba(255,122,184,.18)!important;
}
.section-chip{
  background:linear-gradient(135deg,rgba(0,229,255,.12),rgba(255,122,184,.12));
  border:1px solid rgba(0,229,255,.22);
}

.ai-agent-card,.agent-teaser,.debug-card,.status-card{
  background:rgba(255,255,255,.55);
  backdrop-filter:blur(18px);
  -webkit-backdrop-filter:blur(18px);
  border:1px solid rgba(255,255,255,.48);
  border-radius:24px;
  padding:1.15rem 1.35rem;
  margin:1rem 0;
  color:#3B1F35!important;
  box-shadow:0 20px 46px rgba(8,18,45,.20),0 0 24px rgba(255,122,184,.12), inset 0 1px 0 rgba(255,255,255,.85);
}
.ai-agent-card .agent-title,.agent-teaser .agent-title,.debug-card .agent-title,.status-card .agent-title{
  color:#2B1230!important;
  font-weight:900!important;
  font-size:1rem;
  margin-bottom:.55rem;
}
.ai-agent-card .agent-body,.agent-teaser .agent-body,.debug-card .agent-body,.status-card .agent-body{
  color:#4A2B45!important;
  font-weight:600!important;
  line-height:1.72;
  white-space:normal;
}
.debug-card .agent-body{white-space:pre-wrap;font-family:"Inter","Be Vietnam Pro","Segoe UI",sans-serif;font-size:.84rem;}

.stButton>button{
  background:linear-gradient(135deg,#00E5FF,#FF7AB8 52%,#A78BFA)!important;
  color:#1F1026!important;
  border:1px solid rgba(255,255,255,.66)!important;
  box-shadow:0 14px 30px rgba(8,18,45,.22),0 0 20px rgba(0,229,255,.12)!important;
}
.stButton>button:hover{
  box-shadow:0 18px 36px rgba(8,18,45,.26),0 0 26px rgba(255,122,184,.20)!important;
}

[data-testid="stDataFrame"],div[data-testid="stTable"]{
  border-radius:22px!important;
  border:1px solid rgba(255,255,255,.44)!important;
  box-shadow:0 18px 40px rgba(8,18,45,.18)!important;
  overflow:hidden!important;
}
div[data-testid="stDataFrame"] [role="row"]:nth-child(even){
  background:rgba(255,122,184,.06)!important;
}
div[data-testid="stDataFrame"] [role="columnheader"]{
  background:linear-gradient(135deg,rgba(0,229,255,.12),rgba(255,122,184,.12))!important;
  color:#2B1230!important;
  font-weight:800!important;
}
code,pre{
  background:rgba(255,255,255,.56)!important;
  color:#3B1F35!important;
  border:1px solid rgba(255,122,184,.24)!important;
  border-radius:14px!important;
}
.stApp,.stApp p,.stApp li,.stApp div,.stApp span,.stApp label,[data-testid="stMarkdownContainer"],[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li{
  color:#3B1F35!important;
  font-weight:600!important;
}
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp h6,[data-testid="stMarkdownContainer"] h1,[data-testid="stMarkdownContainer"] h2,[data-testid="stMarkdownContainer"] h3,[data-testid="stMarkdownContainer"] h4{
  color:#2B1230!important;
  font-weight:850!important;
}
.stCaptionContainer,small,.hero-subtitle,.hero-author,.mod-card p,.hero-action-card p{
  color:#4A2B45!important;
  font-weight:650!important;
}

/* AIDEOM v6: pastel kawaii + academic dashboard */
:root{
  --accent-a:#F472B6;
  --accent-b:#A78BFA;
  --accent-c:#67E8F9;
  --ink-strong:#2B1230;
  --ink-main:#3B1F35;
  --ink-soft:#5A3A55;
  --card-glass:rgba(255,255,255,0.75);
  --card-glass-strong:rgba(255,255,255,0.86);
  --pink-line:rgba(244,114,182,0.28);
  --lav-line:rgba(167,139,250,0.26);
}
html,body,[class*="css"],.stApp{
  font-family:"Be Vietnam Pro","Inter","Segoe UI","Arial",sans-serif!important;
}
.stApp{
  color:var(--ink-main)!important;
  background:
    radial-gradient(circle at 8% 10%, rgba(255,205,226,.78), transparent 28%),
    radial-gradient(circle at 82% 12%, rgba(226,214,255,.78), transparent 30%),
    radial-gradient(circle at 72% 86%, rgba(205,246,255,.52), transparent 30%),
    linear-gradient(135deg,#fff7fb 0%,#ffeaf4 38%,#f2ecff 72%,#ecfbff 100%)!important;
}
.stApp::before{
  content:""!important;
  position:fixed!important;
  inset:0!important;
  pointer-events:none!important;
  z-index:0!important;
  opacity:.55!important;
  background-image:
    radial-gradient(circle at 18px 18px, rgba(255,255,255,.95) 0 3px, transparent 4px),
    radial-gradient(circle at 74px 48px, rgba(244,114,182,.24) 0 3px, transparent 4px),
    linear-gradient(90deg, rgba(244,114,182,.08) 1px, transparent 1px),
    linear-gradient(rgba(167,139,250,.10) 1px, transparent 1px),
    linear-gradient(135deg, transparent 0 48%, rgba(103,232,249,.13) 49% 51%, transparent 52%)!important;
  background-size:104px 104px,132px 132px,58px 58px,58px 58px,210px 210px!important;
}
.stApp::after{
  content:"🍩   🥐   🧁   🍪   ♡   🍬"!important;
  position:fixed!important;
  left:4vw!important;
  right:4vw!important;
  top:11vh!important;
  z-index:0!important;
  pointer-events:none!important;
  color:rgba(122,64,112,.13)!important;
  font-size:clamp(2.6rem,7vw,6rem)!important;
  letter-spacing:clamp(.75rem,4vw,4rem)!important;
  line-height:1.9!important;
  opacity:.12!important;
  transform:rotate(-8deg)!important;
  filter:blur(.1px)!important;
}
.main .block-container{
  position:relative!important;
  z-index:1!important;
  padding-top:1.25rem!important;
}
.stApp,.stApp p,.stApp li,.stApp div,.stApp span,.stApp label,
[data-testid="stMarkdownContainer"],[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li{
  color:var(--ink-main)!important;
  font-weight:600!important;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li{
  font-size:18px!important;
  line-height:1.75!important;
}
.stApp h1,.stApp h2,.stApp h3,.stApp h4,.stApp h5,.stApp h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4{
  color:var(--ink-strong)!important;
  font-weight:850!important;
  letter-spacing:0!important;
}
.stApp h1,[data-testid="stMarkdownContainer"] h1{
  font-size:clamp(2.65rem,4.8vw,3.5rem)!important;
}
.stApp h2,[data-testid="stMarkdownContainer"] h2{
  font-size:clamp(1.65rem,3vw,2rem)!important;
}
.stApp h3,[data-testid="stMarkdownContainer"] h3{
  font-size:clamp(1.45rem,2.3vw,1.75rem)!important;
}
.stCaptionContainer,small,.hero-subtitle,.hero-author,.mod-card p,.hero-action-card p,
[data-testid="stCaptionContainer"] *{
  color:var(--ink-soft)!important;
  opacity:1!important;
  font-weight:650!important;
}

[data-testid="stSidebar"]{
  background:
    radial-gradient(circle at 22% 2%, rgba(255,255,255,.92), transparent 22%),
    radial-gradient(circle at 88% 28%, rgba(103,232,249,.20), transparent 28%),
    linear-gradient(180deg,rgba(255,247,251,.96),rgba(247,238,255,.95) 58%,rgba(240,252,255,.94))!important;
  border-right:1px solid rgba(244,114,182,.25)!important;
  box-shadow:10px 0 36px rgba(122,64,112,.12)!important;
}
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label{
  color:var(--ink-main)!important;
  opacity:1!important;
  font-size:18px!important;
  font-weight:700!important;
}
.sidebar-logo{
  background:rgba(255,255,255,.72)!important;
  border:1px solid rgba(244,114,182,.22)!important;
  box-shadow:0 18px 40px rgba(244,114,182,.14)!important;
}
.sidebar-logo .mark{
  background:linear-gradient(135deg,#fff,#ffe0ef 45%,#dff9ff)!important;
  border:1px solid rgba(244,114,182,.30)!important;
  box-shadow:0 12px 28px rgba(244,114,182,.18)!important;
}
.sidebar-logo .title{
  color:var(--ink-strong)!important;
  font-size:1.02rem!important;
  font-weight:900!important;
}
.sidebar-logo .slogan{
  color:var(--ink-soft)!important;
  font-size:.84rem!important;
  font-weight:750!important;
  line-height:1.55!important;
}
.nav-caption{
  color:#B83280!important;
  font-size:.84rem!important;
  font-weight:900!important;
}
.stRadio>div>label{
  min-height:44px!important;
  background:rgba(255,255,255,.78)!important;
  border:1px solid rgba(244,114,182,.18)!important;
  border-radius:18px!important;
  box-shadow:0 8px 20px rgba(122,64,112,.07)!important;
}
.stRadio>div>label p{
  color:var(--ink-main)!important;
  font-size:18px!important;
  font-weight:760!important;
}
.stRadio>div>label:hover{
  background:rgba(255,255,255,.94)!important;
  border-color:var(--accent-a)!important;
  box-shadow:0 0 0 3px color-mix(in srgb, var(--accent-a) 18%, transparent),0 14px 28px rgba(244,114,182,.12)!important;
}
.stRadio>div>label:has(input:checked){
  background:linear-gradient(135deg,rgba(255,255,255,.95),rgba(255,224,239,.92) 46%,rgba(239,249,255,.90))!important;
  border:2px solid var(--accent-a)!important;
  box-shadow:0 0 0 4px color-mix(in srgb, var(--accent-a) 18%, transparent),0 16px 32px rgba(244,114,182,.15)!important;
}
.stRadio>div>label:has(input:checked) p{
  color:var(--ink-strong)!important;
  font-weight:900!important;
}
.stRadio>div>label:nth-of-type(13){
  display:flex!important;
  align-items:center!important;
  gap:.55rem!important;
}
.stRadio>div>label:nth-of-type(13)::before{
  content:"★"!important;
  flex:0 0 auto!important;
  width:28px!important;
  height:19px!important;
  display:inline-grid!important;
  place-items:center!important;
  border-radius:5px!important;
  background:#DA251D!important;
  color:#FFDE00!important;
  font-size:11px!important;
  line-height:1!important;
  box-shadow:0 6px 14px rgba(218,37,29,.18), inset 0 0 0 1px rgba(255,255,255,.42)!important;
}

.policy-hero{
  min-height:500px!important;
  border-radius:34px!important;
  border:1px solid rgba(244,114,182,.28)!important;
  background:
    radial-gradient(circle at 16% 18%, rgba(255,255,255,.98), transparent 17%),
    radial-gradient(circle at 82% 18%, color-mix(in srgb, var(--accent-c) 35%, transparent), transparent 28%),
    radial-gradient(circle at 56% 88%, color-mix(in srgb, var(--accent-a) 30%, transparent), transparent 30%),
    linear-gradient(135deg,#fff9fc 0%,#ffe6f1 38%,#efe7ff 72%,#eefdff 100%)!important;
  box-shadow:0 32px 70px rgba(122,64,112,.18), inset 0 2px 0 rgba(255,255,255,.90)!important;
}
.policy-hero::before{
  opacity:.72!important;
  background:
    radial-gradient(circle at 30px 0, #ffcfe3 0 30px, transparent 31px) 0 0/60px 34px repeat-x,
    linear-gradient(90deg, rgba(244,114,182,.08) 1px, transparent 1px),
    linear-gradient(rgba(167,139,250,.10) 1px, transparent 1px),
    radial-gradient(circle at 18% 72%, rgba(255,255,255,.82) 0 10px, transparent 11px),
    radial-gradient(circle at 19.5% 70%, rgba(255,255,255,.82) 0 10px, transparent 11px),
    radial-gradient(circle at 18.8% 68.4%, #FDE68A 0 5px, transparent 6px)!important;
  background-size:60px 34px,52px 52px,52px 52px,auto,auto,auto!important;
}
.policy-hero::after{
  content:"♡"!important;
  color:rgba(244,114,182,.16)!important;
  text-shadow:none!important;
}
.hero-content{z-index:2!important;}
.kawaii-float{
  opacity:.15!important;
  background:rgba(255,255,255,.76)!important;
  border:1px solid rgba(244,114,182,.22)!important;
  box-shadow:0 16px 34px rgba(122,64,112,.10)!important;
}
.hero-kicker{
  color:#9D174D!important;
  background:rgba(255,255,255,.82)!important;
  border:1px solid rgba(244,114,182,.24)!important;
  font-size:.86rem!important;
  font-weight:900!important;
}
.hero-title-main{
  color:var(--ink-strong)!important;
  font-size:clamp(42px,6vw,56px)!important;
  line-height:1.03!important;
  font-weight:950!important;
  text-shadow:0 3px 0 rgba(255,255,255,.90),0 18px 34px rgba(244,114,182,.16)!important;
}
.hero-subtitle{
  color:var(--ink-soft)!important;
  font-size:20px!important;
  font-weight:700!important;
  line-height:1.75!important;
}
.hero-author{
  color:#8B1E62!important;
  font-size:16px!important;
  font-weight:900!important;
}
.hero-action-card,.mod-card,.glass-panel,.page-header,.math-box,.ai-box,
.lesson-nav-shell,[data-testid="metric-container"],[data-testid="stAlert"],
[data-testid="stExpander"]{
  background:var(--card-glass)!important;
  backdrop-filter:blur(20px)!important;
  -webkit-backdrop-filter:blur(20px)!important;
  border:1px solid rgba(244,114,182,.24)!important;
  border-radius:24px!important;
  box-shadow:0 20px 44px rgba(122,64,112,.12), inset 0 1px 0 rgba(255,255,255,.90)!important;
}
.page-header{
  border-left:7px solid var(--accent-a)!important;
}
.mod-card:hover,.hero-action-card:hover{
  transform:translateY(-3px)!important;
  box-shadow:0 24px 54px rgba(122,64,112,.16),0 0 0 4px color-mix(in srgb, var(--accent-a) 13%, transparent)!important;
}
.mod-card h4,.hero-action-card h4{
  color:var(--ink-strong)!important;
  font-size:1.22rem!important;
  font-weight:900!important;
}
.mod-card p,.hero-action-card p{
  color:var(--ink-soft)!important;
  font-size:1rem!important;
  font-weight:650!important;
}
.home-section-title,.section-chip{
  color:var(--ink-strong)!important;
  font-size:clamp(26px,3vw,32px)!important;
  font-weight:900!important;
}
.section-chip{
  background:linear-gradient(135deg,color-mix(in srgb, var(--accent-a) 18%, white),color-mix(in srgb, var(--accent-b) 16%, white))!important;
  border:1px solid rgba(244,114,182,.25)!important;
  border-radius:999px!important;
}

.stButton>button{
  min-height:46px!important;
  background:linear-gradient(135deg,var(--accent-a),#F9A8D4 48%,var(--accent-b))!important;
  color:var(--ink-strong)!important;
  border:1px solid rgba(255,255,255,.82)!important;
  border-radius:18px!important;
  font-size:18px!important;
  font-weight:900!important;
  box-shadow:0 14px 30px rgba(122,64,112,.14),0 0 0 3px rgba(255,255,255,.40) inset!important;
}
.stButton>button p,.stButton>button span{
  color:var(--ink-strong)!important;
  font-size:18px!important;
  font-weight:900!important;
}
.stButton>button:hover{
  transform:translateY(-1px)!important;
  box-shadow:0 18px 36px rgba(122,64,112,.18),0 0 0 4px color-mix(in srgb, var(--accent-a) 18%, transparent)!important;
}
.stButton>button:disabled{
  color:#7A5C73!important;
  background:rgba(255,255,255,.60)!important;
  border-color:rgba(244,114,182,.18)!important;
  opacity:1!important;
}
.nav-current{
  color:var(--ink-strong)!important;
  font-size:18px!important;
  font-weight:900!important;
}
.nav-current span{
  color:#B83280!important;
  font-size:14px!important;
  font-weight:900!important;
}

.stTabs [data-baseweb="tab"]{
  background:rgba(255,255,255,.56)!important;
  border-radius:16px 16px 0 0!important;
  border:1px solid rgba(244,114,182,.14)!important;
}
.stTabs [data-baseweb="tab"] p{
  color:var(--ink-main)!important;
  font-size:18px!important;
  font-weight:800!important;
}
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,rgba(255,255,255,.94),rgba(255,224,239,.84))!important;
  border-color:var(--accent-a)!important;
}
.stTabs [aria-selected="true"] p{
  color:var(--ink-strong)!important;
  font-weight:900!important;
}

[data-testid="stPlotlyChart"]{
  background:rgba(255,255,255,.72)!important;
  border:1px solid rgba(244,114,182,.22)!important;
  border-radius:24px!important;
  padding:10px!important;
  box-shadow:0 18px 40px rgba(122,64,112,.10)!important;
  overflow:hidden!important;
}
[data-testid="stDataFrame"],div[data-testid="stTable"]{
  background:rgba(255,255,255,.78)!important;
  border:1px solid rgba(244,114,182,.22)!important;
  border-radius:22px!important;
  box-shadow:0 16px 36px rgba(122,64,112,.10)!important;
  overflow:hidden!important;
}
[data-testid="stDataFrame"] *,
div[data-testid="stTable"] *{
  color:var(--ink-strong)!important;
  font-size:15px!important;
  font-weight:650!important;
}
div[data-testid="stDataFrame"] [role="columnheader"],
div[data-testid="stTable"] thead,
div[data-testid="stTable"] th{
  background:linear-gradient(135deg,#ffe0ef,#efe7ff)!important;
  color:var(--ink-strong)!important;
  font-weight:900!important;
}
div[data-testid="stDataFrame"] [role="row"]:nth-child(even),
div[data-testid="stTable"] tbody tr:nth-child(even){
  background:rgba(255,224,239,.30)!important;
}

.ai-agent-card,.agent-teaser,.debug-card,.status-card{
  background:rgba(255,255,255,0.75)!important;
  color:var(--ink-main)!important;
  border:1px solid rgba(244,114,182,.24)!important;
  border-radius:24px!important;
  padding:24px!important;
  line-height:1.8!important;
  box-shadow:0 20px 46px rgba(122,64,112,.12), inset 0 1px 0 rgba(255,255,255,.90)!important;
}
.ai-agent-card .agent-title,.agent-teaser .agent-title,.debug-card .agent-title,.status-card .agent-title{
  color:var(--ink-strong)!important;
  font-size:1.15rem!important;
  font-weight:900!important;
}
.ai-agent-card .agent-body,.agent-teaser .agent-body,.debug-card .agent-body,.status-card .agent-body{
  color:var(--ink-main)!important;
  font-size:18px!important;
  font-weight:650!important;
  line-height:1.8!important;
}

input,textarea,select,[data-baseweb="input"],[data-baseweb="select"],
[data-baseweb="textarea"],[data-baseweb="slider"]{
  color:var(--ink-main)!important;
  background:rgba(255,255,255,.82)!important;
}
[data-baseweb="input"] *,
[data-baseweb="select"] *,
[data-baseweb="slider"] *,
[data-testid="stNumberInput"] *,
[data-testid="stSelectbox"] *{
  color:var(--ink-main)!important;
  font-weight:700!important;
}
[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricLabel"]{
  color:var(--ink-soft)!important;
  font-size:16px!important;
  font-weight:800!important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"]{
  color:#8B1E62!important;
  font-size:clamp(1.65rem,2.5vw,2.25rem)!important;
  font-weight:950!important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"]{
  color:var(--ink-main)!important;
  font-weight:800!important;
}
[data-testid="stSidebar"] [style*="color:#00E5FF"],
[data-testid="stSidebar"] [style*="color:#7D95AA"],
[data-testid="stSidebar"] [style*="color:#455B72"],
.mod-card [style*="color:#00E5FF"]{
  color:#B83280!important;
  opacity:1!important;
  font-weight:900!important;
}
[style*="color:#00D4FF"],
[style*="color: #00D4FF"],
[style*="color:#00E5FF"],
[style*="color: #00E5FF"],
[style*="color:#94A3B8"],
[style*="color: #94A3B8"],
[style*="color:#7D95AA"],
[style*="color: #7D95AA"],
[style*="color:#455B72"],
[style*="color: #455B72"]{
  color:#B83280!important;
  opacity:1!important;
  font-weight:900!important;
}
code,pre{
  background:rgba(255,255,255,.78)!important;
  color:var(--ink-main)!important;
  border:1px solid rgba(244,114,182,.24)!important;
  border-radius:14px!important;
  box-shadow:none!important;
}
@media (max-width: 900px){
  [data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li{font-size:16px!important;}
  [data-testid="stSidebar"] *, .stRadio>div>label p{font-size:16px!important;}
  .policy-hero{min-height:430px!important;}
}

/* AIDEOM v7: readable Streamlit form controls and soft candy stickers */
.sweet-sticker-field{
  position:fixed;
  inset:0;
  z-index:0;
  pointer-events:none;
  overflow:hidden;
}
.sweet-sticker{
  position:absolute;
  width:42px;
  height:42px;
  display:grid;
  place-items:center;
  border-radius:18px;
  background:rgba(255,255,255,.58);
  border:1px solid rgba(243,166,214,.28);
  box-shadow:0 12px 26px rgba(122,64,112,.08);
  font-size:1.45rem;
  opacity:.13;
  animation:sweetFloat 8s ease-in-out infinite;
  backdrop-filter:blur(8px);
}
.sweet-sticker.s1{left:21%;top:12%;animation-delay:.2s;}
.sweet-sticker.s2{right:8%;top:35%;animation-delay:1.4s;}
.sweet-sticker.s3{left:9%;bottom:14%;animation-delay:2.6s;}
.sweet-sticker.s4{right:22%;bottom:8%;animation-delay:3.4s;}
.sweet-sticker.s5{left:48%;top:7%;animation-delay:4.2s;}
.sweet-sticker.s6{right:3%;top:72%;animation-delay:5.1s;}
.sweet-sticker.s7{left:30%;bottom:5%;animation-delay:6s;}
.sweet-sticker.s8{left:3%;top:34%;animation-delay:6.8s;}
.sweet-sticker.s9{right:34%;top:18%;animation-delay:7.4s;}
@keyframes sweetFloat{
  0%,100%{transform:translateY(0) rotate(-5deg);}
  50%{transform:translateY(-16px) rotate(6deg);}
}

.stSelectbox,
.stTextInput,
.stNumberInput,
.stMultiSelect,
.stSlider,
[data-testid="stSelectbox"],
[data-testid="stTextInput"],
[data-testid="stNumberInput"],
[data-testid="stMultiSelect"],
[data-testid="stSlider"]{
  color:#2B1230!important;
  opacity:1!important;
}
.stSelectbox label,
.stTextInput label,
.stNumberInput label,
.stMultiSelect label,
.stSlider label,
[data-testid="stSelectbox"] label,
[data-testid="stTextInput"] label,
[data-testid="stNumberInput"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stSlider"] label{
  color:#2B1230!important;
  font-size:18px!important;
  font-weight:800!important;
  opacity:1!important;
}
.stSelectbox [data-baseweb="select"],
.stMultiSelect [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stMultiSelect"] [data-baseweb="select"]{
  background:rgba(255,255,255,0.92)!important;
  color:#2B1230!important;
  border:2px solid #F3A6D6!important;
  border-radius:18px!important;
  box-shadow:0 12px 26px rgba(122,64,112,.09)!important;
  overflow:hidden!important;
}
.stSelectbox [data-baseweb="select"] > div,
.stMultiSelect [data-baseweb="select"] > div,
[data-baseweb="select"] > div,
[data-baseweb="select"] div{
  background:rgba(255,255,255,0.92)!important;
  color:#2B1230!important;
  border-color:#F3A6D6!important;
  border-radius:18px!important;
  font-size:18px!important;
  font-weight:700!important;
  opacity:1!important;
}
[data-baseweb="select"] input,
[data-baseweb="select"] span,
[data-baseweb="select"] svg,
[data-baseweb="select"] p{
  color:#2B1230!important;
  fill:#2B1230!important;
  font-size:18px!important;
  font-weight:700!important;
  opacity:1!important;
}
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
div[role="listbox"],
ul[role="listbox"]{
  background:#FFF7FC!important;
  color:#2B1230!important;
  border:2px solid #F3A6D6!important;
  border-radius:18px!important;
  box-shadow:0 18px 44px rgba(122,64,112,.16)!important;
  opacity:1!important;
}
div[role="option"],
li[role="option"],
[data-baseweb="menu"] li,
[data-baseweb="menu"] div{
  background:#FFF7FC!important;
  color:#2B1230!important;
  font-size:18px!important;
  font-weight:600!important;
  opacity:1!important;
}
div[role="option"]:hover,
li[role="option"]:hover,
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] div:hover{
  background:linear-gradient(135deg,#FFE0EF,#E8FBFF)!important;
  color:#2B1230!important;
}
div[role="option"][aria-selected="true"],
li[role="option"][aria-selected="true"]{
  background:linear-gradient(135deg,#F3A6D6,#BFEFFF)!important;
  color:#2B1230!important;
  font-weight:800!important;
}
.stTextInput input,
.stNumberInput input,
[data-baseweb="input"],
[data-baseweb="input"] > div,
[data-baseweb="input"] input,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input{
  background:rgba(255,255,255,0.92)!important;
  color:#2B1230!important;
  border-color:#F3A6D6!important;
  border-radius:18px!important;
  font-size:18px!important;
  font-weight:700!important;
  opacity:1!important;
}
.stTextInput [data-baseweb="input"],
.stNumberInput [data-baseweb="input"]{
  border:2px solid #F3A6D6!important;
  box-shadow:0 12px 26px rgba(122,64,112,.09)!important;
}
input::placeholder,
textarea::placeholder,
[data-baseweb="select"] input::placeholder{
  color:#6B4B66!important;
  opacity:1!important;
  font-weight:650!important;
}
[data-baseweb="tag"]{
  background:linear-gradient(135deg,#FFE0EF,#E8FBFF)!important;
  color:#2B1230!important;
  border:1px solid #F3A6D6!important;
  border-radius:999px!important;
  font-size:16px!important;
  font-weight:800!important;
}
[data-baseweb="tag"] span,
[data-baseweb="tag"] svg{
  color:#2B1230!important;
  fill:#2B1230!important;
}
.stSlider [data-baseweb="slider"]{
  background:transparent!important;
}
.stSlider [data-baseweb="slider"] *{
  color:#2B1230!important;
  opacity:1!important;
}
.stSlider [role="slider"]{
  background:#FFFFFF!important;
  border:3px solid #F472B6!important;
  box-shadow:0 0 0 5px rgba(243,166,214,.26),0 8px 20px rgba(122,64,112,.14)!important;
}
.stSlider [data-testid="stTickBar"] *,
.stSlider [data-testid="stThumbValue"],
.stSlider [data-testid="stSliderTickBarMin"],
.stSlider [data-testid="stSliderTickBarMax"]{
  color:#2B1230!important;
  font-size:15px!important;
  font-weight:750!important;
  opacity:1!important;
}
.stTabs [data-baseweb="tab-list"]{
  gap:.35rem!important;
  background:rgba(255,255,255,.45)!important;
  border-radius:20px!important;
  padding:.35rem!important;
  border:1px solid rgba(243,166,214,.22)!important;
}
.stTabs [data-baseweb="tab"]{
  background:rgba(255,255,255,.82)!important;
  color:#2B1230!important;
  border:1px solid rgba(243,166,214,.30)!important;
  border-radius:16px!important;
  min-height:46px!important;
  opacity:1!important;
}
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span{
  color:#2B1230!important;
  font-size:18px!important;
  font-weight:800!important;
  opacity:1!important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"],
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,#F3A6D6 0%,#F9D7EA 48%,#BFEFFF 100%)!important;
  border:2px solid #F3A6D6!important;
  box-shadow:0 12px 24px rgba(122,64,112,.12),0 0 0 3px rgba(255,255,255,.55) inset!important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] p,
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span{
  color:#2B1230!important;
  font-weight:900!important;
}
[data-baseweb="select"],
[data-baseweb="popover"],
[data-baseweb="menu"],
[data-baseweb="input"]{
  caret-color:#2B1230!important;
}

/* AIDEOM v8: visible pastel Streamlit top header */
[data-testid="stHeader"]{
  display:flex!important;
  visibility:visible!important;
  opacity:1!important;
  background:
    linear-gradient(
      90deg,
      rgba(255,245,250,0.95),
      rgba(245,240,255,0.95)
    )!important;
  backdrop-filter:blur(12px)!important;
  -webkit-backdrop-filter:blur(12px)!important;
  border-bottom:1px solid rgba(255,182,220,0.35)!important;
  box-shadow:0 10px 28px rgba(122,64,112,.08)!important;
  color:#3B1F35!important;
}
[data-testid="stHeader"] *,
[data-testid="stHeader"] p,
[data-testid="stHeader"] span,
[data-testid="stHeader"] label,
[data-testid="stToolbar"] *,
[data-testid="stDecoration"] *{
  color:#3B1F35!important;
  fill:#3B1F35!important;
  opacity:1!important;
  font-weight:700!important;
}
[data-testid="stHeader"] button,
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] [role="button"],
[data-testid="stHeader"] [role="button"],
[data-testid="stMainMenu"],
[data-testid="stStatusWidget"],
[data-testid="stDeployButton"],
[data-testid="stAppDeployButton"]{
  background:rgba(255,255,255,0.72)!important;
  color:#3B1F35!important;
  fill:#3B1F35!important;
  border:1px solid rgba(255,182,220,0.42)!important;
  border-radius:12px!important;
  box-shadow:0 8px 20px rgba(122,64,112,.10)!important;
  backdrop-filter:blur(10px)!important;
  -webkit-backdrop-filter:blur(10px)!important;
  font-weight:800!important;
}
[data-testid="stHeader"] button:hover,
[data-testid="stToolbar"] button:hover,
[data-testid="stToolbar"] [role="button"]:hover,
[data-testid="stHeader"] [role="button"]:hover{
  background:linear-gradient(135deg,rgba(255,224,239,.92),rgba(224,249,255,.86))!important;
  border-color:#F3A6D6!important;
  box-shadow:0 10px 24px rgba(122,64,112,.14),0 0 0 3px rgba(243,166,214,.18)!important;
}
[data-testid="stHeader"] svg,
[data-testid="stToolbar"] svg{
  color:#3B1F35!important;
  fill:#3B1F35!important;
  stroke:#3B1F35!important;
}
[data-testid="stHeader"] > div,
[data-testid="stToolbar"],
[data-testid="stToolbar"] > div,
[data-testid="stToolbar"] [data-testid],
[data-testid="stStatusWidget"],
[data-testid="stStatusWidget"] > div,
[data-testid="stStatusWidget"] [role="button"],
[data-testid="stDeployButton"],
[data-testid="stDeployButton"] > div,
[data-testid="stAppDeployButton"],
[data-testid="stAppDeployButton"] > div,
[data-testid="stMainMenu"],
[data-testid="stMainMenu"] > div{
  background:rgba(255,255,255,0.64)!important;
  color:#3B1F35!important;
  border-color:rgba(255,182,220,0.42)!important;
  border-radius:12px!important;
  box-shadow:none!important;
  opacity:1!important;
}
[data-testid="stStatusWidget"]{
  min-width:34px!important;
  min-height:34px!important;
  display:flex!important;
  align-items:center!important;
  justify-content:center!important;
}
[data-testid="stStatusWidget"]::before{
  content:"🏃"!important;
  display:grid!important;
  place-items:center!important;
  width:28px!important;
  height:28px!important;
  border-radius:10px!important;
  background:linear-gradient(135deg,rgba(255,224,239,.88),rgba(224,249,255,.82))!important;
  border:1px solid rgba(255,182,220,.42)!important;
  font-size:16px!important;
  line-height:1!important;
}
[data-testid="stStatusWidget"] canvas,
[data-testid="stStatusWidget"] img,
[data-testid="stStatusWidget"] svg{
  background:transparent!important;
  border-radius:10px!important;
  opacity:.72!important;
}
[data-testid="stHeader"] [style*="background-color: rgb(14"],
[data-testid="stHeader"] [style*="background: rgb(14"],
[data-testid="stHeader"] [style*="background-color: #0"],
[data-testid="stHeader"] [style*="background:#0"],
[data-testid="stToolbar"] [style*="background-color: rgb(14"],
[data-testid="stToolbar"] [style*="background: rgb(14"],
[data-testid="stToolbar"] [style*="background-color: #0"],
[data-testid="stToolbar"] [style*="background:#0"]{
  background:rgba(255,255,255,0.64)!important;
  color:#3B1F35!important;
  border-radius:12px!important;
}

/* Hide the rough Streamlit status square next to Deploy; keep Deploy visible. */
[data-testid="stStatusWidget"],
[data-testid="stStatusWidget"] *,
[data-testid="stStatusWidget"]::before,
[data-testid="stStatusWidget"]::after{
  display:none!important;
  visibility:hidden!important;
  width:0!important;
  height:0!important;
  min-width:0!important;
  min-height:0!important;
  margin:0!important;
  padding:0!important;
  border:0!important;
  opacity:0!important;
  overflow:hidden!important;
}

.sweet-sticker{
  width:50px!important;
  height:50px!important;
  border-radius:20px!important;
  background:rgba(255,255,255,.72)!important;
  border:1px solid rgba(243,166,214,.34)!important;
  box-shadow:0 14px 30px rgba(122,64,112,.10)!important;
  font-size:1.72rem!important;
  opacity:.22!important;
}
.sweet-sticker.s1{left:18%!important;top:15%!important;}
.sweet-sticker.s2{right:7%!important;top:38%!important;}
.sweet-sticker.s3{left:7%!important;bottom:12%!important;}
.sweet-sticker.s4{right:18%!important;bottom:10%!important;}
.sweet-sticker.s5{left:52%!important;top:8%!important;}
.sweet-sticker.s6{right:4%!important;top:68%!important;}
.sweet-sticker.s7{left:30%!important;bottom:5%!important;}
.sweet-sticker.s8{left:4%!important;top:34%!important;}
.sweet-sticker.s9{right:33%!important;top:18%!important;}

/* AIDEOM v9: keep the small status box next to Deploy, but make it pastel. */
[data-testid="stStatusWidget"]{
  display:flex!important;
  visibility:visible!important;
  width:36px!important;
  height:36px!important;
  min-width:36px!important;
  min-height:36px!important;
  margin:0 .25rem!important;
  padding:0!important;
  opacity:1!important;
  overflow:hidden!important;
  align-items:center!important;
  justify-content:center!important;
  border-radius:14px!important;
  border:1px solid rgba(255,182,220,.48)!important;
  background:linear-gradient(135deg,rgba(255,245,250,.90),rgba(236,250,255,.82),rgba(245,240,255,.88))!important;
  box-shadow:0 10px 24px rgba(122,64,112,.12), inset 0 1px 0 rgba(255,255,255,.9)!important;
  backdrop-filter:blur(12px)!important;
  -webkit-backdrop-filter:blur(12px)!important;
}
[data-testid="stStatusWidget"] *{
  display:none!important;
}
[data-testid="stStatusWidget"]::before{
  content:"🏃"!important;
  display:grid!important;
  visibility:visible!important;
  place-items:center!important;
  width:30px!important;
  height:30px!important;
  color:#3B1F35!important;
  font-size:17px!important;
  line-height:1!important;
  opacity:1!important;
}

.sidebar-logo::after{
  content:"🍩  🥐  🧁  ♡"!important;
  display:block!important;
  margin-top:.65rem!important;
  color:rgba(180,50,128,.55)!important;
  font-size:1rem!important;
  letter-spacing:.35rem!important;
  filter:drop-shadow(0 0 8px rgba(255,182,220,.45))!important;
}

.result-note-card{
  margin:.65rem 0 1rem;
  padding:1rem 1.15rem;
  border-radius:22px;
  background:rgba(255,255,255,.76);
  border:1px solid rgba(243,166,214,.34);
  box-shadow:0 16px 34px rgba(122,64,112,.10), inset 0 1px 0 rgba(255,255,255,.92);
  backdrop-filter:blur(16px);
  -webkit-backdrop-filter:blur(16px);
}
.result-note-title{
  color:#2B1230!important;
  font-size:1.02rem;
  font-weight:900!important;
  margin-bottom:.45rem;
}
.result-note-body{
  color:#3B1F35!important;
  font-size:17px;
  line-height:1.75;
  font-weight:650!important;
}

.original-report-shell{
  margin:1.4rem 0 .9rem;
  padding:1.25rem 1.35rem;
  border-radius:26px;
  background:rgba(255,255,255,.78);
  border:1px solid rgba(243,166,214,.32);
  box-shadow:0 20px 46px rgba(122,64,112,.12), inset 0 1px 0 rgba(255,255,255,.92);
  backdrop-filter:blur(18px);
  -webkit-backdrop-filter:blur(18px);
}
.original-report-kicker{
  display:inline-flex;
  align-items:center;
  gap:.35rem;
  padding:.32rem .72rem;
  border-radius:999px;
  background:linear-gradient(135deg,rgba(255,224,239,.92),rgba(232,251,255,.82));
  border:1px solid rgba(243,166,214,.38);
  color:#B83280!important;
  font-size:.82rem;
  font-weight:900!important;
  letter-spacing:.04em;
}
.original-report-title{
  margin-top:.75rem;
  color:#2B1230!important;
  font-size:clamp(1.55rem,2.5vw,2rem);
  font-weight:950!important;
  line-height:1.25;
}
.original-report-note{
  margin-top:.45rem;
  color:#5A3A55!important;
  font-size:1rem;
  line-height:1.7;
  font-weight:650!important;
}
.original-report-card{
  position:relative;
  margin:.85rem 0;
  padding:1.1rem 1.25rem 1.15rem;
  border-radius:24px;
  background:rgba(255,255,255,.80);
  border:1px solid rgba(243,166,214,.26);
  box-shadow:0 16px 34px rgba(122,64,112,.09), inset 0 1px 0 rgba(255,255,255,.92);
  overflow:hidden;
}
.original-report-card::after{
  content:"♡";
  position:absolute;
  right:1rem;
  top:.65rem;
  color:rgba(244,114,182,.16);
  font-size:2.2rem;
  line-height:1;
  pointer-events:none;
}
.original-report-card-title{
  color:#2B1230!important;
  font-size:1.05rem;
  font-weight:900!important;
  padding-right:2.5rem;
  margin-bottom:.55rem;
}
.original-report-card-body{
  color:#3B1F35!important;
  font-size:18px;
  line-height:1.82;
  font-weight:600!important;
  white-space:normal;
}

/* Public Streamlit Cloud UI fixes: toolbar stickers, readable menus, compact cards, responsive charts. */
[data-testid="stHeader"],
[data-testid="stToolbar"]{
  background:linear-gradient(90deg,rgba(255,250,253,.86),rgba(239,250,255,.78),rgba(248,244,255,.84))!important;
  backdrop-filter:blur(18px) saturate(1.18)!important;
  -webkit-backdrop-filter:blur(18px) saturate(1.18)!important;
}
[data-testid="stToolbar"]{
  gap:.38rem!important;
}
[data-testid="stToolbar"] button,
[data-testid="stToolbar"] [role="button"],
[data-testid="stHeader"] button,
[data-testid="stMainMenu"],
[data-testid="stStatusWidget"],
[data-testid="stDeployButton"],
[data-testid="stAppDeployButton"]{
  min-width:36px!important;
  min-height:36px!important;
  border-radius:14px!important;
  border:1px solid rgba(244,114,182,.34)!important;
  background:linear-gradient(135deg,rgba(255,255,255,.78),rgba(255,231,242,.70),rgba(226,249,255,.62))!important;
  color:#2B1230!important;
  fill:#2B1230!important;
  box-shadow:0 10px 24px rgba(122,64,112,.10), inset 0 1px 0 rgba(255,255,255,.90)!important;
  backdrop-filter:blur(14px)!important;
  -webkit-backdrop-filter:blur(14px)!important;
}
[data-testid="stToolbar"] button:hover,
[data-testid="stToolbar"] [role="button"]:hover,
[data-testid="stHeader"] button:hover,
[data-testid="stMainMenu"]:hover,
[data-testid="stStatusWidget"]:hover,
[data-testid="stDeployButton"]:hover,
[data-testid="stAppDeployButton"]:hover{
  background:linear-gradient(135deg,rgba(255,236,246,.96),rgba(230,250,255,.88),rgba(249,242,255,.92))!important;
  border-color:rgba(244,114,182,.62)!important;
  box-shadow:0 12px 28px rgba(122,64,112,.14),0 0 0 4px rgba(244,114,182,.15)!important;
}
[data-testid="stStatusWidget"]{
  display:flex!important;
  visibility:visible!important;
  width:36px!important;
  height:36px!important;
  min-width:36px!important;
  min-height:36px!important;
  margin:0 .18rem!important;
  padding:0!important;
  overflow:hidden!important;
}
[data-testid="stStatusWidget"] *,
[data-testid="stStatusWidget"] canvas,
[data-testid="stStatusWidget"] img,
[data-testid="stStatusWidget"] svg{
  display:none!important;
}
[data-testid="stStatusWidget"]::before{
  content:"🌸"!important;
  display:grid!important;
  place-items:center!important;
  width:100%!important;
  height:100%!important;
  color:#2B1230!important;
  font-size:18px!important;
  line-height:1!important;
  opacity:1!important;
}
[data-testid="stMainMenu"]::before{content:"🍩"!important;font-size:17px!important;line-height:1!important;}
[data-testid="stToolbar"] button[aria-label*="rerun" i]::before,
[data-testid="stToolbar"] button[title*="rerun" i]::before{content:"✨"!important;font-size:17px!important;}
[data-testid="stToolbar"] button[aria-label*="favorite" i]::before,
[data-testid="stToolbar"] button[title*="favorite" i]::before{content:"💖"!important;font-size:17px!important;}
[data-testid="stToolbar"] button[aria-label*="more" i]::before,
[data-testid="stToolbar"] button[title*="more" i]::before{content:"🥐"!important;font-size:17px!important;}
[data-testid="stToolbar"] button:empty,
[data-testid="stToolbar"] [role="button"]:empty{
  display:none!important;
}
[data-baseweb="popover"],
[data-baseweb="popover"] > div,
[data-baseweb="menu"],
[role="menu"],
[data-baseweb="tooltip"]{
  background:linear-gradient(135deg,#fffafd,#fff0f7 44%,#eefbff 100%)!important;
  color:#2B1230!important;
  border:1px solid rgba(244,114,182,.32)!important;
  border-radius:16px!important;
  box-shadow:0 18px 44px rgba(122,64,112,.16)!important;
}
[data-baseweb="menu"] *,
[role="menu"] *,
[data-baseweb="popover"] *,
[data-baseweb="tooltip"] *{
  color:#2B1230!important;
  fill:#2B1230!important;
  font-weight:800!important;
  opacity:1!important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] div:hover,
[role="menuitem"]:hover{
  background:rgba(244,114,182,.15)!important;
  color:#2B1230!important;
}
[data-testid="stPlotlyChart"]{
  width:100%!important;
  max-width:100%!important;
  overflow:hidden!important;
  padding:clamp(4px,1vw,10px)!important;
}
[data-testid="stPlotlyChart"] .js-plotly-plot,
[data-testid="stPlotlyChart"] .plot-container,
[data-testid="stPlotlyChart"] .svg-container{
  width:100%!important;
  max-width:100%!important;
}
.mod-card{
  overflow:hidden!important;
  min-height:178px!important;
}
.mod-kicker{
  display:flex!important;
  align-items:center!important;
  gap:.42rem!important;
  max-width:100%!important;
  color:#B83280!important;
  font-weight:900!important;
  font-size:.72rem!important;
  line-height:1.2!important;
  white-space:nowrap!important;
  overflow:hidden!important;
  text-overflow:ellipsis!important;
}
.mod-kicker-icon{
  flex:0 0 1.25rem!important;
  width:1.25rem!important;
  height:1.25rem!important;
  display:inline-grid!important;
  place-items:center!important;
  border-radius:.5rem!important;
  background:linear-gradient(135deg,rgba(255,255,255,.92),rgba(255,224,239,.86))!important;
  border:1px solid rgba(244,114,182,.25)!important;
  box-shadow:0 5px 12px rgba(122,64,112,.08)!important;
  font-size:.82rem!important;
  line-height:1!important;
}
.mod-card h4,
.mod-card p,
.mod-card .badge{
  max-width:100%!important;
  overflow-wrap:anywhere!important;
}
@media (max-width: 720px){
  [data-testid="stToolbar"] button,
  [data-testid="stToolbar"] [role="button"],
  [data-testid="stMainMenu"],
  [data-testid="stStatusWidget"]{
    width:34px!important;
    height:34px!important;
    min-width:34px!important;
    min-height:34px!important;
  }
  .mod-card{min-height:auto!important;}
  .mod-kicker{font-size:.68rem!important;}
}

</style>
    """,
    unsafe_allow_html=True,
)


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-logo">
              <div class="mark">🇻🇳</div>
              <div class="title">AIDEOM 🇻🇳 Tran Thu Ha</div>
              <div class="slogan">Mô hình ra quyết định phát triển kinh tế số Việt Nam trong kỷ nguyên AI</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="nav-caption">ĐIỀU HƯỚNG</div>', unsafe_allow_html=True)

        selected_label = st.radio(
            "Nav",
            list(PAGES.keys()),
            label_visibility="collapsed",
            key="nav_radio",
        )
        selected_page = PAGES[selected_label]
        if selected_page != st.session_state["current_page"]:
            st.session_state["current_page"] = selected_page
            st.rerun()

        st.markdown("---")
        st.markdown(
            """
            <div class="glass-panel" style="padding:.9rem;">
              <div style="color:#00E5FF;font-size:.74rem;font-weight:900;margin-bottom:.35rem;">📊 Nguồn dữ liệu</div>
              <div style="color:#7D95AA;font-size:.72rem;line-height:1.65;">
                NSO, MoST, MIC, MPI, WB, GII<br>
                Giai đoạn: 2020-2025 | Dự báo: 2030
              </div>
            </div>
            <div style="margin-top:.8rem;text-align:center;color:#455B72;font-size:.68rem;">
              v3.0 — Futuristic Academic Dashboard
            </div>
            """,
            unsafe_allow_html=True,
        )

    return st.session_state["current_page"]


def inject_page_skin(page_id: str) -> None:
    accent_a, accent_b, accent_c = PAGE_ACCENTS.get(page_id, PAGE_ACCENTS["home"])
    st.markdown(
        f"""
        <style>
        :root{{
          --accent-a:{accent_a};
          --accent-b:{accent_b};
          --accent-c:{accent_c};
        }}
        </style>
        <div class="aideom-page-sentinel page-{page_id}"></div>
        <div class="sweet-sticker-field" aria-hidden="true">
          <span class="sweet-sticker s1">🍩</span>
          <span class="sweet-sticker s2">🥐</span>
          <span class="sweet-sticker s3">🧁</span>
          <span class="sweet-sticker s4">🍥</span>
          <span class="sweet-sticker s5">♡</span>
          <span class="sweet-sticker s6">🍬</span>
          <span class="sweet-sticker s7">🍪</span>
          <span class="sweet-sticker s8">💗</span>
          <span class="sweet-sticker s9">🍡</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_lesson_nav(page_id: str, location: str) -> None:
    if page_id not in LESSON_IDS:
        return

    idx = LESSON_IDS.index(page_id)
    prev_id = LESSON_IDS[idx - 1] if idx > 0 else None
    next_id = LESSON_IDS[idx + 1] if idx < len(LESSON_IDS) - 1 else None
    shell_class = "lesson-nav-shell lesson-nav-bottom" if location == "bottom" else "lesson-nav-shell"

    st.markdown(f'<div class="{shell_class}">', unsafe_allow_html=True)
    left, middle, right = st.columns([1.15, 2.7, 1.15], vertical_alignment="center")

    with left:
        if prev_id:
            st.button("← Bài trước", key=f"{location}_prev_{page_id}", use_container_width=True, on_click=set_page, args=(prev_id,))
        else:
            st.button("← Bài trước", key=f"{location}_prev_disabled_{page_id}", use_container_width=True, disabled=True)

    with middle:
        st.markdown(
            f"""
            <div class="nav-current">
              <span>Đang xem {idx + 1:02d}/12</span>
              {LESSON_TITLES[page_id]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        if next_id:
            st.button("Bài tiếp →", key=f"{location}_next_{page_id}", use_container_width=True, on_click=set_page, args=(next_id,))
        else:
            st.button("Bài tiếp →", key=f"{location}_next_disabled_{page_id}", use_container_width=True, disabled=True)

    st.markdown("</div>", unsafe_allow_html=True)

def show_home() -> None:
    st.markdown(
        """
        <section class="policy-hero">
          <div class="kawaii-float float-donut">🍩</div>
          <div class="kawaii-float float-croissant">🥐</div>
          <div class="kawaii-float float-cupcake">🧁</div>
          <div class="kawaii-float float-candy">🍬</div>
          <div class="hero-content">
            <h1 class="hero-title-main">AIDEOM-VN POLICY LAB</h1>
            <div class="hero-subtitle">
              Mô hình ra quyết định phát triển kinh tế số Việt Nam trong kỷ nguyên AI
            </div>
            <div class="hero-author">Designed by Tran Thu Ha</div>
            <div class="hero-actions">
              <div class="hero-action-card">
                <h4>Khởi động chuỗi 12 mô hình</h4>
                <p>Bắt đầu từ Cobb-Douglas + AI, đi qua LP, TOPSIS, NSGA-II, stochastic programming và RL.</p>
              </div>
              <div class="hero-action-card">
                <h4>Mở dashboard tích hợp</h4>
                <p>Xem AIDEOM Score, 5 kịch bản, rủi ro chính sách và AI Policy Agent tổng hợp.</p>
              </div>
            </div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    cta1, cta2, _ = st.columns([1, 1, 2])
    with cta1:
        st.button("Vào Bài 1", key="home_go_bai01", use_container_width=True, on_click=set_page, args=("bai01",))
    with cta2:
        st.button("Mở Dashboard", key="home_go_bai12", use_container_width=True, on_click=set_page, args=("bai12",))

    st.markdown('<div class="home-section-title">Hệ thống chỉ báo nhanh</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GDP Forecast 2030", "Scenario-based", "SP + dynamic")
    c2.metric("Digital Economy %", "30%", "target 2030")
    c3.metric("AI Readiness", "AIDEOM Score", "dashboard")
    c4.metric("NetJob Impact", "8 sectors", "labor model")

    st.markdown('<div class="home-section-title">12 mô hình ra quyết định</div>', unsafe_allow_html=True)
    mods = [
        ("01", "📈", "Cobb-Douglas + AI", "Hàm sản xuất mở rộng D, AI, H. Phân rã tăng trưởng, dự báo 2030.", "numpy/pandas"),
        ("02", "💰", "LP ngân sách số", "Phân bổ ngân sách cho 4 hạng mục. Shadow price và phân tích độ nhạy.", "LP/PuLP"),
        ("03", "🏭", "Priority 10 ngành", "Chỉ số ưu tiên chuyển đổi số với các bộ trọng số chính sách.", "MCDM"),
        ("04", "🗺️", "LP ngành-vùng", "Ngân sách theo 6 vùng và 4 hạng mục, có ràng buộc công bằng.", "PuLP/CVXPY"),
        ("05", "🎯", "MIP 15 dự án", "Lựa chọn danh mục dự án tối ưu với ràng buộc nhị phân.", "MIP"),
        ("06", "🌍", "TOPSIS 6 vùng", "Xếp hạng vùng ưu tiên đầu tư AI theo nhiều tiêu chí.", "TOPSIS"),
        ("07", "⚡", "NSGA-II Pareto", "Tối ưu đa mục tiêu: GDP, bao trùm, phát thải và an ninh.", "pymoo"),
        ("08", "📅", "Tối ưu động", "Quỹ đạo K, D, AI, H tối ưu giai đoạn 2026-2035.", "SLSQP"),
        ("09", "👷", "Lao động & AI", "NetJob = New + Upgrade - Displaced theo 8 ngành.", "LP"),
        ("10", "🎲", "Stochastic SP", "Two-stage stochastic programming với VSS, EVPI, Minimax Regret.", "HiGHS"),
        ("11", "🤖", "Q-learning RL", "MDP, learning curve và policy extraction.", "RL"),
        ("12", "🇻🇳", "AIDEOM tích hợp", "AIDEOM Score, 5 kịch bản, rủi ro và AI Policy Agent.", "Dashboard"),
    ]

    for i in range(0, 12, 4):
        cols = st.columns(4)
        for col, (num, icon, title, desc, tech) in zip(cols, mods[i : i + 4]):
            with col:
                st.markdown(
                    f"""
                    <div class="mod-card">
                      <div class="mod-kicker"><span class="mod-kicker-icon">{icon}</span><span>BÀI {num}</span></div>
                      <h4>{title}</h4>
                      <p>{desc}</p>
                      <div style="margin-top:.55rem;"><span class="badge b-blue">{tech}</span></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="home-section-title">Hướng dẫn sử dụng</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.info("Chọn bài từ sidebar hoặc dùng nút mũi tên ở đầu/cuối từng bài để chuyển nhanh.")
    b.info("Điều chỉnh tham số trong từng bài bằng slider, selectbox và number input.")
    c.info("Đọc phần AI Agent trong mỗi bài để xem nhận xét chính sách mô phỏng.")


def show_module(pid: str) -> None:
    try:
        st.session_state["suppress_auto_result_notes"] = False
        mod = importlib.import_module(f"modules.{pid}")
        mod.show()
    except ModuleNotFoundError as exc:
        st.error(f"Chưa tìm thấy module `modules/{pid}.py`. Chi tiết: {exc}")
    except Exception as exc:
        st.error(f"Lỗi render trong module **{pid}**: `{exc}`. Chi tiết kỹ thuật đã được ẩn để giao diện không hiển thị thông tin gỡ lỗi.")


page_id = render_sidebar()
inject_page_skin(page_id)

if page_id == "home":
    show_home()
else:
    render_lesson_nav(page_id, "top")
    show_module(page_id)
    render_lesson_nav(page_id, "bottom")
