from __future__ import annotations

import os
import traceback
import html
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pandas.io.formats.style import Styler
import streamlit as st


TAB_CONTEXT = "context"
TAB_CONFIG = "config"
TAB_RESULT = "result"
TAB_POLICY = "policy"


def _prepare_solver_path() -> None:
    try:
        import pulp

        cbc_path = Path(pulp.PULP_CBC_CMD().path).resolve()
        if cbc_path.exists():
            os.environ["PATH"] = str(cbc_path.parent) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def _read_colab_code(source_path: str) -> str:
    text = Path(source_path).read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!"):
            indent = line[: len(line) - len(stripped)]
            lines.append(indent + "pass  # Colab shell magic removed for Streamlit")
            continue
        lines.append(line)
    return "\n".join(lines)


def _clean_print_text(text: str) -> str:
    lines = []
    for line in str(text).splitlines():
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if set(stripped) <= {"="}:
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def _target_from_text(text: str) -> str:
    upper = text.upper()
    if "BỐI CẢNH" in upper or "MÔ HÌNH TOÁN" in upper or "MÔ HÌNH MDP" in upper:
        return TAB_CONTEXT
    if "DỮ LIỆU" in upper or "THAM SỐ" in upper or "KỊCH BẢN" in upper:
        return TAB_CONFIG
    if "PHÂN TÍCH CHÍNH SÁCH" in upper or "KẾT LUẬN" in upper:
        return TAB_POLICY
    return TAB_RESULT


def _is_section_text(text: str) -> bool:
    upper = text.upper()
    return any(
        token in upper
        for token in [
            "BỐI CẢNH BÀI TOÁN",
            "MÔ HÌNH TOÁN HỌC",
            "MÔ HÌNH MDP",
            "CÂU ",
            "PHÂN TÍCH CHÍNH SÁCH",
            "KẾT LUẬN",
        ]
    )


def _section_title_from_text(text: str) -> tuple[str, str]:
    clean = _clean_print_text(text)
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    if not lines:
        return "", ""
    title = lines[0]
    if title[:3] in {"0. ", "1. "}:
        title = title[3:]
    body = "\n\n".join(lines[1:]).strip()
    return title, body


def _section(container, title: str, icon: str = "📌") -> None:
    if not title:
        return
    container.markdown(
        f"""
        <div class="section-chip" style="margin-top:1rem;margin-bottom:.7rem;">
          {icon} {html.escape(title)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _result_note(container, title: str, sentences: list[str]) -> None:
    clean_sentences = [s.strip() for s in sentences if s and s.strip()]
    if not clean_sentences:
        return
    body = " ".join(html.escape(sentence) for sentence in clean_sentences[:5])
    container.markdown(
        f"""
        <div class="result-note-card">
          <div class="result-note-title">💬 {html.escape(title)}</div>
          <div class="result-note-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _number(value) -> str:
    try:
        number = float(value)
    except Exception:
        return str(value)
    if abs(number) >= 1000:
        return f"{number:,.1f}"
    if abs(number) >= 10:
        return f"{number:,.2f}"
    return f"{number:,.4f}"


def _dataframe_note(container, df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return
    cols = [c for c in numeric.columns if str(c).lower() not in {"năm", "year"}]
    if not cols:
        cols = list(numeric.columns)
    col = max(cols, key=lambda c: float(numeric[c].max() - numeric[c].min()))
    series = numeric[col].dropna()
    if series.empty:
        return
    top_idx = series.idxmax()
    low_idx = series.idxmin()
    sentences = [
        f"Bảng có {len(df)} dòng và {len(df.columns)} cột; chỉ tiêu đọc nhanh là {col}.",
        f"Giá trị cao nhất thuộc về {top_idx} với {_number(series.loc[top_idx])}, thấp nhất là {low_idx} với {_number(series.loc[low_idx])}.",
        "Khoảng cách này cho thấy mức phân hóa giữa các phương án/vùng/ngành và là cơ sở để ưu tiên nguồn lực chính sách.",
    ]
    if len(series) >= 2:
        trend = "tăng" if float(series.iloc[-1]) >= float(series.iloc[0]) else "giảm"
        sentences.append(f"Theo thứ tự hiển thị, {col} có xu hướng {trend} từ {_number(series.iloc[0])} đến {_number(series.iloc[-1])}.")
    _result_note(container, "Nhận xét kết quả", sentences)


def _safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    safe = df.copy()
    for col in safe.columns:
        if safe[col].dtype == "object":
            sample_types = {type(v).__name__ for v in safe[col].dropna().head(30)}
            if len(sample_types) > 1:
                safe[col] = safe[col].map(lambda v: "" if pd.isna(v) else str(v))
    return safe


def _matplotlib_note(container, fig) -> None:
    try:
        ax = fig.axes[0]
        title = ax.get_title() or "Biểu đồ kết quả"
        sentences = [f"Biểu đồ '{title}' trực quan hóa kết quả chính của câu đang xét."]
        lines = ax.get_lines()
        if lines:
            y = list(lines[0].get_ydata())
            x = list(lines[0].get_xdata())
            if len(y) >= 2:
                trend = "tăng" if y[-1] >= y[0] else "giảm"
                sentences.append(f"Chuỗi chính có xu hướng {trend} từ {_number(y[0])} đến {_number(y[-1])}.")
                top = max(range(len(y)), key=lambda i: y[i])
                sentences.append(f"Điểm cao nhất nằm tại {x[top] if top < len(x) else top} với giá trị {_number(y[top])}.")
        elif ax.containers:
            vals = [patch.get_height() for bar_container in ax.containers for patch in bar_container]
            if vals:
                sentences.append(f"Cột cao nhất đạt {_number(max(vals))}, trong khi cột thấp nhất đạt {_number(min(vals))}.")
        sentences.append("Ý nghĩa chính sách là cần đọc khác biệt giữa nhóm cao và nhóm thấp trước khi quyết định phân bổ nguồn lực.")
        _result_note(container, "Nhận xét kết quả", sentences)
    except Exception:
        pass


def _header(icon: str, title: str, labels: tuple[str, ...]) -> None:
    chips = "".join(f'<span class="badge b-blue">{label}</span>' for label in labels)
    st.markdown(
        f"""
        <div class="page-header">
          <div style="display:flex;align-items:center;gap:1rem;">
            <span style="font-size:2rem;">{icon}</span>
            <div>
              <div style="color:#00D4FF;font-size:.75rem;font-weight:600;letter-spacing:1px;">SOURCE OF TRUTH</div>
              <div style="color:#E2E8F0;font-size:1.5rem;font-weight:800;">{title}</div>
              <div style="margin-top:.3rem;">{chips}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _ai_agent(text: str) -> None:
    state_key = "ai_agent_visible_source_runner"
    if st.button("🤖 Phân tích bằng AI Agent", key=f"{state_key}_button", use_container_width=True):
        st.session_state[state_key] = True
    if not st.session_state.get(state_key, False):
        st.markdown("""
        <div class="agent-teaser">
          <div class="agent-title">🤖 AI Agent sẵn sàng phân tích</div>
          <div class="agent-body">Bấm nút <b>Phân tích bằng AI Agent</b> để hiển thị nhận xét chính sách mô phỏng.</div>
        </div>
        """, unsafe_allow_html=True)
        return
    with st.spinner("AI Agent đang phân tích kết quả mô hình..."):
        import time
        time.sleep(0.35)
    st.markdown(
        f"""
        <div class="ai-agent-card">
          <div class="agent-title">🤖 AI Policy Agent — mô phỏng</div>
          <div class="agent-body">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def run_colab_source(source_path: str, icon: str, title: str, labels: tuple[str, ...], agent_text: str) -> None:
    _header(icon, title, labels)
    tabs = st.tabs(
        [
            "📖 Bối cảnh & Mô hình",
            "🎛️ Tham số",
            "📊 Kết quả",
            "📈 Phân tích chính sách",
            "🤖 AI Agent",
        ]
    )
    containers = {
        TAB_CONTEXT: tabs[0].container(),
        TAB_CONFIG: tabs[1].container(),
        TAB_RESULT: tabs[2].container(),
        TAB_POLICY: tabs[3].container(),
    }
    touched = {k: False for k in containers}
    state = SimpleNamespace(current=TAB_CONTEXT)

    with tabs[1]:
        st.info(
            "Các tham số, dữ liệu và công thức được lấy trực tiếp từ file Python gốc. "
            "Runner chỉ chuyển print/display/plt.show sang Streamlit, không thay đổi logic tính toán."
        )
        st.caption(f"Nguồn logic: `{source_path}`")
        st.slider("Chiều cao biểu đồ Matplotlib", 420, 720, 520, 20, key=f"chart_height_{Path(source_path).stem}")
        st.selectbox(
            "Chế độ đọc báo cáo",
            ["Theo đúng thứ tự câu hỏi trong file gốc", "Tập trung kết quả và chính sách"],
            key=f"read_mode_{Path(source_path).stem}",
        )

    def use(target: str):
        touched[target] = True
        return containers[target]

    def streamlit_print(*args, **kwargs) -> None:
        sep = kwargs.get("sep", " ")
        text = sep.join(str(arg) for arg in args)
        state.current = _target_from_text(text)
        cleaned = _clean_print_text(text)
        if not cleaned:
            return
        target = use(state.current)
        if _is_section_text(cleaned):
            title, body = _section_title_from_text(cleaned)
            icon = "📌"
            if "CÂU" in title.upper():
                icon = "📊"
            elif "PHÂN TÍCH" in title.upper():
                icon = "📈"
            elif "KẾT LUẬN" in title.upper():
                icon = "✅"
            _section(target, title, icon)
            if body:
                target.markdown(body)
        elif state.current == TAB_POLICY:
            target.markdown(
                f"""
                <div class="ai-agent-card">
                  <div class="agent-title">📌 Phân tích chính sách</div>
                  <div class="agent-body">{html.escape(cleaned).replace(chr(10), '<br>')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            target.markdown(cleaned)

    def streamlit_display(obj=None, *args, **kwargs) -> None:
        target = use(state.current if state.current in {TAB_CONTEXT, TAB_CONFIG, TAB_POLICY} else TAB_RESULT)
        if isinstance(obj, Styler):
            target.dataframe(obj, use_container_width=True)
            _dataframe_note(target, obj.data)
        elif isinstance(obj, pd.DataFrame):
            target.dataframe(_safe_dataframe(obj), use_container_width=True)
            _dataframe_note(target, obj)
        elif isinstance(obj, pd.Series):
            target.dataframe(obj.to_frame(), use_container_width=True)
            _dataframe_note(target, obj.to_frame())
        else:
            target.write(obj)

    def streamlit_show(*args, **kwargs) -> None:
        fig = plt.gcf()
        if fig is not None and fig.axes:
            target = use(TAB_RESULT)
            fig.set_size_inches(11, 6)
            try:
                fig.tight_layout()
            except Exception:
                pass
            target.pyplot(fig, clear_figure=True)
            _matplotlib_note(target, fig)
        plt.close(fig)

    def streamlit_plotly_show(fig, *args, **kwargs) -> None:
        target = use(TAB_RESULT)
        try:
            fig.update_layout(height=540, margin=dict(l=70, r=40, t=90, b=80))
        except Exception:
            pass
        target.plotly_chart(fig, use_container_width=True)

    _prepare_solver_path()
    old_show = plt.show
    plt.show = streamlit_show
    try:
        import plotly.graph_objects as go

        old_plotly_show = go.Figure.show
        go.Figure.show = streamlit_plotly_show
    except Exception:
        old_plotly_show = None
    code = _read_colab_code(source_path)
    env = {
        "__name__": "__streamlit_colab__",
        "__file__": source_path,
        "print": streamlit_print,
        "display": streamlit_display,
    }
    try:
        exec(compile(code, source_path, "exec"), env)
    except Exception as exc:
        use(TAB_RESULT).warning(
            "Một phần file gốc không render được trong Streamlit. "
            "Phần tính toán đã chạy đến điểm lỗi và giao diện đã ẩn traceback kỹ thuật để tránh debug text."
        )
    finally:
        plt.show = old_show
        if old_plotly_show is not None:
            try:
                import plotly.graph_objects as go

                go.Figure.show = old_plotly_show
            except Exception:
                pass

    for key, container in containers.items():
        if not touched[key]:
            container.caption("File gốc không có khối hiển thị riêng cho tab này.")

    with tabs[4]:
        _ai_agent(agent_text)
