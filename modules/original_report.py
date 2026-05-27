from __future__ import annotations

import ast
import html
import re
from functools import lru_cache
from pathlib import Path

import streamlit as st


SOURCE_FILES = {
    "bai01": "bai01_cobb_douglas_ai (1).py",
    "bai02": "bai02_lp_phanbongansachso (1).py",
    "bai03": "bai03_priority_index_nganh (1).py",
    "bai04": "bai04_lp_ngansach_nganh_vung (1).py",
    "bai05": "bài_5 (1).py",
    "bai06": "bài_6 (1).py",
    "bai07": "bài_7 (1).py",
    "bai08": "bài_8 (1).py",
    "bai09": "bài_9 (1).py",
    "bai10": "bài_10 (1).py",
    "bai11": "bài_11 (1).py",
    "bai12": "bài_12.py",
}

CATEGORY_META = {
    "context": ("1. Bối cảnh", "🌸"),
    "model": ("2. Mô hình toán học", "📐"),
    "result": ("3. Nhận xét sau từng kết quả", "📊"),
    "policy": ("4. Phân tích chính sách", "🎓"),
    "conclusion": ("5. Kết luận", "💗"),
}


def _source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _strip_colab_magics(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("!"):
            indent = line[: len(line) - len(stripped)]
            lines.append(indent + "pass")
        else:
            lines.append(line)
    return "\n".join(lines)


def _joined_string(node: ast.JoinedStr) -> str:
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        elif isinstance(value, ast.FormattedValue):
            try:
                parts.append("{" + ast.unparse(value.value) + "}")
            except Exception:
                parts.append("{...}")
    return "".join(parts)


def _arg_text(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return _joined_string(node)
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _clean_print_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned.append("")
            continue
        if set(stripped) <= {"=", "-", "─", "═"}:
            continue
        cleaned.append(line.rstrip())
    value = "\n".join(cleaned).strip()
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value


def _category(text: str) -> str:
    upper = text.upper()
    first_lines = "\n".join(text.splitlines()[:3]).upper()
    if "KẾT LUẬN" in first_lines or "KET LUAN" in first_lines or "CONCLUSION" in first_lines:
        return "conclusion"
    if "BỐI CẢNH" in first_lines or "MỤC TIÊU" in first_lines or "BÀI TOÁN ĐẶT RA" in first_lines:
        return "context"
    if "KẾT LUẬN" in upper or "KET LUAN" in upper or "CONCLUSION" in upper:
        return "conclusion"
    if (
        "MÔ HÌNH TOÁN" in first_lines
        or "MÔ HÌNH" in first_lines
        or "HÀM MỤC TIÊU" in first_lines
        or "RÀNG BUỘC" in first_lines
        or "CÔNG THỨC" in first_lines
        or "BIẾN QUYẾT ĐỊNH" in first_lines
    ):
        return "model"
    if (
        "PHÂN TÍCH CHÍNH SÁCH" in upper
        or "HÀM Ý" in upper
        or "KHUYẾN NGHỊ" in upper
        or "ĐỀ XUẤT" in upper
        or "RỦI RO" in upper
        or "POLICY" in upper
    ):
        return "policy"
    if (
        "MÔ HÌNH TOÁN" in upper
        or "MÔ HÌNH" in upper
        or "HÀM MỤC TIÊU" in upper
        or "RÀNG BUỘC" in upper
        or "CÔNG THỨC" in upper
        or "BIẾN QUYẾT ĐỊNH" in upper
    ):
        return "model"
    if "BỐI CẢNH" in upper or "MỤC TIÊU" in upper or "BÀI TOÁN ĐẶT RA" in upper:
        return "context"
    return "result"


def _card_title(text: str, default: str) -> str:
    for line in text.splitlines():
        stripped = line.strip(" -•\t")
        if not stripped:
            continue
        if len(stripped) > 120:
            continue
        return stripped
    return default


def _render_card(text: str, index: int, default_title: str) -> None:
    title = html.escape(_card_title(text, default_title))
    body = html.escape(text).replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="original-report-card">
          <div class="original-report-card-title">{index:02d}. {title}</div>
          <div class="original-report-card-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@lru_cache(maxsize=16)
def extract_print_blocks(page_id: str) -> tuple[tuple[str, str], ...]:
    source_name = SOURCE_FILES.get(page_id)
    if not source_name:
        return tuple()
    source_path = _source_root() / source_name
    if not source_path.exists():
        return tuple()

    code = _strip_colab_magics(source_path.read_text(encoding="utf-8", errors="replace"))
    tree = ast.parse(code)
    blocks: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "print":
            continue
        parts = [_arg_text(arg) for arg in node.args]
        text = _clean_print_text(" ".join(part for part in parts if part))
        if len(text) < 8:
            continue
        blocks.append((_category(text), text))
    return tuple(blocks)


def render_original_report(page_id: str) -> None:
    blocks = list(extract_print_blocks(page_id))
    if not blocks:
        return

    grouped = {key: [] for key in CATEGORY_META}
    for category, text in blocks:
        grouped.setdefault(category, []).append(text)

    st.markdown(
        """
        <div class="original-report-shell">
          <div class="original-report-kicker">Báo cáo học thuật gốc</div>
          <div class="original-report-title">Nội dung phân tích được khôi phục từ file Python gốc</div>
          <div class="original-report-note">
            Phần này đọc trực tiếp các đoạn <b>print(...)</b> trong file gốc và chuyển sang card Streamlit.
            Logic tính toán, bảng và biểu đồ phía trên không bị thay đổi.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs([f"{icon} {title}" for title, icon in CATEGORY_META.values()])
    for tab, (category, (title, _icon)) in zip(tabs, CATEGORY_META.items()):
        with tab:
            items = grouped.get(category, [])
            if category == "model" and not items:
                items = [
                    text
                    for _cat, text in blocks
                    if re.search(r"(^|\n)\s*(\d+\.\s*)?(M[1-6]|NĂM KỊCH BẢN|PHÂN BỔ|AIDEOM SCORE)", text.upper())
                ][:6]
            if not items:
                st.markdown(
                    f"""
                    <div class="original-report-card">
                      <div class="original-report-card-title">{html.escape(title)}</div>
                      <div class="original-report-card-body">
                        File gốc không có đoạn print riêng cho mục này; nội dung liên quan đã nằm trong các mục còn lại.
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                continue
            for index, text in enumerate(items, start=1):
                _render_card(text, index, title)
