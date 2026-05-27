from __future__ import annotations

from pathlib import Path

from .source_runner import run_colab_source


def _source_path() -> str:
    root = next(
        p
        for p in Path("D:/").iterdir()
        if (p / "bai04_lp_ngansach_nganh_vung (1).py").exists()
    )
    return str(root / "bai04_lp_ngansach_nganh_vung (1).py")


def show() -> None:
    run_colab_source(
        _source_path(),
        "🗺️",
        "Bài 4 — LP ngân sách số theo vùng-ngành",
        ("PuLP/CVXPY", "Công bằng vùng", "Logic Colab gốc"),
        """
        <b>Nhận xét chính sách:</b><br>
        Bài 4 đang chạy trực tiếp file Python gốc. Mô hình cho thấy ràng buộc công bằng vùng miền
        tạo đánh đổi rõ giữa hiệu quả GDP gain và mục tiêu bao trùm. Khi hệ số công bằng quá cao,
        vùng có Digital Index thấp phải nhận phần ngân sách chuyển đổi số lớn hơn trần vùng nên
        bài toán có thể infeasible. Tác nhân khuyến nghị dùng nghiệm đã hiệu chỉnh trong file gốc
        để so sánh chi phí công bằng, đồng thời ưu tiên nhân lực số và hạ tầng số ở các vùng nền
        tảng còn yếu trước khi mở rộng đầu tư AI trực tiếp.
        """,
    )
