from __future__ import annotations

from pathlib import Path

from .source_runner import run_colab_source


POLICY_SECTIONS = (
    (
        "1. Nếu bỏ ràng buộc công bằng, vốn sẽ chảy về vùng nào?",
        "📊",
        """
Khi bỏ ràng buộc công bằng, ngân sách có xu hướng chảy về các vùng-ngành có hệ số hiệu quả cao nhất, tức là nơi tạo ra giá trị mục tiêu Z lớn hơn trên mỗi đơn vị vốn đầu tư.

Điều này phản ánh logic tối ưu hóa thuần túy: mô hình sẽ ưu tiên vùng có năng lực hấp thụ công nghệ, hạ tầng số, nhân lực số hoặc hệ số AI cao hơn.

Tuy nhiên, nếu chỉ tối đa hóa hiệu quả mà bỏ qua công bằng vùng, ngân sách có thể tập trung quá nhiều vào các vùng phát triển hơn. Hậu quả xã hội dài hạn là khoảng cách số giữa các vùng tăng lên, vùng yếu hơn khó bắt kịp, và mục tiêu phát triển cân bằng quốc gia bị ảnh hưởng.
        """,
    ),
    (
        "2. Ràng buộc trần ngân sách mỗi vùng C3 làm giảm Z* bao nhiêu phần trăm?",
        "🔍",
        """
Ràng buộc trần ngân sách mỗi vùng C3 có thể được hiểu như một chính sách phân quyền, vì nó ngăn ngân sách tập trung quá mức vào một số vùng có hiệu quả cao.

Để đánh giá tác động của C3, cần so sánh hai giá trị:

* Z* khi có ràng buộc C3
* Z* khi bỏ ràng buộc C3

Tỷ lệ giảm được tính theo công thức:

Mức giảm Z* (%) = ((Z_không_C3 - Z_có_C3) / Z_không_C3) × 100

Nếu mức giảm Z* nhỏ, ràng buộc C3 có thể chấp nhận được vì nó giúp tăng tính cân bằng và phân quyền mà không làm mất quá nhiều hiệu quả. Nếu mức giảm quá lớn, cần xem lại mức trần ngân sách vì chính sách phân quyền đang làm giảm đáng kể hiệu quả đầu tư.
        """,
    ),
    (
        "3. Tây Nguyên có sàn 5.000 tỷ nhưng hệ số AI thấp, nên đầu tư vào AI hay H và I trước?",
        "🎯",
        """
Vùng Tây Nguyên có sàn ngân sách 5.000 tỷ nhưng hệ số AI chỉ ở mức thấp, khoảng 0,45. Điều này cho thấy nếu đầu tư trực tiếp vào AI ngay, hiệu quả biên có thể chưa cao do nền tảng công nghệ và khả năng hấp thụ còn hạn chế.

Mô hình gợi ý rằng nên ưu tiên đầu tư vào H và I trước:

* H: nhân lực số, đào tạo kỹ năng, năng lực sử dụng công nghệ
* I: hạ tầng số, kết nối dữ liệu, nền tảng kỹ thuật

Sau khi H và I được cải thiện, đầu tư vào AI sẽ có khả năng tạo hiệu quả cao hơn. Nói cách khác, Tây Nguyên không nên bị bỏ lại phía sau, nhưng chính sách đầu tư cần đi theo lộ trình: xây nền tảng trước, mở rộng AI sau.
        """,
    ),
    (
        "4. Hàm ý chính sách",
        "📋",
        """
Kết quả mô hình cho thấy phân bổ ngân sách số không chỉ là bài toán tối đa hóa hiệu quả kinh tế, mà còn là bài toán cân bằng giữa hiệu quả, công bằng vùng và khả năng triển khai thực tế.

Nếu bỏ ràng buộc công bằng, vốn sẽ tập trung vào vùng mạnh hơn. Nếu đặt ràng buộc công bằng quá chặt, mô hình có thể kém hiệu quả hoặc không khả thi. Vì vậy, chính sách ngân sách số cần sử dụng các ngưỡng công bằng linh hoạt, phù hợp với năng lực từng vùng.

Đối với các vùng có hệ số AI thấp như Tây Nguyên, chính sách không nên ép đầu tư AI ngay ở mức cao, mà nên ưu tiên nâng cấp hạ tầng số và nhân lực số trước.

**Kết quả nổi bật.** Mô hình sử dụng tổng ngân sách 50.000 tỷ, sàn mỗi vùng 5.000 tỷ, trần mỗi vùng C3 là 12.000 tỷ và hệ số AI của Tây Nguyên khoảng 0,45. Các con số này cho thấy ràng buộc công bằng và trần vùng là hai yếu tố quyết định khả năng triển khai.

**Liên hệ chính sách Việt Nam.** Kết quả gắn với **Quyết định 749/QĐ-TTg** về chuyển đổi số quốc gia và **Quyết định 411/QĐ-TTg** về kinh tế số-xã hội số: phân bổ ngân sách cần đi cùng mục tiêu thu hẹp khoảng cách số vùng.

**Đánh đổi cần lưu ý:** tăng trưởng và công bằng vùng; nới trần ngân sách giúp tăng Z* nhưng có thể làm vốn tập trung vào vùng mạnh hơn.

**Khuyến nghị hành động.** Giữ sàn ngân sách cho vùng yếu; rà soát mức trần C3 bằng so sánh Z* có/không có trần; ưu tiên H và I cho Tây Nguyên trước AI; theo dõi Digital Index sau đầu tư để điều chỉnh ngưỡng công bằng.
        """,
    ),
    (
        "5. Kết luận",
        "✅",
        """
Bài 4 cho thấy mô hình LP là công cụ hữu ích để phân tích bài toán phân bổ ngân sách số theo vùng-ngành.

Kết quả thảo luận chính sách cho thấy:

* Nếu bỏ ràng buộc công bằng, vốn có xu hướng chảy về các vùng hiệu quả cao hơn.
* Ràng buộc trần ngân sách C3 làm giảm Z* nhưng có vai trò phân quyền và hạn chế tập trung vốn.
* Với Tây Nguyên, do hệ số AI thấp, nên ưu tiên đầu tư vào nhân lực số và hạ tầng số trước khi mở rộng đầu tư AI.
* Chính sách ngân sách số cần cân bằng giữa hiệu quả đầu tư, công bằng vùng và khả năng triển khai thực tế.
        """,
    ),
)


def _source_path() -> str:
    """
    Lấy đường dẫn file source bài 4 theo kiểu tương đối trong repo.

    File bai04_lp_ngansach_nganh_vung (1).py đang nằm ở thư mục gốc repo,
    ngang hàng với app.py.
    """
    project_root = Path(__file__).resolve().parent.parent
    source_file = project_root / "bai04_lp_ngansach_nganh_vung (1).py"

    if not source_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file source bài 4: {source_file}. "
            "Hãy kiểm tra file 'bai04_lp_ngansach_nganh_vung (1).py' "
            "có nằm ngang hàng với app.py trong repo GitHub không."
        )

    return str(source_file)


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
        policy_sections=POLICY_SECTIONS,
        format_result_titles=True,
        show_source_controls=False,
    )
