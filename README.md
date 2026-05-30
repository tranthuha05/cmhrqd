# AIDEOM-VN Policy Lab

Web app hỗ trợ phân tích các mô hình ra quyết định phát triển kinh tế số Việt Nam trong kỷ nguyên AI.

Hệ thống tích hợp 12 mô hình, cho phép người dùng:

* lựa chọn mô hình thông qua sidebar;
* điều chỉnh tham số trực tiếp trên giao diện;
* xem bảng kết quả và biểu đồ;
* đọc phần phân tích chính sách;
* tham khảo nội dung diễn giải từ AI Agent.

## 1. Truy cập website trực tuyến

Mở link:

```text
https://cmhrqd-njd53d8tgap2rs2iw5jcwl.streamlit.app
```

## 2. Yêu cầu môi trường

* Python 3.10 trở lên
* Trình duyệt web
* Kết nối internet khi cài thư viện

## 3. Cài đặt và chạy local

### Bước 1. Giải nén project

Giải nén file zip hoặc rar vào một thư mục trên máy tính.

### Bước 2. Mở Terminal tại thư mục project

Ví dụ trên Windows:

```powershell
cd "D:\Các mô hình ra quyết định"
```

### Bước 3. Cài đặt thư viện

```powershell
pip install -r requirements.txt
```

### Bước 4. Khởi chạy web app

```powershell
streamlit run app.py
```

### Bước 5. Mở website local

Nếu trình duyệt không tự mở, truy cập:

```text
http://localhost:8501
```

## 4. Cấu trúc project

```text
cmhrqd/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── .streamlit/
├── modules/
│
├── bai01_...py
├── bai02_...py
├── ...
└── bai12_...py
```

## 5. Nội dung 12 mô hình

1. Cobb-Douglas mở rộng với AI và số hóa
2. LP phân bổ ngân sách số
3. Priority Index cho 10 ngành
4. LP ngân sách theo ngành-vùng
5. MIP lựa chọn 15 dự án
6. TOPSIS xếp hạng 6 vùng
7. NSGA-II Pareto
8. Tối ưu động 2026–2035
9. Lao động và AI
10. Stochastic Programming
11. Q-learning
12. Hệ thống tích hợp AIDEOM-VN

## 6. Lưu ý

* Không xóa các file Python bài 1–12 vì đây là source logic của từng mô hình.
* Không cần đưa thư mục `__pycache__` hoặc file `.pyc` vào bộ cài.
* Để dừng web app local, nhấn:

```text
Ctrl + C
```

## 7. Source code GitHub

```text
https://github.com/tranthuha05/cmhrqd
