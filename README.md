# YOUTUBE TRENDING CATEGORY PREDICTION
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![YouTube API v3](https://img.shields.io/badge/YouTube-Data_API_v3-red.svg)](https://developers.google.com/youtube/v3)
[![Groq API](https://img.shields.io/badge/GenAI-Llama--3_8B_Instant-green.svg)](https://groq.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## GIỚI THIỆU DỰ ÁN
**Dự án được xây dựng trong khuôn khổ môn học Tiền xử lý và xây dựng bộ dữ liệu (DS108) - Thu thập dữ liệu video Youtube Trending và Tiền xử lý phục vụ bài toán dự đoán danh mục dựa trên dữ liệu tương tác.**


Dự án này tập trung vào việc giải quyết bài toán phân loại đa lớp (Multiclass Classification) trên tập dữ liệu mất cân bằng nghiêm trọng, với điểm nhấn đặc biệt ở giai đoạn Tiền xử lý: Sử dụng Hybrid GenAI (Llama-3 8B Instant) & NLP để nội suy dữ liệu khuyết thiếu và MICE để bảo toàn phân phối gốc.


* **Mục tiêu:** Xây dựng hệ thống phân loại danh mục tự động đạt hiệu năng cao, giải quyết triệt để bài toán mất cân bằng dữ liệu (Class Imbalance).
* **Kết quả dự kiến:** Mô hình dự đoán danh mục chính xác, giảm thiểu hiện tượng học vẹt (Shortcut Learning) thông qua kỹ thuật Target Encoding đồng cấp.

---

##  KIẾN TRÚC LUỒNG DỮ LIỆU (DATA PIPELINE) & HUẤN LUYỆN MÔ HÌNH (TRAIN MODEL)

Luồng xử lý dữ liệu được thiết kế tuần tự, đảm bảo tính tái lập (Reproducibility) và chống rò rỉ dữ liệu (Anti-Data Leakage) tuyệt đối:

```text
[Dữ liệu thô gốc - ds108_raw.csv]
      │
      ├──> 1. Phân tách dữ liệu
      │      └──> Cơ chế: Stratified Split (80:20) dựa theo nhãn category_name
      │           Cô lập hoàn toàn tập Train và tập Test ngay từ đầu.
      │
      ├──> 2. Điền khuyết dữ liệu 
      │      ├──> Biến văn bản: Gọi Groq API (Llama-3 8B Instant) nội suy ngữ nghĩa cho cột tags
      │      └──> Biến số (Chiến lược 2 giai đoạn):
      │           ├─ Stage 1: Điền Category Median cho nhóm lỗi hệ thống (sub=0)
      │           └─ Stage 2: Kích hoạt MICE (ExtraTrees) điền khuyết tự nhiên (MCAR)
      │           (*) Lưu ý: Bộ Imputer chỉ .fit() trên Train và .transform() lên Test.
      │
      ├──> 3. Lọc trùng lặp logic 
      │      └──> Áp dụng khóa phức hợp [video_id + region + crawl_date]
      │           Loại bỏ các dòng cào trùng lặp trong cùng một ngày, giữ lại bản ghi đầu tiên.
      │
      ├──> 4. Kỹ thuật đặc trưng 
      │      └──> Áp dụng Target Encoder với Smoothing=10 cho các categorical features
      │           để chuyển chữ thành số dựa trên phân phối nhãn của tập Train.
      │
      └──> 5. Huấn luyện Mô hình & Xử lý mất cân bằng 
             └──> Nạp ma trận số thuần túy vào LightGBM / XGBoost cài đặt Class Weight
```
---

## CẤU TRÚC KHO LƯU TRỮ

```text
ds108-youtube-category-prediction/
│
├── data/
│   ├── raw/                                          <- Dữ liệu thô gốc ban đầu
│   │   └── ds108_raw.csv
│   │
│   └── processed/                                    <- Dữ liệu sau quá trình tiền xử lý
│       ├── ds108_raw_tags_completed.csv              <- Dữ liệu sau khi hoàn tất bổ sung Tags bằng AI
│       ├── ds108_checkpoint_hybrid.csv               <- Dữ liệu checkpoint trong quá trình xử lý missing tags
│       ├── ds108_cleaned.csv                         <- Bộ dữ liệu sạch cuối cùng dùng cho phân tích EDA
│       ├── train_cleaned_ds108.csv                   <- Tập dữ liệu huấn luyện sau khi làm sạch (Train Set)
│       └── test_cleaned_ds108.csv                    <- Tập dữ liệu kiểm thử sau khi làm sạch (Test Set)
│
├── scripts/                                          <- (Tùy chọn) Mã nguồn thu thập & xử lý dữ liệu
│   ├── 00_data_collection.py                         <- Script Python thu thập dữ liệu từ YouTube API
│   └── 01_tag_imputation.py                          <- Script Python sử dụng Groq API để bổ sung Tags bị thiếu
│
├── notebooks/                                        <- (Chạy thực tế) Notebook phục vụ phân tích, xây dựng mô hình
│   ├── 01_initial_eda.ipynb                          <- Khám phá tổng quan dữ liệu thô
│   ├── 02_cleaning_and_imputation.ipynb              <- Tiền xử lý dữ liệu
│   ├── 03_final_eda.ipynb                            <- Phân tích khám phá dữ liệu sau xử lý
│   └── 04_feature_engineering_and_modeling.ipynb     <- Xây dựng đặc trưng, huấn luyện mô hình
│                                                     
│
├── requirements.txt                                  <- Danh sách các thư viện Python cần thiết
├── CODE_BOOK.md                                      <- Từ điển dữ liệu mô tả ý nghĩa và kiểu dữ liệu của các biến
└── README.md                                         <- Tài liệu hướng dẫn sử dụng và mô tả cấu trúc dự án
```

---

## HƯỚNG DẪN CHẠY CODE

Để chạy lại toàn bộ mã nguồn của dự án này mà không gặp lỗi hệ thống, vui lòng thực hiện tuần tự theo hướng dẫn sau:

### 1. Cài đặt môi trường
Mở Terminal/Command Prompt tại thư mục dự án và chạy lệnh cài đặt thư viện:
```bash
pip install -r requirements.txt
```

### 2. Bỏ qua các bước gọi API tốn kém (Scripts)
2 file [`00_data_collection.py`](./scripts/00_data_collection.py) và [`01_tag_imputation.py`](./scripts/01_tag_imputation.py) nằm trong thư mục `scripts/` có sử dụng API giới hạn lượt truy cập (Rate-Limited) và mất rất nhiều thời gian để thực thi. 2 file này đã chạy sẵn toàn bộ tiến trình này và lưu kết quả vào thư mục `data/`. Do đó, không cần thiết phải chạy lại.

### 3. Hướng dẫn chạy Notebooks Phân tích & Mô hình
Toàn bộ phần đánh giá, trực quan hóa và huấn luyện thuật toán được phân bổ logic trong thư mục [`notebooks/`](./notebooks/). Trước khi nhấn *Run All*, vui lòng kiểm tra và trỏ lại đường dẫn file trong các hàm đọc/ghi dữ liệu (ví dụ: `pd.read_csv()`) sao cho khớp với cấu trúc thư mục trên máy tính của bạn.

Quy trình thực hiện được đề xuất theo thứ tự sau:

* **Bước 1**: Chạy file [`01_initial_eda.ipynb`](./notebooks/01_initial_eda.ipynb) để thực hiện khám phá tổng quan trên dữ liệu thô, kiểm tra cấu trúc dữ liệu, kiểu dữ liệu và các vấn đề ban đầu như giá trị khuyết thiếu hoặc bất nhất.
* **Bước 2**: Chạy file [`02_cleaning_and_imputation.ipynb`](./notebooks/02_cleaning_and_imputation.ipynb) để thực hiện quy trình tiền xử lý dữ liệu bao gồm làm sạch dữ liệu, xử lý giá trị khuyết thiếu, kiểm tra dữ liệu trùng lặp.
* **Bước 3**: Chạy file [`03_final_eda.ipynb`](./notebooks/03_final_eda.ipynb) để thực hiện phân tích khám phá trên bộ dữ liệu đã làm sạch, trực quan hóa các đặc trưng và rút ra các nhận định phục vụ quá trình lựa chọn mô hình.
* **Bước 4**: Chạy file [`04_feature_engineering_and_modeling.ipynb`](./notebooks/04_feature_engineering_and_modeling.ipynb) để thực hiện quá trình xây dựng đặc trưng, huấn luyện mô hình học máy, đánh giá hiệu năng và phân tích kết quả dự đoán.

---

## TÀI LIỆU BỔ SUNG & TÀI NGUYÊN

**1. Từ điển Dữ liệu (Data Dictionary)**
Thông tin chi tiết về định nghĩa, kiểu dữ liệu, miền giá trị quét thực tế từ dữ liệu đã làm sạch và vai trò xử lý toán học của từng cột trong mô hình, vui lòng tra cứu chi tiết tại file [CODE_BOOK.md](./CODE_BOOK.md).

**2. Tập Dữ liệu (Dataset)**
Toàn bộ dữ liệu thô (raw data) được thu thập tự động từ YouTube Data API v3 đã được nhóm đóng gói và công bố công khai trên nền tảng Kaggle. Việc này giúp đảm bảo tính minh bạch của số liệu và hỗ trợ cộng đồng dễ dàng tái sử dụng (reproduce) lại toàn bộ quy trình Data Pipeline của dự án.

Truy cập Dataset chính thức của nhóm trên Kaggle **[tại đây](https://www.kaggle.com/datasets/phn217/ds-raw)**
