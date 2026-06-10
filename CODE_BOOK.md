| Tên Cột (Feature) | Kiểu Dữ liệu | Miền Giá Trị | Giải thích Ý nghĩa | Vai trò trong Pipeline |
| :--- | :--- | :--- | :--- | :--- |
| **crawl_date** | str | 04/26/2026 -> 05/03/2026 | Ngày hệ thống thu thập dữ liệu video. | Chỉ dùng EDA / Bị loại bỏ. |
| **region** | str | CA, GB, US, VN, AU, SG, NZ | Quốc gia mà video lọt vào tab thịnh hành. | Đặc trưng phân loại (Target Encoding). |
| **days_trending** | float64 | Từ 1 đến 8 | Số ngày liên tiếp video trụ trên tab thịnh hành. | Đặc trưng số. |
| **video_id** | str | Chuỗi ký tự Alphanumeric | Mã định danh duy nhất của mỗi video. | Định danh / Bị loại bỏ. |
| **title** | str | Văn bản thô (Text) | Tiêu đề của video. | Chỉ dùng EDA / Bị loại bỏ. |
| **channel_id** | str | Chuỗi ký tự Alphanumeric | Mã định danh duy nhất của kênh YouTube. | Định danh kênh / Bị loại bỏ |
| **channel_title** | str | Văn bản thô (Text) | Tên hiển thị của kênh YouTube. | Chỉ dùng EDA / Bị loại bỏ. |
| **published_at** | str | YYYY-MM-DD | Thời gian gốc xuất bản video lên YouTube. | Bị loại bỏ (Đã trích xuất ra các biến thời gian). |
| **publish_hour** | float64 |  0 -> 23 | Khung giờ xuất bản video trong ngày. | Đặc trưng số. |
| **publish_day_of_week** | str | Monday -> Sunday | Thứ trong tuần video được xuất bản. | Đặc trưng phân loại (Target Encoding). |
| **publish_month** | float64 | 3, 4, 5 | Tháng xuất bản video. | Đặc trưng số. |
| **days_since_publish** | float64 | 1 -> 36 | Số ngày tính từ lúc đăng video đến khi crawl. | Đặc trưng số. |
| **view_count** | float64 | 0 -> 66.9 triệu | Tổng lượt xem. Đã xử lý khuyết bằng Hybrid Imputation (Category Median + MICE). | Đặc trưng số. |
| **like_count** | float64 | 0 -> 1.57 triệu | Tổng lượt thích. Đã xử lý khuyết bằng Hybrid Imputation (Category Median + MICE). | Đặc trưng số. |
| **comment_count** | float64 | 0 -> 321 746 | Tổng số lượt bình luận của video. Đã xử lý khuyết bằng Hybrid Imputation (Category Median + MICE). | Đặc trưng số. |
| **like_rate** | float64 | 0 -> 0.458 | Tỷ lệ lượt thích trên mỗi lượt xem (Like/View). | Đặc trưng số. |
| **comment_rate** | float64 | 0 -> 0.098 | Tỷ lệ bình luận trên mỗi lượt xem. | Đặc trưng số. |
| **engagement_rate** | float64 | 0 -> 0.557 | Tổng tỷ lệ tương tác (Like + Comment) / View. | Đặc trưng số. |
| **views_per_day** | float64 | 0 -> 12.3 triệu | Vận tốc tăng view trung bình mỗi ngày. | Đặc trưng số. |
| **description_length** | float64 | 0 -> 5000 | Độ dài ký tự của phần mô tả video. | Đặc trưng số. |
| **tag_count** | float64 | 0 -> 72 | Số lượng từ khóa SEO đính kèm video. | Đặc trưng số. |
| **category_id** | int64 | 1, 2, 10 - 30 | Mã số danh mục gốc do YouTube quy định. | Biến mục tiêu gốc dùng để tạo target_category / Không đưa trực tiếp vào mô hình. |
| **category_name** | str | Music, Gaming, Education, Sports, Film & Animation, People & Blogs, Travel & Events, Science & Technology, Comedy, Others | Chủ đề của video (Các nhóm hiếm được gộp thành Others). | Biến mục tiêu cần dự đoán (Target y) |
| **duration_seconds** | float64 | 2 -> 390 702 | Tổng thời lượng video (tính bằng giây). | Đặc trưng số. |
| **definition** | str | hd, sd | Độ phân giải của video (Cao/Tiêu chuẩn). | Đặc trưng phân loại (Target Encoding). |
| **has_captions** | bool | True, False | Kiểm tra video có phụ đề hay không. | Đặc trưng nhị phân (0/1). |
| **default_language** | str | Mã ngôn ngữ | Ngôn ngữ mặc định được thiết lập cho video. | Đặc trưng phân loại (Target Encoding). |
| **license** | str | youtube, creativeCommon | Giấy phép bản quyền phân phối nội dung. | Đặc trưng phân loại (Target Encoding). |
| **made_for_kids** | bool | True, False | Nội dung có dành riêng cho trẻ em hay không. | Đặc trưng nhị phân (0/1). |
| **thumbnail_url** | str | URL hình ảnh | Đường dẫn ảnh đại diện của video. | Thông tin tham chiếu hình ảnh / Không sử dụng trong mô hình |
| **channel_subscriber_count** | float64 | 0 -> 311 triệu | Số người đăng ký kênh. | Đặc trưng số. |
| **channel_video_count** | float64 | 0 -> 577 546 | Tổng số video trên kênh. | Đặc trưng số. |
| **channel_view_count** | float64 | 0 -> 340 tỷ | Tổng lượt xem lũy kế của toàn bộ kênh. | Đặc trưng số. |
| **channel_age_days** | float64 | 0 -> 7619 | Tuổi thọ của kênh tính bằng ngày. | Đặc trưng số. |
| **url** | str | URL YouTube | Đường dẫn truy cập trực tiếp đến video. | Đường dẫn video / Định danh, bị loại bỏ |
| **title_length** | float64 | 1 -> 100 | Số lượng ký tự trong tiêu đề. | Đặc trưng số. |
| **title_word_count** | float64 | 1 -> 27 | Số lượng từ trong tiêu đề. | Đặc trưng số. |
| **title_has_number** | bool | True, False | Tiêu đề có chứa con số hay không. | Đặc trưng nhị phân (0/1). |
| **title_has_question** | bool | True, False | Tiêu đề có chứa dấu chấm hỏi (?) không. | Đặc trưng nhị phân (0/1). |
| **title_has_exclamation** | bool | True, False | Tiêu đề có chứa dấu chấm than (!) không. | Đặc trưng nhị phân (0/1). |
| **title_all_caps_word** | bool | True, False | Tiêu đề có chứa từ viết hoa toàn bộ (Clickbait). | Đặc trưng nhị phân (0/1). |
| **title_has_brackets** | bool | True, False | Tiêu đề có chứa dấu ngoặc [ ] hoặc ( ). | Đặc trưng nhị phân (0/1). |
| **title_emoji_count** | float64 | 0 -> 5, 17, 19 | Số lượng biểu tượng cảm xúc (Emoji) ở tiêu đề. | Đặc trưng số. |
| **tags** | str | Văn bản thô (Text) | Các từ khóa ngăn cách. | Dùng cho quá trình điền khuyết / EDA, bị loại bỏ khi huấn luyện mô hình. |
| **tags_source** | str | original, groq_llama3, nlp_regex | Minh chứng AI: Nguồn tạo ra tag (LLM/Regex). | Theo dõi nguồn sinh tags trong quá trình xử lý dữ liệu. |