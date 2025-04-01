# Miner G.O.D Tùy Chỉnh để Tối Đa Hóa Điểm Số từ Validator

Đây là phiên bản tùy chỉnh của miner subnet G.O.D, được tối ưu hóa để đạt điểm cao nhất có thể từ các validator.

## Các Tối Ưu Hóa Chính

### 1. Quản Lý Hàng Đợi Công Việc Thông Minh
- **Hàng Đợi Theo Ưu Tiên**: Các công việc được ưu tiên dựa trên chuyên môn với từng họ mô hình
- **Theo Dõi Hiệu Suất Lịch Sử**: Hệ thống học hỏi và nhận biết những họ mô hình nào nó hoạt động tốt
- **Tránh Thất Bại**: Từ chối các công việc thuộc họ mô hình đã thất bại trong quá khứ
- **Xử Lý Song Song**: Có thể xử lý nhiều công việc đồng thời (có thể cấu hình)

### 2. Tối Ưu Hóa Huấn Luyện Theo Từng Loại Mô Hình
- **Cấu Hình Riêng Cho Từng Họ Mô Hình**: Tham số huấn luyện tùy chỉnh cho các loại mô hình khác nhau:
  - Mô hình Llama sử dụng flash attention và bộ lập lịch tỷ lệ học cosine
  - Mô hình Mistral sử dụng các tối ưu hóa tương tự điều chỉnh cho kiến trúc của chúng
  - Mô hình Phi sử dụng batch size khác và bộ lập lịch tỷ lệ học tuyến tính
  - Mô hình Diffusion có tối ưu hóa riêng theo loại (SDXL vs Stable Diffusion)

### 3. Tiêu Chí Chấp Nhận Công Việc Thông Minh
- **Lọc Họ Mô Hình**: Chỉ chấp nhận công việc cho các họ mô hình mà nó có thể xử lý tốt
- **Quản Lý Hàng Đợi**: Kiểm soát kích thước hàng đợi để tránh quá tải hệ thống
- **Ràng Buộc Thời Gian**: Đặt giới hạn hợp lý về thời gian hoàn thành công việc
- **Cấu Hình Biến Môi Trường**: Dễ dàng cấu hình thông qua các biến môi trường:
  ```
  # Giới hạn công việc
  MAX_TEXT_TASK_HOURS=12
  MAX_IMAGE_TASK_HOURS=3
  
  # Họ mô hình được hỗ trợ (phân cách bằng dấu phẩy)
  ACCEPTED_MODEL_FAMILIES=llama,mistral
  
  # Đa xử lý
  MINER_MAX_CONCURRENT_JOBS=1
  
  # Kích thước hàng đợi công việc
  MAX_QUEUE_SIZE=5
  ```

### 4. Quy Trình Huấn Luyện Nâng Cao
- **Tích Hợp Tập Validation**: Tự động dành riêng 5% dữ liệu cho việc kiểm tra chéo
- **Lựa Chọn Checkpoint Tốt Nhất**: Lưu nhiều checkpoint và chọn checkpoint tốt nhất
- **Trích Xuất Metric**: Ghi lại và theo dõi các chỉ số hiệu suất từ quá trình huấn luyện
- **Hyperparameter Tối Ưu**: Cài đặt được cấu hình sẵn cho các loại mô hình khác nhau

## Cách Sử Dụng

1. Thiết lập các biến môi trường trong tệp `.env`
2. Khởi động miner với lệnh `task miner`
3. Theo dõi logs để xem quá trình chấp nhận và xử lý công việc

## Kết Quả Mong Đợi

Phiên bản miner tùy chỉnh này sẽ cải thiện đáng kể điểm số của bạn bằng cách:
1. Chỉ chấp nhận những công việc mà nó có thể hoàn thành một cách đáng tin cậy
2. Tối ưu hóa huấn luyện cho các kiến trúc mô hình khác nhau
3. Sử dụng các phương pháp tốt nhất để fine-tuning từng loại mô hình
4. Ưu tiên các công việc mà nó đã chứng minh được chuyên môn
5. Học hỏi từ hiệu suất trong quá khứ để liên tục cải thiện

Hệ thống sẽ tự động thích ứng với khả năng phần cứng của bạn và tập trung vào các họ mô hình mà nó đạt được kết quả tốt nhất, tối đa hóa điểm số từ validator theo thời gian.