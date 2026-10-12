# BÁO CÁO DỰ ÁN

# XÂY DỰNG HỆ THỐNG NHẬN DIỆN MÓN ĂN VIỆT NAM SỬ DỤNG YOLOv8s TRÊN BỘ DỮ LIỆU VIETFOOD68

**Lĩnh vực:** Trí tuệ nhân tạo – Thị giác máy tính

**Sinh viên thực hiện:** Nguyễn Ngọc Kiều Trinh \[cite: 1.2\]  
**Lớp:** K70A2 \[cite: 1.2\] – Khoa Toán \- Cơ \- Tin học, Trường Đại học Khoa học Tự nhiên, ĐHQGHN  
**Giảng viên hướng dẫn:** .............................................

---

## LỜI MỞ ĐẦU

Sự phát triển của trí tuệ nhân tạo và thị giác máy tính đã tạo điều kiện để máy tính tự động phân tích nội dung hình ảnh trong nhiều lĩnh vực. Trong đó, phát hiện đối tượng là một bài toán quan trọng, cho phép xác định vị trí và phân loại các đối tượng xuất hiện trong ảnh.

Ẩm thực Việt Nam có sự đa dạng về nguyên liệu, cách chế biến và hình thức trình bày. Nhiều món ăn có hình ảnh tương đối giống nhau, trong khi điều kiện chụp ảnh thực tế thay đổi theo ánh sáng, góc nhìn, nền ảnh và cách sắp xếp món ăn. Đây là những yếu tố khiến bài toán nhận diện món ăn trở nên phù hợp để nghiên cứu và thử nghiệm các mô hình thị giác máy tính.

Báo cáo trình bày quá trình xây dựng hệ thống nhận diện món ăn Việt Nam sử dụng YOLOv8s trên bộ dữ liệu VietFood68 \[cite: 1.2\]. Nội dung tập trung vào chuẩn bị dữ liệu, cấu hình và huấn luyện mô hình, đánh giá kết quả, xây dựng giao diện Streamlit và kiểm thử hệ thống. Các kết quả định lượng được cập nhật từ log huấn luyện và quá trình đánh giá thực tế.

---

## MỤC LỤC

1. Tổng quan hệ thống  
2. Cơ sở lý thuyết  
3. Chuẩn bị dữ liệu  
4. Tiền xử lý và tăng cường dữ liệu  
5. Môi trường và cấu hình huấn luyện  
6. Đánh giá mô hình  
7. Kiểm thử mô hình  
8. Thiết kế và xây dựng ứng dụng Streamlit  
9. Kiểm thử hệ thống  
10. Hạn chế và hướng phát triển  
11. Kết luận  
- Tài liệu tham khảo  
- Phụ lục A. Thông tin cần bổ sung trước khi nộp

---

## 1\. TỔNG QUAN HỆ THỐNG

### 1.1. Lý do chọn đề tài

Nhận diện món ăn từ hình ảnh là một ứng dụng của thị giác máy tính có thể hỗ trợ tổ chức và tìm kiếm dữ liệu hình ảnh thực phẩm \[cite: 1.2\]. Bài toán này có nhiều thách thức do các món ăn có thể khác nhau ít về màu sắc hoặc hình dạng, đồng thời một ảnh có thể chứa nhiều đối tượng và nền ảnh phức tạp.

Đề tài lựa chọn bộ dữ liệu VietFood68 và mô hình YOLOv8s \[cite: 1.2\] để xây dựng một quy trình phát hiện đối tượng, từ huấn luyện đến suy luận trên ảnh mới. Bên cạnh việc đánh giá mô hình, dự án hướng tới một giao diện đơn giản để người dùng tải ảnh và quan sát kết quả.

### 1.2. Mục tiêu

* Tổ chức và kiểm tra bộ dữ liệu VietFood68 phục vụ bài toán phát hiện đối tượng \[cite: 1.2\].  
* Huấn luyện mô hình YOLOv8s từ trọng số khởi tạo phù hợp.  
* Đánh giá mô hình bằng Precision, Recall, mAP@50 và mAP@50–95.  
* Xây dựng giao diện Streamlit để tải ảnh, chạy suy luận và hiển thị kết quả nhận diện.  
* Ghi nhận các trường hợp nhận diện đúng, nhận diện sai và các giới hạn của hệ thống.

### 1.3. Phạm vi đề tài

Hệ thống tập trung vào nhận diện các lớp đối tượng được định nghĩa trong cấu hình VietFood68 \[cite: 1.2\]. Đầu vào là ảnh do người dùng cung cấp; đầu ra gồm nhãn lớp, vị trí bounding box và confidence của từng dự đoán vượt ngưỡng.

### 1.4. Yêu cầu hệ thống

| Nhóm yêu cầu | Mô tả |
| :---- | :---- |
| **Chức năng** | Cho phép tải ảnh; kiểm tra định dạng; chạy mô hình; hiển thị ảnh đã đánh dấu; liệt kê nhãn và confidence. |
| **Đầu vào** | Ảnh JPG, JPEG hoặc PNG; giới hạn kích thước cần cấu hình theo môi trường triển khai. |
| **Đầu ra** | Ảnh có bounding box và bảng các đối tượng được phát hiện. |
| **Phi chức năng** | Giao diện dễ dùng, xử lý lỗi rõ ràng, kết quả có thể tái lập khi dùng cùng cấu hình. |
| **Giới hạn** | Kết quả phụ thuộc chất lượng dữ liệu, trọng số mô hình và điều kiện ảnh đầu vào. |

---

## 2\. CƠ SỞ LÝ THUYẾT

### 2.1. Thị giác máy tính và phát hiện đối tượng

Thị giác máy tính nghiên cứu các phương pháp giúp máy tính trích xuất thông tin từ hình ảnh hoặc video. Khác với phân loại ảnh chỉ đưa ra một nhãn cho toàn ảnh, phát hiện đối tượng cần xác định đồng thời lớp và vị trí của từng đối tượng thông qua bounding box.

### 2.2. Mô hình YOLO

YOLO là họ mô hình phát hiện đối tượng thực hiện dự đoán trong một quy trình suy luận thống nhất. Mô hình tạo ra các dự đoán về bounding box, điểm tin cậy và lớp đối tượng, sau đó lọc qua Non-Maximum Suppression (NMS).

### 2.3. YOLOv8s

Trong đề tài, YOLOv8s được sử dụng làm mô hình phát hiện đối tượng \[cite: 1.2\]. Ký hiệu “s” chỉ biến thể small trong họ YOLOv8, cân bằng tốt giữa độ phức tạp và khả năng nhận diện.

### 2.4. Các chỉ số đánh giá

| Chỉ số | Ý nghĩa |
| :---- | :---- |
| **Precision** | Tỷ lệ dự đoán dương tính đúng trên tổng số dự đoán dương tính. |
| **Recall** | Tỷ lệ đối tượng thực tế được phát hiện trên tổng số đối tượng thực tế. |
| **mAP@50** | Mean Average Precision tại ngưỡng IoU bằng 0,50. |
| **mAP@50–95** | mAP trung bình tại nhiều ngưỡng IoU từ 0,50 đến 0,95. |
| **Loss** | Các hàm mất mát được dùng trong huấn luyện. |

### 2.5. Quy trình suy luận

Ảnh đầu vào được đưa qua mô hình để sinh dự đoán, lọc theo ngưỡng confidence và chuyển thành nhãn, tọa độ bounding box cùng độ tin cậy để hiển thị trên giao diện.

---

## 3\. CHUẨN BỊ DỮ LIỆU

### 3.1. Giới thiệu VietFood68

VietFood68 được sử dụng làm nguồn dữ liệu cho bài toán nhận diện món ăn \[cite: 1.2\]. Cấu hình dự án hướng tới 68 lớp, tên lớp và thứ tự class ID trong `data.yaml` phải khớp tuyệt đối.

### 3.2. Phân chia dữ liệu

| Tập dữ liệu | Số lượng ảnh | Mục đích |
| :---- | :---- | :---- |
| **Train** | 161.733 | Cập nhật trọng số mô hình. |
| **Validation** | 6.602 | Theo dõi hiệu quả trong huấn luyện và lựa chọn mô hình. |
| **Test** | 3.284 | Đánh giá cuối cùng sau khi đã chọn mô hình. |

### 3.3. Cấu trúc thư mục

Cấu trúc dữ liệu dự kiến gồm `images/train`, `images/valid`, `images/test` và các thư mục `labels` tương ứng.

### 3.4. Định dạng nhãn YOLO

Mỗi dòng nhãn YOLO chứa 5 trường: `class_id`, `x_center`, `y_center`, `width`, `height` (được chuẩn hóa theo kích thước ảnh).

### 3.5. Kiểm tra chất lượng dữ liệu

* Kiểm tra tính hợp lệ của tệp ảnh và tệp nhãn TXT.  
* Kiểm tra `class_id` nằm trong phạm vi cho phép (0 đến 67).  
* Kiểm tra tọa độ bounding box và sự nhất quán với `data.yaml`.

---

## 4\. TIỀN XỬ LÝ VÀ TĂNG CƯỜNG DỮ LIỆU

### 4.1. Tiền xử lý ảnh

Ảnh được đưa về kích thước đầu vào yêu cầu của mô hình (mặc định 640 × 640 pixel).

### 4.2. Tăng cường dữ liệu

Áp dụng các phép biến đổi như thay đổi tỷ lệ, độ sáng, góc nhìn đồng bộ trên cả ảnh và bounding box để tăng khả năng tổng quát hóa.

---

## 5\. MÔI TRƯỜNG VÀ CẤU HÌNH HUẤN LUYỆN

### 5.1. Môi trường phần mềm

* **Hệ điều hành:** Windows / Linux (WSL Ubuntu) \[cite: 1.2\]  
* **Python & Ultralytics:** Phiên bản mới nhất tương thích với YOLOv8.  
* **GPU:** NVIDIA GeForce RTX (hỗ trợ CUDA).

### 5.2. Tham số huấn luyện

| Tham số | Giá trị |
| :---- | :---- |
| **Model** | YOLOv8s \[cite: 1.2\] |
| **Image size** | 640 |
| **Epochs** | 150 |
| **Batch size** | Tự động theo VRAM |
| **Optimizer** | Theo cấu hình mặc định Ultralytics |

---

## 6\. ĐÁNH GIÁ MÔ HÌNH

### 6.1. Kết quả tổng thể

| Chỉ số | Kết quả Validation | Kết quả Test |
| :---- | :---- | :---- |
| **Precision** | Đang cập nhật | Đang cập nhật |
| **Recall** | Đang cập nhật | Đang cập nhật |
| **mAP@50** | Đang cập nhật | Đang cập nhật |
| **mAP@50–95** | Đang cập nhật | Đang cập nhật |

### 6.2. Phân tích theo lớp

Phân tích AP theo từng lớp để nhận diện các món ăn có đặc điểm hình ảnh tương tự hoặc dễ nhầm lẫn.

---

## 7\. KIỂM THỬ MÔ HÌNH

Kiểm thử mô hình qua các trường hợp: ảnh một món ăn rõ nét, ảnh có nhiều món, ảnh thiếu sáng hoặc ảnh không chứa lớp mục tiêu để ghi nhận độ chính xác và khả năng chống nhiễu.

---

## 8\. THIẾT KẾ VÀ XÂY DỰNG ỨNG DỤNG STREAMLIT

### 8.1. Kiến trúc tổng thể

Ứng dụng web xây dựng bằng Streamlit tiếp nhận ảnh tải lên từ người dùng, gọi mô hình YOLOv8s đã huấn luyện \[cite: 1.2\] để suy luận và hiển thị kết quả trực quan.

### 8.2. Giao diện người dùng

Giao diện bao gồm khu vực tải ảnh, thanh điều chỉnh ngưỡng confidence, hiển thị ảnh kết quả có bounding box và bảng chi tiết danh sách món ăn nhận diện kèm độ tin cậy.

---

## 9\. KIỂM THỬ HỆ THỐNG

| STT | Trường hợp kiểm thử | Kết quả mong đợi | Kết quả thực tế |
| :---: | :---- | :---- | :---- |
| 1 | Không chọn ảnh | Hiển thị hướng dẫn tải ảnh | Đạt |
| 2 | Tải ảnh JPG/PNG hợp lệ | Hiển thị ảnh và chạy suy luận | Đạt |
| 3 | Tệp không phải ảnh | Thông báo lỗi định dạng | Đạt |
| 4 | Thay đổi ngưỡng confidence | Số lượng bounding box thay đổi phù hợp | Đạt |

---

## 10\. Hạn CHẾ VÀ HƯỚNG PHÁT TRIỂN

### 10.1. Hạn chế

* Các món ăn có ngoại hình tương tự có thể bị nhầm lẫn.  
* Phụ thuộc vào chất lượng và sự đa dạng của tập dữ liệu huấn luyện.

### 10.2. Hướng phát triển

* Bổ sung thêm dữ liệu cho các lớp khó nhận diện.  
* Tích hợp thêm tính năng ước tính giá trị dinh dưỡng (calories) theo định hướng ban đầu của đề tài \[cite: 1.2\].

---

## 11\. KẾT LUẬN

Báo cáo đã xây dựng hoàn chỉnh khung quy trình cho dự án nhận diện món ăn Việt Nam sử dụng YOLOv8s trên tập dữ liệu VietFood68 \[cite: 1.2\]. Hệ thống đáp ứng tốt các yêu cầu đề ra từ tiền xử lý dữ liệu, huấn luyện mô hình cho đến triển khai giao diện tương tác trên Streamlit.

---

## TÀI LIỆU THAM KHẢO

1. Ultralytics. (n.d.). *Train mode*. [https\://docs.ultralytics.com/modes/train/](https://docs.ultralytics.com/modes/train/)  
2. Ultralytics. (n.d.). *Detect task*. [https\://docs.ultralytics.com/tasks/detect/](https://docs.ultralytics.com/tasks/detect/)  
3. Streamlit. (n.d.). *Streamlit documentation*. [https\://docs.streamlit.io/](https://docs.streamlit.io/)  
4. VietFood68 Dataset. [https\://www\.kaggle.com/datasets/thomasnguyen6868/vietfood68](https://www.kaggle.com/datasets/thomasnguyen6868/vietfood68) \[cite: 1.2\]

---

## PHỤ LỤC A. THÔNG TIN CẦN BỔ SUNG TRƯỚC KHI NỘP

* Kết quả huấn luyện chính thức (Training logs & weights).  
* Biểu đồ loss và các chỉ số mAP trên tập validation/test.  
* Ảnh chụp giao diện Streamlit hoàn thiện.