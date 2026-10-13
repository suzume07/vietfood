# VietFood68 — RTX 5070 desktop

Baseline detection cho RTX 5070 12 GB: YOLO11s, ảnh 640, AMP,
AutoBatch 0.65, mặc định tối đa 2 workers trên Windows.

## Báo cáo và kết quả thực nghiệm

[Báo cáo VietFood68](Bao_cao_VietFood68.md) trình bày cấu hình RTX 5070,
150 epoch, 6 biểu đồ, kết quả đủ 68 lớp và phân tích lỗi trên tập test.
Trên 3.284 ảnh test: precision **95,34%**, recall **94,54%**, mAP50 **97,23%**,
mAP50–95 **85,32%**. Số liệu và giao thức nằm trong
[reports/evaluation/summary.json](reports/evaluation/summary.json).

Bản trình bày để đọc/in: [PDF báo cáo — 30 trang](Bao_cao_VietFood68.pdf),
kèm [nguồn LaTeX](Bao_cao_VietFood68.tex). Để biên dịch lại trên Windows
(Times New Roman, Arial, Consolas), tải bộ công cụ portable rồi chạy:

```powershell
.venv/Scripts/python.exe scripts/setup_pdf_tools.py
.venv/Scripts/python.exe compile_report.py
```

Lần đầu cần Internet để tải Pandoc, Tectonic và các gói LaTeX. Công cụ và
cache nằm trong `.tools/` và `.cache/`, được loại khỏi Git.

Run `vietfood68_5070-2` đã hoàn tất; checkpoint cuối đã strip optimizer.
Các lệnh/task resume bên dưới dành cho run chưa hoàn tất có checkpoint hợp lệ,
không dùng để tiếp tục trạng thái optimizer của run đã kết thúc này.
Tên lớp hiện là `class_0`–`class_67`; dữ liệu nguồn có cả lớp người và mapping
cục bộ chưa được xác minh đầy đủ. Giao diện Streamlit mới ở mức thiết kế.

## Cài đặt (PowerShell)

Dùng Python 3.12; nếu `python` trỏ tới Windows Store, dùng `uv` có sẵn:

```powershell
uv venv --python 3.12 --seed .venv
uv pip install --python .venv/Scripts/python.exe torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
.venv/Scripts/python.exe check_gpu.py
```

PyTorch 2.8.0 CUDA 12.8 là baseline có hỗ trợ Blackwell, không phải khẳng định
bản mới nhất. Không cần CUDA Toolkit riêng cho wheel; cần driver tương thích.
Nguồn: [PyTorch wheels](https://pytorch.org/get-started/previous-versions/).

## Dataset và train

`path` trong `data.yaml` hiện trỏ tới `D:/archive/dataset`.
Cần `images/train`, `images/valid`, `images/test` cùng labels tương ứng trong
`labels/train`, `labels/valid`, `labels/test`. Label detection mỗi dòng:
`class_id x_center y_center width height`, tọa độ 0–1, ID 0–67.
Thay tên `class_0`…`class_67` theo đúng mapping dataset, không đổi thứ tự.
Nếu chỉ có ảnh chia thư mục theo món, không có bounding box, cần pipeline
classification thay vì detection.

```powershell
.venv/Scripts/python.exe train.py --smoke
.venv/Scripts/python.exe train.py
.venv/Scripts/python.exe train.py --resume D:/vietfood/runs/detect/vietfood68_5070/weights/last.pt
```

Smoke chạy 1 epoch với 5% train, batch 2; vẫn validation toàn bộ val.
Lần đầu cần Internet để tải weights. Trên Windows có ổ D, kết quả mặc định ở
`D:/vietfood/runs/detect/` (máy khác dùng `runs/detect/`), gồm
`weights/best.pt`, `weights/last.pt`, `results.csv`. Run trùng tên sẽ được
Ultralytics tăng số; dùng đường dẫn thực khi resume. Resume giữ cấu hình
checkpoint; ghi đè device/workers/cache và lưu vào thư mục run chứa checkpoint.

`--project` cho phép chọn thư mục đầu ra. Thay đổi mặc định chỉ áp dụng cho
run mới; phiên đang chạy vẫn dùng thư mục đã chọn lúc bắt đầu. Khi resume,
script dùng thư mục run chứa `weights/last.pt`, kể cả sau khi chuyển sang D.
`uv.toml` chuyển cache cài thư viện mới sang `D:/vietfood/uv-cache`.

Môi trường hiện tại được đặt ở `D:/vietfood/.venv`; đường dẫn `.venv` trong repo
là Windows junction nên các lệnh `.venv/Scripts/python.exe` vẫn dùng được.
Để tiếp tục run đã chuyển và hiện log trong terminal:

```powershell
.venv/Scripts/python.exe -u train.py --resume D:/vietfood/runs/detect/vietfood68_5070-2/weights/last.pt --workers 2 --cache off
```

`resume_terminal.ps1` chạy cùng lệnh và ghi thêm log ở
`D:/vietfood/runs/detect/vietfood68_5070-2/resume-terminal.log`.
Không chạy thêm lệnh resume khi đã có một phiên train của run này đang chạy.

Trong VS Code: **Terminal > Run Task > VietFood: Resume training** để chạy
tiếp trong terminal tích hợp, đồng thời lưu log trên D. Task dùng 3 workers,
prefetch 2 và giữ validation không có workers. Nếu đóng terminal khi train,
chạy lại task để tiếp tục từ checkpoint epoch đã hoàn tất gần nhất.

## Tinh chỉnh theo số đo

Chế độ ưu tiên dùng GPU, AutoBatch mục tiêu 90% VRAM; Windows vẫn dùng tối đa
2 workers để giảm áp lực RAM từ dataset lớn:

```powershell
.venv/Scripts/python.exe train.py --max-gpu
```

`--batch` và `--workers` ghi đè preset khi được truyền rõ ràng. Không tự bật
cache với 161.733 ảnh train: RAM và dung lượng SSD cần được đo trước.
90% là mục tiêu ước lượng, không đảm bảo GPU utilization 100% hoặc tốc độ tối ưu.
Nếu OOM, giảm xuống `--max-gpu --batch 0.80` hoặc dùng batch nguyên.
Chế độ này giữ model/độ phân giải, không ép model lớn hơn chỉ để lấp VRAM.
Khi resume, batch vẫn lấy từ checkpoint; preset không đổi batch của run cũ.
cuDNN benchmark chỉ bật sau AutoBatch để không vô hiệu hóa bước chọn batch.

### OpenCV Insufficient memory khi đọc ảnh

Đây là lỗi cấp phát bộ nhớ host khi decode ảnh, không phải thông báo CUDA OOM.
Phiên bản Ultralytics hiện cài prefetch 4 batch mỗi train worker; 8 workers
có thể giữ 32 batch đang chờ, ngoài state dataset và buffer ảnh mỗi process.
Giảm workers trước, giữ AMP và GPU:

```powershell
.venv/Scripts/python.exe train.py --max-gpu --workers 2 --cache off
```

Nếu vẫn lỗi, thử `--workers 0 --batch 8 --cache off` để bỏ subprocess/prefetch
và giảm batch. Không bật RAM cache cho toàn bộ dataset để chữa lỗi này.
Theo dõi Task Manager > Memory > Committed; nếu sát giới hạn, đóng ứng dụng
chiếm RAM và kiểm tra Windows pagefile đang để system-managed, ổ đĩa còn chỗ.
Script không tự thay đổi cấu hình pagefile. Chưa có checkpoint thì chạy lại
run mới; có `last.pt` thì dùng `--resume ... --workers 2`.

`memory_trainer.py` thay đổi cách nạp dữ liệu cho cả run mới và resume:
train giữ số workers đã chọn, mỗi worker mặc định prefetch 2 batch (stock là 4);
validation chạy trên tiến trình chính, không tạo thêm pool workers * 2.
Giữ nguyên batch GPU, AMP, augmentation và checkpoint optimizer.
Điều này giảm áp lực RAM/commit; validation có thể chậm hơn. Vẫn phải đo
thời gian epoch và đỉnh bộ nhớ, không chỉ RAM trống lúc bắt đầu.

`--prefetch 1` giảm RAM dùng cho hàng đợi nếu cần; `--prefetch 2` cho phép
chuẩn bị nhiều batch hơn khi GPU đang chờ dữ liệu. `resume_terminal.ps1`
hiện dùng 3 train workers, prefetch 2 và không có validation workers.

Khi muốn đo thêm tốc độ, thử từng thay đổi sau khi có checkpoint: 4 train
workers, giữ batch 28; rồi thử batch 32 với số workers đã chọn. Khi resume,
`--batch` được truyền chỉ khi bạn chỉ định rõ, mặc định giữ batch checkpoint.
So sánh ảnh/giây hoặc thời gian epoch (batch lớn hơn làm giảm số batch mỗi
epoch nên không so trực tiếp it/s). Tăng batch có thể thay đổi effective batch
và nhịp cập nhật optimizer; theo dõi validation. Giữ cache RAM tắt với dataset
này; VRAM/RAM trống không tự chứng minh rằng batch lớn hơn sẽ nhanh hơn.

- AutoBatch 0.65 chừa VRAM cho desktop nhưng không đảm bảo tránh mọi OOM.
  Nếu OOM, chạy lại `--batch 8` hoặc `--batch 4`. Batch chỉ ước lượng ban đầu.
- So sánh `--workers 2`, `--workers 4`, `--workers 8` theo thời gian epoch
  sau warmup. Main guard đã có để hỗ trợ multiprocessing trên Windows.
- Mặc định không cache. Thử `--cache disk` trên SSD nếu nghẽn đọc ảnh
  (cần thêm dung lượng), hoặc `--cache ram` khi đủ RAM chứa dataset.
- Thử `--model yolo11m.pt` để so sánh mAP/thời gian. `--imgsz 768` có thể
  giúp vật thể nhỏ nhưng dùng nhiều VRAM hơn; giảm batch nếu cần.
- AMP, TF32, cuDNN benchmark được bật; deterministic tắt để ưu tiên tốc độ.
  Seed 42 không đảm bảo tái lập tuyệt đối. Compile chưa bật trên Windows.
- Tránh ảnh trùng giữa train/val/test. Chọn mô hình theo val rồi đánh giá
  test; không suy ra chất lượng từ smoke test.

Tham số theo [Ultralytics Train](https://docs.ultralytics.com/modes/train/).
