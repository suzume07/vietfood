
import torch
from ultralytics import YOLO

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM (GB):", round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
    ))
else:
    print("Dang chay tren CPU")

print("Ultralytics YOLO: OK")