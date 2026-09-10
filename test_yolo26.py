from ultralytics import YOLO
import torch
import time

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

model = YOLO("yolo26x.pt")

print("YOLO26x loaded!")
print("Starting GPU webcam inference...")

results = model.predict(
    source=0,
    show=True,
    conf=0.40,
    imgsz=640,
    device=0,
    stream=True,
    verbose=False
)

for result in results:
    pass