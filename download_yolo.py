from ultralytics import YOLO

print("Loading YOLO26x...")

model = YOLO("yolo26x.pt")

print("YOLO26x loaded successfully!")
print("Number of classes:", len(model.names))