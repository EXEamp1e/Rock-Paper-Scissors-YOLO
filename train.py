from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(data="./datasets/RPS/data.yaml", epochs=1, batch=8, imgsz=640)
