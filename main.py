import cv2
from ultralytics import YOLO
import supervision as sv

cap = cv2.VideoCapture(0)

model = YOLO("runs/detect/train/weights/best.pt")
box = sv.BoxAnnotator(
    thickness=1,
    text_thickness=1,
    text_scale=1
)

strToFrame = ""
lastTotal = 0

while True:
    ret, frame = cap.read()
    result = model(frame)[0]
    detections = sv.Detections.from_yolov8(result).with_nms(threshold=0.5)
    labels = [
        f"{model.model.names[class_id]}"
        for _, _, _, class_id, _ in detections
    ]
    frame = box.annotate(
        scene=frame,
        detections=detections,
        labels=labels
    )
    cv2.imshow('yolo', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
