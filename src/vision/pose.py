from ultralytics import YOLO

model = YOLO("yolov8n-pose.pt")

def detect_pose(frame):
    results = model(frame)
    return results