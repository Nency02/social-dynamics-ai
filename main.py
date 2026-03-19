import cv2
import json
import time

from src.vision.pose import detect_pose
from src.vision.keypoints import extract_keypoints

cap = cv2.VideoCapture(0)
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    # Pose detection
    results = detect_pose(frame)

    # Extract keypoints
    people = extract_keypoints(results)

    # Save keypoints
    payload = {
        "frame_id": frame_id,
        "timestamp": time.time(),
        "people": people,
    }

    with open("outputs/keypoints.json", "w") as f:
        json.dump(payload, f)

    # Show output
    annotated = results[0].plot()
    cv2.imshow("Pose Detection", annotated)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()