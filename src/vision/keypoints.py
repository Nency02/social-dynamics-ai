import math


LEFT_SHOULDER_INDEX = 5
RIGHT_SHOULDER_INDEX = 6


def calculate_shoulder_angle(person_keypoints):
    """Return torso orientation angle in degrees based on shoulder line."""
    if len(person_keypoints) <= RIGHT_SHOULDER_INDEX:
        return None

    left = person_keypoints[LEFT_SHOULDER_INDEX]
    right = person_keypoints[RIGHT_SHOULDER_INDEX]

    # Missing keypoints are commonly returned as zeros.
    if (left[0] == 0 and left[1] == 0) or (right[0] == 0 and right[1] == 0):
        return None

    dx = float(right[0] - left[0])
    dy = float(right[1] - left[1])
    angle_deg = math.degrees(math.atan2(dy, dx))
    return round(angle_deg, 2)


def extract_keypoints(results):
    people = []

    for r in results:
        if r.keypoints is not None:
            keypoints = r.keypoints.xy.cpu().numpy()
            keypoint_conf = None
            if r.keypoints.conf is not None:
                keypoint_conf = r.keypoints.conf.cpu().numpy()

            boxes_xyxy = None
            boxes_conf = None
            if r.boxes is not None and r.boxes.xyxy is not None:
                boxes_xyxy = r.boxes.xyxy.cpu().numpy()
                boxes_conf = r.boxes.conf.cpu().numpy()

            for i, person in enumerate(keypoints):
                orientation_angle_deg = calculate_shoulder_angle(person)

                bbox_xyxy = None
                confidence = None
                if boxes_xyxy is not None and i < len(boxes_xyxy):
                    bbox_xyxy = boxes_xyxy[i].tolist()
                    confidence = round(float(boxes_conf[i]), 4)

                person_keypoint_conf = None
                if keypoint_conf is not None and i < len(keypoint_conf):
                    person_keypoint_conf = keypoint_conf[i].tolist()

                people.append({
                    "id": i,
                    "bbox_xyxy": bbox_xyxy,
                    "confidence": confidence,
                    "keypoints": person.tolist(),
                    "keypoint_confidence": person_keypoint_conf,
                    "orientation_angle_deg": orientation_angle_deg
                })

    return people