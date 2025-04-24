# Updated people_counter_pipeline.py

import cv2
import os
import json
import yaml
import pickle
import numpy as np
import logging
from pathlib import Path
from collections import defaultdict, Counter
from ultralytics import YOLO
from boxmot import DeepOcSort, BotSort, StrongSort
import onnxruntime as ort

# Configure logging
logging.basicConfig(filename="people_counter.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ============ Setup Phase ============
def load_config(path="config.yaml"):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {path}")
        raise

def load_distortion_params(path="distortion_params.txt"):
    try:
        with open(path, "r") as f:
            return map(float, f.read().strip().split(","))
    except FileNotFoundError:
        logging.error("Distortion parameters file not found.")
        raise

def setup_undistortion_map(frame_shape, k1, k2, k3, angle):
    height, width, _ = frame_shape
    K = np.array([[width, 0, width // 2], [0, width, height // 2], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([k1, k2, 0, 0, k3], dtype=np.float32)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (width, height), 1, (width, height))
    map1, map2 = cv2.initUndistortRectifyMap(K, dist_coeffs, None, new_K, (width, height), cv2.CV_16SC2)
    M = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1) if angle != 0 else None
    return map1, map2, M

def undistort_frame(frame, map1, map2, M):
    remapped = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    if M is not None:
        remapped = cv2.warpAffine(remapped, M, (frame.shape[1], frame.shape[0]))
    return remapped

def load_tracker(tracker_type, reid_path, device):
    TRACKERS = {"DeepOcSort": DeepOcSort, "BotSort": BotSort, "StrongSort": StrongSort}
    tracker_cls = TRACKERS.get(tracker_type, DeepOcSort)
    return tracker_cls(reid_weights=Path(reid_path), half=True, device=device)

def load_recognition_model(model_path, db_path, device):
    try:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)
        with open(db_path, "rb") as f:
            db = pickle.load(f)
        db_embeddings, db_names = [], []
        for name, embeds in db.items():
            for e in embeds:
                norm = np.linalg.norm(e)
                if norm > 0:
                    db_embeddings.append(e / norm)
                    db_names.append(name)
        return session, session.get_inputs()[0].name, session.get_outputs()[0].name, np.array(db_embeddings, dtype=np.float32), db_names
    except FileNotFoundError as e:
        logging.error(f"File not found: {e.filename}")
        raise

def load_or_draw_roi(frame):
    roi_file = "roi_coords.json"
    if os.path.exists(roi_file):
        with open(roi_file, "r") as f:
            return json.load(f)
    else:
        coords = draw_polygon_and_get_coordinates(frame)
        with open(roi_file, "w") as f:
            json.dump(coords, f)
        return coords


# ============ Recognition & Detection ============
def recognize_employee(face_img, session, input_name, output_name, db_matrix, db_names, threshold=0.5):
    try:
        img = cv2.resize(face_img, (112, 112))
        img = ((img.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        output = session.run([output_name], {input_name: img})
        emb = output[0].flatten()
        emb = emb / np.linalg.norm(emb)
        sims = db_matrix @ emb
        max_idx = np.argmax(sims)
        if 1.0 - sims[max_idx] < threshold:
            return db_names[max_idx], 1.0 - sims[max_idx]
        return None, 1.0 - sims[max_idx]
    except Exception as e:
        logging.error(f"Recognition failed: {str(e)}")
        return None, None


def box_overlaps_quadr(box, quadr):
    x1, y1, x2, y2 = box
    corners = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    polygon = np.array(quadr, dtype=np.int32)
    if any(cv2.pointPolygonTest(polygon, pt, False) >= 0 for pt in corners + [center]):
        return True
    return False


# ============ Main Pipeline ============
def main(video_path, output_path):
    config = load_config()
    device = config.get("device", "cpu")

    person_model = YOLO(config['model_paths']['person_detector']).to(device)
    face_model = YOLO(config['model_paths']['face_detector']).to(device)
    tracker = load_tracker(config['tracker_type'], config['model_paths']['reid_model'], device)
    session, input_name, output_name, db_matrix, db_names = load_recognition_model(config['model_paths']['face_recognition'], config['employee_db'], device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error("Video file could not be opened.")
        return
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    ret, frame = cap.read()
    if not ret:
        logging.error("Could not read first frame.")
        return

    k1, k2, k3, angle = load_distortion_params()
    map1, map2, M = setup_undistortion_map(frame.shape, k1, k2, k3, angle)
    roi_coords = load_or_draw_roi(frame)

    detection_history = defaultdict(list)
    final_classification = {}
    skip_frames = config.get("frame_skip_interval", 3)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        frame = undistort_frame(frame, map1, map2, M)
        detections = person_model.predict(frame, classes=[0], verbose=False)[0].boxes

        if detections is None or not detections.xyxy.size:
            continue

        boxes = []
        for i, box in enumerate(detections.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box[:4])
            conf = detections.conf[i].item()
            if box_overlaps_quadr((x1, y1, x2, y2), roi_coords):
                boxes.append([x1, y1, x2, y2, conf, 0])  # class id is 0

        tracks = tracker.update(np.array(boxes, dtype=np.float32), frame) if boxes else []

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track[:5])
            if track_id in final_classification:
                continue
            crop = frame[y1:y2, x1:x2]
            face_box = face_model.predict(crop, verbose=False, conf=config['thresholds']['face_confidence'])[0].boxes
            if face_box and len(face_box.xyxy) > 0:
                fx1, fy1, fx2, fy2 = map(int, face_box.xyxy[0].cpu().numpy())
                fx1, fy1 = fx1 + x1, fy1 + y1
                fx2, fy2 = fx2 + x1, fy2 + y1
                face_img = frame[fy1:fy2, fx1:fx2]
                name, _ = recognize_employee(face_img, session, input_name, output_name, db_matrix, db_names, config['thresholds']['recognition'])
                if name:
                    detection_history[track_id].append("employee")
                else:
                    detection_history[track_id].append("visitor")

            # Final classification if enough history exists
            if len(detection_history[track_id]) >= 5:
                most_common = Counter(detection_history[track_id]).most_common(1)[0][0]
                final_classification[track_id] = most_common

            label = final_classification.get(track_id, "unknown")
            color = (0, 255, 0) if label == "employee" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        employee_ids = {tid for tid, label in final_classification.items() if label == 'employee'}
        visitor_ids = {tid for tid, label in final_classification.items() if label == 'visitor'}
        cv2.rectangle(frame, (10, 10), (300, 50), (0, 0, 0), -1)
        cv2.putText(frame, f"Employees: {len(employee_ids)} Visitors: {len(visitor_ids)}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if out.isOpened():
            out.write(frame)
        cv2.imshow("Processed Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("input_video.mp4", "output_video.mp4")
