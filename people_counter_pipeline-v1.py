import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from deepface import DeepFace
import numpy as np
import os
from collections import defaultdict, Counter
import logging
from boxmot import BotSort, StrongSort, DeepOcSort
from pathlib import Path
import onnxruntime as ort
import pickle

# Configure logging
logging.basicConfig(
    filename="perople_counter.log",
    level=logging.INFO,  # Log level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# 1. Preprocess to undistort the frame
def preprocess_frame(frame):
    """
    Preprocesses the input frame by performing distortion correction
    Args:
        frame: input frame
    Returns:
        undistorted frame: processed frame with distortion removed 
    """
    # Load distortion parameters
    with open("distortion_params.txt", "r") as f:
        k1, k2, k3, angle = map(float, f.read().strip().split(","))

    height, width, shape = frame.shape

    K = np.array([[width, 0, width // 2], [0, width, height // 2], [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([k1, k2, 0, 0, k3], dtype=np.float32)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (width, height), 1, (width, height))

    undistorted = cv2.undistort(frame, K, dist_coeffs, None, new_K)

    if angle != 0:
        M = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1)
        undistorted_frame = cv2.warpAffine(undistorted, M, (width, height))
    
    return undistorted_frame

# 2. Select coordinate to draw ROI polygon

def draw_polygon_and_get_coordinates(frame):
    """
    Opens a video, allows the user to draw a polygon area on the
    first frame by clicking four points, and returns the coordinates
    of the four points.
    Args:
        video_path: path of the input video
    Returns:
        points: list of polygon coordinates
    """
    logging.info("Successfully loaded first frame")
    points = []
    drawing = False

    def draw_quad(event, x, y, flags, param):
        nonlocal points, drawing, frame

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                temp_frame = frame.copy()
                # Draw all previously selected points
                for pt in points:
                    cv2.circle(temp_frame, pt, 3, (0, 0, 255), -1)
                # Draw the current point as a preview
                cv2.circle(temp_frame, (x, y), 3, (0, 0, 255), -1)
                cv2.imshow("First Frame", temp_frame)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            # Append the point on mouse button up
            points.append((x, y))
            temp_frame = frame.copy()
            # Draw all selected points
            for pt in points:
                cv2.circle(temp_frame, pt, 3, (0, 0, 255), -1)
            # Draw polygon with 4 points
            if len(points) == 4:
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(temp_frame, [pts], True, (0, 255, 0), 2)
                logging.info("Polygon drawn on frame")
            cv2.imshow("First Frame", temp_frame)

    cv2.namedWindow("First Frame")
    cv2.setMouseCallback("First Frame", draw_quad)
    cv2.imshow("First Frame", frame)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Press 'q' to quit without confirming
            logging.warning("User exited without confirming ROI selection")
            break
        elif key == ord("c"):  # Press 'c' to confirm the selection
            if len(points) == 4:
                logging.info(f"Selected ROI points: {points}")
                cv2.destroyAllWindows()
                return points
            else:
                logging.warning("User pressed 'c' but did not select exactly four points.")

    cv2.destroyAllWindows()
    return None

# 3. Check overlap between person bounding box and ROI polygon

def box_overlaps_quadr(box, quadr):
    """
    Check if the bounding box for a person overlaps with the entrance gate area
    Args:
        box: bounding box of person
        quadr: coordinates of ROI polygon
    Returns:
        True: if there is overlap
        False: if there is no overlap
    """
    try:
        x1, y1, x2, y2 = box
        # Define the centre of the bottom line of bounding box
        bottom_center = (x1+(x2-x1)//2,y1+(y2-y1))
        quadr_np = np.array(quadr, dtype=np.int32)
        # Check if any rectangle corner is inside the quadrilateral.:
        if cv2.pointPolygonTest(quadr_np, bottom_center, False) >= 0:
            return True
        return False
    except Exception as e:
        logging.error(f"ROI check failed: {str(e)}")
        return False

# 4. Detect people in the entrance and get their bounding box coordinates
 
def detect_people(model, frame, quadr):
    """
    Detect persons entering the office and determine their bounding boxes
    Args:
        model: person detection model
        frame: input frame
        quadr: ROI coordinates for detection
    Returns:
        List of tuple for each person to be passed to tracker:
            box_dims: bounding box coordinates
            conf: confidence score
            cls_id: class Id (always 0 for person)
    """
    results = model.predict(frame, classes=[0], verbose=False)
    detections = []
    try:
        if results and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            conf = results[0].boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, conf):
                x1, y1, x2, y2 = map(int, box)
                if quadr:
                    if not box_overlaps_quadr((x1,y1,x2,y2), quadr):
                        continue
                detections.append([x1,y1,x2,y2,conf])
            return detections
    except Exception as e:
        print(f"Error during person detection: {e}")
        return []
    

# 5. Track people in the frame using the bounding box coordinates

def track_people(tracker, frame, bounding_boxes):
    """
        Track the persons in the frame uisng the bounding boxes
        from the person detection block
        Input args:
            tracker: tracker model
            frame: input frame
            bounding_boxes: bounding box for detections obtained from detect_people,
                              each as ([x, y, w, h], confidence, class_id)
        Returns:
            track_boxes: list of tuples [([x, y, w, h], track_id)] 
                                  for persons currently visible in the frame
    """
    detections_np = np.array([[*det[:4], det[4], 0] for det in bounding_boxes], dtype=np.float32)
    print('Detections:', detections_np)
    try:
        frame_contiguous = np.ascontiguousarray(frame)
        tracks = tracker.update(detections_np, frame_contiguous)
        print('Tracks:', tracks)
        if tracks.size == 0:
            return []
        tracked_objs = []
        for track in tracks:
            if len(track) >= 5:
                x1, y1, x2, y2 = map(int, track[:4])
                track_id = int(track[4])
                tracked_objs.append((track_id, x1, y1, x2, y2))
            else:
                print(f"Warning: Unexpected track format from tracker: {track}")
        print('Tracked Objects:', tracked_objs)
        return tracked_objs
    except Exception as e:
        logging.error(f"Tracking failed: {str(e)}")
        return []

# 6. Extract the faces of detected people

def extract_face(model, person_roi, conf=0.5, frame_offset=(0,0)):
    """
    Extracts the face from a detected person in a video frame
    Args:
        - frame: current video frame
        - track_id: unique tracking ID assigned to the detected person
        - box: bounding box coordinates (x, y, width, height) of the detected 
    Returns:
        - bounding box of the detected face if a confident face is found
    """
    face_bbox = None
    max_face_area = -1
    if person_roi is None or person_roi.size == 0:
        return None
    
    try:
        # Perform inference using face detection model on the person ROI
        results = model.predict(person_roi, verbose=False, conf=conf)

        if results and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy() # Bounding boxes of detected faces in ROI coordinates
            confs = results[0].boxes.conf.cpu().numpy()

            for box in boxes:
                # Face coordinates relative to the person_roi
                x1_roi, y1_roi, x2_roi, y2_roi = map(int, box)

                # Ensure validity of the face coordinates
                h_roi, w_roi = person_roi.shape[:2]
                x1_roi = max(0, x1_roi)
                y1_roi = max(0, y1_roi)
                x2_roi = min(w_roi, x2_roi)
                y2_roi = min(h_roi, y2_roi)
                if x1_roi >= x2_roi or y1_roi >= y2_roi: 
                    continue # Invalid box

                # Ensure that the biggest face in the ROI is saved
                area = (x2_roi - x1_roi) * (y2_roi - y1_roi)
                if area > max_face_area:
                    max_face_area = area
                    # Offset the ROI coordinates to get original frame coordinates
                    offset_x, offset_y = frame_offset
                    x1 = x1_roi + offset_x
                    y1 = y1_roi + offset_y
                    x2 = x2_roi + offset_x
                    y2 = y2_roi + offset_y
                    face_bbox = [x1, y1, x2, y2]

    except Exception as e:
        print(f"Error during face detection in ROI: {e}")
        return None
    
    return face_bbox

#7. Recognize employee among detected people

def recognize_employee(model_path, employee_db_path, face_img, threshold=0.5, input_image_size=(112, 112)):
    """
    detects and recognizes employees in a given video frame using a pre-trained model.
    Args:
        - employee_model: model used for employee detection
        - frame: input frame
    Returns:
        - List of tuples containing bounding box and confidence score of detection
    """
    session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    # Load employee embeddings
    with open(employee_db_path, "rb") as f:
        employee_embeddings = pickle.load(f)
    embeddings_list = []
    names_list = []
    for name, embeddings in employee_embeddings.items():
        for embedding in embeddings:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embeddings_list.append(embedding / norm)
                names_list.append(name)
    db_embeddings_matrix = np.array(embeddings_list, dtype=np.float32)
    db_names_list = names_list
    # Preprocess input image
    img = cv2.resize(face_img, input_image_size)
    img = (img.astype(np.float32) - 127.5) / 128.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    # Extract embeddings
    output = session.run([output_name], {input_name: img})
    embedding = output[0].flatten()
    embedding = embedding / np.linalg.norm(embedding)
    # Recognize employee using similarity score
    similarities = db_embeddings_matrix @ embedding
    max_idx = np.argmax(similarities)
    min_distance = 1.0 - similarities[max_idx]
    if min_distance < threshold:
        return db_names_list[max_idx], min_distance
    return None, min_distance

def main(video_path,output_path):
    model = YOLO('models/yolov11n-mod.pt')
    tracker_model_path=Path('models/osnet_x0_25_msmt17.pt')
    TRACKER_CLASS = DeepOcSort
    tracker_params = {
        'reid_weights': tracker_model_path,
        'half': True,
        'device': 'cpu'
    }
    tracker = TRACKER_CLASS(**tracker_params)
    face_detection_model = YOLO('models/yolov11n-face-mod.pt')
    rec_model_path = 'models/model.onnx'
    employee_db_path = 'employees.pkl'
    # tracker = DeepSort(
    #         max_age=15,
    #         n_init=5,
    #         nms_max_overlap=0.6,
    #         max_cosine_distance=0.2
    #     )
    employee_model = YOLO('employee_model.pt')
    logging.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video")
        return None

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Dictionary to store detection history for each track_id
    detection_history = defaultdict(list)
    # Dictionary to store final decision
    final_classification = {}
    skip_frames = 3
    frame_count = 1
    first_frame = True
    while cap.isOpened():
        try:
            ret, frame = cap.read()
            if not ret:
                logging.error("Error reading the first frame")
                break
            frame_count += 1
            if frame_count % skip_frames != 0:
                continue
            frame = preprocess_frame(frame)
            if first_frame:
                quadrilateral = draw_polygon_and_get_coordinates(frame)
                if not quadrilateral:
                    logging.error("ROI Selection failed. Exiting!")
                    return
                first_frame = False

            # Detect persons in frame intersecting the defined area
            detections = detect_people(model, frame, quadrilateral)

            # Update the tracker
            tracked_people = track_people(tracker, frame, detections)
            print(tracked_people)
            for track_id, x1, y1, x2, y2 in tracked_people:
                if track_id in final_classification:
                    continue  # Already classified

                # Extract face region from frame
                person_crop = frame[y1:y2, x1:x2]
                face_bbox = extract_face(face_detection_model, person_crop, frame_offset=(x1, y1))
                if face_bbox:
                    fx1, fy1, fx2, fy2 = face_bbox
                    face_img = frame[fy1:fy2, fx1:fx2]

                    identity, distance = recognize_employee(rec_model_path, employee_db_path, face_img)
                    if identity:
                        final_classification[track_id] = 'employee'
                    else:
                        final_classification[track_id] = 'visitor'

                # Optionally, annotate frame with result
                label = final_classification.get(track_id, 'unknown')
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if label == 'employee' else (0, 0, 255), 2)
                cv2.putText(frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Count current unique employee and visitor IDs
            employee_ids = {tid for tid, label in final_classification.items() if label == 'employee'}
            visitor_ids = {tid for tid, label in final_classification.items() if label == 'visitor'}

            # Draw count box on top-left corner
            info_text = f"Employees: {len(employee_ids)}  Visitors: {len(visitor_ids)}"
            cv2.rectangle(frame, (10, 10), (300, 50), (0, 0, 0), -1)  # Background box
            cv2.putText(frame, info_text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            # Write annotated frame to output
            cv2.imshow('Processed Video', frame)
            out.write(frame)

        except Exception as e:
            logging.error(f"Video processing failed: {str(e)}")   
    try:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    except Exception as e:
        logging.error(f"Cleanup failed: {str(e)}")

if __name__ == "__main__":
    video_path = 'MicrosoftTeams-video (1).mp4'  # Replace with your video path
    output_path = os.path.join(os.getcwd(),'pipeline_output.mp4')
    main(video_path, output_path)