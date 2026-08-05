"""
Tram detector logger — extracts tram crossing timestamps from video.

Usage in a notebook or script:
    from tram_detector.logger import logger, KalmanCentroid, Track, SimpleTracker
    logger(model, location="pcss", footage_id="pcss_07-08", ...)

CSV columns written: location, time_start, time_end, direction, class, footage_id
"""

import os
import time
import numpy as np
import cv2
from datetime import datetime, timedelta
from tqdm import tqdm

try:
    import supervision as sv
except ImportError:
    raise ImportError("supervision is required: pip install supervision")


# Class-merge map applied to every raw detector class name at ingestion,
# before tracking/priority-resolution/traffic-counting/logging ever see it —
# so merged-away classes are simply never produced again, regardless of what
# track_classes/traffic_class_list a caller passes. See
# labelStudio/merge_classes.py for the one-time fix applied to already-logged
# data (labelStudio/output/*.json, labelStudio/input/label_config.xml,
# labeling/log*.csv).
CLASS_MERGE_MAP = {"bus": "truck"}


def _remap_class(cls: str) -> str:
    return CLASS_MERGE_MAP.get(cls, cls)


# ── Kalman centroid tracker ───────────────────────────────────────────────────

class KalmanCentroid:
    def __init__(self, cx, cy, dt=1.0):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], dtype=np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.array([[cx],[cy],[0.],[0.]], dtype=np.float32)

    def predict(self):
        p = self.kf.predict()
        return float(p[0,0]), float(p[1,0])

    def correct(self, cx, cy):
        self.kf.correct(np.array([[cx],[cy]], dtype=np.float32))


class Track:
    _next_id = 0

    def __init__(self, bbox, class_name, frame_index, timestamp_dt, dt_seconds):
        self.id = Track._next_id
        Track._next_id += 1
        self.bbox = bbox
        cx, cy = self._centroid(bbox)
        self.kalman = KalmanCentroid(cx, cy, dt=dt_seconds)
        self.class_name = class_name
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.start_frame = frame_index
        self.time_start = timestamp_dt
        self.last_seen_frame = frame_index
        self.time_last_seen = timestamp_dt
        self.start_centroid = (cx, cy)
        self.last_centroid  = (cx, cy)
        self.logged = False
        self.isNew  = True

    @staticmethod
    def _centroid(bbox):
        x1, y1, x2, y2 = bbox
        return (x1+x2)/2.0, (y1+y2)/2.0

    def predict(self):
        cx, cy = self.kalman.predict()
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        self.bbox = (cx-w/2, cy-h/2, cx+w/2, cy+h/2)
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def update(self, bbox, class_name, frame_index, timestamp_dt):
        self.bbox = bbox
        cx, cy = self._centroid(bbox)
        self.kalman.correct(cx, cy)
        self.hits += 1
        self.time_since_update = 0
        self.age += 1
        self.class_name = class_name
        self.last_seen_frame = frame_index
        self.time_last_seen  = timestamp_dt
        self.last_centroid   = (cx, cy)


class SimpleTracker:
    def __init__(self, max_age_frames=150, dist_threshold=80, dt_seconds=1.0):
        self.tracks = []
        self.max_age = max_age_frames
        self.dist_threshold = dist_threshold
        self.dt_seconds = dt_seconds

    @staticmethod
    def _centroid(bbox):
        x1, y1, x2, y2 = bbox
        return (x1+x2)/2.0, (y1+y2)/2.0

    @staticmethod
    def _dist(a, b):
        return np.hypot(a[0]-b[0], a[1]-b[1])

    def predict(self):
        for t in self.tracks:
            t.predict()

    def update(self, detections, frame_index, timestamp_dt):
        if not self.tracks:
            for bbox, cls in detections:
                self.tracks.append(Track(bbox, cls, frame_index, timestamp_dt, self.dt_seconds))
            return

        dist_matrix = np.full((len(self.tracks), len(detections)), np.inf, dtype=np.float32)
        for i, tr in enumerate(self.tracks):
            tc = self._centroid(tr.bbox)
            for j, (bbox, _) in enumerate(detections):
                dist_matrix[i, j] = self._dist(tc, self._centroid(bbox))

        matched_t, matched_d = set(), set()
        for dist, i, j in sorted(
            [(dist_matrix[i,j], i, j) for i in range(len(self.tracks)) for j in range(len(detections))],
            key=lambda x: x[0]
        ):
            if dist > self.dist_threshold:
                break
            if i in matched_t or j in matched_d:
                continue
            self.tracks[i].update(detections[j][0], detections[j][1], frame_index, timestamp_dt)
            matched_t.add(i)
            matched_d.add(j)

        for j, (bbox, cls) in enumerate(detections):
            if j not in matched_d:
                self.tracks.append(Track(bbox, cls, frame_index, timestamp_dt, self.dt_seconds))


class Annotator:
    def __init__(self):
        self.box_ann   = sv.BoxAnnotator()
        self.label_ann = sv.LabelAnnotator()

    def annotate(self, frame, detections):
        labels = [
            f"{cls} {conf:.2f}"
            for cls, conf in zip(detections['class_name'], detections.confidence)
        ]
        img = self.box_ann.annotate(scene=frame, detections=detections)
        return self.label_ann.annotate(scene=img, detections=detections, labels=labels)


# ── Main logger function ──────────────────────────────────────────────────────

def logger(
    model,
    location               = "pcss",
    begin_date             = "2025-08-06 11:36:00",
    log_filename           = "log.csv",
    video_filename         = "v1.mp4",
    footage_id             = None,        # unique ID for this video file; defaults to filename stem
    reset_log              = True,
    verbose                = True,
    confidence_threshold   = 0.95,
    direction_threshold_px = 10,
    unseen_seconds_to_end  = 5.0,
    dist_threshold         = 50,
    device                 = "cuda",
    locations_csv          = "labeling/locations.csv",
    track_classes          = ["tram"],    # list of classes to track (e.g., ["tram", "bus", "truck"])
    traffic_threshold      = 10,          # log "traffic" when count of (car+bus+truck) > this
    traffic_class_list     = None,        # classes to count for traffic (e.g., ["car", "bus", "truck"])
    auxiliary_models       = None,        # dict of {name: model} for additional detections (e.g., {"vehicles": yolov8m})
):

    # Set defaults
    if traffic_class_list is None:
        traffic_class_list = ["car", "bus", "truck"]

    # Read forward orientation from locations.csv
    try:
        import pandas as _pd
        loc_df = _pd.read_csv(locations_csv)
        loc_row = loc_df[loc_df["id"].str.strip() == location.strip()]
        if len(loc_row) > 0:
            forward = loc_row.iloc[0]["is_forward"]
            if isinstance(forward, str):
                forward = forward.lower() in ("true", "1", "yes")
        else:
            forward = True
            if verbose:
                print(f"Warning: location '{location}' not found in {locations_csv}, assuming forward=True")
    except Exception as e:
        forward = True
        if verbose:
            print(f"Warning: could not read {locations_csv}: {e}, assuming forward=True")

    video = cv2.VideoCapture(video_filename)
    if not video.isOpened():
        print("Cannot open video:", video_filename)
        return

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = video.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delta  = 1.0 / fps
    begin_time   = datetime.strptime(begin_date, "%Y-%m-%d %H:%M:%S")
    dt           = begin_time

    print(f"Video: {video_filename}  |  footage_id: {footage_id}")
    print(f"FPS: {fps}, Total frames: {total_frames}")

    if reset_log:
        event_counter = 0
        with open(log_filename, "w", encoding='utf-8') as f:
            f.write("event_id,location,time_start,time_end,direction,class,footage_id\n")
    else:
        # continue numbering from max existing event_id
        try:
            import pandas as _pd
            _existing = _pd.read_csv(log_filename)
            event_counter = int(_existing["event_id"].max()) + 1 if "event_id" in _existing.columns and len(_existing) > 0 else 0
        except Exception:
            event_counter = 0

    max_age_frames = int(unseen_seconds_to_end * fps)
    tracker   = SimpleTracker(max_age_frames=max_age_frames, dist_threshold=dist_threshold, dt_seconds=frame_delta)
    annotator = Annotator()
    current_frame = 0

    # Traffic state tracking
    traffic_active = False
    traffic_start_time = None
    traffic_event_counter = 0

    img_dir = os.path.join("images", f"results_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(img_dir, exist_ok=True)

    # Store detections for each frame to annotate all models' outputs
    frame_detections = {}

    def resolve_overlapping_detections(bboxes_list, classes_list, confidences_list, iou_threshold=0.3):
        """
        Remove conflicting detections. If tram and bus/car overlap, keep tram.
        If same class overlaps, keep higher confidence.

        Args:
            bboxes_list: list of (x1, y1, x2, y2)
            classes_list: list of class names
            confidences_list: list of confidence scores
            iou_threshold: overlap threshold for considering detections conflicting

        Returns:
            (filtered_bboxes, filtered_classes, filtered_confidences)
        """
        if not bboxes_list:
            return [], [], []

        # Priority: lower number = higher priority
        priority = {"tram": 0, "bus": 1, "truck": 2, "car": 3}

        def iou(box1, box2):
            """Calculate IoU between two boxes."""
            x1a, y1a, x2a, y2a = box1
            x1b, y1b, x2b, y2b = box2

            inter_x1 = max(x1a, x1b)
            inter_y1 = max(y1a, y1b)
            inter_x2 = min(x2a, x2b)
            inter_y2 = min(y2a, y2b)

            if inter_x2 < inter_x1 or inter_y2 < inter_y1:
                return 0.0

            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            box1_area = (x2a - x1a) * (y2a - y1a)
            box2_area = (x2b - x1b) * (y2b - y1b)
            union_area = box1_area + box2_area - inter_area

            return inter_area / (union_area + 1e-6)

        # Keep track of which detections to remove
        keep = [True] * len(bboxes_list)

        for i in range(len(bboxes_list)):
            if not keep[i]:
                continue

            for j in range(i + 1, len(bboxes_list)):
                if not keep[j]:
                    continue

                overlap = iou(bboxes_list[i], bboxes_list[j])
                if overlap > iou_threshold:
                    # Decide which to keep based on priority and confidence
                    class_i = classes_list[i]
                    class_j = classes_list[j]
                    conf_i = confidences_list[i]
                    conf_j = confidences_list[j]

                    pri_i = priority.get(class_i, 999)
                    pri_j = priority.get(class_j, 999)

                    # Higher priority (lower number) wins; ties broken by confidence
                    if pri_i < pri_j or (pri_i == pri_j and conf_i >= conf_j):
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

        filtered_bboxes = [b for b, k in zip(bboxes_list, keep) if k]
        filtered_classes = [c for c, k in zip(classes_list, keep) if k]
        filtered_confs = [c for c, k in zip(confidences_list, keep) if k]

        return filtered_bboxes, filtered_classes, filtered_confs

    def finalize_and_log(track: Track):
        dx = track.last_centroid[0] - track.start_centroid[0]
        if abs(dx) < direction_threshold_px:
            direction = "unknown"
        else:
            direction = "right" if dx > 0 else "left"

        # Invert direction if camera is looking backward (forward=False)
        if not forward and direction != "unknown":
            direction = "left" if direction == "right" else "right"

        t_start  = track.time_start.isoformat(sep=' ', timespec='milliseconds')
        t_end    = track.time_last_seen.isoformat(sep=' ', timespec='milliseconds')
        event_id = getattr(track, 'event_id', -1)
        entry    = f"{event_id},{location},{t_start},{t_end},{direction},{track.class_name},{footage_id}\n"
        with open(log_filename, "a", encoding='utf-8') as f:
            f.write(entry)
        if verbose:
            print(entry, end='')

    def log_traffic_event(t_start, t_end, traffic_event_id):
        """Log a traffic event (high vehicle count)."""
        direction = "unknown"
        t_start_str = t_start.isoformat(sep=' ', timespec='milliseconds')
        t_end_str   = t_end.isoformat(sep=' ', timespec='milliseconds')
        entry = f"{traffic_event_id},{location},{t_start_str},{t_end_str},{direction},traffic,{footage_id}\n"
        with open(log_filename, "a", encoding='utf-8') as f:
            f.write(entry)
        if verbose:
            print(entry, end='')

    with tqdm(total=total_frames, desc="Processing", unit="frame") as pbar:
        while video.isOpened():
            current_frame += 1
            dt += timedelta(seconds=frame_delta)

            ret, frame = video.read()
            if not ret:
                break

            results    = model.predict(frame, conf=confidence_threshold, device=device, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)

            det_bboxes = []
            all_bboxes = []  # for annotation
            all_classes = []  # for annotation
            all_confidences = []  # for conflict resolution

            try:
                if hasattr(detections, "xyxy") and 'class_name' in detections.data:
                    for bbox, cls, conf in zip(detections.xyxy, detections.data['class_name'], detections.confidence):
                        cls = _remap_class(cls)
                        all_bboxes.append(bbox)
                        all_classes.append(cls)
                        all_confidences.append(conf)
                        # Debug: print detected classes on first frame
                        if current_frame == 1 and verbose:
                            print(f"[DEBUG] Primary model detected class: {cls} (conf={conf:.2f})")
            except Exception as e:
                if verbose:
                    print(f"[DEBUG] Exception parsing primary detections: {e}")

            # Merge detections from auxiliary models (e.g., general vehicle detector)
            if auxiliary_models:
                for model_name, aux_model in auxiliary_models.items():
                    try:
                        aux_results = aux_model.predict(frame, conf=confidence_threshold, device=device, verbose=False)[0]
                        aux_detections = sv.Detections.from_ultralytics(aux_results)
                        if hasattr(aux_detections, "xyxy") and 'class_name' in aux_detections.data:
                            for bbox, cls, conf in zip(aux_detections.xyxy, aux_detections.data['class_name'], aux_detections.confidence):
                                cls = _remap_class(cls)
                                # Only add if class is in track_classes (avoid duplicates with primary model)
                                if cls in track_classes:
                                    all_bboxes.append(bbox)
                                    all_classes.append(cls)
                                    all_confidences.append(conf)
                                    if current_frame == 1 and verbose:
                                        print(f"[DEBUG] {model_name} detected class: {cls} (conf={conf:.2f})")
                    except Exception as e:
                        if verbose:
                            print(f"[DEBUG] Exception from {model_name}: {e}")

            # Resolve conflicts (overlapping trams/buses/cars)
            all_bboxes, all_classes, all_confidences = resolve_overlapping_detections(
                all_bboxes, all_classes, all_confidences, iou_threshold=0.3
            )

            # Convert to tracking format
            det_bboxes = [(tuple(map(float, bbox)), cls) for bbox, cls in zip(all_bboxes, all_classes)]

            # Store for annotation
            if all_bboxes:
                frame_detections[current_frame] = (all_bboxes, all_classes)

            tracker.predict()
            tracker.update(det_bboxes, frame_index=current_frame, timestamp_dt=dt)

            # Count current traffic vehicles (car+bus+truck)
            traffic_count = sum(1 for t in tracker.tracks if t.class_name in traffic_class_list)
            is_high_traffic = traffic_count > traffic_threshold

            # Detect traffic state transitions
            if is_high_traffic and not traffic_active:
                traffic_active = True
                traffic_start_time = dt
                if verbose:
                    print(f"[TRAFFIC START] {traffic_count} vehicles at {dt}")
            elif not is_high_traffic and traffic_active:
                traffic_active = False
                log_traffic_event(traffic_start_time, dt, traffic_event_counter)
                traffic_event_counter += 1
                if verbose:
                    print(f"[TRAFFIC END] at {dt}")

            # Handle finished tracks
            to_remove = []
            for t in tracker.tracks:
                if (current_frame - t.last_seen_frame) > max_age_frames:
                    if t.class_name in track_classes and not t.logged:
                        finalize_and_log(t)
                        t.logged = True
                    to_remove.append(t)
            tracker.tracks = [t for t in tracker.tracks if t not in to_remove]

            for t in tracker.tracks:
                if t.class_name in track_classes and t.isNew:
                    t.isNew     = False
                    t.event_id  = event_counter
                    event_counter += 1
                    if verbose:
                        print(f"Detected: event_id={t.event_id}, class={t.class_name}, time={t.time_start}")
                    # Save annotated frame with all detections from all models
                    if current_frame in frame_detections:
                        all_bboxes, all_classes = frame_detections[current_frame]
                        # Draw boxes with labels manually
                        annotated_frame = frame.copy()
                        for bbox, cls in zip(all_bboxes, all_classes):
                            x1, y1, x2, y2 = map(int, bbox)
                            # Draw box
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            # Draw label
                            label = f"{cls}"
                            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        ts = "{:02d}-{:02d}".format(int(current_frame//fps//60), int(current_frame//fps%60))
                        cv2.imwrite(os.path.join(img_dir, f"event_{t.event_id:04d}_{ts}.jpg"), annotated_frame)

            pbar.update(1)

    # End traffic if still active
    if traffic_active:
        log_traffic_event(traffic_start_time, dt, traffic_event_counter)
        if verbose:
            print(f"[TRAFFIC END] at end of video")

    # Finalize any remaining tracks
    for t in tracker.tracks:
        if t.class_name in track_classes and not t.logged:
            finalize_and_log(t)
            t.logged = True

    video.release()
    print("Done.")
