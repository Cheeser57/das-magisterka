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
    class_name             = "tram",
):
    """
    Run YOLO tram detection on a video and write timestamps to a CSV.

    CSV columns: event_id, location, time_start, time_end, direction, class, footage_id
    event_id is a globally rising integer across all logger() runs on the same log file.
    """
    if footage_id is None:
        footage_id = os.path.splitext(os.path.basename(video_filename))[0]

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
        with open(log_filename, "w") as f:
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

    img_dir = os.path.join("images", f"results_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(img_dir, exist_ok=True)

    def finalize_and_log(track: Track):
        dx = track.last_centroid[0] - track.start_centroid[0]
        if abs(dx) < direction_threshold_px:
            direction = "unknown"
        else:
            direction = "right" if dx > 0 else "left"
        t_start  = track.time_start.isoformat(sep=' ', timespec='milliseconds')
        t_end    = track.time_last_seen.isoformat(sep=' ', timespec='milliseconds')
        event_id = getattr(track, 'event_id', -1)
        entry    = f"{event_id},{location},{t_start},{t_end},{direction},{class_name},{footage_id}\n"
        with open(log_filename, "a") as f:
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
            try:
                if hasattr(detections, "xyxy") and 'class_name' in detections.data:
                    for bbox, cls in zip(detections.xyxy, detections.data['class_name']):
                        det_bboxes.append((tuple(map(float, bbox)), cls))
            except Exception:
                pass

            tracker.predict()
            tracker.update(det_bboxes, frame_index=current_frame, timestamp_dt=dt)

            to_remove = []
            for t in tracker.tracks:
                if (current_frame - t.last_seen_frame) > max_age_frames:
                    if t.class_name == class_name and not t.logged:
                        finalize_and_log(t)
                        t.logged = True
                    to_remove.append(t)
            tracker.tracks = [t for t in tracker.tracks if t not in to_remove]

            for t in tracker.tracks:
                if t.class_name == class_name and t.isNew:
                    t.isNew     = False
                    t.event_id  = event_counter
                    event_counter += 1
                    if verbose:
                        print(f"Detected: event_id={t.event_id}, time={t.time_start}")
                    annotated = annotator.annotate(frame, detections)
                    ts = "{:02d}-{:02d}".format(int(current_frame//fps//60), int(current_frame//fps%60))
                    cv2.imwrite(os.path.join(img_dir, f"event_{t.event_id:04d}_{ts}.jpg"), annotated)

            pbar.update(1)

    for t in tracker.tracks:
        if t.class_name == class_name and not t.logged:
            finalize_and_log(t)
            t.logged = True

    video.release()
    print("Done.")
