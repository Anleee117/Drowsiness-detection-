from threading import Thread
import numpy as np
import playsound
import argparse
import time
import cv2
import os

# ============================
# MEDIAPIPE TASKS API (>= 0.10.21)
# ============================
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# ============================
# MEDIAPIPE LANDMARK INDICES (468-point Face Mesh)
# ============================

# Right eye (6 points for EAR)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Left eye (6 points for EAR)
LEFT_EYE = [362, 385, 387, 263, 373, 380]

# Eye contours for drawing
RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Inner lips (for MAR calculation)
INNER_LIP_MAR = {
    "left_corner": 78,
    "right_corner": 308,
    "upper_left": 82,
    "upper_center": 13,
    "upper_right": 312,
    "lower_left": 87,
    "lower_center": 14,
    "lower_right": 317,
}

# Lip contours for drawing
INNER_LIP_CONTOUR = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
OUTER_LIP_CONTOUR = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

# Head pose landmarks (6 key points for solvePnP)
HEAD_POSE_LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}


# ============================
# UTILITY FUNCTIONS
# ============================

def euclidean(p1, p2):
    """Compute euclidean distance between two points (replaces scipy)."""
    return np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float))


def sound_alarm(path):
    """Play an alarm sound repeatedly while ALARM_ON is True."""
    global ALARM_ON
    while ALARM_ON:
        try:
            playsound.playsound(path)
        except Exception as e:
            print("[ERROR] Playsound error:", e)
            break


def get_landmark_coords(landmarks, indices, frame_w, frame_h):
    """Extract (x, y) pixel coordinates from MediaPipe normalized landmarks."""
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        coords.append((int(lm.x * frame_w), int(lm.y * frame_h)))
    return np.array(coords)


def eye_aspect_ratio(eye_points):
    """
    Compute the Eye Aspect Ratio (EAR).
    EAR = (||P2-P6|| + ||P3-P5||) / (2 * ||P1-P4||)
    """
    A = euclidean(eye_points[1], eye_points[5])
    B = euclidean(eye_points[2], eye_points[4])
    C = euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(landmarks, frame_w, frame_h):
    """
    Compute the Mouth Aspect Ratio (MAR) using inner lip landmarks.
    MAR = (A + B + C) / (2 * D)
    """
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * frame_w, lm.y * frame_h])

    A = euclidean(pt(INNER_LIP_MAR["upper_left"]), pt(INNER_LIP_MAR["lower_left"]))
    B = euclidean(pt(INNER_LIP_MAR["upper_center"]), pt(INNER_LIP_MAR["lower_center"]))
    C = euclidean(pt(INNER_LIP_MAR["upper_right"]), pt(INNER_LIP_MAR["lower_right"]))
    D = euclidean(pt(INNER_LIP_MAR["left_corner"]), pt(INNER_LIP_MAR["right_corner"]))

    mar = (A + B + C) / (2.0 * D)
    return mar


def get_head_pose(landmarks, frame_w, frame_h):
    """
    Estimate head pose (pitch, yaw, roll) using cv2.solvePnP.
    Uses 6 key MediaPipe landmarks mapped to a generic 3D face model.
    """
    image_points = np.array([
        (landmarks[HEAD_POSE_LANDMARKS["nose_tip"]].x * frame_w,
         landmarks[HEAD_POSE_LANDMARKS["nose_tip"]].y * frame_h),
        (landmarks[HEAD_POSE_LANDMARKS["chin"]].x * frame_w,
         landmarks[HEAD_POSE_LANDMARKS["chin"]].y * frame_h),
        (landmarks[HEAD_POSE_LANDMARKS["left_eye_outer"]].x * frame_w,
         landmarks[HEAD_POSE_LANDMARKS["left_eye_outer"]].y * frame_h),
        (landmarks[HEAD_POSE_LANDMARKS["right_eye_outer"]].x * frame_w,
         landmarks[HEAD_POSE_LANDMARKS["right_eye_outer"]].y * frame_h),
        (landmarks[HEAD_POSE_LANDMARKS["left_mouth"]].x * frame_w,
         landmarks[HEAD_POSE_LANDMARKS["left_mouth"]].y * frame_h),
        (landmarks[HEAD_POSE_LANDMARKS["right_mouth"]].x * frame_w,
         landmarks[HEAD_POSE_LANDMARKS["right_mouth"]].y * frame_h),
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ])

    focal_length = frame_w
    center = (frame_w / 2, frame_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    pitch = euler_angles[0][0]
    yaw = euler_angles[1][0]
    roll = euler_angles[2][0]

    # Normalize pitch to be around 0 when looking straight (range -90 to 90)
    if pitch > 90:
        pitch = pitch - 180
    elif pitch < -90:
        pitch = pitch + 180
        
    # Invert pitch sign so that looking down is positive (0 to 90)
    pitch = -pitch

    nose_end_3D = np.array([(0.0, 0.0, 1000.0)])
    nose_end_2D, _ = cv2.projectPoints(
        nose_end_3D, rotation_vector, translation_vector,
        camera_matrix, dist_coeffs
    )

    nose_tip_2d = (int(image_points[0][0]), int(image_points[0][1]))
    return (pitch, yaw, roll), nose_end_2D, nose_tip_2d


def draw_contour(frame, landmarks, indices, frame_w, frame_h, color, thickness=1):
    """Draw a closed contour connecting the given landmark indices."""
    points = get_landmark_coords(landmarks, indices, frame_w, frame_h)
    cv2.polylines(frame, [points], isClosed=True, color=color, thickness=thickness)


# ============================
# ARGUMENT PARSING
# ============================

ap = argparse.ArgumentParser()
ap.add_argument("-a", "--alarm", type=str, default="",
                help="Path to alarm .WAV file")
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="Index of webcam on system")
args = vars(ap.parse_args())


# ============================
# THRESHOLDS & CONSTANTS
# ============================

# --- Eyes (EAR) ---
# Blink vs Microsleep discrimination:
#   - Normal blink: EAR < 0.14 for only 0.1-0.4s -> NO alarm
#   - Microsleep:   EAR < 0.14 for >= 0.67s      -> ALARM
#   - Start time resets to None IMMEDIATELY when EAR >= threshold (eye opens)
#   - This eliminates false alarms from normal blinking.
EYE_AR_SEVERE_THRESH = 0.14          # Below this = eyes closed (blink or microsleep)
EYE_AR_MILD_THRESH   = 0.16          # Below this = eyes drooping (kept for mild warning)
EYE_SEVERE_TIME_THRESH = 0.67        # seconds
EYE_MILD_TIME_THRESH   = 1.60        # seconds

# --- Mouth (MAR) ---
MOUTH_AR_MILD_THRESH = 0.55
MOUTH_AR_SEVERE_THRESH = 0.75
MOUTH_MILD_TIME_THRESH = 0.67        # seconds
MOUTH_SEVERE_TIME_THRESH = 0.33      # seconds

# --- Head Pitch (chin-to-chest angle) ---
# Research-based zones (positive = looking down, negative depends on solvePnP convention):
#   0° – 15° down : SAFE      – normal glance at dashboard
#   15° – 20° down: DISTRACTED – looking at phone/screen
#   > 20° down    : DANGER    – neck muscle loss, characteristic of microsleep
#
# Alarm rule: pitch must stay in DANGER zone for 2.5 seconds continuously
#             before triggering, to avoid false alarms.
#
# Note: cv2.decomposeProjectionMatrix pitch sign: negative = head down in this setup
HEAD_PITCH_SAFE_THRESH     = -15     # 0° to 15° down  -> safe zone
HEAD_PITCH_DISTRACT_THRESH = -20     # 15° to 20° down -> distracted zone
HEAD_PITCH_DANGER_THRESH   = -20     # alias for clarity: > 20° down = danger
HEAD_PITCH_MILD_THRESH     = -15     # kept for backward-compat (safe boundary)
HEAD_PITCH_SEVERE_THRESH   = -20     # severe = danger zone (> 20° head-down)
HEAD_PITCH_MILD_TIME_THRESH   = 1.0   # seconds
HEAD_PITCH_SEVERE_TIME_THRESH = 2.5   # seconds

# --- Head Roll ---
HEAD_ROLL_MILD_THRESH  = 20
HEAD_ROLL_SEVERE_THRESH = 35
HEAD_ROLL_MILD_TIME_THRESH   = 1.0   # seconds
HEAD_ROLL_SEVERE_TIME_THRESH = 2.5   # seconds

# --- No Face ---
NO_FACE_TIME_THRESH = 1.67           # seconds


# ============================
# STATE VARIABLES (Time-based tracking)
# ============================

# Start timestamps for each event (None when event is not occurring)
eye_severe_start_time = None
eye_mild_start_time   = None

mouth_severe_start_time = None
mouth_mild_start_time   = None

head_pitch_severe_start_time = None
head_pitch_mild_start_time   = None

head_roll_severe_start_time = None
head_roll_mild_start_time   = None

no_face_start_time = None
ALARM_ON = False

eye_mild_alert         = False
eye_severe_alert       = False   # True = microsleep detected
mouth_mild_alert       = False
mouth_severe_alert     = False
head_pitch_mild_alert  = False
head_pitch_severe_alert = False  # True = prolonged head-down
head_roll_mild_alert   = False
head_roll_severe_alert = False
no_face_alert          = False


# ============================
# DOWNLOAD MODEL IF NEEDED
# ============================

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading face_landmarker model (~4MB)...")
    import urllib.request
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[INFO] Model downloaded to:", MODEL_PATH)


# ============================
# INITIALIZE MEDIAPIPE FACE LANDMARKER
# ============================

print("[INFO] Initializing MediaPipe Face Landmarker (Tasks API)...")

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)


# ============================
# START VIDEO CAPTURE
# ============================

print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(args["webcam"])
time.sleep(1.0)

frame_timestamp_ms = 0


# ============================
# MAIN LOOP
# ============================

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame from webcam.")
        break

    frame_h, frame_w = frame.shape[:2]

    # Convert BGR to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    frame_timestamp_ms += 33  # ~30fps

    results = face_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    face_detected = len(results.face_landmarks) > 0

    # ============================
    # NO FACE DETECTED
    # ============================
    if not face_detected:
        if no_face_start_time is None:
            no_face_start_time = time.time()

        eye_mild_alert = eye_severe_alert = False
        mouth_mild_alert = mouth_severe_alert = False
        head_pitch_mild_alert = head_pitch_severe_alert = False
        head_roll_mild_alert = head_roll_severe_alert = False
        
        eye_severe_start_time = eye_mild_start_time = None
        mouth_severe_start_time = mouth_mild_start_time = None
        head_pitch_severe_start_time = head_pitch_mild_start_time = None
        head_roll_severe_start_time = head_roll_mild_start_time = None

        elapsed_no_face = time.time() - no_face_start_time

        if elapsed_no_face >= NO_FACE_TIME_THRESH:
            no_face_alert = True
            if not ALARM_ON:
                ALARM_ON = True
                if args["alarm"] != "":
                    t = Thread(target=sound_alarm, args=(args["alarm"],))
                    t.daemon = True
                    t.start()

            cv2.putText(frame, "!! NO FACE DETECTED !!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "DROWSY!", (frame_w - 130, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        else:
            no_face_alert = False
            ALARM_ON = False

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, frame_h - 50), (280, frame_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, "No face: {:.2f}s / {:.2f}s".format(elapsed_no_face, NO_FACE_TIME_THRESH),
                    (10, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        cv2.imshow("Drowsiness Detection (MediaPipe)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        continue

    # Face found -> reset
    no_face_start_time = None
    no_face_alert = False

    for face_lm in results.face_landmarks:
        landmarks = face_lm  # list of NormalizedLandmark

        # ------ EYE ASPECT RATIO ------
        left_eye_pts = get_landmark_coords(landmarks, LEFT_EYE, frame_w, frame_h)
        right_eye_pts = get_landmark_coords(landmarks, RIGHT_EYE, frame_w, frame_h)
        leftEAR = eye_aspect_ratio(left_eye_pts)
        rightEAR = eye_aspect_ratio(right_eye_pts)
        ear = (leftEAR + rightEAR) / 2.0

        draw_contour(frame, landmarks, LEFT_EYE_CONTOUR, frame_w, frame_h, (0, 255, 0), 1)
        draw_contour(frame, landmarks, RIGHT_EYE_CONTOUR, frame_w, frame_h, (0, 255, 0), 1)

        # ------ MOUTH ASPECT RATIO ------
        mar = mouth_aspect_ratio(landmarks, frame_w, frame_h)

        draw_contour(frame, landmarks, OUTER_LIP_CONTOUR, frame_w, frame_h, (180, 0, 180), 1)
        draw_contour(frame, landmarks, INNER_LIP_CONTOUR, frame_w, frame_h, (255, 0, 255), 1)

        # ------ HEAD POSE ------
        (pitch, yaw, roll), nose_end_2D, nose_tip = get_head_pose(landmarks, frame_w, frame_h)

        p2 = (int(nose_end_2D[0][0][0]), int(nose_end_2D[0][0][1]))
        cv2.line(frame, nose_tip, p2, (255, 0, 0), 2)

        for key_name, idx in HEAD_POSE_LANDMARKS.items():
            lm = landmarks[idx]
            px = int(lm.x * frame_w)
            py = int(lm.y * frame_h)
            cv2.circle(frame, (px, py), 3, (0, 0, 255), -1)

        # ============================
        # UPDATE START TIMES & ALERTS
        # ============================

        # --- Eyes (Blink vs Microsleep discrimination) ---
        # KEY LOGIC: Start time resets to None IMMEDIATELY when eye opens (EAR >= threshold).
        # This suppresses false alarms from normal blinks (<0.4s).
        # Only sustained closure >= EYE_SEVERE_TIME_THRESH (0.67s) triggers microsleep alert.
        if ear < EYE_AR_SEVERE_THRESH:
            # Eyes fully closed -> record start times
            if eye_severe_start_time is None:
                eye_severe_start_time = time.time()
            if eye_mild_start_time is None:
                eye_mild_start_time = time.time()
        elif ear < EYE_AR_MILD_THRESH:
            # Eyes drooping (half-open) -> mild only, reset severe immediately
            if eye_mild_start_time is None:
                eye_mild_start_time = time.time()
            eye_severe_start_time = None   # <-- IMMEDIATE reset: not a microsleep
        else:
            # Eyes open -> reset BOTH immediately
            eye_mild_start_time = None
            eye_severe_start_time = None   # <-- IMMEDIATE reset: suppresses blink false alarms

        # Calculate elapsed times
        elapsed_eye_severe = time.time() - eye_severe_start_time if eye_severe_start_time is not None else 0.0
        elapsed_eye_mild   = time.time() - eye_mild_start_time if eye_mild_start_time is not None else 0.0

        eye_severe_alert = (elapsed_eye_severe >= EYE_SEVERE_TIME_THRESH)
        eye_mild_alert   = (elapsed_eye_mild >= EYE_MILD_TIME_THRESH)

        # --- Mouth ---
        if mar > MOUTH_AR_SEVERE_THRESH:
            if mouth_severe_start_time is None:
                mouth_severe_start_time = time.time()
            if mouth_mild_start_time is None:
                mouth_mild_start_time = time.time()
        elif mar > MOUTH_AR_MILD_THRESH:
            if mouth_mild_start_time is None:
                mouth_mild_start_time = time.time()
            mouth_severe_start_time = None
        else:
            mouth_mild_start_time = None
            mouth_severe_start_time = None

        elapsed_mouth_severe = time.time() - mouth_severe_start_time if mouth_severe_start_time is not None else 0.0
        elapsed_mouth_mild   = time.time() - mouth_mild_start_time if mouth_mild_start_time is not None else 0.0

        mouth_severe_alert = (elapsed_mouth_severe >= MOUTH_SEVERE_TIME_THRESH)
        mouth_mild_alert   = (elapsed_mouth_mild >= MOUTH_MILD_TIME_THRESH)

        # --- Head Pitch (chin-to-chest / head nodding) ---
        # Zone classification (negative pitch = head bowing down in solvePnP convention):
        #   pitch > -15°            -> SAFE zone        (normal driving glance)
        #   -20° < pitch <= -15°    -> DISTRACTED zone  (phone / infotainment)
        #   pitch <= -20°           -> DANGER zone      (microsleep posture)
        if pitch < HEAD_PITCH_SEVERE_THRESH:      # <=> pitch <= -20° -> DANGER
            if head_pitch_severe_start_time is None:
                head_pitch_severe_start_time = time.time()
            if head_pitch_mild_start_time is None:
                head_pitch_mild_start_time = time.time()
        elif pitch < HEAD_PITCH_MILD_THRESH:      # <=> -20° < pitch <= -15° -> DISTRACTED
            if head_pitch_mild_start_time is None:
                head_pitch_mild_start_time = time.time()
            head_pitch_severe_start_time = None   # left DANGER zone -> reset danger start time
        else:                                     # > -15° -> SAFE -> reset all
            head_pitch_mild_start_time = None
            head_pitch_severe_start_time = None

        elapsed_pitch_severe = time.time() - head_pitch_severe_start_time if head_pitch_severe_start_time is not None else 0.0
        elapsed_pitch_mild   = time.time() - head_pitch_mild_start_time if head_pitch_mild_start_time is not None else 0.0

        head_pitch_severe_alert = (elapsed_pitch_severe >= HEAD_PITCH_SEVERE_TIME_THRESH)
        head_pitch_mild_alert   = (elapsed_pitch_mild >= HEAD_PITCH_MILD_TIME_THRESH)

        # --- Head roll ---
        abs_roll = abs(roll)
        if abs_roll > HEAD_ROLL_SEVERE_THRESH:
            if head_roll_severe_start_time is None:
                head_roll_severe_start_time = time.time()
            if head_roll_mild_start_time is None:
                head_roll_mild_start_time = time.time()
        elif abs_roll > HEAD_ROLL_MILD_THRESH:
            if head_roll_mild_start_time is None:
                head_roll_mild_start_time = time.time()
            head_roll_severe_start_time = None
        else:
            head_roll_mild_start_time = None
            head_roll_severe_start_time = None

        elapsed_roll_severe = time.time() - head_roll_severe_start_time if head_roll_severe_start_time is not None else 0.0
        elapsed_roll_mild   = time.time() - head_roll_mild_start_time if head_roll_mild_start_time is not None else 0.0

        head_roll_severe_alert = (elapsed_roll_severe >= HEAD_ROLL_SEVERE_TIME_THRESH)
        head_roll_mild_alert   = (elapsed_roll_mild >= HEAD_ROLL_MILD_TIME_THRESH)

        # ============================
        # ALARM DECISION LOGIC
        # ============================
        # EYES = primary indicator (gate)
        # 1. ANY severe -> AUTO alarm
        # 2. Eyes mild alone -> alarm
        # 3. Eyes mild + any other mild -> alarm
        # 4. Other mild WITHOUT eyes -> CAUTION only
        # 5. No face -> auto alarm (above)

        any_severe = (eye_severe_alert or mouth_severe_alert or
                      head_pitch_severe_alert or head_roll_severe_alert)

        other_mild_count = sum([
            mouth_mild_alert,
            head_pitch_mild_alert,
            head_roll_mild_alert
        ])

        mild_count = other_mild_count + (1 if eye_mild_alert else 0)

        combined_mild = eye_mild_alert and (other_mild_count >= 1)
        eyes_only_mild = eye_mild_alert and (other_mild_count == 0)

        should_alarm = any_severe or combined_mild or eyes_only_mild

        if should_alarm:
            if not ALARM_ON:
                ALARM_ON = True
                if args["alarm"] != "":
                    t = Thread(target=sound_alarm, args=(args["alarm"],))
                    t.daemon = True
                    t.start()
        else:
            ALARM_ON = False

        # ============================
        # DRAW ALERTS
        # ============================
        alert_y = 30

        if any_severe:
            cv2.putText(frame, "!! CRITICAL DROWSINESS !!", (10, alert_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            alert_y += 28

        if eye_severe_alert:
            cv2.putText(frame, "[MICROSLEEP] EYES SHUT >{:.2f}s".format(EYE_SEVERE_TIME_THRESH),
                        (10, alert_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            alert_y += 25
        elif eye_mild_alert:
            cv2.putText(frame, "[MILD] Eyes Drooping", (10, alert_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            alert_y += 25

        if mouth_severe_alert:
            cv2.putText(frame, "[SEVERE] WIDE YAWN", (10, alert_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            alert_y += 25
        elif mouth_mild_alert:
            cv2.putText(frame, "[MILD] Yawning", (10, alert_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            alert_y += 25

        if head_pitch_severe_alert:
            # DANGER zone (>20° head-down) held for 2-3s -> microsleep posture
            cv2.putText(frame, "[DANGER] HEAD SLUMPED (>20deg ~2-3s)", (10, alert_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            alert_y += 25
        elif head_pitch_mild_alert:
            # DISTRACTED zone (15-20° head-down) held for ~1s
            cv2.putText(frame, "[DISTRACTED] Head Nodding (15-20deg)", (10, alert_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            alert_y += 25

        if head_roll_severe_alert:
            cv2.putText(frame, "[SEVERE] HEAD TILTED", (10, alert_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            alert_y += 25
        elif head_roll_mild_alert:
            cv2.putText(frame, "[MILD] Head Tilting", (10, alert_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            alert_y += 25

        # ============================
        # TELEMETRY (bottom-left)
        # ============================
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, frame_h - 150), (320, frame_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        info_y = frame_h - 130

        def get_color(val, mild_t, severe_t, lower_is_bad=True):
            if lower_is_bad:
                if val < severe_t:
                    return (0, 0, 255)
                elif val < mild_t:
                    return (0, 165, 255)
                else:
                    return (0, 255, 0)
            else:
                if val > severe_t:
                    return (0, 0, 255)
                elif val > mild_t:
                    return (0, 165, 255)
                else:
                    return (0, 255, 0)

        ear_c = get_color(ear, EYE_AR_MILD_THRESH, EYE_AR_SEVERE_THRESH, True)
        mar_c = get_color(mar, MOUTH_AR_MILD_THRESH, MOUTH_AR_SEVERE_THRESH, False)
        pitch_c = get_color(pitch, HEAD_PITCH_MILD_THRESH, HEAD_PITCH_SEVERE_THRESH, True)
        roll_c = get_color(abs_roll, HEAD_ROLL_MILD_THRESH, HEAD_ROLL_SEVERE_THRESH, False)

        cv2.putText(frame, "EAR:   {:.2f}  (blink<{:.2f} / microsleep>={:.2f}s)".format(
            ear, EYE_AR_SEVERE_THRESH, EYE_SEVERE_TIME_THRESH),
            (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ear_c, 1)

        cv2.putText(frame, "EAR dur: {:.2f}s/{:.2f}s".format(
            elapsed_eye_severe, EYE_SEVERE_TIME_THRESH),
            (10, info_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ear_c, 1)

        cv2.putText(frame, "MAR: {:.2f} (>{:.2f})".format(
            mar, MOUTH_AR_MILD_THRESH),
            (180, info_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, mar_c, 1)

        # Pitch zone label
        if pitch > HEAD_PITCH_MILD_THRESH:
            pitch_zone = "SAFE"
        elif pitch > HEAD_PITCH_SEVERE_THRESH:
            pitch_zone = "DISTRACTED"
        else:
            pitch_zone = "DANGER"
        cv2.putText(frame, "Pitch: {:.1f} [{:s}] dur:{:.2f}s/{:.2f}s".format(
            pitch, pitch_zone, elapsed_pitch_severe, HEAD_PITCH_SEVERE_TIME_THRESH),
            (10, info_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, pitch_c, 1)

        cv2.putText(frame, "Roll:  {:.1f}  (mild:+/-{:.0f} sev:+/-{:.0f})".format(
            roll, HEAD_ROLL_MILD_THRESH, HEAD_ROLL_SEVERE_THRESH),
            (10, info_y + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.45, roll_c, 1)

        # Combined logic info
        if eye_mild_alert:
            combo_text = "Eyes + {} other sign(s)".format(other_mild_count)
            combo_color = (0, 0, 255) if combined_mild else (0, 165, 255)
        elif other_mild_count > 0:
            combo_text = "Other signs: {} (eyes OK, no alarm)".format(other_mild_count)
            combo_color = (0, 165, 255)
        else:
            combo_text = "All signs normal"
            combo_color = (0, 255, 0)

        cv2.putText(frame, combo_text,
                    (10, info_y + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45, combo_color, 1)

        cv2.putText(frame, "No-face: {:.2f}s / {:.2f}s".format(
            0.0 if no_face_start_time is None else (time.time() - no_face_start_time), NO_FACE_TIME_THRESH),
                    (10, info_y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # ============================
        # STATUS (top-right)
        # ============================
        if should_alarm:
            status_text = "DROWSY!"
            status_color = (0, 0, 255)
        elif mild_count >= 1:
            status_text = "CAUTION"
            status_color = (0, 165, 255)
        else:
            status_text = "AWAKE"
            status_color = (0, 255, 0)

        (tw, th), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(frame, status_text, (frame_w - tw - 15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 2)

        # Technology label
        cv2.putText(frame, "MediaPipe + OpenCV + EAR + MAR", (frame_w - 310, frame_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    # Show frame
    cv2.imshow("Drowsiness Detection (MediaPipe)", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

# Cleanup
face_landmarker.close()
cap.release()
cv2.destroyAllWindows()
