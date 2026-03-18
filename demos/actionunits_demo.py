import cv2
import mediapipe as mp
import numpy as np
import math
import time

# ---------- Utility Functions ----------

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def draw_bar(panel, y, value, label, color):
    max_width = 200
    width = int(max_width * value)

    cv2.rectangle(panel, (20, y), (20 + max_width, y + 25), (50,50,50), -1)
    cv2.rectangle(panel, (20, y), (20 + width, y + 25), color, -1)

    cv2.putText(panel, f"{label}: {value:.2f}",
                (20, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255),
                1)

# ---------- MediaPipe Setup ----------

cap = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True
)

# ---------- Calibration Storage ----------

neutral_vec = None
happy_vec = None
calm_vec = None

capture_phase = "neutral"
capture_frames = []

countdown = False
countdown_start = 0
countdown_time = 3

# ---------- Mouse Click ----------

def mouse_callback(event, x, y, flags, param):
    global countdown, countdown_start
    if event == cv2.EVENT_LBUTTONDOWN:
        if not countdown:
            countdown = True
            countdown_start = time.time()

cv2.namedWindow("AU Emotion Similarity")
cv2.setMouseCallback("AU Emotion Similarity", mouse_callback)

# ---------- Main Loop ----------

while True:

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    panel = np.zeros((frame.shape[0], 250, 3), dtype=np.uint8)

    if results.multi_face_landmarks:

        lm = results.multi_face_landmarks[0]
        mp_drawing.draw_landmarks(frame, lm, mp_face_mesh.FACEMESH_TESSELATION)

        # ---------- Extract Action Units (Normalized & Expanded) ----------

        # Reference distances for normalization
        face_width = distance(lm.landmark[234], lm.landmark[454])  # approx face left-right
        face_height = distance(lm.landmark[10], lm.landmark[152])  # approx face top-bottom
        ref_scale = (face_width + face_height) / 2  # average scale for robustness

        # Smile / lip
        AU12 = distance(lm.landmark[61], lm.landmark[291]) / face_width  # Lip corner puller
        AU15 = distance(lm.landmark[61], lm.landmark[291]) / face_width  # Lip corner depressor

        # Cheeks / eyes
        AU6 = ((distance(lm.landmark[159], lm.landmark[33]) +
                distance(lm.landmark[386], lm.landmark[263])) / 2) / face_height  # Cheek raiser
        AU5 = distance(lm.landmark[159], lm.landmark[386]) / face_height  # Upper eyelid raiser

        # Brows
        AU1 = ((distance(lm.landmark[70], lm.landmark[63]) +
                distance(lm.landmark[105], lm.landmark[334])) / 2) / face_height  # Inner brow raiser
        AU2 = distance(lm.landmark[55], lm.landmark[285]) / face_height          # Outer brow raiser
        AU4 = ((distance(lm.landmark[70], lm.landmark[105]) +
                distance(lm.landmark[63], lm.landmark[334])) / 2) / face_height  # Brow lowerer

        # Nose wrinkle / upper lip
        AU9 = distance(lm.landmark[195], lm.landmark[5]) / face_height  # Nose wrinkler / upper lip raiser

        # Combine all into live vector
        live_vec = [AU12, AU6, AU1, AU4, AU15, AU2, AU5, AU9]

        # ---------- Countdown ----------

        if countdown:
            remaining = countdown_time - (time.time() - countdown_start)

            if remaining <= 0:
                capture_frames.append(live_vec)

                if len(capture_frames) >= 40:

                    avg_vec = np.mean(capture_frames, axis=0)

                    if capture_phase == "neutral":
                        neutral_vec = avg_vec
                        capture_phase = "happy"
                        print("Neutral captured")

                    elif capture_phase == "happy":
                        happy_vec = avg_vec
                        capture_phase = "calm"
                        print("Happy captured")

                    elif capture_phase == "calm":
                        calm_vec = avg_vec
                        capture_phase = "live"
                        print("Calibration complete")

                    capture_frames = []
                    countdown = False

            else:
                cv2.putText(frame,
                            str(int(remaining)+1),
                            (frame.shape[1]//2, frame.shape[0]//2),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            4,
                            (0,255,255),
                            6)

        # ---------- Live Detection ----------

        if capture_phase == "live" and neutral_vec is not None:

            sim_neutral = cosine_similarity(live_vec, neutral_vec)
            sim_happy = cosine_similarity(live_vec, happy_vec)
            sim_calm = cosine_similarity(live_vec, calm_vec)

            scores = {
                "Neutral": sim_neutral,
                "Happy": sim_happy,
                "Calm": sim_calm
            }

            emotion = max(scores, key=scores.get)

            draw_bar(panel, 60, sim_neutral, "Neutral", (120,120,120))
            draw_bar(panel, 120, sim_happy, "Happy", (0,255,0))
            draw_bar(panel, 180, sim_calm, "Calm", (255,150,0))

            cv2.putText(panel,
                        f"Emotion: {emotion}",
                        (20, 260),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0,255,255),
                        2)

    # ---------- Instructions ----------

    instruction = ""

    if capture_phase == "neutral":
        instruction = "Click to capture NEUTRAL face"

    elif capture_phase == "happy":
        instruction = "Click to capture HAPPY face"

    elif capture_phase == "calm":
        instruction = "Click to capture CALM face"

    elif capture_phase == "live":
        instruction = "Live similarity detection running"

    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (frame.shape[1],50), (0,0,0), -1)
    frame = cv2.addWeighted(overlay,0.6,frame,0.4,0)

    cv2.putText(frame,
                instruction,
                (40,35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255,255,255),
                2)

    combined = np.hstack((frame, panel))
    cv2.imshow("AU Emotion Similarity", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()