import cv2
import time
import numpy as np
import threading
import mediapipe as mp
import speech_recognition as sr
from scipy.spatial import distance
from pygame import mixer
from collections import deque

INTRO_AUDIO = "audio/drowsiness_assistant.mp3" 
TO_CONTINUE_AUDIO = "audio/not_understand.mp3"
ALERT_AUDIO = "audio/alert.wav"
CRITICAL_ALERT_AUDIO = "audio/critical_alert.wav"
YAWN_AUDIO = "audio/yawn_alert.mp3"          
START_AUDIO = "audio/normal_monitoring.mp3" 
ALRIGHT_AUDIO = "audio/alright.mp3" 
STOP_AUDIO = "audio/shutting_down.mp3"  

# threshold
EAR_THRESH = 0.15  
MAR_THRESH = 0.5   
PITCH_THRESH = 8  # head down

EAR_CONSEC_FRAMES = 5  
PERCLOS_THRESH = 0.7
ROLLING_WINDOW = 60
CALIBRATION_FRAMES = 30 

COLOR_CYAN_LIGHT = (255, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_YELLOW = (0, 255, 255)

# UI palette (BGR)
CLAUDE_ORANGE = (35, 107, 255)   
PANEL_DARK = (18, 18, 18)
TEXT_PRI = (235, 235, 235)
TEXT_MUT = (120, 120, 120)
COLOR_EYE = (255, 200, 120)      
COLOR_SAFE = (60, 200, 70)       
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
recognizer = sr.Recognizer()

system_active = False
calibration_mode = False
has_played_intro = False
is_listening_visual = False
audio_protection_end_time = 0 
current_playing_file = None 

flag = 0
yawn_counter = 0
head_down_counter = 0
ear_history = deque(maxlen=30)
perclos_queue = deque(maxlen=ROLLING_WINDOW) 
calibration_frames = []
base_pitch = 0

def get_landmarks(frame_w, frame_h, landmarks, indices):
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        coords.append((int(lm.x * frame_w), int(lm.y * frame_h)))
    return coords

# gesture
def detect_open_palm(hand_landmarks):    
    if not hand_landmarks: return False
    tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
            mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    pips = [mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, 
            mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP]
    return all(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y for tip, pip in zip(tips, pips))

# audio
def play_audio(file, loops=0, protection_seconds=0):
    global audio_protection_end_time, current_playing_file
    try:
        current_time = time.time()
        
        if mixer.music.get_busy() and current_playing_file == file:
            return 

        # priority -> critical alert overrides 
        if file == CRITICAL_ALERT_AUDIO:
             mixer.music.load(file)
             mixer.music.play(loops)
             current_playing_file = file
             return

        if current_time < audio_protection_end_time:
             return

        mixer.music.load(file)
        mixer.music.play(loops)
        current_playing_file = file
        
        if protection_seconds > 0:
            audio_protection_end_time = current_time + protection_seconds
            
    except Exception as e:
        print(f"Audio Error: {e}")

def stop_audio():
    global audio_protection_end_time, current_playing_file
    if time.time() > audio_protection_end_time:
        if mixer.music.get_busy():
            mixer.music.stop()
            current_playing_file = None

def draw_panel(frame, x, y, w, h, alpha=0.72, color=None):
    if color is None: color = PANEL_DARK
    sub = frame[y:y+h, x:x+w]
    bg = np.zeros_like(sub)
    bg[:] = color
    frame[y:y+h, x:x+w] = cv2.addWeighted(bg, alpha, sub, 1 - alpha, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (55, 55, 55), 1)

def draw_metric_bar(frame, x, y, label, value, max_val, color, width=168, threshold=None):
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_MUT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{value:.2f}", (x + width - 28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_PRI, 1, cv2.LINE_AA)
    bar_y, bar_h = y + 5, 7
    cv2.rectangle(frame, (x, bar_y), (x + width, bar_y + bar_h), (45, 45, 45), -1)
    fill_w = int(min(max(value, 0) / max_val, 1.0) * width)
    if fill_w > 0:
        cv2.rectangle(frame, (x, bar_y), (x + fill_w, bar_y + bar_h), color, -1)
    if threshold is not None:
        tx = x + int(min(threshold / max_val, 1.0) * width)
        cv2.line(frame, (tx, bar_y - 2), (tx, bar_y + bar_h + 2), (100, 100, 220), 2)
    cv2.rectangle(frame, (x, bar_y), (x + width, bar_y + bar_h), (65, 65, 65), 1)

def draw_status_pill(frame, x, y, text, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    px, py = 9, 5
    x1, y1, x2, y2 = x, y - th - py, x + tw + 2 * px, y + py
    sub = frame[y1:y2, x1:x2]
    bg = np.zeros_like(sub)
    bg[:] = color
    frame[y1:y2, x1:x2] = cv2.addWeighted(bg, 0.82, sub, 0.18, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    cv2.putText(frame, text, (x + px, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_PRI, 1, cv2.LINE_AA)

def draw_alert_border(frame, color, thickness=8):
    fh, fw = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (fw - 1, fh - 1), color, thickness)

def draw_glow_poly(frame, points, color, thickness=2):
    pts = np.array(points)
    glow = tuple(int(c * 0.4) for c in color)
    cv2.polylines(frame, [pts], True, glow, thickness + 5)
    cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)

def calculate_aspect_ratio(points):
    V = distance.euclidean(points[0], points[1])
    H = distance.euclidean(points[2], points[3])
    if H == 0: return 0
    return V / H

def get_head_pose(landmarks, img_w, img_h):
    face_3d = []
    face_2d = []
    pose_indices = [1, 152, 33, 263, 61, 291]
    for idx in pose_indices:
        lm = landmarks[idx]
        x, y = int(lm.x * img_w), int(lm.y * img_h)
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z])
    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)
    focal_length = 1 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)
    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    rmat, jac = cv2.Rodrigues(rot_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    return angles[0] * 360, angles[1] * 360, angles[2] * 360

def voice_startup_thread():
    global system_active, is_listening_visual, has_played_intro
    
    if not has_played_intro:
        play_audio(INTRO_AUDIO, loops=0, protection_seconds=4.0)
        time.sleep(4.5)
        has_played_intro = True

    with sr.Microphone() as source:
        try:
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            while not system_active: 
                try:
                    is_listening_visual = True
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
                    is_listening_visual = False
                    
                    command = recognizer.recognize_google(audio).lower()
                    print(f"User said: {command}")
                    
                    if "yes" in command or "start" in command:
                        system_active = True
                        play_audio(START_AUDIO, protection_seconds=3.0)
                        return 
                    elif "no" in command:
                        play_audio(STOP_AUDIO, protection_seconds=3.0)
                        return 
                    else:
                        play_audio(TO_CONTINUE_AUDIO, protection_seconds=3.0)
                        time.sleep(3.5)
                        
                except (sr.WaitTimeoutError, sr.UnknownValueError):
                    is_listening_visual = False
                    play_audio(TO_CONTINUE_AUDIO, protection_seconds=3.0)
                    time.sleep(3.5)
        except:
            is_listening_visual = False

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

t = threading.Thread(target=voice_startup_thread)
t.daemon = True
t.start()

print("SYSTEM READY.")

# landmark
LEFT_EYE = [386, 374, 263, 362]
RIGHT_EYE = [159, 145, 33, 133]
MOUTH = [13, 14, 61, 291] 

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # hand detection
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_lms in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            if detect_open_palm(hand_lms):
                if not system_active:
                    system_active = True
                    play_audio(START_AUDIO, loops=0, protection_seconds=3.0) 

    # face detection
    if system_active:
        is_listening_visual = False 
        face_results = face_mesh.process(rgb_frame)
        
        draw_panel(frame, 0, 0, w, 62)
        cv2.putText(frame, "DROWSINESS MONITOR", (12, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_MUT, 1, cv2.LINE_AA)
        draw_status_pill(frame, 12, 52, "ACTIVE", COLOR_SAFE)
        hint = "C: Calibrate   F: Stop   Q: Quit"
        (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(frame, hint, (w - hw - 12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT_MUT, 1, cv2.LINE_AA)
        
        if face_results.multi_face_landmarks:
            for face_lms in face_results.multi_face_landmarks:
                landmarks = face_lms.landmark
                left_eye = get_landmarks(w, h, landmarks, LEFT_EYE)
                right_eye = get_landmarks(w, h, landmarks, RIGHT_EYE)
                mouth = get_landmarks(w, h, landmarks, MOUTH)
                
                # 478-point mesh
                ear = (calculate_aspect_ratio(left_eye) + calculate_aspect_ratio(right_eye)) / 2.0
                mar = calculate_aspect_ratio(mouth) 
                pitch, yaw, roll = get_head_pose(landmarks, w, h)
                relative_pitch = pitch - base_pitch
                
                ear_history.append(ear)

                # calibration
                if calibration_mode:
                    calibration_frames.append((ear, pitch))
                    prog = len(calibration_frames) / CALIBRATION_FRAMES
                    draw_panel(frame, w//2 - 140, h//2 - 45, 280, 90)
                    cv2.putText(frame, "CALIBRATING", (w//2 - 58, h//2 - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, CLAUDE_ORANGE, 1, cv2.LINE_AA)
                    bx = w//2 - 110
                    cv2.rectangle(frame, (bx, h//2), (bx + 220, h//2 + 10), (45, 45, 45), -1)
                    cv2.rectangle(frame, (bx, h//2), (bx + int(220 * prog), h//2 + 10), CLAUDE_ORANGE, -1)
                    cv2.rectangle(frame, (bx, h//2), (bx + 220, h//2 + 10), (70, 70, 70), 1)
                    cv2.putText(frame, f"{len(calibration_frames)} / {CALIBRATION_FRAMES}", (w//2 - 25, h//2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_MUT, 1, cv2.LINE_AA)
                    # personalize
                    if len(calibration_frames) >= CALIBRATION_FRAMES:
                        EAR_THRESH = np.mean([x[0] for x in calibration_frames]) * 0.75  
                        base_pitch = np.mean([x[1] for x in calibration_frames])
                        calibration_mode = False
                        calibration_frames = []
                        play_audio(ALRIGHT_AUDIO, protection_seconds=3.0) 
                    continue 

                # drowsiness status
                is_eyes_closed = False
                is_yawning = False
                is_head_down = False
                
                if ear < EAR_THRESH:
                    flag += 1
                    if flag >= EAR_CONSEC_FRAMES:
                        is_eyes_closed = True
                else:
                    flag = 0
                
                if mar > MAR_THRESH:
                    yawn_counter += 1
                    if yawn_counter > 5:
                        is_yawning = True
                else:
                    yawn_counter = max(0, yawn_counter - 1)
                
                if relative_pitch < -PITCH_THRESH: 
                    head_down_counter += 1
                    if head_down_counter > 10:
                        is_head_down = True
                else:
                    head_down_counter = 0

                perclos_queue.append(1 if ear < EAR_THRESH else 0)
                perclos = sum(perclos_queue) / len(perclos_queue) if perclos_queue else 0

                if is_head_down and is_eyes_closed:
                     # critical
                     play_audio(CRITICAL_ALERT_AUDIO, loops=-1)
                elif is_eyes_closed:
                     # microsleep
                     play_audio(ALERT_AUDIO, loops=-1)
                elif is_yawning:
                     # yawning
                     play_audio(YAWN_AUDIO, loops=0)
                else:
                     stop_audio()

                # metrics panel
                px, py, pw, ph = 10, 72, 195, 155
                draw_panel(frame, px, py, pw, ph)
                mx, my = px + 12, py + 18
                cv2.putText(frame, "METRICS", (mx, my), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CLAUDE_ORANGE, 1, cv2.LINE_AA)
                draw_metric_bar(frame, mx, my + 20, "EAR", ear, 0.5, COLOR_SAFE if ear >= EAR_THRESH else COLOR_RED, threshold=EAR_THRESH)
                draw_metric_bar(frame, mx, my + 50, "MAR", mar, 1.2, CLAUDE_ORANGE if mar > MAR_THRESH else TEXT_MUT, threshold=MAR_THRESH)
                perclos_c = COLOR_RED if perclos > PERCLOS_THRESH else (COLOR_ORANGE if perclos > 0.4 else COLOR_SAFE)
                draw_metric_bar(frame, mx, my + 80, "PERCLOS", perclos, 1.0, perclos_c)
                pitch_c = COLOR_ORANGE if is_head_down else TEXT_MUT
                cv2.putText(frame, f"PITCH  {relative_pitch:+.1f}", (mx, my + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.42, pitch_c, 1, cv2.LINE_AA)

                # Landmark overlays with glow
                draw_glow_poly(frame, left_eye, COLOR_EYE)
                draw_glow_poly(frame, right_eye, COLOR_EYE)
                draw_glow_poly(frame, mouth, CLAUDE_ORANGE)

                # Alert border + bottom banner
                if is_head_down and is_eyes_closed:
                    draw_alert_border(frame, COLOR_RED, 9)
                    draw_panel(frame, 0, h - 52, w, 52, alpha=0.88, color=(25, 15, 15))
                    cv2.putText(frame, "CRITICAL  HEAD DOWN + EYES CLOSED", (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_RED, 2, cv2.LINE_AA)
                elif is_eyes_closed:
                    draw_alert_border(frame, COLOR_RED, 7)
                    draw_panel(frame, 0, h - 52, w, 52, alpha=0.88, color=(25, 15, 15))
                    txt = "ALERT  Eyes Closed" + ("  +  Yawning" if is_yawning else "")
                    cv2.putText(frame, txt, (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_RED, 2, cv2.LINE_AA)
                elif is_yawning:
                    draw_alert_border(frame, COLOR_ORANGE, 5)
                    draw_panel(frame, 0, h - 52, w, 52, alpha=0.85, color=(22, 20, 15))
                    cv2.putText(frame, "WARNING  Yawning Detected", (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_ORANGE, 2, cv2.LINE_AA)
                elif is_head_down:
                    draw_alert_border(frame, COLOR_ORANGE, 5)
                    draw_panel(frame, 0, h - 52, w, 52, alpha=0.85, color=(22, 20, 15))
                    cv2.putText(frame, "WARNING  Head Down", (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_ORANGE, 2, cv2.LINE_AA)
                else:
                    draw_panel(frame, 0, h - 52, w, 52, alpha=0.7)
                    cv2.putText(frame, "DRIVER ALERT", (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_SAFE, 1, cv2.LINE_AA)
                
    else:
        dark = np.zeros_like(frame)
        dark[:] = (8, 8, 8)
        frame[:] = cv2.addWeighted(dark, 0.45, frame, 0.55, 0)
        cw, ch = 360, 115
        cx, cy = w // 2 - cw // 2, h // 2 - ch // 2
        draw_panel(frame, cx, cy, cw, ch, alpha=0.90)
        cv2.putText(frame, "DROWSINESS MONITOR", (cx + 18, cy + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, CLAUDE_ORANGE, 1, cv2.LINE_AA)
        cv2.line(frame, (cx + 18, cy + 36), (cx + cw - 18, cy + 36), (55, 55, 55), 1)
        cv2.putText(frame, "Say 'Yes' or show palm to start", (cx + 18, cy + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_MUT, 1, cv2.LINE_AA)
        cv2.putText(frame, "Press Q to quit", (cx + 18, cy + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_MUT, 1, cv2.LINE_AA)
        if is_listening_visual:
            cv2.circle(frame, (cx + 24, cy + 104), 5, COLOR_YELLOW, -1)
            cv2.putText(frame, "LISTENING", (cx + 36, cy + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_YELLOW, 1, cv2.LINE_AA)

    cv2.imshow("Drowsiness Detection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): 
        play_audio(STOP_AUDIO, loops=0, protection_seconds=0)
        start_wait = time.time()
        time.sleep(0.2)
        while mixer.music.get_busy() and (time.time() - start_wait) < 5.0:
            time.sleep(0.1)
        break
    elif key == ord('f'): 
        system_active = False
        play_audio(STOP_AUDIO, protection_seconds=3.0)
    elif key == ord('c'): 
        calibration_mode = True
        system_active = True
        calibration_frames = []

cap.release()
cv2.destroyAllWindows()