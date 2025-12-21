import cv2
import time
import numpy as np
import threading
import mediapipe as mp
import speech_recognition as sr
from scipy.spatial import distance
from pygame import mixer
from collections import deque

# --- CONFIGURATION ---
INTRO_AUDIO = "audio/drowsiness_assistant.mp3" 
TO_CONTINUE_AUDIO = "audio/not_understand.mp3"
ALERT_AUDIO = "audio/alert.wav"
CRITICAL_ALERT_AUDIO = "audio/critical_alert.wav"
YAWN_AUDIO = "audio/yawn_alert.mp3"          
START_AUDIO = "audio/normal_monitoring.mp3" 
ALRIGHT_AUDIO = "audio/alright.mp3" 
STOP_AUDIO = "audio/shutting_down.mp3"  

# --- DEFAULT THRESHOLDS ---
EAR_THRESH = 0.15  
MAR_THRESH = 0.5   
PITCH_THRESH = 8  # Sensitive Head Down

# --- TUNING ---
EAR_CONSEC_FRAMES = 5  
PERCLOS_THRESH = 0.7
ROLLING_WINDOW = 60
CALIBRATION_FRAMES = 30 

# --- COLORS ---
COLOR_CYAN_LIGHT = (255, 255, 0) 
COLOR_WHITE = (255, 255, 255)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_YELLOW = (0, 255, 255)

# --- SETUP MEDIAPIPE ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# --- SETUP AUDIO & VOICE ---
mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
recognizer = sr.Recognizer()

# --- STATE VARIABLES ---
system_active = False
calibration_mode = False
has_played_intro = False
is_listening_visual = False
audio_protection_end_time = 0 
current_playing_file = None 

# Counters
flag = 0
yawn_counter = 0
head_down_counter = 0
ear_history = deque(maxlen=30)
perclos_queue = deque(maxlen=ROLLING_WINDOW) 
calibration_frames = []
base_pitch = 0

# --- HELPER FUNCTIONS ---

def get_landmarks(frame_w, frame_h, landmarks, indices):
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        coords.append((int(lm.x * frame_w), int(lm.y * frame_h)))
    return coords

# --- GESTURE ACTIVATION ---
def detect_open_palm(hand_landmarks):    
    if not hand_landmarks: return False
    tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
            mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    pips = [mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, 
            mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.PINKY_PIP]
    return all(hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y for tip, pip in zip(tips, pips))

# --- AUDIO FUNCTION ---
def play_audio(file, loops=0, protection_seconds=0):
    global audio_protection_end_time, current_playing_file
    try:
        current_time = time.time()
        
        # 1. DUPLICATE CHECK: 
        # If the requested file is ALREADY playing, do nothing.
        if mixer.music.get_busy() and current_playing_file == file:
            return 

        # 2. PRIORITY CHECK (Critical Alert overrides everything)
        if file == CRITICAL_ALERT_AUDIO:
             mixer.music.load(file)
             mixer.music.play(loops)
             current_playing_file = file
             return

        # 3. PROTECTION CHECK (Don't interrupt Intro/Voice Prompts)
        if current_time < audio_protection_end_time:
             return

        # 4. STANDARD PLAY
        mixer.music.load(file)
        mixer.music.play(loops)
        current_playing_file = file
        
        if protection_seconds > 0:
            audio_protection_end_time = current_time + protection_seconds
            
    except Exception as e:
        print(f"Audio Error: {e}")

def stop_audio():
    global audio_protection_end_time, current_playing_file
    # Only stop if protection time has passed
    if time.time() > audio_protection_end_time:
        if mixer.music.get_busy():
            mixer.music.stop()
            current_playing_file = None

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

# --- VOICE THREAD LOGIC ---
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
                    
                    # --- VOICE ACTIVATION ---
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

# --- MAIN EXECUTION ---

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

t = threading.Thread(target=voice_startup_thread)
t.daemon = True
t.start()

print("SYSTEM READY.")

# Landmark Indices
LEFT_EYE = [386, 374, 263, 362]
RIGHT_EYE = [159, 145, 33, 133]
MOUTH = [13, 14, 61, 291] 

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 1. HAND DETECTION
    hand_results = hands.process(rgb_frame)
    if hand_results.multi_hand_landmarks:
        for hand_lms in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            if detect_open_palm(hand_lms):
                if not system_active:
                    system_active = True
                    play_audio(START_AUDIO, loops=0, protection_seconds=3.0) 

    # 2. FACE DETECTION
    if system_active:
        is_listening_visual = False 
        face_results = face_mesh.process(rgb_frame)
        
        cv2.putText(frame, "MONITORING ACTIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_GREEN, 2)
        cv2.putText(frame, "Press 'c' to Calibrate | Press 'q' to Quit", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        if face_results.multi_face_landmarks:
            for face_lms in face_results.multi_face_landmarks:
                landmarks = face_lms.landmark
                left_eye = get_landmarks(w, h, landmarks, LEFT_EYE)
                right_eye = get_landmarks(w, h, landmarks, RIGHT_EYE)
                mouth = get_landmarks(w, h, landmarks, MOUTH)
                
                # Landmark Mapping from 478-point mesh
                ear = (calculate_aspect_ratio(left_eye) + calculate_aspect_ratio(right_eye)) / 2.0
                mar = calculate_aspect_ratio(mouth) 
                pitch, yaw, roll = get_head_pose(landmarks, w, h)
                relative_pitch = pitch - base_pitch
                
                ear_history.append(ear)

                # --- CALIBRATION ---
                if calibration_mode:
                    calibration_frames.append((ear, pitch))
                    cv2.putText(frame, f"CALIBRATING... {len(calibration_frames)}/{CALIBRATION_FRAMES}", (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_CYAN_LIGHT, 2)
                    # Personalizing thresholds based on 30-frame mean
                    if len(calibration_frames) >= CALIBRATION_FRAMES:
                        EAR_THRESH = np.mean([x[0] for x in calibration_frames]) * 0.75  
                        base_pitch = np.mean([x[1] for x in calibration_frames])
                        calibration_mode = False
                        calibration_frames = []
                        play_audio(ALRIGHT_AUDIO, protection_seconds=3.0) 
                    continue 

                # --- DROWSINESS STATUS ---
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

                # --- AUDIO LOGIC ---
                if is_head_down and is_eyes_closed:
                     # Tier 3: Critical Multimodal Failure
                     play_audio(CRITICAL_ALERT_AUDIO, loops=-1)
                elif is_eyes_closed:
                     # Tier 2: Active Microsleep
                     play_audio(ALERT_AUDIO, loops=-1)
                elif is_yawning:
                     # Tier 1: Yawn Detection
                     play_audio(YAWN_AUDIO, loops=0)
                else:
                     stop_audio()

                # --- UI: STATS (Top Left) ---
                start_y = 85
                line_spacing = 25
                cv2.putText(frame, f"EAR: {ear:.2f} (Thresh: {EAR_THRESH:.2f})", (10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, start_y + line_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
                cv2.putText(frame, f"Pitch: {relative_pitch:.1f} (Base: {base_pitch:.1f})", (10, start_y + 2*line_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)
                cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (10, start_y + 3*line_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1)

                # --- UI: ALERTS (Bottom Left) ---
                bottom_y = h - 20
                if is_head_down and is_eyes_closed:
                     cv2.putText(frame, "!!! CRITICAL DROWSINESS: HEAD DOWN + EYES CLOSED !!!", (10, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
                elif is_eyes_closed:
                    text = "Warning: Eyes Closed" + (" + Yawn alert!" if is_yawning else "")
                    cv2.putText(frame, text, (10, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
                elif is_yawning:
                    cv2.putText(frame, "Warning: Yawn Alert!", (10, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ORANGE, 2)
                elif is_head_down:
                    cv2.putText(frame, "Warning: Head Down", (10, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ORANGE, 2)

                cv2.polylines(frame, [np.array(left_eye)], True, COLOR_CYAN_LIGHT, 1)
                cv2.polylines(frame, [np.array(right_eye)], True, COLOR_CYAN_LIGHT, 1)
                cv2.polylines(frame, [np.array(mouth)], True, COLOR_CYAN_LIGHT, 1)
                
    else:
        cv2.putText(frame, "SYSTEM OFF", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        cv2.putText(frame, "Say 'Yes' or Show Palm to Start | Press 'q' to Exit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        if is_listening_visual:
            cv2.putText(frame, "LISTENING...", (w//2 - 80, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, COLOR_YELLOW, 2)

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