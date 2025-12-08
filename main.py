import cv2
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance
from pygame import mixer
from collections import deque
import time
import numpy as np
import speech_recognition as sr
import pyttsx3
import threading

mixer.init()
mixer.music.load("alert.wav")

# text to speech
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_engine.setProperty('volume', 0.9)

# set up voice recognition
recognizer = sr.Recognizer()
microphoen = sr.Microphone()

system_active = False # drowsiness detection is ON
user_is_sleepy = False # if user said they're sleepy
listening_mode = False 
calibration_done = False
calibrated_ear_threshold = None

# button for on/off
button_rect = None
button_hover = False

def play_audio(file):
    mixer.music.load(file)
    mixer.music.play()
    while mixer.music.get_busy():
        time.sleep(0.1)

def listeningcommand():
    global listening_mode
    listening_mode = True # listening

    with sr.Microphone as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            command = recognizer.recognize_google(audio).lower()
            print(f"You said: {command}")
            listening_mode = False
            return command
        except sr.WaitTimeoutError:
            listening_mode = False
            return ""
        except sr.UnknownValueError:
            listening_mode = False
            return ""
        except sr.RequestError:
            play_audio
            listening_mode = False
            return ""

def input_yesno(command):
    yes_words = ['yes', 'yeah', 'yep', 'sure', 'okay', 'ok']
    no_words = ['no', 'nope', 'nah']

    for word in yes_words:
        if word in command:
            return True

    for word in no_words:
        if word in command:
            return False

    return None # the input unclear

def voice_interaction():
    global system_active, user_is_sleepy

    speak("Hello! I'm your drowsiness detection assistant.")
    # time.sleep(7)

    # ask if user plans to drive
    speak("Are you planning to drive right now?")
    response = listeningcommand()

    if response:
        answer = input_yesno(response)

        # if user says yes
        if answer is True:
            system_active = True
            speak("Great! I'll start monitoring drowsiness.")
            time.sleep(0.5)

            speak("Are you feeling sleepy?")
            is_sleepy = listeningcommand()

            if is_sleepy:
                sleepy_answer = input_yesno(is_sleepy)

                if sleepy_answer is True:
                    user_is_sleepy = True
                    speak("Got it. I will be more cautious. It's always better to drive when you are not sleepy. Please stay safe.")

                elif sleepy_answer is False:
                    user_is_sleepy = False
                    speak("Good! I'll monitor you during your drive. Stay alert!")

                else:
                    speak("I will proceed with normal monitoring.")

            else:
                speak("I will proceed with normal monitoring.")

        # if user says no
        elif answer is False:
            system_active = False
            speak("Okay! The detection system will remain off. Say 'Hey assistant' when you're ready to drive.")

        # if answer is unclear
        else:
            speak("Sorry, I couldn't understand. Please say 'Hey assistant' to continue.")

    # if no voice detected at all
    else:
        speak("I didn't hear you. Please say 'Hey assistant' to restart.")

def listen_for_wake_word():
    global system_active, user_is_sleepy
    
    with sr.Microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5) # adjust background noise
        
        try:
            audio = recognizer.listen(source, timeout=2, phrase_time_limit=3)
            command = recognizer.recognize_google(audio).lower()
            
            if 'hey assistant' in command or 'hey system' in command:
                thread = threading.Thread(target=voice_interaction)
                thread.daemon = True
                thread.start()
                return True
            
            # allow user to turn stop monitoring with voice
            if system_active and ('stop monitoring' in command or 'turn off' in command):
                system_active = False
                speak("Monitoring stopped. Drive safely!")
                return True
                
        except:
            pass # ignore all recognition errors and keep listening
    
    return False

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    
    # horizontal eye distance
    C = distance.euclidean(eye[0], eye[3])
    
    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    # vertical mouth distances (top to bottom)
    A = distance.euclidean(mouth[13], mouth[19])
    B = distance.euclidean(mouth[14], mouth[18])
    C = distance.euclidean(mouth[15], mouth[17])
    
    # horizontal mouth width
    D = distance.euclidean(mouth[12], mouth[16])
    
    # MAR formula
    mar = (A + B + C) / (2.0 * D)
    return mar

def calculate_head_pose(shape, frame_shape):
    
    # 2D facial landmark points used for head pose estimation
    image_pts = np.float32([
        shape[30], # nose tip
        shape[8], # chin
        shape[36], # left eye corner
        shape[45], # right eye corner
        shape[48], # left mouth corner
        shape[54] # right mouth corner
    ])
    
    # 3D model points corresponding to the above landmarks
    model_pts = np.float32([
        [0.0, 0.0, 0.0],
        [0.0, -330.0, -65.0],
        [-225.0, 170.0, -135.0],
        [225.0, 170.0, -135.0],
        [-150.0, -150.0, -125.0],
        [150.0, -150.0, -125.0]
    ])
    
    # camera matrix (approximation using frame size)
    size = frame_shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    
    # assume no lens distortion
    dist_coeffs = np.zeros((4, 1))
    
    # compute rotation and translation vectors
    success, rotation_vec, translation_vec = cv2.solvePnP(
        model_pts, image_pts,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    # convert rotation vector to rotation matrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    
    # combine rotation + translation into a single projection matrix
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    
    # decompose projection matrix to obtain Euler angles
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    
    # extract pitch, yaw, roll values
    pitch, yaw, roll = euler_angles.flatten()[:3]
    return pitch, yaw, roll

def detect_glasses(eye_region, gray_frame):
    eye_roi = gray_frame[
        eye_region[1]:eye_region[3],
        eye_region[0]:eye_region[2]
    ]
    
    if eye_roi.size == 0:
        return False
    
    edges = cv2.Canny(eye_roi, 50, 150)
    edge_density = np.sum(edges) / edges.size
    return edge_density > 15

def assess_lighting(gray_frame):
    mean_intensity = np.mean(gray_frame)
    
    if mean_intensity < 60:
        return "Low", True
    elif mean_intensity > 200:
        return "High", True
    else:
        return "Good", False

def check_blink_pattern(ear_history, window=10):
    if len(ear_history) < window:
        return False
    
    recent = list(ear_history)[-window:]
    min_idx = recent.index(min(recent))
    if 2 <= min_idx <= window - 3:
        before_min = np.mean(recent[:min_idx])
        after_min = np.mean(recent[min_idx+1:])
        if before_min > 0.25 and after_min > 0.25:
            return True
    return False

# configuration parameters (adjusted based on sleepiness)
BASE_EAR_THRESH = 0.25
BASE_MAR_THRESH = 0.6
BASE_EAR_CONSEC_FRAMES = 20
BASE_PERCLOS_THRESH = 0.7
ROLLING_WINDOW = 60

# head pose thresholds
PITCH_THRESH = 20
YAW_THRESH = 30
ROLL_THRESH = 15

# state variables
flag = 0
yawn_counter = 0
head_down_counter = 0
drowsiness_score = 0

perclos_queue = deque(maxlen=ROLLING_WINDOW)
ear_history = deque(maxlen=30)
mar_history = deque(maxlen=30)

# facial landmarks indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS['mouth']

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)
glasses_detected = False
ear_threshold = BASE_EAR_THRESH

print("=" * 60)
print("VOICE-ACTIVATED DROWSINESS DETECTION SYSTEM")
print("=" * 60)
print("Say 'Hey assistant' to activate the system")
print("Press 'q' to quit | 'c' to calibrate for narrow eyes")
print("=" * 60)

# start with voice interaction
initial_thread = threading.Thread(target=voice_interaction)
initial_thread.daemon = True
initial_thread.start()

last_wake_word_check = time.time()
wake_word_check_interval = 2  # check every 2 seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # adjust thresholds based on sleepiness level
    if user_is_sleepy:
        ear_threshold = BASE_EAR_THRESH + 0.03  # More sensitive
        mar_threshold = BASE_MAR_THRESH - 0.1
        ear_consec_frames = BASE_EAR_CONSEC_FRAMES - 5
        perclos_thresh = BASE_PERCLOS_THRESH - 0.1
    else:
        ear_threshold = BASE_EAR_THRESH
        mar_threshold = BASE_MAR_THRESH
        ear_consec_frames = BASE_EAR_CONSEC_FRAMES
        perclos_thresh = BASE_PERCLOS_THRESH
    
    # system status display
    if system_active:
        status_text = "MONITORING ACTIVE"
        status_color = (0, 255, 0)
        if user_is_sleepy:
            status_text += " (HIGH ALERT)"
            status_color = (0, 165, 255)
    else:
        status_text = "SYSTEM OFF - Say 'Hey assistant' to activate"
        status_color = (100, 100, 100)
    
    cv2.putText(frame, status_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # listen for wake word periodically
    if not system_active and time.time() - last_wake_word_check > wake_word_check_interval and not listening_mode:
        wake_thread = threading.Thread(target=listen_for_wake_word)
        wake_thread.daemon = True
        wake_thread.start()
        last_wake_word_check = time.time()
    
    # only process drowsiness detection if system is active
    if system_active:
        lighting_status, poor_lighting = assess_lighting(gray)
        subjects = detect(gray, 0)
        
        eyes_closed = 0
        drowsiness_detected = False
        
        for subject in subjects:
            shape = predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            
            # eye analysis
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEar = eye_aspect_ratio(leftEye)
            rightEar = eye_aspect_ratio(rightEye)
            ear = (leftEar + rightEar) / 2.0
            ear_history.append(ear)
            
            # mouth analysis
            mouth = shape[mStart:mEnd]
            mar = mouth_aspect_ratio(mouth)
            mar_history.append(mar)
            
            # dead pose analysis
            try:
                pitch, yaw, roll = calculate_head_pose(shape, frame.shape)
            except:
                pitch, yaw, roll = 0, 0, 0
            
            # glasses detection
            if len(ear_history) % 30 == 0:
                left_eye_region = [
                    leftEye[:, 0].min(), leftEye[:, 1].min(),
                    leftEye[:, 0].max(), leftEye[:, 1].max()
                ]
                glasses_detected = detect_glasses(left_eye_region, gray)
            
            is_blink = check_blink_pattern(ear_history)
            
            # eye closure detection
            if ear < ear_threshold and not is_blink:
                flag += 1
                eyes_closed = 1
                
                if flag >= ear_consec_frames:
                    drowsiness_score += 2
                    cv2.putText(frame, "DROWSY: Eyes Closed", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                flag = 0
            
            # yawn detection
            if mar > mar_threshold:
                yawn_counter += 1
                if yawn_counter > 15:
                    drowsiness_score += 1
                    cv2.putText(frame, "DROWSY: Yawning", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                yawn_counter = max(0, yawn_counter - 1)
            
            # head pose detection
            if pitch > PITCH_THRESH:
                head_down_counter += 1
                if head_down_counter > 20:
                    drowsiness_score += 1
                    cv2.putText(frame, "DROWSY: Head Down", (10, 120),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                head_down_counter = max(0, head_down_counter -  1)
            
            # PERCLOS Calculation
            perclos_queue.append(1 if eyes_closed > 0 else 0)
            perclos_value = sum(perclos_queue) / len(perclos_queue) if perclos_queue else 0
            
            # combined drowsiness detection
            if perclos_value > perclos_thresh or drowsiness_score > 5:
                drowsiness_detected = True
                cv2.putText(frame, "***** WAKE UP !!! *****", 
                           (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                if not mixer.music.get_busy():
                    mixer.music.play()
            
            drowsiness_score = max(0, drowsiness_score - 0.1)
            
            # visual feedback
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            
            # display metrics
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, frame.shape[0] - 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (10, frame.shape[0] - 95),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"PERCLOS: {perclos_value:.2f}", 
                       (10, frame.shape[0] - 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, frame.shape[0] - 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # status indicators
            status_y = 150
            if glasses_detected:
                cv2.putText(frame, "Glasses: YES", 
                           (frame.shape[1] - 150, status_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                status_y += 25
            
            cv2.putText(frame, f"Light: {lighting_status}", 
                       (frame.shape[1] - 150, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 255) if not poor_lighting else (0, 165, 255), 2)
    
    if listening_mode:
        cv2.putText(frame, "LISTENING...", (frame.shape[1]//2 - 100, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    
    cv2.imshow('Voice-Activated Drowsiness Detection', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("c"):
        ear_threshold = 0.20
        speak("EAR threshold adjusted for narrow eye features")
        print(f"EAR threshold adjusted to {ear_threshold}")

cv2.destroyAllWindows()
cap.release()
speak("System shutting down. Drive safely!")
print("System Stopped")