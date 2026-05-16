# Drowsiness Detection System
A real-time driver drowsiness detection system using computer vision and voice control. It monitors eye closure, yawning, and head position to alert the driver when signs of fatigue are detected.


## Features
- **Eye Aspect Ratio (EAR)**: detects prolonged eye closure / microsleep
- **Mouth Aspect Ratio (MAR)**: detects yawning
- **Head pose estimation**: detects head drooping
- **PERCLOS**: tracks percentage of eye closure over a rolling 60-frame window
- **Voice activation**: say "yes" or "start" to begin monitoring
- **Palm gesture**: show an open palm to the camera to start
- **Calibration mode**: personalizes thresholds to your face
- **Multi-level alerts**: warning, alert, and critical audio cues

## Requirements
- Python 3.11
- Webcam
- Microphone

### Dependencies
```
opencv-python
mediapipe==0.10.14
pygame
SpeechRecognition
pyaudio
scipy
numpy
```

## Setup

**1. Create a virtual environment with Python 3.11**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

**2. Install dependencies**
```bash
pip install mediapipe==0.10.14 opencv-python pygame SpeechRecognition pyaudio scipy numpy
```
> If `pyaudio` fails on Mac, run `brew install portaudio` first.

---

## How to Run
```bash
source venv/bin/activate
python main.py
```

---

## Controls
Say **"yes"** / **"start"** to start monitoring 
Show **open palm** to start monitoring 
`C` : Calibrate thresholds to your face 
`F` : Stop monitoring (standby) 
`Q` : Quit 


## How It Works
1. On launch, the system plays an intro and listens for a voice command or palm gesture to activate
2. Once active, the camera continuously tracks:
   - **EAR** (eye openness ratio): triggers alert if below threshold for 5+ consecutive frames
   - **MAR** (mouth openness ratio): triggers yawn alert if above threshold
   - **Head pitch**: triggers warning if head drops below baseline angle
   - **PERCLOS**: tracks how often eyes are closed over the last 60 frames
3. Alerts escalate: yawn → eyes closed → critical (eyes closed + head down)
4. Press `C` to run a 30-frame calibration that adjusts EAR threshold and baseline pitch to your face

