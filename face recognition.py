import cv2
from simple_facerec import SimpleFacerec
# from cvzone.FaceDetectionModule import FaceDetector 
import pyttsx3
import requests
import time
from datetime import datetime

# --- Configuration ---
ESP32_CAM_IP = "192.168.1.100"  # <<< MUST REPLACE WITH our ESP32 IP
ESP32_CAM_STREAM_URL = f"http://{ESP32_CAM_IP}:81/stream" # URL for the video stream

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("D:/python-opencv/face _recognition/images/")

# Load Camera 
cap = cv2.VideoCapture(ESP32_CAM_STREAM_URL)
if not cap.isOpened():
    print(f"Error: Could not open video stream from {ESP32_CAM_STREAM_URL}. Make sure ESP32-CAM is online and serving stream.")
    exit()

# Flag to track if the door is currently unlocked
door_unlocked = False

# Variables for controlling logging frequency and door state
last_access_log_time = 0
last_failed_log_time = 0
LOG_COOLDOWN_SECONDS = 5
last_known_face_status = False # True if a known face was detected in the previous frame

# --- Logging functions ---
def log_failed_attempt():
    global last_failed_log_time
    current_time = time.time()
    if current_time - last_failed_log_time > LOG_COOLDOWN_SECONDS:
        with open("failed_attempts_log.txt", "a") as log_file:
            log_file.write(f"[{datetime.now()}] ⚠️ Unknown face detected - Access Denied\n")
        last_failed_log_time = current_time

def log_success(name):
    global last_access_log_time
    current_time = time.time()
    if current_time - last_access_log_time > LOG_COOLDOWN_SECONDS:
        with open("access_log.txt", "a") as log_file:
            log_file.write(f"[{datetime.now()}] ✅ Access granted to {name}\n")
        last_access_log_time = current_time

# --- Door Control Functions ---
def unlock_door():
    global door_unlocked
    if not door_unlocked:
        try:
            requests.get(f"http://{ESP32_CAM_IP}/unlock", timeout=1) # Increased timeout
            engine.say("Welcome")
            engine.runAndWait()
            print("Welcome - Door Unlocked")
            door_unlocked = True
            return True
        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to ESP32-CAM at {ESP32_CAM_IP}. Is it online?")
        except requests.exceptions.Timeout:
            print(f"Warning: Connection to ESP32-CAM at {ESP32_CAM_IP} timed out.")
        except Exception as e:
            print(f"An unexpected error occurred while communicating with ESP32-CAM: {e}")
    return False

def lock_door():
    global door_unlocked
    if door_unlocked: # Only try to lock if currently unlocked
        try:
            requests.get(f"http://{ESP32_CAM_IP}/lock", timeout=1) 
            # Only say "Access denied" if it's locking due to an unknown face causing a prior unlock attempt,
           
            engine.say("Access denied")
            engine.runAndWait()
            print("Access Denied - Door Locked")
            door_unlocked = False
            return True
        except requests.exceptions.ConnectionError:
            print(f"Error: Could not connect to ESP32-CAM at {ESP32_CAM_IP}. Is it online?")
        except requests.exceptions.Timeout:
            print(f"Warning: Connection to ESP32-CAM at {ESP32_CAM_IP} timed out.")
        except Exception as e:
            print(f"An unexpected error occurred while communicating with ESP32-CAM: {e}")
    return False

# --- Main loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame. Reconnecting...")
        cap = cv2.VideoCapture(ESP32_CAM_STREAM_URL) # Attempt to reconnect
        time.sleep(1) 
        continue

    # Resize frame for faster processing 
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5) 
    rgb_small_frame = small_frame[:, :, ::-1] # Convert BGR to RGB (SimpleFacerec expects RGB)

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame) 

    current_frame_known_face_detected = False
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        if name != "Unknown":
            current_frame_known_face_detected = True
            if unlock_door(): # Attempt to unlock, and log if successful
                log_success(name)
        # No else here for log_failed_attempt, it's handled when no known faces are detected

    # --- Door Lock/Unlock Logic and Voice Feedback ---
    # Logic to handle locking and "Access Denied" only when no known faces are in view
    if not current_frame_known_face_detected:
        if last_known_face_status:
        
            lock_door()
            log_failed_attempt() # Log a failed attempt when the door locks due to no known face
    
    last_known_face_status = current_frame_known_face_detected # Update status for next frame

    cv2.imshow("Face Recognition Door System", frame)

    key = cv2.waitKey(1)
    if key == 27: # press esc to exit
        break

cap.release()
cv2.destroyAllWindows()

