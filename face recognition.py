import cv2
from simple_facerec import SimpleFacerec
from cvzone.FaceDetectionModule import FaceDetector 
import pyttsx3 # For text-to-speech
import requests # New: For making HTTP requests to the ESP32-CAM
import time # To add a small delay if needed

# Replace with the actual IP address of your ESP32-CAM module
ESP32_CAM_IP = "192.168.1.100" # <<< should be the ESP32-CAM's IP ADDRESS!( this is just for reference)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Initialize Face Detector 
detector = FaceDetector()

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("D:/python-opencv/face _recognition/images/")

# Load Camera 
cap = cv2.VideoCapture(1)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Flag to track if the door is currently unlocked
door_unlocked = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)

    known_face_detected_in_frame = False
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

        if name != "Unknown":
            known_face_detected_in_frame = True

    # --- Door Lock/Unlock Logic and Voice Feedback ---
    if known_face_detected_in_frame:
        if not door_unlocked: 
            try:
                # Send HTTP GET request to ESP32-CAM to unlock
                requests.get(f"http://{ESP32_CAM_IP}/unlock", timeout=0.5) # Small timeout for quick response
                engine.say("Welcome")
                engine.runAndWait()
                print("Welcome - Door Unlocked")
                door_unlocked = True
            except requests.exceptions.ConnectionError:
                print(f"Error: Could not connect to ESP32-CAM at {ESP32_CAM_IP}. Is it online?")
            except requests.exceptions.Timeout:
                print(f"Warning: Connection to ESP32-CAM at {ESP32_CAM_IP} timed out.")
            except Exception as e:
                print(f"An unexpected error occurred while communicating with ESP32-CAM: {e}")
    else:
        # Give a small buffer before locking if no one is detected.
        # This prevents immediate locking if someone briefly moves out of frame.
        # time.sleep(0.1) # small delay before locking

        if door_unlocked:
            try:
                # Send HTTP GET request to ESP32-CAM to lock
                requests.get(f"http://{ESP32_CAM_IP}/lock", timeout=0.5)
                engine.say("Access denied")
                engine.runAndWait()
                print("Access Denied - Door Locked")
                door_unlocked = False
            except requests.exceptions.ConnectionError:
                print(f"Error: Could not connect to ESP32-CAM at {ESP32_CAM_IP}. Is it online?")
            except requests.exceptions.Timeout:
                print(f"Warning: Connection to ESP32-CAM at {ESP32_CAM_IP} timed out.")
            except Exception as e:
                print(f"An unexpected error occurred while communicating with ESP32-CAM: {e}")

    cv2.imshow("Face Recognition Door System", frame)
     key = cv2.waitKey(1)
    if key == 27:  # Press 'Esc' to exit
        break
cap.release()
cv2.destroyAllWindows()
