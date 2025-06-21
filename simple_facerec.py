import face_recognition
import cv2
import os
import numpy as np
from PIL import Image

img = Image.open("me_rgb.jpg").convert("RGB")
img.save("me_rgb.jpg")


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def load_encoding_images(self, images_path):
      
      
        for filename in os.listdir(images_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(images_path, filename)
                print(f"[INFO] Loading image: {filename}")

                # FIX: Load image using OpenCV and convert to RGB manually
                bgr_image = cv2.imread(image_path)
                if bgr_image is None:
                    print(f"[WARNING] Could not read image {filename}. Skipping.")
                    continue

                image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

                encodings = face_recognition.face_encodings(image)
                if encodings:
                    self.known_face_encodings.append(encodings[0])
                    name = os.path.splitext("Jimin")[0]
                    self.known_face_names.append(name)
                    print(f"[INFO] Loaded encoding for {name}")
                else:
                    print(f"[WARNING] No face found in {filename}. Skipping.")

    def detect_known_faces(self, frame):
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)

        # Scale face locations back up
        face_locations = [(top * 4, right * 4, bottom * 4, left * 4)
                          for (top, right, bottom, left) in face_locations]

        return face_locations, face_names
