# SMART-LOCK-UNLOCK
import cv2
import face_recognition

# to Load  known face
known_image = face_recognition.load_image_file("known_face.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# to Access webcam
video = cv2.VideoCapture(0)
print("🔍 Scanning for your face... Press 'q' to quit.")

while True:
    ret, frame = video.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = small_frame[:, :, ::-1] 

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for encoding in face_encodings:
        match = face_recognition.compare_faces([known_encoding], encoding)

        if match[0]:
            print("✅ Access Granted!")
        else:
            print("❌ Access Denied!")

    cv2.imshow("Smart Unlock - Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
