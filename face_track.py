# face_track.py
import cv2
import face_recognition
import mediapipe as mp
import pandas as pd
import os
import numpy as np
from datetime import datetime
from scipy.spatial import distance as dist

# Setup
known_faces_dir = "known_faces"
os.makedirs(known_faces_dir, exist_ok=True)
os.makedirs("logs", exist_ok=True)

known_encodings = []
known_names = []

# Load known faces
for filename in os.listdir(known_faces_dir):
    path = os.path.join(known_faces_dir, filename)
    image = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(os.path.splitext(filename)[0])

# EAR calculation function
def eye_aspect_ratio(landmarks, eye_indices):
    p1 = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
    p2 = np.array([landmarks[eye_indices[1]].x, landmarks[eye_indices[1]].y])
    p3 = np.array([landmarks[eye_indices[2]].x, landmarks[eye_indices[2]].y])
    p4 = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
    p5 = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
    p6 = np.array([landmarks[eye_indices[5]].x, landmarks[eye_indices[5]].y])

    # vertical
    dist1 = dist.euclidean(p2, p6)
    dist2 = dist.euclidean(p3, p5)
    # horizontal
    dist3 = dist.euclidean(p1, p4)

    ear = (dist1 + dist2) / (2.0 * dist3)
    return ear

# Eye landmark indices (for EAR)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Log dictionary to accumulate session data
focus_data = {name: {"focused_frames": 0, "total_frames": 0} for name in known_names}

# Webcam
cap = cv2.VideoCapture(0)

FOCUS_THRESHOLD = 0.21

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    results = face_mesh.process(rgb)

    for encoding, location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, encoding)
        name = "Unknown"
        if True in matches:
            index = matches.index(True)
            name = known_names[index]
            focus_data[name]["total_frames"] += 1

            # Focus check
            focused = False
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    left_ear = eye_aspect_ratio(face_landmarks.landmark, LEFT_EYE)
                    right_ear = eye_aspect_ratio(face_landmarks.landmark, RIGHT_EYE)
                    avg_ear = (left_ear + right_ear) / 2.0
                    if avg_ear > FOCUS_THRESHOLD:
                        focused = True

            if focused:
                focus_data[name]["focused_frames"] += 1

            # Draw box
            top, right, bottom, left = location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Focus Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Write summary
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
rows = []
for name, data in focus_data.items():
    total = data["total_frames"]
    focused = data["focused_frames"]
    percent = round((focused / total) * 100, 2) if total else 0
    rows.append([name, timestamp, total, focused, percent])

summary_df = pd.DataFrame(rows, columns=["Name", "Time", "TotalFrames", "FocusedFrames", "FocusPercentage"])
summary_df.to_csv("attendance_summary.csv", index=False)