import cv2
import numpy as np
import joblib
import datetime
import csv
import os
from keras_facenet import FaceNet
from centroidtracker import CentroidTracker

# =========================
# LOAD MODELS
# =========================
facenet = FaceNet()

model = joblib.load("models/gender_svm.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/label_encoder.pkl")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# =========================
# TRACKER
# =========================
tracker = CentroidTracker(maxDisappeared=40, maxDistance=50)

# =========================
# REAL-TIME CSV LOGGING
# =========================
csv_file = "customers_data.csv"
logged_ids = set()

def log_to_csv(object_id, gender):
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header once
        if not file_exists:
            writer.writerow(["ID", "Gender", "Timestamp"])

        writer.writerow([
            object_id,
            gender,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ])

# =========================
# VIDEO
# =========================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # 🔥 Improve lighting
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    print("Faces detected:", len(faces))

    rects = []
    face_boxes = []

    for (x, y, w, h) in faces:
        if w < 60 or h < 60:
            continue

        rects.append((x, y, x + w, y + h))
        face_boxes.append((x, y, w, h))

    objects = tracker.update(rects)

    # =========================
    # PROCESS TRACKED OBJECTS
    # =========================
    for (objectID, centroid) in objects.items():

        # Handle centroid format safely
        if len(centroid) == 4:
            cX = int((centroid[0] + centroid[2]) / 2)
            cY = int((centroid[1] + centroid[3]) / 2)
        else:
            cX, cY = centroid

        closest_face = None
        min_dist = float("inf")

        for (x, y, w, h) in face_boxes:
            fx = x + w // 2
            fy = y + h // 2

            dist = np.linalg.norm(np.array((cX, cY)) - np.array((fx, fy)))

            if dist < min_dist:
                min_dist = dist
                closest_face = (x, y, w, h)

        if closest_face is None:
            continue

        (x, y, w, h) = closest_face

        face = frame[y:y+h, x:x+w]

        try:
            # Preprocess
            face_resized = cv2.resize(face, (160, 160))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_rgb = np.expand_dims(face_rgb, axis=0)

            embedding = facenet.embeddings(face_rgb)
            embedding = scaler.transform(embedding)

            pred = model.predict(embedding)
            gender = encoder.inverse_transform(pred)[0]

            # 🔥 REAL-TIME LOGGING (only once per ID)
            if objectID not in logged_ids:
                logged_ids.add(objectID)
                log_to_csv(objectID, gender)

            # Draw
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)

            label = f"ID {objectID} | {gender}"
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)

        except Exception as e:
            print("Prediction Error:", e)

    cv2.imshow("Customer Analytics System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
