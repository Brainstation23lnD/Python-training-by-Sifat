import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('../models/cnn_model_v2.h5')

labels = {0: 'Mask', 1: 'No Mask'}

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (128, 128))
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_resized = face_resized.astype('float32') / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)

        pred = model.predict(face_resized)
        label = labels[int(round(pred[0][0]))]

        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow('Mask Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
