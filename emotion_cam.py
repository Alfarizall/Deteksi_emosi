import cv2
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

emotion_model = load_model("emotion_model.h5", compile=False)

emotion_labels = {
    0: "Marah",
    1: "Jijik",
    2: "Takut / Cemas",
    3: "Senang",
    4: "Sedih",
    5: "Terkejut",
    6: "Netral / Bosan"
}

emotion_to_value = {
    "Marah": 0,
    "Jijik": 1,
    "Takut / Cemas": 2,
    "Sedih": 3,
    "Netral / Bosan": 4,
    "Terkejut": 5,
    "Senang": 6
}

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

# =========================
# SETUP GRAFIK REAL-TIME
# =========================
plt.ion()
fig, ax = plt.subplots()
emotion_series = []
line, = ax.plot([])

ax.set_ylim(-0.5, 6.5)
ax.set_yticks(list(emotion_to_value.values()))
ax.set_yticklabels(list(emotion_to_value.keys()))
ax.set_xlabel("Frame")
ax.set_ylabel("Emosi")
ax.set_title("Perubahan Emosi (Real-Time Webcam)")

max_points = 100  # jumlah frame yang ditampilkan di grafik

# =========================
# LOOP WEBCAM
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face / 255.0
        face = np.reshape(face, (1, 64, 64, 1))

        prediction = emotion_model.predict(face, verbose=0)
        emotion_index = np.argmax(prediction)
        confidence = np.max(prediction)

        emotion_text = emotion_labels[emotion_index]
        emotion_value = emotion_to_value.get(emotion_text, 0)

        emotion_series.append(emotion_value)
        if len(emotion_series) > max_points:
            emotion_series.pop(0)

        label = f"{emotion_text} ({confidence*100:.1f}%)"

        cv2.rectangle(frame, (x, y), (x+w, y+h),
                      (0, 255, 0), 2)
        cv2.putText(frame, label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)

        # Update grafik
        line.set_ydata(emotion_series)
        line.set_xdata(range(len(emotion_series)))
        ax.set_xlim(0, max_points)
        fig.canvas.draw()
        fig.canvas.flush_events()

    cv2.imshow("Deteksi Ekspresi Wajah", frame)

    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q')]:
        cap.release()
        cv2.destroyAllWindows()
        plt.close('all')
        break

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()