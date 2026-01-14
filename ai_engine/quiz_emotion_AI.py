import cv2
import numpy as np
import random
import time
import tkinter as tk
from tkinter import messagebox
from keras.models import load_model
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import Counter
import csv

# =====================
# LOAD MODEL
# =====================
emotion_model = load_model("emotion_model.h5", compile=False)
difficulty_model = load_model("difficulty_predictor.keras", compile=False)

emotion_labels = {
    0: "Marah",
    1: "Jijik",
    2: "Cemas",
    3: "Senang",
    4: "Sedih",
    5: "Terkejut",
    6: "Netral"
}

emotion_value = {
    "Marah": 0,
    "Jijik": 1,
    "Cemas": 2,
    "Sedih": 3,
    "Netral": 4,
    "Terkejut": 5,
    "Senang": 6
}

positive = ["Senang", "Netral"]
negative = ["Cemas", "Sedih"]

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# =====================
# GLOBAL STATE
# =====================
difficulty = 1.0
level = 1
score = 0
current_answer = 0
emotion_history = []
emotion_counter = Counter()
wrong_streak = 0
correct_streak = 0
total_questions = 0
correct_answers = 0
ema_emotion_value = 4.0 
dataset_buffer = []

cap = cv2.VideoCapture(0)

# =====================
# QUIZ ENGINE
# =====================
def generate_question():
    global level

    a = random.randint(1, 10 * level)
    b = random.randint(1, 10 * level)

    if level == 1:
        return f"{a} + {b}", a + b
    elif level == 2:
        return f"{a} - {b}", a - b
    elif level == 3:
        return f"{a} x {b}", a * b
    elif level == 4:
        b = random.randint(1, 10)
        return f"{a*b} ÷ {b}", a
    else:
        op = random.choice(["+", "-", "*"])
        if op == "+":
            return f"{a} + {b}", a + b
        if op == "-":
            return f"{a} - {b}", a - b
        return f"{a} x {b}", a * b

# =====================
# CAMERA LOOP
# =====================
def update_camera():
    global difficulty, level, ema_emotion_value

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_camera)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotion_text = "Mendeteksi..."

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (64, 64))
        face = face / 255.0
        face = np.reshape(face, (1, 64, 64, 1))

        pred = emotion_model.predict(face, verbose=0)
        emotion_text = emotion_labels[np.argmax(pred)]
        emotion_counter[emotion_text] += 1
        emotion_history.append(emotion_text)

        alpha = 0.1
        emotion_numeric = emotion_value[emotion_text]

        ema_emotion_value = (
            alpha * emotion_numeric +
            (1 - alpha) * ema_emotion_value
        )

        # Mapping EMA → difficulty target
        # Netral (4) sebagai baseline
        delta = (ema_emotion_value - 4.0) * 0.1

        difficulty = np.clip(difficulty + delta, 1.0, 5.0)
        level = int(round(difficulty))

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
    
    ema_label.config(text=f"EMA Emosi: {ema_emotion_value:.2f}")

    cv2.putText(frame, emotion_text, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = ImageTk.PhotoImage(Image.fromarray(rgb))
    cam_label.imgtk = img
    cam_label.configure(image=img)

    update_graph()
    root.after(30, update_camera)

# =====================
# GRAPH UPDATE
# =====================
def update_graph():
    values = [emotion_value[e] for e in emotion_history[-50:]]
    line.set_data(range(len(values)), values)
    ax.set_xlim(0, 50)
    canvas.draw()

# =====================
# QUIZ LOGIC
# =====================
def next_question():
    global current_answer
    q, current_answer = generate_question()
    question_label.config(text=f"Soal (Level {level}): {q}")
    score_label.config(text=f"Skor: {score}")

def submit_answer():
    global score, wrong_streak, correct_streak
    global difficulty, level

    prev_difficulty = level

    try:
        user_answer = int(answer_entry.get())
    except:
        messagebox.showwarning("Error", "Masukkan angka!")
        return

    if user_answer == current_answer:
        score += 10
        correct_streak += 1
        wrong_streak = 0

        # ⬆️ NAIK KESULITAN JIKA BENAR BERTURUT
        if correct_streak >= 3:
            difficulty = min(difficulty + 1.0, 5.0)
            level = int(round(difficulty))
            correct_streak = 0

    else:
        wrong_streak += 1
        correct_streak = 0

        # ⬇️ TURUN KESULITAN JIKA SALAH BERTURUT
        if wrong_streak >= 3:
            difficulty = max(difficulty - 1.0, 1.0)
            level = int(round(difficulty))
            wrong_streak = 0
    
    next_difficulty = level
    log_difficulty_sample(prev_difficulty, next_difficulty)

    score_label.config(text=f"Skor: {score}")
    answer_entry.delete(0, tk.END)
    next_question()

def finish_quiz():
    global running

    running = True

    cap.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass

    plt.close('all')

    # Hitung persentase emosi
    total = sum(emotion_counter.values())
    emotion_percent = {
        e: (c / total) * 100 if total > 0 else 0
        for e, c in emotion_counter.items()
    }

    # Simpan hasil ke file
    with open("hasil_kuis_emosi.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["================================"])
        writer.writerow(["Skor Akhir", score])
        writer.writerow([])
        writer.writerow(["Emosi", "Persentase"])
        for e, p in emotion_percent.items():
            writer.writerow([e, f"{p:.2f}%"])

    messagebox.showinfo(
        "Kuis Selesai",
        f"Kuis selesai!\n\nSkor akhir: {score}\n\n"
        "Hasil disimpan ke:\nhasil_kuis_emosi.csv"
    )

    with open("difficulty_dataset.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow([
                "ema_emotion",
                "accuracy",
                "correct_streak",
                "wrong_streak",
                "prev_difficulty",
                "next_difficulty"
            ])

        for row in dataset_buffer:
            writer.writerow(row)

    root.destroy()

def log_difficulty_sample(prev_diff, next_diff):
    total = correct_streak + wrong_streak
    accuracy = correct_streak / total if total > 0 else 0

    row = [
        round(ema_emotion_value, 3),
        round(accuracy, 3),
        correct_streak,
        wrong_streak,
        prev_diff,
        next_diff
    ]
    dataset_buffer.append(row)

# =====================
# UI SETUP
# =====================
root = tk.Tk()
root.title("Quiz Matematika Adaptif Berbasis Emosi")

# Kamera
cam_label = tk.Label(root)
cam_label.grid(row=0, column=0, padx=5, pady=5)

# Grafik
fig, ax = plt.subplots(figsize=(4,3))
line, = ax.plot([])
ax.set_ylim(-0.5, 6.5)
ax.set_yticks(list(emotion_value.values()))
ax.set_yticklabels(list(emotion_value.keys()))
ax.set_title("Grafik Emosi")

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)

ema_label = tk.Label(root, text="EMA Emosi: 4.00")
ema_label.grid(row=6, column=0, columnspan=2)

# Quiz UI
question_label = tk.Label(root, text="Klik Mulai", font=("Arial", 16))
question_label.grid(row=1, column=0, columnspan=2, pady=5)

answer_entry = tk.Entry(root, font=("Arial", 14))
answer_entry.grid(row=2, column=0, columnspan=2, pady=5)

submit_btn = tk.Button(root, text="Jawab", command=submit_answer)
submit_btn.grid(row=3, column=0, pady=5)

start_btn = tk.Button(root, text="Mulai", command=next_question)
start_btn.grid(row=3, column=1, pady=5)

finish_btn = tk.Button(
    root,
    text="Selesai Kuis",
    font=("Arial", 12, "bold"),
    bg="red",
    fg="white",
    command=finish_quiz
)
finish_btn.grid(row=5, column=0, columnspan=2, pady=10)

score_label = tk.Label(root, text="Skor: 0")
score_label.grid(row=4, column=0, columnspan=2, pady=5)

# =====================
# START
# =====================
update_camera()
root.mainloop()

cap.release()
