import cv2, base64, numpy as np, mysql.connector
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from tensorflow.keras.models import load_model
from functools import wraps

app = Flask(__name__)
app.secret_key = 'edusense_secret_key_2026'

# --- CONFIG AI (64x64) ---
MODEL_PATH = 'ai_engine/emotion_model.h5'
CASCADE_PATH = 'ai_engine/haarcascade_frontalface_default.xml'
model = load_model(MODEL_PATH, compile=False)
face_haar_cascade = cv2.CascadeClassifier(CASCADE_PATH)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def get_db_connection():
    return mysql.connector.connect(host="localhost", user="root", password="", database="edusense_db")

def login_required(role_target):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if 'user_id' not in session or session.get('role') != role_target:
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# --- ROUTES AUTH ---
@app.route('/')
def home(): return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        nrp, pw = request.form.get('nrp_nip'), request.form.get('password')
        conn = get_db_connection(); cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE nrp_nip=%s AND password=%s", (nrp, pw))
        user = cursor.fetchone(); conn.close()
        if user:
            session.update({'user_id': user['id'], 'nama': user['nama'], 'role': user['role']})
            return redirect(url_for('dashboard_guru' if user['role'] == 'guru' else 'belajar_siswa'))
        return render_template('login.html', error="NRP/NIP atau Password Salah!")
    return render_template('login.html')

@app.route('/logout')
def logout(): session.clear(); return redirect(url_for('login'))

# --- ROUTES SISWA ---
@app.route('/siswa')
@login_required('siswa')
def belajar_siswa():
    conn = get_db_connection(); cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, title FROM learning_resources")
    daftar = cursor.fetchall(); conn.close()
    return render_template('siswa/belajar.html', daftar_materi=daftar, nama=session['nama'])

@app.route('/get_materi/<int:id>')
@login_required('siswa')
def get_materi(id):
    conn = get_db_connection(); cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM learning_resources WHERE id = %s", (id,))
    res = cursor.fetchone(); conn.close()
    return jsonify(res)

@app.route('/kuesioner')
@login_required('siswa')
def kuesioner():
    return render_template('siswa/kuesioner.html')

# --- ROUTES GURU ---
@app.route('/guru')
@login_required('guru')
def dashboard_guru():
    conn = get_db_connection(); cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT users.nama, emotion_data.status_emosi, emotion_data.timestamp FROM emotion_data JOIN users ON users.id = emotion_data.user_id ORDER BY timestamp DESC LIMIT 10")
    logs = cursor.fetchall()
    cursor.execute("SELECT status_emosi, COUNT(*) as jml FROM emotion_data GROUP BY status_emosi")
    stats = cursor.fetchall(); conn.close()
    return render_template('guru/dashboard.html', logs=logs, stats=stats)

@app.route('/guru/tambah_materi', methods=['POST'])
@login_required('guru')
def tambah_materi():
    t, c = request.form.get('title'), request.form.get('content')
    conn = get_db_connection(); cursor = conn.cursor()
    cursor.execute("INSERT INTO learning_resources (title, content) VALUES (%s, %s)", (t, c))
    conn.commit(); conn.close()
    return redirect(url_for('dashboard_guru'))

@app.route('/analyze_emotion', methods=['POST'])
def analyze_emotion():
    if 'user_id' not in session: return jsonify({'error': 'Unauthorized'}), 401
    try:
        data = request.json['image'].split(",")[1]
        nparr = np.frombuffer(base64.b64decode(data), np.uint8)
        gray = cv2.cvtColor(cv2.imdecode(nparr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(gray, 1.1, 5)
        if len(faces) == 0: return jsonify({'emotion': 'Not Detected'})
        (x, y, w, h) = faces[0]
        roi = cv2.resize(gray[y:y+h, x:x+w], (64, 64))
        pixels = np.expand_dims(np.expand_dims(roi.astype('float32')/255.0, axis=0), axis=-1)
        emo = emotion_labels[np.argmax(model.predict(pixels, verbose=0)[0])]
        conn = get_db_connection(); cursor = conn.cursor()
        cursor.execute("INSERT INTO emotion_data (user_id, status_emosi) VALUES (%s, %s)", (session['user_id'], emo))
        conn.commit(); conn.close()
        return jsonify({'emotion': emo, 'anxiety': emo in ['Fear', 'Sad', 'Angry']})
    except: return jsonify({'error': 'AI Error'})

if __name__ == '__main__': app.run(debug=True)