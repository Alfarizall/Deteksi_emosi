import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# ======================
# DUMMY TRAINING DATA
# ======================
# [ema, accuracy, correct_streak, wrong_streak, prev_difficulty]
X = []
y = []

for _ in range(3000):
    ema = np.random.uniform(2, 6)
    acc = np.random.uniform(0, 1)
    cs = np.random.randint(0, 5)
    ws = np.random.randint(0, 5)
    prev = np.random.uniform(1, 5)

    # LOGIC TARGET (teacher)
    diff = prev

    if acc > 0.8 and ema > 4:
        diff += 1
    elif acc < 0.4 or ema < 3:
        diff -= 1

    diff = np.clip(diff, 1, 5)

    X.append([ema, acc, cs, ws, prev])
    y.append(diff)

X = np.array(X)
y = np.array(y)

# ======================
# MODEL
# ======================
model = Sequential([
    Dense(32, activation="relu", input_shape=(5,)),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=Adam(0.001),
    loss="mse"
)

model.fit(X, y, epochs=30, batch_size=32)

model.save("difficulty_predictor.keras")

print("✅ Model saved: difficulty_predictor.keras")
