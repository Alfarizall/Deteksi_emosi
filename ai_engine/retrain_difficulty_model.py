import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

df = pd.read_csv("difficulty_dataset.csv")

X = df[[
    "ema_emotion",
    "accuracy",
    "correct_streak",
    "wrong_streak",
    "prev_difficulty"
]].values

y = df["next_difficulty"].values

model = Sequential([
    Dense(32, activation="relu", input_shape=(5,)),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(
    optimizer=Adam(0.001),
    loss="mse"
)

model.fit(X, y, epochs=50, batch_size=16)

model.save("difficulty_predictor.keras")

print("✅ Model retrained from real user data")
