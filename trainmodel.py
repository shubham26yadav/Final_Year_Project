from split import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import os
import json

# Constants
label_map = {label: num for num, label in enumerate(actions)}
sequences, labels = [], []

# Load preprocessed data
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(
                os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)),
                allow_pickle=True  # ✅ Fix for "Object arrays" error
            )
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

# Prepare data
X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# Define model
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, X.shape[2])))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

# Compile model
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# TensorBoard log directory
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Train model
model.fit(x_train, y_train, epochs=30, callbacks=[tb_callback], validation_data=(x_test, y_test))

# Save model
model.save('model.h5')

# Save model architecture as JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
