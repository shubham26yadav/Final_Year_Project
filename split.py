# Import dependencies
import cv2
import numpy as np
import os
import mediapipe as mp


# MediaPipe modules
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Detection function
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Draw landmarks with style
def draw_styled_landmarks(image, results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

# Extract keypoints from detected hands
def extract_keypoints(results):
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            return keypoints
    return np.zeros(21 * 3)

# Path for exported data
DATA_PATH = os.path.join('MP_Data') 

# Actions to detect
actions = np.array(['A', 'B', 'C','D', 'E', 'F','G', 'H', 'I','J', 'K', 'L','M', 'N', 'O','P', 'Q', 'R','S', 'T', 'U','V', 'W', 'X','Y', 'Z', '1', '2', '3', '4', '5'])

# Number of sequences per action
no_sequences = 30

# Length of each sequence
sequence_length = 30
