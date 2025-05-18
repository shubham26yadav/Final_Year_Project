import tkinter as tk 
from tkinter import ttk, messagebox, filedialog, simpledialog
import cv2
import numpy as np
import mediapipe as mp
import threading
import pyttsx3
from tensorflow.keras.models import model_from_json
from split import *  # your preprocessing functions, actions list, extract_keypoints, mediapipe_detection

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pywhatkit
from datetime import datetime, timedelta
from PIL import Image, ImageTk

# Load model
with open("model.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)
model.load_weights("model.h5")

# Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Voice engine setup
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)  # Female voice

# Detection variables
sequence = []
sentence = []
accuracy = []
predictions = []
threshold = 0.8

dictionary = ["HELLO", "HOW", "ARE", "YOU", "THANK", "PLEASE", "YES", "NO", "GOOD", "MORNING", "NIGHT"]

# Tkinter window
root = tk.Tk()
root.title("Sign Language Translator")
root.attributes('-fullscreen', True)  # Fullscreen

def exit_fullscreen(event):
    root.attributes('-fullscreen', False)
root.bind("<Escape>", exit_fullscreen)

# Configure root grid
root.grid_rowconfigure(0, weight=1)
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)

# Frames
frame_camera = tk.Frame(root, bg="black")
frame_camera.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

frame_controls = tk.Frame(root, bg="#f0f0f0")
frame_controls.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

frame_camera.grid_propagate(False)
frame_controls.grid_propagate(False)

# Video label
label_video = tk.Label(frame_camera)
label_video.pack(expand=True, fill="both")

label_output_title = tk.Label(frame_controls, text="Detected Text:", font=("Segoe UI", 12), bg="#f0f0f0")
label_output_title.pack(anchor="w")

text_output = tk.Text(frame_controls, font=("Segoe UI", 12), height=6)
text_output.pack(expand=True, fill="both", pady=5)

label_suggestion = tk.Label(frame_controls, text="Suggestions:", bg="#f0f0f0")
label_suggestion.pack(anchor="w")

listbox_suggestions = tk.Listbox(frame_controls, height=4)
listbox_suggestions.pack(expand=True, fill="x")

def update_suggestions():
    current_text = text_output.get("1.0", "end-1c").strip().split()[-1] if text_output.get("1.0", "end-1c").strip() else ""
    listbox_suggestions.delete(0, tk.END)
    if current_text:
        matches = [word for word in dictionary if word.startswith(current_text.upper())]
        for w in matches:
            listbox_suggestions.insert(tk.END, w)

def on_suggestion_click(event):
    if listbox_suggestions.curselection():
        selected = listbox_suggestions.get(listbox_suggestions.curselection())
        text = text_output.get("1.0", "end-1c").strip().split()
        if text:
            text[-1] = selected
        else:
            text = [selected]
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, " ".join(text) + " ")
        listbox_suggestions.delete(0, tk.END)

listbox_suggestions.bind("<<ListboxSelect>>", on_suggestion_click)

def save_to_file():
    content = text_output.get("1.0", "end-1c").strip()
    if not content:
        messagebox.showwarning("Save", "No text to save!")
        return
    filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files","*.txt")])
    if filepath:
        with open(filepath, "w") as f:
            f.write(content)
        messagebox.showinfo("Save", f"Saved to {filepath}")

def send_email():
    sender_email = "shubham265yadav@gmail.com"
    sender_password = "your_password"
    receiver_email = "email@gmail.com"
    content = text_output.get("1.0", "end-1c").strip()
    if not content:
        messagebox.showwarning("Email", "No text to send!")
        return

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = "Sign Language Translation Output"
    msg.attach(MIMEText(content, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        messagebox.showinfo("Email", f"Email sent to {receiver_email}!")
    except Exception as e:
        messagebox.showerror("Email Error", f"Failed to send email:\n{str(e)}")

def send_whatsapp():
    content = text_output.get("1.0", "end-1c").strip()
    if not content:
        messagebox.showwarning("WhatsApp", "No text to send!")
        return
    phone = simpledialog.askstring("Phone Number", "Enter phone number with country code (e.g. +911234567890):")
    if not phone:
        return
    now = datetime.now() + timedelta(minutes=1)
    try:
        pywhatkit.sendwhatmsg(phone, content, now.hour, now.minute)
        messagebox.showinfo("WhatsApp", f"Message scheduled to {phone} via WhatsApp Web.")
    except Exception as e:
        messagebox.showerror("WhatsApp Error", str(e))

btn_save = ttk.Button(frame_controls, text="Save to File", command=save_to_file)
btn_save.pack(fill="x", pady=3)
btn_email = ttk.Button(frame_controls, text="Send Email", command=send_email)
btn_email.pack(fill="x", pady=3)
btn_whatsapp = ttk.Button(frame_controls, text="Send WhatsApp", command=send_whatsapp)
btn_whatsapp.pack(fill="x", pady=3)
btn_clear = ttk.Button(frame_controls, text="Clear Text", command=lambda: text_output.delete("1.0", tk.END))
btn_clear.pack(fill="x", pady=3)

lock = threading.Lock()
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def update_frame():
    global sequence, sentence, predictions

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    cropframe = frame[40:400, 0:300].copy()
    image, results = mediapipe_detection(cropframe, hands)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-30:]

    sign_detected = False
    try:
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predicted_index = np.argmax(res)
            predictions.append(predicted_index)

            if np.unique(predictions[-10:])[0] == predicted_index:
                if res[predicted_index] > threshold:
                    sign_detected = True
                    if len(sentence) > 0:
                        if actions[predicted_index] != sentence[-1]:
                            sentence.append(actions[predicted_index])
                            accuracy.append(str(res[predicted_index] * 100))
                            engine.say(actions[predicted_index])
                            engine.runAndWait()
                    else:
                        sentence.append(actions[predicted_index])
                        accuracy.append(str(res[predicted_index] * 100))
    except Exception:
        pass

    disp_frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    disp_frame = cv2.resize(disp_frame, (640, 480))
    border_color = (0, 255, 0) if sign_detected else (0, 0, 255)
    disp_frame = cv2.copyMakeBorder(disp_frame, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=border_color)

    img = cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGBA)
    imgtk = ImageTk.PhotoImage(image=Image.fromarray(img))

    label_video.imgtk = imgtk
    label_video.configure(image=imgtk)

    with lock:
        text_output.delete("1.0", tk.END)
        text_output.insert(tk.END, ' '.join(sentence) + " ")
    update_suggestions()

    root.after(30, update_frame)

root.after(0, update_frame)
root.mainloop()

cap.release()
hands.close()
cv2.destroyAllWindows()
