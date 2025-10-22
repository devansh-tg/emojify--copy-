# gui.py - Professional Emotion Detection GUI
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

# Load the improved model
print("Loading emotion detection model...")
model = load_model('model.h5')
print("Model loaded successfully!")

# Emotion labels and colors
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emoji_map = {
    0: 'emojis/angry.png',
    1: 'emojis/disgust.png',
    2: 'emojis/fear.png',
    3: 'emojis/happy.png',
    4: 'emojis/neutral.png',
    5: 'emojis/sad.png',
    6: 'emojis/surprise.png'
}

# Colors for each emotion (BGR format for OpenCV)
emotion_colors = {
    'Angry': (0, 0, 255),      # Red
    'Disgust': (0, 128, 0),    # Green
    'Fear': (128, 0, 128),     # Purple
    'Happy': (0, 255, 255),    # Yellow
    'Neutral': (192, 192, 192),# Gray
    'Sad': (255, 0, 0),        # Blue
    'Surprise': (255, 165, 0)  # Orange
}

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize main window
root = tk.Tk()
root.title("🎭 Advanced Emotion Detection System")
root.configure(bg='#1e1e1e')
root.geometry("1400x900")
root.resizable(False, False)

# Title frame
title_frame = tk.Frame(root, bg='#2d2d2d', height=80)
title_frame.pack(fill='x', pady=(0, 10))

title_label = tk.Label(
    title_frame, 
    text="🎭 Real-Time Emotion Detection",
    font=('Arial', 28, 'bold'),
    fg='#00d4ff',
    bg='#2d2d2d'
)
title_label.pack(pady=15)

subtitle_label = tk.Label(
    title_frame,
    text="AI-Powered Facial Expression Analysis | 63% Accuracy CNN Model",
    font=('Arial', 12),
    fg='#888888',
    bg='#2d2d2d'
)
subtitle_label.pack()

# Main content frame
content_frame = tk.Frame(root, bg='#1e1e1e')
content_frame.pack(fill='both', expand=True, padx=20, pady=10)

# Left panel - Camera feed
left_panel = tk.Frame(content_frame, bg='#2d2d2d', relief='solid', borderwidth=2)
left_panel.pack(side='left', padx=(0, 10), fill='both', expand=True)

camera_title = tk.Label(
    left_panel,
    text="📹 Live Camera Feed",
    font=('Arial', 16, 'bold'),
    fg='#ffffff',
    bg='#2d2d2d'
)
camera_title.pack(pady=10)

video_label = tk.Label(left_panel, bg='#000000')
video_label.pack(padx=10, pady=10)

# Right panel - Emotion info
right_panel = tk.Frame(content_frame, bg='#2d2d2d', relief='solid', borderwidth=2, width=400)
right_panel.pack(side='right', fill='both', padx=(10, 0))
right_panel.pack_propagate(False)

# Emotion display
emotion_title = tk.Label(
    right_panel,
    text="😊 Detected Emotion",
    font=('Arial', 16, 'bold'),
    fg='#ffffff',
    bg='#2d2d2d'
)
emotion_title.pack(pady=15)

# Emoji display
emoji_label = tk.Label(right_panel, bg='#2d2d2d')
emoji_label.pack(pady=10)

# Emotion name
emotion_name_label = tk.Label(
    right_panel,
    text="No face detected",
    font=('Arial', 32, 'bold'),
    fg='#00d4ff',
    bg='#2d2d2d'
)
emotion_name_label.pack(pady=10)

# Confidence score
confidence_label = tk.Label(
    right_panel,
    text="Confidence: ---%",
    font=('Arial', 18),
    fg='#888888',
    bg='#2d2d2d'
)
confidence_label.pack(pady=5)

# Separator
separator = tk.Frame(right_panel, bg='#444444', height=2)
separator.pack(fill='x', padx=20, pady=15)

# Statistics frame
stats_title = tk.Label(
    right_panel,
    text="📊 Emotion Probabilities",
    font=('Arial', 14, 'bold'),
    fg='#ffffff',
    bg='#2d2d2d'
)
stats_title.pack(pady=10)

# Create progress bars for each emotion
progress_bars = {}
progress_labels = {}

for i, emotion in enumerate(labels):
    frame = tk.Frame(right_panel, bg='#2d2d2d')
    frame.pack(fill='x', padx=20, pady=3)
    
    label = tk.Label(
        frame,
        text=emotion,
        font=('Arial', 11, 'bold'),
        fg='#ffffff',
        bg='#2d2d2d',
        width=10,
        anchor='w'
    )
    label.pack(side='left')
    
    progress = ttk.Progressbar(
        frame,
        length=200,
        mode='determinate'
    )
    progress.pack(side='left', padx=5)
    progress_bars[emotion] = progress
    
    percent_label = tk.Label(
        frame,
        text="0%",
        font=('Arial', 10),
        fg='#888888',
        bg='#2d2d2d',
        width=5
    )
    percent_label.pack(side='left')
    progress_labels[emotion] = percent_label

# Bottom info panel
info_frame = tk.Frame(root, bg='#2d2d2d', height=60)
info_frame.pack(fill='x', pady=(10, 0))

fps_label = tk.Label(
    info_frame,
    text="FPS: --",
    font=('Arial', 11),
    fg='#888888',
    bg='#2d2d2d'
)
fps_label.pack(side='left', padx=20, pady=10)

status_label = tk.Label(
    info_frame,
    text="Status: Ready",
    font=('Arial', 11),
    fg='#00ff00',
    bg='#2d2d2d'
)
status_label.pack(side='left', padx=20, pady=10)

model_label = tk.Label(
    info_frame,
    text="Model: CNN (Trained on Kaggle GPU)",
    font=('Arial', 11),
    fg='#888888',
    bg='#2d2d2d'
)
model_label.pack(side='right', padx=20, pady=10)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# FPS calculation
frame_count = 0
start_time = time.time()
fps = 0

def update():
    global frame_count, start_time, fps
    
    ok, frame = cap.read()
    if not ok:
        status_label.configure(text="Status: Camera Error", fg='#ff0000')
        root.after(10, update)
        return
    
    # Calculate FPS
    frame_count += 1
    if frame_count % 10 == 0:
        end_time = time.time()
        fps = 10 / (end_time - start_time)
        start_time = end_time
        fps_label.configure(text=f"FPS: {fps:.1f}")
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48))
    
    pred_idx = None
    pred_probs = None
    
    if len(faces) > 0:
        # Process first detected face
        (x, y, w, h) = faces[0]
        
        # Draw rectangle around face
        color = (0, 255, 0)  # Default green
        
        # Extract face ROI
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = np.expand_dims(np.expand_dims(roi, -1), 0) / 255.0
        
        # Predict emotion
        pred = model.predict(roi, verbose=0)[0]
        pred_idx = int(np.argmax(pred))
        pred_probs = pred
        
        # Get emotion color
        emotion_name = labels[pred_idx]
        color = emotion_colors[emotion_name]
        
        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        
        # Draw emotion label with background
        label_text = f"{emotion_name}: {pred[pred_idx]*100:.1f}%"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.rectangle(frame, (x, y-35), (x+text_w+10, y), color, -1)
        cv2.putText(frame, label_text, (x+5, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        status_label.configure(text="Status: Face Detected", fg='#00ff00')
    else:
        status_label.configure(text="Status: No Face Detected", fg='#ffaa00')
    
    # Display webcam frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img = img.resize((640, 480))
    imgtk = ImageTk.PhotoImage(img)
    video_label.configure(image=imgtk)
    video_label.image = imgtk
    
    # Update emotion display
    if pred_idx is not None:
        try:
            # Load and display emoji
            em = Image.open(emoji_map[pred_idx])
            em = em.resize((200, 200))
            emtk = ImageTk.PhotoImage(em)
            emoji_label.configure(image=emtk)
            emoji_label.image = emtk
            
            # Update emotion name
            emotion_name_label.configure(text=labels[pred_idx], fg=f'#{emotion_colors[labels[pred_idx]][2]:02x}{emotion_colors[labels[pred_idx]][1]:02x}{emotion_colors[labels[pred_idx]][0]:02x}')
            
            # Update confidence
            confidence = pred_probs[pred_idx] * 100
            confidence_label.configure(text=f"Confidence: {confidence:.1f}%")
            
            # Update progress bars
            for i, emotion in enumerate(labels):
                prob = pred_probs[i] * 100
                progress_bars[emotion]['value'] = prob
                progress_labels[emotion].configure(text=f"{prob:.1f}%")
                
        except Exception as e:
            emotion_name_label.configure(text=f"{labels[pred_idx]}", fg='#ff0000')
            confidence_label.configure(text="Emoji file missing")
    else:
        # No face detected - reset displays
        emoji_label.configure(image='')
        emotion_name_label.configure(text="No face detected", fg='#888888')
        confidence_label.configure(text="Confidence: ---%")
        
        # Reset progress bars
        for emotion in labels:
            progress_bars[emotion]['value'] = 0
            progress_labels[emotion].configure(text="0%")
    
    root.after(10, update)

# Cleanup on close
def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start the update loop
update()
root.mainloop()
