# gui_ultimate.py - The Most Advanced Emotion Detection GUI on Earth! 🌍
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont, ImageFilter
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import math
from collections import deque

# Load the model
print("🚀 Loading Advanced Neural Network...")
model = load_model('model.h5')
print("✅ AI Model Ready!")

# Configuration
labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emoji_map = {
    0: 'emojis/angry.png', 1: 'emojis/disgust.png', 2: 'emojis/fear.png',
    3: 'emojis/happy.png', 4: 'emojis/neutral.png', 5: 'emojis/sad.png',
    6: 'emojis/surprise.png'
}

# Emotion colors (RGB for Tkinter)
emotion_colors = {
    'Angry': '#FF3838', 'Disgust': '#00C853', 'Fear': '#9C27B0',
    'Happy': '#FFD700', 'Neutral': '#9E9E9E', 'Sad': '#2196F3',
    'Surprise': '#FF9800'
}

# Gradient colors
gradient_colors = {
    'Angry': ['#FF3838', '#D50000', '#B71C1C'],
    'Disgust': ['#00C853', '#00E676', '#1DE9B6'],
    'Fear': ['#9C27B0', '#7B1FA2', '#4A148C'],
    'Happy': ['#FFD700', '#FFC107', '#FF9800'],
    'Neutral': ['#9E9E9E', '#757575', '#616161'],
    'Sad': ['#2196F3', '#1976D2', '#1565C0'],
    'Surprise': ['#FF9800', '#F57C00', '#E65100']
}

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# History for smoothing and charts
emotion_history = deque(maxlen=50)
confidence_history = {emotion: deque([0]*50, maxlen=50) for emotion in labels}

# Main window
root = tk.Tk()
root.title("🌟 ULTIMATE AI EMOTION ANALYZER 🌟")
root.configure(bg='#0a0a0a')
root.geometry("1920x1080")
root.state('zoomed')  # Fullscreen

# Custom style
style = ttk.Style()
style.theme_use('clam')
style.configure("Custom.Horizontal.TProgressbar", 
                troughcolor='#1a1a1a', 
                background='#00d4ff', 
                bordercolor='#0a0a0a',
                lightcolor='#00d4ff',
                darkcolor='#0080ff')

# ==================== HEADER ====================
header_frame = tk.Frame(root, bg='#0a0a0a', height=120)
header_frame.pack(fill='x')
header_frame.pack_propagate(False)

# Animated title with gradient effect
title_canvas = tk.Canvas(header_frame, bg='#0a0a0a', height=120, highlightthickness=0)
title_canvas.pack(fill='both')

def create_gradient_text(canvas, text, x, y, colors):
    """Create gradient text effect"""
    canvas.create_text(x+3, y+3, text=text, font=('Arial Black', 42, 'bold'), 
                      fill='#000000', anchor='center')  # Shadow
    canvas.create_text(x, y, text=text, font=('Arial Black', 42, 'bold'), 
                      fill=colors[0], anchor='center')

create_gradient_text(title_canvas, "🌟 ULTIMATE AI EMOTION ANALYZER 🌟", 960, 40, ['#00d4ff'])

subtitle_label = tk.Label(
    header_frame, 
    text="⚡ Next-Gen Deep Learning | Real-Time Analysis | 63% Neural Accuracy ⚡",
    font=('Arial', 14, 'bold'), 
    fg='#00ff88', 
    bg='#0a0a0a'
)
subtitle_label.place(relx=0.5, rely=0.7, anchor='center')

# ==================== MAIN CONTENT ====================
main_container = tk.Frame(root, bg='#0a0a0a')
main_container.pack(fill='both', expand=True, padx=30, pady=20)

# LEFT COLUMN - Camera and Face Detection
left_column = tk.Frame(main_container, bg='#0a0a0a')
left_column.pack(side='left', fill='both', expand=True, padx=(0, 15))

# Camera frame with neon border
camera_container = tk.Frame(left_column, bg='#00d4ff', bd=3, relief='solid')
camera_container.pack(pady=10)

camera_title_frame = tk.Frame(camera_container, bg='#1a1a1a', height=50)
camera_title_frame.pack(fill='x')

camera_title = tk.Label(
    camera_title_frame,
    text="📹 NEURAL VISION SYSTEM",
    font=('Arial', 18, 'bold'),
    fg='#00d4ff',
    bg='#1a1a1a'
)
camera_title.pack(pady=12)

video_canvas = tk.Canvas(camera_container, width=800, height=600, bg='#000000', highlightthickness=0)
video_canvas.pack(padx=3, pady=3)

# Live stats under camera
stats_frame = tk.Frame(left_column, bg='#1a1a1a', bd=2, relief='solid')
stats_frame.pack(fill='x', pady=10)

stats_grid = tk.Frame(stats_frame, bg='#1a1a1a')
stats_grid.pack(pady=15, padx=20)

# Create stat boxes
stat_labels = {}
stat_values = {}
stats_data = [
    ('FPS', '0', '#00d4ff'),
    ('FACES', '0', '#00ff88'),
    ('LATENCY', '0ms', '#FFD700'),
    ('ACCURACY', '63%', '#FF3838')
]

for i, (label, value, color) in enumerate(stats_data):
    frame = tk.Frame(stats_grid, bg='#2a2a2a', bd=1, relief='solid')
    frame.grid(row=0, column=i, padx=15, pady=5)
    
    lbl = tk.Label(frame, text=label, font=('Arial', 10, 'bold'), 
                   fg='#888888', bg='#2a2a2a')
    lbl.pack(pady=(8, 2), padx=20)
    
    val = tk.Label(frame, text=value, font=('Arial', 20, 'bold'), 
                   fg=color, bg='#2a2a2a')
    val.pack(pady=(2, 8), padx=20)
    
    stat_labels[label] = lbl
    stat_values[label] = val

# MIDDLE COLUMN - Main Emotion Display
middle_column = tk.Frame(main_container, bg='#0a0a0a', width=450)
middle_column.pack(side='left', fill='both', padx=15)
middle_column.pack_propagate(False)

# Emotion display card with glow effect
emotion_card = tk.Frame(middle_column, bg='#1a1a1a', bd=3, relief='solid')
emotion_card.pack(pady=10, fill='both', expand=True)

emotion_header = tk.Label(
    emotion_card,
    text="🎭 DETECTED EMOTION",
    font=('Arial', 16, 'bold'),
    fg='#ffffff',
    bg='#1a1a1a'
)
emotion_header.pack(pady=15)

# Large emoji canvas with animation
emoji_canvas = tk.Canvas(emotion_card, width=280, height=280, bg='#1a1a1a', highlightthickness=0)
emoji_canvas.pack(pady=10)

# Circular progress ring
ring_canvas = tk.Canvas(emotion_card, width=320, height=320, bg='#1a1a1a', highlightthickness=0)
ring_canvas.place(x=65, y=70)

# Emotion name with glow
emotion_name_canvas = tk.Canvas(emotion_card, width=400, height=80, bg='#1a1a1a', highlightthickness=0)
emotion_name_canvas.pack(pady=5)

# Confidence meter
confidence_frame = tk.Frame(emotion_card, bg='#1a1a1a')
confidence_frame.pack(pady=10, fill='x', padx=30)

confidence_label = tk.Label(
    confidence_frame,
    text="CONFIDENCE",
    font=('Arial', 11, 'bold'),
    fg='#888888',
    bg='#1a1a1a'
)
confidence_label.pack()

confidence_bar = ttk.Progressbar(
    confidence_frame,
    length=350,
    mode='determinate',
    style="Custom.Horizontal.TProgressbar"
)
confidence_bar.pack(pady=8)

confidence_percent = tk.Label(
    confidence_frame,
    text="0%",
    font=('Arial', 24, 'bold'),
    fg='#00d4ff',
    bg='#1a1a1a'
)
confidence_percent.pack()

# Emotion history mini-chart
history_label = tk.Label(
    emotion_card,
    text="📊 EMOTION TIMELINE",
    font=('Arial', 12, 'bold'),
    fg='#ffffff',
    bg='#1a1a1a'
)
history_label.pack(pady=(15, 5))

history_canvas = tk.Canvas(emotion_card, width=380, height=100, bg='#0a0a0a', highlightthickness=1, highlightbackground='#333333')
history_canvas.pack(pady=5)

# RIGHT COLUMN - Detailed Analysis
right_column = tk.Frame(main_container, bg='#0a0a0a')
right_column.pack(side='right', fill='both', expand=True, padx=(15, 0))

# Probability breakdown
prob_card = tk.Frame(right_column, bg='#1a1a1a', bd=2, relief='solid')
prob_card.pack(fill='both', expand=True, pady=10)

prob_header = tk.Label(
    prob_card,
    text="🧠 NEURAL NETWORK ANALYSIS",
    font=('Arial', 16, 'bold'),
    fg='#ffffff',
    bg='#1a1a1a'
)
prob_header.pack(pady=15)

# Create advanced progress bars for each emotion
emotion_bars = {}
emotion_values = {}
emotion_canvases = {}

for i, emotion in enumerate(labels):
    # Container for each emotion
    container = tk.Frame(prob_card, bg='#1a1a1a')
    container.pack(fill='x', padx=25, pady=8)
    
    # Emotion icon and name
    header = tk.Frame(container, bg='#1a1a1a')
    header.pack(fill='x')
    
    name_label = tk.Label(
        header,
        text=emotion.upper(),
        font=('Arial', 12, 'bold'),
        fg=emotion_colors[emotion],
        bg='#1a1a1a',
        width=12,
        anchor='w'
    )
    name_label.pack(side='left')
    
    value_label = tk.Label(
        header,
        text="0.0%",
        font=('Arial', 11, 'bold'),
        fg='#888888',
        bg='#1a1a1a',
        width=8,
        anchor='e'
    )
    value_label.pack(side='right')
    emotion_values[emotion] = value_label
    
    # Custom canvas progress bar with gradient
    bar_canvas = tk.Canvas(container, width=400, height=20, bg='#0a0a0a', highlightthickness=0)
    bar_canvas.pack(fill='x', pady=3)
    emotion_canvases[emotion] = bar_canvas
    
    # Mini sparkline chart
    spark_canvas = tk.Canvas(container, width=400, height=30, bg='#0a0a0a', highlightthickness=0)
    spark_canvas.pack(fill='x', pady=2)
    emotion_bars[emotion] = spark_canvas

# Status panel at bottom
status_panel = tk.Frame(root, bg='#1a1a1a', height=60, bd=2, relief='solid')
status_panel.pack(fill='x', side='bottom')

status_left = tk.Frame(status_panel, bg='#1a1a1a')
status_left.pack(side='left', fill='both', expand=True)

status_indicator = tk.Label(
    status_left,
    text="🟢 SYSTEM ACTIVE",
    font=('Arial', 12, 'bold'),
    fg='#00ff88',
    bg='#1a1a1a'
)
status_indicator.pack(side='left', padx=30, pady=15)

model_info = tk.Label(
    status_left,
    text="⚡ Model: CNN-Advanced | Platform: Kaggle GPU | Training: 100 Epochs",
    font=('Arial', 10),
    fg='#888888',
    bg='#1a1a1a'
)
model_info.pack(side='left', padx=20, pady=15)

timestamp_label = tk.Label(
    status_panel,
    text="",
    font=('Arial', 10),
    fg='#666666',
    bg='#1a1a1a'
)
timestamp_label.pack(side='right', padx=30, pady=15)

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Animation variables
frame_count = 0
start_time = time.time()
fps = 0
current_emotion = None
current_confidence = 0
rotation_angle = 0

def draw_gradient_rect(canvas, x1, y1, x2, y2, colors):
    """Draw a gradient rectangle"""
    steps = len(colors)
    width = x2 - x1
    step_width = width / steps
    
    for i in range(steps):
        canvas.create_rectangle(
            x1 + i * step_width, y1,
            x1 + (i + 1) * step_width, y2,
            fill=colors[i], outline=''
        )

def draw_circular_progress(canvas, percent, color):
    """Draw animated circular progress ring"""
    canvas.delete('all')
    center_x, center_y = 160, 160
    radius = 140
    
    # Background circle
    canvas.create_oval(
        center_x - radius, center_y - radius,
        center_x + radius, center_y + radius,
        outline='#2a2a2a', width=8
    )
    
    # Progress arc
    if percent > 0:
        extent = -(percent / 100) * 360
        canvas.create_arc(
            center_x - radius, center_y - radius,
            center_x + radius, center_y + radius,
            start=90, extent=extent,
            outline=color, width=8, style='arc'
        )

def draw_sparkline(canvas, data, color):
    """Draw sparkline chart"""
    canvas.delete('all')
    width = 400
    height = 30
    
    if len(data) < 2:
        return
    
    max_val = max(data) if max(data) > 0 else 1
    points = []
    
    for i, val in enumerate(data):
        x = (i / (len(data) - 1)) * width
        y = height - (val / max_val) * height
        points.extend([x, y])
    
    if len(points) >= 4:
        canvas.create_line(points, fill=color, width=2, smooth=True)

def draw_emotion_timeline(canvas):
    """Draw emotion history timeline"""
    canvas.delete('all')
    width = 380
    height = 100
    
    if len(emotion_history) < 2:
        return
    
    # Draw bars for each emotion in history
    bar_width = width / len(emotion_history)
    
    for i, emotion_idx in enumerate(emotion_history):
        if emotion_idx is not None:
            emotion = labels[emotion_idx]
            color = emotion_colors[emotion]
            x = i * bar_width
            canvas.create_rectangle(
                x, 20, x + bar_width - 1, height - 20,
                fill=color, outline=''
            )

def update_frame():
    global frame_count, start_time, fps, current_emotion, current_confidence, rotation_angle
    
    ok, frame = cap.read()
    if not ok:
        root.after(10, update_frame)
        return
    
    # Calculate FPS
    frame_count += 1
    if frame_count % 10 == 0:
        end_time = time.time()
        fps = 10 / (end_time - start_time)
        start_time = end_time
        stat_values['FPS'].configure(text=f"{fps:.1f}")
    
    # Update timestamp
    timestamp_label.configure(text=time.strftime("%Y-%m-%d %H:%M:%S"))
    
    # Process frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(48, 48))
    
    stat_values['FACES'].configure(text=str(len(faces)))
    
    pred_idx = None
    pred_probs = None
    
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        
        # Extract and predict
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (48, 48))
        roi_normalized = np.expand_dims(np.expand_dims(roi_resized, -1), 0) / 255.0
        
        pred_start = time.time()
        pred = model.predict(roi_normalized, verbose=0)[0]
        pred_time = (time.time() - pred_start) * 1000
        
        pred_idx = int(np.argmax(pred))
        pred_probs = pred
        current_emotion = labels[pred_idx]
        current_confidence = pred[pred_idx] * 100
        
        stat_values['LATENCY'].configure(text=f"{pred_time:.0f}ms")
        
        # Draw enhanced face detection
        color_rgb = emotion_colors[current_emotion]
        color_bgr = tuple(int(color_rgb[i:i+2], 16) for i in (5, 3, 1))
        
        # Animated rectangle
        thickness = int(3 + math.sin(frame_count * 0.1) * 2)
        cv2.rectangle(frame, (x-5, y-5), (x+w+5, y+h+5), color_bgr, thickness)
        
        # Glow effect
        cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), color_bgr, 1)
        
        # Label with background
        label_text = f"{current_emotion}: {current_confidence:.1f}%"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        
        # Rounded rectangle background
        cv2.rectangle(frame, (x-5, y-45), (x+text_w+15, y-5), color_bgr, -1)
        cv2.rectangle(frame, (x-5, y-45), (x+text_w+15, y-5), (255, 255, 255), 2)
        cv2.putText(frame, label_text, (x+5, y-15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Add corner markers
        corner_size = 20
        cv2.line(frame, (x, y), (x+corner_size, y), color_bgr, 3)
        cv2.line(frame, (x, y), (x, y+corner_size), color_bgr, 3)
        cv2.line(frame, (x+w, y), (x+w-corner_size, y), color_bgr, 3)
        cv2.line(frame, (x+w, y), (x+w, y+corner_size), color_bgr, 3)
        cv2.line(frame, (x, y+h), (x+corner_size, y+h), color_bgr, 3)
        cv2.line(frame, (x, y+h), (x, y+h-corner_size), color_bgr, 3)
        cv2.line(frame, (x+w, y+h), (x+w-corner_size, y+h), color_bgr, 3)
        cv2.line(frame, (x+w, y+h), (x+w, y+h-corner_size), color_bgr, 3)
        
        status_indicator.configure(text="🟢 FACE DETECTED", fg='#00ff88')
    else:
        status_indicator.configure(text="🟡 SCANNING...", fg='#FFD700')
    
    # Display video frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img = img.resize((800, 600))
    
    # Add scanline effect
    draw = ImageDraw.Draw(img)
    for i in range(0, 600, 4):
        draw.line([(0, i), (800, i)], fill=(0, 200, 255, 30), width=1)
    
    imgtk = ImageTk.PhotoImage(img)
    video_canvas.create_image(0, 0, anchor='nw', image=imgtk)
    video_canvas.image = imgtk
    
    # Update emotion display
    if pred_idx is not None:
        # Update history
        emotion_history.append(pred_idx)
        for i, prob in enumerate(pred_probs):
            confidence_history[labels[i]].append(prob * 100)
        
        # Load and display emoji with animation
        try:
            em = Image.open(emoji_map[pred_idx])
            
            # Rotate emoji slowly
            rotation_angle = (rotation_angle + 2) % 360
            em_rotated = em.rotate(math.sin(rotation_angle * math.pi / 180) * 5, expand=False)
            em_resized = em_rotated.resize((260, 260))
            
            # Add glow effect
            glow = em_resized.filter(ImageFilter.GaussianBlur(10))
            
            emtk = ImageTk.PhotoImage(em_resized)
            emoji_canvas.create_image(140, 140, image=emtk)
            emoji_canvas.image = emtk
            
        except:
            pass
        
        # Draw circular progress
        draw_circular_progress(ring_canvas, current_confidence, emotion_colors[current_emotion])
        
        # Update emotion name with glow effect
        emotion_name_canvas.delete('all')
        # Shadow
        emotion_name_canvas.create_text(252, 42, text=current_emotion.upper(), 
                                       font=('Arial Black', 36, 'bold'),
                                       fill='#000000')
        # Main text
        emotion_name_canvas.create_text(250, 40, text=current_emotion.upper(), 
                                       font=('Arial Black', 36, 'bold'),
                                       fill=emotion_colors[current_emotion])
        
        # Update confidence
        confidence_bar['value'] = current_confidence
        confidence_percent.configure(text=f"{current_confidence:.1f}%")
        
        # Update probability bars
        for i, emotion in enumerate(labels):
            prob = pred_probs[i] * 100
            emotion_values[emotion].configure(text=f"{prob:.1f}%")
            
            # Draw gradient progress bar
            canvas = emotion_canvases[emotion]
            canvas.delete('all')
            
            # Background
            canvas.create_rectangle(0, 0, 400, 20, fill='#0a0a0a', outline='#333333')
            
            # Progress with gradient
            if prob > 0:
                width = int(400 * prob / 100)
                colors = gradient_colors[emotion]
                steps = len(colors)
                step_width = width / steps
                
                for j in range(steps):
                    x1 = j * step_width
                    x2 = min((j + 1) * step_width, width)
                    canvas.create_rectangle(x1, 0, x2, 20, fill=colors[j], outline='')
            
            # Draw sparkline
            draw_sparkline(emotion_bars[emotion], confidence_history[emotion], emotion_colors[emotion])
        
        # Draw timeline
        draw_emotion_timeline(history_canvas)
    
    root.after(10, update_frame)

def on_closing():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

# Start
print("🎬 Launching Ultimate GUI...")
update_frame()
root.mainloop()
