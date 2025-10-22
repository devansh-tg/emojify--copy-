# gui.py (mega project GUI)
import tkinter as tk
from PIL import Image, ImageTk
import cv2, numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# build same CNN model
def build_model():
    m = Sequential([
        Conv2D(32,(3,3),activation='relu',input_shape=(48,48,1)),
        Conv2D(64,(3,3),activation='relu'),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Conv2D(128,(3,3),activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128,(3,3),activation='relu'),
        MaxPooling2D((2,2)),
        Dropout(0.25),

        Flatten(),
        Dense(1024,activation='relu'),
        Dropout(0.5),
        Dense(7,activation='softmax')
    ])
    return m

# load model
model = build_model()
model.load_weights('model.h5')

labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
emoji_map = {
    0:'emojis/angry.png',
    1:'emojis/disgust.png',
    2:'emojis/fear.png',
    3:'emojis/happy.png',
    4:'emojis/neutral.png',
    5:'emojis/sad.png',
    6:'emojis/surprise.png'
}

# face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')

# tkinter window
root = tk.Tk()
root.title("Emoji Creator")
root.configure(bg='black')
root.geometry("1200x800")

video_label = tk.Label(root, bg='black')
video_label.pack(side='left', padx=20, pady=20)

emoji_label = tk.Label(root, bg='black')
emoji_label.pack(side='right', padx=20, pady=20)

text_label = tk.Label(root, text='', font=('Arial', 32, 'bold'),
                      fg='white', bg='black')
text_label.pack(side='top', pady=10)

cap = cv2.VideoCapture(0)

def update():
    ok, frame = cap.read()
    if not ok:
        root.after(10, update)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    pred_idx = None
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48,48))
        roi = np.expand_dims(np.expand_dims(roi, -1), 0) / 255.0
        pred = model.predict(roi, verbose=0)[0]
        pred_idx = int(np.argmax(pred))
        break  # only first face

    # show webcam frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
    video_label.configure(image=imgtk)
    video_label.image = imgtk

    # show emoji
    if pred_idx is not None:
        try:
            em = Image.open(emoji_map[pred_idx])
            em = em.resize((256,256))
            emtk = ImageTk.PhotoImage(em)
            emoji_label.configure(image=emtk)
            emoji_label.image = emtk
            text_label.configure(text=labels[pred_idx])
        except Exception as e:
            text_label.configure(text=f"{labels[pred_idx]} (emoji missing)")

    root.after(10, update)

root.protocol("WM_DELETE_WINDOW", lambda: (cap.release(), root.destroy()))
update()
root.mainloop()
