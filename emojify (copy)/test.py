import cv2
import numpy as np
from tensorflow.keras.models import load_model

# model load
model = load_model("model.h5")

# emotions aur emojis map
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
emoji_paths = {
    'angry': 'emojis/angry.png',
    'disgust': 'emojis/disgust.png',
    'fear': 'emojis/fear.png',
    'happy': 'emojis/happy.png',
    'neutral': 'emojis/neutral.png',
    'sad': 'emojis/sad.png',
    'surprise': 'emojis/surprise.png'
}

# emojis load memory me
emojis = {emo: cv2.imread(path, cv2.IMREAD_UNCHANGED) for emo, path in emoji_paths.items()}

# function: overlay PNG with transparency
def overlay_image(bg, fg, x, y):
    h, w = fg.shape[:2]
    if y+h > bg.shape[0] or x+w > bg.shape[1]:
        return bg
    alpha_fg = fg[:, :, 3] / 255.0
    alpha_bg = 1.0 - alpha_fg
    for c in range(3):
        bg[y:y+h, x:x+w, c] = (alpha_fg * fg[:, :, c] + alpha_bg * bg[y:y+h, x:x+w, c])
    return bg

# webcam open
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face.reshape(1, 48, 48, 1) / 255.0

    prediction = model.predict(face, verbose=0)
    emotion = classes[np.argmax(prediction)]

    # emoji lo
    emoji = emojis[emotion]
    if emoji is not None:
        emoji_resized = cv2.resize(emoji, (150, 150))
        frame = overlay_image(frame, emoji_resized, 10, 10)  # top-left corner me emoji

    # text bhi dikhana chaho to
    cv2.putText(frame, emotion, (200, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Emoji Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

