# train.py (full mega project)
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# dataset paths
train_dir = 'data/train'
val_dir   = 'data/test'

# generators with data augmentation for better accuracy
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)
val_gen   = ImageDataGenerator(rescale=1./255)

batch_size = 64
img_size   = (48, 48)

train_loader = train_gen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size,
    color_mode='grayscale', class_mode='categorical', shuffle=True)

val_loader = val_gen.flow_from_directory(
    val_dir, target_size=img_size, batch_size=batch_size,
    color_mode='grayscale', class_mode='categorical', shuffle=False)

# CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(1024, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

# model training (50 epochs for better accuracy)
history = model.fit(
    train_loader,
    epochs=50,
    validation_data=val_loader
)

# save weights
model.save('model.h5')
print("✅ Training complete, weights saved to model.h5")
