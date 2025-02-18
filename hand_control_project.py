import cv2
import pyautogui
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model



from pathlib import Path
import os

import warnings
warnings.filterwarnings("ignore")

#load data 
dataset = "HG14/HG14-Hand Gesture"

img_size = 64
batch_size = 32
epochs = 30

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split = 0.2
    )

train_generator = train_datagen.flow_from_directory(
    dataset,
    target_size= (img_size, img_size),
    batch_size= batch_size,
    class_mode = 'categorical',
    subset = 'training'
    )

val_generator = train_datagen.flow_from_directory(
    dataset,
    target_size= (img_size, img_size),
    batch_size= batch_size,
    class_mode = 'categorical',
    subset = 'validation'
    )

# Create CNN model
model = Sequential()


model.add(Conv2D(32,(3,3), activation = 'relu', input_shape = (img_size,img_size,3)))
model.add(MaxPooling2D(2,2)) #pool size

model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128,(3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(256,(3,3), activation = 'relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_generator.class_indices),activation="softmax"))


# compile model
model.compile(optimizer = "adam", 
              loss = "categorical_crossentropy", 
              metrics=["accuracy"]
              ) 

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("best_model.h5",monitor="val_loss", save_best_only=True)]
# model training
model.fit(
    train_generator,
    validation_data = val_generator,
    epochs = epochs,
    callbacks=callbacks
    )

# save model
model.save("hand_gesture_model.h5")

class_labels = list(train_generator.class_indices.keys())

command_map = {
    "Gesture_1": lambda: pyautogui.press("volumeup"),
    "Gesture_2": lambda: pyautogui.press("volumedown"),
    "Gesture_0": lambda:  pyautogui.press("playpause"),
    "Gesture_13": lambda: pyautogui.press("space")
    }         

camera_index = int(input("Kullanılacak kamera indeksini girin (varsayılan: 0):") or 0)
cap = cv2.VideoCapture(camera_index)
model = load_model("hand_gesture_model.h5")

while True:
    ret, frame = cap.read()
    if not ret:
        break 
    
    img = cv2.resize(frame, (img_size, img_size))
    img = img /255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    predicted_class = class_labels[np.argmax(prediction)]
    
    cv2.putText(frame, predicted_class, (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)
    
    
    if predicted_class in command_map:
        command_map[predicted_class]()
        
    if not os.environ.get("HEADLESS_MODE"):
        cv2.imshow("Hand Gesture Recognition", frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destrotAllWindows()
