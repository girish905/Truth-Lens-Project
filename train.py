import os
import numpy as np
import tensorflow as tf
from utils.forensic import build_tensor

# Prepare dataset
X, y = [], []
for label, folder in enumerate(["real", "fake"]):
    folder_path = f"dataset/{folder}"
    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)
        try:
            tensor = build_tensor(img_path)
            X.append(tensor)
            y.append(label)
        except:
            continue

X = np.array(X)
y = np.array(y)
print("Dataset shape:", X.shape, "Labels shape:", y.shape)

# Lightweight CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10, zoom_range=0.1, horizontal_flip=True
)

# Train model
model.fit(datagen.flow(X, y, batch_size=8), epochs=15)

# Save model
model.save("truthlens_model.h5")
print("Model trained and saved successfully!")
