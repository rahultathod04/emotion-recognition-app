"""
Custom Emotion Model Trainer
-----------------------------
Trains a CNN on the FER2013 dataset from scratch.
Run this once — it saves your model as 'emotion_model.h5'
Then use it in your app!
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dropout,
                                     Flatten, Dense, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# ── 1. Settings ───────────────────────────────────────────────────────────────

DATA_DIR    = "data"           # folder containing train/ and test/
IMG_SIZE    = 48               # FER2013 images are 48x48
BATCH_SIZE  = 64
EPOCHS      = 50               # will stop early if no improvement
NUM_CLASSES = 7

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

train_datagen = ImageDataGenerator(
    rescale           = 1.0 / 255,      # normalize pixels to 0-1
    rotation_range    = 15,
    width_shift_range = 0.1,
    height_shift_range= 0.1,
    horizontal_flip   = True,
    zoom_range        = 0.1
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size   = (IMG_SIZE, IMG_SIZE),
    color_mode    = "grayscale",
    batch_size    = BATCH_SIZE,
    class_mode    = "categorical",
    shuffle       = True
)

test_gen = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size   = (IMG_SIZE, IMG_SIZE),
    color_mode    = "grayscale",
    batch_size    = BATCH_SIZE,
    class_mode    = "categorical",
    shuffle       = False
)

print(f"\n✅ Training samples : {train_gen.samples}")
print(f"✅ Testing samples  : {test_gen.samples}")
print(f"✅ Classes found    : {list(train_gen.class_indices.keys())}\n")

# ── 3. Build the CNN Model ────────────────────────────────────────────────────
# This is a Convolutional Neural Network — the standard for image tasks

model = Sequential([

    # Block 1 — learn basic edges and shapes
    Conv2D(64, (3,3), activation="relu", padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    Conv2D(64, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # Block 2 — learn more complex features
    Conv2D(128, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(128, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # Block 3 — learn high level face features
    Conv2D(256, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(256, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Dropout(0.25),

    # Flatten and classify
    Flatten(),
    Dense(512, activation="relu"),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation="softmax")   # 7 emotion outputs
])

model.summary()

# ── 4. Compile ────────────────────────────────────────────────────────────────

model.compile(
    optimizer = Adam(learning_rate=0.001),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"]
)

# ── 5. Callbacks — smart training tricks ──────────────────────────────────────

callbacks = [
    # Save the best model automatically
    ModelCheckpoint(
        "emotion_model.h5",
        monitor   = "val_accuracy",
        save_best_only= True,
        verbose   = 1
    ),
    # Stop training if no improvement for 10 epochs
    EarlyStopping(
        monitor  = "val_accuracy",
        patience = 10,
        verbose  = 1,
        restore_best_weights = True
    ),
    # Reduce learning rate when stuck
    ReduceLROnPlateau(
        monitor  = "val_loss",
        factor   = 0.5,
        patience = 5,
        verbose  = 1
    )
]

# ── 6. Train! ─────────────────────────────────────────────────────────────────

print("\n🚀 Starting training...\n")

history = model.fit(
    train_gen,
    epochs            = EPOCHS,
    validation_data   = test_gen,
    callbacks         = callbacks
)

print("\n✅ Training complete! Model saved as emotion_model.h5")

# ── 7. Plot accuracy & loss curves ───────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history.history["accuracy"],     label="Train Accuracy")
ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
ax1.set_title("Model Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()

ax2.plot(history.history["loss"],     label="Train Loss")
ax2.plot(history.history["val_loss"], label="Val Loss")
ax2.set_title("Model Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()

plt.tight_layout()
plt.savefig("training_results.png")
plt.show()
print("📊 Training graph saved as training_results.png")
