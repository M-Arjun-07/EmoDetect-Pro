import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ==============================
# DATASET PATHS
# ==============================
TRAIN_PATH = "dataset/train"
TEST_PATH = "dataset/test"

emotion_labels = sorted(os.listdir(TRAIN_PATH))
print("Emotion Classes:", emotion_labels)


# ==============================
# LOAD DATASET FUNCTION
# ==============================
def load_dataset(path):
    images = []
    labels = []

    for label_index, emotion in enumerate(emotion_labels):
        emotion_path = os.path.join(path, emotion)

        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (48, 48))
            img = img / 255.0
            img = np.expand_dims(img, axis=-1)

            images.append(img)
            labels.append(label_index)

    X = np.array(images, dtype="float32")
    y = np.array(labels)

    print(f"{path} Loaded")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y


# ==============================
# BUILD MODEL
# ==============================
def build_model(input_shape, num_classes):
    model = Sequential()

    model.add(Input(shape=input_shape))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.6))

    model.add(Dense(num_classes, activation='softmax'))

    return model


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    # Load train and test separately
    X_train, y_train = load_dataset(TRAIN_PATH)
    X_test, y_test = load_dataset(TEST_PATH)

    y_train = to_categorical(y_train, num_classes=len(emotion_labels))
    y_test = to_categorical(y_test, num_classes=len(emotion_labels))

    model = build_model((48, 48, 1), len(emotion_labels))

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Data augmentation (train only)
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )

    datagen.fit(X_train)

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=32),
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/emotion_model.keras")

    print("Model Saved Successfully!")
