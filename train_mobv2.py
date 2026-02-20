import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ==============================
# CONFIG
# ==============================
TRAIN_PATH = "dataset/train"
TEST_PATH = "dataset/test"
IMG_SIZE = 96
BATCH_SIZE = 16
EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 40

emotion_labels = sorted(os.listdir(TRAIN_PATH))
NUM_CLASSES = len(emotion_labels)

print("Emotion Classes:", emotion_labels)


# ==============================
# LOAD DATASET
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
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Convert grayscale → RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            img = preprocess_input(img)

            images.append(img)
            labels.append(label_index)

    X = np.array(images, dtype="float32")
    y = np.array(labels)

    print(f"{path} Loaded")
    print("X shape:", X.shape)

    return X, y


# ==============================
# BUILD MODEL
# ==============================
def build_model():

    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base_model.trainable = False  # Stage 1 freeze

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    output = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    return model, base_model


# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    X_train, y_train = load_dataset(TRAIN_PATH)
    X_test, y_test = load_dataset(TEST_PATH)

    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)

    # Class Weights (VERY IMPORTANT for FER2013)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = dict(enumerate(class_weights))
    print("Class Weights:", class_weights)

    model, base_model = build_model()

    # ==============================
    # STAGE 1: Train classifier head
    # ==============================
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True
    )

    datagen.fit(X_train)

    print("\n🚀 Stage 1 Training (Frozen Base)")
    model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
        epochs=EPOCHS_STAGE1,
        validation_data=(X_test, y_test_cat),
        class_weight=class_weights
    )

    # ==============================
    # STAGE 2: Fine-tune deeper layers
    # ==============================
    print("\n🔥 Stage 2 Fine-Tuning")

    base_model.trainable = True

    # Freeze first 100 layers, fine-tune last layers
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        verbose=1
    )

    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
        epochs=EPOCHS_STAGE2,
        validation_data=(X_test, y_test_cat),
        class_weight=class_weights,
        callbacks=[early_stop, reduce_lr]
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/emotion_model_70_percent.keras")

    print("\n✅ Production Model Saved!")