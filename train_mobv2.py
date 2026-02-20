import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.utils.class_weight import compute_class_weight

# =========================
# GPU SAFE SETTINGS
# =========================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# =========================
# SETTINGS
# =========================
TRAIN_PATH = "dataset/train"
TEST_PATH = "dataset/test"
IMG_SIZE = 64
BATCH_SIZE = 32

emotion_labels = sorted(os.listdir(TRAIN_PATH))
print("Emotion Classes:", emotion_labels)

# =========================
# LOAD DATASET
# =========================
def load_dataset(path):
    images = []
    labels = []

    for idx, emotion in enumerate(emotion_labels):
        folder = os.path.join(path, emotion)

        for file in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, file))
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0

            images.append(img)
            labels.append(idx)

    return np.array(images, dtype="float32"), np.array(labels)

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    X_train, y_train = load_dataset(TRAIN_PATH)
    X_test, y_test = load_dataset(TEST_PATH)

    y_train_labels = y_train.copy()

    y_train = to_categorical(y_train, len(emotion_labels))
    y_test = to_categorical(y_test, len(emotion_labels))

    # Handle class imbalance
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_labels),
        y=y_train_labels
    )
    class_weights = dict(enumerate(class_weights))

    # =========================
    # LOAD MOBILENETV2
    # =========================
    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    base_model.trainable = False  # Stage 1

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    output = Dense(len(emotion_labels), activation="softmax")(x)

    model = Model(base_model.input, output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n===== STAGE 1 TRAINING =====\n")

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True
    )

    early_stop = EarlyStopping(patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(factor=0.3, patience=3)

    model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=15,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weights
    )

    # =========================
    # STAGE 2 – FINE TUNING
    # =========================
    print("\n===== STAGE 2 FINE TUNING =====\n")

    base_model.trainable = True

    for layer in base_model.layers[:int(len(base_model.layers)*0.7)]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=20,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weights
    )

    os.makedirs("models", exist_ok=True)
    model.save("models/emotion_mobilenetv2_64.keras")

    print("🔥 MobileNetV2 Model Saved Successfully!")