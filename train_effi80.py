import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.utils.class_weight import compute_class_weight

# ==============================
# CONFIG
# ==============================
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 60

TRAIN_PATH = "dataset/train"
TEST_PATH = "dataset/test"

emotion_labels = sorted(os.listdir(TRAIN_PATH))
NUM_CLASSES = len(emotion_labels)

print("Classes:", emotion_labels)

# ==============================
# LOAD DATASETS (Memory Efficient)
# ==============================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

AUTOTUNE = tf.data.AUTOTUNE

# ==============================
# DATA AUGMENTATION
# ==============================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

def preprocess(ds, training=False):
    def _process(x, y):
        x = tf.cast(x, tf.float32)
        if training:
            x = data_augmentation(x)
        x = preprocess_input(x)
        y = tf.one_hot(y, NUM_CLASSES)
        return x, y

    return ds.map(_process, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

train_ds = preprocess(train_ds, training=True)
val_ds = preprocess(val_ds, training=False)

# ==============================
# CLASS WEIGHTS (SAFE VERSION)
# ==============================
y_train_all = []

for _, labels in tf.keras.preprocessing.image_dataset_from_directory(
        TRAIN_PATH,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False):
    y_train_all.extend(labels.numpy())

y_train_all = np.array(y_train_all)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_all),
    y=y_train_all
)

class_weights = {i: float(class_weights_array[i]) for i in range(len(class_weights_array))}
print("Class Weights:", class_weights)

# ==============================
# BUILD MODEL
# ==============================
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)

# ==============================
# STAGE 1 TRAINING
# ==============================
model.compile(
    optimizer=Adam(1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

print("\n🚀 Stage 1 Training")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weights
)

# ==============================
# STAGE 2 FINE-TUNING
# ==============================
base_model.trainable = True

# Unfreeze more layers (stronger adaptation)
for layer in base_model.layers[:50]:
    layer.trainable = False

model.compile(
    optimizer=Adam(5e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    metrics=["accuracy"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=5,
    verbose=1
)

print("\n🔥 Stage 2 Fine-Tuning")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr]
)

# ==============================
# SAVE MODEL (NO CRASH VERSION)
# ==============================
os.makedirs("models", exist_ok=True)

model.save("models/emotion_efficientnet_pro.keras")
print("Model saved successfully 🚀")