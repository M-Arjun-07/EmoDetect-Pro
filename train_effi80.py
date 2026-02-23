import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.utils.class_weight import compute_class_weight

# =====================================================
# 1️⃣ GPU CONTROL (Stable ~3GB usage)
# =====================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

tf.keras.mixed_precision.set_global_policy("mixed_float16")

# =====================================================
# 2️⃣ PARAMETERS (Balanced for RTX 3050 4GB)
# =====================================================
IMG_SIZE = 224
BATCH_SIZE = 28          # Uses ~3GB VRAM safely
EPOCHS_STAGE1 = 12
EPOCHS_STAGE2 = 25

TRAIN_PATH = "dataset/train"
VAL_PATH = "dataset/test"

# =====================================================
# 3️⃣ DATA LOADING (No RAM explosion)
# =====================================================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_PATH,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="rgb",
    shuffle=False
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
AUTOTUNE = tf.data.AUTOTUNE

# =====================================================
# 4️⃣ FER DATASET CORRECTION + PREPROCESSING
# =====================================================
def preprocess(x, y):
    x = tf.cast(x, tf.float32)

    # Improve low contrast (FER fix)
    x = tf.image.random_contrast(x, 0.8, 1.3)

    # EfficientNet normalization
    x = preprocess_input(x)

    return x, y

train_ds = train_ds.map(preprocess, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(preprocess, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

# =====================================================
# 5️⃣ CLASS WEIGHTS (Fix imbalance)
# =====================================================
y_train = np.concatenate([y.numpy() for x, y in train_ds], axis=0)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights_array))

# =====================================================
# 6️⃣ DATA AUGMENTATION (Safe)
# =====================================================
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.12),
    layers.RandomZoom(0.12),
])

# =====================================================
# 7️⃣ BUILD MODEL
# =====================================================
base_model = EfficientNetB2(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# Force float32 output (mixed precision safe save)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

model = keras.Model(inputs, outputs)

# =====================================================
# 8️⃣ COSINE LEARNING RATE
# =====================================================
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000
)

model.compile(
    optimizer=keras.optimizers.Adam(lr_schedule),
    # FIX: Removed label_smoothing from SparseCategoricalCrossentropy
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

# =====================================================
# 9️⃣ CALLBACKS (Crash Protection)
# =====================================================
os.makedirs("models", exist_ok=True)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "models/best_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=6,
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=3,
        verbose=1
    )
]

# =====================================================
# 🚀 STAGE 1 TRAIN
# =====================================================
print("🚀 Stage 1 Training")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE1,
    class_weight=class_weights,
    callbacks=callbacks
)

# =====================================================
# 🔥 STAGE 2 FINE TUNE
# =====================================================
base_model.trainable = True

for layer in base_model.layers[:int(len(base_model.layers)*0.6)]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(1e-5),
    # FIX: Removed label_smoothing from SparseCategoricalCrossentropy here too
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

print("🔥 Stage 2 Fine Tuning")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_STAGE2,
    class_weight=class_weights,
    callbacks=callbacks
)

# =====================================================
# 10️⃣ FINAL SAVE
# =====================================================
model.save("models/final_emotion_model.h5")

print("✅ Training Complete — Model Saved Safely")