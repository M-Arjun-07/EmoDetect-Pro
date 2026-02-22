import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
import os
import cv2

# Load model
model = load_model("models/emotion_model_70_percent.keras")

TRAIN_PATH = "dataset/train"
TEST_PATH = "dataset/test"
emotion_labels = sorted(os.listdir(TRAIN_PATH))

# Load test data (reuse your function logic)
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

    return np.array(images), np.array(labels)

X_test, y_test = load_dataset(TEST_PATH)

predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=emotion_labels))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=emotion_labels,
            yticklabels=emotion_labels,
            cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
