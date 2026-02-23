import os
import cv2
from tqdm import tqdm

# ==========================
# SETTINGS
# ==========================

SOURCE_DIR = "dataset"
TARGET_DIR = "dataset_resized"
IMG_SIZE = 224  # Resize to 224x224

# ==========================
# FUNCTION TO PROCESS DATA
# ==========================

def process_and_save_images(source_folder, target_folder):
    for emotion in os.listdir(source_folder):
        emotion_path = os.path.join(source_folder, emotion)
        
        if not os.path.isdir(emotion_path):
            continue

        # Create target emotion folder
        target_emotion_path = os.path.join(target_folder, emotion)
        os.makedirs(target_emotion_path, exist_ok=True)

        print(f"Processing {emotion}...")

        for img_name in tqdm(os.listdir(emotion_path)):
            img_path = os.path.join(emotion_path, img_name)

            try:
                # Read image (grayscale)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                if img is None:
                    continue

                # Resize
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                # Convert grayscale → RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

                # Save image
                save_path = os.path.join(target_emotion_path, img_name)
                cv2.imwrite(save_path, img)

            except Exception as e:
                print("Error processing:", img_path)

# ==========================
# MAIN
# ==========================

for split in ["train", "test"]:
    source_split_path = os.path.join(SOURCE_DIR, split)
    target_split_path = os.path.join(TARGET_DIR, split)

    os.makedirs(target_split_path, exist_ok=True)
    process_and_save_images(source_split_path, target_split_path)

print("\n✅ Dataset conversion completed successfully!")
