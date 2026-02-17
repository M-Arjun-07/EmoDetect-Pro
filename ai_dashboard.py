import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import threading
import mediapipe as mp
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

# ==============================
# LOAD MODEL
# ==============================
model = load_model("models/emotion_model_improved.keras")
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ==============================
# GLOBAL VARIABLES
# ==============================
current_camera_index = 0
show_landmarks = True
fullscreen_mode = False

# ==============================
# MEDIAPIPE SETUP
# ==============================
# -------- MediaPipe NEW API --------
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

landmarker = FaceLandmarker.create_from_options(options)

# ==============================
# MAIN APP
# ==============================
class EmotionApp:

    def __init__(self, root):
        self.root = root
        self.root.title("AI Emotion Dashboard")
        self.root.configure(bg="#121212")
        self.root.geometry("1300x750")

        self.cap = cv2.VideoCapture(current_camera_index)

        self.create_ui()
        self.update_frame()

    # ==============================
    # UI
    # ==============================
    def create_ui(self):

        # MENU
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Camera Settings", command=self.open_settings)
        settings_menu.add_command(label="Toggle Fullscreen", command=self.toggle_fullscreen)
        settings_menu.add_separator()
        settings_menu.add_command(label="Exit", command=self.root.quit)

        # MAIN FRAMES
        self.left_frame = tk.Frame(self.root, bg="#1e1e1e", width=400)
        self.left_frame.pack(side="left", fill="y")

        self.right_frame = tk.Frame(self.root, bg="#000000")
        self.right_frame.pack(side="right", expand=True, fill="both")

        # CAMERA LABEL
        self.camera_label = tk.Label(self.right_frame)
        self.camera_label.pack(expand=True)

        # LEFT PANEL CONTENT
        tk.Label(self.left_frame, text="AI Pipeline", fg="white",
                 bg="#1e1e1e", font=("Arial", 16)).pack(pady=10)

        self.status_label = tk.Label(self.left_frame, text="Face: Not Detected",
                                     fg="red", bg="#1e1e1e")
        self.status_label.pack(pady=5)

        tk.Label(self.left_frame, text="Emotion Probabilities",
                 fg="white", bg="#1e1e1e").pack(pady=10)

        self.bars = {}
        for emotion in emotion_labels:
            frame = tk.Frame(self.left_frame, bg="#1e1e1e")
            frame.pack(fill="x", padx=10, pady=2)

            label = tk.Label(frame, text=emotion, width=10,
                             anchor="w", fg="white", bg="#1e1e1e")
            label.pack(side="left")

            bar = tk.Canvas(frame, height=15, bg="#333333", highlightthickness=0)
            bar.pack(side="left", fill="x", expand=True)

            self.bars[emotion] = bar

    # ==============================
    # CAMERA UPDATE LOOP
    # ==============================
    def update_frame(self):

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(10, self.update_frame)
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = landmarker.detect_for_video(mp_image, int(cv2.getTickCount()))

        if results.face_landmarks:
            self.status_label.config(text="Face: Detected", fg="lime")

            for face_landmarks in results.face_landmarks:
                h, w, _ = frame.shape

                # Get bounding box
                xs = [int(lm.x * w) for lm in face_landmarks]
                ys = [int(lm.y * h) for lm in face_landmarks]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                face = frame[y_min:y_max, x_min:x_max]

                if face.size != 0:
                    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face_gray = cv2.resize(face_gray, (48, 48))
                    face_gray = face_gray / 255.0
                    face_gray = np.reshape(face_gray, (1, 48, 48, 1))

                    prediction = model.predict(face_gray, verbose=0)[0]
                    emotion = emotion_labels[np.argmax(prediction)]

                    # Update bars
                    for i, emo in enumerate(emotion_labels):
                        self.update_bar(emo, prediction[i])

                    # Draw simple overlay
                    if show_landmarks:
                        self.draw_clean_overlay(frame, face_landmarks)

                    cv2.putText(frame, emotion,
                                (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)

        else:
            self.status_label.config(text="Face: Not Detected", fg="red")

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)

        self.camera_label.imgtk = imgtk
        self.camera_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    # ==============================
    # CLEAN OVERLAY
    # ==============================
    def draw_clean_overlay(self, frame, landmarks):

        h, w, _ = frame.shape

        # Jawline (0–16)
        jaw_points = [landmarks[i] for i in range(0, 17)]
        self.draw_line(frame, jaw_points, w, h)

        # Eyebrows
        left_eyebrow = [landmarks[i] for i in range(70, 80)]
        right_eyebrow = [landmarks[i] for i in range(300, 310)]
        self.draw_line(frame, left_eyebrow, w, h)
        self.draw_line(frame, right_eyebrow, w, h)

        # Nose bridge
        nose = [landmarks[i] for i in [1, 2, 5, 4]]
        self.draw_line(frame, nose, w, h)

        # Mouth
        mouth = [landmarks[i] for i in range(61, 88)]
        self.draw_line(frame, mouth, w, h)

    def draw_line(self, frame, points, w, h):
        for i in range(len(points) - 1):
            x1, y1 = int(points[i].x * w), int(points[i].y * h)
            x2, y2 = int(points[i+1].x * w), int(points[i+1].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

    # ==============================
    # UPDATE PROBABILITY BARS
    # ==============================
    def update_bar(self, emotion, value):
        bar = self.bars[emotion]
        bar.delete("all")
        width = bar.winfo_width()
        bar.create_rectangle(0, 0, width * value, 15, fill="#00ff99")

    # ==============================
    # SETTINGS WINDOW
    # ==============================
    def open_settings(self):

        settings = tk.Toplevel(self.root)
        settings.title("Settings")
        settings.configure(bg="#1e1e1e")
        settings.geometry("300x200")

        tk.Label(settings, text="Camera Index:", fg="white",
                 bg="#1e1e1e").pack(pady=5)

        camera_entry = tk.Entry(settings)
        camera_entry.insert(0, str(current_camera_index))
        camera_entry.pack(pady=5)

        def apply_camera():
            global current_camera_index
            current_camera_index = int(camera_entry.get())
            self.cap.release()
            self.cap = cv2.VideoCapture(current_camera_index)

        tk.Button(settings, text="Apply", command=apply_camera).pack(pady=10)

        def toggle_landmarks():
            global show_landmarks
            show_landmarks = not show_landmarks

        tk.Button(settings, text="Toggle Landmarks",
                  command=toggle_landmarks).pack(pady=10)

    # ==============================
    # FULLSCREEN TOGGLE
    # ==============================
    def toggle_fullscreen(self):
        global fullscreen_mode
        fullscreen_mode = not fullscreen_mode

        if fullscreen_mode:
            self.left_frame.pack_forget()
        else:
            self.left_frame.pack(side="left", fill="y")


# ==============================
# RUN APP
# ==============================
root = tk.Tk()
app = EmotionApp(root)
root.mainloop()
