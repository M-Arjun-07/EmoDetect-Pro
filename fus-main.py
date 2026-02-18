import tkinter as tk
from tkinter import messagebox, Toplevel, Checkbutton, IntVar
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
from collections import deque

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_1_PATH = "fus-mod/emotion_model_v1.keras"
MODEL_2_PATH = "fus-mod/emotion_model_v2.keras"
TASK_PATH = "fus-mod/face_landmarker.task"

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Ensemble weights
MODEL1_WEIGHT = 0.5
MODEL2_WEIGHT = 0.5

# Temporal smoothing window size
SMOOTHING_WINDOW = 5

# Colors
COLOR_BG = "#1e1e1e"
COLOR_PANEL = "#2d2d2d"
COLOR_TEXT = "#ffffff"
COLOR_ACCENT = "#00ff88"
COLOR_BAR = "#3498db"


class EmotionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1300x720")
        self.window.configure(bg=COLOR_BG)

        self.camera_index = 0
        self.is_running = True
        self.show_logic = True
        self.show_overlay = True

        self.emotion_history = deque(maxlen=SMOOTHING_WINDOW)

        self.load_ai_resources()
        self.create_gui()

        self.cap = cv2.VideoCapture(self.camera_index)
        self.update()

    # ==========================================
    # LOAD BOTH MODELS
    # ==========================================
    def load_ai_resources(self):
        try:
            print("Loading Model 1...")
            self.model1 = load_model(MODEL_1_PATH)

            print("Loading Model 2...")
            self.model2 = load_model(MODEL_2_PATH)

            print("Both Models Loaded Successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Model loading failed:\n{e}")
            self.window.destroy()
            return

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=TASK_PATH),
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

    # ==========================================
    # GUI
    # ==========================================
    def create_gui(self):
        top_bar = tk.Frame(self.window, bg=COLOR_PANEL, height=50)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        tk.Label(top_bar, text="AI Emotion Dashboard (Ensemble)",
                 font=("Arial", 16, "bold"),
                 bg=COLOR_PANEL, fg=COLOR_TEXT).pack(side=tk.LEFT, padx=20)

        btn_style = {"bg": "#444", "fg": "white", "bd": 0, "padx": 10, "pady": 5}

        tk.Button(top_bar, text="Settings",
                  command=self.open_settings, **btn_style).pack(side=tk.RIGHT, padx=10)

        tk.Button(top_bar, text="Switch Camera",
                  command=self.switch_camera, **btn_style).pack(side=tk.RIGHT, padx=10)

        tk.Button(top_bar, text="Exit",
                  command=self.close_app,
                  bg="#e74c3c", fg="white", bd=0,
                  padx=10, pady=5).pack(side=tk.RIGHT, padx=10)

        self.video_frame = tk.Label(self.window, bg="black")
        self.video_frame.pack(fill=tk.BOTH, expand=True)

    # ==========================================
    # OVERLAY
    # ==========================================
    def draw_scanner_overlay(self, frame, landmarks):
        overlay = frame.copy()
        h, w, _ = frame.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        bracket_len = 30
        color = (0, 255, 136)
        thickness = 2

        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), color, 1)

        key_points = [33, 263, 61, 291, 1]
        for idx in key_points:
            if idx < len(points):
                cv2.circle(overlay, points[idx], 4, (255, 255, 255), -1)

        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # ==========================================
    # MAIN LOOP
    # ==========================================
    def update(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()

        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            results = self.landmarker.detect_for_video(
                mp_image, int(time.time() * 1000))

            if results.face_landmarks:
                for face_landmarks in results.face_landmarks:

                    if self.show_overlay:
                        self.draw_scanner_overlay(frame, face_landmarks)

                    h, w, _ = frame.shape
                    x_coords = [lm.x for lm in face_landmarks]
                    y_coords = [lm.y for lm in face_landmarks]

                    x_min = max(0, int(min(x_coords) * w) - 20)
                    x_max = min(w, int(max(x_coords) * w) + 20)
                    y_min = max(0, int(min(y_coords) * h) - 20)
                    y_max = min(h, int(max(y_coords) * h) + 20)

                    face = frame[y_min:y_max, x_min:x_max]

                    if face.size != 0:
                        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        face_resized = cv2.resize(face_gray, (48, 48))
                        face_input = np.reshape(
                            face_resized / 255.0, (1, 48, 48, 1))

                        # ===== ENSEMBLE PREDICTION =====
                        pred1 = self.model1.predict(face_input, verbose=0)[0]
                        pred2 = self.model2.predict(face_input, verbose=0)[0]

                        final_pred = (MODEL1_WEIGHT * pred1) + \
                                     (MODEL2_WEIGHT * pred2)

                        # ===== TEMPORAL SMOOTHING =====
                        self.emotion_history.append(final_pred)
                        smoothed_pred = np.mean(
                            self.emotion_history, axis=0)

                        max_index = np.argmax(smoothed_pred)
                        emotion = EMOTION_LABELS[max_index]

                        cv2.putText(frame,
                                    f"Emotion: {emotion}",
                                    (x_min, y_min - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 255, 136),
                                    2)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(img)
            imgtk = PIL.ImageTk.PhotoImage(image=img)

            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

        self.window.after(10, self.update)

    # ==========================================
    # OTHER
    # ==========================================
    def open_settings(self):
        win = Toplevel(self.window)
        win.title("Settings")
        win.geometry("250x150")
        win.configure(bg=COLOR_PANEL)

        self.overlay_var = IntVar(value=1 if self.show_overlay else 0)

        Checkbutton(win,
                    text="Show Face Scanner",
                    variable=self.overlay_var,
                    command=self.toggle_overlay,
                    bg=COLOR_PANEL,
                    fg="white",
                    selectcolor="#444").pack(pady=20)

    def toggle_overlay(self):
        self.show_overlay = bool(self.overlay_var.get())

    def switch_camera(self):
        self.camera_index += 1
        if self.camera_index > 2:
            self.camera_index = 0
        self.cap.release()
        self.cap = cv2.VideoCapture(self.camera_index)

    def close_app(self):
        self.is_running = False
        self.cap.release()
        self.window.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root, "Advance Emotion Recognition System")
    root.mainloop()
