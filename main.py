import tkinter as tk
from tkinter import messagebox, Toplevel, Checkbutton, IntVar
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
MODEL_PATH = "models/emotion_efficientnet.h5"
TASK_PATH = "face_landmarker.task"
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Visual Colors
COLOR_BG = "#1e1e1e"       # Dark Gray Background
COLOR_PANEL = "#2d2d2d"    # Lighter Panel Background
COLOR_TEXT = "#ffffff"     # White Text
COLOR_ACCENT = "#00ff88"   # Neon Green
COLOR_BAR = "#3498db"      # Blue for Bar Charts

class EmotionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1300x720")
        self.window.configure(bg=COLOR_BG)

        # State Variables
        self.camera_index = 0
        self.is_running = True
        self.show_logic = True
        self.show_overlay = True  # Toggle for the Scanner Overlay
        self.timestamp_ms = 0
        self.last_frame_time = 0

        # --- LOAD RESOURCES ---
        self.load_ai_resources()

        # --- GUI LAYOUT ---
        self.create_gui()

        # --- START VIDEO ---
        self.cap = cv2.VideoCapture(self.camera_index)
        self.update()

    def load_ai_resources(self):
        """Initialize MediaPipe and Keras Model"""
        print("Loading AI Models...")
        try:
            self.model = load_model(MODEL_PATH)
            print("Emotion Model Loaded.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model: {e}\nCheck path: {MODEL_PATH}")
            self.window.destroy()
            return

        # MediaPipe Setup (New API)
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

    def create_gui(self):
        # 1. Top Bar (Header & Settings)
        self.top_bar = tk.Frame(self.window, bg=COLOR_PANEL, height=50)
        self.top_bar.pack(side=tk.TOP, fill=tk.X)
        
        tk.Label(self.top_bar, text="AI Emotion Dashboard", font=("Arial", 16, "bold"), 
                 bg=COLOR_PANEL, fg=COLOR_TEXT).pack(side=tk.LEFT, padx=20, pady=10)

        # Buttons
        btn_style = {"bg": "#444", "fg": "white", "bd": 0, "padx": 10, "pady": 5}
        
        self.btn_toggle = tk.Button(self.top_bar, text="Toggle Fullscreen", command=self.toggle_view, **btn_style)
        self.btn_toggle.pack(side=tk.RIGHT, padx=10)
        
        # Settings Button
        self.btn_settings = tk.Button(self.top_bar, text="Settings", command=self.open_settings, **btn_style)
        self.btn_settings.pack(side=tk.RIGHT, padx=10)

        self.btn_cam = tk.Button(self.top_bar, text="Switch Camera", command=self.switch_camera, **btn_style)
        self.btn_cam.pack(side=tk.RIGHT, padx=10)

        tk.Button(self.top_bar, text="Logout / Exit", command=self.close_app, bg="#e74c3c", fg="white", bd=0, padx=10, pady=5).pack(side=tk.RIGHT, padx=10)

        # 2. Main Content Area
        self.main_container = tk.Frame(self.window, bg=COLOR_BG)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left: Camera Feed
        self.video_frame = tk.Label(self.main_container, bg="black")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Right: Logic Visualization Panel
        self.logic_panel = tk.Frame(self.main_container, bg=COLOR_PANEL, width=400)
        self.logic_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        self.logic_panel.pack_propagate(False)

        self.setup_logic_panel()

    def setup_logic_panel(self):
        """Builds the visualization of internal AI logic"""
        pad = 10
        
        # Title
        tk.Label(self.logic_panel, text="INTERNAL LOGIC VISUALIZER", font=("Courier", 12, "bold"), 
                 bg=COLOR_PANEL, fg=COLOR_ACCENT).pack(pady=pad)

        # 1. Preprocessing View
        tk.Label(self.logic_panel, text="Step 1: Face ROI & Preprocessing", font=("Arial", 10), 
                 bg=COLOR_PANEL, fg=COLOR_TEXT).pack(anchor="w", padx=pad)
        
        self.lbl_roi = tk.Label(self.logic_panel, bg="black", width=100, height=100)
        self.lbl_roi.pack(pady=5)
        
        tk.Label(self.logic_panel, text="Input transformed to 48x48 Grayscale", font=("Arial", 8, "italic"), 
                 bg=COLOR_PANEL, fg="#aaaaaa").pack(pady=(0, 10))

        # 2. Probability Graph
        tk.Label(self.logic_panel, text="Step 2: Neural Net Activation (Softmax)", font=("Arial", 10), 
                 bg=COLOR_PANEL, fg=COLOR_TEXT).pack(anchor="w", padx=pad)

        self.prob_bars = {}
        self.prob_frame = tk.Frame(self.logic_panel, bg=COLOR_PANEL)
        self.prob_frame.pack(fill=tk.X, padx=pad, pady=5)

        for emotion in EMOTION_LABELS:
            row = tk.Frame(self.prob_frame, bg=COLOR_PANEL)
            row.pack(fill=tk.X, pady=2)
            
            tk.Label(row, text=emotion.ljust(10), font=("Courier", 10), width=10, anchor="w",
                     bg=COLOR_PANEL, fg=COLOR_TEXT).pack(side=tk.LEFT)
            
            canvas = tk.Canvas(row, width=150, height=15, bg="#444", highlightthickness=0)
            canvas.pack(side=tk.LEFT, fill=tk.X, expand=True)
            rect = canvas.create_rectangle(0, 0, 0, 15, fill=COLOR_BAR)
            
            lbl_pct = tk.Label(row, text="0%", font=("Arial", 9), width=4, bg=COLOR_PANEL, fg="white")
            lbl_pct.pack(side=tk.RIGHT)
            
            self.prob_bars[emotion] = (canvas, rect, lbl_pct)

        # 3. Status Log
        tk.Label(self.logic_panel, text="Step 3: Decision Status", font=("Arial", 10), 
                 bg=COLOR_PANEL, fg=COLOR_TEXT).pack(anchor="w", padx=pad, pady=(20, 5))
        
        self.status_log = tk.Text(self.logic_panel, height=6, bg="#111", fg="#0f0", font=("Courier", 9), state=tk.DISABLED)
        self.status_log.pack(fill=tk.X, padx=pad)

    def log_status(self, message):
        self.status_log.config(state=tk.NORMAL)
        self.status_log.insert(tk.END, f"> {message}\n")
        self.status_log.see(tk.END)
        self.status_log.config(state=tk.DISABLED)

    def open_settings(self):
        """Opens a small pop-up window for settings"""
        settings_win = Toplevel(self.window)
        settings_win.title("Settings")
        settings_win.geometry("300x200")
        settings_win.configure(bg=COLOR_PANEL)

        tk.Label(settings_win, text="Overlay Settings", font=("Arial", 12, "bold"), bg=COLOR_PANEL, fg="white").pack(pady=10)

        # Checkbox for Overlay
        self.overlay_var = IntVar(value=1 if self.show_overlay else 0)
        chk_mesh = Checkbutton(settings_win, text="Show Face Scanner", variable=self.overlay_var, 
                               command=self.toggle_overlay, bg=COLOR_PANEL, fg="white", selectcolor="#444", activebackground=COLOR_PANEL)
        chk_mesh.pack(pady=10)

        tk.Button(settings_win, text="Close", command=settings_win.destroy, bg="#444", fg="white").pack(pady=20)

    def toggle_overlay(self):
        self.show_overlay = bool(self.overlay_var.get())

    def switch_camera(self):
        self.camera_index += 1
        if self.camera_index > 2: self.camera_index = 0
        self.cap.release()
        self.cap = cv2.VideoCapture(self.camera_index)
        self.log_status(f"Switched to Camera {self.camera_index}")

    def toggle_view(self):
        if self.show_logic:
            self.logic_panel.pack_forget()
            self.btn_toggle.config(text="Show Logic")
        else:
            self.logic_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
            self.btn_toggle.config(text="Fullscreen Cam")
        self.show_logic = not self.show_logic

    # ==========================================
    # SCANNER OVERLAY (Integrated)
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
        color = (0, 255, 136) # Neon Green
        thickness = 2

        # Corner brackets
        cv2.line(overlay, (x_min, y_min), (x_min + bracket_len, y_min), color, thickness)
        cv2.line(overlay, (x_min, y_min), (x_min, y_min + bracket_len), color, thickness)

        cv2.line(overlay, (x_max, y_min), (x_max - bracket_len, y_min), color, thickness)
        cv2.line(overlay, (x_max, y_min), (x_max, y_min + bracket_len), color, thickness)

        cv2.line(overlay, (x_min, y_max), (x_min + bracket_len, y_max), color, thickness)
        cv2.line(overlay, (x_min, y_max), (x_min, y_max - bracket_len), color, thickness)

        cv2.line(overlay, (x_max, y_max), (x_max - bracket_len, y_max), color, thickness)
        cv2.line(overlay, (x_max, y_max), (x_max, y_max - bracket_len), color, thickness)

        # Minimal key landmark dots
        key_points = [33, 263, 61, 291, 1, 199, 10]

        for idx in key_points:
            # Safety check to ensure index exists in detected points
            if idx < len(points):
                cv2.circle(overlay, points[idx], 4, (255, 255, 255), -1)

        # Clean connecting lines
        connections = [(33, 1), (263, 1), (61, 1), (291, 1), (1, 199)]

        for start, end in connections:
            if start < len(points) and end < len(points):
                cv2.line(overlay, points[start], points[end], (255, 255, 255), 1)

        # Transparency effect (Mix overlay with original frame)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def update(self):
        if not self.is_running: return

        ret, frame = self.cap.read()
        if ret:
            # 1. Convert for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # 2. Detect
            self.timestamp_ms += int((time.time() - self.last_frame_time) * 1000)
            self.last_frame_time = time.time()
            results = self.landmarker.detect_for_video(mp_image, int(time.time() * 1000))

            prediction_made = False
            top_emotion = "Waiting..."

            if results.face_landmarks:
                for face_landmarks in results.face_landmarks:
                    h, w, _ = frame.shape
                    
                    # --- DRAWING THE SCANNER OVERLAY ---
                    if self.show_overlay:
                        self.draw_scanner_overlay(frame, face_landmarks)

                    # --- LOGIC: Extract Face ---
                    x_coords = [lm.x for lm in face_landmarks]
                    y_coords = [lm.y for lm in face_landmarks]
                    
                    x_min, x_max = int(min(x_coords) * w) - 20, int(max(x_coords) * w) + 20
                    y_min, y_max = int(min(y_coords) * h) - 20, int(max(y_coords) * h) + 20
                    x_min, x_max = max(0, x_min), min(w, x_max)
                    y_min, y_max = max(0, y_min), min(h, y_max)

                    face = frame[y_min:y_max, x_min:x_max]

                    if face.size != 0:
                        # --- PREPROCESSING ---
                        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        face_resized = cv2.resize(face_gray, (48, 48))
                        face_input = np.reshape(face_resized / 255.0, (1, 48, 48, 1))

                        # --- MODEL PREDICTION ---
                        prediction = self.model.predict(face_input, verbose=0)[0]
                        max_index = np.argmax(prediction)
                        top_emotion = EMOTION_LABELS[max_index]
                        prediction_made = True

                        # Draw Text
                        cv2.putText(frame, f"Emotion: {top_emotion}", (x_min, y_min - 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 136), 2)
                        
                        # --- UPDATE LOGIC PANEL ---
                        if self.show_logic:
                            roi_display = cv2.resize(face_gray, (100, 100))
                            roi_img = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(roi_display))
                            self.lbl_roi.configure(image=roi_img)
                            self.lbl_roi.image = roi_img

                            for i, emotion in enumerate(EMOTION_LABELS):
                                score = prediction[i]
                                canvas, rect, lbl = self.prob_bars[emotion]
                                bar_width = int(score * 150)
                                color = COLOR_ACCENT if i == max_index else COLOR_BAR
                                canvas.coords(rect, 0, 0, bar_width, 15)
                                canvas.itemconfig(rect, fill=color)
                                lbl.config(text=f"{int(score*100)}%")

            if prediction_made and int(time.time()) % 2 == 0:
                self.log_status(f"Detected: {top_emotion}")

            # 3. Convert Frame for Tkinter
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(img)
            imgtk = PIL.ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.configure(image=imgtk)

        self.window.after(10, self.update)

    def close_app(self):
        self.is_running = False
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root, "Advance Emotion Recognition System")
    root.mainloop()