import tkinter as tk
from tkinter import messagebox
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
MODEL_PATH = "models/emotion_model_improved.keras"
TASK_PATH = "face_landmarker.task"
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Visual Colors
COLOR_BG = "#1e1e1e"       # Dark Gray Background
COLOR_PANEL = "#2d2d2d"    # Lighter Panel Background
COLOR_TEXT = "#ffffff"     # White Text
COLOR_ACCENT = "#00ff88"   # Neon Green for Overlay
COLOR_BAR = "#3498db"      # Blue for Bar Charts

# --- HARDCODED FACE CONTOURS (To avoid import errors) ---
# These are the standard MediaPipe indices for the smooth lines of the face
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYEBROW = [276, 283, 300, 293, 334, 296, 336, 285]
RIGHT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105]

ALL_CONTOURS = [FACE_OVAL, LIPS, LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW]

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

        # Settings Buttons
        btn_style = {"bg": "#444", "fg": "white", "bd": 0, "padx": 10, "pady": 5}
        
        self.btn_toggle = tk.Button(self.top_bar, text="Toggle Fullscreen", command=self.toggle_view, **btn_style)
        self.btn_toggle.pack(side=tk.RIGHT, padx=10)

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
        self.logic_panel.pack_propagate(False) # Don't shrink

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
        
        self.lbl_roi = tk.Label(self.logic_panel, bg="black", width=100, height=100) # Placeholder for face image
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
            
            # Progress bar style custom canvas
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
        """Updates the retro-style text log"""
        self.status_log.config(state=tk.NORMAL)
        self.status_log.insert(tk.END, f"> {message}\n")
        self.status_log.see(tk.END)
        self.status_log.config(state=tk.DISABLED)

    def switch_camera(self):
        self.camera_index += 1
        if self.camera_index > 2: self.camera_index = 0
        
        self.cap.release()
        self.cap = cv2.VideoCapture(self.camera_index)
        self.log_status(f"Switched to Camera Index {self.camera_index}")

    def toggle_view(self):
        if self.show_logic:
            self.logic_panel.pack_forget()
            self.btn_toggle.config(text="Show Logic")
        else:
            self.logic_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
            self.btn_toggle.config(text="Fullscreen Cam")
        self.show_logic = not self.show_logic

    def draw_styled_landmarks(self, image, landmarks):
        """Draws smooth lines connecting specific face parts manually"""
        h, w, _ = image.shape
        
        # Create a list of (x, y) tuples for all landmarks
        points = []
        for lm in landmarks:
            points.append((int(lm.x * w), int(lm.y * h)))

        # Draw lines for each contour
        for contour in ALL_CONTOURS:
            for i in range(len(contour) - 1):
                start_idx = contour[i]
                end_idx = contour[i+1]
                
                # Check bounds to avoid index errors
                if start_idx < len(points) and end_idx < len(points):
                    cv2.line(image, points[start_idx], points[end_idx], (0, 255, 136), 1, cv2.LINE_AA)
            
            # Close the loop for the oval and lips
            if contour == FACE_OVAL or contour == LIPS or contour == LEFT_EYE or contour == RIGHT_EYE:
                start_idx = contour[-1]
                end_idx = contour[0]
                if start_idx < len(points) and end_idx < len(points):
                     cv2.line(image, points[start_idx], points[end_idx], (0, 255, 136), 1, cv2.LINE_AA)

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
            
            # Using current time ensures live video works better with the new API
            results = self.landmarker.detect_for_video(mp_image, int(time.time() * 1000))

            prediction_made = False
            top_emotion = "Waiting..."

            if results.face_landmarks:
                for face_landmarks in results.face_landmarks:
                    h, w, _ = frame.shape
                    
                    # --- DRAWING (The Good Graphics) ---
                    self.draw_styled_landmarks(frame, face_landmarks)

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
                        
                        face_input = face_resized / 255.0
                        face_input = np.reshape(face_input, (1, 48, 48, 1))

                        # --- MODEL PREDICTION ---
                        prediction = self.model.predict(face_input, verbose=0)[0]
                        max_index = np.argmax(prediction)
                        top_emotion = EMOTION_LABELS[max_index]
                        prediction_made = True

                        # Draw Text on Screen
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

# --- MAIN ENTRY POINT ---
if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root, "Advance Emotion Recognition System")
    root.mainloop()