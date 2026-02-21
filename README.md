# EmoDetect-Pro
Emotion Recognition from Face is a Computer Vision and Deep Learning project that detects human emotions from facial expressions using a Convolutional Neural Network (CNN).

## Explainable Emotion Recognition from Face using CNN

An advanced Explainable AI (XAI) based Emotion Recognition System that detects human emotions from facial expressions using Deep Learning and visualizes internal CNN operations.

This project is not just a classifier — it is a complete educational + deployment-ready AI system.

---

# 📌 Project Overview

This system:

- Detects human emotions from facial images
- Uses Convolutional Neural Networks (CNN)
- Works as:
  - 🌐 Web Application (Flask)
  - 🖥 Desktop Application (Tkinter)
- Visualizes:
  - Convolution Feature Maps
  - Pooling Outputs
  - Softmax Probability Distribution
- Supports Real-Time Webcam Emotion Detection

---

# 🧠 Technical Domain

Artificial Intelligence  
→ Machine Learning  
→ Deep Learning  
→ Convolutional Neural Networks  
→ Computer Vision  
→ Emotion Recognition  

---

# 📂 Dataset

Dataset Used: **FER2013**

### Dataset Properties:
- 48x48 grayscale facial images
- 7 emotion classes:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral

### Preprocessing Steps:
- Convert pixel strings to image arrays
- Reshape to (48, 48, 1)
- Normalize pixel values (divide by 255)
- One-hot encode labels
- Handle class imbalance (if required)

---

# 🏗 System Architecture

```
Input Image / Webcam
        ↓
Face Detection (OpenCV Haar Cascade)
        ↓
Crop Face Region
        ↓
Resize to 48x48
        ↓
Convert to Grayscale
        ↓
Normalize (0–1)
        ↓
CNN Model
        ↓
Extract Intermediate Layer Outputs
        ↓
Visualization Dashboard
        ↓
Final Emotion Prediction
```

---

# 🧩 CNN Model Architecture

```
Layer 1:
Conv2D (32 filters, 3x3) → ReLU → MaxPooling (2x2)

Layer 2:
Conv2D (64 filters, 3x3) → ReLU → MaxPooling (2x2)

Layer 3:
Conv2D (128 filters, 3x3) → ReLU → MaxPooling (2x2)

Flatten

Dense (128) → ReLU
Dropout (0.5)

Output Layer:
Dense (7) → Softmax
```

### Model Compilation
- Loss: Categorical Crossentropy
- Optimizer: Adam
- Metric: Accuracy

### Expected Accuracy
- Basic CNN: 65–70%
- Improved CNN: 70–75%
- Transfer Learning: 75–85%

---

# 📊 Explainable AI Features

This project visualizes:

1. Convolution Feature Maps  
2. Pooling Outputs  
3. Softmax Probability Bar Graph  
4. Intermediate Layer Activations  

This makes the system educational and interpretable.

---

# 🚀 Features

✅ Emotion Detection from Images  
✅ Real-Time Webcam Detection  
✅ CNN Internal Visualization  
✅ Web-based Interface (Flask)  
✅ Desktop GUI (Tkinter)  
✅ Model Evaluation Metrics  
✅ Ready for Deployment  

---

# 🗂 Project Structure

```
emotion-vision-system/
│
├── dataset/
├── models/
│   └── emotion_model.h5
│
├── static/
├── templates/
│   └── index.html
│
├── train.py
├── evaluate.py
├── visualize.py
├── real_time.py
├── app.py
├── index.py
├── utils.py
├── requirements.txt
└── README.md
```

---

# 🛠 Technologies Used

| Category | Tools |
|----------|--------|
| Programming | Python |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Web Framework | Flask |
| Desktop GUI | Tkinter |

---

# 🔬 Model Development Process

## 1️⃣ Data Preprocessing
- Load FER2013 dataset
- Convert pixel strings to arrays
- Normalize pixel values
- Reshape images
- Split into train/test sets

## 2️⃣ Model Training
- Epochs: 30–50
- Batch Size: 32
- Validation Split used
- Save trained model to `models/emotion_model.h5`

## 3️⃣ Model Evaluation
- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1 Score

## 4️⃣ Real-Time Emotion Detection
- Capture webcam feed
- Detect face using OpenCV
- Preprocess face image
- Predict emotion
- Display emotion label on screen

---

# 📈 Advanced Improvements

- Data Augmentation (Rotation, Flip, Zoom)
- Hyperparameter Tuning
- Transfer Learning (MobileNet, ResNet)
- Model Comparison Experiments
- Cloud Deployment (AWS / Render)

---

# 📅 Development Roadmap

### Phase 1 – Dataset Mastery
- Load and preprocess FER2013
- Perform Exploratory Data Analysis (EDA)

### Phase 2 – Baseline CNN
- Build first working model
- Train and evaluate

### Phase 3 – Visualization Module
- Extract intermediate layer outputs
- Display feature maps
- Plot softmax probabilities

### Phase 4 – Deployment
- Build Flask Web App
- Build Tkinter Desktop App
- Integrate Webcam Detection

---

# 🎯 Interview Preparation

Be ready to answer:

- Why CNN for image tasks?
- What is overfitting?
- Why use dropout?
- Why softmax in output layer?
- What is categorical crossentropy?
- How did you improve model accuracy?
- What challenges did FER2013 dataset present?

---

# 🎓 Learning Outcomes

After completing this project, you will:

- Understand CNN deeply
- Build real-time AI pipelines
- Implement Explainable AI systems
- Train and evaluate deep learning models
- Deploy AI applications
- Create a strong resume-level AI project

---

# 💡 Project Tagline

"Building an Explainable AI Emotion Recognition System with Real-Time CNN Visualization."
