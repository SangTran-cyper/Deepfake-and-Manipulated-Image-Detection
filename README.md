# Deepfake and Manipulated Image Detection 

### 📝 Project Description & Objective
This project implements a hybrid computer vision and deep learning system built to automatically detect and classify fraudulent videos or images (Deepfakes). The system focuses on identifying digital manipulation artifacts across both spatial (frame-by-frame) and temporal (video sequence) dimensions.

---

### 🛠️ Technologies
- **Core Language:** `Python`
- **Deep Learning Frameworks:** `TensorFlow / Keras`
- **Model Backbone & Recurrent Layer:** `Xception` (Pre-trained), `LSTM`
- **Image Processing & Face Detection:** `OpenCV`, `Dlib (HOG Face Detector)`
- **Infrastructure:** `Kaggle Notebooks (NVIDIA Tesla P100 GPU)`

---

### ✨ Core Features
- **Spatial-Temporal Hybrid Network (LRCN):** Leverages the feature-extraction power of a deep CNN (Xception) wrapper inside a `TimeDistributed` layer, combined with an `LSTM` network to capture dynamic biological inconsistencies (e.g., irregular blinking, moving artifacts).
- **Conservative Face Cropping:** Utilizes a Dlib HOG face detector modified with a strict **1.3 padding factor** to catch blending and alignment artifacts often left behind at the outer facial boundaries.
- **Imbalance Mitigation:** Built-in integration of automated `Class Weights` calculated from the training set to prevent the loss function from being heavily biased toward real video classes.

---

### ⚙️ The Process
1. **Data Ingestion:** Processed a subset of 400 videos (200 Real / 200 Fake) from the benchmark **FaceForensics++** dataset.
2. **Frame Sampling & Preprocessing:** Divided each video into equal segments, extracted 15 frames per video, localized/cropped faces with expanded padding, and normalized pixel values to $[0, 1]$.
3. **Two-Stage Fine-Tuning:** 
   - *Phase 1:* Froze the Xception base network and trained only the top LSTM (128 units) and Dense (64 units) layers with a learning rate of $10^{-4}$ for 20 epochs.
   - *Phase 2:* Unfroze the top layers of Xception (keeping Batch Normalization layers frozen for stability) and fine-tuned the entire network with an extremely low learning rate of $10^{-6}$ for 40 epochs.
4. **Evaluation:** Generated comprehensive visualization metrics including Training History (Loss/Accuracy curves), Confusion Matrix, and ROC Curves on a stratified 15% Test Set.

---

### 🧠 What I Learned
- **CNN-RNN Architectural Integration:** Gained practical insight into how a model processes spatial video frames independently before feeding sequentially dependent features into recurrent hidden states.
- **Domain-Specific Preprocessing:** Realized that simple bounding-box crops often cut out the exact edge-blending anomalies created by generative models (GANs), and how minor hyperparameter shifts like expanding face padding can save critical features.
- **Safety-First Metrics (Recall Focus):** Learned that in security-sensitive tasks like spoofing detection, a high Recall rate (83%) is heavily favored over high Precision, as missing a fake video is far more dangerous than manually double-checking a false alarm.

---

### 🔮 How Can I Improve the Project?
- **Adaptive Frame Sampling:** Replace fixed 15-frame static slicing with an adaptive sampling strategy to capture transient artifacts that may appear for only a few milliseconds.
- **Real-Time API Framework:** Port the model out of the experimental notebook environment and deploy it into a high-performance **FastAPI** service containerized with **Docker** for production serving.
- **Attention Mechanisms:** Integrate Spatial and Temporal Attention layers to force the model to explicitly weight highly volatile facial areas like the eyes and lips.

---
