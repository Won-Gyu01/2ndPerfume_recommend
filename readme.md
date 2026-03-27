#  Emotion-Based Personalized Perfume Recommendation Kiosk

> **An AI-powered smart kiosk system that recommends perfumes based on real-time facial emotion recognition.**
> *Presented an Ensemble Recommendation Algorithm combining Random Forest and Cosine Similarity.*

##  Project Overview
This project is an offline smart kiosk system designed to provide highly personalized perfume recommendations. Unlike traditional recommendation systems that rely solely on user input or purchase history, this kiosk utilizes a **webcam to analyze the user's real-time facial expressions**, mapping their current emotional state to specific perfume characteristics.

The system was developed with a full-stack AI approach: training a custom computer vision model, designing a hybrid recommendation algorithm, and optimizing the heavy pipeline for edge deployment on a **Raspberry Pi** hardware environment.

---

##  Core Technical Highlights

### 1. Custom Emotion Detection (YOLOv8 Fine-tuning)
Instead of relying on commercial APIs, I built a custom emotion classifier.
* **Model:** Fine-tuned `YOLOv8` for facial expression recognition.
* **Classes:** Trained on a custom dataset to classify 5 core emotions: `Anger`, `Fear`, `Happy`, `Neutral`, and `Sad`.
* **Details:** The training logs and model evaluation metrics can be found in `research/face_train.ipynb`.

### 2. Ensemble Recommendation Algorithm
To overcome the limitations of simple keyword matching, I designed an advanced ensemble recommendation logic, which was published as an academic paper.
* **Random Forest (30% Weight):** Predicts the baseline perfume rating based on user preferences.
* **Cosine Similarity (70% Weight):** Calculates the vector distance between the user's real-time emotion features (extracted via YOLOv8) and the perfume's characteristic vectors.
* **Details:** The EDA, feature engineering, and algorithm comparison tests are documented in `research/recommendation_model_evaluation.ipynb`.

### 3. Edge AI & Kiosk UI Deployment
Successfully ported the AI models and graphical interface to a resource-constrained hardware environment.
* **Hardware:** Optimized the real-time webcam frame processing to run smoothly on a **Raspberry Pi**.
* **UI/UX:** Developed a touch-friendly kiosk interface using **PyQt5**, featuring full-screen mode and interactive recommendation results.

---

##  Repository Structure

To demonstrate a structured software development lifecycle, the codebase is separated into research, modular PC testing, and final Raspberry Pi deployment versions.

* `main_raspberrypi.py`: **[Main]** The final, fully integrated and hardware-optimized kiosk execution file.
* `main_pc_version.py`: The PC environment version used for stability testing prior to hardware porting.
* `research/`: Contains Jupyter notebooks for YOLOv8 fine-tuning (`face_train.ipynb`) and algorithm design (`recommendation_model_evaluation.ipynb`).
* `pc_modular/`: Contains the modularized components (`camera.py`, `filter.py`, `ui.py`) and alternative algorithm tests (e.g., Apriori).

---

##  Tech Stack
* **AI & Machine Learning:** `YOLOv8`, `Scikit-learn (Random Forest, Cosine Similarity)`
* **Computer Vision:** `OpenCV`
* **Data Processing:** `Pandas`, `NumPy`
* **Hardware & UI:** `Raspberry Pi`, `PyQt5`
* **Language:** `Python`

---

##  Publications
* **Paper:** "A Study on Perfume Recommendation Algorithm using Ensemble Technique" (앙상블 기법을 활용한 향수 추천 알고리즘에 관한 연구)
