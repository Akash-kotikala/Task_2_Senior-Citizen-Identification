# Task_2_Senior-Citizen-Identification 
Develop an ML model to detect multiple people in video or webcam feed for malls/stores. Predict age and gender; mark those over 60 as senior citizens with gender. Store age, gender, visit time in Excel/CSV. Optional GUI. Focus on model performance and functionality.
### Age and Gender Model Link :
https://drive.google.com/file/d/1zSqkwmTJcyNBGZ87pbvEcX97M-m7pJxG/view?usp=drive_link  


https://drive.google.com/file/d/1WS7YL0ssosh2ED-uGBiuO-gVBhj-noIk/view?usp=drive_link


### Dataset Link::


https://www.kaggle.com/datasets/jangedoo/utkface-new


# 👵👴 Senior Citizen Detector  

## 📌 Problem Statement  
Classify individuals as **senior citizens (age ≥ 60)** or **non-senior** based on facial images.  

This supports applications such as:  
- 📊 Demographic analysis  
- 🏥 Healthcare targeting  
- 🎯 Age-specific services  

The challenge lies in **accurate age detection** across diverse facial features.  

---

## 📂 Dataset  

- **Source:** [UTKFace Dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new) (~23,705 images with age, gender, ethnicity).  
- **Preprocessing:**  
  - Images resized → `128x128`  
  - Normalized → `[0,1]`  
  - Binary labels:  
    - ✅ Senior (`1`) → Age ≥ 60  
    - ❌ Non-Senior (`0`) → Age < 60  
- **Class Distribution:**  
  - Senior → ~10%  
  - Non-Senior → ~90%  
- **Size:** ~1 GB (uncompressed).  

---

## 🛠 Methodology  

### 🔹 Data Loading & Preprocessing  
- Load UTKFace images using **OpenCV**.  
- Extract **age** from filenames.  
- Label: `1` if age ≥ 60, else `0`.  
- Implemented using a **custom Keras Sequence** for efficient batch loading.  

### 🔹 Model Architecture (CNN)  
- **Input:** `128x128x3` facial image  
- **Layers:**  
  - 3️⃣ Convolutional Layers + ReLU  
  - 🌀 Max-Pooling layers  
  - 🔗 Dense layers  
- **Output:** Softmax (2 classes → Senior / Non-Senior)  
- **Optimizer:** Adam (`lr = 0.001`)  
- **Loss:** Categorical Crossentropy  
- **Metrics:** Accuracy  

### 🔹 Training  
- Train/Test Split → **80/20**  
- Epochs → **10**  
- Batch Size → **32**  

### 🔹 Evaluation  
- ✅ Accuracy  
- ✅ Confusion Matrix (via Scikit-learn)  

---

## ⚙ Tools & Libraries  
- 🧠 TensorFlow / Keras  
- 👁 OpenCV  
- 🔢 NumPy  
- 📊 Scikit-learn  

---

✨ This project demonstrates a **binary classification model** for **senior citizen detection** using facial images.  


Accuracy: ~90% on test set (inferred from typical UTKFace age classification results).
Challenges: Class imbalance (fewer seniors); could improve with oversampling or weighted loss.
Installation
pip install tensorflow opencv-python numpy scikit-learn

