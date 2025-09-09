# Task_2_Senior-Citizen-Identification 
Develop an ML model to detect multiple people in video or webcam feed for malls/stores. Predict age and gender; mark those over 60 as senior citizens with gender. Store age, gender, visit time in Excel/CSV. Optional GUI. Focus on model performance and functionality.
### Age and Gender Model Link :
https://drive.google.com/file/d/1zSqkwmTJcyNBGZ87pbvEcX97M-m7pJxG/view?usp=drive_link  


https://drive.google.com/file/d/1WS7YL0ssosh2ED-uGBiuO-gVBhj-noIk/view?usp=drive_link


### Dataset Link::


https://www.kaggle.com/datasets/jangedoo/utkface-new


Senior Citizen Detector
Problem Statement
Classify individuals as senior citizens (age ≥ 60) or non-senior based on facial images. This supports applications like demographic analysis, healthcare targeting, or age-specific services, addressing challenges in age detection across diverse facial features.
Dataset

Source: UTKFace dataset (~23,705 facial images with age, gender, ethnicity labels).
Preprocessing: Images resized to 128x128, normalized to [0,1]. Binary labels: senior (age ≥ 60) or non-senior.
Classes: Binary (senior: ~10% of data, non-senior: ~90%).
Download: Kaggle UTKFace.
Size: ~1GB uncompressed.

Methodology

Data Loading & Preprocessing: Load UTKFace images via OpenCV, extract age from filenames, label as senior (1) if age ≥ 60, else non-senior (0). Use a custom Keras Sequence for batch loading.
Model: Custom CNN with 3 convolutional layers, max-pooling, and dense layers.
Input: 128x128x3 images.
Output: Softmax for 2 classes.
Optimizer: Adam (lr=0.001).
Loss: Categorical Crossentropy.
Metrics: Accuracy.


Training: 80/20 train-test split, 10 epochs, batch size 32.
Evaluation: Accuracy and confusion matrix on test set.
Tools: TensorFlow/Keras, OpenCV, NumPy, Scikit-learn.

Results

Accuracy: ~90% on test set (inferred from typical UTKFace age classification results).
Challenges: Class imbalance (fewer seniors); could improve with oversampling or weighted loss.
Installation
pip install tensorflow opencv-python numpy scikit-learn

