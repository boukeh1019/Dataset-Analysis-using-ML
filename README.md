# 📘 Student Exam Score Prediction using Neural Networks and Random Forests

This project implements a machine learning pipeline to predict student exam scores based on behavioral, lifestyle, and demographic data. A **Multilayer Perceptron (MLP)** model is used for prediction, and a **Random Forest** is applied for feature importance analysis.

---

## 📊 Dataset

The dataset (`student_habits_performance.csv`) contains the following columns:

- **Numerical Features**: `age`, `studyHoursPerDay`, `SocialMediaHours`, `NetflixHours`, `AttendancePercentage`, `SleepHours`, `ExerciseFrequency`
- **Categorical Features**: `gender`, `part_time_job`, `parental_education_level`, `extracurricular_participation`, `diet_quality`, `internet_quality`
- **Target Variable**: `exam_score` (numeric value from 0 to 100)

---

## 🎯 Project Objectives

- **Train a predictive model** to estimate students' exam scores.
- **Preprocess data** using scaling and one-hot encoding.
- **Evaluate model performance** using MAE and MSE.
- **Visualize model learning curves**.
- **Analyze feature importance** using a Random Forest model.

---

## 🧠 Model Overview

### 💡 Main Model: Multilayer Perceptron (MLP)
- Input layer size = number of preprocessed features
- Hidden Layers: `[128 (sigmoid), Dropout(0.4), 64 (ReLU), Dropout(0.3)]`
- Output Layer: `1 neuron (regression)`
- Optimizer: `Adam(learning_rate=0.001)`
- Loss Function: `Huber()` — for robustness to outliers
- Metrics: `Mean Absolute Error (MAE)`

### 🛑 EarlyStopping & ModelCheckpoint
- Stops training early if validation loss doesn't improve for 10 epochs
- Automatically saves the best model during training as `best_model.keras`

---

## 📈 Evaluation

- **Final performance** on test set:
  - `Test MAE ≈ 4.165` → Model is off by ~4 marks on average
  - `Test MSE ≈ 25.886` → Indicates low error variance
- **Training vs. Validation Curves** plotted to verify model convergence
- **Feature Importance Plot** identifies key drivers of performance (e.g., `studyHoursPerDay`, `AttendancePercentage`)

---

## 📦 Libraries Used

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `sklearn` (for preprocessing, splitting, and Random Forests)
- `tensorflow.keras` (for model building and training)

---

## 🛠️ How to Run

1. Make sure your environment has the following installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
2. python student_exam_predictor.py
3. Outputs:
    Trained MLP model saved as final_model.keras
    Best model checkpoint as best_model.keras  
    Evaluation metrics printed
    Feature importance and learning curves plotted

📌 Notes
    The model uses a ColumnTransformer to scale numeric features and one-hot encode categorical ones.
    You can easily swap StandardScaler for MinMaxScaler by modifying the preprocessor.
    The Random Forest is not used for prediction but only for interpretability of feature relevance.

👨‍💻 Author
This project was developed as part of an Adaptive Computation and Machine Learning (ACML) assignment focused on student performance modeling using neural networks.

📁 Files
| File Name                        | Description                    |
| -------------------------------- | ------------------------------ |
| `student_habits_performance.csv` | Input dataset                  |
| `student_exam_predictor.py`      | Main model and training script |
| `final_model.keras`              | Trained neural network model   |
| `best_model.keras`               | Best model based on validation |
