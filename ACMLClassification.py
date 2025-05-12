import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ========================
# 1. Load and Clean Dataset
# ========================
df = pd.read_csv('student_habits_performance.csv')

# Drop ID column
df.drop(columns=['student_id'], inplace=True)

# Fill missing values in parental education with the mode
df['parental_education_level'].fillna(df['parental_education_level'].mode()[0], inplace=True)

# ============================
# 2. Convert Exam Score to Grades
# ============================
def categorize_performance(score):
    if score <= 49:
        return "F"
    elif score <= 59:
        return "D"
    elif score <= 69:
        return "C"
    elif score <= 74:
        return "B"
    else:
        return "A"

df['performance'] = df['exam_score'].apply(categorize_performance)

# ========================
# 3. Encode Categorical Features
# ========================
categorical_cols = [
    'gender', 'part_time_job', 'diet_quality',
    'parental_education_level', 'internet_quality',
    'extracurricular_participation'
]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode the target variable
target_le = LabelEncoder()
df['performance'] = target_le.fit_transform(df['performance'])  # A–F to 0–4

# Drop continuous target
df.drop(columns=['exam_score'], inplace=True)

# ========================
# 4. Split and Normalize Data
# ========================
X = df.drop(columns=['performance'])
y = df['performance']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_encoded = to_categorical(y)

# Train-val-test split (70/15/15)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ========================
# 5. Build and Train Model
# ========================
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(y_encoded.shape[1], activation='softmax')  # Number of classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    verbose=1
)

# ========================
# 6. Evaluate Performance
# ========================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.2f}")

# ========================
# 7. Plot Training Curves
# ========================
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# ========================
# 8. Optional MSE (Not usually used for classification)
# ========================
y_pred_probs = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred_probs)
print(f"Mean Squared Error (for reference only): {mse:.3f}")


# Convert predictions from probabilities to class indices
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# =============================
# 9. Confusion Matrix & Metrics
# =============================

# Confusion Matrix
cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_le.classes_)

plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.grid(False)
plt.tight_layout()
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=target_le.classes_))
