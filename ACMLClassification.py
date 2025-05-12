import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ========================
# 1. Load and Clean Dataset
# ========================
df = pd.read_csv('student_habits_performance.csv')
df.drop(columns=['student_id'], inplace=True)
# df['parental_education_level'].fillna(df['parental_education_level'].mode()[0], inplace=True)

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

target_le = LabelEncoder()
df['performance'] = target_le.fit_transform(df['performance'])  # 'A'–'F' to 0–4
df.drop(columns=['exam_score'], inplace=True)

# ========================
# 4. Train/Val/Test Split
# ========================
X = df.drop(columns=['performance'])
y = df['performance']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_encoded = to_categorical(y)

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.5, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ========================
# 5. Define & Compile Model
# ========================
model = Sequential([
    Dense(64, activation='relu', kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(32, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(y_encoded.shape[1], activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ========================
# 6. EarlyStopping Callback
# ========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# ========================
# 7. Train the Model
# ========================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ========================
# 8. Evaluate and Plot
# ========================
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {accuracy:.2f}")

# Plot Accuracy and Loss
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

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
# 9. Confusion Matrix & Report
# ========================
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_classes, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_le.classes_)

# # plt.figure(figsize=(8, 6))
# # disp.plot(cmap='Blues', values_format='d')
# # plt.title('Confusion Matrix')
# # plt.grid(False)
# # plt.tight_layout()
# plt.show()

# print("\nClassification Report:")
# print(classification_report(y_true_classes, y_pred_classes, target_names=target_le.classes_))

# Optional MSE (not critical for classification)
mse = mean_squared_error(y_test, y_pred_probs)
print(f"Mean Squared Error (for reference): {mse:.3f}")
