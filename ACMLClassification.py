import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import geodesic
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

data = pd.read_csv('student_habits_performance.csv')
# data

# Data Exploration

missing_data = data.isnull().sum()
# print("Missing data per column:")
# print(missing_data)

#  Check the percentage of missing data
missing_percentage = (data.isnull().sum() / len(data)) * 100
# print("Percentage of missing data per column:")
# print(missing_percentage)

# #  Visualize missing data
# plt.figure(figsize=(12, 8))
# sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
# plt.title('Missing Data Heatmap')
# plt.show()

#  Descriptive statistics for numerical columns
# print("Descriptive statistics for numerical columns:")
# print(data.describe())


#  Correlation matrix for numerical features
numerical_data = data.select_dtypes(include=[np.number])

# Calculate correlation matrix for numerical features
correlation_matrix = numerical_data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
# plt.show()

#  Check for duplicate rows
duplicates = data.duplicated().sum()
# print(f'Number of duplicate rows: {duplicates}')
# do outliers, feature importance

# fill missing values in the column parental_education_level using mode() which returns the most frequent (common) value in the column
data['parental_education_level'].fillna(data['parental_education_level'].mode()[0], inplace=True)
missing_data = data.isnull().sum()
# print("Missing data per column:")
# print(missing_data)

# classification problem

def categorize_performance(score):
    if score <= 50:
        return "Low"
    elif score <= 75:
        return "Medium"
    else:
        return "High"

data["performance"] = data["exam_score"].apply(categorize_performance)
# data


# Label encode categorical features (numerical values)
categorical_cols = ['gender', 'part_time_job', 'diet_quality', 'parental_education_level',
                    'internet_quality', 'extracurricular_participation']

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Encode the target variable 'performance'
target_le = LabelEncoder()
data["performance"] = target_le.fit_transform(data["performance"])  # Low=0, Medium=1, High=2
# data


from sklearn.model_selection import train_test_split

# Drop non-numeric/non-useful columns
data = data.drop(columns=["student_id", "exam_score"]) 

# Separate Features and Labels
X = data.drop("performance", axis=1)
y = data["performance"]



# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert labels to one-hot encoding
y_encoded = to_categorical(y)

# Train-validation-test split
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
X_scaled

import tensorflow as tf
# Defining the Model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')  # 3 classes: Low, Medium, High
])
# Dropout layer: Helps prevent overfitting by randomly disabling some neurons during training.
# optimizer='adam': adjusts the learning rate and other parameters during training to help the model learn more efficiently.
# loss='categorical_crossentropy': loss function used for multi-class classification problems
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# model.summary()

history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=30,
                    batch_size=32)


loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")


import matplotlib.pyplot as plt

# Plot Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot Loss
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

from sklearn.metrics import mean_squared_error, r2_score

# Evaluate the model on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE) and R2 Score
mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
# print(f'R2 Score: {r2}')

from sklearn.dummy import DummyRegressor

# Train a baseline model (predicting the mean)
baseline_model = DummyRegressor(strategy='mean')
baseline_model.fit(X_train, y_train)
y_baseline_pred = baseline_model.predict(X_test)

# Calculate baseline performance
baseline_mse = mean_squared_error(y_test, y_baseline_pred)
print(f'Baseline Model MSE: {baseline_mse}')