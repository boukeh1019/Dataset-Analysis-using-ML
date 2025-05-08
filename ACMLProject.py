import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ========================
# 1. Load & Prepare Dataset
# ========================
df = pd.read_csv('student_habits_performance.csv')

# Drop non-predictive ID column
df = df.drop(columns=['student_id'])

# Separate features and target
X = df.drop(columns=['exam_score'])
y = df['exam_score']

# Define feature types
categorical_features = [
    'gender',
    'part_time_job',
    'parental_education_level',
    'extracurricular_participation',
    'diet_quality',
    'internet_quality'
]
numerical_features = [col for col in X.columns if col not in categorical_features]

# ==========================
# 2. Preprocessing pipeline
# ==========================
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# =======================
# 3. Train/Val/Test Split
# =======================
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# ======================
# 4. Transform Features
# ======================
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# ======================
# 5. Build & Tune Model
# ======================
input_dim = X_train_processed.shape[1]
model = Sequential([
    Dense(128, activation='relu', input_dim=input_dim),
    Dropout(0.4),  # increased dropout for stronger regularization
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1)  # output for regression
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# ========================
# 6. Callbacks for tuning
# ========================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

# =====================
# 7. Train the model
# =====================
history = model.fit(
    X_train_processed, y_train,
    validation_data=(X_val_processed, y_val),
    epochs=100,
    batch_size=32,  # adjusted for efficiency
    callbacks=[early_stop, checkpoint],
    verbose=1
)

# =====================
# 8. Evaluate & Save
# =====================
test_loss, test_mae = model.evaluate(X_test_processed, y_test, verbose=0)
model.save("final_model.keras")  # Save in native format
print(f"\nTest MAE: {test_mae:.3f}, Test MSE: {test_loss:.3f}")

# ============================
# 9. Plot Training History
# ============================
plt.figure(figsize=(14, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss (MSE)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
