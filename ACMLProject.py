import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.losses import Huber


df = pd.read_csv('student_habits_performance.csv')

# Drop student_ID column
df = df.drop(columns=['student_id'])

# Separate features and target
X = df.drop(columns=['exam_score']) #features
y = df['exam_score'] #target

# Define feature types
# Categorical features are the one's we'll use one-shot encoding for
# numerical features are the ones will normalize

categorical_features = [
    'gender',
    'part_time_job',
    'parental_education_level',
    'extracurricular_participation',
    'diet_quality',
    'internet_quality'
] 

numerical_features = [col for col in X.columns if col not in categorical_features]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

#refining features for processing
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# X_preprocessed = preprocessor.fit_transform(X)


# cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
# all_feature_names = numerical_features + list(cat_feature_names)

# X_preprocessed_df = pd.DataFrame(X_preprocessed.toarray() if hasattr(X_preprocessed, "toarray") else X_preprocessed,
#                                  columns=all_feature_names)


# X_preprocessed_df.to_csv('newdata1.csv', index=False)
# print(X_preprocessed_df.head())

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

input_dim = X_train_processed.shape[1]
model = Sequential([
    Dense(128, activation='relu', input_dim=input_dim),
    Dropout(0.4),  # increased dropout for stronger regularization
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1)  # output for regression
])

# model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(), metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)

history = model.fit(
    X_train_processed, y_train,
    validation_data=(X_val_processed, y_val),
    epochs=200,
    batch_size=32,  # adjusted for efficiency
    callbacks=[early_stop, checkpoint],
    verbose=1
)

test_loss, test_mae = model.evaluate(X_test_processed, y_test, verbose=0)
model.save("final_model.keras")  # Save in native format
print(f"\nTest MAE: {test_mae:.3f}, Test MSE: {test_loss:.3f}")

# plt.figure(figsize=(14, 5))

# # Plot Loss
# plt.subplot(1, 2, 1)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss (MSE)')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)

# # Plot MAE
# plt.subplot(1, 2, 2)
# plt.plot(history.history['mae'], label='Training MAE')
# plt.plot(history.history['val_mae'], label='Validation MAE')
# plt.title('Model Mean Absolute Error')
# plt.xlabel('Epoch')
# plt.ylabel('MAE')
# plt.legend()
# plt.grid(True)


# # Train Random Forest on full data (for interpretability)
# X_all_processed = preprocessor.fit_transform(X)
# rf = RandomForestRegressor(n_estimators=100, random_state=42)
# rf.fit(X_all_processed, y)

# # Get feature names
# cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
# all_feature_names = numerical_features + list(cat_feature_names)

# # Plot top features
# importances = rf.feature_importances_
# indices = np.argsort(importances)[::-1]
# top_n = 10

# plt.figure(figsize=(10, 6))
# sns.barplot(x=importances[indices][:top_n], y=np.array(all_feature_names)[indices][:top_n])
# plt.title("Top 10 Feature Importances (Random Forest)")
# plt.xlabel("Importance Score")
# plt.ylabel("Feature")
# plt.tight_layout()
# plt.show()
