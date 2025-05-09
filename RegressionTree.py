import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

## Creating the regression tree
tree = DecisionTreeRegressor(max_depth=15, random_state=42)
tree.fit(X_train_processed, y_train)

## evaluation part of the tree
y_pred = tree.predict(X_test_processed)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"\n[Regression tree]")
print(f"Test MAE: {mae:.3f}")
print(f"Test MSE: {mse:.3f}")


##Graphs part
feature_names = numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=feature_names, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Regressor (Max Depth = 5)")
plt.tight_layout()
plt.show()