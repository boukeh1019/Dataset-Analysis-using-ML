import pandas as pd

# Load dataset
df = pd.read_csv('student_habits_performance.csv')

# Define function to categorize performance
def categorize_grade(score):
    if score <= 49:
        return 'F'
    elif score <= 59:
        return 'D'
    elif score <= 69:
        return 'C'
    elif score <= 74:
        return 'B'
    else:
        return 'A'

# Create a new column for grade
df['grade'] = df['exam_score'].apply(categorize_grade)

# Split and save to CSV by grade
for grade in ['A', 'B', 'C', 'D', 'F']:
    grade_df = df[df['grade'] == grade]
    grade_df.to_csv(f'students_grade_{grade}.csv', index=False)

print("CSV files created: students_grade_A.csv through students_grade_F.csv")
