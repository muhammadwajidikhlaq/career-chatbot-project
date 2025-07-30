# explore_data.py

import pandas as pd

# Load the dataset
df = pd.read_csv("career_guidance_dataset.csv")

# 1. View basic structure
print(" Dataset Shape:", df.shape)  # (Rows, Columns)
# 2. Preview the first few rows
print("\n Sample Rows:")
print(df.head())

# 3. View all unique career roles
print("\n Unique Roles Count:", df['role'].nunique())
print(df['role'].unique())  # Or use df['Role'].value_counts()

# 4. Check how many questions per role (optional insight)
print("\n Questions per Role:")
print(df['role'].value_counts())

# 5. Check for missing values
print("\n Missing Values:")
print(df.isnull().sum())

# 6. Check for duplicate rows
print("\n Duplicate Rows:", df.duplicated().sum())
