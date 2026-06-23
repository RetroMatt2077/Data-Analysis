import pandas as pd
import numpy as np

print("=" * 60)
print("DATA CLEANING DEMO")
print("=" * 60)

# -----------------------------
# Create messy dataset
# -----------------------------

data = {
    "Name": ["Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "Bob", None],
    "Age": [24, 30, np.nan, 27, 35, 29, 30, 22],
    "Score": [88, 92, 79, None, 84, 91, 92, 75],
    "Country": ["USA", "usa", "UK", "UK ", "Canada", "canada", "USA", "UK"]
}

df = pd.DataFrame(data)

print("\nOriginal Data:")
print(df)

# -----------------------------
# Basic info
# -----------------------------

print("\nShape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# -----------------------------
# Remove duplicates
# -----------------------------

df = df.drop_duplicates()

print("\nAfter removing duplicates:")
print(df)

# -----------------------------
# Handle missing values
# -----------------------------

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Score"] = df["Score"].fillna(df["Score"].median())

df = df.dropna(subset=["Name"])

print("\nAfter handling missing values:")
print(df)

# -----------------------------
# Clean text data
# -----------------------------

df["Country"] = df["Country"].str.strip().str.upper()

print("\nAfter cleaning Country column:")
print(df)

# -----------------------------
# Convert data types
# -----------------------------

df["Age"] = df["Age"].astype(int)

print("\nData types:")
print(df.dtypes)

# -----------------------------
# Feature engineering
# -----------------------------

df["Passed"] = df["Score"] >= 80

def grade(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    else:
        return "D"

df["Grade"] = df["Score"].apply(grade)

print("\nWith new features:")
print(df)

# -----------------------------
# Outlier detection
# -----------------------------

mean = df["Score"].mean()
std = df["Score"].std()

outliers = df[
    (df["Score"] > mean + 2 * std) |
    (df["Score"] < mean - 2 * std)
]

print("\nOutliers detected:")
print(outliers)

# -----------------------------
# Statistics
# -----------------------------

print("\nStatistics:")
print("Average Score:", df["Score"].mean())
print("Max Score:", df["Score"].max())
print("Min Score:", df["Score"].min())

print("\nGrade distribution:")
print(df["Grade"].value_counts())

# -----------------------------
# Grouping
# -----------------------------

grouped = df.groupby("Country")["Score"].mean()

print("\nAverage Score by Country:")
print(grouped)

# -----------------------------
# Sorting
# -----------------------------

df_sorted = df.sort_values(by="Score", ascending=False)

print("\nSorted by Score:")
print(df_sorted)

# -----------------------------
# Save cleaned data
# -----------------------------

df.to_csv("cleaned_data.csv", index=False)

print("\nSaved to cleaned_data.csv")

# -----------------------------
# Load back and verify
# -----------------------------

loaded = pd.read_csv("cleaned_data.csv")

print("\nLoaded cleaned dataset:")
print(loaded)

print("\nDONE - Data cleaning complete!")
print("=" * 60)
