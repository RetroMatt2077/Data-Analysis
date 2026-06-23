import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("VISUALIZATION EXAMPLES")
print("=" * 60)

# -----------------------------
# Create sample dataset
# -----------------------------

np.random.seed(42)

df = pd.DataFrame({
    "Name": ["Alice", "Bob", "Charlie", "Diana", "Ethan", "Fiona", "George", "Hannah"],
    "Age": [24, 30, 22, 27, 35, 29, 31, 26],
    "Score": [88, 92, 79, 95, 84, 91, 73, 89],
    "StudyHours": [4, 6, 2, 7, 5, 6, 3, 4],
    "GamesPlayed": [10, 15, 5, 20, 12, 18, 6, 9]
})

print("\nDataset:")
print(df)

# -----------------------------
# 1. Bar Chart - Scores
# -----------------------------

plt.figure()
plt.bar(df["Name"], df["Score"])
plt.title("Student Scores")
plt.xlabel("Name")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# 2. Line Chart - Study Hours Trend
# -----------------------------

plt.figure()
plt.plot(df["Name"], df["StudyHours"], marker="o")
plt.title("Study Hours Trend")
plt.xlabel("Student")
plt.ylabel("Hours Studied")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# 3. Scatter Plot - Study vs Score
# -----------------------------

plt.figure()
plt.scatter(df["StudyHours"], df["Score"])
plt.title("Study Hours vs Score")
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.tight_layout()
plt.show()

# -----------------------------
# 4. Histogram - Score Distribution
# -----------------------------

plt.figure()
plt.hist(df["Score"], bins=5, edgecolor="black")
plt.title("Score Distribution")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# -----------------------------
# 5. Pie Chart - Game Activity Share
# -----------------------------

plt.figure()
plt.pie(df["GamesPlayed"], labels=df["Name"], autopct="%1.1f%%")
plt.title("Games Played Distribution")
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Seaborn Heatmap - Correlation
# -----------------------------

plt.figure()
corr = df[["Age", "Score", "StudyHours", "GamesPlayed"]].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# -----------------------------
# 7. Seaborn Scatter with trend
# -----------------------------

plt.figure()
sns.regplot(x="StudyHours", y="Score", data=df)
plt.title("Study Hours vs Score (Trend Line)")
plt.tight_layout()
plt.show()

# -----------------------------
# Summary Stats
# -----------------------------

print("\nSummary Statistics:")
print(df.describe())

print("\nDone generating visualizations!")
print("=" * 60)
