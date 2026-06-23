import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("WEATHER DATA ANALYSIS")
print("=" * 60)

# -----------------------------
# Create synthetic weather dataset
# -----------------------------

np.random.seed(42)

dates = pd.date_range(start="2024-01-01", periods=60)

df = pd.DataFrame({
    "Date": dates,
    "Temperature": np.random.normal(70, 10, size=60),
    "Humidity": np.random.randint(40, 90, size=60),
    "WindSpeed": np.random.randint(0, 25, size=60),
    "Rainfall": np.random.choice([0, 0, 0, 0.2, 0.5, 1.0, 2.0], size=60)
})

print("\nFirst 5 rows:")
print(df.head())

# -----------------------------
# Data inspection
# -----------------------------

print("\nData Info:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nSummary statistics:")
print(df.describe())

# -----------------------------
# Date handling
# -----------------------------

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# -----------------------------
# Temperature trend
# -----------------------------

plt.figure()
plt.plot(df["Date"], df["Temperature"])
plt.title("Temperature Over Time")
plt.xlabel("Date")
plt.ylabel("Temperature (°F)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# Humidity trend
# -----------------------------

plt.figure()
plt.plot(df["Date"], df["Humidity"], color="blue")
plt.title("Humidity Over Time")
plt.xlabel("Date")
plt.ylabel("Humidity (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# Rainfall bar chart
# -----------------------------

plt.figure()
plt.bar(df["Date"], df["Rainfall"])
plt.title("Rainfall Over Time")
plt.xlabel("Date")
plt.ylabel("Rainfall (inches)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# Temperature distribution
# -----------------------------

plt.figure()
sns.histplot(df["Temperature"], bins=10, kde=True)
plt.title("Temperature Distribution")
plt.xlabel("Temperature")
plt.ylabel("Frequency")
plt.show()

# -----------------------------
# Correlation heatmap
# -----------------------------

plt.figure()
corr = df[["Temperature", "Humidity", "WindSpeed", "Rainfall"]].corr()

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Weather Feature Correlations")
plt.show()

# -----------------------------
# Relationship plot
# -----------------------------

plt.figure()
sns.scatterplot(x=df["Temperature"], y=df["Humidity"])
plt.title("Temperature vs Humidity")
plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.show()

# -----------------------------
# Rolling averages (real-world skill)
# -----------------------------

df["TempRollingAvg"] = df["Temperature"].rolling(window=7).mean()

plt.figure()
plt.plot(df["Date"], df["Temperature"], label="Temp")
plt.plot(df["Date"], df["TempRollingAvg"], label="7-day Avg")
plt.legend()
plt.title("Temperature vs Rolling Average")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# -----------------------------
# Monthly grouping
# -----------------------------

df.set_index("Date", inplace=True)

monthly_avg = df.resample("M").mean(numeric_only=True)

print("\nMonthly averages:")
print(monthly_avg)

# -----------------------------
# Key insights
# -----------------------------

print("\n=== INSIGHTS ===")

print("Average Temperature:", df["Temperature"].mean())
print("Max Temperature:", df["Temperature"].max())
print("Min Temperature:", df["Temperature"].min())

print("Total Rainfall:", df["Rainfall"].sum())

print("\nMost humid day:")
print(df["Humidity"].idxmax(), df["Humidity"].max())

print("\nAnalysis complete!")
print("=" * 60)
