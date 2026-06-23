import pandas as pd
import numpy as np

print("=" * 60)
print("RPG CHARACTER ANALYSIS")
print("=" * 60)

# -----------------------------
# Generate RPG character dataset
# -----------------------------

np.random.seed(42)

names = [
    "Arin", "Lyra", "Kael", "Mira", "Doran",
    "Selene", "Rex", "Nova", "Vex", "Iris",
    "Thorne", "Luna", "Zane", "Kira", "Orion"
]

classes = ["Warrior", "Mage", "Archer", "Rogue", "Tank", "Healer"]

df = pd.DataFrame({
    "Name": names,
    "Class": np.random.choice(classes, size=len(names)),
    "Level": np.random.randint(1, 100, size=len(names)),
    "HP": np.random.randint(80, 500, size=len(names)),
    "Attack": np.random.randint(20, 150, size=len(names)),
    "Defense": np.random.randint(10, 120, size=len(names)),
    "Speed": np.random.randint(5, 100, size=len(names)),
    "Gold": np.random.randint(0, 10000, size=len(names))
})

print("\nCharacter Dataset:")
print(df)

# -----------------------------
# Basic stats
# -----------------------------

print("\n=== BASIC STATS ===")
print(df.describe())

# -----------------------------
# Strongest characters
# -----------------------------

strongest_attack = df.loc[df["Attack"].idxmax()]
toughest = df.loc[df["Defense"].idxmax()]
fastest = df.loc[df["Speed"].idxmax()]
richest = df.loc[df["Gold"].idxmax()]

print("\nStrongest Attack:")
print(strongest_attack)

print("\nHighest Defense:")
print(toughest)

print("\nFastest Character:")
print(fastest)

print("\nRichest Character:")
print(richest)

# -----------------------------
# Class analysis
# -----------------------------

print("\n=== CLASS ANALYSIS ===")
class_counts = df["Class"].value_counts()
print("\nClass Distribution:")
print(class_counts)

class_avg_stats = df.groupby("Class")[["HP", "Attack", "Defense", "Speed"]].mean()

print("\nAverage Stats by Class:")
print(class_avg_stats)

# -----------------------------
# Level analysis
# -----------------------------

print("\n=== LEVEL ANALYSIS ===")

print("Highest Level Character:")
print(df.loc[df["Level"].idxmax()])

print("\nLowest Level Character:")
print(df.loc[df["Level"].idxmin()])

print("\nAverage Level:", df["Level"].mean())

# -----------------------------
# Power score system
# -----------------------------

df["PowerScore"] = (
    df["HP"] * 0.3 +
    df["Attack"] * 0.4 +
    df["Defense"] * 0.2 +
    df["Speed"] * 0.1
)

print("\n=== POWER SCORES ===")
print(df[["Name", "Class", "PowerScore"]].sort_values(by="PowerScore", ascending=False))

# -----------------------------
# Top 3 strongest overall
# -----------------------------

top3 = df.sort_values(by="PowerScore", ascending=False).head(3)

print("\nTop 3 Characters:")
print(top3[["Name", "Class", "PowerScore"]])

# -----------------------------
# Gold analysis
# -----------------------------

print("\n=== ECONOMY ANALYSIS ===")
print("Total Gold in Game:", df["Gold"].sum())
print("Average Gold:", df["Gold"].mean())

print("\nRichest Player Breakdown:")
print(df.sort_values(by="Gold", ascending=False)[["Name", "Class", "Gold"]])

# -----------------------------
# Class power ranking
# -----------------------------

class_power = df.groupby("Class")["PowerScore"].mean().sort_values(ascending=False)

print("\n=== CLASS POWER RANKING ===")
print(class_power)

# -----------------------------
# Final summary
# -----------------------------

print("\n=== FINAL SUMMARY ===")

print("Most OP Character:")
print(df.loc[df["PowerScore"].idxmax()])

print("\nMost Balanced Character (closest to average stats):")

df["BalanceScore"] = (
    abs(df["HP"] - df["HP"].mean()) +
    abs(df["Attack"] - df["Attack"].mean()) +
    abs(df["Defense"] - df["Defense"].mean()) +
    abs(df["Speed"] - df["Speed"].mean())
)

print(df.loc[df["BalanceScore"].idxmin()])

print("\nAnalysis Complete!")
print("=" * 60)
