import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("GAME STATS ANALYSIS")
print("=" * 60)

# -----------------------------
# Generate simulated game stats
# -----------------------------

np.random.seed(42)

players = [
    "Player1", "Player2", "Player3", "Player4",
    "Player5", "Player6", "Player7", "Player8"
]

df = pd.DataFrame({
    "Player": players,
    "GamesPlayed": np.random.randint(5, 50, size=8),
    "Wins": np.random.randint(1, 30, size=8),
    "Losses": np.random.randint(1, 25, size=8),
    "AvgReactionTime": np.random.normal(0.45, 0.1, size=8),
    "HighScore": np.random.randint(100, 10000, size=8),
    "Accuracy": np.random.randint(50, 100, size=8)
})

print("\nInitial Dataset:")
print(df)

# -----------------------------
# Basic stats
# -----------------------------

print("\n=== BASIC STATS ===")
print(df.describe())

# -----------------------------
# Win rate
# -----------------------------

df["WinRate"] = df["Wins"] / df["GamesPlayed"]

print("\nWin Rates:")
print(df[["Player", "WinRate"]])

# -----------------------------
# Key performers
# -----------------------------

fastest = df.loc[df["AvgReactionTime"].idxmin()]
slowest = df.loc[df["AvgReactionTime"].idxmax()]
top_score = df.loc[df["HighScore"].idxmax()]
best_accuracy = df.loc[df["Accuracy"].idxmax()]

print("\nFastest Player:")
print(fastest)

print("\nSlowest Player:")
print(slowest)

print("\nHighest Score Player:")
print(top_score)

print("\nMost Accurate Player:")
print(best_accuracy)

# -----------------------------
# Skill score (feature engineering)
# -----------------------------

df["SkillScore"] = (
    df["WinRate"] * 100 +
    df["Accuracy"] +
    (1 / df["AvgReactionTime"]) * 10 +
    df["HighScore"] / 1000
)

# -----------------------------
# Leaderboard
# -----------------------------

leaderboard = df.sort_values(by="SkillScore", ascending=False)

print("\n🏆 LEADERBOARD 🏆")
print(leaderboard[["Player", "SkillScore"]])

# -----------------------------
# Summary stats
# -----------------------------

print("\n=== SUMMARY ===")
print("Average Win Rate:", df["WinRate"].mean())
print("Average Accuracy:", df["Accuracy"].mean())
print("Average Reaction Time:", df["AvgReactionTime"].mean())

print("\nTop Player Overall:")
print(df.loc[df["SkillScore"].idxmax()])

# -----------------------------
# Save dataset
# -----------------------------

df.to_csv("game_stats.csv", index=False)
print("\nSaved to game_stats.csv")

# -----------------------------
# Visualization
# -----------------------------

plt.figure()
plt.bar(df["Player"], df["SkillScore"])
plt.title("Player Skill Scores")
plt.xlabel("Player")
plt.ylabel("Skill Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
sns.scatterplot(x=df["WinRate"], y=df["Accuracy"])
plt.title("Win Rate vs Accuracy")
plt.show()

plt.figure()
sns.histplot(df["AvgReactionTime"], bins=5, kde=True)
plt.title("Reaction Time Distribution")
plt.show()

plt.figure()
corr = df[["GamesPlayed", "Wins", "Losses", "AvgReactionTime", "HighScore", "Accuracy", "SkillScore"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Game Stats Correlation")
plt.show()

print("\nAnalysis Complete!")
print("=" * 60)
