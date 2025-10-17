import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('norm_audio_dataset.csv')

# Select numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Ensure 'emotion' and 'gender' are included
target_cols = [col for col in ['emotion', 'gender'] if col in numeric_cols]

# Features are all numeric columns except targets
feature_cols = [col for col in numeric_cols if col not in target_cols]

# Compute correlation of features with emotion and gender
corr_focus = df[feature_cols + target_cols].corr().loc[feature_cols, target_cols]

# -------------------------------
# Print Top 20 Positive and Negative Correlations for Each Target
# -------------------------------
for target in target_cols:
    print(f"\n===== Top 20 Positive Correlations with '{target}' =====")
    print(corr_focus[target].sort_values(ascending=False).head(20))

    print(f"\n===== Top 20 Negative Correlations with '{target}' =====")
    print(corr_focus[target].sort_values(ascending=True).head(20))

# -------------------------------
# Plot Heatmap (same as before)
# -------------------------------
fig_height = max(6, 0.6 * len(feature_cols))  # taller for many features
plt.figure(figsize=(8, fig_height))

sns.heatmap(
    corr_focus,
    annot=True,
    annot_kws={"size": 10},
    cmap='coolwarm',
    linewidths=0.5,
    cbar=True,
    yticklabels=True,
    xticklabels=True
)

plt.xticks(rotation=30, ha='right')
plt.title('Correlation of Features with Emotion and Gender', fontsize=14)
plt.tight_layout()
plt.show()
