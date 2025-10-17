# ==========================================
# 🔍 Feature Importance for Multiple Targets
# ==========================================

import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('norm_audio_dataset.csv')

# Define target columns
target_cols = ['emotion', 'intensity', 'gender']

# Select numeric columns only
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Feature columns = all numeric except targets
feature_cols = [col for col in numeric_cols if col not in target_cols]

# Loop through each target
for target in target_cols:
    print(f"\n==============================")
    print(f" Feature Importance for '{target}'")
    print(f"==============================")
    
    # Split data
    X = df[feature_cols]
    y = df[target]

    # Decide whether it's classification or regression based on unique values
    if y.nunique() <= 10:  # categorical target (like emotion/gender)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:  # continuous target (like intensity)
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Get feature importances
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)

    # Print top 20
    print(importances.head(20))

    # Plot top 20 features
    plt.figure(figsize=(8, 6))
    importances.head(20).plot(kind='barh', color='skyblue')
    plt.title(f"Top 20 Important Features for '{target}'")
    plt.gca().invert_yaxis()
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()
