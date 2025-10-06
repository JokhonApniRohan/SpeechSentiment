# Step 1: Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load your dataset
# Replace 'your_dataset.csv' with your file path
df = pd.read_csv('norm_audio_dataset.csv')

# Step 3: Drop non-numeric / unwanted columns
df_numeric = df.drop(columns=['file'])

# Optional: Keep only numeric columns (exclude categorical ones like 'emotion', 'gender' if needed)
df_numeric = df_numeric.select_dtypes(include=['float64', 'int64'])

# Step 4: Compute correlation matrix
corr = df_numeric.corr()


# Step 5: Plot heatmap
plt.figure(figsize=(15, 12))  # Adjust figure size as needed
sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Relational Heatmap (Correlation Matrix)')
plt.show()
