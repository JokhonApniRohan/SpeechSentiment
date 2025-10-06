import pandas as pd
from sklearn.preprocessing import Normalizer

# ---- Load your dataset ----
# Replace with your actual CSV file path
df = pd.read_csv("ravdess_features.csv")

# ---- Columns to exclude ----
exclude_cols = ['emotion', 'intensity', 'gender', 'file']

# ---- Select columns to normalize ----
cols_to_normalize = [col for col in df.columns if col not in exclude_cols]

# ---- Initialize the L2 Normalizer ----
normalizer = Normalizer(norm='l2')

# ---- Apply normalization ----
# Normalizer works row-wise, so we transpose to apply column-wise normalization
df[cols_to_normalize] = df[cols_to_normalize].apply(
    lambda x: x / ( (x**2).sum()**0.5 ) if (x**2).sum() != 0 else x
)

# ---- Save or view the result ----
print(df.head())


# Optionally save to a new file
df.to_csv("norm_audio_dataset.csv", index=False)


