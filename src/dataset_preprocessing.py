import pandas as pd

# Load dataset
df = pd.read_csv('norm_audio_dataset.csv')

# 1️⃣ Drop all Hammarberg, Shimmer, LogRelF0 related columns
drop_cols = df.columns[df.columns.str.contains(
    'hammarberg|shimmer|logRelF0', case=False, regex=True)].tolist()

# 2️⃣ Drop formant bandwidth & amplitude columns (keep only main frequency columns)
drop_cols += df.columns[df.columns.str.contains(
    'bandwidth|amplitude', case=False, regex=True)].tolist()

# 3️⃣ Drop redundant stddev variants (keep a few key ones like F0_stddevNorm)
# Keep only F0 stddevNorm but drop others containing 'stddev'
drop_cols += [
    col for col in df.columns
    if 'stddev' in col.lower() and 'F0semitoneFrom27.5Hz_sma3nz_stddevNorm' not in col
]

# 4️⃣ Drop duplicate formants (keep F1frequency_sma3nz_amean, drop F2/F3 frequencies)
drop_cols += [
    col for col in df.columns
    if 'F2frequency' in col or 'F3frequency' in col
]

# Remove duplicates from drop list
drop_cols = list(set(drop_cols))

# Create cleaned dataset
df_final = df.drop(columns=drop_cols, errors='ignore')

# Save to new CSV
df_final.to_csv('final_audio_feature_dataset.csv', index=False)

print(f"✅ Cleaned dataset created successfully! Shape: {df_final.shape}")
print(f"🗑️ Dropped {len(drop_cols)} unimportant/redundant columns.")
