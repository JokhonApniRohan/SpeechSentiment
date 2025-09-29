from dataset_builder import build_dataset

X, y = build_dataset("data")

print("Feature shape:", X.shape)    # (num_samples, 40)
print("Labels shape:", y.shape)     # (num_samples,)
print("First 5 labels:", y[:5])
print("First feature vector shape:", X[0].shape)
