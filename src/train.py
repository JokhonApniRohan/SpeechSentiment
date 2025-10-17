import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# ==============================
# 1️⃣ Load the dataset
# ==============================
df = pd.read_csv('final_audio_feature_dataset.csv')
df = df.drop('file', axis = 1)

# Check the columns
print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ==============================
# 2️⃣ Define target and features
# ==============================
# We'll train separate models for emotion, intensity, and gender
target_columns = ['emotion', 'intensity', 'gender']

# Ensure all targets exist in dataset
target_columns = [col for col in target_columns if col in df.columns]

feature_columns = [col for col in df.columns if col not in target_columns]

# ==============================
# 3️⃣ Function to train, test, and validate each target
# ==============================
def train_and_evaluate(df, features, target):
    print(f"\n{'='*30}\n🎯 Training model for target: '{target}'\n{'='*30}")
    
    X = df[features]
    y = df[target]
    
    # Split into train, test (80-20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize model
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluation
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy on Test Set: {acc:.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {target}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    # Cross-validation (5-fold)
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"🔁 5-Fold CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    
    return model

# ==============================
# 4️⃣ Train for each target column
# ==============================
models = {}
for target in target_columns:
    models[target] = train_and_evaluate(df, feature_columns, target)
