# emotion_detection.py
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_builder import build_dataset
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import json

# ---------------------------------------------------
# 1. Build dataset
# ---------------------------------------------------
features, emotion_labels, intensity_labels, gender_labels = build_dataset("data")

# ---------------------------------------------------
# 2. PyTorch Dataset (Multi-task)
# ---------------------------------------------------
class AudioDataset(Dataset):
    def __init__(self, features, emotion_labels, intensity_labels, gender_labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.emotion_labels = torch.tensor(emotion_labels, dtype=torch.long)
        self.intensity_labels = torch.tensor(intensity_labels, dtype=torch.long)
        self.gender_labels = torch.tensor(gender_labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.emotion_labels[idx],
            self.intensity_labels[idx],
            self.gender_labels[idx],
        )

dataset = AudioDataset(features, emotion_labels, intensity_labels, gender_labels)

# ---------------------------------------------------
# 3. Split dataset
# ---------------------------------------------------
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ---------------------------------------------------
# 4. Model (shared base + 3 heads)
# ---------------------------------------------------
class EmotionNetMultiTask(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_emotions):
        super(EmotionNetMultiTask, self).__init__()
        self.shared_fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.shared_fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        # Task heads
        self.emotion_fc = nn.Linear(hidden_dim2, num_emotions)
        self.intensity_fc = nn.Linear(hidden_dim2, 2)
        self.gender_fc = nn.Linear(hidden_dim2, 2)

    def forward(self, x):
        x = self.relu1(self.shared_fc1(x))
        x = self.relu2(self.shared_fc2(x))
        x = self.dropout(x)
        return (
            self.emotion_fc(x),
            self.intensity_fc(x),
            self.gender_fc(x),
        )

# ---------------------------------------------------
# 5. Training and Validation
# ---------------------------------------------------
def train_model(train_loader, val_loader, input_dim, num_emotions, epochs=300, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = EmotionNetMultiTask(input_dim, 128, 128, num_emotions).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Accuracy trackers
    train_acc_emotion, val_acc_emotion = [], []
    train_acc_intensity, val_acc_intensity = [], []
    train_acc_gender, val_acc_gender = [], []

    # F1, Precision, Recall trackers
    val_metrics = {
        "emotion": {"precision": [], "recall": [], "f1": []},
        "intensity": {"precision": [], "recall": [], "f1": []},
        "gender": {"precision": [], "recall": [], "f1": []},
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_emotion = correct_intensity = correct_gender = total = 0

        for features_batch, emotion_batch, intensity_batch, gender_batch in train_loader:
            features_batch = features_batch.to(device)
            emotion_batch = emotion_batch.to(device)
            intensity_batch = intensity_batch.to(device)
            gender_batch = gender_batch.to(device)

            optimizer.zero_grad()
            emotion_out, intensity_out, gender_out = model(features_batch)

            # Weighted total loss
            loss = (
                criterion(emotion_out, emotion_batch)
                + 0.3 * criterion(intensity_out, intensity_batch)
                + 0.3 * criterion(gender_out, gender_batch)
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Training accuracy
            _, pred_emotion = torch.max(emotion_out, 1)
            _, pred_intensity = torch.max(intensity_out, 1)
            _, pred_gender = torch.max(gender_out, 1)

            correct_emotion += (pred_emotion == emotion_batch).sum().item()
            correct_intensity += (pred_intensity == intensity_batch).sum().item()
            correct_gender += (pred_gender == gender_batch).sum().item()
            total += emotion_batch.size(0)

        train_acc_emotion.append(100 * correct_emotion / total)
        train_acc_intensity.append(100 * correct_intensity / total)
        train_acc_gender.append(100 * correct_gender / total)

        # --------------------------
        # Validation
        # --------------------------
        model.eval()
        val_correct_emotion = val_correct_intensity = val_correct_gender = val_total = 0
        y_true_emotion, y_pred_emotion = [], []
        y_true_intensity, y_pred_intensity = [], []
        y_true_gender, y_pred_gender = [], []

        with torch.no_grad():
            for features_batch, emotion_batch, intensity_batch, gender_batch in val_loader:
                features_batch = features_batch.to(device)
                emotion_batch = emotion_batch.to(device)
                intensity_batch = intensity_batch.to(device)
                gender_batch = gender_batch.to(device)

                emotion_out, intensity_out, gender_out = model(features_batch)

                _, pred_emotion = torch.max(emotion_out, 1)
                _, pred_intensity = torch.max(intensity_out, 1)
                _, pred_gender = torch.max(gender_out, 1)

                val_correct_emotion += (pred_emotion == emotion_batch).sum().item()
                val_correct_intensity += (pred_intensity == intensity_batch).sum().item()
                val_correct_gender += (pred_gender == gender_batch).sum().item()
                val_total += emotion_batch.size(0)

                # Store for metrics
                y_true_emotion.extend(emotion_batch.cpu().numpy())
                y_pred_emotion.extend(pred_emotion.cpu().numpy())
                y_true_intensity.extend(intensity_batch.cpu().numpy())
                y_pred_intensity.extend(pred_intensity.cpu().numpy())
                y_true_gender.extend(gender_batch.cpu().numpy())
                y_pred_gender.extend(pred_gender.cpu().numpy())

        # Accuracy per task
        val_acc_emotion.append(100 * val_correct_emotion / val_total)
        val_acc_intensity.append(100 * val_correct_intensity / val_total)
        val_acc_gender.append(100 * val_correct_gender / val_total)

        # Precision, Recall, F1 per task
        for task, y_true, y_pred, avg in [
            ("emotion", y_true_emotion, y_pred_emotion, "weighted"),
            ("intensity", y_true_intensity, y_pred_intensity, "binary"),
            ("gender", y_true_gender, y_pred_gender, "binary"),
        ]:
            val_metrics[task]["precision"].append(
                precision_score(y_true, y_pred, average=avg, zero_division=0)
            )
            val_metrics[task]["recall"].append(
                recall_score(y_true, y_pred, average=avg, zero_division=0)
            )
            val_metrics[task]["f1"].append(
                f1_score(y_true, y_pred, average=avg, zero_division=0)
            )

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"| Emotion Acc: {train_acc_emotion[-1]:.2f}% / {val_acc_emotion[-1]:.2f}% "
            f"| Intensity Acc: {train_acc_intensity[-1]:.2f}% / {val_acc_intensity[-1]:.2f}% "
            f"| Gender Acc: {train_acc_gender[-1]:.2f}% / {val_acc_gender[-1]:.2f}% "
            f"| Emotion F1: {val_metrics['emotion']['f1'][-1]:.3f}"
        )

    # ---------------------------------------------------
    # Save model
    # ---------------------------------------------------
    torch.save(model.state_dict(), "emotion_model_multitask.pth")
    print("\n✅ Model saved as emotion_model_multitask.pth")

    # ---------------------------------------------------
    # Plot Section 1: Training vs Validation Accuracy
    # ---------------------------------------------------
    plt.figure(figsize=(12, 8))
    plt.suptitle("Training vs Validation Accuracy Over Epochs", fontsize=14)

    plt.subplot(3, 1, 1)
    plt.plot(train_acc_emotion, label="Train Emotion Acc")
    plt.plot(val_acc_emotion, label="Val Emotion Acc")
    plt.legend(); plt.title("Emotion Accuracy"); plt.ylabel("Accuracy (%)")

    plt.subplot(3, 1, 2)
    plt.plot(train_acc_intensity, label="Train Intensity Acc")
    plt.plot(val_acc_intensity, label="Val Intensity Acc")
    plt.legend(); plt.title("Intensity Accuracy"); plt.ylabel("Accuracy (%)")

    plt.subplot(3, 1, 3)
    plt.plot(train_acc_gender, label="Train Gender Acc")
    plt.plot(val_acc_gender, label="Val Gender Acc")
    plt.legend(); plt.title("Gender Accuracy"); plt.xlabel("Epochs"); plt.ylabel("Accuracy (%)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # ---------------------------------------------------
    # Plot Section 2: Validation F1 Scores Over Epochs
    # ---------------------------------------------------
    plt.figure(figsize=(12, 8))
    plt.suptitle("F1 Score Trends Over Epochs", fontsize=14)

    plt.plot(val_metrics["emotion"]["f1"], label="Emotion F1")
    plt.plot(val_metrics["intensity"]["f1"], label="Intensity F1")
    plt.plot(val_metrics["gender"]["f1"], label="Gender F1")

    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------------------------------------------
# 6. Train
# ---------------------------------------------------
if __name__ == "__main__":
    train_model(train_loader, val_loader, input_dim=features.shape[1], num_emotions=len(set(emotion_labels)))
