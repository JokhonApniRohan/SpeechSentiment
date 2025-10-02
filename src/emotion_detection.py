# emotion_detection.py
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_builder import build_dataset
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import json


# -----------------------------
# 1. Build dataset
# -----------------------------
features, labels = build_dataset("data")  # adjust path to your data folder
# print("Features shape:", features.shape)
# print("Labels shape:", labels.shape)

# -----------------------------
# 2. PyTorch Dataset
# -----------------------------
class AudioDataset(Dataset):
    def __init__(self, features, labels):
        # Map emotion strings to integers
        self.emotion_to_idx = {emotion: i for i, emotion in enumerate(sorted(set(labels)))}
        labels_idx = [self.emotion_to_idx[label] for label in labels]
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels_idx, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = AudioDataset(features, labels)

# -----------------------------
# 3. Split and create DataLoaders
# -----------------------------
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# -----------------------------
# 4. Define model and training loop
# -----------------------------
class EmotionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes):
        super(EmotionNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout1(x)
        x = self.fc3(x)
        return x

def train_model(train_loader, val_loader, input_dim, num_classes, epochs=600, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = EmotionNet(input_dim=input_dim, hidden_dim1=128, hidden_dim2 = 128, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_acc_lst = []
    val_acc_lst = []

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for features_batch, labels_batch in train_loader:
            features_batch, labels_batch = features_batch.to(device), labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(features_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels_batch).sum().item()
            total += labels_batch.size(0)

        train_acc = 100 * correct / total
        train_acc_lst.append(train_acc)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Acc: {train_acc:.2f}%")

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for features_batch, labels_batch in val_loader:
                features_batch, labels_batch = features_batch.to(device), labels_batch.to(device)
                outputs = model(features_batch)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels_batch).sum().item()
                val_total += labels_batch.size(0)

        val_acc = 100 * val_correct / val_total
        val_acc_lst.append(val_acc)
        print(f"Validation Accuracy: {val_acc:.2f}%")

    torch.save(model.state_dict(), "emotion_model.pth")
    print("Model saved as emotion_model.pth")
    emotions = sorted(set(labels))
    emotion_to_idx = {emotion: i for i, emotion in enumerate(emotions)}

    with open('labels.json', 'w') as f:
        json.dump(emotion_to_idx, f)
    print('Labels saved as json in labels.json')


    # # Plot accuracies
    # plt.plot(train_acc_lst, label="Train Accuracy")
    # plt.plot(val_acc_lst, label="Validation Accuracy")
    # plt.xlabel("Epochs")
    # plt.ylabel("Accuracy (%)")
    # plt.legend()
    # plt.title("Training vs Validation Accuracy")
    # plt.show()

# -----------------------------
# 5. Train
# -----------------------------
train_model(train_loader, val_loader, input_dim=features.shape[1], num_classes=len(set(labels)))
