# Import Libraries
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
import requests

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create Dataset
n_samples = 10000
x, y = make_circles(n_samples, noise=0.02, random_state=42)

# Visualize Dataset
plt.scatter(x=x[:, 0], y=x[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.title("Dataset Visualization")
plt.show()

# Convert Data to Tensors and Move to Device
x = torch.from_numpy(x).type(torch.float).to(device)
y = torch.from_numpy(y).type(torch.float).to(device)

# Split Dataset into Training and Test Sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=34)

# Define Binary Classification Model
class BinaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=2, out_features=10)
        self.linear2 = nn.Linear(in_features=10, out_features=10)
        self.linear3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.linear1(x)
        f1 = self.relu(out1)
        out2 = self.linear2(f1)
        f2 = self.relu(out2)
        out3 = self.linear3(f2)
        return out3

# Model Initialization
torch.manual_seed(42)
torch.cuda.manual_seed(32)
model = BinaryModel().to(device)
print("Model Initialized:\n", model.state_dict())

# Define Loss Function and Optimizer
loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Accuracy Function
def accuracy(output, labels):
    probs = torch.sigmoid(output)
    pred = (probs > 0.5).float()
    return torch.sum(pred == labels).item() / len(labels) * 100

# Download Helper Functions
if not Path("helper_functions.py").is_file():
    print("Downloading helper_functions.py...")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

# Plot Decision Boundaries Before Training
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train Data")
plot_decision_boundary(model, x_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test Data")
plot_decision_boundary(model, x_test, y_test)
plt.show()

# Training Loop
nepochs = 100
train_losses, test_losses = [], []
train_accuracies, test_accuracies = [], []

for epoch in range(nepochs):
    # Training Phase
    model.train()
    y_train_pred = model(x_train)
    train_loss = loss_function(y_train_pred, y_train.unsqueeze(-1))
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Evaluation Phase
    model.eval()
    with torch.inference_mode():
        y_test_pred = model(x_test)
        test_loss = loss_function(y_test_pred, y_test.unsqueeze(-1))

    # Compute Accuracies
    train_acc = accuracy(y_train_pred, y_train.unsqueeze(-1))
    test_acc = accuracy(y_test_pred, y_test.unsqueeze(-1))

    # Store Metrics
    train_losses.append(train_loss.item())
    test_losses.append(test_loss.item())
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    # Print Progress Every 10 Epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{nepochs} | Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")

# Plot Training and Validation Metrics
epochs = range(1, nepochs + 1)
plt.figure(figsize=(12, 6))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Training Loss", color='blue', marker='o')
plt.plot(epochs, test_losses, label="Validation Loss", color='red', linestyle='--', marker='x')
plt.title("Loss Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label="Training Accuracy", color='green', marker='o')
plt.plot(epochs, test_accuracies, label="Validation Accuracy", color='orange', linestyle='--', marker='x')
plt.title("Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Check Predictions
with torch.inference_mode():
    test_pred = model(x_test)
print(torch.round(torch.sigmoid(test_pred[:10])))
print(y_test[:10])

# Plot Decision Boundaries After Training
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train Decision Boundary")
plot_decision_boundary(model, x_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test Decision Boundary")
plot_decision_boundary(model, x_test, y_test)
plt.show()

# Save and Load Model
torch.save(model.state_dict(), "CircleModel.pth")
print("Model Saved!")

saved_model = BinaryModel()
saved_model.load_state_dict(torch.load("CircleModel.pth"))
print("Loaded Model:\n", saved_model.state_dict())
