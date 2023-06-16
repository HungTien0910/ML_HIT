import torch
import torchvision
import torch.nn as nn
import torch.optim as opt
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as transform
from tqdm import tqdm
from torchvision.datasets import FashionMNIST
from torch.utils.data import Dataset, DataLoader, random_split

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding="same")
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same")
        self.fc1 = nn.Linear(in_features=64*7*7, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.flatten = nn.Flatten()
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

# Load FashionMNIST dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])
trainset = torchvision.datasets.FashionMNIST('/files/', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST('/files/', train=False, download=True, transform=transform)

# Split trainset into validation and train subsets
dataset_train_size = int(len(trainset) * 0.8)
dataset_val_size = len(trainset) - dataset_train_size
train_dataset, val_dataset = random_split(trainset, [dataset_train_size, dataset_val_size])

# Define dataloaders
batch_size_train = 128
batch_size_test = 1000
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=True)

# Create an instance of the CNN model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNModel().to(device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# Training loop
n_epochs = 100
log_interval = 10
train_losses, val_losses = [], []
for epoch in range(n_epochs):
    train_loss = 0
    val_loss = 0
    total_correct = 0

    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            val_loss += loss.item() * data.size(0)
            total_correct += torch.sum(torch.argmax(output, dim=1) == target).item()

    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    total_correct /= len(val_loader.dataset)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch % log_interval == 0:
        print(f"Epoch: {epoch+1}/{n_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val Accuracy: {total_correct:.6f}")

# Plot the training and validation losses
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()