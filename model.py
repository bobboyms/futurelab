import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numba import float32
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from futurelabs.lab.chart import Type
from futurelabs.lab.project import Project
import polars as pl
import time

project_log = Project(
        project_name="MNIST",
        work_folder="logs",
        laboratory_name="lab 1",
    ).log()

log_class = project_log.new_logger(
        section_name="Classificação",
        description="Aqui está sendo monitorando o gradiente de 3 camadas do modelo",
    )

log_gradientes = project_log.new_logger(
        section_name="Gradientes do modelo",
        description="Aqui está sendo monitorando o gradiente de 3 camadas do modelo",
        buffer_sleep=4,
    )
# Carregar o dataset Breast Cancer Wisconsin
data = load_breast_cancer()
X = data.data
y = data.target

# Dividir os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar os dados (normalizar as features)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Converter os arrays numpy para tensores PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Criar DataLoader para o conjunto de treino e teste
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class BreastCancerModel(nn.Module):
    def __init__(self):
        super(BreastCancerModel, self).__init__()
        self.fc1 = nn.Linear(30, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x) + x
        x = torch.relu(x)
        x = self.fc2(x) + x
        x = torch.relu(x)
        x = self.fc3(x)
        return x

model = BreastCancerModel()

criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    target_list = []
    output_list = []
    loss_list = []
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # model.fc1.weight.grad

        log_gradientes.log_histogram("FC 1", model.fc1.weight.grad)
        log_gradientes.log_histogram("FC 2", model.fc2.weight.grad)
        log_gradientes.log_histogram("FC 3", model.fc3.weight.grad)

        optimizer.step()

        # print(output)
        loss_list.append(loss.item())
        output_list.append(output)
        target_list.append(target)
        running_loss += loss.item() * data.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    outputs = torch.cat(output_list).flatten().tolist()
    targets = torch.cat(target_list).flatten().tolist()
    mean_loss = np.mean(loss_list).astype(float)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {mean_loss:.4f}')
    log_class.log_classification("Avaliacao",targets, outputs, epoch)
    log_class.log_scalar("Loss", {"train": mean_loss, "teste":0.025 * epoch},epoch)

    audio_path = '127_sample.wav'
    data, sr = librosa.load(audio_path, sr=None)
    log_class.log_audio("Amostras de audio", data, sr, epoch)


time.sleep(6)

