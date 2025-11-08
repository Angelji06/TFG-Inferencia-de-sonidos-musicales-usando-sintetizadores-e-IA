import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from scripts.dataset_torchaudio import SpectrogramTensorDataset

# =============================================================================================================
#  Clase que realiza un entrenamiento sencillo de una CNN para regresión de (carrier, ratio, index). Conviene ajustar hiperparámetros.
#  En el notebook solo se usa la estructura de la clase, no el resto de funciones (era una prueba, las dejo por si acaso)
# =============================================================================================================

class SmallCNNRegressor(nn.Module):
    def __init__(self, in_channels=1, out_dim=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4)),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*4, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


def train_one_epoch(model, loader, optim, loss_fn, device):
    model.train()
    running = 0.0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optim.zero_grad()
        loss.backward()
        optim.step()
        running += loss.item() * xb.size(0)
    return running / len(loader.dataset)


def evaluate(model, loader, loss_fn, device):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            running += loss.item() * xb.size(0)
    return running / len(loader.dataset)


def main():
    # paths
    tensors_dir = os.path.join(os.path.dirname(__file__), '..', 'Datasets', 'datasetFMespec_torchaudio')
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'Datasets', 'datasetFMwav', 'labels.csv')

    dataset = SpectrogramTensorDataset(tensors_dir=tensors_dir, labels_csv=csv_path)

    # small split: 90% train - 10% val
    n = len(dataset)
    val_size = max(1, n // 10)
    train_size = n - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    batch_size = 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SmallCNNRegressor(in_channels=1, out_dim=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epochs = 10
    best_val = float('inf')
    save_path = os.path.join(os.path.dirname(__file__), 'best_regressor.pt')

    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, save_path)
            print('Saved best model to', save_path)

    print('Training finished. Best val loss:', best_val)


if __name__ == '__main__':
    main()
