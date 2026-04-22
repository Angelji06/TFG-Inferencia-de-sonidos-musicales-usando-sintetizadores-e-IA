import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from losses import HybridLoss


# ──────────────────────────────────────────────────────────────────────────────
# CNNRegressorSimple
# ──────────────────────────────────────────────────────────────────────────────
# Arquitectura base: encoder → bottleneck → regresión de los 8 parámetros FM.
#   - forward() devuelve directamente el tensor de parámetros
#   - La función de pérdida es SmoothL1 sobre los parámetros
#   - Entrena rápido, pero el encoder solo aprende lo necesario para predecir
#     parámetros, sin garantía de capturar la estructura completa del espectrograma

# ──────────────────────────────────────────────────────────────────────────────
# CNNRegressor5
# ──────────────────────────────────────────────────────────────────────────────
# Extensión de CNNRegressorSimple: añade un decoder que reconstruye el espectrograma.
#   - head params: igual que en CNNRegressorSimple
#   - head spec:   decoder que reconstruye el espectrograma de entrada
#
# El decoder actúa como regularizador: obliga al encoder a retener toda la
# información del espectrograma y no solo la relevante para los parámetros.

class CNNRegressorSimple(nn.Module):
    def __init__(self, n_params=8, input_channels=1, base_filters=32):
        super().__init__()

        # ENCODER
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # BOTTLENECK
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*8),
            nn.ReLU(inplace=True),
        )

        # CABEZA DE REGRESIÓN
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_params = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters*8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_params)     # 8 salidas FM
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b  = self.bottleneck(e3)
        return self.fc_params(self.global_pool(b))  # (B, n_params)

    def fit(self, train_loader, val_loader=None, device='cpu', epochs=10, lr=1e-3,
            print_every_batches=50, criterion=None, optimizer=None):
        """
        Entrena usando solo pérdida sobre parámetros (SmoothL1).
        El argumento criterion se acepta para mantener compatibilidad de firma con
        CNNRegressor5.fit(), que lo necesita para la pérdida espectral, pero aquí se ignora.
        """
        self.to(device)
        loss_fn   = nn.SmoothL1Loss()
        optimizer = optimizer or optim.Adam(self.parameters(), lr=lr)

        history = {'total': [], 'params': [], 'val_total': []}

        best_val_loss  = float('inf')
        best_state_dict = None

        for epoch in range(epochs):
            self.train()
            running_params = 0.0
            n_batches = 0

            for batch_idx, (batch_spec, batch_params) in enumerate(train_loader):
                batch_spec   = batch_spec.to(device)
                batch_params = batch_params.to(device)

                optimizer.zero_grad()
                pred_params = self(batch_spec)
                loss = loss_fn(pred_params, batch_params)
                loss.backward()
                optimizer.step()

                running_params += loss.item()
                n_batches += 1

                if print_every_batches and print_every_batches > 0:
                    if (batch_idx + 1) % print_every_batches == 0:
                        avg = running_params / n_batches
                        print(f" Epoch {epoch+1}/{epochs}  Batch {batch_idx+1}  Avg params loss: {avg:.6f}")

            avg_params = running_params / max(1, n_batches)
            history['total'].append(avg_params)
            history['params'].append(avg_params)

            msg_val = ""
            if val_loader is not None:
                self.eval()
                val_running = 0.0
                n_val = 0
                with torch.no_grad():
                    for v_spec, v_params in val_loader:
                        v_spec   = v_spec.to(device)
                        v_params = v_params.to(device)
                        v_loss   = loss_fn(self(v_spec), v_params)
                        val_running += v_loss.item()
                        n_val += 1
                avg_val = val_running / max(1, n_val)
                history['val_total'].append(avg_val)
                msg_val = f" | Val Loss: {avg_val:.6f}"
                if avg_val < best_val_loss:
                    best_val_loss   = avg_val
                    best_state_dict = self.state_dict()
                    msg_val += " (*)"

            print(f"Epoch {epoch+1}/{epochs}  Params loss: {avg_params:.6f}{msg_val}")

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)
            print("Entrenamiento finalizado. Se han restaurado los pesos de la mejor época.")

        return history

    def evaluate(self, test_loader, device='cpu', save_dir="eval_results", means=None, stds=None):
        """CNNRegressor5 amplía esta lógica de evaluación para soportar la reconstrucción del espectrograma."""
        return CNNRegressor5.evaluate(self, test_loader, device=device, save_dir=save_dir, means=means, stds=stds)



class CNNRegressor5(nn.Module):
    def __init__(self, n_params=8, input_channels=1, base_filters=32):
        super().__init__()
        # ENCODER — igual que en CNNRegressorSimple
        # Convierte el espectrograma (1, H, W) en una representación comprimida rica en características
        # Tensores espectrograma tal que: (Numero canales, Numero filas (frecuencia), Numero columnas (tiempo))
        # (1, H, W) -> (32, H/2,  W/2) -> (64, H/4,  W/4) -> (128, H/8, W/8)
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, base_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # H/2, W/2
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_filters, base_filters*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # H/4, W/4
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_filters*2, base_filters*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # H/8, W/8
        )

        # BOTTLENECK — igual que en CNNRegressorSimple
        # Duplica el numero de canales manteniendo tamaño espacial
        # (128, H/8, W/8) -> (256, H/8, W/8)
        # Basicamente sirve para mezclar los patrones que el encoder ha extraído
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*8),
            nn.ReLU(inplace=True),
        )

        # CABEZA DE REGRESIÓN — igual que en CNNRegressorSimple
        # GLOBAL POOLING -> Para los parámetros (pues describen todo el sonido completo, no un punto concreto del espectrograma.)
        # Toma cada canal completo y hace la media de todas sus posiciones, quedándose con un único número que resume todo el canal.
        # (C, H, W)  →  (C, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))   # Solo un numero por canal (256, 1, 1)
        self.fc_params = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters*8, 64),              # Conecta los 256 detectores de sonido con 64 neuronas intermedias
            nn.ReLU(inplace=True),
            nn.Linear(64, n_params)                     # Conecta las 64 neuronas intermedias a las 8 salidas
        )

        # DECODER -> su misión es reconstruir el espectrograma original a partir de la representación comprimida del bottleneck.
        # El encoder comprimió tres veces, así que el decoder descomprime tres veces (transposed conv).
        # Si no se reconstruye, el encoder podría ignorar información importante que no afecta directamente a los parámetros.
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_filters*8, base_filters*4, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_filters*4),
            nn.ReLU(inplace=True),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_filters*4, base_filters*2, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_filters*2),
            nn.ReLU(inplace=True),
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_filters*2, base_filters, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )
        # Última capa que convierte lo que sale del decoder en un tensor con la MISMA forma que el espectrograma original.
        self.recon_head = nn.Sequential(
            nn.Conv2d(base_filters, input_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):  # x: (B, 1, H, W)
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)

        # Parámetros
        pooled = self.global_pool(b)  # (B, C, 1,1)
        params = self.fc_params(pooled)  # (B, n_params)

        # Decoder
        d1 = self.dec1(b)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        recon = self.recon_head(d3)  # (B, 1, H, W)

        # Si las dimensiones espaciales no coinciden exactamente (por divisiones enteras en MaxPool),
        # se ajusta la reconstrucción al tamaño original mediante interpolación bilineal:
        if recon.shape[2:] != x.shape[2:]:
            recon = F.interpolate(recon, size=x.shape[2:], mode='bilinear', align_corners=False)

        return params, recon

    # ──────────────────────────────────────────────────────────────────────────
    # ENTRENAMIENTO
    # ──────────────────────────────────────────────────────────────────────────
    def fit(self, train_loader, val_loader=None, device='cpu', epochs=10, lr=1e-3, print_every_batches=50, criterion=None, optimizer=None):
            # - train_loader: DataLoader que devuelve (batch_spec, batch_params)
            # - criterion: instancia de HybridLoss (si None se crea una por defecto)
            # - optimizer: optimizador; si None se crea Adam con lr

            self.to(device)

            # Lo implemento así para poder probar distintos criterios y optimizers
            if criterion is None:
                criterion = HybridLoss()
                criterion.to(device)
            if optimizer is None:
                optimizer = optim.Adam(self.parameters(), lr=lr)

            history = {'total': [],'spec': [],'params': [], 'val_total' : []}

            # Variable que guarda el mejor modelo
            best_val_loss = float('inf')
            best_state_dict = None

            # Entrenamiento
            for epoch in range(epochs):
                self.train()
                running_total = 0.0
                running_spec = 0.0
                running_sc = 0.0
                running_params = 0.0
                n_batches = 0

                for batch_idx, (batch_spec, batch_params) in enumerate(train_loader):
                    # mover a device
                    batch_spec = batch_spec.to(device)       # (B,1,H,W)
                    batch_params = batch_params.to(device)   # (B,3) o (B,n_params)

                    optimizer.zero_grad()  # Reset gradientes
                    pred_params, pred_spec = self(batch_spec)

                    loss_total, loss_spec, loss_sc, loss_params = criterion(pred_spec, batch_spec, pred_params, batch_params)
                    loss_total.backward()  #Backprop
                    optimizer.step()       #Descenso de gradientes

                    running_total += loss_total.item()
                    running_spec += loss_spec.item()
                    running_sc += loss_sc.item()
                    running_params += loss_params.item()
                    n_batches += 1

                    # (opcional) print por batches si el usuario lo habilita mediante print_every_batches
                    if print_every_batches is not None and print_every_batches > 0:
                        if (batch_idx + 1) % print_every_batches == 0:
                            avg_total_sofar = running_total / max(1, n_batches)
                            avg_spec_sofar = running_spec / max(1, n_batches)
                            avg_params_sofar = running_params / max(1, n_batches)
                            print(f" Epoch {epoch+1}/{epochs}  Batch {batch_idx+1}  Avg total so far: {avg_total_sofar:.6f}  Spec: {avg_spec_sofar:.6f}  Params: {avg_params_sofar:.6f}")

                avg_total = running_total / max(1, n_batches)
                avg_spec = running_spec / max(1, n_batches)
                avg_params = running_params / max(1, n_batches)

                history['total'].append(avg_total)
                history['spec'].append(avg_spec)
                history['params'].append(avg_params)

                # Fase de validación
                avg_val_loss = 0.0
                msg_val = ""

                if val_loader is not None:
                    self.eval()
                    val_running = 0.0
                    n_val = 0
                    with torch.no_grad():
                        for v_spec, v_params in val_loader:
                            v_spec = v_spec.to(device)
                            v_params = v_params.to(device)

                            v_p, v_s = self(v_spec)
                            v_loss, _, _, _ = criterion(v_s, v_spec, v_p, v_params)

                            val_running += v_loss.item()
                            n_val += 1

                    avg_val_loss = val_running / max(1, n_val)
                    history['val_total'].append(avg_val_loss)
                    msg_val = f" | Val Loss: {avg_val_loss:.6f}"

                    # Guardar el mejor modelo si mejora
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_state_dict = self.state_dict()
                        msg_val += " (*)"

                print(f"Epoch {epoch+1}/{epochs}  Avg total: {avg_total:.6f}  Spec: {avg_spec:.6f}  Params: {avg_params:.6f}{msg_val}")

            if best_state_dict is not None:
                self.load_state_dict(best_state_dict)
                print("Entrenamiento finalizado. Se han restaurado los pesos de la mejor época.")

            return history  # Retorna: history dict con listas 'total', 'spec', 'params' (valores medios por época)

    # ──────────────────────────────────────────────────────────────────────────
    # EVALUACIÓN
    # ──────────────────────────────────────────────────────────────────────────
    def evaluate(self, test_loader, device='cpu', save_dir="eval_results", means=None, stds=None):
        """
        Evalúa el modelo sobre test_loader siguiendo la celda que proporcionaste.
        - test_loader: DataLoader que devuelve batches con (spec, params) o (spec, audio, params)
        - device: 'cpu' o 'cuda'
        - save_dir: carpeta donde se guardará preds_vs_trues.csv (por defecto 'eval_results')
        - means: array con las medias de entrenamiento (para desnormalizar la salida)
        - stds: array con las desviaciones estándar de entrenamiento (para desnormalizar la salida)
        Retorna: dict con métricas y ruta del CSV.
        """
        os.makedirs(save_dir, exist_ok=True)
        self.to(device)
        self.eval()

        preds_list = []
        trues_list = []
        spec_losses = []
        param_mae_sum = None
        param_mse_sum = None
        n_samples = 0

        spec_loss_fn = nn.L1Loss(reduction='mean')

        example_specs = []      # primeros 5 espectrogramas reales
        example_pred_specs = [] # primeros 5 espectrogramas reconstruidos

        with torch.no_grad():
            for batch in test_loader:
                batch_spec, batch_params = batch

                # mover a device
                batch_spec = batch_spec.to(device)
                batch_params = batch_params.to(device)

                # Forward: CNNRegressor5 devuelve (params, recon), CNNRegressorSimple devuelve params
                out = self(batch_spec)
                if isinstance(out, (tuple, list)):
                    pred_params, pred_spec = out
                else:
                    pred_params = out
                    pred_spec = None

                B = pred_params.shape[0]

                preds_list.append(pred_params.cpu())
                trues_list.append(batch_params.cpu())

                # MSE y MAE por parámetro (sumadas para acumulación)
                mse_batch = ((pred_params - batch_params)**2).sum(dim=0).detach().cpu()  # suma por batch
                mae_batch = torch.abs(pred_params - batch_params).sum(dim=0).detach().cpu()

                if param_mse_sum is None:
                    param_mse_sum = mse_batch.clone()
                    param_mae_sum = mae_batch.clone()
                else:
                    param_mse_sum += mse_batch
                    param_mae_sum += mae_batch

                n_samples += B

                # pérdida espectrograma si hay pred_spec (solo CNNRegressor5)
                if pred_spec is not None:
                    spec_loss = spec_loss_fn(pred_spec, batch_spec)
                    spec_losses.append(spec_loss.item())
                    # Acumular hasta 5 ejemplos
                    needed = 5 - len(example_specs)
                    if needed > 0:
                        example_specs.extend(batch_spec.detach().cpu()[:needed])
                        example_pred_specs.extend(pred_spec.detach().cpu()[:needed])

        preds = torch.cat(preds_list, dim=0)
        trues = torch.cat(trues_list, dim=0)

        # Desnormalizar a escala real si se proporcionan las estadísticas de entrenamiento
        if means is not None and stds is not None:
            means_t = torch.tensor(np.asarray(means, dtype=np.float32))
            stds_t  = torch.tensor(np.asarray(stds,  dtype=np.float32))
            preds = preds * stds_t + means_t
            trues = trues * stds_t + means_t

        # Métricas por parámetro (mean)
        mse_per_param = (param_mse_sum / n_samples).numpy()
        mae_per_param = (param_mae_sum / n_samples).numpy()
        rmse_per_param = np.sqrt(mse_per_param)

        param_names = [
            "carrier", "ratio", "index",
            "amp_att", "amp_sus", "amp_dec", "mod_att", "mod_dec"
        ] if preds.shape[1] == 8 else [f"p{i}" for i in range(preds.shape[1])]

        # Imprimir resultados
        print("=== Resultados de evaluación ===")
        for i, name in enumerate(param_names):
            print(f"{name:8s} | MSE: {mse_per_param[i]:.6f} | RMSE: {rmse_per_param[i]:.6f} | MAE: {mae_per_param[i]:.6f}")

        if len(spec_losses) > 0:
            print(f"Avg spec L1 loss: {np.mean(spec_losses):.6f} (n_batches_with_spec = {len(spec_losses)})")
        else:
            print("No se calcularon pérdidas de espectrograma (modelo no devolvió pred_spec).")

        # Guardar preds/trues en CSV
        df = pd.DataFrame({
            **{f"pred_{name}": preds[:,i].numpy() for i,name in enumerate(param_names)},
            **{f"true_{name}": trues[:,i].numpy() for i,name in enumerate(param_names)}
        })
        csv_path = os.path.join(save_dir, "preds_vs_trues.csv")
        df.to_csv(csv_path, index=False)
        print(f"Predicciones y reales guardados en: {csv_path}")

        # Dibujar scatter true vs pred por parámetro
        plt.figure(figsize=(12,4))
        for i,name in enumerate(param_names):
            plt.subplot(1, len(param_names), i+1)
            plt.scatter(trues[:,i].numpy(), preds[:,i].numpy(), s=6, alpha=0.6)
            mn = float(min(trues[:,i].min().item(), preds[:,i].min().item()))
            mx = float(max(trues[:,i].max().item(), preds[:,i].max().item()))
            plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1)
            plt.xlabel("True")
            plt.ylabel("Pred")
            plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "scatter_params.png"), dpi=150)
        plt.show()

        # Mostrar los primeros 5 pares target / reconstrucción
        if example_specs:
            n = len(example_specs)
            fig, axs = plt.subplots(n, 2, figsize=(10, 3 * n))
            if n == 1:
                axs = [axs]  # asegurar lista de filas
            for row, (spec_t, spec_p) in enumerate(zip(example_specs, example_pred_specs)):
                for col, (tensor, title) in enumerate([(spec_t, f"Target #{row+1}"), (spec_p, f"Pred #{row+1}")]):
                    arr = tensor.squeeze(0).numpy()
                    im = axs[row][col].imshow(arr, origin='lower', aspect='auto')
                    axs[row][col].set_title(title)
                    axs[row][col].set_xlabel("time")
                    axs[row][col].set_ylabel("freq")
                    fig.colorbar(im, ax=axs[row][col], format='%+.0f dB')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "spectrogram_example.png"), dpi=150)
            plt.show()

        print("Evaluación completada.")

        # Devolver un resumen de métricas y la ruta al CSV
        metrics = {
            'param_names': param_names,
            'mse_per_param': mse_per_param,
            'rmse_per_param': rmse_per_param,
            'mae_per_param': mae_per_param,
            'avg_spec_l1': float(np.mean(spec_losses)) if len(spec_losses) > 0 else None,
            'n_samples': int(n_samples),
            'csv_path': csv_path
        }
        return metrics
