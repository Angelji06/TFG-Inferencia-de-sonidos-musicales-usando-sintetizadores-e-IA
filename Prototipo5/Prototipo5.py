import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Función de pérdida hibrida:
#   1) Reconstrucción espectral (L1 en dB): 
#        - Mide la diferencia directa entre espectrogramas
#        - Es la parte principal de la pérdida
#
#   2) Spectral Convergence (SC):
#        - Compara las magnitudes de forma relativa
#        - Captura la estructura global del espectro (armónicos, energía)
#        - Se usa con un peso bajo (mas como refuerzo)
#
#   3) Parámetros (SmoothL1):
#        - Penaliza diferencias en los parámetros del sintetizador
#        - Peso pequeño porque los parámetros son menos importantes y no siempre son únicos para un mismo espectrograma.
class HybridLoss(nn.Module):
    def __init__(self, spec_weight=1.0, sc_weight=0.5, param_weight=0.05, eps=1e-8):
        super().__init__()
        self.spec_weight = spec_weight
        self.sc_weight = sc_weight
        self.param_weight = param_weight
        self.eps = eps

        self.l1 = nn.L1Loss()
        self.param_loss = nn.SmoothL1Loss()

    def spectral_convergence(self, pred_db, tgt_db):
        pred_mag = 10 ** (pred_db / 20)                         # Espectrogramas de vuelta a escala lineal
        tgt_mag = 10 ** (tgt_db / 20)
        num = torch.norm(pred_mag - tgt_mag, p='fro')           # Norma de Frobenius en matriz diferencia (Raiz del sumatorio de cuadrados)
        den = torch.norm(tgt_mag, p='fro').clamp(min=self.eps)  # Norma de Frobenius en matriz target
        return num / den                                        # Error relativo

    def forward(self, pred_spec, tgt_spec, pred_params, tgt_params):
        # 1) Pérdida espectral (diferencia entre espectrogramas)
        loss_spec = self.l1(pred_spec, tgt_spec)

        # 2) Convergencia espectral 
        sc_loss = self.spectral_convergence(pred_spec, tgt_spec)

        # 3) Pérdida paramétrica (bajo peso)
        loss_params = self.param_loss(pred_params, tgt_params)

        # 4) Combinación de pérdidas
        total = (self.spec_weight * loss_spec) + (self.sc_weight * sc_loss) + (self.param_weight * loss_params)

        return total, loss_spec.detach(), sc_loss.detach(), loss_params.detach()

# -------------------------
# CNNRegressor
# Un ejemplo simple: encoder -> latente -> two heads:
#   - head params: regresión (3 valores)
#   - head spec: decoder (reconstrucción del espectrograma)
# -------------------------
class CNNRegressor5(nn.Module):
    def __init__(self, n_params=7, input_channels=1, base_filters=32):
        super().__init__()
        # ENCODER -> Convierte el espectrograma (1, H, W) en una representación comprimida rica en características
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

        # BOTTLENECK -> Duplica el numero de canales manteniendo tamaño espacial
        # (128, H/8, W/8) -> (256, H/8, W/8)
        # Basicamente sirve para mezclar los patrones que el encoder ha extraído
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_filters*4, base_filters*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_filters*8),
            nn.ReLU(inplace=True),
        )

        # GLOBAL POOLING -> Para los parámetros (pues describen todo el sonido completo, no un punto concreto del espectrograma.)
        # Toma cada canal completo y hace la media de todas sus posiciones, quedándose con un único número que resume todo el canal.
        # (C, H, W)  →  (C, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))   # Solo un numero por canal (256, 1, 1)
        self.fc_params = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters*8, 64),              # Conecta los 256 detectores de sonido con 64 neuronas intermedias
            nn.ReLU(inplace=True),
            nn.Linear(64, n_params)                     # Conecta las 64 neuronas intermedias a las 7 salidas
        )

        # DECODER -> su misión es reconstruir el espectrograma original a partir de la representación comprimida del bottleneck.
        # El encoder comprimió tres veces, así que el decoder descomprime tres veces (transposed conv).
        # Si no se reconstruye, el encoder podría ignorar información importante que no afecta directamente a los parámetros.
            # Si el encoder ignora armónicos → el decoder no podrá dibujarlos → pérdida alta → encoder aprende a captar armónicos.
            # Si el encoder no aprende patrones temporales → el decoder produce ruido → pérdida alta → encoder aprende patrones temporales.
            # Si el bottleneck no guarda suficiente información → reconstrucción borrosa → pérdida alta → más capacidad en canales.
            # Si el decoder no sabe expandir bien → reconstrucción deformada → gradiente corrige sus filtros.
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

        # If the spatial dims don't match exactly (due to integer division), we can
        # crop or pad recon to match x's H,W. We'll ensure it's the same shape:
        if recon.shape[2:] != x.shape[2:]:
            recon = F.interpolate(recon, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return params, recon

    # Función que entrena el modelo
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
            best_state = None


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
                            avg_sc = running_sc / print_every_batches
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

            return history #Retorna: history dict con listas 'total', 'spec', 'params' (valores medios por época)
    
    #???
    def evaluate(self, test_loader, device='cpu', save_dir="eval_results"):
        """
        Evalúa el modelo sobre test_loader siguiendo la celda que proporcionaste.
        - test_loader: DataLoader que devuelve batches con (spec, params) o (spec, audio, params)
        - device: 'cpu' o 'cuda'
        - save_dir: carpeta donde se guardará preds_vs_trues.csv (por defecto 'eval_results')
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
        param_mse_fn = nn.MSELoss(reduction='none')  # no usado directamente aquí, calculamos manualmente

        last_pred_spec = None
        last_batch_spec = None

        with torch.no_grad():
            for batch in test_loader:
                # Aceptamos batches de forma (spec, params) o (spec, audio, params)
                if len(batch) == 2:
                    batch_spec, batch_params = batch
                elif len(batch) == 3:
                    batch_spec, _, batch_params = batch
                else:
                    raise ValueError(f"Formato de batch inesperado: {len(batch)} elementos")

                # mover a device
                batch_spec = batch_spec.to(device)
                batch_params = batch_params.to(device)

                # Forward: el modelo puede devolver (params, recon) o solo params
                out = self(batch_spec)
                if isinstance(out, (tuple, list)):
                    pred_params = out[0]
                    pred_spec = out[1] if len(out) > 1 else None
                else:
                    pred_params = out
                    pred_spec = None

                # asegurar dim correcta
                if pred_params.dim() == 1:
                    pred_params = pred_params.unsqueeze(0)

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

                # pérdida espectrograma si hay pred_spec
                if pred_spec is not None:
                    # asegurar misma shape
                    if pred_spec.shape != batch_spec.shape:
                        pred_spec = F.interpolate(pred_spec, size=batch_spec.shape[2:], mode='bilinear', align_corners=False)
                    spec_loss = spec_loss_fn(pred_spec, batch_spec)
                    spec_losses.append(spec_loss.item())
                    last_pred_spec = pred_spec.detach().cpu()
                    last_batch_spec = batch_spec.detach().cpu()

        # Concatenar preds y trues
        if len(preds_list) == 0:
            raise RuntimeError("No se obtuvieron predicciones (preds_list vacío). Revisa test_loader/model.")
        preds = torch.cat(preds_list, dim=0)
        trues = torch.cat(trues_list, dim=0)

        # Métricas por parámetro (mean)
        mse_per_param = (param_mse_sum / n_samples).numpy()
        mae_per_param = (param_mae_sum / n_samples).numpy()
        rmse_per_param = np.sqrt(mse_per_param)

        param_names = [
            "carrier", "ratio", "index", 
            "amp_att", "amp_dec", "mod_att", "mod_dec"
        ] if preds.shape[1] == 7 else [f"p{i}" for i in range(preds.shape[1])]

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
        plt.show()

        # Mostrar una reconstrucción de ejemplo (si el modelo devuelve pred_spec)
        if last_pred_spec is not None and last_batch_spec is not None:
            example_spec = last_batch_spec[0]      # primer ejemplo de la última batch
            example_pred_spec = last_pred_spec[0]
            def show_spec(tensor, title="spec"):
                arr = tensor.squeeze(0).numpy()
                plt.imshow(arr, origin='lower', aspect='auto')
                plt.colorbar()
                plt.title(title)
                plt.xlabel("time")
                plt.ylabel("freq")
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1); show_spec(example_spec, title="Target spec (example)")
            plt.subplot(1,2,2); show_spec(example_pred_spec, title="Predicted spec (example)")
            plt.tight_layout()
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

    #???
    @staticmethod
    def load(path="cnn_spectrogram.pth", device="cpu", n_params=7, input_channels=1, base_filters=32):
        """
        Carga un state_dict y devuelve una instancia en modo eval().
        """
        model = CNNRegressor5(n_params=n_params, input_channels=input_channels, base_filters=base_filters)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        print(f"Modelo cargado desde: {path}")
        return model
