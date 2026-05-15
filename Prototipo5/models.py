import os
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import json


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

    def fit(self, train_loader, val_loader=None, device='cpu', epochs=10, patience=10, print_every_batches=50, criterion=None, optimizer=None, history_path=None):
        self.to(device)

        history = {'total': [], 'params': [], 'val_total': []}

        # Variable que guarda el mejor modelo
        best_val_loss = float('inf')
        best_state_dict = None
        epochs_no_improve = 0

        # Entrenamiento — envuelto en try/except para permitir cancelación con Ctrl+C
        # y restaurar los mejores pesos vistos hasta el momento de la interrupción.
        try:
            for epoch in range(epochs):
                self.train()
                running_params = 0.0
                n_batches = 0

                for batch_idx, (batch_spec, batch_params) in enumerate(train_loader):
                    # mover a device
                    batch_spec   = batch_spec.to(device)       # (B,1,H,W)
                    batch_params = batch_params.to(device)     # (B, n_params)

                    optimizer.zero_grad()                          # Reset gradientes
                    pred_params = self(batch_spec)                 # Forward pass (solo params, sin decoder)
                    loss = criterion(pred_params, batch_params)    # SmoothL1 sobre los 8 parámetros
                    loss.backward()                                # Backprop
                    optimizer.step()                               # Descenso de gradientes

                    running_params += loss.item()
                    n_batches += 1

                    # (opcional) print por batches si el usuario lo habilita mediante print_every_batches
                    if print_every_batches is not None and print_every_batches > 0:
                        if (batch_idx + 1) % print_every_batches == 0:
                            avg = running_params / n_batches
                            print(f" Epoch {epoch+1}/{epochs}  Batch {batch_idx+1}  Avg params loss: {avg:.6f}")

                avg_params = running_params / max(1, n_batches)
                history['total'].append(avg_params)   # total = params (no hay componente espectral)
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
                            v_spec   = v_spec.to(device)
                            v_params = v_params.to(device)
                            v_loss   = criterion(self(v_spec), v_params)
                            val_running += v_loss.item()
                            n_val += 1

                    avg_val_loss = val_running / max(1, n_val)
                    history['val_total'].append(avg_val_loss)
                    msg_val = f" | Val Loss: {avg_val_loss:.6f}"

                    # Guardar el mejor modelo si mejora
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_state_dict = self.state_dict()
                        epochs_no_improve = 0  # Reiniciamos la paciencia
                        msg_val += " (*)"
                    else:
                        epochs_no_improve += 1

                print(f"Epoch {epoch+1}/{epochs}  Params loss: {avg_params:.6f}{msg_val}")

                # --- Guardado en tiempo real ---
                if history_path is not None:
                    with open(history_path, 'w') as f:
                        json.dump(history, f, indent=4)

                # Bloque de Early Stopping
                if epochs_no_improve >= patience:
                    print(f"\n[!] Early Stopping activado en la época {epoch+1}. El modelo no ha mejorado en las últimas {patience} épocas.")
                    break  # Rompe el bucle 'for epoch in range(epochs)'

        except KeyboardInterrupt:
            print("\n[!] Entrenamiento cancelado por el usuario.")

        if best_state_dict is not None:
            self.load_state_dict(best_state_dict)
            print("Se han restaurado los pesos de la mejor época.")

        return history

    def evaluate(self, test_loader, device='cpu', save_dir="eval_results", means=None, stds=None, synth_fn=None, sr=44100):
        return CNNRegressor5.evaluate(self, test_loader, device=device, save_dir=save_dir, means=means, stds=stds, synth_fn=synth_fn, sr=sr)



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

    def fit(self, train_loader, val_loader=None, device='cpu', epochs=10, patience=10, print_every_batches=50, criterion=None, optimizer=None, history_path=None):
            self.to(device)
            criterion.to(device)

            history = {'total': [],'spec': [],'params': [], 'val_total' : []}

            # Variable que guarda el mejor modelo
            best_val_loss = float('inf')
            best_state_dict = None
            epochs_no_improve = 0

            # Entrenamiento — envuelto en try/except para permitir cancelación con Ctrl+C
            # y restaurar los mejores pesos vistos hasta el momento de la interrupción.
            try:
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
                        batch_params = batch_params.to(device)   # (B, n_params)

                        optimizer.zero_grad()                       # Reset gradientes
                        pred_params, pred_spec = self(batch_spec)   # Forward pass
                        loss_total, loss_spec, loss_sc, loss_params = criterion(pred_spec, batch_spec, pred_params, batch_params)
                        loss_total.backward()                       # Backprop
                        optimizer.step()                            # Descenso de gradientes

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
                                avg_sc_sofar = running_sc / max(1, n_batches)
                                avg_params_sofar = running_params / max(1, n_batches)
                                print(f" Epoch {epoch+1}/{epochs}  Batch {batch_idx+1}  Avg total so far: {avg_total_sofar:.6f}  Spec: {avg_spec_sofar:.6f}  SC: {avg_sc_sofar:.6f}  Params: {avg_params_sofar:.6f}")

                    avg_total = running_total / max(1, n_batches)
                    avg_spec = running_spec / max(1, n_batches)
                    avg_sc = running_sc / max(1, n_batches)
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
                            epochs_no_improve = 0  # Reiniciamos la paciencia
                            msg_val += " (*)"
                        else:
                            epochs_no_improve += 1

                    print(f"Epoch {epoch+1}/{epochs}  Avg total: {avg_total:.6f}  Spec: {avg_spec:.6f}  SC: {avg_sc:.6f}  Params: {avg_params:.6f}{msg_val}")

                    # --- Guardado en tiempo real ---
                    if history_path is not None:
                        with open(history_path, 'w') as f:
                            json.dump(history, f, indent=4)

                    # Bloque de Early Stopping
                    if epochs_no_improve >= patience:
                        print(f"\n[!] Early Stopping activado en la época {epoch+1}. El modelo no ha mejorado en las últimas {patience} épocas.")
                        break  # Rompe el bucle 'for epoch in range(epochs)'

            except KeyboardInterrupt:
                print("\n[!] Entrenamiento cancelado por el usuario.")

            if best_state_dict is not None:
                self.load_state_dict(best_state_dict)
                print("Se han restaurado los pesos de la mejor época.")

            return history  # Retorna: history dict con listas 'total', 'spec', 'params' (valores medios por época)


    # ──────────────────────────────────────────────────────────────────────────
    # EVALUACIÓN
    # ──────────────────────────────────────────────────────────────────────────
    # Métricas principales (audio): Mel L1 y MCD entre audio re-sintetizado con params reales vs predichos.
    #   Evitan el problema de no-inyectividad FM: params distintos → mismo sonido → métrica 0.
    # Métricas secundarias (params): MSE/RMSE/MAE por parámetro (diagnóstico, afectadas por no-inyectividad).
    # Métricas decoder (solo CNNRegressor5): L1 espectral entre espectrograma de entrada y reconstrucción.
    # synth_fn: función de síntesis FM (si None se omiten métricas de audio). sr: sample rate (default 44100).
    def evaluate(self, test_loader, device='cpu', save_dir="eval_results", means=None, stds=None, synth_fn=None, sr=44100):
        os.makedirs(save_dir, exist_ok=True)    # Crea la carpeta donde se guardarán los resultados
        self.to(device)                         
        self.eval()                             # Modo evaluación

        # Listas para acumular predicciones y targets de todos los batches
        preds_list, trues_list = [], []
        n_samples = 0

        # L1 del decoder: usa reduction='sum' para no sesgar por tamaño de batch
        # (si un batch tiene 16 muestras y otro 8, con 'mean' pesarían igual; con 'sum' + dividir al final, pesan proporcional)
        spec_loss_fn    = nn.L1Loss(reduction='sum')
        spec_loss_total = 0.0       # acumulador de la suma de errores L1
        spec_loss_count = 0         # acumulador del número total de elementos (pixeles de espectrograma)
        example_specs, example_pred_specs = [], []  # guardamos hasta 5 pares para plotear luego

        # FASE 1: Recopilar predicciones
        with torch.no_grad():
            for batch_spec, batch_params in test_loader:
                # Espectrogramas y parametros del batch
                batch_spec   = batch_spec.to(device)     
                batch_params = batch_params.to(device)   

                # Forward: el modelo devuelve (params, spec_reconstruido) o solo params
                out = self(batch_spec)
                if isinstance(out, (tuple, list)):           # CNNRegressor5 devuelve tupla
                    pred_params, pred_spec = out
                else:                                        # CNNRegressorSimple devuelve solo params
                    pred_params, pred_spec = out, None

                # Acumular predicciones y targets en CPU
                preds_list.append(pred_params.cpu())
                trues_list.append(batch_params.cpu())
                n_samples += pred_params.shape[0]            # contar muestras procesadas

                # Calcular L1 del decoder (solo si hay reconstrucción, es decir, CNNRegressor5)
                if pred_spec is not None:
                    spec_loss_total += spec_loss_fn(pred_spec, batch_spec).item()  # sumar error L1 de este batch
                    spec_loss_count += batch_spec.numel()                          # sumar nº de elementos (pixels)
                    # Guardar hasta 5 pares de espectrogramas para visualizar después
                    needed = 5 - len(example_specs)
                    if needed > 0:
                        example_specs.extend(batch_spec.detach().cpu()[:needed])
                        example_pred_specs.extend(pred_spec.detach().cpu()[:needed])

        # Concatenar todos los batches en un solo tensor (N, n_params)
        preds = torch.cat(preds_list, dim=0)   
        trues = torch.cat(trues_list, dim=0)    

        # Desnormalizar: revertir z-score
        means_t = torch.tensor(np.asarray(means, dtype=np.float32))
        stds_t  = torch.tensor(np.asarray(stds,  dtype=np.float32))
        preds_real = preds * stds_t + means_t   # predicciones en Hz, ratios, segundos, etc.
        trues_real = trues * stds_t + means_t    # targets en escala real

        param_names = (["carrier", "ratio", "index", "amp_att", "amp_sus", "amp_dec", "mod_att", "mod_dec"])

        # FASE 2: Métricas de parámetros 
        diff           = preds_real - trues_real               # Diferencia pred - real por muestra y parámetro
        mse_per_param  = (diff ** 2).mean(dim=0).numpy()       # MSE
        mae_per_param  = diff.abs().mean(dim=0).numpy()        # MAE
        rmse_per_param = np.sqrt(mse_per_param)                # RMSE

        # FASE 3: Métricas de audio: Re-sintetiza audio con params predichos y reales, y compara perceptualmente.
        mel_l1_list, mcd_list = [], []

        # Rangos válidos para clamp: coherentes con el dominio de entrenamiento (GEN_PARAMS)
        param_mins = np.array([100.0, 0.05, 1.0, 0.015, 0.015, 0.015, 0.01, 0.01], dtype=np.float32)
        param_maxs = np.array([2000.0, 2.0, 10.0, 1.9, 1.9, 1.9, 1.9, 1.9], dtype=np.float32)

        if synth_fn is not None:
            n_audio   = min(n_samples, 500)                             # Max 500 muestras (sintetizar es lento)
            rng       = np.random.default_rng(42)                       # Semilla fija
            indices   = rng.choice(n_samples, n_audio, replace=False)   # elegir n_audio índices aleatorios sin repetir
            n_skipped = 0                                               # contador de muestras que fallan al sintetizar
            print(f"Calculando métricas de audio sobre {n_audio} muestras...")

            for idx in indices:
                p = np.clip(preds_real[idx].numpy(), param_mins, param_maxs)   # predicciones clampeadas a rangos válidos
                t = trues_real[idx].numpy()                                    # targets
                try:
                    # Sintetizar 2s de audio FM con los params predichos y los reales
                    audio_pred, _ = synth_fn(*p.tolist(), duration=2.0, sr=sr)
                    audio_true, _ = synth_fn(*t.tolist(), duration=2.0, sr=sr)

                    # ─ Mel L1 ─
                    # Espectrograma mel de amplitud 
                    # L1 = media de |log1p(mel_pred) - log1p(mel_true)|
                    mel_p = librosa.feature.melspectrogram(y=audio_pred, sr=sr, n_fft=1024, hop_length=256, n_mels=128, power=1.0)
                    mel_t = librosa.feature.melspectrogram(y=audio_true, sr=sr, n_fft=1024, hop_length=256, n_mels=128, power=1.0)
                    mel_l1_list.append(float(np.mean(np.abs(np.log1p(mel_p) - np.log1p(mel_t)))))

                    # ─ MCD (Mel-Cepstral Distortion) ─
                    # Extrae 13 coeficientes MFCC y descarta el 0 (energía global) → quedan 12 de timbre
                    # Fórmula estándar MCD en dB: (10/ln10) * media( sqrt(2 * sum(diff²)) )
                    mfcc_p = librosa.feature.mfcc(y=audio_pred, sr=sr, n_mfcc=13)[1:]   
                    mfcc_t = librosa.feature.mfcc(y=audio_true, sr=sr, n_mfcc=13)[1:]
                    min_t  = min(mfcc_p.shape[1], mfcc_t.shape[1])       
                    diff_c = mfcc_p[:, :min_t] - mfcc_t[:, :min_t]       
                    mcd_list.append(float((10 / np.log(10)) * np.mean(np.sqrt(2 * np.sum(diff_c ** 2, axis=0)))))
                except Exception as e:
                    # Si falla la síntesis (p.ej. params extremos), descartamos la muestra
                    n_skipped += 1
                    if n_skipped <= 5:                                     
                        print(f"  [!] Muestra {idx} descartada: {e}")
            if n_skipped > 0:
                print(f"  Muestras descartadas: {n_skipped}/{n_audio}")

        # ── FASE 4: Imprimir resultados por consola ──
        print("\n=== Métricas de audio (principales) ===")
        print(f"  Mel L1  →  media: {np.mean(mel_l1_list):.4f}  |  mediana: {np.median(mel_l1_list):.4f}  |  std: {np.std(mel_l1_list):.4f}  (sobre {len(mel_l1_list)} muestras)")
        print(f"  MCD     →  media: {np.mean(mcd_list):.4f}  |  mediana: {np.median(mcd_list):.4f}  |  std: {np.std(mcd_list):.4f}  dB")

        print("\n=== Métricas de parámetros (afectadas por no-inyectividad FM) ===")
        for i, name in enumerate(param_names):
            print(f"  {name:8s} | MSE: {mse_per_param[i]:.6f} | RMSE: {rmse_per_param[i]:.6f} | MAE: {mae_per_param[i]:.6f}")

        # L1 decoder: dividimos la suma acumulada entre el total de elementos → media global exacta
        avg_spec_l1 = (spec_loss_total / spec_loss_count) if spec_loss_count > 0 else None
        if avg_spec_l1 is not None:
            print(f"\n  L1 espectral decoder: {avg_spec_l1:.6f}")
        else:
            print("\n  (sin reconstrucción del decoder — CNNRegressorSimple)")

        # ── FASE 5: Guardar CSV con predicciones vs valores reales ──
        # Cada fila es una muestra del test set, con columnas pred_carrier, true_carrier, pred_ratio, true_ratio, etc.
        df = pd.DataFrame({
            **{f"pred_{n}": preds_real[:, i].numpy() for i, n in enumerate(param_names)},
            **{f"true_{n}": trues_real[:, i].numpy() for i, n in enumerate(param_names)},
        })
        csv_path = os.path.join(save_dir, "preds_vs_trues.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nPredicciones guardadas en: {csv_path}")

        # ── FASE 6: Gráficas ──
        # Plot 1: Histogramas de distribución de Mel L1 y MCD
        # Permiten ver si hay outliers o si las métricas se concentran en un rango estrecho
        if mel_l1_list:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, values, title, xlabel in [
                (axes[0], mel_l1_list, "Distribución Mel L1", "Mel L1"),
                (axes[1], mcd_list,    "Distribución MCD",    "MCD (dB)"),
            ]:
                ax.hist(values, bins=40, edgecolor='black')                                              # histograma con 40 barras
                ax.axvline(np.mean(values), color='red', linestyle='--', label=f'media={np.mean(values):.3f}')  # línea vertical en la media
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Frecuencia")       # frecuencia = nº de muestras en cada barra
                ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "audio_metrics_dist.png"), dpi=150)
            plt.show()

        # Plot 2: Scatter true vs pred por cada parámetro
        # Si el modelo fuera perfecto, todos los puntos caerían sobre la diagonal roja
        plt.figure(figsize=(12, 4))
        for i, name in enumerate(param_names):
            plt.subplot(1, len(param_names), i + 1)
            plt.scatter(trues_real[:, i].numpy(), preds_real[:, i].numpy(), s=6, alpha=0.3)  # un punto por muestra
            mn = float(min(trues_real[:, i].min(), preds_real[:, i].min()))    # mínimo global para los ejes
            mx = float(max(trues_real[:, i].max(), preds_real[:, i].max()))    # máximo global para los ejes
            plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1)                  # diagonal = predicción perfecta
            plt.xlabel("Real")
            plt.ylabel("Pred")
            plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "scatter_params.png"), dpi=150)
        plt.show()

        # Plot 3: Comparación visual espectrograma entrada vs reconstrucción del decoder
        # Solo para CNNRegressor5 (CNNRegressorSimple no tiene decoder → example_specs estará vacío)
        if example_specs:
            n = len(example_specs)
            fig, axs = plt.subplots(n, 2, figsize=(10, 3 * n))    # n filas x 2 columnas (target | pred)
            if n == 1:
                axs = [axs]                                         # normalizar a lista de filas
            for row, (spec_t, spec_p) in enumerate(zip(example_specs, example_pred_specs)):
                for col, (tensor, title) in enumerate([(spec_t, f"Target #{row+1}"), (spec_p, f"Pred #{row+1}")]):
                    arr = tensor.squeeze(0).numpy()                               # quitar dim de canal → (F, T)
                    im = axs[row][col].imshow(arr, origin='lower', aspect='auto') # origin='lower' = frecuencias bajas abajo
                    axs[row][col].set_title(title)
                    axs[row][col].set_xlabel("tiempo")
                    axs[row][col].set_ylabel("freq")
                    fig.colorbar(im, ax=axs[row][col], format='%+.0f dB')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "spectrogram_example.png"), dpi=150)
            plt.show()

        print("Evaluación completada.")

        # ── Devolver diccionario con todas las métricas numéricas ──
        return {
            'param_names':     param_names,
            'mse_per_param':   mse_per_param,
            'rmse_per_param':  rmse_per_param,
            'mae_per_param':   mae_per_param,
            'avg_spec_l1':     float(avg_spec_l1) if avg_spec_l1 is not None else None,
            'mel_l1_mean':     float(np.mean(mel_l1_list))   if mel_l1_list else None,
            'mel_l1_median':   float(np.median(mel_l1_list)) if mel_l1_list else None,
            'mcd_mean':        float(np.mean(mcd_list))      if mcd_list    else None,
            'mcd_median':      float(np.median(mcd_list))    if mcd_list    else None,
            'n_samples':       int(n_samples),
            'n_audio_samples': len(mel_l1_list),
            'csv_path':        csv_path,
        }
