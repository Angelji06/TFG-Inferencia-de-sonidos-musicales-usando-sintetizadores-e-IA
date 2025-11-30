import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from torchsummary import summary 

# -------------------------
# HybridLoss (espectrograma + parámetros)
# -------------------------
class HybridLoss(nn.Module):
    def __init__(self, param_weight=0.1, spec_weight=1.0):
        """
        param_weight: peso multiplicador para la pérdida de parámetros (MSE)
        spec_weight: peso multiplicador para la pérdida de espectrograma (L1)
        """
        super().__init__()
        self.param_weight = float(param_weight)
        self.spec_weight = float(spec_weight)
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
    
    def forward(self, pred_spec, target_spec, pred_params, target_params):
        """
        pred_spec, target_spec: shape (B, 1, H, W) o (B, C, H, W)
        pred_params, target_params: shape (B, n_params)
        """
        # asegurarse de que shapes son compatibles
        if pred_spec.shape != target_spec.shape:
            # intentar redimensionar si pred tiene un canal distinto (ej: B,H,W)
            # pero lo ideal es que el modelo devuelva exactamente la misma forma
            raise ValueError(f"pred_spec.shape {pred_spec.shape} != target_spec.shape {target_spec.shape}")
        
        loss_spec = self.l1(pred_spec, target_spec) * self.spec_weight
        loss_params = self.mse(pred_params, target_params) * self.param_weight
        total = loss_spec + loss_params
        return total, loss_spec, loss_params

# -------------------------
# SmallCNNRegressor
# Un ejemplo simple: encoder -> latente -> two heads:
#   - head params: regresión (3 valores)
#   - head spec: decoder (reconstrucción del espectrograma)
# -------------------------
class SmallCNNRegressor(nn.Module):
    def __init__(self, n_params=3, input_channels=1, base_filters=32):
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
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_params = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters*8, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_params)
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


