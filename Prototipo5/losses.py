import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# MULTI-SCALE SPECTRAL LOSS
# ──────────────────────────────────────────────────────────────────────────────
# Función de pérdida multi-escala inspirada en DDSP (Engel et al., 2020).
#
# DDSP aplica STFTs con 6 tamaños de ventana distintos (2048..64) sobre audio crudo.
# Nosotros, al tener espectrogramas en lugar de audio, imitamos ese efecto aplicando average pooling temporal
# sobre el espectrograma mel con valores crecientes (1, 2, 4, 8, 16, 32):
#
#   factor=1  → resolución completa (ventana pequeña)
#   factor=32 → 32 frames colapsados en 1 (ventana grande)
#
# Por cada escala s:
#   L_s = ||mag_pred_s - mag_tgt_s||_1          ← L1 sobre magnitudes lineales (como DDSP)
#       + alpha * ||dB_pred_s  - dB_tgt_s||_1   ← L1 sobre log-magnitudes   (como DDSP)
#
# L_total = media(L_s para todo s)  +  param_weight * SmoothL1(params)

# ──────────────────────────────────────────────────────────────────────────────
# HYBRID LOSS
# ──────────────────────────────────────────────────────────────────────────────
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


class MultiScaleSpectralLoss(nn.Module):
    def __init__(self,
                 time_scales=(1, 2, 4, 8, 16, 32),  # factores de pooling temporal (de fino a grueso)
                 alpha=1.0,                            # peso del término log-magnitud
                 param_weight=0.05,
                 eps=1e-8):
        super().__init__()
        self.time_scales  = time_scales
        self.alpha        = alpha
        self.param_weight = param_weight
        self.eps          = eps
        self.param_loss   = nn.SmoothL1Loss()

    def _scale_loss(self, pred_db, tgt_db, pool_factor):
        """Pérdida a una escala temporal dada."""
        if pool_factor > 1:
            # Colapsar 'pool_factor' frames temporales en uno (dimensión W)
            pred_db = F.avg_pool2d(pred_db, kernel_size=(1, pool_factor), stride=(1, pool_factor))
            tgt_db  = F.avg_pool2d(tgt_db,  kernel_size=(1, pool_factor), stride=(1, pool_factor))

        # Convertir de dB a magnitud lineal: mag = 10^(dB/20)
        pred_mag = 10 ** (pred_db / 20)
        tgt_mag  = 10 ** (tgt_db  / 20)

        # Término lineal (L1 sobre amplitudes)
        l_lin = F.l1_loss(pred_mag, tgt_mag)

        # Término logarítmico: los dB ya son escala log, se usan directamente
        l_log = F.l1_loss(pred_db, tgt_db)

        return l_lin, l_log

    def forward(self, pred_spec, tgt_spec, pred_params, tgt_params):
        lin_losses, log_losses = [], []
        for s in self.time_scales:
            l_lin, l_log = self._scale_loss(pred_spec, tgt_spec, s)
            lin_losses.append(l_lin)
            log_losses.append(l_log)

        # Media sobre escalas
        avg_lin = sum(lin_losses) / len(lin_losses)
        avg_log = sum(log_losses) / len(log_losses)
        loss_multiscale = avg_lin + self.alpha * avg_log

        loss_params = self.param_loss(pred_params, tgt_params)
        total = loss_multiscale + self.param_weight * loss_params

        # Devuelve 4 valores para compatibilidad con CNNRegressor5.fit()
        # (total, componente_lineal, componente_log, params)
        return total, avg_lin.detach(), avg_log.detach(), loss_params.detach()


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
        return num / den                                         # Error relativo

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
