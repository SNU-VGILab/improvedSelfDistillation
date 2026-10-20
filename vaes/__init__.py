from vaes.identity import IDAE
from vaes.sdvae import SD_VAE
from vaes.vavae import VA_VAE


VAES = {
    "vavae_f16d32": VA_VAE,
    "sdvae_f8c4": SD_VAE,
    "sdvae_ema_f8c4": SD_VAE,
    "idae_f1c3": IDAE,
}
