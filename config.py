from pathlib import Path

# Paths
DATA_DIR = Path('data/anime_faces')
MODEL_DIR = Path('models')
PLOTS_DIR = Path('plots')
SAMPLES_DIR = Path('samples')

# Image
IMG_SIZE = 64
IMG_CHANNELS = 3
IMG_SHAPE = (64, 64, 3)

# GAN architecture
LATENT_DIM = 128
# The latent space dimension — the size of the random noise vector fed to the
# generator. 128 gives enough capacity for 64x64 colour faces without being
# too large to sample efficiently.

# Training
EPOCHS = 100
BATCH_SIZE = 128
LEARNING_RATE_G = 0.0002
LEARNING_RATE_D = 0.0001
# LEARNING_RATE_D is half of LEARNING_RATE_G. The baseline run showed discriminator
# dominance (D loss → 0, G loss rising to 5) — slowing the discriminator gives
# the generator more useful gradient signal throughout training.

LABEL_SMOOTHING = 0.9
# Real labels use 0.9 instead of 1.0. Prevents the discriminator from becoming
# overconfident on real images, which starves the generator of gradients.
BETA_1 = 0.5
# BETA_1=0.5 is the standard Adam momentum for GANs. The default (0.9) causes
# training instability in GANs because high momentum causes the discriminator
# to overshoot.

SAVE_INTERVAL = 10
# Save sample images and model weights every N epochs. Allows inspecting
# training progress without waiting for full completion.

N_SAMPLES = 16  # number of faces to generate in each sample grid

# Azure ML
AZURE_ML_WORKSPACE = 'anime-face-gan-ws'
AZURE_ML_RESOURCE_GROUP = 'anime-face-gan-rg'
AZURE_ML_COMPUTE = 'anime-face-compute'
AZURE_ML_LOCATION = 'westeurope'

# Azure App Service (inference)
AZURE_APP_NAME = 'anime-face-gan-xoc'
AZURE_RESOURCE_GROUP = 'anime-face-app-rg'
AZURE_LOCATION = 'westeurope'
