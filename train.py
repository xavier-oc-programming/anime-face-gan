import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import (
    BatchNormalization, Conv2D, Conv2DTranspose, Dense, Dropout,
    Flatten, LeakyReLU, Reshape,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from config import (
    BATCH_SIZE, BETA_1, DATA_DIR, EPOCHS, IMG_SHAPE, IMG_SIZE,
    LABEL_SMOOTHING, LATENT_DIM, LEARNING_RATE_D, LEARNING_RATE_G, MODEL_DIR,
    N_SAMPLES, SAMPLES_DIR, SAVE_INTERVAL,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset():
    images = []
    skipped = 0
    paths = list(DATA_DIR.glob('*.jpg')) + list(DATA_DIR.glob('*.png'))
    for p in paths:
        try:
            img = Image.open(p).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
            images.append(np.array(img, dtype=np.float32))
        except Exception:
            skipped += 1
    if skipped:
        print(f'Skipped {skipped} corrupt/unreadable images', flush=True)
    images = np.stack(images)
    # GANs use tanh activation on the generator output, which produces values
    # in [-1, 1]. Normalising input images to the same range is required —
    # without it, the discriminator trivially learns to distinguish real from
    # fake by value range alone.
    images = (images / 127.5) - 1.0
    print(f'Loaded {len(images)} images, shape {images.shape}')
    dataset = tf.data.Dataset.from_tensor_slices(images)
    dataset = dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    print(f'Dataset: {len(list(dataset))} batches of {BATCH_SIZE}')
    return dataset


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def build_generator():
    model = Sequential([
        # Project and reshape noise to 4x4x512 feature maps.
        # 4x4 is the smallest spatial resolution — we build up from here.
        Dense(4 * 4 * 512, use_bias=False, input_shape=(LATENT_DIM,)),
        BatchNormalization(),
        # LeakyReLU(0.2): allows small negative gradients to flow back.
        # Standard ReLU causes "dying ReLU" in GAN generators.
        LeakyReLU(0.2),
        Reshape((4, 4, 512)),

        # Upsample 4x4 -> 8x8.
        # Conv2DTranspose: learned upsampling. Each transposed convolution
        # doubles the spatial dimensions — the inverse of a strided Conv2D.
        Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        # BatchNormalization: stabilises GAN training by reducing internal
        # covariate shift. Critical for GANs — without it training collapses.
        BatchNormalization(),
        LeakyReLU(0.2),

        # Upsample 8x8 -> 16x16
        Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(0.2),

        # Upsample 16x16 -> 32x32
        Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        LeakyReLU(0.2),

        # Upsample 32x32 -> 64x64.
        # tanh output: produces values in [-1, 1] matching the normalised input.
        Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same',
                        use_bias=False, activation='tanh'),
    ], name='generator')
    return model


# ---------------------------------------------------------------------------
# Discriminator
# ---------------------------------------------------------------------------

def build_discriminator():
    model = Sequential([
        # 64x64 -> 32x32.
        # No BatchNorm on first discriminator layer — standard DCGAN practice.
        # The first layer operates directly on pixel values; BatchNorm here
        # can prevent the discriminator from learning low-level statistics.
        Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=IMG_SHAPE),
        LeakyReLU(0.2),
        # Dropout(0.3): prevents discriminator from becoming too strong too fast,
        # which would starve the generator of useful gradients.
        Dropout(0.3),

        # 32x32 -> 16x16
        Conv2D(128, (4, 4), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(0.2),
        Dropout(0.3),

        # 16x16 -> 8x8
        Conv2D(256, (4, 4), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(0.2),
        Dropout(0.3),

        # 8x8 -> 4x4
        Conv2D(512, (4, 4), strides=(2, 2), padding='same'),
        BatchNormalization(),
        LeakyReLU(0.2),
        Dropout(0.3),

        Flatten(),
        # sigmoid output: probability that the input is real
        Dense(1, activation='sigmoid'),
    ], name='discriminator')
    return model


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def generator_loss(cross_entropy, fake_output):
    # The generator wants the discriminator to believe its output is real
    # (label=1). Generator loss is high when the discriminator correctly
    # identifies fakes.
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(cross_entropy, real_output, fake_output):
    # Real labels use LABEL_SMOOTHING (0.9) instead of 1.0 — prevents the
    # discriminator from becoming overconfident, keeping gradients useful for
    # the generator throughout training.
    real_labels = tf.ones_like(real_output) * LABEL_SMOOTHING
    real_loss = cross_entropy(real_labels, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


# ---------------------------------------------------------------------------
# Sample image saving
# ---------------------------------------------------------------------------

def save_sample_grid(generator, epoch, seed_noise, samples_dir):
    predictions = generator(seed_noise, training=False).numpy()
    # Denormalise from [-1, 1] to [0, 255]
    predictions = ((predictions + 1) * 127.5).astype(np.uint8)
    cols = 4
    rows = N_SAMPLES // cols
    grid = np.zeros((rows * IMG_SIZE, cols * IMG_SIZE, 3), dtype=np.uint8)
    for i, img in enumerate(predictions):
        r, c = divmod(i, cols)
        grid[r * IMG_SIZE:(r + 1) * IMG_SIZE, c * IMG_SIZE:(c + 1) * IMG_SIZE] = img
    out = samples_dir / f'epoch_{epoch:04d}.png'
    Image.fromarray(grid).save(out)
    return str(out)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model_dir=None, samples_dir=None):
    # model_dir and samples_dir can be overridden by the notebook to point to
    # persistent storage (e.g. Google Drive on Colab) so checkpoints survive
    # session crashes or disconnections.
    _model_dir   = Path(model_dir)   if model_dir   else MODEL_DIR
    _samples_dir = Path(samples_dir) if samples_dir else SAMPLES_DIR
    _model_dir.mkdir(parents=True, exist_ok=True)
    _samples_dir.mkdir(parents=True, exist_ok=True)

    # Build models inside train() so importing this module does not allocate
    # GPU memory — avoids kernel crashes when the module is imported on Colab.
    generator     = build_generator()
    discriminator = build_discriminator()
    gen_optimizer  = Adam(LEARNING_RATE_G, beta_1=BETA_1)
    disc_optimizer = Adam(LEARNING_RATE_D, beta_1=BETA_1)
    cross_entropy  = BinaryCrossentropy()

    @tf.function
    def train_step(real_images):
        noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = generator(noise, training=True)
            real_output = discriminator(real_images, training=True)
            fake_output = discriminator(fake_images, training=True)
            gen_loss  = generator_loss(cross_entropy, fake_output)
            disc_loss = discriminator_loss(cross_entropy, real_output, fake_output)
        gen_grads  = gen_tape.gradient(gen_loss,  generator.trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        gen_optimizer.apply_gradients(zip(gen_grads,  generator.trainable_variables))
        disc_optimizer.apply_gradients(zip(disc_grads, discriminator.trainable_variables))
        return gen_loss, disc_loss

    print('Loading dataset...', flush=True)
    dataset = load_dataset()

    # Using a fixed seed means the same N_SAMPLES random vectors are used at
    # every save interval. This makes it possible to track how the same
    # "latent points" improve over training — the same face positions becoming
    # sharper and more coherent over epochs.
    seed = tf.random.normal([N_SAMPLES, LATENT_DIM], seed=42)

    training_log = {'gen_loss': [], 'disc_loss': []}
    start = time.time()

    generator.summary()
    discriminator.summary()

    for epoch in range(1, EPOCHS + 1):
        epoch_gen_losses  = []
        epoch_disc_losses = []

        for batch in dataset:
            g_loss, d_loss = train_step(batch)
            epoch_gen_losses.append(float(g_loss))
            epoch_disc_losses.append(float(d_loss))

        mean_g = float(np.mean(epoch_gen_losses))
        mean_d = float(np.mean(epoch_disc_losses))
        training_log['gen_loss'].append(mean_g)
        training_log['disc_loss'].append(mean_d)

        elapsed = time.time() - start
        print(f'Epoch {epoch:03d}/{EPOCHS} | G: {mean_g:.4f} | D: {mean_d:.4f} | {elapsed:.0f}s', flush=True)

        if epoch % SAVE_INTERVAL == 0:
            path = save_sample_grid(generator, epoch, seed, _samples_dir)
            generator.save(_model_dir / f'generator_epoch_{epoch:03d}.keras')
            # Save the log at every checkpoint so a crash mid-training still
            # leaves a complete record of all completed epochs.
            with open(_model_dir / 'training_log.json', 'w') as f:
                json.dump(training_log, f, indent=2)
            print(f'  Checkpoint saved → epoch {epoch} | {path}')

    # Final saves
    generator.save(_model_dir / 'generator.keras')
    discriminator.save(_model_dir / 'discriminator.keras')
    save_sample_grid(generator, EPOCHS, seed, _samples_dir)
    (_samples_dir / f'epoch_{EPOCHS:04d}.png').rename(_samples_dir / 'final_samples.png')

    with open(_model_dir / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)

    total = time.time() - start
    print(f'\nTraining complete in {total/60:.1f} minutes')
    print(f'Final G loss: {training_log["gen_loss"][-1]:.4f}')
    print(f'Final D loss: {training_log["disc_loss"][-1]:.4f}')
    print(f'Models saved to {_model_dir}')
    print(f'Samples saved to {_samples_dir}')


if __name__ == '__main__':
    train()
    # Shut down the Azure ML Compute Instance immediately after training.
    # This prevents idle billing — the machine terminates the moment training
    # completes. Only runs when train.py is executed directly (not when imported).
    # Remove these two lines for local training.
    import subprocess
    subprocess.run(['sudo', 'shutdown', '-h', 'now'])
