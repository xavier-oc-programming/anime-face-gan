import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf

from config import IMG_SIZE, LATENT_DIM, MODEL_DIR, SAMPLES_DIR


def _load_generator():
    model_path = MODEL_DIR / 'generator.keras'
    if not model_path.exists():
        raise FileNotFoundError(f'No trained generator found at {model_path}. Train first.')
    return tf.keras.models.load_model(model_path)


def generate_faces(
    n: int = 16,
    output_path: str = 'samples/generated.png',
    seed: int = None,
) -> str:
    """
    Load the trained generator and generate n faces.
    Save as a grid image to output_path.
    Returns output_path.

    Args:
        n: number of faces to generate
        output_path: where to save the grid
        seed: random seed for reproducibility (None = random)
    """
    generator = _load_generator()

    if seed is not None:
        tf.random.set_seed(seed)

    noise = tf.random.normal([n, LATENT_DIM])
    images = generator(noise, training=False).numpy()
    images = ((images + 1) * 127.5).astype(np.uint8)

    cols = min(4, n)
    rows = (n + cols - 1) // cols
    grid = np.zeros((rows * IMG_SIZE, cols * IMG_SIZE, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid[r * IMG_SIZE:(r + 1) * IMG_SIZE, c * IMG_SIZE:(c + 1) * IMG_SIZE] = img

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(output_path)
    return output_path


def generate_interpolation(
    steps: int = 10,
    output_path: str = 'samples/interpolation.png',
) -> str:
    """
    Generate a latent space interpolation between two random points.
    Shows how the generator smoothly transitions between two faces.
    Save as a 1xsteps grid image.

    Latent space interpolation is one of the most compelling demonstrations
    of GAN learning. If the generator has learned a meaningful latent space,
    interpolating between two noise vectors produces a smooth morphing sequence
    between two faces — not a blurry average, but a coherent transition. This
    would not be possible if the generator had simply memorised training images.
    """
    generator = _load_generator()

    start = tf.random.normal([1, LATENT_DIM])
    end = tf.random.normal([1, LATENT_DIM])

    alphas = np.linspace(0, 1, steps)
    noise_vectors = tf.concat(
        [start + alpha * (end - start) for alpha in alphas], axis=0
    )

    images = generator(noise_vectors, training=False).numpy()
    images = ((images + 1) * 127.5).astype(np.uint8)

    strip = np.concatenate(images, axis=1)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(strip).save(output_path)
    return output_path


if __name__ == '__main__':
    generate_faces(n=16)
    generate_interpolation(steps=10)
    print('Generated faces saved to samples/')
