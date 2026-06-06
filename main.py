import base64
import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

from config import IMG_SIZE, LATENT_DIM, MODEL_DIR, N_SAMPLES, SAMPLES_DIR

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title='Anime Face GAN',
    description=(
        'Generates anime faces using a DCGAN trained on 63,000 images. '
        'POST /generate returns a grid of freshly generated faces — each unique, '
        'none from the training set. Trained on Azure ML Compute Instance.'
    ),
    version='1.0.0',
)

templates = Jinja2Templates(directory='templates')

SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
app.mount('/samples', StaticFiles(directory=str(SAMPLES_DIR)), name='samples')

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

generator = None
MODEL_LOADED = False

model_path = MODEL_DIR / 'generator.keras'
if model_path.exists():
    generator = tf.keras.models.load_model(model_path)
    MODEL_LOADED = True
else:
    print(f'Warning: {model_path} not found. API will return 503 for generation endpoints.')


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    n: int = Field(16, ge=1, le=64, description='Number of faces to generate (1-64)')
    seed: Optional[int] = Field(None, description='Random seed for reproducibility (optional)')


class GenerateResponse(BaseModel):
    image_base64: str
    n_generated: int
    latent_dim: int
    model_info: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    latent_dim: int
    img_size: int


class InterpolationRequest(BaseModel):
    steps: int = Field(10, ge=2, le=30)
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_grid(n: int, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        tf.random.set_seed(seed)
    noise = tf.random.normal([n, LATENT_DIM])
    images = generator(noise, training=False).numpy()
    images = ((images + 1) * 127.5).astype(np.uint8)
    cols = 4
    rows = (n + cols - 1) // cols
    grid = np.zeros((rows * IMG_SIZE, cols * IMG_SIZE, 3), dtype=np.uint8)
    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        grid[r * IMG_SIZE:(r + 1) * IMG_SIZE, c * IMG_SIZE:(c + 1) * IMG_SIZE] = img
    return grid


def _grid_to_png_bytes(grid: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(grid).save(buf, format='PNG')
    return buf.getvalue()


def _grid_to_base64(grid: np.ndarray) -> str:
    return base64.b64encode(_grid_to_png_bytes(grid)).decode('utf-8')


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, 'index.html')


@app.get('/health', response_model=HealthResponse)
async def health():
    return HealthResponse(
        status='ok',
        model_loaded=MODEL_LOADED,
        latent_dim=LATENT_DIM,
        img_size=IMG_SIZE,
    )


@app.post('/generate', response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail='Model not loaded. Train first.')
    grid = _generate_grid(req.n, req.seed)
    return GenerateResponse(
        image_base64=_grid_to_base64(grid),
        n_generated=req.n,
        latent_dim=LATENT_DIM,
        model_info=f'DCGAN generator — {LATENT_DIM}d latent space, 64x64 RGB output',
    )


@app.get('/generate/single')
async def generate_single():
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail='Model not loaded. Train first.')
    grid = _generate_grid(1)
    png = _grid_to_png_bytes(grid)
    return Response(content=png, media_type='image/png')


@app.post('/generate/interpolation')
async def generate_interpolation(req: InterpolationRequest):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail='Model not loaded. Train first.')
    if req.seed is not None:
        tf.random.set_seed(req.seed)
    start = tf.random.normal([1, LATENT_DIM])
    end = tf.random.normal([1, LATENT_DIM])
    alphas = np.linspace(0, 1, req.steps)
    noise_vectors = tf.concat(
        [start + alpha * (end - start) for alpha in alphas], axis=0
    )
    images = generator(noise_vectors, training=False).numpy()
    images = ((images + 1) * 127.5).astype(np.uint8)
    strip = np.concatenate(images, axis=1)

    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(strip).save(buf, format='PNG')
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    return {'image_base64': encoded, 'steps': req.steps}


@app.get('/api/samples')
async def list_samples():
    samples = []
    if SAMPLES_DIR.exists():
        for p in sorted(SAMPLES_DIR.glob('*.png')):
            name = p.stem
            if name == 'final_samples':
                epoch = 'final'
            elif name.startswith('epoch_'):
                try:
                    epoch = int(name.split('_')[1])
                except ValueError:
                    epoch = name
            else:
                epoch = name
            samples.append({
                'filename': p.name,
                'epoch': epoch,
                'url': f'/samples/{p.name}',
            })
    return samples
