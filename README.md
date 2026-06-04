# anime-face-gan

Generates anime faces using a Deep Convolutional GAN trained on 63,000 images.
Every face is generated from random noise — none of them exist. Trained on
Azure ML Compute Instance. Inference served via FastAPI on Azure App Service.

**Live demo → [anime-face-gan-xoc.azurewebsites.net](https://anime-face-gan-xoc.azurewebsites.net)**
&nbsp;&nbsp;·&nbsp;&nbsp;
**API docs → [/docs](https://anime-face-gan-xoc.azurewebsites.net/docs)**
&nbsp;&nbsp;·&nbsp;&nbsp;
**Notebook → [notebook.ipynb](notebook.ipynb)**

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Keras](https://img.shields.io/badge/Keras-DCGAN-red)
![Azure ML](https://img.shields.io/badge/Azure_ML-Compute_Instance-0078D4)
![Azure App Service](https://img.shields.io/badge/Azure-App_Service-0078D4)

---

## 0. Prerequisites

- Python 3.11+
- `pip install -r requirements.txt`
- Trained `models/generator.keras` (committed after training — see section 2)

---

## 1. Quick Start — Inference Only

```bash
git clone https://github.com/xavier-oc-programming/anime-face-gan
cd anime-face-gan
pip install -r requirements.txt
uvicorn main:app --reload
# Open http://localhost:8000
```

The API loads `models/generator.keras` at startup. If the model file is present, every endpoint is live immediately — no retraining required.

Generate faces from the command line:

```bash
python generate.py
# Saves samples/generated.png (4×4 grid) and samples/interpolation.png
```

---

## 2. Training

Three options depending on available compute.

### Option A: Azure ML Compute Instance (Recommended)

Full setup in [azure_ml_setup.md](azure_ml_setup.md).

```bash
# On the compute instance
python train.py
# Instance shuts down automatically when training completes
```

Expected cost: $2–5. The last line of `train.py` issues `sudo shutdown -h now` — remove it for local training.

### Option B: Local CPU

```bash
# Remove the subprocess shutdown line from train.py first
python train.py
```

Expect 8–12 hours for 100 epochs. No cloud costs.

### Option C: Free GPU (Google Colab or Kaggle)

- **Google Colab**: upload dataset, run `train.py` (remove shutdown line), download `generator.keras`
- **Kaggle**: dataset already at `splcher/animefacedataset` — 30 free GPU hours/week

See [azure_ml_setup.md](azure_ml_setup.md) for step-by-step instructions.

---

## 3. Project Structure

```
anime-face-gan/
├── config.py               # Single source of truth for all constants
├── train.py                # DCGAN training script (runs on Azure ML)
├── generate.py             # Standalone generation script
├── main.py                 # FastAPI inference application
├── notebook.ipynb          # Architecture walkthrough and training analysis
├── azure_ml_setup.md       # Azure ML training setup guide
├── Dockerfile              # Inference container
├── startup.txt             # Azure App Service startup command
├── requirements.txt
├── portfolio.yaml
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions — tests inference API only
├── templates/
│   └── index.html          # Demo frontend (dark theme, inline CSS/JS)
├── tests/
│   └── test_api.py         # pytest — all tests pass with or without model
├── models/
│   ├── generator.keras     # Committed after training (~50MB)
│   ├── discriminator.keras # Committed after training
│   └── training_log.json   # gen_loss and disc_loss per epoch
├── samples/
│   ├── epoch_0010.png      # Sample grids at every 10 epochs
│   ├── ...
│   └── final_samples.png
└── data/
    └── anime_faces/        # Gitignored — download separately
```

---

## 4. Architecture

### Generator

Maps a 128-dimensional noise vector to a 64×64 RGB image via transposed convolutions:

| Layer | Output Shape | Notes |
|-------|-------------|-------|
| Dense → Reshape | 4 × 4 × 512 | Projects noise to spatial volume |
| Conv2DTranspose 256 | 8 × 8 × 256 | Learned 2× upsampling |
| Conv2DTranspose 128 | 16 × 16 × 128 | |
| Conv2DTranspose 64 | 32 × 32 × 64 | |
| Conv2DTranspose 3 (tanh) | 64 × 64 × 3 | Output image in [−1, 1] |

BatchNormalization and LeakyReLU(0.2) after every layer except the output.

### Discriminator

Maps a 64×64 RGB image to a real/fake probability via strided convolutions:

| Layer | Output Shape | Notes |
|-------|-------------|-------|
| Conv2D 64 | 32 × 32 × 64 | No BatchNorm on first layer |
| Conv2D 128 | 16 × 16 × 128 | |
| Conv2D 256 | 8 × 8 × 256 | |
| Conv2D 512 | 4 × 4 × 512 | |
| Flatten → Dense (sigmoid) | 1 | Probability real |

BatchNormalization, LeakyReLU(0.2), and Dropout(0.3) after every layer except the first and last.

---

## 5. Training Dynamic

The generator and discriminator have opposing objectives expressed as a minimax game:

```
min_G max_D [ E[log D(x)] + E[log(1 − D(G(z)))] ]
```

where `x` is a real image, `z` is random noise, `G(z)` is a generated image, and `D(x)` is the discriminator's estimate that `x` is real.

**Loss functions:**

```python
gen_loss  = cross_entropy(ones,  fake_output)   # generator wants D to say "real"
disc_loss = cross_entropy(ones,  real_output) \
          + cross_entropy(zeros, fake_output)   # discriminator wants correct labels
```

**Optimizers:** Adam with `lr=0.0002`, `beta_1=0.5` for both networks.

---

## 6. Training Stability

GAN training failure modes and how the architecture prevents them:

| Failure | Sign | Fix Applied |
|---------|------|-------------|
| Mode collapse | All outputs look identical | Dropout(0.3) in discriminator |
| Discriminator dominance | D loss → 0, G loss spikes | Equal learning rates; Dropout slows D |
| Oscillation | Both losses fluctuate without trend | BETA_1=0.5; BatchNorm throughout |

---

## 7. Latent Space Interpolation

```bash
python generate.py  # runs generate_interpolation() alongside generate_faces()
```

Interpolating linearly between two noise vectors produces a smooth morphing sequence. A model that had memorised training images would jump discontinuously — smooth interpolation is evidence of a learned continuous mapping from noise to face space.

---

## 8. Azure ML Training Setup

Brief overview:

1. Create workspace: `az ml workspace create --name anime-face-gan-ws --resource-group anime-face-gan-rg`
2. Create compute: `az ml compute create --name anime-face-compute --type ComputeInstance --size Standard_NC6`
3. Set 15-minute idle shutdown in Azure ML Studio
4. Upload dataset, run `python train.py`
5. Instance shuts down automatically on completion

Full guide: [azure_ml_setup.md](azure_ml_setup.md)

---

## 9. Cost Breakdown

| Item | Cost |
|------|------|
| GPU training (Standard_NC6, ~2hr) | ~$2 |
| CPU training (Standard_DS3_v2, ~10hr) | ~$2.50 |
| Azure App Service F1 (inference) | Free tier |

Cost controls:
- `train.py` last line: `subprocess.run(['sudo', 'shutdown', '-h', 'now'])`
- 15-minute idle shutdown in Azure ML Studio
- $5 budget alert in Azure Cost Management

---

## 10. Deployment

```bash
az group create --name anime-face-app-rg --location westeurope
az appservice plan create --name anime-face-app-plan --resource-group anime-face-app-rg --sku B1 --is-linux
az webapp create --name anime-face-gan-xoc --resource-group anime-face-app-rg --plan anime-face-app-plan --runtime "PYTHON:3.11"
az webapp config set --name anime-face-gan-xoc --resource-group anime-face-app-rg --startup-file "gunicorn main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000 --timeout 600"
az webapp config appsettings set --name anime-face-gan-xoc --resource-group anime-face-app-rg --settings SCM_DO_BUILD_DURING_DEPLOYMENT=true
zip -r deploy.zip . -x "*.git*" -x "venv/*" -x "__pycache__/*" -x "*.ipynb_checkpoints*" -x "data/*"
az webapp deployment source config-zip --name anime-face-gan-xoc --resource-group anime-face-app-rg --src deploy.zip
```

Scale down to F1 (free tier) via Portal after creation if desired.

---

## 11. CI/CD

GitHub Actions runs `pytest tests/ -v` on every push to `main`. Tests are written to pass with or without `generator.keras` — generation endpoints return 503 gracefully when the model is missing.

---

## 12. Design Decisions

**Why anime faces over MNIST or CIFAR-10**
Anime faces have low intra-class variance — all images share a consistent art style, similar face geometry, and uniform scale. This means the generator converges faster and produces cleaner results than on real-face datasets, where variation in lighting, pose, and background creates a much harder distribution to learn. The 63,000-image dataset is large enough to train a 64×64 GAN to convergence.

**Why LATENT_DIM=128**
128 dimensions gives the generator enough capacity to represent the variation in the training data — different face shapes, hair colours, expressions — without being so large that sampling becomes inefficient or training slows. Lower values (64) produce less variety; higher values (256+) add little quality benefit at 64×64 resolution.

**Why BETA_1=0.5 for Adam**
The default Adam momentum (0.9) causes training instability in GANs. High momentum means the optimizer continues in a direction for many steps even after the loss surface changes — in GANs, where the discriminator and generator are constantly shifting each other's loss landscape, this causes the discriminator to overshoot and the training to oscillate. BETA_1=0.5 reduces this momentum, making the optimizer more responsive to the current gradient.

**Why Dropout in the discriminator**
Without Dropout, the discriminator can become too accurate too quickly. When the discriminator is near-perfect, the generator receives gradients close to zero — it gets no useful signal about how to improve. Dropout(0.3) keeps the discriminator imperfect enough that the generator always has something to learn from. It is specifically in the discriminator, not the generator, because the problem is discriminator dominance, not generator dominance.

**Why train on Azure ML rather than locally**
Training 100 epochs on 63,000 64×64 colour images takes 8–12 hours on CPU. Azure ML Compute Instance provides managed cloud compute: a clean environment, no interference with local work, and a GPU option that reduces training time to ~2 hours. The training infrastructure (Azure ML Compute Instance) is also distinct from the inference infrastructure (Azure App Service) — the same separation used in production ML systems.

**Why the subprocess shutdown line is the last line of train.py**
The compute instance runs at $0.25–0.90/hr. If training finishes at 2am and is not manually stopped, the idle cost accumulates until morning. The shutdown line eliminates that possibility — the instance terminates the instant the Python process exits normally. It is the last line so it runs after all model saving is complete. A 15-minute idle shutdown and a $5 budget alert provide additional layers of protection.

---

## 13. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| tensorflow | >=2.15 | DCGAN training and inference |
| numpy | >=1.24,<2.0 | Array operations, image processing |
| matplotlib | >=3.7 | Training visualisation in notebook |
| pillow | >=10.0 | Image loading and grid assembly |
| fastapi | >=0.110 | REST API framework |
| uvicorn | >=0.27 | ASGI server |
| gunicorn | >=21.0 | Production process manager |
| pydantic | >=2.0 | Request/response validation |
| jupyter | >=1.0 | Architecture walkthrough notebook |
| pytest | >=7.0 | API tests |
| httpx | >=0.27 | TestClient dependency for pytest |
| python-multipart | >=0.0.9 | Form data support for FastAPI |
