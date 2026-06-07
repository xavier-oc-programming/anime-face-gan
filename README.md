---
title: Anime Face GAN
emoji: 🎌
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
---

# anime-face-gan

Generates anime faces using a Deep Convolutional GAN trained on 63,000 images.
Every face is generated from random noise — none of them exist. Trained on
Azure ML Compute Instance. Inference served via FastAPI on Hugging Face Spaces.

**Live demo → [xavier-oc-machinelearn-anime-face-gan.hf.space](https://xavier-oc-machinelearn-anime-face-gan.hf.space)**
&nbsp;&nbsp;·&nbsp;&nbsp;
**API docs → [/docs](https://xavier-oc-machinelearn-anime-face-gan.hf.space/docs)**
&nbsp;&nbsp;·&nbsp;&nbsp;
**Notebook → [notebook.ipynb](notebook.ipynb)**

![Python 3.11](https://img.shields.io/badge/Python-3.11-blue)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-inference-lightgrey)
![Keras](https://img.shields.io/badge/Keras-DCGAN-red)
![Azure ML](https://img.shields.io/badge/Azure_ML-Compute_Instance-0078D4)
![Hugging Face Spaces](https://img.shields.io/badge/Hugging_Face-Spaces-yellow)

---

## 0. Prerequisites

- Python 3.11+
- `pip install -r requirements.txt`
- Kaggle account (free) — for dataset download
- Trained `models/generator.onnx` — committed after training, or train it yourself (see section 2)

---

## 1. Quick Start — Inference Only

```bash
git clone https://github.com/xavier-oc-programming/anime-face-gan
cd anime-face-gan
pip install -r requirements.txt
uvicorn main:app --reload
# Open http://localhost:7860
```

The API loads `models/generator.onnx` at startup. If the model file is present every endpoint is live immediately — no retraining required.

Generate faces from the command line:

```bash
python generate.py
# Saves samples/generated.png (4×4 grid) and samples/interpolation.png
```

---

## 2. Dataset

The Anime Face Dataset (~63,000 images, ~330MB) is downloaded via kagglehub. Images go into `data/anime_faces/` which is gitignored — the folder is tracked so it is ready to receive images on clone.

**Set up Kaggle credentials first (one-time):**

Go to [kaggle.com](https://kaggle.com) → profile → Settings → API → **Create New Token**. This downloads `kaggle.json`.

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

Or use environment variables (Colab, CI, other machines):

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

**Download via the notebook** — run the download cell in [notebook.ipynb](notebook.ipynb), or from the command line:

```python
import kagglehub, shutil
from pathlib import Path
src = Path(kagglehub.dataset_download('splcher/animefacedataset'))
for p in src.rglob('*.jpg'):
    shutil.copy(p, 'data/anime_faces/' + p.name)
```

---

## 3. Training

Four options depending on available compute. The notebook supports all of them.

### Option A: Notebook (Colab / Kaggle / local)

Open [notebook.ipynb](notebook.ipynb) and run the training cell. It calls `train()` directly — the Azure ML shutdown line does **not** fire when training is run from the notebook.

```
Google Colab T4 GPU  — ~1-2 hours for 100 epochs   (free)
Kaggle P100 GPU      — ~1-2 hours for 100 epochs   (free, 30hr/week)
Local CPU            — ~8-12 hours for 100 epochs
```

**Why local machines use CPU, not GPU**
Consumer Macs have no NVIDIA GPU, and TensorFlow GPU support requires CUDA — an NVIDIA-only toolkit. Apple Silicon (M1/M2/M3) has a GPU but it uses Metal, not CUDA; TensorFlow's Metal plugin exists but is experimental and not worth the setup friction for a one-off training run. If you have a Windows machine with an NVIDIA GPU, TensorFlow GPU will work but requires matching the exact CUDA + cuDNN versions to your TensorFlow version. For most users, Colab or Kaggle is the path of least resistance for GPU training.

**Google Colab (free T4 GPU):**
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File → Open notebook → GitHub → paste `https://github.com/xavier-oc-programming/anime-face-gan` → open `notebook.ipynb`
3. Runtime → Change runtime type → **T4 GPU** → Save
4. If the dataset download cell fails with an authentication error: click the **key icon** in the Colab left sidebar → add `KAGGLE_USERNAME` and `KAGGLE_KEY` (from kaggle.com → Settings → API → Create New Token). If you already have `~/.kaggle/kaggle.json` locally, kagglehub found it automatically — Colab has no such file so credentials must be provided explicitly.
5. Run all cells in order. When the training cell runs, a popup will appear: **"Permit this notebook to access your Google Drive files?"** — click **Connect to Google Drive** and complete the sign-in. This mounts your Drive so checkpoints are saved to `My Drive/anime-face-gan/` every 10 epochs. If Colab disconnects mid-training, you lose at most the current 10-epoch interval.
6. Download `generator.keras` and `training_log.json` from Google Drive → `anime-face-gan/models/`
7. Commit both files to the repo locally

**Kaggle Notebooks (free P100 GPU, 30hr/week):**
1. Go to [kaggle.com/code](https://kaggle.com/code) → New Notebook
2. File → Import Notebook → GitHub → paste `https://github.com/xavier-oc-programming/anime-face-gan` → import `notebook.ipynb`
3. Settings (right panel) → Accelerator → **GPU P100**
4. Add Dataset: search `splcher/animefacedataset` → Add (the training cell overrides `DATA_DIR` to the Kaggle input path automatically — skip the download cell)
5. Run all cells — checkpoints save to `/kaggle/working/` every 10 epochs
6. **Save Version** (top right) after training completes → output files appear under the Output tab → download `generator.keras` and `training_log.json`
7. Commit both files to the repo locally

### Option B: Azure ML Compute Instance

Full setup in [azure_ml_setup.md](azure_ml_setup.md).

```bash
# On the compute instance
python train.py
# Instance shuts down automatically when training completes
```

`train.py` ends with `subprocess.run(['sudo', 'shutdown', '-h', 'now'])` inside `if __name__ == '__main__':` — the instance terminates the moment the script exits. Expected cost: $2–5.

### Option C: Local CPU

```bash
python train.py
# Remove the shutdown lines at the bottom first, or they will shut down your machine
```

Expect 8–12 hours for 100 epochs.

### After training

Export the trained `.keras` checkpoint to ONNX for inference (one-time, run locally):

```bash
pip install tf2onnx
python -m tf2onnx.convert --saved-model models/generator_saved_model --output models/generator.onnx --opset 13
```

Or export via Python (see `models/` for the committed result):

```python
import tensorflow as tf, tf2onnx, onnx
model = tf.keras.models.load_model('models/generator.keras')
model.export('models/generator_saved_model')
# then run the tf2onnx CLI command above
onnx.save(model_proto, 'models/generator.onnx')
```

Commit the outputs so the inference API and notebook work without retraining:

```bash
git add models/generator.onnx models/discriminator.keras models/training_log.json samples/
git commit -m "Add trained generator — 100 epochs"
git push
```

---

## 4. Project Structure

```
anime-face-gan/
├── config.py               # Single source of truth for all constants
├── train.py                # DCGAN training script (Azure ML or local)
├── generate.py             # Standalone generation and interpolation script
├── main.py                 # FastAPI inference application
├── notebook.ipynb          # Architecture walkthrough, dataset download, training, analysis
├── azure_ml_setup.md       # Azure ML training setup guide
├── Dockerfile              # Inference container (CPU-only, no training)
├── requirements.txt
├── portfolio.yaml
├── .gitignore
├── .gitattributes          # Git LFS tracking for model and image files
├── .github/
│   └── workflows/
│       └── ci.yml          # GitHub Actions — tests inference API only
├── templates/
│   └── index.html          # Demo frontend (dark theme, inline CSS/JS)
├── tests/
│   └── test_api.py         # pytest — passes with or without model file
├── models/
│   ├── generator.onnx      # Committed after training (~15MB, used for inference)
│   ├── generator.keras     # Committed after training (source checkpoint)
│   ├── discriminator.keras # Committed after training
│   └── training_log.json   # gen_loss and disc_loss per epoch
├── samples/
│   ├── epoch_0010.png      # Sample grids saved every 10 epochs during training
│   ├── ...
│   └── final_samples.png
└── data/
    └── anime_faces/        # Gitignored images — download via notebook or kagglehub
```

---

## 5. Architecture

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

## 6. Training Dynamic

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

**Optimizers:** Adam with `beta_1=0.5`. Generator: `lr=0.0002`. Discriminator: `lr=0.0001`.  
The discriminator runs at half the generator's learning rate — the baseline run showed discriminator dominance (D loss → 0, G loss rising), so slowing it down gives the generator more useful gradient signal.

**Label smoothing:** real labels use `0.9` instead of `1.0`. Prevents the discriminator from becoming overconfident on real images.

---

## 7. Training Stability

GAN training failure modes and how the architecture prevents them:

| Failure | Sign | Fix Applied |
|---------|------|-------------|
| Mode collapse | All outputs look identical | Dropout(0.3) in discriminator |
| Discriminator dominance | D loss → 0, G loss spikes | Asymmetric learning rates (D=0.0001, G=0.0002); Dropout slows D |
| Oscillation | Both losses fluctuate without trend | BETA_1=0.5; BatchNorm throughout |

---

## 8. Latent Space Interpolation

```bash
python generate.py  # runs generate_interpolation() alongside generate_faces()
```

Interpolating linearly between two noise vectors produces a smooth morphing sequence. A model that had memorised training images would jump discontinuously — smooth interpolation is evidence of a learned continuous mapping from noise to face space.

---

## 9. Azure ML Training Setup

Brief overview:

1. Create workspace: `az ml workspace create --name anime-face-gan-ws --resource-group anime-face-gan-rg`
2. Create compute: `az ml compute create --name anime-face-compute --type ComputeInstance --size Standard_NC6`
3. Set 15-minute idle shutdown in Azure ML Studio
4. Upload dataset, run `python train.py`
5. Instance shuts down automatically on completion

Full guide: [azure_ml_setup.md](azure_ml_setup.md)

---

## 10. Cost Breakdown

| Item | Cost |
|------|------|
| GPU training (Standard_NC6, ~2hr) | ~$2 |
| CPU training (Standard_DS3_v2, ~10hr) | ~$2.50 |
| Google Colab / Kaggle GPU | Free |
| Hugging Face Spaces (inference) | Free tier |

Azure cost controls:
- `train.py` shutdown line fires only when run as `python train.py` (not from notebook)
- 15-minute idle shutdown in Azure ML Studio
- $5 budget alert in Azure Cost Management

---

## 11. Deployment

The app is deployed on Hugging Face Spaces using Docker. Push to the `hf` remote to redeploy:

```bash
git remote add hf https://huggingface.co/spaces/xavier-oc-machinelearn/anime-face-gan
git push hf main
```

HF Spaces builds the Docker image automatically on push. The app starts in ~30 seconds with no CPU quota — unlike Azure App Service's free tier (F1), which enforces a 60 CPU-minute/day limit that TensorFlow-based cold starts exhaust quickly.

**Why Hugging Face Spaces over Azure App Service**
Azure App Service (F1 free tier) was the initial target, matching the pattern used across other projects in this portfolio. However, the F1 tier enforces a 60 CPU-minute/day quota. TensorFlow's import overhead alone (~2–3 CPU minutes per cold start) exhausted the daily quota within a few app restarts, making the service unreliable. Switching inference to ONNX Runtime (`onnxruntime`, ~10MB vs TensorFlow's ~500MB) significantly reduced startup cost, but F1's shared CPU still made cold starts unpredictable.

Hugging Face Spaces was chosen for cost reasons — not unfamiliarity with Azure. All other projects in this portfolio run on Azure App Service without issue; this project's inference load profile (cold starts with a model load on every restart) is a poor fit for a quota-based free tier. HF Spaces runs the Docker container continuously with no daily CPU limit, which suits the workload.

**Local development:**

```bash
uvicorn main:app --reload --port 7860
# Open http://localhost:7860
```

**Docker (local):**

```bash
docker build -t anime-face-gan .
docker run -p 7860:7860 anime-face-gan
```

---

## 12. CI/CD

GitHub Actions runs `pytest tests/ -v` on every push to `main`. Tests pass with or without `generator.onnx` — generation endpoints return 503 gracefully when the model is missing.

---

## 13. Design Decisions

**Why anime faces over MNIST or CIFAR-10**
Anime faces have low intra-class variance — all images share a consistent art style, similar face geometry, and uniform scale. This means the generator converges faster and produces cleaner results than on real-face datasets, where variation in lighting, pose, and background creates a much harder distribution to learn.

**Why LATENT_DIM=128**
128 dimensions gives the generator enough capacity to represent the variation in the training data — different face shapes, hair colours, expressions — without being so large that sampling becomes inefficient. Lower values (64) produce less variety; higher values (256+) add little quality benefit at 64×64 resolution.

**Why BETA_1=0.5 for Adam**
The default Adam momentum (0.9) causes training instability in GANs. High momentum means the optimizer continues in a direction for many steps even after the loss surface changes — in GANs, where both networks are constantly shifting each other's loss landscape, this causes oscillation. BETA_1=0.5 reduces momentum, making the optimizer more responsive to the current gradient.

**Why Dropout in the discriminator**
Without Dropout, the discriminator becomes too accurate too quickly. When it is near-perfect, the generator receives gradients close to zero — no useful signal about how to improve. Dropout(0.3) keeps the discriminator imperfect enough that the generator always has something to learn from.

**Why train on Azure ML rather than locally**
Training 100 epochs on 63,000 64×64 colour images takes 8–12 hours on CPU. Azure ML Compute Instance provides managed cloud compute with a GPU option that reduces training time to ~2 hours. The training infrastructure (Azure ML) is distinct from the inference infrastructure (Hugging Face Spaces) — the same separation used in production ML systems.

**Why ONNX Runtime for inference**
The trained `.keras` checkpoint is exported to ONNX (`.onnx`) for serving. `onnxruntime` is ~10MB and imports in milliseconds; TensorFlow is ~500MB and takes 2–3 minutes of CPU on a cold start. For inference-only serving, there is no benefit to loading the full TensorFlow runtime — ONNX is the standard format for deploying trained models outside the framework used to train them.

**Why the shutdown line is inside `if __name__ == '__main__':`**
The shutdown only fires when `train.py` is run directly as a script (`python train.py`) on Azure ML — not when `train()` is imported by the notebook. This means the notebook training cell works safely on Colab, Kaggle, or local machines without risk of shutting down the host.

---

## 14. Limitations and Improvements

### Current Limitations

**Discriminator dominance in later epochs**  
The discriminator gradually outpaces the generator after epoch ~50. D loss falls to 0.59 and G loss rises to 3.2 by epoch 100, meaning the generator receives increasingly weak gradient signal in late training. The deployed model uses the epoch 60 checkpoint where the training was most competitive.

**Half dataset used**  
The Colab cache only provided 33,180 of the expected 63,565 images. The generator has seen less variety than it should, which limits face diversity and sharpness.

**64×64 resolution**  
All outputs are 64×64 pixels. At this resolution fine details — eyelashes, sharp hair strands, complex expressions — are not representable regardless of how well training goes.

**DCGAN architecture ceiling**  
DCGAN is a foundational GAN architecture from 2015. It is well-understood and trainable on free GPU compute, but it has a hard quality ceiling compared to modern architectures.

---

### What Would Improve It

| Change | Expected gain | Cost |
|--------|--------------|------|
| Full 63,000-image dataset | More face variety, less D dominance | One more Colab run |
| Update G twice per D step | Extends competitive training phase | One more Colab run |
| Reduce `LEARNING_RATE_D` to 0.00005 | Slower D, more balanced late training | One more Colab run |
| Spectral normalisation in D | Stable gradients without LR tuning | Code change + retrain |
| 128×128 resolution | Sharper details, more complex features | Larger model, longer training |
| StyleGAN2 architecture | Near-photorealistic anime faces | Significantly more complex; pretrained checkpoints available |

---

## 15. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| onnxruntime | >=1.17 | Model inference (exported from trained Keras checkpoint) |
| numpy | >=1.24,<2.0 | Array operations, image processing |
| matplotlib | >=3.7 | Training visualisation in notebook |
| pillow | >=10.0 | Image loading and grid assembly |
| kagglehub | >=0.3 | Dataset download |
| fastapi | >=0.110 | REST API framework |
| uvicorn | >=0.27 | ASGI server |
| gunicorn | >=21.0 | Production process manager |
| pydantic | >=2.0 | Request/response validation |
| jupyter | >=1.0 | Architecture walkthrough notebook |
| pytest | >=7.0 | API tests |
| httpx | >=0.27 | TestClient dependency for pytest |
| python-multipart | >=0.0.9 | Form data support for FastAPI |
