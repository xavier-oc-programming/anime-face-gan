# Azure ML Training Setup

Complete guide for running `train.py` on Azure ML Compute Instance.

---

## 1. Prerequisites

- Azure account with an active subscription
- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) installed (`az --version`)
- Azure ML CLI extension: `az extension add -n ml`
- Logged in: `az login`

---

## 2. Create Azure ML Workspace

```bash
az group create --name anime-face-gan-rg --location westeurope

az ml workspace create \
  --name anime-face-gan-ws \
  --resource-group anime-face-gan-rg
```

---

## 3. Create Compute Instance

**CPU — Standard_DS3_v2 (recommended):**
```bash
az ml compute create \
  --name anime-face-compute \
  --type ComputeInstance \
  --size Standard_DS3_v2 \
  --resource-group anime-face-gan-rg \
  --workspace-name anime-face-gan-ws
```
Cost: ~$0.25/hr · ~10hr training = ~$2.50 total  
Available on all Azure subscriptions by default.

**GPU — Standard_NC6 (faster, but requires quota):**
```bash
az ml compute create \
  --name anime-face-compute \
  --type ComputeInstance \
  --size Standard_NC6 \
  --resource-group anime-face-gan-rg \
  --workspace-name anime-face-gan-ws
```
Cost: ~$0.90/hr · ~2hr training = ~$2 total  
Note: GPU quota is not enabled by default on most Azure subscriptions. You will likely
hit a quota error unless you have previously requested GPU access via Azure Portal →
Subscriptions → Usage + quotas. Use Standard_DS3_v2 if you have not done this.

Note: the compute instance is only used during training and shuts down immediately
after. It has no relation to Azure App Service, which serves inference on CPU.

---

## 4. Set Auto-Shutdown (Important — Prevents Idle Billing)

In **Azure ML Studio** → Compute → your instance → **Edit** → Idle shutdown: **15 minutes**.

This is a safety net. The primary cost control is the shutdown line in `train.py` (see section 7).

---

## 5. Upload Dataset

Download the Anime Face Dataset via Kaggle:

```python
import kagglehub
path = kagglehub.dataset_download('splcher/animefacedataset')
```

Or download from: https://www.kaggle.com/datasets/splcher/animefacedataset

Upload the images to the compute instance's `data/anime_faces/` directory via the Azure ML Studio file browser or `scp`.

---

## 6. Run Training

Connect to the compute instance via Azure ML Studio's terminal or JupyterLab, then:

```bash
git clone https://github.com/xavier-oc-programming/anime-face-gan
cd anime-face-gan
pip install -r requirements.txt
python train.py
```

Training runs end-to-end. The instance shuts itself down when complete.

---

## 7. The Shutdown Line

`train.py` ends with:

```python
import subprocess
subprocess.run(['sudo', 'shutdown', '-h', 'now'])
```

This shuts down the compute instance the moment training completes — no idle billing after training. Combined with the 15-minute idle shutdown setting, costs are tightly controlled.

**Remove this line if training locally** — it will shut down your own machine.

---

## 8. Download generator.keras

After training, download `models/generator.keras` from the compute instance:

- Via Azure ML Studio: Files tab → `models/` → download
- Via `scp` from the compute instance IP

Commit it to the repository: `git add models/generator.keras`

---

## 9. Set Budget Alert

Azure Portal → Cost Management → Budgets → **Create budget**:
- Scope: your subscription or resource group
- Amount: **$5**
- Alert: 80% threshold ($4) — you receive an email before costs reach $5

---

## 10. Cost Summary

| Scenario | Instance | Duration | Cost |
|----------|----------|----------|------|
| GPU training | Standard_NC6 | ~2 hours | ~$2 |
| CPU training | Standard_DS3_v2 | ~10 hours | ~$2.50 |

Expected total cost: **$2–5** depending on instance type and epoch count. The budget alert and auto-shutdown setting ensure costs cannot exceed ~$5 even if training runs longer than expected.

---

## Free Alternatives (No Azure Required)

### Local CPU

Remove the subprocess shutdown line from `train.py`, then:

```bash
python train.py
```

Expect 8–12 hours for 100 epochs. No cloud costs. Your machine stays on.

### Google Colab (Free T4 GPU)

1. Upload dataset to Google Drive
2. Upload `train.py`, `config.py`, `requirements.txt`
3. Remove the subprocess shutdown line
4. Run `!pip install -r requirements.txt && python train.py`
5. Download `models/generator.keras` when done

### Kaggle Notebooks (Free P100 GPU)

The dataset is already available on Kaggle as `splcher/animefacedataset` — no upload needed.

1. Create a new Kaggle notebook
2. Add the dataset: Data → Add Dataset → search `splcher/animefacedataset`
3. Paste `train.py` content, remove the shutdown line
4. Run → download `models/generator.keras` from output
5. 30 free GPU hours/week
