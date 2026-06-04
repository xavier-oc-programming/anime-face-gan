# Inference only — training was done on Azure ML Compute Instance.
# The trained generator.keras is committed to models/ and loaded at startup.
# No GPU or heavy compute required for inference — generating 16 faces
# takes ~100ms on CPU.

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p samples
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
