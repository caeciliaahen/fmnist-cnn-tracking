# Image Classification & Experiment Tracking

This project demonstrates image classification on the [FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) dataset using a SimpleCNN model, with integrated **experiment tracking** via:

- **MLflow** – tracks metrics, parameters, and artifacts
- **TensorBoard** – visualizes loss/accuracy logs and images
- **DVC** – manages dataset versioning, model outputs, and pipeline stages

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run training (standalone)

```bash
python train.py
```

### 3. Run with DVC (reproducible pipeline)

```bash
dvc repro
```