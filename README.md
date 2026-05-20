# MetroVision-MLOps

Production-grade MLOps pipeline built around an academic Paris Metro pictogram detection system.

## Why this project exists

MetroVision started as a university computer-vision project (ISEP IG2405): detect and classify Paris Metro line pictograms from photographs using Hough Circle Transform, HOG descriptors, a binary CNN, and k-NN.

**This repository industrializes that academic code** into a deployable, observable, reproducible ML system — the kind you would operate in a real enterprise environment. The original detection pipeline is untouched; everything added here is MLOps scaffolding:

- REST API serving predictions (FastAPI)
- Containerized builds (Docker, multi-stage)
- Kubernetes-ready deployment (minikube / OpenShift)
- Experiment tracking and model registry (MLflow)
- CI/CD automation (GitHub Actions)
- Monitoring, alerting, and data-drift detection (Prometheus, Grafana, Evidently AI)

## Architecture

```
MetroVision-MLOps/
├── api/                    # FastAPI inference service
├── docker/                 # Dockerfiles (api, training)
├── k8s/                    # Kubernetes manifests (minikube + OpenShift overlay)
├── monitoring/             # Prometheus, Grafana dashboards, Evidently reports
├── mlflow_pipelines/       # Training wrappers with MLflow tracking
├── tests/                  # pytest suite
├── .github/workflows/      # CI/CD pipelines
├── model/                  # Trained model artifacts (.h5, .joblib)
└── [original files]        # Untouched academic code (historical reference)
```

## Quick start

```bash
# Prerequisites: Python 3.11+
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## Run the full stack

```bash
docker-compose up --build
```

| Service | URL | Credentials |
|---------|-----|-------------|
| API (Swagger) | http://localhost:8000/docs | — |
| MLflow | http://localhost:5000 | — |
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / metrovision |

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict` | Upload an image, get detected metro lines |
| `GET` | `/health` | Liveness / readiness probe |
| `GET` | `/metrics` | Prometheus-format metrics |
| `GET` | `/version` | App version and model metadata |

## Tech stack

Python 3.11 · FastAPI · Docker · Kubernetes · MLflow · Prometheus · Grafana · Evidently AI · pytest · GitHub Actions · ruff

## Original academic project

The detection pipeline (Hough + CNN + k-NN) was developed as part of the IG2405 Computer Vision course at ISEP (2025) by Gabriel Esteves and Louis Lenouvel. The original code is preserved as-is in the repository root.

## License

[MIT](LICENSE)
