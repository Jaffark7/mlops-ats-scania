# Scania APS Failure Prediction

*A self-directed MLOps learning project.*

This project predicts failures in the **Air Pressure System (APS)** of Scania
trucks from anonymized sensor readings. The positive class represents failures
caused by a specific APS component; the negative class represents trucks with
other, non-APS failures.

I built this to practice the full path from a tabular machine-learning notebook
to a small deployable inference service: preprocessing, imbalanced
classification, model persistence, API design, tests, Docker, and a local
Kubernetes deployment.

## What the project does

- **Trains an XGBoost classifier** on the Scania APS training set using mean
  imputation, standard scaling, and class-imbalance weighting.
- **Evaluates the model** on the held-out Scania APS test set with F1, recall,
  ROC-AUC, and accuracy.
- **Packages the trained pipeline** as `model.joblib`, including the exact
  training feature order required at inference time.
- **Serves predictions** through a FastAPI service with `/health` and
  `/predict` endpoints.
- **Builds and tests the service** through GitHub Actions, pytest, Docker, and
  Kubernetes manifests for Minikube-style local deployment.

## The model

The production artifact is a scikit-learn `Pipeline`:

```text
SimpleImputer(mean) -> StandardScaler -> XGBClassifier
```

The dataset is highly imbalanced, so the XGBoost classifier uses
`scale_pos_weight = negative_count / positive_count`. The original challenge is
also cost-sensitive: an unnecessary workshop check is much cheaper than missing
a truck with an APS-related failure, so recall and F1 are more meaningful than
accuracy alone.

## Results

The notebook run saved the trained pipeline and produced the following metrics.
The held-out test set is the original Scania APS test CSV.

| Metric | Value |
|--------|------:|
| 5-fold CV F1 mean | 0.8265 |
| Test accuracy | 0.9938 |
| Test F1 | 0.8664 |
| Test recall | 0.8560 |
| Test ROC-AUC | 0.9959 |

The accuracy is high partly because the dataset is dominated by negative
examples. The more important signal is that the model recovers most positive APS
failures while keeping strong F1 on the minority class.

## API

Run the service locally:

```bash
uvicorn aps_failure.api:app --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "aa_000": 0.0,
      "ab_000": 1.0,
      "ac_000": 0.5
    }
  }'
```

The API accepts any subset of APS feature names. Missing training features are
reintroduced as `NaN` in the original column order and handled by the imputer
inside the trained pipeline.

## Project structure

```text
.
├── notebooks/
│   └── 01_train_xgboost_pipeline.ipynb
├── src/aps_failure/
│   ├── api.py
│   ├── data.py
│   ├── evaluate.py
│   ├── model_io.py
│   └── train.py
├── tests/
│   └── test_api.py
├── deployment/k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── data/
│   ├── README.md
│   └── aps_failure_description.txt
├── results/
│   └── metrics.json
├── model.joblib
├── Dockerfile
├── pyproject.toml
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Download the dataset from the UCI Machine Learning Repository:
https://archive.ics.uci.edu/ml/datasets/aps%2Bfailure%2Bat%2Bscania%2Btrucks

Place the CSV files here:

```text
data/aps_failure_training_set.csv
data/aps_failure_test_set.csv
```

The raw CSVs are intentionally git-ignored because they are large source data
files.

## Training recipe

Train and save a new model bundle:

```bash
python -m aps_failure.train \
  --training-csv data/aps_failure_training_set.csv \
  --output model.joblib
```

Evaluate it on the held-out test set:

```bash
python -m aps_failure.evaluate \
  --model model.joblib \
  --test-csv data/aps_failure_test_set.csv \
  --output results/metrics.json
```

The notebook in `notebooks/` contains the same workflow in exploratory form.

## Tests and CI

Run the API tests:

```bash
pytest -q
```

The GitHub Actions workflow installs the package, runs pytest, and builds the
Docker image.

## Docker

```bash
docker build -t scania-aps-failure-api .
docker run -p 8000:8000 scania-aps-failure-api
```

## Kubernetes

For a local Minikube workflow:

```bash
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml
minikube service scania-aps-failure-api
```

## What I learned

- How to turn an imbalanced tabular ML problem into a deployable inference API.
- Why accuracy can be misleading on rare-failure datasets.
- How to persist a full preprocessing and model pipeline safely enough for
  consistent inference.
- How Docker, tests, CI, and Kubernetes manifests fit around a small ML service.
- Why professional ML repositories need reproducible setup instructions and
  honest evaluation, not just a trained artifact.

## Dataset

This project uses the **APS Failure at Scania Trucks** dataset from Scania CV AB,
published through the UCI Machine Learning Repository. The included
`data/aps_failure_description.txt` file preserves the original dataset
description and challenge cost metric.
