![CI](https://github.com/Moazzam-Matin/Gulf_Auto_Price_Intelligence/actions/workflows/ci.yml/badge.svg)

# Gulf Auto Price Intelligence 🚗

An end-to-end MLOps pipeline that predicts used car prices in the UAE market — from raw data to a containerized, CI-tested prediction API.

This isn't just a trained model in a notebook. It's a full engineering pipeline: experiment tracking, automated tests that catch real bugs (including a genuine data leakage issue found and fixed during development), a served API, and a CI/CD pipeline that builds and validates a Docker image on every push.

## Live CI Pipeline

Every push to this repo triggers an automated pipeline (see `.github/workflows/ci.yml`) that:

1. Installs dependencies in a clean environment
2. Runs the full pytest suite
3. Trains the model and builds the Docker image

See it running under the [Actions tab](https://github.com/Moazzam-Matin/Gulf_Auto_Price_Intelligence/actions).

## Architecture

```mermaid
graph TD
    A["Raw CSV: UAE car listings"] --> B("clean_data(): handles dirty data")
    B --> C("train_test_split(): split BEFORE encoding")

    C --> D["fit_target_encoding(): learned only from train"]
    C --> E["apply_target_encoding(): applied to train & test"]

    D --> F["RandomForestRegressor: tracked via MLflow"]
    E --> F

    F --> G["FastAPI /predict: Pydantic-validated HTTP"]
    G --> H["Docker container: built & verified in CI"]

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#bbf,stroke:#333,stroke-width:2px


## A Real Bug I Found and Fixed: Target Leakage

While building this pipeline, I discovered the initial version computed target encoding (average price per car Make/Model) **before** splitting the data into train/test sets. This let information from the test set quietly leak into the features the model was evaluated on.

**Impact, measured directly:** fixing this changed MAPE from 52.3% (leaky, artificially optimistic) to 53.8% (honest). I wrote a regression test (`tests/test_preprocess.py::test_encoding_is_insensitive_to_test_row_values`) that fails if this leak is ever reintroduced — verified by deliberately feeding the encoding an extreme, unmistakable test-row value and asserting it has zero effect on the learned mapping.

This is documented here deliberately: catching and fixing this kind of bug — and writing a test that prevents its return — is a core MLOps skill, not a footnote.

## Project Structure

```

├── research.ipynb # The "lab": EDA, residual analysis, initial model exploration
├── src/
│ ├── features.py # Data loading
│ ├── preprocess.py # Cleaning + leakage-safe target encoding
│ ├── train.py # MLflow-tracked training
│ ├── predict.py # Inference logic (shared feature schema with training)
│ └── api.py # FastAPI service
├── tests/
│ └── test_preprocess.py # Includes the leakage regression test
├── data/raw/
│ └── sample_ci_data.csv # Small synthetic dataset used only by CI (real data is gitignored)
├── .github/workflows/
│ └── ci.yml # Test + Docker build pipeline
├── Dockerfile
└── requirements.txt

```

**Research vs. production:** `research.ipynb` is where the original exploration happened - data cleaning experiments, visualizations, and the residual analysis that surfaced the "Spec Ceiling" finding (missing GCC/Import spec data as the main driver of remaining error). Everything in `src/` is the productionized version of that research: the same modeling decisions, rebuilt as tested, tracked, servable code. Keeping both visible - rather than deleting the notebook once "real" code existed - is deliberate: it shows the reasoning trail from exploration to production, not just the end result.

## Tech Stack

- **Modeling:** scikit-learn (RandomForestRegressor), smoothed target encoding for high-cardinality categoricals
- **Experiment tracking:** MLflow (params, metrics, model registry)
- **Testing:** pytest (including a genuine regression test against a real bug found during development)
- **Serving:** FastAPI + Pydantic input validation
- **Containerization:** Docker
- **CI/CD:** GitHub Actions

## Running Locally

Setup:

```

python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt

```

Train (uses `data/raw/uae_used_cars_10k.csv` — see Data Source below):

```

python src/train.py

```

View experiments:

```

mlflow ui

```

Run tests:

```

pytest tests/ -v

```

Start the API:

```

uvicorn src.api:app --reload

```

Then open `http://127.0.0.1:8000/docs`.

## Data Source

- **Source:** [Kaggle - UAE Used Car Prices](https://www.kaggle.com/datasets/aliiihussain/car-price-prediction/data)
- Raw data is gitignored (standard practice — don't commit data to a code repo). Download it and place it at `data/raw/uae_used_cars_10k.csv` to train locally.

## Current Model Performance

- **MAPE:** ~53.8% (honest, leakage-free)
- **MAE:** ~46,650 AED
- **R²:** ~0.50

Performance varies significantly by price segment — error as a percentage is much higher for cheaper cars than luxury ones, since a fixed AED error matters more on a lower base price. (Segment-level evaluation breakdown via `evaluate.py` — in progress.)

## Roadmap

- [ ] Deploy containerized API to AWS (ECS Fargate)
- [ ] Segment-level evaluation report (`evaluate.py`)
- [ ] Data/model versioning via DVC
- [ ] Model monitoring for data drift

## Acknowledgements

Thanks to the Kaggle community for the UAE market dataset.

```

```

```
