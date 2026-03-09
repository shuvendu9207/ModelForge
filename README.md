<div align="center">

# ⚗️ ModelForge
### *Stop managing files. Start building models.*

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

A self-hosted ML platform — raw CSV to tracked, versioned, comparable models in one command.

</div>

---

## ⚡ Quick Start

```bash
pip install -r requirements.txt

py modelforge.py run      --dataset data/raw/data.csv --model random_forest
py modelforge.py compare
py modelforge.py dashboard
```

> 🌐 Dashboard → `http://localhost:8501` &nbsp;|&nbsp; API docs → `http://localhost:8000/docs`
> 💡 pip issues? Use `pip install --user -r requirements.txt`

---

## 🔁 Pipeline

Every run fires 6 stages automatically:

```
Dataset → Preprocessing → Feature Engineering → Training → Evaluation → Tracking → exp_<id>.pkl
```

**Supported models:** `linear_regression` · `logistic_regression` · `random_forest` · `svm` · `xgboost`

---

## ✨ Features

| | |
|---|---|
| 🔁 **Auto Pipeline** | 6-stage workflow from raw data to tracked results |
| 📦 **Dataset Versioning** | MD5-hashed snapshots in `data/versions/` |
| 🎛️ **Hyperparameter Tuning** | GridSearchCV + Optuna |
| 📊 **Experiment Tracking** | SQLite-backed logs — metrics, params, timings |
| 🗄️ **Model Registry** | `exp_<id>.pkl` per run, loadable by ID |
| 📈 **Dashboard** | 5-page Streamlit UI with charts and comparisons |
| 🌐 **REST API** | FastAPI — `/run`, `/experiments`, `/compare` |

---

## 📁 Project Structure

```
modelforge/
├── modelforge.py          ← Entry point
├── src/
│   ├── loader/            # Load & version datasets
│   ├── preprocessing/     # Clean, encode, scale
│   ├── features/          # Select & transform
│   ├── models/            # Train any supported model
│   ├── pipeline/          # 6-stage executor
│   ├── tracking/          # SQLite experiment logger
│   ├── evaluation/        # Metrics & reports
│   └── api/               # FastAPI routes
├── dashboard/             # Streamlit UI
├── data/                  # raw/ · processed/ · versions/
├── ml/models/             # Versioned .pkl artifacts
├── experiments/           # experiments.db
├── tests/                 # pytest suite
└── config/                # settings.yaml · models.yaml
```

---

## ⚙️ Configuration

```yaml
# config/settings.yaml
data:
  test_size: 0.2
  default_target_column: "label"
preprocessing:
  handle_missing: mean      # mean | median | drop
  scale_method: standard    # standard | minmax | none
  encode_method: onehot     # onehot | label | none
```

---

<div align="center">
Built with ⚗️ by the ModelForge team · <a href="https://github.com/yourusername/ModelForge/issues">Report an Issue</a>

</div>

