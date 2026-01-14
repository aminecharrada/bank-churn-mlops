# 🏦 Bank Churn Prediction - MLOps Project

![CI/CD Pipeline](https://github.com/VOTRE_USERNAME/bank-churn-mlops/workflows/CI%2FCD%20Pipeline/badge.svg)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Description

Projet MLOps complet pour la prédiction du churn bancaire (départ des clients). Ce projet implémente un workflow ML de bout en bout avec :
- Gestion du code (Git)
- Conteneurisation (Docker / Docker Compose)
- Versioning des données (DVC)
- Suivi d'expériences (MLflow)
- Pipeline MLOps (ZenML)
- Optimisation des hyperparamètres (Optuna)
- CI/CD avec GitHub Actions
- Déploiement sur Azure Container Apps
- Monitoring et drift detection

---

## 🏗️ Architecture du Projet

```
bank-churn-mlops/
├── .dvc/                      # Configuration DVC
├── .github/workflows/         # CI/CD Pipeline
│   └── ci-cd.yml
├── app/                       # API FastAPI
│   ├── models.py              # Endpoints API
│   ├── drift_detect.py        # Détection de drift
│   └── __init__.py
├── data/
│   ├── bank_churn.csv         # Dataset (tracké par DVC)
│   └── bank_churn.csv.dvc     # Fichier DVC
├── model/
│   ├── churn_model.pkl        # Modèle entraîné
│   ├── churn_model_optuna.pkl # Modèle optimisé
│   ├── optuna_results.json    # Résultats Optuna
│   └── optuna_trials.csv      # Historique trials
├── mlruns/                    # Tracking MLflow
├── tests/
│   └── test_api.py            # Tests unitaires
├── docker-compose.yml         # Stack locale complète
├── Dockerfile                 # API containerization
├── Dockerfile.streamlit       # Streamlit UI
├── train_model.py             # Entraînement baseline
├── run_experiments.py         # Expériences multiples MLflow
├── optuna_optimize.py         # Optimisation Optuna
├── zenml_pipeline.py          # Pipeline ZenML
├── streamlit_app.py           # Interface utilisateur
├── demo_rollback.sh           # Démo v1→v2 + rollback
├── deploy.sh                  # Déploiement Azure
└── requirements.txt           # Dépendances Python
```

---

## 🚀 Démarrage Rapide

### Prérequis
- Python 3.11+
- Docker & Docker Compose
- Azure CLI (pour déploiement cloud)
- Git

### Installation

```bash
# Cloner le repository
git clone https://github.com/VOTRE_USERNAME/bank-churn-mlops.git
cd bank-churn-mlops

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Récupérer les données avec DVC

```bash
# Pull des données depuis le remote DVC
dvc pull
```

---

## 📦 3.1 Dataset et Modèle

- **Dataset**: Bank Customer Churn (10,000 clients, 11 features)
- **Source**: Public dataset
- **Modèle baseline**: Random Forest Classifier
- **Métrique principale**: ROC-AUC

| Métrique | Baseline | Optuna Best |
|----------|----------|-------------|
| Accuracy | 0.7655   | 0.7615      |
| F1-Score | 0.3290   | 0.1733      |
| ROC-AUC  | 0.7775   | 0.7871      |

---

## 🔀 3.2 Gestion du Code (Git)

```bash
# Structure des branches
git branch -a
# * main                    # Branche principale
# * dev                     # Développement

# Tags de version
git tag -l
# v1.0.0                    # Version initiale
# v2.0.0                    # Avec optimisation
```

---

## 🐳 3.3 Conteneurisation (Docker Compose)

### Démarrer la stack locale complète

```bash
# Démarrer API + Streamlit + MLflow
docker-compose up -d

# Services disponibles:
# - API:       http://localhost:8000
# - Streamlit: http://localhost:8501
# - MLflow:    http://localhost:5000

# Lancer un entraînement
docker-compose run --rm training

# Arrêter tous les services
docker-compose down
```

### Dockerfiles

- `Dockerfile` - API FastAPI de prédiction
- `Dockerfile.streamlit` - Interface utilisateur Streamlit

---

## 📊 3.4 Versioning des Données (DVC)

```bash
# Vérifier le statut DVC
dvc status

# Pousser les données vers le remote
dvc push

# Récupérer les données
dvc pull

# Afficher le DAG des données
dvc dag
```

**Configuration DVC:**
```yaml
# .dvc/config
[core]
    remote = myremote
['remote "myremote"']
    url = /tmp/dvc-remote  # Peut être S3, GCS, Azure Blob, etc.
```

**Fichiers trackés:**
- `data/bank_churn.csv` → `data/bank_churn.csv.dvc`

---

## 📈 3.5 Experiment Tracking (MLflow)

### Lancer MLflow UI

```bash
mlflow ui --port 5000
# Ouvrir http://localhost:5000
```

### Exécuter les expériences

```bash
# Run baseline
python train_model.py

# Runs multiples (5 modèles différents)
python run_experiments.py
```

### Runs disponibles

| Run Name | Modèle | ROC-AUC |
|----------|--------|---------|
| RandomForest-baseline | RF 100 trees | 0.7775 |
| RandomForest-200trees | RF 200 trees | 0.7783 |
| RandomForest-deep | RF depth=20 | 0.7741 |
| GradientBoosting-v1 | GB | 0.7713 |
| LogisticRegression | LR | 0.4862 |
| optuna-best | Optimisé | 0.7871 |

---

## 🔧 3.6 Pipeline MLOps (ZenML)

### Exécuter le pipeline

```bash
# Initialiser ZenML
zenml init

# Exécuter le pipeline complet
python zenml_pipeline.py

# Voir les runs
zenml pipeline runs list

# Dashboard ZenML
zenml show
```

### Structure du Pipeline

```
┌──────────────┐
│  load_data   │ ← Charge le dataset
└──────┬───────┘
       │
┌──────▼───────┐
│  preprocess  │ ← Split train/test
└──────┬───────┘
       │
┌──────▼───────┐
│    train     │ ← Entraîne Random Forest
└──────┬───────┘
       │
┌──────▼───────┐
│   evaluate   │ ← Calcule les métriques
└──────┬───────┘
       │
┌──────▼───────┐
│   export     │ ← Sauvegarde le modèle
└──────┬───────┘
       │
┌──────▼───────┐
│  log_mlflow  │ ← Log dans MLflow
└──────────────┘
```

---

## 🎯 3.7 Optimisation (Optuna)

### Lancer l'optimisation

```bash
python optuna_optimize.py
```

### Résultats

- **Nombre de trials**: 10
- **Meilleurs hyperparamètres**:
  - n_estimators: 269
  - max_depth: 5
  - min_samples_split: 3
  - min_samples_leaf: 3
  - max_features: sqrt

### Fichiers générés

- `model/churn_model_optuna.pkl` - Meilleur modèle
- `model/optuna_results.json` - Résultats complets
- `model/optuna_trials.csv` - Historique des trials

---

## 🔄 3.8 CI/CD (GitHub Actions)

### Pipeline CI/CD

Le fichier `.github/workflows/ci-cd.yml` définit:

1. **CI (Continuous Integration)**
   - Checkout du code
   - Installation Python 3.11
   - Installation des dépendances
   - Exécution des tests avec pytest
   - Code coverage

2. **CD (Continuous Deployment)**
   - Build de l'image Docker
   - Push vers Azure Container Registry
   - Déploiement sur Azure Container Apps
   - Vérification du health check

### Déclenchement

- Push sur `main` → Tests + Build + Deploy
- Pull Request → Tests seulement
- Manuel via `workflow_dispatch`

---

## 🚀 3.9 Déploiement (Serving)

### URLs Production

| Service | URL |
|---------|-----|
| **API** | https://bank-churn.kindisland-3009027b.westeurope.azurecontainerapps.io |
| **Docs** | https://bank-churn.kindisland-3009027b.westeurope.azurecontainerapps.io/docs |
| **Streamlit** | https://bank-churn-streamlit.kindisland-3009027b.westeurope.azurecontainerapps.io |

### Endpoints API

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/` | GET | Info API |
| `/health` | GET | Health check |
| `/predict` | POST | Prédiction unique |
| `/predict/batch` | POST | Prédictions multiples |
| `/drift/check` | POST | Vérification drift |

### Déploiement Azure

```bash
# Déployer l'API
./deploy.sh

# Ou manuellement
az containerapp update \
    --name bank-churn \
    --resource-group rg-mlops-bank-churn \
    --image mlopscharrada.azurecr.io/churn-api:latest
```

### Simulation v1 → v2 + Rollback

```bash
# Exécuter la démo complète
./demo_rollback.sh

# Ou manuellement:
# 1. Déployer v2
az containerapp update --name bank-churn --image mlopscharrada.azurecr.io/churn-api:v2

# 2. Rollback vers v1
az containerapp revision list --name bank-churn  # Voir les révisions
az containerapp revision activate --revision <nom-revision-v1>
az containerapp ingress traffic set --revision-weight <nom-revision-v1>=100
```

---

## 📊 Bonus: Monitoring

### Application Insights

L'API est connectée à Azure Application Insights pour:
- Latence des requêtes
- Nombre de requêtes
- Erreurs et exceptions
- Logs personnalisés

### Drift Detection

```bash
# Vérifier le drift via API
curl -X POST "https://bank-churn.../drift/check" \
  -H "Content-Type: application/json" \
  -d '{"reference_file": "data/bank_churn.csv", "production_file": "data/production_data.csv"}'
```

---

## 🧪 Tests

```bash
# Exécuter les tests
pytest tests/test_api.py -v

# Avec coverage
pytest tests/test_api.py --cov=app --cov-report=term
```

---

## 📁 Livrables

| Livrable | Fichier/Lien |
|----------|--------------|
| **Code source** | GitHub Repository |
| **Dockerfiles** | `Dockerfile`, `Dockerfile.streamlit` |
| **Docker Compose** | `docker-compose.yml` |
| **Configuration DVC** | `.dvc/config`, `data/bank_churn.csv.dvc` |
| **Captures MLflow** | `mlruns/` (lancer `mlflow ui`) |
| **Pipeline ZenML** | `zenml_pipeline.py` |
| **Optimisation Optuna** | `optuna_optimize.py`, `model/optuna_results.json` |
| **CI/CD** | `.github/workflows/ci-cd.yml` |
| **Déploiement** | `deploy.sh`, `demo_rollback.sh` |
| **Documentation** | Ce README |

---

## 👨‍💻 Auteur

**Nom**: Charrada  
**Projet**: Bank Churn MLOps  
**Date**: Janvier 2026

---

## 📝 Licence

MIT License - voir [LICENSE](LICENSE) pour plus de détails.
