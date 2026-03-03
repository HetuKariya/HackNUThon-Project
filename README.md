# NutriSoil — Soil Analysis & Crop Recommendation System

**NutriSoil** is a friendly web app that helps farmers, gardeners, and agritech developers quickly analyze soil properties, predict soil type and crop growth conditions using machine learning, and get practical fertilizer & crop recommendations. It also includes an AI assistant for agricultural advice and tools for generating spectral images and training the models used by the app.

---

## 🚀 Key features

* ✅ User authentication (Firebase)
* ✅ Soil analysis using core parameters: pH, N, P, K, moisture, EC, temperature
* ✅ Machine Learning models for:

  * Soil type classification
  * Growth condition / suitability prediction
* ✅ Fertilizer & crop recommendations based on model outputs
* ✅ AI chatbot assistant (Google AI / Gemini integration) for agricultural Q&A
* ✅ Utilities to train models (`train_models.py`) and generate spectral images (`generate_spectral_images.py`)
* ✅ Simple web UI (Flask) with templates and static assets

---

## Repository structure

```
HackNUThon-Project/
├─ .env                     # environment variables (not committed)
├─ app.py                   # Flask application (web UI, API endpoints)
├─ train_models.py          # training scripts for ML models
├─ generate_spectral_images.py
├─ models/                  # trained model artifacts
├─ data/                    # raw/processed datasets (example CSV/XLSX)
├─ templates/               # Flask HTML templates
├─ static/                  # CSS, JS, images
├─ requirments.txt          # python dependencies (note the filename)
└─ README.md                # ← you are here
```

> ⚠️ The repo currently contains `requirments.txt` (misspelled). When installing, use that filename or rename it to `requirements.txt`.

---

## Quick start

### Prerequisites

* Python 3.8+
* Git
* Firebase account (for authentication + optional database)
* Google AI API key (for the Gemini model / AI assistant)
* (Optional) A virtual environment tool: `venv`, `conda`

### Install

1. Clone the repository

```bash
git clone https://github.com/HetuKariya/HackNUThon-Project.git
cd HackNUThon-Project
```

2. Create & activate a virtual environment (recommended)

```bash
python -m venv venv
# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

3. Install dependencies

```bash
pip install -r requirments.txt
# If you renamed file:
# pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root. Example `.env.template`:

```
# Flask
FLASK_APP=app.py
FLASK_ENV=development
FLASK_SECRET_KEY=replace_with_a_secret_key

# Firebase (example variables commonly used)
FIREBASE_API_KEY=your_firebase_api_key
FIREBASE_AUTH_DOMAIN=your-project.firebaseapp.com
FIREBASE_PROJECT_ID=your_project_id
FIREBASE_DATABASE_URL=https://your_project.firebaseio.com
FIREBASE_STORAGE_BUCKET=your_project.appspot.com
FIREBASE_MESSAGING_SENDER_ID=XXXXXXXXXXXX
FIREBASE_APP_ID=1:XXXXXXXX:web:xxxxxxxxxxxxxxxx

# Google AI / Gemini
GOOGLE_AI_API_KEY=your_google_ai_api_key

# Model & data paths (optional)
MODEL_DIR=models
DATA_DIR=data
```

> Only add keys you actually have. Do **not** commit `.env` to version control.

---

## Run the app (development)

1. Train (or load) models if not already present:

```bash
python train_models.py
```

This script produces trained models in the `models/` folder (see script for options and data paths).

2. Start the web server:

```bash
python app.py
```

3. Open your browser:

```
http://127.0.0.1:5000
```

---

## About the scripts

* **`app.py`** — Flask application. Serves UI pages, handles user auth with Firebase, receives soil parameter inputs, calls ML models to predict soil type & growth conditions, and returns fertilizer/crop recommendations.
* **`train_models.py`** — Preprocesses data from `data/`, trains the ML models (classification/regression as implemented), and saves model artifacts into `models/`.
* **`generate_spectral_images.py`** — Utility to generate or transform spectral imagery that can be used for further model inputs or visualization.
* **`models/`** — Trained model artifacts (pickles, saved weights). Keep these out of version control if they are large; consider using Git LFS or a cloud storage bucket.

---

## Usage & UX flow (high-level)

1. User signs up / logs in (Firebase).
2. User enters soil measurements (pH, N, P, K, moisture, EC, temperature, etc.).
3. App runs the ML models and returns:

   * Predicted soil type
   * Growth condition or suitability score
   * Recommended fertilizers and best-suited crops
4. User can ask follow-up questions to the AI assistant for farming best practices, application schedules, or local recommendations.

---

## Recommendations / Next steps (ideas to improve)

* Add unit and integration tests for model pipelines and Flask routes.
* Add Dockerfile & `docker-compose` for easier local dev and deployment.
* Add CI (GitHub Actions) to run tests and build artifacts.
* Move large models to cloud storage (S3 / Firebase Storage) and load at runtime.
* Add input validation & stronger security for API keys (secrets manager).
* Improve the AI assistant by adding retrieval-augmented-generation (RAG) with a local knowledge base of agronomy resources.

---

## Contributing

Contributions are very welcome! Typical ways to contribute:

* Bug reports & feature requests: open an issue
* Fixes & features: open a PR
* Improve documentation & examples
* Add tests or CI workflows

Please follow these steps for code contributions:

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit changes & push
4. Open a Pull Request describing changes

---

## License

This project ships without an explicit license in the repository. If you want others to reuse and contribute, add a license file (for example, `LICENSE` with the MIT license). Example MIT header:

```
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted...
```

---

## Acknowledgements

* Built for HackNUThon — thanks to the organizers and mentors
* Firebase — authentication & database
* Google AI (Gemini) — for AI assistant capabilities
* Open-source libraries and ML tooling used in `requirments.txt`

---

## Contact

Maintainer: **Hetu Kariya**
Repo: [https://github.com/HetuKariya/HackNUThon-Project](https://github.com/HetuKariya/HackNUThon-Project)

---
