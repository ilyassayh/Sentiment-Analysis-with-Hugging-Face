# 🧠 Sentiment Analysis API  
_FastAPI · Transformers · Postgres · Docker (optional) · Hugging Face Spaces_

Small microservice that predicts if a text is **positive** or **negative** and returns a **confidence score**.  
Includes a minimal **HTML test page**, **Swagger docs**, **Postgres persistence** (save predictions), **tests**, **benchmarks**, and a **GitHub Actions CI**.

**Live Space (Docker SDK):** https://huggingface.co/spaces/iyassayh/sentiment-ia-docker

---

## ✨ Features

- `POST /predict` → `{ text, sentiment, confidence, summary }`  
  `summary` is `text-sentiment-confidence`
- `GET /predictions` → recent rows saved in **Postgres**
- `GET /health` → liveness
- `GET /docs` → Swagger
- Minimal **HTML page** at `/` to try it quickly
- (Bonus) **Benchmarks** (CLI + Gradio UI)
- (Bonus) **Pytest** suite + **GitHub Actions** CI

---

## 🧰 Tech Stack

- **Python 3.10+**, **FastAPI**, **Uvicorn**
- **Transformers** (HF), model: `distilbert-base-uncased-finetuned-sst-2-english`
- **PostgreSQL** (local, no Docker required)
- Optional: **Docker** container for the API (port **7860**)
- Optional: **Gradio** for a tiny benchmark UI

---

## 🗂 Project Structure

.
├─ app/
│ ├─ init.py
│ ├─ main.py # FastAPI app (endpoints + model + DB save/list)
│ ├─ database.py # SQLAlchemy engine/session, reads DATABASE_URL
│ └─ models.py # Prediction model (id, text, sentiment, confidence, created_at)
├─ static/
│ └─ index.html # Minimal UI to call /predict and /predictions
├─ tests/
│ ├─ init.py
│ └─ test_api.py # health, success, error tests
├─ benchmark/
│ ├─ benchmark.py # CLI mini-benchmark
│ └─ benchmark_ui_single_table.py # Gradio UI (one table, 3 models)
├─ Dockerfile
├─ requirements.txt
├─ pytest.ini
└─ .github/
└─ workflows/
└─ ci.yml # CI: run pytest + build Docker image

yaml
Copy
Edit

---

## 🚀 Quickstart (Local, with Postgres)

### 1) Install deps
```bash
pip install -r requirements.txt
If your URL uses psycopg v3 (recommended):

text
Copy
Edit
DATABASE_URL=postgresql+psycopg://...
Make sure psycopg[binary] is in requirements.txt.

If your URL uses psycopg2:

text
Copy
Edit
DATABASE_URL=postgresql+psycopg2://...
Add psycopg2-binary to requirements.txt.

2) Create a Postgres DB (local)
Using psql (or pgAdmin GUI):

sql
Copy
Edit
-- You can reuse your postgres superuser OR create a dedicated one
CREATE USER sentiment_user WITH PASSWORD 'MyStrongPass123!';
CREATE DATABASE sentiment_db OWNER sentiment_user;
GRANT ALL PRIVILEGES ON DATABASE sentiment_db TO sentiment_user;
You already have:
postgresql+psycopg2://postgres:1234@localhost:5432/sentiment
That’s fine too—just install psycopg2-binary and use that string.

3) Set the connection string (Windows PowerShell)
Option A (psycopg v3 driver):

powershell
Copy
Edit
$env:DATABASE_URL = "postgresql+psycopg://postgres:1234@localhost:5432/sentiment"
Option B (psycopg2 driver):

powershell
Copy
Edit
pip install psycopg2-binary
$env:DATABASE_URL = "postgresql+psycopg2://postgres:1234@localhost:5432/sentiment"
4) Run the API
bash
Copy
Edit
uvicorn app.main:app --reload --port 7860
Open:

App (HTML): http://127.0.0.1:7860/

Swagger: http://127.0.0.1:7860/docs

Try a request:

bash
Copy
Edit
curl -X POST "http://127.0.0.1:7860/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"I love this product!\"}"
🐳 Run the API in Docker (optional)
Local Postgres + API in Docker (API connects to host DB):

bash
Copy
Edit
docker build -t sentiment-api .
docker run --rm -p 7860:7860 \
  -e DATABASE_URL="postgresql+psycopg://postgres:1234@host.docker.internal:5432/sentiment" \
  sentiment-api
On Linux, replace host.docker.internal with your host IP (e.g., 172.17.0.1).

☁️ Hugging Face Spaces (Docker SDK)
Create a Space → SDK: Docker → Public

Push to Space repo: Dockerfile, requirements.txt, app/, static/

Add Space Secret: DATABASE_URL (use a managed Postgres: Neon, Supabase, etc.)

Build → Open the App tab

Note (cache permissions in Spaces):
This Dockerfile should set writable HF caches:

dockerfile
Copy
Edit
ENV HF_HOME=/home/user/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/user/.cache/huggingface/transformers
ENV HUGGINGFACE_HUB_CACHE=/home/user/.cache/huggingface/hub
RUN mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE" \
    && chmod -R 777 /home/user/.cache
📘 API Reference
POST /predict
Request

json
Copy
Edit
{ "text": "I love this!" }
Response

json
Copy
Edit
{
  "text": "I love this!",
  "sentiment": "positive",
  "confidence": 0.9973,
  "summary": "I love this!-positive-0.9973"
}
GET /predictions?limit=20
Returns the most recent saved predictions (from Postgres).

GET /health
json
Copy
Edit
{ "status": "ok" }
GET /docs
Interactive Swagger UI.

🧪 Tests
Run all tests:

bash
Copy
Edit
pytest -q
Covers:

/health is OK

/predict success (shape + values)

/predict errors (empty/whitespace, too long, missing)

CI uses SQLite by default if you set DATABASE_URL=sqlite:///./data.db in the workflow (or keep Postgres if you prefer).

🏁 Benchmarks (optional)
CLI
bash
Copy
Edit
python benchmark/benchmark.py
Compares a few SST-2 models and prints a tiny Markdown summary.

Gradio (one table, labels only)
bash
Copy
Edit
python benchmark/benchmark_ui_single_table.py
Opens a local UI showing a single table with positive/negative results from 3 models:

distilbert-base-uncased-finetuned-sst-2-english

textattack/distilbert-base-uncased-SST-2

textattack/roberta-base-SST-2

🤖 CI (GitHub Actions)
Workflow at .github/workflows/ci.yml:

Caches pip + Hugging Face models

Installs deps

Runs pytest

Builds the Docker image (smoke check)

🛠 Troubleshooting
FATAL: password authentication failed
Your DATABASE_URL user/pass mismatch. Fix either the DB password or the URL.

could not connect to server: Connection refused
Postgres isn’t running or wrong port.

Model download is slow on first run
First call downloads weights; cached afterwards.

GET /favicon.ico 404
Harmless—add static/favicon.svg and link it in <head> if you want.

📄 License
MIT (or your preferred license)
