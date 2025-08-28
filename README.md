# 🧠 Sentiment Analysis API (FastAPI · Docker · Hugging Face)

Small **microservice** that predicts whether a text is **positive** or **negative**, and returns a **confidence score**.  
It’s built with **FastAPI**, uses the lightweight **DistilBERT SST-2** model, and includes a **minimal HTML page** and **Swagger** docs.

**Live demo (Hugging Face Space):** https://huggingface.co/spaces/iyassayh/sentiment-ia-docker

---

## ✨ Features

- `POST /predict` → `{ text, sentiment, confidence, summary }`
- `GET /health` → quick liveness check
- `GET /docs` → interactive Swagger UI
- Minimal front page at `/` to test via browser (`/static/index.html`)
- Containerized (Docker) and deployable on **Hugging Face Spaces**
- (Bonus) **Tests** (`pytest`) and **Benchmarks** (CLI + Gradio UI)

---

## 🧰 Tech Stack

- **Python 3.10+**, **FastAPI**, **Uvicorn**
- **transformers** (Hugging Face), model: `distilbert-base-uncased-finetuned-sst-2-english`
- **Docker** (port **7860** inside container)
- Optional: **Gradio** for a simple benchmark UI

---

## 🗂 Project Structure

neww/
├─ app/
│ ├─ init.py
│ └─ main.py # FastAPI app (endpoints + model)
├─ static/
│ └─ index.html # minimal UI to call /predict
├─ tests/
│ ├─ init.py
│ └─ test_api.py # health, success, error tests
├─ benchmark/
│ ├─ benchmark.py # CLI benchmark (3 models)
│ └─ benchmark_ui_single_table.py # Gradio UI (one table, 3 models)
├─ Dockerfile
├─ requirements.txt
├─ pytest.ini
└─ .github/
└─ workflows/
└─ ci.yml # GitHub Actions: run tests + docker build

yaml
Copy code

---

## 🚀 Quickstart (Local)

### 1) Install dependencies
```bash
pip install -r requirements.txt
2) Run the API (Uvicorn)
bash
Copy code
# from the project root
uvicorn app.main:app --reload --port 7860
# open http://127.0.0.1:7860
3) Try it
Front page: http://127.0.0.1:7860/

Swagger: http://127.0.0.1:7860/docs

Health: http://127.0.0.1:7860/health

POST /predict (curl)

bash
Copy code
curl -X POST "http://127.0.0.1:7860/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"I love this product!\"}"
POST /predict (PowerShell)

powershell
Copy code
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:7860/predict" `
  -Body (@{ text = "I love this product!" } | ConvertTo-Json) `
  -ContentType "application/json"
🐳 Run with Docker (Local)
bash
Copy code
# 1) build the image
docker build -t sentiment-api .

# 2) run the container (map 7860 -> 7860)
docker run --rm -p 7860:7860 sentiment-api

# open http://127.0.0.1:7860/
☁️ Deploy on Hugging Face Spaces (Docker)
Create a Space → SDK: Docker → Public.

Push these files to the Space root: Dockerfile, requirements.txt, the whole app/ and static/ folders.

Important (Spaces cache permissions): this project’s Dockerfile sets HF caches to a writable path:

dockerfile
Copy code
ENV HF_HOME=/home/user/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/user/.cache/huggingface/transformers
ENV HUGGINGFACE_HUB_CACHE=/home/user/.cache/huggingface/hub
Commit → watch Build logs → when ready, open App.

Troubleshooting (Spaces)

If you see PermissionError: /root/.cache/huggingface/token, ensure the ENV cache lines above exist and the Dockerfile creates the directories:

dockerfile
Copy code
RUN mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HUGGINGFACE_HUB_CACHE" \
    && chmod -R 777 /home/user/.cache
📘 API Reference
POST /predict
Request

json
Copy code
{ "text": "I love this!" }
Response

json
Copy code
{
  "text": "I love this!",
  "sentiment": "positive",
  "confidence": 0.9973,
  "summary": "I love this!-positive-0.9973"
}
Errors

Empty/whitespace text → 400 with {"detail": "...must not be empty..."}

JSON validation errors (missing field, too long) → 422 (FastAPI/Pydantic)

GET /health
json
Copy code
{ "status": "ok" }
GET /docs
Interactive docs (Swagger UI).

🧪 Tests
Run all tests (uses tests/test_api.py):

bash
Copy code
pytest -q
What’s covered:

/health returns {"status":"ok"}

/predict success shape & values

/predict errors (whitespace, empty string, missing field, too long)

🏎️ Benchmarks
CLI (3 models)
bash
Copy code
python benchmark/benchmark.py
Prints a Markdown summary (avg/min/max/median/p95 latency + simple accuracy) comparing:

distilbert-base-uncased-finetuned-sst-2-english

textattack/distilbert-base-uncased-SST-2

textattack/roberta-base-SST-2 (no sentencepiece needed)

Gradio UI (one table, only labels)
bash
Copy code
python benchmark/benchmark_ui_single_table.py
# opens a local UI, shows a single table with POS/NEG from 3 models
🤖 CI (GitHub Actions)
A simple workflow at .github/workflows/ci.yml:

Caches pip + Hugging Face models

Runs pytest on every push/PR

Builds the Docker image (smoke check)

🖼️ Screenshots (add yours)
/docs Swagger page

/ HTML test page with a sample response

Place images in a /docs/ folder and link them here.

📎 Notes & Tips
Model: distilbert-base-uncased-finetuned-sst-2-english (CPU-friendly)

Container serves on port 7860 (EXPOSE 7860)

For Windows PowerShell users, use backticks ( ` ) for multiline commands.

📄 License
MIT (or your preferred license)

