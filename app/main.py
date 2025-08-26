# app/main.py
# ---------------------------------------------------
# Minimal FastAPI app for sentiment analysis.
# Endpoints:
#   GET  /health     -> quick "ok" check
#   GET  /           -> serves the test page (static/index.html)
#   POST /predict    -> returns sentiment + confidence + summary
# Model:
#   distilbert-base-uncased-finetuned-sst-2-english (CPU friendly)
# ---------------------------------------------------

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from transformers import pipeline

# 1) Create your FastAPI app (the web server)
app = FastAPI(title="Sentiment Analysis API")

# 2) Load the Hugging Face model once at startup
#    - First run may download the model files (cached next runs)
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
nlp = pipeline("sentiment-analysis", model=MODEL_NAME)

# 3) Define the input and output shapes (for validation and docs)
class PredictIn(BaseModel):
    # require at least 1 character; cap size to avoid huge payloads
    text: str = Field(..., min_length=1, max_length=2000, description="English text")

class PredictOut(BaseModel):
    text: str
    sentiment: str          # "positive" or "negative"
    confidence: float       # 0..1
    summary: str            # "text-sentiment-confidence"

# 4) Small helper to find the test page (static/index.html)
def _index_path() -> str:
    # file is at: project_root/static/index.html
    project_root = os.path.dirname(os.path.dirname(__file__))  # go up from app/
    return os.path.join(project_root, "static", "index.html")

# 5) Health check
@app.get("/health")
def health():
    return {"status": "ok"}

# 6) Serve the test page at "/"
@app.get("/")
def root():
    path = _index_path()
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(path)

# 7) Main prediction endpoint
@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    # Clean/sanitize input
    text = payload.text.strip()
    if not text:
        # Send a clear 400 error if empty text
        raise HTTPException(status_code=400, detail="Text must not be empty")

    # Run the model (returns list with one dict: {'label': 'POSITIVE', 'score': 0.99})
    try:
        out = nlp(text)[0]
    except Exception:
        # If something unexpected happens inside the model call
        raise HTTPException(status_code=500, detail="Inference error")

    # Normalize + build the "text-sentiment-confidence" summary
    sentiment = out["label"].lower()    # "positive" or "negative"
    confidence = float(out["score"])    # 0..1
    summary = f"{text}-{sentiment}-{confidence:.4f}"

    # Return a clean JSON
    return PredictOut(
        text=text,
        sentiment=sentiment,
        confidence=confidence,
        summary=summary,
    )

# 8) Allow running with:  python app/main.py
if __name__ == "__main__":
    import uvicorn
    # Port 7860 : pratique pour Hugging Face Spaces aussi
    uvicorn.run(app, host="127.0.0.1", port=7860)
