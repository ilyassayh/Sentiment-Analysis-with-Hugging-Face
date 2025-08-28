"""
tests/test_api.py
-------------------------------------------------------
Small, readable tests for your FastAPI sentiment API.

WHAT WE TEST
1) /health returns {"status":"ok"}.
2) /docs (Swagger UI) is reachable.
3) /predict returns a valid response (200) for normal input.
4) /predict returns clear errors for bad inputs:
   - whitespace-only text -> 400 (our own check in the endpoint)
   - empty string -> 422 (Pydantic validation: min_length=1)
   - missing 'text' field -> 422 (Pydantic validation)
   - too-long text (> 2000 chars) -> 422 (Pydantic validation)

BONUS CHECK
- "summary" follows "text-sentiment-confidence" convention:
  we check it ends with "<sentiment>-<confidence with 4 decimals>".

HOW TO RUN
- Install test dep:  pip install pytest
- Run from project root (where the "app" folder is):  pytest -q

NOTE
- First run may be slow because the model is downloaded once.
"""

from fastapi.testclient import TestClient
from app.main import app  # this imports your FastAPI app

# Create a reusable client for all tests
client = TestClient(app)


# ---------- HEALTH & DOCS ----------

def test_health_ok():
    """Basic liveness check."""
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_docs_available():
    """
    Swagger UI should be available at /docs.
    We just check 200 and the returned content looks like HTML.
    """
    res = client.get("/docs")
    assert res.status_code == 200
    # Fast check it's HTML (content-type may vary across versions)
    assert "text/html" in res.headers.get("content-type", "").lower() \
        or "<!DOCTYPE html>" in res.text


# ---------- PREDICT: HAPPY PATH ----------

def test_predict_success_and_summary_format():
    """
    Send a normal sentence and validate:
    - HTTP 200
    - keys exist
    - sentiment is 'positive' or 'negative'
    - confidence is between 0 and 1
    - summary ends with "<sentiment>-<confidence rounded to 4 decimals>"
    """
    payload = {"text": "I absolutely love this product!"}
    res = client.post("/predict", json=payload)
    assert res.status_code == 200, res.text

    data = res.json()
    # shape
    assert set(data.keys()) == {"text", "sentiment", "confidence", "summary"}

    # types / ranges
    assert isinstance(data["text"], str)
    assert data["sentiment"] in {"positive", "negative"}
    assert 0.0 <= float(data["confidence"]) <= 1.0
    assert isinstance(data["summary"], str)

    # summary format check:
    # server builds summary = f"{text}-{sentiment}-{confidence:.4f}"
    expected_suffix = f"{data['sentiment']}-{data['confidence']:.4f}"
    assert data["summary"].endswith(expected_suffix)


# ---------- PREDICT: ERROR CASES ----------

def test_predict_whitespace_only_returns_400():
    """
    Our endpoint strips the text; if the result is empty,
    we raise HTTP 400 with a clear 'detail' message.
    """
    res = client.post("/predict", json={"text": "   "})
    assert res.status_code == 400
    body = res.json()
    assert "detail" in body
    # optional: assert a specific message substring
    assert "must not be empty" in body["detail"].lower()


def test_predict_empty_string_returns_422():
    """
    Empty string violates Pydantic Field(min_length=1),
    so FastAPI returns a 422 validation error BEFORE our endpoint runs.
    """
    res = client.post("/predict", json={"text": ""})
    assert res.status_code == 422
    body = res.json()
    assert "detail" in body  # standard FastAPI validation error shape


def test_predict_missing_text_field_returns_422():
    """
    Missing the required 'text' field also triggers a 422 validation error.
    """
    res = client.post("/predict", json={"message": "hello"})
    assert res.status_code == 422
    body = res.json()
    assert "detail" in body


def test_predict_too_long_returns_422():
    """
    Text longer than the declared max_length=2000 should be rejected by Pydantic.
    """
    too_long = "a" * 2001
    res = client.post("/predict", json={"text": too_long})
    assert res.status_code == 422
    body = res.json()
    assert "detail" in body
