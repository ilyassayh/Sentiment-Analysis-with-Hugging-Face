# benchmark_ui_single_table.py
# ------------------------------------------------------------
# One-table comparison of 3 SST-2 models.
# Shows ONLY 'positive' / 'negative' per model for each text.
# ------------------------------------------------------------

import gradio as gr
from transformers import pipeline
from transformers.utils import logging as hf_logging

# Optional: hide noisy HF warnings
hf_logging.set_verbosity_error()

# Models (no sentencepiece required)
MODELS = [
    ("distilbert-base-uncased-finetuned-sst-2-english", "DistilBERT (HF)"),
    ("textattack/distilbert-base-uncased-SST-2",        "TA DistilBERT"),
    ("textattack/roberta-base-SST-2",                   "TA RoBERTa"),
]

# Default example texts
SAMPLES = [
    "I absolutely love this!",
    "This is the best thing ever.",
    "Not bad at all, pretty good.",
    "I hated the experience.",
    "This is terrible and disappointing.",
    "I wouldn't recommend it.",
    "It's okay, nothing special.",
    "I am pleasantly surprised.",
]

# Cache the pipelines so we load each model once
_PIPELINES = {}

def _normalize(label: str) -> str:
    t = (label or "").strip().lower()
    if t in {"positive", "pos", "label_1", "1"}:
        return "positive"
    if t in {"negative", "neg", "label_0", "0"}:
        return "negative"
    # SST-2 shouldn’t return neutral; if it does, treat as negative for this demo
    return "negative"

def _get_pipeline(model_id: str):
    if model_id not in _PIPELINES:
        clf = pipeline("sentiment-analysis", model=model_id, device=-1)  # CPU
        _PIPELINES[model_id] = clf
        try:
            clf("warmup")
        except Exception:
            pass
    return _PIPELINES[model_id]

def _extract_texts(df_like) -> list[str]:
    """
    Accepts a Gradio Dataframe payload (list-of-lists or pandas.DataFrame)
    and returns a clean list of non-empty strings.
    """
    try:
        # If it's a pandas DataFrame
        import pandas as pd  # type: ignore
        if isinstance(df_like, pd.DataFrame):
            values = df_like.iloc[:, 0].astype(str).tolist()
        else:
            # Expect list of rows like [[text], [text], ...]
            values = []
            for row in (df_like or []):
                if isinstance(row, (list, tuple)) and row:
                    values.append(str(row[0]))
                elif isinstance(row, str):
                    values.append(row)
    except Exception:
        # Fallback: try to iterate as list
        values = []
        for row in (df_like or []):
            if isinstance(row, (list, tuple)) and row:
                values.append(str(row[0]))
            elif isinstance(row, str):
                values.append(row)

    # Clean blanks and keep order
    return [t.strip() for t in values if isinstance(t, str) and t.strip()]

def run_table(df_input) -> list[list[str]]:
    """
    For each input text, run all 3 models and return rows:
    [Text, DistilBERT (HF), TA DistilBERT, TA RoBERTa]
    """
    texts = _extract_texts(df_input)
    if not texts:
        texts = SAMPLES[:]  # fallback to defaults

    # Load all models once
    clfs = [(_get_pipeline(mid), pretty) for mid, pretty in MODELS]

    rows = []
    for t in texts:
        row = [t]
        for clf, _pretty in clfs:
            out = clf(t)[0]  # {'label': 'POSITIVE'|'NEGATIVE', 'score': float}
            row.append(_normalize(out["label"]))
        rows.append(row)
    return rows


with gr.Blocks(title="One-Table Sentiment Compare (3 Models)") as demo:
    gr.Markdown(
        "## 📊 One-Table Comparison — 3 SST-2 models\n"
        "Outputs **only** `positive` / `negative` per model.\n\n"
        "**Models:** DistilBERT (HF), TextAttack DistilBERT, TextAttack RoBERTa\n"
        "Edit the texts, then click **Run**."
    )

    with gr.Row():
        # IMPORTANT: type='array' ensures we receive a list-of-lists, not pandas
        inp = gr.Dataframe(
            headers=["Text"],
            value=[[t] for t in SAMPLES],
            label="Input texts (edit/add rows as you like)",
            datatype=["str"],
            type="array",              # <<< key fix so texts appear reliably
            row_count=(len(SAMPLES), "dynamic"),
            col_count=(1, "fixed"),
            interactive=True,
            wrap=True,
        )

    with gr.Row():
        btn = gr.Button("▶️ Run", variant="primary")
        btn_defaults = gr.Button("↺ Reset to examples")

    out = gr.Dataframe(
        headers=["Text"] + [pretty for _mid, pretty in MODELS],
        value=[],
        interactive=False,
        label="Predictions (only positive/negative)",
        wrap=True,
    )

    btn.click(run_table, inputs=[inp], outputs=[out])
    btn_defaults.click(
        fn=lambda: [[t] for t in SAMPLES],
        inputs=None,
        outputs=[inp],
    )

if __name__ == "__main__":
    demo.launch()  # add share=True for a temporary public link
