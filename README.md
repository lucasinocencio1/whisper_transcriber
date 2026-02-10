# Whisper Transcriber

An open-source Whisper Transcriber - automatic Speech-to-Text
In this case, we have business rules for a import partner.


## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## API (FastAPI)

```bash
uvicorn app:app --reload
```

- **POST /transcribe** — upload an audio file; optional query params: `model`, `language`, `device`, `compute_type`, `keywords` (comma-separated).

## Batch CLI

```bash
python transcribe_batch.py --input ./audios --out ./output --model small
```

Optional: `--keywords keywords.txt` (one keyword per line).


