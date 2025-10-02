from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
import tempfile, os, re
from typing import List, Dict, Any
from faster_whisper import WhisperModel

app = FastAPI()

# ===== Regex para capturas úteis =====
MONEY_RE = re.compile(r'(?:(?:€|\bEUR\b)\s?[\d\.\,]+|[\d\.\,]+\s?(?:€|\bEUR\b))', re.IGNORECASE)
DATE_RE = re.compile(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+de\s+[A-Za-zçãé]+\s+de\s+\d{4})\b', re.IGNORECASE)

DEFAULT_KEYWORDS: List[str] = [
    "cartão de crédito", "valor da dívida", "salário", "compromisso", "pagamento",
    "acordo", "desconto", "comprovativo", "transferência", "iban",
    "juros", "prestação", "atraso", "litígio", "garantia", "hipoteca", "fiador"
]

def extract_insights(text: str, keywords: List[str]) -> Dict[str, Any]:
    lower = text.lower()
    key_hits = {kw: lower.count(kw) for kw in keywords if kw in lower}
    sent_split = re.split(r'(?<=[\.\!\?])\s+', text)
    highlights = [s.strip() for s in sent_split if any(kw in s.lower() for kw in key_hits)]
    return {
        "key_hits": key_hits,
        "money_mentions": MONEY_RE.findall(text),
        "dates": DATE_RE.findall(text),
        "highlights": highlights[:20],
    }

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Query("small", description="tiny|base|small|medium|large-v3"),
    language: str = Query("pt"),
    device: str = Query("auto", description="auto|cpu|cuda"),
    compute_type: str = Query("int8", description="int8|int8_float32|float16|float32"),
    keywords: str = Query("", description="separadas por vírgula; vazio usa defaults")
):
    # salva arquivo temporário
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".mp3"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # evita device=None (bug que você viu)
        device_arg = "cpu" if (not device or device.lower() == "auto") else device
        km = [k.strip().lower() for k in keywords.split(",") if k.strip()] or DEFAULT_KEYWORDS

        wmodel = WhisperModel(model, device=device_arg, compute_type=compute_type)
        segments, info = wmodel.transcribe(tmp_path, language=language, vad_filter=True)
        full_text = " ".join([s.text.strip() for s in segments]).strip()

        insights = extract_insights(full_text, km)
        payload = {
            "language": info.language or language,
            "duration": float(info.duration) if info.duration else None,
            "transcript": full_text,
            **insights
        }
        return JSONResponse(payload)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass