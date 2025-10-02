import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from datetime import timedelta
import sys

from faster_whisper import WhisperModel

# ===== Regex =====
MONEY_RE = re.compile(r'(?:(?:€|\bEUR\b)\s?[\d\.\,]+|[\d\.\,]+\s?(?:€|\bEUR\b))', re.IGNORECASE)
DATE_RE = re.compile(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+de\s+[A-Za-zçãé]+\s+de\s+\d{4})\b', re.IGNORECASE)

def load_keywords(path: Path | None) -> List[str]:
    if path and path.exists():
        return [ln.strip().lower() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [
        "cartão de crédito", "valor da dívida", "salário", "compromisso", "pagamento",
        "acordo", "desconto", "comprovativo", "transferência", "iban",
        "juros", "prestação", "atraso", "litígio", "garantia", "hipoteca", "fiador"
    ]

def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=float(seconds))
    millis = int((td.total_seconds() - int(td.total_seconds())) * 1000)
    return f"{int(td.total_seconds()//3600):02d}:{int((td.total_seconds()%3600)//60):02d}:{int(td.total_seconds()%60):02d},{millis:03d}"

def extract_insights(text: str, keywords: List[str]) -> Dict[str, Any]:
    lower = text.lower()
    key_hits = {kw: lower.count(kw) for kw in keywords if kw in lower}
    money_mentions = MONEY_RE.findall(text)
    dates = DATE_RE.findall(text)
    sent_split = re.split(r'(?<=[\.\!\?])\s+', text)
    highlights = [s.strip() for s in sent_split if any(kw in s.lower() for kw in key_hits)]
    return {
        "key_hits": key_hits,
        "money_mentions": money_mentions,
        "dates": dates,
        "highlights": highlights[:20],
    }

def transcribe_file(model: WhisperModel, audio_path: Path, language_hint: str = "pt") -> Dict[str, Any]:
    segments, info = model.transcribe(str(audio_path), language=language_hint, vad_filter=True)
    transcript_parts, srt_lines = [], []
    for idx, seg in enumerate(segments, start=1):
        transcript_parts.append(seg.text.strip())
        srt_lines.append(str(idx))
        srt_lines.append(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}")
        srt_lines.append(seg.text.strip())
        srt_lines.append("")
    full_text = " ".join(transcript_parts).strip()
    return {
        "text": full_text,
        "srt": "\n".join(srt_lines).strip(),
        "language": info.language or "pt",
        "duration": float(info.duration) if info.duration else None,
        "word_count": len(full_text.split())
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Pasta com .mp3/.wav")
    ap.add_argument("--out", required=True, help="Pasta de saída")
    ap.add_argument("--model", default="small", help="tiny|base|small|medium|large-v3")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--compute_type", default="int8", help="int8|int8_float32|float16|float32")
    ap.add_argument("--keywords", default=None, help="Arquivo .txt com keywords (uma por linha)")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input), Path(args.out)
    if not in_dir.exists():
        sys.exit(f"ERRO: pasta de entrada não existe: {in_dir.resolve()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    keywords = load_keywords(Path(args.keywords) if args.keywords else None)

    # evita device=None
    device_arg = "cpu" if (not args.device or args.device.lower() == "auto") else args.device
    model = WhisperModel(args.model, device=device_arg, compute_type=args.compute_type)

    audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}
    files = [p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in audio_exts]
    if not files:
        sys.exit(f"ERRO: nenhum áudio encontrado em {in_dir.resolve()} (aceitos: {', '.join(sorted(audio_exts))})")

    results = []
    for p in files:
        print(f"Transcrevendo: {p.name} ...")
        data = transcribe_file(model, p, language_hint="pt")
        insights = extract_insights(data["text"], keywords)

        (out_dir / f"{p.stem}.txt").write_text(data["text"], encoding="utf-8")
        (out_dir / f"{p.stem}.srt").write_text(data["srt"], encoding="utf-8")

        summary = {"file": p.name, "language": data["language"], "duration": data["duration"],
                   "word_count": data["word_count"], **insights}
        (out_dir / f"{p.stem}.summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        results.append(summary)

    (out_dir / "index.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Concluído. Saídas em: {out_dir.resolve()}")

if __name__ == "__main__":
    main()