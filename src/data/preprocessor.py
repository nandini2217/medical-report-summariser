import re
from typing import List, Dict, Optional


def clean_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s*[-_=]{3,}\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def remove_phi(text: str) -> str:
    # Remove dates
    text = re.sub(
        r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"[a-z]*\.?\s+\d{1,2},?\s+\d{4})\b",
        "[DATE]", text, flags=re.IGNORECASE)
    # Remove MRN numbers
    text = re.sub(r"\bMRN[:\s#]+\d+\b", "[MRN]", text, flags=re.IGNORECASE)
    # Remove room/bed numbers
    text = re.sub(r"\b(Room|Bed|Ward)\s+\w+\b", "[LOCATION]", text,
                  flags=re.IGNORECASE)
    return text


def truncate_text(text: str, max_words: int = 400) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    truncated = " ".join(words[:max_words])
    last_period = truncated.rfind(".")
    if last_period > len(truncated) * 0.7:
        truncated = truncated[:last_period + 1]
    return truncated


def preprocess_record(record: Dict, max_input_words=400,
                       apply_phi=True) -> Optional[Dict]:
    clinical = clean_text(record.get("clinical_text", ""))
    summary  = clean_text(record.get("summary", ""))
    if apply_phi:
        clinical = remove_phi(clinical)
    clinical = truncate_text(clinical, max_words=max_input_words)
    if len(clinical.split()) < 20 or len(summary.split()) < 10:
        return None
    return {"clinical_text": clinical, "summary": summary}


def preprocess_dataset(records, max_input_words=400, apply_phi=True):
    original = len(records)
    processed = [r for rec in records
                 if (r := preprocess_record(rec, max_input_words, apply_phi))]
    print(f"  Preprocessing: {len(processed)} kept, {original-len(processed)} dropped")
    return processed


def get_text_statistics(records):
    inp = [len(r["clinical_text"].split()) for r in records]
    tgt = [len(r["summary"].split())       for r in records]
    def st(ls): return {"min":min(ls),"max":max(ls),"avg":round(sum(ls)/len(ls),1)}
    return {"input_words": st(inp), "target_words": st(tgt)}
