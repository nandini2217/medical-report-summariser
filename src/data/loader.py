import os, csv, json, yaml, random
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_sample_data() -> List[Dict]:
    return [
        {
            "clinical_text": (
                "Patient is a 67-year-old male admitted with ST-elevation myocardial "
                "infarction (STEMI). Emergent percutaneous coronary intervention (PCI) "
                "was performed on the LAD with drug-eluting stent placement. Post-procedure "
                "course was uncomplicated. Discharged on aspirin 81mg, clopidogrel 75mg, "
                "metoprolol succinate 50mg, and atorvastatin 80mg. Follow-up with "
                "cardiologist in 2 weeks. Avoid heavy lifting for 4 weeks."
            ),
            "summary": (
                "A 67-year-old man was admitted for a heart attack. A procedure opened "
                "the blocked artery using a small tube called a stent. He is going home "
                "on four heart medicines. He should see his heart doctor in 2 weeks and "
                "avoid heavy lifting for one month."
            )
        },
        {
            "clinical_text": (
                "56-year-old female with type 2 diabetes mellitus and hypertension. "
                "HbA1c of 9.8%, BP 158/96 mmHg. Metformin increased to 1000mg BID. "
                "Lisinopril increased from 10mg to 20mg daily. Referred to nephrology. "
                "Patient advised to monitor blood glucose twice daily and return in 6 weeks."
            ),
            "summary": (
                "A 56-year-old woman with diabetes and high blood pressure came for a "
                "check-up. Her blood sugar and blood pressure were too high, so her "
                "medicines were increased. She should check blood sugar twice a day and "
                "return in 6 weeks for blood tests."
            )
        },
        {
            "clinical_text": (
                "45-year-old male with 3-day history of fever, productive cough, dyspnea. "
                "Chest X-ray shows right lower lobe consolidation consistent with pneumonia. "
                "WBC 14,200. Started azithromycin 500mg day 1 then 250mg days 2-5. "
                "Rest, increase fluids, return to ED if symptoms worsen."
            ),
            "summary": (
                "A 45-year-old man came in with fever, cough, and difficulty breathing. "
                "X-ray showed pneumonia in the right lung. He was given a 5-day antibiotic. "
                "He should rest, drink plenty of fluids, and go to emergency if he gets worse."
            )
        },
    ]


def load_from_huggingface(dataset_name="ccdv/pubmed-summarization",
                           split="train", max_samples=None):
    try:
        from datasets import load_dataset as hf_load
        print(f"Loading {dataset_name} ({split})...")
        ds = hf_load(dataset_name, split=split)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        col_map = {
            "ccdv/pubmed-summarization": ("article", "abstract"),
            "samsum": ("dialogue", "summary"),
        }
        in_col, out_col = col_map.get(dataset_name, ("document", "summary"))
        records = [{"clinical_text": r[in_col].strip(),
                    "summary": r[out_col].strip()}
                   for r in ds if r[in_col].strip() and r[out_col].strip()]
        print(f"  Loaded {len(records):,} samples")
        return records
    except Exception as e:
        print(f"Could not load from HuggingFace: {e}")
        return []


def load_from_csv(filepath: str) -> List[Dict]:
    records = []
    with open(filepath, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row.get("clinical_text","").strip()
            s = row.get("summary","").strip()
            if t and s:
                records.append({"clinical_text": t, "summary": s})
    print(f"  Loaded {len(records):,} samples from CSV")
    return records


def split_dataset(records, train_ratio=0.8, val_ratio=0.1, seed=42):
    random.seed(seed)
    data = records.copy()
    random.shuffle(data)
    n = len(data)
    t = int(n * train_ratio)
    v = int(n * (train_ratio + val_ratio))
    train, val, test = data[:t], data[t:v], data[v:]
    print(f"  Split → train:{len(train)} | val:{len(val)} | test:{len(test)}")
    return train, val, test


def save_split(records, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved {len(records)} records → {path}")


def load_split(filepath):
    with open(filepath, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
