import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  ✅ {name}"); passed += 1
    else:
        print(f"  ❌ {name}"); failed += 1

print("\n========== Phase 3 Model Tests ==========\n")

# Test 1: imports work
print("[Imports]")
try:
    from src.model.trainer import build_model, generate_summary, MedicalDataset
    check("trainer.py imports OK", True)
except Exception as e:
    check(f"trainer.py imports OK ({e})", False)

# Test 2: config loads
print("\n[Config]")
try:
    from src.model.trainer import load_config
    cfg = load_config("config.yaml")
    check("config.yaml loads",          True)
    check("model key exists",           "model" in cfg)
    check("training key exists",        "training" in cfg)
    check("lora key exists",            "lora" in cfg)
    check("base_model key exists",      "base_model" in cfg["model"])
except Exception as e:
    check(f"config loads ({e})", False)

# Test 3: dataset class works
# Test 3: dataset class works
print("\n[Dataset]")
try:
    import json, tempfile
    from transformers import BartTokenizer

    tok = BartTokenizer.from_pretrained("facebook/bart-base")
    sample = {"clinical_text": "Patient admitted with chest pain and shortness of breath.",
              "summary": "Patient came in with chest pain."}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(sample) + "\n")
        fname = f.name

    ds = MedicalDataset(fname, tok, max_input=64, max_target=32)
    item = ds[0]

    check("dataset __len__ works",        len(ds) == 1)
    check("item has input_ids",           "input_ids" in item)
    check("item has attention_mask",      "attention_mask" in item)
    check("item has labels",              "labels" in item)
    check("labels use -100 for padding",  (item["labels"] == -100).any().item())
    os.unlink(fname)

except Exception as e:
    check(f"dataset test ({e})", False)

# Test 4: model builds with LoRA
print("\n[Model build + LoRA]")
try:
    model, tokenizer = build_model("facebook/bart-base")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    pct       = 100 * trainable / total
    check("model builds without error",   True)
    check("tokenizer loads",              tokenizer is not None)
    check("LoRA reduces trainable params",pct < 10.0)
    print(f"     Trainable: {trainable:,} / {total:,} ({pct:.2f}%)")
except Exception as e:
    check(f"model build ({e})", False)

# Test 5: inference works
print("\n[Inference]")
try:
    from transformers import BartForConditionalGeneration, BartTokenizer
    import torch

    # Use base model directly — no fine-tuned model needed for this test
    model_name = "facebook/bart-base"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.eval()

    text = "Patient is a 67yo male admitted with STEMI. PCI performed. Discharged on aspirin."
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=64,
            num_beams=2,
            early_stopping=True,
        )

    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    check("generate returns a string",   isinstance(summary, str))
    check("output is non-empty",         len(summary.strip()) > 5)
    print(f"     Output: {summary[:80]}")

except Exception as e:
    check(f"inference failed: {e}", False)

print(f"\n==========================================")
print(f"  {passed} passed  |  {failed} failed")
if failed == 0:
    print("  Phase 3 tests all passed!")
else:
    print("  Fix the failing tests above.")
print("==========================================\n")
