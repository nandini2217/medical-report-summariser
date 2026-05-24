import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.loader import load_sample_data, split_dataset, save_split, load_split
from src.data.preprocessor import (clean_text, remove_phi, truncate_text,
                                    preprocess_record, preprocess_dataset,
                                    get_text_statistics)
import tempfile

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name}")
        failed += 1

print("\n========== Phase 2 Tests ==========\n")

# Loader tests
print("[Loader]")
s = load_sample_data()
check("returns a list",           isinstance(s, list))
check("has 3+ samples",           len(s) >= 3)
check("has clinical_text key",    all("clinical_text" in r for r in s))
check("has summary key",          all("summary" in r for r in s))
check("texts are non-empty",      all(len(r["clinical_text"]) > 20 for r in s))

# Split tests
print("\n[Splitter]")
big = s * 20
tr, va, te = split_dataset(big)
check("parts sum to total",       len(tr)+len(va)+len(te) == len(big))
check("train is ~80%",            0.75 <= len(tr)/len(big) <= 0.85)
check("val   is ~10%",            0.07 <= len(va)/len(big) <= 0.13)

# Save/load roundtrip
print("\n[Save / Load]")
with tempfile.TemporaryDirectory() as tmp:
    p = os.path.join(tmp, "test.jsonl")
    save_split(s, p)
    loaded = load_split(p)
    check("count matches after reload",  len(loaded) == len(s))
    check("content matches after reload",loaded[0]["summary"] == s[0]["summary"])

# clean_text tests
print("\n[clean_text]")
check("collapses spaces",     clean_text("a   b") == "a b")
check("strips whitespace",    clean_text("  hi  ") == "hi")
check("handles empty string", clean_text("") == "")
check("handles None",         clean_text(None) == "")

# remove_phi tests
print("\n[remove_phi]")
phi = "Admitted 01/15/2024. MRN: 9876. Room 4B. Age 67."
out = remove_phi(phi)
check("removes date",    "[DATE]"     in out)
check("removes MRN",     "[MRN]"      in out)
check("removes room",    "[LOCATION]" in out)
check("keeps age",       "67"         in out)

# truncate_text
print("\n[truncate_text]")
long = " ".join([f"w{i}" for i in range(500)])
check("truncates long text",  len(truncate_text(long, 100).split()) <= 100)
check("short text unchanged", truncate_text("hello world", 100) == "hello world")

# preprocess_record
print("\n[preprocess_record]")
good = s[0].copy()
result = preprocess_record(good)
check("valid record → dict",    isinstance(result, dict))
check("output has both keys",   "clinical_text" in result and "summary" in result)
short = {"clinical_text":"tiny","summary":"also"}
check("too-short → None",       preprocess_record(short) is None)

# Dataset stats
print("\n[get_text_statistics]")
proc = preprocess_dataset(s)
st = get_text_statistics(proc)
check("has input_words key",   "input_words"  in st)
check("has target_words key",  "target_words" in st)
check("input avg > target avg",st["input_words"]["avg"] > st["target_words"]["avg"])

# Summary
print(f"\n====================================")
print(f"  {passed} passed  |  {failed} failed")
if failed == 0:
    print("  🎉 All Phase 2 tests passed!")
else:
    print("  ⚠️  Fix the failing tests above.")
print("====================================\n")
