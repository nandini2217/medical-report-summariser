import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.evaluation.metrics import (
    compute_rouge, compute_readability,
    compute_length_stats, run_full_evaluation
)

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        print(f"  ✅ {name}"); passed += 1
    else:
        print(f"  ❌ {name}"); failed += 1

# Sample data
PREDS = [
    "A 67-year-old man had a heart attack and was treated with a stent procedure.",
    "A 56-year-old woman with diabetes had high blood sugar and her medicine was increased.",
    "A 45-year-old man came in with fever and cough and was found to have pneumonia.",
]
REFS = [
    "A 67-year-old male was admitted for a heart attack and underwent stent placement.",
    "A 56-year-old female with diabetes presented with elevated blood sugar levels.",
    "A 45-year-old male presented with fever, cough, and was diagnosed with pneumonia.",
]
INPUTS = [
    "Patient is a 67yo male with STEMI. Emergent PCI performed on LAD with drug-eluting stent. Discharged on aspirin, clopidogrel, metoprolol, atorvastatin. Follow-up cardiologist 2 weeks.",
    "56yo female with DM2 and HTN. HbA1c 9.8%, BP 158/96. Metformin increased 1000mg BID. Lisinopril 20mg. Monitor glucose twice daily return 6 weeks repeat labs nephrology referral.",
    "45yo male 3-day fever productive cough dyspnea. CXR right lower lobe consolidation pneumonia. WBC 14200. Azithromycin 500mg day 1 then 250mg days 2-5. Rest increase fluids.",
]

print("\n========== Phase 4 Evaluation Tests ==========\n")

# ROUGE tests
print("[ROUGE]")
rouge = compute_rouge(PREDS, REFS)
check("returns dict",           isinstance(rouge, dict))
check("has rouge1 key",         "rouge1" in rouge)
check("has rouge2 key",         "rouge2" in rouge)
check("has rougeL key",         "rougeL" in rouge)
check("rouge1 between 0 and 1", 0 <= rouge["rouge1"] <= 1)
check("rouge2 between 0 and 1", 0 <= rouge["rouge2"] <= 1)
check("rougeL between 0 and 1", 0 <= rouge["rougeL"] <= 1)
check("rouge1 >= rouge2",       rouge["rouge1"] >= rouge["rouge2"])
print(f"     ROUGE-1:{rouge['rouge1']}  ROUGE-2:{rouge['rouge2']}  ROUGE-L:{rouge['rougeL']}")

# Readability tests
print("\n[Readability]")
read = compute_readability(PREDS)
check("returns dict",               isinstance(read, dict))
check("has avg_flesch_ease key",    "avg_flesch_ease"  in read)
check("has avg_grade_level key",    "avg_grade_level"  in read)
check("flesch_ease is a number",    isinstance(read["avg_flesch_ease"], float))

input_read  = compute_readability(INPUTS)
output_read = compute_readability(PREDS)
check("outputs are more readable than inputs",
      output_read["avg_flesch_ease"] > input_read["avg_flesch_ease"])
print(f"     Input ease:{input_read['avg_flesch_ease']}  Output ease:{output_read['avg_flesch_ease']}")

# Length stats tests
print("\n[Length Stats]")
length = compute_length_stats(INPUTS, PREDS)
check("returns dict",                  isinstance(length, dict))
check("has avg_input_words key",       "avg_input_words"  in length)
check("has avg_output_words key",      "avg_output_words" in length)
check("has compression_ratio key",     "compression_ratio" in length)
check("has reduction_pct key",         "reduction_pct" in length)
check("output shorter than input",     length["avg_output_words"] < length["avg_input_words"])
check("reduction_pct is positive",     length["reduction_pct"] > 0)
print(f"     Input:{length['avg_input_words']} words → Output:{length['avg_output_words']} words ({length['reduction_pct']}% reduction)")

# Full evaluation pipeline (skip BERTScore for speed)
print("\n[Full Evaluation Pipeline]")
results = run_full_evaluation(PREDS, REFS, INPUTS, skip_bertscore=True)
check("returns dict",              isinstance(results, dict))
check("has all ROUGE keys",        all(k in results for k in ["rouge1","rouge2","rougeL"]))
check("has readability keys",      "readability_improvement" in results)
check("has compression keys",      "reduction_pct" in results)
check("readability_improvement is float", isinstance(results["readability_improvement"], float))

# Summary
print(f"\n{'='*48}")
print(f"  {passed} passed  |  {failed} failed")
if failed == 0:
    print("  🎉 All Phase 4 tests passed!")
else:
    print("  ⚠️  Fix the failing tests above.")
print(f"{'='*48}\n")
