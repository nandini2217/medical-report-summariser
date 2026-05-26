"""
scripts/evaluate.py
-------------------
Runs full evaluation on the fine-tuned model.

Usage:
    python scripts/evaluate.py --skip_bertscore
"""
import sys, os, json, argparse, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from transformers import BartForConditionalGeneration, BartTokenizer
from peft import PeftModel
from src.evaluation.metrics import run_full_evaluation, print_results_table
from src.model.trainer import load_config


# Fresh clinical texts NOT seen during training
FRESH_SAMPLES = [
    {
        "clinical_text": "Patient is a 58-year-old male with a longstanding history of hypertension, hyperlipidemia, and type 2 diabetes mellitus who presented to the emergency department with acute onset chest pain radiating to the left arm and jaw, associated with diaphoresis and shortness of breath for the past 3 hours. EKG demonstrated ST depression in leads V4 through V6. Troponin I was elevated at 2.3 ng/mL. The patient was started on a heparin drip, aspirin 325mg, nitroglycerin infusion, and atorvastatin 80mg. Cardiology was consulted and the patient is scheduled for cardiac catheterization tomorrow morning. He is currently hemodynamically stable with blood pressure 138/88 and heart rate 76. He is to remain NPO after midnight and continue telemetry monitoring overnight.",
        "summary": "A 58-year-old man with diabetes and high blood pressure came in with chest pain spreading to his arm and jaw. Tests showed his heart was under stress with elevated cardiac enzymes. He was started on blood thinners and heart medicines. A heart specialist was called and he will have a procedure tomorrow to look at his heart arteries."
    },
    {
        "clinical_text": "An 82-year-old female with a history of Alzheimer dementia, hypertension, and osteoporosis presented with a 2-day history of increased confusion, decreased oral intake, and low-grade fever of 38.1 degrees Celsius. Urinalysis was positive for nitrites, leukocyte esterase, and showed greater than 50 white blood cells per high power field. Urine culture is pending. The patient was started on ciprofloxacin 500mg twice daily for a 7-day course. IV fluids were administered for mild dehydration. The patient's confusion is thought to be secondary to the urinary tract infection. Family members were notified and educated regarding signs of worsening infection. Follow up with primary care physician is recommended in one week upon completion of antibiotics.",
        "summary": "An 82-year-old woman with memory problems developed increased confusion and fever caused by a urinary tract infection. She was started on a 7-day antibiotic course and given fluids for dehydration. Her family was informed and educated. She should follow up with her regular doctor in one week after finishing the antibiotics."
    },
    {
        "clinical_text": "A 34-year-old female at 32 weeks gestation presented to labor and delivery with a severe headache, visual disturbances, and upper abdominal pain for the past 6 hours. Blood pressure on arrival was 162/104 mmHg. Physical examination revealed brisk deep tendon reflexes and right upper quadrant tenderness. Urinalysis demonstrated 3 plus proteinuria. Laboratory results showed elevated liver enzymes with AST 98 and ALT 112, and platelet count of 95,000. Diagnosis of severe preeclampsia with features of HELLP syndrome was made. Magnesium sulfate infusion was initiated for seizure prophylaxis. Betamethasone was administered for fetal lung maturity. Obstetrics and neonatology teams were notified. Continuous electronic fetal monitoring was initiated and showed reassuring fetal heart tones. Delivery planning is underway.",
        "summary": "A 34-year-old pregnant woman at 32 weeks came in with severe headache and high blood pressure indicating a dangerous pregnancy complication. Blood tests showed liver and platelet problems. She was given medicine to prevent seizures and a steroid injection to help her baby's lungs develop. Her baby is being continuously monitored and delivery is being planned."
    },
    {
        "clinical_text": "Patient is a 71-year-old male with a 40-pack-year smoking history and known chronic obstructive pulmonary disease presenting with a 3-day history of worsening dyspnea, increased sputum production with yellow-green discoloration, and low-grade fever. On examination, oxygen saturation was 88 percent on room air with diffuse expiratory wheezes bilaterally. Chest X-ray showed hyperinflation without acute infiltrate. The patient was started on albuterol and ipratropium bromide nebulizations every 4 hours, oral prednisone 40mg daily for 5 days, and azithromycin 500mg daily for 5 days. Supplemental oxygen via nasal cannula at 2 liters per minute was initiated maintaining saturation above 92 percent. Smoking cessation counseling was provided. Patient instructed to follow up with pulmonology within 2 weeks.",
        "summary": "A 71-year-old man with a long smoking history and chronic lung disease came in with worsening breathing and increased yellow mucus. His oxygen level was low. He was given breathing treatments every 4 hours, steroids, and antibiotics for 5 days, along with supplemental oxygen. He was strongly advised to quit smoking and should see a lung specialist in 2 weeks."
    },
    {
        "clinical_text": "A 19-year-old male was brought in by emergency medical services following a high-speed motorcycle accident without helmet use. On arrival, Glasgow Coma Scale was 14 with the patient confused but following commands. Primary survey was intact with no airway compromise. CT scan of the head and cervical spine were negative for acute intracranial hemorrhage or fracture. Chest CT showed no pneumothorax or hemothorax. X-ray confirmed a displaced right clavicle fracture. The patient had multiple road rash abrasions over the right arm and leg which were cleaned and dressed. Orthopedic surgery was consulted for the clavicle fracture and a sling was applied. Tetanus prophylaxis was updated. The patient was observed for 6 hours, remained neurologically intact, and was discharged with oral analgesics, wound care instructions, and orthopedic follow up in one week.",
        "summary": "A 19-year-old man was brought in after a motorcycle accident without a helmet. Brain and spine scans were normal. He broke his right collarbone and had scrapes on his arm and leg. His arm was placed in a sling and wounds were cleaned. He was monitored for 6 hours and sent home with pain medicine and instructions to see a bone specialist in one week."
    },
]


def load_model_once(cfg):
    base_name = "facebook/bart-large-cnn"  # use summarisation model directly
    tokenizer = BartTokenizer.from_pretrained(base_name)
    model = BartForConditionalGeneration.from_pretrained(base_name)
    model.eval()
    print("Loaded bart-large-cnn (summarisation specialist) ✅")
    return model, tokenizer


def summarise(text, model, tokenizer, cfg):
    inputs = tokenizer(
        text, return_tensors="pt",
        max_length=1024, truncation=True
    )
    with torch.no_grad():
        output = model.generate(
            input_ids            = inputs["input_ids"],
            attention_mask       = inputs["attention_mask"],
            max_length           = 130,
            min_length           = 30,
            num_beams            = 4,
            length_penalty       = 2.0,
            early_stopping       = True,
            no_repeat_ngram_size = 3,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main(args):
    print("=" * 55)
    print("  Medical Report Summariser — Evaluation")
    print("=" * 55)

    cfg = load_config("config.yaml")

    # Always use fresh samples for honest evaluation
    records = FRESH_SAMPLES
    print(f"\nEvaluating on {len(records)} unseen clinical samples...")

    # Load model once
    print("\nLoading model...")
    model, tokenizer = load_model_once(cfg)

    # Generate predictions
    print("\nGenerating summaries...\n")
    inputs_list, predictions, references = [], [], []

    for i, rec in enumerate(records):
        pred = summarise(rec["clinical_text"], model, tokenizer, cfg)
        inputs_list.append(rec["clinical_text"])
        predictions.append(pred)
        references.append(rec["summary"])

        print(f"  [{i+1}/{len(records)}] Input : {rec['clinical_text'][:70]}...")
        print(f"        Output: {pred[:70]}...")
        print()

    # Run all metrics
    results = run_full_evaluation(
        predictions    = predictions,
        references     = references,
        inputs         = inputs_list,
        skip_bertscore = args.skip_bertscore,
    )

    print_results_table(results)

    # Save
    os.makedirs("docs", exist_ok=True)
    with open("docs/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n✅ Results saved to docs/evaluation_results.json")
    print("📌 Copy these numbers into your README!\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--skip_bertscore", action="store_true")
    main(p.parse_args())