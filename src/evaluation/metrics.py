"""
evaluation/metrics.py
---------------------
Computes ROUGE, BERTScore, and Flesch-Kincaid readability
for the Medical Report Summariser.
"""

import json
from typing import List, Dict


def compute_rouge(predictions: List[str], references: List[str]) -> Dict:
    """
    Compute ROUGE-1, ROUGE-2, ROUGE-L scores.

    Args:
        predictions: List of model-generated summaries
        references:  List of ground-truth summaries

    Returns:
        Dict with rouge1, rouge2, rougeL F1 scores
    """
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
    )
    r1, r2, rL = [], [], []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        r1.append(s['rouge1'].fmeasure)
        r2.append(s['rouge2'].fmeasure)
        rL.append(s['rougeL'].fmeasure)

    return {
        'rouge1': round(sum(r1) / len(r1), 4),
        'rouge2': round(sum(r2) / len(r2), 4),
        'rougeL': round(sum(rL) / len(rL), 4),
    }


def compute_bertscore(predictions: List[str],
                       references: List[str],
                       lang: str = 'en') -> Dict:
    """
    Compute BERTScore F1 (meaning-level similarity).

    Args:
        predictions: Model-generated summaries
        references:  Ground-truth summaries
        lang:        Language code (default 'en')

    Returns:
        Dict with bertscore_f1 (average)
    """
    from bert_score import score as bert_score
    print("  Computing BERTScore (this takes ~1 min)...")
    P, R, F1 = bert_score(predictions, references, lang=lang, verbose=False)
    return {
        'bertscore_precision': round(P.mean().item(), 4),
        'bertscore_recall':    round(R.mean().item(), 4),
        'bertscore_f1':        round(F1.mean().item(), 4),
    }


def compute_readability(texts: List[str]) -> Dict:
    """
    Compute Flesch Reading Ease score.
    Higher = easier to read.
      90-100: Very easy (5th grade)
      60-70:  Standard (8th-9th grade)
      30-50:  Difficult (college level)
      0-30:   Very confusing (professional)

    Args:
        texts: List of text strings to score

    Returns:
        Dict with avg_flesch_ease and avg_grade_level
    """
    import textstat
    ease_scores  = [textstat.flesch_reading_ease(t)      for t in texts]
    grade_scores = [textstat.flesch_kincaid_grade(t)     for t in texts]
    return {
        'avg_flesch_ease':  round(sum(ease_scores)  / len(ease_scores),  2),
        'avg_grade_level':  round(sum(grade_scores) / len(grade_scores), 2),
    }


def compute_length_stats(inputs: List[str],
                          predictions: List[str]) -> Dict:
    """
    Compare input vs output length (compression ratio).

    Args:
        inputs:      Original clinical texts
        predictions: Generated summaries

    Returns:
        Dict with avg word counts and compression ratio
    """
    input_lens = [len(t.split()) for t in inputs]
    pred_lens  = [len(t.split()) for t in predictions]
    avg_in  = sum(input_lens)  / len(input_lens)
    avg_out = sum(pred_lens)   / len(pred_lens)
    return {
        'avg_input_words':  round(avg_in,  1),
        'avg_output_words': round(avg_out, 1),
        'compression_ratio': round(avg_out / avg_in, 3),
        'reduction_pct':    round((1 - avg_out / avg_in) * 100, 1),
    }


def run_full_evaluation(predictions: List[str],
                         references:  List[str],
                         inputs:      List[str],
                         skip_bertscore: bool = False) -> Dict:
    """
    Run all metrics and return a combined results dict.

    Args:
        predictions:     Model outputs
        references:      Ground truth summaries
        inputs:          Original clinical texts
        skip_bertscore:  Set True to skip slow BERTScore

    Returns:
        Combined dict with all metric scores
    """
    print(f"\nRunning evaluation on {len(predictions)} samples...")
    results = {}

    print("  [1/4] ROUGE scores...")
    results.update(compute_rouge(predictions, references))

    if not skip_bertscore:
        print("  [2/4] BERTScore...")
        results.update(compute_bertscore(predictions, references))
    else:
        print("  [2/4] BERTScore skipped")

    print("  [3/4] Readability — inputs...")
    input_read  = compute_readability(inputs)
    print("  [3/4] Readability — outputs...")
    output_read = compute_readability(predictions)

    results['input_flesch_ease']   = input_read['avg_flesch_ease']
    results['output_flesch_ease']  = output_read['avg_flesch_ease']
    results['input_grade_level']   = input_read['avg_grade_level']
    results['output_grade_level']  = output_read['avg_grade_level']
    results['readability_improvement'] = round(
        output_read['avg_flesch_ease'] - input_read['avg_flesch_ease'], 2
    )

    print("  [4/4] Length statistics...")
    results.update(compute_length_stats(inputs, predictions))

    return results


def print_results_table(results: Dict) -> None:
    """Pretty-print evaluation results to terminal."""
    print("\n" + "="*50)
    print("  EVALUATION RESULTS")
    print("="*50)

    sections = [
        ("ROUGE Scores", [
            ("ROUGE-1",  results.get('rouge1',  'N/A')),
            ("ROUGE-2",  results.get('rouge2',  'N/A')),
            ("ROUGE-L",  results.get('rougeL',  'N/A')),
        ]),
        ("BERTScore", [
            ("Precision", results.get('bertscore_precision', 'N/A')),
            ("Recall",    results.get('bertscore_recall',    'N/A')),
            ("F1",        results.get('bertscore_f1',        'N/A')),
        ]),
        ("Readability (Flesch Ease — higher = simpler)", [
            ("Input  (clinical text)", results.get('input_flesch_ease',  'N/A')),
            ("Output (summary)",       results.get('output_flesch_ease', 'N/A')),
            ("Improvement",            results.get('readability_improvement', 'N/A')),
        ]),
        ("Length / Compression", [
            ("Avg input words",   results.get('avg_input_words',  'N/A')),
            ("Avg output words",  results.get('avg_output_words', 'N/A')),
            ("Reduction",         str(results.get('reduction_pct', 'N/A')) + '%'),
        ]),
    ]

    for section_name, metrics in sections:
        print(f"\n  {section_name}")
        print("  " + "-"*40)
        for name, val in metrics:
            print(f"  {name:<35} {val}")

    print("\n" + "="*50)
