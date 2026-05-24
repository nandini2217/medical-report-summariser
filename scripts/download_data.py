import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data.loader import (load_from_huggingface, load_sample_data,
                              split_dataset, save_split)
from src.data.preprocessor import preprocess_dataset, get_text_statistics


def main(args):
    print("=" * 50)
    print("  Medical Report Summariser — Data Pipeline")
    print("=" * 50)

    # Step 1: Load
    if args.use_samples:
        print("\n[1/4] Using built-in sample data...")
        raw = load_sample_data() * 100
    else:
        print(f"\n[1/4] Downloading from HuggingFace...")
        raw = load_from_huggingface(
            dataset_name=args.dataset,
            split="train",
            max_samples=args.max_samples,
        )

    # Step 2: Clean
    print("\n[2/4] Preprocessing...")
    processed = preprocess_dataset(raw, max_input_words=400)

    # Step 3: Statistics
    stats = get_text_statistics(processed)
    print(f"\n[3/4] Statistics:")
    print(f"  Input  — min:{stats['input_words']['min']} "
          f"max:{stats['input_words']['max']} "
          f"avg:{stats['input_words']['avg']}")
    print(f"  Target — min:{stats['target_words']['min']} "
          f"max:{stats['target_words']['max']} "
          f"avg:{stats['target_words']['avg']}")

    # Step 4: Split and save
    print("\n[4/4] Splitting and saving...")
    train, val, test = split_dataset(processed)
    save_split(train, "data/processed/train.jsonl")
    save_split(val,   "data/processed/val.jsonl")
    save_split(test,  "data/processed/test.jsonl")

    # Save sample for inspection
    os.makedirs("data/samples", exist_ok=True)
    save_split(processed[:5], "data/samples/sample_records.jsonl")

    print("\n✅ Pipeline complete!")
    print(f"   Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",     default="ccdv/pubmed-summarization")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--use_samples", action="store_true")
    main(p.parse_args())
