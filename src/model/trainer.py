import os, json, yaml
from pathlib import Path
from typing import List, Dict

import torch
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import Dataset


def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


class MedicalDataset(Dataset):
    """PyTorch dataset that reads from a JSONL file."""

    def __init__(self, filepath, tokenizer, max_input=512, max_target=128):
        self.tokenizer  = tokenizer
        self.max_input  = max_input
        self.max_target = max_target
        with open(filepath, encoding="utf-8") as f:
            self.records = [json.loads(l) for l in f if l.strip()]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        enc = self.tokenizer(
            rec["clinical_text"],
            max_length=self.max_input,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        dec = self.tokenizer(
            text_target=rec["summary"],
            max_length=self.max_target,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        labels = dec["input_ids"].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         labels,
        }


def build_model(model_name="facebook/bart-large-cnn", lora_config=None):
    """
    Load BART and wrap it with LoRA adapters.
    LoRA adds tiny trainable layers — we only train those,
    not the full 400M parameter model.
    """
    print(f"Loading base model: {model_name}")
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model     = BartForConditionalGeneration.from_pretrained(model_name)

    print(f"Base model params: {sum(p.numel() for p in model.parameters()):,}")

    # Apply LoRA
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=lora_config.get("r", 16) if lora_config else 16,
        lora_alpha=lora_config.get("lora_alpha", 32) if lora_config else 32,
        lora_dropout=lora_config.get("lora_dropout", 0.1) if lora_config else 0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} ({100*trainable/total:.2f}% of total)")

    return model, tokenizer


def train(config_path="config.yaml"):
    """Full training pipeline — call this to start fine-tuning."""
    cfg = load_config(config_path)

    # Build model + tokenizer
    model, tokenizer = build_model(
        model_name=cfg["model"]["base_model"],
        lora_config=cfg.get("lora"),
    )

    # Load datasets
    print("\nLoading datasets...")
    train_ds = MedicalDataset(
        os.path.join(cfg["data"]["processed_dir"], "train.jsonl"),
        tokenizer,
        max_input=cfg["data"]["max_input_length"],
        max_target=cfg["data"]["max_target_length"],
    )
    val_ds = MedicalDataset(
        os.path.join(cfg["data"]["processed_dir"], "val.jsonl"),
        tokenizer,
        max_input=cfg["data"]["max_input_length"],
        max_target=cfg["data"]["max_target_length"],
    )
    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg["model"]["output_dir"],
        num_train_epochs=cfg["training"]["epochs"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        per_device_eval_batch_size=cfg["training"]["batch_size"],
        learning_rate=cfg["training"]["learning_rate"],
        warmup_steps=cfg["training"]["warmup_steps"],
        weight_decay=cfg["training"]["weight_decay"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        evaluation_strategy="steps",
        eval_steps=cfg["training"]["eval_steps"],
        save_steps=cfg["training"]["save_steps"],
        logging_steps=cfg["training"]["logging_steps"],
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),      # Use fp16 only on GPU
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",                    # Set to "wandb" if you want tracking
    )

    # Data collator handles padding within each batch
    collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train
    print("\nStarting training...")
    print(f"  Epochs:     {cfg['training']['epochs']}")
    print(f"  Batch size: {cfg['training']['batch_size']}")
    print(f"  Device:     {'GPU' if torch.cuda.is_available() else 'CPU (slow - use Colab for real training)'}")
    trainer.train()

    # Save final model
    final_dir = cfg["model"]["final_dir"]
    Path(final_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\nModel saved to {final_dir}")

    return trainer


def generate_summary(clinical_text, model_dir=None, config_path="config.yaml"):
    """
    Generate a plain-English summary from clinical text.
    Falls back to base model if fine-tuned model is not found.
    """
    cfg = load_config(config_path)
    base_model_name = cfg["model"]["base_model"]
    final_dir = model_dir or cfg["model"]["final_dir"]

    print(f"Loading tokenizer from base model: {base_model_name}")
    tokenizer = BartTokenizer.from_pretrained(base_model_name)

    # Try loading fine-tuned model, fall back to base
    model_loaded = False
    if os.path.isdir(final_dir) and os.listdir(final_dir):
        try:
            from peft import PeftModel
            base = BartForConditionalGeneration.from_pretrained(base_model_name)
            model = PeftModel.from_pretrained(base, final_dir)
            tokenizer = BartTokenizer.from_pretrained(final_dir)
            print("Loaded fine-tuned LoRA model")
            model_loaded = True
        except Exception as e:
            print(f"Could not load fine-tuned model: {e}")

    if not model_loaded:
        print("Using base model (no fine-tuned model found yet)")
        model = BartForConditionalGeneration.from_pretrained(base_model_name)

    model.eval()

    inputs = tokenizer(
        clinical_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=cfg["app"]["max_length"],
            num_beams=cfg["app"]["num_beams"],
            no_repeat_ngram_size=cfg["app"]["no_repeat_ngram_size"],
            early_stopping=True,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


if __name__ == "__main__":
    print("Testing model build (no training)...")
    model, tokenizer = build_model("facebook/bart-large-cnn")
    print("Model built successfully")

    sample = (
        "Patient is a 67-year-old male with STEMI. "
        "PCI performed on LAD. Discharged on aspirin, clopidogrel, metoprolol."
    )
    summary = generate_summary(sample)
    print(f"\nSample output:\n{summary}")
