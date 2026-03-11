"""
Rosales, Kean Louis R. 
Ranigo, Gerome
Cipriaso, James

Akkadian Translation
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
import evaluate

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("deep-past-initiative-machine-translation")
OUTPUT_DIR = Path("outputs/akkadian-mt")

MODEL_NAME  = "Helsinki-NLP/opus-mt-mul-en" 
MAX_SRC_LEN = 512
MAX_TGT_LEN = 512
BATCH_SIZE  = 8
GRAD_ACC    = 2          
EPOCHS      = 5
LR          = 5e-5
SEED        = 42

def load_data():
    df = pd.read_csv(DATA_DIR / "train.csv")
    print(f"Loaded {len(df)} training examples")

    # Drop rows where either field is empty
    df = df.dropna(subset=["transliteration", "translation"])
    df = df[df["transliteration"].str.strip() != ""]
    df = df[df["translation"].str.strip() != ""]
    print(f"After cleaning: {len(df)} examples")

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=SEED)
    print(f"Train: {len(train_df)}  |  Val: {len(val_df)}")

    train_ds = Dataset.from_dict({
        "src": train_df["transliteration"].tolist(),
        "tgt": train_df["translation"].tolist(),
    })
    val_ds = Dataset.from_dict({
        "src": val_df["transliteration"].tolist(),
        "tgt": val_df["translation"].tolist(),
    })

    return DatasetDict({"train": train_ds, "validation": val_ds})


def tokenize_dataset(dataset, tokenizer):
    def preprocess(examples):
        model_inputs = tokenizer(
            examples["src"],
            text_target=examples["tgt"],  
            max_length=MAX_SRC_LEN,
            truncation=True,
            padding=False,
        )
        return model_inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=["src", "tgt"],
        desc="Tokenizing",
    )
    return tokenized
    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=["src", "tgt"],
        desc="Tokenizing",
    )
    return tokenized

def train(tokenized_ds, tokenizer, model):
    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, label_pad_token_id=-100, pad_to_multiple_of=8
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC,
        learning_rate=LR,
        warmup_steps=20,
        weight_decay=0.01,
        predict_with_generate=True,  
        generation_max_length=MAX_TGT_LEN,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),   
        seed=SEED,
        report_to="none",               
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(tokenizer),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting training …")
    trainer.train()
    print(f"\nTraining complete. Best model saved to: {OUTPUT_DIR}/best")
    trainer.save_model(str(OUTPUT_DIR / "best"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "best"))
    return trainer

def make_compute_metrics(tokenizer):
    bleu_metric = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds  = tokenizer.batch_decode(preds,   skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels,  skip_special_tokens=True)

        result = bleu_metric.compute(
            predictions=decoded_preds,
            references=[[ref] for ref in decoded_labels],
        )
        return {"bleu": result["score"]}

    return compute_metrics

def generate_submission(tokenizer, model):
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    print(f"\nGenerating translations for {len(test_df)} test sentences …")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = model.to(device)
    model.eval()

    translations = []
    for text in test_df["transliteration"]:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=MAX_SRC_LEN,
            truncation=True,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_TGT_LEN,
                num_beams=4,
                early_stopping=True,
            )
        translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        translations.append(translation)

    submission = pd.DataFrame({
        "id":          test_df["id"],
        "translation": translations,
    })
    submission.to_csv("submission.csv", index=False)
    print("submission.csv written.")
    return submission

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer and model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    raw_ds      = load_data()
    tokenized_ds = tokenize_dataset(raw_ds, tokenizer)

    trainer = train(tokenized_ds, tokenizer, model)

    print("\nFinal evaluation on validation set:")
    metrics = trainer.evaluate()
    print(metrics)

    best_model = AutoModelForSeq2SeqLM.from_pretrained(str(OUTPUT_DIR / "best"))
    generate_submission(tokenizer, best_model)


if __name__ == "__main__":
    main()
