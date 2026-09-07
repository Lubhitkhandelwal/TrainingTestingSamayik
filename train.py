#!/usr/bin/env python3
"""
Fine-tune NLLB-200 distilled 1.3B on the Sāmayik dataset.

Default direction: Sanskrit (san_Deva) -> English (eng_Latn)
Method: 4-bit QLoRA / LoRA

Examples:
    python train.py --smoke_test
    python train.py
    python train.py --epochs 5 --learning_rate 1e-4
    python train.py --resume_from_checkpoint ./outputs/nllb-samayik/checkpoint-1000
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import sacrebleu
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
DATASET_NAME = "acomquest/Saamayik"

SOURCE_LANG = "san_Deva"
TARGET_LANG = "eng_Latn"

MAX_SOURCE_LENGTH = 256
MAX_TARGET_LENGTH = 256

SMOKE_TEST_SAMPLES = 100
SMOKE_TEST_VAL_RATIO = 0.20

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

FULL_EPOCHS = 3
SMOKE_EPOCHS = 3

LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8

NUM_BEAMS = 5
DEFAULT_OUTPUT_DIR = "./outputs/nllb-samayik"


def print_gpu_info():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. Run this script on an NVIDIA CUDA GPU."
        )

    print("=" * 70)
    print("GPU INFORMATION")
    print("=" * 70)
    print("PyTorch :", torch.__version__)
    print("GPU     :", torch.cuda.get_device_name(0))
    print(
        "VRAM    :",
        round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2),
        "GB",
    )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def preprocess_function(examples, tokenizer):
    # Sāmayik format:
    # examples["translation"]["sa"] -> Sanskrit
    # examples["translation"]["en"] -> English
    source_texts = examples["translation"]["sa"]
    target_texts = examples["translation"]["en"]

    model_inputs = tokenizer(
        source_texts,
        max_length=MAX_SOURCE_LENGTH,
        truncation=True,
    )

    labels = tokenizer(
        text_target=target_texts,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def build_model(tokenizer):
    print("=" * 70)
    print("LOADING NLLB-200 1.3B IN 4-BIT")
    print("=" * 70)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    target_id = tokenizer.convert_tokens_to_ids(TARGET_LANG)
    model.config.forced_bos_token_id = target_id
    model.generation_config.forced_bos_token_id = target_id

    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    model = get_peft_model(model, lora_config)

    print("\nTrainable parameters:")
    model.print_trainable_parameters()
    return model


def compute_metrics(eval_preds, tokenizer):
    predictions, labels = eval_preds

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    predictions = tokenizer.batch_decode(
        predictions, skip_special_tokens=True
    )
    labels = tokenizer.batch_decode(
        labels, skip_special_tokens=True
    )

    predictions = [x.strip() for x in predictions]
    labels = [x.strip() for x in labels]

    bleu = sacrebleu.corpus_bleu(predictions, [labels])
    chrf = sacrebleu.corpus_chrf(predictions, [labels])

    return {"bleu": bleu.score, "chrf": chrf.score}


def generate_and_evaluate(model, tokenizer, dataset, output_file, split_name):
    print("=" * 70)
    print("GENERATING:", split_name)
    print("=" * 70)

    model.eval()
    device = next(model.parameters()).device

    predictions = []
    references = []
    sanskrit = []

    for i, example in enumerate(dataset):
        sa = example["translation"]["sa"]
        ref = example["translation"]["en"]

        inputs = tokenizer(
            sa,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SOURCE_LENGTH,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(TARGET_LANG),
                max_length=MAX_TARGET_LENGTH,
                num_beams=NUM_BEAMS,
            )

        pred = tokenizer.batch_decode(
            output, skip_special_tokens=True
        )[0].strip()

        sanskrit.append(sa)
        references.append(ref.strip())
        predictions.append(pred)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(dataset)}")

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references])

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        f.write("sanskrit\treference_english\tmodel_english\n")
        for sa, ref, pred in zip(sanskrit, references, predictions):
            f.write(
                sa.replace("\t", " ")
                + "\t"
                + ref.replace("\t", " ")
                + "\t"
                + pred.replace("\t", " ")
                + "\n"
            )

    metrics = {
        "split": split_name,
        "num_examples": len(dataset),
        "BLEU": bleu.score,
        "ChrF": chrf.score,
    }

    print(f"{split_name} BLEU : {bleu.score:.4f}")
    print(f"{split_name} ChrF  : {chrf.score:.4f}")
    print("Predictions:", output_file)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning of NLLB-200 on Sāmayik."
    )

    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=GRADIENT_ACCUMULATION_STEPS,
    )
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_gpu_info()
    seed_everything(args.seed)

    print("=" * 70)
    print("EXPERIMENT")
    print("=" * 70)
    print("Model   :", MODEL_NAME)
    print("Dataset :", DATASET_NAME)
    print("Source  :", SOURCE_LANG)
    print("Target  :", TARGET_LANG)
    print("Smoke   :", args.smoke_test)

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------
    dataset = load_dataset(DATASET_NAME)
    print("\nDataset:")
    print(dataset)

    for split in dataset:
        print(f"{split:12s}: {len(dataset[split])}")

    if args.smoke_test:
        smoke = dataset["train"].select(range(SMOKE_TEST_SAMPLES))
        splits = smoke.train_test_split(
            test_size=SMOKE_TEST_VAL_RATIO,
            seed=args.seed,
        )
        train_dataset = splits["train"]
        eval_dataset = splits["test"]
    else:
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]

    print("\nTraining examples  :", len(train_dataset))
    print("Validation examples:", len(eval_dataset))

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        src_lang=SOURCE_LANG,
    )

    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data",
    )

    tokenized_eval = eval_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing validation data",
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------
    model = build_model(tokenizer)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    epochs = args.epochs
    if epochs is None:
        epochs = SMOKE_EPOCHS if args.smoke_test else FULL_EPOCHS

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        learning_rate=args.learning_rate,
        weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        fp16=True,
        predict_with_generate=True,
        generation_max_length=MAX_TARGET_LENGTH,
        logging_steps=10,
        report_to="none",
        seed=args.seed,
        remove_unused_columns=True,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # --------------------------------------------------------
    # SAVE ADAPTER
    # --------------------------------------------------------
    final_dir = output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    print("Saved adapter:", final_dir)

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------
    final_metrics = {}

    val_metrics = generate_and_evaluate(
        model,
        tokenizer,
        eval_dataset,
        output_dir / "validation_predictions.tsv",
        "validation",
    )
    final_metrics["validation"] = val_metrics

    if not args.smoke_test:
        test_metrics = generate_and_evaluate(
            model,
            tokenizer,
            dataset["test"],
            output_dir / "test_predictions.tsv",
            "test",
        )
        final_metrics["test"] = test_metrics

        if "test_ood" in dataset:
            ood_metrics = generate_and_evaluate(
                model,
                tokenizer,
                dataset["test_ood"],
                output_dir / "mkb_ood_predictions.tsv",
                "test_ood_MKB",
            )
            final_metrics["test_ood_MKB"] = ood_metrics

    metrics_file = output_dir / "metrics.json"
    metrics_file.write_text(
        json.dumps(final_metrics, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(json.dumps(final_metrics, indent=2))
    print("\nIMPORTANT: final_adapter contains LoRA adapter weights,")
    print("not a standalone merged 1.3B model.")


if __name__ == "__main__":
    main()
