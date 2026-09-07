# NLLB-200 + Sāmayik GPU Training

Model: `facebook/nllb-200-distilled-1.3B`  
Dataset: `acomquest/Saamayik`  
Direction: Sanskrit -> English  
Source language: `san_Deva`  
Target language: `eng_Latn`

The script uses 4-bit QLoRA/LoRA so the 1.3B model can be fine-tuned
on a single NVIDIA GPU more practically than full-parameter fine-tuning.

## Install

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Use a CUDA-enabled PyTorch build appropriate for the GPU/driver.

## 100-example smoke test

```bash
python train.py --smoke_test
```

This selects 100 examples from the Sāmayik training split and creates
an 80/20 train/validation split.

This is a pipeline test, not a publishable evaluation.

## Full training

```bash
python train.py
```

The script uses:
- `train` for training
- `validation` for validation/model selection
- `test` only for final in-domain evaluation
- `test_ood` / Mann Ki Baat only for out-of-domain evaluation

## Resume

```bash
python train.py --resume_from_checkpoint ./outputs/nllb-samayik/checkpoint-XXXX
```

## Change training settings

```bash
python train.py --epochs 5 --learning_rate 1e-4
```

## Outputs

```text
outputs/nllb-samayik/
├── checkpoint-*/
├── final_adapter/
├── validation_predictions.tsv
├── test_predictions.tsv
├── mkb_ood_predictions.tsv
└── metrics.json
```

`final_adapter/` is a LoRA adapter, not a merged standalone 1.3B model.

## Research note

The Sāmayik paper evaluates BLEU and ChrF and reserves Mann Ki Baat as an
out-of-domain test set. The paper's reported NLLB result is a zero-shot
comparison; this project is an extension that fine-tunes NLLB on Sāmayik.
Do not compare the 100-example smoke-test scores directly with the paper's
benchmark results.
