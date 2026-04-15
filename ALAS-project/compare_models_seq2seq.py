'''
This code assumes model: final_checkpoint exists, but this file is too large for upload so
realistically it must be re-trained which may create a different checkpoint name.
After re-training and finding best or final checkpoint, please re-name the model on
line 77 from "final_checkpoint" to whatever the best model checkpoint was during training.
(Assuming training finished and final_checkpoint was created, you may leave the model name
as default for getting the code to just run, but this does not guarantee best model results)
'''

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

SPECIAL_TOKENS = [
    "[INTRO]",
    "[METHODS]",
    "[RESULTS]",
    "[CONCLUSION]",
    "[GLOBAL]",
    "[FINAL]",
    "[TITLE]",
    "[ABSTRACT]",
    "[DIGEST]",
    "[INTRO_SUMMARY]",
    "[METHODS_SUMMARY]",
    "[RESULTS_SUMMARY]",
    "[CONCLUSION_SUMMARY]",
]
DEFAULT_EVAL_CSV = "phase3_outputs/silver_test_hybrid.csv"
DEFAULT_OUTPUT_CSV = "outputs/silver_test_predictions_clean.csv"
DEFAULT_METRICS_CSV = "outputs/silver_test_metrics_clean.csv"

# -------------------------
# reuse from phase2
# -------------------------
try:
    from phase2_preprocess import safe_readability_metrics, split_glued_tokens_in_text
except Exception:
    # fallback so compare script still works even if phase2 imports fail
    def split_glued_tokens_in_text(text: str) -> str:
        return str(text or "").strip()

    def safe_readability_metrics(text: str) -> Dict[str, float]:
        text = str(text or "").strip()
        words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())
        return {
            "word_count": len(words),
            "unique_word_count": len(set(words)),
            "avg_zipf_frequency": 0.0,
            "low_zipf_ratio": 0.0,
            "very_low_zipf_ratio": 0.0,
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "gunning_fog": 0.0,
            "smog_index": 0.0,
        }

# -------------------------
# data structures
# -------------------------
@dataclass
class ModelSpec:
    name: str
    kind: str
    path_or_id: str
    base_model: str = "facebook/bart-large-cnn"

MODELS = [
        # checkpoint below may change name if you had to re-train. check both name and 3rd argument (path or id) both have the correct checkpoint name attached
        ModelSpec("final_checkpoint", "peft_seq2seq", "phase3_outputs/model_outputs/final_checkpoint", "facebook/bart-large-cnn"),
        ModelSpec("bart_large_cnn_base", "seq2seq", "facebook/bart-large-cnn"),
        ModelSpec("pegasus_arxiv", "seq2seq", "google/pegasus-arxiv"),
        ModelSpec("led_base_16384", "seq2seq", "allenai/led-base-16384")
    ]

# -------------------------
# parsing helpers
# -------------------------
def parse_model_spec(raw) -> ModelSpec:
    '''parts = raw.split("|")
    if len(parts) != 4:
        raise ValueError(
            "Each --model_spec must look like: name|kind|path_or_id|base_model"
        )
    name, kind, path_or_id, base_model = [p.strip() for p in parts]
    if base_model == "-":
        base_model = "facebook/bart-large-cnn"'''
    
    return ModelSpec(
        name=raw.name,
        kind=raw.kind,
        path_or_id=raw.path_or_id,
        base_model=raw.base_model,
    )

def pick_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)

def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()

def count_sentences(text: str) -> int:
    text = safe_text(text)
    if not text:
        return 0
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if p and p.strip()]
    return len(parts)

def count_words_simple(text: str) -> int:
    text = safe_text(text)
    if not text:
        return 0
    return len(text.split())

# -------------------------
# model loading
# -------------------------
def load_seq2seq_model(spec: ModelSpec, device: torch.device):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer_source = spec.path_or_id
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    except Exception as exc:
        msg = str(exc).lower()
        if "protobuf" in msg or "sentencepiece" in msg or "convert_slow_tokenizer" in msg:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=False)
        else:
            raise
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    if spec.kind == "peft_seq2seq":
        from peft import PeftModel

        base = AutoModelForSeq2SeqLM.from_pretrained(spec.base_model)
        base.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base, spec.path_or_id)

    elif spec.kind == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(spec.path_or_id)
        model.resize_token_embeddings(len(tokenizer))

    else:
        raise ValueError(f"Unsupported seq2seq model kind: {spec.kind}")

    model.to(device)
    model.eval()
    return model, tokenizer

def load_causal_model(spec: ModelSpec, device: torch.device, load_in_4bit: bool = False):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.path_or_id)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise ImportError(
                "4-bit causal loading needs bitsandbytes-compatible transformers support."
            ) from exc

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            spec.path_or_id,
            quantization_config=quant_cfg,
            device_map="auto",
        )
    else:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else None
        model = AutoModelForCausalLM.from_pretrained(
            spec.path_or_id,
            torch_dtype=torch_dtype,
        )
        model.to(device)

    model.eval()
    return model, tokenizer

def load_model(spec: ModelSpec, device: torch.device, load_in_4bit_causal: bool):
    if spec.kind in {"peft_seq2seq", "seq2seq"}:
        return load_seq2seq_model(spec, device)

    if spec.kind == "causal_lm":
        return load_causal_model(spec, device, load_in_4bit=load_in_4bit_causal)

    if spec.kind == "causal_lm_4bit":
        return load_causal_model(spec, device, load_in_4bit=True)

    raise ValueError(f"Unsupported model kind: {spec.kind}")

# -------------------------
# prompt / generation helpers
# -------------------------
def build_causal_prompt(tokenizer, task: str, input_text: str) -> str:
    system_text = (
        "You explain computer science research papers to non-experts. "
        "Write a faithful plain-language summary that is easier to understand. "
        "Do not use bullet points. Do not mention task tags like [GLOBAL] or [FINAL] in the answer."
    )
    user_text = (
        f"Task tag: {safe_text(task)}\n\n"
        f"Model input:\n{safe_text(input_text)}\n\n"
        "Write one concise layman summary:"
    )

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ]
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    return f"{system_text}\n\n{user_text}"

def generate_seq2seq_one(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_source_length: int,
    max_new_tokens: int,
    num_beams: int,
) -> str:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_source_length,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    # LED needs global attention for summarization
    if getattr(model.config, "model_type", "") == "led":
        global_attention_mask = torch.zeros_like(encoded["input_ids"])
        global_attention_mask[:, 0] = 1
        encoded["global_attention_mask"] = global_attention_mask.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

def generate_causal_one(
    model,
    tokenizer,
    device: torch.device,
    prompt: str,
    max_source_length: int,
    max_new_tokens: int,
    num_beams: int,
) -> str:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_source_length,
    )
    # device_map="auto" models may already be sharded, so only move inputs when safe
    if hasattr(model, "device") and model.device.type != "meta":
        encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=max(1, num_beams),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
        )

    input_len = encoded["input_ids"].shape[1]
    new_tokens = output_ids[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def generate_for_row(
    spec: ModelSpec,
    model,
    tokenizer,
    device: torch.device,
    row: pd.Series,
    max_source_length: int,
    max_new_tokens: int,
    num_beams: int,
) -> str:
    task = safe_text(row.get("task", ""))
    input_text = safe_text(row.get("input_text", ""))

    if spec.kind in {"peft_seq2seq", "seq2seq"}:
        return generate_seq2seq_one(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=input_text,
            max_source_length=max_source_length,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

    if spec.kind in {"causal_lm", "causal_lm_4bit"}:
        prompt = build_causal_prompt(tokenizer, task=task, input_text=input_text)
        return generate_causal_one(
            model=model,
            tokenizer=tokenizer,
            device=device,
            prompt=prompt,
            max_source_length=max_source_length,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )

    raise ValueError(f"Unsupported model kind: {spec.kind}")

# -------------------------
# metric helpers
# -------------------------
def build_row_metrics(input_text: str, target_text: str, prediction: str) -> Dict[str, object]:
    cleaned_prediction = split_glued_tokens_in_text(safe_text(prediction))
    pred_metrics = safe_readability_metrics(cleaned_prediction)

    input_word_count = count_words_simple(input_text)
    target_word_count = count_words_simple(target_text)
    prediction_sentence_count = count_sentences(cleaned_prediction)

    prediction_word_count = int(pred_metrics.get("word_count", 0))
    prediction_unique_word_count = int(pred_metrics.get("unique_word_count", 0))

    avg_sentence_len = (
        round(prediction_word_count / prediction_sentence_count, 4)
        if prediction_sentence_count > 0 else 0.0
    )

    compression_vs_input = (
        round(prediction_word_count / max(input_word_count, 1), 4)
        if input_word_count > 0 else 0.0
    )

    compression_vs_target = (
        round(prediction_word_count / max(target_word_count, 1), 4)
        if target_word_count > 0 else None
    )

    abs_gap_vs_target = (
        abs(prediction_word_count - target_word_count)
        if target_word_count > 0 else None
    )

    return {
        "prediction_cleaned_for_metrics": cleaned_prediction,
        "prediction_word_count": prediction_word_count,
        "prediction_unique_word_count": prediction_unique_word_count,
        "prediction_avg_zipf_frequency": pred_metrics.get("avg_zipf_frequency", 0.0),
        "prediction_low_zipf_ratio": pred_metrics.get("low_zipf_ratio", 0.0),
        "prediction_very_low_zipf_ratio": pred_metrics.get("very_low_zipf_ratio", 0.0),
        "prediction_flesch_reading_ease": pred_metrics.get("flesch_reading_ease", 0.0),
        "prediction_flesch_kincaid_grade": pred_metrics.get("flesch_kincaid_grade", 0.0),
        "prediction_gunning_fog": pred_metrics.get("gunning_fog", 0.0),
        "prediction_smog_index": pred_metrics.get("smog_index", 0.0),
        "prediction_sentence_count": prediction_sentence_count,
        "prediction_avg_sentence_length": avg_sentence_len,
        "input_word_count": input_word_count,
        "target_word_count": target_word_count,
        "compression_ratio_vs_input": compression_vs_input,
        "compression_ratio_vs_target": compression_vs_target,
        "abs_word_count_gap_vs_target": abs_gap_vs_target,
    }

def maybe_compute_rouge(df: pd.DataFrame) -> pd.DataFrame:
    if "target_text" not in df.columns:
        return pd.DataFrame()

    usable = df[df["target_text"].fillna("").astype(str).str.strip() != ""].copy()
    if usable.empty:
        return pd.DataFrame()

    try:
        import evaluate
    except Exception:
        return pd.DataFrame()

    rouge = evaluate.load("rouge")
    rows = []

    for model_name, sub in usable.groupby("model_name"):
        refs = sub["target_text"].fillna("").astype(str).tolist()
        preds = sub["prediction"].fillna("").astype(str).tolist()
        scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
        rows.append({"model_name": model_name, **scores})

    return pd.DataFrame(rows)

def maybe_compute_bertscore(df: pd.DataFrame, do_compute: bool = False) -> pd.DataFrame:
    if not do_compute:
        return pd.DataFrame()

    if "target_text" not in df.columns:
        return pd.DataFrame()

    usable = df[df["target_text"].fillna("").astype(str).str.strip() != ""].copy()
    if usable.empty:
        return pd.DataFrame()

    try:
        import evaluate
    except Exception:
        return pd.DataFrame()

    bertscore = evaluate.load("bertscore")
    rows = []

    for model_name, sub in usable.groupby("model_name"):
        refs = sub["target_text"].fillna("").astype(str).tolist()
        preds = sub["prediction"].fillna("").astype(str).tolist()
        scores = bertscore.compute(
            predictions=preds,
            references=refs,
            lang="en",
        )
        rows.append(
            {
                "model_name": model_name,
                "bertscore_precision": round(float(sum(scores["precision"]) / len(scores["precision"])), 4),
                "bertscore_recall": round(float(sum(scores["recall"]) / len(scores["recall"])), 4),
                "bertscore_f1": round(float(sum(scores["f1"]) / len(scores["f1"])), 4),
            }
        )

    return pd.DataFrame(rows)

def compute_descriptive_metrics(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "prediction_word_count",
        "prediction_unique_word_count",
        "prediction_avg_zipf_frequency",
        "prediction_low_zipf_ratio",
        "prediction_very_low_zipf_ratio",
        "prediction_flesch_reading_ease",
        "prediction_flesch_kincaid_grade",
        "prediction_gunning_fog",
        "prediction_smog_index",
        "prediction_sentence_count",
        "prediction_avg_sentence_length",
        "input_word_count",
        "target_word_count",
        "compression_ratio_vs_input",
        "compression_ratio_vs_target",
        "abs_word_count_gap_vs_target",
    ]

    rows = []
    for model_name, sub in df.groupby("model_name"):
        row = {
            "model_name": model_name,
            "n_rows": len(sub),
        }
        for col in metric_cols:
            if col in sub.columns:
                series = pd.to_numeric(sub[col], errors="coerce")
                row[f"avg_{col}"] = round(float(series.mean()), 4) if series.notna().any() else None
        rows.append(row)

    return pd.DataFrame(rows)

def merge_metric_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [x for x in frames if x is not None and not x.empty]
    if not nonempty:
        return pd.DataFrame()

    out = nonempty[0].copy()
    for frame in nonempty[1:]:
        out = out.merge(frame, on="model_name", how="outer")

    # keep model_name first
    cols = ["model_name"] + [c for c in out.columns if c != "model_name"]
    return out[cols]

# -------------------------
# main
# -------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_csv", type=str, default=DEFAULT_EVAL_CSV)
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--metrics_csv", type=str, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--max_source_length", type=int, default=768)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--compute_bertscore", action="store_true")
    parser.add_argument("--load_in_4bit_causal", action="store_true")
    parser.add_argument("--model_spec", action="append", default=MODELS, help="Models should be formatted as: name|kind|path_or_id|base_model")
    args = parser.parse_args()

    eval_path = Path(args.eval_csv)
    if not eval_path.exists():
        raise FileNotFoundError(f"Could not find eval csv: {eval_path}")

    df = pd.read_csv(eval_path)
    needed = {"paper_id", "task", "input_text"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    device = pick_device(args.device)
    specs = [parse_model_spec(x) for x in args.model_spec]

    all_rows = []

    for spec in specs:
        print(f"\nLoading model: {spec.name} ({spec.kind})")
        try:
            model, tokenizer = load_model(
                spec=spec,
                device=device,
                load_in_4bit_causal=args.load_in_4bit_causal,
            )
        except Exception as exc:
            print(f"Skipping model {spec.name} because loading failed: {exc}")
            continue

        try:
            for _, row in df.iterrows():
                prediction = generate_for_row(
                    spec=spec,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    row=row,
                    max_source_length=args.max_source_length,
                    max_new_tokens=args.max_new_tokens,
                    num_beams=args.num_beams,
                )

                row_metrics = build_row_metrics(
                    input_text=safe_text(row.get("input_text", "")),
                    target_text=safe_text(row.get("target_text", "")),
                    prediction=prediction,
                )

                out_row = {
                    "model_name": spec.name,
                    "model_kind": spec.kind,
                    "paper_id": safe_text(row.get("paper_id", "")),
                    "task": safe_text(row.get("task", "")),
                    "input_text": safe_text(row.get("input_text", "")),
                    "target_text": safe_text(row.get("target_text", "")),
                    "prediction": prediction,
                }
                out_row.update(row_metrics)
                all_rows.append(out_row)
        except Exception as exc:
            print(f"Skipping remaining rows for model {spec.name} because generation failed: {exc}")
        finally:
            try:
                del model
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache() # free memory before the next model loads

    out_df = pd.DataFrame(all_rows)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved predictions to: {args.output_csv}")

    metrics_csv = args.metrics_csv.strip()
    if metrics_csv:
        desc_df = compute_descriptive_metrics(out_df)
        rouge_df = maybe_compute_rouge(out_df)
        bert_df = maybe_compute_bertscore(out_df, do_compute=args.compute_bertscore)

        metrics_df = merge_metric_frames([desc_df, rouge_df, bert_df])

        if not metrics_df.empty:
            Path(metrics_csv).parent.mkdir(parents=True, exist_ok=True)
            metrics_df.to_csv(metrics_csv, index=False)
            print(f"Saved metrics to: {metrics_csv}")
        else:
            print("Skipped metrics csv because no metrics could be computed.")


if __name__ == "__main__":
    main()