'''
Inference pipeline for the hybrid lay-summary system.

Updated notes
-------------
- support loading a LoRA adapter checkpoint
- computes digests_count on the fly if the column is missing
- move model + inputs to the selected device
'''
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

SECTION_ORDER = ["introduction", "methods", "results", "conclusion"]
SECTION_TAGS = {
    "introduction": "[INTRO]",
    "methods": "[METHODS]",
    "results": "[RESULTS]",
    "conclusion": "[CONCLUSION]",
}
SECTION_INPUT_LABELS = {
    "introduction": "[INTRO_SUMMARY]",
    "methods": "[METHODS_SUMMARY]",
    "results": "[RESULTS_SUMMARY]",
    "conclusion": "[CONCLUSION_SUMMARY]",
}
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
INFERENCE_REQUIRED_COLUMNS = [
    "title",
    "abstract",
    "introduction_digest",
    "methods_digest",
    "results_digest",
    "conclusion_digest",
    "global_digest",
]
INFERENCE_OPTIONAL_COLUMNS = [
    "entry_id",
    "route_hint",
    "section_route_confidence",
    "digests_count",
    "training_quality_bucket",
]
DEFAULT_MODEL_DIR = Path("phase3_outputs/model_outputs/final_checkpoint") # change this to correct checkpoint
DEFAULT_PHASE2_CSV = "inference_testset.csv"
DEFAULT_OUTPUT_CSV = "inference_outputs.csv"
DEFAULT_DEVICE = "cuda"
DEFAULT_ROUTE_THRESHOLD = 0.72
INFERENCE_KEEP_COLUMNS = INFERENCE_REQUIRED_COLUMNS + INFERENCE_OPTIONAL_COLUMNS

def clean_cell(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()

def truncate_words(text: str, max_words: int) -> str:
    tokens = clean_cell(text).split()
    if len(tokens) <= max_words:
        return " ".join(tokens)
    return " ".join(tokens[:max_words]).strip()

def make_section_input(section_name: str, title: str, abstract: str, digest: str) -> str:
    parts = [
        SECTION_TAGS[section_name],
        f"[TITLE] {clean_cell(title)}" if clean_cell(title) else "",
        f"[ABSTRACT] {truncate_words(clean_cell(abstract), 120)}" if clean_cell(abstract) else "",
        f"[DIGEST] {clean_cell(digest)}" if clean_cell(digest) else "",
    ]
    return "\n".join(p for p in parts if p).strip()

def make_global_input(title: str, abstract: str, global_digest: str) -> str:
    parts = [
        "[GLOBAL]",
        f"[TITLE] {clean_cell(title)}" if clean_cell(title) else "",
        f"[ABSTRACT] {truncate_words(clean_cell(abstract), 120)}" if clean_cell(abstract) else "",
        f"[DIGEST] {clean_cell(global_digest)}" if clean_cell(global_digest) else "",
    ]
    return "\n".join(p for p in parts if p).strip()

def make_final_input(section_outputs: Dict[str, str]) -> str:
    parts = ["[FINAL]"]
    for section_name in SECTION_ORDER:
        text = clean_cell(section_outputs.get(section_name, ""))
        if text:
            parts.append(f"{SECTION_INPUT_LABELS[section_name]} {text}")
    return "\n".join(parts).strip()

def pick_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)

def load_model(model_dir: str, base_model: str = "facebook/bart-large-cnn", device: str = "auto"):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_path = Path(model_dir)
    device_obj = pick_device(device)

    has_adapter_cfg = (model_path / "adapter_config.json").exists()
    has_adapter_weights = (
        (model_path / "adapter_model.safetensors").exists()
        or (model_path / "adapter_model.bin").exists()
    )

    if not model_path.exists():
        print(f"[warning] model_dir '{model_dir}' does not exist, falling back to base model: {base_model}")
    elif model_path.exists() and not (has_adapter_cfg and has_adapter_weights):
        print(f"[warning] model_dir '{model_dir}' exists but does not look like a LoRA adapter checkpoint")
        print("[warning] trying to load it as a regular full model folder instead")
    else:
        print(f"[info] loading fine-tuned LoRA checkpoint from: {model_dir}")

    tokenizer_source = str(model_path) if (model_path / "tokenizer_config.json").exists() else base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    if has_adapter_cfg and has_adapter_weights:
        from peft import PeftModel

        base = AutoModelForSeq2SeqLM.from_pretrained(base_model)
        base.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(base, str(model_path))
    else:
        full_model_source = str(model_path) if model_path.exists() else base_model
        model = AutoModelForSeq2SeqLM.from_pretrained(full_model_source)
        model.resize_token_embeddings(len(tokenizer))

    model.to(device_obj)
    model.eval()
    return model, tokenizer, device_obj

def generate_text(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int = 160, num_beams: int = 4, length_penalty: float = 1.0,) -> str:
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=768,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

def summarize_row(row: pd.Series, model, tokenizer, device: torch.device, route_threshold: float = 0.72, run_hybrid_compare: bool = False,) -> Dict[str, object]:
    title = clean_cell(row.get("title", ""))
    abstract = clean_cell(row.get("abstract", ""))
    global_digest = clean_cell(row.get("global_digest", ""))

    confidence = float(row.get("section_route_confidence", 0.0) or 0.0)
    route_hint = clean_cell(row.get("route_hint", ""))

    raw_digests_count = row.get("digests_count", None)
    if raw_digests_count is not None and str(raw_digests_count).strip() != "":
        digests_count = int(float(raw_digests_count))
    else:
        digests_count = sum(
            1 for section_name in SECTION_ORDER
            if len(clean_cell(row.get(f"{section_name}_digest", "")).split()) >= 12
        )

    use_section_route = (
        route_hint == "section_plus_global"
        and confidence >= route_threshold
        and digests_count >= 3
    )

    section_outputs: Dict[str, str] = {}
    for section_name in SECTION_ORDER:
        digest = clean_cell(row.get(f"{section_name}_digest", ""))
        if not use_section_route or len(digest.split()) < 12:
            continue

        prompt = make_section_input(section_name, title, abstract, digest)
        section_outputs[section_name] = generate_text(
            model,
            tokenizer,
            prompt,
            device=device,
            max_new_tokens=96,
            num_beams=4,
        )

    final_summary = ""
    global_summary = ""

    if len(section_outputs) >= 3:
        final_prompt = make_final_input(section_outputs)
        final_summary = generate_text(
            model,
            tokenizer,
            final_prompt,
            device=device,
            max_new_tokens=160,
            num_beams=4,
        )

    if (not final_summary) or run_hybrid_compare:
        global_prompt = make_global_input(title, abstract, global_digest)
        global_summary = generate_text(
            model,
            tokenizer,
            global_prompt,
            device=device,
            max_new_tokens=160,
            num_beams=4,
        )

    chosen_route = "section_plus_final" if final_summary else "global_only"
    chosen_summary = final_summary if final_summary else global_summary

    return {
        "title": title,
        "route_hint": route_hint,
        "section_route_confidence": confidence,
        "digests_count_used": digests_count,
        "chosen_route": chosen_route,
        "chosen_summary": chosen_summary,
        "section_outputs": section_outputs,
        "final_summary": final_summary,
        "global_summary": global_summary,
    }

def parse_args():
    parser = argparse.ArgumentParser()

    # keep these optional so normal usage can just be: python inference_pipeline.py
    parser.add_argument("--model_dir", type=str, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--phase2_csv", type=str, default=DEFAULT_PHASE2_CSV)
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV)

    # default behavior stays "run one row"
    parser.add_argument("--row_index", type=int, default=0)
    parser.add_argument("--entry_id", type=str, default="")

    # if max_rows > 0, run a small batch starting at row_index
    parser.add_argument("--max_rows", type=int, default=0)

    parser.add_argument("--run_hybrid_compare", action="store_true")
    return parser.parse_args()

def get_rows_to_run(df: pd.DataFrame, row_index: int, entry_id: str, max_rows: int):
    if entry_id:
        sub = df[df["entry_id"].astype(str) == str(entry_id)]
        if sub.empty:
            raise ValueError(f"Could not find entry_id={entry_id}")
        idx = int(sub.index[0])
        return [(idx, sub.iloc[0])]

    if row_index < 0 or row_index >= len(df):
        raise IndexError(f"row_index out of range: {row_index}")

    # default behavior: one row only
    if max_rows <= 0:
        return [(row_index, df.iloc[row_index])]

    end_index = min(row_index + max_rows, len(df))
    return [(i, df.iloc[i]) for i in range(row_index, end_index)]

def result_to_record(row_index: int, row: pd.Series, result: Dict[str, object]) -> Dict[str, object]:
    section_outputs = result.get("section_outputs", {}) or {}

    return {
        "row_index": row_index,
        "entry_id": clean_cell(row.get("entry_id", "")),
        "title": result.get("title", ""),
        "route_hint": result.get("route_hint", ""),
        "section_route_confidence": result.get("section_route_confidence", ""),
        "digests_count_used": result.get("digests_count_used", ""),
        "chosen_route": result.get("chosen_route", ""),
        "chosen_summary": result.get("chosen_summary", ""),
        "final_summary": result.get("final_summary", ""),
        "global_summary": result.get("global_summary", ""),
        "introduction_summary": clean_cell(section_outputs.get("introduction", "")),
        "methods_summary": clean_cell(section_outputs.get("methods", "")),
        "results_summary": clean_cell(section_outputs.get("results", "")),
        "conclusion_summary": clean_cell(section_outputs.get("conclusion", "")),
    }

def prepare_inference_df(df: pd.DataFrame) -> pd.DataFrame:
    missing_required = [c for c in INFERENCE_REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(
            "Inference csv is missing required columns: "
            + ", ".join(missing_required)
        )

    out = df.copy()

    for col in INFERENCE_OPTIONAL_COLUMNS:
        if col not in out.columns:
            if col == "section_route_confidence":
                out[col] = 0.0
            elif col == "digests_count":
                out[col] = ""
            else:
                out[col] = ""

    text_cols = [
        "entry_id",
        "title",
        "abstract",
        "introduction_digest",
        "methods_digest",
        "results_digest",
        "conclusion_digest",
        "global_digest",
        "route_hint",
        "training_quality_bucket",
    ]
    for col in text_cols:
        out[col] = out[col].fillna("").astype(str)

    out["section_route_confidence"] = pd.to_numeric(
        out["section_route_confidence"],
        errors="coerce",
    ).fillna(0.0)
    return out[INFERENCE_KEEP_COLUMNS].copy()

def main():
    args = parse_args()

    csv_path = Path(args.phase2_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find csv: {csv_path}")

    raw_df = pd.read_csv(csv_path)
    df = prepare_inference_df(raw_df)

    rows_to_run = get_rows_to_run(
        df=df,
        row_index=args.row_index,
        entry_id=args.entry_id,
        max_rows=args.max_rows,
    )
    model, tokenizer, device = load_model(args.model_dir)

    all_records = []

    for i, row in rows_to_run:
        result = summarize_row(
            row=row,
            model=model,
            tokenizer=tokenizer,
            device=device,
            run_hybrid_compare=args.run_hybrid_compare,
        )
        all_records.append(result_to_record(i, row, result))

        # print full details only when running one paper
        if len(rows_to_run) == 1:
            print("\nTITLE\n-----")
            print(result["title"])
            print("\nCHOSEN ROUTE\n------------")
            print(result["chosen_route"])
            print("\nSUMMARY\n-------")
            print(result["chosen_summary"])

            if args.run_hybrid_compare:
                print("\nSECTION OUTPUTS\n---------------")
                for name, text in result["section_outputs"].items():
                    print(f"{name.upper()}: {text}\n")

                print("FINAL SUMMARY")
                print(result["final_summary"])
                print("\nGLOBAL SUMMARY")
                print(result["global_summary"])
        else:
            print(f"[{i}] {result['title']} -> {result['chosen_route']}")

    out_df = pd.DataFrame(all_records)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"\nSaved {len(out_df)} result(s) to: {output_path}")

if __name__ == "__main__":
    main()