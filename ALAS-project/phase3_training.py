'''
Build weakly supervised training examples from the phase-2 CSV, exclude the
silver set papers from synthetic training, prepare silver validation/test
JSONL files, and train BART with --train

Training pair examples
-------------------------
- [INTRO]       digest -> pseudo lay intro summary
- [METHODS]     digest -> pseudo lay methods summary
- [RESULTS]     digest -> pseudo lay results summary
- [CONCLUSION]  digest -> pseudo lay conclusion summary
- [GLOBAL]      global_digest -> pseudo full lay summary
- [FINAL]       section mini-summaries -> pseudo full lay summary

To create validation and test sets run the code normally with py phase3_training.py
to train model run the code with py phase3_training.py --train
'''
from __future__ import annotations

import argparse
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import numpy as np
except Exception:
    np = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception as exc:
    raise ImportError(
        "scikit-learn is required for weak label alignment\n"
        "Install with: pip install scikit-learn"
    ) from exc

# -------------------------
# setting block
# -------------------------
PHASE2_DATA = "phase2_preprocessed.csv"
SILVER_DATASET = "silverset_summaries.csv"
OUT_DIR = "phase3_outputs"
BASE_MODEL = "facebook/bart-large-cnn"

RANDOM_SEED = 13

MIN_GLOBAL_FRACTION = 0.12
MAX_ABSTRACT_SENTENCES_PER_SECTION = 2
MIN_ABSTRACT_SENTENCE_CHARS = 30
MIN_DIGEST_WORDS = 12
MIN_TARGET_WORDS = 8

DEFAULT_SILVER_TEXT_COL_CANDIDATES = [
    "second_pass_summary",
    "silver_reference_draft",
]
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
SECTION_KEYWORDS = {
    "introduction": {
        "motivation", "problem", "challenge", "goal", "aim", "task", "why",
        "background", "question", "contribution",
    },
    "methods": {
        "method", "approach", "framework", "model", "algorithm", "architecture",
        "training", "pipeline", "system", "design", "procedure",
    },
    "results": {
        "result", "results", "experiment", "evaluation", "performance", "accuracy",
        "improve", "improvement", "outperform", "better", "benchmark", "analysis",
    },
    "conclusion": {
        "conclusion", "future", "limitation", "impact", "summary", "finding",
        "takeaway", "overall",
    },
}

ARXIV_ID_RE = re.compile(r"([0-9]{4}\.[0-9]{4,5})(?:v\d+)?", re.IGNORECASE)
# -------------------------
# text helpers
# -------------------------
def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)

def clean_cell(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()

def coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    text = clean_cell(value).lower()
    return text in {"1", "true", "t", "yes", "y"}

def word_count(text: str) -> int:
    return len(text.split()) if text else 0

def split_sentences(text: str) -> List[str]:
    text = clean_cell(text)
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    parts = [p.strip() for p in parts if p and p.strip()]
    return [p for p in parts if len(p) >= MIN_ABSTRACT_SENTENCE_CHARS]

def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", clean_cell(text)).strip()

def truncate_words(text: str, max_words: int) -> str:
    tokens = clean_cell(text).split()
    if len(tokens) <= max_words:
        return " ".join(tokens)
    return " ".join(tokens[:max_words]).strip()

def gentle_plain_english_rewrite(text: str) -> str:
    """
    rewrite so the pseudo-target is not copy
    meant to nudge style a bit,
    not hallucinate or heavily paraphrase
    """
    text = normalize_ws(text)

    replacements = [
        (r"\bthis paper\b", "this study"),
        (r"\bwe propose\b", "the authors present"),
        (r"\bwe present\b", "the authors present"),
        (r"\bwe introduce\b", "the authors introduce"),
        (r"\bwe investigate\b", "the study investigates"),
        (r"\bwe study\b", "the study examines"),
        (r"\bwe show\b", "the results show"),
        (r"\bwe demonstrate\b", "the results show"),
        (r"\bobtain\b", "get"),
        (r"\butilize\b", "use"),
        (r"\bapproximately\b", "about"),
        (r"\bnovel\b", "new"),
    ]

    out = text
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out, flags=re.IGNORECASE)

    out = re.sub(r"\s+,\s+", ", ", out)
    out = re.sub(r"\s+\.\s*", ". ", out)
    out = re.sub(r"\s+\)\s*", ") ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out
# -------------------------
# path and key helpers
# -------------------------
def extract_arxiv_key(*values: object) -> str:
    for value in values:
        text = clean_cell(value)
        if not text:
            continue

        match = ARXIV_ID_RE.search(text)
        if match:
            return match.group(1)

        stem = Path(text).stem
        match = ARXIV_ID_RE.search(stem)
        if match:
            return match.group(1)

    return ""

def paper_id_from_row(row: pd.Series) -> str:
    entry_id = clean_cell(row.get("entry_id", ""))
    if entry_id:
        return entry_id

    pdf_url = clean_cell(row.get("pdf_url", ""))
    if pdf_url:
        return pdf_url

    title = clean_cell(row.get("title", ""))
    year = clean_cell(row.get("target_year", ""))
    return f"{title}::{year}"

def add_merge_key_from_phase2(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["merge_key"] = out.apply(
        lambda row: extract_arxiv_key(
            row.get("entry_id", ""),
            row.get("pdf_url", ""),
            row.get("pdf_path", ""),
            row.get("title", ""),
        ),
        axis=1,
    )
    return out

def add_merge_key_from_silver(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["merge_key"] = out.apply(
        lambda row: extract_arxiv_key(
            row.get("entry_id", ""),
            row.get("pdf_filename", ""),
            row.get("title", ""),
        ),
        axis=1,
    )
    return out

def pick_silver_text_column(df: pd.DataFrame, preferred: str = "") -> str:
    preferred = clean_cell(preferred)
    if preferred and preferred in df.columns:
        return preferred

    for col in DEFAULT_SILVER_TEXT_COL_CANDIDATES:
        if col in df.columns:
            return col

    raise ValueError(
        "Could not find a silver reference text column. Expected one of: "
        + ", ".join(DEFAULT_SILVER_TEXT_COL_CANDIDATES)
    )

# -------------------------
# alignment logic
# -------------------------
def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
    )

def keyword_bonus(sentence: str, section_name: str) -> float:
    sent_lc = sentence.lower()
    hits = sum(1 for kw in SECTION_KEYWORDS[section_name] if kw in sent_lc)
    return min(0.20, 0.05 * hits)

def position_bonus(index: int, section_name: str, n_sentences: int) -> float:
    if n_sentences <= 1:
        return 0.0

    rel = index / max(n_sentences - 1, 1)

    if section_name == "introduction":
        return max(0.0, 0.12 - 0.12 * rel)
    if section_name == "methods":
        return 0.06 if 0.15 <= rel <= 0.70 else 0.0
    if section_name == "results":
        return 0.08 if rel >= 0.45 else 0.0
    if section_name == "conclusion":
        return 0.10 if rel >= 0.70 else 0.0
    return 0.0

def align_section_target(
    abstract: str,
    section_digest: str,
    section_name: str,
    max_sentences: int = MAX_ABSTRACT_SENTENCES_PER_SECTION,
) -> str:
    abstract = clean_cell(abstract)
    section_digest = clean_cell(section_digest)

    if not abstract or not section_digest:
        return ""

    if word_count(section_digest) < MIN_DIGEST_WORDS:
        return ""

    sentences = split_sentences(abstract)
    if not sentences:
        return ""

    docs = [section_digest] + sentences
    try:
        tfidf = build_vectorizer().fit_transform(docs)
        sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    except ValueError:
        sims = np.zeros(len(sentences)) if np is not None else [0.0] * len(sentences)

    scored: List[Tuple[float, int, str]] = []
    for i, sent in enumerate(sentences):
        score = float(sims[i])
        score += keyword_bonus(sent, section_name)
        score += position_bonus(i, section_name, len(sentences))
        scored.append((score, i, sent))

    scored.sort(key=lambda x: (-x[0], x[1]))

    chosen = []
    used_idx = set()
    for score, i, sent in scored:
        if score < 0.05 and chosen:
            continue
        if i in used_idx:
            continue
        chosen.append((i, sent))
        used_idx.add(i)
        if len(chosen) >= max_sentences:
            break

    chosen.sort(key=lambda x: x[0])
    target = " ".join(sent for _, sent in chosen).strip()
    target = gentle_plain_english_rewrite(target)
    target = truncate_words(target, 80)

    if word_count(target) < MIN_TARGET_WORDS:
        return ""
    return target

def build_global_target(title: str, abstract: str) -> str:
    abstract = clean_cell(abstract)
    if not abstract:
        return ""

    text = gentle_plain_english_rewrite(abstract)

    if title and not re.match(r"(?i)^(this|the|in|we|our)\b", text):
        text = f"This study is about {title}. {text}"

    return truncate_words(text, 160)

def build_section_pseudo_targets_from_row(row: pd.Series) -> Dict[str, str]:
    abstract = clean_cell(row.get("abstract", ""))
    targets: Dict[str, str] = {}

    for section_name in SECTION_ORDER:
        digest = clean_cell(row.get(f"{section_name}_digest", ""))
        if word_count(digest) < MIN_DIGEST_WORDS:
            continue

        target = align_section_target(abstract, digest, section_name)
        if target:
            targets[section_name] = target

    return targets

# -------------------------
# multitask formatting
# -------------------------
def make_section_input(section_name: str, title: str, abstract: str, digest: str) -> str:
    tag = SECTION_TAGS[section_name]
    parts = [
        tag,
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
# -------------------------
# example building
# -------------------------
@dataclass
class Example:
    paper_id: str
    example_id: str
    task: str
    input_text: str
    target_text: str
    route_hint: str
    quality_bucket: str
    source_type: str
    weight: float
    split: str = "train"
    merge_key: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "paper_id": self.paper_id,
            "example_id": self.example_id,
            "task": self.task,
            "input_text": self.input_text,
            "target_text": self.target_text,
            "route_hint": self.route_hint,
            "quality_bucket": self.quality_bucket,
            "source_type": self.source_type,
            "weight": self.weight,
            "split": self.split,
            "merge_key": self.merge_key,
        }

def build_examples_for_row(row: pd.Series) -> List[Example]:
    if not coerce_bool(row.get("keep_for_training", False)):
        return []

    title = clean_cell(row.get("title", ""))
    abstract = clean_cell(row.get("abstract", ""))
    global_digest = clean_cell(row.get("global_digest", ""))
    route_hint = clean_cell(row.get("route_hint", ""))
    quality_bucket = clean_cell(row.get("training_quality_bucket", ""))
    paper_id = paper_id_from_row(row)
    merge_key = clean_cell(row.get("merge_key", ""))

    examples: List[Example] = []
    section_targets = build_section_pseudo_targets_from_row(row)

    for section_name, target in section_targets.items():
        digest = clean_cell(row.get(f"{section_name}_digest", ""))
        examples.append(
            Example(
                paper_id=paper_id,
                example_id=f"{paper_id}::{section_name}",
                task=SECTION_TAGS[section_name],
                input_text=make_section_input(section_name, title, abstract, digest),
                target_text=target,
                route_hint=route_hint,
                quality_bucket=quality_bucket,
                source_type="section_alignment",
                weight=1.0 if quality_bucket == "good" else 0.7,
                split="train",
                merge_key=merge_key,
            )
        )

    global_target = build_global_target(title, abstract)
    if global_target and word_count(global_digest) >= 25:
        examples.append(
            Example(
                paper_id=paper_id,
                example_id=f"{paper_id}::global",
                task="[GLOBAL]",
                input_text=make_global_input(title, abstract, global_digest),
                target_text=global_target,
                route_hint=route_hint,
                quality_bucket=quality_bucket,
                source_type="global_from_abstract",
                weight=1.0 if quality_bucket == "good" else 0.8,
                split="train",
                merge_key=merge_key,
            )
        )

    if len(section_targets) >= 3 and global_target:
        examples.append(
            Example(
                paper_id=paper_id,
                example_id=f"{paper_id}::final",
                task="[FINAL]",
                input_text=make_final_input(section_targets),
                target_text=global_target,
                route_hint=route_hint,
                quality_bucket=quality_bucket,
                source_type="final_from_section_pseudo_targets",
                weight=1.15 if quality_bucket == "good" else 0.85,
                split="train",
                merge_key=merge_key,
            )
        )

    return examples

def enforce_min_global_fraction(examples: List[Example], min_fraction: float = MIN_GLOBAL_FRACTION) -> List[Example]:
    if not examples:
        return examples

    globals_now = [ex for ex in examples if ex.task == "[GLOBAL]"]
    fraction_now = len(globals_now) / max(len(examples), 1)

    if fraction_now >= min_fraction or not globals_now:
        return examples

    needed = math.ceil((min_fraction * len(examples) - len(globals_now)) / max(1 - min_fraction, 1e-9))
    candidates = [ex for ex in globals_now if ex.quality_bucket == "good"]
    if not candidates:
        candidates = globals_now

    extras: List[Example] = []
    for i in range(max(0, needed)):
        base = candidates[i % len(candidates)]
        extras.append(
            Example(
                paper_id=base.paper_id,
                example_id=f"{base.example_id}::dup{i+1}",
                task=base.task,
                input_text=base.input_text,
                target_text=base.target_text,
                route_hint=base.route_hint,
                quality_bucket=base.quality_bucket,
                source_type=f"{base.source_type}_dup",
                weight=base.weight,
                split=base.split,
                merge_key=base.merge_key,
            )
        )

    return examples + extras

def build_synthetic_dataset(df: pd.DataFrame) -> pd.DataFrame:
    all_examples: List[Example] = []
    for _, row in df.iterrows():
        all_examples.extend(build_examples_for_row(row))

    all_examples = enforce_min_global_fraction(all_examples)
    out = pd.DataFrame([ex.to_dict() for ex in all_examples])

    if out.empty:
        return out

    out["target_word_count"] = out["target_text"].fillna("").astype(str).str.split().str.len()
    out["input_word_count"] = out["input_text"].fillna("").astype(str).str.split().str.len()
    return out

# -------------------------
# silver eval building
# -------------------------
def choose_silver_eval_input(row: pd.Series, mode: str) -> Tuple[Optional[str], str, str]:
    """
    Returns (task, input_text, source_type)
    mode:
      - hybrid: prefer [FINAL] when section route looks usable, else [GLOBAL]
      - global: always build [GLOBAL] when possible
    """
    title = clean_cell(row.get("title", ""))
    abstract = clean_cell(row.get("abstract", ""))
    global_digest = clean_cell(row.get("global_digest", ""))
    route_hint = clean_cell(row.get("route_hint", ""))

    if mode == "global":
        if word_count(global_digest) >= 25:
            return "[GLOBAL]", make_global_input(title, abstract, global_digest), "silver_eval_global"
        return None, "", ""

    section_targets = build_section_pseudo_targets_from_row(row)
    if route_hint == "section_plus_global" and len(section_targets) >= 3:
        return "[FINAL]", make_final_input(section_targets), "silver_eval_hybrid_final"

    if word_count(global_digest) >= 25:
        return "[GLOBAL]", make_global_input(title, abstract, global_digest), "silver_eval_hybrid_global_fallback"

    if len(section_targets) >= 3:
        return "[FINAL]", make_final_input(section_targets), "silver_eval_hybrid_final_fallback"

    return None, "", ""

def build_silver_eval_dataset(
    phase2_df: pd.DataFrame,
    silver_df: pd.DataFrame,
    split_name: str,
    silver_text_col: str,
    mode: str,
) -> pd.DataFrame:
    silver_split = silver_df[silver_df["gold_split"].astype(str).str.lower() == split_name.lower()].copy()
    if silver_split.empty:
        return pd.DataFrame()

    merged = silver_split.merge(
        phase2_df,
        on="merge_key",
        how="left",
        suffixes=("_silver", ""),
    )

    rows: List[Dict[str, object]] = []
    for _, row in merged.iterrows():
        ref_summary = clean_cell(row.get(silver_text_col, ""))
        if not ref_summary:
            continue

        task, input_text, source_type = choose_silver_eval_input(row, mode=mode)
        if not task or not input_text:
            continue

        paper_id = clean_cell(row.get("entry_id_silver", "")) or clean_cell(row.get("entry_id", ""))
        quality_bucket = clean_cell(row.get("training_quality_bucket_silver", "")) or clean_cell(row.get("training_quality_bucket", ""))
        route_hint = clean_cell(row.get("route_hint_silver", "")) or clean_cell(row.get("route_hint", ""))
        merge_key = clean_cell(row.get("merge_key", ""))

        rows.append(
            Example(
                paper_id=paper_id,
                example_id=f"{paper_id}::{split_name.lower()}::{mode}",
                task=task,
                input_text=input_text,
                target_text=ref_summary,
                route_hint=route_hint,
                quality_bucket=quality_bucket,
                source_type=source_type,
                weight=1.0,
                split=split_name.lower(),
                merge_key=merge_key,
            ).to_dict()
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["target_word_count"] = out["target_text"].fillna("").astype(str).str.split().str.len()
    out["input_word_count"] = out["input_text"].fillna("").astype(str).str.split().str.len()
    return out

# -------------------------
# saving helpers
# -------------------------
def save_stats_csv(examples_df: pd.DataFrame, path: Path) -> None:
    if examples_df.empty:
        stats_rows = [{"metric": "n_examples", "value": 0}]
    else:
        stats_rows = [
            {"metric": "n_examples", "value": int(len(examples_df))},
            {"metric": "avg_input_words", "value": round(float(examples_df["input_word_count"].mean()), 2)},
            {"metric": "avg_target_words", "value": round(float(examples_df["target_word_count"].mean()), 2)},
            {"metric": "global_fraction", "value": round(float((examples_df["task"] == "[GLOBAL]").mean()), 4)},
            {"metric": "final_fraction", "value": round(float((examples_df["task"] == "[FINAL]").mean()), 4)},
        ]

        for key, value in examples_df["task"].value_counts().to_dict().items():
            stats_rows.append({"metric": f"task_count::{key}", "value": int(value)})
        for key, value in examples_df["quality_bucket"].value_counts().to_dict().items():
            stats_rows.append({"metric": f"quality_count::{key}", "value": int(value)})
        for key, value in examples_df["route_hint"].value_counts().to_dict().items():
            stats_rows.append({"metric": f"route_count::{key}", "value": int(value)})
        for key, value in examples_df["source_type"].value_counts().to_dict().items():
            stats_rows.append({"metric": f"source_type_count::{key}", "value": int(value)})

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(stats_rows).to_csv(path, index=False)

# -------------------------
# training flag
# -------------------------
def train_model(
    train_csv: Path,
    valid_csv: Optional[Path],
    output_dir: Path,
    base_model: str = BASE_MODEL,
    max_source_length: int = 768,
    max_target_length: int = 192,
    learning_rate: float = 1e-4,
    train_batch_size: int = 4,
    eval_batch_size: int = 4,
    num_train_epochs: int = 3,
    grad_accum: int = 4,
) -> None:
    try:
        from datasets import load_dataset
        import evaluate
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            AutoModelForSeq2SeqLM,
        )
    except ImportError as exc:
        raise ImportError(
            "Training requires: transformers datasets peft evaluate accelerate rouge_score\n"
            "Install with:\n"
            "pip install transformers datasets peft evaluate accelerate rouge_score"
        ) from exc

    data_files = {"train": str(train_csv)}
    if valid_csv and valid_csv.exists():
        data_files["validation"] = str(valid_csv)

    dataset = load_dataset("csv", data_files=data_files)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    def tokenize_batch(batch: Dict[str, Sequence[str]]) -> Dict[str, Sequence[List[int]]]:
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        # some trainer/eval setups return a tuple
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # replace any negative ids before decoding
        if np is not None:
            predictions = np.where(predictions < 0, tokenizer.pad_token_id, predictions)
            labels = np.where(labels < 0, tokenizer.pad_token_id, labels)
            predictions = predictions.astype("int64")
            labels = labels.astype("int64")
        else:
            predictions = [
                [token if token >= 0 else tokenizer.pad_token_id for token in row]
                for row in predictions
            ]
            labels = [
                [token if token >= 0 else tokenizer.pad_token_id for token in row]
                for row in labels
            ]

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        return rouge.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        warmup_ratio=0.03,
        lr_scheduler_type="linear",
        save_total_limit=2,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        generation_max_length=max_target_length,
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="epoch" if "validation" in tokenized else "no",
        load_best_model_at_end=True if "validation" in tokenized else False,
        metric_for_best_model="eval_rougeL" if "validation" in tokenized else None,
        greater_is_better=True,
        report_to="none",
        fp16=False,
        bf16=True,
        tf32=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"] if "validation" in tokenized else None,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=compute_metrics if "validation" in tokenized else None,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "final_checkpoint"))
    tokenizer.save_pretrained(str(output_dir / "final_checkpoint"))
    
# -------------------------
# main
# -------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver_text_col", type=str, default="", help="optional override for the silver reference text column")
    parser.add_argument("--out_dir", type=str, default=OUT_DIR)
    parser.add_argument("--train", action="store_true", help="also fine-tune the model after building CSV datasets")
    parser.add_argument("--valid_csv", type=str, default="", help="optional explicit validation csv override")
    parser.add_argument("--train_quality_buckets", type=str, default="good,borderline", help="comma-separated phase-2 quality buckets to include for synthetic training")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    return parser.parse_args()

def parse_bucket_list(raw: str) -> List[str]:
    buckets = [clean_cell(x).lower() for x in str(raw).split(",") if clean_cell(x)]
    return [b for b in buckets if b]

def main() -> None:
    set_seed(RANDOM_SEED)
    args = parse_args()

    phase2_path = Path(PHASE2_DATA)
    silver_path = Path(SILVER_DATASET)

    if not phase2_path.exists():
        raise FileNotFoundError(f"Could not find pre-processed csv data: {phase2_path}")

    if not silver_path.exists():
        raise FileNotFoundError(f"Could not find silver summary csv dataset: {silver_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase2_df = pd.read_csv(phase2_path)
    phase2_df = add_merge_key_from_phase2(phase2_df)

    silver_df = pd.read_csv(silver_path)
    silver_df = add_merge_key_from_silver(silver_df)
    silver_text_col = pick_silver_text_column(silver_df, preferred=args.silver_text_col)
    heldout_keys = set(k for k in silver_df["merge_key"].astype(str) if k)

    train_phase2_df = phase2_df.copy()
    if heldout_keys:
        train_phase2_df = train_phase2_df[~train_phase2_df["merge_key"].isin(heldout_keys)].copy()

    selected_buckets = set(parse_bucket_list(args.train_quality_buckets))
    if selected_buckets:
        train_phase2_df = train_phase2_df[
            train_phase2_df["training_quality_bucket"].astype(str).str.lower().isin(selected_buckets)
        ].copy()

    synth_df = build_synthetic_dataset(train_phase2_df)
    if synth_df.empty:
        raise ValueError("No synthetic examples were built. Check your phase-2 CSV and filters.")

    train_csv = out_dir / "synthetic_train.csv"
    stats_csv = out_dir / "synthetic_stats.csv"

    synth_df.to_csv(train_csv, index=False)
    save_stats_csv(synth_df, stats_csv)

    print(f"phase2 csv: {phase2_path}")
    print(f"train phase2 rows after holdout exclusion + bucket filter: {len(train_phase2_df)}")
    print(f"training buckets included: {sorted(selected_buckets) if selected_buckets else 'all'}")
    print(f"built synthetic examples: {len(synth_df)}")
    print(f"saved train csv: {train_csv}")
    print(f"saved stats csv: {stats_csv}")

    silver_manifest = silver_df[[c for c in silver_df.columns if c in {
        "gold_split", "target_year", "entry_id", "title", "main_category",
        "route_hint", "training_quality_bucket", silver_text_col, "merge_key"
    }]].copy()
    silver_manifest.to_csv(out_dir / "silver_holdout_manifest.csv", index=False)

    val_hybrid = build_silver_eval_dataset(
        phase2_df=phase2_df,
        silver_df=silver_df,
        split_name="validation",
        silver_text_col=silver_text_col,
        mode="hybrid",
    )
    test_hybrid = build_silver_eval_dataset(
        phase2_df=phase2_df,
        silver_df=silver_df,
        split_name="test",
        silver_text_col=silver_text_col,
        mode="hybrid",
    )
    val_global = build_silver_eval_dataset(
        phase2_df=phase2_df,
        silver_df=silver_df,
        split_name="validation",
        silver_text_col=silver_text_col,
        mode="global",
    )
    test_global = build_silver_eval_dataset(
        phase2_df=phase2_df,
        silver_df=silver_df,
        split_name="test",
        silver_text_col=silver_text_col,
        mode="global",
    )

    for df_obj, stem in [
        (val_hybrid, "silver_validation_hybrid"),
        (test_hybrid, "silver_test_hybrid"),
        (val_global, "silver_validation_global"),
        (test_global, "silver_test_global"),
    ]:
        if not df_obj.empty:
            df_obj.to_csv(out_dir / f"{stem}.csv", index=False)
            save_stats_csv(df_obj, out_dir / f"{stem}_stats.csv")
            print(f"saved {stem}: {len(df_obj)} rows")

    if val_hybrid.empty:
        raise ValueError("silver_validation_hybrid.csv came out empty")
    auto_valid_csv = out_dir / "silver_validation_hybrid.csv"

    exact_overlap = phase2_df[phase2_df["merge_key"].isin(heldout_keys)].shape[0]
    print(f"silver csv: {silver_path}")
    print(f"silver text column: {silver_text_col}")
    print(f"silver holdout papers: {len(heldout_keys)}")
    print(f"phase2 rows matched to silver holdout: {exact_overlap}")

    if args.train:
        if clean_cell(args.valid_csv):
            valid_csv = Path(args.valid_csv)
        else:
            valid_csv = auto_valid_csv

        train_model(
            train_csv=train_csv,
            valid_csv=valid_csv,
            output_dir=out_dir / "model_outputs",
            base_model=args.base_model,
        )

if __name__ == "__main__":
    ''' uncomment to check whether gpu optimization is available
    import torch
    print("cuda available:", torch.cuda.is_available())
    print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
    print("torch cuda version:", torch.version.cuda)
    print("bf16 supported:", torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)
    print("tf32 supported:", torch.cuda.is_tf32_supported() if torch.cuda.is_available() else False)
    '''
    main()