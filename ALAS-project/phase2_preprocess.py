import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet" # silence spacy's git features
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import time
from wordfreq import zipf_frequency
from functools import lru_cache
import textstat

import pandas as pd
from pypdf import PdfReader

''' 
HARD CODED PATHS
To run this code with no issues file path should be ".\ProjectFolder\"
and inside this folder there should be 2 files, main.py, and the file in METADATA_CSV
and 1 folder "arxiv_cs_2026_pdfs"
'''
METADATA_ROOT = "arxiv_database_csvs"
METADATA_CSV = "arxiv_cs_2022_2026_raw_papers_combined.csv"
PDF_ROOT = r"all_years_pdfs"
OUTPUT_CSV = "phase2_preprocessed.csv"
SPACY_MODEL = "en_core_web_sm"

# biased TextRank settings
BIASED_BIAS = 10.0
BIASED_DEFAULT_BIAS = 0.0
SECTION_WORD_BUDGET = 220
GLOBAL_WORD_BUDGET = 420
SECTION_LIMIT_PHRASES = 20
SECTION_LIMIT_SENTENCES = 6
GLOBAL_LIMIT_PHRASES = 35
GLOBAL_LIMIT_SENTENCES = 10
MAX_SECTION_CHARS = 120000
MAX_GLOBAL_CHARS = 160000

# terms slightly change the focus
SECTION_NAMES = ["abstract", "introduction", "methods", "results", "conclusion"]
SECTION_BIAS_TERMS = {
    "introduction": [
        "problem", "motivation", "challenge", "goal", "address", "aim", "contribution"
    ],
    "methods": [
        "method", "approach", "framework", "model", "architecture", "training", "algorithm"
    ],
    "results": [
        "results", "experiments", "evaluation", "performance", "accuracy", "outperforms", "improvement"
    ],
    "conclusion": [
        "conclusion", "summary", "future work", "limitations", "impact", "takeaway"
    ],
    "global": [
        "problem", "approach", "method", "results", "performance", "conclusion", "impact"
    ],
}

GENERIC_SECTION_HINTS = {
    "introduction": {
        "introduction", "motivation", "background", "preliminaries",
        "overview", "problem statement", "problem formulation",
    },
    "methods": {
        "method", "methods", "methodology", "approach", "framework",
        "algorithm", "implementation", "setup", "training",
        "model architecture",
    },
    "results": {
        "results", "experiments", "evaluation", "analysis",
        "ablation", "findings", "empirical study", "performance",
    },
    "conclusion": {
        "conclusion", "conclusions", "discussion", "future work",
        "limitations", "closing remarks", "broader impact",
        "broader impacts",
    },
}

# section alias detection
SECTION_ALIASES = {
    "abstract": [
        r"abstract",
        r"summary",
        r"executive summary",
    ],
    "introduction": [
        r"introduction",
        r"background",
        r"motivation",
        r"overview",
        r"preliminaries",
        r"problem formulation",
        r"problem setup",
    ],
    "methods": [
        r"method",
        r"methods",
        r"methodology",
        r"approach",
        r"approaches",
        r"proposed method",
        r"proposed approach",
        r"materials and methods",
        r"experimental setup",
        r"implementation",
        r"implementation details",
        r"training details",
        r"technical approach",
        r"method overview",
        r"system overview",
        r"framework",
        r"model",
        r"models",
    ],
    "results": [
        r"results",
        r"experiments",
        r"experimental results",
        r"evaluation",
        r"experimental evaluation",
        r"empirical evaluation",
        r"analysis",
        r"ablation",
        r"ablation study",
        r"findings",
        r"performance evaluation",
        r"quantitative results",
        r"qualitative results",
    ],
    "conclusion": [
        r"conclusion",
        r"conclusions",
        r"discussion",
        r"conclusion and discussion",
        r"limitations",
        r"future work",
        r"closing remarks",
        r"broader impacts",
    ],
}

HEADING_BAD_TOKEN_RE = re.compile(
    r"(?i)\b(?:references?|bibliography|appendix|acknowledg(?:e)?ments?|table|figure|fig\.|algorithm)\b"
)
HEADING_PREFIX_RE = re.compile(
    r"^(?:[IVXLC]+\.|\d+(?:\.\d+)*\.?|[A-Z]\. |\(?[a-z]\))\s+",
    re.IGNORECASE,
)
CITATIONISH_RE = re.compile(
    r"(?i)(?:\bet al\.?\b|\barxiv\b|\bdoi\b|\bjournal\b|\bproceedings\b|\bpreprint\b|\bvol\.?\b|\bno\.?\b|\bpp\.?\b)"
)
MATH_INLINE_RE = re.compile(
    r"(?:(?<!\w)[=+\-*/<>≤≥≈∈∑∏√∂λμσΩαβγθ][^\n]{0,120})"
)
LATEX_INLINE_RE = re.compile(
    r"\\(?:begin|end|frac|sum|int|alpha|beta|gamma|theta|lambda|mu|sigma|omega|mathbf|mathrm|mathbb)\b"
)
SYMBOL_HEAVY_LINE_RE = re.compile(
    r"^[^A-Za-z\n]*[=+\-*/<>≤≥≈∈∑∏√∂λμσΩαβγθ0-9][^A-Za-z\n]*$"
)
NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.?)\s+([A-Z][A-Za-z0-9 ,&:\-]{2,80})\s*$"
)
REFERENCES_START_RE = re.compile(
    r"(?im)^\s*(references|bibliography)\s*$"
)
CAMEL_GLUE_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
TOKEN_WITH_PUNCT_RE = re.compile(r"^([^A-Za-z]*)([A-Za-z]+)([^A-Za-z]*)$")
LOWER_GLUE_RE = re.compile(r"^[A-Za-z]{14,}$")
LETTER_DIGIT_GLUE_RE = re.compile(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])")
TITLE_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
BRACKET_CIT_RE = re.compile(r"\[(?:\d+[\s,;\-]*){1,8}\]")
AUTHOR_YEAR_CIT_RE = re.compile(
    r"\((?:[A-Z][A-Za-z'\-]+(?:\s+et\s+al\.)?(?:,\s*)?)+(?:19|20)\d{2}[a-z]?\)",
)
PAGE_NUM_LINE_RE = re.compile(r"^\s*(?:page\s+)?\d+\s*(?:/\s*\d+)?\s*$", re.IGNORECASE)
ARXIV_LINE_RE = re.compile(r"^\s*arXiv:\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b\S+@\S+\b")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
BRACKET_REF_TEXT_RE = re.compile(r"\[[A-Za-z0-9,\-; ]{2,}\]")
THEOREMISH_RE = re.compile(r"(?i)^\s*(?:theorem|lemma|proof|corollary|proposition|definition|remark)\b")
LONG_NOSPACE_RE = re.compile(r"[A-Za-z]{18,}")
EMAIL_OR_URL_RE = re.compile(r"(?i)(?:https?://|www\.|@\w+)")
SENTENCEISH_START_RE = re.compile(r"(?i)^\s*(?:we|our|this paper|in this paper|to|for|here|let us)\b")
# these single-word aliases are too risky to soft-match loosely
SOFT_MATCH_EXACT_ONLY = {"model", "models", "analysis", "findings", "overview", "background"}

_NLP = None

# generic helpers
def normalize_text(text: str) -> str: # this normalizes weird unicode stuff that shows up a lot in pdf text
    text = unicodedata.normalize("NFKC", text)
    replacements = {
        "\u00ad": "",
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\u2013": "-",
        "\u2014": " - ",
        "\u2212": "-",
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def clean_cell(value) -> str:
    if pd.isna(value): # keep missing csv cells from turning into nan strings
        return ""
    return str(value).strip() # trim spaces so the metadata stays consistent

def safe_int(value, default: int = 0) -> int: # convert floats or blank values from pandas into numeric columns
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default
    
def get_word_list(text: str) -> List[str]:
    if not text:
        return []
    return WORD_RE.findall(text.lower())

def split_glued_tokens_in_text(text: str) -> str:
    fixed_tokens = []

    for raw in text.split():
        m = TOKEN_WITH_PUNCT_RE.match(raw)
        if not m:
            fixed_tokens.append(raw)
            continue

        prefix, core, suffix = m.groups()
        fixed_core = split_glued_token(core)
        fixed_tokens.append(prefix + fixed_core + suffix)

    return " ".join(fixed_tokens)

@lru_cache(maxsize=50000)
def split_glued_token(token: str) -> str:
    if not LOWER_GLUE_RE.fullmatch(token):
        return token

    if token.isupper():
        return token

    lower = token.lower()
    whole_zipf = zipf_frequency(lower, "en")
    if whole_zipf >= 4.5:
        return token

    @lru_cache(maxsize=None)
    def solve(i: int, parts_used: int):
        if i == len(lower):
            return (0.0, [])
        if parts_used >= 8:   # was 5
            return (-1e9, None)

        best_score = -1e9
        best_parts = None

        for j in range(i + 2, min(len(lower), i + 20) + 1):
            piece = lower[i:j]
            z = zipf_frequency(piece, "en")

            if z < 1.5:   # was 2.0
                continue

            local_score = z - 2.0   # was z - 2.5

            if j == len(lower):
                total_score = local_score
                total_parts = [piece]
            else:
                rest_score, rest_parts = solve(j, parts_used + 1)
                if rest_parts is None:
                    continue
                total_score = local_score + rest_score
                total_parts = [piece] + rest_parts

            if total_score > best_score:
                best_score = total_score
                best_parts = total_parts

        return best_score, best_parts

    split_score, split_parts = solve(0, 0)

    if split_parts and len(split_parts) >= 2 and split_score >= (whole_zipf - 2.5) + 2.0:
        rebuilt = " ".join(split_parts)
        if token[0].isupper() and token[1:].islower():
            rebuilt = rebuilt.capitalize()
        return rebuilt

    return token

def safe_readability_metrics(text: str) -> Dict[str, float]:
    # keep empty text from crashing the readability calls
    if not text or not text.strip():
        return {
            "word_count": 0,
            "unique_word_count": 0,
            "avg_zipf_frequency": 0.0,
            "low_zipf_ratio": 0.0,
            "very_low_zipf_ratio": 0.0,
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "gunning_fog": 0.0,
            "smog_index": 0.0,
        }

    words = get_word_list(text)
    if not words:
        return {
            "word_count": 0,
            "unique_word_count": 0,
            "avg_zipf_frequency": 0.0,
            "low_zipf_ratio": 0.0,
            "very_low_zipf_ratio": 0.0,
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
            "gunning_fog": 0.0,
            "smog_index": 0.0,
        }

    zipf_scores = [zipf_frequency(word, "en") for word in words]

    # rough rarity buckets
    # around 4+ is common everyday vocab
    # around 3 or below starts getting rarer / more technical
    low_zipf_ratio = sum(score < 4.0 for score in zipf_scores) / len(zipf_scores)
    very_low_zipf_ratio = sum(score < 3.0 for score in zipf_scores) / len(zipf_scores)

    # textstat sometimes throws odd errors on weird extracted text
    try:
        flesch = float(textstat.flesch_reading_ease(text))
    except Exception:
        flesch = 0.0

    try:
        fk_grade = float(textstat.flesch_kincaid_grade(text))
    except Exception:
        fk_grade = 0.0

    try:
        gunning_fog = float(textstat.gunning_fog(text))
    except Exception:
        gunning_fog = 0.0

    try:
        smog = float(textstat.smog_index(text))
    except Exception:
        smog = 0.0

    return {
        "word_count": len(words),
        "unique_word_count": len(set(words)),
        "avg_zipf_frequency": round(sum(zipf_scores) / len(zipf_scores), 4),
        "low_zipf_ratio": round(low_zipf_ratio, 4),
        "very_low_zipf_ratio": round(very_low_zipf_ratio, 4),
        "flesch_reading_ease": round(flesch, 4),
        "flesch_kincaid_grade": round(fk_grade, 4),
        "gunning_fog": round(gunning_fog, 4),
        "smog_index": round(smog, 4),
    }

def guess_canonical_heading(raw_heading: str) -> Optional[Tuple[str, str]]:
    # first try your stricter existing alias logic
    direct = canonical_heading(raw_heading)
    if direct:
        return direct

    heading = normalize_heading_candidate(raw_heading)
    heading_lc = heading.lower()

    if not is_valid_heading_candidate(heading):
        return None

    # fallback soft mapping for papers that use valid but not explicitly listed titles
    for canonical, keywords in GENERIC_SECTION_HINTS.items():
        if heading_lc in keywords:
            return canonical, heading

        if any(heading_lc.startswith(k + " ") or heading_lc.endswith(" " + k) for k in keywords):
            return canonical, heading

        if any(f" {k} " in f" {heading_lc} " for k in keywords if len(k) > 4):
            return canonical, heading

    return None


def split_long_abstract_if_needed(sections: Dict[str, str], section_titles: Dict[str, str],) -> Tuple[Dict[str, str], Dict[str, str]]:
    abstract = sections.get("abstract", "")
    if len(abstract.split()) < 1200:
        return sections, section_titles

    lines = [ln.strip() for ln in abstract.splitlines() if ln.strip()]

    for i, line in enumerate(lines[:400]):
        guessed = guess_canonical_heading(line)
        if guessed and guessed[0] != "abstract":
            canonical, original = guessed
            kept_abstract = normalize_section_body("\n".join(lines[:i]))
            body = normalize_section_body("\n".join(lines[i + 1:]))

            if len(body.split()) >= 40:
                sections["abstract"] = kept_abstract
                sections[canonical] = body
                section_titles[canonical] = original
                return sections, section_titles

        m = NUMBERED_HEADING_RE.match(line)
        if m:
            heading_text = normalize_heading_candidate(m.group(1))
            kept_abstract = normalize_section_body("\n".join(lines[:i]))
            body = normalize_section_body("\n".join(lines[i + 1:]))

            if len(body.split()) >= 40:
                sections["abstract"] = kept_abstract
                sections["introduction"] = body
                section_titles["introduction"] = heading_text
                return sections, section_titles

    return sections, section_titles

def repair_spacing_artifacts(text: str) -> str:
    # fixes common pdf extraction glue like "Thesecomplexalgorithms"
    # and things like "Definition3.2"
    text = CAMEL_GLUE_RE.sub(" ", text)
    text = LETTER_DIGIT_GLUE_RE.sub(" ", text)
    text = re.sub(r"(?<=[,.;:])(?=[A-Za-z])", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # new pass for lowercase glued words
    text = split_glued_tokens_in_text(text)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_equationish_noise(text: str) -> str:
    cleaned_lines: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue

        # remove latex commands, don't drop whole line
        line = LATEX_INLINE_RE.sub(" ", line)

        # remove inline $...$ math
        line = re.sub(r"\$[^$]{1,200}\$", " ", line)

        # remove obvious equation fragments, but keep prose around them
        line = MATH_INLINE_RE.sub(" ", line)

        alpha = sum(ch.isalpha() for ch in line)
        symbolish = sum((not ch.isalnum()) and (not ch.isspace()) for ch in line)

        # only drop truly equation-only lines
        if alpha == 0 and symbolish >= 3:
            continue

        if SYMBOL_HEAVY_LINE_RE.match(line):
            continue

        line = re.sub(r"\s+", " ", line).strip()
        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()

def light_sanitize(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def remove_references_tail(text: str) -> str:
    m = REFERENCES_START_RE.search(text)
    if m:
        return text[:m.start()].strip()
    return text

def looks_like_broken_text(text: str) -> bool:
    if not text:
        return True

    words = text.split()
    if not words:
        return True

    long_glued = sum(1 for w in words if LONG_NOSPACE_RE.search(w))
    symboly_lines = 0

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        alpha = sum(ch.isalpha() for ch in line)
        symbolish = sum((not ch.isalnum()) and (not ch.isspace()) for ch in line)
        if len(line) < 160 and alpha > 0 and symbolish / max(alpha + symbolish, 1) > 0.55:
            symboly_lines += 1

    return (long_glued / max(len(words), 1) > 0.08) or (symboly_lines >= 8)


def section_is_usable(section_name: str, body: str) -> bool:
    cleaned = sanitize_for_summarizer(body)
    word_count = len(cleaned.split())

    min_words = 20 if section_name == "conclusion" else 35
    if word_count < min_words:
        return False

    # only reject if text is really bad and also short-ish
    if looks_like_broken_text(cleaned) and word_count < 150:
        return False

    return True

def keyword_set(*chunks: str) -> List[str]:
    words: List[str] = []
    for chunk in chunks: # filter out generic academic words so the bias terms stay more useful
        words.extend(w.lower() for w in TITLE_WORD_RE.findall(chunk))
    stop = {
        "with", "from", "that", "this", "using", "based", "into", "under", "their",
        "study", "paper", "model", "models", "approach", "approaches", "method", "methods",
        "results", "result", "analysis", "data", "learning", "research", "system", "systems",
    }
    seen = [] # seen logic also keeps the order while removing duplicates
    added = set()
    for word in words:
        if word in stop or word in added:
            continue
        seen.append(word)
        added.add(word)
    return seen

# spaCy + PyTextRank loader
def get_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP

    '''
    Separately import TextRank dependencies because spacy raises many errors
    Like trying to use Git features without it being installed and having to be
    installed in virtual environments... the code bellow and git silencing at the start
    should handle errors but in case of any unexpected errors please contact me.
    Additional commands that might fix issues if they arise
    are: python -m pip install "spacy==3.7.5" "pytextrank==3.3.0"
    and: python -m pip install --upgrade pip setuptools wheel
    '''
    try:
        import spacy
        import pytextrank 
    except ImportError as exc:
        raise ImportError(
            "The program encountered an error trying to load spacy and pytextrank\n"
            "Please install them using the following commands:\n"
            "pip install spacy pytextrank\n"
            f"python -m spacy download {SPACY_MODEL}"
        ) from exc

    try:
        nlp = spacy.load(SPACY_MODEL, disable=["ner"])
    except OSError as exc:
        raise OSError(
            f"Could not load spaCy model '{SPACY_MODEL}'.\n"
            f"Run: python -m spacy download {SPACY_MODEL}"
        ) from exc

    nlp.max_length = max(nlp.max_length, 5_000_000)

    if "biasedtextrank" not in nlp.pipe_names:
        nlp.add_pipe("biasedtextrank")
    _NLP = nlp
    return _NLP

# pdf extraction
def arxiv_id_from_pdf_url(pdf_url: str) -> Optional[str]: # turn pdf url into the arxiv id so it can match local filenames
    if not pdf_url:
        return None
    tail = re.sub(r"^https?://(?:export\.)?arxiv\.org/pdf/", "", pdf_url)
    tail = tail.replace(".pdf", "")
    return re.sub(r"v\d+$", "", tail) # normalize pdf versions so files can still match

def build_pdf_index(pdf_root: Path) -> Dict[str, str]: # walk through the pdf folder and build a quick lookup table
    index: Dict[str, str] = {}
    if not pdf_root.exists():
        return index
    for path in pdf_root.rglob("*.pdf"):
        base = re.sub(r"v\d+$", "", path.stem)
        index[base] = str(path)
    return index

def safe_extract_text(page) -> str: # try layout extraction first because it usually keeps research papers cleaner
    try:
        return page.extract_text(extraction_mode="layout") or ""
    except TypeError:
        return page.extract_text() or ""
    except Exception:
        try:
            return page.extract_text() or ""
        except Exception:
            return "" # stop one bad page from killing the whole paper

def extract_pages(pdf_path: str) -> List[str]: # read every page from the pdf and normalize the extracted text
    reader = PdfReader(pdf_path, strict=False)
    pages = [normalize_text(safe_extract_text(page)) for page in reader.pages]
    return [p for p in pages if p and p.strip()]

# cleaning
def detect_repeated_margin_lines(pages: Sequence[str], top_k: int = 3, bottom_k: int = 3) -> set[str]:
    # find headers or footers that repeat across many pages
    counts: Counter[str] = Counter()
    n_pages = 0
    for page in pages:
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        if len(lines) < 5:
            continue
        n_pages += 1
        margin_lines = lines[:top_k] + lines[-bottom_k:]
        for line in margin_lines:
            if 3 <= len(line) <= 160:
                counts[line] += 1
    return {
        line for line, count in counts.items()
        if n_pages >= 3 and count / max(n_pages, 1) >= 0.35
    }

def clean_page_text(page_text: str, repeated_margin_lines: set[str]) -> str:
    # remove page level noise before all pages get merged together
    # things like page numbers emails and arxiv header lines are not useful for summarizing
    cleaned_lines: List[str] = []
    for raw in page_text.splitlines():
        line = raw.strip()
        if not line:
            cleaned_lines.append("")
            continue
        if line in repeated_margin_lines:
            continue
        if PAGE_NUM_LINE_RE.match(line):
            continue
        if ARXIV_LINE_RE.match(line):
            continue
        if EMAIL_RE.search(line) and len(line) < 120:
            continue
        if line.lower().startswith(("permission to make digital", "copyright", "manuscript submitted")):
            continue
        cleaned_lines.append(line)

    # the regex cleanup also fixes broken line wraps from pdf extraction
    text = "\n".join(cleaned_lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n(?=[a-z])", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def clean_full_text(pages: Sequence[str]) -> str:
    repeated = detect_repeated_margin_lines(pages)
    text = "\n\n".join(clean_page_text(page, repeated) for page in pages)
    text = URL_RE.sub("", text)
    text = BRACKET_CIT_RE.sub("", text)
    text = AUTHOR_YEAR_CIT_RE.sub("", text)
    text = re.sub(
        r"(?im)^\s*(?:figure|fig\.|table|algorithm|appendix)\s+[A-Za-z0-9.:-]+\s*.*$",
        "",
        text,
    )
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = remove_references_tail(text)
    return text.strip()

def normalize_heading_candidate(raw_heading: str) -> str:
    heading = normalize_text(raw_heading)
    heading = HEADING_PREFIX_RE.sub("", heading).strip()
    heading = re.sub(r"^[^\w]+|[^\w]+$", "", heading)
    heading = re.sub(r"\s+", " ", heading)
    return heading

def is_valid_heading_candidate(raw_heading: str) -> bool:
    heading = normalize_heading_candidate(raw_heading)
    heading_lc = heading.lower()

    if len(heading_lc) < 3 or len(heading_lc) > 100:
        return False
    word_count = len(heading_lc.split())
    if word_count == 0 or word_count > 8:
        return False
    if HEADING_BAD_TOKEN_RE.search(heading_lc):
        return False
    if CITATIONISH_RE.search(heading_lc):
        return False
    if YEAR_RE.search(heading_lc):
        return False
    if BRACKET_REF_TEXT_RE.search(heading):
        return False
    if THEOREMISH_RE.search(heading):
        return False
    if EMAIL_OR_URL_RE.search(heading):
        return False
    if LONG_NOSPACE_RE.search(heading):
        return False
    if SENTENCEISH_START_RE.search(heading_lc):
        return False
    # headings usually should not look like full running sentences
    if heading.endswith((".", "?", "!")):
        return False
    # reject lines with too much punctuation / symbol noise
    punct_chars = sum(1 for ch in heading if not ch.isalnum() and not ch.isspace())
    if punct_chars / max(len(heading), 1) > 0.18:
        return False
    return True

def alias_soft_match(alias: str, heading_lc: str) -> bool:
    # exact match is always safest
    if re.fullmatch(alias, heading_lc):
        return True

    if alias in SOFT_MATCH_EXACT_ONLY:
        return False

    # allow short controlled variants like:
    # "results and discussion", "discussion and future work", "technical approach details"
    if len(heading_lc.split()) > 6:
        return False

    patterns = [
        rf"^(?:{alias})\s+(?:and|&)\s+[a-z][a-z0-9\-]*(?:\s+[a-z][a-z0-9\-]*){{0,2}}$",
        rf"^(?:{alias})\s+[a-z][a-z0-9\-]*(?:\s+[a-z][a-z0-9\-]*){{0,2}}$",
        rf"^[a-z][a-z0-9\-]*(?:\s+[a-z][a-z0-9\-]*){{0,2}}\s+(?:{alias})$",
    ]
    return any(re.fullmatch(pat, heading_lc) for pat in patterns)

def canonical_heading(raw_heading: str) -> Optional[Tuple[str, str]]:
    heading = normalize_heading_candidate(raw_heading)
    heading_lc = heading.lower()

    if not is_valid_heading_candidate(heading):
        return None

    for canonical, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            if alias_soft_match(alias, heading_lc):
                return canonical, heading

    return None

def insert_heading_breaks(text: str) -> str:
    heading_terms = set()

    for aliases in SECTION_ALIASES.values():
        heading_terms.update(aliases)

    for hints in GENERIC_SECTION_HINTS.values():
        heading_terms.update(re.escape(h) for h in hints)

    heading_union = "|".join(sorted(heading_terms, key=len, reverse=True))

    patterns = [
        rf"(?i)(?<!\n)(\b[IVXLC]+\.\s*(?:{heading_union})(?:\s+(?:and|&)\s+[A-Za-z][A-Za-z\-]*)?\b)",
        rf"(?i)(?<!\n)(\b\d+(?:\.\d+)*\.?\s*(?:{heading_union})(?:\s+(?:and|&)\s+[A-Za-z][A-Za-z\-]*)?\b)",
        rf"(?i)(?<!\n)(\b(?:{heading_union})(?:\s+(?:and|&)\s+[A-Za-z][A-Za-z\-]*)?\b)(?=\s+[A-Z])",
    ]

    for pattern in patterns:
        text = re.sub(pattern, r"\n\1\n", text)

    # numbered heading rescue even if heading text is not in alias list
    text = re.sub(
        r"(?m)(?<!\n)(\b(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.?)\s+[A-Z][A-Za-z0-9 ,&:\-]{2,80}\b)",
        r"\n\1\n",
        text,
    )
    return text

def normalize_section_body(text: str) -> str: # collapse extra blank lines and repeated spaces
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def sectionize_text(text: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    # still keep your heading-break trick because it helps when a heading
    # got glued into the previous paragraph
    text = insert_heading_breaks(text)

    lines = [line.rstrip() for line in text.splitlines()]
    sections: Dict[str, List[str]] = defaultdict(list)
    section_titles: Dict[str, str] = {}
    current = "front_matter"

    for raw_line in lines:
        stripped = raw_line.strip()
        guessed = guess_canonical_heading(stripped)

        if guessed:
            current, original = guessed
            section_titles[current] = original
            continue

        sections[current].append(raw_line)

    if "abstract" not in sections:
        front = "\n".join(sections.get("front_matter", []))
        match = re.search(
            r"(?is)\babstract\b\s*[-—:]?\s*(.+?)(?=\n\s*(?:index terms|keywords|[IVXLC]+\.|1\.\s*introduction|introduction)\b)",
            front,
        )
        if match:
            sections["abstract"] = [match.group(1).strip()]
            section_titles["abstract"] = "Abstract"

    cleaned_sections = {
        name: normalize_section_body("\n".join(chunk_lines))
        for name, chunk_lines in sections.items()
        if name != "front_matter" and normalize_section_body("\n".join(chunk_lines))
    }

    cleaned_sections, section_titles = split_long_abstract_if_needed(cleaned_sections, section_titles)
    return cleaned_sections, section_titles

def recover_missing_sections(cleaned_text: str, sections: Dict[str, str], section_titles: Dict[str, str]):
    missing = [
        name for name in ["introduction", "methods", "results", "conclusion"]
        if not sections.get(name, "").strip()
    ]
    if not missing:
        return sections, section_titles

    raw_lines = [ln.rstrip() for ln in cleaned_text.splitlines()]

    candidates: List[Tuple[int, str, str]] = []
    for i, line in enumerate(raw_lines):
        stripped = line.strip()
        guessed = guess_canonical_heading(stripped)
        if not guessed:
            continue

        canonical, original = guessed
        if canonical in missing:
            candidates.append((i, canonical, original))

    if not candidates:
        return sections, section_titles

    candidates = sorted(candidates, key=lambda x: x[0])

    # keep earliest reasonable candidate for each missing section
    deduped: List[Tuple[int, str, str]] = []
    seen = set()
    for item in candidates:
        if item[1] in seen:
            continue
        deduped.append(item)
        seen.add(item[1])

    for idx, (start_line, canonical, original) in enumerate(deduped):
        if sections.get(canonical, "").strip():
            continue

        end_line = deduped[idx + 1][0] if idx + 1 < len(deduped) else len(raw_lines)
        body = normalize_section_body("\n".join(raw_lines[start_line + 1:end_line]))

        min_words = 20 if canonical == "conclusion" else 40
        if len(body.split()) < min_words:
            continue

        sections[canonical] = body
        section_titles[canonical] = original

    sections, section_titles = split_long_abstract_if_needed(sections, section_titles)
    return sections, section_titles

# biased TextRank digest creation
def sanitize_for_summarizer(text: str) -> str:
    text = normalize_text(text)
    text = strip_equationish_noise(text)
    text = repair_spacing_artifacts(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def truncate_words(text: str, word_budget: int) -> str:
    words = text.split()
    if len(words) <= word_budget:
        return text.strip()
    return " ".join(words[:word_budget]).strip()

def unique_terms(terms: Sequence[str]) -> List[str]:
    output: List[str] = []
    seen = set()
    for term in terms:
        term = term.strip().lower()
        if not term or term in seen:
            continue
        output.append(term)
        seen.add(term)
    return output

def build_focus_terms(section_name: str, title: str, abstract: str) -> List[str]:
    base = SECTION_BIAS_TERMS.get(section_name, [])
    title_and_abstract_terms = keyword_set(title, abstract)
    return unique_terms(base + title_and_abstract_terms[:18])

def cap_chars(text: str, max_chars: int) -> str:
    text = sanitize_for_summarizer(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]

def summarize_with_biased_textrank(text, focus_terms, word_budget, limit_phrases, limit_sentences, max_chars: int = MAX_SECTION_CHARS,) -> str:
    cleaned = sanitize_for_summarizer(text)
    cleaned = cap_chars(cleaned, max_chars)

    if not cleaned:
        return ""

    word_count = len(cleaned.split())
    if word_count < 25:
        return ""

    # match section_is_usable logic instead of being harsher
    if looks_like_broken_text(cleaned) and word_count < 150:
        return ""

    nlp = get_nlp()
    doc = nlp(cleaned)
    tr = doc._.textrank

    focus = " ".join(unique_terms(focus_terms))
    if focus:
        tr.change_focus(focus=focus, bias=BIASED_BIAS, default_bias=BIASED_DEFAULT_BIAS)

    summary_sentences = list(
        tr.summary(
            limit_phrases=limit_phrases,
            limit_sentences=limit_sentences,
            preserve_order=True,
        )
    )

    summary_text = " ".join(str(sent).strip() for sent in summary_sentences if str(sent).strip())
    summary_text = sanitize_for_summarizer(summary_text)

    # lighter threshold
    if len(summary_text.split()) < 12:
        return ""

    return truncate_words(summary_text, word_budget)

def build_global_from_section_digests(title: str, abstract: str, section_digests: Dict[str, str]) -> str:
    chunks: List[str] = []
    if title:
        chunks.append(f"Title: {title}")
    if abstract:
        chunks.append(f"Abstract: {abstract}")

    for section_name in ["introduction", "methods", "results", "conclusion"]:
        digest = section_digests.get(section_name, "")
        if digest:
            chunks.append(f"{section_name.title()}: {digest}")

    return sanitize_for_summarizer("\n\n".join(chunk for chunk in chunks if chunk.strip()))

def build_global_from_raw_text(title: str, abstract: str, sections: Dict[str, str], cleaned_text: str) -> str:
    chunks: List[str] = []
    if title:
        chunks.append(f"Title: {title}")
    if abstract:
        chunks.append(f"Abstract: {abstract}")

    for section_name in ["introduction", "methods", "results", "conclusion"]:
        body = sections.get(section_name, "")
        if body:
            chunks.append(f"{section_name.title()}: {body}")

    if len(chunks) <= 2 and cleaned_text:
        chunks.append(cleaned_text)

    return sanitize_for_summarizer("\n\n".join(chunk for chunk in chunks if chunk.strip()))

''' Combined version of section + raw digests
def build_global_source_text(title: str, abstract: str, section_digests: Dict[str, str], cleaned_text: str) -> str:
    chunks = []
    if title:
        chunks.append(f"Title: {title}")
    if abstract:
        chunks.append(f"Abstract: {abstract}")

    for section_name in ["introduction", "methods", "results", "conclusion"]:
        digest = section_digests.get(section_name, "")
        if digest:
            chunks.append(f"{section_name.title()}: {digest}")

    if not chunks and cleaned_text:
        chunks.append(cap_chars(cleaned_text, MAX_GLOBAL_CHARS))
    return sanitize_for_summarizer("\n\n".join(chunks))
'''

# compute a rough score for how complete and usable the section route looks
def compute_route_scores(section_digests: Dict[str, str], abstract: str, full_text: str, sections: Optional[Dict[str, str]] = None,) -> Dict[str, float]:
    usable_sections = 0
    if sections:
        for name in ["introduction", "methods", "results", "conclusion"]:
            if section_is_usable(name, sections.get(name, "")) and section_digests.get(name, "").strip():
                usable_sections += 1
    else:
        usable_sections = sum(
            1 for name in ["introduction", "methods", "results", "conclusion"]
            if section_digests.get(name)
        )
    coverage = usable_sections / 4.0
    digest_words = sum(len(text.split()) for text in section_digests.values() if text)
    full_text_words = max(len(full_text.split()), 1)
    compression = min(digest_words / full_text_words, 1.0)
    has_abstract = 1.0 if abstract.strip() else 0.0

    confidence = (
        0.60 * coverage
        + 0.15 * has_abstract
        + 0.25 * min(compression / 0.18, 1.0)
    )
    return {
        "section_route_confidence": round(confidence, 4),
        "coverage": round(coverage, 4),
        "compression_ratio": round(compression, 4),
        "has_abstract": round(has_abstract, 4),
    }

def count_nonempty_fields(values: Sequence[str]) -> int:
    return sum(1 for v in values if clean_cell(v))


def count_long_glued_tokens(text: str) -> int:
    if not text:
        return 0
    return sum(1 for tok in text.split() if LONG_NOSPACE_RE.search(tok))


def combined_model_text_for_flags(
    section_digests: Dict[str, str],
    global_digest: str,
) -> str:
    parts = [
        section_digests.get("introduction", ""),
        section_digests.get("methods", ""),
        section_digests.get("results", ""),
        section_digests.get("conclusion", ""),
        global_digest,
    ]
    return " ".join(p for p in parts if p).strip()


def compute_training_quality_flags(
    sections: Dict[str, str],
    section_titles: Dict[str, str],
    section_digests: Dict[str, str],
    global_digest: str,
    cleaned_text: str,
    route_hint: str,
) -> Dict[str, object]:
    title_count = count_nonempty_fields([
        section_titles.get("introduction", ""),
        section_titles.get("methods", ""),
        section_titles.get("results", ""),
        section_titles.get("conclusion", ""),
    ])

    digest_count = count_nonempty_fields([
        section_digests.get("introduction", ""),
        section_digests.get("methods", ""),
        section_digests.get("results", ""),
        section_digests.get("conclusion", ""),
    ])

    combined_text = combined_model_text_for_flags(section_digests, global_digest)
    long_glued_token_count = count_long_glued_tokens(combined_text)

    cleaned_word_count = len(cleaned_text.split())
    global_digest_word_count = len(global_digest.split())

    # simple bucket logic
    if (
        cleaned_word_count >= 400
        and title_count >= 3
        and digest_count >= 3
        and global_digest_word_count >= 40
        and long_glued_token_count <= 8
    ):
        quality_bucket = "good"
        keep_for_training = True
        exclude_reason = ""
    elif (
        cleaned_word_count >= 250
        and title_count >= 2
        and digest_count >= 2
        and global_digest_word_count >= 25
        and long_glued_token_count <= 20
    ):
        quality_bucket = "borderline"
        keep_for_training = True
        exclude_reason = ""
    else:
        quality_bucket = "bad"
        keep_for_training = False

        reasons = []
        if cleaned_word_count < 250:
            reasons.append("low_cleaned_word_count")
        if title_count < 2:
            reasons.append("too_few_titles")
        if digest_count < 2:
            reasons.append("too_few_digests")
        if global_digest_word_count < 25:
            reasons.append("weak_global_digest")
        if long_glued_token_count > 20:
            reasons.append("many_glued_tokens")

        exclude_reason = "|".join(reasons)

    return {
        "titles_count": title_count,
        "digests_count": digest_count,
        "long_glued_token_count": long_glued_token_count,
        "global_digest_word_count": global_digest_word_count,
        "training_quality_bucket": quality_bucket,
        "keep_for_training": keep_for_training,
        "exclude_reason": exclude_reason,
        "is_good_training_row": quality_bucket == "good",
        "is_borderline_training_row": quality_bucket == "borderline",
        "is_bad_training_row": quality_bucket == "bad",
        "used_section_route_for_training": route_hint == "section_plus_global",
    }

# main paper processing
def build_record(row: pd.Series, pdf_path: Optional[str]) -> Dict[str, object]:
    entry_id = clean_cell(row.get("entry_id", ""))
    title = clean_cell(row.get("title", ""))
    abstract = normalize_text(clean_cell(row.get("abstract", "")))
    published = clean_cell(row.get("published", ""))
    updated = clean_cell(row.get("updated", ""))
    authors = clean_cell(row.get("authors", ""))
    all_categories = clean_cell(row.get("all_categories", ""))
    main_category = clean_cell(row.get("main_category", ""))
    pdf_url = clean_cell(row.get("pdf_url", ""))
    target_year = safe_int(row.get("target_year", 0), default=0)

    cleaned_text = ""
    sections: Dict[str, str] = {}
    section_titles: Dict[str, str] = {}
    extraction_ok = False

    if pdf_path:
        try:
            pages = extract_pages(pdf_path)
            cleaned_text = clean_full_text(pages)
            sections, section_titles = sectionize_text(cleaned_text)
            sections, section_titles = recover_missing_sections(cleaned_text, sections, section_titles)
            extraction_ok = bool(cleaned_text)
        except Exception:
            cleaned_text = ""
            sections = {}
            section_titles = {}
            extraction_ok = False

    if abstract and "abstract" not in sections:
        sections["abstract"] = abstract
        section_titles.setdefault("abstract", "Abstract")

    section_digests = {
        "introduction": "",
        "methods": "",
        "results": "",
        "conclusion": "",
    }

    if extraction_ok:
        for section_name in ["introduction", "methods", "results", "conclusion"]:
            body = sections.get(section_name, "")

            if not section_is_usable(section_name, body):
                continue

            focus_terms = build_focus_terms(section_name, title, abstract)
            digest = summarize_with_biased_textrank(
                text=body,
                focus_terms=focus_terms,
                word_budget=SECTION_WORD_BUDGET,
                limit_phrases=SECTION_LIMIT_PHRASES,
                limit_sentences=SECTION_LIMIT_SENTENCES,
                max_chars=MAX_SECTION_CHARS,
            )

            if digest and len(digest.split()) >= 12:
                section_digests[section_name] = digest

    route_scores = compute_route_scores(
        section_digests,
        abstract,
        cleaned_text,
        sections=sections,
    ) if extraction_ok else {
        "section_route_confidence": 0.0,
        "coverage": 0.0,
        "compression_ratio": 0.0,
        "has_abstract": 1.0 if abstract else 0.0,
    }

    if extraction_ok:
        usable_section_count = sum(
            1 for name in ["introduction", "methods", "results", "conclusion"]
            if section_digests.get(name)
        )

        if usable_section_count >= 3:
            global_source_text = build_global_from_section_digests(title, abstract, section_digests)
        else:
            global_source_text = build_global_from_raw_text(title, abstract, sections, cleaned_text)

        global_digest = summarize_with_biased_textrank(
            text=global_source_text,
            focus_terms=build_focus_terms("global", title, abstract),
            word_budget=GLOBAL_WORD_BUDGET,
            limit_phrases=GLOBAL_LIMIT_PHRASES,
            limit_sentences=GLOBAL_LIMIT_SENTENCES,
            max_chars=MAX_GLOBAL_CHARS,
        )
    else:
        global_digest = ""

    if len(global_digest.split()) < 25:
        global_digest = light_sanitize(f"Title: {title}. Abstract: {abstract}")

    route_hint = (
        "section_plus_global"
        if route_scores["section_route_confidence"] >= 0.60
        and sum(1 for v in section_digests.values() if v) >= 3
        else "global_only"
    )

    precompression_text = cleaned_text

    if route_hint == "section_plus_global":
        model_input_text = "\n\n".join(
            x for x in [
                section_digests["introduction"],
                section_digests["methods"],
                section_digests["results"],
                section_digests["conclusion"],
                global_digest,
            ] if x.strip()
        )
    else:
        model_input_text = global_digest
       
    quality_flags = compute_training_quality_flags(
        sections=sections,
        section_titles=section_titles,
        section_digests=section_digests,
        global_digest=global_digest,
        cleaned_text=cleaned_text,
        route_hint=route_hint,
    )

    precompression_metrics = safe_readability_metrics(precompression_text)
    model_input_metrics = safe_readability_metrics(model_input_text)

    return {
        "entry_id": entry_id,
        "title": title,
        "abstract": abstract,
        "published": published,
        "updated": updated,
        "authors": authors,
        "all_categories": all_categories,
        "main_category": main_category,
        "pdf_url": pdf_url,
        "target_year": target_year,
        "pdf_path": pdf_path or "",
        "extraction_ok": extraction_ok,
        "digest_engine": "biased_textrank",
        "cleaned_text_word_count": len(cleaned_text.split()),
        "abstract_section_title": section_titles.get("abstract", ""),
        "introduction_section_title": section_titles.get("introduction", ""),
        "methods_section_title": section_titles.get("methods", ""),
        "results_section_title": section_titles.get("results", ""),
        "conclusion_section_title": section_titles.get("conclusion", ""),
        "abstract_word_count": len(sections.get("abstract", "").split()),
        "introduction_word_count": len(sections.get("introduction", "").split()),
        "methods_word_count": len(sections.get("methods", "").split()),
        "results_word_count": len(sections.get("results", "").split()),
        "conclusion_word_count": len(sections.get("conclusion", "").split()),
        "introduction_digest": section_digests["introduction"],
        "methods_digest": section_digests["methods"],
        "results_digest": section_digests["results"],
        "conclusion_digest": section_digests["conclusion"],
        "global_digest": global_digest,
        "route_hint": route_hint,
        "section_route_confidence": route_scores["section_route_confidence"],
        "coverage": route_scores["coverage"],
        "compression_ratio": route_scores["compression_ratio"],
        "has_abstract": route_scores["has_abstract"],

        "titles_count": quality_flags["titles_count"],
        "digests_count": quality_flags["digests_count"],
        "long_glued_token_count": quality_flags["long_glued_token_count"],
        "global_digest_word_count": quality_flags["global_digest_word_count"],
        "training_quality_bucket": quality_flags["training_quality_bucket"],
        "keep_for_training": quality_flags["keep_for_training"],
        "exclude_reason": quality_flags["exclude_reason"],
        "is_good_training_row": quality_flags["is_good_training_row"],
        "is_borderline_training_row": quality_flags["is_borderline_training_row"],
        "is_bad_training_row": quality_flags["is_bad_training_row"],
        "used_section_route_for_training": quality_flags["used_section_route_for_training"],

        "precompression_word_count": precompression_metrics["word_count"],
        "precompression_unique_word_count": precompression_metrics["unique_word_count"],
        "precompression_avg_zipf_frequency": precompression_metrics["avg_zipf_frequency"],
        "precompression_low_zipf_ratio": precompression_metrics["low_zipf_ratio"],
        "precompression_very_low_zipf_ratio": precompression_metrics["very_low_zipf_ratio"],
        "precompression_flesch_reading_ease": precompression_metrics["flesch_reading_ease"],
        "precompression_flesch_kincaid_grade": precompression_metrics["flesch_kincaid_grade"],
        "precompression_gunning_fog": precompression_metrics["gunning_fog"],
        "precompression_smog_index": precompression_metrics["smog_index"],

        "model_input_word_count": model_input_metrics["word_count"],
        "model_input_unique_word_count": model_input_metrics["unique_word_count"],
        "model_input_avg_zipf_frequency": model_input_metrics["avg_zipf_frequency"],
        "model_input_low_zipf_ratio": model_input_metrics["low_zipf_ratio"],
        "model_input_very_low_zipf_ratio": model_input_metrics["very_low_zipf_ratio"],
        "model_input_flesch_reading_ease": model_input_metrics["flesch_reading_ease"],
        "model_input_flesch_kincaid_grade": model_input_metrics["flesch_kincaid_grade"],
        "model_input_gunning_fog": model_input_metrics["gunning_fog"],
        "model_input_smog_index": model_input_metrics["smog_index"],
    }

def record_key_from_metadata_row(row: pd.Series) -> str:
    # entry_id is the cleanest key when it exists
    entry_id = clean_cell(row.get("entry_id", ""))
    if entry_id:
        return f"entry_id::{entry_id}"

    pdf_url = clean_cell(row.get("pdf_url", ""))
    if pdf_url:
        return f"pdf_url::{pdf_url}"

    title = clean_cell(row.get("title", ""))
    published = clean_cell(row.get("published", ""))
    return f"fallback::{title}::{published}"


def record_key_from_output_row(row: pd.Series) -> str:
    entry_id = clean_cell(row.get("entry_id", ""))
    if entry_id:
        return f"entry_id::{entry_id}"

    pdf_url = clean_cell(row.get("pdf_url", ""))
    if pdf_url:
        return f"pdf_url::{pdf_url}"

    title = clean_cell(row.get("title", ""))
    published = clean_cell(row.get("published", ""))
    return f"fallback::{title}::{published}"


def output_row_is_complete(row: pd.Series) -> bool:
    global_digest = clean_cell(row.get("global_digest", ""))
    route_hint = clean_cell(row.get("route_hint", ""))
    extraction_ok = clean_cell(row.get("extraction_ok", "")).lower() in {"true", "1"}

    intro_digest = clean_cell(row.get("introduction_digest", ""))
    methods_digest = clean_cell(row.get("methods_digest", ""))
    results_digest = clean_cell(row.get("results_digest", ""))
    conclusion_digest = clean_cell(row.get("conclusion_digest", ""))

    digest_count = sum(bool(x) for x in [intro_digest, methods_digest, results_digest, conclusion_digest])

    cleaned_word_count = safe_int(row.get("cleaned_text_word_count", 0), default=0)

    return bool(
        global_digest
        and route_hint
        and extraction_ok
        and cleaned_word_count >= 400
        and digest_count >= 2
    )

def load_existing_output(output_path: Path) -> Tuple[Dict[str, Dict[str, object]], set[str]]:
    existing_records_by_key: Dict[str, Dict[str, object]] = {}
    completed_keys: set[str] = set()

    if not output_path.exists():
        return existing_records_by_key, completed_keys

    try:
        existing_df = pd.read_csv(output_path)
    except Exception as e:
        print(f"warning: could not read existing output csv, starting fresh: {e}")
        return existing_records_by_key, completed_keys

    for _, row in existing_df.iterrows():
        key = record_key_from_output_row(row)
        row_dict = row.to_dict()
        existing_records_by_key[key] = row_dict

        if output_row_is_complete(row):
            completed_keys.add(key)

    print(f"found existing output csv: {output_path}")
    print(f"existing rows loaded: {len(existing_records_by_key)}")
    print(f"completed rows that will be skipped: {len(completed_keys)}")

    return existing_records_by_key, completed_keys

# Runner
def main() -> None: # tie the csv metadata and local pdf folder together
    metadata_path = Path(METADATA_CSV)
    metadata_root = Path(METADATA_ROOT)
    full_path = os.path.join(metadata_root, metadata_path)
    pdf_root = Path(PDF_ROOT)
    output_path = Path(OUTPUT_CSV)

    '''if not metadata_path.exists():
        raise FileNotFoundError(
            f"Could not find metadata CSV: {metadata_path}\n"
            "Make sure this file is in the same folder as main.py, or edit METADATA_CSV at the top of the script."
        )
        df = pd.read_csv(full_path)'''

    try:
        df = pd.read_csv(full_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Could not find metadata CSV: {metadata_path}\n"
            "Make sure this file is in the same folder as main.py, or edit METADATA_CSV at the top of the script."
        ) from exc
    except Exception as e:
        print(f"Error {e} trying to read metadata csv: {metadata_path}")
        return
    
    pdf_index = build_pdf_index(pdf_root)

    existing_records_by_key, completed_keys = load_existing_output(output_path)

    records: List[Dict[str, object]] = []
    total_rows = len(df)
    skipped_count = 0
    processed_count = 0

    for i, (_, row) in enumerate(df.iterrows(), start=1): # each paper row gets matched to a local pdf by arxiv id before processing
        key = record_key_from_metadata_row(row)

        # if this row already exists and looks complete, reuse it and skip recomputing
        if key in completed_keys and key in existing_records_by_key:
            records.append(existing_records_by_key[key])
            skipped_count += 1

            if i % 25 == 0 or i == total_rows:
                print(f"[{i}/{total_rows}] skipped completed row")
            continue

        arxiv_id = arxiv_id_from_pdf_url(clean_cell(row.get("pdf_url", "")))
        pdf_path = pdf_index.get(arxiv_id) if arxiv_id else None

        print(f"[{i}/{total_rows}] starting...")
        t0 = time.time()

        try:
            record = build_record(row, pdf_path)
            records.append(record)
            processed_count += 1
            print(f"[{i}/{total_rows}] done in {time.time() - t0:.1f}s | extraction successful = {record['extraction_ok']}")
        except Exception as e:
            print(f"[{i}/{total_rows}] failed in {time.time() - t0:.1f}s | {type(e).__name__}: {e}")

            # if there was an old incomplete row, keep it instead of losing it
            if key in existing_records_by_key:
                records.append(existing_records_by_key[key])
            else:
                # last resort: store a minimal placeholder row
                records.append({
                    "entry_id": clean_cell(row.get("entry_id", "")),
                    "title": clean_cell(row.get("title", "")),
                    "abstract": clean_cell(row.get("abstract", "")),
                    "published": clean_cell(row.get("published", "")),
                    "updated": clean_cell(row.get("updated", "")),
                    "authors": clean_cell(row.get("authors", "")),
                    "all_categories": clean_cell(row.get("all_categories", "")),
                    "main_category": clean_cell(row.get("main_category", "")),
                    "pdf_url": clean_cell(row.get("pdf_url", "")),
                    "target_year": safe_int(row.get("target_year", 0), default=0),
                    "pdf_path": pdf_path or "",
                    "extraction_ok": False,
                    "digest_engine": "biased_textrank",
                    "global_digest": "",
                    "route_hint": "",
                })
        # checkpoint every 50 processed metadata rows
        if i % 50 == 0 or i == total_rows:
            checkpoint_df = pd.DataFrame(records)
            checkpoint_df.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(
                f"[{i}/{total_rows}] checkpoint saved | "
                f"processed this run: {processed_count} | skipped: {skipped_count}"
            )
    output_df = pd.DataFrame(records)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved Phase 2 CSV to: {output_path.resolve()}")
    print(f"Rows written: {len(output_df)}")
    print(f"Processed this run: {processed_count}")
    print(f"Skipped from existing output: {skipped_count}")

if __name__ == "__main__":
    main()