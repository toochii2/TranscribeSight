"""
TranscribeSight Re-Scorer (CPU + crash-safe resume)
===================================================
Recompute normalised WER and CER, BERTScore and SeMaScore from an Excel file of
transcriptions (one sheet per prompt condition, one column per system).

Crash safety
------------
BERTScore and SeMaScore are the slow steps and the ones that crash. Every chunk
of scores is written to disk (atomically) as it is produced. If the process dies,
just re-upload the same workbook and click Run again: the app reloads the
checkpoint and computes only the rows that are still missing. WER and CER are
cached the same way, so a resume is near-instant up to the point it stopped.

CPU
---
Force CPU is on by default. BERTScore and the SentenceTransformer are both pinned
to the CPU device, which avoids CUDA out-of-memory crashes. roberta-large on CPU
is slow, so two things help: the BERTScore batch size slider lets you trade RAM
for speed (a larger forward batch is faster, and with tens of GB of RAM you can
push it to 256 to 1024 since the references are short), and the checkpoint lets
you run in stages across several sessions.

Metric definitions
-------------------
* WER and CER  -> jiwer Levenshtein alignment, (S + D + I) / N, capped at 100.
* BERTScore    -> bert_score with rescale_with_baseline=True, F1 * 100 (matches
                  TranscribeSight; the baseline rescaling is the fix for the high
                  floor in the un-rescaled run).
* SeMaScore    -> SentenceTransformer('all-MiniLM-L6-v2') cosine * 100 (matches
                  TranscribeSight).

Run with:
    pip install streamlit pandas numpy jiwer bert-score sentence-transformers openpyxl
    streamlit run transcribesight_rescore.py
"""

import os
import sys
import asyncio

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

import re
import string
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st

# Lazily loaded heavy objects (loaded only when there is work to do).
_st_model = None
_bert_scorer = None
_bert_key = None


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

SEVERITY = {
    "F04": "Mild", "M03": "Mild",
    "F03": "Moderate", "M05": "Moderate",
    "F01": "Severe", "M01": "Severe", "M02": "Severe", "M04": "Severe",
}
SEVERITY_ORDER = ["Mild", "Moderate", "Severe"]

DEFAULT_SHEET_CONDITION = {
    "V0": "No prompt", "V1": "Full prompt",
    "V2": "No-clause prompt", "V3": "Minimal prompt",
}

NON_SYSTEM_COLS = {"Speaker", "File Name", "Ground Truth", "Reference"}
PLACEHOLDER_PREFIXES = ("error:",)
PLACEHOLDER_EXACT = ("no transcription available", "")

CKPT_DIR = os.path.join(os.path.expanduser("~"), ".transcribesight_rescore")
CKPT_FILE = os.path.join(CKPT_DIR, "scores_checkpoint.pkl")

METRIC_COLS = ["WER", "CER", "S_word", "D_word", "I_word", "Reference_words",
               "Hypothesis_words", "Length_ratio", "BERTScore", "SeMaScore"]
KEY_COLS = ["ref_norm", "hyp_norm"]

CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "shan't": "shall not",
    "ain't": "is not", "y'all": "you all", "let's": "let us",
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "he's": "he is", "she's": "she is", "it's": "it is",
    "that's": "that is", "what's": "what is", "who's": "who is",
    "there's": "there is", "here's": "here is", "where's": "where is",
    "how's": "how is", "she'd": "she would", "he'd": "he would",
    "we're": "we are", "they're": "they are", "you're": "you are",
    "we've": "we have", "they've": "they have", "you've": "you have",
    "we'll": "we will", "they'll": "they will", "you'll": "you will",
    "we'd": "we would", "they'd": "they would", "you'd": "you would",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "don't": "do not", "doesn't": "does not",
    "didn't": "did not", "haven't": "have not", "hasn't": "has not",
    "hadn't": "had not", "wouldn't": "would not", "couldn't": "could not",
    "shouldn't": "should not", "mustn't": "must not", "mightn't": "might not",
    "needn't": "need not", "o'clock": "oclock",
}

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]
_ORDINAL = {"one": "first", "two": "second", "three": "third", "five": "fifth",
            "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}


# --------------------------------------------------------------------------- #
# Number to words
# --------------------------------------------------------------------------- #

def _int_to_words(n):
    if n < 0:
        return "minus " + _int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return (_TENS[n // 10] + (" " + _ONES[n % 10] if n % 10 else "")).strip()
    if n < 1000:
        rest = n % 100
        return (_ONES[n // 100] + " hundred"
                + (" " + _int_to_words(rest) if rest else "")).strip()
    rest = n % 1000
    return (_int_to_words(n // 1000) + " thousand"
            + (" " + _int_to_words(rest) if rest else "")).strip()


def _ordinal_to_words(n):
    words = _int_to_words(n)
    last = words.split()[-1]
    if last in _ORDINAL:
        return " ".join(words.split()[:-1] + [_ORDINAL[last]]).strip()
    if last.endswith("y"):
        return " ".join(words.split()[:-1] + [last[:-1] + "ieth"]).strip()
    return words + "th"


def _numbers_to_words(text):
    text = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b",
                  lambda m: _ordinal_to_words(int(m.group(1))), text)
    text = re.sub(r"\b\d+\b",
                  lambda m: _int_to_words(int(m.group(0))), text)
    return text


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #

def _expand_contractions(text):
    for k, v in CONTRACTIONS.items():
        text = re.sub(r"\b" + re.escape(k) + r"\b", v, text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"'re\b", " are", text)
    text = re.sub(r"'ve\b", " have", text)
    text = re.sub(r"'ll\b", " will", text)
    text = re.sub(r"'d\b", " would", text)
    text = re.sub(r"'m\b", " am", text)
    return text


def basic_normalise(text, lower=True, contractions=True, numbers=True,
                    punctuation=True, collapse=True):
    if not isinstance(text, str):
        return ""
    if lower:
        text = text.lower()
    if contractions:
        text = _expand_contractions(text)
    if numbers:
        text = _numbers_to_words(text)
    if punctuation:
        text = re.sub(r"[-/]", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
    if collapse:
        text = re.sub(r"\s+", " ", text).strip()
    return text


def get_whisper_normaliser():
    for mod, attr in (("whisper.normalizers", "EnglishTextNormalizer"),
                      ("whisper_normalizer.english", "EnglishTextNormalizer")):
        try:
            m = __import__(mod, fromlist=[attr])
            return getattr(m, attr)()
        except Exception:
            continue
    return None


def normalise(text, scheme, opts, whisper_norm=None):
    if scheme == "Whisper English normaliser" and whisper_norm is not None:
        try:
            return whisper_norm(text if isinstance(text, str) else "")
        except Exception:
            pass
    if scheme == "None (raw)":
        return text if isinstance(text, str) else ""
    return basic_normalise(text, **opts)


def is_placeholder(text):
    if not isinstance(text, str):
        return True
    t = text.strip().lower()
    return t in PLACEHOLDER_EXACT or any(t.startswith(p) for p in PLACEHOLDER_PREFIXES)


# --------------------------------------------------------------------------- #
# Model loaders (CPU aware, loaded once)
# --------------------------------------------------------------------------- #

def get_bert_scorer(device="cpu", batch_size=128):
    """Load roberta-large BERTScorer (once per device and batch size), with
    baseline rescaling. A larger batch size processes more pairs per forward
    pass, which uses more RAM and is faster on CPU."""
    global _bert_scorer, _bert_key
    key = (device, int(batch_size))
    if _bert_scorer is None or _bert_key != key:
        from bert_score import BERTScorer
        import torch
        if device == "cpu":
            try:
                torch.set_num_threads(max(1, os.cpu_count() or 1))
            except Exception:
                pass
        _bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True,
                                  device=device, batch_size=int(batch_size))
        _bert_key = key
    return _bert_scorer


def get_st_model(device="cpu"):
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return _st_model


# --------------------------------------------------------------------------- #
# Per-row WER / CER
# --------------------------------------------------------------------------- #

def wer_cer_counts(ref_norm, hyp_norm, cap=True):
    import jiwer
    out = {k: np.nan for k in METRIC_COLS}
    rw = len(ref_norm.split())
    hw = len(hyp_norm.split())
    out["Reference_words"] = rw
    out["Hypothesis_words"] = hw
    out["Length_ratio"] = (hw / rw) if rw else np.nan
    if rw == 0:
        return out
    w = jiwer.process_words(ref_norm, hyp_norm)
    wer = (w.substitutions + w.deletions + w.insertions) / rw * 100.0
    try:
        cer = jiwer.cer(ref_norm, hyp_norm) * 100.0
    except Exception:
        cer = np.nan
    if cap:
        wer = min(wer, 100.0)
        if not np.isnan(cer):
            cer = min(cer, 100.0)
    out.update({"WER": wer, "CER": cer, "S_word": w.substitutions,
                "D_word": w.deletions, "I_word": w.insertions})
    return out


# --------------------------------------------------------------------------- #
# Checkpoint
# --------------------------------------------------------------------------- #

def ckpt_save(scores):
    try:
        os.makedirs(CKPT_DIR, exist_ok=True)
        tmp = CKPT_FILE + ".tmp"
        scores.to_pickle(tmp)
        os.replace(tmp, CKPT_FILE)
    except Exception as e:
        print(f"Warning: could not save checkpoint: {e}")


def ckpt_load():
    if not os.path.exists(CKPT_FILE):
        return None
    try:
        return pd.read_pickle(CKPT_FILE)
    except Exception as e:
        print(f"Warning: could not load checkpoint: {e}")
        return None


def ckpt_clear():
    try:
        if os.path.exists(CKPT_FILE):
            os.remove(CKPT_FILE)
    except Exception as e:
        print(f"Warning: could not clear checkpoint: {e}")


def ckpt_merge(scores, cached):
    """Copy cached metric values into `scores` for rows whose row_id and
    normalised text both match (so stale or re-normalised rows recompute)."""
    if cached is None:
        return scores
    m = cached.reindex(scores.index)
    match = ((m["ref_norm"].fillna("\0") == scores["ref_norm"].fillna("\0"))
             & (m["hyp_norm"].fillna("\0") == scores["hyp_norm"].fillna("\0")))
    for col in METRIC_COLS:
        if col in cached.columns:
            scores.loc[match, col] = m.loc[match, col]
    return scores


# --------------------------------------------------------------------------- #
# Resumable scoring helpers
# --------------------------------------------------------------------------- #

def run_wer(scores, excluded, cap, prog, save_every=500):
    todo = scores.index[
        scores["WER"].isna() & (~excluded)
        & scores["ref_norm"].fillna("").str.len().gt(0)
    ].tolist()
    if not todo:
        prog(1.0)
        return
    since = 0
    for n, rid in enumerate(todo, 1):
        r = wer_cer_counts(scores.at[rid, "ref_norm"],
                           scores.at[rid, "hyp_norm"] or "", cap=cap)
        for k, v in r.items():
            scores.at[rid, k] = v
        since += 1
        if since >= save_every:
            ckpt_save(scores)
            since = 0
        if n % 200 == 0 or n == len(todo):
            prog(n / len(todo))
    ckpt_save(scores)


def run_batched(scores, metric, compute_chunk, chunk_size, prog,
                save_every=200):
    """Compute `metric` for rows where it is NaN and both sides are non-empty."""
    todo = scores.index[
        scores[metric].isna()
        & scores["ref_norm"].fillna("").str.len().gt(0)
        & scores["hyp_norm"].fillna("").str.len().gt(0)
    ].tolist()
    if not todo:
        prog(1.0)
        return
    since = 0
    for start in range(0, len(todo), chunk_size):
        ids = todo[start:start + chunk_size]
        refs = [scores.at[i, "ref_norm"] for i in ids]
        hyps = [scores.at[i, "hyp_norm"] for i in ids]
        vals = compute_chunk(refs, hyps)
        for i, v in zip(ids, vals):
            scores.at[i, metric] = v
        since += len(ids)
        if since >= save_every:
            ckpt_save(scores)
            since = 0
        prog(min((start + len(ids)) / len(todo), 1.0))
    ckpt_save(scores)


def bert_chunk_fn(device, batch_size=128):
    scorer = get_bert_scorer(device, batch_size)

    def _fn(refs, hyps):
        # bert_score order is (candidates, references); candidate = hypothesis.
        try:
            _, _, f1 = scorer.score(hyps, refs, batch_size=int(batch_size))
        except TypeError:
            _, _, f1 = scorer.score(hyps, refs)
        return (f1.cpu().numpy() * 100.0).tolist()
    return _fn


def sema_chunk_fn(device, batch_size=256):
    model = get_st_model(device)
    import torch

    def _fn(refs, hyps):
        er = model.encode(refs, convert_to_tensor=True, batch_size=batch_size,
                          show_progress_bar=False)
        eh = model.encode(hyps, convert_to_tensor=True, batch_size=batch_size,
                          show_progress_bar=False)
        cos = torch.nn.functional.cosine_similarity(eh, er, dim=1).cpu().numpy()
        return (cos * 100.0).tolist()
    return _fn


# --------------------------------------------------------------------------- #
# Load, reshape, duplication, aggregate
# --------------------------------------------------------------------------- #

def speaker_from_filename(fn):
    return str(fn).split("-")[0].split(".")[0].strip().upper()


def load_long(xls_bytes, sheet_condition_map):
    frames = []
    for sheet, condition in sheet_condition_map.items():
        try:
            df = pd.read_excel(BytesIO(xls_bytes), sheet_name=sheet)
        except Exception:
            continue
        ref_col = "Ground Truth" if "Ground Truth" in df.columns else "Reference"
        systems = [c for c in df.columns if c not in NON_SYSTEM_COLS]
        for _, row in df.iterrows():
            fn = row.get("File Name")
            spk = speaker_from_filename(fn)
            for sysname in systems:
                frames.append({
                    "Condition": condition, "Sheet": sheet, "Speaker": spk,
                    "Severity": SEVERITY.get(spk, "Unknown"), "File Name": fn,
                    "System": sysname, "Reference": row.get(ref_col, ""),
                    "Hypothesis": row.get(sysname, ""),
                })
    return pd.DataFrame(frames)


def duplication_report(long_df, threshold=0.95):
    rows = []
    conds = sorted(long_df["Condition"].unique())
    systems = sorted(long_df["System"].unique())
    for sev in SEVERITY_ORDER:
        for sysname in systems:
            sub = long_df[(long_df["Severity"] == sev) & (long_df["System"] == sysname)]
            by_cond = {c: sub[sub["Condition"] == c].set_index("File Name")["Hypothesis"]
                       for c in conds}
            for i in range(len(conds)):
                for j in range(i + 1, len(conds)):
                    a, b = by_cond.get(conds[i]), by_cond.get(conds[j])
                    if a is None or b is None or len(a) == 0:
                        continue
                    common = a.index.intersection(b.index)
                    if len(common) == 0:
                        continue
                    frac = (a.loc[common].astype(str).values
                            == b.loc[common].astype(str).values).mean()
                    if frac >= threshold:
                        rows.append({"Severity": sev, "System": sysname,
                                     "Condition A": conds[i], "Condition B": conds[j],
                                     "Identical fraction": round(float(frac), 3),
                                     "n": int(len(common))})
    return pd.DataFrame(rows)


SUMM_METRICS = ["WER", "CER", "BERTScore", "SeMaScore", "Length_ratio"]


def summarise(scored, group_cols):
    agg = {}
    for m in SUMM_METRICS:
        if m in scored.columns:
            agg[f"{m}_mean"] = (m, "mean")
            agg[f"{m}_std"] = (m, "std")
            agg[f"{m}_count"] = (m, "count")
    return scored.groupby(group_cols, dropna=False).agg(**agg).reset_index()


def pivot_metric(summary_by_sev, metric):
    if f"{metric}_mean" not in summary_by_sev.columns:
        return pd.DataFrame()
    p = summary_by_sev.pivot_table(index=["System", "Severity"], columns="Condition",
                                   values=f"{metric}_mean").reset_index()
    p["Severity"] = pd.Categorical(p["Severity"], SEVERITY_ORDER, ordered=True)
    return p.sort_values(["System", "Severity"])


def build_workbook(scored, sev_summary, spk_summary, dup_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if not dup_df.empty:
            dup_df.to_excel(writer, sheet_name="Duplication_Check", index=False)
        sev_summary.to_excel(writer, sheet_name="Summary_BySeverity", index=False)
        spk_summary.to_excel(writer, sheet_name="Summary_BySpeaker", index=False)
        for m, short in [("WER", "WER"), ("CER", "CER"), ("BERTScore", "BERT"),
                         ("SeMaScore", "SeMa")]:
            p = pivot_metric(sev_summary, m)
            if not p.empty:
                p.to_excel(writer, sheet_name=f"Prompt_{short}_Pivot", index=False)
        scored.to_excel(writer, sheet_name="RowScores", index=False)
    return output.getvalue()


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #

def main():
    st.set_page_config(page_title="TranscribeSight Re-Scorer", layout="wide")
    st.title("TranscribeSight Re-Scorer")
    st.caption("Normalised WER/CER, BERTScore and SeMaScore. CPU, crash-safe resume.")

    up = st.file_uploader("Transcription workbook (.xlsx)", type=["xlsx"])

    # Checkpoint status is shown even before upload so a stale one can be cleared.
    cached = ckpt_load()
    with st.sidebar:
        st.header("Checkpoint")
        if cached is not None:
            nb = int(cached["BERTScore"].notna().sum()) if "BERTScore" in cached else 0
            ns = int(cached["SeMaScore"].notna().sum()) if "SeMaScore" in cached else 0
            st.success(f"Found a checkpoint: {len(cached):,} rows "
                       f"({nb:,} BERTScore, {ns:,} SeMaScore already done). "
                       f"A new run resumes automatically.")
        else:
            st.info("No checkpoint yet.")
        st.caption(f"Stored at {CKPT_FILE}")
        if st.button("Clear checkpoint"):
            ckpt_clear()
            st.rerun()

    if up is None:
        st.info("Upload your transcription Excel to begin.")
        return

    xls_bytes = up.getvalue()
    xls = pd.ExcelFile(BytesIO(xls_bytes))

    with st.sidebar:
        st.header("1  Sheet to condition")
        sheet_condition_map = {s: st.text_input(f"Sheet '{s}' is condition",
                                                value=DEFAULT_SHEET_CONDITION.get(s, s),
                                                key=f"cond_{s}") for s in xls.sheet_names}

        st.header("2  Normalisation")
        scheme = st.selectbox("Scheme", ["Basic (documented, no extra deps)",
                                         "Whisper English normaliser", "None (raw)"])
        scheme_key = ("Basic" if scheme.startswith("Basic")
                      else "Whisper English normaliser" if scheme.startswith("Whisper")
                      else "None (raw)")
        opts = dict(lower=True, contractions=True, numbers=True, punctuation=True,
                    collapse=True)
        whisper_norm = None
        if scheme_key == "Basic":
            c1, c2 = st.columns(2)
            opts["lower"] = c1.checkbox("Lower-case", True)
            opts["contractions"] = c2.checkbox("Expand contractions", True)
            opts["numbers"] = c1.checkbox("Numbers to words", True)
            opts["punctuation"] = c2.checkbox("Remove punctuation", True)
        elif scheme_key == "Whisper English normaliser":
            whisper_norm = get_whisper_normaliser()
            if whisper_norm is None:
                st.warning("Whisper normaliser not installed; using Basic. "
                           "Install with: pip install openai-whisper")
                scheme_key = "Basic"

        st.header("3  Metrics and compute")
        do_wer = st.checkbox("WER and CER", True)
        do_bert = st.checkbox("BERTScore", True)
        do_sema = st.checkbox("SeMaScore", True)
        force_cpu = st.checkbox("Force CPU", True,
                                help="Pins BERTScore and SeMaScore to the CPU to "
                                     "avoid CUDA out-of-memory crashes.")
        cap = st.checkbox("Cap per-utterance WER/CER at 100 percent", True)
        bert_bs = st.slider("BERTScore batch size", 16, 1024, 256, step=16,
                            help="Pairs per forward pass. Higher uses more RAM and "
                                 "runs faster on CPU. With ample RAM (you have about "
                                 "94 GB) 256 to 512 works well; the references here "
                                 "are short so even 1024 is safe. Progress is saved "
                                 "after each batch.")
        failed = st.radio("Failed or empty transcriptions",
                          ["Exclude from metrics (NaN)", "Score as 100 percent WER"])

    device = "cpu" if force_cpu else "cuda"

    long_df = load_long(xls_bytes, sheet_condition_map)
    if long_df.empty:
        st.error("No rows parsed. Check the sheet to condition mapping.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(long_df):,}")
    c2.metric("Conditions", long_df["Condition"].nunique())
    c3.metric("Systems", long_df["System"].nunique())
    c4.metric("Files", long_df["File Name"].nunique())

    unknown = long_df[long_df["Severity"] == "Unknown"]["Speaker"].unique()
    if len(unknown):
        st.warning("Speakers with no severity mapping: " + ", ".join(map(str, unknown)))

    dup_df = duplication_report(long_df)
    if not dup_df.empty:
        st.error("Duplication check: near-identical hypotheses between conditions. "
                 "The affected condition is likely a copy and should be re-run.")
        st.dataframe(dup_df, use_container_width=True)
    else:
        st.success("Duplication check passed.")

    if not st.button("Run scoring", type="primary"):
        st.dataframe(long_df.head(20), use_container_width=True)
        return

    # Build the work frame with a stable row_id and normalised text.
    work = long_df.copy()
    work["row_id"] = (work["Condition"].astype(str) + "||"
                      + work["System"].astype(str) + "||"
                      + work["File Name"].astype(str))
    if work["row_id"].duplicated().any():
        st.warning("Duplicate row_ids detected; keeping the first of each.")
        work = work.drop_duplicates("row_id")

    with st.spinner("Normalising..."):
        work["ref_norm"] = work["Reference"].apply(
            lambda t: normalise(t, scheme_key, opts, whisper_norm))
        excluded_mask = work["Hypothesis"].apply(is_placeholder)
        if failed.startswith("Exclude"):
            work["hyp_norm"] = work.apply(
                lambda r: "" if is_placeholder(r["Hypothesis"])
                else normalise(r["Hypothesis"], scheme_key, opts, whisper_norm), axis=1)
            excluded = excluded_mask.values
        else:
            work["hyp_norm"] = work["Hypothesis"].apply(
                lambda t: "" if is_placeholder(t)
                else normalise(t, scheme_key, opts, whisper_norm))
            excluded = np.zeros(len(work), dtype=bool)  # score empties as 100% WER

    scores = work.set_index("row_id")[KEY_COLS].copy()
    for col in METRIC_COLS:
        scores[col] = np.nan
    excluded_series = pd.Series(excluded, index=work["row_id"].values)
    excluded_series = excluded_series.reindex(scores.index).fillna(False)

    # Auto-resume: pull in any matching cached scores.
    scores = ckpt_merge(scores, ckpt_load())
    nb0 = int(scores["BERTScore"].notna().sum())
    ns0 = int(scores["SeMaScore"].notna().sum())
    if nb0 or ns0:
        st.info(f"Resuming from checkpoint: {nb0:,} BERTScore and {ns0:,} SeMaScore "
                f"rows reused.")

    if do_wer:
        p = st.progress(0.0, text="WER and CER")
        run_wer(scores, excluded_series.values, cap,
                lambda f: p.progress(min(f, 1.0), text="WER and CER"))
        p.progress(1.0, text="WER and CER complete")

    if do_bert:
        p = st.progress(0.0, text="BERTScore (loads roberta-large on first run; "
                                   "slow on CPU, progress is saved each chunk)")
        run_batched(scores, "BERTScore", bert_chunk_fn(device, bert_bs), bert_bs,
                    lambda f: p.progress(min(f, 1.0), text="BERTScore"))
        p.progress(1.0, text="BERTScore complete")

    if do_sema:
        p = st.progress(0.0, text="SeMaScore")
        run_batched(scores, "SeMaScore", sema_chunk_fn(device), 256,
                    lambda f: p.progress(min(f, 1.0), text="SeMaScore"))
        p.progress(1.0, text="SeMaScore complete")

    # Merge scores back onto the long frame.
    scored = work.merge(scores.drop(columns=KEY_COLS), left_on="row_id",
                        right_index=True, how="left")
    scored["Reference_norm"] = scored["ref_norm"]
    scored["Hypothesis_norm"] = scored["hyp_norm"]
    scored = scored.drop(columns=["ref_norm", "hyp_norm"])

    st.success("Scoring complete.")
    st.subheader("Per-utterance scores")
    st.dataframe(scored.head(30), use_container_width=True)

    sev_summary = summarise(scored, ["Condition", "Severity", "System"])
    spk_summary = summarise(scored, ["Condition", "Severity", "Speaker", "System"])
    st.subheader("Summary by severity")
    st.dataframe(sev_summary, use_container_width=True)
    wp = pivot_metric(sev_summary, "WER")
    if not wp.empty:
        st.subheader("WER pivot")
        st.dataframe(wp, use_container_width=True)

    st.download_button("Download scored workbook (.xlsx)",
                       data=build_workbook(scored, sev_summary, spk_summary, dup_df),
                       file_name="rescored_results.xlsx",
                       mime=("application/vnd.openxmlformats-officedocument."
                             "spreadsheetml.sheet"))
    st.caption("Checkpoint retained so you can re-run without recomputing. "
               "Use Clear checkpoint in the sidebar to start fresh.")


if __name__ == "__main__":
    main()