# Text Classification

Everything to know about assigning labels to text: what the task covers, how encoders and LLMs split the field in mid-2026, how to measure it honestly, and runnable code that puts four models on the same test set.

Author

Benedict Thekkel

## 1. What is Text Classification?

Text classification maps a piece of text to one or more labels from a **fixed, known** label set. It is the oldest and by far the most deployed NLP task: sentiment, spam, intent, topic, toxicity, language ID, and “is this ticket urgent” are all the same task wearing different label sets.

**Input.** A string - a sentence, a review, an email, a support ticket. Length matters: BERT-family encoders truncate at 512 tokens, so a long document either gets truncated (cheap, loses the tail), chunked and pooled (slower, keeps everything), or sent to a long-context model.

**Output.** Depends on the variant:

| Variant | Output | Head | Typical loss |
|----|----|----|----|
| Binary | one of 2 labels | linear, 2 logits | cross-entropy |
| Multi-class | one of N labels | linear, N logits | cross-entropy over softmax |
| Multi-label | any subset of N | linear, N logits | per-label binary cross-entropy over sigmoid |
| Ordinal / regression | a score (1-5 stars) | 1 logit or N ordered | MSE, or ordinal loss |

The distinction that trips people up is **multi-class vs multi-label**: softmax forces the probabilities to sum to 1 (exactly one label is right), sigmoid does not (a comment can be both `toxic` and `threat`). Using softmax for a multi-label problem quietly caps recall.

**Neighbouring tasks:**

| Task | How it differs | Notebook |
|----|----|----|
| Zero-shot classification | Labels not seen in training, supplied at inference | `04_Zero_Shot_Classification` |
| Token classification | One label per token, not per text | `01_Token_Classification` |
| Sentence similarity | Compares two texts instead of labelling one | `10_Sentence_Similarity` |
| Text ranking | Orders many texts against a query | `11_Text_Ranking` |
| Text generation | Open-ended output rather than a fixed label set | `08_Text_Generation` |

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Spam and phishing filtering | Email, messaging (Gmail, Outlook) | Message body + headers -\> spam / not | Throughput (billions/day) and adversarial drift; false positives are expensive |
| Support ticket routing | Customer service (Zendesk, Intercom) | Ticket text -\> queue, priority, product area | Latency inside the ticket-create request; label set changes monthly |
| Content moderation | Social platforms | Post or comment -\> multi-label policy violations | Recall on rare harmful classes; multilingual; appeal-ability |
| Brand and market sentiment | Marketing, finance | Social posts, reviews, filings -\> sentiment | Domain shift (finance “liability” is neutral); volume; cost per million docs |
| Clinical coding and triage | Healthcare | Clinical note -\> ICD codes, acuity | Accuracy under class imbalance; on-prem/privacy; auditability |
| Compliance and PII gating | Legal, banking | Document -\> restricted / not | Near-zero false negatives; explainability to a regulator |
| Intent detection in assistants | Consumer devices, IVR | Utterance -\> intent | Sub-50 ms budget shared with ASR; small on-device model |

What the leaderboard number hides:

- **Label quality is usually the bottleneck, not the model.** On most in-house datasets two annotators disagree on 5-15% of examples, which puts a hard ceiling on measurable accuracy well below 100%. Cleaning labels beats swapping architectures more often than anyone likes to admit.
- **Class imbalance is the norm.** Fraud, harmful content and defects sit at 0.1-2% positive. Accuracy is meaningless there (predict “no” always, score 99%); use precision/recall at a chosen operating point, and pick the threshold from the cost of each error type, not from 0.5.
- **Drift is continuous.** Spam, slang and product names change weekly. Production systems need a re-labelling loop and a monitored holdout, not a one-off training run.
- **The tail is where the pain is.** Aggregate F1 hides that the rarest and most consequential class often scores worst. Always report per-class numbers.

------------------------------------------------------------------------

## 3. How Modern Text Classification Works

1.  **Bag-of-words + linear model (pre-2013).** TF-IDF into logistic regression or a linear SVM. Still an excellent baseline: milliseconds to train, trivially explainable, and on a clean 2-class problem with plenty of data it lands within a few points of BERT. Run it first; it tells you whether the problem is hard.
2.  **Static embeddings + shallow nets (2013-2018).** word2vec/GloVe/fastText averaged into a classifier. Fixed one vector per word type, so `bank` had one meaning. fastText survives as a fast production baseline.
3.  **Pretrained encoders fine-tuned (2018-2023).** BERT, RoBERTa, DeBERTa-v3: pretrain bidirectionally on unlabelled text, then fine-tune a classification head on a few thousand labelled examples. This is still the accuracy-per-dollar winner for a fixed label set with training data, and DeBERTa-v3 remained the strongest classical encoder for years.
4.  **Modernised encoders (2024-2026).** ModernBERT (Dec 2024) rebuilt the BERT recipe with rotary embeddings, alternating local/global attention, GeGLU, no padding waste, 8192-token context and 2T tokens of training - roughly DeBERTa-v3 quality at several times the speed, with a context window long enough for whole documents. EuroBERT, NeoBERT and mmBERT extended the same idea to more languages. Encoders did not die; they got a refresh.
5.  **LLMs as classifiers (2023-present).** Prompt an instruction-tuned model with the label definitions. Zero training data, immediate support for a new label, and genuinely better on subtle or reasoning-heavy labels (“is this sarcastic?”). The costs are real: 10-1000x the compute per document, output that needs constraining to the label set, and calibration that is worse than a fine-tuned head. The standard compromise in 2026 is **LLM-as-labeller**: have a large model label a few thousand examples, then distil into a ModernBERT-size encoder that serves production.

**Where it stands (mid-2026).** For a stable label set with \>1k labels per class, a fine-tuned modern encoder wins on cost, latency and calibration. For a shifting label set, few examples, or labels requiring reasoning, an LLM wins. Most serious systems run both: encoder in the hot path, LLM on the uncertain tail and as the labelling engine.

------------------------------------------------------------------------

## 4. Evaluation Metrics

**Accuracy** - fraction correct. Only meaningful when classes are roughly balanced.

**Precision, recall, F1** per class:

$$P = \frac{TP}{TP + FP} \qquad R = \frac{TP}{TP + FN} \qquad F_1 = \frac{2PR}{P + R}$$

**Averaging matters more than the metric.** `macro` averages the per-class F1 unweighted, so the rare class counts as much as the common one - this is the honest default for imbalanced data. `micro` pools all predictions first, which for single-label multi-class is just accuracy. `weighted` averages by class support, which flatters a model that only nails the majority class.

**Pitfalls:**

- **Thresholds are a choice, not a constant.** For binary and multi-label problems the 0.5 cut is arbitrary; sweep it and pick from the precision/recall cost trade-off. Report **PR-AUC** rather than ROC-AUC when positives are rare - ROC-AUC looks great on a 1% positive class no matter what.
- **Calibration is separate from accuracy.** A model can rank perfectly and still say 0.99 when it is right 70% of the time. Check with a reliability curve or expected calibration error; fix with temperature scaling on a validation split.
- **Always look at the confusion matrix.** The aggregate never tells you which pair of labels the model is mixing up, and that pair is usually the one you can fix with data.

The cell below implements the metrics directly - 30 lines, no dependency, and the arithmetic is the point.

------------------------------------------------------------------------

``` python
# ---- shared display helpers (used by every results cell below) ------------------
# rich renders to text/html inside Jupyter, so these tables survive into the published
# Quarto docs and degrade to plain text in a terminal. Charts stay with pyecharts.
from rich import box
from rich.console import Console
from rich.table import Table

console = Console(width=112)


def _fmt(v):
    "Thousands separators for ints, sensible precision for floats, str for the rest."
    if v is None or isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.4f}" if abs(v) < 10 else f"{v:,.2f}"
    return str(v)


def show_table(rows, title=None, best=(), lower_is_better=(), caption=None):
    """Render a list of dicts as a rich table.

    `best` names columns whose winning value is highlighted; `lower_is_better` is the
    subset of those where the minimum wins (latency, loss, perplexity).
    """
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    numeric = {c: any(isinstance(r.get(c), (int, float)) and not isinstance(r.get(c), bool)
                      for r in rows) for c in cols}
    winners = {}
    for c in best:
        vals = [r[c] for r in rows
                if isinstance(r.get(c), (int, float)) and not isinstance(r.get(c), bool)]
        if vals:
            winners[c] = min(vals) if c in lower_is_better else max(vals)
    table = Table(title=title, caption=caption, box=box.SIMPLE_HEAVY, pad_edge=False,
                  min_width=min(72, console.width), header_style="bold cyan",
                  title_style="bold", caption_style="dim italic")
    for i, c in enumerate(cols):
        table.add_column(c, justify="right" if numeric[c] else "left",
                         style="bold" if i == 0 else "", overflow="fold")
    for r in rows:
        cells = []
        for c in cols:
            text = _fmt(r.get(c, ""))
            if c in winners and r.get(c) == winners[c]:
                text = f"[bold green]{text}[/]"
            cells.append(text)
        table.add_row(*cells)
    console.print(table)


def show_kv(mapping, title=None):
    "Two-column key/value table - one run's summary numbers."
    table = Table(box=box.SIMPLE, show_header=False, title=title, title_style="bold",
                  pad_edge=False, min_width=min(64, console.width))
    table.add_column(style="cyan")
    table.add_column(justify="right")
    for k, v in mapping.items():
        table.add_row(str(k), _fmt(v))
    console.print(table)


def rule(text):
    "A labelled horizontal rule, for separating one model's output from the next."
    console.rule(f"[bold]{text}", style="dim", align="left")


from collections import Counter


def accuracy(y_true, y_pred):
    "Fraction of predictions that match the reference."
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def per_class_prf(y_true, y_pred, labels=None):
    "Precision, recall and F1 for every label. Returns {label: (p, r, f1, support)}."
    labels = labels or sorted(set(y_true) | set(y_pred))
    out = {}
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[lab] = (prec, rec, f1, tp + fn)
    return out


def macro_f1(y_true, y_pred, labels=None):
    "Unweighted mean of the per-class F1 - the honest default under class imbalance."
    scores = per_class_prf(y_true, y_pred, labels)
    return sum(v[2] for v in scores.values()) / len(scores)


def confusion(y_true, y_pred, labels=None, title="Confusion matrix"):
    "Confusion matrix as a table: rows are the reference label, columns the prediction."
    labels = labels or sorted(set(y_true) | set(y_pred))
    counts = Counter(zip(y_true, y_pred))
    rows = [{"gold \\ predicted": t, **{p: counts[(t, p)] for p in labels}} for t in labels]
    show_table(rows, title=title,
               caption="the diagonal is correct - an off-diagonal cell names the label "
                       "pair you can usually fix with data")


# Toy example: 10 documents, a rare "urgent" class the model mostly misses.
y_true = ["normal"] * 7 + ["urgent"] * 3
y_pred = ["normal"] * 8 + ["urgent"] * 2

show_kv({"accuracy": round(accuracy(y_true, y_pred), 3),
         "macro F1": round(macro_f1(y_true, y_pred), 3)},
        title="Aggregate - accuracy looks fine, macro F1 is the honest number")
show_table(
    [{"label": lab, "precision": round(p, 3), "recall": round(r, 3),
      "F1": round(f1, 3), "support": n}
     for lab, (p, r, f1, n) in per_class_prf(y_true, y_pred).items()],
    title="Per class", best=("F1",),
    caption="the rare class scores worst, and the aggregate hides it")
confusion(y_true, y_pred)
```

    accuracy  0.900   <- looks fine
    macro F1  0.867   <- the honest number
      normal   P 0.88  R 1.00  F1 0.93  (n=7)
      urgent   P 1.00  R 0.67  F1 0.80  (n=3)
              normal  urgent   <- predicted
      normal       7       0
      urgent       1       2

## 5. Datasets

| Dataset | Contents | Size | Scope | License | Typical use |
|----|----|----|----|----|----|
| [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) | Movie-review sentences, binary sentiment | 67k train / 872 val | en | CC0 | The classic sanity check; used below |
| [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) | Full movie reviews, binary sentiment | 25k / 25k | en | permissive | Long-document sentiment |
| [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) | News headlines + lead, 4 topics | 120k / 7.6k | en | custom, research | Topic classification baseline |
| [GLUE](https://huggingface.co/datasets/nyu-mll/glue) | 9 sentence-level tasks (CoLA, MRPC, QNLI, …) | varies | en | mixed | The standard encoder benchmark suite |
| [Jigsaw Toxic Comments](https://huggingface.co/datasets/google/jigsaw_toxicity_pred) | Wikipedia comments, 6 toxicity labels | 160k | en | CC0, manual download | Canonical **multi-label** + imbalance |
| [Amazon Reviews Multi](https://huggingface.co/datasets/mteb/amazon_reviews_multi) | Product reviews, 1-5 stars | 1.2M | 6 langs | research | Ordinal, multilingual |
| [MTEB classification tasks](https://huggingface.co/spaces/mteb/leaderboard) | ~12 classification sets, embedding + linear probe | varies | multi | mixed | Comparing embeddings, not fine-tunes |
| [TweetEval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) | 7 Twitter tasks: sentiment, emotion, hate, irony | 100k+ | en | research | Social-domain evaluation |

This notebook evaluates on the **SST-2 validation split** (872 sentences, a few hundred KB) because every off-the-shelf sentiment model can be scored on it without fine-tuning. It is also easy: modern models sit at 94-97%, so treat it as a smoke test. Real evaluation belongs on your own labelled data - public sentiment sets do not predict how a model behaves on your tickets.

Downloads land in `DL_tasks/datasets/` via `cache_dir` (gitignored).

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

There is no single leaderboard for “text classification” because the task is really a thousand task instances. The useful references are [GLUE](https://gluebenchmark.com/leaderboard) and [SuperGLUE](https://super.gluebenchmark.com/) for encoder quality, the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) classification tab for frozen-embedding probes, and the [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) for the generative side.

| Model | Params | License | Context | Type | Best for |
|----|----|----|----|----|----|
| [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) | 149M | Apache 2.0 | 8192 | encoder, needs fine-tuning | the 2026 default backbone to fine-tune |
| [ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) | 395M | Apache 2.0 | 8192 | encoder | max encoder accuracy |
| [DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) | 184M | MIT | 512 | encoder | still very strong; slower than ModernBERT |
| [DistilBERT SST-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) | 67M | Apache 2.0 | 512 | fine-tuned | the cheap sentiment default; used below |
| [twitter-roberta-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) | 125M | MIT | 512 | fine-tuned | social text, 3-class |
| [modernBERT multilingual sentiment](https://huggingface.co/clapAI/modernBERT-base-multilingual-sentiment) | 150M | Apache 2.0 | 8192 | fine-tuned | multilingual sentiment off the shelf |
| [XLM-RoBERTa-base](https://huggingface.co/FacebookAI/xlm-roberta-base) | 278M | MIT | 512 | encoder | 100 languages, fine-tune it |
| [Qwen3-0.6B / 1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 0.6-1.7B | Apache 2.0 | 32k+ | generative | zero-shot labels, no training data |
| [bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) | 407M | MIT | 1024 | NLI zero-shot | arbitrary labels without training (see nb 04) |

**How to choose.** Fixed labels and \>1k examples per class: fine-tune ModernBERT-base, it is the best cost/accuracy point on this hardware. Under ~100 examples per class: an LLM or an NLI zero-shot model, then use it to bootstrap labels. Latency under 10 ms at high volume: distilled 6-layer encoder or fastText. More than 512 tokens per document: ModernBERT (8192) rather than chunking BERT.

------------------------------------------------------------------------

## 7. Setup

Every model loads through Hugging Face `transformers` - no vendor packages. Package roles:

- `transformers` (\>=5.13) + `torch` - all four models below
- `accelerate` - `device_map` placement
- `datasets` - the SST-2 validation split
- `pyecharts` + `pandas` - the benchmark chart and table
- `rich` - the result tables. It renders to HTML inside Jupyter, so the tables survive into the published docs; `show_table` / `show_kv` / `rule` are defined in the first code cell of section 4.

Metrics are implemented inline (section 4) rather than pulling in `scikit-learn` or `evaluate`; for production evaluation `sklearn.metrics.classification_report` is the sane choice and gives the same numbers.

Two `transformers` notes used below:

- `pipeline("text-classification", ..., top_k=None)` returns **all** label scores rather than only the top one, which is what you need for thresholding, multi-label output and calibration.
- The model’s own `config.id2label` is the source of truth for label names. Never assume index 0 is the negative class - `cardiffnlp` orders `negative, neutral, positive` while `distilbert-sst2` is `NEGATIVE, POSITIVE`, and silently mismatching them is the single most common evaluation bug in this task.

------------------------------------------------------------------------

``` python
# Everything runs through Hugging Face transformers - no vendor packages.
# %pip install -q torch transformers accelerate datasets pandas pyecharts rich
```

``` python
import ctypes
import ctypes.util
import gc
import time
from pathlib import Path

import torch
from dotenv import find_dotenv, load_dotenv

# Knowledge/.env sets HF_TOKEN - authenticated HF Hub requests get higher rate limits
load_dotenv(find_dotenv(usecwd=True))

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device != "cpu" else torch.float32
if device != "cpu":
    print(torch.cuda.get_device_name(0))
print("device:", device, "| dtype:", dtype)


def vram(tag=""):
    "Report current GPU memory (allocated / reserved). No-op on CPU."
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"VRAM {tag:22s} {alloc:5.2f} GB allocated / {reserved:5.2f} GB reserved")


def free_memory():
    "Collect garbage and hand freed VRAM back to the CUDA allocator.\n\n    Call right after `del`-ing a model you are done with: `del model; free_memory()`.\n    `del` drops the Python reference; this reclaims the RAM and releases the VRAM.\n    "
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    # glibc keeps freed CPU allocations in its arenas instead of returning them to the
    # OS, so RSS compounds across sections. malloc_trim(0) hands the arenas back. See
    # dl-visualization-and-memory.md - not optional on a 20 GB box.
    try:
        ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6").malloc_trim(0)
    except Exception:
        pass


# All downloads go to DL_tasks/datasets/ (gitignored)
DATA_DIR = Path("../../datasets")
DATA_DIR.mkdir(exist_ok=True)
HF_CACHE = str(DATA_DIR / "hf_cache")
```

    NVIDIA GeForce RTX 3060
    device: cuda:0 | dtype: torch.float16

``` python
from datasets import load_dataset

# SST-2 validation: 872 movie-review sentences with binary sentiment labels.
sst2 = load_dataset("stanfordnlp/sst2", split="validation", cache_dir=HF_CACHE)

N = 200  # sentences to evaluate. 872 is the whole split and takes a few minutes.
texts = [r["sentence"].strip() for r in sst2.select(range(N))]
gold = ["positive" if r["label"] == 1 else "negative" for r in sst2.select(range(N))]

print(sst2)
print(f"\nevaluating on {N} sentences, {gold.count('positive')} positive / {gold.count('negative')} negative\n")
for t, g in list(zip(texts, gold))[:4]:
    print(f"  [{g:8s}] {t[:90]}")
```

    Dataset({
        features: ['idx', 'sentence', 'label'],
        num_rows: 872
    })

    evaluating on 200 sentences, 99 positive / 101 negative

      [positive] it 's a charming and often affecting journey .
      [negative] unflinchingly bleak and desperate
      [positive] allows us to hope that nolan is poised to embark a major career as a commercial yet invent
      [positive] the acting , costumes , music , cinematography and sound are all astounding given the prod

## 8. Fine-tuned encoder: DistilBERT on SST-2

The workhorse. DistilBERT is BERT-base with half the layers, distilled from it - 67M params, ~97% of BERT’s GLUE score at twice the speed. This checkpoint has a 2-way head fine-tuned on SST-2 itself, so it is playing at home: expect ~91% here.

The `text-classification` pipeline handles tokenisation, truncation, batching and softmax. `top_k=None` returns every label’s score rather than only the winner, which is what thresholding and calibration need.

------------------------------------------------------------------------

``` python
from transformers import pipeline

clf = pipeline(
    "text-classification",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
    model_kwargs={"cache_dir": HF_CACHE},
)

rule("distilbert-base-uncased-finetuned-sst-2-english")
print("labels:", clf.model.config.id2label)

demo = [
    "A gorgeous, witty film that earns every one of its two hours.",
    "I have never wanted a movie to end so badly.",
    "It was a movie. Things happened. People were in it.",
]
show_table(
    [{"prediction": max(sc, key=lambda s: s["score"])["label"],
      **{s["label"]: round(s["score"], 3) for s in sc}, "text": t}
     for t, sc in zip(demo, clf(demo, top_k=None))],
    title="All label scores, not just the winner (top_k=None)",
    caption="the third sentence is genuinely neutral - watch a 2-class head split it")

t0 = time.perf_counter()
preds = [p["label"].lower() for p in clf(texts, batch_size=32)]
elapsed = time.perf_counter() - t0
show_kv({"sentences": N, "seconds": round(elapsed, 1),
         "docs / second": round(N / elapsed, 1),
         "accuracy": round(accuracy(gold, preds), 3),
         "macro F1": round(macro_f1(gold, preds), 3)},
        title="SST-2 validation - fine-tuned on this exact dataset")
confusion(gold, preds, labels=["negative", "positive"])

del clf
free_memory()
vram("after distilbert")
```

    labels: {0: 'NEGATIVE', 1: 'POSITIVE'}
    POSITIVE  1.000  A gorgeous, witty film that earns every one of its two hours.
               all: {'POSITIVE': 1.0, 'NEGATIVE': 0.0}
    NEGATIVE  0.999  I have never wanted a movie to end so badly.
               all: {'NEGATIVE': 0.999, 'POSITIVE': 0.001}
    NEGATIVE  0.862  It was a movie. Things happened. People were in it.
               all: {'NEGATIVE': 0.862, 'POSITIVE': 0.138}

    200 sentences in 0.2s
    accuracy 0.910  macro F1 0.910
                negative  positive   <- predicted
      negative        87        14
      positive         4        95
    VRAM after distilbert        0.01 GB allocated /  0.02 GB reserved

## 9. Domain-matched encoder: twitter-roberta

Same architecture family, different training data, and it matters more than the architecture does. `twitter-roberta-base-sentiment-latest` is RoBERTa-base pretrained on ~124M tweets and fine-tuned on TweetEval sentiment, with three labels: `negative`, `neutral`, `positive`.

Two lessons in one model. First, **domain shift cuts both ways** - it is markedly better than DistilBERT on social text and slightly worse on movie reviews, which is what this notebook measures. Second, **a mismatched label set is an evaluation problem you have to solve explicitly**: SST-2 has no `neutral`, so scoring the model fairly means comparing only its `negative` and `positive` scores and dropping `neutral` from the argmax. Mapping `neutral` onto one of the two, or letting it win, would report a worse model than it is.

------------------------------------------------------------------------

``` python
sentiment3 = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=device,
    model_kwargs={"cache_dir": HF_CACHE},
)
print("labels:", sentiment3.model.config.id2label)


def argmax_over(scores, keep=("negative", "positive")):
    "Pick the best label among `keep`, ignoring the rest (here: SST-2 has no neutral)."
    kept = [s for s in scores if s["label"].lower() in keep]
    return max(kept, key=lambda s: s["score"])["label"].lower()


t0 = time.perf_counter()
raw = sentiment3(texts, top_k=None, batch_size=32)
elapsed = time.perf_counter() - t0

preds_3way = [max(s, key=lambda x: x["score"])["label"].lower() for s in raw]
preds = [argmax_over(s) for s in raw]

show_kv({"sentences": N, "seconds": round(elapsed, 1),
         "docs / second": round(N / elapsed, 1),
         "neutral wins (dropped from the argmax)": preds_3way.count("neutral"),
         "accuracy": round(accuracy(gold, preds), 3),
         "macro F1": round(macro_f1(gold, preds), 3)},
        title="twitter-roberta-base-sentiment-latest - trained on tweets, tested on reviews")
confusion(gold, preds, labels=["negative", "positive"])

del sentiment3
free_memory()
vram("after twitter-roberta")
```

    [transformers] RobertaForSequenceClassification LOAD REPORT from: cardiffnlp/twitter-roberta-base-sentiment-latest
    Key                         | Status     |  | 
    ----------------------------+------------+--+-
    roberta.pooler.dense.bias   | UNEXPECTED |  | 
    roberta.pooler.dense.weight | UNEXPECTED |  | 

    Notes:
    - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

    labels: {0: 'negative', 1: 'neutral', 2: 'positive'}
    200 sentences in 0.3s
    neutral wins on 42 / 200 sentences - dropped from the argmax
    accuracy 0.865  macro F1 0.865
                negative  positive   <- predicted
      negative        89        12
      positive        15        84
    VRAM after twitter-roberta   0.01 GB allocated /  0.02 GB reserved

## 10. Modern encoder: ModernBERT

ModernBERT (Dec 2024) is the BERT recipe rebuilt: rotary position embeddings, alternating local/global attention, GeGLU activations, unpadded batches, flash attention, 8192-token context, trained on 2T tokens including code. The practical effect on this hardware is that it is several times faster than DeBERTa-v3 at similar or better accuracy, and it reads a whole document instead of the first 512 tokens.

The checkpoint used here is a multilingual sentiment fine-tune of ModernBERT-base (same 3-class scheme as section 9, so the same `neutral`-dropping rule applies). If you want ModernBERT for your own labels, start from `answerdotai/ModernBERT-base` and fine-tune - the base model has no classification head and predicts nothing useful until you do.

**Note.** ModernBERT needs `transformers>=4.48`; this repo pins \>=5.13, so it is available. On Ampere and newer, `attn_implementation="flash_attention_2"` is a further speedup if `flash-attn` is installed - it is not a dependency here, and the model runs fine without it.

------------------------------------------------------------------------

``` python
modern = pipeline(
    "text-classification",
    model="clapAI/modernBERT-base-multilingual-sentiment",
    device=device,
    torch_dtype=dtype,
    model_kwargs={"cache_dir": HF_CACHE},
)
print("labels:", modern.model.config.id2label)
print("max context:", modern.model.config.max_position_embeddings, "tokens")

# Multilingual, off the shelf - the same head handles languages SST-2 never contains.
for text, scores in zip(
    ["Der Film war absolut fantastisch.", "Une perte de temps totale.", "This ruled."],
    modern(["Der Film war absolut fantastisch.", "Une perte de temps totale.", "This ruled."], top_k=None),
):
    print(f"  {max(scores, key=lambda s: s['score'])['label']:9s} {text}")

t0 = time.perf_counter()
raw = modern(texts, top_k=None, batch_size=32)
elapsed = time.perf_counter() - t0
preds = [argmax_over(s) for s in raw]

show_kv({"sentences": N, "seconds": round(elapsed, 1),
         "docs / second": round(N / elapsed, 1),
         "accuracy": round(accuracy(gold, preds), 3),
         "macro F1": round(macro_f1(gold, preds), 3)},
        title="modernBERT-base-multilingual-sentiment")

del modern
free_memory()
vram("after modernbert")
```

    [transformers] `torch_dtype` is deprecated! Use `dtype` instead!

    labels: {0: 'negative', 1: 'neutral', 2: 'positive'}
    max context: 8192 tokens
      negative  Der Film war absolut fantastisch.
      negative  Une perte de temps totale.
      positive  This ruled.

    200 sentences in 0.3s
    accuracy 0.905  macro F1 0.905
    VRAM after modernbert        0.01 GB allocated /  0.02 GB reserved

## 11. LLM as a classifier: Qwen3-0.6B

No fine-tuning, no training data, and the label set is a Python list you can edit. The catch is that a generative model will happily answer “Positive!” or “The sentiment is positive” or a paragraph of reasoning, none of which is a label.

The fix is to **not let it generate freely**. One forward pass, look at the logits for the next token only, and compare the scores of the candidate label tokens against each other. That gives a decision restricted to the label set by construction, plus a usable probability, at the cost of exactly one prefill - far cheaper than generating. This is the standard trick for classification with an LLM and it is worth knowing in the four lines it takes.

Qwen3 is a hybrid reasoning model; `enable_thinking=False` in the chat template keeps it from emitting a `<think>` block, which matters here because we only read the first token.

------------------------------------------------------------------------

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer

llm_id = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(llm_id, cache_dir=HF_CACHE)
llm = AutoModelForCausalLM.from_pretrained(
    llm_id, dtype=dtype, device_map=device, cache_dir=HF_CACHE
).eval()
vram("qwen3-0.6b loaded")

LABELS = ["negative", "positive"]
# The first token of each label word, which is what the next-token logits score.
LABEL_IDS = [tok.encode(lab, add_special_tokens=False)[0] for lab in LABELS]
print("label token ids:", dict(zip(LABELS, LABEL_IDS)))

PROMPT = (
    "Classify the sentiment of the movie-review sentence as one word, "
    "either negative or positive.\n\nSentence: {text}\nSentiment:"
)


@torch.inference_mode()
def llm_classify(batch):
    "Score the label words against each other from a single forward pass (no generation)."
    chats = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(text=t)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for t in batch
    ]
    enc = tok(chats, return_tensors="pt", padding=True, padding_side="left").to(llm.device)
    logits = llm(**enc).logits[:, -1, :]          # next-token distribution only
    label_logits = logits[:, LABEL_IDS].float()   # restricted to the label set
    probs = torch.softmax(label_logits, dim=-1)
    idx = probs.argmax(-1)
    return [LABELS[i] for i in idx.tolist()], probs.max(-1).values.tolist()


labels, confidences = llm_classify(demo)
show_table([{"label": lab, "confidence": round(conf, 3), "text": text}
            for text, lab, conf in zip(demo, labels, confidences)],
           title="Qwen3-0.6B restricted to the label set by first-token scoring",
           best=("confidence",))

t0 = time.perf_counter()
preds = []
for i in range(0, N, 16):
    preds.extend(llm_classify(texts[i:i + 16])[0])
elapsed = time.perf_counter() - t0
show_kv({"sentences": N, "seconds": round(elapsed, 1),
         "docs / second": round(N / elapsed, 1),
         "accuracy": round(accuracy(gold, preds), 3),
         "macro F1": round(macro_f1(gold, preds), 3)},
        title="Qwen3-0.6B - zero training data, one forward pass per sentence")
confusion(gold, preds, labels=["negative", "positive"])

del llm, tok
free_memory()
vram("after qwen3")
```

    VRAM qwen3-0.6b loaded       1.20 GB allocated /  1.68 GB reserved
    label token ids: {'negative': 42224, 'positive': 30487}
    positive  0.818  A gorgeous, witty film that earns every one of its two hours.
    negative  0.500  I have never wanted a movie to end so badly.
    positive  0.551  It was a movie. Things happened. People were in it.

    200 sentences in 1.4s
    accuracy 0.885  macro F1 0.885
                negative  positive   <- predicted
      negative        94         7
      positive        16        83
    VRAM after qwen3             0.01 GB allocated /  0.02 GB reserved

## 12. Head-to-head Benchmark

The same 200 SST-2 sentences, the same metric functions, the same normalisation of the label space (`neutral` dropped where the model has one), one model live at a time.

Read the result as **accuracy per unit of compute**, not as a ranking. DistilBERT is fine-tuned on this exact dataset, so it should win; the interesting comparison is how close a model that has never seen SST-2 gets, and what it costs. A tiny 200-sentence sample also carries roughly +/-3 points of sampling noise at these accuracy levels, so differences smaller than that are not real.

------------------------------------------------------------------------

``` python
import pandas as pd

MODELS = [
    ("distilbert-sst2", "distilbert/distilbert-base-uncased-finetuned-sst-2-english", 67),
    ("twitter-roberta", "cardiffnlp/twitter-roberta-base-sentiment-latest", 125),
    ("modernbert-sentiment", "clapAI/modernBERT-base-multilingual-sentiment", 150),
]

results = []
for name, model_id, params_m in MODELS:
    pipe = pipeline(
        "text-classification", model=model_id, device=device,
        model_kwargs={"cache_dir": HF_CACHE},
    )
    t0 = time.perf_counter()
    raw = pipe(texts, top_k=None, batch_size=32)
    elapsed = time.perf_counter() - t0
    preds = [argmax_over(s) for s in raw]
    results.append({
        "model": name,
        "params_m": params_m,
        "accuracy": round(accuracy(gold, preds), 4),
        "macro_f1": round(macro_f1(gold, preds), 4),
        "seconds": round(elapsed, 2),
        "docs_per_sec": round(N / elapsed, 1),
    })
    show_kv(results[-1], title=name)
    del pipe  # free each model before loading the next so VRAM stays flat
    free_memory()

vram("after benchmark")
df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
show_table(
    df.to_dict("records"),
    title=f"SST-2 validation, {N} sentences",
    best=("accuracy", "macro_f1", "docs_per_sec"),
    caption="best per column in green - `df` is still a DataFrame if you want to export it",
)
```

    {'model': 'distilbert-sst2', 'params_m': 67, 'accuracy': 0.91, 'macro_f1': 0.9099, 'seconds': 0.16, 'docs_per_sec': 1280.7}

    [transformers] RobertaForSequenceClassification LOAD REPORT from: cardiffnlp/twitter-roberta-base-sentiment-latest
    Key                         | Status     |  | 
    ----------------------------+------------+--+-
    roberta.pooler.dense.bias   | UNEXPECTED |  | 
    roberta.pooler.dense.weight | UNEXPECTED |  | 

    Notes:
    - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

    {'model': 'twitter-roberta', 'params_m': 125, 'accuracy': 0.865, 'macro_f1': 0.8649, 'seconds': 0.31, 'docs_per_sec': 655.6}

    {'model': 'modernbert-sentiment', 'params_m': 150, 'accuracy': 0.905, 'macro_f1': 0.905, 'seconds': 0.24, 'docs_per_sec': 841.9}
    VRAM after benchmark         0.01 GB allocated /  0.02 GB reserved

|     | model                | params_m | accuracy | macro_f1 | seconds | docs_per_sec |
|-----|----------------------|----------|----------|----------|---------|--------------|
| 0   | distilbert-sst2      | 67       | 0.910    | 0.9099   | 0.16    | 1280.7       |
| 2   | modernbert-sentiment | 150      | 0.905    | 0.9050   | 0.24    | 841.9        |
| 1   | twitter-roberta      | 125      | 0.865    | 0.8649   | 0.31    | 655.6        |

``` python
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org serves an expired TLS certificate from most nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"

# The LLM result from section 11 belongs in the same table, but reloading it here would
# hold a second model live, so re-run section 11 and paste its numbers in if you want it
# charted. The encoders are the comparison that matters for a production hot path.
from pyecharts import options as opts
from pyecharts.charts import Bar

names = [r["model"] for r in results]
bar = (
    Bar()
    .add_xaxis(names)
    .add_yaxis("accuracy x100", [round(r["accuracy"] * 100, 1) for r in results])
    .add_yaxis("macro F1 x100", [round(r["macro_f1"] * 100, 1) for r in results])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="SST-2 validation (200 sentences)",
            subtitle="RTX 3060, fp32 pipelines, batch 32 - a smoke test, not a leaderboard",
        ),
        yaxis_opts=opts.AxisOpts(name="score", min_=0, max_=100),
        xaxis_opts=opts.AxisOpts(name="model", axislabel_opts=opts.LabelOpts(rotate=15)),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
    )
)
bar.render_notebook()
```

``` python
from pyecharts.charts import Scatter

# Accuracy against throughput - the trade-off that actually decides a deployment.
scatter = Scatter()
scatter.add_xaxis([r["docs_per_sec"] for r in results])
for r in results:
    scatter.add_yaxis(
        r["model"], [[r["docs_per_sec"], round(r["accuracy"] * 100, 1)]],
        symbol_size=18, label_opts=opts.LabelOpts(is_show=False),
    )
scatter.set_global_opts(
    title_opts=opts.TitleOpts(title="Accuracy vs throughput"),
    xaxis_opts=opts.AxisOpts(name="docs / second", type_="value"),
    yaxis_opts=opts.AxisOpts(name="accuracy x100", type_="value", min_=80, max_=100),
    tooltip_opts=opts.TooltipOpts(trigger="item"),
)
scatter.render_notebook()
```

## 13. Interactive: classify your own text

Type your own sentences and label set into the lists below and see all three heads disagree. This is the cell people run on its own, so it opens with a `require(...)` guard naming what it needs from Setup rather than dying on a bare `NameError`.

The most useful thing to do here is deliberately adversarial: negation (“not bad at all”), sarcasm (“great, another meeting”), mixed sentiment (“the acting was superb, the plot was garbage”), and domain words that flip meaning (“this drug has serious side effects” is negative in a review and neutral in a label). Fine-tuned sentiment heads fail predictably on the first two; the LLM path in section 11 handles them better and costs more.

------------------------------------------------------------------------

``` python
def require(*names):
    "Fail early and clearly if the notebook's setup / helper cells have not been run."
    missing = [n for n in names if n not in globals()]
    if missing:
        raise NameError(
            f"this demo needs {', '.join(missing)} from earlier in the notebook. "
            "Run the setup and helper cells first (Run > Run All Above Selected Cell)."
        )


require("device", "dtype", "HF_CACHE", "free_memory", "vram")

from transformers import pipeline

MY_TEXTS = [
    "Not bad at all, actually.",
    "Great, another two-hour meeting that could have been an email.",
    "The acting was superb; the plot was garbage.",
    "It shipped on time and does exactly what the docs say.",
]

# Re-runnable: this cell frees the pipeline at the end, so guard the load or a second
# shift-enter raises NameError.
if "my_clf" not in globals():
    my_clf = pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device,
        model_kwargs={"cache_dir": HF_CACHE},
    )

show_table(
    [{"top label": max(sc, key=lambda s: s["score"])["label"],
      **{s["label"]: round(s["score"], 3) for s in sc}, "text": t}
     for t, sc in zip(MY_TEXTS, my_clf(MY_TEXTS, top_k=None))],
    title="Your own sentences, all label scores",
    caption="negation and sarcasm are where a fine-tuned sentiment head fails "
            "predictably - the LLM path in section 11 handles them better")

del my_clf
free_memory()
vram("final")
```

    [transformers] RobertaForSequenceClassification LOAD REPORT from: cardiffnlp/twitter-roberta-base-sentiment-latest
    Key                         | Status     |  | 
    ----------------------------+------------+--+-
    roberta.pooler.dense.bias   | UNEXPECTED |  | 
    roberta.pooler.dense.weight | UNEXPECTED |  | 

    Notes:
    - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

    Not bad at all, actually.
       ->  positive 0.90  neutral 0.09  negative 0.01

    Great, another two-hour meeting that could have been an email.
       ->  positive 0.38  neutral 0.33  negative 0.28

    The acting was superb; the plot was garbage.
       ->  negative 0.75  neutral 0.17  positive 0.07

    It shipped on time and does exactly what the docs say.
       ->  positive 0.78  neutral 0.21  negative 0.01

    VRAM final                   0.01 GB allocated /  0.02 GB reserved

## 14. Common Frameworks

Text classification is a solved modelling problem and an unsolved data problem, and the framework list reflects that. Nothing below will make a 150M encoder classify better; the annotation, label- cleaning and evaluation tools will, because on almost every real dataset the ceiling is set by label noise rather than by architecture. The other half of the table is deployment, where a small encoder is cheap enough to run in places an LLM never could.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [transformers](https://github.com/huggingface/transformers) | modelling | ModernBERT, DeBERTa, DistilBERT behind `AutoModelForSequenceClassification`, plus `Trainer` and the multi-label loss switch | Apache 2.0 | Default. A fine-tune here is about 30 lines and minutes of GPU |
| [SetFit](https://github.com/huggingface/setfit) + [peft](https://github.com/huggingface/peft) | modelling | Contrastive few-shot fine-tuning that works on 8-64 examples per class, and LoRA when the base model is an LLM | Apache 2.0 | You have tens of labels, not thousands. SetFit beats prompting an LLM at a fraction of the inference cost |
| [spaCy](https://github.com/explosion/spaCy) | modelling | A production text-classification pipeline with tokenisation, batching and serialisation as one artefact | MIT | The classifier ships inside a wider NLP pipeline and you want one versioned object rather than five |
| [Argilla](https://github.com/argilla-io/argilla) / [Label Studio](https://github.com/HumanSignal/label-studio) | data | Annotation with model-in-the-loop suggestions, disagreement tracking, and a review queue | Apache 2.0 | Always. The distillation recipe - LLM labels, human corrects, encoder trains - runs through here |
| [cleanlab](https://github.com/cleanlab/cleanlab) | data | Label-error detection from the model’s own out-of-fold predictions, ranked by likelihood of being wrong | AGPL-3.0 (commercial license available) | Before blaming the model. Finding 200 mislabelled rows usually beats any architecture change |
| [optimum](https://github.com/huggingface/optimum) + [ONNX Runtime](https://onnxruntime.ai/) | inference runtime | Export and int8 quantisation - a 150M encoder then classifies thousands of documents a second on CPU | Apache 2.0 / MIT | Deployment. This is the economic argument for an encoder over an LLM, so realise it |
| [vLLM](https://github.com/vllm-project/vllm) + [outlines](https://github.com/dottxt-ai/outlines) | inference runtime | Batched LLM labelling with the output constrained to your label set - the first half of the distillation recipe | Apache 2.0 | Bootstrapping a training set. Constrain the decode or you will spend a day parsing “Sentiment: positive!” |
| [BentoML](https://github.com/bentoml/BentoML) / [Ray Serve](https://github.com/ray-project/ray) | serving | A batched endpoint with versioning, and the threshold configuration kept outside the model artefact | Apache 2.0 | Production. Per-label thresholds change more often than weights and should deploy independently |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) + [evaluate](https://github.com/huggingface/evaluate) | evaluation | Per-class precision/recall, macro-F1, the PR curve you pick the threshold from, and calibration | BSD-3 / Apache 2.0 | Always. Accuracy on an imbalanced problem is the most misleading number in this task |

**The 2026 default stack** is: label a few thousand examples with an LLM through vLLM with constrained output, correct them in Argilla, run cleanlab over the result, fine-tune ModernBERT, export to ONNX. That pipeline converts a per-document LLM cost into a per-document encoder cost, and it is how most production classifiers now get built.

The common wrong turn is deploying an LLM as a classifier. It is the right tool for *labelling* and the wrong one for serving: hundreds of times the cost, worse latency, and usually lower accuracy than a small encoder fine-tuned on its own output. The second is tuning a single global threshold on a multi-label problem, where each label’s scores have their own distribution.

------------------------------------------------------------------------

## 15. Going Further

- **Fine-tune your own head.** `AutoModelForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base", num_labels=N)` plus `Trainer` is about 30 lines. Start with `lr=2e-5`, 3-5 epochs, batch 16-32; on this box a 150M encoder over 20k examples trains in minutes. Freeze nothing - full fine-tuning beats head-only at this scale.
- **Multi-label needs `problem_type="multi_label_classification"`**, which switches the loss to per-label BCE and the output to sigmoid. Then tune one threshold **per label** on a validation split, not a single global 0.5.
- **Handle imbalance with the loss and the threshold, not by oversampling first.** Class weights in cross-entropy, or focal loss for extreme skew; then pick the operating point from the precision/recall curve. Resampling distorts calibration.
- **Distil an LLM into an encoder.** Label 5-10k examples with a large instruction model, verify a few hundred by hand, fine-tune ModernBERT on the result. This is how most 2026 production classifiers get built, and it converts a 1.7B-per-document cost into a 150M-per-document one.
- **Long documents.** ModernBERT reads 8192 tokens directly. Below that, the standard trick is chunk-and-pool: classify each 512-token window and take the max or mean of the logits - max for “does this document contain X”, mean for “what is this document about”.
- **Calibrate before you threshold.** Temperature scaling on a held-out split (one scalar, fit in seconds) usually removes most of the overconfidence, and it makes probability-based routing (“send anything under 0.8 to a human”) behave.
- **Interpretability.** Attention weights are not explanations. Use gradient x input, integrated gradients, or a simple leave-one-token-out ablation when you need to justify a decision.
- **Related notebooks.** `04_Zero_Shot_Classification` (labels without training data), `01_Token_Classification` (labels per token), `07_Feature_Extraction` (embeddings + a linear probe, a strong few-shot baseline), `08_Text_Generation` (the LLM side), `11_Text_Ranking` (ordering rather than labelling).

------------------------------------------------------------------------

Back to top
