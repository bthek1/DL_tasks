# Zero-Shot Classification

Classifying text into labels the model was never trained on: the NLI reformulation that makes it work, why the hypothesis template is a hyperparameter, the cost model nobody mentions, and runnable code that puts three NLI models and an LLM on the same AG News sample.

Author

Benedict Thekkel

## 1. What is Zero-Shot Classification?

Zero-shot classification assigns a label to a text where **the label set is supplied at inference time** and the model has never been trained on it. You hand it `["billing", "shipping", "returns"]` today and `["urgent", "routine"]` tomorrow, with no training data and no fine-tuning in between.

**Input.** A text plus a list of candidate label strings. The labels are natural language, not indices - `"a complaint about a late delivery"` is a valid label and usually a better one than `"shipping"`.

**Output.** A score per candidate label. Two modes, and confusing them is the most common bug in this task:

| Mode | Scores | Use when |
|----|----|----|
| Single-label (`multi_label=False`) | softmax across labels, sums to 1 | exactly one label is correct |
| Multi-label (`multi_label=True`) | independent sigmoid per label | any subset can be correct |

**How it works: the NLI trick.** The insight (Yin et al., 2019) is that natural language inference is a general-purpose classifier in disguise. An NLI model takes a **premise** and a **hypothesis** and predicts `entailment` / `neutral` / `contradiction`. So set:

- premise = the text to classify
- hypothesis = a template filled with the candidate label, e.g. `"This example is sports."`

and read off `P(entailment)`. The label whose hypothesis is most entailed wins. Nothing about the model is task-specific - it just happens that “does this text imply that it is about sports?” *is* the classification question, written out.

**The cost model follows directly and surprises people.** One forward pass per (text, label) pair. Ten labels means ten passes per document, so a 400M NLI model classifying against 10 labels costs about the same as a 4B model doing one pass. This is the real reason zero-shot does not go straight into a high-volume hot path.

**What “zero-shot” does not mean.** These models are heavily trained - on MNLI, on XNLI, on dozens of aggregated classification sets. The zero-shot part is only that they have not seen *your* labels. A model that has seen “sports vs politics” style tasks in some form will do much better than the phrase suggests, and one facing a genuinely alien domain will do worse.

**Neighbouring tasks:**

| Task | How it differs | Notebook |
|----|----|----|
| Text classification | Fixed label set, trained head | `00_Text_Classification` |
| Sentence similarity | Compares two texts, no labels | `10_Sentence_Similarity` |
| Feature extraction | Embeddings + a probe is the other few-shot path | `07_Feature_Extraction` |
| Text generation | LLM prompting is the other zero-shot path | `08_Text_Generation` |
| Zero-shot image classification | CLIP does the same thing for images | `Computer_Vision/11_Zero_Shot_Image_Classification` |

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Cold-start ticket routing | Customer support | Ticket + a queue list that changes monthly -\> queue | No labelled data at launch; label set is not stable enough to fine-tune |
| Content moderation policy rollout | Social platforms | Post + a new policy category -\> flag | New categories ship weekly; recall on rare classes |
| Taxonomy exploration | Research, media monitoring | Document + candidate topics -\> topic distribution | Analysts want to try 30 taxonomies in an afternoon |
| Bootstrapping training data | Any ML team | Unlabelled corpus -\> weak labels for a fine-tune | Throughput over precision; humans verify a sample |
| Intent detection for new products | Assistants, chatbots | Utterance + intent list -\> intent | Zero examples on day one; must work before traffic exists |
| Compliance screening | Legal, finance | Document + regulation categories -\> match | Categories are legally defined text, not code labels |
| Multilingual triage | Global orgs | Text in 20 languages + English labels -\> label | Cross-lingual transfer; no per-language training data |

What the benchmark number hides:

- **Zero-shot is a bridge, not a destination.** The standard trajectory is: ship zero-shot on day one, log the predictions, have humans correct a few thousand, fine-tune a small encoder, and replace it. The zero-shot model is usually 10-20 points behind a fine-tuned one on the same labels, and 100x more expensive per document. Its value is that it exists before your data does.
- **Label wording is a tuning knob with real range.** Swapping `"tech"` for `"science and technology"` can move accuracy by 10+ points. This is not a quirk - the label string is literally part of the model’s input, so writing labels is prompt engineering. Most reported zero-shot numbers are quietly the best of several phrasings.
- **The cost scales with the taxonomy.** A 200-label taxonomy means 200 forward passes per document. In practice you retrieve a candidate shortlist with embeddings first (`07_Feature_Extraction`) and only run NLI on the top 10.
- **Confidence is not calibrated.** Entailment probability is a ranking signal. Thresholding it directly to decide “none of these labels apply” works poorly without calibration on a held-out set.

------------------------------------------------------------------------

## 3. How Modern Zero-Shot Classification Works

1.  **Embedding-space similarity (2016-2019).** Embed the text and the label name in a shared space and take the nearest label. Cheap and still a good shortlist generator, but label names are short and out of distribution as sentences, so accuracy is mediocre.
2.  **The NLI reformulation (2019).** Yin, Hay and Roth showed that an off-the-shelf MNLI model used as an entailment scorer beat purpose-built zero-shot methods. `facebook/bart-large-mnli` became the default implementation and, remarkably, is still a reasonable baseline seven years later.
3.  **Prompt- and pattern-based fine-tuning (2020-2021).** PET and its relatives reframed classification as cloze completion over a masked-LM - `"The article is about ___"` - which sits between zero-shot and fine-tuning and works well in the few-shot regime. `09_Fill_Mask` covers the mechanism.
4.  **Instruction-tuned NLI models (2022-2024).** Rather than train only on MNLI, train on dozens of classification datasets converted into NLI format. Moritz Laurer’s `deberta-v3-*-zeroshot-v2.0` line is the widely used result: same NLI interface, substantially better zero-shot transfer, and commercially usable data provenance. mDeBERTa and XLM-R variants extended this cross-lingually via XNLI, so English labels work on non-English text.
5.  **LLM prompting (2023-2026).** Ask an instruction-tuned decoder to pick a label. This handles labels that need reasoning (“is this sarcastic?”, “does this violate clause 4.2?”), accepts label *definitions* rather than label names, and needs one pass regardless of label count - which inverts the cost comparison as the taxonomy grows. The catch is constraining the output to the label set; scoring the label tokens from a single forward pass (as section 11 does) solves that cleanly.
6.  **Distillation back to a small model (2024-2026).** The mature pattern: an LLM labels 5-10k documents, humans audit a sample, and a 150M encoder is fine-tuned on the result. This turns a zero-shot capability into a production classifier at 1/100th the cost.

**Where it stands (mid-2026).** NLI models remain the best accuracy-per-FLOP option for a small taxonomy (up to ~10 labels) with no training data, and they are small enough to run on CPU. LLMs win when labels are complex, defined by policy text, or numerous. Neither should be the permanent answer for a high-volume stable task - both are outperformed on cost and accuracy by a fine-tuned encoder as soon as you have a few thousand labelled examples, and the fastest way to get those is to use the zero-shot model to make them.

------------------------------------------------------------------------

## 4. Evaluation Metrics

Zero-shot classification is scored with ordinary classification metrics - accuracy, macro F1, per-class precision/recall - so `00_Text_Classification` section 4 applies in full. What is different is **what you must control for**, and it is easy to report a number that means nothing.

**Macro F1 is the right default.** Zero-shot models have systematic per-label biases (some label wordings attract far more predictions than others), so accuracy on an imbalanced set flatters them. Macro F1 exposes the label the model never predicts.

**The prediction is an argmax over hypothesis scores**, so the comparison is between labels, not against an absolute threshold. That has a consequence worth internalising: the model is nearly always *forced* to pick something. “None of these labels apply” requires either `multi_label=True` with a tuned threshold, or an explicit `"none of the above"` candidate label - and the latter works surprisingly well.

**Report the template you used.** A zero-shot number without its hypothesis template is unreproducible. `"This example is {}."` and `"This news article is about {}."` are different systems. Section 9 measures the spread directly.

**Fair-comparison rules that are routinely broken:**

- **The label set must be identical across models.** Adding a `"none"` option or renaming a label changes the task, not the model.
- **Do not tune label wording on the test set** and then report test accuracy. Tune on a validation split, exactly as you would a learning rate.
- **State the label count with the latency.** “45 ms per document” means nothing without “over 4 labels”; the same model over 20 labels is 5x slower.
- **Compare against the right baseline.** The interesting question is not “does zero-shot beat random?” but “how far behind a fine-tuned model is it, and is the gap worth the labelling effort?” Section 12 keeps that framing.

The cell below defines the metrics used throughout - the same accuracy and macro-F1 implementations as `00_Text_Classification`, plus a per-label prediction-count report that makes label bias visible.

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
    "Unweighted mean of the per-class F1 - exposes the label the model never picks."
    scores = per_class_prf(y_true, y_pred, labels)
    return sum(v[2] for v in scores.values()) / len(scores)


def label_bias(y_true, y_pred, labels, title="Label bias"):
    "Predicted vs true count per label. Zero-shot models over-predict attractive wordings."
    pred_n, true_n = Counter(y_pred), Counter(y_true)
    show_table([{"label": lab, "predicted": pred_n[lab], "true": true_n[lab],
                 "skew": round(pred_n[lab] / true_n[lab], 2) if true_n[lab] else None}
                for lab in labels],
               title=title,
               caption="skew of 1.0 is balanced; a label never predicted scores 0.0 and "
                       "is invisible in the accuracy number")


# Toy example: a zero-shot model that loves the label "world" and never picks "business".
labels = ["world", "sports", "business", "science and technology"]
y_true = ["world"] * 5 + ["sports"] * 5 + ["business"] * 5 + ["science and technology"] * 5
y_pred = (["world"] * 5 + ["sports"] * 4 + ["world"] + ["world"] * 3 + ["science and technology"] * 2
          + ["science and technology"] * 4 + ["world"])

show_kv({"accuracy": round(accuracy(y_true, y_pred), 3),
         "macro F1": round(macro_f1(y_true, y_pred, labels), 3)},
        title="Accuracy looks acceptable; macro F1 is the honest number")
label_bias(y_true, y_pred, labels,
           title="'business' is never predicted - accuracy hides it, macro F1 does not")
```

```
    Accuracy looks acceptable; macro F1 is the honest number    
                                                                
 accuracy                                                0.6500 
 macro F1                                                0.5710 
                                                                
```

```
  'business' is never predicted - accuracy hides it, macro F1 does not  
                                                                        
 label                                   predicted      true       skew 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 world                                          10         5     2.0000 
 sports                                          4         5     0.8000 
 business                                        0         5     0.0000 
 science and technology                          6         5     1.2000 
                                                                        
   skew of 1.0 is balanced; a label never predicted scores 0.0 and is   
                    invisible in the accuracy number                    
```

## 5. Datasets

Two kinds of dataset matter here, and they play different roles. **NLI datasets** are what the models are trained on; **classification datasets** are what zero-shot performance is measured on, precisely because the model never saw them.

| Dataset | Role | Contents | Size | Scope | License |
|----|----|----|----|----|----|
| [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) | training | Premise/hypothesis pairs, 3-way entailment, 10 genres | 393k | en | OANC / mixed |
| [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) | training | Image-caption-derived NLI pairs | 570k | en | CC BY-SA 4.0 |
| [XNLI](https://huggingface.co/datasets/facebook/xnli) | training | MNLI dev/test translated into 15 languages | 112k | 15 langs | CC BY-NC 4.0 |
| [ANLI](https://huggingface.co/datasets/facebook/anli) | training | Adversarially collected NLI, 3 rounds | 163k | en | CC BY-NC 4.0 |
| [FEVER-NLI](https://huggingface.co/datasets/pietrolesci/nli_fever) | training | Fact-verification recast as NLI | 180k | en | CC BY-SA 3.0 |
| [WANLI](https://huggingface.co/datasets/alisawuffles/WANLI) | training | Worker-and-AI generated NLI, harder patterns | 103k | en | CC BY 4.0 |
| [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) | evaluation | News headline + lead, 4 topics | 120k / 7.6k | en | custom, research |
| [Yahoo Answers Topics](https://huggingface.co/datasets/community-datasets/yahoo_answers_topics) | evaluation | Questions, 10 topics | 1.4M | en | Apache 2.0 |
| [Emotion](https://huggingface.co/datasets/dair-ai/emotion) | evaluation | Tweets, 6 emotions | 20k | en | educational |
| [Banking77](https://huggingface.co/datasets/PolyAI/banking77) | evaluation | Banking queries, **77 fine-grained intents** | 13k | en | CC BY 4.0 |
| [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) | evaluation | Assistant utterances, 60 intents | 1M | 51 langs | CC BY 4.0 |
| [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | evaluation | Reddit comments, 27 **multi-label** emotions | 58k | en | Apache 2.0 |

This notebook evaluates on the **AG News test split** with 4 topic labels. It is a deliberately friendly choice: four well-separated topics with natural-language names, which is close to the best case for zero-shot. Banking77 (77 intents) or GoEmotions (27 overlapping multi-label emotions) are the honest stress tests, and zero-shot accuracy on those is far lower - if you want a realistic picture of your own hard taxonomy, swap the dataset in the Setup cell and expect a shock.

The label names below are the **verbalised** forms, not the raw dataset labels: AG News ships `World / Sports / Business / Sci/Tech`, and `"Sci/Tech"` is a poor hypothesis phrase. Section 9 measures exactly how much that rewording is worth.

Downloads land in `DL_tasks/datasets/` via `cache_dir` (gitignored).

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

There is no single zero-shot leaderboard. The useful references are the model cards in Moritz Laurer’s [zeroshot collection](https://huggingface.co/collections/MoritzLaurer/zeroshot-classifiers-6548b4ff407bb19ff5c3ad6f) (which report averaged accuracy over ~28 held-out classification sets), the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for the embedding-based alternative, and general LLM boards for the prompting route.

| Model | Params | License | Type | Languages | Best for |
|----|----|----|----|----|----|
| [bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli) | 407M | MIT | NLI (MNLI only) | en | the classic default; used below |
| [deberta-v3-base-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/deberta-v3-base-zeroshot-v2.0) | 184M | MIT | NLI + 28 task mix | en | best small zero-shot; used below |
| [deberta-v3-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0) | 435M | MIT | NLI + task mix | en | best English accuracy in this family |
| [mDeBERTa-v3-base-xnli](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli) | 278M | MIT | XNLI multilingual | 100 | English labels on non-English text; used below |
| [bge-m3-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/bge-m3-zeroshot-v2.0) | 568M | MIT | NLI over a multilingual encoder | 100+ | multilingual with long context (8192) |
| [nli-deberta-v3-small](https://huggingface.co/cross-encoder/nli-deberta-v3-small) | 142M | Apache 2.0 | NLI | en | CPU-latency-bound deployments |
| [Qwen3-0.6B / 1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 0.6-1.7B | Apache 2.0 | decoder LLM | 100+ | complex labels, big taxonomies; used below |
| [flan-t5-large](https://huggingface.co/google/flan-t5-large) | 780M | Apache 2.0 | instruction seq2seq | en | prompted classification without a chat template |
| Frontier LLMs (Claude, GPT, Gemini) | \- | proprietary | decoder LLM | 100+ | policy-defined labels, nuance, and label bootstrapping |

**How to choose.** Up to ~10 well-named labels, English, no training data: `deberta-v3-base-zeroshot-v2.0` - it is smaller than bart-large-mnli and clearly better. Non-English text with English labels: `mDeBERTa-v3-base-xnli` or `bge-m3-zeroshot-v2.0`. More than ~20 labels: either shortlist with embeddings and run NLI on the top-k, or switch to an LLM, whose cost does not scale with the label count. Labels that need a paragraph to define: LLM, no contest - NLI takes a short hypothesis, not a policy document.

------------------------------------------------------------------------

## 7. Setup

Everything loads through Hugging Face `transformers` - no vendor packages. Package roles:

- `transformers` + `torch` - the three NLI models and the LLM
- `accelerate` - `device_map` placement
- `datasets` - the AG News test split
- `pandas` + `pyecharts` - the benchmark table and charts
- `rich` - the result tables. It renders to HTML inside Jupyter, so the tables survive into the published docs; `show_table` / `show_kv` / `rule` are defined in the first code cell of section 4.

Three `transformers` details that decide whether this works:

- **`pipeline("zero-shot-classification")` runs one forward pass per (text, label) pair.** With 4 labels and 200 texts that is 800 passes. Batch size applies to those pairs, not to the texts, and this is why the throughput numbers below look low for a 184M model.
- **`hypothesis_template` is a constructor-level knob** with the default `"This example is {}."`. It is the most under-used tuning parameter in the task; section 9 sweeps it.
- **`multi_label=False`** (the default) softmaxes the entailment logits across the candidate labels, so scores sum to 1 and exactly one label wins. **`multi_label=True`** runs an independent entailment-vs-contradiction softmax per label, so each score is standalone and any subset can pass a threshold. Using the default for a genuinely multi-label problem silently caps recall, and is the most common misuse of this pipeline.

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
    """Collect garbage and hand freed VRAM back to the CUDA allocator.

    Call right after `del`-ing a model you are done with: `del model; free_memory()`.
    `del` drops the Python reference; this reclaims the RAM and releases the VRAM.
    """
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

# AG News test split: news headline + lead paragraph, 4 topics. None of the models
# below has been fine-tuned on it - that is the whole point.
ag = load_dataset("fancyzhx/ag_news", split="test", cache_dir=HF_CACHE)

# Verbalised label names. The raw dataset labels are World / Sports / Business / Sci/Tech;
# "Sci/Tech" is a bad hypothesis phrase, so it is spelled out. Section 9 measures the cost
# of getting this wrong.
LABELS = ["world news", "sports", "business", "science and technology"]
RAW_TO_LABEL = dict(zip(range(4), LABELS))

N = 200  # texts to evaluate. Remember: N x len(LABELS) forward passes per NLI model.
sample = ag.shuffle(seed=0).select(range(N))
texts = [r["text"].strip() for r in sample]
gold = [RAW_TO_LABEL[r["label"]] for r in sample]

print(ag)
print(f"\n{N} texts x {len(LABELS)} labels = {N * len(LABELS)} NLI forward passes per model")
print("label distribution:", dict(sorted(Counter(gold).items())), "\n")
for t, g in list(zip(texts, gold))[:3]:
    print(f"  [{g:22s}] {t[:100]}")
```

    Dataset({
        features: ['text', 'label'],
        num_rows: 7600
    })

    200 texts x 4 labels = 800 NLI forward passes per model
    label distribution: {'business': 50, 'science and technology': 54, 'sports': 47, 'world news': 49} 

      [business              ] McTeer: Lonesome Dove to be an Aggie NEW YORK (CNN/Money) - A New Economy champion, a lover of the T
      [world news            ] Peru Gov't: Police Killed in Self-Defense Peru's interior minister said Wednesday that police acted 
      [science and technology] SpaceShipOne Rolls Toward Victory MOJAVE, California -- A Southern California aerospace team took a 

## 8. The classic: bart-large-mnli

The model that made zero-shot classification a one-liner, and still the first hit for the task. It is BART-large fine-tuned on MultiNLI - **only** MNLI, no classification-task mixture - so it is the purest demonstration of the reformulation: a model trained solely to judge entailment, used as a general classifier.

The cell below prints the underlying NLI call before using the pipeline, because the pipeline hides the mechanism and the mechanism is the interesting part. `premise = text`, `hypothesis = "This example is sports."`, read `P(entailment)`. That is all a zero-shot classifier is.

At 407M params it is also the slowest model here per pass, and the least accurate of the three NLI models - seven years of better training data has more than made up for a smaller backbone.

------------------------------------------------------------------------

``` python
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# First, the raw mechanism, with no pipeline in the way.
nli_id = "facebook/bart-large-mnli"
nli_tok = AutoTokenizer.from_pretrained(nli_id, cache_dir=HF_CACHE)
nli = AutoModelForSequenceClassification.from_pretrained(nli_id, cache_dir=HF_CACHE).to(device).eval()
print("NLI label order:", nli.config.id2label)

premise = "Manchester United beat Arsenal 3-1 at Old Trafford on Sunday."
with torch.inference_mode():
    for label in LABELS:
        hypothesis = f"This example is {label}."
        enc = nli_tok(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
        probs = nli(**enc).logits.softmax(-1)[0]
        ent = probs[nli.config.label2id["entailment"]].item()
        print(f"  P(entailment) {ent:.3f}   hypothesis: {hypothesis!r}")
print("  ^ argmax over these is the zero-shot prediction\n")

del nli, nli_tok
free_memory()

# Now the same thing through the pipeline, which does exactly this for every pair.
zs = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=device,
    model_kwargs={"cache_dir": HF_CACHE},
)
vram("bart-large-mnli loaded")


def run_zeroshot(pipe, labels=LABELS, template="This example is {}.", batch_size=16):
    "Classify every text, returning predictions and wall-clock seconds."
    t0 = time.perf_counter()
    outs = pipe(texts, candidate_labels=labels, hypothesis_template=template,
                multi_label=False, batch_size=batch_size)
    return [o["labels"][0] for o in outs], time.perf_counter() - t0


bart_preds, bart_secs = run_zeroshot(zs)
show_kv({"texts": N, "NLI forward passes": N * len(LABELS),
         "seconds": round(bart_secs, 1),
         "docs / second": round(N / bart_secs, 1),
         "NLI passes / second": round(N * len(LABELS) / bart_secs, 1),
         "accuracy": round(accuracy(gold, bart_preds), 3),
         "macro F1": round(macro_f1(gold, bart_preds, LABELS), 3)},
        title="facebook/bart-large-mnli - trained on MNLI alone")
label_bias(gold, bart_preds, LABELS, title="Label bias, bart-large-mnli")

del zs
free_memory()
vram("after bart-large-mnli")
```

    NLI label order: {0: 'contradiction', 1: 'neutral', 2: 'entailment'}
      P(entailment) 0.064   hypothesis: 'This example is world news.'
      P(entailment) 0.470   hypothesis: 'This example is sports.'
      P(entailment) 0.101   hypothesis: 'This example is business.'
      P(entailment) 0.013   hypothesis: 'This example is science and technology.'
      ^ argmax over these is the zero-shot prediction

    VRAM bart-large-mnli loaded  1.64 GB allocated /  1.65 GB reserved

```
        facebook/bart-large-mnli - trained on MNLI alone        
                                                                
 texts                                                      200 
 NLI forward passes                                         800 
 seconds                                                 7.9000 
 docs / second                                            25.20 
 NLI passes / second                                     100.70 
 accuracy                                                0.6650 
 macro F1                                                0.6310 
                                                                
```

```
                      Label bias, bart-large-mnli                       
                                                                        
 label                                   predicted      true       skew 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 world news                                     81        49     1.6500 
 sports                                         46        47     0.9800 
 business                                       65        50     1.3000 
 science and technology                          8        54     0.1500 
                                                                        
   skew of 1.0 is balanced; a label never predicted scores 0.0 and is   
                    invisible in the accuracy number                    
```

    VRAM after bart-large-mnli   0.01 GB allocated /  0.02 GB reserved

## 9. Modern zero-shot: deberta-v3-base-zeroshot-v2.0

Half the parameters of bart-large-mnli and consistently better. The gain comes entirely from training data: instead of MNLI alone, this was trained on a mixture of NLI corpora **plus ~28 classification datasets converted into the NLI format** (each label becomes a hypothesis, correct label = entailment, others = not-entailment). The model therefore learned the *shape* of “is this text about X?” across many taxonomies, not just textual entailment on Wikipedia-style pairs.

The `v2.0` line also has a deliberate licensing story: the variants without a `-c` suffix are trained only on commercially permissive data, which matters because MNLI-derived models inherit non-commercial constraints from XNLI and ANLI.

**The template sweep is the substance of this section.** The hypothesis is part of the model’s input, so changing it changes the prediction - and the swing is large enough that any zero-shot number reported without its template is not reproducible. The sweep below includes the raw dataset label names (`Sci/Tech`) against the verbalised ones, which is the single highest-leverage change available in this task and costs nothing.

------------------------------------------------------------------------

``` python
zs2 = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
    device=device,
    model_kwargs={"cache_dir": HF_CACHE},
)
vram("deberta-v3-zeroshot loaded")

deberta_preds, deberta_secs = run_zeroshot(zs2)
show_kv({"texts": N, "seconds": round(deberta_secs, 1),
         "docs / second": round(N / deberta_secs, 1),
         "accuracy": round(accuracy(gold, deberta_preds), 3),
         "macro F1": round(macro_f1(gold, deberta_preds, LABELS), 3)},
        title="deberta-v3-base-zeroshot-v2.0 - half the parameters, better training data")

# The hypothesis template is a hyperparameter. Sweep it on a subset (the full sweep is
# len(templates) x N x len(labels) forward passes).
TEMPLATES = [
    "This example is {}.",                       # the library default
    "This text is about {}.",
    "This news article is about {}.",            # domain-matched
    "The topic of this article is {}.",
    "{}",                                        # bare label, no sentence frame
]
SUB = 80
sub_texts, sub_gold = texts[:SUB], gold[:SUB]

template_scores = []
for tmpl in TEMPLATES:
    outs = zs2(sub_texts, candidate_labels=LABELS, hypothesis_template=tmpl,
               multi_label=False, batch_size=16)
    preds = [o["labels"][0] for o in outs]
    template_scores.append({"template": tmpl, "labels": "verbalised",
                            "accuracy": round(accuracy(sub_gold, preds), 3),
                            "macro_f1": round(macro_f1(sub_gold, preds, LABELS), 3)})

# And the label wording itself, holding the template fixed. AG News ships "Sci/Tech".
RAW_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
raw_gold = [RAW_LABELS[LABELS.index(g)] for g in sub_gold]
outs = zs2(sub_texts, candidate_labels=RAW_LABELS, hypothesis_template="This example is {}.",
           multi_label=False, batch_size=16)
raw_preds = [o["labels"][0] for o in outs]
template_scores.append({"template": "This example is {}.", "labels": "raw dataset names",
                        "accuracy": round(accuracy(raw_gold, raw_preds), 3),
                        "macro_f1": round(macro_f1(raw_gold, raw_preds, RAW_LABELS), 3)})
show_table(template_scores, title=f"Same model, same {SUB} texts - only the wording changed",
           best=("accuracy", "macro_f1"),
           caption="the last row uses the raw dataset labels (World / Sports / Business / "
                   "Sci-Tech); rewording them is free accuracy in either direction")

del zs2, outs
free_memory()
vram("after deberta-v3-zeroshot")
```

    VRAM deberta-v3-zeroshot loaded  0.38 GB allocated /  0.39 GB reserved

```
  deberta-v3-base-zeroshot-v2.0 - half the parameters, better   
                         training data                          
                                                                
 texts                                                      200 
 seconds                                                 1.7000 
 docs / second                                           115.80 
 accuracy                                                0.9150 
 macro F1                                                0.9170 
                                                                
```

```
            Same model, same 80 texts - only the wording changed            
                                                                            
 template                           labels              accuracy   macro_f1 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 This example is {}.                verbalised            0.9000     0.8940 
 This text is about {}.             verbalised            0.9000     0.8940 
 This news article is about {}.     verbalised            0.9000     0.8940 
 The topic of this article is {}.   verbalised            0.9000     0.8940 
 {}                                 verbalised            0.8870     0.8860 
 This example is {}.                raw dataset names     0.8870     0.8800 
                                                                            
   the last row uses the raw dataset labels (World / Sports / Business /    
       Sci-Tech); rewording them is free accuracy in either direction       
```

    VRAM after deberta-v3-zeroshot  0.01 GB allocated /  0.02 GB reserved

``` python
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org serves an expired TLS certificate from most nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"

from pyecharts import options as opts
from pyecharts.charts import Bar

# The template/label sweep, drawn. The spread is the point: a zero-shot number without
# its template is not a reproducible number.
names = [f"{s['labels']}: {s['template']}" for s in template_scores]
bar = (
    Bar()
    .add_xaxis(names)
    .add_yaxis("accuracy x100", [round(s["accuracy"] * 100, 1) for s in template_scores])
    .add_yaxis("macro F1 x100", [round(s["macro_f1"] * 100, 1) for s in template_scores])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="deberta-v3-base-zeroshot: hypothesis template sensitivity",
            subtitle=f"AG News, {SUB} texts, 4 labels - the model is identical in every bar",
        ),
        yaxis_opts=opts.AxisOpts(name="score", min_=0, max_=100),
        xaxis_opts=opts.AxisOpts(name="template / label wording",
                                 axislabel_opts=opts.LabelOpts(rotate=25, font_size=9)),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="8%"),
    )
)
bar.render_notebook()
```

## 10. Cross-lingual: mDeBERTa-v3-base-xnli

The same NLI trick, but the backbone is multilingual (mDeBERTa-v3, 100 languages) and the fine-tuning includes **XNLI** - MNLI translated into 15 languages. The practical consequence is the interesting one: because the encoder shares a representation space across languages, you can keep the **candidate labels in English** and classify text in German, Chinese or Arabic. You do not need labels, training data, or even a tokeniser decision per language.

This is genuinely useful and genuinely uneven. Transfer is strong for the 15 XNLI languages and for high-resource languages close to them; it degrades for low-resource languages, and it degrades quietly - the model returns confident scores either way. Before deploying it on a language, label 200 examples in that language and measure. That is still 100x less labelling than training a classifier per language.

The cell below scores the same English AG News sample (so it is comparable in the benchmark) and then demonstrates the cross-lingual property directly on a handful of non-English sentences with English labels.

------------------------------------------------------------------------

``` python
zs3 = pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    device=device,
    model_kwargs={"cache_dir": HF_CACHE},
)
vram("mdeberta-xnli loaded")

mdeberta_preds, mdeberta_secs = run_zeroshot(zs3)
show_kv({"texts": N, "seconds": round(mdeberta_secs, 1),
         "docs / second": round(N / mdeberta_secs, 1),
         "accuracy": round(accuracy(gold, mdeberta_preds), 3),
         "macro F1": round(macro_f1(gold, mdeberta_preds, LABELS), 3)},
        title="mDeBERTa-v3-base-xnli, scored on the same English texts")

# The actual selling point: non-English text, English labels, no training data.
MULTILINGUAL = [
    ("de", "Der Bundestag hat gestern ein neues Klimagesetz verabschiedet."),
    ("fr", "Le PSG a remporte le match hier soir contre Marseille."),
    ("es", "La empresa anuncio beneficios record en el tercer trimestre."),
    ("it", "I ricercatori hanno sviluppato un nuovo chip quantistico."),
]
show_table([{"lang": lang, "predicted label": o["labels"][0],
             "score": round(o["scores"][0], 3), "text": text}
            for lang, text, o in ((l, t, zs3(t, candidate_labels=LABELS, multi_label=False))
                                  for l, t in MULTILINGUAL)],
           title="English labels, non-English text, no training data", best=("score",),
           caption="transfer is strong for the 15 XNLI languages and degrades quietly "
                   "for low-resource ones - measure before deploying on a language")

del zs3
free_memory()
vram("after mdeberta-xnli")
```

    VRAM mdeberta-xnli loaded    0.57 GB allocated /  0.58 GB reserved

```
    mDeBERTa-v3-base-xnli, scored on the same English texts     
                                                                
 texts                                                      200 
 seconds                                                 1.4000 
 docs / second                                           139.30 
 accuracy                                                0.6250 
 macro F1                                                0.6020 
                                                                
```

```
                           English labels, non-English text, no training data                            
                                                                                                         
 lang   predicted label           score   text                                                           
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 de     world news               0.6820   Der Bundestag hat gestern ein neues Klimagesetz verabschiedet. 
 fr     sports                   0.5540   Le PSG a remporte le match hier soir contre Marseille.         
 es     business                 0.9390   La empresa anuncio beneficios record en el tercer trimestre.   
 it     science and technology   0.7280   I ricercatori hanno sviluppato un nuovo chip quantistico.      
                                                                                                         
transfer is strong for the 15 XNLI languages and degrades quietly for low-resource ones - measure before 
                                         deploying on a language                                         
```

    VRAM after mdeberta-xnli     0.01 GB allocated /  0.02 GB reserved

## 11. The other zero-shot: an LLM (Qwen3-0.6B)

An instruction-tuned decoder does the same job through a completely different mechanism, and the cost model inverts. NLI needs one forward pass **per label**; the LLM needs one pass **per document** regardless of how many labels there are. At 4 labels NLI is cheaper; somewhere around 8-15 labels they cross, and at 77 labels (Banking77) it is not close.

The problem with a generative classifier is that it generates. Ask for a label and you may get `"Sports!"`, `"This article is about sports"`, or a paragraph of reasoning. The fix used here is the same one as in `00_Text_Classification`: **do not let it generate**. Run one forward pass, take the next-token logits, and compare only the scores of the first token of each candidate label. The output is restricted to the label set by construction, comes with a usable probability, and costs a single prefill.

The catch worth knowing: this works only when the candidate labels have **distinct first tokens**. `"business"` and `"business travel"` share theirs, so the trick silently conflates them. Check with the printed token ids below; where labels collide, either rename them, score the full label sequence, or fall back to constrained generation.

The LLM’s real advantage is not accuracy on four clean topics - it is that a label can be a *definition*. `"a complaint where the customer explicitly asks for a refund"` is a valid label for an LLM and an unusable hypothesis for an NLI model.

`enable_thinking=False` keeps Qwen3 from emitting a `<think>` block, which matters because only the first generated token is read.

------------------------------------------------------------------------

``` python
from transformers import AutoModelForCausalLM

llm_id = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(llm_id, cache_dir=HF_CACHE)
llm = AutoModelForCausalLM.from_pretrained(
    llm_id, dtype=dtype, device_map=device, cache_dir=HF_CACHE
).eval()
vram("qwen3-0.6b loaded")

# The trick needs distinct first tokens per label - check, do not assume.
LABEL_IDS = [tok.encode(" " + lab, add_special_tokens=False)[0] for lab in LABELS]
print("first-token ids:", dict(zip(LABELS, LABEL_IDS)))
assert len(set(LABEL_IDS)) == len(LABELS), "labels share a first token - rename or score full sequences"

PROMPT = (
    "Classify the news article into exactly one category: "
    + ", ".join(LABELS)
    + ".\n\nArticle: {text}\n\nCategory:"
)


@torch.inference_mode()
def llm_classify(batch):
    "One forward pass per document; score the label words against each other."
    chats = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(text=t[:1200])}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for t in batch
    ]
    enc = tok(chats, return_tensors="pt", padding=True, padding_side="left",
              truncation=True, max_length=768).to(llm.device)
    logits = llm(**enc).logits[:, -1, :]           # next-token distribution only
    probs = torch.softmax(logits[:, LABEL_IDS].float(), dim=-1)
    idx = probs.argmax(-1)
    return [LABELS[i] for i in idx.tolist()], probs.max(-1).values.tolist()

t0 = time.perf_counter()
llm_preds = []
for i in range(0, N, 16):
    llm_preds.extend(llm_classify(texts[i:i + 16])[0])
llm_secs = time.perf_counter() - t0

show_kv({"texts": N, "forward passes": N,
         "seconds": round(llm_secs, 1),
         "docs / second": round(N / llm_secs, 1),
         "accuracy": round(accuracy(gold, llm_preds), 3),
         "macro F1": round(macro_f1(gold, llm_preds, LABELS), 3)},
        title="Qwen3-0.6B - one pass per document, not per label")
label_bias(gold, llm_preds, LABELS, title="Label bias, qwen3-0.6b")

# What NLI cannot express: a label that is a definition rather than a name.
RICH_LABELS = [
    "a report about an armed conflict or a government decision",
    "a report about a professional sports result",
    "a report about company earnings, markets or trade",
    "a report about a scientific finding or a new technology",
]
rich_ids = [tok.encode(" " + lab, add_special_tokens=False)[0] for lab in RICH_LABELS]
print(f"\nrich labels share first tokens ({len(set(rich_ids))} distinct of {len(RICH_LABELS)}) "
      "- this is where first-token scoring breaks and constrained generation is needed")

del llm, tok
free_memory()
vram("after qwen3")
```

    VRAM qwen3-0.6b loaded       1.20 GB allocated /  1.67 GB reserved
    first-token ids: {'world news': 1879, 'sports': 9833, 'business': 2562, 'science and technology': 8038}

```
       Qwen3-0.6B - one pass per document, not per label        
                                                                
 texts                                                      200 
 forward passes                                             200 
 seconds                                                 2.4000 
 docs / second                                            83.40 
 accuracy                                                0.4800 
 macro F1                                                0.4120 
                                                                
```

```
                         Label bias, qwen3-0.6b                         
                                                                        
 label                                   predicted      true       skew 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 world news                                     52        49     1.0600 
 sports                                          0        47     0.0000 
 business                                       73        50     1.4600 
 science and technology                         75        54     1.3900 
                                                                        
   skew of 1.0 is balanced; a label never predicted scores 0.0 and is   
                    invisible in the accuracy number                    
```

    rich labels share first tokens (1 distinct of 4) - this is where first-token scoring breaks and constrained generation is needed
    VRAM after qwen3             0.01 GB allocated /  0.02 GB reserved

## 12. Head-to-head Benchmark

The same 200 AG News texts, the same four verbalised labels, the same default template, one model live at a time. Sections 8-11 produced the numbers; this collects them.

Two things to read out of it, neither of which is “which model is best”:

- **Throughput is not comparable per pass.** The NLI models each ran `200 x 4 = 800` forward passes; the LLM ran 200. The `docs/s` column is the honest comparison for *this* label count, and it would move against the NLI models on every label you add. The chart plots both so the crossover is visible.
- **The gap to a fine-tuned model is the number that matters.** A ModernBERT-base fine-tuned on AG News reaches ~94-95% accuracy with a few thousand labels and runs at 10-50x the throughput of any model here. Everything in this notebook exists to cover the period before you have those labels, or to create them.

At n=200 with 4 classes, sampling noise is roughly +/-3.5 points; smaller differences are not real.

------------------------------------------------------------------------

``` python
import pandas as pd

RUNS = [
    ("bart-large-mnli", 407, "NLI", bart_preds, bart_secs, N * len(LABELS)),
    ("deberta-v3-base-zeroshot-v2.0", 184, "NLI", deberta_preds, deberta_secs, N * len(LABELS)),
    ("mDeBERTa-v3-base-xnli", 278, "NLI", mdeberta_preds, mdeberta_secs, N * len(LABELS)),
    ("qwen3-0.6b (label logits)", 596, "LLM", llm_preds, llm_secs, N),
]

results = []
for name, params_m, kind, preds, secs, passes in RUNS:
    results.append({
        "model": name,
        "params_m": params_m,
        "type": kind,
        "accuracy": round(accuracy(gold, preds), 4),
        "macro_f1": round(macro_f1(gold, preds, LABELS), 4),
        "fwd_passes": passes,
        "seconds": round(secs, 2),
        "docs_per_sec": round(N / secs, 1),
    })

df_results = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
show_table(
    df_results.to_dict("records"),
    title=f"AG News test, zero-shot, {N} texts x {len(LABELS)} labels",
    best=("accuracy", "macro_f1", "docs_per_sec"),
    lower_is_better=("fwd_passes", "seconds"),
    caption="fwd_passes is the cost model: NLI pays per label, the LLM pays per document",
)
```

```
                                AG News test, zero-shot, 200 texts x 4 labels                                
                                                                                                             
 model                           params_m   type   accuracy   macro_f1   fwd_passes   seconds   docs_per_sec 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 deberta-v3-base-zeroshot-v2.0        184   NLI      0.9150     0.9171          800    1.7300         115.80 
 bart-large-mnli                      407   NLI      0.6650     0.6307          800    7.9500          25.20 
 mDeBERTa-v3-base-xnli                278   NLI      0.6250     0.6025          800    1.4400         139.30 
 qwen3-0.6b (label logits)            596   LLM      0.4800     0.4122          200    2.4000          83.40 
                                                                                                             
                 fwd_passes is the cost model: NLI pays per label, the LLM pays per document                 
```

``` python
bar = (
    Bar()
    .add_xaxis([r["model"] for r in results])
    .add_yaxis("accuracy x100", [round(r["accuracy"] * 100, 1) for r in results])
    .add_yaxis("macro F1 x100", [round(r["macro_f1"] * 100, 1) for r in results])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"AG News test, zero-shot ({N} texts, 4 labels)",
            subtitle="RTX 3060 - none of these models was trained on AG News; "
                     "a fine-tuned encoder reaches ~95%",
        ),
        yaxis_opts=opts.AxisOpts(name="score", min_=0, max_=100),
        xaxis_opts=opts.AxisOpts(name="model", axislabel_opts=opts.LabelOpts(rotate=18, font_size=9)),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="10%"),
    )
)
bar.render_notebook()
```

``` python
from pyecharts.charts import Line

# The cost model, extrapolated: NLI cost grows linearly with the label count, the LLM's
# does not. Measured single-pass rates from the runs above, projected over taxonomy size.
nli_pass_rate = (N * len(LABELS)) / deberta_secs   # NLI forward passes per second
llm_doc_rate = N / llm_secs                        # documents per second (1 pass each)
label_counts = [2, 4, 8, 16, 32, 64, 77]

line = (
    Line()
    .add_xaxis([str(k) for k in label_counts])
    .add_yaxis("deberta-v3-zeroshot (NLI)", [round(nli_pass_rate / k, 1) for k in label_counts])
    .add_yaxis("qwen3-0.6b (one pass/doc)", [round(llm_doc_rate, 1) for _ in label_counts])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="Throughput vs taxonomy size",
            subtitle="measured rates projected over label count - NLI pays per label, the LLM does not",
        ),
        xaxis_opts=opts.AxisOpts(name="number of candidate labels"),
        yaxis_opts=opts.AxisOpts(name="documents / second", type_="log"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
    )
)
line.render_notebook()
```

## 13. Interactive: classify with your own labels

Edit `MY_TEXTS` and `MY_LABELS` below - that is the whole point of zero-shot, and it needs no retraining. This is the cell people run on its own, so it opens with a `require(...)` guard naming what it needs from Setup rather than dying on a bare `NameError`.

Things worth trying, in rough order of how much they will teach you:

- **Rewrite one label and watch everything move.** `"tech"` versus `"science and technology"` versus `"an article about a new technology"`. This is the highest-leverage knob in the task.
- **Add `"none of the above"` as a candidate.** Zero-shot models are forced to pick from what you give them, so with a fixed set they will always find a winner - even for a text about none of your topics. An explicit escape label is the cheapest fix for that and works better than thresholding.
- **Switch `MULTI_LABEL` to `True`.** Scores stop summing to 1 and become independent, which is what you need when a document can be about two topics at once. Then the threshold below 0.5 becomes yours to tune, per label.
- **Try overlapping labels** (`"business"` and `"finance"`). The model will split probability between them roughly arbitrarily; overlapping taxonomies are a data-design problem no model solves.

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
    "My package was supposed to arrive on Tuesday and it is still not here.",
    "The card was declined twice but the amount left my account both times.",
    "Can you tell me how to export my data before I close the account?",
    "The new dashboard looks great, well done to whoever built it.",
]

MY_LABELS = ["shipping problem", "billing problem", "account question", "none of the above"]
TEMPLATE = "This customer message is about {}."
MULTI_LABEL = False  # True -> independent per-label scores that do not sum to 1

# Re-runnable: this cell frees the pipeline at the end, so guard the load or a second
# shift-enter raises NameError.
if "my_zs" not in globals():
    my_zs = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
        device=device, model_kwargs={"cache_dir": HF_CACHE},
    )

print(f"template: {TEMPLATE!r}   multi_label={MULTI_LABEL}\n")
for text in MY_TEXTS:
    out = my_zs(text, candidate_labels=MY_LABELS, hypothesis_template=TEMPLATE,
                multi_label=MULTI_LABEL)
    scored = "  ".join(f"{l} {s:.2f}" for l, s in zip(out["labels"], out["scores"]))
    print(f"{text}\n   -> {out['labels'][0]}\n      {scored}\n")

del my_zs
free_memory()
vram("final")
```

    template: 'This customer message is about {}.'   multi_label=False

    My package was supposed to arrive on Tuesday and it is still not here.
       -> shipping problem
          shipping problem 0.84  none of the above 0.13  billing problem 0.02  account question 0.02

    The card was declined twice but the amount left my account both times.
       -> billing problem
          billing problem 0.83  none of the above 0.14  account question 0.01  shipping problem 0.01

    Can you tell me how to export my data before I close the account?
       -> account question
          account question 0.98  none of the above 0.01  billing problem 0.00  shipping problem 0.00

    The new dashboard looks great, well done to whoever built it.
       -> none of the above
          none of the above 0.45  account question 0.20  shipping problem 0.18  billing problem 0.17

    VRAM final                   0.01 GB allocated /  0.02 GB reserved

## 14. Common Frameworks

Zero-shot classification is best understood as a bootstrapping tool rather than a deployment target, and the framework list follows from that. What you need is something to run the NLI or LLM pass at volume, something to shortlist labels when the taxonomy is large, somewhere for a human to correct the low-confidence predictions, and a training loop to distil the result into a small encoder you can actually afford to serve.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [transformers](https://github.com/huggingface/transformers) | modelling | The `zero-shot-classification` pipeline over BART-MNLI and the DeBERTa zeroshot family, plus the multilingual XNLI models | Apache 2.0 | Default. Section 8 shows the pipeline is a hypothesis template plus an entailment score |
| [SetFit](https://github.com/huggingface/setfit) + [peft](https://github.com/huggingface/peft) | modelling | The distillation target: contrastive few-shot training on 8-64 corrected examples per class | Apache 2.0 | The moment you have any labels. This is where the 10-20 points of accuracy come from |
| [sentence-transformers](https://github.com/UKPLab/sentence-transformers) | data | Embedding the labels and the document to shortlist candidates before the expensive NLI pass | Apache 2.0 | Taxonomies past ~15 labels. Accuracy barely moves and cost drops by the shortlist ratio - see `07_Feature_Extraction` |
| [Argilla](https://github.com/argilla-io/argilla) / [Label Studio](https://github.com/HumanSignal/label-studio) | data | The correction loop: surface the low-confidence zero-shot predictions to a human, keep the disagreements | Apache 2.0 | Always. Correcting uncertain predictions is far cheaper per unit of accuracy than labelling from scratch |
| [datasets](https://github.com/huggingface/datasets) + [cleanlab](https://github.com/cleanlab/cleanlab) | data | Streaming the unlabelled corpus, and finding which of the model’s confident predictions are confidently wrong | Apache 2.0 / AGPL-3.0 (commercial license available) | Before training on your own pseudo-labels, which is exactly when errors get baked in |
| [vLLM](https://github.com/vllm-project/vllm) + [outlines](https://github.com/dottxt-ai/outlines) | inference runtime | Batched LLM classification with the output constrained to the label set, or first-token scoring for a calibrated score | Apache 2.0 | Labels that need a sentence of policy to define. NLI cannot represent a definition; an LLM can |
| [optimum](https://github.com/huggingface/optimum) + [ONNX Runtime](https://onnxruntime.ai/) | inference runtime | The NLI pass quantised - it runs once per (document, label) pair, so cost scales with taxonomy size | Apache 2.0 / MIT | Running zero-shot at volume, where the label-count multiplier is what hurts |
| [BentoML](https://github.com/bentoml/BentoML) / [Ray Serve](https://github.com/ray-project/ray) | serving | An endpoint where the label set and its per-label thresholds are configuration, not baked into the artefact | Apache 2.0 | Production. The taxonomy changes weekly; that is the whole reason you chose zero-shot |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) + [evaluate](https://github.com/huggingface/evaluate) | evaluation | Per-label precision/recall, the threshold sweep, and temperature scaling for calibration | BSD-3 / Apache 2.0 | Always. Entailment probabilities are not calibrated across label sets, so “reject below 0.7” means nothing untuned |

**The 2026 default stack** is DeBERTa zeroshot for the first pass, a sentence-transformers shortlist when the taxonomy is large, Argilla for correction, and SetFit or ModernBERT for the distilled production model. An LLM through vLLM when labels are definitional rather than lexical.

The common wrong turn is shipping zero-shot as the production classifier. It is 50x slower than the encoder you could distil it into and 10-20 points less accurate on your own data - it is a labelling engine, and treating it as one is the highest-value thing you can do here. The second is spending on a bigger model before tuning the label strings: three to five phrasings evaluated on 200 examples is half an hour of work and usually wins.

------------------------------------------------------------------------

## 15. Going Further

- **Treat zero-shot as a labelling engine.** Run it over your unlabelled corpus, have a human correct a few thousand of the low-confidence predictions, and fine-tune a 150M encoder on the result. You get 10-20 points of accuracy and 50x the throughput. This is the single most valuable thing to do with a zero-shot model, and `00_Text_Classification` section 14 covers the fine-tune.
- **Shortlist before you score.** For taxonomies past ~15 labels, embed the labels and the document (`07_Feature_Extraction`), retrieve the top 10 candidate labels by cosine similarity, and run NLI only on those. Accuracy barely moves and cost drops by the ratio of taxonomy size to shortlist size.
- **Tune the label strings on a validation set.** Write 3-5 phrasings per label, evaluate on 200 labelled examples, and keep the best. Half an hour of this typically beats moving to a model 3x larger, and it is the cheapest experiment in the notebook.
- **Use `multi_label=True` properly.** Independent sigmoid scores need a threshold **per label**, fitted on validation data. A single global 0.5 across labels with different score distributions is the second most common bug in this task.
- **Calibrate before you threshold.** Entailment probabilities are not calibrated across label sets. Temperature scaling on a small labelled sample makes “reject below 0.7” behave the way you expect.
- **For big or definitional taxonomies, use an LLM.** When a label needs a sentence of policy to define, NLI cannot represent it. Give the LLM the definitions, constrain the output with first-token scoring or a grammar, and accept the cost - which no longer scales with label count.
- **Test the escape hatch.** Whatever mechanism you choose for “none of these”, evaluate it explicitly with out-of-taxonomy documents. Every model here will confidently label a weather report as one of your four news topics, and only an explicit test surfaces that.
- **Related notebooks.** `00_Text_Classification` (the fine-tuned destination), `07_Feature_Extraction` (embedding-based shortlisting and few-shot probes), `08_Text_Generation` (prompting and constrained decoding), `03_Question_Answering` (NLI is also the groundedness checker for RAG), `Computer_Vision/11_Zero_Shot_Image_Classification` (the CLIP version of the same idea).

------------------------------------------------------------------------

Back to top
