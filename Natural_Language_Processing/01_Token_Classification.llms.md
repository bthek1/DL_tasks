# Token Classification

Everything to know about labelling every token in a sequence: NER and its relatives, the BIO scheme and the span-level F1 that goes with it, the mid-2026 landscape from BERT heads to zero-shot extractors, and runnable code that scores three models on CoNLL-2003.

Author

Benedict Thekkel

## 1. What is Token Classification?

Token classification assigns a label to **every token** in a sequence rather than one label to the whole text. The dominant instance is **named entity recognition** (NER): find the people, organisations, locations and dates in a document and say what each one is.

**Input.** A string, tokenised by the model’s own tokenizer. This creates the central complication of the task: your labels are attached to *words*, the model works on *subwords*. `Washington` may become `Wash` + `##ington`, and only the first subword carries the label - the rest are masked out with `-100` during training and dropped at inference.

**Output.** One label per token, encoded with a tagging scheme so that multi-token entities can be reconstructed:

| Scheme | Tags | Note |
|----|----|----|
| IOB2 (“BIO”) | `B-TYPE`, `I-TYPE`, `O` | The default. `B-` opens an entity, `I-` continues it |
| IOBES / BILOU | adds `E-`, `S-` | Marks ends and single-token entities; slightly better, more classes |
| Raw / no scheme | `TYPE`, `O` | Simpler, but two adjacent entities of the same type merge into one |

The scheme is not decoration: **spans are the unit that matters**, and the tags are only the encoding that lets a per-token classifier emit them. `New York` is one `LOC`, not two, and a system that tags it `B-LOC I-LOC` is right while one that tags it `B-LOC B-LOC` is wrong even though every token got the right type.

**Task family.** The same head, different label sets:

| Task | Labels | Typical use |
|----|----|----|
| NER | PER, ORG, LOC, MISC (or 18+ in OntoNotes) | Entity extraction, linking, redaction |
| Part-of-speech tagging | NOUN, VERB, ADJ, … | Linguistic preprocessing, grammar tools |
| Chunking / shallow parsing | NP, VP, PP | Phrase extraction |
| PII detection | NAME, EMAIL, SSN, … | Redaction, compliance |
| Slot filling | domain slots (departure_city) | Task-oriented dialogue |
| Keyphrase extraction | KEY, O | Indexing, SEO |

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| PII redaction | Healthcare, legal, any GDPR/HIPAA surface | Document -\> spans of names, addresses, IDs | Recall above all - one missed SSN is the incident; on-prem processing |
| Clinical concept extraction | Healthcare | Clinical note -\> drugs, dosages, conditions, negation | Domain vocabulary; negation and hedging (“no evidence of X”); auditability |
| Resume and job parsing | HR tech | CV -\> skills, employers, dates, titles | Layout noise from PDFs; entity linking to a taxonomy |
| Financial event extraction | Trading, research | Filings and news -\> tickers, amounts, dates | Latency (a news headline is worth money for seconds); precision |
| Search query understanding | E-commerce | “red nike running shoes size 10” -\> brand, colour, category, size | Sub-10 ms; very short, ungrammatical input |
| Legal contract review | Legal tech | Contract -\> parties, obligations, dates, governing law | Long documents; low tolerance for false negatives; explainability |
| Log and telemetry parsing | Ops, security | Log line -\> IPs, hostnames, user ids | Throughput; templates change with every release |
| Knowledge graph construction | Enterprise search | Corpus -\> entities, then relations | Consistency across documents; linking, not just tagging |

What the leaderboard number hides:

- **CoNLL-2003 is solved and unrepresentative.** State of the art passed 94 F1 years ago on well-formed 1996 Reuters newswire with four entity types. Real inputs are lowercase, misspelled, layout-mangled or code-switched, and the same model can drop 20-30 F1 on them.
- **Your entity types are not the benchmark’s.** Almost every deployment needs custom types (product SKUs, gene names, contract clauses), which means either annotation or a zero-shot extractor. This is the main reason GLiNER-style models exist.
- **Boundaries cause most errors.** Getting `LOC` right but tagging `New York City Council` as `LOC` instead of `ORG`, or including a trailing `Inc.` or not, dominates the error budget - which is exactly why span-level F1 is the metric and token accuracy is not.
- **Extraction is only step one.** Production systems need **entity linking** (this `Apple` is the company, id Q312) and often **coreference** (“the company”, “it”). Tagging is the easy half.

------------------------------------------------------------------------

## 3. How Modern Token Classification Works

1.  **Rules and gazetteers (pre-2003).** Regexes plus lists of known names. Still shipped for structured entities - a regex beats any neural model on an IBAN or a UK postcode, deterministically and for free.
2.  **CRFs over hand-built features (2003-2015).** Conditional random fields model the tag sequence jointly, so illegal transitions (`O` -\> `I-PER`) cost probability mass. Capitalisation, prefixes, gazetteer hits as features. Strong, interpretable, and still the fallback with tiny data.
3.  **BiLSTM-CRF (2015-2018).** Learned word + character embeddings through a bidirectional LSTM, CRF on top. The first architecture to beat feature engineering, and the origin of the “encoder plus CRF” pattern.
4.  **Pretrained encoder + linear head (2018-2024).** BERT and descendants with a per-token linear classifier. The CRF layer mostly disappeared - a strong encoder learns the transition constraints implicitly - though a CRF still buys a point on small datasets. This is what almost every deployed NER model still is.
5.  **Zero-shot span extractors (2023-2026).** GLiNER (2023) matches entity-*type* embeddings against span embeddings in a bidirectional encoder, so the types are a runtime argument: pass `["disease", "dosage", "manufacturer"]` and it extracts them without training. NuNER, UniversalNER and GLiNER v2 followed. The quality is below a fine-tuned model on that model’s own types, and far above nothing at all on new ones - a 200M-param model doing what previously needed annotation.
6.  **LLM extraction (2023-present).** Prompt a generative model for JSON. Handles arbitrary schemas, nested structure and reasoning-heavy fields, and is the only practical option for “extract the obligations and who owes them”. Costs orders of magnitude more per document, can hallucinate spans that are not in the text, and needs constrained decoding or post-hoc verification that every extracted string actually appears in the source.

**Where it stands (mid-2026).** Fixed types with annotated data: fine-tune a modern encoder (ModernBERT, DeBERTa-v3, XLM-R for multilingual) - still the accuracy and latency leader. New types with no data: GLiNER-family zero-shot, optionally distilled into a fine-tune once you have labels. Complex or nested schemas: an LLM, with span verification.

------------------------------------------------------------------------

## 4. Evaluation Metrics

**Span-level (entity-level) F1** is the metric. An entity counts as correct only if **both** the type and the exact boundaries match; partial overlaps score zero. This is what CoNLL scored in 2003, what `seqeval` computes, and what every paper reports.

$$P = \frac{\text{correct spans}}{\text{predicted spans}} \qquad R = \frac{\text{correct spans}}{\text{gold spans}} \qquad F_1 = \frac{2PR}{P+R}$$

**Why not token accuracy.** In CoNLL-2003 about 83% of tokens are `O`. A model that predicts `O` everywhere scores 83% token accuracy and 0.0 entity F1. Token accuracy cannot be compared across datasets, and it rewards exactly the failure mode you care about least.

**Pitfalls:**

- **Micro vs macro.** Micro F1 (pool all spans, the CoNLL default) is dominated by frequent types; macro F1 (average per-type F1) exposes the rare type that is failing. Report both.
- **Partial credit is a different metric.** Relaxed schemes (MUC-5, SemEval-2013 `type` / `partial` modes) give credit for right-type-wrong-boundary. They are legitimate and they are *not* comparable to strict F1 - always say which one you used.
- **Tokenisation must match.** Comparing a model’s character offsets against gold word-level tags requires an explicit alignment. Getting this wrong (off-by-one on whitespace) silently costs several F1 points and looks like a bad model.
- **Nested and overlapping entities** break BIO entirely (`Bank of [China]` is `ORG` containing `LOC`). Those need a span-based or multi-head model, and the flat F1 above does not apply.

The cell below implements BIO decoding plus strict span F1 in about 40 lines - the same numbers `seqeval` would give, without the dependency.

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


def bio_to_spans(tags):
    "Decode a BIO tag sequence into a set of (type, start, end_exclusive) spans.\n\n    Tolerates the two malformed cases real models produce: an `I-X` that opens a span\n    with no preceding `B-X` (treated as a `B-X`, the conll2003 convention), and an\n    `I-Y` following a `B-X` of a different type (closes the old span, opens a new one).\n    "
    spans, start, kind = set(), None, None
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, _, typ = tag.partition("-")
        if kind is not None and (prefix in ("B", "O") or typ != kind):
            spans.add((kind, start, i))
            start, kind = None, None
        if prefix == "B" or (prefix == "I" and kind is None):
            start, kind = i, typ
    return spans


def span_f1(gold_spans, pred_spans):
    "Strict micro precision/recall/F1 over sets of (type, start, end) spans."
    tp = sum(len(g & p) for g, p in zip(gold_spans, pred_spans))
    n_pred = sum(len(p) for p in pred_spans)
    n_gold = sum(len(g) for g in gold_spans)
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gold if n_gold else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "pred": n_pred, "gold": n_gold}


def span_f1_by_type(gold_spans, pred_spans):
    "Per-entity-type strict F1 - where the aggregate hides the type that is failing."
    types = {t for spans in list(gold_spans) + list(pred_spans) for t, _, _ in spans}
    return {
        t: span_f1(
            [{s for s in g if s[0] == t} for g in gold_spans],
            [{s for s in p if s[0] == t} for p in pred_spans],
        )
        for t in sorted(types)
    }


# Toy example: the boundary error that token accuracy cannot see.
tokens = ["Steve", "Jobs", "founded", "Apple", "in", "Cupertino"]
gold = ["B-PER", "I-PER", "O", "B-ORG", "O", "B-LOC"]
pred = ["B-PER", "B-PER", "O", "B-ORG", "O", "B-LOC"]   # splits the person in two

tok_acc = sum(g == p for g, p in zip(gold, pred)) / len(gold)
show_table([{"token": t, "gold tag": g, "predicted tag": p, "same": g == p}
            for t, g, p in zip(tokens, gold, pred)],
           title="One boundary error: the person is split into two spans")
show_kv({"gold spans": str(sorted(bio_to_spans(gold))),
         "predicted spans": str(sorted(bio_to_spans(pred))),
         "token accuracy": round(tok_acc, 3),
         **{k: round(v, 3) for k, v in
            span_f1([bio_to_spans(gold)], [bio_to_spans(pred)]).items()}},
        title="Token accuracy barely notices; strict span F1 charges for it twice")
```

```
         One boundary error: the person is split into two spans         
                                                                        
 token              gold tag          predicted tag            same     
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 Steve              B-PER             B-PER                    True     
 Jobs               I-PER             B-PER                    False    
 founded            O                 O                        True     
 Apple              B-ORG             B-ORG                    True     
 in                 O                 O                        True     
 Cupertino          B-LOC             B-LOC                    True     
                                                                        
```

```
       Token accuracy barely notices; strict span F1 charges for it twice       
                                                                                
 gold spans                       [('LOC', 5, 6), ('ORG', 3, 4), ('PER', 0, 2)] 
 predicted spans   [('LOC', 5, 6), ('ORG', 3, 4), ('PER', 0, 1), ('PER', 1, 2)] 
 token accuracy                                                          0.8330 
 precision                                                               0.5000 
 recall                                                                  0.6670 
 f1                                                                      0.5710 
 tp                                                                           2 
 pred                                                                         4 
 gold                                                                         3 
                                                                                
```

## 5. Datasets

| Dataset | Contents | Size | Scope | License | Typical use |
|----|----|----|----|----|----|
| [CoNLL-2003](https://huggingface.co/datasets/eriktks/conll2003) | Reuters newswire, 4 types (PER/ORG/LOC/MISC) | 14k / 3.3k / 3.5k sentences | en (+ de) | research only, Reuters terms | The reference NER benchmark; used below |
| [OntoNotes 5.0](https://huggingface.co/datasets/tner/ontonotes5) | News, broadcast, web, telephone; 18 types | 76k sentences | en, zh, ar | LDC (license required) | The serious English benchmark |
| [WikiANN / PAN-X](https://huggingface.co/datasets/unimelb-nlp/wikiann) | Wikipedia-derived, silver labels, 3 types | 20k-40k per lang | 176 languages | ODC-BY | Multilingual and cross-lingual transfer |
| [Few-NERD](https://huggingface.co/datasets/DFKI-SLT/few-nerd) | Wikipedia, 8 coarse / 66 fine types | 188k sentences | en | CC BY-SA 4.0 | Fine-grained and few-shot NER |
| [MultiNERD](https://huggingface.co/datasets/Babelscape/multinerd) | Wikipedia + WikiNews, 15 types | 2.6M sentences | 10 languages | CC BY-NC-SA | Large multilingual training |
| [Universal Dependencies](https://universaldependencies.org/) | Treebanks with UPOS tags | 200+ treebanks | 100+ langs | mixed | POS tagging and parsing |
| [BC5CDR / NCBI-disease](https://huggingface.co/datasets/tner/bc5cdr) | PubMed abstracts, chemicals and diseases | 1.5k abstracts | en | public domain | Biomedical NER |
| [PII masking datasets](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) | Synthetic PII in context | 200k+ | multi | varies | Redaction models |

This notebook evaluates on the **CoNLL-2003 validation split**. Note the loading detail below: the Hub copy is a legacy loading *script*, which `datasets` 4.x no longer executes, so the notebook reads the auto-converted parquet branch directly. The dataset is research-licensed newswire from 1996 - fine for comparing models, useless for predicting behaviour on your own text.

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

CoNLL-2003 has no live leaderboard worth linking; [Papers with Code NER](https://paperswithcode.com/task/named-entity-recognition-ner) tracks the academic numbers and the [GLiNER collection](https://huggingface.co/collections/urchade/gliner-6533b6f8ae8b7c14b3d9f527) tracks the zero-shot side.

| Model | Params | License | Types | Approach | Best for |
|----|----|----|----|----|----|
| [bert-base-NER](https://huggingface.co/dslim/bert-base-NER) | 108M | MIT | 4 (CoNLL) | fine-tuned encoder | the cheap default; used below |
| [roberta-large-ner-english](https://huggingface.co/Jean-Baptiste/roberta-large-ner-english) | 355M | MIT | 4 | fine-tuned encoder | best strict F1 of the three below |
| [wikineural-multilingual-ner](https://huggingface.co/Babelscape/wikineural-multilingual-ner) | 177M | CC BY-NC-SA | 4 | fine-tuned mBERT | 9 languages in one model |
| [xlm-roberta-large-finetuned-conll03](https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll03-english) | 560M | MIT | 4 | fine-tuned encoder | strongest multilingual encoder base |
| [GLiNER multi-v2.1](https://huggingface.co/urchade/gliner_multi-v2.1) | 209M | Apache 2.0 | **arbitrary** | zero-shot span matching | custom types with no training data |
| [NuNER Zero](https://huggingface.co/numind/NuNER_Zero) | 449M | MIT | arbitrary | zero-shot | zero-shot with longer entity spans |
| [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) | 149M | Apache 2.0 | yours | backbone to fine-tune | 8192-token context, fast |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 1.7B | Apache 2.0 | arbitrary | generative JSON | nested or reasoning-heavy schemas |
| [stanza / spaCy](https://spacy.io/) | 20-500M | MIT | 18 (OntoNotes) | pipeline | CPU-only production, plus tokenisation and linking |

**How to choose.** Standard types, English, latency-sensitive: `bert-base-NER` or a ModernBERT fine-tune. Multilingual: XLM-R or wikineural. Custom types, no annotation budget: GLiNER (it needs the `gliner` package, so it is named here rather than run - see Going Further). Nested or relational output: an LLM.

------------------------------------------------------------------------

## 7. Setup

All three benchmarked models load through the `transformers` `token-classification` pipeline. Package roles:

- `transformers` (\>=5.13) + `torch` - the three NER models
- `accelerate` - device placement
- `datasets` - CoNLL-2003 validation
- `pandas` + `pyecharts` - the benchmark table and chart
- `rich` - the result tables. It renders to HTML inside Jupyter, so the tables survive into the published docs; `show_table` / `show_kv` / `rule` are defined in the first code cell of section 4.

`seqeval` is not used; the span decoder and strict F1 in section 4 do the same job in 40 readable lines.

**Two mechanics worth understanding before the results make sense:**

1.  **`aggregation_strategy`.** The pipeline emits one prediction per *subword*. `aggregation_strategy="simple"` groups them back into entity spans with character offsets - which is what you almost always want. `"first"` (label from the word’s first subword), `"average"` and `"max"` differ only in how they resolve disagreement inside a word. With `aggregation_strategy=None` you get raw subword tags and must reassemble them yourself.
2.  **The alignment problem.** Gold labels are per *word*; the pipeline returns *character* offsets. The evaluation below joins CoNLL tokens with single spaces, records where each token starts, and maps predicted character spans back to token indices. Joining with spaces is itself a small distortion - real CoNLL text has `Germany 's` - which costs every model the same amount and is worth knowing about when comparing to published numbers.

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
    "Collect garbage and hand freed VRAM back to the CUDA allocator.\n\n    Call right after `del`-ing a model you are done with: `del model; free_memory()`.\n    "
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    # glibc keeps freed CPU allocations in its arenas instead of returning them to the
    # OS, so RSS compounds across sections. malloc_trim(0) hands the arenas back.
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

# The Hub copy of CoNLL-2003 is a legacy loading *script*, which datasets 4.x refuses to
# execute. The Hub's auto-converted parquet branch holds the same data - read it directly.
conll = load_dataset(
    "parquet",
    data_files={
        "validation": "hf://datasets/eriktks/conll2003@refs%2Fconvert%2Fparquet/conll2003/validation/0000.parquet"
    },
    split="validation",
    cache_dir=HF_CACHE,
)

# ner_tags are ints into this list (the CoNLL-2003 ClassLabel order).
TAGS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

N = 200  # sentences to evaluate (3250 in the full split)
rows = [r for r in conll.select(range(N * 2)) if len(r["tokens"]) > 3][:N]
sentences = [r["tokens"] for r in rows]
gold_spans = [bio_to_spans([TAGS[t] for t in r["ner_tags"]]) for r in rows]

print(conll)
print(f"\n{len(sentences)} sentences, {sum(len(g) for g in gold_spans)} gold entities")
for r in rows[:2]:
    print("\n ", " ".join(r["tokens"])[:110])
    print("  ", sorted(bio_to_spans([TAGS[t] for t in r["ner_tags"]])))
```

    Dataset({
        features: ['id', 'tokens', 'pos_tags', 'chunk_tags', 'ner_tags'],
        num_rows: 3250
    })

    200 sentences, 408 gold entities

      CRICKET - LEICESTERSHIRE TAKE OVER AT TOP AFTER INNINGS VICTORY .
       [('ORG', 2, 3)]

      West Indian all-rounder Phil Simmons took four for 38 on Friday as Leicestershire beat Somerset by an innings 
       [('MISC', 0, 2), ('ORG', 12, 13), ('ORG', 14, 15), ('PER', 3, 5)]

``` python
def char_starts(tokens):
    "Character offset of each token in ' '.join(tokens)."
    offsets, pos = [], 0
    for tok in tokens:
        offsets.append(pos)
        pos += len(tok) + 1
    return offsets


def pipeline_spans(tokens, entities):
    "Map a pipeline's character-offset entities back onto (type, start_tok, end_tok).\n\n    The pipeline works on the joined string, the gold labels are per word - this is the\n    alignment step, and getting it wrong silently costs several F1 points.\n    "
    starts = char_starts(tokens)
    ends = [s + len(t) for s, t in zip(starts, tokens)]
    spans = set()
    for ent in entities:
        first = next((i for i, e in enumerate(ends) if e > ent["start"]), None)
        last = next((i for i in range(len(starts) - 1, -1, -1) if starts[i] < ent["end"]), None)
        if first is not None and last is not None and last >= first:
            spans.add((ent["entity_group"], first, last + 1))
    return spans


def evaluate_ner(pipe, sentences, gold_spans, batch_size=16):
    "Run a token-classification pipeline over the sentences and score strict span F1."
    texts = [" ".join(toks) for toks in sentences]
    t0 = time.perf_counter()
    outputs = pipe(texts, batch_size=batch_size)
    elapsed = time.perf_counter() - t0
    preds = [pipeline_spans(toks, ents) for toks, ents in zip(sentences, outputs)]
    return span_f1(gold_spans, preds), elapsed, preds
```

## 8. The default: bert-base-NER

BERT-base fine-tuned on CoNLL-2003 itself - 108M params, four types, and the model most “add NER to this” tickets get closed with. Published test F1 is around 91; it is playing at home on this validation split.

Look at the entity output rather than the score: each result carries `entity_group`, a confidence, the matched string and character offsets. Those offsets are the useful part - they let you redact, highlight or link back into the original document without re-tokenising.

------------------------------------------------------------------------

``` python
from transformers import pipeline

ner = pipeline(
    "token-classification",
    model="dslim/bert-base-NER",
    aggregation_strategy="simple",   # regroup subwords into whole entity spans
    device=device,
    model_kwargs={"cache_dir": HF_CACHE},
)
rule("dslim/bert-base-NER")
print("labels:", ner.model.config.id2label)

demo = "Satya Nadella said Microsoft will open an office in Sydney next March."
show_table([{"type": e["entity_group"], "score": round(e["score"], 3),
             "text": demo[e["start"]:e["end"]],
             "chars": f"[{e['start']}:{e['end']}]"} for e in ner(demo)],
           title="Entities with character offsets - what makes the output actionable",
           best=("score",))

scores, elapsed, preds = evaluate_ner(ner, sentences, gold_spans)
show_kv({"sentences": len(sentences), "seconds": round(elapsed, 1),
         "sentences / second": round(len(sentences) / elapsed, 1),
         **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in scores.items()}},
        title=f"CoNLL-2003 validation, strict span F1 ({len(sentences)} sentences)")
show_table([{"type": typ, "precision": round(s["precision"], 3),
             "recall": round(s["recall"], 3), "F1": round(s["f1"], 3), "gold": s["gold"]}
            for typ, s in span_f1_by_type(gold_spans, preds).items()],
           title="Per entity type", best=("F1",),
           caption="the aggregate hides which type is failing - and it is nearly always MISC")

del ner
free_memory()
vram("after bert-base-NER")
```

    [transformers] BertForTokenClassification LOAD REPORT from: dslim/bert-base-NER
    Key                      | Status     |  | 
    -------------------------+------------+--+-
    bert.pooler.dense.bias   | UNEXPECTED |  | 
    bert.pooler.dense.weight | UNEXPECTED |  | 

    Notes:
    - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

```
dslim/bert-base-NER ────────────────────────────────────────────────────────────────────────────────────────────
```

    labels: {0: 'O', 1: 'B-MISC', 2: 'I-MISC', 3: 'B-PER', 4: 'I-PER', 5: 'B-ORG', 6: 'I-ORG', 7: 'B-LOC', 8: 'I-LOC'}

```
   Entities with character offsets - what makes the output actionable   
                                                                        
 type         score          text                       chars           
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 PER          1.0            Sa                         [0:2]           
 PER          0.893          tya Nadella                [2:13]          
 ORG          0.998          Microsoft                  [19:28]         
 LOC          0.999          Sydney                     [52:58]         
                                                                        
```

```
     CoNLL-2003 validation, strict span F1 (200 sentences)      
                                                                
 sentences                                                  200 
 seconds                                                 0.4000 
 sentences / second                                      457.00 
 precision                                               0.8422 
 recall                                                  0.9681 
 f1                                                      0.9008 
 tp                                                         395 
 pred                                                       469 
 gold                                                       408 
                                                                
```

```
                            Per entity type                             
                                                                        
 type                precision          recall             F1      gold 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 LOC                    0.9830          1.0000         0.9920       117 
 MISC                   0.7580          0.8930         0.8200        28 
 ORG                    0.9490          0.9490         0.9490       137 
 PER                    0.6830          0.9760         0.8040       126 
                                                                        
the aggregate hides which type is failing - and it is nearly always MISC
```

    VRAM after bert-base-NER     0.01 GB allocated /  0.02 GB reserved

## 9. Scaling up: roberta-large-ner-english

Same task, 3x the parameters (355M), and a different label convention: this checkpoint predicts bare types (`PER`, `ORG`, `LOC`, `MISC`) with no `B-`/`I-` prefixes. The pipeline’s aggregation handles it, but the consequence is real - **two adjacent entities of the same type merge into one span**, because nothing marks where the first ends. On newswire that is rare; on a comma-separated list of names it is not.

Expect roughly +2-3 strict F1 over BERT-base for roughly 3x the compute, which is the usual shape of the size/accuracy curve on a saturated benchmark.

------------------------------------------------------------------------

``` python
ner_large = pipeline(
    "token-classification",
    model="Jean-Baptiste/roberta-large-ner-english",
    aggregation_strategy="simple",
    device=device,
    torch_dtype=dtype,
    model_kwargs={"cache_dir": HF_CACHE},
)
print("labels:", ner_large.model.config.id2label, "  <- no B-/I- prefixes")

# The merge failure mode, made visible.
tricky = "Angela Merkel Emmanuel Macron and Justin Trudeau met in Ottawa."
for ent in ner_large(tricky):
    print(f"  {ent['entity_group']:5s} {tricky[ent['start']:ent['end']]!r}")

scores, elapsed, preds = evaluate_ner(ner_large, sentences, gold_spans)
show_kv({"sentences": len(sentences), "seconds": round(elapsed, 1),
         "sentences / second": round(len(sentences) / elapsed, 1),
         **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in scores.items()}},
        title="Jean-Baptiste/roberta-large-ner-english")

del ner_large
free_memory()
vram("after roberta-large")
```

    [transformers] `torch_dtype` is deprecated! Use `dtype` instead!

    labels: {0: 'O', 1: 'PER', 2: 'ORG', 3: 'LOC', 4: 'MISC'}   <- no B-/I- prefixes
      PER   'Angela Merkel Emmanuel Macron'
      PER   'Justin Trudeau'
      LOC   'Ottawa'

```
            Jean-Baptiste/roberta-large-ner-english             
                                                                
 sentences                                                  200 
 seconds                                                 0.4000 
 sentences / second                                      540.40 
 precision                                               0.9707 
 recall                                                  0.9755 
 f1                                                      0.9731 
 tp                                                         398 
 pred                                                       410 
 gold                                                       408 
                                                                
```

    VRAM after roberta-large     0.01 GB allocated /  0.02 GB reserved

## 10. Multilingual: wikineural

Multilingual BERT fine-tuned on WikiNEuRal, a silver-standard corpus built automatically from Wikipedia. One model covers English, German, Spanish, French, Italian, Dutch, Polish, Portuguese and Russian - which is the whole point, because maintaining nine per-language models is nine times the work.

Two honest caveats. It is trained on **Wikipedia-style** text, so it is strong on encyclopaedic prose and weaker on newswire than a CoNLL-trained model - which is what the benchmark below will show. And it is **CC BY-NC-SA**: non-commercial, unlike the MIT-licensed models above. Check the licence before it reaches a product.

------------------------------------------------------------------------

``` python
ner_multi = pipeline(
    "token-classification",
    model="Babelscape/wikineural-multilingual-ner",
    aggregation_strategy="simple",
    device=device,
    model_kwargs={"cache_dir": HF_CACHE},
)

for text in [
    "Angela Merkel besuchte die Universitaet Heidelberg in Baden-Wuerttemberg.",
    "Le siege social de Renault se trouve a Boulogne-Billancourt.",
]:
    print(text)
    for ent in ner_multi(text):
        print(f"   {ent['entity_group']:5s} {ent['score']:.3f}  {text[ent['start']:ent['end']]!r}")

scores, elapsed, preds = evaluate_ner(ner_multi, sentences, gold_spans)
show_kv({"sentences": len(sentences), "seconds": round(elapsed, 1),
         "sentences / second": round(len(sentences) / elapsed, 1),
         **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in scores.items()}},
        title="Babelscape/wikineural-multilingual-ner (scored on English)")

del ner_multi
free_memory()
vram("after wikineural")
```

    Angela Merkel besuchte die Universitaet Heidelberg in Baden-Wuerttemberg.
       PER   1.000  'Angela Merkel'
       ORG   0.876  'Universitaet Heidelberg'
       LOC   0.999  'Baden-Wuerttemberg'
    Le siege social de Renault se trouve a Boulogne-Billancourt.
       ORG   0.807  'Renault'
       LOC   0.999  'Boulogne-Billancourt'

```
   Babelscape/wikineural-multilingual-ner (scored on English)   
                                                                
 sentences                                                  200 
 seconds                                                 0.3000 
 sentences / second                                      636.00 
 precision                                               0.6326 
 recall                                                  0.6667 
 f1                                                      0.6492 
 tp                                                         272 
 pred                                                       430 
 gold                                                       408 
                                                                
```

    VRAM after wikineural        0.01 GB allocated /  0.02 GB reserved

## 11. Head-to-head Benchmark

Same 200 CoNLL-2003 validation sentences, same whitespace joining, same strict span F1, one model live at a time.

The comparison to keep in mind: two of these models were fine-tuned on CoNLL-2003’s own training set and one was not. That is not a flaw in the benchmark, it *is* the benchmark - it puts a number on how much a domain- and label-matched fine-tune is worth, which is the decision you actually face when choosing between an off-the-shelf model and annotating your own data.

------------------------------------------------------------------------

``` python
import pandas as pd

MODELS = [
    ("bert-base-NER", "dslim/bert-base-NER", 108),
    ("roberta-large-ner", "Jean-Baptiste/roberta-large-ner-english", 355),
    ("wikineural-multi", "Babelscape/wikineural-multilingual-ner", 177),
]

results = []
for name, model_id, params_m in MODELS:
    pipe = pipeline(
        "token-classification", model=model_id, aggregation_strategy="simple",
        device=device, model_kwargs={"cache_dir": HF_CACHE},
    )
    scores, elapsed, preds = evaluate_ner(pipe, sentences, gold_spans)
    by_type = span_f1_by_type(gold_spans, preds)
    results.append({
        "model": name,
        "params_m": params_m,
        "precision": round(scores["precision"], 4),
        "recall": round(scores["recall"], 4),
        "f1": round(scores["f1"], 4),
        "sents_per_sec": round(len(sentences) / elapsed, 1),
        **{f"f1_{t}": round(s["f1"], 3) for t, s in by_type.items() if t in ("PER", "ORG", "LOC", "MISC")},
    })
    show_kv(results[-1], title=name)
    del pipe  # free each model before loading the next so VRAM stays flat
    free_memory()

vram("after benchmark")
df = pd.DataFrame(results).sort_values("f1", ascending=False)
show_table(
    df.to_dict("records"),
    title=f"CoNLL-2003 validation, strict span F1, {len(sentences)} sentences",
    best=("precision", "recall", "f1", "sents_per_sec",
          "f1_PER", "f1_ORG", "f1_LOC", "f1_MISC"),
    caption="best per column in green - `df` is still a DataFrame if you want to export it",
)
```

    [transformers] BertForTokenClassification LOAD REPORT from: dslim/bert-base-NER
    Key                      | Status     |  | 
    -------------------------+------------+--+-
    bert.pooler.dense.bias   | UNEXPECTED |  | 
    bert.pooler.dense.weight | UNEXPECTED |  | 

    Notes:
    - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

```
                         bert-base-NER                          
                                                                
 model                                            bert-base-NER 
 params_m                                                   108 
 precision                                               0.8422 
 recall                                                  0.9681 
 f1                                                      0.9008 
 sents_per_sec                                           604.00 
 f1_LOC                                                  0.9920 
 f1_MISC                                                 0.8200 
 f1_ORG                                                  0.9490 
 f1_PER                                                  0.8040 
                                                                
```

```
                       roberta-large-ner                        
                                                                
 model                                        roberta-large-ner 
 params_m                                                   355 
 precision                                               0.9707 
 recall                                                  0.9755 
 f1                                                      0.9731 
 sents_per_sec                                           251.40 
 f1_LOC                                                  0.9910 
 f1_MISC                                                 0.9090 
 f1_ORG                                                  0.9740 
 f1_PER                                                  0.9690 
                                                                
```

```
                        wikineural-multi                        
                                                                
 model                                         wikineural-multi 
 params_m                                                   177 
 precision                                               0.6326 
 recall                                                  0.6667 
 f1                                                      0.6492 
 sents_per_sec                                           638.60 
 f1_LOC                                                  0.7120 
 f1_MISC                                                 0.3090 
 f1_ORG                                                  0.4060 
 f1_PER                                                  0.8800 
                                                                
```

    VRAM after benchmark         0.01 GB allocated /  0.02 GB reserved

```
                              CoNLL-2003 validation, strict span F1, 200 sentences                              
                                                                                                                
 model              params_m   precision   recall       f1   sents_per_sec   f1_LOC   f1_MISC   f1_ORG   f1_PER 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 roberta-large-ne        355      0.9707   0.9755   0.9731          251.40   0.9910    0.9090   0.9740   0.9690 
 r                                                                                                              
 bert-base-NER           108      0.8422   0.9681   0.9008          604.00   0.9920    0.8200   0.9490   0.8040 
 wikineural-multi        177      0.6326   0.6667   0.6492          638.60   0.7120    0.3090   0.4060   0.8800 
                                                                                                                
                 best per column in green - `df` is still a DataFrame if you want to export it                  
```

``` python
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org serves an expired TLS certificate from most nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"

from pyecharts import options as opts
from pyecharts.charts import Bar

types = ["PER", "ORG", "LOC", "MISC"]
bar = Bar().add_xaxis([r["model"] for r in results])
bar.add_yaxis("micro F1 x100", [round(r["f1"] * 100, 1) for r in results])
for t in types:
    key = f"f1_{t}"
    if all(key in r for r in results):
        bar.add_yaxis(f"{t} F1 x100", [round(r[key] * 100, 1) for r in results])
bar.set_global_opts(
    title_opts=opts.TitleOpts(
        title="CoNLL-2003 validation, strict span F1 (200 sentences)",
        subtitle="RTX 3060 - two of the three were fine-tuned on this dataset",
    ),
    yaxis_opts=opts.AxisOpts(name="F1 x100", min_=0, max_=100),
    xaxis_opts=opts.AxisOpts(name="model", axislabel_opts=opts.LabelOpts(rotate=15)),
    tooltip_opts=opts.TooltipOpts(trigger="axis"),
    legend_opts=opts.LegendOpts(pos_top="8%"),
)
bar.render_notebook()
```

``` python
from pyecharts.charts import Scatter

# Strict F1 against throughput - the deployment trade-off, on a saturated benchmark.
scatter = Scatter()
scatter.add_xaxis([r["sents_per_sec"] for r in results])
for r in results:
    scatter.add_yaxis(
        r["model"], [[r["sents_per_sec"], round(r["f1"] * 100, 1)]],
        symbol_size=18, label_opts=opts.LabelOpts(is_show=False),
    )
scatter.set_global_opts(
    title_opts=opts.TitleOpts(title="Span F1 vs throughput"),
    xaxis_opts=opts.AxisOpts(name="sentences / second", type_="value"),
    yaxis_opts=opts.AxisOpts(name="strict F1 x100", type_="value"),
    tooltip_opts=opts.TooltipOpts(trigger="item"),
)
scatter.render_notebook()
```

## 12. Interactive: tag and redact your own text

Paste your own text below and get both the entity list and a redacted version - the PII use case from section 2 in its simplest form. This is the cell people run on its own, so it opens with a `require(...)` guard naming what it needs from Setup rather than dying on a bare `NameError`.

Worth trying: all-lowercase text (case is a huge feature for NER - most models lose 10+ F1 without it), names the model has never seen, and text where an entity is ambiguous between types (`Washington` as person, state or city). The redaction loop also shows why character offsets matter: rebuilding the string from tokens would lose the original spacing and punctuation.

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


require("device", "HF_CACHE", "free_memory", "vram")

from transformers import pipeline

MY_TEXT = (
    "Dr. Priya Raman joined Atlassian in Sydney last year after six years at "
    "Google DeepMind in London. She can be reached at the Melbourne office."
)
REDACT = {"PER", "LOC"}   # entity types to mask
MIN_SCORE = 0.5           # drop low-confidence spans before redacting

# Re-runnable: this cell frees the pipeline at the end, so guard the load or a second
# shift-enter raises NameError.
if "my_ner" not in globals():
    my_ner = pipeline(
        "token-classification",
        model="dslim/bert-base-NER",
        aggregation_strategy="simple",
        device=device,
        model_kwargs={"cache_dir": HF_CACHE},
    )

ents = [e for e in my_ner(MY_TEXT) if e["score"] >= MIN_SCORE]
show_table([{"type": e["entity_group"], "score": round(e["score"], 3),
             "text": MY_TEXT[e["start"]:e["end"]],
             "redacted": e["entity_group"] in REDACT} for e in ents],
           title=f"Entities above score {MIN_SCORE}", best=("score",),
           caption="anything the model misses is silently NOT redacted - which is why "
                   "recall, not precision, is the metric for a redaction pipeline")

# Redact back-to-front so earlier character offsets stay valid as the string shrinks.
redacted = MY_TEXT
for e in sorted(ents, key=lambda e: -e["start"]):
    if e["entity_group"] in REDACT:
        redacted = redacted[:e["start"]] + f"[{e['entity_group']}]" + redacted[e["end"]:]
print("\n" + redacted)

del my_ner
free_memory()
vram("final")
```

    [transformers] BertForTokenClassification LOAD REPORT from: dslim/bert-base-NER
    Key                      | Status     |  | 
    -------------------------+------------+--+-
    bert.pooler.dense.bias   | UNEXPECTED |  | 
    bert.pooler.dense.weight | UNEXPECTED |  | 

    Notes:
    - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

```
                        Entities above score 0.5                        
                                                                        
 type       score         text                           redacted       
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 PER        0.999         P                              True           
 PER        0.997         riya Raman                     True           
 ORG        0.999         Atlassian                      False          
 LOC        1.0           Sydney                         True           
 ORG        0.998         Google DeepMind                False          
 LOC        0.999         London                         True           
 LOC        0.999         Melbourne                      True           
                                                                        
   anything the model misses is silently NOT redacted - which is why    
     recall, not precision, is the metric for a redaction pipeline      
```

    Dr. [PER][PER] joined Atlassian in [LOC] last year after six years at Google DeepMind in [LOC]. She can be reached at the [LOC] office.
    VRAM final                   0.01 GB allocated /  0.02 GB reserved

## 13. Common Frameworks

Token classification is the oldest production NLP task, and it has the deepest non-transformer ecosystem of anything in this folder: spaCy pipelines, rule engines and gazetteers still do real work, often alongside a neural tagger rather than under it. The other thing that shapes this table is what the tags are usually *for* - a large share of NER in production is PII detection, which has purpose-built tooling that handles the redaction as well as the tagging.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [transformers](https://github.com/huggingface/transformers) | modelling | BERT/RoBERTa/ModernBERT taggers behind `AutoModelForTokenClassification`, with `DataCollatorForTokenClassification` and the aggregation strategies of section 8 | Apache 2.0 | Default. Get `word_ids()` label alignment right and the rest is standard |
| [spaCy](https://github.com/explosion/spaCy) | modelling | A full pipeline - tokeniser, tagger, NER, rules, `EntityRuler` gazetteers - as one serialisable object | MIT | Production systems where rules and a model must coexist. Half of real NER accuracy comes from a gazetteer nobody publishes |
| [GLiNER](https://github.com/urchade/GLiNER) | modelling | Entity types passed as a runtime list, no training - an encoder-sized zero-shot tagger | Apache 2.0 | New entity types with no labels. Best used offline to bootstrap a training set, then distilled into a fine-tuned encoder |
| [Presidio](https://github.com/microsoft/presidio) | modelling | PII detection and *de-identification*: recognisers, checksum validators, and the redaction/anonymisation step | MIT | The tags are personal data. Tagging is the easy half; consistent pseudonymisation is the half that gets audited |
| [Argilla](https://github.com/argilla-io/argilla) / [Label Studio](https://github.com/HumanSignal/label-studio) / [doccano](https://github.com/doccano/doccano) | data | Span annotation with model pre-labelling, and inter-annotator agreement on boundaries | Apache 2.0 / MIT | Always. Span boundaries are where annotators disagree, and disagreement caps your F1 |
| [tokenizers](https://github.com/huggingface/tokenizers) offsets + `stride` windows | data | Character offsets back to the source text, and sliding windows with overlap for long documents | Apache 2.0 | Always. An entity you cannot map back to a character span in the original document is not usable downstream |
| [optimum](https://github.com/huggingface/optimum) + [ONNX Runtime](https://onnxruntime.ai/) | inference runtime | A quantised tagger running over a document stream on CPU at high throughput | Apache 2.0 / MIT | Deployment. NER usually runs over everything, so per-document cost matters more than peak accuracy |
| [vLLM](https://github.com/vllm-project/vllm) + [outlines](https://github.com/dottxt-ai/outlines) | inference runtime | Schema-constrained LLM extraction for nested or relational structures a BIO tagger cannot express | Apache 2.0 | The schema has nesting or relations. Verify every extracted string occurs in the source - that check removes most hallucinated spans |
| [seqeval](https://github.com/chakki-works/seqeval) + [evaluate](https://github.com/huggingface/evaluate) | evaluation | Entity-level precision/recall/F1 with correct BIO handling, not token-level accuracy | MIT / Apache 2.0 | Always. Token accuracy is inflated by the `O` class and will tell you a useless model is 97% correct |

**The 2026 default stack** is a fine-tuned ModernBERT tagger through `transformers`, a spaCy pipeline around it when rules and gazetteers are in play, Presidio if the entities are PII, Argilla for the annotation loop, and ONNX for serving. GLiNER to bootstrap types you have no labels for.

The common wrong turn is evaluating with token-level accuracy, which the `O` class dominates and which says nothing about whether entities were found. The second is treating extraction as finished at the tag: `Apple` as `ORG` is not the same as knowing *which* Apple, and entity linking against a real knowledge base is usually the step that makes the output actionable.

------------------------------------------------------------------------

## 14. Going Further

- **Fine-tune on your own types.** `AutoModelForTokenClassification.from_pretrained(base, num_labels=len(tags))` plus `Trainer`. The one thing to get right is label alignment: tokenize with `is_split_into_words=True`, call `encoding.word_ids()`, and set the label to `-100` on every subword after a word’s first so the loss ignores it. `DataCollatorForTokenClassification` handles the padding. A few thousand annotated sentences is usually enough for 85+ F1 on a narrow domain.
- **Zero-shot for new types.** GLiNER takes the entity types as a runtime list - `model.predict_entities(text, ["drug", "dosage", "adverse effect"])`. It needs the `gliner` package rather than plain `transformers`, so it is not run here; install it separately if you want it, or use it offline to bootstrap labels and then distil into a fine-tuned encoder.
- **LLM extraction with verification.** For nested or relational schemas, prompt for JSON and then **check every extracted string actually occurs in the source text** before you trust it. That one check removes most hallucinated spans. Constrained decoding (a JSON-schema grammar) removes the rest of the parse failures.
- **Add a CRF layer** if your dataset is small (under ~2k sentences) or your tags have hard structural constraints. It buys a point or two of F1 and eliminates illegal transitions outright.
- **Long documents.** Sliding windows with `stride` in the tokenizer (`return_overflowing_tokens=True`), then de-duplicate entities in the overlap. ModernBERT’s 8192-token context removes the need for most documents.
- **Entity linking is the next step.** Tagging `Apple` as `ORG` is not the same as knowing which Apple. Look at BLINK, ReFinED, or a simple embedding lookup against your own entity catalogue.
- **Rules still win on structured entities.** Emails, IBANs, phone numbers, order ids: use a regex, and reserve the model for the entities that genuinely need context.
- **Related notebooks.** `00_Text_Classification` (one label for the whole text), `03_Question_Answering` (extractive spans selected by a question rather than a type), `08_Text_Generation` (LLM extraction), `07_Feature_Extraction` (the encoders these heads sit on).

------------------------------------------------------------------------

Back to top
