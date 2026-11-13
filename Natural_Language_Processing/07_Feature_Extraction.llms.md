# Feature Extraction

Turning text into vectors: why raw BERT embeddings are bad and contrastive training fixes them, what pooling and normalisation actually do, how Matryoshka lets you cut a vector in half for free, and runnable code that probes four embedding models on the same task.

Author

Benedict Thekkel

## 1. What is Feature Extraction?

Feature extraction turns text into a fixed-size dense vector - an **embedding** - that can be compared, indexed, clustered or fed to a downstream model. It is the task that quietly underpins search, RAG, recommendation, deduplication and clustering, and unlike most tasks in this folder its output is never shown to a user.

**Input.** A string. Often with a prefix or instruction, because most modern embedding models are asymmetric (see below).

**Output.** A vector, typically 384-4096 dimensions, usually L2-normalised so that a dot product is a cosine similarity.

**Where the vector comes from: pooling.** A transformer produces one vector *per token*. Collapsing those into one vector per text is the pooling step, and the choice is model-specific, not a preference:

| Pooling | How | Used by |
|----|----|----|
| **CLS** | take the `[CLS]` token’s vector | BGE, GTE, most BERT-based embedders |
| **Mean** | average the token vectors, masking padding | MiniLM / sentence-transformers, E5 |
| **Last token** | take the final token’s vector | decoder-based embedders (Qwen3-Embedding) |
| **Max** | element-wise max | rare, mostly historical |

Using the wrong pooling for a checkpoint does not error - it silently produces bad vectors. This is the most common bug in hand-rolled embedding code, and it is why the cell in section 8 reads the pooling from the model card rather than defaulting to one.

**The thing that makes embedding models work is not the architecture, it is the training.** Pretrained BERT’s `[CLS]` vector is close to useless for similarity: the representation space is **anisotropic**, squeezed into a narrow cone where almost every pair of sentences has cosine similarity around 0.9. Contrastive training fixes this by pulling paired texts together and pushing unrelated ones apart, which spreads the space out. Section 8 measures the difference directly, and it is large enough to be startling.

**Symmetric versus asymmetric.** Comparing two sentences of the same kind (“is A similar to B?”) is symmetric. Matching a short query to a long document is asymmetric, and the two roles need different treatment - which is why E5 wants `"query: "` and `"passage: "` prefixes, BGE wants an instruction on queries only, and Qwen3-Embedding takes a task description. Dropping those prefixes typically costs several points of retrieval quality, silently.

**Neighbouring tasks:**

| Task | How it differs | Notebook |
|----|----|----|
| Sentence similarity | Uses these vectors to score text pairs | `10_Sentence_Similarity` |
| Text ranking | Uses them to retrieve and order documents | `11_Text_Ranking` |
| Text classification | A trained head instead of a frozen vector + probe | `00_Text_Classification` |
| Zero-shot classification | Embedding-vs-label is the cheap alternative to NLI | `04_Zero_Shot_Classification` |
| Image feature extraction | The same idea for images | `Computer_Vision/16_Image_Feature_Extraction` |

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| RAG retrieval | Every LLM application | Chunks -\> vectors -\> nearest neighbours per query | Recall@k; embedding and index must be rebuilt on model change |
| Semantic search | E-commerce, docs, support | Query + catalogue -\> ranked results | Latency at query time; index size in RAM |
| Deduplication and near-duplicate detection | Content platforms, data pipelines | Documents -\> clusters above a threshold | Precision; threshold calibration per corpus |
| Recommendation | Media, retail | Item text -\> vectors -\> similar items | Cold-start items with no interaction data |
| Clustering and topic discovery | Analytics, research | Corpus -\> clusters -\> labels | Interpretability; cluster count is not known in advance |
| Few-shot classification | Any ML team pre-labelling | Embeddings + a linear probe on 50 examples | Beats fine-tuning when data is tiny; used below |
| Anomaly and drift detection | Ops, moderation | Stream -\> distance from a reference distribution | Stability of the embedding space over time |
| Cross-lingual matching | Global orgs | Text in any language -\> shared space | Alignment quality across language pairs |

What the MTEB score hides:

- **Changing the embedding model means re-embedding everything.** Vectors from two models are not comparable, so a model upgrade is a full re-index: compute cost, storage churn, and a migration window where both indexes must exist. Teams underestimate this and then stay on a mediocre model for years.
- **Storage is a real budget.** Ten million chunks at 1024 float32 dimensions is 40 GB. The same vectors at 256 dimensions in int8 are 2.5 GB. This is what Matryoshka embeddings and quantization are for, and section 10 measures what accuracy that trade actually costs.
- **Chunking dominates retrieval quality.** How you split documents matters more than which top-10 embedding model you pick. A chunk that splits a sentence from its subject is unretrievable no matter how good the encoder is.
- **MTEB is contaminated and over-fitted.** It is the standard board and it is also a target: several models are trained on data closely resembling its tasks. Treat the leaderboard as a shortlist generator, then evaluate the top few on **your** queries and documents. A 200-query hand-labelled set from your own corpus is worth more than the whole leaderboard.
- **Domain shift is severe and quiet.** General embedders underperform badly on code, legal citations, medical abbreviations and product SKUs, and there is no error - just worse neighbours.

------------------------------------------------------------------------

## 3. How Modern Feature Extraction Works

1.  **Count vectors (1970s-2000s).** TF-IDF and LSA. Sparse, high-dimensional, no notion of synonymy. BM25 is the surviving descendant and remains a strong retrieval baseline - see `11_Text_Ranking`.
2.  **Static word embeddings (2013-2017).** word2vec, GloVe, fastText. One vector per word type, so `bank` had a single meaning. Sentence vectors came from averaging, which discards word order entirely and still works embarrassingly well as a baseline.
3.  **Contextual embeddings, used naively (2018-2019).** ELMo and BERT gave one vector per token *in context*. People pooled them for similarity and found the results poor - the anisotropy problem. This is the era’s key negative result: **pretraining alone does not produce a useful similarity space.**
4.  **Sentence-BERT and contrastive fine-tuning (2019-2021).** SBERT trained BERT with a siamese objective on NLI pairs, making cosine similarity meaningful and dropping the cost of comparing 10,000 sentences from 65 hours (cross-encoder) to 5 seconds. SimCSE then showed that *unsupervised* contrastive learning - the same sentence twice through the model with different dropout masks as a positive pair - almost matched supervised training. The lesson stuck: the objective, not the data, was the missing piece.
5.  **Large-scale contrastive pretraining (2022-2023).** E5, GTE, BGE and the OpenAI embedding models scaled the recipe: hundreds of millions of weakly-supervised pairs (title-body, question-answer, citation pairs) for stage one, then curated pairs with **hard negatives** for stage two. In-batch negatives with very large batches made the contrastive signal strong. MTEB (2022) gave the field a common scoreboard across 58 datasets and 8 task types.
6.  **Instruction-aware and LLM-based embedders (2023-2026).** Two changes. First, embeddings conditioned on a task instruction, so one model serves retrieval, clustering and classification with different prefixes. Second, decoder LLMs converted into embedders (E5-Mistral, GTE-Qwen, Qwen3-Embedding, NV-Embed) by last-token pooling plus contrastive fine-tuning, often with synthetic training pairs generated by a larger LLM. These top MTEB, at 1.5B-7B params and 1024-4096 dimensions.
7.  **Efficiency work (2023-2026).** **Matryoshka Representation Learning** trains so that the first 64/128/256/512 dimensions are each independently useful, letting you truncate a vector at serving time instead of running a smaller model. **Binary and int8 quantization** cuts storage 4-32x for a few points of recall, usually recovered by re-ranking the top-100 with full-precision vectors. **ColBERT-style late interaction** keeps one vector per token for much better quality at much higher storage cost.

**Where it stands (mid-2026).** The 100-300M encoder models (BGE, GTE, E5, and the ModernBERT-based ones) are the practical default: strong, fast, 384-768 dimensions, and cheap to run over millions of chunks. The 0.6-7B LLM-based embedders lead the benchmarks and cost 10-50x more per document to index - worth it when retrieval quality is the product, not when it is a feature. Matryoshka truncation and int8 quantization are close to free wins and should be the default for any index over a few million vectors.

------------------------------------------------------------------------

## 4. Evaluation Metrics

Embeddings are never evaluated directly - there is no “correct” vector. They are evaluated by **how well they serve a downstream task**, which is why MTEB spans eight task types rather than one.

**Retrieval** (the metric that matters for RAG): nDCG@10, Recall@k, MRR. Covered in `11_Text_Ranking`.

**Semantic similarity**: Spearman correlation between cosine similarity and human ratings on STS. Covered in `10_Sentence_Similarity`.

**Classification (linear probe)** - the metric used in this notebook. Freeze the embeddings, train a logistic regression on top, report accuracy. It answers the question “how much task-relevant information is linearly available in this vector?”, which is exactly what you want to know before using embeddings as features. It is also the cheapest realistic few-shot baseline in existence: 100 labelled examples and a 30-line probe frequently beat a fine-tune.

**Clustering**: V-measure over k-means clusters against gold labels.

**Two diagnostics that are not benchmark metrics but tell you more about a model than any of them:**

- **Alignment** - mean distance between paired (similar) texts. Lower is better.
- **Uniformity** - how evenly the embeddings spread over the hypersphere. An anisotropic model scores badly here, and this is the number that exposes raw BERT.

A blunter version of the same idea, used below: the gap between **mean intra-class cosine similarity** and **mean inter-class cosine similarity**. A model whose same-label texts are barely closer than its different-label texts is not encoding the distinction, whatever its probe accuracy says after enough training.

**Pitfalls:**

- **Normalise, then compare.** Cosine similarity requires unit vectors. If you skip normalisation and use a dot product, vector *magnitude* leaks into the score, and magnitude often tracks text length.
- **Absolute cosine values are not comparable across models.** One model’s “similar” is 0.85, another’s is 0.45. Thresholds must be recalibrated per model, always.
- **Use the model’s own prefixes.** E5 without `"query: "`/`"passage: "`, or BGE without its query instruction, loses several points. The prefix is part of the model, not a suggestion.
- **A probe measures linear separability, not the embedding’s ceiling.** A stronger head can extract more. Compare probes to probes.

The cell below implements the linear probe (pure torch, ~25 lines) and the separation diagnostic used throughout.

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


import torch


def linear_probe(train_x, train_y, test_x, test_y, n_classes, epochs=300, lr=0.5, wd=1e-4):
    "Multinomial logistic regression on frozen embeddings. Returns test accuracy."
    # Clone the features into ordinary tensors first. `embed` below runs under
    # torch.inference_mode(), and an inference tensor may never take part in an autograd
    # graph - not even as an input saved for backward - so `train_x @ w` would raise
    # "Inference tensors cannot be saved for backward". `.float()` does NOT launder one:
    # on an already-float32 tensor it is a no-op that hands back the same tensor.
    train_x = train_x.detach().clone().float()
    test_x = test_x.detach().clone().float()
    w = torch.zeros(train_x.shape[1], n_classes, requires_grad=True, device=train_x.device)
    b = torch.zeros(n_classes, requires_grad=True, device=train_x.device)
    opt = torch.optim.LBFGS([w, b], lr=lr, max_iter=epochs, history_size=10)
    loss_fn = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = loss_fn(train_x @ w + b, train_y) + wd * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.inference_mode():
        pred = (test_x @ w + b).argmax(-1)
    return (pred == test_y).float().mean().item()


def separation(emb, labels):
    """Mean intra-class minus mean inter-class cosine similarity.

    Assumes `emb` is L2-normalised, so emb @ emb.T is the cosine similarity matrix.
    A near-zero gap means the space does not encode the distinction at all.
    """
    sim = emb.float() @ emb.float().T
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    intra = sim[same & ~eye].mean().item()
    inter = sim[~same].mean().item()
    return {"intra": round(intra, 3), "inter": round(inter, 3),
            "gap": round(intra - inter, 3)}


# Toy example: 3 classes in 8 dimensions, one well separated and one collapsed into a
# narrow cone (the anisotropy failure mode that raw BERT embeddings actually exhibit).
torch.manual_seed(0)
labels = torch.arange(3).repeat_interleave(40)
centres = torch.randn(3, 8)

good = torch.nn.functional.normalize(centres[labels] + 0.35 * torch.randn(120, 8), dim=-1)
cone = torch.nn.functional.normalize(
    torch.ones(120, 8) + 0.08 * centres[labels] + 0.05 * torch.randn(120, 8), dim=-1)

show_table([{"embedding space": name, **separation(emb, labels),
             "probe_acc": round(linear_probe(emb[::2], labels[::2],
                                             emb[1::2], labels[1::2], 3), 3)}
            for name, emb in [("well separated", good), ("anisotropic cone", cone)]],
           title="Two synthetic spaces, both linearly separable",
           caption="only one has a usable cosine similarity. Probe accuracy and "
                   "similarity quality are different questions - measure both")
```

```
             Two synthetic spaces, both linearly separable              
                                                                        
 embedding space             intra       inter        gap     probe_acc 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 well separated             0.8740      0.2330     0.6410        1.0000 
 anisotropic cone           0.9980      0.9940     0.0040        0.9830 
                                                                        
 only one has a usable cosine similarity. Probe accuracy and similarity 
             quality are different questions - measure both             
```

## 5. Datasets

Embedding models are trained on **pairs**, and evaluated on downstream tasks. Both kinds are listed here because the training data explains far more about a model’s behaviour than its architecture does.

| Dataset | Role | Contents | Size | Scope | License |
|----|----|----|----|----|----|
| [MTEB](https://huggingface.co/spaces/mteb/leaderboard) | evaluation | 58+ datasets, 8 task types, 112 languages | \- | multi | mixed |
| [BEIR](https://huggingface.co/datasets/BeIR/beir) | evaluation | 18 zero-shot retrieval datasets | varies | en | mixed |
| [STS Benchmark](https://huggingface.co/datasets/sentence-transformers/stsb) | evaluation | Sentence pairs with human 0-5 similarity | 8.6k | en | mixed |
| [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) | evaluation | Tweets, 6 emotions | 20k | en | educational |
| [MS MARCO](https://huggingface.co/datasets/microsoft/ms_marco) | training | Bing queries + relevant passages | 8.8M passages | en | non-commercial |
| [Natural Questions](https://huggingface.co/datasets/sentence-transformers/natural-questions) | training | Question + Wikipedia passage pairs | 100k | en | CC BY-SA 3.0 |
| [SNLI + MultiNLI](https://huggingface.co/datasets/sentence-transformers/all-nli) | training | Entailment triplets (anchor, positive, negative) | 940k | en | mixed |
| [S2ORC / citation pairs](https://huggingface.co/datasets/sentence-transformers/s2orc) | training | Scientific title-abstract and citation pairs | 100M+ | en | CC BY-NC |
| [Reddit / StackExchange pairs](https://huggingface.co/datasets/sentence-transformers/reddit-title-body) | training | Title-body and question-answer pairs | 700M+ | en | mixed |
| [Embedding training mixes](https://huggingface.co/collections/sentence-transformers/embedding-model-datasets-6644d7a3673a511914aa7552) | training | Curated collection of the above, ready to use | billions | multi | mixed |

This notebook evaluates with a **linear probe on `dair-ai/emotion`** - 6 emotion classes over short tweets. It is a good probe target for three reasons: the classes are semantically close (joy/love, anger/fear), so it discriminates between models rather than saturating; the texts are short, so embedding is fast; and it is nothing like the retrieval data these models were trained on, so it measures transfer rather than memorisation.

**A note on what this measures.** A probe on 2,000 examples is a *few-shot* evaluation. A fine-tuned ModernBERT with the full 16k training set reaches ~93% on this dataset. The probe numbers below will be lower, and the useful comparison is between the models, plus the observation of how close a frozen 22M-parameter encoder plus a logistic regression gets to a full fine-tune.

Downloads land in `DL_tasks/datasets/` via `cache_dir` (gitignored).

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

The reference board is [MTEB](https://huggingface.co/spaces/mteb/leaderboard) - read the task-type tabs, not the average, because a model tuned for retrieval is not necessarily good for clustering.

| Model | Params | Dims | License | Context | Pooling | Best for |
|----|----|----|----|----|----|----|
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 22M | 384 | Apache 2.0 | 256 | mean | the cheap default, CPU-friendly; used below |
| [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | 109M | 768 | MIT | 512 | CLS | strong English baseline; used below |
| [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) | 335M | 1024 | MIT | 512 | CLS | more accuracy at the same interface |
| [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) | 109M | 768 | MIT | 512 | mean | requires query:/passage: prefixes |
| [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | 149M | 768 | Apache 2.0 | 8192 | CLS | long chunks, modern backbone; used below |
| [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) | 137M | 768 (MRL) | Apache 2.0 | 8192 | mean | open data + weights, Matryoshka |
| [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 596M | 1024 (MRL) | Apache 2.0 | 32k | last token | instruction-aware, multilingual; used below |
| [Qwen3-Embedding-4B / 8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | 4-8B | 2560-4096 | Apache 2.0 | 32k | last token | top of MTEB (needs more VRAM than this box) |
| [bge-m3](https://huggingface.co/BAAI/bge-m3) | 568M | 1024 | MIT | 8192 | CLS | multilingual + dense/sparse/ColBERT in one |
| [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) | 560M | 1024 | MIT | 512 | mean | 100 languages, well tested |
| [jina-colbert-v2](https://huggingface.co/jinaai/jina-colbert-v2) | 560M | per-token | CC BY-NC | 8192 | late interaction | best quality, ~100x the storage |

**How to choose.** Start with a 100-300M encoder at 768 dimensions - `bge-base-en-v1.5` or `gte-modernbert-base` - and only move up if a measurement on your own data justifies it. Take `gte-modernbert-base` when chunks exceed 512 tokens; its 8192-token window removes a chunking constraint. Take an LLM-based embedder when retrieval quality *is* the product and you can afford 10-50x the indexing cost. Take `all-MiniLM-L6-v2` when you are embedding tens of millions of items on CPU - it is 15 years of progress in 22M parameters and 384 dimensions.

**The three questions that decide it in practice** are rarely accuracy: how many vectors will you store (dimensions x count x bytes), how long are your chunks (context window), and how often will you re-index (model stability). Ask those first.

------------------------------------------------------------------------

## 7. Setup

Everything loads through Hugging Face `transformers` with `AutoModel` - deliberately **not** `sentence-transformers`. The library would collapse loading, pooling and normalisation into one line, and those three lines are the content of this notebook. `sentence-transformers` is the right choice in production; here it would hide the mechanism.

Package roles:

- `transformers` + `torch` - the four encoders, pooling and normalisation
- `accelerate` - `device_map` placement
- `datasets` - the emotion split used for the probe
- `pandas` + `pyecharts` - the benchmark table and charts
- `rich` - the result tables. It renders to HTML inside Jupyter, so the tables survive into the published docs; `show_table` / `show_kv` / `rule` are defined in the first code cell of section 4.

Three details that decide correctness:

- **Pooling is per-checkpoint, not per-preference.** BGE and GTE use CLS, MiniLM and E5 use mean, Qwen3-Embedding uses the last token. The model card states it; guessing produces plausible vectors with degraded quality and no error.
- **Mean pooling must respect the attention mask.** Averaging over padding tokens is a real and common bug: it makes a text’s embedding depend on the longest text in its batch, so the same input embeds differently in different batches.
- **Normalise before comparing.** `torch.nn.functional.normalize(v, dim=-1)` makes the dot product a cosine. Everything downstream - thresholds, index choices, quantization - assumes unit vectors.

------------------------------------------------------------------------

``` python
# Everything runs through Hugging Face transformers - no sentence-transformers, so the
# pooling and normalisation stay visible.
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

# dair-ai/emotion: short tweets, 6 emotion classes. Close semantic classes (joy/love,
# anger/fear) make it discriminate between models instead of saturating.
emotion = load_dataset("dair-ai/emotion", "split", cache_dir=HF_CACHE)
CLASSES = emotion["train"].features["label"].names

N_TRAIN, N_TEST = 2000, 800
train_ds = emotion["train"].shuffle(seed=0).select(range(N_TRAIN))
test_ds = emotion["test"].shuffle(seed=0).select(range(N_TEST))

train_texts = [r["text"] for r in train_ds]
test_texts = [r["text"] for r in test_ds]
train_y = torch.tensor([r["label"] for r in train_ds], device=device)
test_y = torch.tensor([r["label"] for r in test_ds], device=device)

print(emotion)
print(f"\nclasses: {CLASSES}")
print(f"probe: {N_TRAIN} train / {N_TEST} test embeddings (a few-shot setting)\n")
for r in train_ds.select(range(4)):
    print(f"  [{CLASSES[r['label']]:8s}] {r['text'][:90]}")
```

    DatasetDict({
        train: Dataset({
            features: ['text', 'label'],
            num_rows: 16000
        })
        validation: Dataset({
            features: ['text', 'label'],
            num_rows: 2000
        })
        test: Dataset({
            features: ['text', 'label'],
            num_rows: 2000
        })
    })

    classes: ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    probe: 2000 train / 800 test embeddings (a few-shot setting)

      [joy     ] i may feel that way but the fact that stories created by adults that are meant for childre
      [joy     ] i feel some people shouldn t answer if they are not considerate and serious
      [love    ] i began feeling amorous towards everyone on stage towards the people around me as i experi
      [sadness ] im feeling defeated or doubtful

## 8. Pooling, normalisation, and why raw BERT is not an embedding model

Before comparing trained embedders, it is worth seeing what the training buys, because the answer is “almost everything”.

The cell below runs **`bert-base-uncased`** - a strong pretrained encoder that was never contrastively trained - and pools it three ways. Then it runs `all-MiniLM-L6-v2`, which is the *same size class* with the same architecture family, trained contrastively on a billion sentence pairs.

Watch the `inter` column, which is the mean cosine similarity between texts of **different** classes. Raw BERT puts almost every pair of sentences at very high similarity regardless of content - the **anisotropy** problem. The vectors are not meaningless (the probe still extracts a signal from them, because a linear classifier does not care about the scale of the cone) but cosine similarity over them is nearly useless, and cosine similarity is what every vector database computes.

This is the single most important fact in the notebook: **an embedding model is defined by its contrastive training objective, not by its encoder.** “Just use BERT and take the CLS token” is the standard first attempt and it does not work.

------------------------------------------------------------------------

``` python
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def pool(hidden, mask, how):
    "Collapse per-token vectors into one vector per text. Mask-aware for mean pooling."
    if how == "cls":
        return hidden[:, 0]
    if how == "mean":                      # never average over padding - see Setup
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
    if how == "last":                      # decoder embedders: last non-pad token
        idx = mask.sum(1) - 1
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), idx]
    raise ValueError(f"unknown pooling: {how}")


@torch.inference_mode()
def embed(model, tok, texts, how="mean", batch_size=64, max_length=256,
          prefix="", normalize=True):
    "Encode texts to one vector each. Returns a (len(texts), dim) tensor."
    out = []
    for i in range(0, len(texts), batch_size):
        batch = [prefix + t for t in texts[i:i + batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length).to(model.device)
        hidden = model(**enc).last_hidden_state
        v = pool(hidden, enc["attention_mask"], how)
        out.append(F.normalize(v.float(), dim=-1) if normalize else v.float())
    return torch.cat(out)


# Raw pretrained BERT, pooled three ways - no contrastive training anywhere.
bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased", cache_dir=HF_CACHE)
bert = AutoModel.from_pretrained("bert-base-uncased", cache_dir=HF_CACHE).to(device).eval()
vram("bert-base loaded")

pooling_rows = []
for how in ("cls", "mean"):
    tr = embed(bert, bert_tok, train_texts, how=how)
    te = embed(bert, bert_tok, test_texts, how=how)
    pooling_rows.append({"model / pooling": f"bert-base ({how} pool)",
                         **separation(te, test_y),
                         "probe_acc": round(linear_probe(tr, train_y, te, test_y,
                                                         len(CLASSES)), 3)})

del bert, bert_tok, tr, te
free_memory()

# Same architecture family, same size class, trained contrastively on ~1B pairs.
mini_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", cache_dir=HF_CACHE)
mini = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2",
                                 cache_dir=HF_CACHE).to(device).eval()
tr = embed(mini, mini_tok, train_texts, how="mean")
te = embed(mini, mini_tok, test_texts, how="mean")
pooling_rows.append({"model / pooling": "all-MiniLM-L6-v2 (mean pool)",
                     **separation(te, test_y),
                     "probe_acc": round(linear_probe(tr, train_y, te, test_y,
                                                     len(CLASSES)), 3)})
show_table(pooling_rows, title="Same architecture family, one trained contrastively",
           best=("gap", "probe_acc"), lower_is_better=("inter",),
           caption="raw BERT's inter-class similarity is near its intra-class similarity - "
                   "almost every pair looks similar. That cone is what contrastive "
                   "training unfolds, and the probe barely notices while cosine "
                   "similarity (what a vector DB uses) notices enormously")

# Mean pooling without the attention mask - the silent bug from Setup, measured.
enc = mini_tok(test_texts[:64], return_tensors="pt", padding=True, truncation=True,
               max_length=256).to(device)
with torch.inference_mode():
    hidden = mini(**enc).last_hidden_state
naive = F.normalize(hidden.mean(1).float(), dim=-1)      # averages padding too
correct = F.normalize(pool(hidden, enc["attention_mask"], "mean").float(), dim=-1)
print(f"\nmask-aware vs naive mean pooling: mean cosine agreement "
      f"{(naive * correct).sum(-1).mean().item():.4f}")
print("Not 1.0 - and the error depends on the longest text in the batch, so the same")
print("input embeds differently in different batches. Silent, and it breaks caching.")

del mini, mini_tok, hidden, enc, naive, correct, tr, te
free_memory()
vram("after pooling demo")
```

    [transformers] BertModel LOAD REPORT from: bert-base-uncased
    Key                                        | Status     |  | 
    -------------------------------------------+------------+--+-
    cls.seq_relationship.bias                  | UNEXPECTED |  | 
    cls.predictions.bias                       | UNEXPECTED |  | 
    cls.predictions.transform.LayerNorm.bias   | UNEXPECTED |  | 
    cls.predictions.transform.LayerNorm.weight | UNEXPECTED |  | 
    cls.predictions.transform.dense.weight     | UNEXPECTED |  | 
    cls.predictions.transform.dense.bias       | UNEXPECTED |  | 
    cls.seq_relationship.weight                | UNEXPECTED |  | 

    Notes:
    - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

    VRAM bert-base loaded        0.44 GB allocated /  0.49 GB reserved

```
          Same architecture family, one trained contrastively           
                                                                        
 model / pooling                    intra    inter      gap   probe_acc 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 bert-base (cls pool)              0.8260   0.8220   0.0040      0.5320 
 bert-base (mean pool)             0.6500   0.6420   0.0080      0.5820 
 all-MiniLM-L6-v2 (mean pool)      0.1930   0.1770   0.0160      0.6260 
                                                                        
 raw BERT's inter-class similarity is near its intra-class similarity - 
almost every pair looks similar. That cone is what contrastive training 
 unfolds, and the probe barely notices while cosine similarity (what a  
                   vector DB uses) notices enormously                   
```

    mask-aware vs naive mean pooling: mean cosine agreement 0.6564
    Not 1.0 - and the error depends on the longest text in the batch, so the same
    input embeds differently in different batches. Silent, and it breaks caching.
    VRAM after pooling demo      0.02 GB allocated /  0.04 GB reserved

## 9. Four embedding models, same probe

Now the comparison proper: four models spanning 22M to 596M parameters and 384 to 1024 dimensions, each with its **own** pooling and its **own** prefix convention, evaluated with the identical probe on the identical texts.

The prefixes are not decoration:

- **BGE** wants an instruction on *queries* only (`"Represent this sentence for searching relevant passages: "`), and nothing on documents. For a symmetric task like this probe, no prefix is correct.
- **E5** wants `"query: "` or `"passage: "` on everything - the model was trained with them always present, and omitting them measurably degrades output.
- **Qwen3-Embedding** takes a natural-language task instruction on queries, and uses `last`-token pooling because it is a decoder.
- **MiniLM and GTE** take no prefix.

Expect the differences to be smaller than the parameter counts suggest. A well-trained 100M encoder is close to a 600M one on a task like this; the 600M model earns its keep on hard retrieval, long context and multilingual input, none of which this probe tests. That is a useful calibration: **do not pay for embedding size without measuring on your own task.**

One model is loaded at a time and freed before the next, so VRAM stays flat.

------------------------------------------------------------------------

``` python
MODELS = [
    # name, checkpoint, pooling, doc prefix, params (M), download (GB)
    ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", "mean", "", 22, 0.09),
    ("bge-base-en-v1.5", "BAAI/bge-base-en-v1.5", "cls", "", 109, 0.44),
    ("gte-modernbert-base", "Alibaba-NLP/gte-modernbert-base", "cls", "", 149, 0.60),
    ("qwen3-embedding-0.6b", "Qwen/Qwen3-Embedding-0.6B", "last", "", 596, 1.20),
]

embeddings = {}     # keep the test embeddings for the Matryoshka / quantization section
results = []

for name, model_id, how, prefix, params_m, gb in MODELS:
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE)
    model = AutoModel.from_pretrained(
        model_id, dtype=dtype, cache_dir=HF_CACHE).to(device).eval()

    t0 = time.perf_counter()
    tr = embed(model, tok, train_texts, how=how, prefix=prefix)
    te = embed(model, tok, test_texts, how=how, prefix=prefix)
    secs = time.perf_counter() - t0

    s = separation(te, test_y)
    acc = linear_probe(tr, train_y, te, test_y, len(CLASSES))
    dim = te.shape[1]
    results.append({
        "model": name, "params_m": params_m, "dims": dim, "pooling": how,
        "probe_acc": round(acc, 4), "intra": s["intra"], "inter": s["inter"],
        "gap": s["gap"],
        "texts_per_sec": round((N_TRAIN + N_TEST) / secs, 1),
        "index_gb_per_10m": round(dim * 4 * 1e7 / 1e9, 2),   # float32 storage
    })
    embeddings[name] = te.cpu()      # off-GPU so it survives free_memory()
    show_kv(results[-1], title=name)

    del model, tok, tr, te          # one model live at a time
    free_memory()

vram("after all embedders")
```

```
                        all-MiniLM-L6-v2                        
                                                                
 model                                         all-MiniLM-L6-v2 
 params_m                                                    22 
 dims                                                       384 
 pooling                                                   mean 
 probe_acc                                               0.6262 
 intra                                                   0.1930 
 inter                                                   0.1770 
 gap                                                     0.0160 
 texts_per_sec                                         6,411.20 
 index_gb_per_10m                                         15.36 
                                                                
```

```
                        bge-base-en-v1.5                        
                                                                
 model                                         bge-base-en-v1.5 
 params_m                                                   109 
 dims                                                       768 
 pooling                                                    cls 
 probe_acc                                               0.6862 
 intra                                                   0.6060 
 inter                                                   0.5840 
 gap                                                     0.0220 
 texts_per_sec                                         1,941.40 
 index_gb_per_10m                                         30.72 
                                                                
```

```
                      gte-modernbert-base                       
                                                                
 model                                      gte-modernbert-base 
 params_m                                                   149 
 dims                                                       768 
 pooling                                                    cls 
 probe_acc                                               0.6737 
 intra                                                   0.5710 
 inter                                                   0.5480 
 gap                                                     0.0230 
 texts_per_sec                                           929.10 
 index_gb_per_10m                                         30.72 
                                                                
```

```
                      qwen3-embedding-0.6b                      
                                                                
 model                                     qwen3-embedding-0.6b 
 params_m                                                   596 
 dims                                                     1,024 
 pooling                                                   last 
 probe_acc                                               0.6725 
 intra                                                   0.4750 
 inter                                                   0.4590 
 gap                                                     0.0170 
 texts_per_sec                                           269.20 
 index_gb_per_10m                                         40.96 
                                                                
```

    VRAM after all embedders     0.02 GB allocated /  0.04 GB reserved

## 10. Matryoshka truncation and quantization

Two ways to make an index smaller, both nearly free, and both under-used.

**Matryoshka Representation Learning (MRL)** trains the model so that the *first* k dimensions of a vector are independently a good embedding - the information is packed front-loaded rather than spread evenly. So you can slice `v[:256]`, renormalise, and get most of the quality at a quarter of the storage. `Qwen3-Embedding` and `nomic-embed-text-v1.5` are trained this way.

**On a non-MRL model, truncation still works better than it has any right to** - the leading dimensions of most embedders carry more variance - but the degradation is steeper and unprincipled. The cell below runs the sweep on every model so the difference between “trained for this” and “getting away with it” is visible.

**Quantization** is the other axis. Storing int8 instead of float32 is a 4x reduction; binary (1 bit per dimension, compared with Hamming distance) is 32x and is genuinely usable when followed by a rescoring pass over the top-100 with full-precision vectors.

The arithmetic that makes this matter: 10 million chunks at 1024 float32 dimensions is **40 GB** of RAM for the index. At 256 dimensions in int8 it is **2.5 GB** - the difference between a machine and a laptop, for a few points of recall you can often win back with reranking (`11_Text_Ranking`).

------------------------------------------------------------------------

``` python
DIMS = [64, 128, 256, 512, 768, 1024]

mrl_curves = {}
for name, te in embeddings.items():
    full_dim = te.shape[1]
    accs = []
    for d in DIMS:
        if d > full_dim:
            accs.append(None)
            continue
        # Slice, then RE-normalise - a truncated unit vector is no longer unit length.
        sub = F.normalize(te[:, :d].float(), dim=-1).to(device)
        # Probe train/test split reuses the same texts, so re-embed is not needed: the
        # probe is trained on the first half of the test embeddings for this sweep.
        acc = linear_probe(sub[::2], test_y[::2], sub[1::2], test_y[1::2], len(CLASSES))
        accs.append(round(acc, 4))
    mrl_curves[name] = accs

show_table([{"model": name, **{f"{d}d": a for d, a in zip(DIMS, accs) if a is not None}}
            for name, accs in mrl_curves.items()],
           title="Probe accuracy after truncating the vector and renormalising",
           caption="qwen3-embedding is MRL-trained and degrades gracefully; the others "
                   "are getting away with it")

# Quantization: what int8 and binary actually cost, measured as similarity agreement.
ref_name = "bge-base-en-v1.5"
ref = embeddings[ref_name].to(device).float()
full_sim = ref @ ref.T

int8 = torch.round(ref.clamp(-1, 1) * 127).to(torch.int8)
int8_sim = F.normalize(int8.float(), dim=-1) @ F.normalize(int8.float(), dim=-1).T
binary = (ref > 0).float()
bin_sim = F.normalize(binary, dim=-1) @ F.normalize(binary, dim=-1).T


def top_k_overlap(a, b, k=10):
    "Fraction of each row's top-k neighbours preserved - what retrieval actually cares about."
    ta = a.topk(k + 1, dim=-1).indices[:, 1:]
    tb = b.topk(k + 1, dim=-1).indices[:, 1:]
    return sum(len(set(x.tolist()) & set(y.tolist())) for x, y in zip(ta, tb)) / (len(ta) * k)


dim = ref.shape[1]
quant_rows = [
    {"precision": label, "bytes_per_dim": bytes_per_dim,
     "gb_per_10m_vectors": round(dim * bytes_per_dim * 1e7 / 1e9, 2),
     "top10_overlap_vs_fp32": round(top_k_overlap(full_sim, sim), 3)}
    for label, sim, bytes_per_dim in [("float32", full_sim, 4), ("int8", int8_sim, 1),
                                      ("binary", bin_sim, 1 / 8)]
]
show_table(quant_rows, title=f"{ref_name} ({dim} dims), 10M-vector index",
           best=("top10_overlap_vs_fp32", "gb_per_10m_vectors"),
           lower_is_better=("gb_per_10m_vectors",),
           caption="binary keeps most neighbours at 1/32 the storage - rescore the top-100 "
                   "with float vectors and the recall loss largely disappears (see nb 11)")

del ref, full_sim, int8, int8_sim, binary, bin_sim
free_memory()
```

```
        Probe accuracy after truncating the vector and renormalising        
                                                                            
 model                     64d     128d     256d     512d     768d    1024d 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 all-MiniLM-L6-v2       0.4550   0.5000   0.5325                            
 bge-base-en-v1.5       0.5525   0.5750   0.6000   0.6475   0.6450          
 gte-modernbert-base    0.5450   0.6025   0.6050   0.6025   0.6100          
 qwen3-embedding-0.6b   0.5375   0.5625   0.5975   0.6200   0.6375   0.6375 
                                                                            
   qwen3-embedding is MRL-trained and degrades gracefully; the others are   
                            getting away with it                            
```

```
             bge-base-en-v1.5 (768 dims), 10M-vector index              
                                                                        
 precision   bytes_per_dim   gb_per_10m_vectors   top10_overlap_vs_fp32 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 float32                 4                30.72                  1.0000 
 int8                    1               7.6800                  0.9510 
 binary             0.1250               0.9600                  0.5600 
                                                                        
 binary keeps most neighbours at 1/32 the storage - rescore the top-100 
 with float vectors and the recall loss largely disappears (see nb 11)  
```

``` python
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org serves an expired TLS certificate from most nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"

from pyecharts import options as opts
from pyecharts.charts import Bar

# Storage against neighbour preservation. The bars are on different scales deliberately:
# the storage drop is multiplicative and the quality drop is not.
bar = (
    Bar()
    .add_xaxis([r["precision"] for r in quant_rows])
    .add_yaxis("GB per 10M vectors", [r["gb_per_10m_vectors"] for r in quant_rows])
    .add_yaxis("top-10 neighbours kept x100",
               [round(r["top10_overlap_vs_fp32"] * 100, 1) for r in quant_rows])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"Quantization: storage vs neighbour preservation ({ref_name})",
            subtitle="int8 is 4x smaller and binary 32x - the recall you lose is "
                     "recoverable by rescoring the top-100",
        ),
        yaxis_opts=opts.AxisOpts(name="value", type_="log"),
        xaxis_opts=opts.AxisOpts(name="stored precision"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="10%"),
    )
)
bar.render_notebook()
```

## 11. Head-to-head Benchmark

The same texts, the same probe, the same separation diagnostic, one model live at a time. Sections 9-10 produced the numbers; this collects and charts them.

Four columns and each answers a different production question:

- **probe accuracy** - how much task signal is linearly available in the vector.
- **gap (intra minus inter cosine)** - whether cosine similarity is meaningful, which is what a vector database actually uses. A model can score well on the first and poorly on the second; raw BERT in section 8 is the extreme case.
- **texts/second** - your indexing cost. Embedding 10 million chunks at 200/s is 14 hours.
- **GB per 10M vectors** - your serving cost, which dimensions determine and which Matryoshka truncation can cut.

The chart plots probe accuracy against index size, because that is the trade-off nobody makes deliberately and everybody lives with. At n=800 test items, probe accuracy carries roughly +/-1.7 points of noise.

------------------------------------------------------------------------

``` python
import pandas as pd

df_results = pd.DataFrame(results).sort_values("probe_acc", ascending=False)
show_table(
    df_results.to_dict("records"),
    title=f"dair-ai/emotion linear probe ({N_TRAIN} train / {N_TEST} test)",
    best=("probe_acc", "gap", "texts_per_sec"),
    lower_is_better=("index_gb_per_10m",),
    caption="probe_acc = task signal in the vector; gap = whether cosine similarity is "
            "meaningful; index_gb_per_10m = what it costs to serve",
)
```

```
                              dair-ai/emotion linear probe (2000 train / 800 test)                              
                                                                                                                
                                                                                    texts_per_s   index_gb_per_ 
 model          params_m    dims   pooling   probe_acc    intra    inter      gap            ec             10m 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 bge-base-en-        109     768   cls          0.6862   0.6060   0.5840   0.0220      1,941.40           30.72 
 v1.5                                                                                                           
 gte-modernbe        149     768   cls          0.6737   0.5710   0.5480   0.0230        929.10           30.72 
 rt-base                                                                                                        
 qwen3-embedd        596   1,024   last         0.6725   0.4750   0.4590   0.0170        269.20           40.96 
 ing-0.6b                                                                                                       
 all-MiniLM-L         22     384   mean         0.6262   0.1930   0.1770   0.0160      6,411.20           15.36 
 6-v2                                                                                                           
                                                                                                                
probe_acc = task signal in the vector; gap = whether cosine similarity is meaningful; index_gb_per_10m = what it
                                                 costs to serve                                                 
```

``` python
from pyecharts import options as opts
from pyecharts.charts import Bar

bar = (
    Bar()
    .add_xaxis([r["model"] for r in results])
    .add_yaxis("probe accuracy x100", [round(r["probe_acc"] * 100, 1) for r in results])
    .add_yaxis("intra-inter cosine gap x100", [round(r["gap"] * 100, 1) for r in results])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"dair-ai/emotion linear probe ({N_TRAIN} train / {N_TEST} test)",
            subtitle="RTX 3060, frozen embeddings + logistic regression - "
                     "a few-shot setting; a full fine-tune reaches ~93%",
        ),
        yaxis_opts=opts.AxisOpts(name="score"),
        xaxis_opts=opts.AxisOpts(name="model", axislabel_opts=opts.LabelOpts(rotate=15, font_size=9)),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="10%"),
    )
)
bar.render_notebook()
```

``` python
from pyecharts.charts import Line

# The Matryoshka sweep. A flat curve means you can truncate for free; a steep one means
# the model's information is spread across all dimensions and truncation costs you.
line = Line().add_xaxis([str(d) for d in DIMS])
for name, accs in mrl_curves.items():
    line.add_yaxis(name, [round(a * 100, 1) if a is not None else None for a in accs],
                   is_connect_nones=False)
line.set_global_opts(
    title_opts=opts.TitleOpts(
        title="Probe accuracy vs truncated dimensions",
        subtitle="slice the vector, renormalise - MRL-trained models degrade gracefully",
    ),
    xaxis_opts=opts.AxisOpts(name="dimensions kept"),
    yaxis_opts=opts.AxisOpts(name="probe accuracy x100"),
    tooltip_opts=opts.TooltipOpts(trigger="axis"),
    legend_opts=opts.LegendOpts(pos_top="10%"),
)
line.render_notebook()
```

``` python
from pyecharts.charts import Scatter

# Quality against index size - the trade-off that decides a deployment and that nobody
# makes on purpose.
scatter = Scatter()
scatter.add_xaxis([r["index_gb_per_10m"] for r in results])
for r in results:
    scatter.add_yaxis(
        r["model"], [[r["index_gb_per_10m"], round(r["probe_acc"] * 100, 1)]],
        symbol_size=18, label_opts=opts.LabelOpts(is_show=False),
    )
scatter.set_global_opts(
    title_opts=opts.TitleOpts(title="Probe accuracy vs index size",
                              subtitle="float32 storage for a 10M-vector index"),
    xaxis_opts=opts.AxisOpts(name="GB per 10M vectors (float32)", type_="value"),
    yaxis_opts=opts.AxisOpts(name="probe accuracy x100", type_="value"),
    tooltip_opts=opts.TooltipOpts(trigger="item"),
)
scatter.render_notebook()
```

## 12. Interactive: embed and search your own text

Edit `MY_CORPUS` and `MY_QUERIES` below. This is the cell people run on its own, so it opens with a `require(...)` guard naming what it needs from Setup rather than dying on a bare `NameError`.

It builds a tiny semantic index - embed the corpus, embed the query, cosine, sort - which is a vector database with the hard parts (persistence, approximate nearest neighbours, filtering) removed. That is genuinely all the retrieval math there is.

The inputs worth trying are the ones that show where embeddings differ from keyword search:

- **A query with no shared words** with the right document (“how do I get my money back?” against a document about refunds). This is what embeddings buy you and what BM25 cannot do.
- **Negation** (“laptops without a touchscreen”). Embeddings are famously weak here - the vector for a sentence and its negation are close, because they share nearly all their content. Expect the wrong result and remember it before shipping a filter as a semantic query.
- **A rare exact identifier** (a SKU, an error code, a surname). Now BM25 wins and the embedding flounders. This asymmetry is why production retrieval is nearly always **hybrid** - see `11_Text_Ranking`.
- **The query prefix.** Set `USE_QUERY_PREFIX = True` to switch on BGE’s query instruction and watch the scores move. Asymmetric models expect it; the ranking can change.
- **The absolute scores.** Note that a *bad* match still scores 0.5-0.6. Cosine similarity has no natural zero, so “is this relevant?” needs a threshold calibrated on your data, not a universal constant.

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


require("device", "dtype", "HF_CACHE", "free_memory", "vram", "embed", "pool")

import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MY_CORPUS = [
    "To request a refund, open Settings > Billing and choose 'Cancel and refund'.",
    "Our laptops ship with a matte non-touch display by default.",
    "The XR-7741 sensor requires firmware 2.3 or later to report humidity.",
    "Password resets are sent to the address on file within five minutes.",
    "Touchscreen models are available in the Pro line only.",
    "Shipping to the EU takes 3-5 business days and is tracked end to end.",
]

MY_QUERIES = [
    "how do I get my money back?",       # no shared words with the refund document
    "laptops without a touchscreen",     # negation - watch this fail
    "XR-7741",                           # rare identifier - BM25 territory
    "when will my parcel arrive",
]

USE_QUERY_PREFIX = False   # True switches on BGE's query instruction
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Re-runnable: this cell frees the model at the end, so guard the load or a second
# shift-enter raises NameError.
if "my_emb_model" not in globals():
    my_emb_tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5", cache_dir=HF_CACHE)
    my_emb_model = AutoModel.from_pretrained(
        "BAAI/bge-base-en-v1.5", dtype=dtype, cache_dir=HF_CACHE).to(device).eval()

# Documents get no prefix; queries optionally get the instruction. BGE uses CLS pooling.
doc_vecs = embed(my_emb_model, my_emb_tok, MY_CORPUS, how="cls")
q_prefix = BGE_QUERY_PREFIX if USE_QUERY_PREFIX else ""
q_vecs = embed(my_emb_model, my_emb_tok, MY_QUERIES, how="cls", prefix=q_prefix)

scores = q_vecs @ doc_vecs.T          # unit vectors, so this is cosine similarity
print(f"query prefix: {q_prefix!r}\n")
for q, row in zip(MY_QUERIES, scores):
    order = row.argsort(descending=True)
    print(f"Q: {q}")
    for rank, i in enumerate(order[:3].tolist(), 1):
        print(f"   {rank}. {row[i]:.3f}  {MY_CORPUS[i][:78]}")
    print()

print("Note how high the *wrong* answers score - cosine has no natural zero, so a")
print("relevance threshold has to be calibrated per model and per corpus.")

del my_emb_model, my_emb_tok, doc_vecs, q_vecs, scores
free_memory()
vram("final")
```

    query prefix: ''

    Q: how do I get my money back?
       1. 0.751  To request a refund, open Settings > Billing and choose 'Cancel and refund'.
       2. 0.446  Shipping to the EU takes 3-5 business days and is tracked end to end.
       3. 0.432  Password resets are sent to the address on file within five minutes.

    Q: laptops without a touchscreen
       1. 0.716  Our laptops ship with a matte non-touch display by default.
       2. 0.645  Touchscreen models are available in the Pro line only.
       3. 0.436  Password resets are sent to the address on file within five minutes.

    Q: XR-7741
       1. 0.660  The XR-7741 sensor requires firmware 2.3 or later to report humidity.
       2. 0.492  Password resets are sent to the address on file within five minutes.
       3. 0.467  Shipping to the EU takes 3-5 business days and is tracked end to end.

    Q: when will my parcel arrive
       1. 0.670  Shipping to the EU takes 3-5 business days and is tracked end to end.
       2. 0.466  Password resets are sent to the address on file within five minutes.
       3. 0.404  To request a refund, open Settings > Billing and choose 'Cancel and refund'.

    Note how high the *wrong* answers score - cosine has no natural zero, so a
    relevance threshold has to be calibrated per model and per corpus.
    VRAM final                   0.02 GB allocated /  0.04 GB reserved

## 13. Common Frameworks

This notebook used raw `transformers` to keep pooling visible, and that is the one thing you should not do in production - the pooling strategy, the query/document prefixes and the normalisation are checkpoint-specific, and getting any of them wrong degrades results quietly. `sentence-transformers` reads all three from the model config. The rest of the ecosystem is storage: at scale, the embedding is cheap and the index is the bill.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [sentence-transformers](https://github.com/UKPLab/sentence-transformers) | modelling | `encode(texts, normalize_embeddings=True)` with the correct pooling and prefixes read from the checkpoint, plus batching and multi-GPU | Apache 2.0 | Production, always. Section 8 exists to show what this library is doing for you |
| [transformers](https://github.com/huggingface/transformers) | modelling | The raw towers when you need the token embeddings, a custom pooling, or a model with no ST wrapper | Apache 2.0 | Research, or a model that has not been packaged yet |
| [peft](https://github.com/huggingface/peft) + `MultipleNegativesRankingLoss` | modelling | Contrastive fine-tuning on a few thousand (query, passage) pairs - under an hour on this box | Apache 2.0 | Your own data. This typically beats moving to a model five times larger, and hard negatives are what make it work |
| [faiss](https://github.com/facebookresearch/faiss) / [hnswlib](https://github.com/nmslib/hnswlib) | data | Exact and approximate indexes, product quantisation, and the recall/latency knobs | MIT / Apache 2.0 | Past a million vectors, or when memory is the constraint |
| [Qdrant](https://github.com/qdrant/qdrant) / [Milvus](https://github.com/milvus-io/milvus) / [LanceDB](https://github.com/lancedb/lancedb) / pgvector | data | A database: persistence, metadata filtering, incremental updates, replication | Apache 2.0 / PostgreSQL | The corpus changes while it is being queried. FAISS handles updates poorly and that is usually what forces the move |
| [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) (TEI) | inference runtime | A purpose-built embedding server: dynamic batching, ONNX/candle backends, far higher throughput than a Python loop | Apache 2.0 | Indexing a corpus or serving live queries. The single largest easy win in this task |
| [optimum](https://github.com/huggingface/optimum) + [ONNX Runtime](https://onnxruntime.ai/) | inference runtime | Export and int8 for CPU embedding, plus [model2vec](https://github.com/MinishLab/model2vec) static distillation when even that is too slow | Apache 2.0 / MIT | Edge or CPU-only indexing, where a static embedding model is hundreds of times faster for a real quality cost |
| [BentoML](https://github.com/bentoml/BentoML) / [Ray Serve](https://github.com/ray-project/ray) | serving | The embedding service versioned separately from the index, so a model change is a deliberate re-index | Apache 2.0 | Production. Silently changing the encoder invalidates every vector you have stored |
| [MTEB](https://github.com/embeddings-benchmark/mteb) + your own Recall@10 | evaluation | The public leaderboard for shortlisting, and the 200 hand-labelled queries that actually decide | Apache 2.0 | Always both. MTEB narrows the field; your own data picks the winner, and it disagrees often enough to justify the afternoon |

**The 2026 default stack** is a mid-sized `sentence-transformers` model served by TEI, normalised vectors in FAISS below a million and Qdrant above it, Matryoshka truncation plus int8 storage before sharding, and a fine-tune on your own pairs before any thought of a bigger model.

The common wrong turn is picking a model from the MTEB leaderboard and skipping the pooling and prefix details - `bge` and `e5` models want an instruction prefix on queries, and omitting it costs more than the gap between the top ten models. The second is forgetting normalisation: every index here computes inner product, so unnormalised vectors silently give you dot product instead of cosine.

------------------------------------------------------------------------

## 14. Going Further

- **Fine-tune on your own pairs - it is cheaper than it sounds.** `MultipleNegativesRankingLoss` over a few thousand (query, relevant-passage) pairs from your logs, on a `bge-base`-size model, takes under an hour on this box and typically beats moving to a model 5x larger. **Hard negatives** (wrong answers that a base model retrieves) are the highest-value ingredient; mine them with the base model itself.
- **Measure on your own data, not MTEB.** Hand-label 200 queries against your corpus and compute Recall@10 for three candidate models. It takes an afternoon and it will disagree with the leaderboard often enough to justify itself.
- **Truncate and quantize by default at scale.** MRL truncation to 256 dimensions plus int8 storage is a 16x index reduction for a few points, and rescoring the top-100 with full vectors recovers most of it. Do this before you shard.
- **Go hybrid.** Dense embeddings lose to BM25 on rare identifiers, exact codes and names. Run both, fuse with Reciprocal Rank Fusion, and rerank the union with a cross-encoder. This is the standard 2026 retrieval stack, and `11_Text_Ranking` builds it.
- **Chunk deliberately.** 300-500 tokens with ~20% overlap is a sane starting point; retrieve on the chunk and pass the parent section downstream. If chunks routinely exceed 512 tokens, move to an 8192-context embedder rather than shrinking the chunks.
- **Plan for re-indexing.** Store the model id and version alongside every vector, and build the pipeline so a full re-embed is a routine job rather than a project. This one decision determines whether you can adopt a better model in two years.
- **For maximum quality, consider late interaction.** ColBERT-style models keep one vector per token and score with MaxSim. Quality is materially better on hard retrieval; storage is ~100x. Worth it for small, high-value corpora.
- **Related notebooks.** `10_Sentence_Similarity` (what these vectors are for, and bi- vs cross-encoders), `11_Text_Ranking` (retrieval, hybrid search, reranking, nDCG), `00_Text_Classification` (the fine-tuned alternative to a probe), `04_Zero_Shot_Classification` (embedding-vs-label as a cheap zero-shot method), `03_Question_Answering` (RAG, which is this notebook plus a reader), `Computer_Vision/16_Image_Feature_Extraction` (the same ideas for images).

------------------------------------------------------------------------

Back to top
