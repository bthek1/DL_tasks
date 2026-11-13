# Sentence Similarity

Scoring how close two texts are in meaning: the bi-encoder / cross-encoder trade-off that decides every retrieval architecture, why Spearman and not Pearson, the adversarial case where high word overlap means opposite meanings, and runnable code that measures all of it on STS-B and PAWS.

Author

Benedict Thekkel

## 1. What is Sentence Similarity?

Sentence similarity scores how close two texts are in meaning. It is the task that turns the vectors from `07_Feature_Extraction` into decisions - is this a duplicate, does this FAQ answer that question, are these two paragraphs the same claim.

**Input.** Two texts. Sometimes both are sentences of the same kind, sometimes one is a short query and the other a long document, and the difference matters more than it looks.

**Output.** A score. Either a continuous similarity (usually cosine, in `[-1, 1]`) or a binary judgement (paraphrase or not).

**Two architectures, and choosing between them is the central engineering decision:**

|  | **Bi-encoder** | **Cross-encoder** |
|----|----|----|
| How | encode each text separately, compare vectors | encode the **pair jointly**, output one score |
| Attention between texts | none | full, every token to every token |
| Cost for N texts, all pairs | N encodings + cheap dot products | N(N-1)/2 full forward passes |
| Precomputable | yes - the whole point | no, ever |
| Accuracy | good | consistently better |
| Used for | retrieval, clustering, dedup at scale | reranking a shortlist, final scoring |

The numbers make the trade concrete. Finding the most similar pair among 10,000 sentences takes 10,000 encodings plus a matrix multiply with a bi-encoder - seconds. With a cross-encoder it takes **50 million forward passes** - the original SBERT paper measured this as 65 hours against 5 seconds. That single fact is why every production retrieval system is a bi-encoder for recall followed by a cross-encoder for precision on the top 50-100.

**Symmetric versus asymmetric.** “Are these two questions duplicates?” is symmetric - both sides are the same kind of text. “Does this passage answer this query?” is asymmetric - a short query against a long document, and the model needs to know which is which. Most modern embedding models expose this through prefixes or instructions (see `07_Feature_Extraction`), and using a symmetric setup for an asymmetric task quietly costs several points.

**The hard part is that “similar” is underspecified.** “The cat sat on the mat” and “The mat sat on the cat” share every word and mean different things. “I need to cancel” and “How do I stop my subscription?” share almost nothing and mean the same thing. Word overlap and meaning are close to independent, and section 10 measures a case where they actively point in opposite directions.

**Neighbouring tasks:**

| Task | How it differs | Notebook |
|----|----|----|
| Feature extraction | Produces the vectors this task compares | `07_Feature_Extraction` |
| Text ranking | Orders many candidates against one query | `11_Text_Ranking` |
| Zero-shot classification | Compares text to labels, not to text | `04_Zero_Shot_Classification` |
| Question answering | Extracts an answer rather than scoring a pair | `03_Question_Answering` |
| Text classification | One text, fixed labels | `00_Text_Classification` |

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Duplicate question detection | Q&A platforms (Quora, Stack Overflow) | New question + archive -\> duplicate or not | Precision; a wrong merge destroys content |
| FAQ and intent matching | Customer support | User message + FAQ bank -\> best match | Latency; must abstain when nothing matches |
| Semantic caching for LLMs | Any LLM application | New prompt + cache -\> hit or miss | Threshold calibration; a false hit returns a wrong answer |
| Deduplication of training data | ML teams, data engineering | Corpus -\> near-duplicate clusters | Scale (billions of pairs); needs approximate NN, not exhaustive |
| Plagiarism and reuse detection | Education, publishing | Document + corpus -\> matched passages | Recall on paraphrased reuse; must survive rewording |
| Record linkage / entity matching | Data integration | Two records -\> same entity or not | Structured fields; embeddings alone are not enough |
| Recommendation by content | Media, e-commerce | Item text -\> similar items | Cold start; similarity is not the same as substitutability |
| Evaluating generated text | ML evaluation | Output + reference -\> similarity | Correlates better than n-gram metrics; still reference-bound |
| Contradiction and consistency checks | Compliance, fact-checking | Claim + source -\> agree / disagree | Similarity cannot see negation - needs NLI |

What the STS number hides:

- **Similarity is task-relative and there is no universal notion of it.** For deduplication, two articles about the same event by different outlets are *not* duplicates. For clustering, they are. The same model and the same score serve both, and only your threshold distinguishes them.
- **The threshold is the product, not the model.** Cosine similarity has no natural zero: an unrelated pair often scores 0.3-0.6. “Similar enough” must be calibrated per model and per corpus with labelled examples, and a threshold copied from a blog post will be wrong.
- **Negation and antonymy are near-invisible.** “The drug is effective” and “The drug is not effective” embed very close together, because they share almost all content. If your task depends on that distinction, similarity is the wrong tool - use NLI (`04_Zero_Shot_Classification`).
- **High overlap and same meaning are different things**, and adversarial datasets like PAWS were built precisely because models learned to conflate them. Section 10 runs it.
- **Symmetry assumptions break silently.** A model tuned for query-document matching used to compare two documents underperforms with no error message.

------------------------------------------------------------------------

## 3. How Modern Sentence Similarity Works

1.  **String and set overlap (pre-2013).** Edit distance, Jaccard over token sets, TF-IDF cosine. Fast, interpretable, and blind to synonymy - the persistent failure that motivated everything after.
2.  **Averaged static embeddings (2013-2018).** Mean of word2vec or GloVe vectors. Captures topic, ignores word order and negation. Still a shockingly hard baseline to beat on some STS sets, which says more about STS than about the method.
3.  **BERT used naively (2018-2019).** Two obvious approaches, both bad in different ways. Feeding a pair into BERT with a regression head (a cross-encoder) works very well and is computationally hopeless for search. Pooling BERT’s token vectors and taking cosine works computationally and is *worse than averaged GloVe* - the anisotropy problem covered in `07_Feature_Extraction` section 8.
4.  **Sentence-BERT (2019).** Train BERT in a **siamese** configuration: encode both sentences with shared weights, and optimise cosine similarity against a target directly. This made pooled vectors meaningful and reduced the 10,000-sentence problem from 65 hours to 5 seconds. It is the single most consequential result in this task’s history, and the modern bi-encoder is a direct descendant.
5.  **Contrastive objectives and hard negatives (2020-2022).** MultipleNegativesRankingLoss - treat other items in the batch as negatives - proved far more effective than regressing on similarity scores, and scaled to huge batches. **SimCSE** showed that even unsupervised contrastive training (the same sentence twice with different dropout masks) nearly matched supervised results. Mining **hard negatives** (plausible-but-wrong pairs) became the highest-value data work.
6.  **Adversarial evaluation (2019-2021).** PAWS constructed sentence pairs with ~95% word overlap and different meanings, by swapping arguments around. Models trained on QQP fell from 90%+ accuracy to near chance. The lesson stuck: benchmarks that do not decorrelate lexical overlap from meaning do not measure what they claim.
7.  **The modern stack (2022-2026).** Large contrastively pretrained bi-encoders (BGE, GTE, E5, Qwen3-Embedding) with instruction prefixes for symmetric/asymmetric roles, plus strong cross-encoder rerankers (bge-reranker, ms-marco MiniLM, mxbai-rerank). LLMs used as pairwise judges outperform both on nuanced judgements and cost far more per pair. The architecture has not changed since SBERT; the training data and scale have.

**Where it stands (mid-2026).** The default is a two-stage pipeline: a bi-encoder retrieves and a cross-encoder reranks. Bi-encoder quality has improved enough that the reranker’s marginal gain has narrowed on easy tasks, but on anything adversarial - high lexical overlap, subtle negation, fine distinctions - the cross-encoder’s cross-attention still wins by a lot, because it can compare tokens across the pair and a bi-encoder structurally cannot.

------------------------------------------------------------------------

## 4. Evaluation Metrics

**Spearman rank correlation** is the standard for continuous similarity, and the reason it beats Pearson is worth understanding rather than accepting.

$$\rho = 1 - \frac{6\sum d_i^2}{n(n^2-1)} \qquad (d_i = \text{rank}(x_i) - \text{rank}(y_i))$$

Human STS annotations are on a 0-5 scale; model outputs are cosine similarities on a compressed and model-specific range - one model’s “identical” is 0.95, another’s is 0.75. **Pearson measures whether the relationship is linear**, so it punishes a model whose scores are perfectly ordered but non-linearly scaled. **Spearman measures only the ordering**, which is exactly what matters: every downstream use (rank, threshold, retrieve top-k) depends on order, not on absolute value. Reporting Pearson on STS conflates calibration with quality.

**For binary paraphrase detection** (MRPC, QQP, PAWS), use accuracy, F1 and **average precision**. AP is the more informative choice because it summarises performance across all thresholds rather than at one arbitrary cut. Report the class balance too - QQP is ~37% positive, and accuracy is misleading without that context.

**Threshold-dependent metrics need the threshold stated.** “94% accurate at deduplication” is meaningless without the cosine cutoff and the corpus. Tune the threshold on a validation split and report it.

**Pitfalls, in the order people hit them:**

- **Never compare raw cosine values across models.** Their scales differ. Only ranks are comparable, which is another argument for Spearman.
- **Ties matter.** A proper Spearman implementation assigns *average* ranks to tied values. A naive `argsort` breaks ties arbitrarily and inflates or deflates the correlation, especially on coarse human labels where ties are common. The implementation below handles this explicitly.
- **STS-B is nearly saturated and its sentences are short and simple.** Strong models cluster around 0.85-0.90 Spearman, where differences are within noise. Use it as a sanity check, and use an adversarial set (PAWS) plus your own data to actually discriminate.
- **Evaluate on your own pairs.** Similarity is task-relative (section 2). Two hundred hand-labelled pairs from your corpus will disagree with STS-B rankings often enough to justify the afternoon.

The cell below implements Spearman with proper tie handling, Pearson for contrast, and average precision - about 40 lines, and the tie-handling is the part worth reading.

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


import math


def _ranks(values):
    "Ranks with average ranks for ties - the part naive implementations get wrong."
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1          # 1-based average rank across the tied block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(x, y):
    "Linear correlation. Punishes a model whose scores are ordered but non-linearly scaled."
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = math.sqrt(sum((a - mx) ** 2 for a in x))
    vy = math.sqrt(sum((b - my) ** 2 for b in y))
    return cov / (vx * vy) if vx and vy else 0.0


def spearman(x, y):
    "Rank correlation - the STS standard, because only the ordering is used downstream."
    return pearson(_ranks(x), _ranks(y))


def average_precision(scores, labels):
    "Area under the precision-recall curve, computed exactly over every threshold."
    pairs = sorted(zip(scores, labels), key=lambda p: -p[0])
    n_pos = sum(labels)
    if not n_pos:
        return 0.0
    tp, total = 0, 0.0
    for i, (_, lab) in enumerate(pairs, start=1):
        if lab:
            tp += 1
            total += tp / i          # precision at each positive hit
    return total / n_pos


def best_f1(scores, labels):
    "Best achievable F1 over all thresholds, and the threshold that achieves it."
    pairs = sorted(zip(scores, labels), key=lambda p: -p[0])
    n_pos, tp, fp, best = sum(labels), 0, 0, (0.0, 0.0)
    for score, lab in pairs:
        tp, fp = tp + (lab == 1), fp + (lab == 0)
        prec, rec = tp / (tp + fp), tp / n_pos
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        if f1 > best[0]:
            best = (f1, score)
    return best


# Toy example: two models with identical ORDERING and different scales.
gold = [5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
model_a = [0.95, 0.88, 0.71, 0.55, 0.40, 0.22]     # roughly linear in the gold score
model_b = [0.99, 0.97, 0.94, 0.90, 0.85, 0.70]     # same order, compressed and curved

show_table([{"model": name, "scores": str(pred),
             "Pearson": round(pearson(gold, pred), 4),
             "Spearman": round(spearman(gold, pred), 4)}
            for name, pred in [("A (linear scale)", model_a), ("B (compressed)", model_b)]],
           title=f"Two models, identical ordering, against gold {gold}",
           caption="identical Spearman, different Pearson - Pearson is measuring "
                   "calibration, which no downstream use of a similarity score depends on")

# Ties: the case a naive implementation gets wrong.
tied_gold = [3.0, 3.0, 3.0, 1.0, 1.0]
tied_pred = [0.8, 0.7, 0.9, 0.4, 0.3]
print(f"with ties: ranks {_ranks(tied_gold)}  ->  Spearman {spearman(tied_gold, tied_pred):.4f}")
print("  (the three tied 3.0s share ranks 3-5 and all get 4.0, the two 1.0s get 1.5;")
print("   arbitrary tie-breaking would assign 3, 4, 5 and distort the correlation)\n")

# Binary paraphrase metrics.
scores = [0.95, 0.91, 0.60, 0.88, 0.30, 0.85, 0.20, 0.75]
labels = [1, 1, 0, 0, 0, 1, 0, 1]
f1, thr = best_f1(scores, labels)
show_kv({"average precision": round(average_precision(scores, labels), 4),
         "best F1": round(f1, 4), "at threshold": round(thr, 2)},
        title="Binary paraphrase metrics - always report the threshold")
```

```
 Two models, identical ordering, against gold [5.0, 4.0, 3.0, 2.0, 1.0, 0.0] 
                                                                             
 model              scores                                Pearson   Spearman 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 A (linear scale)   [0.95, 0.88, 0.71, 0.55, 0.4, 0.22]    0.9948     1.0000 
 B (compressed)     [0.99, 0.97, 0.94, 0.9, 0.85, 0.7]     0.9287     1.0000 
                                                                             
  identical Spearman, different Pearson - Pearson is measuring calibration,  
          which no downstream use of a similarity score depends on           
```

    with ties: ranks [4.0, 4.0, 4.0, 1.5, 1.5]  ->  Spearman 0.8660
      (the three tied 3.0s share ranks 3-5 and all get 4.0, the two 1.0s get 1.5;
       arbitrary tie-breaking would assign 3, 4, 5 and distort the correlation)

```
    Binary paraphrase metrics - always report the threshold     
                                                                
 average precision                                       0.8875 
 best F1                                                 0.8889 
 at threshold                                            0.7500 
                                                                
```

## 5. Datasets

| Dataset | Contents | Size | Label | License | Typical use |
|----|----|----|----|----|----|
| [STS Benchmark](https://huggingface.co/datasets/sentence-transformers/stsb) | Sentence pairs from news, captions, forums | 5.7k / 1.4k test | 0-5 human similarity | mixed | The standard; used below |
| [STS12-16](https://huggingface.co/datasets/mteb/sts12-sts) | Earlier SemEval STS tasks | ~14k total | 0-5 | mixed | MTEB reports all of them |
| [SICK-R](https://huggingface.co/datasets/RobZamp/sick) | Sentences varied by controlled transformations | 10k | 1-5 + NLI label | CC BY-NC-SA | Similarity **and** entailment on the same pairs |
| [MRPC](https://huggingface.co/datasets/nyu-mll/glue) | News sentence pairs | 5.8k | paraphrase y/n | custom | GLUE paraphrase task; small |
| [QQP](https://huggingface.co/datasets/nyu-mll/glue) | Quora question pairs | 400k | duplicate y/n | custom | The big paraphrase set; ~37% positive |
| [PAWS](https://huggingface.co/datasets/google-research-datasets/paws) | **~95% word overlap**, meaning often differs | 65k | paraphrase y/n | custom, free | **Adversarial**; used below |
| [PAWS-X](https://huggingface.co/datasets/google-research-datasets/paws-x) | PAWS in 7 languages | 49k | paraphrase y/n | custom, free | Cross-lingual adversarial |
| [BIOSSES](https://huggingface.co/datasets/mteb/biosses-sts) | Biomedical sentence pairs | 100 | 0-4 | custom | Domain-shift check; tiny |
| [SemRel / STS multilingual](https://huggingface.co/datasets/mteb/stsb_multi_mt) | STS-B translated into 15 languages | 1.4k each | 0-5 | mixed | Multilingual similarity |
| [AllNLI triplets](https://huggingface.co/datasets/sentence-transformers/all-nli) | (anchor, positive, negative) from SNLI+MNLI | 940k | triplet | mixed | The standard training data |

This notebook uses two, deliberately:

- **STS-B test** (1,379 pairs) for Spearman correlation. It is the standard and it is close to saturated - strong models land at 0.85-0.90 and the differences between them are within noise.
- **PAWS `labeled_final` test** (8,000 pairs) for the adversarial check. PAWS was built by swapping words and arguments to produce pairs with near-identical vocabulary and often opposite meanings. It is the dataset that separates “measures meaning” from “measures word overlap”, and models that look equivalent on STS-B are not equivalent here.

Running both is the point. A single benchmark number for this task is not informative.

Downloads land in `DL_tasks/datasets/` via `cache_dir` (gitignored).

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

The reference board is [MTEB](https://huggingface.co/spaces/mteb/leaderboard) - the **STS** tab for this task and the **Reranking** tab for cross-encoders. The `sentence-transformers` [pretrained model list](https://sbert.net/docs/sentence_transformer/pretrained_models.html) is the practical index.

**Bi-encoders** (encode separately, compare vectors):

| Model | Params | Dims | License | Best for |
|----|----|----|----|----|
| [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 22M | 384 | Apache 2.0 | the cheap default, CPU-friendly; used below |
| [all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) | 109M | 768 | Apache 2.0 | the classic symmetric-STS strong baseline; used below |
| [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | 109M | 768 | MIT | strong general English; used below |
| [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | 149M | 768 | Apache 2.0 | long inputs (8192 tokens) |
| [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 596M | 1024 | Apache 2.0 | instruction-aware, multilingual |
| [multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) | 560M | 1024 | MIT | 100 languages |

**Cross-encoders** (encode the pair jointly, no precomputation possible):

| Model | Params | License | Output | Best for |
|----|----|----|----|----|
| [stsb-roberta-base](https://huggingface.co/cross-encoder/stsb-roberta-base) | 125M | Apache 2.0 | 0-1 similarity | direct STS scoring; used below |
| [stsb-TinyBERT-L4](https://huggingface.co/cross-encoder/stsb-TinyBERT-L-4) | 14M | Apache 2.0 | 0-1 similarity | latency-bound reranking |
| [ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | 22M | Apache 2.0 | relevance logit | query-document reranking (see nb 11) |
| [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | 568M | Apache 2.0 | relevance | best open multilingual reranker |
| [mxbai-rerank-base-v2](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2) | 500M | Apache 2.0 | relevance | strong 2025 reranker |
| Frontier LLMs as judges | \- | proprietary | any rubric | nuanced pairwise judgements, high cost |

**How to choose - and it is not either/or.** Comparing more than a few hundred texts, or needing precomputed vectors: bi-encoder, no alternative. Scoring a small fixed set of pairs, or reranking a shortlist someone else retrieved: cross-encoder, it is simply more accurate. Production systems do both, and the ratio to remember is that a bi-encoder over 10,000 texts is a matrix multiply while a cross-encoder over the same set is 50 million forward passes.

------------------------------------------------------------------------

## 7. Setup

Everything loads through Hugging Face `transformers` with `AutoModel` - deliberately not `sentence-transformers`, so pooling and normalisation stay visible (the same choice as `07_Feature_Extraction`). In production, use `sentence-transformers`; it handles pooling, prefixes and batching correctly.

Package roles:

- `transformers` + `torch` - the three bi-encoders and the cross-encoder
- `accelerate` - `device_map` placement
- `datasets` - STS-B and PAWS
- `pandas` + `pyecharts` - the benchmark table and charts
- `rich` - the result tables. It renders to HTML inside Jupyter, so the tables survive into the published docs; `show_table` / `show_kv` / `rule` are defined in the first code cell of section 4.

Three details that decide correctness:

- **Pooling is per-checkpoint.** MiniLM and mpnet use mean pooling, BGE uses CLS. Guessing produces degraded vectors with no error - see `07_Feature_Extraction` section 8.
- **A cross-encoder is `AutoModelForSequenceClassification` with a text *pair*.** `tok(list_a, list_b, ...)` builds the pair encoding with the right separator tokens; passing pre-concatenated strings instead is a silent bug.
- **`stsb-roberta-base` has one output logit in `[0, 1]`** rather than class probabilities. Applying a softmax to a single logit yields 1.0 everywhere - a genuinely confusing failure. Check `model.config.num_labels` before deciding how to read the head.

------------------------------------------------------------------------

``` python
# Everything runs through Hugging Face transformers - no sentence-transformers, so the
# pooling and pair encoding stay visible.
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

BI_ENCODERS = [
    # name, checkpoint, pooling, params (M)
    ("all-MiniLM-L6-v2", "sentence-transformers/all-MiniLM-L6-v2", "mean", 22),
    ("all-mpnet-base-v2", "sentence-transformers/all-mpnet-base-v2", "mean", 109),
    ("bge-base-en-v1.5", "BAAI/bge-base-en-v1.5", "cls", 109),
]
CROSS_ENCODER = ("stsb-roberta-base", "cross-encoder/stsb-roberta-base", 125)
```

    NVIDIA GeForce RTX 3060
    device: cuda:0 | dtype: torch.float16

``` python
from datasets import load_dataset

# STS-B test: 1,379 pairs with human similarity, already normalised to [0, 1].
stsb = load_dataset("sentence-transformers/stsb", split="test", cache_dir=HF_CACHE)
N_STS = 600                                   # a subset keeps the cross-encoder tractable
sts = stsb.select(range(N_STS))
sts_a = [r["sentence1"] for r in sts]
sts_b = [r["sentence2"] for r in sts]
sts_gold = [float(r["score"]) for r in sts]

# PAWS labeled_final test: ~95% word overlap, meaning often differs. The adversarial set.
paws = load_dataset("google-research-datasets/paws", "labeled_final", split="test",
                    cache_dir=HF_CACHE)
N_PAWS = 600
pw = paws.shuffle(seed=0).select(range(N_PAWS))
paws_a = [r["sentence1"] for r in pw]
paws_b = [r["sentence2"] for r in pw]
paws_gold = [int(r["label"]) for r in pw]

print(stsb)
print(paws)
print(f"\nSTS-B:  {N_STS} pairs, gold in [{min(sts_gold):.2f}, {max(sts_gold):.2f}]")
print(f"PAWS:   {N_PAWS} pairs, {sum(paws_gold)} paraphrase / "
      f"{N_PAWS - sum(paws_gold)} not\n")


def word_overlap(a, b):
    "Jaccard over lowercased word sets - the lexical-overlap baseline."
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return len(sa & sb) / max(len(sa | sb), 1)


print("STS-B examples (gold, word overlap):")
for i in (0, 3, 7):
    print(f"  {sts_gold[i]:.2f}  overlap {word_overlap(sts_a[i], sts_b[i]):.2f}")
    print(f"     A: {sts_a[i][:80]}\n     B: {sts_b[i][:80]}")

print("\nPAWS examples - note the overlap is near 1.0 in BOTH classes:")
for i in (0, 1, 2):
    lab = "paraphrase" if paws_gold[i] else "NOT paraphrase"
    print(f"  {lab:14s} overlap {word_overlap(paws_a[i], paws_b[i]):.2f}")
    print(f"     A: {paws_a[i][:95]}\n     B: {paws_b[i][:95]}")
```

    Dataset({
        features: ['sentence1', 'sentence2', 'score'],
        num_rows: 1379
    })
    Dataset({
        features: ['id', 'sentence1', 'sentence2', 'label'],
        num_rows: 8000
    })

    STS-B:  600 pairs, gold in [0.00, 1.00]
    PAWS:   600 pairs, 277 paraphrase / 323 not

    STS-B examples (gold, word overlap):
      0.50  overlap 0.71
         A: A girl is styling her hair.
         B: A girl is brushing her hair.
      0.84  overlap 0.57
         A: A man is cutting up a cucumber.
         B: A man is slicing a cucumber.
      0.44  overlap 0.71
         A: A man is playing the drums.
         B: A man is playing the guitar.

    PAWS examples - note the overlap is near 1.0 in BOTH classes:
      paraphrase     overlap 0.73
         A: Philippe Maystadt was born in 1948 in Belgium .
         B: Philippe Maystadt was born in Belgium ( Verviers ) in 1948 .
      paraphrase     overlap 0.82
         A: Winters are cold , and summers are dry and extremely hot .
         B: Winters are cold and the summers are dry and extremely hot .
      NOT paraphrase overlap 0.86
         A: It was chosen as the 19th best film at the 7th Yokohama Film Festival .
         B: It was elected as the 7th best film at the 19th Yokohama Film Festival .

## 8. The lexical baseline, and why it is not enough

Before any model, measure what word overlap alone achieves. This is the baseline every similarity result must beat, and skipping it is how people convince themselves a model works when it is matching vocabulary.

On STS-B, Jaccard overlap correlates decently with human similarity - human annotators rate rewordings as similar, and rewordings share words. That correlation is real and it is a trap: it means **a model can score well on STS-B by learning word overlap**, without representing meaning at all.

On PAWS the baseline collapses to nothing, by construction. Every pair has near-identical vocabulary, and half of them mean different things, so overlap carries no signal at all - average precision sits at the class prior.

Running both is what makes the comparison honest. A model that beats overlap on STS-B and matches the prior on PAWS has learned overlap with extra steps.

------------------------------------------------------------------------

``` python
sts_overlap = [word_overlap(a, b) for a, b in zip(sts_a, sts_b)]
paws_overlap = [word_overlap(a, b) for a, b in zip(paws_a, paws_b)]

ap = average_precision(paws_overlap, paws_gold)
f1, thr = best_f1(paws_overlap, paws_gold)
prior = sum(paws_gold) / len(paws_gold)
show_kv({"STS-B Spearman": round(spearman(sts_gold, sts_overlap), 4),
         "STS-B Pearson": round(pearson(sts_gold, sts_overlap), 4),
         "PAWS average precision": round(ap, 4),
         "PAWS class prior": round(prior, 4),
         "PAWS best F1": round(f1, 4),
         "PAWS mean overlap, paraphrase": round(
             sum(o for o, l in zip(paws_overlap, paws_gold) if l) / max(sum(paws_gold), 1), 3),
         "PAWS mean overlap, non-paraphrase": round(
             sum(o for o, l in zip(paws_overlap, paws_gold) if not l)
             / max(len(paws_gold) - sum(paws_gold), 1), 3)},
        title="Lexical overlap baseline - no model at all")
print("Overlap is a real signal on STS-B and none at all on PAWS, where both classes have")
print("the same vocabulary by construction. Any model must beat the first, and should not")
print("collapse to the class prior on the second.")
```

```
           Lexical overlap baseline - no model at all           
                                                                
 STS-B Spearman                                          0.5070 
 STS-B Pearson                                           0.5003 
 PAWS average precision                                  0.4552 
 PAWS class prior                                        0.4617 
 PAWS best F1                                            0.6339 
 PAWS mean overlap, paraphrase                           0.8870 
 PAWS mean overlap, non-paraphrase                       0.8870 
                                                                
```

    Overlap is a real signal on STS-B and none at all on PAWS, where both classes have
    the same vocabulary by construction. Any model must beat the first, and should not
    collapse to the class prior on the second.

## 9. Bi-encoders: one pass per text

The architecture that makes similarity search possible. Encode each sentence independently into a vector, then compare vectors with a dot product. The two texts never see each other - all the work happened at encoding time, which is exactly why the vectors can be precomputed and indexed.

The consequences are worth stating explicitly:

- **Cost is linear in the number of texts, not the number of pairs.** N encodings serve all N(N-1)/2 comparisons.
- **Vectors are reusable.** Index once, query forever. This is what a vector database stores.
- **The model must compress everything relevant into one fixed vector before it knows what it will be compared against.** That is the structural limitation, and it is why cross-encoders win on hard pairs.

The cell scores all three bi-encoders on STS-B and PAWS, one model live at a time.

**Expect a specific and instructive pattern:** strong Spearman on STS-B, and on PAWS something much closer to the lexical baseline than to the cross-encoder in section 10. A bi-encoder must decide what a sentence means before seeing its partner, and PAWS pairs differ only in *which* argument attaches to which predicate - information that survives poorly in a single pooled vector.

------------------------------------------------------------------------

``` python
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def pool(hidden, mask, how):
    "Collapse per-token vectors into one per text. Mask-aware for mean pooling."
    if how == "cls":
        return hidden[:, 0]
    m = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)


@torch.inference_mode()
def embed(model, tok, texts, how="mean", batch_size=64, max_length=128):
    "Encode texts to L2-normalised vectors, so a dot product is a cosine."
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=max_length).to(model.device)
        v = pool(model(**enc).last_hidden_state, enc["attention_mask"], how)
        out.append(F.normalize(v.float(), dim=-1))
    return torch.cat(out)


bi_results = []
for name, model_id, how, params_m in BI_ENCODERS:
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE)
    model = AutoModel.from_pretrained(model_id, dtype=dtype,
                                      cache_dir=HF_CACHE).to(device).eval()

    t0 = time.perf_counter()
    va, vb = embed(model, tok, sts_a, how), embed(model, tok, sts_b, how)
    sts_pred = (va * vb).sum(-1).tolist()          # unit vectors -> cosine
    sts_secs = time.perf_counter() - t0

    t0 = time.perf_counter()
    pa, pb = embed(model, tok, paws_a, how), embed(model, tok, paws_b, how)
    paws_pred = (pa * pb).sum(-1).tolist()
    paws_secs = time.perf_counter() - t0

    f1, thr = best_f1(paws_pred, paws_gold)
    bi_results.append({
        "model": name, "type": "bi-encoder", "params_m": params_m,
        "sts_spearman": round(spearman(sts_gold, sts_pred), 4),
        "sts_pearson": round(pearson(sts_gold, sts_pred), 4),
        "paws_ap": round(average_precision(paws_pred, paws_gold), 4),
        "paws_f1": round(f1, 4),
        "pairs_per_sec": round(N_STS / sts_secs, 1),
        "sts_scores": sts_pred,
    })
    show_kv({k: v for k, v in bi_results[-1].items() if k != "sts_scores"}, title=name)

    del model, tok, va, vb, pa, pb        # one model live at a time
    free_memory()

vram("after bi-encoders")
show_table([{k: v for k, v in r.items() if k != "sts_scores"} for r in bi_results]
           + [{"model": "word overlap (no model)", "type": "lexical", "params_m": 0,
               "sts_spearman": round(spearman(sts_gold, sts_overlap), 4),
               "sts_pearson": round(pearson(sts_gold, sts_overlap), 4),
               "paws_ap": round(ap, 4)}],
           title="Three bi-encoders against the lexical baseline",
           best=("sts_spearman", "paws_ap", "pairs_per_sec"),
           caption="a model that beats overlap on STS-B and matches it on PAWS has "
                   "learned overlap with extra steps")
```

```
                        all-MiniLM-L6-v2                        
                                                                
 model                                         all-MiniLM-L6-v2 
 type                                                bi-encoder 
 params_m                                                    22 
 sts_spearman                                            0.9219 
 sts_pearson                                             0.9159 
 paws_ap                                                 0.6487 
 paws_f1                                                 0.6589 
 pairs_per_sec                                         1,385.10 
                                                                
```

```
                       all-mpnet-base-v2                        
                                                                
 model                                        all-mpnet-base-v2 
 type                                                bi-encoder 
 params_m                                                   109 
 sts_spearman                                            0.9270 
 sts_pearson                                             0.9258 
 paws_ap                                                 0.6484 
 paws_f1                                                 0.6536 
 pairs_per_sec                                         1,771.10 
                                                                
```

```
                        bge-base-en-v1.5                        
                                                                
 model                                         bge-base-en-v1.5 
 type                                                bi-encoder 
 params_m                                                   109 
 sts_spearman                                            0.9083 
 sts_pearson                                             0.8948 
 paws_ap                                                 0.6189 
 paws_f1                                                 0.6477 
 pairs_per_sec                                         2,208.50 
                                                                
```

    VRAM after bi-encoders       0.01 GB allocated /  0.02 GB reserved

```
                                 Three bi-encoders against the lexical baseline                                 
                                                                                                                
 model                   type         params_m   sts_spearman   sts_pearson   paws_ap   paws_f1   pairs_per_sec 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 all-MiniLM-L6-v2        bi-encoder         22         0.9219        0.9159    0.6487    0.6589        1,385.10 
 all-mpnet-base-v2       bi-encoder        109         0.9270        0.9258    0.6484    0.6536        1,771.10 
 bge-base-en-v1.5        bi-encoder        109         0.9083        0.8948    0.6189    0.6477        2,208.50 
 word overlap (no        lexical             0         0.5070        0.5003    0.4552                           
 model)                                                                                                         
                                                                                                                
        a model that beats overlap on STS-B and matches it on PAWS has learned overlap with extra steps         
```

## 10. Cross-encoders: one pass per pair

Feed both sentences into the model together, separated by a special token, and read a single score off a regression head. Every token in sentence A can attend to every token in sentence B at every layer.

That cross-attention is the entire advantage. The model does not have to guess in advance what will matter about a sentence - it can compare “who did what to whom” between the pair directly. On PAWS, where the pairs differ only in argument structure, this is decisive.

**And it costs everything the bi-encoder saved.** No precomputation is possible, because the representation depends on both texts. Scoring N texts against each other is N(N-1)/2 forward passes. The cell measures this concretely on a 200-sentence set, where the bi-encoder does 200 encodings and a matrix multiply while the cross-encoder does 19,900 forward passes - and 200 sentences is a toy corpus.

**One API detail worth flagging:** `stsb-roberta-base` has `num_labels=1` and outputs a similarity in `[0, 1]` directly. Applying a softmax over a single logit gives 1.0 for everything, which produces a perfectly uniform and perfectly useless score. Always check `config.num_labels` before deciding how to read a cross-encoder’s head.

------------------------------------------------------------------------

``` python
from transformers import AutoModelForSequenceClassification

ce_name, ce_id, ce_params = CROSS_ENCODER
ce_tok = AutoTokenizer.from_pretrained(ce_id, cache_dir=HF_CACHE)
ce = AutoModelForSequenceClassification.from_pretrained(
    ce_id, dtype=dtype, cache_dir=HF_CACHE).to(device).eval()
print(f"{ce_name}: num_labels={ce.config.num_labels} "
      f"({'single regression output in [0,1]' if ce.config.num_labels == 1 else 'class logits'})")
vram("cross-encoder loaded")


@torch.inference_mode()
def cross_score(texts_a, texts_b, batch_size=64, max_length=128):
    "Score pairs jointly. tok(a, b) builds the pair encoding with the right separators."
    out = []
    for i in range(0, len(texts_a), batch_size):
        enc = ce_tok(texts_a[i:i + batch_size], texts_b[i:i + batch_size],
                     return_tensors="pt", padding=True, truncation=True,
                     max_length=max_length).to(ce.device)
        logits = ce(**enc).logits.float()
        # num_labels == 1 -> already a similarity; do NOT softmax a single logit.
        out.extend(logits[:, 0].tolist() if ce.config.num_labels == 1
                   else logits.softmax(-1)[:, -1].tolist())
    return out


t0 = time.perf_counter()
ce_sts = cross_score(sts_a, sts_b)
ce_sts_secs = time.perf_counter() - t0
ce_paws = cross_score(paws_a, paws_b)
ce_f1, ce_thr = best_f1(ce_paws, paws_gold)

show_kv({"STS-B Spearman": round(spearman(sts_gold, ce_sts), 4),
         "STS-B Pearson": round(pearson(sts_gold, ce_sts), 4),
         "PAWS average precision": round(average_precision(ce_paws, paws_gold), 4),
         "PAWS best F1": round(ce_f1, 4),
         "pairs / second": round(N_STS / ce_sts_secs, 1)},
        title=f"{ce_name} - the pair is encoded jointly")

cross_results = {
    "model": ce_name, "type": "cross-encoder", "params_m": ce_params,
    "sts_spearman": round(spearman(sts_gold, ce_sts), 4),
    "sts_pearson": round(pearson(sts_gold, ce_sts), 4),
    "paws_ap": round(average_precision(ce_paws, paws_gold), 4),
    "paws_f1": round(ce_f1, 4),
    "pairs_per_sec": round(N_STS / ce_sts_secs, 1),
    "sts_scores": ce_sts,
}

# The scaling wall, measured rather than asserted.
M = 200
corpus = sts_a[:M]
n_pairs = M * (M - 1) // 2
print(f"\nAll-pairs comparison over {M} sentences = {n_pairs:,} pairs")

bi_tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", cache_dir=HF_CACHE)
bi = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", dtype=dtype,
                               cache_dir=HF_CACHE).to(device).eval()
if device != "cpu":
    torch.cuda.synchronize()
t0 = time.perf_counter()
vecs = embed(bi, bi_tok, corpus, "mean")
sim_matrix = vecs @ vecs.T                       # every pair, in one matmul
if device != "cpu":
    torch.cuda.synchronize()
bi_all_secs = time.perf_counter() - t0
del bi, bi_tok, vecs
free_memory()

# Time a small slice of the cross-encoder's pairs and extrapolate - running all of them
# is the point being made, not something to actually sit through. SLICE is larger than
# the corpus on purpose (a rate wants more than M samples), so both sides wrap with % M.
SLICE = 400
pa = [corpus[i % M] for i in range(SLICE)]
pb = [corpus[(i + 7) % M] for i in range(SLICE)]
if device != "cpu":
    torch.cuda.synchronize()
t0 = time.perf_counter()
cross_score(pa, pb)
if device != "cpu":
    torch.cuda.synchronize()
ce_rate = SLICE / (time.perf_counter() - t0)

show_table([{"approach": "bi-encoder", "work": f"{M} encodings + 1 matmul",
             "seconds": round(bi_all_secs, 3), "measured": True},
            {"approach": "cross-encoder", "work": f"{n_pairs:,} forward passes",
             "seconds": round(n_pairs / ce_rate, 1),
             "measured": False}],
           title=f"All-pairs comparison over {M} sentences ({n_pairs:,} pairs)",
           lower_is_better=("seconds",), best=("seconds",),
           caption=f"{(n_pairs / ce_rate) / bi_all_secs:,.0f}x apart, and the ratio grows "
                   "linearly with corpus size. This is why production retrieval is "
                   "bi-encoder first, cross-encoder on the top-k")

del sim_matrix
free_memory()
vram("after cross-encoder")
```

    stsb-roberta-base: num_labels=1 (single regression output in [0,1])
    VRAM cross-encoder loaded    0.27 GB allocated /  0.27 GB reserved

```
        stsb-roberta-base - the pair is encoded jointly         
                                                                
 STS-B Spearman                                          0.9421 
 STS-B Pearson                                           0.8890 
 PAWS average precision                                  0.5834 
 PAWS best F1                                            0.6494 
 pairs / second                                        2,495.80 
                                                                
```

    All-pairs comparison over 200 sentences = 19,900 pairs

```
         All-pairs comparison over 200 sentences (19,900 pairs)         
                                                                        
 approach           work                            seconds   measured  
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 bi-encoder         200 encodings + 1 matmul         0.0550   True      
 cross-encoder      19,900 forward passes            6.9000   False     
                                                                        
 124x apart, and the ratio grows linearly with corpus size. This is why 
  production retrieval is bi-encoder first, cross-encoder on the top-k  
```

    VRAM after cross-encoder     0.27 GB allocated /  0.27 GB reserved

## 11. Head-to-head Benchmark

Four models, two datasets, the same metrics. Sections 8-10 produced the numbers; this collects and charts them.

**Read the two datasets as one result.** On STS-B every trained model clusters near 0.85 Spearman and the differences are within noise - the benchmark is saturated and the sentences are short and simple. On PAWS the same models separate sharply, and the ordering reflects the architecture rather than the parameter count.

Three things the pair of columns says that neither says alone:

- **Cross-attention is what handles argument structure.** The cross-encoder’s PAWS advantage comes from comparing tokens across the pair, which a pooled vector cannot represent.
- **STS-B rewards lexical overlap enough that it cannot detect the failure.** Compare each model’s STS-B score against the lexical baseline in section 8 to see how much headroom the benchmark actually has.
- **Throughput inverts the ranking.** The cross-encoder is the most accurate and the least deployable at scale, which is precisely why the standard architecture uses both.

At 600 pairs, Spearman carries roughly +/-0.02 and PAWS AP roughly +/-0.03 of sampling noise.

------------------------------------------------------------------------

``` python
import pandas as pd

results = bi_results + [cross_results]
df_results = pd.DataFrame([{k: v for k, v in r.items() if k != "sts_scores"}
                           for r in results]).sort_values("paws_ap", ascending=False)
show_table(
    df_results.to_dict("records"),
    title=f"STS-B ({N_STS} pairs) and PAWS ({N_PAWS} pairs)",
    best=("sts_spearman", "sts_pearson", "paws_ap", "paws_f1", "pairs_per_sec"),
    caption="STS-B is saturated and rewards word overlap; PAWS is where the models separate",
)
```

```
                                    STS-B (600 pairs) and PAWS (600 pairs)                                     
                                                                                                               
 model               type            params_m   sts_spearman   sts_pearson   paws_ap   paws_f1   pairs_per_sec 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 all-MiniLM-L6-v2    bi-encoder            22         0.9219        0.9159    0.6487    0.6589        1,385.10 
 all-mpnet-base-v2   bi-encoder           109         0.9270        0.9258    0.6484    0.6536        1,771.10 
 bge-base-en-v1.5    bi-encoder           109         0.9083        0.8948    0.6189    0.6477        2,208.50 
 stsb-roberta-base   cross-encoder        125         0.9421        0.8890    0.5834    0.6494        2,495.80 
                                                                                                               
                STS-B is saturated and rewards word overlap; PAWS is where the models separate                 
```

``` python
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org serves an expired TLS certificate from most nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"

from pyecharts import options as opts
from pyecharts.charts import Bar

names = [r["model"] for r in results] + ["word overlap (no model)"]
sts_scores = [r["sts_spearman"] for r in results] + [round(spearman(sts_gold, sts_overlap), 4)]
paws_scores = [r["paws_ap"] for r in results] + [round(ap, 4)]

bar = (
    Bar()
    .add_xaxis(names)
    .add_yaxis("STS-B Spearman x100", [round(s * 100, 1) for s in sts_scores])
    .add_yaxis("PAWS average precision x100", [round(s * 100, 1) for s in paws_scores])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"STS-B ({N_STS} pairs) vs PAWS ({N_PAWS} pairs)",
            subtitle="RTX 3060 - STS-B is saturated and rewards word overlap; "
                     "PAWS decorrelates overlap from meaning",
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
from pyecharts.charts import Scatter

# Accuracy against throughput. The cross-encoder sits top-left: most accurate, least
# deployable at scale - which is the whole argument for a two-stage pipeline.
scatter = Scatter()
scatter.add_xaxis([r["pairs_per_sec"] for r in results])
for r in results:
    scatter.add_yaxis(
        f"{r['model']} ({r['type']})",
        [[r["pairs_per_sec"], round(r["paws_ap"] * 100, 1)]],
        symbol_size=18, label_opts=opts.LabelOpts(is_show=False),
    )
scatter.set_global_opts(
    title_opts=opts.TitleOpts(
        title="PAWS average precision vs throughput",
        subtitle="pairs/second here understates the gap - a bi-encoder amortises "
                 "encoding across all pairs, a cross-encoder cannot",
    ),
    xaxis_opts=opts.AxisOpts(name="pairs / second", type_="log"),
    yaxis_opts=opts.AxisOpts(name="PAWS AP x100", type_="value"),
    tooltip_opts=opts.TooltipOpts(trigger="item"),
    legend_opts=opts.LegendOpts(pos_top="12%"),
)
scatter.render_notebook()
```

## 12. Interactive: compare your own pairs

Edit `MY_PAIRS` below and see the bi-encoder and the cross-encoder disagree. This is the cell people run on its own, so it opens with a `require(...)` guard naming what it needs from Setup rather than dying on a bare `NameError`.

The second half runs a **deduplication** demo over `MY_CORPUS`, which is the most common real use of this task, and it shows the threshold problem directly: change `DEDUP_THRESHOLD` and watch clusters merge or split. There is no correct value - it depends on what you count as a duplicate, which is a product decision.

The pairs worth trying:

- **Negation.** “The drug is effective” vs “The drug is not effective”. Both models will score these high. Similarity does not encode negation, and if your task depends on it you need NLI (`04_Zero_Shot_Classification`), not this.
- **Argument swap.** “The dog chased the cat” vs “The cat chased the dog”. Identical words, opposite meaning - the PAWS pattern. The cross-encoder handles it much better.
- **Paraphrase with no shared words.** “I want to cancel my plan” vs “How do I stop being billed?”. This is what embeddings buy you over keyword matching.
- **Same topic, different claim.** “Rates rose in March” vs “Rates fell in March”. High similarity, contradictory content - the failure mode that breaks semantic caching and naive fact-checking.
- **Watch the absolute scores.** Unrelated pairs still score 0.3-0.6 with a bi-encoder. There is no natural zero, which is why the threshold has to be calibrated on labelled examples from your own data.

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


require("device", "dtype", "HF_CACHE", "free_memory", "vram", "embed", "pool", "word_overlap")

import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

MY_PAIRS = [
    ("The drug is effective against the infection.",
     "The drug is not effective against the infection."),          # negation
    ("The dog chased the cat across the garden.",
     "The cat chased the dog across the garden."),                 # argument swap
    ("I want to cancel my plan.",
     "How do I stop being billed every month?"),                   # no shared words
    ("Interest rates rose sharply in March.",
     "Interest rates fell sharply in March."),                     # contradictory
    ("The server returned a 500 error.",
     "The API responded with an internal server error."),          # true paraphrase
]

MY_CORPUS = [
    "How do I reset my password?",
    "I forgot my password, how can I get back in?",
    "Password reset is not working for me.",
    "What are your business hours?",
    "When is your support team available?",
    "How much does the Pro plan cost?",
    "What is the price of the Pro subscription?",
]
DEDUP_THRESHOLD = 0.75   # no correct value - this is a product decision

# Re-runnable: this cell frees both models at the end, so guard the loads or a second
# shift-enter raises NameError.
if "my_bi" not in globals():
    my_bi_tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5", cache_dir=HF_CACHE)
    my_bi = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5", dtype=dtype,
                                      cache_dir=HF_CACHE).to(device).eval()
if "my_ce" not in globals():
    my_ce_tok = AutoTokenizer.from_pretrained("cross-encoder/stsb-roberta-base", cache_dir=HF_CACHE)
    my_ce = AutoModelForSequenceClassification.from_pretrained(
        "cross-encoder/stsb-roberta-base", dtype=dtype, cache_dir=HF_CACHE).to(device).eval()

a_list = [p[0] for p in MY_PAIRS]
b_list = [p[1] for p in MY_PAIRS]
va = embed(my_bi, my_bi_tok, a_list, "cls")
vb = embed(my_bi, my_bi_tok, b_list, "cls")
bi_scores = (va * vb).sum(-1).tolist()

with torch.inference_mode():
    enc = my_ce_tok(a_list, b_list, return_tensors="pt", padding=True, truncation=True,
                    max_length=128).to(my_ce.device)
    ce_scores = my_ce(**enc).logits.float()[:, 0].tolist()

print(f"{'bi':>6s} {'cross':>7s} {'overlap':>8s}   pair")
for (a, b), s_bi, s_ce in zip(MY_PAIRS, bi_scores, ce_scores):
    print(f"{s_bi:6.3f} {s_ce:7.3f} {word_overlap(a, b):8.3f}   {a[:52]}")
    print(f"{'':23s}   {b[:52]}")
print("\nWhere bi and cross disagree, the cross-encoder saw the two sentences together.")

# Deduplication: the most common production use, and the threshold problem made visible.
cv = embed(my_bi, my_bi_tok, MY_CORPUS, "cls")
sim = (cv @ cv.T).tolist()
clusters, assigned = [], set()
for i, text in enumerate(MY_CORPUS):
    if i in assigned:
        continue
    group = [i] + [j for j in range(i + 1, len(MY_CORPUS))
                   if j not in assigned and sim[i][j] >= DEDUP_THRESHOLD]
    assigned.update(group)
    clusters.append(group)

print(f"\ndeduplication at threshold {DEDUP_THRESHOLD}: "
      f"{len(MY_CORPUS)} texts -> {len(clusters)} clusters")
for group in clusters:
    print(f"  cluster: {MY_CORPUS[group[0]]}")
    for j in group[1:]:
        print(f"     +{sim[group[0]][j]:.3f}  {MY_CORPUS[j]}")

del my_bi, my_bi_tok, my_ce, my_ce_tok, va, vb, cv, enc
free_memory()
vram("final")
```

        bi   cross  overlap   pair
     0.868  -0.745    0.857   The drug is effective against the infection.
                              The drug is not effective against the infection.
     0.993  -0.049    1.000   The dog chased the cat across the garden.
                              The cat chased the dog across the garden.
     0.712  -0.035    0.077   I want to cancel my plan.
                              How do I stop being billed every month?
     0.824  -0.646    0.714   Interest rates rose sharply in March.
                              Interest rates fell sharply in March.
     0.764  -0.014    0.273   The server returned a 500 error.
                              The API responded with an internal server error.

    Where bi and cross disagree, the cross-encoder saw the two sentences together.

    deduplication at threshold 0.75: 7 texts -> 4 clusters
      cluster: How do I reset my password?
         +0.872  I forgot my password, how can I get back in?
         +0.866  Password reset is not working for me.
      cluster: What are your business hours?
      cluster: When is your support team available?
      cluster: How much does the Pro plan cost?
         +0.881  What is the price of the Pro subscription?
    VRAM final                   0.27 GB allocated /  0.27 GB reserved

## 13. Common Frameworks

Sentence similarity shares its stack almost entirely with feature extraction and ranking - the same encoders, the same indexes - so what is worth isolating here is the part specific to *pairs*: the cross-encoder that scores two texts jointly, the adversarial data that exposes what a bi-encoder cannot represent, and the threshold calibration that turns a similarity score into a decision.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [sentence-transformers](https://github.com/UKPLab/sentence-transformers) | modelling | Bi-encoders and `CrossEncoder` under one API, with correct pooling, prefixes and the training losses attached | Apache 2.0 | Default in production. Section 8 shows what it does for you; do not reimplement it |
| [transformers](https://github.com/huggingface/transformers) | modelling | The raw encoders when you need custom pooling or a checkpoint without an ST wrapper | Apache 2.0 | Research, or a new model before the ecosystem catches up |
| `MultipleNegativesRankingLoss` + [peft](https://github.com/huggingface/peft) | modelling | Contrastive fine-tuning on a few thousand (anchor, positive) pairs from your own logs | Apache 2.0 | Almost always worth it. The contrastive objective works; regressing on similarity scores is the intuitive choice and the weaker one |
| Hard-negative mining (via the base model itself) | data | The wrong answers your current model ranks highly, which is the highest-value training data in this task | Apache 2.0 | Always. A model that has never seen a hard negative has no reason to learn one - this is what closes the PAWS-style gap |
| [faiss](https://github.com/facebookresearch/faiss) / [hnswlib](https://github.com/nmslib/hnswlib) / [Qdrant](https://github.com/qdrant/qdrant) | data | Approximate nearest neighbours, because all-pairs comparison is quadratic and stops working quickly | MIT / Apache 2.0 | Deduplication or clustering over more than a few thousand items - which is the usual reason people are here |
| [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) (TEI) | inference runtime | A dedicated embedding and reranking server with dynamic batching, for both encoder types | Apache 2.0 | Serving. It handles cross-encoders too, which is the piece people usually hand-roll |
| [optimum](https://github.com/huggingface/optimum) + [ONNX Runtime](https://onnxruntime.ai/) | inference runtime | Quantised bi-encoders on CPU, cheap enough to embed an entire stream | Apache 2.0 / MIT | High-volume deduplication where a GPU is not justified |
| [BentoML](https://github.com/bentoml/BentoML) / [Ray Serve](https://github.com/ray-project/ray) | serving | The two-stage pipeline as one service: bi-encoder retrieves 100, cross-encoder reranks 10 | Apache 2.0 | Production. This gets most of the cross-encoder’s accuracy at close to the bi-encoder’s cost |
| [scipy](https://github.com/scipy/scipy) / [scikit-learn](https://github.com/scikit-learn/scikit-learn) + [PAWS](https://github.com/google-research-datasets/paws) | evaluation | Spearman correlation for STS, the threshold sweep, and an adversarial set of high-overlap non-paraphrases | BSD-3 | Always. A model evaluated only on a saturated benchmark like STS-B looks fine right up until production |

**The 2026 default stack** is a `sentence-transformers` bi-encoder fine-tuned on your own pairs with mined hard negatives, served through TEI, an ANN index for retrieval, a cross-encoder rerank on the top 100, and a threshold calibrated on 200 hand-labelled pairs from your data.

The common wrong turn is carrying a threshold across models or corpora. Similarity scores are not comparable between encoders, and a cutoff tuned elsewhere is a guess - sweep it against the relative cost of a false merge versus a missed duplicate. The second is using similarity where direction matters: cosine is symmetric and blind to negation, so “A entails B” and “A contradicts B” both look similar to it (see `04_Zero_Shot_Classification`).

------------------------------------------------------------------------

## 14. Going Further

- **Fine-tune a bi-encoder on your own pairs.** `sentence-transformers` with `MultipleNegativesRankingLoss` over a few thousand (anchor, positive) pairs from your logs takes under an hour on this box and typically beats moving to a model five times larger. Regressing on similarity scores (`CosineSimilarityLoss`) is the intuitive choice and the weaker one - the contrastive objective is what works.
- **Mine hard negatives.** Retrieve top-k with the base model, label the wrong ones, and train against them. This is the single highest-value data-work step in the task, and it is what closes the PAWS-style gap: a model that has never seen a hard negative has no reason to learn one.
- **Build the two-stage pipeline.** Bi-encoder retrieves 100 candidates, cross-encoder reranks to 10. This gets most of the cross-encoder’s accuracy at close to the bi-encoder’s cost, and it is the default architecture in `11_Text_Ranking`.
- **Calibrate the threshold, and re-calibrate per model.** Label 200 pairs from your own data, sweep the cutoff, and pick the operating point from the relative cost of a false merge versus a missed duplicate. A threshold from a different model or corpus is a guess.
- **Use NLI when direction matters.** Similarity is symmetric and blind to negation. “A entails B”, “A contradicts B” and “A is similar to B” are different questions - see `04_Zero_Shot_Classification`.
- **Evaluate on an adversarial set, always.** PAWS or a hand-built set of your own hard pairs. A model that only ever sees a saturated benchmark like STS-B will look fine right up to production.
- **Related notebooks.** `07_Feature_Extraction` (the vectors, pooling, and Matryoshka/quantization), `11_Text_Ranking` (the retrieve-then-rerank pipeline and its metrics), `04_Zero_Shot_Classification` (NLI, for entailment and contradiction), `03_Question_Answering` (RAG, where this pipeline feeds a reader), `06_Summarization` (embedding similarity as a generation metric).

------------------------------------------------------------------------

Back to top
