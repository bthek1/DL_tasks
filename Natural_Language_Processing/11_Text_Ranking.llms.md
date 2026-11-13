# Text Ranking

Retrieval and reranking, the machinery behind search and RAG: BM25 implemented from scratch, dense retrieval, why hybrid beats either alone, what a cross-encoder reranker buys, and nDCG/MRR/Recall measured end to end on BEIR SciFact.

Author

Benedict Thekkel

## 1. What is Text Ranking?

Text ranking orders a set of documents by relevance to a query. It is the retrieval half of search and of every RAG system, and it is where those systems actually fail - a reader cannot answer from a passage it never received.

**Input.** A query, and a corpus (thousands to billions of documents).

**Output.** An ordered list, usually with scores. Only the top few positions matter, which is why the metrics in section 4 are all rank-weighted.

**The pipeline, and every production system is some version of it:**

| Stage | Scores | Candidates | Cost per query | Typical model |
|----|----|----|----|----|
| **First-stage retrieval** | the whole corpus | -\> 100-1000 | sublinear (an index) | BM25, dense bi-encoder, or both |
| **Reranking** | the candidate list | -\> 10-50 | ~100 forward passes | cross-encoder |
| **Optional second rerank** | the top few | -\> 3-10 | a few LLM calls | listwise LLM reranker |

The reason for stages is arithmetic. A cross-encoder is the most accurate scorer available and needs one forward pass **per (query, document) pair** - over a 5-million-document corpus that is 5 million passes per query. An index lookup gets the candidate set to 100 first, and then the accurate model is affordable.

**Three families of first-stage retriever**, and the differences are not stylistic:

| Family | Representation | Strong at | Weak at |
|----|----|----|----|
| **Sparse** (BM25, SPLADE) | term weights over the vocabulary | exact terms, rare words, IDs, names | synonyms, paraphrase |
| **Dense** (bi-encoders) | one vector per text | paraphrase, semantics, cross-lingual | rare identifiers, exact match, out-of-domain |
| **Late interaction** (ColBERT) | one vector per **token** | both, largely | storage - roughly 100x a dense index |

**Sparse and dense fail on different queries, and that is the single most useful fact in this notebook.** BM25 cannot match “how do I get my money back” to a document about refunds; a dense retriever cannot reliably find document `XR-7741`. Combining them is not a marginal tuning trick - it is a structural fix, and section 11 measures it.

**Neighbouring tasks:**

| Task | How it differs | Notebook |
|----|----|----|
| Feature extraction | Produces the vectors dense retrieval indexes | `07_Feature_Extraction` |
| Sentence similarity | Scores a pair; ranking orders a corpus | `10_Sentence_Similarity` |
| Question answering | Reads the retrieved passage to produce an answer | `03_Question_Answering` |
| Summarization | Condenses what ranking selected | `06_Summarization` |
| Zero-shot classification | Ranks labels rather than documents | `04_Zero_Shot_Classification` |

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Web and site search | Search engines, e-commerce, docs | Query + index -\> ranked results | Sub-100 ms at enormous QPS; head queries dominate traffic |
| RAG retrieval | Every LLM application | Query + chunks -\> top-k for the reader | Recall@k is the ceiling on the whole system |
| Product search | E-commerce | Query + catalogue -\> ranked SKUs | Revenue-weighted relevance; filters and business rules on top |
| Code search | Developer tools | Natural language or code -\> files, symbols | Exact identifiers matter, so sparse is essential |
| Legal and patent discovery | Legal | Query + case corpus -\> ranked documents | **Recall** above all - missing a document is the failure |
| Enterprise knowledge search | Any large org | Query + intranet -\> ranked docs | Permissions filtering; freshness; conflicting versions |
| Recommendation ranking | Media, social | User context -\> ranked items | Personalisation; feedback loops; not purely textual |
| Scientific literature search | Research, pharma | Claim -\> supporting papers | Domain vocabulary; general embedders underperform |

What the BEIR score hides:

- **Retrieval is the ceiling of every RAG system.** End-to-end quality is roughly `P(right passage retrieved) x P(reader uses it correctly)`, and the first term is almost always the smaller one. Teams debug the prompt for weeks when the passage was never in the context.
- **Chunking decides retrieval quality more than the model does.** A chunk that separates a claim from its subject is unretrievable regardless of the encoder. Chunk size, overlap and boundary strategy are the highest-leverage knobs, and they are usually set once and never revisited.
- **Head queries and tail queries are different problems.** A small number of queries carry most traffic and can be cached or hand-tuned; the tail is where a semantic retriever earns its keep. Averaged metrics hide both.
- **Filtering and permissions interact badly with approximate nearest neighbours.** Retrieve-then-filter can return nothing; filter-then-retrieve can be slow. This is a real engineering problem that no benchmark measures.
- **Freshness and deletion are the operational reality.** An index is a cache of a corpus. Update lag, tombstoning deleted documents and re-embedding after a model upgrade consume more engineering time than model selection ever does.

------------------------------------------------------------------------

## 3. How Modern Text Ranking Works

1.  **Boolean and TF-IDF (1960s-1990s).** Term matching, then weighting terms by how rare they are. The vocabulary-mismatch problem - users and documents use different words for the same thing - was identified here and drives everything since.
2.  **BM25 (1994).** Robertson and Sparck Jones’s probabilistic weighting: term frequency with **saturation** (the tenth occurrence of a word adds almost nothing) and **length normalisation** (a long document should not win by containing everything). Thirty years later it is still the baseline every neural retriever is measured against, it needs no training data, and on out-of-domain corpora it still beats many dense models. Section 9 implements it in 30 lines.
3.  **Learning to rank (2005-2015).** LambdaMART and friends: gradient-boosted trees over hand-engineered features (BM25 score, PageRank, clicks, freshness), trained to optimise nDCG directly. This ran commercial search for a decade and remains how business signals get combined with relevance.
4.  **Dense retrieval (2019-2021).** **DPR** trained two BERT encoders contrastively on question-passage pairs and beat BM25 on open-domain QA. Semantics without exact terms became possible. **ANCE** and **RocketQA** showed that **hard negatives** - plausible wrong answers mined with the model itself - were what made it work; in-batch negatives alone are too easy.
5.  **The BEIR correction (2021).** BEIR evaluated retrievers **zero-shot** across 18 domains and found that dense models trained on MS MARCO often **lost to BM25** out of domain. This reframed the field: in-domain dense retrieval is excellent, out-of-domain generalisation was the hard problem, and hybrid was the pragmatic answer.
6.  **Late interaction and learned sparse (2020-2023).** **ColBERT** kept one vector per token and scored with MaxSim - most of a cross-encoder’s quality at index-time cost, for roughly 100x the storage. **SPLADE** learned *sparse* representations, expanding a document into weighted vocabulary terms, giving semantic matching that still runs on an inverted index.
7.  **Cross-encoder reranking as standard (2020-2026).** monoBERT, then the MiniLM ms-marco rerankers, then bge-reranker and mxbai-rerank. Reranking the top 100 is the cheapest large quality gain in the whole pipeline and it is the least-skipped step in serious systems.
8.  **LLM rerankers (2023-2026).** RankGPT-style **listwise** reranking - show the model 20 passages and ask for an ordering - beats pointwise cross-encoders because the model can compare candidates against each other. Expensive, and typically applied to the top 20-50 only.
9.  **Hybrid as the default (2022-2026).** Run BM25 and dense in parallel, fuse with **Reciprocal Rank Fusion** (no score calibration needed, just ranks), then rerank. This is the 2026 default stack, and it is what section 11 builds.

**Where it stands (mid-2026).** The consensus architecture is hybrid retrieval into a cross-encoder reranker. Dense-only is a reasonable simplification in a narrow domain with in-domain training data; BM25-only remains defensible when queries are keyword-shaped and there is no GPU. What has genuinely changed since 2021 is that the reranker is no longer optional and that the retrieval stage is expected to be fused rather than singular.

------------------------------------------------------------------------

## 4. Evaluation Metrics

All ranking metrics are **rank-weighted**: a relevant document at position 1 is worth far more than the same document at position 20, because users and LLM readers both look at the top.

**Recall@k** - the fraction of relevant documents that appear in the top k. **This is the metric that matters for RAG**, because it is the ceiling on the reader: if the passage is not in the top k, nothing downstream can recover it. Measure it at your actual k.

**MRR@k** (mean reciprocal rank) - `1/rank` of the *first* relevant document, averaged over queries. The right metric when there is one correct answer and the user stops at it (navigational search, question answering).

**nDCG@k** (normalised discounted cumulative gain) - the general-purpose metric, and the one BEIR reports:

$$\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+1)}, \qquad \text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$

Two ideas in one formula. The **discount** `1/log2(i+1)` makes rank 1 worth ~1.0, rank 5 worth ~0.39, rank 10 worth ~0.29 - a smooth, principled decay. The **normalisation** by the ideal DCG puts every query on `[0, 1]` regardless of how many relevant documents it has, which makes averaging across queries meaningful. It also handles graded relevance (`rel = 0, 1, 2, 3`), which Recall and MRR cannot.

**MAP** (mean average precision) - averages precision at every relevant hit. Binary relevance only; largely superseded by nDCG.

**Pitfalls, and the first one invalidates a lot of published comparisons:**

- **Unjudged documents are scored as irrelevant.** Test collections are pooled: only documents retrieved by the *original* participating systems were judged. A new model that finds a genuinely relevant unjudged document is punished for it. This systematically understates novel retrievers, and it is why “beats BM25 by 2 points” claims deserve scepticism.
- **Report k, and report it at the k you deploy.** nDCG@10 and nDCG@100 rank systems differently. A reranker that fixes the top 10 barely moves nDCG@100.
- **Recall@k of the retriever and nDCG@10 of the reranker measure different stages.** Optimising the wrong one is common: if Recall@100 is 0.70, no reranker can exceed 0.70 - the fix is retrieval, not reranking.
- **Latency belongs next to the score.** A reranker that adds 300 ms is a different product. Always report both.

The cell below implements Recall@k, MRR@k, nDCG@k and MAP directly - the discount and the ideal-DCG normalisation are the parts worth reading.

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


def recall_at_k(ranked_ids, relevant_ids, k=10):
    "Fraction of relevant documents found in the top k. The ceiling on any RAG reader."
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & set(relevant_ids)) / len(relevant_ids)


def mrr_at_k(ranked_ids, relevant_ids, k=10):
    "1/rank of the FIRST relevant hit. For when the user stops at the first good result."
    for i, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant_ids:
            return 1 / i
    return 0.0


def ndcg_at_k(ranked_ids, relevance, k=10):
    """Normalised discounted cumulative gain.

    `relevance` maps doc_id -> graded relevance (0 = irrelevant). The 1/log2(i+1)
    discount makes rank 1 worth ~1.0, rank 5 ~0.39, rank 10 ~0.29; dividing by the
    ideal DCG puts every query on [0, 1] so averaging across queries is meaningful.
    """
    dcg = sum((2 ** relevance.get(doc_id, 0) - 1) / math.log2(i + 1)
              for i, doc_id in enumerate(ranked_ids[:k], start=1))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / math.log2(i + 1) for i, rel in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def average_precision(ranked_ids, relevant_ids, k=100):
    "Precision at each relevant hit, averaged. Binary relevance only."
    if not relevant_ids:
        return 0.0
    hits, total = 0, 0.0
    for i, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id in relevant_ids:
            hits += 1
            total += hits / i
    return total / len(relevant_ids)


def evaluate(run, qrels, ks=(1, 5, 10, 100)):
    """Aggregate metrics over a run: {query_id: [ranked doc_ids]} against
    {query_id: {doc_id: relevance}}."""
    out = {}
    for k in ks:
        out[f"nDCG@{k}"] = round(sum(
            ndcg_at_k(run[q], qrels[q], k) for q in qrels) / len(qrels), 4)
        out[f"Recall@{k}"] = round(sum(
            recall_at_k(run[q], set(qrels[q]), k) for q in qrels) / len(qrels), 4)
    out["MRR@10"] = round(sum(mrr_at_k(run[q], set(qrels[q]), 10) for q in qrels) / len(qrels), 4)
    out["MAP"] = round(sum(average_precision(run[q], set(qrels[q])) for q in qrels) / len(qrels), 4)
    return out


# Toy example: three rankings of the same 5 candidates, one relevant document ("d2").
qrels_toy = {"q1": {"d2": 1}}
runs = {
    "relevant at rank 1": ["d2", "d1", "d3", "d4", "d5"],
    "relevant at rank 3": ["d1", "d3", "d2", "d4", "d5"],
    "relevant at rank 10": ["d1", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d0", "d2"],
    "not in top 10": ["d1", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d0", "dx"],
}
show_table([{"ranking": name,
             "nDCG@10": round(ndcg_at_k(ranked, qrels_toy["q1"], 10), 4),
             "MRR@10": round(mrr_at_k(ranked, {"d2"}, 10), 4),
             "Recall@10": round(recall_at_k(ranked, {"d2"}, 10), 4)}
            for name, ranked in runs.items()],
           title="One relevant document, four rankings of the same candidates",
           best=("nDCG@10", "MRR@10", "Recall@10"),
           caption="Recall@10 cannot tell rank 1 from rank 10 - it only asks whether the "
                   "document is there at all. That makes it right for a RAG retriever "
                   "(the reader sees all k) and wrong for a results page")
show_table([{"rank": i, "nDCG weight": round(1 / math.log2(i + 1), 3)}
            for i in (1, 2, 3, 5, 10, 20)],
           title="The nDCG discount by position", best=("nDCG weight",))
```

```
      One relevant document, four rankings of the same candidates       
                                                                        
 ranking                            nDCG@10       MRR@10      Recall@10 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 relevant at rank 1                  1.0000       1.0000         1.0000 
 relevant at rank 3                  0.5000       0.3333         1.0000 
 relevant at rank 10                 0.2891       0.1000         1.0000 
 not in top 10                       0.0000       0.0000         0.0000 
                                                                        
  Recall@10 cannot tell rank 1 from rank 10 - it only asks whether the  
 document is there at all. That makes it right for a RAG retriever (the 
            reader sees all k) and wrong for a results page             
```

```
                     The nDCG discount by position                      
                                                                        
                 rank                                       nDCG weight 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
                    1                                            1.0000 
                    2                                            0.6310 
                    3                                            0.5000 
                    5                                            0.3870 
                   10                                            0.2890 
                   20                                            0.2280 
                                                                        
```

## 5. Datasets

| Dataset | Contents | Corpus size | Queries | License | Typical use |
|----|----|----|----|----|----|
| [BEIR](https://github.com/beir-cellar/beir) | 18 retrieval datasets, **zero-shot** evaluation | varies | varies | mixed | The standard generalisation benchmark |
| [SciFact](https://huggingface.co/datasets/BeIR/scifact) | Scientific claims + abstracts that support them | 5.2k | 300 test | CC BY-NC 2.0 | Small BEIR task; used below |
| [NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus) | Nutrition/medical queries + PubMed docs | 3.6k | 323 | open | Small, hard, graded relevance |
| [MS MARCO passage](https://huggingface.co/datasets/microsoft/ms_marco) | Bing queries + passages | 8.8M | 500k train | non-commercial | The dense-retrieval training standard |
| [TREC Deep Learning](https://microsoft.github.io/msmarco/TREC-Deep-Learning) | MS MARCO with **deep** human judgements | 8.8M | 43-54/year | non-commercial | The most trustworthy judgements |
| [Natural Questions (open)](https://huggingface.co/datasets/google-research-datasets/nq_open) | Real Google queries + Wikipedia | 21M passages | 91k | CC BY-SA 3.0 | Open-domain QA retrieval |
| [HotpotQA](https://huggingface.co/datasets/BeIR/hotpotqa) | Multi-hop questions needing 2 documents | 5.2M | 7.4k | CC BY-SA 4.0 | Multi-hop retrieval |
| [FiQA](https://huggingface.co/datasets/BeIR/fiqa) | Financial opinion questions | 57k | 648 | open | Domain shift; dense models struggle |
| [MIRACL](https://huggingface.co/datasets/miracl/miracl) | Multilingual retrieval, 18 languages | 77M | 40k | Apache 2.0 | Multilingual evaluation |
| [LoTTE](https://huggingface.co/datasets/colbertv2/lotte) | Long-tail StackExchange topics | 2.4M | 3.5k | MIT | Out-of-domain, long-tail queries |

This notebook uses **BEIR SciFact**: 5,183 scientific abstracts and 300 test claims, each with one or two supporting abstracts. It is chosen deliberately for three reasons - it is small enough to index exhaustively on this box (no approximate nearest neighbours needed, so the measurements are exact), its judgements are reliable, and it is a **domain-shift** task where general-purpose dense models do not automatically beat BM25. That last property is what makes the hybrid result in section 11 meaningful rather than decorative.

Note that SciFact has sparse judgements (roughly 1.1 relevant documents per query), so Recall@k and nDCG@10 track each other closely here. On a dataset with graded relevance like NFCorpus they diverge, and nDCG earns its complexity.

Downloads land in `DL_tasks/datasets/` via `cache_dir` (gitignored).

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

The reference boards are [BEIR](https://github.com/beir-cellar/beir) and the [MTEB retrieval and reranking tabs](https://huggingface.co/spaces/mteb/leaderboard).

**First-stage retrievers:**

| Model | Params | License | Type | Best for |
|----|----|----|----|----|
| BM25 | 0 | \- | sparse, untrained | the baseline; exact terms, zero-shot; used below |
| [bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) | 109M | MIT | dense | strong general English; used below |
| [gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | 149M | Apache 2.0 | dense, 8192 ctx | long chunks |
| [e5-base-v2](https://huggingface.co/intfloat/e5-base-v2) | 109M | MIT | dense | needs query:/passage: prefixes |
| [bge-m3](https://huggingface.co/BAAI/bge-m3) | 568M | MIT | dense + sparse + ColBERT | all three signals from one model |
| [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 596M | Apache 2.0 | dense, instruction-aware | top MTEB retrieval at a runnable size |
| [SPLADE-v3](https://huggingface.co/naver/splade-v3) | 110M | CC BY-NC | learned sparse | inverted index + semantics |
| [jina-colbert-v2](https://huggingface.co/jinaai/jina-colbert-v2) | 560M | CC BY-NC | late interaction | near-reranker quality at index time, ~100x storage |

**Rerankers:**

| Model | Params | License | Type | Best for |
|----|----|----|----|----|
| [ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) | 22M | Apache 2.0 | cross-encoder | the cheap default; used below |
| [ms-marco-MiniLM-L12-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2) | 33M | Apache 2.0 | cross-encoder | a little better, still cheap |
| [bge-reranker-base / v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3) | 278M / 568M | Apache 2.0 | cross-encoder | best open multilingual reranking |
| [mxbai-rerank-base-v2](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v2) | 500M | Apache 2.0 | cross-encoder | strong 2025 reranker |
| [monoT5-base](https://huggingface.co/castorini/monot5-base-msmarco) | 220M | Apache 2.0 | seq2seq reranker | robust zero-shot |
| RankGPT-style listwise | \- | proprietary | LLM | top-20 reordering; best quality, highest cost |

**How to choose.** Start with **BM25 + a dense bi-encoder fused by RRF, then a MiniLM cross-encoder over the top 100**. That stack is cheap, needs no training data, and gets most of the available quality. Add training data only when you have query logs: fine-tuning the bi-encoder on your own (query, relevant-passage) pairs with hard negatives is worth more than any model swap. Reach for ColBERT or an LLM reranker when retrieval quality is the product and the corpus is small enough to afford them.

------------------------------------------------------------------------

## 7. Setup

Everything loads through Hugging Face `transformers` - no vendor packages, and BM25 is implemented inline rather than pulled from `rank_bm25`, because it is 30 lines and the formula is the content.

Package roles:

- `transformers` + `torch` - the dense bi-encoder and the cross-encoder reranker
- `accelerate` - `device_map` placement
- `datasets` - BEIR SciFact corpus, queries and qrels
- `pandas` + `pyecharts` - the benchmark table and charts
- `rich` - the result tables. It renders to HTML inside Jupyter, so the tables survive into the published docs; `show_table` / `show_kv` / `rule` are defined in the first code cell of section 4.

Three details that decide correctness:

- **The corpus is indexed exhaustively here.** 5,183 documents fit in a single matrix, so dense retrieval is an exact matrix multiply and every number below is exact. At a million-plus documents you need approximate nearest neighbours (FAISS, HNSW, a vector database), which introduces its own recall loss - measure it rather than assuming it away.
- **BEIR documents are `title` + `text`.** Concatenating them is the standard preprocessing and it matters: titles carry disproportionate signal, and dropping them costs several points.
- **Cross-encoders trained on MS MARCO output an unbounded relevance logit**, not a probability. Use it for ordering, never as a calibrated confidence, and never softmax across documents - the scores are pointwise and independent.

------------------------------------------------------------------------

``` python
# Everything runs through Hugging Face transformers - no vendor packages, and BM25 is
# implemented inline rather than imported from rank_bm25.
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

# BEIR SciFact: 5,183 scientific abstracts, 300 test claims with supporting documents.
corpus_ds = load_dataset("BeIR/scifact", "corpus", split="corpus", cache_dir=HF_CACHE)
queries_ds = load_dataset("BeIR/scifact", "queries", split="queries", cache_dir=HF_CACHE)
qrels_ds = load_dataset("BeIR/scifact-qrels", split="test", cache_dir=HF_CACHE)

# BEIR documents are title + text; concatenating is standard and titles carry real signal.
doc_ids = [r["_id"] for r in corpus_ds]
doc_texts = [f"{r['title']} {r['text']}".strip() for r in corpus_ds]
doc_pos = {d: i for i, d in enumerate(doc_ids)}

query_text = {r["_id"]: r["text"] for r in queries_ds}

# qrels: {query_id: {doc_id: graded relevance}}
qrels = {}
for r in qrels_ds:
    if str(r["corpus-id"]) in doc_pos:
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])

QUERY_IDS = sorted(qrels, key=int)
QUERIES = [query_text[q] for q in QUERY_IDS]

print(f"corpus  {len(doc_ids):,} documents")
print(f"queries {len(QUERY_IDS):,} test claims with judgements")
print(f"judged  {sum(len(v) for v in qrels.values()):,} (query, doc) pairs, "
      f"{sum(len(v) for v in qrels.values()) / len(qrels):.2f} relevant per query")
print(f"mean document length: {sum(len(t.split()) for t in doc_texts) / len(doc_texts):.0f} words\n")
print("QUERY:", QUERIES[0])
gold0 = list(qrels[QUERY_IDS[0]])[0]
print("GOLD :", doc_texts[doc_pos[gold0]][:260], "...")
```

    corpus  5,183 documents
    queries 300 test claims with judgements
    judged  339 (query, doc) pairs, 1.13 relevant per query
    mean document length: 215 words

    QUERY: 0-dimensional biomaterials show inductive properties.
    GOLD : New opportunities: the use of nanotechnologies to manipulate and track stem cells. Nanotechnologies are emerging platforms that could be useful in measuring, understanding, and manipulating stem cells. Examples include magnetic nanoparticles and quantum dots f ...

## 8. BM25 from scratch

Thirty years old, no training data, no GPU, and still the baseline every neural retriever is measured against. It is worth implementing because the formula encodes two genuinely good ideas that neural models had to relearn.

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

**Term-frequency saturation** (the `k1` term). A word appearing ten times should not score ten times a word appearing once - the marginal information of each repeat falls off. The fraction saturates toward `k1 + 1`, so extra occurrences add progressively less. Plain TF-IDF has no such bound and can be gamed by repetition.

**Length normalisation** (the `b` term). A long document contains more words and would otherwise win everything. Dividing the term frequency by the document’s length relative to the average corrects this. `b = 0.75` partially normalises, which works better in practice than either extreme.

**IDF** weights rare terms up: a query term appearing in 5 documents is far more discriminative than one appearing in 5,000. This is precisely the property dense retrievers lack, and it is why BM25 finds `XR-7741` and a bi-encoder does not.

The implementation below is exhaustive (score every document per query), which is fine at 5,183 documents. Real deployments use an inverted index and only touch documents containing a query term - the same scores, sublinear cost.

------------------------------------------------------------------------

``` python
import math
import re
from collections import Counter, defaultdict

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in", "is",
    "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with",
}


def tokenize(text):
    "Lowercase alphanumeric tokens, stopwords removed. No stemming - keep it legible."
    return [t for t in re.findall(r"[a-z0-9]+", text.lower())
            if t not in STOPWORDS and len(t) > 1]


class BM25:
    "Okapi BM25 with term-frequency saturation (k1) and length normalisation (b)."

    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in documents]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avgdl = sum(self.lengths) / len(self.lengths)
        # Inverted index: term -> [(doc index, term frequency)]
        self.postings = defaultdict(list)
        for i, counts in enumerate(self.docs):
            for term, freq in counts.items():
                self.postings[term].append((i, freq))
        n = len(self.docs)
        # Robertson-Sparck Jones IDF: rare terms are far more discriminative.
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.postings.items()}

    def score(self, query):
        "Scores for every document. Only documents containing a query term are touched."
        scores = [0.0] * len(self.docs)
        for term in tokenize(query):
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for i, freq in self.postings[term]:
                norm = 1 - self.b + self.b * self.lengths[i] / self.avgdl
                scores[i] += idf * freq * (self.k1 + 1) / (freq + self.k1 * norm)
        return scores

    def search(self, query, k=100):
        "Top-k document indices by BM25 score."
        scores = self.score(query)
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:k], scores


t0 = time.perf_counter()
bm25 = BM25(doc_texts)
index_secs = time.perf_counter() - t0
print(f"indexed {len(doc_texts):,} documents in {index_secs:.1f}s "
      f"({len(bm25.postings):,} unique terms, no GPU, no training data)\n")

# The IDF intuition, made concrete.
show_table([{"term": t, "IDF": round(bm25.idf[t], 2),
             "documents containing it": len(bm25.postings[t])}
            for t in ["cells", "protein", "microrna", "1", "assessed"] if t in bm25.idf],
           title="IDF: rarer terms are more discriminative", best=("IDF",),
           caption="this is precisely the property a dense retriever has no analogue for, "
                   "and why BM25 finds an identifier that an embedding cannot")

t0 = time.perf_counter()
bm25_run = {}
bm25_scores_all = {}
for qid, qtext in zip(QUERY_IDS, QUERIES):
    idxs, scores = bm25.search(qtext, k=100)
    bm25_run[qid] = [doc_ids[i] for i in idxs]
    bm25_scores_all[qid] = {doc_ids[i]: scores[i] for i in idxs}
bm25_secs = time.perf_counter() - t0

bm25_metrics = evaluate(bm25_run, qrels)
show_kv({"queries": len(QUERY_IDS), "seconds": round(bm25_secs, 1),
         "queries / second": round(len(QUERY_IDS) / bm25_secs, 0),
         **bm25_metrics},
        title="BM25 - no training data, no GPU, no embeddings to store")
```

    indexed 5,183 documents in 0.5s (35,675 unique terms, no GPU, no training data)

```
                IDF: rarer terms are more discriminative                
                                                                        
 term                      IDF                  documents containing it 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 cells                  0.9700                                    1,969 
 protein                1.4900                                    1,164 
 microrna               4.3600                                       66 
 assessed               2.8800                                      291 
                                                                        
 this is precisely the property a dense retriever has no analogue for,  
       and why BM25 finds an identifier that an embedding cannot        
```

```
    BM25 - no training data, no GPU, no embeddings to store     
                                                                
 queries                                                    300 
 seconds                                                 0.4000 
 queries / second                                        855.00 
 nDCG@1                                                  0.5467 
 Recall@1                                                0.5286 
 nDCG@5                                                  0.6401 
 Recall@5                                                0.7176 
 nDCG@10                                                 0.6616 
 Recall@10                                               0.7756 
 nDCG@100                                                0.6852 
 Recall@100                                              0.8759 
 MRR@10                                                  0.6312 
 MAP                                                     0.6252 
                                                                
```

## 9. Dense retrieval

Encode every document into a vector once, encode the query at search time, and rank by cosine similarity. The mechanics are `07_Feature_Extraction` plus an argsort; the interesting part is what changes relative to BM25.

**What dense retrieval gains.** It matches meaning without shared words. A query about “getting money back” retrieves a document about refunds; BM25 cannot, because it has no term in common. This is the vocabulary-mismatch problem that motivated fifty years of retrieval research, and dense retrieval is the first approach that genuinely addresses it.

**What it loses.** Rare identifiers, product codes, surnames and exact numbers. A 768-dimensional vector cannot preserve an arbitrary token, and it has no analogue of IDF - nothing says “this term is rare, therefore decisive”. BM25 gets these right by construction.

**Note the asymmetric setup.** BGE was trained with an instruction on queries and nothing on documents, so that is what the cell does. Skipping the query instruction costs real points and produces no error - the single most common mistake when wiring up a dense retriever.

SciFact is a **domain-shift** test: scientific abstracts, and the model was not trained on them. This is exactly the setting where BEIR found dense models losing to BM25, so treat whichever wins below as a fact about this corpus rather than a general ranking.

------------------------------------------------------------------------

``` python
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

dense_id = "BAAI/bge-base-en-v1.5"
d_tok = AutoTokenizer.from_pretrained(dense_id, cache_dir=HF_CACHE)
d_model = AutoModel.from_pretrained(dense_id, dtype=dtype,
                                    cache_dir=HF_CACHE).to(device).eval()
vram("bge-base loaded")

# BGE is asymmetric: an instruction on queries, nothing on documents. Omitting this
# costs real points and raises nothing.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@torch.inference_mode()
def encode(texts, batch_size=64, max_length=256, prefix=""):
    "CLS-pooled, L2-normalised vectors so a dot product is a cosine."
    out = []
    for i in range(0, len(texts), batch_size):
        enc = d_tok([prefix + t for t in texts[i:i + batch_size]], return_tensors="pt",
                    padding=True, truncation=True, max_length=max_length).to(device)
        v = d_model(**enc).last_hidden_state[:, 0]        # BGE uses CLS pooling
        out.append(F.normalize(v.float(), dim=-1))
    return torch.cat(out)


t0 = time.perf_counter()
doc_vecs = encode(doc_texts)
encode_secs = time.perf_counter() - t0
print(f"encoded {len(doc_texts):,} documents in {encode_secs:.1f}s "
      f"({len(doc_texts) / encode_secs:.0f} docs/s), index "
      f"{doc_vecs.numel() * 4 / 1e6:.1f} MB in float32")

t0 = time.perf_counter()
q_vecs = encode(QUERIES, prefix=QUERY_PREFIX)
sims = q_vecs @ doc_vecs.T                    # exhaustive: exact, no ANN approximation
top = sims.topk(100, dim=-1)
dense_secs = time.perf_counter() - t0

dense_run, dense_scores_all = {}, {}
for row, qid in enumerate(QUERY_IDS):
    idxs = top.indices[row].tolist()
    dense_run[qid] = [doc_ids[i] for i in idxs]
    dense_scores_all[qid] = {doc_ids[i]: float(s)
                             for i, s in zip(idxs, top.values[row].tolist())}

dense_metrics = evaluate(dense_run, qrels)
show_table([{"retriever": "BM25 (sparse)", **bm25_metrics},
            {"retriever": "bge-base (dense)", **dense_metrics}],
           title=f"BEIR SciFact, {len(QUERY_IDS)} queries - a domain-shift corpus",
           best=tuple(dense_metrics),
           caption="SciFact is scientific abstracts and neither model was trained on them; "
                   "this is the setting where BEIR found dense models losing to BM25")

# Where they disagree - the reason section 11 fuses them rather than picking one.
bm25_only = sum(1 for q in qrels
                if recall_at_k(bm25_run[q], set(qrels[q]), 10) > recall_at_k(dense_run[q], set(qrels[q]), 10))
dense_only = sum(1 for q in qrels
                 if recall_at_k(dense_run[q], set(qrels[q]), 10) > recall_at_k(bm25_run[q], set(qrels[q]), 10))
show_kv({"queries where BM25 beats dense at Recall@10": bm25_only,
         "queries where dense beats BM25 at Recall@10": dense_only,
         "queries where they tie": len(qrels) - bm25_only - dense_only},
        title="Per-query, not aggregate")
print("Both numbers are large. They fail on different queries - which is the whole")
print("argument for hybrid retrieval, and why the aggregate comparison is a distraction.")

del d_model, d_tok, sims
free_memory()
vram("after dense")
```

    VRAM bge-base loaded         0.23 GB allocated /  0.24 GB reserved
    encoded 5,183 documents in 11.4s (454 docs/s), index 15.9 MB in float32

```
                               BEIR SciFact, 300 queries - a domain-shift corpus                                
                                                                                                                
                                                               Recall@1              Recall@1                   
 retriever   nDCG@1   Recall@1   nDCG@5   Recall@5   nDCG@10          0   nDCG@100         00   MRR@10      MAP 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 BM25        0.5467     0.5286   0.6401     0.7176    0.6616     0.7756     0.6852     0.8759   0.6312   0.6252 
 (sparse)                                                                                                       
 bge-base    0.6100     0.5805   0.7125     0.8036    0.7362     0.8703     0.7584     0.9700   0.6981   0.6941 
 (dense)                                                                                                        
                                                                                                                
  SciFact is scientific abstracts and neither model was trained on them; this is the setting where BEIR found   
                                          dense models losing to BM25                                           
```

```
                    Per-query, not aggregate                    
                                                                
 queries where BM25 beats dense at Recall@10                 12 
 queries where dense beats BM25 at Recall@10                 44 
 queries where they tie                                     244 
                                                                
```

    Both numbers are large. They fail on different queries - which is the whole
    argument for hybrid retrieval, and why the aggregate comparison is a distraction.
    VRAM after dense             0.03 GB allocated /  0.03 GB reserved

## 10. Hybrid retrieval with Reciprocal Rank Fusion

Sparse and dense fail on different queries, so run both and combine. The question is how, and the answer is less obvious than it looks.

**Score fusion does not work out of the box.** BM25 produces unbounded positive scores whose scale depends on the query’s IDF; cosine similarity lives in `[-1, 1]`. Adding them means the BM25 score dominates arbitrarily. Normalising per query (min-max or z-score) helps but is unstable when one list is short or flat.

**Reciprocal Rank Fusion** sidesteps the problem by discarding the scores entirely and fusing **ranks**:

$$\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}$$

with `k = 60` by convention (from Cormack et al., 2009). A document at rank 1 in either list contributes `1/61`; at rank 50, `1/110`. Because only ranks enter, no calibration is needed, rankers of wildly different score scales combine cleanly, and adding a third ranker is a one-line change. The `k` constant damps the top of each list so that one ranker’s confident-but-wrong first result cannot dominate.

It is crude, it has one hyperparameter that essentially nobody tunes, and it is extremely hard to beat. It is the default fusion method in Elasticsearch, OpenSearch, Weaviate and Qdrant for exactly that reason.

------------------------------------------------------------------------

``` python
def reciprocal_rank_fusion(rankings, k=60):
    """Fuse ranked lists by rank, not by score.

    `rankings` is a list of ranked doc_id lists. Scores never enter, so rankers with
    incomparable score scales (BM25's unbounded sums, cosine's [-1,1]) combine cleanly.
    """
    scores = defaultdict(float)
    for ranked in rankings:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] += 1 / (k + rank)
    return sorted(scores, key=lambda d: -scores[d])


hybrid_run = {qid: reciprocal_rank_fusion([bm25_run[qid], dense_run[qid]])[:100]
              for qid in QUERY_IDS}
hybrid_metrics = evaluate(hybrid_run, qrels)

best_single = max(bm25_metrics["Recall@100"], dense_metrics["Recall@100"])
show_table([{"system": name, "nDCG@10": m["nDCG@10"], "Recall@10": m["Recall@10"],
             "Recall@100": m["Recall@100"], "MRR@10": m["MRR@10"], "MAP": m["MAP"]}
            for name, m in [("BM25", bm25_metrics), ("dense", dense_metrics),
                            ("hybrid RRF", hybrid_metrics)]],
           title="Fusing two retrievers that fail on different queries",
           best=("nDCG@10", "Recall@10", "Recall@100", "MRR@10", "MAP"),
           caption=f"hybrid Recall@100 gain over the better single retriever: "
                   f"{hybrid_metrics['Recall@100'] - best_single:+.4f}. Recall@100 is the "
                   "ceiling for the reranker in section 11 and for any reader downstream")

# Why naive score fusion is a bad idea, shown rather than asserted.
qid = QUERY_IDS[0]
bm25_vals = list(bm25_scores_all[qid].values())[:5]
dense_vals = list(dense_scores_all[qid].values())[:5]
print(f"\nscore scales for one query:")
print(f"  BM25  top-5: {[round(v, 2) for v in bm25_vals]}")
print(f"  dense top-5: {[round(v, 3) for v in dense_vals]}")
print("  adding these directly would let BM25 decide everything - hence rank fusion.")
```

```
          Fusing two retrievers that fail on different queries          
                                                                        
 system          nDCG@10     Recall@10    Recall@100    MRR@10      MAP 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 BM25             0.6616        0.7756        0.8759    0.6312   0.6252 
 dense            0.7362        0.8703        0.9700    0.6981   0.6941 
 hybrid RRF       0.7169        0.8261        0.9693    0.6887   0.6840 
                                                                        
   hybrid Recall@100 gain over the better single retriever: -0.0007.    
  Recall@100 is the ceiling for the reranker in section 11 and for any  
                           reader downstream                            
```

    score scales for one query:
      BM25  top-5: [9.68, 9.59, 9.45, 9.11, 8.55]
      dense top-5: [0.578, 0.571, 0.571, 0.552, 0.551]
      adding these directly would let BM25 decide everything - hence rank fusion.

## 11. Cross-encoder reranking

The last stage, and the cheapest large quality gain in the pipeline.

A cross-encoder encodes the (query, document) pair **jointly**, so every query token can attend to every document token at every layer. It does not have to compress the document into a vector before knowing what the query is, which is the structural limitation of any bi-encoder. On hard cases - subtle distinctions, negation, which entity did what - the difference is decisive.

The cost is that nothing can be precomputed: one forward pass per pair, every query. That is why it runs on the top 100 rather than on the corpus. The arithmetic here: 100 documents x 300 queries = 30,000 forward passes for the whole evaluation, versus 5,183 x 300 = 1.5 million if it were the retriever. On a real corpus of 5 million documents the second number is 1.5 billion, which is the reason the pipeline has stages at all.

**Two things to read carefully in the output:**

- **`Recall@100` does not move.** Reranking reorders the candidate list; it cannot add anything to it. If a relevant document was not retrieved, no reranker recovers it. This makes retrieval Recall@k the true ceiling, and it is the single most common misdiagnosis in RAG work - people add a reranker when their retriever’s Recall@100 is 0.6.
- **`nDCG@10` and `MRR@10` move a lot**, because the reranker is doing exactly what it is for: pulling the right document from rank 30 to rank 2.

The cell also sweeps the candidate depth, which is the parameter that actually trades quality against latency: reranking 200 candidates costs twice reranking 100 and usually gains little.

------------------------------------------------------------------------

``` python
from transformers import AutoModelForSequenceClassification

rr_id = "cross-encoder/ms-marco-MiniLM-L6-v2"
rr_tok = AutoTokenizer.from_pretrained(rr_id, cache_dir=HF_CACHE)
rr = AutoModelForSequenceClassification.from_pretrained(
    rr_id, dtype=dtype, cache_dir=HF_CACHE).to(device).eval()
print(f"{rr_id}: num_labels={rr.config.num_labels} "
      "(an unbounded relevance logit - use it for ordering, not as a probability)")
vram("reranker loaded")


@torch.inference_mode()
def rerank(query, candidate_ids, batch_size=64, max_length=320):
    "Score every (query, document) pair jointly and re-order by the logit."
    texts = [doc_texts[doc_pos[d]] for d in candidate_ids]
    scores = []
    for i in range(0, len(texts), batch_size):
        enc = rr_tok([query] * len(texts[i:i + batch_size]), texts[i:i + batch_size],
                     return_tensors="pt", padding=True, truncation=True,
                     max_length=max_length).to(device)
        scores.extend(rr(**enc).logits.float()[:, 0].tolist())
    order = sorted(range(len(candidate_ids)), key=lambda j: -scores[j])
    return [candidate_ids[j] for j in order]


# Rerank the hybrid candidates - the standard production configuration.
DEPTH = 100
t0 = time.perf_counter()
rerank_run = {qid: rerank(query_text[qid], hybrid_run[qid][:DEPTH]) for qid in QUERY_IDS}
rerank_secs = time.perf_counter() - t0
rerank_metrics = evaluate(rerank_run, qrels)

show_table([{"system": name, "nDCG@10": m["nDCG@10"], "Recall@10": m["Recall@10"],
             "Recall@100": m["Recall@100"], "MRR@10": m["MRR@10"]}
            for name, m in [("hybrid RRF", hybrid_metrics),
                            (f"+ rerank top-{DEPTH}", rerank_metrics)]],
           title=f"Reranking the top {DEPTH} with a cross-encoder "
                 f"({len(QUERY_IDS) * DEPTH / rerank_secs:.0f} pairs/s, "
                 f"{rerank_secs / len(QUERY_IDS) * 1000:.0f} ms/query)",
           best=("nDCG@10", "Recall@10", "Recall@100", "MRR@10"),
           caption=f"nDCG@10 {rerank_metrics['nDCG@10'] - hybrid_metrics['nDCG@10']:+.4f}, "
                   f"Recall@100 "
                   f"{rerank_metrics['Recall@100'] - hybrid_metrics['Recall@100']:+.4f} - "
                   "unchanged by construction, because reranking reorders the candidate "
                   "list and cannot add to it")

# Candidate depth: the real latency/quality knob.
depth_sweep = []
for depth in (10, 25, 50, 100):
    t0 = time.perf_counter()
    run = {qid: rerank(query_text[qid], hybrid_run[qid][:depth]) for qid in QUERY_IDS}
    secs = time.perf_counter() - t0
    m = evaluate(run, qrels, ks=(10,))
    depth_sweep.append({"depth": depth, "nDCG@10": m["nDCG@10"],
                        "ms_per_query": round(secs / len(QUERY_IDS) * 1000, 1)})
show_table(depth_sweep, title="Reranker candidate depth", best=("nDCG@10", "ms_per_query"),
           lower_is_better=("ms_per_query",),
           caption="latency is linear in depth and quality saturates - pick from this curve")

del rr, rr_tok
free_memory()
vram("after reranker")
```

    cross-encoder/ms-marco-MiniLM-L6-v2: num_labels=1 (an unbounded relevance logit - use it for ordering, not as a probability)
    VRAM reranker loaded         0.07 GB allocated /  0.08 GB reserved

```
 Reranking the top 100 with a cross-encoder (1143 pairs/s, 87 ms/query) 
                                                                        
 system                   nDCG@10     Recall@10     Recall@100   MRR@10 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 hybrid RRF                0.7169        0.8261         0.9693   0.6887 
 + rerank top-100          0.6934        0.8246         0.9693   0.6599 
                                                                        
nDCG@10 -0.0235, Recall@100 +0.0000 - unchanged by construction, because
       reranking reorders the candidate list and cannot add to it       
```

```
                        Reranker candidate depth                        
                                                                        
          depth                nDCG@10                     ms_per_query 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
             10                 0.6962                            10.40 
             25                 0.6990                            22.00 
             50                 0.6960                            43.50 
            100                 0.6934                            87.00 
                                                                        
latency is linear in depth and quality saturates - pick from this curve 
```

    VRAM after reranker          0.03 GB allocated /  0.03 GB reserved

## 12. Head-to-head Benchmark

Five systems on the same 300 SciFact queries, the same corpus, the same metrics. Sections 8-11 produced the numbers; this collects and charts them.

Read it as a pipeline rather than a competition:

- **BM25 is free** - no training, no GPU, no embeddings to store - and on a domain-shifted corpus like this it is not far off a general-purpose dense model. Any retrieval project that has not measured it does not know what its neural model is worth.
- **Dense finds what BM25 cannot**, and vice versa. The per-query comparison in section 9 shows both directions are common, which is what makes fusion productive rather than cosmetic.
- **Hybrid RRF raises the ceiling.** Recall@100 is the number that matters here, because it bounds every later stage.
- **The reranker converts recall into precision.** It moves nDCG@10 and MRR substantially and moves Recall@100 by exactly zero.

The chart plots nDCG@10 against Recall@100 so the two roles are visible: retrievers move right, rerankers move up.

At 300 queries with ~1.1 relevant documents each, nDCG@10 carries roughly +/-0.02 of sampling noise. And remember the pooling caveat from section 4 - unjudged relevant documents count as misses, which understates every system here, most of all the ones least like the systems that built the pool.

------------------------------------------------------------------------

``` python
import pandas as pd

SYSTEMS = [
    ("BM25 (sparse, no training)", bm25_metrics, round(bm25_secs / len(QUERY_IDS) * 1000, 1)),
    ("dense bge-base", dense_metrics, round(dense_secs / len(QUERY_IDS) * 1000, 1)),
    ("hybrid (RRF)", hybrid_metrics,
     round((bm25_secs + dense_secs) / len(QUERY_IDS) * 1000, 1)),
    (f"hybrid + rerank top-{DEPTH}", rerank_metrics,
     round((bm25_secs + dense_secs + rerank_secs) / len(QUERY_IDS) * 1000, 1)),
]

results = []
for name, m, ms in SYSTEMS:
    results.append({
        "system": name,
        "nDCG@10": m["nDCG@10"],
        "Recall@10": m["Recall@10"],
        "Recall@100": m["Recall@100"],
        "MRR@10": m["MRR@10"],
        "MAP": m["MAP"],
        "ms_per_query": ms,
    })

df_results = pd.DataFrame(results)
show_table(
    results,
    title=f"BEIR SciFact, {len(QUERY_IDS)} queries over {len(doc_ids):,} documents",
    best=("nDCG@10", "Recall@10", "Recall@100", "MRR@10", "MAP"),
    lower_is_better=("ms_per_query",),
    caption="retrieval sets Recall@100; reranking moves nDCG@10 and leaves Recall@100 alone",
)
```

```
                         BEIR SciFact, 300 queries over 5,183 documents                         
                                                                                                
 system                       nDCG@10   Recall@10   Recall@100   MRR@10      MAP   ms_per_query 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 BM25 (sparse, no training)    0.6616      0.7756       0.8759   0.6312   0.6252         1.2000 
 dense bge-base                0.7362      0.8703       0.9700   0.6981   0.6941         0.7000 
 hybrid (RRF)                  0.7169      0.8261       0.9693   0.6887   0.6840         1.9000 
 hybrid + rerank top-100       0.6934      0.8246       0.9693   0.6599   0.6532          89.40 
                                                                                                
         retrieval sets Recall@100; reranking moves nDCG@10 and leaves Recall@100 alone         
```

``` python
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org serves an expired TLS certificate from most nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"

from pyecharts import options as opts
from pyecharts.charts import Bar

bar = (
    Bar()
    .add_xaxis([r["system"] for r in results])
    .add_yaxis("nDCG@10 x100", [round(r["nDCG@10"] * 100, 1) for r in results])
    .add_yaxis("Recall@10 x100", [round(r["Recall@10"] * 100, 1) for r in results])
    .add_yaxis("Recall@100 x100", [round(r["Recall@100"] * 100, 1) for r in results])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"BEIR SciFact ({len(QUERY_IDS)} queries, {len(doc_ids):,} documents)",
            subtitle="RTX 3060 - Recall@100 is set by retrieval and untouched by reranking",
        ),
        yaxis_opts=opts.AxisOpts(name="score", min_=0, max_=100),
        xaxis_opts=opts.AxisOpts(name="system",
                                 axislabel_opts=opts.LabelOpts(rotate=18, font_size=9)),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="10%"),
    )
)
bar.render_notebook()
```

``` python
from pyecharts.charts import Scatter

# Retrievers move right (more recall), rerankers move up (better ordering of what is
# already there). Reading the pipeline this way makes the next fix obvious.
scatter = Scatter()
scatter.add_xaxis([round(r["Recall@100"] * 100, 1) for r in results])
for r in results:
    scatter.add_yaxis(
        r["system"],
        [[round(r["Recall@100"] * 100, 1), round(r["nDCG@10"] * 100, 1)]],
        symbol_size=18, label_opts=opts.LabelOpts(is_show=False),
    )
scatter.set_global_opts(
    title_opts=opts.TitleOpts(
        title="nDCG@10 vs Recall@100",
        subtitle="retrieval sets the horizontal position, reranking sets the vertical one",
    ),
    xaxis_opts=opts.AxisOpts(name="Recall@100 x100", type_="value"),
    yaxis_opts=opts.AxisOpts(name="nDCG@10 x100", type_="value"),
    tooltip_opts=opts.TooltipOpts(trigger="item"),
    legend_opts=opts.LegendOpts(pos_top="12%"),
)
scatter.render_notebook()
```

``` python
from pyecharts.charts import Line

# The reranker depth curve: latency is linear in depth, quality saturates.
line = (
    Line()
    .add_xaxis([str(d["depth"]) for d in depth_sweep])
    .add_yaxis("nDCG@10 x100", [round(d["nDCG@10"] * 100, 1) for d in depth_sweep])
    .add_yaxis("ms per query", [d["ms_per_query"] for d in depth_sweep])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="Reranker candidate depth: quality vs latency",
            subtitle="cost is linear in depth and quality saturates - pick from the curve",
        ),
        xaxis_opts=opts.AxisOpts(name="candidates reranked"),
        yaxis_opts=opts.AxisOpts(name="value"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="10%"),
    )
)
line.render_notebook()
```

## 13. Interactive: search your own corpus

Edit `MY_CORPUS` and `MY_QUERIES` below to run the full pipeline - BM25, dense, RRF fusion and cross-encoder reranking - over your own documents. This is the cell people run on its own, so it opens with a `require(...)` guard naming what it needs from Setup rather than dying on a bare `NameError`.

The corpus is small and deliberately constructed so each retriever’s characteristic failure is visible in one screen:

- **`"how do I get my money back"`** shares no words with the refund document. BM25 cannot find it; dense can. This is vocabulary mismatch, the problem that motivated fifty years of research.
- **`"XR-7741"`** is a rare identifier. BM25 nails it via IDF; dense flounders, because a 768-dimensional vector has no mechanism for “this exact token is decisive”.
- **`"laptop without a touchscreen"`** contains a negation. Neither retriever handles it - both will happily return the touchscreen document. Negation is a filter, not a similarity, and no amount of embedding quality fixes it.
- **RRF** should recover the first two, which is the point of hybrid.
- **The reranker** reorders what fusion produced; watch it fix ordering and never add a missing document.

Then try your own documents and queries. Watch where the pipeline breaks - and note that on a real corpus, chunking would be the next thing to fix, before any model.

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


require("device", "dtype", "HF_CACHE", "free_memory", "vram", "BM25",
        "reciprocal_rank_fusion")

import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

MY_CORPUS = [
    "To request a refund, open Settings > Billing and choose 'Cancel and refund'.",
    "Our laptops ship with a matte non-touch display by default.",
    "Touchscreen models are available in the Pro line only.",
    "The XR-7741 sensor requires firmware 2.3 or later to report humidity correctly.",
    "Password resets are sent to the address on file within five minutes.",
    "Shipping to the EU takes 3-5 business days and is tracked end to end.",
    "Annual plans are billed once a year and can be cancelled before renewal.",
    "The humidity sensor calibration procedure is documented in appendix C.",
]

MY_QUERIES = [
    "how do I get my money back",        # no shared words with the refund doc
    "XR-7741",                           # rare identifier - IDF territory
    "laptop without a touchscreen",      # negation - both retrievers fail
]
TOP_K = 3

# Re-runnable: this cell frees its models at the end, so guard the loads or a second
# shift-enter raises NameError.
if "my_dense" not in globals():
    my_d_tok = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5", cache_dir=HF_CACHE)
    my_dense = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5", dtype=dtype,
                                         cache_dir=HF_CACHE).to(device).eval()
if "my_rr" not in globals():
    my_rr_tok = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L6-v2",
                                              cache_dir=HF_CACHE)
    my_rr = AutoModelForSequenceClassification.from_pretrained(
        "cross-encoder/ms-marco-MiniLM-L6-v2", dtype=dtype,
        cache_dir=HF_CACHE).to(device).eval()


@torch.inference_mode()
def _encode(texts, prefix=""):
    enc = my_d_tok([prefix + t for t in texts], return_tensors="pt", padding=True,
                   truncation=True, max_length=256).to(device)
    return F.normalize(my_dense(**enc).last_hidden_state[:, 0].float(), dim=-1)


my_bm25 = BM25(MY_CORPUS)
my_vecs = _encode(MY_CORPUS)
ids = [str(i) for i in range(len(MY_CORPUS))]

for query in MY_QUERIES:
    sparse_idx, _ = my_bm25.search(query, k=len(MY_CORPUS))
    sparse_ids = [str(i) for i in sparse_idx]
    qv = _encode([query], prefix="Represent this sentence for searching relevant passages: ")
    dense_ids = [str(i) for i in (qv @ my_vecs.T)[0].argsort(descending=True).tolist()]
    fused = reciprocal_rank_fusion([sparse_ids, dense_ids])

    with torch.inference_mode():
        cands = fused[:5]
        enc = my_rr_tok([query] * len(cands), [MY_CORPUS[int(i)] for i in cands],
                        return_tensors="pt", padding=True, truncation=True,
                        max_length=320).to(device)
        rr_scores = my_rr(**enc).logits.float()[:, 0].tolist()
    reranked = [c for _, c in sorted(zip(rr_scores, cands), key=lambda p: -p[0])]

    print(f"QUERY: {query}")
    for label, order in [("BM25", sparse_ids), ("dense", dense_ids),
                         ("hybrid RRF", fused), ("+ rerank", reranked)]:
        print(f"  {label:12s}")
        for i in order[:TOP_K]:
            print(f"     {MY_CORPUS[int(i)][:74]}")
    print()

print("Query 1: BM25 misses (no shared words), dense finds it.")
print("Query 2: BM25 finds it via IDF, dense struggles with the identifier.")
print("Query 3: both return the touchscreen document - negation is a filter, not a")
print("         similarity, and no retriever here handles it.")

del my_dense, my_d_tok, my_rr, my_rr_tok, my_vecs
free_memory()
vram("final")
```

    QUERY: how do I get my money back
      BM25        
         To request a refund, open Settings > Billing and choose 'Cancel and refund
         Our laptops ship with a matte non-touch display by default.
         Touchscreen models are available in the Pro line only.
      dense       
         To request a refund, open Settings > Billing and choose 'Cancel and refund
         Annual plans are billed once a year and can be cancelled before renewal.
         Shipping to the EU takes 3-5 business days and is tracked end to end.
      hybrid RRF  
         To request a refund, open Settings > Billing and choose 'Cancel and refund
         Touchscreen models are available in the Pro line only.
         Our laptops ship with a matte non-touch display by default.
      + rerank    
         To request a refund, open Settings > Billing and choose 'Cancel and refund
         Shipping to the EU takes 3-5 business days and is tracked end to end.
         Annual plans are billed once a year and can be cancelled before renewal.

    QUERY: XR-7741
      BM25        
         The XR-7741 sensor requires firmware 2.3 or later to report humidity corre
         To request a refund, open Settings > Billing and choose 'Cancel and refund
         Our laptops ship with a matte non-touch display by default.
      dense       
         The XR-7741 sensor requires firmware 2.3 or later to report humidity corre
         Password resets are sent to the address on file within five minutes.
         Shipping to the EU takes 3-5 business days and is tracked end to end.
      hybrid RRF  
         The XR-7741 sensor requires firmware 2.3 or later to report humidity corre
         Password resets are sent to the address on file within five minutes.
         Our laptops ship with a matte non-touch display by default.
      + rerank    
         The XR-7741 sensor requires firmware 2.3 or later to report humidity corre
         Password resets are sent to the address on file within five minutes.
         Our laptops ship with a matte non-touch display by default.

    QUERY: laptop without a touchscreen
      BM25        
         Touchscreen models are available in the Pro line only.
         To request a refund, open Settings > Billing and choose 'Cancel and refund
         Our laptops ship with a matte non-touch display by default.
      dense       
         Our laptops ship with a matte non-touch display by default.
         Touchscreen models are available in the Pro line only.
         Password resets are sent to the address on file within five minutes.
      hybrid RRF  
         Touchscreen models are available in the Pro line only.
         Our laptops ship with a matte non-touch display by default.
         To request a refund, open Settings > Billing and choose 'Cancel and refund
      + rerank    
         Our laptops ship with a matte non-touch display by default.
         Touchscreen models are available in the Pro line only.
         The XR-7741 sensor requires firmware 2.3 or later to report humidity corre

    Query 1: BM25 misses (no shared words), dense finds it.
    Query 2: BM25 finds it via IDF, dense struggles with the identifier.
    Query 3: both return the touchscreen document - negation is a filter, not a
             similarity, and no retriever here handles it.
    VRAM final                   0.03 GB allocated /  0.03 GB reserved

## 14. Common Frameworks

Ranking is the task with the most infrastructure per unit of modelling, because a production retriever is three systems at once: a sparse index, a dense index, and a reranker, each with its own scaling behaviour. It is also the task where the oldest tool on the list - BM25, from 1994 - is still in every serious 2026 stack, because it catches exactly what dense retrieval misses.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [sentence-transformers](https://github.com/UKPLab/sentence-transformers) | modelling | Bi-encoders and cross-encoders with the right pooling and prefixes, plus the contrastive training loop | Apache 2.0 | Default for both stages. Fine-tuning on query logs beats any model swap |
| [transformers](https://github.com/huggingface/transformers) | modelling | The raw cross-encoders, the late-interaction models, and any LLM used as a listwise reranker | Apache 2.0 | Custom scoring, or a RankGPT-style prompt over the top 20 |
| [rerankers](https://github.com/AnswerDotAI/rerankers) | modelling | One interface over cross-encoders, ColBERT, FlashRank and LLM rerankers, so you can swap them in a line | Apache 2.0 | Comparing reranking approaches, which is otherwise four different APIs |
| BM25: [Pyserini](https://github.com/castorini/pyserini) / [rank_bm25](https://github.com/dorianbrown/rank_bm25) / [Elasticsearch](https://github.com/elastic/elasticsearch) / [OpenSearch](https://github.com/opensearch-project/OpenSearch) | data | The sparse half, plus RRF fusion built in on the two search engines | Apache 2.0 / Elastic License | Always. Hybrid beats dense alone, BM25 costs nothing to run, and it is what finds rare identifiers and exact codes |
| [faiss](https://github.com/facebookresearch/faiss) / [Qdrant](https://github.com/qdrant/qdrant) / [Vespa](https://github.com/vespa-engine/vespa) / pgvector | data | The dense half at scale, with metadata filtering - and in Vespa’s case sparse, dense and late interaction in one engine | MIT / Apache 2.0 | Past the exhaustive matmul of section 9. Measure the recall your approximate index gives up; it is rarely zero |
| Chunking via [Docling](https://github.com/docling-project/docling) / [LlamaIndex](https://github.com/run-llama/llama_index) node parsers | data | Structure-aware splitting with overlap and parent-document retrieval | MIT | Always. A claim separated from its subject by a chunk boundary is unretrievable at any model size |
| [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) (TEI) + [vLLM](https://github.com/vllm-project/vllm) | inference runtime | Batched embedding and reranking, and batched generation when the reranker is an LLM | Apache 2.0 | Serving. Reranking is the latency-critical stage and TEI is built for it |
| [Haystack](https://github.com/deepset-ai/haystack) / [LlamaIndex](https://github.com/run-llama/llama_index) | orchestration | Retrieve, fuse, rerank, filter and cite as a declarative pipeline, with permissions and fallbacks in one place | Apache 2.0 / MIT | Building the system. Pre-filter versus post-filter with an ANN index is a decision that belongs somewhere explicit |
| [ranx](https://github.com/AmenRa/ranx) / [BEIR](https://github.com/beir-cellar/beir) / trec_eval | evaluation | nDCG@k, MRR, Recall@k, and statistical significance testing between runs | MIT / Apache 2.0 | Always. Recall@k is the ceiling on everything downstream, and it is the first number to measure |

**The 2026 default stack** is hybrid retrieval - a fine-tuned bi-encoder through TEI plus BM25 - fused with RRF, a cross-encoder reranking the top 100, and ranx reporting nDCG and Recall separately. A listwise LLM reranker on the top 20 when the budget allows, because comparing candidates against each other beats scoring them independently.

The common wrong turn is improving the reranker while Recall@100 is the bottleneck. If the right document is not retrieved, no reranker, prompt or reader can recover it - this single diagnostic redirects more misspent RAG effort than any other. The second is dropping BM25 because the dense model scores better on a benchmark: the benchmark queries are natural language, and your users paste error codes and part numbers.

------------------------------------------------------------------------

## 15. Going Further

- **Measure Recall@k before touching anything else.** It is the ceiling on your whole system. If Recall@100 is 0.65, a better reranker, a better prompt and a bigger reader are all wasted effort - fix retrieval. This single diagnostic redirects more misspent RAG work than any other.
- **Fix chunking before you fix models.** 300-500 tokens with ~20% overlap is a sane default; retrieve on the chunk and pass the parent section to the reader. A claim separated from its subject by a chunk boundary is unretrievable at any model size.
- **Fine-tune the bi-encoder on your query logs.** `sentence-transformers` with `MultipleNegativesRankingLoss` over a few thousand (query, clicked-document) pairs beats any model swap, and **hard negatives** - wrong documents your current retriever ranks highly - are the ingredient that makes it work.
- **Keep BM25.** Even with an excellent dense model, hybrid is better and BM25 costs nothing to run. Elasticsearch and OpenSearch give you both plus RRF out of the box.
- **Try a listwise LLM reranker on the top 20.** RankGPT-style prompting - show the model the candidates and ask for an ordering - beats pointwise cross-encoders because the model compares candidates against each other. Expensive, so use it on a short list.
- **Handle filters and permissions deliberately.** Pre-filtering with an ANN index can silently return too few results; post-filtering can return none. Test it with restrictive filters before it reaches production.
- **Consider ColBERT for small, high-value corpora.** Late interaction gets close to cross-encoder quality at index time. The storage cost (~100x dense) rules it out at web scale and is irrelevant for 100k documents.
- **Related notebooks.** `07_Feature_Extraction` (the vectors, pooling, quantization and index-size arithmetic), `10_Sentence_Similarity` (bi- vs cross-encoders and the metrics for pairs), `03_Question_Answering` (the reader this pipeline feeds, and RAG end to end), `02_Table_Question_Answering` (schema retrieval as a ranking problem), `06_Summarization` (condensing what ranking selected).

------------------------------------------------------------------------

Back to top
