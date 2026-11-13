# Question Answering

Answering questions from a passage of text: extractive span models versus generative readers, why “no answer” is the hard half of the task, how EM and token-F1 actually work, and runnable code that puts four models on the same SQuAD 2.0 sample.

Author

Benedict Thekkel

## 1. What is Question Answering?

Question answering maps a question - and usually a passage - to an answer. The single most useful distinction is **where the answer comes from**, because it decides the architecture, the metric and the failure modes:

| Variant | Input | Output | Typical model |
|----|----|----|----|
| **Extractive** (reading comprehension) | question + context passage | a **span** of the passage, or “no answer” | encoder with start/end heads (BERT, DeBERTa) |
| **Abstractive / generative** | question + context | free text, may paraphrase | seq2seq or decoder LLM |
| **Open-domain (retrieval-augmented)** | question only | answer + cited passages | retriever + reader; this is RAG |
| **Closed-book** | question only | answer from parameters alone | LLM, no retrieval |
| **Multi-hop** | question + a corpus | answer requiring 2+ passages chained | iterative retrieve-and-read |

**The extractive formulation is worth understanding even in an LLM world**, because it is what makes a QA system auditable. The model outputs two integers - the start and end token of the answer - so the answer is a *pointer into the source*. It cannot hallucinate: a wrong answer is still a real quote from the document. Systems that must justify themselves (legal, clinical, compliance) still ship this way.

**How a span head works.** The encoder produces one vector per token. Two learned vectors, `S` and `E`, score each token as a start and as an end:

$$P_{\text{start}}(i) = \frac{e^{S \cdot h_i}}{\sum_j e^{S \cdot h_j}} \qquad P_{\text{end}}(j) = \frac{e^{E \cdot h_j}}{\sum_k e^{E \cdot h_k}}$$

Decoding picks the valid span `(i, j)` with `i <= j` and `j - i < max_answer_len` maximising `P_start(i) * P_end(j)`. That is the entire task-specific machinery: two vectors on top of a frozen architecture.

**Answerability is the hard half.** SQuAD 1.1 guaranteed every question had an answer, so models learned to always point somewhere. SQuAD 2.0 added 53k unanswerable questions written to *look* answerable, and scores fell off a cliff. The standard fix is the null score: the span `(CLS, CLS)` competes with every real span, and you answer only when the best real span beats it by a threshold. Tuning that threshold is a real decision with a real precision/recall trade-off, not a detail.

**Neighbouring tasks:**

| Task | How it differs | Notebook |
|----|----|----|
| Table question answering | Source is a structured table | `02_Table_Question_Answering` |
| Zero-shot classification | Fixed label set, not a span | `04_Zero_Shot_Classification` |
| Text ranking | Finds the passage; QA reads it | `11_Text_Ranking` |
| Summarization | Condenses a document, no question | `06_Summarization` |
| Text generation | Open-ended, no grounding requirement | `08_Text_Generation` |
| Visual document QA | Question over a page image | `Multimodal/05_Document_Question_Answering` |

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Enterprise document search | Any large org (Glean, Copilot, Vertex Search) | Question + retrieved chunks -\> answer + citations | Grounding and citation accuracy; stale-document detection |
| Customer self-service | SaaS, telco, retail | Question + help-centre articles -\> answer | Deflection rate vs wrong-answer cost; must abstain, not guess |
| Clinical evidence lookup | Healthcare | Clinician question + guidelines/notes -\> answer + source | Zero tolerance for fabrication; on-prem; must cite the paragraph |
| Legal and contract review | Legal | “What is the termination notice period?” + contract -\> span | Extractive by requirement - the answer must be a literal quote |
| Financial filings analysis | Finance | Question + 10-K -\> figure or statement | Numeric exactness; long documents (300+ pages) |
| Voice assistants | Consumer devices | Spoken question -\> short factual answer | End-to-end latency budget shared with ASR and TTS |
| Search engine snippets | Web search | Query + top page -\> highlighted span | Throughput at web scale; a 100M encoder, not an LLM |
| Compliance QA over policy | Regulated industries | Question + policy corpus -\> answer + provenance | Auditability; answer must be reproducible from the cited text |

What the leaderboard number hides:

- **Retrieval, not reading, is where open-domain systems fail.** Readers score in the 80s-90s on a passage that *contains* the answer. End-to-end accuracy is roughly `P(retrieved the right passage) x P(read it correctly)`, and the first term is usually the smaller one. Fixing retrieval beats swapping readers.
- **Abstention is a product feature, not a metric artifact.** In production, “I could not find this in the documents” is a good answer and a confident wrong one is a support escalation, a compliance incident, or worse. SQuAD 2.0’s `NoAns` bucket is the closest public proxy, and it is exactly the part models are weakest at.
- **Chunking decides the ceiling.** Split a document at 512 tokens and any answer spanning the boundary becomes unreachable. Overlapping windows, semantic chunking and parent-document retrieval exist because of this, and the choice matters more than the reader’s parameter count.
- **Benchmarks are cleaner than reality.** SQuAD passages are single-topic Wikipedia paragraphs with one clear answer. Real corpora contain three versions of the same policy from different years, and the correct behaviour is to notice the conflict - which no standard metric rewards.

------------------------------------------------------------------------

## 3. How Modern Question Answering Works

1.  **Feature-engineered and IR-based QA (pre-2016).** Retrieve documents, match question type to named-entity type, score candidate spans by hand-built features. IBM Watson’s Jeopardy system was the high-water mark of this era and needed a room of machines.
2.  **Neural readers with attention (2016-2018).** BiDAF, DrQA and R-Net learned question-passage attention end to end. SQuAD 1.1 drove the field, and by 2018 these were approaching human EM on it.
3.  **Pretrained encoders with span heads (2018-2020).** BERT put two vectors on a pretrained encoder, beat every bespoke architecture by a wide margin, and made the reader a solved-looking problem. RoBERTa, ALBERT and ELECTRA followed; **DeBERTa-v3** ended up the strongest extractive reader and still is in its size class.
4.  **Answerability and adversarial data (2018-2021).** SQuAD 2.0, AdversarialQA and CheckList exposed that high-scoring models were pattern-matching. Handling “unanswerable” properly became the differentiator, and it remains the thing that separates a demo from a product.
5.  **Retrieval-augmented QA (2020-2023).** DPR replaced BM25 with dense dual encoders; RAG and FiD combined retrieval with a generative reader; ColBERT added late interaction. Open-domain QA became a pipeline: retrieve, rerank, read, cite.
6.  **LLM readers (2023-2026).** Instruction-tuned decoders read the passage and answer in natural language. They handle multi-sentence reasoning, aggregate across passages, and answer questions whose answer is not a contiguous span - all things extractive models structurally cannot do. In exchange they can fabricate, they are 10-100x the compute, and their abstention behaviour depends on prompt wording.
7.  **Long-context and agentic QA (2024-2026).** Context windows of 128k-1M tokens made “just put the whole document in the prompt” viable for single documents, which removed chunking for many cases but not retrieval for corpora. The current frontier is agentic: the model issues its own searches, reads results, decides it needs more, and stops when it can cite an answer. On multi-hop benchmarks this beats single-shot RAG by a wide margin.

**Where it stands (mid-2026).** For a single passage and a short factual answer at high volume, a fine-tuned 100-400M extractive encoder is still the cost and latency winner, and its answers are quotes by construction. For anything requiring synthesis, multiple passages, or a natural-language response, an LLM reader on top of good retrieval wins on quality. The strongest production pattern combines them: retrieve, read with an LLM, and verify the generated answer is supported by the retrieved text - a check an extractive model can do cheaply.

------------------------------------------------------------------------

## 4. Evaluation Metrics

The SQuAD pair, used almost universally for extractive QA:

**Exact Match (EM)** - percentage of predictions that equal a gold answer exactly, after normalisation. Binary and unforgiving: `"the 1990s"` vs `"1990s"` scores 0 without the article-stripping rule.

**Token F1** - the harmonic mean of token precision and recall between prediction and gold, so partial credit exists:

$$P = \frac{|\text{pred} \cap \text{gold}|}{|\text{pred}|} \qquad R = \frac{|\text{pred} \cap \text{gold}|}{|\text{gold}|} \qquad F_1 = \frac{2PR}{P+R}$$

Both are computed against **every** gold answer (SQuAD gives 3+ crowd annotations) and the maximum is taken - a prediction only has to match one annotator.

**Normalisation is part of the metric.** The official SQuAD script lowercases, strips punctuation, removes the articles `a`/`an`/`the`, and collapses whitespace. Skipping it costs several points and makes cross-paper comparison meaningless.

**Report `HasAns` and `NoAns` separately.** On SQuAD 2.0 the aggregate hides everything interesting: a model that never abstains scores ~0 on the unanswerable half and ~85 on the answerable half, and the aggregate looks mid-40s - the same aggregate as a genuinely balanced model. Any honest evaluation splits them.

**Beyond SQuAD.** Generative and RAG systems need different instruments: **groundedness / faithfulness** (is every claim supported by the retrieved text?), **answer relevance**, and **citation precision**. These are typically measured with an LLM judge or an NLI entailment model, because no n-gram metric detects a fluent unsupported sentence. For retrieval itself, see `11_Text_Ranking` (Recall@k, nDCG, MRR).

**Pitfalls:**

- **F1 rewards verbosity.** A generative model that answers in a sentence gets recall for free and loses precision; an extractive model that returns the minimal span does the opposite. Comparing the two on raw F1 without normalising answer length is an apples-to-oranges comparison, and it is the single most common mistake in “LLMs beat BERT at QA” claims.
- **EM is brittle to formatting.** Dates, numbers with units, and lists all fail EM for cosmetic reasons. Read F1 alongside it, always.
- **The null threshold is a tuned hyperparameter.** SQuAD 2.0 scores are reported at the *best* threshold on the dev set. Quoting a paper’s number while running at the default threshold will not reproduce it.

The cell below implements the official SQuAD normalisation, EM and F1 - about 30 lines, and the normalisation is the part that matters.

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


import re
import string
from collections import Counter


def normalize_answer(s):
    "Official SQuAD normalisation: lowercase, drop punctuation, articles and extra space."
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, golds):
    "1 if the prediction matches any gold answer exactly after normalisation."
    return float(any(normalize_answer(pred) == normalize_answer(g) for g in golds))


def token_f1(pred, golds):
    "Max token-overlap F1 against any gold answer. Partial credit for partial spans."
    def f1(p, g):
        p_toks, g_toks = normalize_answer(p).split(), normalize_answer(g).split()
        if not p_toks or not g_toks:          # empty == empty is a correct abstention
            return float(p_toks == g_toks)
        common = Counter(p_toks) & Counter(g_toks)
        n_same = sum(common.values())
        if n_same == 0:
            return 0.0
        prec, rec = n_same / len(p_toks), n_same / len(g_toks)
        return 2 * prec * rec / (prec + rec)
    return max(f1(pred, g) for g in golds)


def squad_score(preds, golds_list):
    "Overall EM/F1 plus the HasAns / NoAns split that the aggregate hides."
    buckets = {"all": [], "HasAns": [], "NoAns": []}
    for pred, golds in zip(preds, golds_list):
        golds = golds or [""]                  # no gold answer == the empty string
        em, f1 = exact_match(pred, golds), token_f1(pred, golds)
        key = "NoAns" if all(not g for g in golds) else "HasAns"
        buckets["all"].append((em, f1))
        buckets[key].append((em, f1))
    return {
        k: {"n": len(v),
            "EM": round(100 * sum(e for e, _ in v) / len(v), 2),
            "F1": round(100 * sum(f for _, f in v) / len(v), 2)}
        for k, v in buckets.items() if v
    }


# Toy example: five predictions against a gold set, including one unanswerable question.
preds = ["the 1990s", "Denver Broncos", "1962", "a very long answer about Paris France", ""]
golds = [["1990s"], ["Broncos", "Denver Broncos"], ["1963"], ["Paris"], [""]]
show_table([{"prediction": p, "gold": str(g),
             "EM": exact_match(p, g), "F1": round(token_f1(p, g), 3)}
            for p, g in zip(preds, golds)],
           title="Five predictions scored",
           caption="row 4 is correct but verbose: it keeps recall and loses precision, "
                   "which is exactly how a generative reader loses to a span model on F1")
show_table([{"bucket": b, **s} for b, s in squad_score(preds, golds).items()],
           title="Aggregate, split by answerability",
           caption="always split HasAns from NoAns - the overall row averages two "
                   "completely different abilities")
```

```
                                 Five predictions scored                                 
                                                                                         
 prediction                              gold                                EM       F1 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 the 1990s                               ['1990s']                       1.0000   1.0000 
 Denver Broncos                          ['Broncos', 'Denver Broncos']   1.0000   1.0000 
 1962                                    ['1963']                        0.0000   0.0000 
 a very long answer about Paris France   ['Paris']                       0.0000   0.2860 
                                         ['']                            1.0000   1.0000 
                                                                                         
row 4 is correct but verbose: it keeps recall and loses precision, which is exactly how a
                      generative reader loses to a span model on F1                      
```

```
                   Aggregate, split by answerability                    
                                                                        
 bucket                    n                     EM                  F1 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 all                       5                  60.00               65.71 
 HasAns                    4                  50.00               57.14 
 NoAns                     1                 100.00              100.00 
                                                                        
always split HasAns from NoAns - the overall row averages two completely
                          different abilities                           
```

## 5. Datasets

| Dataset | Contents | Size | Scope | License | Typical use |
|----|----|----|----|----|----|
| [SQuAD 2.0](https://huggingface.co/datasets/rajpurkar/squad_v2) | Wikipedia paragraphs + spans, incl. 53k unanswerable | 130k train / 11.9k dev | en | CC BY-SA 4.0 | The extractive standard; used below |
| [SQuAD 1.1](https://huggingface.co/datasets/rajpurkar/squad) | Same, every question answerable | 88k / 10.6k | en | CC BY-SA 4.0 | Legacy; saturated |
| [Natural Questions](https://huggingface.co/datasets/google-research-datasets/natural_questions) | Real Google queries + full Wikipedia pages | 307k | en | CC BY-SA 3.0 | Open-domain; long/short answers |
| [TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa) | Trivia questions + distantly-supervised evidence | 95k | en | Apache 2.0 | Open-domain and closed-book |
| [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) | Questions needing 2 documents + supporting facts | 113k | en | CC BY-SA 4.0 | Multi-hop reasoning |
| [MS MARCO](https://huggingface.co/datasets/microsoft/ms_marco) | Bing queries + passages, human-written answers | 1M | en | custom, non-commercial | Passage ranking + generative QA |
| [BEIR](https://huggingface.co/datasets/BeIR/beir) | 18 retrieval datasets, zero-shot | varies | en | mixed | Retrieval side of open-domain QA |
| [TyDi QA](https://huggingface.co/datasets/google-research-datasets/tydiqa) | Typologically diverse, information-seeking | 204k | 11 langs | Apache 2.0 | Multilingual QA |
| [Natural Questions-Open](https://huggingface.co/datasets/google-research-datasets/nq_open) | NQ without the passage | 91k | en | CC BY-SA 3.0 | Closed-book / RAG evaluation |
| [AdversarialQA](https://huggingface.co/datasets/UCLNLP/adversarial_qa) | Questions written to fool a model in the loop | 30k | en | MIT | Robustness; scores drop sharply |
| [ELI5](https://huggingface.co/datasets/sentence-transformers/eli5) | Long-form “explain like I’m five” answers | 270k | en | BSD | Abstractive long-form QA |
| [RAGTruth](https://huggingface.co/datasets/wandb/RAGTruth-processed) | RAG responses annotated for hallucination | 18k | en | MIT | Groundedness evaluation |

This notebook evaluates on the **SQuAD 2.0 validation split**, sampled to keep both answerable and unanswerable questions in roughly their natural proportion (about a third of dev is unanswerable). The sample is small enough to run in a couple of minutes and large enough to show the `HasAns`/`NoAns` split that makes this task interesting.

Downloads land in `DL_tasks/datasets/` via `cache_dir` (gitignored).

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

The reference boards are the [SQuAD 2.0 leaderboard](https://rajpurkar.github.io/SQuAD-explorer/) (extractive, now historical), [BEIR](https://github.com/beir-cellar/beir) for the retrieval half, and the general LLM boards for generative readers. There is no single “QA leaderboard” in 2026 because the task fragmented into retrieval quality, reader quality and groundedness.

| Model | Params | License | Type | SQuAD 2.0 dev F1 | Best for |
|----|----|----|----|----|----|
| [distilbert-base-cased-distilled-squad](https://huggingface.co/distilbert/distilbert-base-cased-distilled-squad) | 66M | Apache 2.0 | extractive (SQuAD 1.1) | n/a - cannot abstain | fastest reader; used below |
| [roberta-base-squad2](https://huggingface.co/deepset/roberta-base-squad2) | 125M | CC BY 4.0 | extractive | ~83 | the practical default; used below |
| [tinyroberta-squad2](https://huggingface.co/deepset/tinyroberta-squad2) | 82M | CC BY 4.0 | extractive, distilled | ~81 | near-base quality at 2x speed |
| [deberta-v3-large-squad2](https://huggingface.co/deepset/deberta-v3-large-squad2) | 435M | CC BY 4.0 | extractive | ~88 | best extractive accuracy; used below |
| [flan-t5-base / large](https://huggingface.co/google/flan-t5-large) | 250M-780M | Apache 2.0 | seq2seq reader | \- | abstractive answers, small |
| [Qwen3-0.6B / 1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 0.6-1.7B | Apache 2.0 | decoder LLM reader | \- | generative reading; used below |
| [Qwen3-4B / 8B](https://huggingface.co/Qwen/Qwen3-8B) | 4-8B | Apache 2.0 | decoder LLM | \- | strong local RAG reader (4B fits in 4-bit here) |
| [bge-m3 / bge-reranker-v2](https://huggingface.co/BAAI/bge-m3) | 568M / 568M | MIT | retriever / reranker | \- | the retrieval half of open-domain QA |
| Frontier LLMs (Claude, GPT, Gemini) | \- | proprietary | long-context + agentic reader | \- | multi-hop, synthesis, citation-heavy production RAG |

**How to choose.** Single passage, short factual answer, high volume, answers must be quotes: `roberta-base-squad2` at 125M, tens of milliseconds, and you get a character offset into the source for free. Maximum extractive accuracy on one GPU: `deberta-v3-large-squad2`. Answers needing synthesis across passages or a natural-language response: an LLM reader, with retrieval quality as the thing you actually invest in. Note that **the extractive models are only competitive because they are fine-tuned on this exact dataset** - on your domain, an LLM with a good prompt usually beats an off-the-shelf SQuAD model, and a SQuAD model fine-tuned on 2k of your own examples beats both.

------------------------------------------------------------------------

## 7. Setup

Everything loads through Hugging Face `transformers` - no vendor packages. Package roles:

- `transformers` + `torch` - the three extractive readers and the LLM reader
- `accelerate` - `device_map` placement
- `datasets` - the SQuAD 2.0 validation split
- `pandas` + `pyecharts` - the benchmark table and chart
- `rich` - the result tables. It renders to HTML inside Jupyter, so the tables survive into the published docs; `show_table` / `show_kv` / `rule` are defined in the first code cell of section 4.

**There is no `question-answering` pipeline any more.** transformers 5.x removed it - `pipeline("question-answering", ...)` now raises `KeyError: Unknown task question-answering`, and only the `table-question-answering` and `document-question-answering` variants survive. Extractive QA is now `AutoModelForQuestionAnswering` plus the decoding the pipeline used to do for you, which section 8 writes out as a small `SpanReader` class and the rest of the notebook reuses. That is a gain for a notebook about this task: the three things that decide whether a reader works are no longer hidden behind a keyword argument.

- **The `[CLS]` null score** is what lets a model return `""`. The start and end distributions are softmaxed over the context tokens *plus* `[CLS]`, and `P_start([CLS]) * P_end([CLS])` is the model’s own “the answer is not in this passage” vote, competing directly against the best real span. Drop `[CLS]` from that softmax - an easy mistake, since it is not part of the answer - and the model can never abstain, so it scores ~0 on the `NoAns` half no matter what it was trained on. This is the single most consequential detail in the notebook, and `handle_impossible_answer=False` turns it off.
- **`max_seq_len` + `doc_stride`** control the sliding window over long contexts. A passage that exceeds the model’s window is tokenised into overlapping chunks (`return_overflowing_tokens=True`), scored independently, and the best span across all of them wins; `doc_stride` is the overlap, and too small a value makes boundary-spanning answers unreachable. The SQuAD sample below reaches ~3,200 characters, well past a 384-token window, so this path runs for real.
- **`top_k`** returns the n-best spans with scores, which is what you need for a confidence threshold or for showing alternatives.

Answers come back with `start`/`end` **character** offsets into the original context, via the fast tokenizer’s `return_offsets_mapping=True` - keep them. They are what turns an answer into a highlight in the source document, and they are the reason to use an extractive model at all.

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

# SQuAD 2.0 validation: 11,873 questions, about a third of them deliberately unanswerable.
squad = load_dataset("rajpurkar/squad_v2", split="validation", cache_dir=HF_CACHE)

N = 200  # questions to evaluate. The full split takes tens of minutes on this box.
sample = squad.shuffle(seed=0).select(range(N))

questions = [r["question"] for r in sample]
contexts = [r["context"] for r in sample]
gold = [r["answers"]["text"] or [""] for r in sample]  # [] means unanswerable -> [""]

n_noans = sum(1 for g in gold if all(not x for x in g))
print(squad)
print(f"\nevaluating {N} questions: {N - n_noans} answerable / {n_noans} unanswerable")
print(f"median context length: {int(sorted(len(c) for c in contexts)[N // 2])} characters\n")

for i in (0, 1):
    print(f"Q: {questions[i]}")
    print(f"A: {gold[i]}")
    print(f"   ...{contexts[i][:180]}...\n")
```

    Dataset({
        features: ['id', 'title', 'context', 'question', 'answers'],
        num_rows: 11873
    })

    evaluating 200 questions: 105 answerable / 95 unanswerable
    median context length: 748 characters

    Q: How much support is there for the UN approach to economic development?
    A: ['']
       ...John Schmitt and Ben Zipperer (2006) of the CEPR point to economic liberalism and the reduction of business regulation along with the decline of union membership as one of the caus...

    Q: What principle highlights the significance of primes in number theory
    A: ['local-global principle', 'local-global', 'local-global principle', 'local-global principle']
       ...In particular, this norm gets smaller when a number is multiplied by p, in sharp contrast to the usual absolute value (also referred to as the infinite prime). While completing Q (...

## 8. Extractive baseline: DistilBERT (SQuAD 1.1)

The cheapest useful reader: 66M params, a distilled BERT with a span head fine-tuned on **SQuAD 1.1** - the version where every question has an answer. That training data is the point of including it here.

Because it never saw an unanswerable question, it has no notion of abstaining. `handle_impossible_answer=True` is set anyway, and it changes almost nothing: the model’s `[CLS]` null score is never competitive, so it points at *something* for every question. On the `HasAns` half it is respectable; on the `NoAns` half it scores near zero, and the aggregate lands in the middle looking mediocre for the wrong reason.

This is the most important lesson in the notebook and it is a data lesson, not an architecture one. Two models with identical architectures score 40 points apart on SQuAD 2.0 purely because one was trained to say “no”.

------------------------------------------------------------------------

``` python
# transformers 5.x removed the extractive `question-answering` pipeline - only the
# table- and document- variants survive - so the span decoding it used to hide is
# written out here. Read it once: it is section 1's start/end head plus the three
# details that make a reader usable - a sliding window over long contexts, the [CLS]
# null score that lets a model abstain, and token -> character offsets.
import numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


class SpanReader:
    "An encoder with a span head, plus n-best decoding over a sliding window."

    def __init__(self, model_id, device="cpu", dtype=None, cache_dir=None,
                 max_seq_len=384, doc_stride=128):
        self.tok = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_id, dtype=dtype, cache_dir=cache_dir,
        ).to(device).eval()
        self.device, self.max_seq_len, self.doc_stride = device, max_seq_len, doc_stride

    def _encode(self, question, context):
        "A passage longer than the window becomes several overlapping features."
        return self.tok(
            question, context, truncation="only_second", max_length=self.max_seq_len,
            stride=self.doc_stride, return_overflowing_tokens=True,
            return_offsets_mapping=True, padding="max_length", return_tensors="pt",
        )

    @staticmethod
    def _probs(logits, keep):
        "Softmax over the candidate positions only; everything else gets zero mass."
        z = np.where(keep, logits, -1e4)
        p = np.exp(z - z.max())
        return p / p.sum()

    @torch.inference_mode()
    def __call__(self, question, context, handle_impossible_answer=True,
                 top_k=1, max_answer_len=30, batch_size=16):
        single = isinstance(question, str)
        qs = [question] if single else list(question)
        cs = [context] if single else list(context)
        out = []
        for i in range(0, len(qs), batch_size):
            out.extend(self._answer(qs[i:i + batch_size], cs[i:i + batch_size],
                                    handle_impossible_answer, top_k, max_answer_len))
        return out[0] if single else out

    def _answer(self, questions, contexts, handle_impossible, top_k, max_answer_len):
        encs = [self._encode(q, c) for q, c in zip(questions, contexts)]
        owner = [i for i, e in enumerate(encs) for _ in range(e["input_ids"].shape[0])]
        fields = ["input_ids", "attention_mask"] + (
            ["token_type_ids"] if "token_type_ids" in encs[0] else [])
        batch = {k: torch.cat([e[k] for e in encs]).to(self.device) for k in fields}
        logits = self.model(**batch)
        starts = logits.start_logits.float().cpu().numpy()
        ends = logits.end_logits.float().cpu().numpy()

        # sequence_ids() says which tokens came from the context (1) and which from the
        # question; an answer may only start and end inside the context.
        ctx_masks, offsets, cls_at = [], [], []
        for e in encs:
            for f in range(e["input_ids"].shape[0]):
                ctx_masks.append(np.array([s == 1 for s in e.sequence_ids(f)]))
                offsets.append(e["offset_mapping"][f].numpy())
                hit = (e["input_ids"][f] == self.tok.cls_token_id).nonzero()
                cls_at.append(int(hit[0][0]) if len(hit) else 0)

        found = [{"spans": [], "null": 1.0} for _ in questions]
        for f, ex in enumerate(owner):
            ctx = ctx_masks[f]
            c = cls_at[f]
            # [CLS] stays inside the softmax next to the context tokens: its
            # start x end probability IS "no answer in this window". Mask it out and the
            # model can never abstain, however it was trained - the NoAns half goes to 0.
            keep = ctx.copy()
            keep[c] = True
            s, e_ = self._probs(starts[f], keep), self._probs(ends[f], keep)
            found[ex]["null"] = min(found[ex]["null"], float(s[c] * e_[c]))
            s, e_ = s.copy(), e_.copy()
            s[c] = e_[c] = 0.0
            # Every (start, end) pair at once, then drop the malformed ones:
            # end before start, longer than max_answer_len, or outside the context.
            scores = np.outer(s, e_)
            ok = np.triu(np.ones_like(scores, dtype=bool))
            ok &= ~np.triu(np.ones_like(scores, dtype=bool), max_answer_len)
            ok &= ctx[:, None] & ctx[None, :]
            scores = np.where(ok, scores, 0.0).ravel()
            for idx in np.argsort(scores)[-top_k:][::-1]:
                if scores[idx] <= 0:
                    continue
                si, ei = divmod(int(idx), len(s))
                found[ex]["spans"].append(
                    (float(scores[idx]), int(offsets[f][si][0]), int(offsets[f][ei][1])))

        out = []
        for ex, r in enumerate(found):
            answers, seen = [], set()
            for sc, a, b in sorted(r["spans"], reverse=True):
                # Sentencepiece models (DeBERTa) fold the leading space into the token,
                # so trim it or the offsets highlight one character too wide.
                while a < b and contexts[ex][a].isspace():
                    a += 1
                if (a, b) in seen:              # overlapping windows find the same span
                    continue
                seen.add((a, b))
                answers.append({"score": sc, "start": a, "end": b,
                                "answer": contexts[ex][a:b]})
                if len(answers) == top_k:
                    break
            if handle_impossible:               # "" competes with the best real span
                answers.append({"score": r["null"], "start": 0, "end": 0, "answer": ""})
                answers = sorted(answers, key=lambda d: -d["score"])[:top_k]
            out.append(answers if top_k > 1 else answers[0])
        return out


qa = SpanReader("distilbert/distilbert-base-cased-distilled-squad",
                device=device, cache_dir=HF_CACHE)
vram("distilbert loaded")

# One worked example - note the character offsets, which turn an answer into a highlight.
demo_ctx = (
    "The Amazon rainforest covers most of the Amazon basin of South America. This basin "
    "encompasses 7,000,000 square kilometres, of which 5,500,000 square kilometres are "
    "covered by the rainforest. Brazil holds about 60% of the rainforest."
)
_demo_qs = ["How large is the Amazon basin?", "Which country holds most of it?",
            "When was the rainforest formed?"]
show_table([{"question": q, "answer": (o["answer"] or "<abstained>"),
             "score": round(o["score"], 3), "chars": f"[{o['start']}:{o['end']}]"}
            for q, o in ((q, qa(question=q, context=demo_ctx,
                                handle_impossible_answer=True)) for q in _demo_qs)],
           title="Extractive answers come with character offsets into the source",
           caption="the third question is unanswerable from this passage - "
                   "watch a SQuAD 1.1 model answer it anyway")


def show_squad(title, preds, secs):
    "Render one reader's SQuAD result: the HasAns/NoAns split plus throughput."
    rows = [{"bucket": b, **s} for b, s in squad_score(preds, gold).items()]
    show_table(rows, title=title, best=("EM", "F1"),
               caption=f"{N} questions in {secs:.1f}s "
                       f"({N / secs:.1f} q/s), abstained on "
                       f"{sum(1 for p in preds if not p.strip())}/{N}")


def run_extractive(reader, batch_size=16):
    "Answer every question, returning predictions and wall-clock seconds."
    t0 = time.perf_counter()
    outs = reader(question=questions, context=contexts,
                  handle_impossible_answer=True, batch_size=batch_size)
    return [o["answer"] for o in outs], time.perf_counter() - t0


distil_preds, distil_secs = run_extractive(qa)
show_squad("distilbert (SQuAD 1.1) - never trained to abstain", distil_preds, distil_secs)

del qa
free_memory()
vram("after distilbert")
```

    VRAM distilbert loaded       0.26 GB allocated /  0.28 GB reserved

```
           Extractive answers come with character offsets into the source           
                                                                                    
 question                          answer                         score   chars     
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 How large is the Amazon basin?    7,000,000 square kilometres   0.9720   [95:122]  
 Which country holds most of it?   Brazil                        0.9810   [192:198] 
 When was the rainforest formed?   Amazon                        0.0960   [4:10]    
                                                                                    
   the third question is unanswerable from this passage - watch a SQuAD 1.1 model   
                                  answer it anyway                                  
```

```
           distilbert (SQuAD 1.1) - never trained to abstain            
                                                                        
 bucket                        n                   EM                F1 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 all                         200                42.50             46.86 
 HasAns                      105                80.95             89.27 
 NoAns                        95               0.0000            0.0000 
                                                                        
          200 questions in 5.6s (35.8 q/s), abstained on 0/200          
```

    VRAM after distilbert        0.01 GB allocated /  0.02 GB reserved

## 9. Trained to abstain: roberta-base-squad2

The same architecture family and only twice the parameters, but fine-tuned on **SQuAD 2.0**, so it has seen 53k questions whose correct answer is nothing. That single difference in the training data is worth more than any architectural change in this notebook.

The mechanism is the null score. During training, unanswerable questions are labelled with the span `(CLS, CLS)`, so `[CLS]` learns to be a strong start-and-end candidate exactly when no real span works. At inference the best real span only wins if it beats the null score:

    answer if  score(best_span) - score(null)  >  threshold  else  ""

**That threshold is a product decision.** Raise it and the model abstains more: higher precision on the answers it does give, lower recall. Lower it and you get the opposite. Papers report SQuAD 2.0 numbers at the dev-set-optimal threshold, which is why reproducing a headline number at the default rarely works. The cell below sweeps it so the trade-off is visible rather than assumed.

------------------------------------------------------------------------

``` python
qa2 = SpanReader("deepset/roberta-base-squad2", device=device, cache_dir=HF_CACHE)
vram("roberta-squad2 loaded")

show_table([{"question": q, "answer": (o["answer"] or "<abstained>"),
             "score": round(o["score"], 3)}
            for q, o in ((q, qa2(question=q, context=demo_ctx,
                                 handle_impossible_answer=True))
                         for q in ["How large is the Amazon basin?",
                                   "When was the rainforest formed?"])],
           title="Same passage, a model trained on SQuAD 2.0",
           caption="an empty answer is the model abstaining, not failing")

roberta_preds, roberta_secs = run_extractive(qa2)
show_squad("roberta-base-squad2 - 53k unanswerable questions in its training data",
           roberta_preds, roberta_secs)

# The abstention threshold is a tunable trade-off, not a constant. Sweep it on the
# reader's confidence: below `t`, refuse to answer.
raw = qa2(question=questions, context=contexts, handle_impossible_answer=True,
          batch_size=16)
sweep = []
for t in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
    preds_t = [o["answer"] if o["score"] >= t else "" for o in raw]
    s = squad_score(preds_t, gold)
    sweep.append({"threshold": t, "overall_F1": s["all"]["F1"],
                  "HasAns_F1": s.get("HasAns", {}).get("F1", 0.0),
                  "NoAns_F1": s.get("NoAns", {}).get("F1", 0.0),
                  "answered": sum(1 for p in preds_t if p.strip())})
show_table(sweep, title="Abstention threshold sweep", best=("overall_F1",),
           caption="papers report SQuAD 2.0 at the dev-set-optimal threshold - running at "
                   "the default will not reproduce their number")

del qa2, raw
free_memory()
vram("after roberta-squad2")
```

    VRAM roberta-squad2 loaded   0.51 GB allocated /  0.56 GB reserved

```
               Same passage, a model trained on SQuAD 2.0               
                                                                        
 question                          answer                         score 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 How large is the Amazon basin?    7,000,000 square kilometres   0.8620 
 When was the rainforest formed?   <abstained>                   0.9660 
                                                                        
          an empty answer is the model abstaining, not failing          
```

```
 roberta-base-squad2 - 53k unanswerable questions in its training data  
                                                                        
 bucket                          n                  EM               F1 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 all                           200               81.00            84.28 
 HasAns                        105               78.10            84.35 
 NoAns                          95               84.21            84.21 
                                                                        
         200 questions in 7.5s (26.6 q/s), abstained on 90/200          
```

```
                       Abstention threshold sweep                       
                                                                        
    threshold       overall_F1      HasAns_F1      NoAns_F1    answered 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
       0.0000            84.28          84.35         84.21         110 
       0.1000            82.66          81.26         84.21         106 
       0.2000            81.83          77.78         86.32          99 
       0.3000            78.53          69.59         88.42          87 
       0.5000            76.50          59.05         95.79          66 
       0.7000            68.00          40.95         97.89          45 
       0.9000            58.00          20.95         98.95          23 
                                                                        
 papers report SQuAD 2.0 at the dev-set-optimal threshold - running at  
              the default will not reproduce their number               
```

    VRAM after roberta-squad2    0.01 GB allocated /  0.02 GB reserved

``` python
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org serves an expired TLS certificate from most nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"

from pyecharts import options as opts
from pyecharts.charts import Line

# The abstention trade-off, drawn. HasAns falls and NoAns rises as the threshold climbs;
# the overall peak is where a deployment would sit if both error types cost the same.
line = (
    Line()
    .add_xaxis([str(s["threshold"]) for s in sweep])
    .add_yaxis("overall F1", [s["overall_F1"] for s in sweep])
    .add_yaxis("HasAns F1", [s["HasAns_F1"] for s in sweep])
    .add_yaxis("NoAns F1", [s["NoAns_F1"] for s in sweep])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="roberta-base-squad2: abstention threshold sweep",
            subtitle=f"SQuAD 2.0, {N} questions - answering less makes NoAns rise and HasAns fall",
        ),
        xaxis_opts=opts.AxisOpts(name="confidence threshold"),
        yaxis_opts=opts.AxisOpts(name="F1", min_=0, max_=100),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
    )
)
line.render_notebook()
```

## 10. Best extractive: deberta-v3-large-squad2

DeBERTa-v3 is the strongest classical encoder for span extraction, and it earns that with two ideas. **Disentangled attention** keeps content and position as separate vectors and computes content-to-content, content-to-position and position-to-content attention separately, which helps for a task that is fundamentally about *where* something is. And the **ELECTRA-style replaced-token-detection** pretraining objective is far more sample-efficient than masked-LM, so the same compute buys a better encoder.

At 435M params it is ~3.5x roberta-base and lands around 88 F1 on full SQuAD 2.0 dev, several points ahead. In fp16 it is under 1 GB of VRAM, so it fits this box comfortably; the cost is throughput, which is where the benchmark below earns its keep.

If you need one off-the-shelf extractive reader and can afford the latency, this is it.

------------------------------------------------------------------------

``` python
qa3 = SpanReader("deepset/deberta-v3-large-squad2",
                 device=device, dtype=dtype, cache_dir=HF_CACHE)
vram("deberta-v3-large loaded")

deberta_preds, deberta_secs = run_extractive(qa3, batch_size=8)
show_squad("deberta-v3-large-squad2 - the best extractive reader here",
           deberta_preds, deberta_secs)

del qa3
free_memory()
vram("after deberta-v3-large")
```

    VRAM deberta-v3-large loaded  0.88 GB allocated /  0.90 GB reserved

```
       deberta-v3-large-squad2 - the best extractive reader here        
                                                                        
 bucket                          n                  EM               F1 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 all                           200               86.50            90.51 
 HasAns                        105               80.95            88.59 
 NoAns                          95               92.63            92.63 
                                                                        
         200 questions in 8.5s (23.6 q/s), abstained on 92/200          
```

    VRAM after deberta-v3-large  0.01 GB allocated /  0.02 GB reserved

## 11. Generative reader: Qwen3-0.6B

A decoder LLM reads the same passage and writes the answer. No span head, no fine-tuning on SQuAD, no start/end offsets - just a prompt.

What it gains: answers that are not contiguous spans (a computed number, a synthesis of two sentences, a rephrasing), and abstention controlled by instruction rather than by a threshold. What it loses: the guarantee that the answer appears in the source, and roughly an order of magnitude of throughput.

Two practical notes that decide whether this works at all:

- **The prompt is the abstention mechanism.** Telling it to reply exactly `unanswerable` when the passage does not contain the answer is the entire `NoAns` strategy. Remove that sentence and the `NoAns` score collapses - the model will always find something plausible to say.
- **Verbosity is a metric problem, not a quality problem.** SQuAD F1 punishes “The Amazon basin covers 7,000,000 square kilometres” against the gold `7,000,000 square kilometres`. The instruction to answer with the shortest exact phrase, plus stripping trailing punctuation, is not cheating - it is aligning the output format with what the metric measures. When people report that LLMs underperform BERT on SQuAD, this accounts for a large part of the gap.

`enable_thinking=False` keeps Qwen3 from emitting a `<think>` block; the reasoning tokens would be pure cost here.

------------------------------------------------------------------------

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer

llm_id = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(llm_id, cache_dir=HF_CACHE)
llm = AutoModelForCausalLM.from_pretrained(
    llm_id, dtype=dtype, device_map=device, cache_dir=HF_CACHE
).eval()
vram("qwen3-0.6b loaded")

PROMPT = (
    "Answer the question using ONLY the passage below. Reply with the shortest exact "
    "phrase from the passage that answers it - no sentence, no explanation. If the "
    "passage does not contain the answer, reply exactly: unanswerable\n\n"
    "Passage: {context}\n\nQuestion: {question}\nAnswer:"
)


@torch.inference_mode()
def llm_read(batch_q, batch_c, max_new_tokens=24):
    "Generate a short answer per (question, context); map the refusal token to ''."
    chats = [
        tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(context=c, question=q)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        for q, c in zip(batch_q, batch_c)
    ]
    enc = tok(chats, return_tensors="pt", padding=True, padding_side="left",
              truncation=True, max_length=1024).to(llm.device)
    out = llm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                       pad_token_id=tok.eos_token_id)
    gen = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    answers = []
    for g in gen:
        a = g.strip().split("\n")[0].strip().strip('."')
        answers.append("" if a.lower().startswith("unanswerable") else a)
    return answers


_demo_llm = ["How large is the Amazon basin?", "When was the rainforest formed?"]
show_table([{"question": q, "answer": a or "<abstained>"}
            for q, a in zip(_demo_llm, llm_read(_demo_llm, [demo_ctx, demo_ctx]))],
           title="Qwen3-0.6B reading the same passage",
           caption="abstention here comes entirely from one sentence in the prompt")

t0 = time.perf_counter()
llm_preds = []
for i in range(0, N, 8):
    llm_preds.extend(llm_read(questions[i:i + 8], contexts[i:i + 8]))
llm_secs = time.perf_counter() - t0

show_squad("qwen3-0.6b (generative) - EM well below F1 is the verbosity effect",
           llm_preds, llm_secs)

del llm, tok
free_memory()
vram("after qwen3")
```

    VRAM qwen3-0.6b loaded       1.20 GB allocated /  1.68 GB reserved

```
                  Qwen3-0.6B reading the same passage                   
                                                                        
 question                               answer                          
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 How large is the Amazon basin?         7,000,000 square kilometres     
 When was the rainforest formed?        <abstained>                     
                                                                        
     abstention here comes entirely from one sentence in the prompt     
```

```
   qwen3-0.6b (generative) - EM well below F1 is the verbosity effect   
                                                                        
 bucket                          n                  EM               F1 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 all                           200               41.50            48.16 
 HasAns                        105               27.62            40.31 
 NoAns                          95               56.84            56.84 
                                                                        
         200 questions in 12.9s (15.5 q/s), abstained on 74/200         
```

    VRAM after qwen3             0.01 GB allocated /  0.02 GB reserved

## 12. Head-to-head Benchmark

The same 200 SQuAD 2.0 questions, the same normalisation, the same scorer, one model live at a time. Sections 8-11 already produced the numbers, so this collects them rather than reloading four models to reproduce what we have.

Read the `HasAns` and `NoAns` columns before the overall one. The overall column is a weighted average of two very different abilities, and every interesting difference between these models lives in the split:

- DistilBERT is fine on `HasAns` and near-zero on `NoAns` - it was never trained to abstain.
- The two SQuAD 2.0 encoders trade a little `HasAns` for a great deal of `NoAns`.
- The LLM’s `NoAns` behaviour comes entirely from one sentence in the prompt, and its `EM`-to-`F1` gap is a formatting artifact, not a comprehension gap.

At n=200 the sampling noise is roughly +/-4 F1, so treat small differences as ties. And note the encoders are fine-tuned on this exact dataset while the LLM has never seen it - a genuinely fair comparison would fine-tune the LLM or evaluate all four on out-of-domain data, which is what section 13 invites you to do by hand.

------------------------------------------------------------------------

``` python
import pandas as pd

RUNS = [
    ("distilbert-squad1.1", 66, distil_preds, distil_secs),
    ("roberta-base-squad2", 125, roberta_preds, roberta_secs),
    ("deberta-v3-large-squad2", 435, deberta_preds, deberta_secs),
    ("qwen3-0.6b (generative)", 596, llm_preds, llm_secs),
]

results = []
for name, params_m, preds, secs in RUNS:
    s = squad_score(preds, gold)
    results.append({
        "model": name,
        "params_m": params_m,
        "EM": s["all"]["EM"],
        "F1": s["all"]["F1"],
        "HasAns_F1": s.get("HasAns", {}).get("F1", 0.0),
        "NoAns_F1": s.get("NoAns", {}).get("F1", 0.0),
        "abstained": sum(1 for p in preds if not p.strip()),
        "q_per_sec": round(N / secs, 1),
    })

df_results = pd.DataFrame(results).sort_values("F1", ascending=False)
show_table(
    df_results.to_dict("records"),
    title=f"SQuAD 2.0 validation, {N} questions",
    best=("EM", "F1", "HasAns_F1", "NoAns_F1", "q_per_sec"),
    caption="read HasAns_F1 and NoAns_F1 before the aggregate - that is where the models differ",
)
```

```
                                SQuAD 2.0 validation, 200 questions                                
                                                                                                   
 model                     params_m      EM      F1   HasAns_F1   NoAns_F1   abstained   q_per_sec 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 deberta-v3-large-squad2        435   86.50   90.51       88.59      92.63          92       23.60 
 roberta-base-squad2            125   81.00   84.28       84.35      84.21          90       26.60 
 qwen3-0.6b (generative)        596   41.50   48.16       40.31      56.84          74       15.50 
 distilbert-squad1.1             66   42.50   46.86       89.27     0.0000           0       35.80 
                                                                                                   
        read HasAns_F1 and NoAns_F1 before the aggregate - that is where the models differ         
```

``` python
from pyecharts.charts import Bar

# The split is the story: NoAns is where the models differ, and it is a training-data
# property (SQuAD 1.1 vs 2.0) or a prompt property, not an architecture property.
bar = (
    Bar()
    .add_xaxis([r["model"] for r in results])
    .add_yaxis("overall F1", [r["F1"] for r in results])
    .add_yaxis("HasAns F1", [r["HasAns_F1"] for r in results])
    .add_yaxis("NoAns F1", [r["NoAns_F1"] for r in results])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"SQuAD 2.0 validation ({N} questions)",
            subtitle="RTX 3060 - a smoke test, not a leaderboard; +/-4 F1 of noise at n=200",
        ),
        yaxis_opts=opts.AxisOpts(name="F1", min_=0, max_=100),
        xaxis_opts=opts.AxisOpts(name="model", axislabel_opts=opts.LabelOpts(rotate=15)),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="8%"),
    )
)
bar.render_notebook()
```

``` python
from pyecharts.charts import Scatter

# F1 against throughput - the curve a deployment actually picks from.
scatter = Scatter()
scatter.add_xaxis([r["q_per_sec"] for r in results])
for r in results:
    scatter.add_yaxis(
        r["model"], [[r["q_per_sec"], r["F1"]]],
        symbol_size=18, label_opts=opts.LabelOpts(is_show=False),
    )
scatter.set_global_opts(
    title_opts=opts.TitleOpts(title="F1 vs throughput",
                              subtitle="extractive readers are 10-50x cheaper per question"),
    xaxis_opts=opts.AxisOpts(name="questions / second", type_="value"),
    yaxis_opts=opts.AxisOpts(name="overall F1", type_="value"),
    tooltip_opts=opts.TooltipOpts(trigger="item"),
)
scatter.render_notebook()
```

## 13. Interactive: ask your own passage

Paste your own passage and questions below. This is the cell people run on its own, so it opens with a `require(...)` guard naming what it needs from Setup rather than dying on a bare `NameError`.

The questions worth trying are the ones that break a span model:

- **Answerable but not a span** (“how many years between the two events?”) - the answer must be computed, so an extractive model cannot express it at all. It will return the nearest date.
- **Unanswerable but adjacent** - ask about an entity the passage mentions but never describes. This is exactly how SQuAD 2.0’s unanswerable questions were written, and it is where the confidence score becomes the only defence.
- **Answer spanning two sentences** - extractive models return one contiguous span, so they pick the better half.
- **Contradicted or superseded facts** - put two conflicting statements in the passage. Neither model notices; both answer from one of them without flagging the conflict, which is the failure mode that matters most in real document QA.

The printed confidence is what a production system would threshold on. Watch it stay high on questions the model gets wrong - confidence is a ranking signal, not a correctness probability, unless you calibrate it.

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


require("device", "dtype", "HF_CACHE", "free_memory", "vram", "squad_score", "SpanReader")

MY_CONTEXT = (
    "The Knowledge Lab container runs on a Proxmox host and is allocated 4 vCPU, 20 GB "
    "of RAM and 4 GB of swap. An RTX 3060 with 12 GB of VRAM is passed through to it, "
    "along with a USB camera and its microphone. JupyterLab is served on port 8888. "
    "The root disk was moved to a 4 TB NVMe drive in August 2026 and grown to 500 GB."
)

MY_QUESTIONS = [
    "How much VRAM does the GPU have?",                 # plain span
    "What port does JupyterLab use?",                   # plain span
    "How much more RAM than swap does it have?",        # answerable, but not a span
    "Who administers the container?",                   # unanswerable, adjacent topic
    "What size is the root disk and what drive is it on?",  # answer spans two clauses
]

# Re-runnable: this cell frees the reader at the end, so guard the load or a second
# shift-enter raises NameError.
if "my_qa" not in globals():
    my_qa = SpanReader("deepset/roberta-base-squad2", device=device, cache_dir=HF_CACHE)

print(MY_CONTEXT, "\n")
for q in MY_QUESTIONS:
    out = my_qa(question=q, context=MY_CONTEXT, handle_impossible_answer=True, top_k=3)
    best = out[0]
    shown = best["answer"] if best["answer"].strip() else "<abstained>"
    print(f"Q: {q}")
    print(f"   -> {shown!r}  (confidence {best['score']:.3f}, "
          f"chars[{best['start']}:{best['end']}])")
    alts = ", ".join(f"{o['answer']!r} {o['score']:.2f}" for o in out[1:])
    print(f"   alternatives: {alts}\n")

del my_qa
free_memory()
vram("final")
```

    The Knowledge Lab container runs on a Proxmox host and is allocated 4 vCPU, 20 GB of RAM and 4 GB of swap. An RTX 3060 with 12 GB of VRAM is passed through to it, along with a USB camera and its microphone. JupyterLab is served on port 8888. The root disk was moved to a 4 TB NVMe drive in August 2026 and grown to 500 GB. 

    Q: How much VRAM does the GPU have?
       -> '12 GB'  (confidence 0.361, chars[124:129])
       alternatives: '' 0.13, '12 GB of VRAM' 0.03

    Q: What port does JupyterLab use?
       -> '8888'  (confidence 0.711, chars[236:240])
       alternatives: 'port 8888' 0.28, 'JupyterLab is served on port 8888' 0.00

    Q: How much more RAM than swap does it have?
       -> '20 GB'  (confidence 0.691, chars[76:81])
       alternatives: '20 GB of RAM and 4 GB' 0.09, '20 GB of RAM' 0.03

    Q: Who administers the container?
       -> '<abstained>'  (confidence 0.208, chars[0:0])
       alternatives: 'Proxmox host' 0.13, 'Proxmox' 0.05

    Q: What size is the root disk and what drive is it on?
       -> '500 GB'  (confidence 0.240, chars[315:321])
       alternatives: '4 TB NVMe drive in August 2026 and grown to 500 GB' 0.17, '4 TB NVMe drive' 0.09

    VRAM final                   0.01 GB allocated /  0.02 GB reserved

## 14. Common Frameworks

Single-passage QA needs almost no infrastructure; open-domain QA needs all of it. The moment the passage is not handed to you, the system becomes a retrieval pipeline with a reader on the end, and the retrieval half is where both the cost and the accuracy live. That is why most of this table is retrieval and orchestration - and why the most useful diagnostic in the task, Recall@k, belongs to the retriever rather than the model this notebook is about.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [transformers](https://github.com/huggingface/transformers) | modelling | Extractive readers behind `AutoModelForQuestionAnswering` with the SQuAD offset-mapping recipe, and any LLM for the generative path | Apache 2.0 | Default. Both reader styles of sections 8-11 |
| [peft](https://github.com/huggingface/peft) + [trl](https://github.com/huggingface/trl) | modelling | LoRA on a generative reader, and plain `Trainer` fine-tuning for the extractive ones | Apache 2.0 | Two thousand domain examples typically beat any off-the-shelf SQuAD checkpoint, because answer granularity is a domain convention |
| [sentence-transformers](https://github.com/UKPLab/sentence-transformers) | data | The retriever: bi-encoder embeddings, and the contrastive fine-tuning that adapts it to your queries | Apache 2.0 | Open-domain QA. See `07_Feature_Extraction` |
| [faiss](https://github.com/facebookresearch/faiss) / [Qdrant](https://github.com/qdrant/qdrant) / pgvector + BM25 ([Elasticsearch](https://github.com/elastic/elasticsearch), [rank_bm25](https://github.com/dorianbrown/rank_bm25)) | data | Dense and sparse indexes, and the hybrid fusion that beats either alone on names, codes and rare terms | MIT / Apache 2.0 / Elastic License | Always for open-domain. Keep BM25 - it costs nothing and catches exactly what dense retrieval misses |
| Chunking via [Docling](https://github.com/docling-project/docling) / [LlamaIndex](https://github.com/run-llama/llama_index) node parsers | data | Structure-aware splitting with overlap, and parent-document retrieval so the reader sees context the boundary cut | MIT | Always. A claim separated from its subject by a chunk boundary is unretrievable at any model size |
| [vLLM](https://github.com/vllm-project/vllm) / [SGLang](https://github.com/sgl-project/sglang) | inference runtime | Batched generation with prefix caching, which pays when many questions hit the same retrieved context | Apache 2.0 | Serving the generative path |
| [optimum](https://github.com/huggingface/optimum) + [ONNX Runtime](https://onnxruntime.ai/) | inference runtime | An extractive reader on CPU - fast enough that the whole answer path can run without a GPU | Apache 2.0 / MIT | The extractive path, which stays remarkably cheap |
| [LlamaIndex](https://github.com/run-llama/llama_index) / [Haystack](https://github.com/deepset-ai/haystack) / [LangGraph](https://github.com/langchain-ai/langgraph) | orchestration | Retrieve, rerank, read, cite - with the retry and fallback logic in one place instead of scattered | MIT / Apache 2.0 | Building the system rather than the model. Haystack was built for exactly this pipeline |
| [ragas](https://github.com/explodinggradients/ragas) + [evaluate](https://github.com/huggingface/evaluate) (SQuAD EM/F1) | evaluation | Reader EM/F1, plus retrieval recall, answer faithfulness and context precision measured separately | Apache 2.0 | Always. An end-to-end score cannot tell you whether the retriever or the reader failed, and they need different fixes |

**The 2026 default stack** is hybrid retrieval (dense plus BM25) fused with RRF, a cross-encoder rerank, a generative reader served by vLLM, groundedness checked with an NLI model, and ragas reporting retrieval and generation separately. The extractive path remains the right answer when you need a span and a confidence rather than prose.

The common wrong turn is optimising the reader when Recall@k is the ceiling. If the right passage is not in the top-k, no reader can recover it, and this one diagnostic redirects more misspent RAG effort than anything else. The second is not calibrating abstention: in most enterprise settings a wrong answer costs 5-20x a “not found”, which puts the threshold well above the F1-optimal point.

------------------------------------------------------------------------

## 15. Going Further

- **Fine-tune a reader on your own domain.** `AutoModelForQuestionAnswering.from_pretrained("microsoft/deberta-v3-base")` plus the SQuAD preprocessing recipe (offset mapping, sliding window, `(CLS, CLS)` labels for unanswerable) is the standard path. Two thousand domain examples typically beat any off-the-shelf SQuAD checkpoint on that domain, because question phrasing and answer granularity are domain conventions.
- **Build the retrieval half.** Single-passage QA is the easy part. Open-domain means embed your corpus (`07_Feature_Extraction`), retrieve top-k (`11_Text_Ranking`), rerank with a cross-encoder, then read. Measure Recall@k of the retriever separately - if the right passage is not in the top-k, no reader can save you.
- **Chunk with overlap, and store the parent.** 300-500 token chunks with ~20% overlap is a sane default; retrieve on the chunk and pass the surrounding parent section to the reader, so the answer has context the chunk boundary cut off.
- **Verify groundedness.** After a generative answer, check that it is entailed by the retrieved passages using an NLI model (`04_Zero_Shot_Classification` covers the mechanism) or an LLM judge. This catches fluent unsupported sentences, which no n-gram metric will.
- **Calibrate abstention deliberately.** Sweep the null threshold on your own validation set and pick the operating point from the relative cost of a wrong answer versus a missed one. In most enterprise settings a wrong answer costs 5-20x a “not found”, which puts the threshold far above the F1-optimal point.
- **Long documents: try the context window first.** Before building a retrieval pipeline for a single 50-page document, check whether it fits a 128k-token model. It usually does, and it removes chunking entirely. Retrieval is for *corpora*, not for long files.
- **Multi-hop needs iteration.** HotpotQA-style questions do not yield to one retrieve-and-read pass. Let the model issue a second query conditioned on what the first hop returned; this simple loop beats any single-shot approach.
- **Related notebooks.** `02_Table_Question_Answering` (structured sources), `06_Summarization` (the other grounded-generation task, same faithfulness problem), `07_Feature_Extraction` and `11_Text_Ranking` (the retrieval half), `08_Text_Generation` (decoding and prompting), `04_Zero_Shot_Classification` (NLI, used for groundedness checking).

------------------------------------------------------------------------

Back to top
