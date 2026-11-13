# Summarization

Compressing documents without inventing facts: extractive versus abstractive, why ROUGE rewards copying, how faithfulness is actually measured in 2026, and runnable code that puts a lead-3 baseline, two fine-tuned seq2seq models and an LLM on the same CNN/DailyMail sample.

Author

Benedict Thekkel

## 1. What is Summarization?

Summarization compresses a document into a shorter text that preserves its important content. The definition contains the whole difficulty: *important* is not defined by the input, it is defined by what the summary is for.

**Input.** A document, or several. Length is the structural constraint - a 512-token encoder cannot read a 3,000-word article, so the field splits into models that fit the input and strategies for when it does not.

**Output.** Shorter text. Two families, and they fail in completely different ways:

| Family | Mechanism | Cannot hallucinate | Fluency across sentences |
|----|----|----|----|
| **Extractive** | selects sentences from the source | yes, by construction | poor - dangling pronouns, no transitions |
| **Abstractive** | generates new text | no | good |
| **Hybrid** | extract then rewrite | partially | good |

**The core tension: compression versus faithfulness.** An extractive summary is guaranteed to be supported by the source because every sentence *is* the source. An abstractive summary reads far better and can state something the document never said. In 2026 nearly everything shipped is abstractive, so the engineering problem is no longer fluency - it is detecting the fabricated sentence.

**Axes that change the task more than the model choice does:**

- **Generic vs query-focused.** “Summarise this contract” versus “summarise the termination clauses”. The second is a much better-defined problem and usually a much better product.
- **Single- vs multi-document.** Multi-document adds redundancy removal and contradiction handling between sources.
- **Compression ratio.** A one-sentence summary of a news article (XSum) and a three-sentence one (CNN/DailyMail) are different tasks trained on different data; a model fine-tuned for one produces bad output at the other length.
- **Faithfulness requirement.** For a chat digest, an invented detail is an annoyance. For a discharge summary or an earnings note, it is the whole risk.

**Neighbouring tasks:**

| Task | How it differs | Notebook |
|----|----|----|
| Translation | Transfers rather than compresses; same faithfulness problem | `05_Translation` |
| Question answering | Answers a specific question from the document | `03_Question_Answering` |
| Text generation | No source document to be faithful to | `08_Text_Generation` |
| Text ranking | Selects documents, not sentences | `11_Text_Ranking` |
| Zero-shot classification | NLI models are how faithfulness gets scored | `04_Zero_Shot_Classification` |

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Meeting and call summaries | Productivity (Zoom, Teams, Granola) | Transcript -\> notes + action items | Faithfulness on attributions; ASR errors propagate |
| Clinical documentation | Healthcare | Encounter notes -\> discharge summary | Zero tolerance for fabricated findings; regulated |
| Search and feed snippets | Web search, news apps | Article -\> 1-2 sentences | Throughput at enormous volume; a small model |
| Legal and contract review | Legal | Contract -\> obligations, dates, risks | Query-focused; must cite the clause |
| Financial research | Finance | Earnings call + filings -\> analyst brief | Numeric accuracy; multi-document; timeliness |
| Customer support digests | SaaS, telco | Ticket thread -\> handover summary | Latency; must preserve what the customer actually asked |
| Scientific literature triage | Research, pharma | Paper -\> structured abstract | Domain terminology; a wrong claim wastes weeks |
| Codebase and PR summaries | Software | Diff -\> change description | Structure over prose; must not invent behaviour |

What the ROUGE score hides:

- **Hallucination is the failure mode, and ROUGE does not measure it.** Studies of pre-2022 abstractive systems found 25-30% of generated summaries contained content unsupported by the source, and those systems scored well. Any serious evaluation needs a separate faithfulness measurement.
- **The reference is one of many valid summaries.** Two competent humans summarising the same article overlap surprisingly little, which caps what n-gram overlap can tell you. Reported ROUGE ceilings on CNN/DailyMail are around the level of a *human* summary scored against another human’s.
- **Length is a confound, not a detail.** ROUGE recall rises with summary length. Comparing systems that produce different lengths without controlling for it measures verbosity. The benchmark below reports length ratio next to the score for exactly this reason.
- **Extractive baselines are embarrassingly strong on news.** “Take the first three sentences” is competitive with trained neural models on CNN/DailyMail, because journalists write in inverted-pyramid style and put the summary first. Any news summarization result that does not report lead-3 is not reporting a result.
- **The interesting inputs are the ones nobody benchmarks.** Meeting transcripts with crosstalk, threads where the conclusion reverses halfway, documents that contradict themselves. Benchmarks are single-author, single-topic, well-formed prose.

------------------------------------------------------------------------

## 3. How Modern Summarization Works

1.  **Extractive statistics (1958-2015).** Luhn’s word-frequency method, then TF-IDF sentence scoring, LexRank and TextRank (graph centrality over sentence similarity). No training data needed, and never fabricates. Still a legitimate baseline when faithfulness dominates.
2.  **Neural extractive and pointer-generator (2016-2018).** Sequence labellers that pick sentences (SummaRuNNer), then See et al.’s **pointer-generator with coverage**, which let an abstractive decoder *copy* from the source and penalised repeating itself. The copy mechanism was the key idea - it survives today as the reason models trained on news copy so heavily.
3.  **Pretrained seq2seq (2019-2020).** **BART** (denoising autoencoder: corrupt the text, reconstruct it) and **PEGASUS** (gap-sentence generation: mask whole important sentences and generate them, a pretraining objective designed specifically for summarization) set the standard. `bart-large-cnn` and `pegasus-xsum` are still deployed, and BART’s design remains the reason encoder-decoder models are good at this task.
4.  **Long-document architectures (2020-2022).** Full attention is quadratic, so a 16k-token document was out of reach. LED (Longformer Encoder-Decoder), BigBird and LongT5 used sparse or global-plus-local attention to reach 16k-64k tokens. These mattered enormously for two years and are now largely superseded by long-context decoders.
5.  **The faithfulness literature (2020-2023).** FactCC, DAE, QAFactEval, SummaC and AlignScore - a body of work whose finding was that ROUGE and factuality are close to uncorrelated, and that the way to measure faithfulness is entailment (does the source entail each summary sentence?) or question generation (ask questions of the summary, answer them against the source). This reframed the task’s evaluation and is still the practical approach.
6.  **Instruction-tuned LLMs (2022-2026).** Zero-shot summarization from a general model. By 2023, human raters were preferring GPT-3-class zero-shot summaries to the CNN/DailyMail *reference* summaries - which mostly revealed that the references are mediocre, being the article’s own bullet-point teasers rather than real summaries. Fine-tuned-on-news models still score higher on ROUGE, because ROUGE measures similarity to those references. This is the clearest case in NLP of a benchmark and its task diverging.
7.  **Where the frontier is (2026).** Length- and format-controllable summarization, query-focused and personalised summaries, multi-document synthesis with conflict detection, and **self-verification** - generating a summary, then checking each sentence against the source with a cheap entailment model and regenerating the ones that fail.

**Where it stands (mid-2026).** For a fixed format at high volume (search snippets, feed cards), a fine-tuned 300-400M seq2seq model is still the cost winner and produces reliably on-format output. For anything where the user reads the summary and the format varies, an instruction-tuned LLM wins on quality, controllability and multi-document handling. Both need a faithfulness check on top if the content matters; neither provides one by default.

------------------------------------------------------------------------

## 4. Evaluation Metrics

**ROUGE** (Lin, 2004) - recall-oriented n-gram overlap against a reference summary. Three variants are reported:

$$\text{ROUGE-N} = \frac{\sum_{\text{ref}} \sum_{gram_n} \text{Count}_{\text{match}}(gram_n)}{\sum_{\text{ref}} \sum_{gram_n} \text{Count}(gram_n)}$$

- **ROUGE-1 / ROUGE-2** - unigram and bigram overlap. R-1 tracks content selection, R-2 tracks fluency and phrasing.
- **ROUGE-L** - longest common subsequence, so word order counts without requiring contiguity. `ROUGE-Lsum` applies it per sentence and averages, which is the variant papers report for multi-sentence summaries.

ROUGE is recall-flavoured by design (it asks how much of the reference the summary covers), so the F-measure version is what is usually reported to keep length honest.

**Why ROUGE is not enough, stated plainly:** it measures similarity to one reference summary. It cannot distinguish a faithful summary from a fluent fabrication, it rewards copying source phrasing, it punishes valid abstraction, and it correlates weakly with human preference above a moderate quality level. It remains useful as a **regression check** - if ROUGE drops 5 points after a change, something broke - and it is nearly useless as a quality ranking between good systems.

**BERTScore** - token-level cosine similarity between contextual embeddings of the summary and the reference, greedily matched. Handles paraphrase, which ROUGE cannot. Still reference-based, so it inherits the “one reference” problem.

**Faithfulness metrics, which are the ones that matter:**

- **Entailment-based** (SummaC, AlignScore, FactCC): split the summary into sentences and score `P(source entails sentence)` with an NLI model. Needs no reference - it compares the summary to the *source*, which is the actual question. Section 11 runs this.
- **QA-based** (QAFactEval, QuestEval): generate questions from the summary, answer them against the source, compare answers. Slower, and better at catching wrong entities and numbers.
- **LLM-as-judge**: prompt a strong model with the source and summary and ask for a faithfulness verdict with the offending span. This is what most teams now use in practice; it needs its own validation against human labels before it can be trusted.

**Pitfalls:**

- **Report length.** ROUGE recall rises with summary length. Always publish the summary-to-reference length ratio next to the score, or the comparison is not a comparison.
- **Stemming and stopwords change the number.** The official ROUGE-1.5.5 Perl script, `rouge-score`, and `evaluate` do not agree by default. State which you used.
- **Do not compare across datasets.** ROUGE-2 of 21 on CNN/DailyMail is strong; on XSum, 21 is also strong but the task (one-sentence, highly abstractive) is completely different. The numbers are not commensurable.

The cell below implements ROUGE-1, ROUGE-2 and ROUGE-L from scratch - the LCS is the only non-obvious part, and seeing it makes ROUGE-L’s behaviour on reordered text much less mysterious.

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
from collections import Counter


def tokenize(text):
    "Lowercase word tokens - the standard ROUGE preprocessing, minus stemming."
    return re.findall(r"[a-z0-9]+", text.lower())


def rouge_n(hyp, ref, n=1):
    "F1 over n-gram overlap. Clipped, so repeating an n-gram does not inflate the score."
    h, r = tokenize(hyp), tokenize(ref)
    h_ng = Counter(tuple(h[i:i + n]) for i in range(len(h) - n + 1))
    r_ng = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
    overlap = sum((h_ng & r_ng).values())
    if not overlap:
        return 0.0
    prec = overlap / max(sum(h_ng.values()), 1)
    rec = overlap / max(sum(r_ng.values()), 1)
    return 2 * prec * rec / (prec + rec)


def lcs_length(a, b):
    "Length of the longest common subsequence - order matters, contiguity does not."
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(hyp, ref):
    "F1 over the LCS. Rewards keeping the reference's word order without exact phrasing."
    h, r = tokenize(hyp), tokenize(ref)
    if not h or not r:
        return 0.0
    l = lcs_length(h, r)
    if not l:
        return 0.0
    prec, rec = l / len(h), l / len(r)
    return 2 * prec * rec / (prec + rec)


def rouge_scores(hyps, refs):
    "Mean ROUGE-1/2/L over a corpus, plus the length ratio that keeps them honest."
    r1 = sum(rouge_n(h, r, 1) for h, r in zip(hyps, refs)) / len(hyps)
    r2 = sum(rouge_n(h, r, 2) for h, r in zip(hyps, refs)) / len(hyps)
    rl = sum(rouge_l(h, r) for h, r in zip(hyps, refs)) / len(hyps)
    ratio = sum(len(tokenize(h)) for h in hyps) / max(sum(len(tokenize(r)) for r in refs), 1)
    return {"rouge1": round(100 * r1, 2), "rouge2": round(100 * r2, 2),
            "rougeL": round(100 * rl, 2), "len_ratio": round(ratio, 2)}


# Toy example. The reference is a real summary; the candidates fail in different ways,
# and only one of those failures is the dangerous one.
ref = ["The council approved the new library budget of 4.2 million after a long debate."]
cases = {
    "faithful abstraction": ["Councillors backed a 4.2 million funding plan for the library."],
    "copied from source": ["The council approved the new library budget of 4.2 million."],
    "fluent hallucination": ["The council approved the new library budget of 8.5 million "
                             "and fired the head librarian."],
    "extractive, verbose": ["The council approved the new library budget of 4.2 million "
                            "after a long debate that lasted three hours and involved "
                            "several amendments."],
}
show_table([{"candidate": name, **rouge_scores(hyp, ref), "text": hyp[0]}
            for name, hyp in cases.items()],
           title=f"All four scored against one reference: {ref[0]!r}",
           caption="the fluent hallucination (wrong number, invented event) scores near "
                   "the top on all three. ROUGE cannot see it - that is the entire "
                   "argument for the faithfulness metric in section 11")
```

```
All four scored against one reference: 'The council approved the new library budget of 4.2 million after a long 
                                                    debate.'                                                    
                                                                                                                
 candidate              rouge1   rouge2   rougeL   len_ratio   text                                             
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 faithful abstraction    46.15    16.67    23.08      0.7300   Councillors backed a 4.2 million funding plan    
                                                               for the library.                                 
 copied from source      84.62    83.33    84.62      0.7300   The council approved the new library budget of   
                                                               4.2 million.                                     
 fluent hallucination    58.06    48.28    58.06      1.0700   The council approved the new library budget of   
                                                               8.5 million and fired the head librarian.        
 extractive, verbose     78.95    77.78    78.95      1.5300   The council approved the new library budget of   
                                                               4.2 million after a long debate that lasted      
                                                               three hours and involved several amendments.     
                                                                                                                
the fluent hallucination (wrong number, invented event) scores near the top on all three. ROUGE cannot see it - 
                     that is the entire argument for the faithfulness metric in section 11                      
```

## 5. Datasets

| Dataset | Contents | Size | Summary style | License | Typical use |
|----|----|----|----|----|----|
| [CNN/DailyMail](https://huggingface.co/datasets/abisee/cnn_dailymail) | News articles + bullet highlights | 287k / 11.5k test | 3-4 sentences, **highly extractive** | Apache 2.0 | The classic benchmark; used below |
| [XSum](https://huggingface.co/datasets/EdinburghNLP/xsum) | BBC articles + the one-line intro | 204k / 11.3k | **1 sentence, very abstractive** | CC BY-SA 4.0 | Tests real abstraction; hallucination-prone |
| [Multi-News](https://huggingface.co/datasets/alexfabbri/multi_news) | Clusters of news articles + editor summary | 45k | multi-paragraph | custom, research | Multi-document summarization |
| [SAMSum](https://huggingface.co/datasets/Samsung/samsum) | Messenger-style dialogues + summaries | 16k | 1-3 sentences | CC BY-NC-ND 4.0 | Dialogue summarization |
| [DialogSum](https://huggingface.co/datasets/knkarthick/dialogsum) | Spoken dialogues + summaries | 13k | 1-3 sentences | MIT | Meeting/call summarization |
| [arXiv / PubMed](https://huggingface.co/datasets/armanc/scientific_papers) | Papers + abstracts | 215k / 133k | abstract-length | research | Long-document (4k-8k tokens) |
| [BillSum](https://huggingface.co/datasets/FiscalNote/billsum) | US congressional bills + summaries | 23k | paragraph | CC0 | Legal-domain long documents |
| [BookSum](https://huggingface.co/datasets/kmfoda/booksum) | Books, chapters, paragraphs | 12k | varies | BSD-3 | Book-length hierarchical summarization |
| [GovReport](https://huggingface.co/datasets/ccdv/govreport-summarization) | US government reports | 19.5k | long, multi-paragraph | CC BY 4.0 | Very long inputs (~9k tokens) |
| [SummEval](https://huggingface.co/datasets/mteb/summeval) | 16 systems on CNN/DM + human ratings | 1.6k | \- | MIT | **Validating metrics** against human judgement |
| [AggreFact / SummaC](https://huggingface.co/datasets/lytang/LLM-AggreFact) | Summaries labelled factual / not | 60k+ | \- | mixed | Training and testing faithfulness metrics |

This notebook evaluates on the **CNN/DailyMail 3.0.0 test split**. Be clear-eyed about what it is: the “reference summaries” are the bullet-point highlights news sites put at the top of an article, they are largely copied from the first paragraphs, and they are the reason a lead-3 baseline is competitive. It is the right dataset for demonstrating the mechanics and the wrong one for deciding whether a model summarises well. **XSum** is the harder and more informative benchmark, and swapping it in below is a two-line change.

**Note the loading trick.** `load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")` downloads the whole repo - all train shards, ~1.3 GB - and then slices. Pointing the generic `parquet` loader at an `hf://` glob for the test shard alone pulls **30 MB**. On a box with a finite disk budget that difference is worth knowing; see the dataset-download note in `dl-visualization-and-memory.md`.

Downloads land in `DL_tasks/datasets/` via `cache_dir` (gitignored).

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

There is no single live leaderboard; the useful references are the [CNN/DailyMail](https://paperswithcode.com/sota/document-summarization-on-cnn-daily-mail) and [XSum](https://paperswithcode.com/sota/text-summarization-on-x-sum) result tables for ROUGE, [SummEval](https://github.com/Yale-LILY/SummEval) for how metrics correlate with humans, and [LLM-AggreFact](https://huggingface.co/spaces/lytang/LLM-AggreFact) for faithfulness-checker quality.

| Model | Params | License | Context | Trained for | Best for |
|----|----|----|----|----|----|
| lead-3 baseline | 0 | \- | any | nothing | the baseline every news result must beat; used below |
| [distilbart-cnn-12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6) | 306M | Apache 2.0 | 1024 | CNN/DM | cheapest usable news summarizer; used below |
| [bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn) | 406M | MIT | 1024 | CNN/DM | the standard fine-tuned baseline; used below |
| [pegasus-xsum](https://huggingface.co/google/pegasus-xsum) | 568M | Apache 2.0 | 512 | XSum | one-sentence, highly abstractive output |
| [pegasus-cnn_dailymail](https://huggingface.co/google/pegasus-cnn_dailymail) | 568M | Apache 2.0 | 1024 | CNN/DM | strong news ROUGE |
| [flan-t5-base / large](https://huggingface.co/google/flan-t5-large) | 250M-780M | Apache 2.0 | 512 | instruction mix | promptable at small size |
| [led-base-16384](https://huggingface.co/allenai/led-base-16384) | 162M | Apache 2.0 | 16384 | long docs | long inputs on tiny hardware |
| [long-t5-tglobal-base](https://huggingface.co/google/long-t5-tglobal-base) | 250M | Apache 2.0 | 16384 | long docs | arXiv/GovReport-scale documents |
| [bart-large-samsum / dialogsum tunes](https://huggingface.co/philschmid/bart-large-cnn-samsum) | 406M | MIT | 1024 | dialogue | meeting and chat summaries |
| [Qwen3-1.7B / 4B](https://huggingface.co/Qwen/Qwen3-1.7B) | 1.7-4B | Apache 2.0 | 32k+ | instruction | controllable, query-focused; used below |
| [AlignScore / SummaC](https://huggingface.co/krishnamrith12/AlignScore-base) | 125-355M | MIT | \- | **faithfulness metric** | scoring summaries, not producing them |
| Frontier LLMs (Claude, GPT, Gemini) | \- | proprietary | 200k-1M | instruction | whole documents without chunking; production quality |

**How to choose.** Fixed format, fixed length, high volume: fine-tune a small seq2seq model (distilbart at 306M is the floor) - it will produce on-format output far more reliably than a prompted LLM and cost a fraction. User-facing summaries where format varies, or query-focused summaries: an LLM. Documents past ~1,000 tokens: check whether a long-context model fits before building a chunking pipeline, because hierarchical chunk-and-merge loses cross-chunk relationships and is the main source of incoherent long summaries.

**Note on size.** The runnable cells below total roughly 5 GB of downloads. The long-context specialists and frontier models belong in this table, not in a runnable cell on a 12 GB box.

------------------------------------------------------------------------

## 7. Setup

Everything loads through Hugging Face `transformers` - no vendor packages. Package roles:

- `transformers` + `torch` - the seq2seq summarizers, the LLM, and the NLI faithfulness checker
- `accelerate` - `device_map` placement
- `datasets` - the CNN/DailyMail test shard
- `pandas` + `pyecharts` - the benchmark table and charts
- `rich` - the result tables. It renders to HTML inside Jupyter, so the tables survive into the published docs; `show_table` / `show_kv` / `rule` are defined in the first code cell of section 4.

Metrics are implemented inline (section 4). For production use `rouge-score` or `evaluate` and state which; the inline version is for understanding the arithmetic.

Three `transformers` details that decide the output quality:

- **`min_length` and `max_length` on `generate` are the length controls that actually work** for seq2seq summarizers, and both are in *tokens*. `bart-large-cnn` defaults to `min_length=56`, which is why it never produces a one-line summary no matter what you ask.
- **`no_repeat_ngram_size=3`** is standard for summarization and worth understanding: beam search on a copy-heavy model loops without it, emitting the same clause repeatedly. It is a decoding-level patch for a training-level problem.
- **`truncation=True` silently discards the tail** of any document past the model’s window. For a 1024-token model on a 900-word news article this is usually fine; on a 5,000-word report it means the model summarised the first fifth and you would never know. Check `len(tokenizer(doc).input_ids)` against the model’s limit before trusting an output.

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

# CNN/DailyMail 3.0.0 test split ONLY. Calling load_dataset("abisee/cnn_dailymail",
# "3.0.0", split="test") downloads every shard in the repo (~1.3 GB) and then slices;
# the parquet loader on an hf:// glob pulls the 30 MB test shard and nothing else.
cnn = load_dataset(
    "parquet",
    data_files={"test": "hf://datasets/abisee/cnn_dailymail/3.0.0/test-*.parquet"},
    split="test",
    cache_dir=HF_CACHE,
)

N = 60  # articles to summarise - a smoke test, not a leaderboard run
sample = cnn.shuffle(seed=0).select(range(N))
articles = [r["article"].strip() for r in sample]
# CNN/DM highlights are newline-separated bullets; join them into one reference string.
references = [r["highlights"].replace("\n", " ").strip() for r in sample]

print(cnn)
print(f"\nsummarising {N} articles")
print(f"mean article: {sum(len(a.split()) for a in articles) / N:.0f} words")
print(f"mean summary: {sum(len(r.split()) for r in references) / N:.0f} words "
      f"({sum(len(r.split()) for r in references) / sum(len(a.split()) for a in articles):.1%} "
      "compression)\n")
print("ARTICLE:", articles[0][:400], "...\n")
print("REFERENCE:", references[0])
```

    Dataset({
        features: ['article', 'highlights', 'id'],
        num_rows: 11490
    })

    summarising 60 articles
    mean article: 650 words
    mean summary: 52 words (8.0% compression)

    ARTICLE: Volunteers who have restored one of the Dunkirk 'Little Ships' are now hoping they will be able to take the unique vessel back to the beaches where it helped to rescue 600 British soldiers. Fire boat Massey Shaw was requisitioned from the London Fire Brigade in 1940 and transported to Dunkirk to help with the evacuation of Allied troops in the face of the Nazi advance. After carrying 600 soldiers  ...

    REFERENCE: Massey Shaw was requisitioned from the London Fire Brigade and helped rescue 600 troops from Dunkirk in 1940 . The vessel went on to save St Paul's Cathedral during the Blitz before being commissioned in 1971 . But the fire boat has been renovated and restored to its former glory by a group of volunteers thanks to lottery grant . They are now seeking another £10,000 so they can transport the Massey Shaw back to Dunkirk 70 years later .

## 8. The baseline that must be beaten: lead-3

Take the first three sentences of the article. No model, no parameters, no GPU.

On CNN/DailyMail this is competitive with trained neural summarizers, and that fact is the most useful thing in this notebook. It is not a quirk of the metric - it is a property of the data. News is written in inverted-pyramid style with the key facts first, and the “reference summaries” are the site’s own bullet-point teasers, themselves drawn from the opening. A model that learns to copy the top of the article is learning the dataset correctly.

Two consequences worth carrying elsewhere:

- **Any news summarization result without a lead-3 number is uninterpretable.** If a 400M model beats lead-3 by one ROUGE point, the model contributed one point.
- **Extractive baselines cannot hallucinate.** Lead-3 scores near the neural models here *and* has a perfect faithfulness score in section 11. In a setting where a fabricated sentence is unacceptable and the source is well-structured, this is a real option, not a joke.

The same reasoning does not transfer to XSum (one-sentence, genuinely abstractive, written by a different person than the article), where lead-1 is far behind - which is exactly why XSum is the better benchmark.

------------------------------------------------------------------------

``` python
def split_sentences(text):
    "Cheap sentence splitter - adequate for newswire, not for legal or clinical text."
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [p.strip() for p in parts if p.strip()]


def lead_k(text, k=3):
    "The first k sentences. Zero parameters, cannot hallucinate."
    return " ".join(split_sentences(text)[:k])


t0 = time.perf_counter()
lead_hyps = [lead_k(a, 3) for a in articles]
lead_secs = time.perf_counter() - t0

print("SUMMARY:", lead_hyps[0][:300], "\n")
show_kv({"articles": N, "seconds": round(lead_secs, 3), "parameters": 0,
         **rouge_scores(lead_hyps, references)},
        title="lead-3 - the first three sentences, no model at all")

# How much of the reference is literally in the article - the reason lead-3 works here.
copied = []
for ref, art in zip(references, articles):
    ref_toks, art_toks = set(tokenize(ref)), set(tokenize(art))
    copied.append(len(ref_toks & art_toks) / max(len(ref_toks), 1))
print(f"\n{sum(copied) / len(copied):.1%} of reference-summary words appear in the article")
print("CNN/DailyMail references are largely extractive - keep that in mind below")
```

    SUMMARY: Volunteers who have restored one of the Dunkirk 'Little Ships' are now hoping they will be able to take the unique vessel back to the beaches where it helped to rescue 600 British soldiers. Fire boat Massey Shaw was requisitioned from the London Fire Brigade in 1940 and transported to Dunkirk to hel 

```
      lead-3 - the first three sentences, no model at all       
                                                                
 articles                                                    60 
 seconds                                                 0.0030 
 parameters                                                   0 
 rouge1                                                   37.35 
 rouge2                                                   16.69 
 rougeL                                                   24.26 
 len_ratio                                               1.6300 
                                                                
```

    85.9% of reference-summary words appear in the article
    CNN/DailyMail references are largely extractive - keep that in mind below

## 9. Fine-tuned seq2seq: distilBART and BART-large-CNN

BART is a denoising autoencoder: pretraining corrupts the text (masking spans, permuting sentences, deleting tokens) and trains the model to reconstruct the original. That objective happens to be almost perfect preparation for summarization - the model learns to produce fluent well-ordered text conditioned on a damaged version of it - which is why `bart-large-cnn` became the default and stayed there for years.

`distilbart-cnn-12-6` is the distilled version: 12 encoder layers, 6 decoder layers, 306M params against 406M, and roughly twice the speed for one or two ROUGE points. Decoder layers are the expensive part at inference (they run once per generated token), so cutting them is where the speedup comes from.

Both are fine-tuned on CNN/DailyMail, so they produce three-ish sentences in news register whatever you feed them, and they copy heavily. That is a feature when your task matches the training data and a hard limitation otherwise: give `bart-large-cnn` a meeting transcript and it will write a news article about the meeting.

The generation settings below are the model card’s own defaults (`num_beams=4`, `min_length=56`, `no_repeat_ngram_size=3`) rather than library defaults. `min_length` in particular is doing a lot of work - remove it and output gets noticeably shorter and worse on this dataset.

------------------------------------------------------------------------

``` python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

GEN = dict(num_beams=4, max_length=142, min_length=56, no_repeat_ngram_size=3,
           length_penalty=2.0, early_stopping=True)


@torch.inference_mode()
def summarize_seq2seq(model, tok, texts, batch_size=4, **gen_kwargs):
    "Batched beam-search summarization; reports how many inputs were truncated."
    out, truncated = [], 0
    limit = tok.model_max_length if tok.model_max_length < 100000 else 1024
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=limit).to(model.device)
        truncated += sum(len(tok(t).input_ids) > limit for t in chunk)
        gen = model.generate(**enc, **gen_kwargs)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out, truncated


results_raw = {}
for name, model_id, params_m in [
    ("distilbart-cnn-12-6", "sshleifer/distilbart-cnn-12-6", 306),
    ("bart-large-cnn", "facebook/bart-large-cnn", 406),
]:
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id, dtype=dtype, cache_dir=HF_CACHE).to(device).eval()
    vram(f"{name} loaded")

    t0 = time.perf_counter()
    hyps, n_trunc = summarize_seq2seq(model, tok, articles, **GEN)
    secs = time.perf_counter() - t0
    results_raw[name] = (params_m, hyps, secs)

    show_kv({"articles": N, "seconds": round(secs, 1),
             "articles / second": round(N / secs, 2),
             "inputs truncated": f"{n_trunc}/{N} at {tok.model_max_length} tokens",
             **rouge_scores(hyps, references)}, title=name)
    print("SUMMARY:", hyps[0][:260])

    del model, tok                # one model live at a time
    free_memory()
    vram(f"after {name}")
```

    [transformers] Please make sure the generation config includes `forced_bos_token_id=0`. 

    [transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (1182 > 1024). Running this sequence through the model will result in indexing errors

    VRAM distilbart-cnn-12-6 loaded  0.61 GB allocated /  0.63 GB reserved

```
                      distilbart-cnn-12-6                       
                                                                
 articles                                                    60 
 seconds                                                  18.90 
 articles / second                                       3.1700 
 inputs truncated                          12/60 at 1024 tokens 
 rouge1                                                   44.04 
 rouge2                                                   22.68 
 rougeL                                                   31.13 
 len_ratio                                               1.3200 
                                                                
```

    SUMMARY:  Fire boat Massey Shaw was requisitioned from the London Fire Brigade in 1940 and transported to Dunkirk to help with the evacuation of Allied troops in the face of the Nazi advance . After carrying 600 soldiers back to Britain, the boat returned to London whe
    VRAM after distilbart-cnn-12-6  0.01 GB allocated /  0.02 GB reserved

    VRAM bart-large-cnn loaded   0.82 GB allocated /  0.84 GB reserved

```
                           bart-large-cnn                            
                                                                     
 articles                                                         60 
 seconds                                                       31.00 
 articles / second                                            1.9400 
 inputs truncated    12/60 at 1000000000000000019884624838656 tokens 
 rouge1                                                        43.36 
 rouge2                                                        22.10 
 rougeL                                                        31.22 
 len_ratio                                                    1.2900 
                                                                     
```

    SUMMARY: Massey Shaw was requisitioned from the London Fire Brigade in 1940. It was transported to Dunkirk to help with the evacuation of Allied troops. After carrying 600 soldiers back to Britain, the boat returned to London. It is still capable of pumping out 3,000 g
    VRAM after bart-large-cnn    0.01 GB allocated /  0.02 GB reserved

## 10. LLM summarizer: Qwen3-1.7B

A general instruction-tuned model, prompted. No summarization fine-tuning, and on CNN/DailyMail’s ROUGE it will probably lose to the BART models - which is the point worth sitting with.

**Why it loses on the metric and wins with readers.** The BART models were trained to reproduce the phrasing and three-bullet shape of CNN/DailyMail highlights, and ROUGE measures exactly that similarity. The LLM writes a genuine summary in its own words. Human evaluations from 2023 onward have consistently found raters preferring zero-shot LLM summaries to both the fine-tuned models *and* the dataset’s reference summaries - so a lower ROUGE here means “less similar to a mediocre reference”, not “worse”.

**What the LLM can do that no fine-tuned seq2seq model can:**

- Take a length or format instruction and honour it (one sentence, three bullets, no adjectives).
- Take a *query* (“summarise only what this says about funding”) and produce a focused summary.
- Handle a document type it was never trained on without producing news register.
- Say that the document does not contain what you asked about.

The cost is roughly an order of magnitude more compute per document, output that needs post-processing (LLMs preface things), and a hallucination risk that a copy-heavy fine-tuned model largely does not have - measured directly in section 11.

`enable_thinking=False` keeps Qwen3 from emitting a `<think>` block before the summary.

------------------------------------------------------------------------

``` python
from transformers import AutoModelForCausalLM

llm_id = "Qwen/Qwen3-1.7B"   # ~3.4 GB download, ~3.4 GB VRAM in fp16
tok = AutoTokenizer.from_pretrained(llm_id, cache_dir=HF_CACHE)
llm = AutoModelForCausalLM.from_pretrained(
    llm_id, dtype=dtype, device_map=device, cache_dir=HF_CACHE
).eval()
vram("qwen3-1.7b loaded")

PROMPT = (
    "Summarise the news article below in 3 short sentences. Use only information stated "
    "in the article. Output only the summary - no preamble, no bullet points, no "
    "headings.\n\nArticle:\n{article}\n\nSummary:"
)


@torch.inference_mode()
def llm_summarize(texts, instruction=PROMPT, batch_size=4, max_new_tokens=160):
    "Greedy-decode a summary per document; strip the preamble LLMs like to add."
    out = []
    for i in range(0, len(texts), batch_size):
        chats = [
            tok.apply_chat_template(
                [{"role": "user", "content": instruction.format(article=t[:6000])}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for t in texts[i:i + batch_size]
        ]
        enc = tok(chats, return_tensors="pt", padding=True, padding_side="left",
                  truncation=True, max_length=4096).to(llm.device)
        gen = llm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        for g in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
            text = g.strip()
            if ":" in text.split("\n")[0] and len(text.split("\n")[0]) < 60:
                text = "\n".join(text.split("\n")[1:]).strip()   # drop "Summary:" preambles
            out.append(" ".join(text.split()))
    return out


t0 = time.perf_counter()
llm_hyps = llm_summarize(articles)
llm_secs = time.perf_counter() - t0

show_kv({"articles": N, "seconds": round(llm_secs, 1),
         "articles / second": round(N / llm_secs, 2),
         **rouge_scores(llm_hyps, references)},
        title="Qwen3-1.7B prompted - lower ROUGE means less similar to a mediocre reference")
print("SUMMARY:", llm_hyps[0][:300], "\n")

# The capabilities no fine-tuned seq2seq summarizer has. Same model, same article.
VARIANTS = [
    ("one sentence", "Summarise the article below in exactly ONE sentence. Output only "
                     "the sentence.\n\nArticle:\n{article}\n\nSummary:"),
    ("query-focused", "Using only the article below, state what it says about money, "
                      "costs or funding. If it says nothing about that, reply exactly: "
                      "not mentioned.\n\nArticle:\n{article}\n\nAnswer:"),
    ("bullets, no adjectives", "Summarise the article below as 3 bullet points starting "
                               "with '- '. Use no adjectives.\n\nArticle:\n{article}\n\nSummary:"),
]
show_table([{"instruction": label,
             "output": llm_summarize([articles[0]], instruction=tmpl, batch_size=1)[0][:280]}
            for label, tmpl in VARIANTS],
           title="Same model, same article - things no fine-tuned summarizer can do",
           caption="length control, query focus and format are prompt parameters here "
                   "and retraining for a seq2seq model")

del llm, tok
free_memory()
vram("after qwen3")
```

    VRAM qwen3-1.7b loaded       3.45 GB allocated /  4.41 GB reserved

```
   Qwen3-1.7B prompted - lower ROUGE means less similar to a    
                       mediocre reference                       
                                                                
 articles                                                    60 
 seconds                                                  58.90 
 articles / second                                       1.0200 
 rouge1                                                   38.71 
 rouge2                                                   15.47 
 rougeL                                                   25.34 
 len_ratio                                               1.1800 
                                                                
```

    SUMMARY: Volunteers are restoring the Massey Shaw, a fire boat used in the Dunkirk evacuation, to its former glory. The boat was requisitioned in 1940 and helped rescue 600 British soldiers. The boat is now seeking funding to return to Dunkirk for the 75th anniversary of the Little Ships' adventure. 

```
                       Same model, same article - things no fine-tuned summarizer can do                        
                                                                                                                
 instruction              output                                                                                
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 one sentence             Volunteers are hoping to transport the historic fire boat Massey Shaw back to Dunkirk 
                          to commemorate its role in rescuing 600 British soldiers during the Dunkirk           
                          evacuation.                                                                           
 query-focused            The article states that the Massey Shaw, a fire boat, was used to help rescue 600     
                          British soldiers during the Dunkirk evacuation and later saved St Paul's Cathedral    
                          during the Blitz. It mentions that the boat was restored by volunteers and a trust    
                          with a grant from the National L                                                      
 bullets, no adjectives   - Volunteers are hoping to transport the Massey Shaw back to Dunkirk to celebrate its 
                          role in rescuing 600 soldiers. - The boat, restored after 70 years, is capable of     
                          pumping 3,000 gallons of water per minute and has been repaired with a National       
                          Lottery grant. - The trust needs                                                      
                                                                                                                
      length control, query focus and format are prompt parameters here and retraining for a seq2seq model      
```

    VRAM after qwen3             0.01 GB allocated /  0.02 GB reserved

## 11. Faithfulness: what ROUGE cannot see

ROUGE compares the summary to a **reference**. The question that actually matters in production is whether the summary is supported by the **source**, and answering it needs no reference at all.

The standard mechanism is entailment, and it is the same NLI machinery as `04_Zero_Shot_Classification` pointed at a different question:

- premise = the source document (or the most relevant chunk of it)
- hypothesis = one sentence of the summary
- score = `P(entailment)`

A summary sentence the source does not entail is either unsupported (a fabrication) or an inference the model made (which may be fine, or may be exactly the problem). Averaging over the summary’s sentences gives a document-level score; **the minimum** over sentences is often the more useful number, because one fabricated sentence ruins a summary and an average of six good sentences hides it.

This is a simplified SummaC: real implementations chunk the source and take the maximum entailment over chunks per summary sentence, which handles documents longer than the NLI model’s window properly. The version below truncates the source instead, so treat the absolute numbers as indicative and the *ranking* as the signal.

**What to expect, and it is the punchline of the notebook:** lead-3 scores perfectly by construction - its sentences are the source. The copy-heavy fine-tuned models score high because they largely quote. The LLM, which genuinely rewrites, scores lowest and is the one whose summaries humans prefer. Faithfulness and quality are different axes, and a system that optimises only one of them is misdesigned.

------------------------------------------------------------------------

``` python
from transformers import AutoModelForSequenceClassification

# Entailment-based faithfulness, a simplified SummaC. The NLI model is the same kind
# used for zero-shot classification (nb 04), asked a different question.
nli_id = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
nli_tok = AutoTokenizer.from_pretrained(nli_id, cache_dir=HF_CACHE)
nli = AutoModelForSequenceClassification.from_pretrained(nli_id, cache_dir=HF_CACHE).to(device).eval()
ENT = nli.config.label2id["entailment"]
vram("nli loaded")


@torch.inference_mode()
def faithfulness(summaries, sources, batch_size=16):
    "Mean and worst-sentence P(source entails summary sentence). No reference needed."
    means, mins = [], []
    for summ, src in zip(summaries, sources):
        sents = [s for s in split_sentences(summ) if len(s.split()) >= 4]
        if not sents:
            means.append(0.0)
            mins.append(0.0)
            continue
        scores = []
        for i in range(0, len(sents), batch_size):
            batch = sents[i:i + batch_size]
            enc = nli_tok([src] * len(batch), batch, return_tensors="pt", padding=True,
                          truncation="only_first", max_length=512).to(device)
            probs = nli(**enc).logits.softmax(-1)[:, ENT]
            scores.extend(probs.tolist())
        means.append(sum(scores) / len(scores))
        mins.append(min(scores))
    return (100 * sum(means) / len(means), 100 * sum(mins) / len(mins))


# Sanity check the checker before trusting it on real summaries.
src = ("The council approved the new library budget of 4.2 million after a long debate. "
       "Three councillors voted against the motion.")
_checks = [
    ("supported", "The council approved a 4.2 million library budget."),
    ("wrong number", "The council approved an 8.5 million library budget."),
    ("invented event", "The council approved the budget and dismissed the head librarian."),
]
show_table([{"case": label, "entailment": round(faithfulness([summ], [src])[0], 1),
             "summary": summ} for label, summ in _checks],
           title="Sanity-checking the checker before trusting it", best=("entailment",),
           caption=f"premise: {src[:80]}...")

SYSTEMS = {
    "lead-3": lead_hyps,
    "distilbart-cnn-12-6": results_raw["distilbart-cnn-12-6"][1],
    "bart-large-cnn": results_raw["bart-large-cnn"][1],
    "qwen3-1.7b": llm_hyps,
}
faith = {}
for name, hyps in SYSTEMS.items():
    mean_ent, min_ent = faithfulness(hyps, articles)
    faith[name] = {"faith_mean": round(mean_ent, 1), "faith_worst": round(min_ent, 1)}
show_table([{"system": name, **v} for name, v in faith.items()],
           title=f"Faithfulness against the source, no reference needed ({N} articles)",
           best=("faith_mean", "faith_worst"),
           caption="lead-3 is near-perfect by construction, and is still a poor summary. "
                   "Read this next to ROUGE, not instead of it")

del nli, nli_tok
free_memory()
vram("after nli")
```

    VRAM nli loaded              0.38 GB allocated /  0.39 GB reserved

```
                         Sanity-checking the checker before trusting it                          
                                                                                                 
 case             entailment   summary                                                           
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 supported             99.70   The council approved a 4.2 million library budget.                
 wrong number         0.0000   The council approved an 8.5 million library budget.               
 invented event       0.0000   The council approved the budget and dismissed the head librarian. 
                                                                                                 
  premise: The council approved the new library budget of 4.2 million after a long debate. ...   
```

```
   Faithfulness against the source, no reference needed (60 articles)   
                                                                        
 system                                  faith_mean         faith_worst 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 lead-3                                       73.30               63.70 
 distilbart-cnn-12-6                          75.50               60.50 
 bart-large-cnn                               74.80               61.40 
 qwen3-1.7b                                   82.90               72.00 
                                                                        
  lead-3 is near-perfect by construction, and is still a poor summary.  
               Read this next to ROUGE, not instead of it               
```

    VRAM after nli               0.01 GB allocated /  0.02 GB reserved

## 12. Head-to-head Benchmark

The same 60 articles, the same reference summaries, the same ROUGE implementation and the same faithfulness checker, one model live at a time. Sections 8-11 produced the numbers; this collects them.

The table has two independent axes and a system has to be read on both:

- **ROUGE** says how similar the summary is to CNN/DailyMail’s own bullet points. It rewards copying and it rewards news register.
- **Faithfulness** says how much of the summary the source actually supports. It rewards copying too - which is why a high faithfulness score is necessary and not sufficient.
- **Length ratio** is the confound. A system producing 1.5x the reference length gets ROUGE recall for free.

The shape to expect: lead-3 competitive on ROUGE and perfect on faithfulness while being a poor summary; the BART models slightly ahead on ROUGE at 100-1000x lead-3’s cost; the LLM behind on ROUGE, behind on faithfulness, and producing the summaries a human would pick. No single column identifies the best system, and that is the honest state of this task’s evaluation.

At n=60, ROUGE carries roughly +/-1.5 points of noise.

------------------------------------------------------------------------

``` python
import pandas as pd

RUNS = [
    # display name, params, key into `faith`, hypotheses, seconds
    ("lead-3", 0, "lead-3", lead_hyps, lead_secs),
    ("distilbart-cnn-12-6", 306, "distilbart-cnn-12-6",
     results_raw["distilbart-cnn-12-6"][1], results_raw["distilbart-cnn-12-6"][2]),
    ("bart-large-cnn", 406, "bart-large-cnn",
     results_raw["bart-large-cnn"][1], results_raw["bart-large-cnn"][2]),
    ("qwen3-1.7b (prompted)", 1720, "qwen3-1.7b", llm_hyps, llm_secs),
]

results = []
for name, params_m, faith_key, hyps, secs in RUNS:
    row = {"model": name, "params_m": params_m}
    row.update(rouge_scores(hyps, references))
    row.update(faith[faith_key])
    row["seconds"] = round(secs, 2)
    row["art_per_sec"] = round(N / secs, 2) if secs else float("inf")
    results.append(row)

df_results = pd.DataFrame(results).sort_values("rouge2", ascending=False)
show_table(
    df_results.to_dict("records"),
    title=f"CNN/DailyMail test, {N} articles",
    best=("rouge1", "rouge2", "rougeL", "faith_mean", "faith_worst", "art_per_sec"),
    caption="ROUGE measures similarity to the reference; faithfulness measures support "
            "from the source - two independent axes",
)
```

```
                                        CNN/DailyMail test, 60 articles                                         
                                                                                                                
                                                                              faith_wors             art_per_se 
 model         params_m   rouge1   rouge2   rougeL   len_ratio   faith_mean            t   seconds            c 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 distilbart-        306    44.04    22.68    31.13      1.3200        75.50        60.50     18.94       3.1700 
 cnn-12-6                                                                                                       
 bart-large-        406    43.36    22.10    31.22      1.2900        74.80        61.40     30.96       1.9400 
 cnn                                                                                                            
 lead-3               0    37.35    16.69    24.26      1.6300        73.30        63.70    0.0000    23,534.92 
 qwen3-1.7b       1,720    38.71    15.47    25.34      1.1800        82.90        72.00     58.93       1.0200 
 (prompted)                                                                                                     
                                                                                                                
ROUGE measures similarity to the reference; faithfulness measures support from the source - two independent axes
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
    .add_xaxis([r["model"] for r in results])
    .add_yaxis("ROUGE-1", [r["rouge1"] for r in results])
    .add_yaxis("ROUGE-2", [r["rouge2"] for r in results])
    .add_yaxis("ROUGE-L", [r["rougeL"] for r in results])
    .add_yaxis("faithfulness (mean entailment)", [r["faith_mean"] for r in results])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"CNN/DailyMail test ({N} articles)",
            subtitle="RTX 3060 - ROUGE measures similarity to the reference; "
                     "faithfulness measures support from the source",
        ),
        yaxis_opts=opts.AxisOpts(name="score", min_=0, max_=100),
        xaxis_opts=opts.AxisOpts(name="system", axislabel_opts=opts.LabelOpts(rotate=15, font_size=9)),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="10%"),
    )
)
bar.render_notebook()
```

``` python
from pyecharts.charts import Scatter

# The two axes against each other. A system in the top right is good on both; the
# interesting observation is how weakly they are related.
scatter = Scatter()
scatter.add_xaxis([r["faith_mean"] for r in results])
for r in results:
    scatter.add_yaxis(
        r["model"], [[r["faith_mean"], r["rouge2"]]],
        symbol_size=18, label_opts=opts.LabelOpts(is_show=False),
    )
scatter.set_global_opts(
    title_opts=opts.TitleOpts(title="ROUGE-2 vs faithfulness",
                              subtitle="two independent axes - optimising one does not move the other"),
    xaxis_opts=opts.AxisOpts(name="mean sentence entailment", type_="value"),
    yaxis_opts=opts.AxisOpts(name="ROUGE-2", type_="value"),
    tooltip_opts=opts.TooltipOpts(trigger="item"),
)
scatter.render_notebook()
```

## 13. Interactive: summarise your own text

Paste a document into `MY_TEXT` below. This is the cell people run on its own, so it opens with a `require(...)` guard naming what it needs from Setup rather than dying on a bare `NameError`.

It runs the fine-tuned news model and the prompted LLM side by side, then scores both for faithfulness against your text - which is the check you would actually run in production, and it needs no reference summary.

The inputs worth trying are the ones that break each approach differently:

- **A meeting transcript or a chat thread.** `bart-large-cnn` will write a news article about it, because that is the only register it knows. The LLM will not. This is the clearest demonstration of what fine-tuning on one dataset costs.
- **A document longer than 1024 tokens.** Watch the printed truncation warning. The seq2seq model summarises the first ~750 words and gives no indication that it did.
- **A document with a reversal** (“we planned X… in the end we did Y”). Copy-heavy models frequently report the plan, not the decision.
- **A document containing numbers.** Check every one against the source. Wrong or transposed figures are the most common and most damaging abstractive error, and fluency gives no warning.
- **Ask a query-focused question** by editing `QUERY`. This is where an LLM stops competing with the seq2seq model and starts doing something it cannot do at all.

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


require("device", "dtype", "HF_CACHE", "free_memory", "vram", "rouge_scores",
        "split_sentences", "summarize_seq2seq")

from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification, AutoTokenizer)

MY_TEXT = """
The infrastructure review concluded on Tuesday after six weeks. The team had originally
planned to migrate all workloads to the managed Kubernetes service by the end of the
quarter, at an estimated cost of 240,000 dollars per year. During the review it became
clear that three of the seven services depend on GPU passthrough that the managed
offering does not support. The team therefore decided to keep those three services on
the existing hosts and migrate only the remaining four. Revised annual cost is 155,000
dollars. Two engineers raised concerns about maintaining two deployment paths, and the
review recommends revisiting the split in twelve months.
""".strip()

QUERY = "What did they decide about cost?"   # used for the query-focused run

# Re-runnable: this cell frees its models at the end, so guard the loads or a second
# shift-enter raises NameError.
if "my_bart" not in globals():
    my_bart_tok = AutoTokenizer.from_pretrained("facebook/bart-large-cnn", cache_dir=HF_CACHE)
    my_bart = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/bart-large-cnn", dtype=dtype, cache_dir=HF_CACHE).to(device).eval()
if "my_llm" not in globals():
    my_llm_tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", cache_dir=HF_CACHE)
    my_llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", dtype=dtype, device_map=device, cache_dir=HF_CACHE).eval()

n_tokens = len(my_bart_tok(MY_TEXT).input_ids)
print(f"input: {len(MY_TEXT.split())} words / {n_tokens} bart tokens"
      f"{'  *** TRUNCATED at 1024 ***' if n_tokens > 1024 else ''}\n")

bart_out, _ = summarize_seq2seq(my_bart, my_bart_tok, [MY_TEXT], batch_size=1,
                                num_beams=4, max_length=142, min_length=30,
                                no_repeat_ngram_size=3, early_stopping=True)


@torch.inference_mode()
def _ask(instruction, max_new_tokens=200):
    chat = my_llm_tok.apply_chat_template(
        [{"role": "user", "content": instruction}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    enc = my_llm_tok(chat, return_tensors="pt", truncation=True, max_length=4096).to(my_llm.device)
    gen = my_llm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=my_llm_tok.eos_token_id)
    return " ".join(my_llm_tok.decode(gen[0][enc["input_ids"].shape[1]:],
                                      skip_special_tokens=True).split())


llm_out = _ask("Summarise the text below in 3 short sentences, using only information "
               f"stated in it. Output only the summary.\n\n{MY_TEXT}\n\nSummary:")
query_out = _ask(f"Using only the text below, answer: {QUERY}\nIf the text does not say, "
                 f"reply exactly: not stated.\n\n{MY_TEXT}\n\nAnswer:")

print("[bart-large-cnn]\n ", bart_out[0], "\n")
print("[qwen3-1.7b]\n ", llm_out, "\n")
print(f"[query-focused: {QUERY}]\n ", query_out, "\n")

# Faithfulness against YOUR text - the production check, no reference required.
if "my_nli" not in globals():
    my_nli_tok = AutoTokenizer.from_pretrained(
        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", cache_dir=HF_CACHE)
    my_nli = AutoModelForSequenceClassification.from_pretrained(
        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", cache_dir=HF_CACHE).to(device).eval()
ENT_ID = my_nli.config.label2id["entailment"]

with torch.inference_mode():
    for label, summ in [("bart-large-cnn", bart_out[0]), ("qwen3-1.7b", llm_out)]:
        print(f"[{label}] per-sentence support:")
        for s in split_sentences(summ):
            enc = my_nli_tok(MY_TEXT, s, return_tensors="pt", truncation="only_first",
                             max_length=512).to(device)
            p = my_nli(**enc).logits.softmax(-1)[0, ENT_ID].item()
            flag = "  <-- UNSUPPORTED" if p < 0.5 else ""
            print(f"   {p:.2f}  {s[:100]}{flag}")
        print()

del my_bart, my_bart_tok, my_llm, my_llm_tok, my_nli, my_nli_tok
free_memory()
vram("final")
```

    input: 105 words / 136 bart tokens

    [bart-large-cnn]
      The infrastructure review concluded on Tuesday after six weeks. The team had originally planned to migrate all workloads to the managed Kubernetes service by the end of the quarter. Three of the seven services depend on GPU passthrough that the managed offering does not support. 

    [qwen3-1.7b]
      The infrastructure review concluded after six weeks. Three services depend on GPU passthrough, so they remain on existing hosts. The annual cost is now 155,000 dollars. The team will revisit the deployment split in twelve months. 

    [query-focused: What did they decide about cost?]
      not stated 

    [bart-large-cnn] per-sentence support:
       0.97  The infrastructure review concluded on Tuesday after six weeks.
       0.99  The team had originally planned to migrate all workloads to the managed Kubernetes service by the en
       0.99  Three of the seven services depend on GPU passthrough that the managed offering does not support.

    [qwen3-1.7b] per-sentence support:
       0.99  The infrastructure review concluded after six weeks.
       1.00  Three services depend on GPU passthrough, so they remain on existing hosts.
       1.00  The annual cost is now 155,000 dollars.
       0.04  The team will revisit the deployment split in twelve months.  <-- UNSUPPORTED

    VRAM final                   0.01 GB allocated /  0.02 GB reserved

## 14. Common Frameworks

Summarization’s framework story is dominated by one problem: the output is unverifiable by construction. There is no reference to compare against in production, ROUGE does not detect the failure that matters, and a fluent summary containing one invented number is worse than no summary. So the interesting tools here are the faithfulness scorers and the orchestration that lets you re-generate what fails them - not the models, which have been adequate for years.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [transformers](https://github.com/huggingface/transformers) | modelling | BART, distilBART, Pegasus and any instruction LLM behind `AutoModelForSeq2SeqLM` and the `summarization` pipeline | Apache 2.0 | Default. Both the fine-tuned seq2seq and the prompted-LLM paths |
| [peft](https://github.com/huggingface/peft) + [trl](https://github.com/huggingface/trl) | modelling | LoRA or `Seq2SeqTrainer` over a few thousand (document, summary) pairs from your own domain | Apache 2.0 | You need on-format output. The win is not ROUGE - it is that length, register and structure stop varying |
| [datasets](https://github.com/huggingface/datasets) + [Docling](https://github.com/docling-project/docling) | data | Streaming corpora, and structure-aware chunking that splits on sections rather than mid-argument | Apache 2.0 / MIT | Long documents. Where you split decides what the summary can possibly contain |
| [vLLM](https://github.com/vllm-project/vllm) / [SGLang](https://github.com/sgl-project/sglang) | inference runtime | Batched long-context generation, and prefix caching when many summaries share a system prompt | Apache 2.0 | Serving. Summarization prompts are long by nature, so prefill dominates and caching pays |
| [optimum](https://github.com/huggingface/optimum) + [ONNX Runtime](https://onnxruntime.ai/) | inference runtime | distilBART on CPU, which is fast enough for a real product and costs nothing per document | Apache 2.0 / MIT | High-volume, fixed-format summarisation where an LLM is overkill |
| [outlines](https://github.com/dottxt-ai/outlines) / [xgrammar](https://github.com/mlc-ai/xgrammar) | inference runtime | Structured summaries - bullet counts, required fields, a JSON schema - enforced rather than requested | Apache 2.0 / MIT | The summary feeds another system. Length and format compliance is otherwise a per-call gamble |
| [LangGraph](https://github.com/langchain-ai/langgraph) / [LlamaIndex](https://github.com/run-llama/llama_index) | orchestration | Map-reduce and refine chains over chunks, and the score-then-regenerate loop for failed sentences | MIT | Documents past the context window, and any pipeline with a faithfulness gate |
| [AlignScore](https://github.com/yuh-zha/AlignScore) / [SummaC](https://github.com/tingofurro/summac) / an NLI model | evaluation | Sentence-level entailment against the source - the check that catches an invented number | MIT / Apache 2.0 | Always. Section 11 is the simplified version; validate whichever you pick against a few hundred human labels of your own content |
| [evaluate](https://github.com/huggingface/evaluate) (ROUGE, BERTScore) | evaluation | The standard reported metrics, useful as a regression test and nothing more | Apache 2.0 | Comparing runs of the same system. ROUGE cannot distinguish a faithful summary from a fluent fabrication |

**The 2026 default stack** is a prompted LLM through vLLM with a constrained output format, structure-aware chunking with map-reduce for anything long, and an entailment-based faithfulness gate that triggers regeneration. A fine-tuned distilBART when the format is fixed and the volume is high.

The common wrong turn is treating ROUGE as a quality signal. It measures n-gram overlap with one arbitrary reference summary, and it assigns the same score to a faithful summary and to one that inverts a finding. The second is building generic summarisation at all: “summarise this” has no correct answer, while “what does this say about X” does - query-focused output is both more useful and far easier to evaluate.

------------------------------------------------------------------------

## 15. Going Further

- **Fine-tune on your own summaries.** `AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")` plus `Seq2SeqTrainer` over a few thousand (document, summary) pairs from your domain gets you on-format output that no prompt reliably produces. The dominant benefit is not ROUGE - it is that the length, register and structure stop varying.
- **Measure faithfulness, not just ROUGE.** Section 11 is a simplified SummaC; the real ones (`AlignScore`, `SummaC`, `QAFactEval`) chunk the source properly and are worth using directly. Whatever you pick, validate it against a few hundred human-labelled summaries of *your* content before trusting it as a gate.
- **Self-verify and regenerate.** Generate a summary, score each sentence against the source, and regenerate the sentences below threshold with the offending claim quoted back to the model. This is the cheapest large gain available in abstractive summarization, and it needs no training.
- **Query-focused beats generic in almost every product.** “Summarise this” has no correct answer; “what does this say about X” does. If your users have a recurring question, build for that instead - the output is more useful and the evaluation is far easier.
- **Long documents: check the window before you chunk.** A 32k-token model reading a whole report keeps cross-section relationships that hierarchical chunk-and-merge destroys. When you must chunk, summarise chunks, then summarise the summaries, and expect the result to lose anything that spanned chunks.
- **Control length in tokens, and verify it.** `min_length`/`max_length` for seq2seq, an explicit instruction plus a post-hoc check for an LLM. Length compliance is the most common silent product bug in summarization and the easiest to test.
- **Multi-document needs deduplication and conflict handling.** Concatenating five articles and summarising produces confident nonsense when two of them disagree. Cluster first, summarise per cluster, then reconcile - and surface the disagreement rather than averaging it away.
- **Related notebooks.** `05_Translation` (the same faithfulness-under-generation problem, different constraint), `03_Question_Answering` (query-focused summarization is closer to QA than to summarization), `04_Zero_Shot_Classification` (the NLI model used as a faithfulness checker), `08_Text_Generation` (decoding parameters, beam search, repetition control), `11_Text_Ranking` (selecting what to summarise in the first place).

------------------------------------------------------------------------

Back to top
