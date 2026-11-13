# Table Question Answering

Answering natural-language questions against structured tables: how table encoders, table-to-text seq2seq models and text-to-SQL LLMs split the field in mid-2026, how denotation accuracy is measured, and runnable code that puts three approaches on the same WikiTableQuestions sample.

Author

Benedict Thekkel

## 1. What is Table Question Answering?

Table QA answers a natural-language question using a **table** as the knowledge source, rather than a passage of prose. “Which driver won the most races in 2011?” over a results table is a table QA problem; the same question over a Wikipedia article is extractive QA.

**Input.** A question plus a table: a header row and N data rows, all cells nominally strings. Real tables carry types (numbers, dates, currencies) that are not marked up, which is most of the difficulty.

**Output.** One of three shapes, and the shape decides the architecture:

| Output | Example | Approach that fits |
|----|----|----|
| Cell selection | “Rafael Nadal” (one cell) | table encoder with a per-cell head (TAPAS) |
| Aggregation over cells | “3” from `COUNT`, “17.5” from `AVG` | cell selection + an aggregation-operator head |
| Free-form string | “Nadal, by two titles” | seq2seq over a linearised table (TAPEX), or an LLM |
| Executable program | `SELECT driver FROM t ORDER BY wins DESC LIMIT 1` | text-to-SQL, then run the query |

**The central problem is that tables are not sequences.** A transformer reads a flat token stream, so a table has to be linearised, and doing that naively destroys the row/column structure that the question depends on. Every model below is a different answer to “how do you tell the transformer what is a row and what is a column”.

**The second problem is size.** A 512-token encoder holds roughly a 10x10 table. Anything bigger has to be truncated (drop rows and hope), retrieved over (select rows first), or handed to a text-to-SQL model that never reads the data at all - only the schema.

**Neighbouring tasks:**

| Task | How it differs | Notebook |
|----|----|----|
| Question answering | Source is unstructured prose | `03_Question_Answering` |
| Document question answering | Source is a page image with layout | `Multimodal/05_Document_Question_Answering` |
| Text generation | No structured source, open-ended output | `08_Text_Generation` |
| Tabular classification | Predicts a label per row, no language | `Tabular/00_Tabular_Classification` |
| Visual document retrieval | Finds the right table/page first | `Multimodal/07_Visual_Document_Retrieval` |

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Natural-language BI (“ask your data”) | Analytics (Databricks Genie, Snowflake Cortex Analyst, Power BI Copilot) | Question + warehouse schema -\> SQL -\> result set | Correctness on joins and filters; a wrong number that looks right is worse than an error |
| Spreadsheet assistants | Office software (Excel Copilot, Sheets) | Question + selected range -\> value or formula | Latency inside a keystroke loop; user sees the table, so hallucination is instantly visible |
| Financial report QA | Finance, audit | 10-K/10-Q tables -\> figures, ratios | Numeric exactness and auditability; must cite the source cell |
| Clinical and lab tables | Healthcare | Lab result tables -\> “was potassium ever above 5.5?” | Aggregation over time; privacy forces on-prem models |
| Product catalogue search | E-commerce | Spec tables -\> “laptops under \$1200 with 32 GB RAM” | Throughput; the constraint is really a filter, not a question |
| Sports and reference lookup | Consumer assistants | Wiki tables -\> a cell value | Coverage of messy, unnormalised web tables |
| Enterprise data governance | Any regulated org | Question -\> SQL with row-level security applied | The generated query must never read what the user cannot see |

What the leaderboard number hides:

- **Text-to-SQL benchmarks and production BI are not the same task.** Spider and BIRD give the model a clean, documented schema. A real warehouse has 4,000 tables, columns named `dt_flg_2`, three tables that all look like “customers”, and tribal knowledge about which one is current. Schema linking, not SQL syntax, is where accuracy is lost.
- **A plausible wrong answer is the failure mode that matters.** The model that returns `1,284,331` when the truth is `1,284,133` produces a number nobody double-checks. Systems that show the generated SQL and the source rows are safer than systems that only show the answer, even at equal accuracy.
- **Aggregation is where small models break.** Selecting a cell is easy; `AVG` over a filtered subset, or a comparison across two groups, is where TAPAS-class models fall off and program-generating approaches pull ahead.
- **Execution changes the safety model.** Text-to-SQL means running generated code against a live database. Read-only credentials, a statement timeout, `LIMIT` injection and a query allow-list are not optional extras; they are the feature.

------------------------------------------------------------------------

## 3. How Modern Table QA Works

1.  **Semantic parsing to logical forms (2013-2018).** WikiTableQuestions arrived with parsers that mapped a question to a lambda-DCS or SQL-like program and executed it. Correct in principle, but trained from denotations (answers) only, so learning was a brutal search over programs with spurious ones that got the right answer for the wrong reason.
2.  **Table-aware encoders (2020).** **TAPAS** extended BERT with row, column and rank embeddings, pretrained on millions of Wikipedia tables, and put two heads on top: which cells to select, and which aggregation to apply (`NONE`, `COUNT`, `SUM`, `AVERAGE`). No program to search; end-to-end from denotations. TaBERT and TURL explored the same idea for joint text-table representations.
3.  **Table linearisation with seq2seq (2021-2022).** **TAPEX** flipped the pretraining objective: take BART and pretrain it to be a *SQL executor* - feed it a flattened table plus a SQL query and make it output the result. That teaches table reasoning without any table-specific architecture, and it beat TAPAS on WTQ. Output is free-form text, so it handles answers no cell contains.
4.  **Text-to-SQL with code LLMs (2023-2026).** The dominant approach now. The model never reads the data, only the schema, so table size is irrelevant; the database does the arithmetic, so aggregation is exact rather than predicted. **Spider** then **BIRD** (harder, dirty real-world schemas, execution-time-aware) became the benchmarks. Frontier LLMs pushed BIRD execution accuracy from ~40% (2023) past ~75% (2025-2026), with the gains coming from schema linking, self-correction on execution errors, and majority voting over sampled queries rather than from bigger models alone.
5.  **Agentic table analysis (2024-2026).** Rather than emit one query, the model writes and runs code (pandas, SQL) in a loop, inspects intermediate results, and repairs itself. This is what “ask your data” products actually ship: retrieval over the schema, a plan, generated code, execution, and a check pass. It is slower and far more accurate on multi-step questions.

**Where it stands (mid-2026).** For a large or live database, text-to-SQL with a code-capable LLM wins outright - it is the only approach whose accuracy does not degrade with row count, and its arithmetic is exact. For a small self-contained table (a spreadsheet range, a 10-row wiki table) where you cannot run a database and latency matters, TAPAS and TAPEX still do the job in 150-400M params. The encoder approaches are not obsolete; they are the cheap end of a spectrum whose expensive end got very good.

------------------------------------------------------------------------

## 4. Evaluation Metrics

**Denotation accuracy** is the standard for WikiTableQuestions: does the predicted answer set equal the gold answer set, after normalisation? It sidesteps the fact that many different programs produce the right answer.

$$\text{DenotationAcc} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\big[\,\text{norm}(\hat{y}_i) = \text{norm}(y_i)\,\big]$$

**Execution accuracy (EX)** is the text-to-SQL equivalent: run the predicted SQL and the gold SQL, compare the result sets. This is the Spider/BIRD headline metric. **Exact-match on the SQL string** is the weaker alternative and is largely abandoned - two correct queries can differ in join order, aliasing or whitespace.

**Valid efficiency score (VES)**, introduced with BIRD, weights execution accuracy by how fast the generated query runs relative to the gold one, because a correct query that takes 40 seconds is not production-usable.

**Normalisation is the whole game, and it is where the honest numbers live.** The predictions `"3"`, `"3.0"`, `" 3 "` and `"three"` are the same answer; `"$1,200"` and `"1200"` usually are too. A scorer that compares raw strings under-reports every model, and one that normalises too aggressively (stripping units, rounding floats) over-reports them. The official WTQ evaluator normalises unicode, case, punctuation and numbers, and compares **sets** because answers can be multi-cell.

**Pitfalls:**

- **Multi-cell answers are sets, not strings.** `"Nadal, Federer"` vs `"Federer, Nadal"` must score as correct; joining cells with a comma and comparing strings does not.
- **Aggregation answers are floats.** Compare numerically with a tolerance, not textually - `12.333333` vs `12.33` is a formatting difference, not a wrong answer.
- **Spurious correctness inflates small samples.** On a 3-row table, `COUNT` and “select the only matching cell” often coincide. A model can score well while reasoning wrongly, which is why WTQ is evaluated over thousands of tables.

The cell below implements the normaliser and the set comparison - the arithmetic is trivial, the normalisation is the part worth reading.

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
import unicodedata


def normalize_answer(s):
    "Lowercase, strip punctuation/articles/units, and canonicalise numbers (WTQ-style)."
    s = unicodedata.normalize("NFKD", str(s)).strip().lower()
    s = s.replace(",", "") if re.fullmatch(r"[\d,]+(\.\d+)?", s.replace(" ", "")) else s
    s = re.sub(r"^(a|an|the)\s+", "", s)
    s = re.sub(r"[^\w\s.\-]", "", s)  # drops $, %, quotes, brackets
    s = re.sub(r"\s+", " ", s).strip()
    try:  # canonicalise numbers: "3.0" == "3" == "3 "
        f = float(s)
        return str(int(f)) if f.is_integer() else f"{f:.4f}".rstrip("0")
    except ValueError:
        return s


def answers_match(pred, gold, tol=1e-4):
    "Set comparison over normalised answers, with a numeric tolerance for aggregates."
    p = {normalize_answer(x) for x in (pred if isinstance(pred, (list, tuple)) else [pred])}
    g = {normalize_answer(x) for x in (gold if isinstance(gold, (list, tuple)) else [gold])}
    if p == g:
        return True
    if len(p) == len(g) == 1:  # single numeric answer: compare with tolerance
        try:
            return abs(float(next(iter(p))) - float(next(iter(g)))) <= tol
        except ValueError:
            return False
    return False


def denotation_accuracy(preds, golds):
    "Fraction of questions whose predicted answer set matches the gold set."
    return sum(answers_match(p, g) for p, g in zip(preds, golds)) / len(golds)


# Toy example: every pair below is the *same* answer wearing different clothes.
cases = [
    ("3.0", ["3"]),
    ("$1,200", ["1200"]),
    ("The Netherlands", ["Netherlands"]),
    (["Federer", "Nadal"], ["Nadal", "Federer"]),
    ("12.333333", ["12.3333"]),
    ("4", ["5"]),  # genuinely wrong
]
show_table([{"match": answers_match(pred, gold), "prediction": str(pred),
             "gold": str(gold), "normalised prediction": str(normalize_answer(pred)
                                                             if not isinstance(pred, list) else
                                                             sorted(normalize_answer(x) for x in pred))}
            for pred, gold in cases],
           title="Every pair but the last is the same answer wearing different clothes",
           caption=f"denotation accuracy "
                   f"{denotation_accuracy([c[0] for c in cases], [c[1] for c in cases]):.3f} - "
                   "raw string comparison would score 0.167, so normalisation is most of the metric")
```

```
    Every pair but the last is the same answer wearing different clothes     
                                                                             
 match   prediction             gold                   normalised prediction 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 True    3.0                    ['3']                  3                     
 True    $1,200                 ['1200']               1200                  
 True    The Netherlands        ['Netherlands']        netherlands           
 True    ['Federer', 'Nadal']   ['Nadal', 'Federer']   ['federer', 'nadal']  
 True    12.333333              ['12.3333']            12.3333               
 False   4                      ['5']                  4                     
                                                                             
   denotation accuracy 0.833 - raw string comparison would score 0.167, so   
                     normalisation is most of the metric                     
```

## 5. Datasets

| Dataset | Contents | Size | Scope | License | Typical use |
|----|----|----|----|----|----|
| [WikiTableQuestions](https://huggingface.co/datasets/stanfordnlp/wikitablequestions) | Wikipedia tables + crowd questions, free-form answers | 22k questions / 2.1k tables | en | CC BY-SA 4.0 | The classic table QA benchmark; used below via a parquet mirror |
| [WikiSQL](https://huggingface.co/datasets/Salesforce/wikisql) | Simple single-table SQL over wiki tables | 80k | en | BSD-3 | Easy text-to-SQL; largely saturated |
| [Spider](https://huggingface.co/datasets/xlangai/spider) | Cross-domain, multi-table SQL with joins | 10k / 200 DBs | en | CC BY-SA 4.0 | The standard text-to-SQL benchmark |
| [BIRD](https://huggingface.co/datasets/birdsql/bird_train) | Large dirty real-world DBs, external knowledge, efficiency | 12.7k / 95 DBs | en | CC BY-SA 4.0 | The hard, current text-to-SQL benchmark |
| [TabFact](https://huggingface.co/datasets/wenhu/tab_fact) | Table + statement -\> entailed / refuted | 118k | en | CC BY 4.0 | Table fact verification, not QA |
| [FeTaQA](https://huggingface.co/datasets/DongfuJiang/FeTaQA) | Free-form *sentence* answers over tables | 10k | en | CC BY-SA 4.0 | Generative table QA; ROUGE/BLEU scored |
| [HybridQA](https://huggingface.co/datasets/wenhu/hybrid_qa) | Questions needing a table **and** linked text | 70k | en | CC BY 4.0 | Hybrid table+text reasoning |
| [SQA](https://huggingface.co/datasets/microsoft/wiki_table_questions) | Sequential, conversational follow-ups over one table | 17k | en | MSR-LA | Multi-turn table QA |
| [TAT-QA](https://huggingface.co/datasets/next-tat/tat-qa) | Financial reports: tables + paragraphs, numeric reasoning | 16k | en | CC BY 4.0 | Finance-domain arithmetic |

This notebook evaluates on **WikiTableQuestions**, sampled down to tables small enough for a 512-token encoder. WTQ is a genuinely hard benchmark - it deliberately includes comparison, aggregation and superlative questions, and TAPAS-class models sit around 45-50% denotation accuracy on the full set, not the 90% that “just look up a cell” would suggest.

**Loading it needs a mirror, and the reason generalises.** The canonical repo, [`stanfordnlp/wikitablequestions`](https://huggingface.co/datasets/stanfordnlp/wikitablequestions), is a legacy loading **script**, and `datasets` 4.x refuses to execute scripts at all:

    RuntimeError: Dataset scripts are no longer supported, but found wikitablequestions.py

There are two fixes for a script-based dataset and it is worth knowing both, because the first one is cheaper and does not always exist:

1.  **Read the Hub’s auto-converted parquet branch.** Most script datasets have one at `refs/convert/parquet`, and you can point the generic `parquet` loader straight at it - this is what `01_Token_Classification` does for CoNLL-2003.
2.  **Find a parquet mirror.** WTQ has no convert branch, so that is the route here: [`lighteval/wikitablequestions`](https://huggingface.co/datasets/lighteval/wikitablequestions) carries the identical schema (`question`, `answers`, `table` as `{header, rows}`) and holds 18.5k of WTQ’s ~22k questions in a single `test` split - so it is **not** the official 4,344-question test split, and a number measured on it is not comparable to a published WTQ result. That does not matter for the 40-question smoke test here, and it would matter enormously for a leaderboard claim.

Check the mirror’s schema before trusting it. A mirror that silently drops the table structure and keeps only a markdown rendering is common, and it breaks TAPAS (which needs a real `DataFrame`) while appearing to work for the text-to-SQL path.

Downloads land in `DL_tasks/datasets/` via `cache_dir` (gitignored).

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

Two leaderboards matter and they measure different things: [Spider](https://yale-lily.github.io/spider) / [BIRD](https://bird-bench.github.io/) for text-to-SQL execution accuracy, and the [WikiTableQuestions](https://paperswithcode.com/sota/semantic-parsing-on-wikitablequestions) results for end-to-end table QA.

| Model | Params | License | Approach | Answer shape | Best for |
|----|----|----|----|----|----|
| [tapas-base-finetuned-wtq](https://huggingface.co/google/tapas-base-finetuned-wtq) | 110M | Apache 2.0 | table encoder + cell/aggregation heads | cells + operator | cheap cell lookup; used below |
| [tapas-large-finetuned-wtq](https://huggingface.co/google/tapas-large-finetuned-wtq) | 340M | Apache 2.0 | same, larger | cells + operator | best TAPAS accuracy |
| [tapex-base-finetuned-wtq](https://huggingface.co/microsoft/tapex-base-finetuned-wtq) | 140M | MIT | BART pretrained as a SQL executor | free-form text | small free-form answers; used below |
| [tapex-large-finetuned-wtq](https://huggingface.co/microsoft/tapex-large-finetuned-wtq) | 400M | MIT | same, larger | free-form text | strongest small table-QA model |
| [omnitab-large](https://huggingface.co/neulab/omnitab-large-finetuned-wtq) | 400M | MIT | TAPEX + synthetic NL pretraining | free-form text | few-shot table QA |
| [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct) | 1.5B | Apache 2.0 | text-to-SQL / code generation | SQL to execute | small local text-to-SQL; used below |
| [Qwen2.5-Coder-7B / 32B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) | 7-32B | Apache 2.0 | text-to-SQL | SQL | strong open text-to-SQL (needs more VRAM than this box) |
| [XiYanSQL-QwenCoder-32B](https://huggingface.co/XGenerationLab/XiYanSQL-QwenCoder-32B-2412) | 32B | Apache 2.0 | SQL-specialised fine-tune | SQL | near-frontier open BIRD scores (server-class) |
| Frontier LLMs (Claude, GPT, Gemini) | \- | proprietary | agentic text-to-SQL + self-repair | SQL | production “ask your data”; top of BIRD |

**How to choose.** Table fits in 512 tokens, no database, latency matters: TAPEX-base or TAPAS-base, 140-340M params, tens of milliseconds. Table lives in a database, or is bigger than a few hundred rows: text-to-SQL, always - the accuracy gap widens with every row. Multi-step analytical questions (“compare Q3 growth across regions”): an agentic loop that writes code, executes it and checks itself, not a single-shot model.

**Note on size.** The 32B SQL specialists are the accuracy leaders and do **not** fit this box (32B in fp16 is ~64 GB). The runnable text-to-SQL cell below uses Qwen2.5-Coder-1.5B-Instruct (~3.1 GB download, ~3.1 GB VRAM in fp16), which is enough to demonstrate the approach on a single-table schema.

------------------------------------------------------------------------

## 7. Setup

Everything loads through Hugging Face `transformers` - no vendor packages. Package roles:

- `transformers` + `torch` - TAPAS, TAPEX and the code LLM
- `accelerate` - `device_map` placement
- `datasets` - the WikiTableQuestions sample, read from a parquet mirror (see section 5)
- `pandas` - tables are `DataFrame`s; also the benchmark table
- `sqlite3` (stdlib) - executes the generated SQL
- `pyecharts` - the benchmark chart
- `rich` - the result tables. It renders to HTML inside Jupyter, so the tables survive into the published docs; `show_table` / `show_kv` / `rule` are defined in the first code cell of section 4.

Three `transformers` details that matter here:

- **TAPAS wants every cell as a `str`.** `pipeline("table-question-answering")` raises if the `DataFrame` holds ints or floats. Cast with `df.astype(str)` - the model recovers numeric meaning from its rank embeddings, not from the dtype.
- **TAPAS returns cells plus an aggregator**, e.g. `{"answer": "SUM > 12, 5", "cells": ["12", "5"], "aggregator": "SUM"}`. It does **not** do the arithmetic; the `answer` string is a prefix plus the raw cells. Applying the operator yourself is part of using the model, and section 8 does it.
- **TAPEX is a BART seq2seq, and its tokenizer class is gone.** `transformers` 5.x removed `TapexTokenizer`, so `AutoTokenizer.from_pretrained("microsoft/tapex-base-finetuned-wtq")` raises `Couldn't instantiate the backend tokenizer from one of: ...`. This is less dramatic than it looks: the checkpoint is a plain BART and the repo ships BART’s BPE files, so `BartTokenizerFast` loads it, and the only thing the removed class contributed was Python-level table linearisation - which section 9 now does in six lines. Two details the old class handled silently and you now handle yourself: **lowercasing** (the repo sets `do_lower_case=True`, which a BPE tokenizer ignores) and **row-level truncation** for oversized tables.

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
import pandas as pd
from datasets import load_dataset

# WikiTableQuestions: wiki tables + crowd questions with free-form answers.
#
# The canonical repo (stanfordnlp/wikitablequestions) is a legacy loading *script*, and
# datasets 4.x refuses to run scripts - "RuntimeError: Dataset scripts are no longer
# supported, but found wikitablequestions.py". It has no refs/convert/parquet branch to
# read directly either (the trick 01_Token_Classification uses for CoNLL-2003), so this
# reads a parquet mirror with the same schema: question, answers, table {header, rows}.
wtq = load_dataset("lighteval/wikitablequestions", split="test", cache_dir=HF_CACHE)

MAX_CELLS = 120  # a 512-token encoder holds roughly this much table; bigger gets truncated
N = 40           # questions to evaluate - a smoke test, not a leaderboard run


def to_frame(table):
    "WTQ tables are {'header': [...], 'rows': [[...]]}. TAPAS needs every cell as str."
    return pd.DataFrame(table["rows"], columns=table["header"]).astype(str)


examples = []
for r in wtq:
    df = to_frame(r["table"])
    if df.size <= MAX_CELLS and len(df.columns) >= 2:
        examples.append({"question": r["question"], "answers": r["answers"], "df": df})
    if len(examples) >= N:
        break

questions = [e["question"] for e in examples]
gold = [e["answers"] for e in examples]
tables = [e["df"] for e in examples]

print(wtq)
print(f"\n{len(examples)} questions kept (tables of <= {MAX_CELLS} cells)")
print(f"median table: {int(pd.Series([t.size for t in tables]).median())} cells\n")
print("example question:", questions[0])
print("gold answer:     ", gold[0])
tables[0].head()
```

    Dataset({
        features: ['id', 'question', 'answers', 'table', 'table_md'],
        num_rows: 18486
    })

    40 questions kept (tables of <= 120 cells)
    median table: 73 cells

    example question: how many total points did the bombers score against the bc lions?
    gold answer:      ['58']

|     | Week | Date    | Opponent                     | Score | Result | Record |
|-----|------|---------|------------------------------|-------|--------|--------|
| 0   | 1    | July 9  | vs. Saskatchewan Roughriders | 31–21 | Win    | 1–0    |
| 1   | 2    | July 16 | at Montreal Concordes        | 36–0  | Win    | 2–0    |
| 2   | 3    | July 24 | vs. Hamilton Tiger-Cats      | 36–25 | Loss   | 2–1    |
| 3   | 4    | Aug 1   | at Edmonton Eskimos          | 32–26 | Win    | 3–1    |
| 4   | 5    | Aug 8   | vs. BC Lions                 | 29–16 | Win    | 4–1    |

## 8. Table encoder: TAPAS

TAPAS is BERT with extra position embeddings that tell it the table structure: which row a token is in, which column, and the numeric **rank** of a cell within its column (so “highest” and “before 2005” are learnable). It was pretrained on millions of Wikipedia tables with a masked-LM objective over the whole table, then fine-tuned on WTQ with weak supervision - only the answers, never the programs.

Two heads sit on top. The **cell-selection head** scores every cell independently. The **aggregation head** predicts one of `NONE`, `COUNT`, `SUM`, `AVERAGE`. Crucially, **TAPAS does not compute the aggregate** - it tells you which operator and which cells, and you do the arithmetic. The pipeline’s `answer` field is a string like `"SUM > 12, 5"`, which is a prefix and the raw cells, not `17`. Treating that string as the answer is the standard beginner bug and it scores ~0 on aggregation questions.

Pick TAPAS when the table is small, you want the *provenance* (it names the cells it used), and you need an answer in ~20 ms on CPU.

------------------------------------------------------------------------

``` python
from transformers import pipeline

tapas = pipeline(
    "table-question-answering",
    model="google/tapas-base-finetuned-wtq",
    device=device,
    model_kwargs={"cache_dir": HF_CACHE},
)
vram("tapas loaded")


def apply_aggregator(out):
    "TAPAS names cells and an operator but does not compute it. Do the arithmetic here."
    cells, agg = out["cells"], out.get("aggregator", "NONE")
    if agg == "NONE" or not cells:
        return cells  # a set of cell values
    nums = []
    for c in cells:
        try:
            nums.append(float(str(c).replace(",", "").replace("$", "")))
        except ValueError:
            pass
    if agg == "COUNT":
        return [str(len(cells))]
    if not nums:
        return cells
    return [str(sum(nums))] if agg == "SUM" else [str(sum(nums) / len(nums))]


# One worked example, showing what the model actually returns.
out = tapas(table=tables[0], query=questions[0])
show_kv({"question": questions[0],
         **{k: str(out[k]) for k in ("answer", "cells", "aggregator") if k in out},
         "resolved (aggregator applied)": str(apply_aggregator(out)),
         "gold": str(gold[0])},
        title="What TAPAS actually returns")

t0 = time.perf_counter()
tapas_preds = [apply_aggregator(tapas(table=t, query=q)) for t, q in zip(tables, questions)]
tapas_secs = time.perf_counter() - t0

show_kv({"questions": len(questions), "seconds": round(tapas_secs, 1),
         "questions / second": round(len(questions) / tapas_secs, 1),
         "denotation accuracy": round(denotation_accuracy(tapas_preds, gold), 3)},
        title="google/tapas-base-finetuned-wtq")

del tapas
free_memory()
vram("after tapas")
```

    VRAM tapas loaded            0.44 GB allocated /  0.50 GB reserved

```
                                    What TAPAS actually returns                                    
                                                                                                   
 question                        how many total points did the bombers score against the bc lions? 
 answer                                                                                SUM > 5, 12 
 cells                                                                                 ['5', '12'] 
 aggregator                                                                                    SUM 
 resolved (aggregator applied)                                                            ['17.0'] 
 gold                                                                                       ['58'] 
                                                                                                   
```

    [transformers] You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset

```
                google/tapas-base-finetuned-wtq                 
                                                                
 questions                                                   40 
 seconds                                                 1.1000 
 questions / second                                       36.60 
 denotation accuracy                                     0.8750 
                                                                
```

    VRAM after tapas             0.01 GB allocated /  0.02 GB reserved

## 9. Table-to-text seq2seq: TAPEX

TAPEX takes the opposite bet: no table-specific architecture at all. It is plain BART, pretrained on a synthetic corpus of (flattened table, SQL query) -\> execution result. In other words it was taught to **be a SQL engine** before it ever saw a natural-language question. The table gets linearised by the tokenizer into `col : a | b | c row 1 : ... row 2 : ...` and the decoder emits the answer as text.

That single change fixes TAPAS’s weakest point: because the answer is generated, aggregation comes out as a number rather than as an operator you have to apply, and answers that appear in no cell (“2 more than Spain”) are expressible. It beat TAPAS on WTQ and remains the strongest sub-500M option.

The cost is provenance. TAPEX gives you a string with no indication of which cells it came from, so you cannot show a user the evidence, and a hallucinated number is indistinguishable from a computed one.

**The cell below builds the linearised string by hand**, because `transformers` 5.x removed `TapexTokenizer` (see Setup). That is a happy accident for this section: the format `col : a | b | c row 1 : ... row 2 : ...` is the entire table-specific part of TAPEX, and it is now visible rather than hidden inside a tokenizer call.

------------------------------------------------------------------------

``` python
from transformers import AutoModelForSeq2SeqLM, BartTokenizerFast

tapex_id = "microsoft/tapex-base-finetuned-wtq"

# TAPEX shipped a `TapexTokenizer`, and transformers 5.x removed that class, so
# AutoTokenizer fails here with "Couldn't instantiate the backend tokenizer from one
# of: ...". Nothing is actually missing: TAPEX *is* a plain BART, the repo carries
# BART's BPE files (vocab.json + merges.txt), and the only thing TapexTokenizer added
# was Python-level table linearisation. Doing that by hand below turns the section's
# claim into code - the linearisation is the whole table-specific part of TAPEX.
tapex_tok = BartTokenizerFast.from_pretrained(tapex_id, cache_dir=HF_CACHE)
tapex = AutoModelForSeq2SeqLM.from_pretrained(tapex_id, cache_dir=HF_CACHE).to(device).eval()
vram("tapex loaded")


def linearize(df):
    "Flatten a table into the string TAPEX was pretrained to read."
    out = "col : " + " | ".join(str(c) for c in df.columns)
    for i, row in enumerate(df.astype(str).values.tolist(), start=1):
        out += f" row {i} : " + " | ".join(row)
    return out


def tapex_input(df, query):
    "Question + linearised table, lowercased - the repo sets do_lower_case=True, and a\n\n    BPE tokenizer will not apply that for you.\n    "
    return (query + " " + linearize(df)).lower()


# Look at what the model actually reads.
print("linearised table (first 300 chars):")
print(" ", tapex_input(tables[0].head(3), questions[0])[:300], "...\n")


@torch.inference_mode()
def tapex_answer(df, query):
    """Linearise the table, generate the answer as free text, strip the leading space.

    TapexTokenizer used to drop *rows* when a table overflowed the window; truncating
    tokens is cruder, but the tables here are filtered to MAX_CELLS so it rarely fires.
    `max_length` (not `max_new_tokens`) overrides the repo's generation_config cleanly.
    """
    enc = tapex_tok(tapex_input(df, query), return_tensors="pt",
                    truncation=True, max_length=1024).to(device)
    out = tapex.generate(**enc, max_length=32, num_beams=1)
    return [tapex_tok.decode(out[0], skip_special_tokens=True).strip()]


show_kv({"question": questions[0],
         "tapex": str(tapex_answer(tables[0], questions[0])),
         "gold": str(gold[0])},
        title="TAPEX generates the answer as free text - no cells, no operator")

t0 = time.perf_counter()
tapex_preds = [tapex_answer(t, q) for t, q in zip(tables, questions)]
tapex_secs = time.perf_counter() - t0

show_kv({"questions": len(questions), "seconds": round(tapex_secs, 1),
         "questions / second": round(len(questions) / tapex_secs, 1),
         "denotation accuracy": round(denotation_accuracy(tapex_preds, gold), 3)},
        title="microsoft/tapex-base-finetuned-wtq")

del tapex, tapex_tok
free_memory()
vram("after tapex")
```

    VRAM tapex loaded            0.57 GB allocated /  0.62 GB reserved
    linearised table (first 300 chars):
      how many total points did the bombers score against the bc lions? col : week | date | opponent | score | result | record row 1 : 1 | july 9 | vs. saskatchewan roughriders | 31–21 | win | 1–0 row 2 : 2 | july 16 | at montreal concordes | 36–0 | win | 2–0 row 3 : 3 | july 24 | vs. hamilton tiger-cats  ...

```
       TAPEX generates the answer as free text - no cells, no operator        
                                                                              
 question   how many total points did the bombers score against the bc lions? 
 tapex                                                                 ['58'] 
 gold                                                                  ['58'] 
                                                                              
```

```
               microsoft/tapex-base-finetuned-wtq               
                                                                
 questions                                                   40 
 seconds                                                 0.9000 
 questions / second                                       44.80 
 denotation accuracy                                     0.9000 
                                                                
```

    VRAM after tapex             0.01 GB allocated /  0.02 GB reserved

## 10. Text-to-SQL: Qwen2.5-Coder-1.5B + SQLite

The approach that scales. The model never sees the table contents - only the **schema** (column names, types, and a couple of sample rows for disambiguation). It emits SQL, SQLite executes it, and the database returns the answer. Three consequences follow immediately:

- **Table size stops mattering.** A million-row table has the same prompt cost as a ten-row one.
- **Arithmetic is exact.** `AVG` is computed by the database, not predicted by a language model. This is the single biggest accuracy win over TAPAS/TAPEX.
- **Errors become visible.** A malformed query raises instead of returning a confident wrong string, which gives you something to retry on. The `try/except` below is the seed of the self-repair loops that production systems run.

The safety story is the flip side: you are executing generated code. The cell below is safe because SQLite is in-memory and thrown away, but a real deployment needs a read-only connection, a statement timeout, a forced `LIMIT`, and a rejection of anything that is not a single `SELECT`. The `is_safe_select` check here is the minimum version of that.

WTQ is an unfair test for this approach - its tables are tiny and its columns are untyped strings scraped from Wikipedia, so numeric comparisons need casting the model has to guess at. Expect it to look merely competitive here and to pull far ahead on anything with real types and real size.

------------------------------------------------------------------------

``` python
import re
import sqlite3

from transformers import AutoModelForCausalLM, AutoTokenizer

sql_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # ~3.1 GB download, ~3.1 GB VRAM in fp16
sql_tok = AutoTokenizer.from_pretrained(sql_id, cache_dir=HF_CACHE)
sql_llm = AutoModelForCausalLM.from_pretrained(
    sql_id, dtype=dtype, device_map=device, cache_dir=HF_CACHE
).eval()
vram("qwen2.5-coder-1.5b loaded")

PROMPT = (
    "You are a SQLite expert. Given the table schema and a question, write ONE SQLite "
    "SELECT query that answers it. Return only the SQL, no explanation, no markdown.\n\n"
    "The table is named `t`. All columns are TEXT, so CAST to REAL for numeric "
    "comparisons or aggregation.\n\nSchema:\n{schema}\n\nSample rows:\n{sample}\n\n"
    "Question: {question}\nSQL:"
)


def safe_columns(df):
    "SQLite-safe column names, kept in order, deduplicated."
    seen, cols = {}, []
    for c in df.columns:
        name = re.sub(r"\W+", "_", str(c)).strip("_").lower() or "col"
        seen[name] = seen.get(name, 0) + 1
        cols.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    return cols


def is_safe_select(sql):
    "Minimum guard for executing generated SQL: one statement, read-only."
    s = sql.strip().rstrip(";").lower()
    banned = ("insert", "update", "delete", "drop", "alter", "attach", "pragma", "create")
    return s.startswith("select") and ";" not in s and not any(b in s for b in banned)


@torch.inference_mode()
def text_to_sql_answer(df, question, max_new_tokens=96):
    "Generate SQL from the schema alone, then let SQLite compute the answer."
    frame = df.copy()
    frame.columns = safe_columns(frame)
    schema = "\n".join(f"  {c} TEXT" for c in frame.columns)
    sample = frame.head(2).to_string(index=False)

    chat = sql_tok.apply_chat_template(
        [{"role": "user", "content": PROMPT.format(schema=schema, sample=sample, question=question)}],
        tokenize=False, add_generation_prompt=True,
    )
    enc = sql_tok(chat, return_tensors="pt").to(sql_llm.device)
    out = sql_llm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=sql_tok.eos_token_id)
    sql = sql_tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    sql = re.sub(r"```(?:sql)?|```", "", sql).strip().split("\n\n")[0].strip()

    if not is_safe_select(sql):
        return [""], sql
    con = sqlite3.connect(":memory:")
    try:
        frame.to_sql("t", con, index=False)
        rows = con.execute(sql).fetchall()
        return [str(v) for row in rows for v in row][:10] or [""], sql
    except Exception as exc:  # a broken query is a *visible* failure - retry material
        return [""], f"{sql}   -- ERROR: {type(exc).__name__}"
    finally:
        con.close()


ans, sql = text_to_sql_answer(tables[0], questions[0])
show_kv({"question": questions[0], "generated SQL": sql,
         "answer (from SQLite)": str(ans), "gold": str(gold[0])},
        title="Text-to-SQL: the model never sees the data, only the schema")

t0 = time.perf_counter()
sql_out = [text_to_sql_answer(t, q) for t, q in zip(tables, questions)]
sql_secs = time.perf_counter() - t0
sql_preds = [a for a, _ in sql_out]
n_broken = sum(1 for _, s in sql_out if "ERROR" in s or not is_safe_select(s.split("   --")[0]))

show_kv({"questions": len(questions), "seconds": round(sql_secs, 1),
         "questions / second": round(len(questions) / sql_secs, 1),
         "denotation accuracy": round(denotation_accuracy(sql_preds, gold), 3),
         "queries that failed to execute": f"{n_broken}/{len(questions)}"},
        title="Qwen2.5-Coder-1.5B-Instruct + SQLite")
print("A failed query is a *visible* failure - it is what a self-repair loop retries on.")

del sql_llm, sql_tok
free_memory()
vram("after text-to-sql")
```

    VRAM qwen2.5-coder-1.5b loaded  3.10 GB allocated /  3.16 GB reserved

```
                Text-to-SQL: the model never sees the data, only the schema                
                                                                                           
 question                how many total points did the bombers score against the bc lions? 
 generated SQL          SELECT SUM(CAST(score AS REAL)) FROM t WHERE opponent = 'BC Lions' 
 answer (from SQLite)                                                             ['None'] 
 gold                                                                               ['58'] 
                                                                                           
```

```
              Qwen2.5-Coder-1.5B-Instruct + SQLite              
                                                                
 questions                                                   40 
 seconds                                                  19.10 
 questions / second                                      2.1000 
 denotation accuracy                                     0.1250 
 queries that failed to execute                            2/40 
                                                                
```

    A failed query is a *visible* failure - it is what a self-repair loop retries on.
    VRAM after text-to-sql       0.01 GB allocated /  0.02 GB reserved

## 11. Head-to-head Benchmark

The same questions, the same tables, the same normaliser, one model live at a time. The numbers already exist from sections 8-10, so this section collects and charts them rather than re-running the models - reloading three models to re-measure would just spend VRAM to reproduce what we have.

Read it as a shape, not a ranking. Forty questions carries roughly +/-8 points of sampling noise at these accuracy levels, and WTQ’s tiny untyped tables are the *least* favourable ground for text-to-SQL and the most favourable for cell selection. The generalisable findings are the ones that come from the mechanism, not the sample:

- TAPAS is the fastest and the only one that names its evidence cells.
- TAPEX beats it on anything needing an actual computed value, because it generates rather than selects.
- Text-to-SQL is the slowest per question here and the only one whose cost is flat in table size - on a 10,000-row table the other two cannot run at all.

------------------------------------------------------------------------

``` python
import pandas as pd

results = [
    {"model": "tapas-base-wtq", "params_m": 110, "approach": "cell selection",
     "accuracy": round(denotation_accuracy(tapas_preds, gold), 4), "seconds": round(tapas_secs, 2)},
    {"model": "tapex-base-wtq", "params_m": 140, "approach": "seq2seq",
     "accuracy": round(denotation_accuracy(tapex_preds, gold), 4), "seconds": round(tapex_secs, 2)},
    {"model": "qwen2.5-coder-1.5b", "params_m": 1540, "approach": "text-to-SQL",
     "accuracy": round(denotation_accuracy(sql_preds, gold), 4), "seconds": round(sql_secs, 2)},
]
for r in results:
    r["q_per_sec"] = round(len(questions) / r["seconds"], 2)

df_results = pd.DataFrame(results).sort_values("accuracy", ascending=False)
show_table(
    df_results.to_dict("records"),
    title=f"WikiTableQuestions, {len(questions)} questions on small tables",
    best=("accuracy", "q_per_sec"),
    lower_is_better=("seconds",),
    caption="best per column in green - n=40, so treat gaps under ~8 points as ties",
)
```

```
                WikiTableQuestions, 40 questions on small tables                 
                                                                                 
 model                params_m   approach         accuracy   seconds   q_per_sec 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 tapex-base-wtq            140   seq2seq            0.9000    0.8900       44.94 
 tapas-base-wtq            110   cell selection     0.8750    1.0900       36.70 
 qwen2.5-coder-1.5b      1,540   text-to-SQL        0.1250     19.14      2.0900 
                                                                                 
     best per column in green - n=40, so treat gaps under ~8 points as ties      
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
    .add_yaxis("denotation accuracy x100", [round(r["accuracy"] * 100, 1) for r in results])
    .add_yaxis("questions / sec", [r["q_per_sec"] for r in results])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"WikiTableQuestions ({len(questions)} questions, small tables)",
            subtitle="RTX 3060 - a smoke test, not a leaderboard; +/-8 points of noise at n=40",
        ),
        yaxis_opts=opts.AxisOpts(name="score"),
        xaxis_opts=opts.AxisOpts(name="model", axislabel_opts=opts.LabelOpts(rotate=15)),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
    )
)
bar.render_notebook()
```

``` python
from pyecharts.charts import Scatter

# Accuracy against throughput - the trade-off that decides which one you deploy.
scatter = Scatter()
scatter.add_xaxis([r["q_per_sec"] for r in results])
for r in results:
    scatter.add_yaxis(
        r["model"], [[r["q_per_sec"], round(r["accuracy"] * 100, 1)]],
        symbol_size=18, label_opts=opts.LabelOpts(is_show=False),
    )
scatter.set_global_opts(
    title_opts=opts.TitleOpts(title="Accuracy vs throughput",
                              subtitle="text-to-SQL is the only one flat in table size"),
    xaxis_opts=opts.AxisOpts(name="questions / second", type_="value"),
    yaxis_opts=opts.AxisOpts(name="denotation accuracy x100", type_="value"),
    tooltip_opts=opts.TooltipOpts(trigger="item"),
)
scatter.render_notebook()
```

## 12. Interactive: ask your own table

Edit `MY_TABLE` and `MY_QUESTIONS` below and watch the two mechanisms disagree. This is the cell people run on its own, so it opens with a `require(...)` guard naming what it needs from Setup rather than dying on a bare `NameError`.

The questions worth trying are the ones that separate selection from computation:

- **Lookup** (“who has the most points?”) - both get it, TAPAS names the cell.
- **Aggregation** (“what is the total revenue?”) - TAPAS returns the `SUM` operator plus the four revenue cells, and `apply_aggregator` computes 1845 from them. TAPEX prints a single number with no arithmetic behind it, and on this table it gets it wrong. That is section 9’s provenance problem in one line: nothing in TAPEX’s output distinguishes a computed number from a fabricated one.
- **Comparison** (“how many earned more than 500?”) - the classic TAPAS failure; it selects plausible cells and mislabels the operator.
- **Not in the table** (“what is the CEO’s name?”) - neither model abstains. Both will confidently return a cell. Refusal is not a behaviour these models have, which is a real argument for the text-to-SQL path, where an empty result set is at least honest.

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


require("device", "dtype", "HF_CACHE", "free_memory", "vram", "normalize_answer",
        "apply_aggregator", "tapex_input")

import pandas as pd
from transformers import AutoModelForSeq2SeqLM, BartTokenizerFast, pipeline

MY_TABLE = pd.DataFrame({
    "region": ["North", "South", "East", "West"],
    "reps": ["12", "7", "19", "4"],
    "revenue_k": ["480", "310", "905", "150"],
    "quarter": ["Q1", "Q1", "Q1", "Q1"],
}).astype(str)

MY_QUESTIONS = [
    "which region has the highest revenue?",
    "what is the total revenue?",
    "how many regions have more than 10 reps?",
    "who is the CEO?",  # not answerable from this table - watch neither model say so
]

# Re-runnable: this cell frees both models at the end, so guard the loads or a second
# shift-enter raises NameError.
if "my_tapas" not in globals():
    my_tapas = pipeline(
        "table-question-answering", model="google/tapas-base-finetuned-wtq",
        device=device, model_kwargs={"cache_dir": HF_CACHE},
    )
if "my_tapex" not in globals():
    # BartTokenizerFast, not AutoTokenizer - see section 9.
    my_tapex_tok = BartTokenizerFast.from_pretrained(
        "microsoft/tapex-base-finetuned-wtq", cache_dir=HF_CACHE)
    my_tapex = AutoModelForSeq2SeqLM.from_pretrained(
        "microsoft/tapex-base-finetuned-wtq", cache_dir=HF_CACHE
    ).to(device).eval()

print(MY_TABLE.to_string(index=False), "\n")
for q in MY_QUESTIONS:
    out = my_tapas(table=MY_TABLE, query=q)
    with torch.inference_mode():
        enc = my_tapex_tok(tapex_input(MY_TABLE, q), return_tensors="pt",
                           truncation=True, max_length=1024).to(device)
        tx = my_tapex_tok.decode(
            my_tapex.generate(**enc, max_length=32)[0], skip_special_tokens=True
        ).strip()
    print(f"Q: {q}")
    print(f"   tapas  cells={out['cells']} agg={out.get('aggregator', 'NONE')} "
          f"-> {apply_aggregator(out)}")
    print(f"   tapex  {tx}\n")

del my_tapas, my_tapex, my_tapex_tok
free_memory()
vram("final")
```

    region reps revenue_k quarter
     North   12       480      Q1
     South    7       310      Q1
      East   19       905      Q1
      West    4       150      Q1 

    Q: which region has the highest revenue?
       tapas  cells=['East'] agg=NONE -> ['East']
       tapex  east

    Q: what is the total revenue?
       tapas  cells=['480', '310', '905', '150'] agg=SUM -> ['1845.0']
       tapex  485

    Q: how many regions have more than 10 reps?
       tapas  cells=['North', 'East'] agg=COUNT -> ['2']
       tapex  2

    Q: who is the CEO?
       tapas  cells=['East'] agg=NONE -> ['East']
       tapex  north

    VRAM final                   0.01 GB allocated /  0.03 GB reserved

## 13. Common Frameworks

Table QA is the task where the model is the least interesting component. Once text-to-SQL became the dominant approach, the hard parts moved to the database side: finding the right three tables in a warehouse of four thousand, validating that generated SQL is safe and syntactically legal, executing it under a permission boundary, and retrying on the error message. The frameworks below are mostly data infrastructure, which is the honest shape of this task.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [transformers](https://github.com/huggingface/transformers) | modelling | TAPAS and TAPEX for the encoder path, and any code LLM for the text-to-SQL path, under one API | Apache 2.0 | Default. The three architectures of sections 8-10 all load from here |
| [DuckDB](https://github.com/duckdb/duckdb) / SQLite + [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy) | data | The execution engine the generated SQL actually runs against, plus dialect handling and connection safety | MIT | Always for text-to-SQL. DuckDB in particular runs analytical queries over a DataFrame with no server at all |
| [SQLGlot](https://github.com/tobymao/sqlglot) | data | Parse, validate and transpile the generated SQL before executing it - catch illegal queries and rewrite across dialects | MIT | Always. Executing unvalidated model output against a database is the security problem of this task, not a style issue |
| [pandas](https://github.com/pandas-dev/pandas) / [polars](https://github.com/pola-rs/polars) | data | Table loading, type inference, and the linearisation that TAPAS and TAPEX consume | BSD-3 / MIT | Always. Column dtypes decide whether a comparison works, and the models see strings either way |
| A vector index over the schema ([faiss](https://github.com/facebookresearch/faiss) / [Qdrant](https://github.com/qdrant/qdrant)) | data | Schema linking: retrieve the relevant tables and columns per question before generating | MIT / Apache 2.0 | Any real warehouse. The model’s SQL ability is rarely the bottleneck; finding the right three tables is |
| [vLLM](https://github.com/vllm-project/vllm) / [SGLang](https://github.com/sgl-project/sglang) | inference runtime | Batched sampling, which makes execution-guided majority voting - generate 5-8 queries, run them all, take the modal result - affordable | Apache 2.0 | Text-to-SQL in production. Voting is a standard BIRD-leaderboard technique and costs only compute |
| [outlines](https://github.com/dottxt-ai/outlines) / [xgrammar](https://github.com/mlc-ai/xgrammar) | inference runtime | Grammar-constrained decoding, the general version of PICARD - only tokens that can continue a valid parse are sampled | Apache 2.0 / MIT | Always worth it. Syntactic validity by construction beats retrying on syntax errors |
| [LlamaIndex](https://github.com/run-llama/llama_index) / [LangGraph](https://github.com/langchain-ai/langgraph) | orchestration | The retrieve-schema, generate, execute, repair-on-error loop, with the retry budget and permissions in one place | MIT | Building the product. One self-repair retry recovers a third to a half of failed queries |
| Execution accuracy via the [Spider](https://github.com/taoyds/spider) / [BIRD](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird) harnesses | evaluation | Comparing *result sets* rather than SQL strings, which is the only meaningful correctness test | Apache 2.0 / MIT | Always. Two different queries returning the same rows are both right, and string match calls one of them wrong |

**The 2026 default stack** is a code LLM through vLLM with grammar-constrained decoding, SQLGlot validating before execution, DuckDB or the real warehouse executing under a read-only role, schema linking via a vector index, and one self-repair retry. The encoder models remain useful for small tables where no database exists.

The common wrong turn is executing model-generated SQL with write permissions and no parse check. This is the one task in the folder with a genuine security surface, and the fix is boring: read-only credentials, a parse-and-validate step, a row limit, and a timeout. The second is evaluating on SQL string match instead of execution results.

------------------------------------------------------------------------

## 14. Going Further

- **Fine-tune TAPEX on your own tables.** `AutoModelForSeq2SeqLM.from_pretrained("microsoft/tapex-base")` plus `TapexTokenizer` and `Seq2SeqTrainer` is a standard seq2seq recipe. A few thousand (table, question, answer) triples from your domain beats any off-the-shelf checkpoint, because column naming conventions are domain-specific and the model has to learn yours.
- **Retrieve rows before you encode.** For tables past ~50 rows, the encoder approaches need a row-selection stage: embed each row, retrieve the top-k against the question, and encode only those. This is `07_Feature_Extraction` plus `11_Text_Ranking` used as a pre-filter, and it is how you stretch a 512-token model over a real spreadsheet.
- **Self-repair is the cheapest text-to-SQL win.** Feed the SQLite error message back to the model and regenerate. One retry typically recovers a third to a half of failed queries, which is a bigger accuracy jump than moving up a model size.
- **Sample and vote.** Generate 5-8 queries at `temperature=0.7`, execute all of them, and return the most common result set. This “execution-guided majority vote” is standard on BIRD leaderboards and costs only compute.
- **Schema linking is where real accuracy comes from.** On a 4,000-table warehouse, put the schema in a vector index, retrieve the relevant tables per question, and only then generate. The model’s SQL ability is rarely the bottleneck; finding the right three tables is.
- **Constrain the decode.** `PICARD`-style incremental parsing (reject tokens that cannot continue a valid SQL parse) guarantees syntactically valid output. In 2026 the general version of this is grammar-constrained generation with a SQL GBNF, available in most local-inference runtimes.
- **Hybrid table+text.** TAT-QA and HybridQA need a table *and* the surrounding prose. The workable pattern is retrieval over both, then a long-context LLM - no specialised architecture has beaten that.
- **Related notebooks.** `03_Question_Answering` (the unstructured-source version), `08_Text_Generation` (decoding and constrained generation), `11_Text_Ranking` (row and schema retrieval), `Multimodal/05_Document_Question_Answering` (tables that live inside page images), `Tabular/00_Tabular_Classification` (tables without language).

------------------------------------------------------------------------

Back to top
