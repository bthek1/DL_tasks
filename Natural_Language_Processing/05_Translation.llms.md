# Translation

Machine translation in mid-2026: bilingual specialists against massively multilingual models against LLMs, why BLEU stopped being the metric that decides anything, and runnable code that scores three models on the same WMT22 test set.

Author

Benedict Thekkel

## 1. What is Translation?

Machine translation maps text in a **source language** to text in a **target language**, preserving meaning. It is the oldest task in NLP - the Georgetown-IBM experiment translated 60 Russian sentences in 1954 - and the one where neural methods delivered the most visible break with what came before.

**Input.** A sentence, or increasingly a document. The unit matters more than it looks: sentence-level translation cannot resolve a pronoun whose antecedent is in the previous sentence, cannot keep terminology consistent across a page, and cannot pick a formality register from context. Document-level translation can, and it is the main quality frontier that remains.

**Output.** Text in the target language. Unlike classification there is no single correct answer - a good sentence has many valid translations, which is the root of every difficulty in evaluating this task.

**The three model families**, and the choice between them is the whole engineering decision:

| Family | Shape | Languages | Typical size |
|----|----|----|----|
| **Bilingual** (Marian / OPUS-MT) | encoder-decoder, one direction | 1 pair | 40-80M |
| **Massively multilingual** (NLLB, M2M-100, mBART) | encoder-decoder + language tokens | 50-200 | 400M-54B |
| **LLM** (Qwen, Llama, Gemma, frontier models) | decoder-only, prompted | 100+ | 0.5B-500B+ |

Multilingual models work by prepending a **target-language token** to the decoder, so one set of weights covers every direction. That buys **zero-shot transfer** between pairs never seen together in training, and it costs the **curse of multilinguality**: fixed capacity split across 200 languages means each high-resource pair is slightly worse than a dedicated bilingual model of the same size.

**What “quality” means here** is contested in a way it is not for classification. A translation can be adequate but unnatural, fluent but wrong, or correct but in the wrong register. The current professional standard, MQM, has human annotators mark *errors* with categories and severities rather than score a sentence 1-5, precisely because a single number hides which of those failures happened.

**Neighbouring tasks:**

| Task | How it differs | Notebook |
|----|----|----|
| Summarization | Compresses rather than transfers | `06_Summarization` |
| Text generation | Open-ended, no source to be faithful to | `08_Text_Generation` |
| Speech translation | Source is audio, not text | `Audio/02_Automatic_Speech_Recognition` |
| Sentence similarity | Cross-lingual embeddings score alignment, not produce it | `10_Sentence_Similarity` |

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Web page and UI translation | Consumer web (Google Translate, browser built-ins) | Page text -\> target language | Latency and cost at enormous volume; increasingly on-device |
| Localisation pipelines | Software, gaming, documentation | Source strings + glossary + TM -\> target strings | Terminology consistency and placeholder preservation, not fluency |
| Post-editing for translators | Language services industry | Source -\> draft; human edits | Edit distance, not BLEU. A near-miss costs more than an obvious miss |
| Customer support across languages | SaaS, e-commerce | Ticket/chat -\> agent language, and back | Real-time latency; domain jargon; must not invert negation |
| Subtitling and dubbing | Media | Transcript + timing -\> target with length constraints | Character-per-second limits force compression, not just translation |
| E-commerce listings | Retail marketplaces | Product titles/descriptions -\> 30 languages | Cost per million; noisy, fragmentary source text |
| Humanitarian and public health | NGOs, government | Critical information -\> low-resource languages | Coverage of languages no commercial system serves; correctness is life-safety |
| Legal and patent translation | Legal | Contracts, filings -\> certified translation | Literalness and auditability; a human signs it |

What the BLEU score hides:

- **Placeholders, markup and formatting break more deployments than fluency does.** `{count} items` becoming `{Anzahl} Artikel`, an HTML tag swallowed, a trailing space lost - each is a production bug that no quality metric measures and every localisation engineer spends time on.
- **Terminology is a hard constraint, not a preference.** Your product name, legal terms, and drug names have exactly one correct rendering. Systems solve this with glossary injection or constrained decoding, and general model quality does not help.
- **Fluent errors are the dangerous ones.** Modern MT rarely produces gibberish; it produces confident, natural sentences with a dropped negation or a swapped number. Human reviewers catch broken output easily and fluent-but-wrong output rarely, which makes hallucination in MT a safety issue in medical and legal settings.
- **Low-resource means genuinely worse, not slightly worse.** For the ~180 NLLB languages outside the top 30, output is often usable for gist and not for publication. Aggregate multilingual scores average this away.
- **Domain shift is severe.** A general model translating clinical notes or patent claims underperforms a small model fine-tuned on 50k in-domain sentence pairs. In MT, more than most tasks, in-domain data beats scale.

------------------------------------------------------------------------

## 3. How Modern Translation Works

1.  **Rule-based and statistical MT (1954-2014).** Hand-written transfer rules, then IBM alignment models and phrase-based SMT (Moses). Phrase tables plus a target-language model, tuned with MERT against BLEU. Robust, interpretable, and permanently mediocre at word order between distant language pairs.
2.  **Neural encoder-decoder with attention (2014-2017).** Sutskever’s seq2seq, then Bahdanau attention, which let the decoder look back at any source position. Quality jumped, and attention - invented here, for this task - went on to reorganise the whole field.
3.  **The Transformer (2017).** “Attention Is All You Need” was a **translation** paper. Removing recurrence made training parallel, and the resulting architecture beat the state of the art on WMT En-De at a fraction of the compute. Everything after this is downstream of a machine translation result.
4.  **Massively multilingual NMT (2019-2022).** One model, many directions, with a target-language token. mBART, M2M-100 (100 languages, non-English-centric), and **NLLB-200** (2022, 200 languages, explicitly targeting low-resource pairs with a mixture-of-experts model and heavy data mining). Zero-shot transfer between unseen pairs became real.
5.  **Metrics caught up, and it mattered (2020-2022).** COMET and BLEURT - neural metrics trained on human judgements - correlated with human ratings far better than BLEU. The WMT metrics task showed BLEU actively misranking systems, and the field’s conclusions changed as a result: several “improvements” of the 2010s did not survive re-evaluation.
6.  **LLMs as translators (2022-2026).** General instruction-tuned models overtook dedicated NMT systems on high-resource pairs. GPT-4-class models matched or beat commercial MT on WMT22 for English-centric directions, and by 2024-2026 LLM-based systems led most WMT tracks. Their real advantages are contextual: document-level consistency, honouring a style instruction, using a supplied glossary, and explaining an ambiguity - none of which an encoder-decoder NMT model can do at all. Their weaknesses are low-resource languages, where dedicated models like NLLB still lead, and cost.
7.  **Where the frontier is (2026).** Document- and context-aware translation, quality estimation without references (predicting a COMET-like score with no gold translation, so systems can flag their own bad output), and adaptive translation that learns from a translator’s edits within a session.

**Where it stands (mid-2026).** For a high-resource pair with context and style requirements, an LLM wins on quality. For a single high-volume pair with a fixed domain, a small bilingual Marian model still gives the best quality per FLOP by a wide margin - 75M params, milliseconds, and quality that a 2019 research lab would not have believed. For breadth across 200 languages, especially low-resource ones, NLLB remains the specialist. Most production systems route: cheap model by default, LLM for the content that matters.

------------------------------------------------------------------------

## 4. Evaluation Metrics

**BLEU** (Papineni et al., 2002) - modified n-gram precision against one or more references, with a brevity penalty:

$$\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{4} w_n \log p_n\right), \qquad \text{BP} = \min\left(1,\ e^{1 - r/c}\right)$$

where $p_n$ is clipped n-gram precision and $r/c$ is the reference-to-candidate length ratio. It is fast, deterministic and reference-based, and it has been the default for twenty years.

**chrF / chrF++** (Popovic, 2015) - the same idea over **character** n-grams, with recall weighted more heavily than precision. Because it works at character level it degrades gracefully for morphologically rich languages (Finnish, Turkish, Czech) where a single wrong suffix zeroes a BLEU word match. It is the better of the two string metrics and should be reported alongside or instead of BLEU.

**COMET / BLEURT** - neural metrics. COMET encodes the source, the hypothesis and the reference with a multilingual encoder and regresses onto human quality judgements. It correlates far better with human ratings than any n-gram metric, and it is the metric the WMT community actually uses to rank systems. **COMET-QE** does it without a reference, which makes it usable in production to flag low-confidence output. The cost is that it is a 580M-2B model, so scoring is a GPU job, and its scores are not comparable across versions.

**MQM** - the human gold standard. Annotators mark errors by category (accuracy, fluency, terminology, style) and severity (minor, major, critical), and the score is a weighted error count. This is what “human evaluation” means in serious MT work; 5-point adequacy ratings have been retired for being too noisy.

**The pitfalls, and they are severe enough that BLEU numbers are routinely incomparable:**

- **Tokenisation changes the number.** BLEU on tokenised text is not BLEU on detokenised text. This is why **sacreBLEU** exists: it fixes the tokenisation and emits a version string with the score. A BLEU number without that signature cannot be compared to anything.
- **Reference count changes the number.** More references means higher BLEU. Comparing a single-reference score to a four-reference score is meaningless.
- **BLEU misranks good systems.** Above roughly 30 BLEU, differences correlate poorly with human judgement, and BLEU systematically penalises valid paraphrase - which is exactly what an LLM produces. Reporting only BLEU when comparing an LLM to an NMT model understates the LLM, sometimes badly. The benchmark below shows this directly.
- **BLEU is corpus-level.** Sentence-level BLEU is unstable (a 12-word sentence with no 4-gram match scores 0 regardless of quality) and should only be used with smoothing, if at all.

The cell below implements corpus BLEU and chrF from scratch - roughly 40 lines, and seeing the clipping and the brevity penalty spelled out is worth more than importing them.

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
from collections import Counter


def ngrams(tokens, n):
    "All n-grams of a token list, as a Counter."
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(hyps, refs, max_n=4):
    "Corpus BLEU: clipped n-gram precision up to max_n, with a brevity penalty."
    num = [0] * max_n
    den = [0] * max_n
    hyp_len = ref_len = 0
    for hyp, ref in zip(hyps, refs):
        h_toks, r_toks = hyp.split(), ref.split()
        hyp_len += len(h_toks)
        ref_len += len(r_toks)
        for n in range(1, max_n + 1):
            h_ng, r_ng = ngrams(h_toks, n), ngrams(r_toks, n)
            # "clipped": an n-gram counts at most as often as it appears in the reference
            num[n - 1] += sum(min(c, r_ng[g]) for g, c in h_ng.items())
            den[n - 1] += sum(h_ng.values())
    if min(num) == 0:
        return 0.0
    log_p = sum((1 / max_n) * math.log(num[i] / den[i]) for i in range(max_n) if den[i])
    bp = min(1.0, math.exp(1 - ref_len / hyp_len)) if hyp_len else 0.0
    return 100 * bp * math.exp(log_p)


def corpus_chrf(hyps, refs, max_n=6, beta=2):
    "chrF: character n-gram F-score, recall weighted beta times precision."
    f_scores = []
    for hyp, ref in zip(hyps, refs):
        h, r = hyp.replace(" ", ""), ref.replace(" ", "")
        per_n = []
        for n in range(1, max_n + 1):
            h_ng, r_ng = ngrams(h, n), ngrams(r, n)
            overlap = sum(min(c, r_ng[g]) for g, c in h_ng.items())
            prec = overlap / max(sum(h_ng.values()), 1)
            rec = overlap / max(sum(r_ng.values()), 1)
            if prec + rec:
                per_n.append((1 + beta ** 2) * prec * rec / (beta ** 2 * prec + rec))
            else:
                per_n.append(0.0)
        f_scores.append(sum(per_n) / max_n)
    return 100 * sum(f_scores) / len(f_scores)


# Toy example: three hypotheses against one reference. The third is a valid paraphrase -
# the kind of output an LLM produces and BLEU punishes.
ref = ["The committee approved the new budget on Tuesday morning."]
cases = {
    "near-identical": ["The committee approved the new budget on Tuesday morning."],
    "one word wrong": ["The committee rejected the new budget on Tuesday morning."],
    "valid paraphrase": ["On Tuesday morning, the new budget was approved by the committee."],
    "fluent but empty": ["The weather on Tuesday morning was quite pleasant indeed."],
}
show_table([{"hypothesis type": name, "BLEU": round(corpus_bleu(hyp, ref), 2),
             "chrF": round(corpus_chrf(hyp, ref), 2), "text": hyp[0]}
            for name, hyp in cases.items()],
           title=f"All four scored against one reference: {ref[0]!r}",
           caption="'one word wrong' inverts the meaning and outscores a correct "
                   "paraphrase on BLEU - this is why COMET exists and why BLEU alone "
                   "no longer decides anything")
```

```
   All four scored against one reference: 'The committee approved the new budget on Tuesday morning.'   
                                                                                                        
 hypothesis type      BLEU     chrF   text                                                              
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 near-identical     100.00   100.00   The committee approved the new budget on Tuesday morning.         
 one word wrong      66.06    81.90   The committee rejected the new budget on Tuesday morning.         
 valid paraphrase   0.0000    75.39   On Tuesday morning, the new budget was approved by the committee. 
 fluent but empty   0.0000    41.29   The weather on Tuesday morning was quite pleasant indeed.         
                                                                                                        
  'one word wrong' inverts the meaning and outscores a correct paraphrase on BLEU - this is why COMET   
                          exists and why BLEU alone no longer decides anything                          
```

## 5. Datasets

| Dataset | Contents | Size | Scope | License | Typical use |
|----|----|----|----|----|----|
| [WMT22 test sets](https://huggingface.co/datasets/haoranxu/WMT22-Test) | Official WMT22 general-domain test sets | ~2k per pair | 10+ pairs | CC BY-SA 4.0 | Evaluation; used below |
| [FLORES-200](https://huggingface.co/datasets/facebook/flores) | 3k sentences from Wikipedia, translated into 200 languages | 2009 devtest | 200 langs | CC BY-SA 4.0 | The standard multilingual eval; fully many-to-many |
| [WMT19 / WMT21](https://huggingface.co/datasets/wmt/wmt19) | Parallel training corpora + test sets | 10M-40M pairs | ~10 pairs | mixed | Training and historical comparison |
| [OPUS](https://opus.nlpl.eu/) | Aggregator: 700+ corpora, 700+ languages | billions | huge | mixed per corpus | The main source of NMT training data |
| [OPUS-100](https://huggingface.co/datasets/Helsinki-NLP/opus-100) | English-centric sample of OPUS, 100 pairs | 55M | 100 langs | mixed | Multilingual training baseline |
| [Europarl](https://huggingface.co/datasets/Helsinki-NLP/europarl) | European Parliament proceedings | 60M | 21 EU langs | open | Classic clean parallel data; formal register only |
| [CCMatrix / CCAligned](https://huggingface.co/datasets/yhavinga/ccmatrix) | Web-mined bitext at scale | 4.5B | 90+ | CC BY-NC | Where low-resource data comes from; noisy |
| [Tatoeba](https://huggingface.co/datasets/Helsinki-NLP/tatoeba_mt) | Community sentence pairs, many pairs | 10M+ | 400+ | CC BY 2.0 | Small-pair coverage; short sentences |
| [MQM annotations](https://github.com/google/wmt-mqm-human-evaluation) | Human error annotations for WMT systems | 100k+ segments | several pairs | Apache 2.0 | Training and validating neural metrics |
| [IWSLT](https://huggingface.co/datasets/IWSLT/iwslt2017) | TED talk transcripts, spoken register | 200k+ | 10+ pairs | CC BY-NC-ND | Spoken-language and speech-translation eval |

This notebook evaluates **English to German** on the **WMT22 general test set** (1,984 segments, sampled down). It is a deliberately favourable pair - En-De is among the highest-resource directions in existence, and every model below has seen enormous amounts of it. Read the results as “what a strong pair looks like”, not as a claim about the 200 languages NLLB covers, where scores are far lower and the ranking between these models changes.

**FLORES-200** is the better choice for multilingual evaluation because it is the same 2,009 sentences in all 200 languages, making every direction directly comparable. It sits behind an auto-approved gate on the Hub, so it needs an accepted licence and a token; WMT22 needs neither, which is why it is used here.

Downloads land in `DL_tasks/datasets/` via `cache_dir` (gitignored).

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

The reference points are the annual [WMT shared task](https://www2.statmt.org/wmt24/) results (human MQM evaluation, the only ranking that fully counts), the [WMT metrics task](https://www2.statmt.org/wmt24/metrics-task.html) for how to score, and the [OPUS-MT model list](https://huggingface.co/Helsinki-NLP) for the bilingual zoo.

| Model | Params | License | Coverage | Type | Best for |
|----|----|----|----|----|----|
| [opus-mt-en-de](https://huggingface.co/Helsinki-NLP/opus-mt-en-de) | 74M | Apache 2.0 | 1 pair | Marian encoder-decoder | one high-volume pair, minimal cost; used below |
| [opus-mt-tc-big-en-de](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-de) | 232M | CC BY 4.0 | 1 pair | Marian, larger | best bilingual quality per pair |
| [nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | 615M | CC BY-NC 4.0 | 200 langs | multilingual enc-dec | broad coverage on one GPU; used below |
| [nllb-200-3.3B](https://huggingface.co/facebook/nllb-200-3.3B) | 3.3B | CC BY-NC 4.0 | 200 langs | multilingual enc-dec | best open low-resource quality (~6.6 GB fp16) |
| [m2m100_418M / 1.2B](https://huggingface.co/facebook/m2m100_418M) | 418M-1.2B | MIT | 100 langs | multilingual, non-English-centric | direct X-to-Y without pivoting through English |
| [mbart-large-50-many-to-many](https://huggingface.co/facebook/mbart-large-50-many-to-many-mmt) | 611M | MIT | 50 langs | multilingual enc-dec | older but permissively licensed |
| [madlad400-3b-mt](https://huggingface.co/google/madlad400-3b-mt) | 3B | Apache 2.0 | 400+ langs | T5 encoder-decoder | widest language coverage, commercial licence |
| [Qwen3-1.7B / 4B](https://huggingface.co/Qwen/Qwen3-1.7B) | 1.7-4B | Apache 2.0 | 100+ langs | decoder LLM | context, style, glossary control; used below |
| [TowerInstruct-7B](https://huggingface.co/Unbabel/TowerInstruct-7B-v0.2) | 7B | CC BY-NC 4.0 | 10 langs | translation-specialised LLM | strongest open LLM MT on its pairs (needs \>12 GB) |
| [wmt22-comet-da](https://huggingface.co/Unbabel/wmt22-comet-da) | 580M | Apache 2.0 | multilingual | **metric**, not a translator | the score that actually ranks systems |
| Frontier LLMs (Claude, GPT, Gemini) | \- | proprietary | 100+ | decoder LLM | document-level, top of recent WMT human evaluations |

**How to choose.** One pair, high volume, fixed domain: a Marian model, fine-tuned on your data - 74M params translating in single-digit milliseconds, and fine-tuning it on 50k in-domain pairs is a couple of GPU-hours. Many languages including low-resource: NLLB-200 (non-commercial licence) or MADLAD-400 (Apache 2.0). Quality on high-resource pairs where context, terminology or register matter: an LLM. Note the **licence trap** in this task specifically - NLLB and Tower are CC BY-NC, so a commercial product cannot use them, and this rules out the best open low-resource models for many teams.

**Note on size.** NLLB-3.3B (~6.6 GB in fp16) and TowerInstruct-7B (~14 GB) exceed a comfortable fit on this box’s 12 GB of VRAM alongside anything else; they belong in this table, not in a runnable cell. The three models below total roughly 6 GB of downloads.

------------------------------------------------------------------------

## 7. Setup

Everything loads through Hugging Face `transformers` - no vendor packages. Package roles:

- `transformers` + `torch` - Marian, NLLB and the LLM
- `accelerate` - `device_map` placement
- `datasets` - the WMT22 En-De test set
- `pandas` + `pyecharts` - the benchmark table and charts
- `rich` - the result tables. It renders to HTML inside Jupyter, so the tables survive into the published docs; `show_table` / `show_kv` / `rule` are defined in the first code cell of section 4.

Metrics are implemented inline (section 4). For real work use **`sacrebleu`** (`pip install sacrebleu`), which gives reproducible, signature-stamped BLEU and chrF, and **`unbabel-comet`** for COMET - neither is a repo dependency, and the inline versions here are for understanding the arithmetic, not for publishing numbers.

Three `transformers` details that decide whether the code below works:

- **Language tokens are model-specific.** Marian models are single-direction and need no language marker. NLLB needs `tokenizer.src_lang` set *before* encoding and `forced_bos_token_id` set to the target language token at generation. M2M-100 uses a different attribute again. Getting this wrong produces output in the wrong language rather than an error.
- **`tokenizer.lang_code_to_id` was removed** in recent `transformers`. The current way to get NLLB’s target token id is `tokenizer.convert_tokens_to_ids("deu_Latn")`, which is what the cell below uses.
- **Beam search is the NMT default and matters here.** `num_beams=4` is the standard setting for encoder-decoder MT and is typically worth 1-2 BLEU over greedy. LLMs are usually run greedy for translation (sampling adds nothing but variance when there is a source to be faithful to).

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

# WMT22 general test set, English -> German. Ungated parquet; rows are {"en-de": {...}}.
PAIR = "en-de"
SRC, TGT = "en", "de"
wmt = load_dataset("haoranxu/WMT22-Test", PAIR, split="test", cache_dir=HF_CACHE)

N = 120  # segments to translate. The full set is 2k and takes a while with beam search.
pairs = [r[PAIR] for r in wmt]
pairs = [p for p in pairs if 20 <= len(p[SRC]) <= 300][:N]  # drop fragments and outliers

sources = [p[SRC].strip() for p in pairs]
references = [p[TGT].strip() for p in pairs]
N = len(sources)

print(wmt)
print(f"\ntranslating {N} segments, {SRC} -> {TGT}")
print(f"mean source length: {sum(len(s.split()) for s in sources) / N:.1f} words\n")
for s, r in list(zip(sources, references))[:3]:
    print(f"  src: {s[:95]}")
    print(f"  ref: {r[:95]}\n")
```

    Dataset({
        features: ['en-de'],
        num_rows: 2037
    })

    translating 120 segments, en -> de
    mean source length: 17.8 words

      src: You can come back any time as our chat service window is open 24/7
      ref: Sie können jederzeit wiederkommen, da unser Chat-Service-Fenster täglich rund um die Uhr geöffn

      src: I sincerely hope you get to find a resolution
      ref: Ich hoffe wirklich, dass Sie eine Lösung finden werden

      src: Thank you for contacting #PRS_ORG#, it was my pleasure to assist you today.
      ref: Vielen Dank, dass Sie #PRS_ORG# kontaktiert haben, es hat mich gefreut, Ihnen helfen zu können.

## 8. Bilingual specialist: Marian / OPUS-MT

The Helsinki-NLP OPUS-MT models are small Transformer encoder-decoders trained on OPUS data, one model per direction, published for over a thousand language pairs. `opus-mt-en-de` is **74M parameters** - a fifth of the smallest model elsewhere in this notebook and about 300 MB on disk.

That size is the point. It translates in single-digit milliseconds per sentence on a GPU, runs perfectly well on a CPU, and on a high-resource pair like En-De the quality gap to a model ten times larger is small. If you have one direction and volume, this is the correct engineering answer, and fine-tuning it on 50k in-domain sentence pairs takes a couple of GPU-hours and typically beats every general model on your domain.

The limitations are exactly what you would expect: one direction per model (En-De and De-En are separate downloads), no context beyond the sentence, no way to instruct it, and no glossary support without constrained decoding. Marian also has a known tendency to **drop content on long inputs** rather than truncate visibly - the sentence comes out fluent and shorter than it should be, which the brevity penalty catches in aggregate and a human reviewer might not.

------------------------------------------------------------------------

``` python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

marian_id = "Helsinki-NLP/opus-mt-en-de"
marian_tok = AutoTokenizer.from_pretrained(marian_id, cache_dir=HF_CACHE)
marian = AutoModelForSeq2SeqLM.from_pretrained(marian_id, cache_dir=HF_CACHE).to(device).eval()
print(f"parameters: {marian.num_parameters() / 1e6:.0f}M")
vram("marian loaded")


@torch.inference_mode()
def translate_seq2seq(model, tok, texts, batch_size=16, num_beams=4, **gen_kwargs):
    "Batched beam-search translation for any encoder-decoder model."
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(model.device)
        gen = model.generate(**enc, num_beams=num_beams, max_new_tokens=256, **gen_kwargs)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return out


demo = ["The committee approved the new budget on Tuesday morning.",
        "She did not say that the report was ready."]
show_table([{"english": s, "german": t}
            for s, t in zip(demo, translate_seq2seq(marian, marian_tok, demo))],
           title="Two sentences - the second tests whether negation scope survives")

t0 = time.perf_counter()
marian_hyps = translate_seq2seq(marian, marian_tok, sources)
marian_secs = time.perf_counter() - t0

show_kv({"segments": N, "seconds": round(marian_secs, 1),
         "segments / second": round(N / marian_secs, 2),
         "BLEU": round(corpus_bleu(marian_hyps, references), 2),
         "chrF": round(corpus_chrf(marian_hyps, references), 2)},
        title="Helsinki-NLP/opus-mt-en-de (74M params, one direction)")

del marian, marian_tok
free_memory()
vram("after marian")
```

    /home/bthek1/Knowledge/.venv/lib/python3.14/site-packages/transformers/models/marian/tokenization_marian.py:176: UserWarning: Recommended: pip install sacremoses.
      warnings.warn("Recommended: pip install sacremoses.")

    parameters: 74M
    VRAM marian loaded           0.30 GB allocated /  0.30 GB reserved

    [transformers] Both `max_new_tokens` (=256) and `max_length`(=512) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

```
                        Two sentences - the second tests whether negation scope survives                        
                                                                                                                
 english                                                 german                                                 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 The committee approved the new budget on Tuesday        Der Ausschuß billigte den neuen Haushaltsplan am       
 morning.                                                Dienstag vormittag.                                    
 She did not say that the report was ready.              Sie sagte nicht, dass der Bericht fertig sei.          
                                                                                                                
```

    [transformers] Both `max_new_tokens` (=256) and `max_length`(=512) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=512) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=512) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=512) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=512) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=512) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=512) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=512) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

```
     Helsinki-NLP/opus-mt-en-de (74M params, one direction)     
                                                                
 segments                                                   120 
 seconds                                                 2.8000 
 segments / second                                        42.24 
 BLEU                                                     29.36 
 chrF                                                     58.23 
                                                                
```

    VRAM after marian            0.01 GB allocated /  0.02 GB reserved

## 9. Massively multilingual: NLLB-200-distilled-600M

“No Language Left Behind” (Meta, 2022) is one model covering **200 languages in every direction** - about 40,000 translation directions from a single checkpoint. The 600M distilled version used here is the practical one; the full model is a 54B-parameter mixture of experts, and the 3.3B dense version is the usual quality choice when VRAM allows.

The technical interest is in what it took to get there. Most of those 200 languages have no meaningful parallel corpus, so NLLB was built on large-scale **bitext mining** (finding translation pairs by embedding similarity across web crawls), backtranslation, and a training regime that deliberately upsamples low-resource languages against the curse of multilinguality. The FLORES-200 benchmark was built alongside it because no evaluation set covering those languages existed.

**The mechanics differ from Marian in ways that silently break code.** You must set `tokenizer.src_lang` before encoding, and pass `forced_bos_token_id` for the target language at generation. Get the target token wrong and the model produces fluent output in the wrong language with no error. Language codes are `FLORES-200` style - `eng_Latn`, `deu_Latn`, `zho_Hans` - script included, because several languages are written in more than one.

On En-De expect it to land near the 74M Marian model despite being eight times the size. That is the curse of multilinguality made concrete, and it is a fair trade: what you bought with those parameters is the other 39,999 directions.

**Licence note:** NLLB is **CC BY-NC 4.0**. It cannot be used in a commercial product. MADLAD-400 (Apache 2.0) is the closest permissively licensed alternative.

------------------------------------------------------------------------

``` python
nllb_id = "facebook/nllb-200-distilled-600M"   # ~2.5 GB download
nllb_tok = AutoTokenizer.from_pretrained(nllb_id, cache_dir=HF_CACHE, src_lang="eng_Latn")
nllb = AutoModelForSeq2SeqLM.from_pretrained(
    nllb_id, dtype=dtype, cache_dir=HF_CACHE
).to(device).eval()
print(f"parameters: {nllb.num_parameters() / 1e6:.0f}M")
vram("nllb-600M loaded")

# The target language is a forced first decoder token. lang_code_to_id was removed from
# recent transformers - convert_tokens_to_ids is the current way.
DEU = nllb_tok.convert_tokens_to_ids("deu_Latn")
print("forced_bos_token_id for deu_Latn:", DEU)

t0 = time.perf_counter()
nllb_hyps = translate_seq2seq(nllb, nllb_tok, sources, batch_size=8, forced_bos_token_id=DEU)
nllb_secs = time.perf_counter() - t0

show_kv({"segments": N, "seconds": round(nllb_secs, 1),
         "segments / second": round(N / nllb_secs, 2),
         "BLEU": round(corpus_bleu(nllb_hyps, references), 2),
         "chrF": round(corpus_chrf(nllb_hyps, references), 2)},
        title="facebook/nllb-200-distilled-600M (8x the parameters, 200 languages)")

# One model, many directions - the thing the extra parameters actually bought.
sentence = "The committee approved the new budget on Tuesday morning."
show_table([{"language": name, "code": code,
             "translation": translate_seq2seq(
                 nllb, nllb_tok, [sentence], batch_size=1,
                 forced_bos_token_id=nllb_tok.convert_tokens_to_ids(code))[0]}
            for name, code in [("French", "fra_Latn"), ("Japanese", "jpn_Jpan"),
                               ("Swahili", "swh_Latn"), ("Icelandic", "isl_Latn")]],
           title=f"Same checkpoint, four more directions: {sentence!r}",
           caption="this is what the extra parameters bought - not en-de quality, "
                   "but the other 39,999 directions")

del nllb, nllb_tok
free_memory()
vram("after nllb")
```

    [transformers] Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

    parameters: 615M
    VRAM nllb-600M loaded        1.24 GB allocated /  1.26 GB reserved
    forced_bos_token_id for deu_Latn: 256042

    [transformers] Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

```
    facebook/nllb-200-distilled-600M (8x the parameters, 200    
                           languages)                           
                                                                
 segments                                                   120 
 seconds                                                 8.5000 
 segments / second                                        14.17 
 BLEU                                                     27.22 
 chrF                                                     57.57 
                                                                
```

    [transformers] Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

```
  Same checkpoint, four more directions: 'The committee approved the new budget on   
                                  Tuesday morning.'                                  
                                                                                     
 language    code       translation                                                  
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 French      fra_Latn   Le comité a approuvé le nouveau budget mardi matin.          
 Japanese    jpn_Jpan   委員会は火曜日の朝 新しい予算を承認した.                     
 Swahili     swh_Latn   Kamati iliidhinisha bajeti mpya Jumanne asubuhi.             
 Icelandic   isl_Latn   Nefndin samþykkti nýja fjárhagsáætlunina á þriðjudagsmorgun. 
                                                                                     
 this is what the extra parameters bought - not en-de quality, but the other 39,999  
                                     directions                                      
```

    VRAM after nllb              0.01 GB allocated /  0.02 GB reserved

## 10. LLM translator: Qwen3-1.7B

A general instruction-tuned decoder, prompted to translate. No encoder-decoder, no language tokens, no translation-specific training beyond whatever multilingual text was in its pretraining mix.

On a high-resource pair this is competitive, and on the things that actually cause rework in localisation it is not close:

- **Instructions work.** “Use the formal register”, “keep `{placeholder}` tokens verbatim”, “translate `dashboard` as `Ubersicht`” are all just text in the prompt. No NMT model can accept any of these without constrained decoding or retraining.
- **Context works.** Give it the previous two sentences and pronoun resolution and terminology consistency improve. This is the main open quality gap in MT and encoder-decoder models structurally cannot address it at the sentence level.
- **Ambiguity can be surfaced.** It can be asked to flag a source sentence it finds ambiguous - a quality-estimation signal for free.

The costs are equally concrete: roughly an order of magnitude more compute per sentence than Marian, output that needs post-processing (models love to add “Here is the translation:”), and a real risk of the model *answering* the source sentence instead of translating it when the source is a question. The prompt below is built to suppress exactly that.

**Watch the BLEU-versus-chrF gap in the benchmark.** An LLM paraphrases more freely than an NMT model trained to match reference style, so it loses n-gram matches it did not need to lose. The character-level metric is more forgiving, and a neural metric like COMET would be more forgiving still - the ranking between these three models can flip depending on which metric you report, which is the single most important practical lesson in section 4.

`enable_thinking=False` keeps Qwen3 from emitting a `<think>` block before the translation.

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
    "Translate the following English text into German. Output ONLY the German "
    "translation - no explanation, no quotation marks, no commentary. If the text is a "
    "question, translate the question; do not answer it.\n\nEnglish: {src}\nGerman:"
)


@torch.inference_mode()
def llm_translate(texts, batch_size=8, max_new_tokens=192, extra=""):
    "Greedy decode a translation per source segment; strip the usual LLM preamble."
    out = []
    for i in range(0, len(texts), batch_size):
        chats = [
            tok.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(src=s) + extra}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for s in texts[i:i + batch_size]
        ]
        enc = tok(chats, return_tensors="pt", padding=True, padding_side="left",
                  truncation=True, max_length=1024).to(llm.device)
        gen = llm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tok.eos_token_id)
        for g in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True):
            line = g.strip().split("\n")[0].strip().strip('"')
            out.append(line)
    return out


t0 = time.perf_counter()
llm_hyps = llm_translate(sources)
llm_secs = time.perf_counter() - t0

show_kv({"segments": N, "seconds": round(llm_secs, 1),
         "segments / second": round(N / llm_secs, 2),
         "BLEU": round(corpus_bleu(llm_hyps, references), 2),
         "chrF": round(corpus_chrf(llm_hyps, references), 2)},
        title="Qwen3-1.7B prompted - a lower BLEU than chrF suggests is the paraphrase penalty")

# The capability no encoder-decoder model has: instructions that change the output.
sample = "Could you please confirm whether the dashboard shows the {count} latest items?"
_variants = [
    ("plain", ""),
    ("formal register", "\n\nUse the formal register (Sie), not the informal one."),
    ("glossary + placeholders",
     "\n\nKeep any {placeholder} tokens exactly as they appear. "
     "Translate 'dashboard' as 'Dashboard', not as 'Ubersicht'."),
]
show_table([{"instruction": label,
             "translation": llm_translate([sample], batch_size=1, extra=extra)[0]}
            for label, extra in _variants],
           title=f"Same source, three instructions: {sample!r}",
           caption="no encoder-decoder NMT model can accept any of these without "
                   "constrained decoding or retraining")

del llm, tok
free_memory()
vram("after qwen3")
```

    VRAM qwen3-1.7b loaded       3.45 GB allocated /  4.41 GB reserved

```
  Qwen3-1.7B prompted - a lower BLEU than chrF suggests is the  
                       paraphrase penalty                       
                                                                
 segments                                                   120 
 seconds                                                  25.00 
 segments / second                                       4.8000 
 BLEU                                                     13.16 
 chrF                                                     47.11 
                                                                
```

```
   Same source, three instructions: 'Could you please confirm whether the dashboard shows the {count} latest   
                                                    items?'                                                    
                                                                                                               
 instruction               translation                                                                         
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 plain                     Könntest du bitte bestätigen, ob der Dashboard die {count} neuesten Elemente zeigt? 
 formal register           Könnten Sie bitte bestätigen, ob der Dashboard die letzten {count} Elemente zeigt?  
 glossary + placeholders   Könntest du bitte bestätigen, ob das Dashboard die {count} neuesten Items zeigt?    
                                                                                                               
        no encoder-decoder NMT model can accept any of these without constrained decoding or retraining        
```

    VRAM after qwen3             0.01 GB allocated /  0.02 GB reserved

## 11. Head-to-head Benchmark

The same source segments, the same references, the same two metrics, one model live at a time. Sections 8-10 produced the numbers; this collects and charts them.

How to read it honestly:

- **These are string metrics on one high-resource pair with one reference.** They rank systems that translate in a similar style. They do not rank systems that translate in different styles, which is exactly the Marian-versus-LLM comparison. A COMET score would likely reorder the table, and a human MQM evaluation would reorder it again.
- **The size-to-quality relationship is almost flat here.** 74M, 615M and 1.7B params land close together on En-De. Parameters bought language coverage (NLLB) and controllability (the LLM), not En-De quality - and on a low-resource pair the same three models would be tens of points apart.
- **Throughput is the real differentiator.** Marian is one to two orders of magnitude cheaper per segment than the LLM. At a billion segments a month that is the entire decision.
- At n=120 segments, BLEU carries roughly +/-1.5 points of noise; treat small differences as ties.

------------------------------------------------------------------------

``` python
import pandas as pd

RUNS = [
    ("opus-mt-en-de (Marian)", 74, "bilingual", marian_hyps, marian_secs),
    ("nllb-200-distilled-600M", 615, "multilingual (200)", nllb_hyps, nllb_secs),
    ("qwen3-1.7b (prompted)", 1720, "LLM", llm_hyps, llm_secs),
]

results = []
for name, params_m, kind, hyps, secs in RUNS:
    results.append({
        "model": name,
        "params_m": params_m,
        "type": kind,
        "BLEU": round(corpus_bleu(hyps, references), 2),
        "chrF": round(corpus_chrf(hyps, references), 2),
        "seconds": round(secs, 2),
        "seg_per_sec": round(N / secs, 2),
        "len_ratio": round(sum(len(h.split()) for h in hyps)
                           / sum(len(r.split()) for r in references), 3),
    })

df_results = pd.DataFrame(results).sort_values("chrF", ascending=False)
show_table(
    df_results.to_dict("records"),
    title=f"WMT22 en-de, {N} segments, 1 reference",
    best=("BLEU", "chrF", "seg_per_sec"),
    caption="string metrics only - a COMET score would likely reorder this table",
)
```

```
                                   WMT22 en-de, 120 segments, 1 reference                                    
                                                                                                             
 model                     params_m   type                  BLEU    chrF   seconds   seg_per_sec   len_ratio 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 opus-mt-en-de (Marian)          74   bilingual            29.36   58.23    2.8400         42.24      0.9930 
 nllb-200-distilled-600M        615   multilingual (200)   27.22   57.57    8.4700         14.17      0.9850 
 qwen3-1.7b (prompted)        1,720   LLM                  13.16   47.11     25.00        4.8000      0.9870 
                                                                                                             
                     string metrics only - a COMET score would likely reorder this table                     
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
    .add_yaxis("BLEU", [r["BLEU"] for r in results])
    .add_yaxis("chrF", [r["chrF"] for r in results])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"WMT22 en-de ({N} segments, 1 reference)",
            subtitle="RTX 3060, beam=4 for NMT / greedy for the LLM - string metrics only, "
                     "COMET would rank differently",
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
from pyecharts.charts import Scatter

# Quality against throughput on a log axis - the range here is the whole story.
scatter = Scatter()
scatter.add_xaxis([r["seg_per_sec"] for r in results])
for r in results:
    scatter.add_yaxis(
        r["model"], [[r["seg_per_sec"], r["chrF"]]],
        symbol_size=18, label_opts=opts.LabelOpts(is_show=False),
    )
scatter.set_global_opts(
    title_opts=opts.TitleOpts(title="chrF vs throughput",
                              subtitle="a 74M bilingual model is 1-2 orders of magnitude cheaper"),
    xaxis_opts=opts.AxisOpts(name="segments / second", type_="log"),
    yaxis_opts=opts.AxisOpts(name="chrF", type_="value"),
    tooltip_opts=opts.TooltipOpts(trigger="item"),
)
scatter.render_notebook()
```

## 12. Interactive: translate your own text

Edit `MY_TEXTS` and `TARGET` below. This is the cell people run on its own, so it opens with a `require(...)` guard naming what it needs from Setup rather than dying on a bare `NameError`.

It uses NLLB because a single checkpoint reaches all 200 languages - change `TARGET` to any [FLORES-200 code](https://github.com/facebookresearch/flores/blob/main/flores200/README.md) (`fra_Latn`, `arb_Arab`, `hin_Deva`, `zho_Hans`, `swh_Latn`, `yor_Latn`, …) and it just works.

The inputs worth trying are the ones that expose how sentence-level MT actually fails:

- **Negation and scope** - “She did not say that the report was ready” versus “She said that the report was not ready”. Getting these confused is rare but catastrophic, and it is invisible to a reader who does not speak the target language.
- **Placeholders and markup** - put `{count}` or `<b>bold</b>` in the text. NMT models mangle them routinely; this is the number one source of localisation bugs and no quality metric measures it.
- **Ambiguous pronouns across sentences** - “The doctor called the nurse. She was late.” Sentence-level models must guess the gender agreement, and in many target languages there is no neutral option.
- **A low-resource target** - try `yor_Latn` or `amh_Ethi` and compare with `deu_Latn`. The quality difference between the top 30 languages and the rest is the honest picture of what “200 languages” means.
- **Round-tripping** - translate to the target and back to English. Meaning that survives a round trip is usually safe; meaning that does not is worth a human look. It is a crude quality-estimation signal, but it is free.

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


require("device", "dtype", "HF_CACHE", "free_memory", "vram", "translate_seq2seq")

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MY_TEXTS = [
    "She did not say that the report was ready.",
    "The dashboard shows the {count} most recent alerts.",
    "The doctor called the nurse because she was running late.",
    "Please confirm receipt of the invoice by the end of the week.",
]
TARGET = "deu_Latn"     # any FLORES-200 code: fra_Latn, jpn_Jpan, swh_Latn, yor_Latn, ...
ROUND_TRIP = True       # translate back to English as a crude quality-estimation signal

# Re-runnable: this cell frees the model at the end, so guard the load or a second
# shift-enter raises NameError.
if "my_mt" not in globals():
    my_mt_tok = AutoTokenizer.from_pretrained(
        "facebook/nllb-200-distilled-600M", cache_dir=HF_CACHE, src_lang="eng_Latn")
    my_mt = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M", dtype=dtype, cache_dir=HF_CACHE
    ).to(device).eval()

tgt_id = my_mt_tok.convert_tokens_to_ids(TARGET)
outs = translate_seq2seq(my_mt, my_mt_tok, MY_TEXTS, batch_size=4, forced_bos_token_id=tgt_id)

back = None
if ROUND_TRIP:
    my_mt_tok.src_lang = TARGET
    eng_id = my_mt_tok.convert_tokens_to_ids("eng_Latn")
    back = translate_seq2seq(my_mt, my_mt_tok, outs, batch_size=4, forced_bos_token_id=eng_id)
    my_mt_tok.src_lang = "eng_Latn"

for i, (src, out) in enumerate(zip(MY_TEXTS, outs)):
    print(f"en  : {src}")
    print(f"{TARGET[:3]} : {out}")
    if back:
        print(f"back: {back[i]}")
        print(f"      chrF(src, back) {corpus_chrf([back[i]], [src]):.1f} "
              "- low means meaning was lost somewhere in the loop")
    print()

del my_mt, my_mt_tok
free_memory()
vram("final")
```

    [transformers] Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=256) and `max_length`(=200) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

    en  : She did not say that the report was ready.
    deu : Sie sagte nicht, daß der Bericht fertig sei.
    back: She did not say that the report was finished.
          chrF(src, back) 80.8 - low means meaning was lost somewhere in the loop

    en  : The dashboard shows the {count} most recent alerts.
    deu : Das Dashboard zeigt die jüngsten Warnungen an.
    back: The dashboard shows the latest warnings.
          chrF(src, back) 49.1 - low means meaning was lost somewhere in the loop

    en  : The doctor called the nurse because she was running late.
    deu : Der Arzt rief die Krankenschwester an, weil sie zu spät kam.
    back: The doctor called the nurse because she was late.
          chrF(src, back) 81.5 - low means meaning was lost somewhere in the loop

    en  : Please confirm receipt of the invoice by the end of the week.
    deu : Bitte bestätigen Sie den Eingang der Rechnung bis zum Ende der Woche.
    back: Please confirm receipt of the invoice by the end of the week.
          chrF(src, back) 100.0 - low means meaning was lost somewhere in the loop

    VRAM final                   0.01 GB allocated /  0.02 GB reserved

## 13. Common Frameworks

Translation has the most mature evaluation ecosystem of any task in this folder, and that is the thing worth taking from it: learned metrics (COMET) correlate with human judgement well enough that a reference-free version can gate production output. It also has an unusually strong dedicated inference runtime, because bilingual models are small, run at enormous volume, and predate the LLM serving stack by years.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [transformers](https://github.com/huggingface/transformers) | modelling | Marian/OPUS-MT, NLLB-200, M2M-100 and any LLM translator behind `AutoModelForSeq2SeqLM` and the `translation` pipeline | Apache 2.0 | Default. Covers the bilingual, massively multilingual and LLM paths |
| [peft](https://github.com/huggingface/peft) + `Seq2SeqTrainer` | modelling | Domain fine-tuning: 50k in-domain sentence pairs and a couple of GPU-hours | Apache 2.0 | The highest-value move in the task. In MT, in-domain data beats scale more reliably than anywhere else in this repo |
| [sentencepiece](https://github.com/google/sentencepiece) + [tokenizers](https://github.com/huggingface/tokenizers) | data | The subword models these checkpoints ship with, and the language-code handling NLLB requires | Apache 2.0 | Always. A wrong or missing target language token produces fluent output in the wrong language, silently |
| Translation memory / TMS ([Weblate](https://github.com/WeblateOrg/weblate), [OmegaT](https://omegat.org/)) | data | Segment-level reuse, glossaries, and the human review workflow real localisation runs on | GPL-3.0 | Localisation rather than raw translation. Exact TM matches should never reach the model at all |
| [CTranslate2](https://github.com/OpenNMT/CTranslate2) | inference runtime | Marian and NLLB weights at int8 with several times the throughput and a fraction of the memory | MIT | Bulk translation. This is the purpose-built runtime for exactly this model family |
| [vLLM](https://github.com/vllm-project/vllm) / [SGLang](https://github.com/sgl-project/sglang) | inference runtime | Batched serving for the LLM path, where document context and glossary instructions live in the prompt | Apache 2.0 | You chose an LLM translator for its context handling and now have to serve it |
| [Triton Inference Server](https://github.com/triton-inference-server/server) / [BentoML](https://github.com/bentoml/BentoML) | serving | Multiple language-pair models behind one endpoint, with routing and per-model scaling | BSD-3 / Apache 2.0 | A bilingual-model architecture, where “the model” is really thirty models |
| [sacreBLEU](https://github.com/mjpost/sacrebleu) | evaluation | A reproducible BLEU/chrF with a signature string recording tokenisation and settings | Apache 2.0 | Any number you report. The inline implementation in section 4 is for understanding, not publishing |
| [COMET](https://github.com/Unbabel/COMET) (`unbabel-comet`) | evaluation | Learned scoring that actually tracks human judgement, plus **reference-free** quality estimation | Apache 2.0 | Always. CometKiwi flags the segments your model probably got wrong, so you route those to a human and publish the rest |

**The 2026 default stack** is a fine-tuned Marian model per high-volume pair served through CTranslate2, NLLB for the long tail, an LLM where document context and terminology matter, and COMET-Kiwi gating what ships without review. sacreBLEU for regression tests, never for decisions.

The common wrong turn is choosing a model on BLEU. It correlates poorly with human judgement at the quality level modern systems operate at, and it will happily rank a fluent, accurate LLM translation below a stilted one that matches the reference’s word choice. The second is sentence-level thinking: feeding the previous two or three sentences as context is free with an LLM and fixes pronouns, terminology consistency and register - none of which a sentence-level metric will show you.

------------------------------------------------------------------------

## 14. Going Further

- **Fine-tune a Marian model on your domain.** This is the highest-value move in the whole task and it is cheap: `AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")` plus `Seq2SeqTrainer` over 50k in-domain sentence pairs, a couple of GPU-hours, and it will beat every general model on your content. In MT, in-domain data beats scale more reliably than in any other task in this repo.
- **Score with COMET, not BLEU.** `pip install unbabel-comet`, then `Unbabel/wmt22-comet-da` for reference-based scoring or `Unbabel/wmt22-cometkiwi-da` for reference-free quality estimation. The reference-free version is the one that changes your production system: it flags segments the model probably got wrong, so you route those to a human and publish the rest.
- **Use sacreBLEU for any number you report.** `sacrebleu.corpus_bleu(hyps, [refs])` gives a signature string that makes the score reproducible. The inline implementation in section 4 is for understanding, not for publishing.
- **Enforce terminology.** Two workable mechanisms: constrained beam search (`force_words_ids` in `generate`) for NMT, or glossary injection into the prompt for an LLM. Post-hoc find-and-replace is what people actually do first and it breaks on inflected languages.
- **Go document-level.** Feed the previous 2-3 sentences as context. On pronoun resolution, terminology consistency and register this is the largest remaining quality gap, it needs no retraining with an LLM, and no sentence-level metric will show you the improvement - use a contrastive test set like ContraPro.
- **Quality estimation is the production feature.** A system that knows which of its outputs are bad is worth more than one that is slightly better on average. COMET-QE, round-trip agreement, and output-length ratio are all cheap signals; combine them into a routing threshold.
- **Backtranslation for low-resource pairs.** Translate monolingual target-language text into the source language with a weaker model and train on the synthetic pairs. It is the single most effective data-augmentation technique in MT and it is how NLLB covered languages with almost no bitext.
- **Watch the licences.** NLLB and TowerInstruct are CC BY-NC. For commercial use, MADLAD-400 (Apache 2.0), OPUS-MT (Apache 2.0 / CC BY 4.0 depending on the model) and Apache-licensed LLMs are the options.
- **Related notebooks.** `06_Summarization` (the other faithfulness-constrained generation task), `08_Text_Generation` (decoding strategies, beam search, constrained generation), `10_Sentence_Similarity` (cross-lingual embeddings, which are how bitext is mined), `Audio/02_Automatic_Speech_Recognition` (speech translation starts here).

------------------------------------------------------------------------

Back to top
