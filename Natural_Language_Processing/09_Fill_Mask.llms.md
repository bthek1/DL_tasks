# Fill-Mask

The objective that built every encoder in this folder: how masked language modelling works, why `[MASK]` predicts a token and not a word, what pseudo-perplexity measures, how to probe a model for knowledge and for bias, and runnable code comparing four encoders on the same masked sentences.

Author

Benedict Thekkel

## 1. What is Fill-Mask?

Fill-mask predicts the token hidden behind a mask, using **both** sides of the context:

> “The capital of France is `[MASK]`.” -\> `Paris`

This is **masked language modelling** (MLM), and it is far more important than its modest task page suggests: it is the pretraining objective that produced BERT, RoBERTa, DeBERTa, ModernBERT and every encoder used elsewhere in this folder. Section 8 of `07_Feature_Extraction`, the NLI models in `04_Zero_Shot_Classification`, the span readers in `03_Question_Answering` - all of them are MLM-pretrained encoders with a head bolted on.

**Input.** Text containing the model’s mask token. Note that the mask token is model-specific: BERT uses `[MASK]`, RoBERTa uses `<mask>`, and hardcoding either one breaks the other.

**Output.** A probability distribution over the whole vocabulary at each masked position, usually reported as the top-k tokens with scores.

**MLM versus causal LM**, the fork that defines the two halves of modern NLP:

|  | Masked LM (encoder) | Causal LM (decoder) |
|----|----|----|
| Sees | left **and** right context | left only |
| Predicts | masked positions (~15% of tokens) | every next token |
| Learning signal per pass | ~15% of tokens | 100% of tokens |
| Natural use | understanding: classify, tag, retrieve, rank | generation |
| Can generate? | not naturally | yes |

Bidirectionality is the whole point. For deciding whether “bank” means a riverbank or a financial institution, the words *after* it matter as much as those before, and a causal model cannot see them. That is why encoders remain better per parameter at classification, tagging, retrieval and reranking - and why a 150M encoder still beats a 7B decoder on those tasks at 1/50th the cost.

**The one thing everyone gets wrong: `[MASK]` is one *token*, not one *word*.** Subword tokenizers split rare words into pieces, so a single mask cannot produce `unbelievable` if that word is three tokens. Section 8 measures this directly, and it explains most “why is the model giving nonsense?” reports for this task.

**Neighbouring tasks:**

| Task | How it differs | Notebook |
|----|----|----|
| Text generation | Left-to-right, open-ended | `08_Text_Generation` |
| Feature extraction | Uses the encoder’s vectors, not its LM head | `07_Feature_Extraction` |
| Text classification | Adds a supervised head to the same encoder | `00_Text_Classification` |
| Token classification | Labels every token instead of predicting it | `01_Token_Classification` |
| Zero-shot classification | Prompt-based cloze is the close cousin of MLM | `04_Zero_Shot_Classification` |

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| **Pretraining and domain adaptation** | Every ML team with a specialised corpus | Unlabelled domain text -\> a better encoder | The dominant real use; needs GPU-days, no labels |
| Prompt-based few-shot classification (PET) | Low-data classification | “This review was `[MASK]`.” -\> good/bad | Verbaliser choice; beats fine-tuning under ~100 labels |
| Knowledge probing | Interpretability research | Cloze template -\> what the model memorised | Templates confound knowledge with phrasing |
| Bias and fairness auditing | Responsible AI, compliance | Stereotype template -\> token distribution | The standard audit; template-sensitive |
| Spelling and grammar correction | Writing tools, OCR post-processing | Corrupted token -\> candidates | Needs a candidate generator; MLM ranks, does not detect |
| Data augmentation | Any small-data NLP task | Sentence -\> variants with words swapped | Can flip the label; must be filtered |
| Lexical substitution / paraphrase | Search query expansion | Word in context -\> alternatives | Contextual synonyms only, no multi-word phrases |
| Cloze-based educational tools | EdTech | Passage -\> fill-in-the-blank exercises | Distractor quality matters more than the answer |

What the demo hides:

- **Almost nobody deploys fill-mask as a product.** Its overwhelming real use is as a *training objective*. When someone says they use MLM in production, they usually mean they ran continued pretraining on their own corpus - and that is one of the highest-return, least-glamorous moves available in applied NLP.
- **Continued pretraining is the cheap domain win.** Take `ModernBERT-base`, run MLM over a few hundred MB of your clinical notes, legal filings or logs for a few GPU-hours, then fine-tune. It routinely beats fine-tuning the general checkpoint, needs zero labels, and costs less than labelling.
- **Fill-mask makes a model’s priors legible.** It is one of the few interfaces where you see the raw distribution rather than a decision. Ask it to complete “The nurse said `[MASK]` was tired” and you are reading the training corpus’s statistics back, which is what bias audits exploit.
- **A cloze probe measures phrasing as much as knowledge.** The same fact succeeds or fails depending on the template - which is a finding about probing, not about the model, and it is why LAMA-style “BERT knows facts” results were later heavily qualified.

------------------------------------------------------------------------

## 3. How Modern Fill-Mask Works

1.  **Distributional semantics and CBOW (1950s-2015).** “You shall know a word by the company it keeps” (Firth, 1957). word2vec’s continuous-bag-of-words objective is literally fill-mask with a linear model and a tiny window - predict the centre word from its neighbours.
2.  **BERT and the MLM objective (2018).** Mask 15% of tokens and predict them from full bidirectional context. The masking recipe is oddly specific and the reason is worth knowing: of the 15% chosen, **80% become `[MASK]`, 10% become a random token, 10% are left unchanged**. The 10% unchanged and 10% random exist to combat the **pretrain/finetune mismatch** - `[MASK]` never appears in downstream data, so a model that only ever predicts at `[MASK]` positions learns to do nothing elsewhere. BERT also had a next-sentence-prediction loss, later shown to be useless.
3.  **RoBERTa (2019).** Same architecture, better training: drop NSP, mask **dynamically** (a new mask pattern each epoch instead of one fixed pattern), train 10x longer on 10x the data, use bigger batches. It beat BERT by a wide margin with no architectural change - one of the clearest demonstrations in NLP that training recipe beats architecture.
4.  **Better corruption objectives (2019-2021).** SpanBERT masked contiguous spans (better for extractive QA). **ELECTRA** replaced MLM with replaced-token detection: a small generator proposes replacements, the discriminator classifies every token as original or replaced. Because the loss covers **100% of tokens** rather than 15%, it is dramatically more sample-efficient - ELECTRA-small matched much larger BERT models. **DeBERTa** added disentangled attention (content and position as separate vectors) and DeBERTa-v3 combined it with ELECTRA-style pretraining, giving the strongest classical encoder for years.
5.  **The quiet years (2021-2024).** Attention moved to decoders. Encoders stayed on 2019 recipes: 512-token limits, old tokenizers, no code in the pretraining data - even as everyone kept using them for retrieval and classification.
6.  **The encoder refresh (Dec 2024-2026).** **ModernBERT** rebuilt the recipe with everything learned since: rotary position embeddings, alternating local/global attention, GeGLU, unpadded batches, FlashAttention, **8192-token context**, and 2 trillion training tokens including code. Result: DeBERTa-v3-class quality at several times the speed and 16x the context. **NeoBERT**, **EuroBERT** and **mmBERT** (multilingual, 1800+ languages) followed through 2025-2026. MLM did not die; it got a decade of accumulated improvements applied at once.

**Where it stands (mid-2026).** MLM remains the pretraining objective of choice for understanding tasks, and ModernBERT-class models are the default backbone for classification, retrieval and reranking. The interesting current question is not MLM versus CLM but *where each belongs*: encoders in the high-volume hot path, decoders where generation or complex reasoning is required. Fill-mask as an end-user task is niche; fill-mask as the thing that makes those encoders exist is foundational.

------------------------------------------------------------------------

## 4. Evaluation Metrics

**Top-k accuracy on masked tokens** - mask a token, check whether the gold token is in the model’s top-k predictions. Simple, interpretable, and the standard way to compare fill-mask quality directly. Report k, because top-1 and top-5 tell different stories.

**Pseudo-perplexity (PPPL)** - the MLM analogue of perplexity. A masked LM has no `P(sentence)` to factorise, so instead you mask each token in turn, score it from its bidirectional context, and sum:

$$\text{PLL}(W) = \sum_{t=1}^{|W|} \log P(w_t \mid W_{\setminus t}), \qquad \text{PPPL} = \exp\left(-\frac{\text{PLL}(W)}{|W|}\right)$$

**This is not comparable to a causal model’s perplexity, ever.** Each MLM prediction sees the *entire rest of the sentence*, both directions; each causal prediction sees only the left. The MLM’s task is far easier per token, so its number is systematically lower and means something different. Comparing BERT’s pseudo-perplexity to GPT-2’s perplexity is a category error that appears regularly in blog posts.

It also costs one forward pass **per token**, so a 30-token sentence needs 30 passes. Fine for hundreds of sentences, hopeless for a corpus.

**What pseudo-perplexity is genuinely good for:** comparing the *same* model across domains (is my clinical corpus out of distribution for this encoder?), measuring the effect of continued pretraining, and scoring sentence acceptability - it correlates well with human grammaticality judgements, which is what BLiMP-style evaluation uses.

**Bias metrics** are built on exactly this machinery. StereoSet, CrowS-Pairs and WinoGender present minimal pairs that differ only in a demographic term and compare the model’s assigned probability. The measurement is easy; the interpretation is not, and the standard caveats are real: results are highly template-sensitive, “bias” is operationalised very narrowly, and a low score on one benchmark does not mean a model is fair in deployment.

**Pitfalls:**

- **Multi-token targets break naive scoring.** If the gold word is two tokens, a single-mask evaluation cannot ever be right. Either restrict the evaluation to single-token targets (what section 12 does, and it reports how many were dropped) or use multiple masks and score the sequence.
- **Case and subword prefixes matter.** In byte-level BPE the leading space is part of the token, so RoBERTa’s id for `" paris"` differs from its id for `"paris"` (the vocabulary prints the space as a marker glyph). Comparing predictions without normalising for this makes models look worse than they are.
- **Vocabulary differences make top-k accuracy only roughly comparable across models.** A model with a finer tokenizer faces an easier per-token task, the same effect as in `08_Text_Generation` section 4.

The cell below implements top-k accuracy and pseudo-log-likelihood on a toy distribution, with no model, so the arithmetic is visible before any GPU work.

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


def top_k_accuracy(ranked_predictions, gold_tokens, k=5):
    "Fraction of masked positions whose gold token appears in the top-k predictions."
    hits = sum(gold in preds[:k] for preds, gold in zip(ranked_predictions, gold_tokens))
    return hits / len(gold_tokens)


def mean_reciprocal_rank(ranked_predictions, gold_tokens):
    "1/rank of the gold token, averaged. Rewards being close when not exactly right."
    total = 0.0
    for preds, gold in zip(ranked_predictions, gold_tokens):
        if gold in preds:
            total += 1 / (preds.index(gold) + 1)
    return total / len(gold_tokens)


def pseudo_perplexity_from_logprobs(token_logprobs):
    """PPPL from per-token log P(token | rest of sentence).

    Not comparable to a causal model's perplexity: every prediction here saw the whole
    sentence in both directions, so the per-token task is far easier. See section 4.
    """
    return math.exp(-sum(token_logprobs) / len(token_logprobs))


# Toy example: five masked positions, the model's top-5 guesses at each.
predictions = [
    ["paris", "lyon", "france", "marseille", "nice"],
    ["dog", "cat", "man", "boy", "horse"],
    ["hospital", "clinic", "school", "office", "house"],
    ["running", "walking", "sitting", "standing", "sleeping"],
    ["blue", "red", "green", "white", "black"],
]
gold = ["paris", "cat", "hospital", "swimming", "blue"]

show_table([{"gold token": g, "found at rank": (p.index(g) + 1) if g in p else None,
             "top-5 predictions": ", ".join(p)}
            for p, g in zip(predictions, gold)],
           title="Five masked positions and the model's top-5 guesses",
           caption="'swimming' is not in the top 5 at all - top-k and MRR both charge for it")
show_kv({f"top-{k} accuracy": round(top_k_accuracy(predictions, gold, k), 3)
         for k in (1, 3, 5)} | {"MRR": round(mean_reciprocal_rank(predictions, gold), 3)},
        title="Aggregate")

# Pseudo-perplexity for two sentences, one fluent and one not.
fluent = [-0.10, -0.35, -0.05, -1.20, -0.40]        # log P per token, bidirectional
awkward = [-0.10, -4.10, -0.05, -3.80, -0.40]
show_table([{"sentence": name, "per-token log P": str(lp),
             "PPPL": round(pseudo_perplexity_from_logprobs(lp), 2)}
            for name, lp in [("fluent", fluent), ("awkward", awkward)]],
           title="Pseudo-perplexity from per-token bidirectional log-probabilities",
           lower_is_better=("PPPL",), best=("PPPL",))
print("\nPPPL tracks acceptability well, which is what makes it useful for judging")
print("whether a corpus is in distribution for an encoder - not for ranking encoders")
print("against decoders, which is a category error.")
```

```
            Five masked positions and the model's top-5 guesses             
                                                                            
 gold token   found at rank   top-5 predictions                             
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 paris                    1   paris, lyon, france, marseille, nice          
 cat                      2   dog, cat, man, boy, horse                     
 hospital                 1   hospital, clinic, school, office, house       
 swimming              None   running, walking, sitting, standing, sleeping 
 blue                     1   blue, red, green, white, black                
                                                                            
  'swimming' is not in the top 5 at all - top-k and MRR both charge for it  
```

```
                           Aggregate                            
                                                                
 top-1 accuracy                                          0.6000 
 top-3 accuracy                                          0.8000 
 top-5 accuracy                                          0.8000 
 MRR                                                     0.7000 
                                                                
```

```
    Pseudo-perplexity from per-token bidirectional log-probabilities    
                                                                        
 sentence       per-token log P                                    PPPL 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 fluent         [-0.1, -0.35, -0.05, -1.2, -0.4]                 1.5200 
 awkward        [-0.1, -4.1, -0.05, -3.8, -0.4]                  5.4200 
                                                                        
```

    PPPL tracks acceptability well, which is what makes it useful for judging
    whether a corpus is in distribution for an encoder - not for ranking encoders
    against decoders, which is a category error.

## 5. Datasets

Fill-mask needs no labelled data - any text is training data, which is the objective’s entire appeal. The datasets that matter split into pretraining corpora and diagnostic probes.

| Dataset | Role | Contents | Size | License | Notes |
|----|----|----|----|----|----|
| [wikitext-103 / -2](https://huggingface.co/datasets/Salesforce/wikitext) | pretraining / eval | Clean Wikipedia articles | 100M / 2M tokens | CC BY-SA 3.0 | The standard small benchmark; used below |
| [BookCorpus + Wikipedia](https://huggingface.co/datasets/bookcorpus/bookcorpus) | pretraining | BERT’s original 3.3B-word mix | 3.3B words | unclear provenance | Historical; licence is genuinely murky |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | pretraining | Education-filtered web text | 1.3T tokens | ODC-By | What modern encoders train on |
| [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) | pretraining | Source code | 900B tokens | mixed | ModernBERT’s code ability comes from here |
| [C4](https://huggingface.co/datasets/allenai/c4) | pretraining | Cleaned Common Crawl | 750 GB | ODC-By | T5’s corpus; still widely used |
| [LAMA](https://huggingface.co/datasets/facebook/lama) | probe | Cloze templates over facts (Google-RE, T-REx, ConceptNet) | 50k+ | CC BY-NC 4.0 | “What does the LM know?” |
| [BLiMP](https://huggingface.co/datasets/nyu-mll/blimp) | probe | 67 minimal-pair sets for grammar | 67k pairs | CC BY 4.0 | Acceptability via pseudo-log-likelihood |
| [StereoSet](https://huggingface.co/datasets/McGill-NLP/stereoset) | bias probe | Stereotype/anti-stereotype/unrelated triples | 17k | CC BY-SA 4.0 | Gender, race, religion, profession |
| [CrowS-Pairs](https://huggingface.co/datasets/nyu-mll/crows_pairs) | bias probe | Minimal pairs differing in a demographic term | 1.5k | CC BY-SA 4.0 | 9 bias categories |
| [WinoGender / WinoBias](https://huggingface.co/datasets/oskarvanderwal/winogender) | bias probe | Coreference with occupational gender | 720 / 3.2k | MIT / CC | Pronoun resolution bias |
| [GLUE / SuperGLUE](https://huggingface.co/datasets/nyu-mll/glue) | downstream eval | 9+ sentence-level tasks | varies | mixed | What MLM pretraining is judged by |

This notebook uses **wikitext-2** for masked-token accuracy and pseudo-perplexity, and **hand-written templates** for the knowledge and bias probes. Hand-written templates are used deliberately rather than loading StereoSet: the templates are visible in the cell, so you can see exactly what is being measured, edit them, and watch the conclusion change - which is the honest lesson about template-based probing.

Downloads land in `DL_tasks/datasets/` via `cache_dir` (gitignored).

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

Encoders are judged by what they enable downstream, so the references are [GLUE](https://gluebenchmark.com/leaderboard) and [SuperGLUE](https://super.gluebenchmark.com/) for classification and [MTEB](https://huggingface.co/spaces/mteb/leaderboard) for retrieval - not by fill-mask quality itself.

| Model | Params | License | Context | Objective | Mask token | Best for |
|----|----|----|----|----|----|----|
| [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) | 110M | Apache 2.0 | 512 | MLM + NSP | `[MASK]` | the historical reference; used below |
| [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) | 66M | Apache 2.0 | 512 | distilled MLM | `[MASK]` | cheapest usable encoder; used below |
| [roberta-base](https://huggingface.co/FacebookAI/roberta-base) | 125M | MIT | 512 | MLM, dynamic masking | `<mask>` | better-trained BERT; used below |
| [roberta-large](https://huggingface.co/FacebookAI/roberta-large) | 355M | MIT | 512 | MLM | `<mask>` | strong classical encoder |
| [deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) | 184M | MIT | 512 | ELECTRA-style RTD | `[MASK]` | best pre-2024 accuracy; **no MLM head** |
| [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) | 149M | Apache 2.0 | **8192** | MLM | `[MASK]` | the 2026 default backbone; used below |
| [ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) | 395M | Apache 2.0 | 8192 | MLM | `[MASK]` | best open encoder accuracy |
| [NeoBERT](https://huggingface.co/chandar-lab/NeoBERT) | 250M | MIT | 4096 | MLM | `[MASK]` | strong 2025 alternative |
| [xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) | 278M | MIT | 512 | MLM | `<mask>` | 100 languages |
| [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) | 307M | MIT | 8192 | MLM | `[MASK]` | 1800+ languages, modern recipe |
| [EuroBERT](https://huggingface.co/EuroBERT/EuroBERT-210m) | 210M-2.1B | Apache 2.0 | 8192 | MLM | `[MASK]` | European languages, modern recipe |

**How to choose.** For a new project in 2026, start at `ModernBERT-base`: it is faster than BERT-base, better than DeBERTa-v3-base, and its 8192-token window removes the chunking constraint that shaped a decade of encoder pipelines. Take `xlm-roberta` or `mmBERT` for multilingual work. Take `distilbert` only when CPU latency is the binding constraint.

**A trap worth naming: DeBERTa-v3 has no usable MLM head.** It was pretrained with ELECTRA-style replaced-token detection, not masked-token prediction, so `pipeline("fill-mask", model="microsoft/deberta-v3-base")` either fails or returns garbage. It is an excellent encoder to *fine-tune* and a bad one to fill masks with - a clean illustration that the pretraining objective determines which heads exist.

------------------------------------------------------------------------

## 7. Setup

Everything loads through Hugging Face `transformers` - no vendor packages. Package roles:

- `transformers` + `torch` - the four encoders and their MLM heads
- `accelerate` - `device_map` placement
- `datasets` - wikitext-2 for the masked-token evaluation
- `pandas` + `pyecharts` - the benchmark table and charts
- `rich` - the result tables. It renders to HTML inside Jupyter, so the tables survive into the published docs; `show_table` / `show_kv` / `rule` are defined in the first code cell of section 4.

Four details that decide whether fill-mask code works at all:

- **Never hardcode the mask token.** Use `tok.mask_token` (`[MASK]` for BERT/ModernBERT, `<mask>` for RoBERTa) and `tok.mask_token_id`. A template with a literal `[MASK]` fed to RoBERTa is tokenised as ordinary text, so the model sees no mask and the pipeline raises.
- **One mask predicts one token.** For BERT-family tokenizers a “word” is often several tokens. Predicting a multi-token word needs multiple consecutive masks, and even then the pieces are predicted semi-independently, which is why the output is frequently incoherent.
- **The leading space is part of the token in byte-level BPE.** RoBERTa gives `" Paris"` and `"Paris"` different ids (the vocabulary renders the leading space as a marker glyph). Building candidate lists without the leading space silently compares against tokens the model will never predict mid-sentence.
- **`AutoModelForMaskedLM`, not `AutoModel`.** The base class drops the LM head, and loading a checkpoint without one gives randomly initialised output weights with only a warning.

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

MODELS = [
    # name, checkpoint, params (M)
    ("distilbert-base", "distilbert/distilbert-base-uncased", 66),
    ("bert-base", "google-bert/bert-base-uncased", 110),
    ("roberta-base", "FacebookAI/roberta-base", 125),
    ("modernbert-base", "answerdotai/ModernBERT-base", 149),
]
```

    NVIDIA GeForce RTX 3060
    device: cuda:0 | dtype: torch.float16

``` python
from datasets import load_dataset

# wikitext-2 test: clean Wikipedia prose. Sentences are used for masked-token accuracy
# and pseudo-perplexity in section 12.
wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test",
                        cache_dir=HF_CACHE)

lines = [t.strip() for t in wikitext["text"]
         if 80 <= len(t.strip()) <= 260 and not t.strip().startswith("=")]
SENTENCES = lines[:120]

print(wikitext)
print(f"\n{len(SENTENCES)} evaluation sentences kept\n")
for s in SENTENCES[:3]:
    print(" ", s[:140])
```

    Dataset({
        features: ['text'],
        num_rows: 4358
    })

    120 evaluation sentences kept

      Brooding on what I have lived through , if even I know such suffering , the common man must surely be rattled by the winds .
      I am about to scream madly in the office / Especially when they bring more papers to pile higher on my desk .
      Hung summarises his life by concluding that , " He appeared to be a filial son , an affectionate father , a generous brother , a faithful hu

## 8. The mechanism, and the token-versus-word trap

Start with the raw model rather than the pipeline, because the pipeline hides the two things worth understanding: the output is a distribution over the **entire vocabulary** at the masked position, and that position holds exactly **one token**.

The cell does three things:

1.  Fills a mask and shows the top-k with probabilities, so the shape of the distribution is visible - sometimes sharply peaked, sometimes nearly flat, and the flat case is the model telling you it does not know.
2.  Demonstrates the **token, not word** problem directly: a common word is one token and fills fine; a rare or morphologically complex word is several tokens and *cannot* be produced by a single mask no matter how well the model knows it.
3.  Shows what multiple consecutive masks do - each position is predicted from the same bidirectional context, roughly independently, so the pieces often do not compose into a real word. This is the structural reason MLM is not a generation method.

The fourth part is a knowledge probe (LAMA-style). It works impressively often, and the caveat is essential: the model is not consulting a knowledge base, it is completing a pattern that occurred in its training corpus. Change the template wording and answers change - which is why the “language models as knowledge bases” claim was substantially walked back once template sensitivity was measured.

------------------------------------------------------------------------

``` python
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", cache_dir=HF_CACHE)
mlm = AutoModelForMaskedLM.from_pretrained(
    "google-bert/bert-base-uncased", cache_dir=HF_CACHE).to(device).eval()
vram("bert-base loaded")

M = tok.mask_token          # never hardcode "[MASK]" - roberta uses "<mask>"
print(f"mask token {M!r} (id {tok.mask_token_id}), vocabulary {len(tok):,}\n")


@torch.inference_mode()
def fill(model, tokenizer, text, k=5):
    "Top-k predictions at every mask position, with probabilities."
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    logits = model(**enc).logits[0]
    positions = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero().flatten()
    out = []
    for pos in positions.tolist():
        probs = logits[pos].float().softmax(-1)
        top = probs.topk(k)
        out.append([(tokenizer.decode([i]).strip(), round(p, 4))
                    for p, i in zip(top.values.tolist(), top.indices.tolist())])
    return out


# 1. The distribution: peaked when the model knows, flat when it does not.
_probes = [f"The capital of France is {M}.",
           f"She poured the coffee into the {M}.",
           f"The {M} is on the table."]
show_table([{"text": t.replace(M, "___"), "top-1 mass": preds[0][1],
             **{f"#{i}": f"{w} ({p:.3f})" for i, (w, p) in enumerate(preds, start=1)}}
            for t, preds in ((t, fill(mlm, tok, t)[0]) for t in _probes)],
           title="The distribution is peaked when the model knows and flat when it does not",
           lower_is_better=(), best=("top-1 mass",),
           caption="a low top-1 mass is the model telling you the context underdetermines "
                   "the answer - useful information a single argmax throws away")

# 2. One mask is one TOKEN. Check before you evaluate.
show_table([{"word": w, "tokens": len(tok.tokenize(w)), "pieces": str(tok.tokenize(w)),
             "fillable with one mask": len(tok.tokenize(w)) == 1}
            for w in ["paris", "hospital", "unbelievable", "acetaminophen", "kubernetes"]],
           title="One mask is one TOKEN, not one word",
           caption="a multi-token word cannot be produced by a single mask no matter how "
                   "well the model knows it - check before you evaluate")

# 3. Multiple masks: each position predicted from the same context, near-independently.
multi = f"The medication {M} {M} was prescribed."
print(f"\n{multi}")
for i, preds in enumerate(fill(mlm, tok, multi, k=4)):
    print(f"  mask {i}: " + "  ".join(f"{w} {p:.3f}" for w, p in preds))
print("  the pieces rarely compose into a real word - MLM is not a generation method\n")

# 4. Knowledge probing (LAMA-style), and its fragility.
FACT_TEMPLATES = [
    "The capital of {} is [M].",
    "{} has the capital city [M].",
    "[M] is the capital of {}.",
]
show_table([{"country": country,
             **{t.replace("[M]", "___"):
                (lambda p: f"{p[0][0]} ({p[0][1]:.2f})")(
                    fill(mlm, tok, t.replace("[M]", M).format(country), k=1)[0])
                for t in FACT_TEMPLATES}}
            for country in ["France", "Japan", "Brazil"]],
           title="One fact, three phrasings",
           caption="where the columns disagree, the model is completing a pattern rather "
                   "than consulting knowledge - the core caveat of cloze-based probing")

del mlm, tok
free_memory()
vram("after bert-base")
```

    [transformers] BertForMaskedLM LOAD REPORT from: google-bert/bert-base-uncased
    Key                         | Status     |  | 
    ----------------------------+------------+--+-
    bert.pooler.dense.weight    | UNEXPECTED |  | 
    bert.pooler.dense.bias      | UNEXPECTED |  | 
    cls.seq_relationship.bias   | UNEXPECTED |  | 
    cls.seq_relationship.weight | UNEXPECTED |  | 

    Notes:
    - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

    VRAM bert-base loaded        0.44 GB allocated /  0.49 GB reserved
    mask token '[MASK]' (id 103), vocabulary 30,522

```
                   The distribution is peaked when the model knows and flat when it does not                    
                                                                                                                
 text               top-1 mass   #1               #2              #3             #4               #5            
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 The capital of         0.4168   paris (0.417)    lille (0.071)   lyon (0.063)   marseille        tours (0.030) 
 France is ___.                                                                  (0.044)                        
 She poured the         0.3667   cup (0.367)      mug (0.305)     pot (0.067)    coffee (0.033)   glass (0.027) 
 coffee into the                                                                                                
 ___.                                                                                                           
 The ___ is on          0.0852   coffee (0.085)   phone (0.056)   food (0.050)   money (0.048)    book (0.029)  
 the table.                                                                                                     
                                                                                                                
 a low top-1 mass is the model telling you the context underdetermines the answer - useful information a single 
                                               argmax throws away                                               
```

```
                             One mask is one TOKEN, not one word                             
                                                                                             
 word            tokens   pieces                                      fillable with one mask 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 paris                1   ['paris']                                   True                   
 hospital             1   ['hospital']                                True                   
 unbelievable         1   ['unbelievable']                            True                   
 acetaminophen        5   ['ace', '##tam', '##ino', '##ph', '##en']   False                  
 kubernetes           4   ['ku', '##ber', '##net', '##es']            False                  
                                                                                             
a multi-token word cannot be produced by a single mask no matter how well the model knows it 
                                 - check before you evaluate                                 
```

    The medication [MASK] [MASK] was prescribed.
      mask 0: the 0.214  that 0.161  for 0.040  he 0.037
      mask 1: he 0.054  she 0.020  i 0.019  it 0.015
      the pieces rarely compose into a real word - MLM is not a generation method

```
                                   One fact, three phrasings                                    
                                                                                                
 country   The capital of {} is ___.   {} has the capital city ___.   ___ is the capital of {}. 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 France    paris (0.42)                paris (0.30)                   paris (0.62)              
 Japan     tokyo (0.56)                tokyo (0.44)                   tokyo (0.61)              
 Brazil    salvador (0.16)             victoria (0.10)                it (0.59)                 
                                                                                                
where the columns disagree, the model is completing a pattern rather than consulting knowledge -
                             the core caveat of cloze-based probing                             
```

    VRAM after bert-base         0.01 GB allocated /  0.02 GB reserved

## 9. Different tokenizers, different masks

Swapping to RoBERTa changes three things at once, and each is a bug waiting to happen in code that assumed BERT.

**The mask token is `<mask>`, not `[MASK]`.** A template containing a literal `[MASK]` is tokenised as ordinary text - the model sees no mask at all, and the pipeline raises `No mask_token ... found`.

**Byte-level BPE puts the leading space inside the token.** RoBERTa gives `" Paris"` one id and `"Paris"` (at the start of a string) another; the vocabulary renders that leading space as a marker glyph, which the cell below prints. These are distinct entries with different embeddings. Any candidate list built without the leading space is comparing against tokens the model essentially never predicts mid-sentence, which silently deflates the score.

**Casing.** `bert-base-uncased` lowercases everything, so it can never output `Paris`; RoBERTa and ModernBERT are cased and can. Comparing them without normalising case measures the tokenizer, not the model.

The cell runs the same probes on all four models with each model’s own mask token, so the comparison is fair. Notice that ModernBERT - trained in 2024 on 2 trillion tokens including code - is not uniformly better at this particular task. Fill-mask quality on short generic templates is not what modern encoders were optimised for; their advantage shows up in downstream fine-tuning and in the 8192-token context, which section 12 and the landscape table cover.

------------------------------------------------------------------------

``` python
PROBES = [
    "The capital of France is {mask}.",
    "She works at the hospital as a {mask}.",
    "The patient was given a {mask} to reduce the fever.",
    "def compute_total(items): return {mask}(items)",   # code - ModernBERT saw code
]

loaded = {}
for name, model_id, params_m in MODELS:
    t = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE)
    m = AutoModelForMaskedLM.from_pretrained(model_id, cache_dir=HF_CACHE).to(device).eval()

    show_table([{"probe": probe.format(mask="___"),
                 **{f"#{i}": f"{w} ({p:.2f})" for i, (w, p) in
                    enumerate(fill(m, t, probe.format(mask=t.mask_token), k=4)[0], start=1)}}
                for probe in PROBES],
               title=f"{name} ({params_m}M) - mask {t.mask_token!r}, vocab {len(t):,}, "
                     f"{'uncased' if 'uncased' in model_id else 'cased'}")

    del m, t                      # one model live at a time
    free_memory()

# The leading-space trap, shown concretely on RoBERTa's byte-level BPE.
rt = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", cache_dir=HF_CACHE)
show_table([{"input": repr(s), "pieces": str(rt.tokenize(s)),
             "ids": str(rt.encode(s, add_special_tokens=False))}
            for s in ["Paris", " Paris", "paris", " paris"]],
           title="RoBERTa byte-level BPE - the leading space is part of the token",
           caption="building a candidate list as ['paris', 'lyon'] instead of "
                   "[' Paris', ' Lyon'] compares against tokens the model almost never "
                   "predicts mid-sentence")

del rt
free_memory()
vram("after probes")
```

```
                          distilbert-base (66M) - mask '[MASK]', vocab 30,522, uncased                          
                                                                                                                
 probe                            #1                 #2                    #3                 #4                
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 The capital of France is ___.    marseille (0.14)   nantes (0.09)         toulouse (0.09)    paris (0.09)      
 She works at the hospital as a   nurse (0.60)       receptionist (0.05)   volunteer (0.04)   consultant (0.03) 
 ___.                                                                                                           
 The patient was given a ___ to   vaccine (0.25)     medication (0.17)     dose (0.05)        remedy (0.04)     
 reduce the fever.                                                                                              
 def compute_total(items):        value (0.23)       item (0.04)           return (0.03)      total (0.02)      
 return ___(items)                                                                                              
                                                                                                                
```

    [transformers] BertForMaskedLM LOAD REPORT from: google-bert/bert-base-uncased
    Key                         | Status     |  | 
    ----------------------------+------------+--+-
    bert.pooler.dense.weight    | UNEXPECTED |  | 
    bert.pooler.dense.bias      | UNEXPECTED |  | 
    cls.seq_relationship.bias   | UNEXPECTED |  | 
    cls.seq_relationship.weight | UNEXPECTED |  | 

    Notes:
    - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

```
                            bert-base (110M) - mask '[MASK]', vocab 30,522, uncased                             
                                                                                                                
 probe                            #1                  #2                    #3              #4                  
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 The capital of France is ___.    paris (0.42)        lille (0.07)          lyon (0.06)     marseille (0.04)    
 She works at the hospital as a   nurse (0.87)        receptionist (0.03)   doctor (0.02)   psychologist (0.01) 
 ___.                                                                                                           
 The patient was given a ___ to   medication (0.07)   cream (0.06)          pill (0.05)     drug (0.05)         
 reduce the fever.                                                                                              
 def compute_total(items):        total (0.57)        value (0.02)          set (0.02)      sum (0.01)          
 return ___(items)                                                                                              
                                                                                                                
```

```
                            roberta-base (125M) - mask '<mask>', vocab 50,265, cased                            
                                                                                                                
 probe                                #1                  #2                 #3                  #4             
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 The capital of France is ___.        Paris (0.90)        Lyon (0.08)        Nice (0.00)         Nancy (0.00)   
 She works at the hospital as a       nurse (0.60)        secretary (0.14)   translator (0.03)   manager (0.02) 
 ___.                                                                                                           
 The patient was given a ___ to       medication (0.20)   medicine (0.20)    pill (0.12)         drug (0.11)    
 reduce the fever.                                                                                              
 def compute_total(items): return     total (0.65)        array (0.02)       random (0.02)       list (0.02)    
 ___(items)                                                                                                     
                                                                                                                
```

    [transformers] Ignoring clean_up_tokenization_spaces=True for BPE tokenizer TokenizersBackend. The clean_up_tokenization post-processing step is designed for WordPiece tokenizers and is destructive for BPE (it strips spaces before punctuation). Set clean_up_tokenization_spaces=False to suppress this warning, or set clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output=True to force cleanup anyway.

```
                          modernbert-base (149M) - mask '[MASK]', vocab 50,368, cased                           
                                                                                                                
 probe                                #1                #2                  #3                 #4               
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 The capital of France is ___.        Paris (0.92)      Lyon (0.04)         Nancy (0.02)       Nice (0.01)      
 She works at the hospital as a       nurse (0.77)      doctor (0.02)       secretary (0.02)   physician (0.01) 
 ___.                                                                                                           
 The patient was given a ___ to       medicine (0.20)   medication (0.10)   drug (0.09)        vaccine (0.05)   
 reduce the fever.                                                                                              
 def compute_total(items): return     sum (0.65)        len (0.31)          max (0.01)         total (0.01)     
 ___(items)                                                                                                     
                                                                                                                
```

```
    RoBERTa byte-level BPE - the leading space is part of the token     
                                                                        
 input              pieces                         ids                  
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 'Paris'            ['Paris']                      [32826]              
 ' Paris'           ['ĠParis']                     [2201]               
 'paris'            ['par', 'is']                  [5489, 354]          
 ' paris'           ['Ġpar', 'is']                 [2242, 354]          
                                                                        
 building a candidate list as ['paris', 'lyon'] instead of [' Paris', ' 
     Lyon'] compares against tokens the model almost never predicts     
                              mid-sentence                              
```

    VRAM after probes            0.01 GB allocated /  0.02 GB reserved

## 10. Reading the training corpus back: bias probing

Fill-mask is one of the few interfaces where a model’s raw priors are directly legible, which is why it became the standard instrument for bias auditing. The method is simple: write a template that differs only in a demographic term, and compare the probability the model assigns to the same completion.

The cell below does two versions:

1.  **Occupation to pronoun** - “The `<occupation>` said that `[MASK]` was tired” and compare `P(he)` against `P(she)`. This mirrors WinoGender/WinoBias. The result is a legible number, and for most models trained on web text it is not close to balanced.
2.  **Group to attribute** - a stereotype template with the demographic term varied, comparing the same attribute token. This mirrors StereoSet and CrowS-Pairs.

**Read this carefully, in both directions.** The measurement is real: these associations are in the training corpora, and they propagate into any classifier fine-tuned from the encoder. This is a genuine and well-documented harm pathway, not a hypothetical one.

The measurement is also narrow, and overclaiming it is its own failure mode:

- Results are **highly template-sensitive**. Change the verb or the sentence frame and the ratio moves. A single template proves very little.
- Statistical association is not the same thing as discriminatory behaviour in a deployed system. A model with skewed pronoun priors may or may not produce unfair decisions downstream - that has to be measured on the downstream task, on your data.
- A balanced score on this probe does **not** certify a model as fair. It certifies that these particular templates came out balanced.

The right use is diagnostic: run it on a candidate backbone before adopting it, run it again on your fine-tuned model, and measure fairness properly on the task you actually deploy.

------------------------------------------------------------------------

``` python
OCCUPATIONS = ["nurse", "doctor", "engineer", "teacher", "secretary", "programmer",
               "cleaner", "scientist"]
FRAME = "The {occ} finished the shift and said that {mask} was exhausted."


@torch.inference_mode()
def prob_of_tokens(model, tokenizer, text, candidates):
    "P(token) at the single mask position for each candidate word."
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    pos = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero().flatten()[0]
    probs = model(**enc).logits[0, pos].float().softmax(-1)
    out = {}
    for word in candidates:
        # Try the raw and leading-space forms; byte-level BPE makes them different ids
        # (see section 9). Keep whichever is a single token, take the higher probability.
        best = 0.0
        for form in (word, " " + word):
            ids = tokenizer.encode(form, add_special_tokens=False)
            if len(ids) == 1:
                best = max(best, probs[ids[0]].item())
        out[word] = best
    return out


bias_rows, bias_curves = [], {}
for name, model_id, _ in MODELS:
    t = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE)
    m = AutoModelForMaskedLM.from_pretrained(model_id, cache_dir=HF_CACHE).to(device).eval()

    curve = {}
    for occ in OCCUPATIONS:
        p = prob_of_tokens(m, t, FRAME.format(occ=occ, mask=t.mask_token), ["he", "she"])
        total = p["he"] + p["she"]
        if total >= 1e-6:
            curve[occ] = round(p["she"] / total, 3)
    bias_curves[name] = curve
    if curve:
        ratios = list(curve.values())
        bias_rows.append({"model": name,
                          "mean_abs_skew": round(
                              sum(abs(r - 0.5) for r in ratios) / len(ratios), 3),
                          "min_p_she": round(min(ratios), 3),
                          "max_p_she": round(max(ratios), 3)})

    del m, t
    free_memory()

show_table([{"model": name, **curve} for name, curve in bias_curves.items()],
           title="P(she | he or she) at the pronoun mask, by occupation",
           caption=f"frame: {FRAME.format(occ='<occupation>', mask='___')} "
                   "- 0.500 would be balanced")
show_table(bias_rows, title="Summary", best=("mean_abs_skew",),
           lower_is_better=("mean_abs_skew",),
           caption="mean |deviation from 0.5| across occupations - 0.0 would be "
                   "perfectly balanced")

print("This measures an association present in the pretraining corpus. It is a real")
print("harm pathway into any downstream classifier - and it is one template. Change the")
print("verb and the numbers move, so treat it as a diagnostic, not a fairness certificate.")
vram("after bias probe")
```

    [transformers] BertForMaskedLM LOAD REPORT from: google-bert/bert-base-uncased
    Key                         | Status     |  | 
    ----------------------------+------------+--+-
    bert.pooler.dense.weight    | UNEXPECTED |  | 
    bert.pooler.dense.bias      | UNEXPECTED |  | 
    cls.seq_relationship.bias   | UNEXPECTED |  | 
    cls.seq_relationship.weight | UNEXPECTED |  | 

    Notes:
    - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

```
                         P(she | he or she) at the pronoun mask, by occupation                         
                                                                                                       
 model              nurse   doctor   engineer   teacher   secretary   programmer   cleaner   scientist 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 distilbert-base   0.8350   0.4740     0.2210    0.5700      0.4870       0.3440    0.5510      0.4450 
 bert-base         0.9370   0.2460     0.0130    0.3860      0.4450       0.0650    0.3280      0.1200 
 roberta-base      0.9400   0.2130     0.1100    0.5800      0.8950       0.0800    0.3870      0.2190 
 modernbert-base   0.8040   0.1370     0.0720    0.5480      0.3240       0.1230    0.5260      0.1980 
                                                                                                       
 frame: The <occupation> finished the shift and said that ___ was exhausted. - 0.500 would be balanced 
```

```
                                Summary                                 
                                                                        
 model                       mean_abs_skew      min_p_she     max_p_she 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 distilbert-base                    0.1230         0.2210        0.8350 
 bert-base                          0.2920         0.0130        0.9370 
 roberta-base                       0.3010         0.0800        0.9400 
 modernbert-base                    0.2530         0.0720        0.8040 
                                                                        
 mean |deviation from 0.5| across occupations - 0.0 would be perfectly  
                                balanced                                
```

    This measures an association present in the pretraining corpus. It is a real
    harm pathway into any downstream classifier - and it is one template. Change the
    verb and the numbers move, so treat it as a diagnostic, not a fairness certificate.
    VRAM after bias probe        0.01 GB allocated /  0.02 GB reserved

``` python
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org serves an expired TLS certificate from most nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"

from pyecharts import options as opts
from pyecharts.charts import Bar

# The same numbers as a chart. The 0.5 line is what balance would look like; distance
# from it is the association, and the ordering of occupations is the shape to notice.
bar = Bar().add_xaxis(list(OCCUPATIONS))
for name, curve in bias_curves.items():
    bar.add_yaxis(name, [curve.get(occ) for occ in OCCUPATIONS],
                  label_opts=opts.LabelOpts(is_show=False))
bar.set_global_opts(
    title_opts=opts.TitleOpts(
        title="P(she | he or she) at the pronoun mask, by occupation",
        subtitle="0.5 is balanced - this is one template, and the numbers move when the "
                 "verb changes, so read it as a diagnostic and not a fairness certificate",
    ),
    xaxis_opts=opts.AxisOpts(name="occupation",
                             axislabel_opts=opts.LabelOpts(rotate=25, font_size=9)),
    yaxis_opts=opts.AxisOpts(name="P(she)", min_=0, max_=1),
    tooltip_opts=opts.TooltipOpts(trigger="axis"),
    legend_opts=opts.LegendOpts(pos_top="14%"),
)
bar.set_series_opts(
    markline_opts=opts.MarkLineOpts(
        data=[opts.MarkLineItem(y=0.5, name="balanced")]))
bar.render_notebook()
```

## 11. Pseudo-perplexity

The MLM analogue of perplexity, and the cost is the first thing to notice: **one forward pass per token**. A 30-token sentence needs 30 passes, so this is a diagnostic for tens or hundreds of sentences, never for a corpus.

The implementation below batches all the single-token maskings of one sentence into a single forward pass, which is the standard trick and makes it tractable - one batched pass per sentence instead of one per token.

Two things to watch:

- **The acceptability signal is strong.** Grammatical sentences score far lower than scrambled or ungrammatical ones. This is what BLiMP-style evaluation uses, and it is a genuinely useful application: it needs no labels and it correlates well with human judgements.
- **The out-of-domain signal is also strong**, and this is the practically valuable one. Score a hundred sentences from your own corpus against a candidate encoder. A high pseudo-perplexity means the text is out of distribution, which is precisely the case where continued MLM pretraining on your corpus pays for itself.

And once more, because it is the most common misuse: these numbers are **not** comparable to a causal model’s perplexity from `08_Text_Generation`, and they are only roughly comparable across models with different tokenizers.

------------------------------------------------------------------------

``` python
@torch.inference_mode()
def pseudo_perplexity(model, tokenizer, text, max_tokens=64):
    """Mask each token in turn, score it from bidirectional context, exponentiate.

    All maskings of one sentence go through as a single batch: n copies of the sentence
    with a different position masked in each. One batched pass per sentence, not n passes.
    """
    ids = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=max_tokens).input_ids[0].to(model.device)
    special = set(tokenizer.all_special_ids)
    positions = [i for i, t in enumerate(ids.tolist()) if t not in special]
    if not positions:
        return float("nan"), 0

    batch = ids.repeat(len(positions), 1)
    rows = torch.arange(len(positions), device=ids.device)
    cols = torch.tensor(positions, device=ids.device)
    gold = batch[rows, cols].clone()
    batch[rows, cols] = tokenizer.mask_token_id

    logits = model(batch, attention_mask=torch.ones_like(batch)).logits[rows, cols]
    logprobs = logits.float().log_softmax(-1)[rows, gold]
    return math.exp(-logprobs.mean().item()), len(positions)


ACCEPTABILITY = [
    ("grammatical    ", "The committee approved the new budget after a long debate."),
    ("scrambled      ", "budget the approved committee long a after debate new the."),
    ("agreement error", "The committee approve the new budget after a long debate."),
    ("nonsense words ", "The flarn approved the glimt budget after a long trebble."),
    ("clinical domain", "Patient presented with acute dyspnea and bilateral crackles on auscultation."),
    ("code           ", "for i in range(len(items)): total += items[i].price"),
]

ppl_rows = {}
for name, model_id, _ in MODELS:
    t = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE)
    m = AutoModelForMaskedLM.from_pretrained(model_id, cache_dir=HF_CACHE).to(device).eval()

    row, _detail = {}, []
    for label, sent in ACCEPTABILITY:
        ppl, n_tok = pseudo_perplexity(m, t, sent)
        row[label.strip()] = round(ppl, 2)
        _detail.append({"sentence type": label.strip(), "tokens": n_tok,
                        "PPPL": round(ppl, 2), "text": sent[:70]})
    show_table(_detail, title=name, best=("PPPL",), lower_is_better=("PPPL",))

    # Corpus-level number over the wikitext sentences, for the benchmark table.
    t0 = time.perf_counter()
    corpus = [pseudo_perplexity(m, t, s)[0] for s in SENTENCES[:40]]
    secs = time.perf_counter() - t0
    row["wikitext_pppl"] = round(sum(corpus) / len(corpus), 2)
    row["seconds_40_sents"] = round(secs, 2)
    ppl_rows[name] = row
    print(f"  wikitext-2 mean PPPL over 40 sentences: {row['wikitext_pppl']:.2f} "
          f"({secs:.1f}s)\n")

    del m, t
    free_memory()

print("Grammatical << ungrammatical is the acceptability signal (no labels needed).")
print("High PPPL on the clinical or code line means that text is out of distribution -")
print("which is exactly when continued MLM pretraining on your own corpus pays off.")
vram("after pseudo-perplexity")
```

```
                                                distilbert-base                                                
                                                                                                               
 sentence type     tokens        PPPL   text                                                                   
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 grammatical           11      6.6500   The committee approved the new budget after a long debate.             
 scrambled             11   12,197.74   budget the approved committee long a after debate new the.             
 agreement error       11       15.45   The committee approve the new budget after a long debate.              
 nonsense words        16      154.44   The flarn approved the glimt budget after a long trebble.              
 clinical domain       18       15.46   Patient presented with acute dyspnea and bilateral crackles on auscult 
 code                  20      227.12   for i in range(len(items)): total += items.price                       
                                                                                                               
```

      wikitext-2 mean PPPL over 40 sentences: 12.23 (1.4s)

    [transformers] BertForMaskedLM LOAD REPORT from: google-bert/bert-base-uncased
    Key                         | Status     |  | 
    ----------------------------+------------+--+-
    bert.pooler.dense.weight    | UNEXPECTED |  | 
    bert.pooler.dense.bias      | UNEXPECTED |  | 
    cls.seq_relationship.bias   | UNEXPECTED |  | 
    cls.seq_relationship.weight | UNEXPECTED |  | 

    Notes:
    - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

```
                                                  bert-base                                                   
                                                                                                              
 sentence type     tokens       PPPL   text                                                                   
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 grammatical           11     3.4800   The committee approved the new budget after a long debate.             
 scrambled             11   3,871.61   budget the approved committee long a after debate new the.             
 agreement error       11     7.3300   The committee approve the new budget after a long debate.              
 nonsense words        16      96.27   The flarn approved the glimt budget after a long trebble.              
 clinical domain       18     8.6500   Patient presented with acute dyspnea and bilateral crackles on auscult 
 code                  20     147.44   for i in range(len(items)): total += items.price                       
                                                                                                              
```

      wikitext-2 mean PPPL over 40 sentences: 8.21 (2.4s)

```
                                                 roberta-base                                                  
                                                                                                               
 sentence type     tokens        PPPL   text                                                                   
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 grammatical           11      3.8300   The committee approved the new budget after a long debate.             
 scrambled             11   16,750.76   budget the approved committee long a after debate new the.             
 agreement error       11       10.62   The committee approve the new budget after a long debate.              
 nonsense words        15      133.62   The flarn approved the glimt budget after a long trebble.              
 clinical domain       18      9.7800   Patient presented with acute dyspnea and bilateral crackles on auscult 
 code                  17      6.8900   for i in range(len(items)): total += items.price                       
                                                                                                               
```

      wikitext-2 mean PPPL over 40 sentences: 16.83 (2.7s)

```
                                              modernbert-base                                               
                                                                                                            
 sentence type     tokens     PPPL   text                                                                   
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 grammatical           11   3.6300   The committee approved the new budget after a long debate.             
 scrambled             12   520.76   budget the approved committee long a after debate new the.             
 agreement error       11   7.2300   The committee approve the new budget after a long debate.              
 nonsense words        16   148.19   The flarn approved the glimt budget after a long trebble.              
 clinical domain       16   2.6600   Patient presented with acute dyspnea and bilateral crackles on auscult 
 code                  16   1.6000   for i in range(len(items)): total += items.price                       
                                                                                                            
```

      wikitext-2 mean PPPL over 40 sentences: 11.77 (3.7s)

    Grammatical << ungrammatical is the acceptability signal (no labels needed).
    High PPPL on the clinical or code line means that text is out of distribution -
    which is exactly when continued MLM pretraining on your own corpus pays off.
    VRAM after pseudo-perplexity  0.01 GB allocated /  0.02 GB reserved

## 12. Head-to-head Benchmark

The last measurement is the most direct one: take real sentences, mask one token at a time, and check whether the model recovers it. Same sentences, same masked positions, same scoring for every model.

**The evaluation only uses single-token gold words**, and the cell reports how many candidates were dropped for being multi-token. That number is not an inconvenience - it is the section 8 lesson quantified, and it differs per tokenizer, which is also why top-k accuracy is only roughly comparable across models.

Read the table with the landscape table in mind. ModernBERT is the best *backbone* here by a comfortable margin on downstream tasks and context length, and it may well not top this particular column. Fill-mask accuracy on generic Wikipedia prose is a narrow measurement of a general-purpose encoder, and the honest conclusion is that all four are close on it while differing enormously on the things you would actually choose between them for: speed, context window, and downstream fine-tuned accuracy.

At 300 masked positions, top-1 accuracy carries roughly +/-3 points of noise.

------------------------------------------------------------------------

``` python
import random

random.seed(0)

# Build the evaluation set once, in raw text, so every model sees identical sentences.
EVAL_SENTS = SENTENCES[:100]


@torch.inference_mode()
def masked_token_accuracy(model, tokenizer, sentences, per_sentence=3, k=5):
    "Mask single-token words at random positions and check top-k recovery."
    ranked, gold_words, dropped = [], [], 0
    for idx, sent in enumerate(sentences):
        words = [w for w in sent.split() if w.isalpha() and len(w) > 2]
        # Seed by index, not hash(sent): string hashing is randomised per process, so a
        # hash-based seed would pick different words after every kernel restart.
        random.Random(idx).shuffle(words)
        picked = 0
        for word in words:
            if picked >= per_sentence:
                break
            # Only single-token targets are scorable with one mask (see section 8).
            forms = [word, " " + word, word.lower(), " " + word.lower()]
            single = [f for f in forms
                      if len(tokenizer.encode(f, add_special_tokens=False)) == 1]
            if not single:
                dropped += 1
                continue
            masked = sent.replace(word, tokenizer.mask_token, 1)
            enc = tokenizer(masked, return_tensors="pt", truncation=True,
                            max_length=128).to(model.device)
            pos = (enc["input_ids"][0] == tokenizer.mask_token_id).nonzero().flatten()
            if len(pos) != 1:
                dropped += 1
                continue
            top = model(**enc).logits[0, pos[0]].float().topk(k)
            ranked.append([tokenizer.decode([i]).strip().lower()
                           for i in top.indices.tolist()])
            gold_words.append(word.lower())
            picked += 1
    return ranked, gold_words, dropped


results = []
for name, model_id, params_m in MODELS:
    t = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE)
    m = AutoModelForMaskedLM.from_pretrained(model_id, cache_dir=HF_CACHE).to(device).eval()

    t0 = time.perf_counter()
    ranked, gold_words, dropped = masked_token_accuracy(m, t, EVAL_SENTS)
    secs = time.perf_counter() - t0

    results.append({
        "model": name,
        "params_m": params_m,
        "vocab": len(t),
        "context": min(getattr(m.config, "max_position_embeddings", 512), 8192),
        "n_scored": len(gold_words),
        "dropped_multitoken": dropped,
        "top1": round(100 * top_k_accuracy(ranked, gold_words, 1), 2),
        "top5": round(100 * top_k_accuracy(ranked, gold_words, 5), 2),
        "mrr": round(mean_reciprocal_rank(ranked, gold_words), 4),
        "wikitext_pppl": ppl_rows[name]["wikitext_pppl"],
        "masks_per_sec": round(len(gold_words) / secs, 1),
    })
    show_kv(results[-1], title=name)

    del m, t
    free_memory()

vram("after benchmark")
```

```
                        distilbert-base                         
                                                                
 model                                          distilbert-base 
 params_m                                                    66 
 vocab                                                   30,522 
 context                                                    512 
 n_scored                                                   297 
 dropped_multitoken                                          53 
 top1                                                     46.46 
 top5                                                     70.37 
 mrr                                                     0.5557 
 wikitext_pppl                                            12.23 
 masks_per_sec                                           388.70 
                                                                
```

    [transformers] BertForMaskedLM LOAD REPORT from: google-bert/bert-base-uncased
    Key                         | Status     |  | 
    ----------------------------+------------+--+-
    bert.pooler.dense.weight    | UNEXPECTED |  | 
    bert.pooler.dense.bias      | UNEXPECTED |  | 
    cls.seq_relationship.bias   | UNEXPECTED |  | 
    cls.seq_relationship.weight | UNEXPECTED |  | 

    Notes:
    - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

```
                           bert-base                            
                                                                
 model                                                bert-base 
 params_m                                                   110 
 vocab                                                   30,522 
 context                                                    512 
 n_scored                                                   297 
 dropped_multitoken                                          53 
 top1                                                     54.21 
 top5                                                     76.43 
 mrr                                                     0.6319 
 wikitext_pppl                                           8.2100 
 masks_per_sec                                           243.20 
                                                                
```

```
                          roberta-base                          
                                                                
 model                                             roberta-base 
 params_m                                                   125 
 vocab                                                   50,265 
 context                                                    514 
 n_scored                                                   297 
 dropped_multitoken                                          69 
 top1                                                     54.88 
 top5                                                     74.41 
 mrr                                                     0.6228 
 wikitext_pppl                                            16.83 
 masks_per_sec                                           225.00 
                                                                
```

```
                        modernbert-base                         
                                                                
 model                                          modernbert-base 
 params_m                                                   149 
 vocab                                                   50,368 
 context                                                  8,192 
 n_scored                                                   297 
 dropped_multitoken                                          76 
 top1                                                     54.21 
 top5                                                     74.41 
 mrr                                                     0.6171 
 wikitext_pppl                                            11.77 
 masks_per_sec                                           154.70 
                                                                
```

    VRAM after benchmark         0.01 GB allocated /  0.02 GB reserved

``` python
import pandas as pd

df_results = pd.DataFrame(results).sort_values("top5", ascending=False)
show_table(
    df_results.to_dict("records"),
    title=f"Masked-token recovery on wikitext-2 ({results[0]['n_scored']} positions)",
    best=("top1", "top5", "mrr", "context", "masks_per_sec"),
    lower_is_better=("wikitext_pppl", "dropped_multitoken"),
    caption="dropped_multitoken is the section-8 lesson quantified, and it differs per "
            "tokenizer - which is why top-k is only roughly comparable across models",
)
```

```
                              Masked-token recovery on wikitext-2 (297 positions)                               
                                                                                                                
                                                      dropped_                                                  
                                                      multitok                            wikitext   masks_per_ 
 model       params_m    vocab   context   n_scored         en    top1    top5      mrr      _pppl          sec 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 bert-base        110   30,522       512        297         53   54.21   76.43   0.6319     8.2100       243.20 
 roberta-b        125   50,265       514        297         69   54.88   74.41   0.6228      16.83       225.00 
 ase                                                                                                            
 modernber        149   50,368     8,192        297         76   54.21   74.41   0.6171      11.77       154.70 
 t-base                                                                                                         
 distilber         66   30,522       512        297         53   46.46   70.37   0.5557      12.23       388.70 
 t-base                                                                                                         
                                                                                                                
dropped_multitoken is the section-8 lesson quantified, and it differs per tokenizer - which is why top-k is only
                                        roughly comparable across models                                        
```

``` python
from pyecharts import options as opts
from pyecharts.charts import Bar

bar = (
    Bar()
    .add_xaxis([r["model"] for r in results])
    .add_yaxis("top-1 accuracy", [r["top1"] for r in results])
    .add_yaxis("top-5 accuracy", [r["top5"] for r in results])
    .add_yaxis("MRR x100", [round(r["mrr"] * 100, 1) for r in results])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"Masked-token recovery on wikitext-2 ({results[0]['n_scored']} positions)",
            subtitle="RTX 3060, single-token targets only - a narrow measurement of a "
                     "general-purpose encoder",
        ),
        yaxis_opts=opts.AxisOpts(name="score"),
        xaxis_opts=opts.AxisOpts(name="model", axislabel_opts=opts.LabelOpts(rotate=12)),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="10%"),
    )
)
bar.render_notebook()
```

``` python
from pyecharts.charts import Line

# Pseudo-perplexity across text types. The shape of each line says what a model finds
# in distribution - which is the practical use of PPPL.
labels = [lab.strip() for lab, _ in ACCEPTABILITY]
line = Line().add_xaxis(labels)
for name in ppl_rows:
    line.add_yaxis(name, [ppl_rows[name][lab] for lab in labels])
line.set_global_opts(
    title_opts=opts.TitleOpts(
        title="Pseudo-perplexity by text type (lower = more expected)",
        subtitle="not comparable to a causal model's perplexity - every prediction here "
                 "saw both directions",
    ),
    xaxis_opts=opts.AxisOpts(name="sentence type",
                             axislabel_opts=opts.LabelOpts(rotate=20, font_size=9)),
    yaxis_opts=opts.AxisOpts(name="pseudo-perplexity", type_="log"),
    tooltip_opts=opts.TooltipOpts(trigger="axis"),
    legend_opts=opts.LegendOpts(pos_top="10%"),
)
line.render_notebook()
```

## 13. Interactive: mask your own text

Edit `MY_TEXTS` below, using `{mask}` where you want the blank - the cell substitutes each model’s own mask token, so the same template works for BERT and RoBERTa. This is the cell people run on its own, so it opens with a `require(...)` guard naming what it needs from Setup rather than dying on a bare `NameError`.

Things worth trying:

- **Move the mask around one sentence.** “The `___` bit the man” versus “The dog bit the `___`”. Watch how much the right-hand context contributes - that is the bidirectionality a causal model does not have, and it is the whole argument for encoders.
- **Probe your own domain.** Write a sentence in your jargon and see whether the completions are plausible. If they are not, that encoder has not seen your domain, and continued pretraining will pay for itself.
- **Try a multi-token target.** Put the mask where the answer is a rare word or a proper noun. It cannot work with one mask, and the printed token counts show why.
- **Vary the template for the same fact** and watch the answer change. This is the fragility that defines cloze-based knowledge probing.
- **Write your own bias template.** Swap the demographic term and compare probabilities. Then change the verb and run it again - if the conclusion moves, you have learned something about the method as well as the model.
- **Try code and non-English.** ModernBERT saw code; BERT did not. `xlm-roberta` and `mmBERT` handle non-English; the models here mostly do not.

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


require("device", "dtype", "HF_CACHE", "free_memory", "vram", "fill", "pseudo_perplexity")

from transformers import AutoModelForMaskedLM, AutoTokenizer

# Use {mask} - it is replaced with each model's own mask token.
MY_TEXTS = [
    "The {mask} bit the man on the leg.",
    "The dog bit the man on the {mask}.",
    "The container is allocated 4 vCPU and 20 GB of {mask}.",
    "The nurse told the resident that {mask} needed a break.",
]

COMPARE = ["google-bert/bert-base-uncased", "answerdotai/ModernBERT-base"]
TOP_K = 5

# Re-runnable: this cell frees its models at the end, so guard the loads or a second
# shift-enter raises NameError.
if "my_models" not in globals():
    my_models = {}
    for mid in COMPARE:
        t = AutoTokenizer.from_pretrained(mid, cache_dir=HF_CACHE)
        m = AutoModelForMaskedLM.from_pretrained(mid, cache_dir=HF_CACHE).to(device).eval()
        my_models[mid] = (m, t)

for text in MY_TEXTS:
    print(text.format(mask="___"))
    for mid, (m, t) in my_models.items():
        filled = text.format(mask=t.mask_token)
        preds = fill(m, t, filled, k=TOP_K)
        for i, row in enumerate(preds):
            tag = f"{mid.split('/')[-1]}" + (f" [mask {i}]" if len(preds) > 1 else "")
            print(f"  {tag:22s} " + "  ".join(f"{w} {p:.3f}" for w, p in row))
        ppl, n = pseudo_perplexity(m, t, text.format(mask="something"))
        print(f"  {'':22s} PPPL of the filled sentence: {ppl:.1f} ({n} tokens)")
    print()

print("Token counts for anything you want to predict (1 token = fillable with one mask):")
_, ref_tok = next(iter(my_models.values()))
for w in ["memory", "RAM", "kubernetes", "acetaminophen"]:
    print(f"  {w:16s} {ref_tok.tokenize(w)}")

for m, t in my_models.values():
    del m, t
del my_models
free_memory()
vram("final")
```

    [transformers] BertForMaskedLM LOAD REPORT from: google-bert/bert-base-uncased
    Key                         | Status     |  | 
    ----------------------------+------------+--+-
    bert.pooler.dense.weight    | UNEXPECTED |  | 
    bert.pooler.dense.bias      | UNEXPECTED |  | 
    cls.seq_relationship.bias   | UNEXPECTED |  | 
    cls.seq_relationship.weight | UNEXPECTED |  | 

    Notes:
    - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

    The ___ bit the man on the leg.
      bert-base-uncased      knife 0.086  dog 0.079  man 0.038  wolf 0.029  vampire 0.028
                             PPPL of the filled sentence: 28.3 (9 tokens)
      ModernBERT-base        dog 0.327  cat 0.141  boy 0.042  woman 0.028  man 0.024
                             PPPL of the filled sentence: 20.7 (9 tokens)

    The dog bit the man on the ___.
      bert-base-uncased      head 0.252  shoulder 0.117  nose 0.108  cheek 0.081  neck 0.069
                             PPPL of the filled sentence: 40.7 (9 tokens)
      ModernBERT-base        street 0.100  head 0.081  floor 0.051  shoulder 0.048  sidewalk 0.043
                             PPPL of the filled sentence: 15.0 (9 tokens)

    The container is allocated 4 vCPU and 20 GB of ___.
      bert-base-uncased      ram 0.611  memory 0.164  storage 0.156  bandwidth 0.024  capacity 0.010
                             PPPL of the filled sentence: 175.4 (13 tokens)
      ModernBERT-base        RAM 0.802  memory 0.119  ram 0.023  storage 0.018  space 0.005
                             PPPL of the filled sentence: 29.5 (13 tokens)

    The nurse told the resident that ___ needed a break.
      bert-base-uncased      she 0.595  he 0.297  they 0.059  i 0.007  everyone 0.005
                             PPPL of the filled sentence: 13.8 (11 tokens)
      ModernBERT-base        he 0.540  she 0.282  they 0.145  it 0.012  patients 0.001
                             PPPL of the filled sentence: 11.7 (11 tokens)

    Token counts for anything you want to predict (1 token = fillable with one mask):
      memory           ['memory']
      RAM              ['ram']
      kubernetes       ['ku', '##ber', '##net', '##es']
      acetaminophen    ['ace', '##tam', '##ino', '##ph', '##en']
    VRAM final                   0.45 GB allocated /  0.49 GB reserved

## 14. Common Frameworks

Fill-mask is not a product, it is an *objective*, and the framework list should be read that way. Nobody deploys a mask filler; they run masked language modelling over their own corpus to produce a better backbone for everything else. That makes the training stack - corpus handling, a tokenizer that fits your vocabulary, and distributed training - the substance here, and the inference tooling almost an afterthought.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [transformers](https://github.com/huggingface/transformers) | modelling | `AutoModelForMaskedLM`, the `fill-mask` pipeline, and `DataCollatorForLanguageModeling` / `DataCollatorForWholeWordMask` | Apache 2.0 | Default. Continued pretraining is about 30 lines with these pieces |
| [accelerate](https://github.com/huggingface/accelerate) + [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) | modelling | Mixed precision, gradient accumulation, checkpoint resumption, and ZeRO sharding when the run outgrows one card | Apache 2.0 / MIT | Any real continued-pretraining run. On this box, a few hundred MB of domain text and a few GPU-hours is the realistic scale |
| [SetFit](https://github.com/huggingface/setfit) / PET-style cloze prompting | modelling | Turning classification into a fill-mask problem, which reuses the pretrained MLM head instead of training a new one | Apache 2.0 | Under ~100 labels per class. This beats fine-tuning a fresh classification head - see `04_Zero_Shot_Classification` |
| [datasets](https://github.com/huggingface/datasets) + [tokenizers](https://github.com/huggingface/tokenizers) | data | Streaming and chunking the domain corpus, and training a vocabulary that actually contains your domain’s words | Apache 2.0 | Always. If your terms tokenise into five subwords each, a new or extended vocabulary is worth more than more steps |
| [Argilla](https://github.com/argilla-io/argilla) / [datatrove](https://github.com/huggingface/datatrove) | data | Corpus inspection, deduplication and filtering - the part that decides whether continued pretraining helps or memorises | Apache 2.0 | Before training. Near-duplicate documents are the most common reason a domain-adapted model overfits |
| [optimum](https://github.com/huggingface/optimum) + [ONNX Runtime](https://onnxruntime.ai/) | inference runtime | The adapted encoder exported for whatever downstream task consumes it | Apache 2.0 / MIT | Deployment of the *downstream* model, which is the only thing that ships |
| [Weights & Biases](https://github.com/wandb/wandb) / [MLflow](https://github.com/mlflow/mlflow) | evaluation | Loss curves, checkpoint tracking, and the downstream-probe results that tell you whether pretraining is helping | MIT / Apache 2.0 | Every pretraining run. MLM loss alone will not tell you when to stop |
| [evaluate](https://github.com/huggingface/evaluate) + a pseudo-perplexity script | evaluation | Perplexity for causal comparison, and the pseudo-perplexity of section 11 for scoring sentences with a bidirectional model | Apache 2.0 | Comparing checkpoints on your own text. Note it is not comparable across tokenizers - section 9 shows why |

**The 2026 default stack** is ModernBERT as the starting checkpoint, `datasets` streaming your domain corpus with deduplication applied first, `Trainer` plus `accelerate` for the MLM run, and a downstream probe - not the MLM loss - deciding when to stop. Whole-word masking if the downstream tasks are extractive.

The common wrong turn is skipping this step entirely. Continued pretraining on unlabelled domain text needs **zero labels**, costs a few GPU-hours, and produces a backbone that beats the general checkpoint on every downstream task in that domain - on clinical, legal and log data it is one of the highest-return moves in applied NLP and it is routinely left undone. The second is using MLM for generation: multiple masks are filled near-independently and do not compose into a coherent phrase.

------------------------------------------------------------------------

## 15. Going Further

- **Continued pretraining is the payoff of this whole notebook.** Run MLM over your own unlabelled corpus starting from `ModernBERT-base`: `AutoModelForMaskedLM` + `DataCollatorForLanguageModeling(mlm_probability=0.15)` + `Trainer` is about 30 lines. A few hundred MB of domain text and a few GPU-hours produces a backbone that beats the general checkpoint on every downstream task in your domain, with **zero labels**. On clinical, legal and log data this is one of the highest-return moves in applied NLP and it is routinely skipped.
- **Whole-word masking helps for extraction tasks.** Masking all the subword pieces of a word together (rather than individual tokens) makes the task harder and the representations better for QA and NER. `DataCollatorForWholeWordMask` does it.
- **Prompt-based few-shot classification (PET).** Under ~100 labels per class, converting classification into a cloze - “This review was `[MASK]`.” with a verbaliser mapping `great`/`terrible` to the labels - beats fine-tuning a classification head, because it reuses the pretrained MLM head instead of training a new one from scratch. See `04_Zero_Shot_Classification` for the related NLI framing.
- **Data augmentation, carefully.** Masking and refilling random words generates paraphrases cheaply, and it will happily flip your label (replace “not” and the sentiment inverts). Always filter augmented examples with the original model or a heuristic before training on them.
- **Do not use MLM for generation.** Multiple masks are filled near-independently and do not compose. For generation, use a decoder (`08_Text_Generation`) or a seq2seq model (`06_Summarization`).
- **Remember DeBERTa-v3 has no MLM head.** It is an excellent fine-tuning backbone and cannot fill masks, because it was pretrained with replaced-token detection. Check the pretraining objective before assuming a head exists.
- **Audit bias on your fine-tuned model, not just the backbone.** The probe in section 10 measures the pretrained priors; fine-tuning changes them, sometimes for the worse. Measure fairness on the deployed task with task-appropriate metrics - the cloze probe is a smoke test, not an assessment.
- **Related notebooks.** `07_Feature_Extraction` (what these encoders are mostly used for), `00_Text_Classification` and `01_Token_Classification` (the fine-tuned heads), `04_Zero_Shot_Classification` (cloze and NLI as label-free classifiers), `08_Text_Generation` (the causal counterpart and the perplexity that is not comparable to this one), `03_Question_Answering` (span heads on the same backbones).

------------------------------------------------------------------------

Back to top
