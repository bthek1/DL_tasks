# Text Generation

How autoregressive language models actually produce text: the decoding parameters that matter more than the model choice, what perplexity does and does not tell you, why the KV cache is the whole cost model, and runnable code that measures all of it on models that fit a 12 GB GPU.

Author

Benedict Thekkel

## 1. What is Text Generation?

Text generation produces text one token at a time, each conditioned on everything before it. The model is a function from a token sequence to a probability distribution over the next token:

$$P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})$$

Training maximises the likelihood of real text under that factorisation - it is next-token prediction, nothing more. Everything a large language model appears to do (answering, reasoning, translating, coding) is that single objective applied to text where those behaviours were present.

**Input.** A prompt: a token sequence. For instruction-tuned models the prompt is structured by a **chat template** into roles, which is not cosmetic - a model fed raw text instead of its template is being used out of distribution and behaves noticeably worse.

**Output.** A token distribution per step. **Turning that distribution into text is a separate decision from the model**, and it is where most practical quality lives:

| Strategy | Rule | Good for | Failure mode |
|----|----|----|----|
| **Greedy** | argmax each step | extraction, classification, code | repetition loops; bland |
| **Beam search** | keep k partial sequences by total score | translation, summarization | degenerate on open-ended text |
| **Top-k** | sample from the k likeliest | creative text | k is wrong at both ends of the distribution |
| **Top-p (nucleus)** | sample from the smallest set with mass \>= p | the general default | still admits low-probability tails at high p |
| **Min-p** | keep tokens above `p * max_prob` | robust across entropies | newer, less tooling |
| **Temperature** | scale logits by 1/T before softmax | tunes randomness | high T degrades coherence fast |

**Two phases, two cost models.** Generation splits into **prefill** (encode the prompt - one parallel forward pass over all prompt tokens, compute-bound) and **decode** (produce tokens one at a time, memory-bandwidth-bound because each step reads the entire model’s weights to produce a single token). This asymmetry explains nearly every performance fact about LLM serving, including why batching helps decode enormously and prefill barely at all.

**The KV cache is the reason decode is not quadratic.** Without it, generating token 500 would recompute attention over all 500 prior tokens from scratch. The cache stores each layer’s keys and values so each new token attends to stored state and computes only its own. The cost is memory that grows linearly with sequence length and batch size - and in practice the KV cache, not the weights, is what limits how many concurrent requests a GPU can serve.

**Neighbouring tasks:**

| Task | How it differs | Notebook |
|----|----|----|
| Fill-mask | Bidirectional, predicts a masked middle token | `09_Fill_Mask` |
| Summarization | Grounded in a source document | `06_Summarization` |
| Translation | Constrained by a source meaning | `05_Translation` |
| Question answering | Grounded, and usually short | `03_Question_Answering` |
| Text classification | Fixed label set instead of open text | `00_Text_Classification` |

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Coding assistants | Software (Copilot, Claude Code, Cursor) | Code context -\> completion or edit | Time-to-first-token; correctness beats fluency; huge context |
| Conversational assistants | Consumer and enterprise | Chat history -\> reply | Streaming latency; safety; cost per conversation |
| Agents and tool use | Automation, ops | Instruction + tool schemas -\> structured calls | Output must parse; a malformed call breaks the loop |
| Content drafting | Marketing, support, docs | Brief -\> draft | Brand voice consistency; factual grounding |
| Structured extraction | Data engineering | Document -\> JSON | Schema compliance; constrained decoding, not prompting |
| Synthetic data generation | ML teams | Seed examples -\> training data | Diversity (needs sampling); cost per million tokens |
| On-device assistants | Phones, laptops, cars | Local prompt -\> reply | 1-4B params, quantized; memory bandwidth is the ceiling |
| Reasoning-heavy analysis | Finance, research, engineering | Problem -\> chain of thought -\> answer | Token budget; thinking tokens dominate the cost |

What the benchmark number hides:

- **Decoding parameters change output quality more than a model-size step does.** The same model at `temperature=1.4` and `temperature=0.2` produces work of visibly different quality. Teams that upgrade models without ever tuning sampling are leaving the cheaper win on the table.
- **Throughput is dominated by memory bandwidth, not FLOPs.** Decode reads every weight per token, so a 3B model in fp16 on a 360 GB/s card has a hard ceiling near 60 tokens/s single-stream no matter how fast the arithmetic is. Quantization speeds up decode mainly by shrinking bytes read.
- **The KV cache decides your concurrency.** At long contexts it exceeds the model weights. This is why paged attention (vLLM) and GQA/MLA attention variants exist, and why “context window” and “how many users can I serve” are the same question.
- **Reasoning models moved the cost from parameters to tokens.** A model that emits 3,000 thinking tokens before a 100-token answer costs 30x a direct answer. Budget in tokens, not parameters.
- **Structured output needs constrained decoding, not prompting.** “Reply with valid JSON” fails a few percent of the time, which is a broken pipeline at scale. Grammar-constrained decoding makes it impossible to emit invalid output.

------------------------------------------------------------------------

## 3. How Modern Text Generation Works

1.  **N-gram models (1950s-2010).** Count what followed what. Shannon’s 1948 paper generated English this way. Sparse, no generalisation beyond the window, but the maximum-likelihood objective and perplexity as its metric both start here and neither has changed.
2.  **Neural and recurrent LMs (2003-2016).** Bengio’s neural LM, then RNNs, LSTMs and GRUs. Fixed-size hidden state carried context, but sequential computation blocked parallel training and long dependencies vanished.
3.  **The Transformer decoder (2017-2019).** Self-attention with a causal mask: full context access, fully parallel training. GPT-1 (2018) and GPT-2 (2019) showed that a large decoder pretrained on enough text does tasks it was never trained on, purely by continuation.
4.  **Scale and in-context learning (2020-2022).** GPT-3 (175B) made few-shot prompting a usable interface. Kaplan’s scaling laws, then **Chinchilla** (2022), which corrected them: most large models were badly undertrained, and the compute-optimal ratio is roughly 20 tokens per parameter. Everything after Chinchilla is smaller and trained far longer, which is why a 2026 3B model outperforms a 2020 175B one.
5.  **Instruction tuning and preference optimisation (2022-2024).** SFT on instruction data, then RLHF (PPO) and later the simpler **DPO** and its successors aligned outputs with human preference. This is the step that turned a text continuer into an assistant, and it changed usefulness far more than the underlying capability.
6.  **Efficiency architecture (2023-2026).** Grouped-query attention (GQA) and multi-head latent attention (MLA) shrank the KV cache. **Mixture of experts** (Mixtral, DeepSeek-V3, Qwen3-MoE) decoupled parameter count from per-token compute - hundreds of billions of parameters, tens of billions active. RoPE scaling and YaRN pushed context to 128k-1M tokens. FlashAttention made attention memory-linear in practice.
7.  **Reasoning models (2024-2026).** Models trained with RL on verifiable outcomes (maths, code) to produce long chains of thought before answering - OpenAI o1/o3, DeepSeek-R1, Qwen3’s thinking mode, Claude’s extended thinking. **Test-time compute** became a scaling axis alongside parameters and data: the same model gets better by thinking longer. Hybrid models (Qwen3) expose it as a switch, which is why `enable_thinking=False` appears throughout this repo.
8.  **Serving as a discipline (2023-2026).** vLLM’s paged attention, continuous batching, speculative decoding (a small draft model proposes, the big one verifies in parallel), prefix caching, and 4-bit weight quantization (GPTQ, AWQ, bitsandbytes). These deliver order-of-magnitude throughput gains with no quality change, and they are where most production effort now goes.

**Where it stands (mid-2026).** Open models in the 1-8B range are genuinely useful for extraction, classification, routing and drafting, and they run on consumer hardware. 30-70B dense and MoE models handle most production reasoning at a fraction of frontier cost. Frontier models lead on hard reasoning, long-horizon agentic work and code. The practical skills are no longer “which model” - they are prompt structure, decoding configuration, constrained output, and serving economics.

------------------------------------------------------------------------

## 4. Evaluation Metrics

**Perplexity** - the exponentiated mean negative log-likelihood per token:

$$\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i \mid x_{<i})\right)$$

Read it as “the effective number of equally likely choices the model faces at each token”. A perplexity of 10 means the model is about as uncertain as if it were picking uniformly among 10 tokens.

**What perplexity is good for:** comparing the *same model* across checkpoints, quantization levels, or context lengths. Quantized a model and want to know if it broke? Perplexity answers that precisely.

**What perplexity cannot do, and this is important:** compare different models with different tokenizers. Perplexity is per **token**, and tokenizers differ in how many tokens a sentence costs. A model with a bigger vocabulary compresses text into fewer tokens and gets a lower perplexity for free. Cross-model perplexity comparisons are only meaningful with the identical tokenizer, or after converting to **bits per byte**, which normalises the units away. The benchmark below reports both, and the gap between the two rankings is the lesson.

**Perplexity also does not measure helpfulness.** An instruction-tuned model usually has *worse* perplexity on raw web text than its base model, while being far more useful. Alignment trades distributional fit for behaviour.

**What is actually used to evaluate generation in 2026:**

| Metric | Measures | Caveat |
|----|----|----|
| **Task benchmarks** (MMLU, GSM8K, HumanEval, GPQA) | knowledge, maths, code | contamination; saturation; multiple-choice is not usage |
| **LMArena Elo** | human pairwise preference | style and length bias; slow to update |
| **LLM-as-judge** (MT-Bench, AlpacaEval) | instruction following, quality | judge bias toward its own family and toward longer answers |
| **Pass@k** | code correctness by execution | only for verifiable tasks |
| **Agentic benchmarks** (SWE-bench, tau-bench) | multi-step tool use | expensive; closest to real usage |

**Throughput metrics, which decide deployments:**

- **TTFT** (time to first token) - prefill latency; what a user perceives as responsiveness.
- **TPOT** / inter-token latency - decode speed; determines whether streaming reads smoothly (above ~20 tokens/s feels fine).
- **Throughput** (tokens/s across concurrent requests) - determines cost per million tokens, and improves dramatically with batching because decode is bandwidth-bound.

**Pitfalls:**

- **Report the stride for perplexity.** Sliding-window perplexity with a small stride gives every token more context and a lower number. Papers using stride 512 and stride 1 are not comparable.
- **Benchmark contamination is real and unfixable.** Public test sets appear in pretraining corpora. Treat single-benchmark claims sceptically and prefer held-out or private evaluations.
- **Measure decode and prefill separately.** A single “tokens/s” figure conflates two different bottlenecks and hides which one you should optimise.

The cell below implements perplexity with an explicit sliding window - the stride argument is the part worth understanding.

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

import torch


@torch.inference_mode()
def perplexity(model, tok, text, max_length=1024, stride=512):
    """Sliding-window perplexity, the standard recipe.

    A model with a 1024-token window cannot score a 300k-token document in one pass, so
    the text is scored in overlapping windows and only the *new* `stride` tokens in each
    window contribute to the loss. Smaller stride => more context per scored token =>
    lower (better-looking) perplexity, so always report it alongside the number.
    """
    ids = tok(text, return_tensors="pt").input_ids.to(model.device)
    n = ids.size(1)
    nll_sum, n_tokens, prev_end = 0.0, 0, 0
    for begin in range(0, n, stride):
        end = min(begin + max_length, n)
        trg_len = end - prev_end               # tokens scored in this window
        window = ids[:, begin:end]
        target = window.clone()
        target[:, :-trg_len] = -100            # -100 => ignored by the loss
        out = model(window, labels=target)
        # out.loss is the mean over scored tokens; multiply back to a sum.
        n_scored = trg_len - 1 if begin == 0 else trg_len
        nll_sum += out.loss.float().item() * n_scored
        n_tokens += n_scored
        prev_end = end
        if end == n:
            break
    return math.exp(nll_sum / n_tokens), n_tokens


def bits_per_byte(ppl, n_tokens, n_bytes):
    """Tokenizer-independent version of perplexity.

    Perplexity is per token, and a model with a larger vocabulary needs fewer tokens for
    the same text - so it wins on perplexity without being a better model. Bits per byte
    divides the same total information by the *text's* size instead, which is comparable
    across tokenizers.
    """
    total_nats = math.log(ppl) * n_tokens
    return total_nats / math.log(2) / n_bytes


# Worked example with no model: two tokenizers over the same 100-byte text.
text_bytes = 100
show_table([{"tokenizer": name, "tokens": n_tok, "bytes": text_bytes,
             "perplexity": ppl, "bits_per_byte": round(bits_per_byte(ppl, n_tok, text_bytes), 4)}
            for name, n_tok, ppl in [("coarse (25 tokens)", 25, 30.25),
                                     ("fine (50 tokens)", 50, 5.50)]],
           title="The same 100 bytes of text, two tokenizers",
           caption="the coarse tokenizer looks 5x worse on perplexity and is identical in "
                   "bits per byte - it just makes fewer, harder predictions. Never compare "
                   "perplexity across tokenizers; convert to bits per byte first")
```

```
               The same 100 bytes of text, two tokenizers               
                                                                        
 tokenizer               tokens    bytes    perplexity    bits_per_byte 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 coarse (25 tokens)          25      100         30.25           1.2297 
 fine (50 tokens)            50      100        5.5000           1.2297 
                                                                        
 the coarse tokenizer looks 5x worse on perplexity and is identical in  
 bits per byte - it just makes fewer, harder predictions. Never compare 
      perplexity across tokenizers; convert to bits per byte first      
```

## 5. Datasets

Three kinds matter and they correspond to the three training stages. Pretraining corpora build the capability; instruction and preference data shape the behaviour; evaluation suites measure both.

| Dataset | Stage | Contents | Size | License | Notes |
|----|----|----|----|----|----|
| [FineWeb / FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb) | pretraining | Filtered CommonCrawl web text | 15T / 1.3T tokens | ODC-By | The current open pretraining standard |
| [The Pile](https://huggingface.co/datasets/EleutherAI/pile) | pretraining | 22 mixed sources incl. code, papers | 825 GB | mixed | Historically important; licence issues |
| [Dolma](https://huggingface.co/datasets/allenai/dolma) | pretraining | Open corpus for OLMo | 3T tokens | ODC-By | Fully documented provenance |
| [The Stack v2](https://huggingface.co/datasets/bigcode/the-stack-v2) | pretraining | Permissively licensed source code | 900B tokens | mixed | Code capability comes from here |
| [wikitext-2 / wikitext-103](https://huggingface.co/datasets/Salesforce/wikitext) | evaluation | Clean Wikipedia articles | 2M / 100M tokens | CC BY-SA 3.0 | The perplexity standard; used below |
| [Tulu 3 SFT mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) | instruction | Curated multi-task instruction data | 940k | ODC-By | Strong open SFT recipe |
| [OpenHermes 2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | instruction | Mostly synthetic instructions | 1M | MIT | Widely used community SFT set |
| [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) | preference | Rated response pairs for DPO | 64k | MIT | The standard DPO dataset |
| [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) | preference | Helpfulness/harmlessness comparisons | 170k | MIT | The original RLHF preference data |
| [MMLU](https://huggingface.co/datasets/cais/mmlu) | evaluation | 57 subjects, multiple choice | 16k | MIT | Saturated and contaminated; still reported |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | evaluation | Grade-school maths word problems | 8.5k | MIT | Reasoning; largely saturated |
| [HumanEval / MBPP](https://huggingface.co/datasets/openai/openai_humaneval) | evaluation | Python problems with unit tests | 164 / 974 | MIT | Execution-based, pass@k |
| [GPQA](https://huggingface.co/datasets/Idavidrein/gpqa) | evaluation | Graduate-level science, google-proof | 448 | CC BY 4.0 | Hard and not yet saturated |

This notebook measures perplexity on **wikitext-2-raw-v1 test** (~1.3 MB, 280k tokens) - the long-standing default for language-model perplexity, clean, and small enough to score in seconds.

Downloads land in `DL_tasks/datasets/` via `cache_dir` (gitignored).

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

References: [LMArena](https://lmarena.ai/) for human preference Elo, the [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) for benchmark aggregates, and [SWE-bench](https://www.swebench.com/) for agentic coding.

| Model | Params | Active | License | Context | Best for |
|----|----|----|----|----|----|
| [gpt2](https://huggingface.co/openai-community/gpt2) | 124M | dense | MIT | 1024 | historical anchor; used below |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 596M | dense | Apache 2.0 | 32k | smallest usable assistant; used below |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 1.7B | dense | Apache 2.0 | 32k | the sweet spot on this box; used below |
| [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | 3B | dense | Apache 2.0 | 64k | fully open data + recipe |
| [Qwen3-4B / 8B](https://huggingface.co/Qwen/Qwen3-8B) | 4-8B | dense | Apache 2.0 | 32k-128k | 4B fits here in 4-bit; 8B does not |
| [Llama 3.x 1B / 3B / 8B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 1-8B | dense | Llama licence | 128k | huge ecosystem; gated download |
| [Gemma 3 1B / 4B / 12B](https://huggingface.co/google/gemma-3-4b-it) | 1-12B | dense | Gemma licence | 128k | strong small multilingual; gated |
| [Mistral Small 3.x](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) | 24B | dense | Apache 2.0 | 32k | strong open mid-size |
| [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) | 30B | 3B | Apache 2.0 | 128k | MoE: 30B quality at ~3B decode cost |
| [DeepSeek-V3 / R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) | 671B | 37B | MIT | 128k | frontier-class open reasoning (server-class) |
| Frontier (Claude, GPT, Gemini) | \- | \- | proprietary | 200k-1M | hard reasoning, agentic work, long horizons |

**How to choose.** Extraction, classification, routing, short structured output: 0.5-2B, and the win comes from constrained decoding rather than model size. General assistant behaviour on one consumer GPU: 3-8B in 4-bit. Serious reasoning or code on your own hardware: a 30B-class MoE, where decode cost tracks *active* parameters. Hard novel problems and long agentic runs: frontier models - the gap is real and it is largest exactly there.

**What fits this box (12 GB VRAM).** Roughly 6B params in fp16, or ~13B in 4-bit. The runnable cells below use `gpt2` (0.5 GB), `Qwen3-0.6B` (1.2 GB) and `Qwen3-1.7B` plus its base (~6.8 GB) - about 8.5 GB of downloads in total. Anything larger belongs in this table, not in a runnable cell.

------------------------------------------------------------------------

## 7. Setup

Everything loads through Hugging Face `transformers` - no vendor packages. Package roles:

- `transformers` + `torch` - all models, `generate`, and the streamer
- `accelerate` - `device_map` placement
- `datasets` - wikitext-2 for perplexity
- `pandas` + `pyecharts` - the benchmark table and charts
- `rich` - the result tables. It renders to HTML inside Jupyter, so the tables survive into the published docs; `show_table` / `show_kv` / `rule` are defined in the first code cell of section 4.

Four `transformers` details that decide whether generation behaves:

- **`do_sample=False` means greedy, and it is the default.** Passing `temperature=0.8` *without* `do_sample=True` silently does nothing, which is one of the most common and most confusing bugs in this API. Recent versions warn; older ones do not.
- **Chat templates are not optional for instruct models.** `tok.apply_chat_template(messages, add_generation_prompt=True)` produces exactly the format the model was trained on. Feeding a raw string instead puts the model out of distribution.
- **`use_cache=True` (the default) is the KV cache.** Turning it off makes generation quadratic; section 10 measures how much.
- **Left padding for batched generation.** Decoder-only models continue from the last position, so right padding puts pad tokens between the prompt and the continuation. `padding_side="left"` is required, not a preference.

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
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM {props.total_memory / 1e9:.1f} GB")
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
    VRAM 12.5 GB
    device: cuda:0 | dtype: torch.float16

``` python
from datasets import load_dataset

# wikitext-2-raw-v1 test: clean Wikipedia prose, the long-standing perplexity benchmark.
wikitext = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test",
                        cache_dir=HF_CACHE)
ppl_text = "\n\n".join(wikitext["text"])
PPL_BYTES = len(ppl_text.encode("utf-8"))

print(wikitext)
print(f"\nperplexity corpus: {PPL_BYTES / 1e6:.2f} MB of text\n")
print(ppl_text[:400].strip(), "...")
```

    Dataset({
        features: ['text'],
        num_rows: 4358
    })

    perplexity corpus: 1.30 MB of text

    = Robert Boulter = 




     Robert Boulter is an English film , television and theatre actor . He had a guest @-@ starring role on the television series The Bill in 2000 . This was followed by a starring role in the play Herons written by Simon Stephens , which was performed in 2001 at the Royal Court Theatre . He had a guest role in the television series Judge John Deed in 2002 . In 2004 Boulter ...

## 8. Decoding strategies: the parameters that matter most

Load one model and change nothing about it. Every difference below comes from how the next-token distribution is turned into a token.

**What each knob does, mechanically:**

- **Temperature** divides the logits before the softmax. `T < 1` sharpens the distribution (more deterministic), `T > 1` flattens it (more random). `T = 0` is greedy. Above about 1.2 most small models lose coherence quickly.
- **Top-k** keeps the k highest-probability tokens and renormalises. Its weakness is that k is fixed while the distribution’s shape is not: when the model is confident, k=50 admits 49 bad tokens; when it is uncertain, k=50 cuts off good ones.
- **Top-p (nucleus)** keeps the smallest set of tokens whose cumulative probability reaches p. This adapts to the distribution’s shape, which is why it became the default. `p = 0.9-0.95` with `T = 0.7-1.0` is the standard configuration.
- **Min-p** keeps tokens with probability at least `p * max_prob`. It scales the threshold to the model’s own confidence, and it holds up better at high temperature than top-p does.
- **Repetition and no-repeat-ngram penalties** attack the degenerate-repetition failure directly. `no_repeat_ngram_size=3` forbids repeating any trigram - effective and blunt, since some trigrams legitimately repeat.
- **Beam search** keeps k sequences by cumulative log-probability. It helps when there is a single right answer (translation, summarization) and actively hurts open-ended generation, where it produces bland high-likelihood text. The “likelihood trap” is the reason sampling exists.

The cell prints the distribution at a single step first, so the abstract knobs become concrete numbers, then generates under each configuration from the same prompt and seed.

------------------------------------------------------------------------

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer

gen_id = "Qwen/Qwen3-1.7B"
tok = AutoTokenizer.from_pretrained(gen_id, cache_dir=HF_CACHE)
model = AutoModelForCausalLM.from_pretrained(
    gen_id, dtype=dtype, device_map=device, cache_dir=HF_CACHE
).eval()
vram("qwen3-1.7b loaded")

# What the model actually produces at one step: a distribution over ~150k tokens.
prompt = "The three most important qualities in a good engineer are"
enc = tok(prompt, return_tensors="pt").to(model.device)
with torch.inference_mode():
    logits = model(**enc).logits[0, -1].float()
probs = logits.softmax(-1)

top = probs.topk(15)
# Kept for the chart in the next cell.
NEXT_TOKENS = [(tok.decode([i]), round(p, 5))
               for p, i in zip(top.values.tolist(), top.indices.tolist())]
srt = probs.sort(descending=True).values
NUCLEUS = [(k, round(float(srt[:k].sum()), 4)) for k in (1, 2, 5, 10, 25, 50, 100, 250, 500)]
N_FOR_90 = int((srt.cumsum(0) < 0.9).sum()) + 1

show_table([{"rank": r, "token": repr(t), "probability": p}
            for r, (t, p) in enumerate(NEXT_TOKENS[:8], start=1)],
           title=f"Next-token distribution for {prompt!r} (top 8 of {len(probs):,})",
           best=("probability",))
show_kv({"top-1 mass": round(float(srt[0]), 3),
         "top-10 mass": round(float(srt[:10].sum()), 3),
         "top-50 mass": round(float(srt[:50].sum()), 3),
         "tokens for 90% mass (what top-p=0.9 keeps)": N_FOR_90,
         "entropy (nats)": round(float(-(probs * probs.clamp_min(1e-12).log()).sum()), 2)},
        title="Distribution shape - high entropy means many plausible continuations")

del logits, probs, srt, top
free_memory()
```

    VRAM qwen3-1.7b loaded       3.44 GB allocated /  4.40 GB reserved

```
  Next-token distribution for 'The three most important qualities in a  
                 good engineer are' (top 8 of 151,936)                  
                                                                        
       rank   token                                         probability 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
          1   ':'                                                0.3822 
          2   ' ('                                               0.0534 
          3   ':\n'                                              0.0390 
          4   ' curiosity'                                       0.0355 
          5   ' the'                                             0.0286 
          6   ' creativity'                                      0.0222 
          7   ' patience'                                        0.0190 
          8   '...'                                              0.0182 
                                                                        
```

```
     Distribution shape - high entropy means many plausible     
                         continuations                          
                                                                
 top-1 mass                                              0.3820 
 top-10 mass                                             0.6280 
 top-50 mass                                             0.8400 
 tokens for 90% mass (what top-p=0.9 keeps)                  99 
 entropy (nats)                                          3.6800 
                                                                
```

``` python
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org serves an expired TLS certificate from most nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"

from pyecharts import options as opts
from pyecharts.charts import Bar, Line

# What top-k and top-p are actually cutting. The bars are the head of the distribution;
# the line is the cumulative mass, and top-p=0.9 is a horizontal cut across it.
bar = (
    Bar()
    .add_xaxis([repr(t) for t, _ in NEXT_TOKENS])
    .add_yaxis("probability", [p for _, p in NEXT_TOKENS],
               label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="The head of the next-token distribution",
            subtitle=f"top 15 of {len(tok):,} tokens - top-p=0.9 keeps {N_FOR_90} of them",
        ),
        xaxis_opts=opts.AxisOpts(name="token",
                                 axislabel_opts=opts.LabelOpts(rotate=35, font_size=9)),
        yaxis_opts=opts.AxisOpts(name="probability"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
    )
)
bar.render_notebook()
```

``` python
# Cumulative mass against the number of tokens kept - the curve top-k slices vertically
# and top-p slices horizontally. A steep curve means the model is confident.
line = (
    Line()
    .add_xaxis([str(k) for k, _ in NUCLEUS])
    .add_yaxis("cumulative probability mass", [m for _, m in NUCLEUS],
               markline_opts=opts.MarkLineOpts(
                   data=[opts.MarkLineItem(y=0.9, name="top-p = 0.9")]))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="Nucleus size: how many tokens hold how much mass",
            subtitle="top-k fixes the x-axis cut; top-p fixes the y-axis cut and lets k "
                     "vary with the model's confidence",
        ),
        xaxis_opts=opts.AxisOpts(name="tokens kept (k)"),
        yaxis_opts=opts.AxisOpts(name="cumulative mass", max_=1),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
    )
)
line.render_notebook()
```

``` python
# The same prompt and seed under six decoding configurations.
CONFIGS = [
    ("greedy", dict(do_sample=False)),
    ("beam search (4)", dict(do_sample=False, num_beams=4, early_stopping=True)),
    ("temp 0.7 + top-p 0.9", dict(do_sample=True, temperature=0.7, top_p=0.9)),
    ("temp 1.5 + top-p 0.95", dict(do_sample=True, temperature=1.5, top_p=0.95)),
    ("top-k 5", dict(do_sample=True, temperature=1.0, top_k=5)),
    ("min-p 0.1, temp 1.2", dict(do_sample=True, temperature=1.2, min_p=0.1, top_p=1.0, top_k=0)),
]

gen_rows = []
with torch.inference_mode():
    for name, cfg in CONFIGS:
        torch.manual_seed(0)                     # same seed - differences are the config
        out = model.generate(**enc, max_new_tokens=48, pad_token_id=tok.eos_token_id, **cfg)
        text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        gen_rows.append({"decoding": name, "continuation": " ".join(text.split())})

show_table(gen_rows, title=f"{prompt!r} - same model, same seed, six decoders",
           caption="greedy and beam are deterministic and flatter; high temperature is "
                   "more varied and less coherent. Nothing about the model changed "
                   "between these rows - only the decoder")
```

```
       'The three most important qualities in a good engineer are' - same model, same seed, six decoders        
                                                                                                                
 decoding                continuation                                                                           
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 greedy                  : 1) the ability to think critically, 2) the ability to communicate effectively, and   
                         3) the ability to work in a team. Is this a valid conclusion? Why or why not? The      
                         answer should be in                                                                    
 beam search (4)         : 1) the ability to think critically, 2) the ability to communicate effectively, and   
                         3) the ability to work in a team. These are the three most important qualities in a    
                         good engineer. These are the three most                                                
 temp 0.7 + top-p 0.9    : 1) ability to solve complex problems, 2) ability to think critically, and 3) the     
                         ability to communicate effectively. These qualities are essential for success in the   
                         field. However, the ability to solve complex problems is often                         
 temp 1.5 + top-p 0.95   attention to details, strong logical thinking, and ability to solve problem Answer:    
                         correct The three most important qualities in a good engineer are attention to         
                         details, strong logical thinking, and the ability to solve problems. This statement is 
                         **                                                                                     
 top-k 5                 : 1. The ability to learn from mistakes, 2. The ability to think critically and        
                         creatively, and 3. A strong sense of ethics and responsibility. Why do you think these 
                         are important? These three qualities are vital for                                     
 min-p 0.1, temp 1.2     : 1. The ability to learn from mistakes, 2. The ability to think and solve problems in 
                         the face of uncertainty, and 3. The ability to work in a team. These three qualities   
                         are all essential for an engineer                                                      
                                                                                                                
greedy and beam are deterministic and flatter; high temperature is more varied and less coherent. Nothing about 
                            the model changed between these rows - only the decoder                             
```

## 9. Repetition, and why greedy decoding loops

The most visible pathology in text generation: a greedy or beam decoder falls into a loop and emits the same clause forever. It is worth understanding because the cause is not a bug.

Holtzman et al. (2019) showed that **human text is not the highest-likelihood text**. People routinely choose surprising words; maximum-likelihood decoding chooses safe ones, and safe continuations of safe continuations converge to a cycle. The model is behaving exactly as trained - the objective and the decoding rule are simply mismatched.

Three fixes, in rough order of preference:

1.  **Sample.** Top-p at a moderate temperature avoids the loop by construction, because a loop requires repeatedly picking the argmax.
2.  **`repetition_penalty`** divides the logits of already-generated tokens by a factor (1.1-1.2 is typical). Cheap, and slightly distorts the distribution for legitimate repetition.
3.  **`no_repeat_ngram_size=3`** forbids any repeated trigram outright. Very effective, and wrong for text that legitimately repeats phrases - a list of dates, a legal document, code.

The cell measures the failure directly rather than asserting it: it runs a repetition-prone prompt with a long budget and counts distinct trigrams as a fraction of total trigrams.

------------------------------------------------------------------------

``` python
def distinct_n(text, n=3):
    "Fraction of n-grams that are unique. Near 1.0 is healthy; low means looping."
    toks = text.split()
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


loop_prompt = "List of things to remember: be kind, be kind, be kind,"
enc2 = tok(loop_prompt, return_tensors="pt").to(model.device)

FIXES = [
    ("greedy (no fix)", dict(do_sample=False)),
    ("greedy + repetition_penalty 1.15", dict(do_sample=False, repetition_penalty=1.15)),
    ("greedy + no_repeat_ngram 3", dict(do_sample=False, no_repeat_ngram_size=3)),
    ("sampling, temp 0.8 top-p 0.9", dict(do_sample=True, temperature=0.8, top_p=0.9)),
]

fix_rows = []
with torch.inference_mode():
    for name, cfg in FIXES:
        torch.manual_seed(0)
        out = model.generate(**enc2, max_new_tokens=90, pad_token_id=tok.eos_token_id, **cfg)
        text = " ".join(tok.decode(out[0][enc2["input_ids"].shape[1]:],
                                   skip_special_tokens=True).split())
        fix_rows.append({"fix": name, "distinct-3": round(distinct_n(text), 3),
                         "continuation": text[:170]})

show_table(fix_rows, title=f"{loop_prompt!r} - a repetition-prone prompt",
           best=("distinct-3",),
           caption="distinct-3 near 1.0 means no trigram repeats; a low value is a decoder "
                   "in a loop. The weights are unchanged - maximum-likelihood decoding is "
                   "what loops")
```

```
              'List of things to remember: be kind, be kind, be kind,' - a repetition-prone prompt              
                                                                                                                
 fix                                distinct-3   continuation                                                   
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 greedy (no fix)                        0.0340   be kind, be kind, be kind, be kind, be kind, be kind, be kind, 
                                                 be kind, be kind, be kind, be kind, be kind, be kind, be kind, 
                                                 be kind, be kind, be kind, be kind, be kind,                   
 greedy + repetition_penalty 1.15       1.0000   and so on. The first time I saw this phrase in a book by the   
                                                 author who was also known for writing about kindness. It's     
                                                 been around since at least 1980s or maybe even ear             
 greedy + no_repeat_ngram 3             1.0000   and be kind. This is a common saying that is often used to     
                                                 encourage people to be kind to others. It is also a reminder   
                                                 that kindness is a powerful force that can change              
 sampling, temp 0.8 top-p 0.9           0.8550   be kind. This is what I've been taught, and I've always        
                                                 followed it. It's the best way to deal with life, and I have a 
                                                 lot of friends. What is the best way to deal with l            
                                                                                                                
 distinct-3 near 1.0 means no trigram repeats; a low value is a decoder in a loop. The weights are unchanged -  
                                   maximum-likelihood decoding is what loops                                    
```

## 10. The KV cache and streaming

Two mechanical facts that explain most of what generation costs.

**The KV cache.** Attention at step `t` needs the keys and values of all previous positions. Recomputing them every step makes generation quadratic in output length; caching them makes each step linear in context and constant in work per token. `use_cache=True` is the default and the cell below measures what turning it off costs - a large multiple, growing with length.

The cache is not free. Its size is:

    2 (K and V) x layers x kv_heads x head_dim x seq_len x batch x bytes_per_element

For a 1.7B model at 4k tokens this is comparable to a few hundred MB; for a 70B model with long context and a real batch, the cache exceeds the weights. That is why **grouped-query attention** (share K/V across query heads) and **paged attention** (vLLM’s non-contiguous cache blocks) exist, and why context length and concurrency are the same budget.

**Prefill versus decode.** Prefill processes the whole prompt in one parallel pass - compute-bound, and fast per token. Decode produces one token at a time, and each step reads the entire model from memory to produce a single token - **memory-bandwidth-bound**. The practical consequences:

- Single-stream decode speed is roughly `memory_bandwidth / model_bytes`. On this card (~360 GB/s) a 3.4 GB fp16 model tops out near 100 tokens/s, and no amount of extra compute changes that.
- Batching helps decode enormously (the same weight read serves many sequences) and prefill barely at all.
- Quantization speeds up decode mostly by shrinking bytes read, not by making arithmetic faster.

**Streaming** exists because of this split: TTFT is prefill, and everything after arrives at decode speed. `TextIteratorStreamer` yields tokens as they are produced, which is what makes a chat interface feel responsive even when total generation takes ten seconds.

------------------------------------------------------------------------

``` python
import threading

from transformers import TextIteratorStreamer

# 1. What the KV cache is worth, measured.
probe = tok("Write a short paragraph about the sea." * 4, return_tensors="pt").to(model.device)
timings = {}
with torch.inference_mode():
    for use_cache in (True, False):
        torch.manual_seed(0)
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**probe, max_new_tokens=64, do_sample=False,
                             use_cache=use_cache, pad_token_id=tok.eos_token_id)
        if device != "cpu":
            torch.cuda.synchronize()
        timings[use_cache] = time.perf_counter() - t0
        n_new = out.shape[1] - probe["input_ids"].shape[1]
        print(f"use_cache={str(use_cache):5s}  {timings[use_cache]:5.2f}s for {n_new} tokens "
              f"({n_new / timings[use_cache]:5.1f} tok/s)")
print(f"  -> the cache is worth {timings[False] / timings[True]:.1f}x here, "
      "and the gap widens with output length\n")

# 2. KV cache size for this model, from the config.
cfg = model.config
kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
bytes_per = 2 if dtype == torch.float16 else 4
per_token = 2 * cfg.num_hidden_layers * kv_heads * head_dim * bytes_per
GQA_RATIO = cfg.num_attention_heads // kv_heads
KV_CTX = [512, 1024, 4096, 8192, 16384, 32768]
KV_ROWS = [{"context tokens": ctx,
            "batch 1 (GB)": round(per_token * ctx / 1e9, 3),
            "batch 16 (GB)": round(per_token * ctx * 16 / 1e9, 2),
            "batch 16 without GQA (GB)": round(per_token * ctx * 16 * GQA_RATIO / 1e9, 2)}
           for ctx in KV_CTX]
show_kv({"model": gen_id, "layers": cfg.num_hidden_layers,
         "query heads": cfg.num_attention_heads, "kv heads": kv_heads,
         "GQA ratio": f"{GQA_RATIO}x", "KV cache per token": f"{per_token / 1e3:.1f} KB"},
        title="What the config says about the cache")
show_table(KV_ROWS, title="KV cache size - the budget that decides your concurrency",
           lower_is_better=("batch 16 (GB)",),
           caption="the last column is what this model would cost without grouped-query "
                   "attention; at long context the cache exceeds the weights")

# 3. Prefill vs decode, separated.
long_prompt = "In the beginning, " + "the system was designed to be simple. " * 120
enc3 = tok(long_prompt, return_tensors="pt").to(model.device)
n_prompt = enc3["input_ids"].shape[1]
with torch.inference_mode():
    if device != "cpu":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    model(**enc3)                                     # prefill only
    if device != "cpu":
        torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = model.generate(**enc3, max_new_tokens=64, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    if device != "cpu":
        torch.cuda.synchronize()
    total_s = time.perf_counter() - t0

n_new = out.shape[1] - n_prompt
_dec_rate = n_new / (total_s - prefill_s)
show_table([{"phase": "prefill (prompt)", "tokens": n_prompt,
             "milliseconds": round(prefill_s * 1000, 1),
             "tokens / second": round(n_prompt / prefill_s, 0),
             "bound by": "compute (parallel)"},
            {"phase": "decode (generation)", "tokens": n_new,
             "milliseconds": round((total_s - prefill_s) * 1000, 1),
             "tokens / second": round(_dec_rate, 1),
             "bound by": "memory bandwidth (sequential)"}],
           title="The two phases have different cost models", best=("tokens / second",),
           caption=f"prefill is ~{(n_prompt / prefill_s) / _dec_rate:.0f}x faster per token. "
                   "TTFT is prefill; everything a user reads after that arrives at "
                   "decode speed")

# 4. Streaming - what makes a chat UI feel fast.
messages = [{"role": "user", "content": "In two sentences, why is the sky blue?"}]
chat = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                               enable_thinking=False)
enc4 = tok(chat, return_tensors="pt").to(model.device)
streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
thread = threading.Thread(target=model.generate, kwargs=dict(
    **enc4, max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.9,
    streamer=streamer, pad_token_id=tok.eos_token_id))

t0 = time.perf_counter()
thread.start()
ttft, n = None, 0
print("streamed:", end=" ", flush=True)
for chunk in streamer:
    if ttft is None and chunk.strip():
        ttft = time.perf_counter() - t0
    n += 1
    print(chunk, end="", flush=True)
thread.join()
print(f"\n\nTTFT {ttft * 1000:.0f} ms, then {n / (time.perf_counter() - t0 - ttft):.1f} chunks/s")

del probe, enc2, enc3, enc4, out
free_memory()
vram("after cache demo")
```

    use_cache=True    0.99s for 64 tokens ( 64.4 tok/s)
    use_cache=False   1.42s for 64 tokens ( 45.1 tok/s)
      -> the cache is worth 1.4x here, and the gap widens with output length

```
              What the config says about the cache              
                                                                
 model                                          Qwen/Qwen3-1.7B 
 layers                                                      28 
 query heads                                                 16 
 kv heads                                                     8 
 GQA ratio                                                   2x 
 KV cache per token                                    114.7 KB 
                                                                
```

```
         KV cache size - the budget that decides your concurrency          
                                                                           
 context tokens   batch 1 (GB)   batch 16 (GB)   batch 16 without GQA (GB) 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
            512         0.0590          0.9400                      1.8800 
          1,024         0.1170          1.8800                      3.7600 
          4,096         0.4700          7.5200                       15.03 
          8,192         0.9400           15.03                       30.06 
         16,384         1.8790           30.06                       60.13 
         32,768         3.7580           60.13                      120.26 
                                                                           
    the last column is what this model would cost without grouped-query    
         attention; at long context the cache exceeds the weights          
```

```
                           The two phases have different cost models                           
                                                                                               
 phase                 tokens   milliseconds   tokens / second   bound by                      
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 prefill (prompt)         965         192.40          5,016.00   compute (parallel)            
 decode (generation)       64       1,022.30             62.60   memory bandwidth (sequential) 
                                                                                               
 prefill is ~80x faster per token. TTFT is prefill; everything a user reads after that arrives 
                                        at decode speed                                        
```

    streamed: The sky appears blue because molecules in the Earth's atmosphere scatter shorter wavelengths of light, such as blue, more efficiently than longer wavelengths. This scattering, called Rayleigh scattering, makes the sky appear blue to our eyes.

    TTFT 38 ms, then 65.4 chunks/s
    VRAM after cache demo        3.46 GB allocated /  4.09 GB reserved

``` python
from pyecharts import options as opts
from pyecharts.charts import Line

# The cache grows linearly in context and in batch, and it is the real limit on how many
# concurrent requests a GPU can serve - not the weights.
total_vram = (torch.cuda.get_device_properties(0).total_memory / 1e9
              if torch.cuda.is_available() else 12.0)
line = (
    Line()
    .add_xaxis([str(c) for c in KV_CTX])
    .add_yaxis("batch 1", [r["batch 1 (GB)"] for r in KV_ROWS])
    .add_yaxis("batch 16", [r["batch 16 (GB)"] for r in KV_ROWS])
    .add_yaxis(f"batch 16, no GQA ({GQA_RATIO}x)",
               [r["batch 16 without GQA (GB)"] for r in KV_ROWS],
               markline_opts=opts.MarkLineOpts(
                   data=[opts.MarkLineItem(y=round(total_vram, 1), name="total VRAM")]))
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"KV cache growth for {gen_id}",
            subtitle="linear in context and in batch - this, not the weights, is what "
                     "caps concurrency",
        ),
        xaxis_opts=opts.AxisOpts(name="context length (tokens)"),
        yaxis_opts=opts.AxisOpts(name="GB", type_="log"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="12%"),
    )
)
line.render_notebook()
```

## 11. Chat templates and thinking mode

Base models continue text. Instruct models answer. The difference is instruction tuning plus preference optimisation, and the *interface* to that difference is the **chat template** - a model-specific format of role markers and special tokens that the model was fine-tuned to expect.

Feeding an instruct model a raw string instead of its template is a real error with a soft failure: the model still produces text, just worse and less controllably. `tok.apply_chat_template(messages, add_generation_prompt=True)` emits the exact string, and printing it once for any new model is worth the ten seconds.

**Thinking mode.** Qwen3 is a hybrid reasoning model: the same weights either answer directly or first emit a `<think>` block of reasoning tokens. The switch is `enable_thinking` in the template. The trade is stark and worth measuring rather than assuming - reasoning costs a multiple of the tokens (and therefore the latency and money) and buys accuracy on problems that need multiple steps, while buying nothing at all on lookup or formatting tasks.

This is why `enable_thinking=False` appears in every other notebook in this folder: those cells want a label or a span, and thinking tokens would be pure cost.

------------------------------------------------------------------------

``` python
# What the template actually produces - print it once for any new model.
messages = [
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user", "content": "Name the capital of Australia."},
]
print("=== chat template, thinking OFF ===")
print(repr(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)))
print("\n=== chat template, thinking ON ===")
print(repr(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True))[-220:])
print()

# The cost/benefit of thinking, measured on a problem that needs steps.
QUESTION = ("A shop sells pens at 3 for 7 dollars and notebooks at 2 for 11 dollars. "
            "If I buy 9 pens and 6 notebooks, how much change do I get from 100 dollars?")

think_rows = []
for label, thinking, budget in [("direct", False, 120), ("thinking", True, 700)]:
    msgs = [{"role": "user", "content": QUESTION}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=thinking)
    enc = tok(chat, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(**enc, max_new_tokens=budget, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        if device != "cpu":
            torch.cuda.synchronize()
        secs = time.perf_counter() - t0
    text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    n_new = out.shape[1] - enc["input_ids"].shape[1]
    answer = text.split("</think>")[-1].strip() if "</think>" in text else text.strip()
    think_rows.append({"mode": label, "tokens": n_new, "seconds": round(secs, 1),
                       "tokens / second": round(n_new / secs, 1),
                       "answer": " ".join(answer.split())[:220]})

show_table(think_rows, title="Direct answer vs thinking, same weights and same question",
           lower_is_better=("tokens", "seconds"),
           caption="correct answer: pens 9/3 x 7 = 21, notebooks 6/2 x 11 = 33, "
                   "change = 100 - 54 = 46")
print("Thinking costs a multiple of the tokens. Spend it on problems with steps, not on")
print("lookups or formatting - which is why enable_thinking=False is used elsewhere here.")

del model, tok
free_memory()
vram("after generation model")
```

    === chat template, thinking OFF ===
    '<|im_start|>system\nYou are a terse assistant.<|im_end|>\n<|im_start|>user\nName the capital of Australia.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

    === chat template, thinking ON ===
    '<|im_start|>system\nYou are a terse assistant.<|im_end|>\n<|im_start|>user\nName the capital of Australia.<|im_end|>\n<|im_start|>assistant\n'

```
                           Direct answer vs thinking, same weights and same question                            
                                                                                                                
 mode       tokens   seconds   tokens / second   answer                                                         
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 direct        120    1.9000             63.90   We are given: - **Pens**: 3 pens for $7 - **Notebooks**: 2     
                                                 notebooks for $11 - **Buy**: 9 pens and 6 notebooks - **Total  
                                                 money**: $100 - **Question**: How much change do you get? ---  
                                                 ### Step 1: Find the cost of 9 pens -                          
 thinking      700     11.10             63.20   <think> Okay, let's see. I need to figure out how much change  
                                                 I get from $100 after buying 9 pens and 6 notebooks. The shop  
                                                 sells pens at 3 for $7 and notebooks at 2 for $11. Hmm, let me 
                                                 break this down step by step. Fir                              
                                                                                                                
               correct answer: pens 9/3 x 7 = 21, notebooks 6/2 x 11 = 33, change = 100 - 54 = 46               
```

    Thinking costs a multiple of the tokens. Spend it on problems with steps, not on
    lookups or formatting - which is why enable_thinking=False is used elsewhere here.
    VRAM after generation model  0.02 GB allocated /  3.47 GB reserved

## 12. Head-to-head Benchmark

Three models, one live at a time: perplexity on wikitext-2, bits per byte, and measured decode throughput.

**Read the two quality columns against each other.** Perplexity ranks by tokens; bits per byte ranks by information per byte of text and is the tokenizer-independent version. GPT-2’s tokenizer is much coarser than Qwen3’s, so the two columns can disagree - and when they do, bits per byte is the honest one. This is the single most misused number in language modelling and the reason section 4 spends so long on it.

**Base models, not instruct models.** Instruction tuning makes a model *more useful* and usually *worse* at predicting raw Wikipedia, because it has been reshaped toward assistant-style text. Comparing an instruct model’s perplexity to a base model’s measures the alignment tax, not capability, so all three rows here are base checkpoints.

**Decode throughput is bandwidth, not intelligence.** Tokens per second tracks model bytes almost exactly, which is the point of section 10’s arithmetic. Expect a roughly inverse-linear relationship between size and speed - and note that this ceiling is why quantization is the first thing anyone reaches for.

Perplexity is measured at `stride=512, max_length=1024` for every model; changing the stride changes every number.

------------------------------------------------------------------------

``` python
BENCH = [
    ("gpt2", "openai-community/gpt2", 124),
    ("qwen3-0.6b-base", "Qwen/Qwen3-0.6B-Base", 596),
    ("qwen3-1.7b-base", "Qwen/Qwen3-1.7B-Base", 1720),
]

DECODE_PROMPT = "The history of computing begins with"
results = []

for name, model_id, params_m in BENCH:
    b_tok = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE)
    b_model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=device, cache_dir=HF_CACHE).eval()

    ctx = min(getattr(b_model.config, "max_position_embeddings", 1024), 1024)
    ppl, n_tokens = perplexity(b_model, b_tok, ppl_text, max_length=ctx, stride=512)
    bpb = bits_per_byte(ppl, n_tokens, PPL_BYTES)

    # Single-stream decode throughput - the bandwidth-bound number from section 10.
    enc = b_tok(DECODE_PROMPT, return_tensors="pt").to(b_model.device)
    with torch.inference_mode():
        b_model.generate(**enc, max_new_tokens=8, do_sample=False,
                         pad_token_id=b_tok.eos_token_id)   # warm up kernels
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = b_model.generate(**enc, max_new_tokens=128, do_sample=False,
                               pad_token_id=b_tok.eos_token_id)
        if device != "cpu":
            torch.cuda.synchronize()
        decode_s = time.perf_counter() - t0
    n_new = out.shape[1] - enc["input_ids"].shape[1]

    results.append({
        "model": name,
        "params_m": params_m,
        "vocab": len(b_tok),
        "ctx_used": ctx,
        "ppl_tokens": n_tokens,
        "perplexity": round(ppl, 2),
        "bits_per_byte": round(bpb, 4),
        "decode_tok_per_sec": round(n_new / decode_s, 1),
        "fp16_gb": round(params_m * 2 / 1000, 2),
    })
    show_kv(results[-1], title=name)

    del b_model, b_tok, enc, out       # one model live at a time
    free_memory()

vram("after benchmark")
```

    [transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (287644 > 1024). Running this sequence through the model will result in indexing errors
    [transformers] `loss_type=None` was set in the config but it is unrecognized. Using the default loss: `ForCausalLMLoss`.

```
                              gpt2                              
                                                                
 model                                                     gpt2 
 params_m                                                   124 
 vocab                                                   50,257 
 ctx_used                                                 1,024 
 ppl_tokens                                             287,643 
 perplexity                                               25.18 
 bits_per_byte                                           1.0327 
 decode_tok_per_sec                                      376.10 
 fp16_gb                                                 0.2500 
                                                                
```

    [transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (299078 > 131072). Running this sequence through the model will result in indexing errors

```
                        qwen3-0.6b-base                         
                                                                
 model                                          qwen3-0.6b-base 
 params_m                                                   596 
 vocab                                                  151,669 
 ctx_used                                                 1,024 
 ppl_tokens                                             299,077 
 perplexity                                               12.09 
 bits_per_byte                                           0.8295 
 decode_tok_per_sec                                       83.10 
 fp16_gb                                                 1.1900 
                                                                
```

    [transformers] Token indices sequence length is longer than the specified maximum sequence length for this model (299078 > 131072). Running this sequence through the model will result in indexing errors

```
                        qwen3-1.7b-base                         
                                                                
 model                                          qwen3-1.7b-base 
 params_m                                                 1,720 
 vocab                                                  151,669 
 ctx_used                                                 1,024 
 ppl_tokens                                             299,077 
 perplexity                                              8.9700 
 bits_per_byte                                           0.7302 
 decode_tok_per_sec                                       62.70 
 fp16_gb                                                 3.4400 
                                                                
```

    VRAM after benchmark         0.02 GB allocated /  3.46 GB reserved

``` python
import pandas as pd

df_results = pd.DataFrame(results).sort_values("bits_per_byte")
show_table(
    df_results.to_dict("records"),
    title="wikitext-2 test, base checkpoints, stride 512 / window 1024",
    best=("decode_tok_per_sec",),
    lower_is_better=("perplexity", "bits_per_byte"),
    caption="bits_per_byte is the tokenizer-independent column - trust it over perplexity "
            "when the vocab sizes differ",
)
```

```
                          wikitext-2 test, base checkpoints, stride 512 / window 1024                           
                                                                                                                
                                                                           bits_per_by   decode_tok_p           
 model           params_m     vocab   ctx_used   ppl_tokens   perplexity            te         er_sec   fp16_gb 
 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 
 qwen3-1.7b-ba      1,720   151,669      1,024      299,077       8.9700        0.7302          62.70    3.4400 
 se                                                                                                             
 qwen3-0.6b-ba        596   151,669      1,024      299,077        12.09        0.8295          83.10    1.1900 
 se                                                                                                             
 gpt2                 124    50,257      1,024      287,643        25.18        1.0327         376.10    0.2500 
                                                                                                                
    bits_per_byte is the tokenizer-independent column - trust it over perplexity when the vocab sizes differ    
```

``` python
from pyecharts import options as opts
from pyecharts.charts import Bar

# Two quality views of the same runs. Perplexity is per token and rewards a coarse
# tokenizer; bits per byte normalises that away.
bar = (
    Bar()
    .add_xaxis([r["model"] for r in results])
    .add_yaxis("perplexity (lower better)", [r["perplexity"] for r in results])
    .add_yaxis("bits per byte x10 (lower better)",
               [round(r["bits_per_byte"] * 10, 3) for r in results])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="wikitext-2 test: perplexity vs bits per byte",
            subtitle="RTX 3060, base checkpoints, stride 512 / window 1024 - "
                     "perplexity is not comparable across tokenizers",
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

# Decode speed against model size in bytes. Decode reads every weight per token, so this
# curve is memory bandwidth, not compute - see section 10.
ordered = sorted(results, key=lambda r: r["fp16_gb"])
line = (
    Line()
    .add_xaxis([f"{r['model']}\n({r['fp16_gb']} GB)" for r in ordered])
    .add_yaxis("decode tokens / second", [r["decode_tok_per_sec"] for r in ordered])
    .add_yaxis("bits per byte x100",
               [round(r["bits_per_byte"] * 100, 1) for r in ordered])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="Decode throughput vs model size",
            subtitle="single stream, greedy, 128 new tokens - bandwidth-bound, "
                     "so speed tracks bytes read per token",
        ),
        xaxis_opts=opts.AxisOpts(name="model (fp16 size)",
                                 axislabel_opts=opts.LabelOpts(font_size=9)),
        yaxis_opts=opts.AxisOpts(name="value"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="10%"),
    )
)
line.render_notebook()
```

## 13. Interactive: chat with your own settings

Edit `MY_MESSAGES` and the sampling parameters below. This is the cell people run on its own, so it opens with a `require(...)` guard naming what it needs from Setup rather than dying on a bare `NameError`.

It streams the reply, so you can watch decode speed directly, and it reports TTFT and tokens/second - the two numbers a user actually feels.

Things worth doing here:

- **Sweep temperature on the same prompt.** Run at 0.2, 0.7 and 1.4 with the same seed. The 0.2 output is reliable and dull; 1.4 is varied and starts to drift. Pick per task: near-0 for extraction and code, 0.7-1.0 for drafting, higher only for deliberate variety like synthetic data.
- **Turn `THINKING` on for a multi-step question.** Watch the token count and the latency. Then turn it on for “what is the capital of France” and watch it waste 200 tokens agreeing with itself.
- **Grow the conversation.** Append turns to `MY_MESSAGES` and watch TTFT rise as prefill grows while tokens/second stays roughly flat - prefill scales with prompt length, decode does not.
- **Try to get invalid JSON.** Ask for a strict JSON object and run it ten times at `temperature=1.0`. It will mostly work, which is the trap: “mostly” is a broken pipeline at scale. The fix is grammar-constrained decoding, not a better prompt.
- **Set `SEED` to `None`** to see run-to-run variance under sampling. Reproducibility requires a fixed seed *and* fixed batch composition; batching changes floating-point reduction order and can change output.

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

import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MY_MESSAGES = [
    {"role": "system", "content": "You are a concise technical assistant."},
    {"role": "user", "content": "Explain the KV cache in three sentences."},
]

MODEL_ID = "Qwen/Qwen3-1.7B"
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_NEW_TOKENS = 220
THINKING = False        # True lets Qwen3 emit a <think> block first
SEED = 0                # None for a fresh sample each run

# Re-runnable: this cell frees the model at the end, so guard the load or a second
# shift-enter raises NameError.
if "my_llm" not in globals():
    my_tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=HF_CACHE)
    my_llm = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=dtype, device_map=device, cache_dir=HF_CACHE).eval()

chat = my_tok.apply_chat_template(MY_MESSAGES, tokenize=False, add_generation_prompt=True,
                                  enable_thinking=THINKING)
enc = my_tok(chat, return_tensors="pt").to(my_llm.device)
n_prompt = enc["input_ids"].shape[1]

if SEED is not None:
    torch.manual_seed(SEED)

streamer = TextIteratorStreamer(my_tok, skip_prompt=True, skip_special_tokens=True)
gen_kwargs = dict(**enc, max_new_tokens=MAX_NEW_TOKENS, streamer=streamer,
                  pad_token_id=my_tok.eos_token_id,
                  **(dict(do_sample=True, temperature=TEMPERATURE, top_p=TOP_P)
                     if TEMPERATURE > 0 else dict(do_sample=False)))
thread = threading.Thread(target=my_llm.generate, kwargs=gen_kwargs)

print(f"prompt: {n_prompt} tokens | temp {TEMPERATURE} | top_p {TOP_P} | "
      f"thinking {THINKING}\n" + "-" * 70)
t0 = time.perf_counter()
thread.start()
ttft, pieces = None, []
for chunk in streamer:
    if ttft is None and chunk.strip():
        ttft = time.perf_counter() - t0
    pieces.append(chunk)
    print(chunk, end="", flush=True)
thread.join()
elapsed = time.perf_counter() - t0

reply = "".join(pieces)
n_out = len(my_tok(reply).input_ids)
print("\n" + "-" * 70)
print(f"TTFT {ttft * 1000:6.0f} ms   (prefill of {n_prompt} tokens)")
print(f"decode {n_out / max(elapsed - ttft, 1e-6):6.1f} tok/s  ({n_out} tokens in "
      f"{elapsed - ttft:.1f}s)")
print(f"total  {elapsed:6.1f} s")

del my_llm, my_tok, enc
free_memory()
vram("final")
```

    prompt: 33 tokens | temp 0.7 | top_p 0.9 | thinking False
    ----------------------------------------------------------------------
    The KV cache is a mechanism used in large language models to store and reuse previously accessed key-value pairs (KV) during inference. It improves efficiency by avoiding repeated computation of the same tokens. This cache is typically managed by the model's inference engine and is shared across multiple requests.
    ----------------------------------------------------------------------
    TTFT     52 ms   (prefill of 33 tokens)
    decode   52.8 tok/s  (56 tokens in 1.1s)
    total     1.1 s
    VRAM final                   0.02 GB allocated /  3.46 GB reserved

## 14. Common Frameworks

Text generation has the largest and fastest-moving ecosystem in this repo, and almost none of it is about the model. `model.generate` is a reference implementation; everything below exists because serving, quantising, constraining and adapting an LLM are each their own engineering problem. If you take one thing from this table, take the serving row - it is the highest-leverage change available in this task and it requires no model work at all.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [transformers](https://github.com/huggingface/transformers) | modelling | Every open model behind one API, chat templates, the KV cache and streaming of sections 10-11, and speculative decoding via `assistant_model` | Apache 2.0 | Default for exploration and for anything where you need to see the mechanism |
| [peft](https://github.com/huggingface/peft) + [trl](https://github.com/huggingface/trl) | modelling | LoRA/QLoRA, `SFTTrainer`, and `DPOTrainer` for preference tuning with no reward model and no PPO | Apache 2.0 | Teaching format and style, which LoRA does very well. Do not use it to add knowledge - retrieval is for knowledge |
| [axolotl](https://github.com/axolotl-ai-cloud/axolotl) / [unsloth](https://github.com/unslothai/unsloth) / [torchtune](https://github.com/pytorch/torchtune) | modelling | Config-driven fine-tuning with the memory tricks already applied - unsloth in particular roughly halves VRAM for a LoRA run | Apache 2.0 | Fine-tuning on constrained hardware. A 1.7B model fine-tunes comfortably on this box with these |
| [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes) / GPTQ / AWQ | data | 4-bit and 8-bit weights - roughly a quarter of the memory, and faster decode because decode is bandwidth-bound | MIT / Apache 2.0 | Almost always on consumer hardware. Verify with perplexity on your own data, which is exactly what perplexity is good at |
| [vLLM](https://github.com/vllm-project/vllm) / [SGLang](https://github.com/sgl-project/sglang) / [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | inference runtime | Paged attention, continuous batching and prefix caching: typically 5-20x the throughput on the same hardware with identical output | Apache 2.0 | Production, without exception. This is the single highest-leverage change in the task |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) / [Ollama](https://github.com/ollama/ollama) / [MLX](https://github.com/ml-explore/mlx) | inference runtime | GGUF quants on CPU, a one-command local server, and Apple silicon support | MIT | Local, offline or on-device. Ollama is how most people run a model on a laptop, and it wraps llama.cpp |
| [outlines](https://github.com/dottxt-ai/outlines) / [xgrammar](https://github.com/mlc-ai/xgrammar) | inference runtime | Grammar-constrained decoding - invalid JSON becomes impossible rather than unlikely | Apache 2.0 / MIT | Anything that parses the output. The difference between a 99% pipeline and a 100% one |
| [LangGraph](https://github.com/langchain-ai/langgraph) / [DSPy](https://github.com/stanfordnlp/dspy) | orchestration | Multi-step flows with state and retries, and - in DSPy’s case - optimising the prompts instead of hand-tuning them | MIT / Apache 2.0 | The model is one step in a system, which it almost always is |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | evaluation | The standard benchmarks with the standard prompt formats, plus your own tasks in the same harness | MIT | Comparing models. Then ignore it: fifty hand-written examples of your actual task, judged by a validated rubric, will guide decisions better |

**The 2026 default stack** is vLLM for serving, a 4-bit quant if the model does not fit, constrained decoding wherever output is parsed, LoRA through trl for format and style, and DPO after SFT when behaviour needs changing. Ollama for anything local.

The common wrong turn is serving `model.generate` from a request handler. It holds a whole GPU per request, cannot batch, and re-prefills shared prompts every time - a serving runtime gives you multiples of the throughput for a configuration change. The second is reporting one tokens-per-second number: time-to-first-token and time-per-output-token have different causes (prefill versus decode) and different fixes, and a single figure tells you which to work on.

------------------------------------------------------------------------

## 15. Going Further

- **Quantize.** 4-bit (`bitsandbytes`, GPTQ, AWQ) roughly quarters memory and speeds up decode, because decode is bandwidth-bound. Quality loss at 4-bit is small for 7B+ models and more noticeable below 3B. **Verify with perplexity on your own data** - that is exactly the comparison perplexity is good at.
- **Constrain structured output.** For JSON, use grammar-constrained decoding (`outlines`, `xgrammar`, llama.cpp GBNF, or `transformers`’ logits processors). It makes invalid output impossible rather than unlikely - the difference between a 99% pipeline and a 100% one.
- **Fine-tune with LoRA.** `peft` (already a dependency) trains a small adapter instead of all the weights: a 1.7B model fine-tunes comfortably on this box. Use it to teach format and style, which it does very well; do not use it to add knowledge, which it does poorly. Retrieval is for knowledge - see `03_Question_Answering`.
- **Try DPO after SFT.** `trl`’s `DPOTrainer` optimises preference pairs directly, with no reward model and no PPO. On a small model with a few thousand preference pairs it is a weekend project and it changes behaviour more than more SFT does.
- **Speculative decoding is free throughput.** A small draft model proposes several tokens, the large model verifies them in one pass. Output is provably identical to the large model’s, at 2-3x the speed. `transformers` supports it via `assistant_model=`.
- **Measure TTFT and TPOT separately.** They have different fixes: TTFT is prefill (prompt length, prefix caching), TPOT is decode (quantization, bandwidth, batch size). A single tokens/second number tells you nothing about which one to work on.
- **Evaluate on your own task, always.** Public benchmarks are contaminated and saturated. Fifty hand-written examples of your actual task, scored by a rubric or an LLM judge you have validated, will guide decisions better than the entire Open LLM Leaderboard.
- **Related notebooks.** `09_Fill_Mask` (the bidirectional counterpart and the objective that trained encoders), `06_Summarization` and `05_Translation` (generation with a source to be faithful to), `03_Question_Answering` (RAG - grounding generation in retrieved text), `00_Text_Classification` (constraining an LLM to a label set), `Multimodal/01_Image_Text_to_Text` (the same decoder with an image encoder attached).

------------------------------------------------------------------------

Back to top
