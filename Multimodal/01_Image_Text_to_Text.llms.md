# Image-Text-to-Text

Everything to know about vision-language models: how pixels get into an LLM’s token stream, what resolution and tiling actually cost, the mid-2026 landscape of models that fit on one consumer card, and runnable code that puts three of them on the same prompts.

Author

Benedict Thekkel

## 1. What is Image-Text-to-Text?

Image-text-to-text is the **image(s) plus a text prompt in, text out** family - what everyone means by “VLM” or “multimodal LLM”. It is the largest and most consequential multimodal task, because it swallowed most of the others: captioning, VQA, OCR, chart reading, document understanding, screenshot agents and referring expressions are all now “prompt a VLM and read the text it emits”.

**Input.** One or more RGB images plus a chat-formatted text prompt. The image is resized (and usually **tiled**) to a grid, cut into patches by a vision encoder, projected into the LLM’s embedding space, and spliced into the token stream at the position of an `<image>` placeholder. The prompt is ordinary chat text around it.

**Output.** Free text. Everything structured - JSON, bounding boxes, markdown tables, code - is text that the model was trained to emit in that shape.

**The number that governs everything: visual tokens.** A 384 px SigLIP encoder at patch 14 gives ~729 patches per tile. Native-resolution models (Qwen3-VL, InternVL3) tile a large image into many such crops, so a 4K screenshot can cost **thousands** of tokens before the prompt starts. Every design in this family is a different answer to “how do I spend fewer tokens per image without going blind”: pixel shuffle (SmolVLM, InternVL), Q-Former resampling (BLIP-2, MiniCPM), token merging, or simply capping the tile count.

| Neighbouring task | Difference | Typical tools |
|----|----|----|
| Image-to-text (`Computer_Vision/05`) | **No prompt.** Fixed captioner or OCR output | BLIP, Florence-2, GOT-OCR |
| Visual question answering (`Multimodal/04`) | Same mechanics, but scored as short-answer accuracy on a fixed benchmark | ViLT, BLIP-VQA, any VLM |
| Document QA (`Multimodal/05`) | Input is a *page*; layout and OCR fidelity dominate | Donut, dots.ocr, Qwen3-VL |
| Video-text-to-text (`Multimodal/06`) | Many frames plus time; token budget is the whole problem | Qwen3-VL, SmolVLM2 |
| Zero-shot classification (`Computer_Vision/11`) | Embedding similarity, no generation | CLIP, SigLIP |
| Image-text-to-image (`Multimodal/02`) | Image *out*, not text out | FLUX Kontext, Qwen-Image-Edit |

**Why “VLM” replaced most vision task-specific models.** Not because it is more accurate - a dedicated detector still beats a VLM at detection - but because one model with a prompt covers a hundred long-tail tasks that would each need their own dataset, and because the output is text your downstream code can already parse. The cost is latency, VRAM, and a model that hallucinates fluently when it cannot see something.

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| GUI and browser agents | Software automation (OpenAI Operator, Claude computer use, UI-TARS) | Screenshot + goal -\> next click coordinates | **Resolution** (small UI text) and grounding precision; latency per step |
| Insurance and damage assessment | Insurance (Tractable, CCC) | Claim photos + policy prompt -\> structured damage report | Consistency across adjusters; auditability; refusal to guess |
| Retail and e-commerce cataloguing | Marketplaces | Seller photo + schema -\> attributes as JSON | Schema adherence at scale; cost per listing |
| Accessibility assistants | Consumer (Be My Eyes + GPT-4o, Envision) | Live camera + spoken question -\> spoken answer | End-to-end latency; never fabricate safety-critical detail |
| Medical imaging triage | Healthcare (research; LLaVA-Med, MedGemma) | Scan + clinical question -\> findings draft | Regulatory approval; hallucination is a patient-safety event |
| Industrial inspection | Manufacturing, energy | Line or drone image + checklist -\> pass/fail with reason | On-prem, edge VRAM budget; false-negative cost |
| Content moderation | Social platforms | Image + policy text -\> label with justification | Throughput; adversarial evasion; consistent policy reading |
| Robotics perception | Robotics (RT-2, OpenVLA, pi-0) | Camera + instruction -\> action tokens or subgoals | Control-loop latency (tens of ms); on-robot compute |
| Data extraction from screenshots | Fintech, ops tooling | Screenshot of a dashboard -\> numbers as JSON | OCR of small text; numeric accuracy |
| Education and tutoring | EdTech | Photo of homework + question -\> worked explanation | Reasoning quality on diagrams and maths |

**What the leaderboard hides.** Four realities.

*Resolution is the hidden cost driver.* A model that scores well on MMBench at 448 px can be useless on a 1440p screenshot, and the fix (tiling) multiplies the token count and therefore the latency and the price. Before choosing a model, work out how many visual tokens your actual images cost.

*Hallucination is the deployment blocker, not accuracy.* VLMs answer confidently about objects that are not present, especially when the prompt presupposes them (“what colour is the car?” on an image with no car). POPE and HallusionBench exist precisely to measure this, and production systems add a refusal path rather than trusting the model.

*Structured output compliance is a real, measurable property.* Two models with the same MMMU score can differ hugely in how often they return parseable JSON under a schema prompt. That number decides your engineering cost, and section 12 measures it.

*Small models won the deployment argument.* A 2B VLM at 4-5 GB of VRAM that answers in under a second is what actually ships on-prem and on-device; the 70B+ models are API products. Everything runnable in this notebook is under 4B for that reason.

------------------------------------------------------------------------

## 3. How Modern VLMs Work

1.  **Dual-encoder contrastive (CLIP, 2021).** Not generative at all - an image tower and a text tower trained to agree. It cannot answer a question, but its vision tower became the eyes of nearly everything after it, and SigLIP (2023) replaced the softmax contrastive loss with a sigmoid one to train better at scale.

2.  **Frozen-LLM bridges (Flamingo 2022, BLIP-2 2023).** Keep a strong LLM frozen, learn a small module that maps image features into its embedding space: Flamingo used gated cross-attention layers, BLIP-2 a **Q-Former** that compresses an image to 32 query tokens. Cheap to train; the bottleneck throws away fine detail.

3.  **Linear projector + instruction tuning (LLaVA, 2023).** The result that reset the field: a *single linear layer* (later a 2-layer MLP) from CLIP features into Vicuna, trained on GPT-4-generated instruction data. Simpler than a Q-Former and better. Every open VLM since is a variation on it.

4.  **Dynamic resolution and tiling (2024).** Fixed 336 px was the ceiling on OCR and charts. LLaVA-NeXT (AnyRes) and InternVL cut a large image into tiles plus a thumbnail; **Qwen2-VL** went further with **naive dynamic resolution** (the ViT accepts arbitrary sizes, producing a variable number of tokens) and **M-RoPE** (separate positional components for time, height and width). This is where “VLMs can read screenshots” came from.

5.  **Token-efficiency architectures (2024-2025).** If tokens are the cost, compress them. **Pixel shuffle** (space-to-depth) trades spatial resolution for channels and cuts token count 4x or 9x - SmolVLM uses it aggressively enough to run a 256M model in under 1 GB. Perceiver resamplers, token pruning, and native-resolution ViTs with 2x2 merging (Qwen2.5-VL) are the other levers.

6.  **Reasoning and agentic VLMs (2025-2026).** RL post-training arrives in vision: models emit a thinking trace before answering, which lifts maths-on-diagram and chart benchmarks sharply. **Qwen3-VL** (2025) ships Instruct and Thinking variants from 2B to 235B with 256k context, interleaved-MRoPE for video, DeepStack for fine detail, and explicit GUI-agent training. **InternVL3/3.5** made native multimodal *pretraining* the recipe (train on text and vision jointly from the start rather than bolting vision onto a finished LLM). **Gemma 3** (2025) brought a 128k window, 140 languages and a pan-and-scan tiling scheme to 1B-27B open weights.

7.  **Where it is going.** Unified understanding *and* generation in one model (see `Multimodal/08_Any_to_Any`), agentic loops where the VLM controls a UI, and on-device: Gemma 3n and SmolVLM2 exist because the phone is the target.

**Mid-2026 state.** The 2-4B tier is startlingly good: Qwen3-VL-2B beats 2023’s 13B models on almost everything, reads dense screenshots, grounds objects to boxes, and returns JSON reliably. The remaining gaps against frontier API models are long-horizon reasoning, dense-text documents at very high resolution, and hallucination under pressure.

------------------------------------------------------------------------

## 4. Evaluation Metrics

There is no single number, and any single number is a red flag. Four axes matter.

**Aggregate capability benchmarks.** MMMU (college-level multi-discipline reasoning), MMBench (fine-grained ability circular-eval), MMStar, MathVista (visual maths), AI2D (diagrams), ChartQA, DocVQA and OCRBench (text in images), RealWorldQA. All are accuracy on short answers, so scoring is exact match after normalisation, and **circular evaluation** (MMBench rotating the answer options) exists because VLMs have strong positional bias toward option A.

**Hallucination.** POPE asks yes/no object-existence questions balanced between present and absent objects, and reports precision/recall/F1 plus the **yes-ratio** - a model that answers “yes” 90% of the time scores well on accuracy and is useless. CHAIR measures the fraction of mentioned objects that are not in the image. HallusionBench targets visual illusions and counterfactual prompts.

**Instruction and format compliance.** Under-reported and operationally decisive: does the model obey “answer with one word”, and does a schema prompt return valid JSON? Both are deterministic to measure, which is why the benchmark in section 13 measures them.

**Cost.** Visual tokens per image, prefill vs decode latency, tokens/second, peak VRAM. A model that is 2 points better on MMMU and 4x slower is usually the wrong choice.

The cell below implements the compliance scorers plus a POPE-style yes-ratio check, all deterministic and reusable in the benchmark.

------------------------------------------------------------------------

``` python
import json
import re


def normalise(text):
    "Lowercase, strip punctuation and articles - the standard VQA-style normalisation."
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(w for w in text.split() if w not in {"a", "an", "the"})


def obeys_word_limit(answer, limit):
    "Did the model respect 'answer in at most N words'? A blunt but honest compliance test."
    return len(normalise(answer).split()) <= limit


def parseable_json(answer):
    "Return the parsed object if the answer contains valid JSON, else None.\n\n    Models fence JSON in ```json blocks, prefix it with 'Sure!', or emit trailing prose.\n    Real pipelines strip the fence and take the first balanced object; anything that\n    still fails to parse is a genuine compliance failure, not a parsing bug.\n    "
    text = re.sub(r"^```(?:json)?|```$", "", answer.strip(), flags=re.MULTILINE).strip()
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    if start == -1:
        return None
    depth, opener = 0, text[start]
    closer = "}" if opener == "{" else "]"
    for i, ch in enumerate(text[start:], start):
        depth += ch == opener
        depth -= ch == closer
        if depth == 0:
            try:
                return json.loads(text[start:i + 1])
            except json.JSONDecodeError:
                return None
    return None


def yes_no(answer):
    "Map a free-text answer onto yes / no / other - the POPE scoring convention."
    n = normalise(answer)
    if n.startswith("yes") or " yes " in f" {n} ":
        return "yes"
    if n.startswith("no") or " no " in f" {n} ":
        return "no"
    return "other"


def pope_scores(preds, golds):
    "Accuracy, F1 and the yes-ratio. The yes-ratio is the one that exposes a lazy model."
    tp = sum(p == "yes" and g == "yes" for p, g in zip(preds, golds))
    fp = sum(p == "yes" and g == "no" for p, g in zip(preds, golds))
    fn = sum(p != "yes" and g == "yes" for p, g in zip(preds, golds))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "accuracy": round(sum(p == g for p, g in zip(preds, golds)) / len(golds), 3),
        "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 3),
        "yes_ratio": round(sum(p == "yes" for p in preds) / len(preds), 3),
    }


print("word limit  :", obeys_word_limit("Two cats.", 3), obeys_word_limit("There are two cats on a couch.", 3))
print("json        :", parseable_json('Sure! ```json\n{"objects": ["cat", "remote"]}\n```'))
print("json (bad)  :", parseable_json("I can see two cats and a couch."))
print("pope        :", pope_scores(["yes", "yes", "yes", "no"], ["yes", "no", "yes", "no"]))
```

    word limit  : True False
    json        : {'objects': ['cat', 'remote']}
    json (bad)  : None
    pope        : {'accuracy': 0.75, 'f1': 0.8, 'yes_ratio': 0.75}

## 5. Datasets

| Dataset | Contents | Size | Scope | License | Typical use |
|----|----|----|----|----|----|
| [LLaVA-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | GPT-4-generated conversations about COCO images | 150k | en | CC-BY-NC 4.0 | The original visual-instruction-tuning set |
| [The Cauldron](https://huggingface.co/datasets/HuggingFaceM4/the_cauldron) | 50 vision-language datasets unified into one chat format | 30M+ turns | en, mixed | per-source | The standard open SFT mixture (Idefics2, SmolVLM) |
| [LAION / DataComp / COYO](https://huggingface.co/datasets/mlfoundations/datacomp_1b) | Web image-text pairs | 1B+ | multilingual, noisy | CC-BY (metadata) | Contrastive and alignment pretraining |
| [PixMo](https://huggingface.co/collections/allenai/pixmo-674746ea613028006285687b) | Human-narrated dense captions, pointing, docs | ~1M | en | ODC-BY | Molmo’s fully-open alternative to distilled data |
| [MMMU](https://huggingface.co/datasets/MMMU/MMMU) | College exam questions with figures, 30 subjects | 11.5k | en | Apache 2.0 | **The** capability benchmark; MMMU-Pro is the harder version |
| [MMBench](https://huggingface.co/datasets/lmms-lab/MMBench) | Fine-grained ability MCQs with circular eval | 3k+ | en, zh | Apache 2.0 | Ability breakdown; robust to option-order bias |
| [POPE](https://huggingface.co/datasets/lmms-lab/POPE) | Balanced object-existence yes/no questions | 9k | en | MIT | **Hallucination**; report the yes-ratio with it |
| [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) | Questions over real charts, incl. arithmetic | 32k | en | GPL-3.0 | Chart reading and numeric reasoning |
| [AI2D](https://huggingface.co/datasets/lmms-lab/ai2d) | Grade-school science diagrams | 5k | en | CC-BY-SA | Diagram understanding |
| [RealWorldQA](https://huggingface.co/datasets/xai-org/RealworldQA) | Driving/real photos with spatial questions | 765 | en | CC-BY-ND | Practical spatial reasoning |
| [DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA) | Scanned document pages | 50k | en | non-commercial | Text-heavy input; see `Multimodal/05` |
| [COCO 2017 val](https://huggingface.co/datasets/lmms-lab/COCO-Caption2017) | Everyday photos, 5 captions each | 5k val | en | CC-BY 4.0 | **This notebook’s images**; convenient and ungated |

This notebook prompts a handful of **COCO val2017** photos plus one text-bearing fixture image, and the benchmark scores deterministic compliance rather than a benchmark accuracy. That is deliberate: a real MMMU or POPE run needs the [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval) harness and hundreds of GPU-minutes, and a 12-image reimplementation of it would be a worse number dressed up as a better one.

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

Leaderboards: **[OpenVLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)** (VLMEvalKit, 20+ benchmarks, the default reference), **[LMArena Vision](https://lmarena.ai/)** for human preference, and **[OCRBench](https://github.com/Yuliang-Liu/MultimodalOCR)** for text-in-image.

| Model | Params | License | Context | Architecture | Best for |
|----|----|----|----|----|----|
| LLaVA-1.5 / 1.6 | 7B-34B | LLaMA / Apache | 4k | CLIP + MLP + Vicuna/Mistral | historical reference; still a fine baseline |
| [SmolVLM2-256M / 500M / 2.2B](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) | 0.26-2.2B | Apache 2.0 | 16k | SigLIP + **pixel shuffle** + SmolLM2 | the smallest usable VLM; video too; under 1 GB at 256M |
| [Moondream 2/3](https://huggingface.co/vikhyatk/moondream2) | 1.9B (3: 9B MoE) | Apache 2.0 | 2k+ | SigLIP + Phi-derived | tiny, fast, strong pointing/detection; needs `trust_remote_code` |
| **[Qwen3-VL-2B/4B/8B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)** | 2-235B | Apache 2.0 | **256k** | native-res ViT + interleaved M-RoPE + DeepStack | **the default choice in 2026**; OCR, grounding, GUI, video |
| [Qwen2.5-VL-3B/7B](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | 3-72B | Apache 2.0 (3B: research) | 32k | native dynamic resolution, window attention | the 2025 workhorse; huge ecosystem support |
| [InternVL3 / 3.5](https://huggingface.co/OpenGVLab/InternVL3-2B-hf) | 1-241B | MIT (varies) | 32k+ | InternViT + pixel shuffle, **native multimodal pretraining** | strong open competitor; clean transformers support |
| [Gemma 3 4B/12B/27B](https://huggingface.co/google/gemma-3-4b-it) | 1-27B | Gemma terms | 128k | SigLIP-400M + pan-and-scan | 140 languages, long context, permissive-ish |
| [Gemma 3n E2B/E4B](https://huggingface.co/google/gemma-3n-E2B-it) | 5.4B raw / 2B effective | Gemma terms | 32k | MatFormer + per-layer embeddings | on-device, and it hears audio too |
| [Phi-4-multimodal](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | 5.6B | MIT | 128k | per-modality LoRA on one backbone | vision + speech in one small model |
| [Molmo](https://huggingface.co/allenai/Molmo-7B-D-0924) | 1-72B | Apache 2.0 | 4k | fully open data (PixMo), pointing | reproducibility; pointing as a primitive |
| [Idefics3 / SmolVLM lineage](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3) | 8B | Apache 2.0 | 10k | SigLIP + Llama 3 | fully documented open recipe |
| [Pixtral](https://huggingface.co/mistralai/Pixtral-12B-2409) | 12B | Apache 2.0 | 128k | native-res ViT trained from scratch | long multi-image contexts |

**Who wins what.** On aggregate capability per parameter, Qwen3-VL leads the open field at every size in 2026, with InternVL3.5 close and Gemma 3 ahead on language coverage and context length. On **tokens per image** (and therefore cost), SmolVLM is in its own class. On **grounding and pointing**, Molmo and Moondream punch far above their size. On **document text**, a specialist (`Multimodal/05`) still beats a generalist.

**What fits this 12 GB box.** Qwen3-VL-2B (4.3 GB download, ~5 GB VRAM), InternVL3-2B (4.2 GB), SmolVLM2-2.2B (9 GB download, ~5 GB VRAM) and SmolVLM-500M (1 GB) all run comfortably; sections 8-10 use them. Gemma 3 4B (8.6 GB download, ~9 GB VRAM in bf16) fits alone but leaves no headroom for high-resolution tiling, so it sits behind `RUN_HEAVY`. Anything 7B+ needs 4-bit here, and the 30B+ models are out of reach.

------------------------------------------------------------------------

## 7. Setup

Everything below runs on a 12 GB RTX 3060 (or CPU, slowly), and **every model loads through Hugging Face `transformers`** - no vendor packages, no `trust_remote_code`. Package roles:

- `transformers` (\>=5.13) + `torch` - all four VLMs via `AutoModelForImageTextToText`
- `accelerate` - `device_map` placement
- `pillow` - image loading and display
- `datasets` - not needed for the images (they are plain URLs), but used by neighbouring notebooks
- `pyecharts` + `pandas` - benchmark chart and table
- `bitsandbytes` - optional, only if you want to try a 7B+ checkpoint in 4-bit

**The one API note that saves an hour.** In transformers 5.x every one of these models is driven the same way:

``` python
messages = [{"role": "user", "content": [
    {"type": "image", "image": pil_image},
    {"type": "text",  "text": "your prompt"},
]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device)
```

`apply_chat_template` with `tokenize=True, return_dict=True` does the image preprocessing too, so there is no separate `processor(images=..., text=...)` call and no per-model prompt string to get wrong. Slice the generated ids past `inputs["input_ids"].shape[1]` to drop the prompt echo.

All downloads land in `DL_tasks/datasets/`, which is gitignored.

------------------------------------------------------------------------

``` python
# Everything runs through Hugging Face transformers - no model-specific packages.
# %pip install -q torch transformers accelerate pillow pandas pyecharts

# Optional: 4-bit quantization if you want to try a 7B+ VLM on a 12 GB card
# %pip install -q bitsandbytes
```

``` python
import ctypes
import ctypes.util
import gc
import time
import urllib.request
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

# Sections whose *download* is over ~8 GB are gated behind this flag.
RUN_HEAVY = False


def vram(tag=""):
    "Report current GPU memory (allocated / reserved). No-op on CPU."
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"VRAM {tag:20s} {alloc:5.2f} GB allocated / {reserved:5.2f} GB reserved")


def free_memory():
    "Collect garbage and hand freed VRAM back to the CUDA allocator.\n\n    Call right after `del`-ing a model you are done with: `del model; free_memory()`.\n    `del` drops the Python reference; this reclaims the RAM and releases the VRAM.\n    "
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    # glibc keeps freed CPU allocations in its arenas instead of returning them
    # to the OS, so RSS compounds across model sections (cpu-offloaded weights
    # live in system RAM). malloc_trim(0) hands the freed arenas back. See
    # dl-visualization-and-memory.md - not optional on a 12 GB box.
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
from IPython.display import display
from PIL import Image

# Three images, chosen to exercise three different abilities.
SOURCES = {
    # 1. A photo with countable objects and clear relations - captioning, counting, POPE.
    "cats": ("http://images.cocodataset.org/val2017/000000039769.jpg", "coco_cats.jpg"),
    # 2. A cluttered indoor scene - spatial reasoning and object lists.
    "room": ("http://images.cocodataset.org/val2017/000000000139.jpg", "coco_room.jpg"),
    # 3. A text-bearing image - OCR and small-text resolution limits.
    "doc": ("https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg",
            "sample_ocr.jpg"),
}

images = {}
for key, (url, fname) in SOURCES.items():
    path = DATA_DIR / fname
    if not path.exists():
        urllib.request.urlretrieve(url, path)
    images[key] = Image.open(path).convert("RGB")
    print(f"{key:5s} {images[key].size}")

cats, room, doc = images["cats"], images["room"], images["doc"]
for img in (cats, room, doc):
    display(img.resize((320, int(320 * img.height / img.width))))
```

    cats  (640, 480)
    room  (640, 426)
    doc   (762, 1000)

![](01_Image_Text_to_Text_files/figure-html/cell-5-output-2.png)

![](01_Image_Text_to_Text_files/figure-html/cell-5-output-3.png)

![](01_Image_Text_to_Text_files/figure-html/cell-5-output-4.png)

``` python
# One driver for every model in this notebook. Every VLM below is loaded with
# AutoModelForImageTextToText and prompted through apply_chat_template, so the only
# thing that changes between sections is the checkpoint id.
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_vlm(model_id, **kw):
    "Load a VLM and return (ask, handles). `ask(images, prompt) -> text`."
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=HF_CACHE)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=dtype, device_map=device, low_cpu_mem_usage=True,
        cache_dir=HF_CACHE, **kw,
    )
    model.eval()

    def ask(imgs, prompt, max_new_tokens=200, return_tokens=False):
        "One or more images plus a prompt -> the model's text (prompt echo stripped)."
        imgs = [imgs] if isinstance(imgs, Image.Image) else list(imgs)
        content = [{"type": "image", "image": im} for im in imgs]
        content.append({"type": "text", "text": prompt})
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        n_prompt = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = processor.batch_decode(out[:, n_prompt:], skip_special_tokens=True)[0].strip()
        return (text, n_prompt, out.shape[1] - n_prompt) if return_tokens else text

    return ask, [model, processor]


def timed(ask, imgs, prompt, **kw):
    "Run one prompt and print it with its wall-clock time. Returns the answer."
    t0 = time.perf_counter()
    answer = ask(imgs, prompt, **kw)
    print(f"> {prompt}\n  [{time.perf_counter() - t0:4.1f}s] {answer}\n")
    return answer
```

## 8. SmolVLM2 2.2B - the token-efficiency argument

SmolVLM2 (Hugging Face, 2025) exists to answer one question: how few visual tokens can you get away with? A SigLIP encoder feeds a **pixel-shuffle** projector that rearranges a 2x2 (or 3x3) block of spatial positions into the channel dimension, cutting the token count 4x or 9x before the LLM ever sees it. That is why 2.2B parameters fit in ~5 GB and why the 256M sibling runs image inference in under 1 GB of VRAM.

What you trade: fine detail. Small text and dense charts are where aggressive compression shows, and it will lose to Qwen3-VL on OCR. What you gain: it runs on hardware where the alternatives do not exist at all, and it handles video (see `Multimodal/06`).

Checkpoints: `HuggingFaceTB/SmolVLM2-2.2B-Instruct` (used here), `-500M-Video-Instruct`, and `HuggingFaceTB/SmolVLM-256M-Instruct` for the extreme end.

------------------------------------------------------------------------

``` python
smol_ask, smol_handles = load_vlm("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
vram("smolvlm2 loaded")

timed(smol_ask, cats, "Write one short alt-text sentence for a screen reader.", max_new_tokens=60)
timed(smol_ask, room, "List every distinct object you can see, comma separated, nothing else.")
timed(smol_ask, doc, "Read all the text in this image exactly as it appears.", max_new_tokens=300)

# How many visual tokens did one image actually cost? This is the number that decides
# latency and price, and it is the whole reason pixel shuffle exists.
_, n_prompt, _ = smol_ask(cats, "hi", max_new_tokens=1, return_tokens=True)
_, n_text_only, _ = smol_ask([], "hi", max_new_tokens=1, return_tokens=True)
print(f"prompt tokens with one image: {n_prompt}  | text-only: {n_text_only}"
      f"  -> ~{n_prompt - n_text_only} visual tokens for a {cats.size[0]}x{cats.size[1]} image")

for h in smol_handles:
    del h
del smol_ask, smol_handles
free_memory()
vram("after smolvlm2")
```

    [transformers] Model config: pad_token_id must be `None` or an integer within the vocabulary (between 0 and 31999), got 128002. This may result in unexpected behavior.

    [transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`

    VRAM smolvlm2 loaded       4.49 GB allocated /  4.50 GB reserved

    [transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`

    > Write one short alt-text sentence for a screen reader.
      [ 1.9s] Two tabby cats are sleeping on a pink blanket.

    [transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`

    > List every distinct object you can see, comma separated, nothing else.
      [ 4.6s] tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv, tv,

    [transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`

    > Read all the text in this image exactly as it appears.
      [ 5.2s] R&D QUALITY IMPROVEMENT SUGGESTION/SOLUTION OF FM Name/Phone Ext: M. Haan, P. Harper, P. Marine: Date: 9/3/92 Supervisor/Manager: J.S. Wigand R&D Group: Liceasee-e Suggestion: Disintegrating coal retention product samples. Retention testing is not performed by most licensees. Other B&W physical measurements as stability tests are sufficient for materials to assure physical integrity. The proposed action will increase laboratory productivity. Suggested solutions: Delete coal retention from the list of standard analysis performed on license submitted analysis performed on license submitted product samples. Special requests for cali retention testing still be submitted except as this. Have you contacted you Manager/Supervisor? Yes No Manager, please contact suggester and forward comments to the Quality Council.

    [transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`

    prompt tokens with one image: 1092  | text-only: 9  -> ~1083 visual tokens for a 640x480 image
    VRAM after smolvlm2        0.01 GB allocated /  0.01 GB reserved

## 9. Qwen3-VL-2B - the 2026 default

Qwen3-VL (Alibaba, 2025) is the model to reach for first at this size. The 2B Instruct checkpoint is 4.3 GB, runs in ~5 GB of VRAM, and carries the full feature set of the family:

- **Native dynamic resolution.** The ViT accepts the image at its own aspect ratio and emits a variable number of tokens (2x2 merged), so a wide screenshot is not squashed into a square.
- **Interleaved M-RoPE.** Positional encoding split across time, height and width, which is what makes long video and multi-image contexts coherent.
- **DeepStack.** Multi-level ViT features fused into the LLM instead of only the final layer, which recovers fine detail that a single-layer projection loses.
- **256k context** (extensible to 1M), 32 OCR languages, and explicit training for **grounding** (boxes as text) and **GUI agents** (click coordinates).

There is also a **Thinking** variant (`Qwen/Qwen3-VL-2B-Thinking`) that emits a reasoning trace before the answer - better on visual maths and charts, several times the tokens.

The four prompts below are the ones worth remembering: describe, ground, extract as JSON, and read text.

------------------------------------------------------------------------

``` python
qwen_ask, qwen_handles = load_vlm("Qwen/Qwen3-VL-2B-Instruct")
vram("qwen3-vl loaded")

timed(qwen_ask, cats, "Describe this image in detail: objects, colours, materials, spatial layout.")

# Grounding: boxes come back as text, normalised to a 0-1000 grid by convention.
timed(qwen_ask, cats, "Locate every cat. Return JSON only: [{\"label\": ..., \"bbox_2d\": [x1,y1,x2,y2]}].")

# Structured extraction: the prompt pattern behind most production VLM pipelines.
timed(qwen_ask, room, 'Return ONLY valid JSON: {"objects": [...], "room_type": "...", "n_people": 0}')

# OCR: this is where native resolution earns its keep against a pixel-shuffled model.
timed(qwen_ask, doc, "Read all the text in this image exactly as it appears.", max_new_tokens=300)
```

    VRAM qwen3-vl loaded       4.26 GB allocated /  4.27 GB reserved
    > Describe this image in detail: objects, colours, materials, spatial layout.
      [ 3.4s] This is a detailed description of the image provided.

    ### Objects and Subjects

    The image features two cats resting on a vibrant pink surface, which appears to be a couch or sofa. The cats are the central subjects, and they are positioned side-by-side.

    - **Cat on the left**: This is a smaller, younger-looking tabby cat, possibly a kitten. It is lying on its back with its legs stretched out. Its fur is a mix of dark brown and black stripes, with a lighter, cream-colored patch on its chest and belly. The cat is wearing a small, light green collar. It is sleeping with its eyes closed, and its tail is curled up behind it. The cat's paws are slightly curled, and its body is relaxed.

    - **Cat on the right**: This is a larger, adult tabby cat with a more prominent, striped pattern. It is lying on its side, facing the viewer, with its head turned slightly to the right. Its fur is

    > Locate every cat. Return JSON only: [{"label": ..., "bbox_2d": [x1,y1,x2,y2]}].
      [ 1.5s] ```json
    [
        {"label": "cat", "bbox_2d": [538, 58, 1000, 773], "label": "cat"},
        {"label": "cat", "bbox_2d": [14, 111, 494, 987], "label": "cat"}
    ]
    ```

    > Return ONLY valid JSON: {"objects": [...], "room_type": "...", "n_people": 0}
      [ 3.3s] ```json
    {
      "objects": [
        "person",
        "television",
        "refrigerator",
        "dining table",
        "chairs",
        "vase",
        "cabinet",
        "window",
        "radiator",
        "fireplace",
        "ceiling light",
        "floor",
        "candle",
        "wall",
        "potted plant",
        "vase",
        "coffee table",
        "cabinet",
        "refrigerator",
        "television",
        "candle",
        "wall",
        "window",
        "radiator",
        "fireplace",
        "ceiling light",
        "floor",
        "candle",
        "wall",
        "window",
        "radiator",
        "fireplace",
        "ceiling light",
        "floor",
        "candle",
        "wall",
        "window",
        "radiator",
        "fireplace",

    > Read all the text in this image exactly as it appears.
      [ 3.7s] R&D
    QUALITY IMPROVEMENT
    SUGGESTION/SOLUTION FORM

    Name/Phone Ext.: M. Hamann, P. Harper, P. Martinez Date: 9/3/92
    Supervisor/Manager: J. S. Wigand R&D Group: Licensee

    Suggestion: Discontinue coal retention analyses on licensee submitted product samples. (Note: Coal Retention testing is not performed by most licensees. Other B&W physical measurements as ends stability and inspection for soft spots in cigarettes are thought to be sufficient measures to assure cigarette physical integrity. The proposed action will increase laboratory productivity.)

    Suggested Solution(s): Delete coal retention from the list of standard analyses performed on licensee submitted product samples. Special requests for coal retention testing could still be submitted on an exception basis.

    Have you contacted your Manager/Supervisor? Yes No

    Manager Comments: Manager, please contact suggester and forward comments to the Quality Council.

    qip.wp
    597005708

    'R&D\nQUALITY IMPROVEMENT\nSUGGESTION/SOLUTION FORM\n\nName/Phone Ext.: M. Hamann, P. Harper, P. Martinez Date: 9/3/92\nSupervisor/Manager: J. S. Wigand R&D Group: Licensee\n\nSuggestion: Discontinue coal retention analyses on licensee submitted product samples. (Note: Coal Retention testing is not performed by most licensees. Other B&W physical measurements as ends stability and inspection for soft spots in cigarettes are thought to be sufficient measures to assure cigarette physical integrity. The proposed action will increase laboratory productivity.)\n\nSuggested Solution(s): Delete coal retention from the list of standard analyses performed on licensee submitted product samples. Special requests for coal retention testing could still be submitted on an exception basis.\n\nHave you contacted your Manager/Supervisor? Yes No\n\nManager Comments: Manager, please contact suggester and forward comments to the Quality Council.\n\nqip.wp\n597005708'

``` python
# Multi-image in one turn: the images are just two <image> placeholders in the same
# prompt, and the model attends across both. This is how comparison, before/after and
# few-shot visual prompting all work.
timed(qwen_ask, [cats, room],
      "These are two photos. In two sentences: what is in each, and what do they have in common?")

# Resolution is a knob, not a constant. Downscaling the OCR image by 4x costs the model
# the small text - the single most common cause of "the VLM cannot read my document".
small_doc = doc.resize((doc.width // 4, doc.height // 4))
print(f"full {doc.size} vs downscaled {small_doc.size}")
timed(qwen_ask, small_doc, "Read all the text in this image exactly as it appears.", max_new_tokens=300)

# Hallucination probe: ask about something that is not there. A well-behaved model says
# no; a lazy one agrees with the premise of the question (see POPE, section 4).
for q in ["Is there a dog in this image? Answer yes or no.",
          "Is there a cat in this image? Answer yes or no.",
          "What colour is the car in this image?"]:
    timed(qwen_ask, cats, q, max_new_tokens=30)

for h in qwen_handles:
    del h
del qwen_ask, qwen_handles
free_memory()
vram("after qwen3-vl")
```

    > These are two photos. In two sentences: what is in each, and what do they have in common?
      [ 1.6s] The first image shows two cats resting on a bright pink couch, each with a remote control beside them. The second image displays a woman in a kitchen, standing at a dining table, with a television and a fireplace visible in the background.

    Both photos feature a domestic interior setting, with the first focusing on a cozy living room and the second on a kitchen and dining area.

    full (762, 1000) vs downscaled (190, 250)
    > Read all the text in this image exactly as it appears.
      [ 4.8s] THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHURCH OF ENGLAND
    THE CHUR

    > Is there a dog in this image? Answer yes or no.
      [ 0.2s] no

    > Is there a cat in this image? Answer yes or no.
      [ 0.2s] yes

    > What colour is the car in this image?
      [ 0.6s] Based on the image provided, there is no car visible. The image shows two cats sleeping on a bright pink couch. There are also two remote controls

    VRAM after qwen3-vl        0.01 GB allocated /  0.01 GB reserved

## 10. InternVL3 2B - the other open recipe

InternVL3 (OpenGVLab, 2025) is the main open alternative, and its architectural claim is worth understanding: instead of taking a finished text LLM and bolting vision onto it, InternVL3 does **native multimodal pretraining** - text and vision from the start, in one stage. The stated benefit is that it avoids the catastrophic-forgetting-and-realignment dance that a bolted-on projector needs, and it shows up as unusually strong text ability for a VLM this size.

Mechanically it uses an InternViT encoder with **pixel unshuffle** to 256 tokens per 448 px tile, dynamic tiling with a thumbnail, and variable visual position encoding (V2PE) for long contexts. The `-hf` checkpoints are transformers-native, so no `trust_remote_code`.

Reach for it when you want a second opinion from a different training lineage, when you need its stronger text-only behaviour in the same model, or when the license (MIT for several sizes) matters.

------------------------------------------------------------------------

``` python
intern_ask, intern_handles = load_vlm("OpenGVLab/InternVL3-2B-hf")
vram("internvl3 loaded")

timed(intern_ask, cats, "Describe this image in detail: objects, colours, materials, spatial layout.")
timed(intern_ask, room, 'Return ONLY valid JSON: {"objects": [...], "room_type": "...", "n_people": 0}')
timed(intern_ask, doc, "Read all the text in this image exactly as it appears.", max_new_tokens=300)

# The native-multimodal-pretraining claim in one probe: ask a text-only question of a
# vision-language model. Bolted-on VLMs often degrade here; this one should not.
timed(intern_ask, [], "In one sentence, what is the difference between precision and recall?",
      max_new_tokens=80)

for h in intern_handles:
    del h
del intern_ask, intern_handles
free_memory()
vram("after internvl3")
```

    [transformers] The tied weights mapping and config for this model specifies to tie model.language_model.embed_tokens.weight to lm_head.weight, but both are present in the checkpoints with different values, so we will NOT tie them. You should update the config with `tie_word_embeddings=False` to silence this warning.

    VRAM internvl3 loaded      4.19 GB allocated /  4.46 GB reserved
    > Describe this image in detail: objects, colours, materials, spatial layout.
      [ 4.0s] In the image, two cats are lying on a pink blanket, seemingly asleep. The blanket covers most of the surface, providing a soft and cozy resting place for the cats. 

    The cat on the left is smaller and has a striped pattern with shades of brown, black, and white. It is curled up with its head resting on the blanket, appearing to be in a deep sleep. The cat on the right is larger and has a similar striped pattern, with a mix of brown, black, and tan colors. This cat is also lying on its side, with its head turned slightly towards the left cat, possibly in a similar state of relaxation.

    Next to the cats, there are two remote controls. One remote is placed near the head of the smaller cat, while the other is positioned closer to the larger cat. The remote controls are white with colored buttons, indicating they are likely for a television or similar device.

    The background includes a red cushion or part of a couch, adding a vibrant

    > Return ONLY valid JSON: {"objects": [...], "room_type": "...", "n_people": 0}
      [ 1.9s] ```json
    {
      "objects": [
        "TV",
        "radiator",
        "vases",
        "flowers",
        "cabinets",
        "kitchen appliances",
        "dining table",
        "chairs",
        "clock",
        "floor",
        "walls",
        "ceiling",
        "light fixture",
        "basket"
      ],
      "room_type": "living room",
      "n_people": 0
    }
    ```

    > Read all the text in this image exactly as it appears.
      [ 3.9s] R&D QUALITY IMPROVEMENT SUGGESTION/SOLUTION FORM

    Name/Phone Ext.: M. Hamann, P. Harper, P. Martinez  
    Date: 9/3/92

    Supervisor/Manager: J. S. Wigand  
    R&D Group: Licensee

    Suggestion: Discontinue coal retention analyses on licensee submitted product samples. (Note: Coal Retention test is not performed by most licensees. Other B&W physical measurements as ends stability and inspection for soft spots in cigarettes are thought to be sufficient measures to assure cigarette physical integrity. The proposed action will increase laboratory productivity.)

    Suggested Solution(s):  
    Delete coal retention from the list of standard analyses performed on licensee submitted product samples. Special requests for coal retention testing could still be submitted on an exception basis.

    Have you contacted your Manager/Supervisor?  
    Yes  
    No

    Manager Comments: Manager, please contact suggester and forward comments to the Quality Council.

    qip.wp  
    597005708

    > In one sentence, what is the difference between precision and recall?
      [ 0.4s] Precision measures the proportion of true positive results among all the results produced, while recall measures the proportion of true positive results among all the actual positive cases.

    VRAM after internvl3       0.01 GB allocated /  0.01 GB reserved

## 11. Gemma 3 4B - long context and 140 languages

Google’s Gemma 3 (2025) is the pick when the constraint is **context or language coverage** rather than raw benchmark position: a 128k window, 140+ languages, and a **pan-and-scan** preprocessing scheme that crops non-square and high-resolution images into overlapping windows so the fixed 896 px SigLIP encoder is not forced to squash them.

At 4B it is a 8.6 GB download and about 9 GB of VRAM in bf16, which fits this card alone but leaves little headroom once a high-resolution image tiles out. That is why it sits behind `RUN_HEAVY` - flip the flag in Setup to run it.

Its on-device sibling **Gemma 3n E2B** goes further (MatFormer nesting, per-layer embeddings, and an audio encoder as well - see `Multimodal/00_Audio_Text_to_Text`), at a 10.9 GB download.

------------------------------------------------------------------------

``` python
if not RUN_HEAVY:
    print("skipped: gemma-3-4b-it is an 8.6 GB download and ~9 GB of VRAM.\n"
          "Set RUN_HEAVY = True in the Setup cell to run it.")
else:
    gemma_ask, gemma_handles = load_vlm("google/gemma-3-4b-it")
    vram("gemma3 loaded")

    timed(gemma_ask, cats, "Describe this image in detail: objects, colours, materials, spatial layout.")
    timed(gemma_ask, room, 'Return ONLY valid JSON: {"objects": [...], "room_type": "...", "n_people": 0}')
    # Language coverage is the differentiator - ask for the answer in another script.
    timed(gemma_ask, cats, "Describe this image in one sentence, in Japanese.", max_new_tokens=80)

    for h in gemma_handles:
        del h
    del gemma_ask, gemma_handles
    free_memory()
    vram("after gemma3")
```

    skipped: gemma-3-4b-it is an 8.6 GB download and ~9 GB of VRAM.
    Set RUN_HEAVY = True in the Setup cell to run it.

## 12. Head-to-head Benchmark

Three VLMs on the **same images and the same prompt suite**, with deterministic scoring - no judge model, no hand-waving. The suite has four kinds of prompt, each scoring a property that matters operationally:

| Prompt kind | Scored by | What it tells you |
|----|----|----|
| “answer in at most 3 words” | `obeys_word_limit` | instruction compliance |
| “return ONLY valid JSON: …” | `parseable_json` | can you build a pipeline on it |
| POPE-style existence yes/no | `pope_scores` (accuracy + **yes-ratio**) | hallucination under a leading question |
| free description | latency and tokens/s only | throughput |

Each model is loaded, measured, and freed **before** the next one loads, so VRAM stays flat.

**Read this as a smoke test, not a leaderboard.** Three images and a dozen prompts is far too small for a stable accuracy estimate, and the real comparison lives on the OpenVLM Leaderboard over 20+ benchmarks. What the sample *does* measure honestly is compliance, latency and tokens/second on this specific 12 GB card - and those are exactly the numbers a leaderboard never gives you.

------------------------------------------------------------------------

``` python
# The suite. `check` is a deterministic scorer; None means "timing only".
SUITE = [
    {"image": "cats", "prompt": "How many cats are in this image? Answer in at most 3 words.",
     "kind": "word_limit", "check": lambda a: obeys_word_limit(a, 3)},
    {"image": "room", "prompt": "What room is this? Answer in at most 3 words.",
     "kind": "word_limit", "check": lambda a: obeys_word_limit(a, 3)},
    {"image": "cats", "prompt": 'Return ONLY valid JSON: {"objects": [...], "n_animals": 0}',
     "kind": "json", "check": lambda a: parseable_json(a) is not None},
    {"image": "room", "prompt": 'Return ONLY valid JSON: {"objects": [...], "n_people": 0}',
     "kind": "json", "check": lambda a: parseable_json(a) is not None},
    {"image": "doc", "prompt": 'Return ONLY valid JSON: {"title": "...", "lines": [...]}',
     "kind": "json", "check": lambda a: parseable_json(a) is not None},
    # POPE-style: half the questions are about objects that are present, half are not.
    {"image": "cats", "prompt": "Is there a cat in this image? Answer yes or no.", "kind": "pope", "gold": "yes"},
    {"image": "cats", "prompt": "Is there a dog in this image? Answer yes or no.", "kind": "pope", "gold": "no"},
    {"image": "cats", "prompt": "Is there a remote control in this image? Answer yes or no.", "kind": "pope", "gold": "yes"},
    {"image": "room", "prompt": "Is there a boat in this image? Answer yes or no.", "kind": "pope", "gold": "no"},
    {"image": "room", "prompt": "Is there a chair in this image? Answer yes or no.", "kind": "pope", "gold": "yes"},
    {"image": "room", "prompt": "Is there an elephant in this image? Answer yes or no.", "kind": "pope", "gold": "no"},
    {"image": "cats", "prompt": "Describe this image in detail.", "kind": "free", "check": None},
    {"image": "doc", "prompt": "Read all the text in this image exactly as it appears.", "kind": "free", "check": None},
]

IMAGES = {"cats": cats, "room": room, "doc": doc}


def run_suite(name, model_id):
    "Load one VLM, run every prompt in the suite, score it, free it."
    ask, handles = load_vlm(model_id)
    answers, latencies, gen_tokens = [], [], []
    for item in SUITE:
        t0 = time.perf_counter()
        text, _, n_new = ask(IMAGES[item["image"]], item["prompt"],
                             max_new_tokens=200, return_tokens=True)
        latencies.append(time.perf_counter() - t0)
        gen_tokens.append(n_new)
        answers.append(text)

    def rate(kind):
        items = [(i, it) for i, it in enumerate(SUITE) if it["kind"] == kind]
        return round(sum(it["check"](answers[i]) for i, it in items) / max(len(items), 1), 3)

    pope_idx = [i for i, it in enumerate(SUITE) if it["kind"] == "pope"]
    pope = pope_scores([yes_no(answers[i]) for i in pope_idx],
                       [SUITE[i]["gold"] for i in pope_idx])

    for h in handles:
        del h
    del ask, handles
    free_memory()
    vram(f"after {name}")
    return {
        "model": name,
        "word_limit": rate("word_limit"),
        "json_ok": rate("json"),
        "pope_acc": pope["accuracy"],
        "yes_ratio": pope["yes_ratio"],
        "sec_per_prompt": round(sum(latencies) / len(latencies), 2),
        "tok_per_sec": round(sum(gen_tokens) / sum(latencies), 1),
        "answers": answers,
    }


results = [
    run_suite("smolvlm2-2.2b", "HuggingFaceTB/SmolVLM2-2.2B-Instruct"),
    run_suite("qwen3-vl-2b", "Qwen/Qwen3-VL-2B-Instruct"),
    run_suite("internvl3-2b", "OpenGVLab/InternVL3-2B-hf"),
]
vram("benchmark done")
```

    [transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`
    [transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`
    [transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`
    [transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`
    [transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`
    [transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`
    [transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`
    [transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`
    [transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`
    [transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`
    [transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`
    [transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`
    [transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`

    VRAM after smolvlm2-2.2b   0.01 GB allocated /  0.01 GB reserved

    VRAM after qwen3-vl-2b     0.01 GB allocated /  0.01 GB reserved

    [transformers] The tied weights mapping and config for this model specifies to tie model.language_model.embed_tokens.weight to lm_head.weight, but both are present in the checkpoints with different values, so we will NOT tie them. You should update the config with `tie_word_embeddings=False` to silence this warning.

    VRAM after internvl3-2b    0.01 GB allocated /  0.01 GB reserved
    VRAM benchmark done        0.01 GB allocated /  0.01 GB reserved

``` python
import pandas as pd

df = pd.DataFrame([{k: v for k, v in r.items() if k != "answers"} for r in results])
df = df.sort_values("json_ok", ascending=False)
df
```

|  | model | word_limit | json_ok | pope_acc | yes_ratio | sec_per_prompt | tok_per_sec |
|----|----|----|----|----|----|----|----|
| 2 | internvl3-2b | 1.0 | 0.667 | 1.0 | 0.5 | 1.54 | 35.0 |
| 0 | smolvlm2-2.2b | 1.0 | 0.333 | 1.0 | 0.5 | 1.75 | 36.9 |
| 1 | qwen3-vl-2b | 1.0 | 0.333 | 1.0 | 0.5 | 1.34 | 55.0 |

``` python
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org serves an expired TLS certificate from most nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"

from pyecharts import options as opts
from pyecharts.charts import Bar

names = [r["model"] for r in results]
bar = (
    Bar()
    .add_xaxis(names)
    .add_yaxis("word-limit compliance %", [round(r["word_limit"] * 100) for r in results])
    .add_yaxis("valid JSON %", [round(r["json_ok"] * 100) for r in results])
    .add_yaxis("POPE accuracy %", [round(r["pope_acc"] * 100) for r in results])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="Compliance and hallucination on a 13-prompt suite",
            subtitle="RTX 3060 12 GB, fp16, greedy decoding - smoke test, not a leaderboard",
        ),
        xaxis_opts=opts.AxisOpts(name="model"),
        yaxis_opts=opts.AxisOpts(name="percent", max_=100),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
    )
)
bar.render_notebook()
```

``` python
from pyecharts.charts import Scatter

# The trade every deployment actually makes: how compliant, and how fast.
scatter = Scatter()
scatter.add_xaxis([round(r["sec_per_prompt"], 2) for r in results])
for r in results:
    scatter.add_yaxis(
        r["model"],
        [[round(r["sec_per_prompt"], 2), round(r["json_ok"] * 100)]],
        symbol_size=18,
        label_opts=opts.LabelOpts(is_show=False),
    )
scatter.set_global_opts(
    title_opts=opts.TitleOpts(title="JSON compliance vs latency",
                              subtitle="up and to the left is better"),
    xaxis_opts=opts.AxisOpts(type_="value", name="seconds / prompt"),
    yaxis_opts=opts.AxisOpts(type_="value", name="valid JSON %"),
    tooltip_opts=opts.TooltipOpts(trigger="item"),
)
scatter.render_notebook()
```

``` python
# The rates hide the interesting part: read what each model actually said.
for idx in (0, 2, 6, 12):
    item = SUITE[idx]
    print(f"[{item['kind']}] {item['prompt']}")
    for r in results:
        print(f"  {r['model']:16s}: {r['answers'][idx][:150]}")
    print()
```

    [word_limit] How many cats are in this image? Answer in at most 3 words.
      smolvlm2-2.2b   : 2.
      qwen3-vl-2b     : 2
      internvl3-2b    : 2

    [json] Return ONLY valid JSON: {"objects": [...], "n_animals": 0}
      smolvlm2-2.2b   : {
        "objects": [
            {
                "type": "animal",
                "name": "cat1",
                "image": "https://example.com/cat1.jpg",
              
      qwen3-vl-2b     : ```json
    {
      "objects": [
        {
          "name": "cat",
          "description": "striped cat lying on a pink blanket, sleeping with its eyes closed"
        },
     
      internvl3-2b    : ```json
    {
      "objects": [
        "remote",
        "remote",
        "cat",
        "cat"
      ],
      "n_animals": 2
    }
    ```

    [pope] Is there a dog in this image? Answer yes or no.
      smolvlm2-2.2b   : No.
      qwen3-vl-2b     : no
      internvl3-2b    : No.

    [free] Read all the text in this image exactly as it appears.
      smolvlm2-2.2b   : R&D QUALITY IMPROVEMENT SUGGESTION/SOLUTION OF FM Name/Phone Ext: M. Haan, P. Harper, P. Marine: Date: 9/3/92 Supervisor/Manager: J.S. Wigand R&D Grou
      qwen3-vl-2b     : R&D
    QUALITY IMPROVEMENT
    SUGGESTION/SOLUTION FORM

    Name/Phone Ext.: M. Hamann, P. Harper, P. Martinez Date: 9/3/92
    Supervisor/Manager: J. S. Wigand R&D
      internvl3-2b    : R&D QUALITY IMPROVEMENT SUGGESTION/SOLUTION FORM

    Name/Phone Ext.: M. Hamann, P. Harper, P. Martinez  
    Date: 9/3/92

    Supervisor/Manager: J. S. Wigand 

## 13. Live Demo: point a camera at the model

Streams the webcam through Qwen3-VL-2B and burns the answer to your own question onto each frame. At ~1 second per frame on a 3060 this streams at roughly 1 FPS, not 15 - which is the honest number for a 2B VLM and exactly why production camera products put the VLM behind a shutter button rather than in the preview loop.

This is the cell people run on its own, so it opens with a `require(...)` guard naming what it needs from Setup instead of dying on a bare `NameError`. Capture notes, all measured on the knowledge-lab container:

- **V4L2 backend, MJPEG, and a warm-up read.** Without the warm-up the first frames come back black while auto-exposure settles.
- **Never set `CAP_PROP_BUFFERSIZE`** - it halves the delivered frame rate (67 ms -\> 134 ms per read here) and does not make frames fresher.
- **No `cv2.imshow`** - there is no GUI in the container. Frames go into the notebook through `IPython.display` handles that update in place.

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


require("device", "dtype", "HF_CACHE", "load_vlm", "free_memory", "vram")

import time

# opencv-python-headless is a project dependency; the headless build captures from
# V4L2 fine, it only drops the GUI windows.
import io

import cv2
from IPython.display import Image as IPyImage
from IPython.display import Pretty, display
from PIL import Image, ImageDraw, ImageFont

CAM = 0              # /dev/video0
WARMUP = 10          # throwaway reads - auto-exposure and white balance need to settle
STREAM_SECONDS = 20  # how long the live demo runs; interrupt the kernel to stop early
QUESTION = "In one short sentence, what is happening in this image?"


def open_camera(index=CAM, width=640, height=480, auto_exposure=True, exposure=150):
    "Open a V4L2 webcam in MJPEG mode, let it settle, and return the capture handle."
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(
            f"/dev/video{index} did not open - no camera attached, "
            "or it is not passed through into this container"
        )
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))  # MJPEG unlocks the higher modes
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # UVC exposure is DEVICE state and persists between processes: if anything left this
    # camera in manual mode, every frame comes back dark and never adapts. Ask for the
    # mode explicitly. auto (3) = correct brightness but 15 FPS in a dim room;
    # manual (1) = locked 30 FPS at whatever `exposure` suits the lighting.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if auto_exposure else 1)
    if not auto_exposure:
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    # Deliberately no CAP_PROP_BUFFERSIZE: on the V4L2 backend it HALVES the delivered
    # frame rate and does not make frames any fresher.
    for _ in range(WARMUP):
        if not cap.read()[0]:
            cap.release()
            raise RuntimeError(f"/dev/video{index} opened but delivered no frames")
    return cap


def grab(cap):
    "Read one frame off an open camera as an RGB PIL image (OpenCV hands back BGR)."
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("failed to read a frame")
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


_FONT = ImageFont.load_default(size=15)


def draw_lines(img, lines, pad=6):
    "Burn a few lines of text into a band across the top of a copy of `img`."
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, out.width, 18 * len(lines) + 2 * pad], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        d.text((pad, pad + 18 * i), line, fill=(255, 255, 255), font=_FONT)
    return out


def pair_view(left, right, gap=8):
    "Raw frame and annotated frame side by side on one canvas - the live view."
    right = right.convert("RGB")
    if right.size != left.size:
        right = right.resize(left.size)
    canvas = Image.new("RGB", (left.width * 2 + gap, left.height), (20, 20, 20))
    canvas.paste(left.convert("RGB"), (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    return canvas


def _jpeg(img, quality=80):
    "Encode a PIL image to JPEG bytes - what actually goes over the wire each frame."
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def live_stream(annotate, seconds=STREAM_SECONDS, width=640, height=480):
    """Stream `raw | annotated` into the notebook output until `seconds` elapse.

    `annotate(rgb)` returns `(annotated_image, info_string)`. The image and the status
    line each own a display handle and update in place, so this needs no GUI and no
    `cv2.imshow` - it works over JupyterLab against a headless container. Interrupt the
    kernel (the stop button) to end early; the camera is still released.
    """
    cap = open_camera(width=width, height=height)
    view = status = None  # created from the FIRST real frame, so no placeholder flashes up
    n, t0 = 0, time.perf_counter()
    try:
        while time.perf_counter() - t0 < seconds:
            rgb = grab(cap)
            annotated, info = annotate(rgb)
            n += 1
            frame = IPyImage(data=_jpeg(pair_view(rgb, annotated)))
            line = Pretty(f"frame {n:4d}   {n / (time.perf_counter() - t0):5.2f} FPS   {info}")
            if view is None:
                view = display(frame, display_id=True)
                status = display(line, display_id=True)
            else:
                view.update(frame)
                status.update(line)
    except KeyboardInterrupt:
        if status is not None:
            status.update(Pretty(f"stopped at frame {n}"))
    finally:
        cap.release()  # always hand the device back
    elapsed = time.perf_counter() - t0
    print(f"{n} frames in {elapsed:.1f}s -> {n / max(elapsed, 1e-9):.2f} FPS end-to-end "
          "(camera + VLM + JPEG encode)")


def wrap(text, width=52, max_lines=4):
    "Wrap a sentence to something that fits across a 640 px frame."
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    return lines[:max_lines]


# Re-runnable: this cell frees the model at the end, so guard the load or a second
# shift-enter raises NameError on `live_ask`.
if "live_ask" not in globals():
    live_ask, live_handles = load_vlm("Qwen/Qwen3-VL-2B-Instruct")
    vram("live model")


def annotate(rgb):
    "One frame -> (frame with the model's answer wrapped onto it, the answer again)."
    text = live_ask(rgb, QUESTION, max_new_tokens=48)
    return draw_lines(rgb, wrap(text)), text[:80]


live_stream(annotate)

for h in live_handles:
    del h
del live_ask, live_handles
free_memory()
vram("final")
```

    VRAM live model            4.26 GB allocated /  4.27 GB reserved

![](01_Image_Text_to_Text_files/figure-html/cell-17-output-3.jpeg)

    frame   38    1.88 FPS   A computer monitor is displaying a terminal window with text, and a ring light i

    38 frames in 20.3s -> 1.88 FPS end-to-end (camera + VLM + JPEG encode)
    VRAM final                 0.01 GB allocated /  0.01 GB reserved

## 14. Common Frameworks

A VLM is an LLM with an image encoder bolted on, and its ecosystem is the LLM ecosystem with image handling added at the edges. That means the serving, quantisation, structured-decoding and agent frameworks all transfer directly - which is the good news. The bad news is the parts that do not transfer: image tokens make prompts enormous, and the resolution and tiling decisions that control that cost are made in the processor, where most people never look.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [transformers](https://github.com/huggingface/transformers) | modelling | Qwen3-VL, SmolVLM2, InternVL3, Gemma 3 under `AutoModelForImageTextToText`, with chat templates that place images correctly in the conversation | Apache 2.0 | Default. The processor is where resolution and tiling live - read it before blaming the model |
| [peft](https://github.com/huggingface/peft) + [trl](https://github.com/huggingface/trl) | modelling | LoRA/QLoRA with `SFTTrainer` on a 2-3B VLM: freeze the ViT, train the projector plus LoRA on the LLM | Apache 2.0 | A few thousand image/instruction pairs. Full vision-tower fine-tunes rarely pay below ~100k examples |
| [datasets](https://github.com/huggingface/datasets) + Pillow | data | Streaming image-text corpora, and the resize/tile control that decides your token count and therefore your bill | Apache 2.0 / MIT-CMU | Always. Raising input resolution is often a bigger accuracy win than a model upgrade, and a bigger cost too |
| [vLLM](https://github.com/vllm-project/vllm) / [SGLang](https://github.com/sgl-project/sglang) | inference runtime | Continuous batching and prefix caching - a large win when many prompts share one image | Apache 2.0 | Serving. Multiple questions about one image should encode it once, and only these do that for you |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) / [MLX](https://github.com/ml-explore/mlx) / [optimum](https://github.com/huggingface/optimum) | inference runtime | GGUF quants for edge, Apple silicon support, and ONNX export for the small SmolVLM checkpoints on CPU | MIT / Apache 2.0 | On-device deployment. SmolVLM exists precisely for this and runs on a phone |
| [outlines](https://github.com/dottxt-ai/outlines) / [xgrammar](https://github.com/mlc-ai/xgrammar) / lm-format-enforcer | inference runtime | Grammar-constrained decoding: schema-valid JSON by construction rather than by luck | Apache 2.0 / MIT | Any pipeline that parses the output. Prompting gets 80-95%; this gets 100% |
| [BentoML](https://github.com/bentoml/BentoML) / [Ray Serve](https://github.com/ray-project/ray) | serving | An endpoint around vLLM with autoscaling, plus the image preprocessing colocated so you are not shipping raw pixels twice | Apache 2.0 | Production. Preprocessing on the serving side is a real bandwidth saving at image scale |
| [LangGraph](https://github.com/langchain-ai/langgraph) / [LlamaIndex](https://github.com/run-llama/llama_index) / [DSPy](https://github.com/stanfordnlp/dspy) | orchestration | Multi-step flows, tool calls, retrieval, and prompt optimisation over a VLM instead of hand-tuning strings | MIT / Apache 2.0 | The VLM is one step in a system. DSPy in particular optimises the prompts this notebook writes by hand |
| [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) / [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) | evaluation | MMMU, MMBench, DocVQA and the rest as one harness, with the prompt formats the leaderboards used | Apache 2.0 | Comparing models. Reimplementing a benchmark prompt slightly differently is the most common source of unreproducible VLM numbers |

**The 2026 default stack** is Qwen3-VL through `transformers` while you are exploring, vLLM the moment there is more than one user, constrained decoding wherever output is parsed, and LoRA before any thought of a bigger model. SmolVLM2 when the deployment target is small.

The common wrong turn is upgrading the model when the prompt or the resolution is the problem. Raising input resolution, asking for an explicit JSON schema, giving one worked example, or splitting a compound question into two turns each routinely beat a tier upgrade. The second is serving with `generate`: image prompts are long, prefix caching is worth multiples here, and only a real serving runtime gives it to you.

------------------------------------------------------------------------

## 15. Going Further

- **Fine-tuning.** LoRA/QLoRA through `peft` + `trl`’s `SFTTrainer` on a 2-3B VLM fits this 12 GB card if you keep images small and freeze the vision tower. The usual recipe: freeze the ViT, train the projector plus LoRA on the LLM, on a few thousand image/instruction pairs. Hugging Face’s [VLM fine-tuning guide](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl) and the SmolVLM training scripts are the cleanest starting points. Full fine-tunes of the vision tower rarely pay for themselves below ~100k examples.
- **Prompting is most of the gain.** Before changing model, try: raising the input resolution (or turning off aggressive downscaling), asking for JSON with an explicit schema and `Return ONLY valid JSON`, giving one worked example in the prompt, and splitting a compound question into two turns. Each of those routinely beats a one-tier model upgrade.
- **Models that still want `trust_remote_code`.** Moondream, several InternVL non-`-hf` repos, and MiniCPM-V ship custom modelling code. They are good models; they are deliberately not imported here, and the `-hf` mirrors exist precisely to avoid it.
- **Related notebooks.** `Computer_Vision/05_Image_to_Text` (the promptless captioners and OCR specialists), `Multimodal/04_Visual_Question_Answering` (the same models scored as short-answer accuracy), `Multimodal/05_Document_Question_Answering` (pages rather than photos), `Multimodal/06_Video_Text_to_Text` (frames plus time), `Multimodal/07_Visual_Document_Retrieval` (finding the page before you ask about it), and `Multimodal/08_Any_to_Any` (models that also emit images).

------------------------------------------------------------------------

Back to top
