# Visual Question Answering

Everything to know about VQA: the accuracy metric that looks trivial and is not, why a 0.1B classifier can beat a 2B VLM on the benchmark while being useless in practice, the mid-2026 landscape, and runnable code that scores three models on real VQAv2 items.

Author

Benedict Thekkel

## 1. What is Visual Question Answering?

VQA is **an image plus a natural-language question in, an answer out**. “How many cats are there?” -\> “2”. “What colour is the couch?” -\> “pink”. “Is the man wearing a hat?” -\> “no”.

Mechanically that is a subset of `Multimodal/01_Image_Text_to_Text`, and in 2026 most people answer VQA questions by prompting a general VLM. So why does it still have its own task page, its own benchmark and its own notebook? Because **VQA is a scoring protocol, not an architecture**. It is defined by:

- **Short answers.** Almost always one to three words, drawn from a long-tailed but highly repetitive distribution (“yes”, “no”, “2”, “white”, “tennis”).
- **Ten human annotators per question**, and an accuracy metric built around their disagreement.
- **A benchmark culture** that shaped a decade of multimodal research and still gates most model releases.

Understanding the metric is most of understanding the task, which is why section 4 is unusually long.

**Input.** One RGB image and a question string. (Video QA is `Multimodal/06`; document QA is `Multimodal/05`.)

**Output.** A short answer. Two very different ways to produce it:

- **Classification**: pick from a fixed vocabulary of the ~3129 most common training answers. Fast, tiny, cannot ever say anything else.
- **Generation**: decode free text. Unbounded, and it must then be normalised into something the metric will accept.

| Neighbouring task | Difference | Typical tools |
|----|----|----|
| Image-text-to-text (`Multimodal/01`) | Same inputs, but long free-form answers and no fixed scoring | Qwen3-VL, InternVL3 |
| Document QA (`Multimodal/05`) | Input is a **page**; answers are spans of text; scored with ANLS | Donut, dots.ocr |
| Video QA (`Multimodal/06`) | Question spans time, not just space | Qwen3-VL, VideoLLaMA 3 |
| Image-to-text (`Computer_Vision/05`) | **No question**; describes rather than answers | BLIP, Florence-2 |
| Zero-shot classification (`Computer_Vision/11`) | Fixed label set, no question at all | CLIP, SigLIP |

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Assistive apps for blind and low-vision users | Accessibility (Be My Eyes, Seeing AI, VizWiz) | Phone photo + spoken question -\> short spoken answer | **Answerability**: real user photos are blurry and the honest answer is often “cannot tell”; latency |
| Retail visual search and support | E-commerce | Product photo + “does this come in blue?” -\> answer | Grounding in a catalogue, not just the pixels; refusal to invent stock |
| Insurance and claims triage | Insurance | Damage photo + structured question set -\> answers | Consistency; auditability; a wrong answer is a payout error |
| Quality assurance on production lines | Manufacturing | Line image + “is the seal intact?” -\> yes/no | False negatives; on-prem latency; fixed camera means fine-tuning pays |
| Medical imaging Q&A | Healthcare (research) | Scan + clinical question -\> answer | Calibration and refusal; regulatory approval |
| Autonomous-vehicle scene interrogation | Automotive | Camera frame + “is the crossing clear?” -\> answer | Real-time latency; safety-critical false confidence |
| Content moderation triage | Social platforms | Image + policy question -\> yes/no with reason | Throughput; consistent policy application |
| Education and assessment | EdTech | Diagram + question -\> answer with explanation | Reasoning quality; explanation matters more than the token |
| Dataset curation and auto-labelling | ML engineering | Image + attribute question -\> label | Cost per label; agreement with human annotators |
| Robotic task verification | Robotics | Camera + “did the gripper pick up the cup?” -\> yes/no | Control-loop latency; reliability under distribution shift |

**What the benchmark hides.** Three things dominate real deployments and are absent from VQAv2.

*Answerability.* VQAv2 guarantees every question is answerable from the image. VizWiz - built from photographs actually taken by blind users - does not, and roughly 28% of its questions are unanswerable. A model that never says “unanswerable” is dangerous in exactly the application VQA was invented for.

*Language priors.* Models learn that “what colour is the banana” is “yellow” and “is there a …” is “yes” about 70% of the time, and they answer from the question alone. VQAv2 was specifically built to fight this (every question has two images with different answers); VQA-CP rearranges the splits so the train and test priors differ, and accuracy collapses. If your deployment distribution differs from your training distribution, this is what will bite you.

*Calibration and refusal.* Production systems need a confidence and an escape hatch. A classification model gives you a softmax you can threshold; a generative VLM gives you a fluent sentence with no calibrated confidence attached, which is a real operational regression.

------------------------------------------------------------------------

## 3. How Modern VQA Works

1.  **CNN + LSTM fusion (2015-2017).** The original VQA paper (Antol et al., 2015) encoded the image with a CNN, the question with an LSTM, fused them by element-wise product, and classified over the top answers. Stacked attention and bilinear pooling (MCB, MUTAN) refined the fusion. Crucially, the whole field adopted **classification over ~3129 answers**, and that choice shaped the benchmark for a decade.

2.  **Bottom-up attention (2018).** Run a Faster R-CNN, attend over its object regions rather than a uniform grid. Won the 2017 VQA Challenge and became the default until transformers arrived. Two-stage, slow, and welded to a detector vocabulary.

3.  **Vision-language transformers (2019-2021).** ViLBERT, LXMERT, UNITER and friends pretrained a joint transformer on image-text pairs with masked-language and image-text-matching objectives, then fine-tuned a VQA head. **ViLT** (2021) is the pared-down endpoint: no detector, no convolutions - just a linear patch projection and a single transformer, which made it ~10x faster than region-based models at similar accuracy. Section 8 runs it.

4.  **Generative VQA (2022).** BLIP reframed the answer as *text to decode* rather than a class to pick, so the model can produce answers outside any fixed vocabulary. **BLIP-2** (2023) attached a frozen LLM through a Q-Former and made zero-shot VQA competitive without any VQA fine-tuning.

5.  **Instruction-tuned VLMs (2023-2026).** LLaVA and its descendants absorbed VQA entirely: no VQA-specific architecture, just a prompt. Modern models (Qwen3-VL, InternVL3, Gemma 3) post VQAv2-class accuracy without ever being trained on the VQAv2 training split, and additionally *explain* their answers, count reliably, read text in the image, and ground objects - none of which the metric rewards.

6.  **Where it stands in mid-2026.** VQAv2 is effectively saturated (human accuracy is ~80.8%; good models are in the mid-80s) and has largely been retired as a headline benchmark in favour of MMMU, MMBench, MMStar and RealWorldQA. What replaced it is not a better VQA dataset but a **different scoring philosophy**: multiple choice with circular evaluation, or LLM-judged free-form answers. VQA survives as a *capability* inside every VLM evaluation, and as a deployment pattern with its own operational demands (answerability, calibration, cost).

------------------------------------------------------------------------

## 4. Evaluation Metrics

**VQA accuracy.** Ten annotators answer each question. The official metric for a predicted answer $a$ is

$$\mathrm{Acc}(a) = \min\!\left(\frac{\#\{\text{humans who said } a\}}{3},\; 1\right)$$

averaged over all $\binom{10}{9}$ leave-one-annotator-out subsets - which, worked through, is the same as averaging the above over the ten “one annotator held out” variants. Three humans agreeing gives full credit; one gives a third.

The consequences are worth internalising:

- **The ceiling is not 100%.** Humans score ~80.8% against each other, because annotators disagree (“2” vs “two”, “beach” vs “seaside”).
- **It rewards the modal answer**, not the correct one. If the crowd is wrong, the model must be wrong too.
- **It is exact-match after normalisation**, so *the normaliser is part of the metric*. The official one lowercases, strips articles and most punctuation, expands contractions (“dont” -\> “do not”), and maps number words to digits (“two” -\> “2”). Skip it and a correct model loses 5-15 points.

**The generative-model penalty.** The metric was designed for classifiers. A VLM that answers “There are two cats.” scores **zero** against the reference “2” unless you either prompt it into short-answer mode or post-process. That is not a capability gap; it is a formatting gap, and section 11 measures exactly how large it is.

**Answerability (VizWiz).** Predicting *whether* a question can be answered is scored separately, with average precision. Ignore it and your assistive app confidently answers questions about a photo of a thumb.

**Consistency (VQAv2) and priors (VQA-CP).** VQAv2 pairs each question with two images that have different answers, so answering from the question alone is punished. VQA-CP goes further by making the train and test answer distributions differ per question type; accuracy on it is the honest measure of whether a model is looking at the image.

The cell below implements the official normaliser and the accuracy formula, and demonstrates the generative penalty.

------------------------------------------------------------------------

``` python
import re

# The official VQA normalisation, condensed. It IS part of the metric: publish it with
# any number you report, because a different normaliser is a different benchmark.
_CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't", "couldve": "could've",
    "couldnt": "couldn't", "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hasnt": "hasn't", "havent": "haven't", "hes": "he's", "isnt": "isn't",
    "its": "it's", "shes": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
    "thats": "that's", "theres": "there's", "theyre": "they're", "wasnt": "wasn't",
    "werent": "weren't", "whats": "what's", "wont": "won't", "youre": "you're",
}
_DIGITS = {
    "none": "0", "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[;/\[\]\"{}()=+\\_\-><@`,?!.']")
_COMMA_NUM = re.compile(r"(\d)(,)(\d)")


def vqa_normalise(answer):
    "The official VQA answer normalisation: punctuation, articles, contractions, digits."
    text = answer.replace("\n", " ").replace("\t", " ").strip().lower()
    text = _COMMA_NUM.sub(r"\1\3", text)          # 1,000 -> 1000 before punctuation strip
    text = _PUNCT.sub("", text)
    words = []
    for w in text.split():
        w = _DIGITS.get(w, w)
        if w in _ARTICLES:
            continue
        words.append(_CONTRACTIONS.get(w, w))
    return " ".join(words)


def vqa_accuracy(prediction, human_answers):
    "Official VQA accuracy: min(#matching humans / 3, 1), averaged leave-one-annotator-out."
    pred = vqa_normalise(prediction)
    golds = [vqa_normalise(a) for a in human_answers]
    scores = []
    for i in range(len(golds)):
        others = golds[:i] + golds[i + 1:]          # hold out one annotator
        scores.append(min(sum(g == pred for g in others) / 3.0, 1.0))
    return sum(scores) / len(scores)


HUMANS = ["2", "2", "2", "two", "2", "2", "2", "2", "2", "3"]
print(f"{'prediction':44s} {'accuracy':>9s}")
for pred in ["2", "two", "There are two cats.", "The image shows 2 cats.", "3", "several"]:
    print(f"{pred!r:44s} {vqa_accuracy(pred, HUMANS):9.2f}")
print("\n'two' scores 1.0 only because the normaliser maps it to '2'.")
print("'There are two cats.' scores 0.00 - the whole string must match. That is the\n"
      "generative-model penalty, and it is a formatting problem, not a vision problem.")
```

    prediction                                    accuracy
    '2'                                               1.00
    'two'                                             1.00
    'There are two cats.'                             0.00
    'The image shows 2 cats.'                         0.00
    '3'                                               0.30
    'several'                                         0.00

    'two' scores 1.0 only because the normaliser maps it to '2'.
    'There are two cats.' scores 0.00 - the whole string must match. That is the
    generative-model penalty, and it is a formatting problem, not a vision problem.

## 5. Datasets

| Dataset | Contents | Size | Scope | License | Typical use |
|----|----|----|----|----|----|
| [VQAv2](https://huggingface.co/datasets/lmms-lab/VQAv2) | COCO images, 10 answers per question, balanced pairs | 1.1M QA / 200k images | en | CC-BY 4.0 | **The** benchmark; this notebook evaluates on its validation split |
| [VizWiz-VQA](https://huggingface.co/datasets/lmms-lab/VizWiz-VQA) | Photos taken by **blind users** + spoken questions | 31k | en | CC-BY 4.0 | Real-world quality; **answerability** prediction |
| [GQA](https://huggingface.co/datasets/lmms-lab/GQA) | Compositional questions from scene graphs, with programs | 22M | en | CC-BY 4.0 | Relational and multi-step reasoning; less prior-exploitable |
| [OK-VQA](https://okvqa.allenai.org/) / [A-OKVQA](https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA) | Questions needing **outside knowledge** | 14k / 25k | en | CC-BY 4.0 | Knowledge-augmented VQA |
| [TextVQA](https://huggingface.co/datasets/lmms-lab/TextVQA) | Questions requiring reading text in the image | 45k | en | CC-BY 4.0 | OCR-dependent VQA |
| [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) | Questions over charts, incl. arithmetic | 32k | en | GPL-3.0 | Numeric reasoning on plots |
| [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA) | Multimodal science MCQs with explanations | 21k | en | CC-BY-NC-SA | Chain-of-thought VQA |
| [VQA-CP v2](https://www.cs.virginia.edu/~an7s3/vqacp/) | VQAv2 re-split so train/test priors differ | 600k | en | CC-BY 4.0 | **Language-prior** diagnosis |
| [POPE](https://huggingface.co/datasets/lmms-lab/POPE) | Balanced object-existence yes/no questions | 9k | en | MIT | Hallucination, framed as VQA |
| [MMMU](https://huggingface.co/datasets/MMMU/MMMU) | College-level multi-discipline MCQs | 11.5k | en | Apache 2.0 | What replaced VQAv2 as the headline benchmark |

This notebook streams the **VQAv2 validation** split (ungated parquet on the Hub) and scores a small sample with the official metric. Streaming matters: the full validation split is tens of gigabytes of COCO images, and we need a few dozen.

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

Where to look: the **[OpenVLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)** (VQAv2 is no longer a headline column - MMBench, MMMU and MMStar replaced it), the **[VQA Challenge leaderboard](https://eval.ai/web/challenges/challenge-page/830/leaderboard)** for the historical numbers, and **[VizWiz](https://eval.ai/web/challenges/challenge-page/2185/overview)** for the assistive setting.

| Model | Params | License | Answer mode | VQAv2 (test-dev) | Best for |
|----|----|----|----|----|----|
| [ViLT-B/32 VQA](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa) | 0.11B | Apache 2.0 | classification, 3129 answers | ~71% | the fast baseline; ~0.5 GB and milliseconds per question |
| [BLIP VQA base](https://huggingface.co/Salesforce/blip-vqa-base) | 0.38B | BSD-3 | generative (short) | ~78% | small generative VQA; open vocabulary |
| [BLIP-2 OPT-2.7B](https://huggingface.co/Salesforce/blip2-opt-2.7b) | 3.7B | MIT | generative | ~82% zero-shot | historical; the frozen-LLM bridge |
| [LLaVA-1.5-7B](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | 7B | LLaMA | generative | ~78-80% | the instruction-tuned turning point |
| [SmolVLM-500M](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) | 0.5B | Apache 2.0 | generative | competitive | edge deployment; ~1 GB |
| **[Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)** | 2B | Apache 2.0 | generative | well past 80% | this notebook’s modern model; also counts, reads and grounds |
| [InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B-hf) | 2B | MIT | generative | well past 80% | the other open lineage |
| [PaliGemma 2](https://huggingface.co/google/paligemma2-3b-pt-448) | 3-10B | Gemma terms | generative | strong when fine-tuned | a *base* model built to be fine-tuned per task |
| Frontier VLMs (GPT-5.x, Gemini 3, Claude) | closed | API | generative | saturated | the ceiling; not measured on VQAv2 any more |

**Who wins what.** On the **metric**, a fine-tuned classifier is startlingly competitive: ViLT at 0.11B gets ~71% where a 2B general VLM gets mid-80s, at a hundredth of the compute - because the metric rewards exactly the short modal answers a classifier is built to emit. On **capability**, it is not close: ViLT can only ever say one of 3129 strings, cannot read text in an image, cannot count past what it memorised, and cannot say “I do not know”. On **cost per question**, ViLT is milliseconds and a general VLM is hundreds of milliseconds to seconds.

That gap between “wins the benchmark cheaply” and “is actually useful” is the most interesting thing about VQA, and section 11 makes it concrete.

**What fits this 12 GB box.** Everything in sections 8-10: ViLT (0.5 GB), BLIP-VQA-base (1.5 GB) and Qwen3-VL-2B (4.3 GB). BLIP-2 OPT-2.7B is a 30 GB download for a 3.7B model (the repo carries fp32 and fp16 copies) and LLaVA-1.5-7B needs 4-bit here; both are landscape entries rather than runnable cells.

------------------------------------------------------------------------

## 7. Setup

Every model loads through Hugging Face `transformers` - no vendor packages. Package roles:

- `transformers` (\>=5.13) + `torch` - ViLT, BLIP-VQA, Qwen3-VL
- `accelerate` - `device_map` placement
- `datasets` - the streamed VQAv2 validation sample
- `pillow` - image handling
- `pyecharts` + `pandas` - benchmark chart and table

One dataset note: VQAv2 validation is streamed with `streaming=True` and `.take(N)`, so only the images actually scored are downloaded. Materialising the split would pull tens of gigabytes onto a machine with 12 GB of RAM.

All downloads land in `DL_tasks/datasets/`, which is gitignored.

------------------------------------------------------------------------

``` python
# Everything runs through Hugging Face transformers - no model-specific packages.
# %pip install -q torch transformers accelerate datasets pillow pandas pyecharts
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
        print(f"VRAM {tag:20s} {alloc:5.2f} GB allocated / {reserved:5.2f} GB reserved")


def free_memory():
    "Collect garbage and hand freed VRAM back to the CUDA allocator.\n\n    Call right after `del`-ing a model you are done with: `del model; free_memory()`.\n    `del` drops the Python reference; this reclaims the RAM and releases the VRAM.\n    "
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    # glibc keeps freed CPU allocations in its arenas instead of returning them
    # to the OS, so RSS compounds across model sections. malloc_trim(0) hands the
    # freed arenas back. See dl-visualization-and-memory.md.
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
from IPython.display import display

# VQAv2 validation, streamed: only the items we score get downloaded. The full split is
# tens of GB of COCO images and would not fit comfortably on this box.
N_EVAL = 24
stream = load_dataset("lmms-lab/VQAv2", split="validation", streaming=True, cache_dir=HF_CACHE)

items = []
for row in stream.take(N_EVAL):
    items.append({
        "image": row["image"].convert("RGB"),
        "question": row["question"],
        "answers": [a["answer"] for a in row["answers"]],   # 10 human annotators
        "answer_type": row["answer_type"],                  # yes/no | number | other
        "gold": row["multiple_choice_answer"],
    })

print(f"{len(items)} VQAv2 validation items")
from collections import Counter
print("answer types:", Counter(i["answer_type"] for i in items))
for it in items[:3]:
    display(it["image"].resize((260, int(260 * it["image"].height / it["image"].width))))
    print(f"Q: {it['question']}\n  humans: {it['answers']}\n  gold  : {it['gold']}\n")
```

    24 VQAv2 validation items
    answer types: Counter({'other': 11, 'yes/no': 8, 'number': 5})

![](04_Visual_Question_Answering_files/figure-html/cell-5-output-8.png)

    Q: Where is he looking?
      humans: ['down', 'down', 'at table', 'skateboard', 'down', 'table', 'down', 'down', 'down', 'down']
      gold  : down

![](04_Visual_Question_Answering_files/figure-html/cell-5-output-10.png)

    Q: What are the people in the background doing?
      humans: ['spectating', 'watching', 'watching', 'watching', 'watching', 'watching', 'watching', 'watching', 'watching', 'watching']
      gold  : watching

![](04_Visual_Question_Answering_files/figure-html/cell-5-output-12.png)

    Q: What is he on top of?
      humans: ['table', 'table', 'table', 'picnic table', 'picnic table', 'picnic table', 'picnic table', 'picnic table', 'skateboard', 'picnic table']
      gold  : picnic table

## 8. ViLT - VQA as classification over 3129 answers

ViLT (Kim et al., ICML 2021) is the minimal vision-language transformer: no CNN, no region detector, just a linear projection of image patches concatenated with word embeddings into a single transformer. Dropping the detector made it roughly **10x faster** than the region-based models it matched, and it is still one of the cheapest ways to answer a visual question - 0.11B parameters, ~0.5 GB, milliseconds per question.

The VQA head is a classifier over the **3129 most frequent training answers**. That single design decision explains everything about its behaviour:

- It is fast and it gives you a **calibrated-ish softmax** you can threshold or take a top-k from - genuinely useful in production, and something no generative VLM offers for free.
- It **cannot** answer anything outside those 3129 strings. Not a name, not a phrase, not “I cannot tell”.
- It has no idea what text in the image says, and it counts only as well as it memorised.

The cell below prints the top-5 with probabilities, because that distribution is the honest picture of what a classifier VQA model knows.

------------------------------------------------------------------------

``` python
from transformers import ViltForQuestionAnswering, ViltProcessor

vilt_id = "dandelin/vilt-b32-finetuned-vqa"
vilt_proc = ViltProcessor.from_pretrained(vilt_id, cache_dir=HF_CACHE)
vilt = ViltForQuestionAnswering.from_pretrained(vilt_id, cache_dir=HF_CACHE).to(device).eval()
vram("vilt loaded")
print("answer vocabulary:", len(vilt.config.id2label), "classes")


def vilt_answer(image, question, top_k=5):
    "Return (best answer, [(answer, probability), ...]) - the softmax is the point."
    inputs = vilt_proc(image, question, return_tensors="pt").to(device)
    with torch.inference_mode():
        logits = vilt(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, top_k)
    ranked = [(vilt.config.id2label[i.item()], round(p.item(), 3))
              for p, i in zip(top.values, top.indices)]
    return ranked[0][0], ranked


for it in items[:4]:
    t0 = time.perf_counter()
    best, ranked = vilt_answer(it["image"], it["question"])
    print(f"Q: {it['question']}")
    print(f"  [{(time.perf_counter() - t0) * 1000:.0f} ms] {best!r}  "
          f"acc {vqa_accuracy(best, it['answers']):.2f}   top5 {ranked}")

# What a fixed vocabulary means in practice: ask something outside it.
print("\nout-of-vocabulary probe")
for q in ["What is written on the sign?", "What is the name of this person?"]:
    best, ranked = vilt_answer(items[0]["image"], q)
    print(f"  {q!r} -> {best!r} (top5 {[a for a, _ in ranked]})")
print("It always answers, always from the same 3129 strings, and never says 'I cannot tell'.")

del vilt, vilt_proc
free_memory()
vram("after vilt")
```

    VRAM vilt loaded           0.47 GB allocated /  0.52 GB reserved
    answer vocabulary: 3129 classes
    Q: Where is he looking?
      [442 ms] 'down'  acc 1.00   top5 [('down', 0.76), ('up', 0.08), ('ground', 0.06), ('air', 0.012), ('skateboard', 0.01)]
    Q: What are the people in the background doing?
      [77 ms] 'watching'  acc 1.00   top5 [('watching', 0.939), ('standing', 0.021), ('skateboarding', 0.012), ('sitting', 0.006), ('skating', 0.004)]
    Q: What is he on top of?
      [11 ms] 'table'  acc 0.90   top5 [('table', 0.59), ('picnic table', 0.311), ('bench', 0.072), ('skateboard', 0.015), ('rail', 0.002)]
    Q: What website copyrighted the picture?
      [13 ms] 'unknown'  acc 0.00   top5 [('unknown', 0.101), ('photographer', 0.096), ("don't know", 0.064), ('2012', 0.061), ('not sure', 0.048)]

    out-of-vocabulary probe
      'What is written on the sign?' -> 'nothing' (top5 ['nothing', 'vans', 'stop', 'unknown', "can't tell"])
      'What is the name of this person?' -> 'skateboarder' (top5 ['skateboarder', 'skateboarding', "don't know", 'male', 'unknown'])
    It always answers, always from the same 3129 strings, and never says 'I cannot tell'.
    VRAM after vilt            0.01 GB allocated /  0.01 GB reserved

## 9. BLIP-VQA - the small generative answer

BLIP (Salesforce, 2022) reframed the answer as **text to decode**. The VQA variant encodes the image, cross-attends the question, and generates an answer with a decoder - so the output vocabulary is unbounded. `Salesforce/blip-vqa-base` is 0.38B parameters and was fine-tuned on VQAv2, which means it also learned the *style* the benchmark wants: one or two words, no sentence.

That combination - generative but benchmark-formatted - is why it beats ViLT on the metric while remaining small. It still cannot read text well, still cannot refuse, and it has no calibrated confidence.

The larger `Salesforce/blip-vqa-capfilt-large` (0.47B) trades a little speed for a couple of accuracy points.

------------------------------------------------------------------------

``` python
from transformers import BlipForQuestionAnswering, BlipProcessor

blip_id = "Salesforce/blip-vqa-base"
blip_proc = BlipProcessor.from_pretrained(blip_id, cache_dir=HF_CACHE)
blip = BlipForQuestionAnswering.from_pretrained(
    blip_id, dtype=dtype, cache_dir=HF_CACHE
).to(device).eval()
vram("blip-vqa loaded")


def blip_answer(image, question, max_new_tokens=10):
    "Generate a short answer. BLIP-VQA was fine-tuned to answer in benchmark style."
    inputs = blip_proc(image, question, return_tensors="pt").to(device, dtype)
    with torch.inference_mode():
        out = blip.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=3)
    return blip_proc.decode(out[0], skip_special_tokens=True).strip()


for it in items[:4]:
    t0 = time.perf_counter()
    ans = blip_answer(it["image"], it["question"])
    print(f"Q: {it['question']}\n  [{(time.perf_counter() - t0) * 1000:.0f} ms] {ans!r}  "
          f"acc {vqa_accuracy(ans, it['answers']):.2f}  (humans: {it['gold']!r})")

# Open vocabulary in action: the same out-of-vocabulary probe ViLT could not touch.
print("\nout-of-vocabulary probe")
for q in ["What is written on the sign?", "Describe the weather in one word."]:
    print(f"  {q!r} -> {blip_answer(items[0]['image'], q)!r}")

del blip, blip_proc
free_memory()
vram("after blip-vqa")
```

    VRAM blip-vqa loaded       0.76 GB allocated /  0.76 GB reserved
    Q: Where is he looking?
      [427 ms] 'down'  acc 1.00  (humans: 'down')
    Q: What are the people in the background doing?
      [92 ms] 'watching'  acc 1.00  (humans: 'watching')
    Q: What is he on top of?
      [35 ms] 'picnic table'  acc 1.00  (humans: 'picnic table')
    Q: What website copyrighted the picture?
      [53 ms] 'foodiebakercom'  acc 1.00  (humans: 'foodiebakercom')

    out-of-vocabulary probe
      'What is written on the sign?' -> 'jordi verdugo'
      'Describe the weather in one word.' -> 'cloudy'
    VRAM after blip-vqa        0.01 GB allocated /  0.01 GB reserved

## 10. Qwen3-VL-2B - VQA by prompting a general model

The 2026 answer: there is no VQA model, only a VLM and a prompt. Qwen3-VL-2B was never fine-tuned on the VQAv2 training split, has no answer vocabulary and no VQA head - and it answers better than either model above while also counting reliably, reading text in the image, and explaining itself.

The catch is entirely about **format**. Ask it a VQA question and it will answer in a sentence, which scores zero on the metric. So the prompt has to do the work the fine-tune used to do:

    Answer the question using a single word or phrase. <question>

That exact line (or a close variant) is what every published VLM VQAv2 number uses. It is not cheating - it is telling the model which of its abilities you are measuring - but it *is* a reminder that the benchmark measures a formatting convention as much as a capability.

Section 11 quantifies what the prompt is worth.

------------------------------------------------------------------------

``` python
from transformers import AutoModelForImageTextToText, AutoProcessor

qwen_id = "Qwen/Qwen3-VL-2B-Instruct"
qwen_proc = AutoProcessor.from_pretrained(qwen_id, cache_dir=HF_CACHE)
qwen = AutoModelForImageTextToText.from_pretrained(
    qwen_id, dtype=dtype, device_map=device, low_cpu_mem_usage=True, cache_dir=HF_CACHE
).eval()
vram("qwen3-vl loaded")

# The prompt every published VLM VQAv2 number uses, in one form or another.
SHORT_ANSWER = "Answer the question using a single word or phrase."


def qwen_answer(image, question, style=SHORT_ANSWER, max_new_tokens=48):
    "Ask a general VLM a VQA question. `style` is what turns prose into a benchmark answer."
    prompt = f"{style} {question}" if style else question
    inputs = qwen_proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image", "image": image},
                                      {"type": "text", "text": prompt}]}],
        add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
    ).to(qwen.device)
    n = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = qwen.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return qwen_proc.batch_decode(out[:, n:], skip_special_tokens=True)[0].strip()


for it in items[:4]:
    t0 = time.perf_counter()
    ans = qwen_answer(it["image"], it["question"])
    print(f"Q: {it['question']}\n  [{(time.perf_counter() - t0) * 1000:.0f} ms] {ans!r}  "
          f"acc {vqa_accuracy(ans, it['answers']):.2f}  (humans: {it['gold']!r})")

# Everything the metric cannot see, from the same weights and the same image.
it = items[0]
print("\nwhat the benchmark never asks for")
for q in ["Answer the question, then explain your reasoning in one sentence: " + it["question"],
          "What text, if any, is visible in this image?",
          "Could this question be answered from this image? Reply 'answerable' or 'unanswerable': "
          "What is the person's name?"]:
    print(f"  > {q[:80]}\n    {qwen_answer(it['image'], q, style=None, max_new_tokens=80)}")
```

    VRAM qwen3-vl loaded       4.26 GB allocated /  4.27 GB reserved
    Q: Where is he looking?
      [239 ms] 'down'  acc 1.00  (humans: 'down')
    Q: What are the people in the background doing?
      [165 ms] 'watching'  acc 1.00  (humans: 'watching')
    Q: What is he on top of?
      [180 ms] 'picnic table'  acc 1.00  (humans: 'picnic table')
    Q: What website copyrighted the picture?
      [228 ms] 'http://foodiebaker.com'  acc 0.30  (humans: 'foodiebakercom')

    what the benchmark never asks for
      > Answer the question, then explain your reasoning in one sentence: Where is he lo
        He is looking down at the skateboard as he is in mid-air, likely to maintain balance and control during the trick.

    The reasoning is based on the visual evidence: the skateboarder is in a dynamic, airborne position, and his body is oriented forward with his arms extended, which is typical for maintaining balance during a trick. His gaze is directed downward, which is consistent with the action of looking at
      > What text, if any, is visible in this image?
        © Retil Johansen
      > Could this question be answered from this image? Reply 'answerable' or 'unanswer
        unanswerable

## 11. What the prompt is worth

Before the benchmark, one measurement that explains most of the confusion around VQA numbers.

Take the same model, the same images and the same questions, and change only the instruction:

1.  **No style prompt** - just the question. The model answers naturally, in a sentence.
2.  **Short-answer prompt** - the standard benchmark instruction.
3.  **Short-answer prompt plus a light post-process** - strip a trailing period, drop a leading “The answer is”.

The gap between (1) and (2) is the **generative-model penalty**: capability held constant, points appearing purely because of output format. It is routinely 20-40 accuracy points, which is larger than the gap between most models on any leaderboard.

The honest reading: VQAv2 accuracy compares *formatting-plus-capability*, and any comparison across models must fix the prompt. That is exactly what the benchmark in section 12 does.

------------------------------------------------------------------------

``` python
def post_process(answer):
    "Strip the conversational scaffolding a chat model puts around a short answer."
    text = answer.strip().strip(".").strip()
    for prefix in ("the answer is", "answer:", "it is", "there are", "there is"):
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()
    return text.split("\n")[0].strip().strip(".")


styles = {
    "no style prompt": (None, False),
    "short-answer prompt": (SHORT_ANSWER, False),
    "short-answer + post-process": (SHORT_ANSWER, True),
}

print(f"{'condition':30s} {'VQA accuracy':>13s}")
examples = {}
for name, (style, post) in styles.items():
    scores, first = [], None
    for it in items[:12]:
        raw = qwen_answer(it["image"], it["question"], style=style)
        ans = post_process(raw) if post else raw
        scores.append(vqa_accuracy(ans, it["answers"]))
        if first is None:
            first = raw
    examples[name] = first
    print(f"{name:30s} {sum(scores) / len(scores):13.3f}")

print(f"\nsame question ({items[0]['question']!r}), same weights, three prompts:")
for name, ans in examples.items():
    print(f"  {name:30s} -> {ans!r}")
```

    condition                       VQA accuracy
    no style prompt                        0.000
    short-answer prompt                    0.850
    short-answer + post-process            0.850

    same question ('Where is he looking?'), same weights, three prompts:
      no style prompt                -> 'Based on the image provided, the skateboarder is looking down at the skateboard and the ground beneath him.\n\nThis is a common technique for a skateboarder during a trick, especially when performing a jump or a grind. The purpose of looking down'
      short-answer prompt            -> 'down'
      short-answer + post-process    -> 'down'

## 12. Head-to-head Benchmark

Three models on the **same VQAv2 validation items**, the **same official normaliser**, and the **same accuracy formula**: ViLT (classification), BLIP-VQA (small generative, VQA-fine-tuned) and Qwen3-VL-2B (general VLM with the standard short-answer prompt). Each model is loaded, measured and **freed before the next one loads**, so VRAM stays flat.

Reported per model: overall VQA accuracy, accuracy broken down by **answer type** (yes/no, number, other - they behave very differently), milliseconds per question, and model size. The answer-type breakdown is the most informative column: yes/no is where language priors live, number is where models are worst, and “other” is where open vocabulary pays off.

**Read this as a smoke test, not a leaderboard.** Two dozen items gives an accuracy with an uncertainty of roughly plus or minus 10 points, and the published test-dev numbers come from 107k questions through the EvalAI server. What the sample *does* show honestly is the shape of the trade: cost per question, the answer-type profile, and what each architecture can and cannot say.

------------------------------------------------------------------------

``` python
del qwen, qwen_proc
free_memory()
vram("before benchmark")

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BlipForQuestionAnswering,
    BlipProcessor,
    ViltForQuestionAnswering,
    ViltProcessor,
)


def load_vilt():
    "Classification over 3129 answers. 0.11B."
    proc = ViltProcessor.from_pretrained(vilt_id, cache_dir=HF_CACHE)
    model = ViltForQuestionAnswering.from_pretrained(vilt_id, cache_dir=HF_CACHE).to(device).eval()

    def answer(image, question):
        inputs = proc(image, question, return_tensors="pt").to(device)
        with torch.inference_mode():
            logits = model(**inputs).logits[0]
        return model.config.id2label[int(logits.argmax())]

    return answer, [model, proc]


def load_blip():
    "Generative, fine-tuned on VQAv2 so it already answers in benchmark style. 0.38B."
    proc = BlipProcessor.from_pretrained(blip_id, cache_dir=HF_CACHE)
    model = BlipForQuestionAnswering.from_pretrained(
        blip_id, dtype=dtype, cache_dir=HF_CACHE).to(device).eval()

    def answer(image, question):
        inputs = proc(image, question, return_tensors="pt").to(device, dtype)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=10, num_beams=3)
        return proc.decode(out[0], skip_special_tokens=True).strip()

    return answer, [model, proc]


def load_qwen():
    "General VLM, never trained on VQAv2, with the standard short-answer prompt. 2B."
    proc = AutoProcessor.from_pretrained(qwen_id, cache_dir=HF_CACHE)
    model = AutoModelForImageTextToText.from_pretrained(
        qwen_id, dtype=dtype, device_map=device, low_cpu_mem_usage=True, cache_dir=HF_CACHE).eval()

    def answer(image, question):
        inputs = proc.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": image},
                                          {"type": "text", "text": f"{SHORT_ANSWER} {question}"}]}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        ).to(model.device)
        n = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        return post_process(proc.batch_decode(out[:, n:], skip_special_tokens=True)[0])

    return answer, [model, proc]


def benchmark(name, loader):
    "Load, answer every item, score with the official metric, free."
    answer_fn, handles = loader()
    preds, t0 = [], time.perf_counter()
    for it in items:
        preds.append(answer_fn(it["image"], it["question"]))
    elapsed = time.perf_counter() - t0
    scores = [vqa_accuracy(p, it["answers"]) for p, it in zip(preds, items)]

    def by_type(kind):
        s = [sc for sc, it in zip(scores, items) if it["answer_type"] == kind]
        return round(sum(s) / len(s), 3) if s else None

    for h in handles:
        del h
    del answer_fn, handles
    free_memory()
    vram(f"after {name}")
    return {"model": name,
            "accuracy": round(sum(scores) / len(scores), 3),
            "yes/no": by_type("yes/no"), "number": by_type("number"), "other": by_type("other"),
            "ms_per_q": round(elapsed / len(items) * 1000, 1),
            "preds": preds}


results = [
    benchmark("vilt-b32 (classifier)", load_vilt),
    benchmark("blip-vqa-base", load_blip),
    benchmark("qwen3-vl-2b (prompted)", load_qwen),
]
vram("benchmark done")
```

    VRAM before benchmark      0.01 GB allocated /  0.01 GB reserved

    VRAM after vilt-b32 (classifier)  0.01 GB allocated /  0.01 GB reserved

    VRAM after blip-vqa-base   0.01 GB allocated /  0.01 GB reserved

    VRAM after qwen3-vl-2b (prompted)  0.01 GB allocated /  0.01 GB reserved
    VRAM benchmark done        0.01 GB allocated /  0.01 GB reserved

``` python
import pandas as pd

df = pd.DataFrame([{k: v for k, v in r.items() if k != "preds"} for r in results])
df = df.sort_values("accuracy", ascending=False)
df
```

|     | model                  | accuracy | yes/no | number | other | ms_per_q |
|-----|------------------------|----------|--------|--------|-------|----------|
| 0   | vilt-b32 (classifier)  | 0.888    | 1.0    | 0.8    | 0.845 | 22.0     |
| 1   | blip-vqa-base          | 0.888    | 1.0    | 0.8    | 0.845 | 42.2     |
| 2   | qwen3-vl-2b (prompted) | 0.808    | 1.0    | 0.8    | 0.673 | 156.6    |

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
    .add_yaxis("overall", [round(r["accuracy"] * 100, 1) for r in results])
    .add_yaxis("yes/no", [round((r["yes/no"] or 0) * 100, 1) for r in results])
    .add_yaxis("number", [round((r["number"] or 0) * 100, 1) for r in results])
    .add_yaxis("other", [round((r["other"] or 0) * 100, 1) for r in results])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"VQA accuracy by answer type on {len(items)} VQAv2 val items",
            subtitle="RTX 3060 12 GB, official normaliser - smoke test, not a leaderboard "
                     "(human ceiling is ~80.8%)",
        ),
        xaxis_opts=opts.AxisOpts(name="model", axislabel_opts=opts.LabelOpts(rotate=12)),
        yaxis_opts=opts.AxisOpts(name="VQA accuracy %", max_=100),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
    )
)
bar.render_notebook()
```

``` python
from pyecharts.charts import Scatter

# Accuracy against cost - the trade that decides which of these you would actually deploy.
scatter = Scatter()
scatter.add_xaxis([round(r["ms_per_q"], 1) for r in results])
for r in results:
    scatter.add_yaxis(
        r["model"], [[round(r["ms_per_q"], 1), round(r["accuracy"] * 100, 1)]],
        symbol_size=18, label_opts=opts.LabelOpts(is_show=False),
    )
scatter.set_global_opts(
    title_opts=opts.TitleOpts(title="VQA accuracy vs latency",
                              subtitle="up and to the left is better; note the log-ish spread in cost"),
    xaxis_opts=opts.AxisOpts(type_="value", name="milliseconds / question"),
    yaxis_opts=opts.AxisOpts(type_="value", name="VQA accuracy %"),
    tooltip_opts=opts.TooltipOpts(trigger="item"),
)
scatter.render_notebook()
```

``` python
# The numbers hide the interesting part: read the answers, especially where they disagree.
from IPython.display import display

shown = 0
for i, it in enumerate(items):
    answers = [r["preds"][i] for r in results]
    scores = [vqa_accuracy(a, it["answers"]) for a in answers]
    if len(set(scores)) == 1 and shown >= 2:      # prefer cases where the models disagree
        continue
    display(it["image"].resize((240, int(240 * it["image"].height / it["image"].width))))
    print(f"Q: {it['question']}   humans: {it['answers'][:5]}")
    for r, a, s in zip(results, answers, scores):
        print(f"  {r['model']:24s} {a!r:28s} acc {s:.2f}")
    print()
    shown += 1
    if shown >= 5:
        break
```

![](04_Visual_Question_Answering_files/figure-html/cell-14-output-1.png)

    Q: Where is he looking?   humans: ['down', 'down', 'at table', 'skateboard', 'down']
      vilt-b32 (classifier)    'down'                       acc 1.00
      blip-vqa-base            'down'                       acc 1.00
      qwen3-vl-2b (prompted)   'down'                       acc 1.00

![](04_Visual_Question_Answering_files/figure-html/cell-14-output-3.png)

    Q: What are the people in the background doing?   humans: ['spectating', 'watching', 'watching', 'watching', 'watching']
      vilt-b32 (classifier)    'watching'                   acc 1.00
      blip-vqa-base            'watching'                   acc 1.00
      qwen3-vl-2b (prompted)   'watching'                   acc 1.00

![](04_Visual_Question_Answering_files/figure-html/cell-14-output-5.png)

    Q: What is he on top of?   humans: ['table', 'table', 'table', 'picnic table', 'picnic table']
      vilt-b32 (classifier)    'table'                      acc 0.90
      blip-vqa-base            'picnic table'               acc 1.00
      qwen3-vl-2b (prompted)   'picnic table'               acc 1.00

![](04_Visual_Question_Answering_files/figure-html/cell-14-output-7.png)

    Q: What website copyrighted the picture?   humans: ['foodiebakercom', 'foodiebakercom', 'foodiebaker', 'foodiebakercom', 'foodiebakercom']
      vilt-b32 (classifier)    'unknown'                    acc 0.00
      blip-vqa-base            'foodiebakercom'             acc 1.00
      qwen3-vl-2b (prompted)   'http://foodiebaker.com'     acc 0.30

![](04_Visual_Question_Answering_files/figure-html/cell-14-output-9.png)

    Q: What is the man doing in the street?   humans: ['crossing it', 'walking', 'walking', 'crossing', 'crossing road']
      vilt-b32 (classifier)    'walking'                    acc 1.00
      blip-vqa-base            'crossing street'            acc 0.00
      qwen3-vl-2b (prompted)   'crossing'                   acc 0.90

## 13. Live Demo: ask questions about the camera

Streams the webcam and answers a question about every frame with Qwen3-VL-2B, burning the answer onto the picture. A 2B VLM answers a short question in roughly half a second on this card, so this runs at about 1-2 FPS - fast enough to feel responsive for a yes/no question, and a good illustration of why assistive apps run VQA on a captured photo rather than on a live stream.

This is the cell people run on its own, so it opens with a `require(...)` guard naming what it needs from Setup instead of dying on a bare `NameError`. Capture notes, all measured on the knowledge-lab container: V4L2 backend with MJPEG and a warm-up read (auto-exposure needs frames to settle), never `CAP_PROP_BUFFERSIZE` (it halves the frame rate without making frames fresher), and no `cv2.imshow` because there is no GUI - frames go through `IPython.display` handles that update in place.

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


require("device", "dtype", "HF_CACHE", "free_memory", "vram", "qwen_id",
        "SHORT_ANSWER", "post_process")

import time
import torch

# opencv-python-headless is a project dependency; the headless build captures from
# V4L2 fine, it only drops the GUI windows.
import io

import cv2
from IPython.display import Image as IPyImage
from IPython.display import Pretty, display
from PIL import Image, ImageDraw, ImageFont

from transformers import AutoModelForImageTextToText, AutoProcessor

CAM = 0              # /dev/video0
WARMUP = 10          # throwaway reads - auto-exposure and white balance need to settle
STREAM_SECONDS = 20  # how long the live demo runs; interrupt the kernel to stop early
QUESTION = "What are the objects in this image?"


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
    # camera in manual mode every frame comes back dark and never adapts, so ask for the
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


_FONT = ImageFont.load_default(size=16)


def draw_lines(img, lines, pad=6):
    "Burn a few lines of text into a band across the top of a copy of `img`."
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    d.rectangle([0, 0, out.width, 19 * len(lines) + 2 * pad], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        d.text((pad, pad + 19 * i), line, fill=(255, 255, 255), font=_FONT)
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


# Re-runnable: this cell frees the model at the end, so guard the load or a second
# shift-enter raises NameError on `live_model`.
if "live_model" not in globals():
    live_proc = AutoProcessor.from_pretrained(qwen_id, cache_dir=HF_CACHE)
    live_model = AutoModelForImageTextToText.from_pretrained(
        qwen_id, dtype=dtype, device_map=device, low_cpu_mem_usage=True, cache_dir=HF_CACHE
    ).eval()
    vram("live model")


def annotate(rgb):
    "One frame -> (frame with the answer burned on, the answer again for the status line)."
    inputs = live_proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image", "image": rgb},
                                      {"type": "text", "text": f"{SHORT_ANSWER} {QUESTION}"}]}],
        add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
    ).to(live_model.device)
    n = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = live_model.generate(**inputs, max_new_tokens=24, do_sample=False)
    answer = post_process(live_proc.batch_decode(out[:, n:], skip_special_tokens=True)[0])
    return draw_lines(rgb, [f"Q: {QUESTION}"[:52], f"A: {answer}"[:52]]), answer[:40]


live_stream(annotate)

del live_model, live_proc
free_memory()
vram("final")
```

    VRAM live model            4.26 GB allocated /  4.27 GB reserved

![](04_Visual_Question_Answering_files/figure-html/cell-15-output-3.jpeg)

    frame   86    4.29 FPS   bottle

    86 frames in 20.1s -> 4.28 FPS end-to-end (camera + VLM + JPEG encode)
    VRAM final                 0.01 GB allocated /  0.01 GB reserved

## 14. Common Frameworks

VQA is the task where the ecosystem question is really an architecture question: a fine-tuned classifier, a small generative model, or a prompted general VLM. The first two are cheap enough to run on CPU and need almost no infrastructure; the third brings the entire LLM serving stack with it. The frameworks below are grouped so you can see what each choice actually costs you in dependencies.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [transformers](https://github.com/huggingface/transformers) | modelling | ViLT, BLIP-VQA and every general VLM behind one API, plus the `vqa` pipeline for the classification path | Apache 2.0 | Default, and it spans all three architectures in this notebook |
| [peft](https://github.com/huggingface/peft) + [trl](https://github.com/huggingface/trl) | modelling | LoRA on a 2-3B VLM, and plain `Trainer` fine-tuning for ViLT and BLIP-VQA on this card | Apache 2.0 | A fixed camera and a small question set. Fine-tuning then wins on accuracy, latency and cost at once |
| [datasets](https://github.com/huggingface/datasets) + [FiftyOne](https://github.com/voxel51/fiftyone) | data | Streaming VQAv2/VizWiz/OK-VQA, and looking at the images your model answers wrongly | Apache 2.0 | Always. VQA labels are ten human answers with real disagreement, and the metric is built around that |
| A retriever ([faiss](https://github.com/facebookresearch/faiss) / [LlamaIndex](https://github.com/run-llama/llama_index)) | data | Knowledge-augmented VQA: caption or embed the image, retrieve documents, answer with both in context | MIT | OK-VQA-style questions that need facts not in the image. This is a RAG problem, not a bigger-model problem |
| [vLLM](https://github.com/vllm-project/vllm) / [SGLang](https://github.com/sgl-project/sglang) | inference runtime | Continuous batching, and prefix caching that pays off enormously when many questions share one image | Apache 2.0 | Serving the VLM path. Asking twenty questions about one photo should encode it once |
| [optimum](https://github.com/huggingface/optimum) + [ONNX Runtime](https://onnxruntime.ai/) | inference runtime | ViLT and BLIP on CPU, where they are still fast enough to be practical | Apache 2.0 / MIT | The classification path. A 3129-way head on CPU is a genuinely cheap product |
| [outlines](https://github.com/dottxt-ai/outlines) / [xgrammar](https://github.com/mlc-ai/xgrammar) | inference runtime | Constrained decoding to a closed answer set, which turns a generative VLM into a classifier when you need one | Apache 2.0 | The answer must come from a known vocabulary. Also the cleanest way to force an “unanswerable” option |
| [BentoML](https://github.com/bentoml/BentoML) / [Ray Serve](https://github.com/ray-project/ray) | serving | An endpoint with image preprocessing colocated, and separate scaling for the vision and language halves | Apache 2.0 | Production |
| [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) / [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) | evaluation | VQA accuracy with the official ten-answer soft scoring, ANLS, and the VizWiz unanswerable handling | Apache 2.0 | Always. Exact-match against one reference answer understates every generative model, as section 4 shows |

**The 2026 default stack** is a prompted VLM through `transformers` while the question set is open, constrained decoding as soon as the answers are closed, vLLM once there are users, and a fine-tuned small model the moment the camera and the questions stop changing.

The common wrong turn is deploying a general VLM for a fixed, narrow question set - it costs orders of magnitude more than a fine-tuned ViLT and is often less accurate on exactly those questions. The second is not measuring the language prior: feed your model the questions with a blank image, and whatever accuracy survives is the part of your number that has nothing to do with vision.

------------------------------------------------------------------------

## 15. Going Further

- **Fine-tuning still wins on a fixed camera.** If your questions come from a small template set and the camera never moves (production line, kiosk, drone inspection), fine-tuning a small model on a few thousand of your own QA pairs beats prompting a general VLM on accuracy, latency and cost, all three. ViLT and BLIP-VQA both fine-tune on this 12 GB card; PaliGemma 2 is explicitly published as a *base* model for exactly this. LoRA through `peft` + `trl` covers the 2-3B VLMs.
- **Add an escape hatch.** Train or prompt for “unanswerable” and threshold on it. For a classifier, the softmax gives you a usable confidence directly; for a generative VLM, ask for the answer plus a confidence word, or score the answer’s log-probability. VizWiz exists because this is the difference between a useful assistive app and a harmful one.
- **Knowledge-augmented VQA.** OK-VQA questions need facts that are not in the image. The production pattern is retrieval: caption or embed the image, retrieve documents, then answer with both in context. That is a RAG pipeline with a VLM at both ends, not a bigger model.
- **Measure the prior, not just the accuracy.** Run your model on VQA-CP, or simply feed it the questions with a **blank image** and see how much accuracy survives. Whatever it scores blind is the part of your number that has nothing to do with vision.
- **Related notebooks.** `Multimodal/01_Image_Text_to_Text` (the same models, unconstrained answers, and the compliance/hallucination measurements), `Multimodal/05_Document_Question_Answering` (pages instead of photos, and ANLS instead of VQA accuracy), `Multimodal/06_Video_Text_to_Text` (questions about time), `Computer_Vision/05_Image_to_Text` (describing rather than answering), and `Computer_Vision/11_Zero_Shot_Image_Classification` (when a fixed label set is genuinely enough).

------------------------------------------------------------------------

Back to top
