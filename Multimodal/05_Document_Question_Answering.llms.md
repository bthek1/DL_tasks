# Document Question Answering

Everything to know about DocVQA: why ANLS and not accuracy, the three architectures (OCR+layout, OCR-free, and prompt-a-VLM) with runnable code for all of them, what resolution really costs on a page, and a head-to-head on real DocVQA items.

Author

Benedict Thekkel

## 1. What is Document Question Answering?

DocQA is **a document page plus a question in, an answer out**. “What is the invoice total?” -\> “\$1,240.00”. “Who is the recipient of this letter?” -\> “Dr. J. Smith”. “What was the 1974 budget for tobacco research?” -\> “\$2,500,000”.

It looks like VQA with a different picture, and it is not. Three things make it its own task:

- **The answer is usually *in* the image, as text.** So the model must read, not describe. That makes it an OCR problem wrapped in a reasoning problem, and resolution becomes the dominant variable: an 8 pt footnote is a few pixels tall at 224 px input.
- **Layout carries meaning.** In a table, “1,240.00” means something only because of which row and column it sits in. A model that reads a page as a flat left-to-right string loses the association between a key and its value.
- **Answers are text spans with messy surface forms.** “\$1,240.00” vs “1240.00” vs “1,240”. Exact match would be brutal and uninformative, so the field uses **ANLS** (section 4) instead.

**Input.** A page image - a scan, a photo, or a rendered PDF page - plus a question. Multi-page variants (MP-DocVQA, DUDE) take a whole document.

**Output.** A short answer, normally extracted from the page rather than invented.

| Neighbouring task | Difference | Typical tools |
|----|----|----|
| Visual question answering (`Multimodal/04`) | Natural photos; answers are visual facts, not text spans | ViLT, BLIP-VQA |
| Image-to-text / OCR (`Computer_Vision/05`) | Transcribes the **whole** page; no question | GOT-OCR, Florence-2, dots.ocr |
| Visual document retrieval (`Multimodal/07`) | Finds *which page* to ask about, across thousands | ColPali, ColQwen2 |
| Table question answering (`NLP/02`) | Table already parsed into text/structure | TAPAS, TAPEX |
| Image-text-to-text (`Multimodal/01`) | Same mechanics, general images, free-form answers | Qwen3-VL, InternVL3 |

**The three architectures**, all still deployed in 2026 and all runnable below:

1.  **OCR + layout-aware encoder** (LayoutLM family). Run an OCR engine, feed words *plus their bounding boxes* into a transformer, extract the answer span. Cheap, interpretable, and only as good as your OCR.
2.  **OCR-free end-to-end** (Donut, Pix2Struct). Read pixels, generate the answer. No OCR engine, no error propagation, but weak on tiny text and hard to debug.
3.  **Prompt a VLM** (Qwen3-VL, InternVL3, dots.ocr). What most new systems do in 2026, at a much higher compute cost.

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Invoice and receipt processing | Finance, AP automation (SAP Concur, Ramp, Rossum) | Invoice scan + field schema -\> line items, totals, tax | Numeric exactness; a decimal error is a payment error; audit trail |
| Insurance claims intake | Insurance | Claim forms and photos + question set -\> structured fields | Handwriting; degraded faxes; regulatory retention |
| Contract review and diligence | Legal | Contract page + “what is the termination notice period?” -\> clause | Multi-page context; citation back to the source span |
| KYC and identity verification | Banking, fintech | ID document + field questions -\> extracted fields | Fraud detection; PII handling; on-prem or in-region processing |
| Clinical records abstraction | Healthcare | Scanned chart + question -\> value | Handwriting, dense tables, PHI; wrong values are patient-safety events |
| Patent and scientific search | Research, IP | Paper page + question -\> answer with figure/table context | Formulas and tables; long documents |
| Government forms processing | Public sector | Form scan -\> field values | Volume and cost per page; accessibility obligations |
| Financial reporting and analysis | Finance | 10-K page + “what was FY24 opex?” -\> number | Table structure; numerical reasoning across rows |
| Logistics documents | Shipping | Bill of lading, customs form -\> fields | Rotation, stamps, poor scan quality; throughput |
| Enterprise knowledge assistants | Any large company | Any internal PDF + question -\> answer with citation | Retrieval first (see `Multimodal/07`), then DocQA; hallucination control |

**What the benchmark hides.** Four realities.

*Real documents are not DocVQA.* DocVQA pages are scanned industry documents at reasonable quality. Production input is phone photos at an angle, faxes at 200 dpi, stamped and handwritten annotations, and 60-page PDFs where the answer is on page 43. Deskewing, page classification and retrieval come *before* the model.

*Extraction beats generation for auditability.* A regulated pipeline usually needs to point at *where* on the page the answer came from. Extractive models give you a span and a box for free; a generative VLM gives you a fluent string that may or may not appear anywhere in the document. Grounded generation (asking the model to also return the bounding box) is the current workaround.

*Cost per page decides architecture.* LayoutLM plus a cheap OCR engine is a few milliseconds and fractions of a cent per page. A 3B VLM at high resolution is a second or more of GPU time and thousands of visual tokens. At a million pages a month that difference is the entire business case.

*Multi-page is the unsolved part.* Most benchmarks are single-page. Real questions (“what is the total across all invoices in this bundle?”) need retrieval, page ranking and cross-page aggregation - a pipeline, not a model.

------------------------------------------------------------------------

## 3. How Modern Document QA Works

1.  **OCR plus a text QA model (pre-2020).** Run Tesseract, concatenate the words, run a SQuAD-style extractive reader. Loses all layout, so key-value pairs and tables fail.

2.  **Layout-aware pretraining (LayoutLM, 2020-2022).** The idea that defined the field: add **2-D positional embeddings** for each token’s bounding box, so the model knows *where* on the page a word sits. LayoutLMv2 added the image as a third modality with a visual backbone; **LayoutLMv3** (2022) unified text and image masking and dropped the CNN in favour of linear patch embeddings. Related: LiLT (language-independent layout), DocFormer, ERNIE-Layout. All are **extractive**: they predict start and end token positions, so the answer is provably from the page.

3.  **OCR-free end-to-end (Donut, 2022).** Donut (“Document Understanding Transformer”) reads the raw page with a Swin encoder and generates the answer with a BART decoder - no OCR engine anywhere. That removes error propagation and the OCR licensing/latency cost, at the price of needing large synthetic pretraining and struggling with very small text. **Pix2Struct** (2023) generalised it by pretraining on screenshot-to-HTML parsing.

4.  **Generalist VLMs discover documents (2023-2024).** Once VLMs got **dynamic resolution and tiling** (LLaVA-NeXT AnyRes, Qwen2-VL native resolution, InternVL tiles) they could finally resolve small print, and DocVQA became just another prompt. Numbers jumped past the specialists.

5.  **The OCR specialists come back (2024-2026).** For dense pages, a small model trained at document resolution on structured targets beats a general VLM many times its size. **GOT-OCR 2.0** (0.58B) emits markdown, LaTeX and HTML tables; **dots.ocr** (1.7B), **olmOCR 2**, **DeepSeek-OCR** and **PaddleOCR-VL** are the 2025-26 generation. The winning production pattern is often a cascade: specialist OCR to structured markdown, then a text LLM to answer.

6.  **Retrieval-first document AI (2024-2026).** **ColPali** (2024) showed you can skip parsing entirely for *retrieval*: embed the page image directly with a VLM and match against the query with late interaction. The modern enterprise stack is ColPali-style retrieval -\> DocQA on the retrieved pages (see `Multimodal/07_Visual_Document_Retrieval`).

**Mid-2026 state.** DocVQA test-set ANLS is above 0.95 for the leaders, which means the benchmark is saturated and the interesting problems moved: multi-page and multi-document reasoning (MP-DocVQA, DUDE, MMLongBench-Doc), charts and infographics (ChartQA, InfographicVQA), handwriting, and grounded answers with citations. On the cost axis, the specialists are winning back ground from the generalists.

------------------------------------------------------------------------

## 4. Evaluation Metrics

**ANLS (Average Normalized Levenshtein Similarity)** is the DocVQA metric, and it exists because exact match punishes irrelevant OCR noise. For a prediction $p$ and the set of gold answers $\{g_i\}$:

$$\mathrm{NL}(p, g) = \frac{\mathrm{lev}(p, g)}{\max(|p|, |g|)}, \qquad
s = \max_i \big(1 - \mathrm{NL}(p, g_i)\big)$$

$$\mathrm{ANLS} = \begin{cases} s & \text{if } s \ge \tau \\ 0 & \text{otherwise}\end{cases}, \qquad \tau = 0.5$$

Two properties worth understanding:

- **It is character-level and forgiving of small errors.** “Washington” vs “Wash1ngton” scores 0.9, which is the point: a single OCR character slip should not zero a correct answer.
- **The threshold makes it unforgiving of *wrong* answers.** Anything below 0.5 similarity is scored 0, not 0.49. So it does not reward a model that emits vaguely similar strings, and it is not a smooth metric.

Both sides are compared **lowercased with whitespace stripped**, which is where a chatty model quietly loses everything: “The total is \$1,240.00” against “\$1,240.00” scores about 0.6 - it survives, but barely, and a longer sentence drops below threshold.

**Other metrics you will see.** Exact match and F1 on token overlap (for extractive models, following SQuAD); **field-level accuracy** for information extraction (CORD, FUNSD); **tree edit distance** (Donut’s original metric, for parsing a whole document into a structure); and for retrieval-first pipelines, nDCG@k and Recall@k on the retrieval stage (`Multimodal/07`).

**Cost.** Milliseconds per page and, for VLMs, **visual tokens per page** - the number that decides whether a million-page month is affordable.

The cell below implements ANLS and shows both of its edges.

------------------------------------------------------------------------

``` python
def levenshtein(a, b):
    "Classic edit distance, iterative two-row version. No dependency needed for this."
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1,        # deletion
                               current[j - 1] + 1,     # insertion
                               previous[j - 1] + (ca != cb)))  # substitution
        previous = current
    return previous[-1]


def anls(prediction, golds, threshold=0.5):
    "DocVQA's ANLS: best 1 - normalised edit distance over the gold answers, thresholded."
    p = " ".join(str(prediction).lower().split())
    best = 0.0
    for g in golds:
        g = " ".join(str(g).lower().split())
        if not p and not g:
            return 1.0
        nl = levenshtein(p, g) / max(len(p), len(g), 1)
        best = max(best, 1.0 - nl)
    return best if best >= threshold else 0.0


GOLD = ["$1,240.00", "1,240.00"]
print(f"{'prediction':52s} {'ANLS':>6s}")
for pred in [
    "$1,240.00",                          # exact
    "1,240.00",                           # matches the second gold
    "$1,24O.00",                          # OCR confused O for 0
    "$1240",                              # dropped separators
    "The total is $1,240.00",             # a chatty VLM
    "The invoice total shown at the bottom right of the page is $1,240.00.",
    "$980.00",                            # simply wrong
]:
    print(f"{pred!r:52s} {anls(pred, GOLD):6.2f}")
print("\nBoth edges are visible: a one-character OCR slip barely costs anything, while a\n"
      "wrong answer and a long sentence both fall under the 0.5 threshold and score zero.")
```

    prediction                                             ANLS
    '$1,240.00'                                            1.00
    '1,240.00'                                             1.00
    '$1,24O.00'                                            0.89
    '$1240'                                                0.56
    'The total is $1,240.00'                               0.00
    'The invoice total shown at the bottom right of the page is $1,240.00.'   0.00
    '$980.00'                                              0.56

    Both edges are visible: a one-character OCR slip barely costs anything, while a
    wrong answer and a long sentence both fall under the 0.5 threshold and score zero.

## 5. Datasets

| Dataset | Contents | Size | Scope | License | Typical use |
|----|----|----|----|----|----|
| [DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA) | Scanned industry documents (UCSF tobacco archive) | 50k QA / 12k pages | en | non-commercial research | **The** benchmark; ANLS on the hidden test set via RRC |
| [nielsr/docvqa_1200_examples](https://huggingface.co/datasets/nielsr/docvqa_1200_examples) | A 1200-item DocVQA slice **with OCR words and boxes** | 1000 train / 200 test | en | research | **This notebook’s eval set**; the boxes let LayoutLM run without Tesseract |
| [InfographicVQA](https://www.docvqa.org/datasets/infographicvqa) | Infographics: charts, icons, dense layout | 30k QA | en | research | Layout + numeric reasoning |
| [ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) | Real charts, incl. arithmetic questions | 32k | en | GPL-3.0 | Plot reading |
| [FUNSD](https://guillaumejaume.github.io/FUNSD/) | Noisy scanned forms with entity/relation labels | 199 forms | en | research | Form understanding, key-value linking |
| [CORD](https://huggingface.co/datasets/naver-clova-ix/cord-v2) | Receipts with structured field annotations | 1k | en, id | CC-BY 4.0 | Receipt parsing; Donut’s canonical fine-tune |
| [SROIE](https://rrc.cvc.uab.es/?ch=13) | Scanned receipts, OCR + key extraction | 1k | en | research | Information extraction |
| [MP-DocVQA](https://huggingface.co/datasets/rubentito/mp-docvqa) | Multi-page DocVQA | 46k QA | en | research | Page retrieval + answering |
| [DUDE](https://rrc.cvc.uab.es/?ch=23) | Diverse, multi-page, multi-domain, incl. unanswerable | 41k QA | en | research | The realistic hard benchmark |
| [OmniDocBench](https://github.com/opendatalab/OmniDocBench) | Page parsing quality across 9 document types | 981 pages | en, zh | Apache 2.0 | Evaluating the OCR stage itself |

This notebook evaluates on the **`nielsr/docvqa_1200_examples` test split**, chosen deliberately: it ships each page’s OCR **words and bounding boxes**, so the LayoutLM path in section 8 runs without installing Tesseract, and every architecture below sees exactly the same pages.

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

Leaderboards: **[DocVQA on RRC](https://rrc.cvc.uab.es/?ch=17&com=evaluation)** (the official ANLS test server), the **[OpenVLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)** DocVQA/OCRBench columns, and **[OmniDocBench](https://github.com/opendatalab/OmniDocBench)** for page parsing.

| Model | Params | License | Architecture | Needs OCR? | Best for |
|----|----|----|----|----|----|
| [LayoutLMv3-base](https://huggingface.co/microsoft/layoutlmv3-base) | 0.13B | CC-BY-NC-SA | text + layout + patch embeddings, **extractive** | **yes** | cheap, auditable span extraction; fine-tunes on a laptop |
| [impira/layoutlm-document-qa](https://huggingface.co/impira/layoutlm-document-qa) | 0.13B | MIT | LayoutLM + QA head, extractive | **yes** | this notebook’s extractive model; runs from provided boxes |
| [LiLT](https://huggingface.co/SCUT-DLVCLab/lilt-roberta-en-base) | 0.13B | MIT | layout stream decoupled from language | yes | multilingual forms without retraining layout |
| **[Donut base-finetuned-docvqa](https://huggingface.co/naver-clova-ix/donut-base-finetuned-docvqa)** | 0.2B | MIT | Swin encoder + BART decoder, **OCR-free** | no | this notebook’s OCR-free model; 0.8 GB |
| [Pix2Struct](https://huggingface.co/google/pix2struct-docvqa-base) | 0.28B | Apache 2.0 | screenshot parsing pretraining | no | charts and UI screenshots as well as docs |
| [GOT-OCR 2.0](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf) | 0.58B | Apache 2.0 | high-compression ViT + Qwen2-0.5B | is the OCR | page -\> markdown/LaTeX/tables, then answer with a text LLM |
| [dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) | 1.7B | MIT | layout + OCR in one VLM | is the OCR | 2025-generation parsing, multilingual |
| **[Qwen3-VL-2B](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)** | 2B | Apache 2.0 | native-resolution VLM | no | this notebook’s VLM; also explains and grounds |
| [InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B-hf) | 2B | MIT | tiled VLM | no | the other open lineage |
| [olmOCR 2](https://huggingface.co/allenai/olmOCR-2-7B-1025) | 7B | Apache 2.0 | Qwen2.5-VL fine-tune for PDFs | is the OCR | high-fidelity PDF linearisation at scale |

**Who wins what.** On **ANLS**, the big VLMs lead (0.9+ on DocVQA test) and the specialists are close behind at a fraction of the cost. On **cost per page**, LayoutLM plus a commodity OCR engine is two to three orders of magnitude cheaper than a 2B VLM at high resolution. On **auditability**, only the extractive models give you a span and a box without extra prompting. On **dense or unusual layouts** (multi-column science, Chinese newspapers, forms with stamps), the 2025-26 OCR specialists beat generalists many times their size.

**What fits this 12 GB box.** All four runnable sections: LayoutLM (1 GB), Donut (0.8 GB), Qwen3-VL-2B (4.3 GB) and GOT-OCR 2.0 (1.1 GB). None of them is close to the memory limit - the constraint in this notebook is page **resolution**, not parameters, and section 11 measures what dropping it costs.

------------------------------------------------------------------------

## 7. Setup

Every model loads through Hugging Face `transformers` - no vendor packages. Package roles:

- `transformers` (\>=5.13) + `torch` - LayoutLM (via the `document-question-answering` pipeline), Donut, Qwen3-VL, GOT-OCR 2.0
- `accelerate` - `device_map` placement
- `datasets` + `pillow` - the DocVQA slice and page images
- `pyecharts` + `pandas` - benchmark chart and table

**The Tesseract note.** The `document-question-answering` pipeline will call **Tesseract** through `pytesseract` if you hand it an image with no `word_boxes` - and Tesseract is a system binary that is not installed here. This notebook sidesteps that entirely by passing the dataset’s own OCR words and boxes into the pipeline. In production you would use a real OCR engine (Tesseract, PaddleOCR, Azure/Google Document AI, or a GOT-OCR-style model) and the quality of that stage would set the ceiling for the whole extractive path.

**Box normalisation, which fails in two different ways.** LayoutLM expects boxes as integers on a **0-1000 grid**, not pixels, and there are two distinct bugs here - one loud, one silent. Both are handled in the next cell and both are worth knowing.

*The loud one: inverted boxes crash the model.* LayoutLM embeds each box’s **height and width** (`y1 - y0`, `x1 - x0`) via an embedding lookup, so a box with `y1 < y0` becomes a negative index. That is `IndexError: index out of range in self` on CPU and a **device-side assert** on CUDA - which also corrupts the CUDA context, so the kernel has to be restarted before anything else will run. This is not a hypothetical: **21% of the boxes in this dataset arrive inverted**, because the OCR emits near-zero-height boxes for baseline text. Sorting each box’s corners before use is mandatory, not defensive.

*The silent one: normalising already-normalised boxes.* Scaling by the page’s pixel dimensions is the obvious move and is wrong for this dataset - its boxes are already on the 0-1000 grid (max coordinates 842-994 on pages spanning 850-2156 px). Dividing again never raises; it just shrinks every box toward the origin and quietly degrades the layout signal. Measured over all 200 test items it costs 0.030 ANLS. Always check whether your source boxes are pixels or already normalised, rather than assuming.

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

# A 200-item DocVQA test slice that ships its own OCR words and boxes, so the extractive
# path runs without a Tesseract install and every model sees identical pages.
ds = load_dataset("nielsr/docvqa_1200_examples", split="test", cache_dir=HF_CACHE)

N_EVAL = 20


def clamp_box(box):
    """This dataset's boxes are ALREADY on LayoutLM's 0-1000 grid. Sort, round, clamp.

    Two traps, and the loud one is the `sorted` calls. LayoutLM does not embed the box
    corners; it embeds the box *dimensions*, as embedding LOOKUPS:

        h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])   # y1 - y0
        w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])   # x1 - x0

    So a box with y1 < y0 is a NEGATIVE index into an embedding table. Here 21% of boxes
    (911 of 4407 over the first 20 pages) arrive inverted - the OCR emits near-zero-height
    boxes for baseline text and rounding lands the wrong way. Clamping each coordinate
    independently preserves the inversion, so the model dies on page 0: `IndexError: index
    out of range in self` on CPU, and on CUDA a device-side assert, which also poisons the
    CUDA context so the kernel must be restarted before anything else will run.

    The quiet trap is dividing by the page's pixel width/height, which looks obviously
    right and is wrong: these boxes are already normalised. Max coordinates sit at 842-994
    on pages spanning 850-2156 px, so they cannot be pixels - dividing again shrinks every
    box toward the origin. It never raises. Measured over all 200 test items, the extra
    division costs 0.030 ANLS (0.396 vs 0.426) and 6 answered questions.
    """
    x0, y0, x1, y1 = box
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return [max(0, min(1000, int(round(v)))) for v in (x0, y0, x1, y1)]


items = []
for row in ds.select(range(N_EVAL)):
    page = row["image"].convert("RGB")
    items.append({
        "id": row["id"],
        "image": page,
        "question": row["query"]["en"],          # the queries are provided in several languages
        "answers": row["answers"],               # several accepted surface forms
        "word_boxes": [(word, clamp_box(box))
                       for word, box in zip(row["words"], row["bounding_boxes"])],
    })

print(f"{len(items)} DocVQA test items")
print("page sizes:", sorted({it['image'].size for it in items})[:4], "...")
print("words on page 0:", len(items[0]["word_boxes"]))
inverted = sum(b[3] < b[1] or b[2] < b[0]
               for row in ds.select(range(N_EVAL)) for b in row["bounding_boxes"])
print(f"inverted boxes repaired by the sort: {inverted}")

for it in items[:2]:
    display(it["image"].resize((420, int(420 * it["image"].height / it["image"].width))))
    print(f"Q: {it['question']}\n  answers: {it['answers']}\n")
```

    20 DocVQA test items
    page sizes: [(850, 1644), (1370, 1480), (1653, 2339), (1679, 2340)] ...
    words on page 0: 104
    inverted boxes repaired by the sort: 919

![](05_Document_Question_Answering_files/figure-html/cell-5-output-2.png)

    Q: What the location address of NSDA?
      answers: ['1128 SIXTEENTH ST., N. W., WASHINGTON, D. C. 20036', '1128 sixteenth st., N. W., washington, D. C. 20036']

![](05_Document_Question_Answering_files/figure-html/cell-5-output-4.png)

    Q: According to budget request summary what is total amount of other expenses??
      answers: ['$975.00', '975.00']

## 8. LayoutLM - extractive QA over words and boxes

The classical document-AI architecture, and still the cheapest by a wide margin. LayoutLM (Microsoft, 2020) is BERT with **2-D positional embeddings**: each token carries its bounding box, so the model knows that “Total” sits directly to the left of “\$1,240.00” even though a flat reading order might separate them by half a page.

The head is **extractive** - it predicts a start and an end token in the OCR word sequence - which has two consequences that matter operationally:

- The answer is provably a span of the page. You get the exact words, and the boxes tell you *where*, which is what an auditor wants.
- It **cannot** answer anything that is not literally written there. No arithmetic, no summarising, no “not stated”.

`impira/layoutlm-document-qa` is a LayoutLM fine-tuned on DocVQA and SQuAD. The `document-question-answering` pipeline drives it; passing `word_boxes` skips its Tesseract call, which is what makes this cell runnable here.

------------------------------------------------------------------------

``` python
from transformers import pipeline

layout_id = "impira/layoutlm-document-qa"
layout = pipeline("document-question-answering", model=layout_id, device=device,
                  model_kwargs={"cache_dir": HF_CACHE})
vram("layoutlm loaded")


def layout_answer(item, top_k=3):
    "Extractive answer plus the pipeline's score. `word_boxes` avoids the Tesseract call."
    out = layout(image=item["image"], question=item["question"],
                 word_boxes=item["word_boxes"], top_k=top_k)
    out = out if isinstance(out, list) else [out]
    return out[0]["answer"], [(o["answer"], round(o["score"], 3)) for o in out]


for it in items[:4]:
    t0 = time.perf_counter()
    best, ranked = layout_answer(it)
    print(f"Q: {it['question']}")
    print(f"  [{(time.perf_counter() - t0) * 1000:4.0f} ms] {best!r}  ANLS {anls(best, it['answers']):.2f}")
    print(f"  candidates: {ranked}   gold: {it['answers'][:2]}\n")

# What "extractive" costs you: a question whose answer is not a span on the page.
print("questions an extractive model structurally cannot answer")
for q in ["How many pages does this document have in total?",
          "Summarise this document in one sentence."]:
    probe = dict(items[0], question=q)
    print(f"  {q!r} -> {layout_answer(probe)[0]!r}")

del layout
free_memory()
vram("after layoutlm")
```

    VRAM layoutlm loaded       0.51 GB allocated /  0.56 GB reserved
    Q: What the location address of NSDA?
      [ 188 ms] '1128 SIXTEENTH ST.,'  ANLS 0.00
      candidates: [('1128 SIXTEENTH ST.,', 0.035), ('SIXTEENTH ST., N. W., WASHINGTON,', 0.025), ('1128 SIXTEENTH ST., N. W.,', 0.011)]   gold: ['1128 SIXTEENTH ST., N. W., WASHINGTON, D. C. 20036', '1128 sixteenth st., N. W., washington, D. C. 20036']

    Q: According to budget request summary what is total amount of other expenses??
      [  30 ms] '$15000 .00'  ANLS 0.00
      candidates: [('$15000 .00', 0.853), ('$15000 .00', 0.055), ('15,000.00 $15000 .00', 0.04)]   gold: ['$975.00', '975.00']

    Q: Who is ‘presiding’ TRRF GENERAL SESSION (PART 1)?
      [  23 ms] 'Lee A. Waller'  ANLS 1.00
      candidates: [('Lee A. Waller', 1.0), ('Lee A. Waller TRRF Vice President', 0.0), ('Waller', 0.0)]   gold: ['TRRF Vice President', 'lee a. waller']

    Q: How many nomination committee meetings has Y. C. Deveshwar attended?
      [  49 ms] 'three'  ANLS 0.00
      candidates: [('three', 0.118), ('5th April, 2012', 0.059), ('2012', 0.055)]   gold: ['2']

    questions an extractive model structurally cannot answer
      'How many pages does this document have in total?' -> '20036'
      'Summarise this document in one sentence.' -> 'NATIONAL SOFT DRINK ASSOCIATIONS'
    VRAM after layoutlm        0.01 GB allocated /  0.02 GB reserved

## 9. Donut - OCR-free, end to end

Donut (Naver Clova, ECCV 2022) removed the OCR engine entirely: a Swin Transformer reads the page image, a BART decoder writes the answer. The name is the argument - “Document Understanding Transformer” with no OCR.

What that buys: no OCR licensing or latency, no error propagation from a bad scan, and one model to deploy instead of two systems. What it costs: it needs heavy synthetic pretraining (Donut was trained on 11M synthetic documents in several languages), it is weaker on very small text, and when it is wrong you have no intermediate artefact to debug.

Mechanically it is a `VisionEncoderDecoderModel` driven by **task prompts**: `<s_docvqa><s_question>...</s_question><s_answer>` for question answering, other tags for parsing (CORD receipts) or classification. The output is a tagged string that a small regex turns back into fields.

Input resolution is fixed at 2560x1920 for this checkpoint, which is the real reason it can read at all - and also why it is slower per page than LayoutLM despite being a similar size.

------------------------------------------------------------------------

``` python
import re

from transformers import DonutProcessor, VisionEncoderDecoderModel

donut_id = "naver-clova-ix/donut-base-finetuned-docvqa"
donut_proc = DonutProcessor.from_pretrained(donut_id, cache_dir=HF_CACHE)
donut = VisionEncoderDecoderModel.from_pretrained(
    donut_id, dtype=dtype, cache_dir=HF_CACHE
).to(device).eval()
vram("donut loaded")
print("input resolution:", donut_proc.image_processor.size)


def donut_answer(image, question, max_new_tokens=64):
    "Donut's task-prompt protocol: a tagged prompt in, a tagged string out."
    prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
    pixel_values = donut_proc(image, return_tensors="pt").pixel_values.to(device, dtype)
    decoder_ids = donut_proc.tokenizer(prompt, add_special_tokens=False,
                                       return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        out = donut.generate(
            pixel_values, decoder_input_ids=decoder_ids, max_new_tokens=max_new_tokens,
            pad_token_id=donut_proc.tokenizer.pad_token_id,
            eos_token_id=donut_proc.tokenizer.eos_token_id,
            use_cache=True, bad_words_ids=[[donut_proc.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
    text = donut_proc.batch_decode(out.sequences)[0]
    text = text.replace(donut_proc.tokenizer.eos_token, "").replace(donut_proc.tokenizer.pad_token, "")
    text = re.sub(r"<.*?>", "", text, count=1).strip()      # drop the leading task token
    parsed = donut_proc.token2json(text)
    return parsed.get("answer", text) if isinstance(parsed, dict) else text


for it in items[:4]:
    t0 = time.perf_counter()
    ans = donut_answer(it["image"], it["question"])
    print(f"Q: {it['question']}\n  [{(time.perf_counter() - t0) * 1000:4.0f} ms] {ans!r}  "
          f"ANLS {anls(ans, it['answers']):.2f}   gold: {it['answers'][:2]}")

del donut, donut_proc
free_memory()
vram("after donut")
```

    VRAM donut loaded          0.41 GB allocated /  0.43 GB reserved
    input resolution: SizeDict(height=2560, width=1920, longest_edge=None, shortest_edge=None, max_height=None, max_width=None)

    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

    Q: What the location address of NSDA?
      [1183 ms] 'national soft drink association'  ANLS 0.00   gold: ['1128 SIXTEENTH ST., N. W., WASHINGTON, D. C. 20036', '1128 sixteenth st., N. W., washington, D. C. 20036']

    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

    Q: According to budget request summary what is total amount of other expenses??
      [ 353 ms] '$ 975.00'  ANLS 0.88   gold: ['$975.00', '975.00']

    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

    Q: Who is ‘presiding’ TRRF GENERAL SESSION (PART 1)?
      [ 359 ms] 'lee a. waller'  ANLS 1.00   gold: ['TRRF Vice President', 'lee a. waller']
    Q: How many nomination committee meetings has Y. C. Deveshwar attended?
      [ 361 ms] '2'  ANLS 1.00   gold: ['2']
    VRAM after donut           0.01 GB allocated /  0.02 GB reserved

## 10. Qwen3-VL-2B - prompt a VLM, and the cascade built from one

The 2026 default, and two architectures for the price of one model.

**Direct.** Show the page, ask the question, add the short-answer instruction that DocVQA scoring needs (same lesson as `Multimodal/04`: without it, a chat model answers in a sentence and ANLS falls below threshold). Qwen3-VL’s **native dynamic resolution** is what makes this work on documents - the page is not squashed to a square, and 32 OCR languages were in its training mix.

**Cascade.** The pattern most production document pipelines actually use: a small OCR specialist turns the page into structured markdown, and a language model answers from that text. Here **GOT-OCR 2.0** (0.58B) does the reading and the *same* Qwen3-VL model answers from the transcript with no image at all. The cascade is cheaper per question when you ask many questions of one page (transcribe once, answer many times), it produces a reusable artefact, and it is auditable - but it inherits every OCR error, exactly as in `Multimodal/00_Audio_Text_to_Text`.

------------------------------------------------------------------------

``` python
from transformers import AutoModelForImageTextToText, AutoProcessor

qwen_id = "Qwen/Qwen3-VL-2B-Instruct"
qwen_proc = AutoProcessor.from_pretrained(qwen_id, cache_dir=HF_CACHE)
qwen = AutoModelForImageTextToText.from_pretrained(
    qwen_id, dtype=dtype, device_map=device, low_cpu_mem_usage=True, cache_dir=HF_CACHE
).eval()
vram("qwen3-vl loaded")

SHORT = ("Answer the question using text taken directly from the document. "
         "Reply with the answer only, no explanation.")


def qwen_ask(question, image=None, max_new_tokens=64):
    "Ask Qwen3-VL with a page image, or with text only when `image` is None (the cascade)."
    content = ([{"type": "image", "image": image}] if image is not None else [])
    content.append({"type": "text", "text": question})
    inputs = qwen_proc.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
    ).to(qwen.device)
    n = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = qwen.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return qwen_proc.batch_decode(out[:, n:], skip_special_tokens=True)[0].strip()


for it in items[:4]:
    t0 = time.perf_counter()
    ans = qwen_ask(f"{SHORT} {it['question']}", it["image"])
    print(f"Q: {it['question']}\n  [{(time.perf_counter() - t0) * 1000:4.0f} ms] {ans!r}  "
          f"ANLS {anls(ans, it['answers']):.2f}   gold: {it['answers'][:2]}")

# How many visual tokens does one page cost? On documents this is the cost driver, and
# it is what separates a VLM from LayoutLM economically.
page = items[0]["image"]
inputs = qwen_proc.apply_chat_template(
    [{"role": "user", "content": [{"type": "image", "image": page},
                                  {"type": "text", "text": "hi"}]}],
    add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
text_only = qwen_proc.apply_chat_template(
    [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
print(f"\n{page.size} page -> ~{inputs['input_ids'].shape[1] - text_only['input_ids'].shape[1]} "
      "visual tokens before the question is even read")
```

    VRAM qwen3-vl loaded       4.26 GB allocated /  4.27 GB reserved
    Q: What the location address of NSDA?
      [1785 ms] '1128 SIXTEENTH ST., N. W., WASHINGTON, D. C. 20036'  ANLS 1.00   gold: ['1128 SIXTEENTH ST., N. W., WASHINGTON, D. C. 20036', '1128 sixteenth st., N. W., washington, D. C. 20036']
    Q: According to budget request summary what is total amount of other expenses??
      [2523 ms] '$ 975.00'  ANLS 0.88   gold: ['$975.00', '975.00']
    Q: Who is ‘presiding’ TRRF GENERAL SESSION (PART 1)?
      [ 714 ms] 'Lee A. Waller'  ANLS 1.00   gold: ['TRRF Vice President', 'lee a. waller']
    Q: How many nomination committee meetings has Y. C. Deveshwar attended?
      [2269 ms] '2'  ANLS 1.00   gold: ['2']

    (1370, 1480) page -> ~1980 visual tokens before the question is even read

``` python
# The cascade: GOT-OCR 2.0 reads the page to markdown, then the SAME model answers from
# the text with no image. Load OCR, transcribe every page once, free it.
del qwen
free_memory()

got_id = "stepfun-ai/GOT-OCR-2.0-hf"
got_proc = AutoProcessor.from_pretrained(got_id, use_fast=True, cache_dir=HF_CACHE)
got = AutoModelForImageTextToText.from_pretrained(
    got_id, dtype=dtype, device_map=device, cache_dir=HF_CACHE
).eval()
vram("got-ocr loaded")


def transcribe(image, max_new_tokens=1024, formatted=True):
    "Page image -> text. `format=True` asks for markdown/LaTeX structure instead of a flat string."
    inputs = got_proc(image, return_tensors="pt", format=formatted).to(got.device, dtype)
    with torch.inference_mode():
        ids = got.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens,
                           tokenizer=got_proc.tokenizer, stop_strings="<|im_end|>")
    return got_proc.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)


t0 = time.perf_counter()
transcripts = {it["id"]: transcribe(it["image"]) for it in items}
print(f"transcribed {len(transcripts)} pages in {time.perf_counter() - t0:.0f}s")
print(f"\npage 0 transcript ({len(transcripts[items[0]['id']])} chars):\n"
      f"{transcripts[items[0]['id']][:400]}...")

del got, got_proc
free_memory()
vram("after got-ocr")
```

    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.

    VRAM got-ocr loaded        1.13 GB allocated /  1.13 GB reserved

    [transformers] Ignoring clean_up_tokenization_spaces=True for BPE tokenizer TokenizersBackend. The clean_up_tokenization post-processing step is designed for WordPiece tokenizers and is destructive for BPE (it strips spaces before punctuation). Set clean_up_tokenization_spaces=False to suppress this warning, or set clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output=True to force cleanup anyway.

    transcribed 20 pages in 97s

    page 0 transcript (651 chars):
    \title{
    The best thing between two sandwiches.
    }
    Soft drinks go with all kinds of sandwiches. Round ones, square ones, fat ones and lean ones.
    Not only do they quench large thirsts in a fun way; they also help balance the diet. After all, healthy bodies need 5 to 6 glasses of water a day. Soft drinks contain the purest, filtered water. So sandwich soft drinks among your sandwiches. And celebrate N...
    VRAM after got-ocr         0.01 GB allocated /  0.02 GB reserved

``` python
# Now answer from the transcripts alone. Same weights as the direct VLM path, no image.
qwen = AutoModelForImageTextToText.from_pretrained(
    qwen_id, dtype=dtype, device_map=device, low_cpu_mem_usage=True, cache_dir=HF_CACHE
).eval()
vram("qwen reloaded")

for it in items[:4]:
    prompt = (f"Document text:\n\"\"\"\n{transcripts[it['id']][:6000]}\n\"\"\"\n\n"
              f"{SHORT} {it['question']}")
    t0 = time.perf_counter()
    ans = qwen_ask(prompt)
    print(f"Q: {it['question']}\n  [{(time.perf_counter() - t0) * 1000:4.0f} ms] {ans!r}  "
          f"ANLS {anls(ans, it['answers']):.2f}   gold: {it['answers'][:2]}")
print("\nWhere the cascade wins: one transcription, then many cheap text-only questions.\n"
      "Where it loses: anything the OCR dropped is gone, and the answer can no longer be\n"
      "pointed back at a box on the page.")
```

    VRAM qwen reloaded         4.26 GB allocated /  4.27 GB reserved
    Q: What the location address of NSDA?
      [ 468 ms] '112B SIXTEENTH ST., N. W., WASHINGTON, D.C. 20039'  ANLS 0.94   gold: ['1128 SIXTEENTH ST., N. W., WASHINGTON, D. C. 20036', '1128 sixteenth st., N. W., washington, D. C. 20036']
    Q: According to budget request summary what is total amount of other expenses??
      [ 241 ms] '$1,957.00'  ANLS 0.56   gold: ['$975.00', '975.00']
    Q: Who is ‘presiding’ TRRF GENERAL SESSION (PART 1)?
      [ 170 ms] 'Lee A. Waller'  ANLS 1.00   gold: ['TRRF Vice President', 'lee a. waller']
    Q: How many nomination committee meetings has Y. C. Deveshwar attended?
      [ 154 ms] '3'  ANLS 0.00   gold: ['2']

    Where the cascade wins: one transcription, then many cheap text-only questions.
    Where it loses: anything the OCR dropped is gone, and the answer can no longer be
    pointed back at a box on the page.

## 11. What resolution costs

The single most important variable in document AI, and the easiest to get wrong.

A page is mostly whitespace with small glyphs. Downscale it and the glyphs disappear before the layout does, so the model still “sees a document” and confidently answers from a blurred approximation. This is the root cause of most “the VLM cannot read my PDF” reports - not the model, the preprocessing.

The cell below asks the same questions at full resolution, half, and quarter, and reports ANLS at each. Expect a sharp cliff rather than a gentle slope: text is legible or it is not.

The corollary for the extractive path: LayoutLM does not see pixels at all, so its quality tracks **OCR quality** rather than image resolution. Feed a 150 dpi scan to a good OCR engine and LayoutLM is fine; feed a blurry phone photo and both paths fail, for different reasons.

------------------------------------------------------------------------

``` python
def scaled(image, factor):
    "Downscale a page by `factor` (1.0 = original)."
    if factor == 1.0:
        return image
    return image.resize((max(1, int(image.width * factor)), max(1, int(image.height * factor))))


print(f"{'scale':>7s} {'page size':>14s} {'ANLS':>6s}")
for factor in (1.0, 0.5, 0.25):
    scores = []
    for it in items[:8]:
        ans = qwen_ask(f"{SHORT} {it['question']}", scaled(it["image"], factor))
        scores.append(anls(ans, it["answers"]))
    size = scaled(items[0]["image"], factor).size
    print(f"{factor:7.2f} {str(size):>14s} {sum(scores) / len(scores):6.3f}")
print("\nThe drop is a cliff, not a slope. Resolution is the first thing to check when a\n"
      "document pipeline underperforms, and the last thing to economise on.")
```

      scale      page size   ANLS
       1.00   (1370, 1480)  0.984
       0.50     (685, 740)  0.992
       0.25     (342, 370)  0.734

    The drop is a cliff, not a slope. Resolution is the first thing to check when a
    document pipeline underperforms, and the last thing to economise on.

## 12. Head-to-head Benchmark

Four systems on the **same DocVQA pages**, the **same questions** and the **same ANLS implementation**:

| System | Architecture | Sees |
|----|----|----|
| LayoutLM | extractive, layout-aware | OCR words + boxes |
| Donut | OCR-free seq2seq | the page image |
| Qwen3-VL-2B (direct) | generalist VLM | the page image |
| GOT-OCR -\> Qwen3-VL (cascade) | OCR specialist + text LLM | the transcript only |

Each model is loaded, measured and **freed before the next one loads**, so VRAM stays flat. Reported: mean ANLS, milliseconds per question, and - for the cascade - the transcription cost is amortised across all questions on a page, which is the honest way to count it.

**Read this as a smoke test, not a leaderboard.** Twenty questions gives an ANLS with an uncertainty of roughly plus or minus 0.1, and the published numbers come from the RRC test server over 5,349 questions. What the sample *does* show honestly is the cost profile of the four architectures and the kinds of question each one gets wrong.

------------------------------------------------------------------------

``` python
del qwen, qwen_proc
free_memory()
vram("before benchmark")

from transformers import DonutProcessor, VisionEncoderDecoderModel, pipeline


def load_layoutlm():
    "Extractive over provided OCR words and boxes. 0.13B."
    pipe = pipeline("document-question-answering", model=layout_id, device=device,
                    model_kwargs={"cache_dir": HF_CACHE})

    def answer(item):
        out = pipe(image=item["image"], question=item["question"],
                   word_boxes=item["word_boxes"], top_k=1)
        out = out if isinstance(out, list) else [out]
        return out[0]["answer"]

    return answer, [pipe]


def load_donut():
    "OCR-free seq2seq at a fixed 2560x1920 input. 0.2B."
    proc = DonutProcessor.from_pretrained(donut_id, cache_dir=HF_CACHE)
    model = VisionEncoderDecoderModel.from_pretrained(
        donut_id, dtype=dtype, cache_dir=HF_CACHE).to(device).eval()

    def answer(item):
        prompt = f"<s_docvqa><s_question>{item['question']}</s_question><s_answer>"
        pv = proc(item["image"], return_tensors="pt").pixel_values.to(device, dtype)
        ids = proc.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        with torch.inference_mode():
            out = model.generate(pv, decoder_input_ids=ids, max_new_tokens=64,
                                 pad_token_id=proc.tokenizer.pad_token_id,
                                 eos_token_id=proc.tokenizer.eos_token_id, use_cache=True,
                                 bad_words_ids=[[proc.tokenizer.unk_token_id]],
                                 return_dict_in_generate=True)
        text = proc.batch_decode(out.sequences)[0]
        text = text.replace(proc.tokenizer.eos_token, "").replace(proc.tokenizer.pad_token, "")
        text = re.sub(r"<.*?>", "", text, count=1).strip()
        parsed = proc.token2json(text)
        return parsed.get("answer", text) if isinstance(parsed, dict) else text

    return answer, [model, proc]


def load_qwen_direct():
    "Generalist VLM reading the page image. 2B."
    proc = AutoProcessor.from_pretrained(qwen_id, cache_dir=HF_CACHE)
    model = AutoModelForImageTextToText.from_pretrained(
        qwen_id, dtype=dtype, device_map=device, low_cpu_mem_usage=True, cache_dir=HF_CACHE).eval()

    def ask(content):
        inputs = proc.apply_chat_template([{"role": "user", "content": content}],
                                          add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt").to(model.device)
        n = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        return proc.batch_decode(out[:, n:], skip_special_tokens=True)[0].strip()

    def answer(item):
        return ask([{"type": "image", "image": item["image"]},
                    {"type": "text", "text": f"{SHORT} {item['question']}"}])

    answer.ask = ask
    return answer, [model, proc]


def load_cascade():
    "GOT-OCR transcripts (already computed above) + the same text LLM."
    answer_direct, handles = load_qwen_direct()
    ask = answer_direct.ask

    def answer(item):
        prompt = (f"Document text:\n\"\"\"\n{transcripts[item['id']][:6000]}\n\"\"\"\n\n"
                  f"{SHORT} {item['question']}")
        return ask([{"type": "text", "text": prompt}])

    return answer, handles


def benchmark(name, loader):
    "Load, answer every item, score with ANLS, free."
    answer_fn, handles = loader()
    preds, t0 = [], time.perf_counter()
    for it in items:
        preds.append(answer_fn(it))
    elapsed = time.perf_counter() - t0
    scores = [anls(p, it["answers"]) for p, it in zip(preds, items)]
    for h in handles:
        del h
    del answer_fn, handles
    free_memory()
    vram(f"after {name}")
    return {"system": name,
            "anls": round(sum(scores) / len(scores), 3),
            "exact": round(sum(p.strip().lower() in [a.lower() for a in it["answers"]]
                               for p, it in zip(preds, items)) / len(items), 3),
            "ms_per_q": round(elapsed / len(items) * 1000, 1),
            "preds": preds}


results = [
    benchmark("layoutlm (extractive)", load_layoutlm),
    benchmark("donut (ocr-free)", load_donut),
    benchmark("qwen3-vl-2b (direct)", load_qwen_direct),
    benchmark("got-ocr -> qwen3-vl (cascade)", load_cascade),
]
vram("benchmark done")
```

    VRAM before benchmark      0.01 GB allocated /  0.02 GB reserved

    [transformers] You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset

    VRAM after layoutlm (extractive)  0.01 GB allocated /  0.02 GB reserved

    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=64) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

    VRAM after donut (ocr-free)  0.01 GB allocated /  0.02 GB reserved

    VRAM after qwen3-vl-2b (direct)  0.01 GB allocated /  0.02 GB reserved

    VRAM after got-ocr -> qwen3-vl (cascade)  0.01 GB allocated /  0.02 GB reserved
    VRAM benchmark done        0.01 GB allocated /  0.02 GB reserved

``` python
import pandas as pd

df = pd.DataFrame([{k: v for k, v in r.items() if k != "preds"} for r in results])
df = df.sort_values("anls", ascending=False)
df
```

|     | system                         | anls  | exact | ms_per_q |
|-----|--------------------------------|-------|-------|----------|
| 2   | qwen3-vl-2b (direct)           | 0.992 | 0.90  | 2186.4   |
| 1   | donut (ocr-free)               | 0.769 | 0.65  | 344.9    |
| 3   | got-ocr -\> qwen3-vl (cascade) | 0.700 | 0.60  | 186.0    |
| 0   | layoutlm (extractive)          | 0.326 | 0.20  | 26.8     |

``` python
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org serves an expired TLS certificate from most nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"

from pyecharts import options as opts
from pyecharts.charts import Bar

names = [r["system"] for r in results]
bar = (
    Bar()
    .add_xaxis(names)
    .add_yaxis("ANLS x100", [round(r["anls"] * 100, 1) for r in results])
    .add_yaxis("exact match %", [round(r["exact"] * 100, 1) for r in results])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"DocVQA on {len(items)} test items",
            subtitle="RTX 3060 12 GB - smoke test, not a leaderboard (published leaders are ANLS 0.9+)",
        ),
        xaxis_opts=opts.AxisOpts(name="system", axislabel_opts=opts.LabelOpts(rotate=15)),
        yaxis_opts=opts.AxisOpts(name="score", max_=100),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
    )
)
bar.render_notebook()
```

``` python
from pyecharts.charts import Scatter

# Quality against cost per question - the axis on which these four genuinely differ.
scatter = Scatter()
scatter.add_xaxis([round(r["ms_per_q"], 1) for r in results])
for r in results:
    scatter.add_yaxis(r["system"], [[round(r["ms_per_q"], 1), round(r["anls"] * 100, 1)]],
                      symbol_size=18, label_opts=opts.LabelOpts(is_show=False))
scatter.set_global_opts(
    title_opts=opts.TitleOpts(title="ANLS vs latency per question",
                              subtitle="up and to the left is better; the cascade amortises its OCR over many questions"),
    xaxis_opts=opts.AxisOpts(type_="value", name="milliseconds / question"),
    yaxis_opts=opts.AxisOpts(type_="value", name="ANLS x100"),
    tooltip_opts=opts.TooltipOpts(trigger="item"),
)
scatter.render_notebook()
```

``` python
# The numbers hide the interesting part: look at where the four architectures disagree.
from IPython.display import display

shown = 0
for i, it in enumerate(items):
    scores = [anls(r["preds"][i], it["answers"]) for r in results]
    if len(set(round(s, 1) for s in scores)) == 1 and shown >= 1:
        continue
    display(it["image"].resize((380, int(380 * it["image"].height / it["image"].width))))
    print(f"Q: {it['question']}   gold: {it['answers'][:2]}")
    for r, s in zip(results, scores):
        print(f"  {r['system']:30s} {r['preds'][i][:60]!r:62s} ANLS {s:.2f}")
    print()
    shown += 1
    if shown >= 4:
        break
```

![](05_Document_Question_Answering_files/figure-html/cell-16-output-1.png)

    Q: What the location address of NSDA?   gold: ['1128 SIXTEENTH ST., N. W., WASHINGTON, D. C. 20036', '1128 sixteenth st., N. W., washington, D. C. 20036']
      layoutlm (extractive)          '1128 SIXTEENTH ST.,'                                          ANLS 0.00
      donut (ocr-free)               'national soft drink association'                              ANLS 0.00
      qwen3-vl-2b (direct)           '1128 SIXTEENTH ST., N. W., WASHINGTON, D. C. 20036'           ANLS 1.00
      got-ocr -> qwen3-vl (cascade)  '112B SIXTEENTH ST., N. W., WASHINGTON, D.C. 20039'            ANLS 0.94

![](05_Document_Question_Answering_files/figure-html/cell-16-output-3.png)

    Q: According to budget request summary what is total amount of other expenses??   gold: ['$975.00', '975.00']
      layoutlm (extractive)          '$15000 .00'                                                   ANLS 0.00
      donut (ocr-free)               '$ 975.00'                                                     ANLS 0.88
      qwen3-vl-2b (direct)           '$ 975.00'                                                     ANLS 0.88
      got-ocr -> qwen3-vl (cascade)  '$1,957.00'                                                    ANLS 0.56

![](05_Document_Question_Answering_files/figure-html/cell-16-output-5.png)

    Q: How many nomination committee meetings has Y. C. Deveshwar attended?   gold: ['2']
      layoutlm (extractive)          'three'                                                        ANLS 0.00
      donut (ocr-free)               '2'                                                            ANLS 1.00
      qwen3-vl-2b (direct)           '2'                                                            ANLS 1.00
      got-ocr -> qwen3-vl (cascade)  '3'                                                            ANLS 0.00

![](05_Document_Question_Answering_files/figure-html/cell-16-output-7.png)

    Q: How many nomination committee meetings has S. Banerjee attended?   gold: ['2']
      layoutlm (extractive)          '2012'                                                         ANLS 0.00
      donut (ocr-free)               '2'                                                            ANLS 1.00
      qwen3-vl-2b (direct)           '2'                                                            ANLS 1.00
      got-ocr -> qwen3-vl (cascade)  '3'                                                            ANLS 0.00

## 13. Live Demo: photograph a document and ask about it

Point the webcam at a printed page, receipt or form, capture one frame, and ask a question about it with Qwen3-VL-2B. This is the realistic mobile-capture case, and it exposes everything section 11 was about: a 640x480 webcam frame of an A4 page gives you roughly 60 dpi, which is well below what any model needs for body text. Hold the page close, fill the frame, and light it evenly - or capture at the camera’s full 2592x1520 mode by raising `WIDTH`/`HEIGHT`.

This is the cell people run on its own, so it opens with a `require(...)` guard naming what it needs from Setup instead of dying on a bare `NameError`. Capture notes, all measured on the knowledge-lab container: V4L2 backend with MJPEG and a warm-up read (auto-exposure needs frames to settle), never `CAP_PROP_BUFFERSIZE` (it halves the frame rate without making frames fresher), and no `cv2.imshow` because there is no GUI - the framing preview goes through `IPython.display` handles that update in place.

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


require("device", "dtype", "HF_CACHE", "free_memory", "vram", "qwen_id", "SHORT", "anls")

import time
import torch

# opencv-python-headless is a project dependency; the headless build captures from
# V4L2 fine, it only drops the GUI windows.
import io

import cv2
from IPython.display import Image as IPyImage
from IPython.display import Pretty, display
from PIL import Image

from transformers import AutoModelForImageTextToText, AutoProcessor

CAM = 0             # /dev/video0
WARMUP = 10         # throwaway reads - auto-exposure and white balance need to settle
FRAME_SECONDS = 6   # how long the framing preview runs before the shot is taken
WIDTH, HEIGHT = 1280, 720   # a page needs pixels; 640x480 is ~60 dpi on A4 and unreadable
QUESTIONS = [
    "What is the title of this document?",
    "What is the total amount?",
]


def open_camera(index=CAM, width=WIDTH, height=HEIGHT, auto_exposure=True, exposure=150):
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


def _jpeg(img, quality=85):
    "Encode a PIL image to JPEG bytes - what actually goes over the wire each frame."
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def preview(seconds=FRAME_SECONDS):
    "Stream the raw camera so you can frame the page, then return the final frame."
    cap = open_camera()
    view = status = None  # created from the FIRST real frame, so no placeholder flashes up
    last, n, t0 = None, 0, time.perf_counter()
    try:
        while time.perf_counter() - t0 < seconds:
            last = grab(cap)
            n += 1
            view_img = IPyImage(data=_jpeg(last.resize((640, int(640 * last.height / last.width)))))
            line = Pretty(f"framing - {seconds - (time.perf_counter() - t0):4.1f}s left, "
                          f"{n} frames at {last.size} (fill the frame with the page)")
            if view is None:
                view = display(view_img, display_id=True)
                status = display(line, display_id=True)
            else:
                view.update(view_img)
                status.update(line)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()  # always hand the device back
    if status is not None:
        status.update(Pretty(f"captured the last of {n} frames at {last.size}"))
    return last


# Re-runnable: this cell frees the model at the end, so guard the load or a second
# shift-enter raises NameError on `live_model`.
if "live_model" not in globals():
    live_proc = AutoProcessor.from_pretrained(qwen_id, cache_dir=HF_CACHE)
    live_model = AutoModelForImageTextToText.from_pretrained(
        qwen_id, dtype=dtype, device_map=device, low_cpu_mem_usage=True, cache_dir=HF_CACHE
    ).eval()
    vram("live model")


def live_ask(image, question, max_new_tokens=96):
    "One page image plus one question through the live model."
    inputs = live_proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "image", "image": image},
                                      {"type": "text", "text": question}]}],
        add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
    ).to(live_model.device)
    n = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = live_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return live_proc.batch_decode(out[:, n:], skip_special_tokens=True)[0].strip()


page = preview()
display(page.resize((640, int(640 * page.height / page.width))))

# First, read it back: if the transcript is garbage the answers will be too, and this
# tells you it was the capture, not the model.
t0 = time.perf_counter()
print(f"[{time.perf_counter() - t0:4.1f}s] transcript:\n"
      f"{live_ask(page, 'Transcribe all the text in this image exactly as it appears.', 400)[:600]}\n")

for q in QUESTIONS:
    t0 = time.perf_counter()
    print(f"Q: {q}\n  [{time.perf_counter() - t0:4.1f}s] {live_ask(page, f'{SHORT} {q}')}")

del live_model, live_proc
free_memory()
vram("final")
```

    VRAM live model            4.26 GB allocated /  4.27 GB reserved

![](05_Document_Question_Answering_files/figure-html/cell-17-output-3.jpeg)

    captured the last of 90 frames at (1280, 720)

![](05_Document_Question_Answering_files/figure-html/cell-17-output-5.png)

    [ 0.0s] transcript:
    STANLEY

    Q: What is the title of this document?
      [ 0.0s] The title of this document is "The Water Bottle".
    Q: What is the total amount?
      [ 0.0s] 100
    VRAM final                 0.01 GB allocated /  0.02 GB reserved

## 14. Common Frameworks

Document QA is a pipeline task wearing a model’s clothes. In production the model is rarely the bottleneck: capture quality, page segmentation, reading order and table structure are, and each has its own mature tooling that predates VLMs. The table below is therefore weighted towards the document stack rather than the model stack - which matches where the accuracy actually comes from.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [transformers](https://github.com/huggingface/transformers) | modelling | LayoutLM/LayoutLMv3, Donut, GOT-OCR and every VLM, behind `document-question-answering` and `AutoModelForImageTextToText` | Apache 2.0 | Default. All three architectures of sections 8-10 |
| [peft](https://github.com/huggingface/peft) + [trl](https://github.com/huggingface/trl) | modelling | LoRA on a 2-3B VLM, and plain fine-tuning for LayoutLMv3 and Donut, both of which fit here | Apache 2.0 | Narrow, repetitive document sets - one supplier’s invoice, one government form. A few hundred pages gets near-perfect extraction |
| [Docling](https://github.com/docling-project/docling) / [unstructured](https://github.com/Unstructured-IO/unstructured) | data | Full document conversion: layout, reading order, tables and figures out of a PDF, calling OCR only where the page needs it | MIT / Apache 2.0 | Almost always. Born-digital PDFs have a text layer, and running OCR over it is pure waste |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) / [docTR](https://github.com/mindee/doctr) / [Tesseract](https://github.com/tesseract-ocr/tesseract) | data | Word boxes and confidences for the extractive path, fast and on CPU | Apache 2.0 | LayoutLM needs boxes, and these produce them. Also the cheap answer when the text is clean |
| [OpenCV](https://github.com/opencv/opencv) + [pypdfium2](https://github.com/pypdfium2-team/pypdfium2) / pdf2image | data | Deskew, dewarp, crop to page, and render PDF pages at a chosen dpi | Apache 2.0 / BSD-3 | Always. Section 11 shows resolution is a cliff, and capture quality moves ANLS more than any model swap |
| [vLLM](https://github.com/vllm-project/vllm) / [SGLang](https://github.com/sgl-project/sglang) | inference runtime | Batched VLM pages, with prefix caching that pays when many questions share one page | Apache 2.0 | Serving the VLM path. Document pages are large prompts, so caching matters more here than usual |
| [optimum](https://github.com/huggingface/optimum) + [ONNX Runtime](https://onnxruntime.ai/) | inference runtime | LayoutLM and Donut exported for CPU serving at a fraction of the cost | Apache 2.0 / MIT | The extractive path, which is cheap enough to run without a GPU at all |
| [LlamaIndex](https://github.com/run-llama/llama_index) / [LangGraph](https://github.com/langchain-ai/langgraph) | orchestration | The multi-page flow: retrieve the relevant pages, answer on those, aggregate across them | MIT | Real questions span pages. Pair with visual retrieval - see `Multimodal/07_Visual_Document_Retrieval` |
| ANLS via [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) / the DocVQA scripts | evaluation | Edit-distance-tolerant scoring, which is what DocVQA and its successors report | Apache 2.0 | Always. Exact match punishes a correct answer for a single OCR character, which is not the failure you care about |

**The 2026 default stack** is Docling to convert, a cheap OCR engine where there is no text layer, LayoutLMv3 or Donut fine-tuned when the layouts repeat, and a VLM through vLLM when they do not. Visual retrieval in front of it as soon as documents exceed one page.

The common wrong turn is fixing the model when the pipeline is broken. Deskewing, dewarping and raising dpi cost nothing and move accuracy more than a model tier. The second is asking a VLM to do arithmetic over a table image: parse the table into structure first and query it as a table (`Natural_Language_Processing/02_Table_Question_Answering`) - that is a solved problem, and pixel-space arithmetic is not.

------------------------------------------------------------------------

## 15. Going Further

- **Fine-tuning pays here more than almost anywhere.** Document sets are narrow and repetitive: one supplier’s invoice template, one government form. A few hundred annotated pages fine-tune LayoutLMv3 or Donut to near-perfect field extraction on *your* documents, at a cost per page nothing generalist can match. The HF [document QA task guide](https://huggingface.co/docs/transformers/tasks/document_question_answering) and Donut’s CORD recipe are the templates; LoRA on a 2-3B VLM through `peft` + `trl` is the alternative when the layouts vary.
- **Fix the pipeline before the model.** Deskew, dewarp, crop to the page, and check the dpi. Section 11 showed that resolution is a cliff; in production, capture quality moves ANLS more than any model swap. `opencv` plus a document-boundary detector covers most of it.
- **Multi-page and multi-document.** Real questions span pages. The 2026 stack is: ColPali-style visual retrieval (`Multimodal/07_Visual_Document_Retrieval`) to find the relevant pages, then DocQA on those pages, then an LLM to aggregate. MP-DocVQA and DUDE are the benchmarks that measure it; MMLongBench-Doc is the hard one.
- **Grounded answers.** If you need to cite the source, either use an extractive model (span and box for free) or prompt a grounding-capable VLM for the answer *plus* its bounding box - Qwen3-VL and Florence-2 both emit boxes as text. That turns a fluent answer into an auditable one.
- **Tables specifically.** For a table-heavy page, parsing the table into structure and then querying it (`NLP/02_Table_Question_Answering`) beats asking a VLM to do arithmetic over pixels. GOT-OCR and dots.ocr both emit HTML tables; TAPAS and TAPEX answer over them.
- **Related notebooks.** `Multimodal/07_Visual_Document_Retrieval` (find the page first), `Computer_Vision/05_Image_to_Text` (GOT-OCR and the OCR specialists in depth), `Multimodal/04_Visual_Question_Answering` (natural photos and the VQA metric), `Multimodal/01_Image_Text_to_Text` (general VLM prompting, resolution and structured output), and `NLP/02_Table_Question_Answering` (once the table is text).

------------------------------------------------------------------------

Back to top
