# Image-to-Text

Everything to know about image captioning and OCR: what “image in, text out” covers, the mid-2026 model landscape, why n-gram metrics break on dense captions, and runnable code to test the leading open models.

Author

Benedict Thekkel

## 1. What is Image-to-Text?

Image-to-text is the **image in, text out** family: the model sees pixels and only pixels, and emits natural language. No text prompt is required. Three jobs live under this umbrella:

- **Captioning** - one sentence of alt-text (“two cats asleep on a pink couch”).
- **Dense description** - a paragraph that enumerates objects, attributes, spatial relations, style and any visible text. This is what feeds text-to-image training sets and accessibility tools.
- **OCR / document text extraction** - transcribe the text *in* the image, ideally with structure (markdown, LaTeX, table cells) preserved.

**Input.** A single RGB image. Every model resizes it to a fixed grid (224/384/448/896 px, or a dynamic tiling scheme for OCR models that need to resolve small glyphs) and cuts it into patches; the patch embeddings are the “visual tokens” the decoder attends over. Resolution is the single biggest lever for OCR quality - a 224 px encoder physically cannot read 8 pt text.

**Output.** A token sequence. Sometimes plain text; sometimes structured text (markdown, LaTeX, HTML tables) or text interleaved with coordinates (`<OCR_WITH_REGION>`, dense region captions).

**Where the boundary is blurring.** The honest 2026 position: “caption this image” is increasingly answered by **prompting a general VLM**, not by running a dedicated captioner. A dedicated captioner (BLIP, ViT-GPT2) takes no prompt and always produces the same style of output; a VLM takes a prompt and will give you alt-text, a paragraph, JSON, or a transcript depending on what you ask. Image-to-text is therefore best understood as the **zero-prompt slice** of image-text-to-text, and the two tasks now share most of their model list.

| Neighbouring task | What it does | See |
|----|----|----|
| Image-text-to-text | Image **plus** an instruction, out comes text (VLM chat) | `Multimodal/01_Image_Text_to_Text` |
| Visual question answering | Image plus a question, out comes a short answer | `Multimodal/04_Visual_Question_Answering` |
| Zero-shot image classification | Image plus candidate labels, out comes a label (CLIP) | `11_Zero_Shot_Image_Classification` |
| Text-to-image | The inverse task: text in, pixels out | `04_Text_to_Image` |
| Object detection | Image in, boxes + class labels out (not free text) | `02_Object_Detection` |
| Image feature extraction | Image in, embedding out (no language) | `16_Image_Feature_Extraction` |

------------------------------------------------------------------------

## 2. Real-World Use Cases

The three sub-tasks (alt-text, dense caption, OCR) are deployed by completely different teams under completely different constraints. “Which captioner is best” is not a question with one answer.

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Automatic alt-text for accessibility | Social / web (Facebook AAT, Instagram, iOS VoiceOver) | Photo -\> one short, safe sentence | Latency and cost at billions of images/day; never hallucinate a person’s identity |
| Recaptioning training data for generative models | AI research (DALL-E 3, SD3, FLUX pipelines) | Web image -\> long dense synthetic caption | Throughput (hundreds of millions of images); descriptive completeness beats brevity |
| Document parsing / IDP | Fintech, insurance, legal (Mistral OCR, Azure Document Intelligence) | Scanned PDF page -\> markdown + tables + reading order | Layout fidelity, table structure, tolerance to skew/scan noise; auditability |
| Receipt and invoice extraction | Retail, expense tools (Expensify, Ramp) | Phone photo of a receipt -\> line items, totals | Robustness to crumpled paper and bad lighting; a wrong total is a financial error |
| Media asset search and moderation | Stock photo, streaming, ad-tech | Image -\> caption + tags indexed for retrieval | Recall of rare concepts; cost per asset; consistency of vocabulary |
| E-commerce product listings | Marketplaces | Seller photo -\> title, attributes, description | Attribute accuracy (colour, material, brand); hallucinated attributes cause returns |
| Screen understanding for agents | GUI agents, RPA | Screenshot -\> text + element positions | Small-font OCR at 1080p+; coordinates must be pixel-accurate |
| Handwriting and archive digitisation | Libraries, healthcare records | Handwritten page -\> transcript | CER on cursive; domain shift between scribes/centuries |

**What the CIDEr number hides.** A COCO CIDEr score is a measure of how well a model writes *one short sentence about a photo of a common object*, scored against five crowdworker captions written in 2014. It tells you nothing about the three things that actually break production systems. First, **hallucination**: captioners confidently invent objects that are not in the image (the CHAIR metric exists precisely to measure this), and a fluent hallucination propagates further than an obvious error because downstream indexes trust it. Second, **resolution and domain shift**: a model that captions Flickr photos beautifully will read a 300 DPI invoice as gibberish, because its encoder downsamples the page to 224 px. Third, **style is not accuracy**: for dense captioning there is no reference to compare against, so n-gram metrics collapse (see section 4) and teams end up with LLM-as-judge or human review, which is slow and expensive. Add the deployment fork - batch recaptioning of a billion images wants a 0.2B model at 1000 img/s, while a document pipeline wants a 3B+ model at 1 page/s with perfect table structure - and the leaderboard ranking becomes almost irrelevant to the choice.

------------------------------------------------------------------------

## 3. How Modern Image-to-Text Works

Six generations, all still in use somewhere:

1.  **CNN encoder + LSTM decoder (Show and Tell, 2015).** A CNN pools the image into one vector; an LSTM unrolls a sentence from it. **Show, Attend and Tell** (2015) added soft attention so each generated word could look at a different image region - the first “the model looks where it writes” result. Fluent but shallow; the vector bottleneck loses everything but the gist.
2.  **Region features + transformer (bottom-up attention, 2018-2020).** Run a Faster R-CNN, feed its region features to a transformer decoder. Strong COCO numbers, but a slow two-stage pipeline welded to a fixed detector vocabulary.
3.  **End-to-end ViT encoder + text decoder (2021-2022).** Drop the detector: patch embeddings straight into a seq2seq decoder. `nlpconnect/vit-gpt2-image-captioning` is the archetype (ViT encoder, GPT-2 decoder); **TrOCR** is the same recipe aimed at text lines; **Donut** (2022) applied it to whole documents *without* an OCR engine (“OCR-free document understanding”), and **Nougat** (2023) to academic PDFs (LaTeX out).
4.  **Contrastive + generative hybrids (BLIP, 2022).** BLIP trains image-text contrastive, image-text matching and captioning objectives together, and bootstraps its own training data (CapFilt: caption the web images, filter the noisy alt-texts). Still the default small captioner in 2026 because it is 0.25-0.47B, permissive, and needs no prompt.
5.  **Frozen-tower bridges (BLIP-2, 2023).** The key efficiency idea of the era: **freeze the vision encoder and freeze the LLM**, and train only a tiny bridge - BLIP-2’s **Q-Former**, a set of ~32 learned query tokens that cross-attend to the ViT features and emit soft prompts into the LLM’s embedding space. You get an LLM’s world knowledge and fluency for the cost of training ~100M parameters instead of billions. Every later VLM is a variant of this bridge (LLaVA replaced the Q-Former with a plain MLP projector and showed that visual **instruction tuning** on GPT-4-generated data mattered more than the bridge’s cleverness; InstructBLIP kept the Q-Former and made it instruction-aware).
6.  **Small unified VLMs (2024-2026) - where we are now.** One model does captioning, OCR, grounding, detection and VQA, selected by prompt. **Florence-2** (2024, 0.23B/0.77B) is the purest example: a seq2seq model whose task is chosen by a special token (`<CAPTION>`, `<DETAILED_CAPTION>`, `<OCR>`, `<OD>`), trained on FLD-5B (5.4B annotations over 126M images). **PaliGemma 2**, **Qwen2.5-VL / Qwen3-VL**, **InternVL3**, **Moondream**, and **SmolVLM/SmolVLM2** occupy the 0.25B-8B band; they are prompted, not task-tokened, and they are what most people now reach for.

**The OCR specialist line runs in parallel** and has not been absorbed: TrOCR (2021, line-level) -\> Donut (2022, OCR-free docs) -\> Nougat (2023, PDFs to LaTeX) -\> **GOT-OCR 2.0** (2024, 0.58B, “OCR-2.0”: plain text, markdown, LaTeX, tables, sheet music, molecular formulas from one 0.58B model) -\> the 2025-2026 crop of document VLMs (**dots.ocr**, **olmOCR 2**, **DeepSeek-OCR**, **PaddleOCR-VL**, **GLM-OCR**), which are sub-1B-to-3B models that beat far larger general VLMs on document parsing because they train on high-resolution tiled pages and structured targets. Focus beats generality when the task is narrow.

**Trade-off cheat sheet:**

| Approach | Caption quality | OCR | Speed | Size | Prompted? | Example |
|----|----|----|----|----|----|----|
| CNN/ViT + LM decoder | basic, generic | no | fastest | 0.2B | no | ViT-GPT2 |
| Contrastive+generative captioner | good short captions | no | fast | 0.25-0.47B | prefix only | BLIP |
| Frozen-tower bridge | good, LLM-fluent | weak | medium | 3-12B | yes | BLIP-2 |
| Task-token unified model | short **and** dense | decent | fast | 0.23-0.77B | task tokens | Florence-2 |
| Instruction-tuned small VLM | best dense captions | good | medium | 0.25-8B | yes | SmolVLM2, Qwen3-VL |
| Document OCR specialist | n/a | best | medium | 0.5-3B | format flags | GOT-OCR 2.0, dots.ocr |

------------------------------------------------------------------------

## 4. Evaluation Metrics

**BLEU-4** - modified n-gram precision of the candidate against the references, with a brevity penalty. For n-grams up to N (=4):

$$\mathrm{BLEU} = BP \cdot \exp\!\left(\frac{1}{N}\sum_{n=1}^{N} \log p_n\right), \qquad p_n = \frac{\sum_{g \in \mathrm{cand}} \min\!\big(c(g),\, \max_r c_r(g)\big)}{\sum_{g \in \mathrm{cand}} c(g)}$$

$$BP = \begin{cases} 1 & \text{if } |c| > |r| \\ e^{1 - |r|/|c|} & \text{otherwise}\end{cases}$$

Precision-oriented, so a short safe caption scores well. Weak signal on its own.

**METEOR** - unigram alignment with stemming and synonym matching, F-mean weighted toward recall. Correlates better with humans than BLEU but is slow and English-centric.

**ROUGE-L** - longest-common-subsequence F-measure. Recall-oriented; borrowed from summarisation.

**CIDEr** - the captioning standard. Represent the candidate and each reference as **TF-IDF-weighted n-gram vectors** (IDF computed over the whole reference corpus, so n-grams that appear in every caption - “a”, “of a”, “there is” - are down-weighted toward zero), then average the cosine similarity to the references over n = 1..4:

$$\mathrm{CIDEr}_n(c, S) = \frac{1}{|S|}\sum_{j} \frac{g^n(c) \cdot g^n(s_j)}{\lVert g^n(c)\rVert \, \lVert g^n(s_j)\rVert}, \qquad \mathrm{CIDEr}(c, S) = \frac{10}{N}\sum_{n=1}^{N}\mathrm{CIDEr}_n(c,S)$$

CIDEr-D, the version actually used on COCO, adds a Gaussian length penalty and clips n-gram counts to stop models gaming it by repeating salient words. Scores are conventionally x100 (COCO SOTA is roughly 130-150 CIDEr).

**SPICE** - parse both captions into scene graphs (objects, attributes, relations) and take the F1 over graph tuples. Semantic rather than lexical, so it rewards getting the *content* right regardless of phrasing; the cost is a dependency parser in the loop.

**Reference-free: CLIPScore** = $2.5 \cdot \max(\cos(E_I, E_T), 0)$ - cosine similarity between the CLIP image embedding and the CLIP text embedding, no references needed. RefCLIPScore harmonic-means that with the reference similarity. Cheap and useful, but it inherits CLIP’s blindness to counting, spatial relations and negation, and it saturates on long captions.

**For OCR: CER / WER**, the same edit-distance metrics as ASR. $\mathrm{CER} = (S + D + I) / N$ over characters. Document parsing adds structural metrics: **TEDS** (tree edit distance similarity) for HTML tables and **CDM** for formulas, which is what OmniDocBench reports.

**The honest pitfalls.**

- **Normalisation dominates**, exactly as in ASR: lowercasing, punctuation stripping and tokenizer choice move BLEU/CIDEr by several points. Always compare under one normaliser (the COCO eval toolkit’s PTB tokenizer is the convention).
- **n-gram metrics are near-useless for dense captions.** They were designed for a 10-word sentence with 5 references. A 150-word paragraph has no references, and two equally correct paragraphs can share almost no 4-grams. [CapArena (ACL 2025 Findings)](https://arxiv.org/abs/2503.12329) showed that on detailed captioning the classic metrics do not track human preference at all, and that arena-style **LLM-as-judge** (or reference-free CLIP-based scores) is what actually correlates. If you are optimising dense captions against CIDEr, you are optimising the wrong thing.
- **Hallucination is invisible to CIDEr.** A caption can score well while inventing an object. Use **CHAIR** (fraction of mentioned objects not in the ground-truth annotation) alongside it.
- **Speed metrics:** images/second at a fixed resolution, and time-to-first-token if you stream. Note that OCR models process *tiles*, so latency scales with page area, not with image count.

The cell below implements BLEU-4 and a simplified CIDEr in numpy (no extra dependency), plus CER via `jiwer`, on a toy example.

------------------------------------------------------------------------

``` python
import math
from collections import Counter

import jiwer
import numpy as np


def ngrams(tokens, n):
    "Counter of the n-grams in a token list."
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu(candidates, references, max_n=4):
    "Corpus BLEU-n: clipped modified precision over all candidates + brevity penalty."
    p_num, p_den = [0] * max_n, [0] * max_n
    cand_len = ref_len = 0
    for cand, refs in zip(candidates, references):
        c = cand.lower().split()
        rs = [r.lower().split() for r in refs]
        cand_len += len(c)
        closest = min(rs, key=lambda r: (abs(len(r) - len(c)), len(r)))  # BLEU uses the closest ref length
        ref_len += len(closest)
        for n in range(1, max_n + 1):
            cand_ng = ngrams(c, n)
            max_ref = Counter()  # clip each n-gram at its max count in any reference
            for r in rs:
                for g, cnt in ngrams(r, n).items():
                    max_ref[g] = max(max_ref[g], cnt)
            p_num[n - 1] += sum(min(cnt, max_ref[g]) for g, cnt in cand_ng.items())
            p_den[n - 1] += max(sum(cand_ng.values()), 1)
    precisions = [num / den for num, den in zip(p_num, p_den)]
    if min(precisions) == 0.0:
        return 0.0  # a missing 4-gram zeroes BLEU - the usual complaint about it
    bp = 1.0 if cand_len > ref_len else math.exp(1 - ref_len / max(cand_len, 1))
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)


def cider(candidates, references, max_n=4):
    "Simplified CIDEr: TF-IDF weighted n-gram cosine, averaged over n=1..4, x10.\n\n    IDF is estimated from the reference set you pass in, so it is only meaningful\n    over a corpus - on 3 images the IDF term is noise. Smoothed IDF and no length\n    penalty, so this is 'CIDEr-lite', not the official CIDEr-D implementation.\n    "
    n_docs = len(references)
    per_n = []
    for n in range(1, max_n + 1):
        df = Counter()
        for refs in references:
            seen = set()
            for r in refs:
                seen |= set(ngrams(r.lower().split(), n))
            for g in seen:
                df[g] += 1

        def tfidf(text):
            counts = ngrams(text.lower().split(), n)
            v = {g: c * (math.log((n_docs + 1) / (df.get(g, 0) + 1)) + 1.0) for g, c in counts.items()}
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            return v, norm

        img_scores = []
        for cand, refs in zip(candidates, references):
            cv, cn = tfidf(cand)
            sims = []
            for r in refs:
                rv, rn = tfidf(r)
                sims.append(sum(cv[g] * rv.get(g, 0.0) for g in cv) / (cn * rn))
            img_scores.append(float(np.mean(sims)) if sims else 0.0)
        per_n.append(float(np.mean(img_scores)))
    return 10.0 * float(np.mean(per_n))


# Toy captioning example: one image, five human references, two candidate captions.
refs = [[
    "two cats are sleeping on a pink couch",
    "a pair of cats lying on a pink sofa",
    "two cats laying on a couch next to two remotes",
    "a couple of cats asleep on a pink blanket",
    "two cats sleeping on a couch with remote controls",
]]
good = ["two cats sleeping on a pink couch"]
bad = ["a dog standing in a field of grass"]

print(f"good  BLEU-4 {bleu(good, refs):.3f}   CIDEr-lite {cider(good, refs):5.2f}")
print(f"bad   BLEU-4 {bleu(bad, refs):.3f}   CIDEr-lite {cider(bad, refs):5.2f}")

# OCR is scored with edit distance instead - character error rate, same as ASR.
truth = "Invoice #4417 - Total: $1,284.50"
ocr_out = "Invoice #4417 - Total: $l,284.5O"  # classic 1/l and 0/O confusions
print(f"\nOCR   CER {jiwer.cer(truth, ocr_out):.3f}   WER {jiwer.wer(truth, ocr_out):.3f}")
```

    good  BLEU-4 0.867   CIDEr-lite  3.97
    bad   BLEU-4 0.000   CIDEr-lite  0.56

    OCR   CER 0.062   WER 0.200

## 5. Datasets

| Dataset | Contents | Size | Scope | License | Typical use |
|----|----|----|----|----|----|
| [COCO Captions](https://huggingface.co/datasets/lmms-lab/COCO-Caption2017) | Everyday photos, 5 human captions each | 123k images / 616k captions | en, 80 common object classes | CC-BY 4.0 (annotations) | **The** captioning benchmark; the Karpathy split (113k train / 5k val / 5k test) is the convention |
| [Flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k) | Flickr photos of people/actions, 5 captions each | 31k images | en | research-only terms | Second standard benchmark; also the entity-grounding variant Flickr30k-Entities |
| [nocaps](https://nocaps.org/) | Open Images photos with **novel objects** unseen in COCO training | 15k val/test images | en | CC-BY 2.0 | Tests generalisation past COCO’s 80 classes |
| [TextCaps](https://textvqa.org/textcaps/) | Images whose caption **requires reading the text in them** | 28k images / 145k captions | en | CC-BY 4.0 | Joint OCR + captioning |
| [Conceptual Captions / CC12M](https://huggingface.co/datasets/google-research-datasets/conceptual_captions) | Web alt-text, machine-cleaned | 3.3M / 12M pairs | en, web-scale | image URLs only | Pretraining, not evaluation (noisy) |
| [DOCCI](https://huggingface.co/datasets/google/docci) | **Dense** human-written descriptions, ~136 words each | 15k images | en | CC-BY 4.0 | Detailed-captioning eval; the DCI dataset is its sibling |
| [LAION-COCO / recaptioned sets](https://huggingface.co/datasets/laion/laion-coco) | Web images with synthetic model-written captions | 600M | en | CC-BY 4.0 (metadata) | Training text-to-image models |
| [IAM Handwriting](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) | Handwritten English lines | 13k lines | en, cursive | research, registration required | TrOCR-style handwriting OCR (CER) |
| [FUNSD](https://guillaumejaume.github.io/FUNSD/) | Noisy scanned forms with boxes + entity links | 199 forms | en | research use | Form understanding / KIE |
| [DocVQA](https://huggingface.co/datasets/lmms-lab/DocVQA) | Document images with questions | 50k Q / 12k images | en | non-commercial research | Document VLM eval |
| [SROIE](https://rrc.cvc.uab.es/?ch=13) | Scanned receipts | 1k receipts | en | competition terms | Receipt OCR + field extraction |
| [OmniDocBench](https://github.com/opendatalab/OmniDocBench) | Diverse PDF pages with layout, tables, formulas | 981 pages | en/zh | Apache 2.0 | The 2025-2026 document-parsing leaderboard |

**This notebook evaluates on** [`lmms-lab/COCO-Caption2017`](https://huggingface.co/datasets/lmms-lab/COCO-Caption2017) (val split, 5 references per image, ungated parquet), streamed so we only pull the handful of images we score. Gated / restricted: Flickr30k and IAM require accepting terms; DocVQA and SROIE are non-commercial research only; `google/paligemma2-*` is a **gated model** (accept the Gemma licence on the Hub before it will download).

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

Captioning has no single authoritative leaderboard any more, because the task fragmented. Use three:

- **[CapArena](https://github.com/njucckevin/CapArena)** (arena-style, GPT-judge) for **detailed captioning** - the only ranking that tracks human preference on long captions (93.4% correlation with human rankings).
- **[OpenVLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)** for general VLM capability (MMBench, MMMU, and the OCR-heavy OCRBench / DocVQA columns).
- **[OmniDocBench](https://github.com/opendatalab/OmniDocBench)** and **[olmOCR-Bench](https://huggingface.co/allenai/olmOCR-2-7B-1025)** for **document OCR**.

| Model | Params | License | Scope | Architecture | Best for |
|----|----|----|----|----|----|
| ViT-GPT2 (`nlpconnect`) | 0.24B | Apache 2.0 | short captions, en | ViT encoder + GPT-2 decoder | the 2021 baseline; cheap, generic |
| BLIP-base / -large | 0.25B / 0.47B | BSD-3 | short captions, en | ViT + BERT-ish decoder, CapFilt | zero-prompt alt-text at volume; the safe default |
| BLIP-2 OPT-2.7b | ~3.7B | MIT | captions + VQA | frozen ViT-g + **Q-Former** + frozen OPT | historical importance; ~7.5 GB in fp16, marginal on a 12 GB card |
| **Florence-2-base / -large** | 0.23B / 0.77B | MIT | caption, dense caption, OCR, detection, grounding | seq2seq (DaViT + BART), **task tokens** | short **and** dense captions from one tiny model; this notebook’s workhorse |
| Moondream 2/3 | ~2B (3 is MoE, ~2B active) | Apache 2.0 (2) | caption, VQA, OCR, pointing | small VLM | edge captioning; explicit short/normal/long caption lengths |
| **SmolVLM2-2.2B** | 2.2B | Apache 2.0 | caption, VQA, video | SigLIP + SmolLM2, pixel-shuffle | prompted captioning at ~5 GB VRAM |
| SmolVLM-256M / -500M | 0.26B / 0.5B | Apache 2.0 | caption, VQA | same, tiny | “smallest VLM in the world”; \<1 GB VRAM |
| PaliGemma 2 (3B/10B/28B) | 3B+ | Gemma (gated) | caption, OCR, detection | SigLIP-So400m + Gemma 2 | fine-tuning base; ships explicit long-caption checkpoints |
| Qwen3-VL-2B / -4B | 2B / 4B | Apache 2.0 | everything | ViT + Qwen3, dynamic resolution | best small general VLM; strong OCR and grounding |
| Qwen2.5-VL-3B/7B | 3B / 7B | Apache 2.0 (3B/7B) | everything | ViT + Qwen2.5 | the 2025 workhorse; 3B fits, 7B does not (fp16) |
| InternVL3 (1B-78B) | 1B-78B | Apache 2.0 (weights) | everything | InternViT + LLM | the small end is competitive; 8B+ needs quantization here |
| Molmo 7B/72B | 7B / 72B | Apache 2.0 | caption + pointing | ViT + LLM, PixMo dense captions | dense captions (trained on human speech-transcribed descriptions) |
| **GOT-OCR 2.0** | 0.58B | Apache 2.0 | OCR only | high-compression ViT + Qwen-0.5B decoder | plain/markdown/LaTeX/table/formula OCR in one small model |
| TrOCR base/large | 0.33B / 0.56B | MIT | OCR, single text **line** | ViT encoder + RoBERTa decoder | handwriting line recognition after a line detector |
| Donut / Nougat | 0.2B / 0.35B | MIT / CC-BY-NC | OCR-free doc / PDF understanding | Swin + BART | receipts and forms (Donut), academic PDFs to LaTeX (Nougat) |
| dots.ocr / olmOCR 2 / DeepSeek-OCR / PaddleOCR-VL | 1B-7B | Apache/MIT (varies) | document parsing | high-res tiled VLMs | the 2025-2026 document-parsing SOTA; the 7B ones need quantization here |

**Who wins what.**

- **Accuracy on dense captions:** the big general VLMs (Qwen3-VL-235B, InternVL3-78B, Molmo-72B, and the closed GPT/Gemini/Claude models). None of them fit on this box. Among things that do fit, Qwen3-VL-4B and SmolVLM2-2.2B write the best paragraphs.
- **Accuracy per parameter:** **Florence-2**. 0.23B parameters that do short captions, dense captions, OCR and grounding is still, two years on, an absurd deal - and its `<MORE_DETAILED_CAPTION>` output is what a lot of recaptioning pipelines quietly use.
- **Speed / cost at volume:** ViT-GPT2 and BLIP-base, or SmolVLM-256M if you need a prompt.
- **OCR:** the specialists, not the generalists. A 0.58B GOT-OCR 2.0 beats a 7B general VLM on document text, because it was trained at document resolution on structured targets.

Tying that back to section 2: alt-text at billions of images/day is a BLIP/SmolVLM-256M problem; recaptioning a training set is a Florence-2 `<MORE_DETAILED_CAPTION>` problem; an invoice pipeline is a GOT-OCR / dots.ocr problem; a GUI agent is a Qwen3-VL problem. **Rows too big for a 12 GB card in fp16:** BLIP-2 OPT-2.7b (~7.5 GB, works but leaves little headroom), Qwen2.5-VL-7B (~15 GB), Molmo-7B (~15 GB), InternVL3-8B+ and everything above it, and the 7B document models - all need 4-bit quantization or a bigger GPU.

------------------------------------------------------------------------

## 7. Setup

Everything below runs on a 12 GB RTX 3060 (or CPU, slowly), and **every model loads through Hugging Face `transformers`** - no vendor packages. Package roles:

- `transformers` (\>=5.13) + `torch` - all four runnable models (BLIP, Florence-2, SmolVLM2, GOT-OCR 2.0)
- `accelerate` - `device_map` placement
- `datasets` - the COCO Captions eval slice (streamed)
- `pillow` - image loading and display
- `jiwer` - CER for the OCR toy example
- `pyecharts` + `pandas` - benchmark chart and table
- `numpy` - the BLEU / CIDEr implementations above

**Two transformers-5 gotchas worth knowing.** (1) Florence-2 is now **natively supported**: use the `florence-community/Florence-2-base` mirror with `Florence2ForConditionalGeneration` and **no `trust_remote_code`**. The original `microsoft/Florence-2-*` repos still ship custom modelling code and require `trust_remote_code=True`, which is the source of most Florence-2 error reports. (2) The old `image-to-text` pipeline is legacy; modern VLMs go through `AutoModelForImageTextToText` / the `image-text-to-text` pipeline. This notebook uses explicit processor + model classes throughout, which works either way.

All downloads (sample images, HF model + dataset cache) land in `DL_tasks/datasets/`, which is gitignored.

------------------------------------------------------------------------

``` python
# Everything runs through Hugging Face transformers - no model-specific packages.
# %pip install -q torch transformers accelerate datasets pillow jiwer numpy pandas pyecharts num2words

# Optional: 4-bit quantization if you want to try a model bigger than ~3B on a 12 GB card
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

# 1. A photo: the canonical COCO val2017 cats image (two cats on a pink couch, two remotes).
PHOTO = DATA_DIR / "coco_cats.jpg"
if not PHOTO.exists():
    urllib.request.urlretrieve("http://images.cocodataset.org/val2017/000000039769.jpg", PHOTO)

# 2. A text-bearing image: the transformers GOT-OCR test fixture.
DOC = DATA_DIR / "sample_ocr.jpg"
if not DOC.exists():
    urllib.request.urlretrieve(
        "https://huggingface.co/datasets/hf-internal-testing/fixtures_got_ocr/resolve/main/image_ocr.jpg",
        DOC,
    )

photo = Image.open(PHOTO).convert("RGB")
doc = Image.open(DOC).convert("RGB")
print("photo:", photo.size, "| doc:", doc.size)
display(photo.resize((320, 240)))
display(doc.resize((480, int(480 * doc.height / doc.width))))

from datasets import load_dataset

# Eval set: COCO Captions 2017 val, 5 human references per image. Ungated parquet.
# Streamed so we download only the images we score, not the whole 789 MB split.
N_EVAL = 12
stream = load_dataset(
    "lmms-lab/COCO-Caption2017", split="val", streaming=True, cache_dir=HF_CACHE
)

eval_images, eval_refs = [], []
for row in stream.take(N_EVAL):
    eval_images.append(row["image"].convert("RGB"))
    eval_refs.append([c.strip() for c in row["answer"]])

print(f"{len(eval_images)} eval images, {len(eval_refs[0])} references each")
print("refs[0]:", eval_refs[0][:2])
display(eval_images[0].resize((240, int(240 * eval_images[0].height / eval_images[0].width))))
```

    photo: (640, 480) | doc: (762, 1000)

![](05_Image_to_Text_files/figure-html/cell-5-output-2.png)

![](05_Image_to_Text_files/figure-html/cell-5-output-3.png)

    12 eval images, 5 references each
    refs[0]: ['A black Honda motorcycle parked in front of a garage.', 'A Honda motorcycle parked in a grass driveway']

![](05_Image_to_Text_files/figure-html/cell-5-output-5.png)

## 8. BLIP - the zero-prompt workhorse

BLIP (Salesforce, 2022) is still the model most production alt-text pipelines run, and for good reason: 0.25B/0.47B parameters, BSD-3 licensed, no prompt required, and a caption in ~50 ms on a modern GPU. Its CapFilt bootstrapping (caption the web images with the model, then filter the noisy human alt-text with a matcher) was the trick that let it beat models trained on far more data.

It supports **conditional** captioning: pass a text prefix and the decoder continues it, which is a poor man’s prompt (“a photography of …”, “a picture of the weather …”). Checkpoints: `Salesforce/blip-image-captioning-base` (0.25B) and `-large` (0.47B). What you get is a short, safe, generic sentence - it will tell you “two cats laying on a couch”, never “a tabby and a calico asleep on a pink velvet sofa next to two TV remotes”.

------------------------------------------------------------------------

``` python
from transformers import BlipForConditionalGeneration, BlipProcessor

blip_id = "Salesforce/blip-image-captioning-large"
blip_proc = BlipProcessor.from_pretrained(blip_id, cache_dir=HF_CACHE)
blip = BlipForConditionalGeneration.from_pretrained(
    blip_id, dtype=dtype, cache_dir=HF_CACHE
).to(device)
vram("blip loaded")

with torch.inference_mode():
    # Unconditional: image only, no prompt. This is pure image-to-text.
    t0 = time.perf_counter()
    inputs = blip_proc(images=photo, return_tensors="pt").to(device, dtype)
    out = blip.generate(**inputs, max_new_tokens=40, num_beams=3)
    print(f"[{time.perf_counter() - t0:.2f}s] unconditional:",
          blip_proc.decode(out[0], skip_special_tokens=True))

    # Conditional: seed the decoder with a prefix and let it continue.
    for prefix in ["a photography of", "the furniture in this picture is"]:
        inputs = blip_proc(images=photo, text=prefix, return_tensors="pt").to(device, dtype)
        out = blip.generate(**inputs, max_new_tokens=40, num_beams=3)
        print(f"  prefix {prefix!r:35s} ->", blip_proc.decode(out[0], skip_special_tokens=True))

del blip, blip_proc, inputs, out
free_memory()
vram("after blip")
```

    VRAM blip loaded           0.90 GB allocated /  0.91 GB reserved
    [0.60s] unconditional: there are two cats laying on a couch with remote controls
      prefix 'a photography of'                  -> a photography of a couple of cats laying on top of a pink couch
      prefix 'the furniture in this picture is'  -> the furniture in this picture is pink and has two cats sleeping on it
    VRAM after blip            0.01 GB allocated /  0.02 GB reserved

## 9. Florence-2 - short caption vs dense caption vs OCR from one 0.23B model

Florence-2 (Microsoft, 2024) is the most interesting model in this notebook. It is a plain seq2seq model (DaViT vision encoder, BART-style text encoder-decoder) whose **task is selected by a special token** rather than by a natural-language prompt, and it was trained on FLD-5B: 5.4 billion annotations over 126 million images, auto-generated and iteratively refined. The result is a 0.23B model that captions, densely captions, does OCR, detects objects and grounds phrases.

The three caption task tokens are exactly the short-vs-dense contrast this notebook is about:

- `<CAPTION>` - one clause of alt-text
- `<DETAILED_CAPTION>` - a sentence or two with attributes and setting
- `<MORE_DETAILED_CAPTION>` - a paragraph enumerating objects, colours, positions, background

Plus `<OCR>` and `<OCR_WITH_REGION>` for text, `<OD>` / `<DENSE_REGION_CAPTION>` / `<CAPTION_TO_PHRASE_GROUNDING>` for boxes.

**Checkpoint note.** Use the transformers-native mirrors `florence-community/Florence-2-base` (0.23B) or `florence-community/Florence-2-large` (0.77B) with `Florence2ForConditionalGeneration` - **no `trust_remote_code`**. The original `microsoft/Florence-2-*` repos still carry custom modelling code and need `trust_remote_code=True`. The raw output is a tagged string; `processor.post_process_generation(text, task=..., image_size=...)` parses it.

------------------------------------------------------------------------

``` python
from transformers import AutoProcessor, Florence2ForConditionalGeneration

flor_id = "florence-community/Florence-2-base"  # 0.23B; swap for -large (0.77B)
flor_proc = AutoProcessor.from_pretrained(flor_id, cache_dir=HF_CACHE)
flor = Florence2ForConditionalGeneration.from_pretrained(
    flor_id, dtype=dtype, device_map=device, cache_dir=HF_CACHE
)
vram("florence loaded")

def florence(image, task, max_new_tokens=256):
    "Run one Florence-2 task token on an image and return the parsed output."
    inputs = flor_proc(text=task, images=image, return_tensors="pt").to(flor.device, dtype)
    with torch.inference_mode():
        ids = flor.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=3, do_sample=False)
    text = flor_proc.batch_decode(ids, skip_special_tokens=False)[0]
    return flor_proc.post_process_generation(text, task=task, image_size=image.size)[task]

for task in ["<CAPTION>", "<DETAILED_CAPTION>", "<MORE_DETAILED_CAPTION>"]:
    t0 = time.perf_counter()
    caption = florence(photo, task)
    print(f"{task:24s} [{time.perf_counter() - t0:4.1f}s] {caption}\n")

# Same model, same weights, different task token: now read the text in the document image.
t0 = time.perf_counter()
print("<OCR> ->", florence(doc, "<OCR>", max_new_tokens=512))
print(f"({time.perf_counter() - t0:.1f}s)")

# <OCR_WITH_REGION> additionally returns quad boxes for each text span:
regions = florence(doc, "<OCR_WITH_REGION>", max_new_tokens=1024)
print("\nspans:", regions["labels"][:5])
print("first quad box:", [round(v, 1) for v in regions["quad_boxes"][0]])

del flor, flor_proc
free_memory()
vram("after florence")
```

    VRAM florence loaded       0.47 GB allocated /  0.49 GB reserved
    <CAPTION>                [ 0.8s] Two cats laying on a pink couch next to a remote control.

    <DETAILED_CAPTION>       [ 0.2s] The image shows two cats laying on top of a pink blanket on a red couch, with two remotes beside them.

    <MORE_DETAILED_CAPTION>  [ 0.4s] The image shows two cats lying on a pink blanket on a red couch. The cat on the left is lying on its back with its head resting on its front paws and its tail curled around its body. It appears to be sleeping or resting, with its eyes closed and its mouth slightly open. On the right side of the image, there is another cat lying down with its paws on the blanket. Both cats have brown and black stripes on their bodies. There are two white remote controls resting on the couch next to them.

    <OCR> -> REDRED QUALITY IMPROVEMENTSUGGESTION/SOLUTION FORMName/Phone Ex: : M. Hammam P. Harper, P. MartinezDate: 9/3/92Supervisor/Manager: J. S. VigandRed Group: LicenseeSuggestion:Discontinue coal retention analyzes on license submittedproduct samples. (Notes: Coal Extension testing is notproduct samples as stability and inspection testing is soft.other inspection system is notamounts as assets stability and Inspection testing is requiredspots in cigarettes physical integrity. The proposedaction to increase laboratory productivity. The print proposedSuggested Solution(a): Delicate coal retention from the list of standardanalysts performed on license admittedproduct samples, Special requests for coalretention costs could still be submitted onan exception basis.Have you contacted your Manager/Supervisor?Yes- NoManager Comments: Manager, please contact suggester and forwardcomments to the Quality Council II.$1p.wp597005708
    (0.7s)

    spans: ['RED', 'RED QUALITY IMPROVEMENT', 'SUGGESTION/SOLUTION FORM', 'Name/Phone Ex: : M. Hammam, P. Harper, P ., Martinez', 'Date: 9/3/92']
    first quad box: [321, 104, 347, 104, 347, 115, 321, 115]
    VRAM after florence        0.01 GB allocated /  0.02 GB reserved

## 10. SmolVLM2 - captioning by prompting a VLM

This is the modern answer to “caption this image”: do not use a captioner, **prompt a small VLM**. SmolVLM2-2.2B (Hugging Face, 2025) is a SigLIP vision encoder plus a SmolLM2 text decoder, joined by a pixel-shuffle projector that compresses visual tokens aggressively - which is why 2.2B parameters fit in ~5 GB of VRAM and still handle video.

The advantage over BLIP/Florence-2 is **control**: the caption style, length, focus and output format are all just words in the prompt (“describe this image in one sentence for a screen reader”, “list every object you can see as JSON”). The cost is latency and VRAM, and a tendency to be chatty. Note that this is technically image-**text**-to-text (see `Multimodal/01_Image_Text_to_Text`) - the task boundary really has dissolved, and pretending otherwise would be dishonest.

Smaller siblings if 2.2B is too much: `HuggingFaceTB/SmolVLM-500M-Instruct` and `-256M-Instruct` (the latter runs image inference in under 1 GB of VRAM). Bigger alternatives that still fit here: `Qwen/Qwen3-VL-2B-Instruct` and `Qwen/Qwen2.5-VL-3B-Instruct` (~6 GB in fp16 - it fits, but leave headroom for the vision tokens of a high-resolution image).

------------------------------------------------------------------------

``` python
from transformers import AutoModelForImageTextToText, AutoProcessor

smol_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
smol_proc = AutoProcessor.from_pretrained(smol_id, cache_dir=HF_CACHE)
smol = AutoModelForImageTextToText.from_pretrained(
    smol_id, dtype=dtype, device_map=device, low_cpu_mem_usage=True, cache_dir=HF_CACHE
)
vram("smolvlm loaded")

def smolvlm(image, prompt, max_new_tokens=200):
    "Prompt SmolVLM2 with one image and return just the generated text."
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    inputs = smol_proc.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(smol.device, dtype)
    with torch.inference_mode():
        ids = smol.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return smol_proc.batch_decode(
        ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()

prompts = [
    "Write one short alt-text sentence for a screen reader.",
    "Describe this image in detail: objects, colours, materials, spatial layout, mood.",
    "List every distinct object you can see, comma separated, nothing else.",
]
for p in prompts:
    t0 = time.perf_counter()
    print(f"> {p}\n  [{time.perf_counter() - t0:4.1f}s] {smolvlm(photo, p)}\n")

del smol, smol_proc
free_memory()
vram("after smolvlm")
```

    [transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`

    VRAM smolvlm loaded        4.50 GB allocated /  4.50 GB reserved

    [transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`

    > Write one short alt-text sentence for a screen reader.
      [ 0.0s] Two tabby cats are sleeping on a pink blanket.

    [transformers] Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict, not in `**kwargs`

    > Describe this image in detail: objects, colours, materials, spatial layout, mood.
      [ 0.0s] In the image, two tabby cats are comfortably sprawled out on a pink blanket, their bodies relaxed and at ease. The cat on the left, with its fur a mix of black and brown stripes, is lying on its side, its head resting on its paws. Its companion, a slightly larger cat with a similar pattern, is sprawled out on its stomach, its head also resting on its paws. Both cats are facing the same direction, their eyes closed in a state of peaceful slumber. The pink blanket they're resting on is soft and inviting, providing a stark contrast to the cats' striped fur. In the background, a remote control lies on the couch, perhaps indicating that the cats were watching something before they decided to take a nap. The overall mood of the image is one of tranquility and contentment, as the cats seem to be enjoying their rest in each other's company.

    > List every distinct object you can see, comma separated, nothing else.
      [ 0.0s] cat, remote, pink, blanket

    VRAM after smolvlm         0.01 GB allocated /  0.02 GB reserved

## 11. GOT-OCR 2.0 - the OCR specialist

The counter-argument to “just prompt a VLM”: on documents, a small specialist wins. GOT-OCR 2.0 (StepFun, 2024) is **0.58B parameters** - a high-compression ViT encoder (1024x1024 page into ~256 tokens) feeding a Qwen2-0.5B decoder - and it beats general VLMs many times its size at reading pages, because it was trained at document resolution on structured targets. “OCR-2.0” means it emits not just characters but **structure**: markdown, LaTeX formulas, HTML tables, sheet music, molecular formulas.

Two modes: plain text (default) and `format=True`, which asks for markdown/LaTeX. It also supports interactive OCR (pass a box or a colour to transcribe only that region) and multi-page/crop modes for large pages.

For **single text lines** (a handwriting pipeline, after a line detector), `microsoft/trocr-base-handwritten` / `-printed` is the classic choice - `TrOCRProcessor` + `VisionEncoderDecoderModel`, 0.33B, trained on IAM. For **full document parsing** in 2026 the field has moved to `rednote-hilab/dots.ocr`, `allenai/olmOCR-2-*`, `deepseek-ai/DeepSeek-OCR` and PaddleOCR-VL; they are 1B-7B and mostly need quantization to sit comfortably in 12 GB alongside anything else.

------------------------------------------------------------------------

``` python
from transformers import AutoModelForImageTextToText, AutoProcessor

got_id = "stepfun-ai/GOT-OCR-2.0-hf"  # 0.58B
got_proc = AutoProcessor.from_pretrained(got_id, use_fast=True, cache_dir=HF_CACHE)
got = AutoModelForImageTextToText.from_pretrained(
    got_id, dtype=dtype, device_map=device, cache_dir=HF_CACHE
)
vram("got-ocr loaded")

t0 = time.perf_counter()
inputs = got_proc(doc, return_tensors="pt").to(got.device, dtype)
with torch.inference_mode():
    ids = got.generate(
        **inputs, do_sample=False, max_new_tokens=1024,
        tokenizer=got_proc.tokenizer, stop_strings="<|im_end|>",
    )
text = got_proc.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"[{time.perf_counter() - t0:.1f}s] plain OCR:\n{text}\n")

# format=True asks for structured output (markdown / LaTeX) instead of a flat string.
inputs = got_proc(doc, return_tensors="pt", format=True).to(got.device, dtype)
with torch.inference_mode():
    ids = got.generate(
        **inputs, do_sample=False, max_new_tokens=1024,
        tokenizer=got_proc.tokenizer, stop_strings="<|im_end|>",
    )
print("formatted OCR:\n", got_proc.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[:600])

del got, got_proc, inputs, ids
free_memory()
vram("after got-ocr")
```

    [transformers] The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.

    VRAM got-ocr loaded        1.13 GB allocated /  1.13 GB reserved

    [transformers] Ignoring clean_up_tokenization_spaces=True for BPE tokenizer TokenizersBackend. The clean_up_tokenization post-processing step is designed for WordPiece tokenizers and is destructive for BPE (it strips spaces before punctuation). Set clean_up_tokenization_spaces=False to suppress this warning, or set clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output=True to force cleanup anyway.

    [2.6s] plain OCR:
    R&D QUALITY IMPROVEMENT
    SUGGESTION/SOLUTION FORM
    Name/Phone Ext. : M. Hamann, P. Harper, P. Martinez
    Date: 9/3/92
    Supervisor/Manager: J. S. Wigand
    R&D Group: Licensee
    Suggestion:
    Discontinue coal retention analyses on licensee submitted
    product samples. (Note: Coal Retention testing is not
    performed by most licensees. Other B&W physical
    measurements as ends stability and inspection for soft
    spots in cigarettes are thought to be sufficient measures
    to assure cigarette physical integrity. The proposed
    action will increase laboratory productivity.)
    Suggested Solution(s) : Delete coal retention from the list of standard
    analyses performed on licensee submitted
    product samples. Special requests for coal
    retention testing could still be submitted on
    an exception basis.
    Have you contacted your Manager/Supervisor?
    Yes
    No
    Manager Comments: Manager, please contact suggester and forward
    comments to the Quality Council.
    qip.wp
    597005708

    formatted OCR:
     \title{
    R\&D QUALITY IMPROVEMENT
    }
    SUGGESTION/SOLUTION FORM
    Name/Phone Ext.: M. Hamann, P. Harper, P. Martinez Date: \(9 / 3 / 92\)
    Supervisor/Manager: J. S. Wigand
    R\&D Group: Licensee
    Suggestion: Discontinue coal retention analyses on licensee submitted product samples. (Note: Coal Retention testing is not performed by most licensees. Other B&W physical measurements as ends stability and inspection for soft spots in cigarettes are thought to be sufficient measures to assure cigarette physical integrity. The proposed action will increase laboratory productivity.)
    Suggested Solution(s): Delete
    VRAM after got-ocr         0.01 GB allocated /  0.02 GB reserved

## 12. Head-to-head Benchmark

Four captioners on the **same 12 COCO val2017 images**, scored against the same 5 human references each, under the same normalisation (lowercase, split on whitespace), with wall-clock latency. Each model is loaded, measured, and **freed before the next one loads**, so VRAM stays flat.

Models: ViT-GPT2 (the 2021 baseline), BLIP-base, BLIP-large, and Florence-2-base `<CAPTION>`.

**Read this as a smoke test, not a leaderboard.** 12 images is far too few for a stable CIDEr - and our CIDEr-lite estimates IDF from those same 12 images, which real CIDEr would estimate over the full 5k-image reference corpus. The published numbers to compare against come from the full Karpathy test split. What the sample *does* show honestly is the relative latency and the shape of each model’s output.

------------------------------------------------------------------------

``` python
from transformers import (
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipProcessor,
    Florence2ForConditionalGeneration,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
)


def load_vit_gpt2():
    "ViT encoder + GPT-2 decoder, the 2021-era captioner. 0.24B."
    mid = "nlpconnect/vit-gpt2-image-captioning"
    model = VisionEncoderDecoderModel.from_pretrained(mid, dtype=dtype, cache_dir=HF_CACHE).to(device)
    img_proc = ViTImageProcessor.from_pretrained(mid, cache_dir=HF_CACHE)
    tok = AutoTokenizer.from_pretrained(mid, cache_dir=HF_CACHE)

    def caption(image):
        px = img_proc(images=image, return_tensors="pt").pixel_values.to(device, dtype)
        with torch.inference_mode():
            ids = model.generate(px, max_new_tokens=32, num_beams=3)
        return tok.decode(ids[0], skip_special_tokens=True).strip()

    return caption, [model, img_proc, tok]


def load_blip(size="base"):
    "BLIP captioner, unconditional (no prompt). 0.25B (base) / 0.47B (large)."
    mid = f"Salesforce/blip-image-captioning-{size}"
    proc = BlipProcessor.from_pretrained(mid, cache_dir=HF_CACHE)
    model = BlipForConditionalGeneration.from_pretrained(mid, dtype=dtype, cache_dir=HF_CACHE).to(device)

    def caption(image):
        inputs = proc(images=image, return_tensors="pt").to(device, dtype)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=32, num_beams=3)
        return proc.decode(ids[0], skip_special_tokens=True).strip()

    return caption, [model, proc]


def load_florence(task="<CAPTION>"):
    "Florence-2-base with a caption task token. 0.23B."
    mid = "florence-community/Florence-2-base"
    proc = AutoProcessor.from_pretrained(mid, cache_dir=HF_CACHE)
    model = Florence2ForConditionalGeneration.from_pretrained(
        mid, dtype=dtype, device_map=device, cache_dir=HF_CACHE
    )

    def caption(image):
        inputs = proc(text=task, images=image, return_tensors="pt").to(model.device, dtype)
        with torch.inference_mode():
            ids = model.generate(**inputs, max_new_tokens=64, num_beams=3, do_sample=False)
        raw = proc.batch_decode(ids, skip_special_tokens=False)[0]
        return proc.post_process_generation(raw, task=task, image_size=image.size)[task].strip()

    return caption, [model, proc]


def benchmark(name, loader):
    "Load a model, caption every eval image, score it, free it. Returns a result dict."
    caption_fn, handles = loader()
    t0 = time.perf_counter()
    caps = [caption_fn(img) for img in eval_images]
    elapsed = time.perf_counter() - t0
    for h in handles:
        del h
    del caption_fn, handles
    free_memory()
    res = {
        "model": name,
        "bleu4": bleu(caps, eval_refs),
        "cider": cider(caps, eval_refs),
        "sec_per_image": elapsed / len(eval_images),
        "captions": caps,
    }
    print(f"{name:22s} BLEU-4 {res['bleu4']:.3f}  CIDEr-lite {res['cider']:5.2f}  "
          f"{res['sec_per_image']:.2f} s/img")
    vram(f"after {name}")
    return res
```

``` python
results = [
    benchmark("vit-gpt2", load_vit_gpt2),
    benchmark("blip-base", lambda: load_blip("base")),
    benchmark("blip-large", lambda: load_blip("large")),
    benchmark("florence-2-base", load_florence),
]
vram("benchmark done")

# SmolVLM2 slots in the same way if you want a prompted VLM in the comparison - it is
# ~10x slower per image and its captions are long, which n-gram metrics punish hard:
#
# def load_smol():
#     proc = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct", cache_dir=HF_CACHE)
#     model = AutoModelForImageTextToText.from_pretrained(
#         "HuggingFaceTB/SmolVLM2-2.2B-Instruct", dtype=dtype, device_map=device, cache_dir=HF_CACHE)
#     def caption(image):
#         msgs = [{"role": "user", "content": [{"type": "image", "image": image},
#                  {"type": "text", "text": "Caption this image in one short sentence."}]}]
#         inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
#                                        return_dict=True, return_tensors="pt").to(model.device, dtype)
#         with torch.inference_mode():
#             ids = model.generate(**inp, max_new_tokens=48, do_sample=False)
#         return proc.batch_decode(ids[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
#     return caption, [model, proc]
# results.append(benchmark("smolvlm2-2.2b", load_smol))
```

    [transformers] VisionEncoderDecoderModel LOAD REPORT from: nlpconnect/vit-gpt2-image-captioning
    Key                                                       | Status     |  | 
    ----------------------------------------------------------+------------+--+-
    decoder.transformer.h.{0...11}.crossattention.masked_bias | UNEXPECTED |  | 
    decoder.transformer.h.{0...11}.attn.masked_bias           | UNEXPECTED |  | 

    Notes:
    - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.

    [transformers] Both `max_new_tokens` (=32) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] We strongly recommend passing in an `attention_mask` since your input_ids may be padded. See https://huggingface.co/docs/transformers/troubleshooting#incorrect-output-when-padding-tokens-arent-masked.
    You may ignore this warning if your `pad_token_id` (50256) is identical to the `bos_token_id` (50256), `eos_token_id` (50256), or the `sep_token_id` (None), and your input is not padded.
    [transformers] Both `max_new_tokens` (=32) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=32) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=32) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=32) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=32) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=32) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=32) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=32) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=32) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=32) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)
    [transformers] Both `max_new_tokens` (=32) and `max_length`(=20) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)

    vit-gpt2               BLEU-4 0.269  CIDEr-lite  1.22  0.07 s/img
    VRAM after vit-gpt2        0.01 GB allocated /  0.02 GB reserved

    blip-base              BLEU-4 0.350  CIDEr-lite  1.49  0.10 s/img
    VRAM after blip-base       0.01 GB allocated /  0.02 GB reserved

    blip-large             BLEU-4 0.294  CIDEr-lite  1.23  0.12 s/img
    VRAM after blip-large      0.01 GB allocated /  0.02 GB reserved

    florence-2-base        BLEU-4 0.429  CIDEr-lite  1.73  0.11 s/img
    VRAM after florence-2-base  0.01 GB allocated /  0.02 GB reserved
    VRAM benchmark done        0.01 GB allocated /  0.02 GB reserved

``` python
import pandas as pd

df = pd.DataFrame([
    {k: v for k, v in r.items() if k != "captions"} for r in results
]).sort_values("cider", ascending=False)
df
```

|     | model           | bleu4    | cider    | sec_per_image |
|-----|-----------------|----------|----------|---------------|
| 3   | florence-2-base | 0.428649 | 1.729557 | 0.111076      |
| 1   | blip-base       | 0.350264 | 1.493558 | 0.096031      |
| 2   | blip-large      | 0.294068 | 1.232794 | 0.117645      |
| 0   | vit-gpt2        | 0.269456 | 1.217764 | 0.072834      |

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
    .add_yaxis("BLEU-4", [round(r["bleu4"] * 100, 2) for r in results])
    .add_yaxis("CIDEr-lite", [round(r["cider"], 2) for r in results])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="Captioning quality on 12 COCO val2017 images",
            subtitle="RTX 3060 12 GB, fp16, 5 references/image - smoke test, not a leaderboard",
        ),
        xaxis_opts=opts.AxisOpts(name="model"),
        yaxis_opts=opts.AxisOpts(name="score"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
    )
)
bar.render_notebook()
```

``` python
from pyecharts.charts import Scatter

# Quality vs speed: the decision every deployment actually makes (see section 2).
scatter = Scatter()
scatter.add_xaxis([round(r["sec_per_image"], 3) for r in results])
for r in results:
    scatter.add_yaxis(
        r["model"],
        [[round(r["sec_per_image"], 3), round(r["cider"], 2)]],
        symbol_size=18,
        label_opts=opts.LabelOpts(is_show=False),
    )
scatter.set_global_opts(
    title_opts=opts.TitleOpts(title="CIDEr-lite vs latency", subtitle="up and to the left is better"),
    xaxis_opts=opts.AxisOpts(type_="value", name="seconds / image"),
    yaxis_opts=opts.AxisOpts(type_="value", name="CIDEr-lite"),
    tooltip_opts=opts.TooltipOpts(trigger="item"),
)
scatter.render_notebook()
```

``` python
# The numbers hide the interesting part: read the actual captions side by side.
for i in range(3):
    display(eval_images[i].resize((300, int(300 * eval_images[i].height / eval_images[i].width))))
    print("  human ref :", eval_refs[i][0])
    for r in results:
        print(f"  {r['model']:16s}: {r['captions'][i]}")
    print()
```

![](05_Image_to_Text_files/figure-html/cell-15-output-1.png)

      human ref : A black Honda motorcycle parked in front of a garage.
      vit-gpt2        : a black motorcycle parked in a driveway
      blip-base       : a motorcycle parked in a yard next to a house
      blip-large      : there is a motorcycle that is parked on the grass in the yard
      florence-2-base : A black motorcycle parked in front of a house.

![](05_Image_to_Text_files/figure-html/cell-15-output-3.png)

      human ref : An office cubicle with four different types of computers.
      vit-gpt2        : a desk with a computer monitor, keyboard and mouse
      blip-base       : a room with a computer desk and a chair
      blip-large      : there is a computer desk with a chair and a computer monitor on it
      florence-2-base : A view of an office cubicle with two computers and a chair.

![](05_Image_to_Text_files/figure-html/cell-15-output-5.png)

      human ref : A small closed toilet in a cramped space.
      vit-gpt2        : a bathroom with a toilet and a sink
      blip-base       : a bathroom with a white toilet and a green rug
      blip-large      : this is a picture of a bathroom with a toilet and a bidet
      florence-2-base : A white toilet sitting next to a sink in a bathroom.

## 13. Caption Any Image (interactive)

Point the cell below at any image URL or local path. It reloads Florence-2-base (0.23B, the best size/capability trade in this notebook), runs all three caption lengths plus OCR, and frees the model afterwards. The webcam branch is guarded so a headless server skips it cleanly.

------------------------------------------------------------------------

``` python
# opencv-python-headless is a project dependency; the headless build captures from
# V4L2 fine, it only drops the GUI windows.
import io
import time

import cv2
import numpy as np
import torch
from IPython.display import Image as IPyImage
from IPython.display import Pretty, display
from PIL import Image, ImageDraw, ImageFont

CAM = 0              # /dev/video0
WARMUP = 10          # throwaway reads - auto-exposure and white balance need to settle
STREAM_SECONDS = 15  # how long a live demo runs; interrupt the kernel to stop early


def bootstrap(*names, notebook, sections):
    """Make this demo runnable on a cold kernel, without duplicating the notebook.

    The demo builds on the notebook's setup and helper cells. Instead of making you
    run them by hand - or copying them in here and letting the copies drift - this
    reads the notebook file and executes those sections itself, and only when a name
    is actually missing. Run the notebook top to bottom and it does nothing at all.
    It stops as soon as every required name exists, so trailing benchmark cells in a
    section are not run.
    """
    if all(n in globals() for n in names):
        return
    import json
    from pathlib import Path

    from IPython.utils.capture import capture_output

    path = Path(notebook)
    if not path.exists():
        raise NameError(
            f"this demo needs {', '.join(n for n in names if n not in globals())}, and cannot "
            f"find {notebook} to bootstrap from (cwd is {Path.cwd()}, expected the notebook's "
            f"own directory). Run section(s) {'; '.join(sections)} by hand instead."
        )
    print(f"cold start: running {'; '.join(sections)} from {notebook} (output suppressed)")
    heading = None
    for cell in json.loads(path.read_text())["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "markdown" and src.lstrip().startswith("## "):
            heading = src.lstrip().splitlines()[0][3:].strip()
            continue
        if cell["cell_type"] != "code" or not heading or "def bootstrap(" in src:
            continue
        if not any(heading.startswith(s) for s in sections):
            continue
        code = "".join("" if l.lstrip().startswith(("%", "!")) else l
                       for l in src.splitlines(keepends=True))
        # The setup cells print tables and display sample images. This demo only
        # wants the live stream, so swallow their output - errors still propagate.
        with capture_output():
            exec(compile(code, f"{notebook} [{heading}]", "exec"), globals())
        if all(n in globals() for n in names):
            break
    still = [n for n in names if n not in globals()]
    if still:
        raise NameError(f"bootstrapped {'; '.join(sections)} but {', '.join(still)} "
                        "are still undefined - the notebook layout may have changed.")


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
    # UVC exposure is DEVICE state and persists between processes: if anything left
    # this camera in manual mode, every frame comes back dark and never adapts
    # (measured here: mean 13/255 stuck, vs 109/255 on auto). So ask for the mode
    # explicitly instead of inheriting whatever the last program set.
    #   auto (3): correct brightness, but a dim room throttles the sensor to 15 FPS
    #   manual (1): locked 30 FPS, at whatever `exposure` level suits your lighting
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3 if auto_exposure else 1)
    if not auto_exposure:
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    # Deliberately no CAP_PROP_BUFFERSIZE: on the V4L2 backend it HALVES the
    # delivered frame rate (measured here: 67 -> 134 ms per read) and does not make
    # frames any fresher.
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


def capture_frame(**kw):
    "Open the camera, grab one settled frame, and release the device."
    cap = open_camera(**kw)
    try:
        return grab(cap)
    finally:
        cap.release()


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

    `annotate(rgb)` returns `(annotated_image, info_string)`. The image and the
    status line each own a display handle and update in place, so this needs no GUI
    and no `cv2.imshow` - it works over JupyterLab against a headless container.
    Interrupt the kernel (the stop button) to end early; the camera is still released.
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
            line = Pretty(f"frame {n:4d}   {n / (time.perf_counter() - t0):5.1f} FPS   {info}")
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
    print(f"{n} frames in {elapsed:.1f}s -> {n / max(elapsed, 1e-9):.1f} FPS end-to-end "
          "(camera + model + JPEG encode)")


def preview(seconds=5, width=640, height=480):
    "Stream the raw camera so you can frame the shot, then return the final frame."
    cap = open_camera(width=width, height=height)
    view = status = None  # created from the FIRST real frame, so no placeholder flashes up
    last, n, t0 = None, 0, time.perf_counter()
    try:
        while time.perf_counter() - t0 < seconds:
            last = grab(cap)
            n += 1
            frame = IPyImage(data=_jpeg(last))
            line = Pretty(f"framing - {seconds - (time.perf_counter() - t0):4.1f}s left, "
                          f"{n} frames (the last one is the one that gets used)")
            if view is None:
                view = display(frame, display_id=True)
                status = display(line, display_id=True)
            else:
                view.update(frame)
                status.update(line)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
    if status is not None:
        status.update(Pretty(f"captured the last of {n} frames"))
    return last


# Everything below builds on the notebook's setup and helper cells.
bootstrap("load_florence", "free_memory", "vram",
          notebook="05_Image_to_Text.ipynb",
          sections=["4. Evaluation Metrics", "7. Setup", "12. Head-to-head Benchmark"])



# Florence-2-base captions in roughly a second per frame, so this streams at ~1 FPS
# rather than 15. That is the honest number for a 0.23B VLM on a 3060, and it is why
# captioning lives behind a shutter button in real products, not in the preview loop.
caption_fn, handles = load_florence("<MORE_DETAILED_CAPTION>")


def annotate(rgb):
    "One frame -> (frame with its caption wrapped onto it, the caption again)."
    text = caption_fn(rgb)
    words, lines, line = text.split(), [], ""
    for w in words:  # wrap to something that fits a 640 px frame
        if len(line) + len(w) + 1 > 52:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    return draw_lines(rgb, lines[:4]), text[:90]


live_stream(annotate)

for h in handles:
    del h
del caption_fn, handles
free_memory()
vram("final")
```

    cold start: running 4. Evaluation Metrics; 7. Setup; 12. Head-to-head Benchmark from 05_Image_to_Text.ipynb (output suppressed)

![](05_Image_to_Text_files/figure-html/cell-16-output-3.jpeg)

    frame   49     3.2 FPS   The image shows a large water bottle sitting on top of a desk next to a computer monitor. 

    49 frames in 15.2s -> 3.2 FPS end-to-end (camera + model + JPEG encode)
    VRAM final                 0.01 GB allocated /  0.02 GB reserved

## 14. Common Frameworks

Image-to-text sits on a fault line. One half of the task is *captioning*, which is now just a VLM with a prompt and inherits the whole LLM serving stack; the other half is *OCR*, which has a mature, fast, purpose-built ecosystem that predates transformers entirely and still wins whenever the text is clean. Choosing the wrong half is the expensive mistake here: running a 2B VLM to read a receipt costs orders of magnitude more than Tesseract and is often less accurate.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [transformers](https://github.com/huggingface/transformers) | modelling | BLIP, Florence-2, GOT-OCR 2.0, TrOCR and every VLM behind `image-to-text` and `AutoModelForVision2Seq` | Apache 2.0 | Default, and unusually broad here - it covers both halves of the task |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) / [docTR](https://github.com/mindee/doctr) / [Tesseract](https://github.com/tesseract-ocr/tesseract) | modelling | Classical detect-then-recognise OCR: fast, CPU-friendly, per-word boxes and confidences | Apache 2.0 | The text is printed and clean. These are 10-100x cheaper than a VLM and give you coordinates a VLM will not |
| [peft](https://github.com/huggingface/peft) + [trl](https://github.com/huggingface/trl) | modelling | LoRA/QLoRA fine-tuning of a 2-3B captioner on this box, which is the standard recipe for domain captions | Apache 2.0 | Your captions have a house style, a vocabulary, or a required format. This teaches format extremely well |
| [Docling](https://github.com/docling-project/docling) / [unstructured](https://github.com/Unstructured-IO/unstructured) | data | Full document conversion: page layout, reading order, tables and figures out of a PDF, calling OCR only where needed | MIT / Apache 2.0 | The input is a document rather than a photograph - see `Multimodal/05_Document_Question_Answering` |
| [datasets](https://github.com/huggingface/datasets) + [img2dataset](https://github.com/rom1504/img2dataset) | data | Streaming and sharding of image-text corpora, and downloading web-scale pairs without filling the disk | Apache 2.0 / MIT | Recaptioning or training. This is where the 8 GB download rule bites hardest |
| [vLLM](https://github.com/vllm-project/vllm) / [SGLang](https://github.com/sgl-project/sglang) | inference runtime | Continuous batching for the VLM captioners - many images per GPU instead of one at a time | Apache 2.0 | Bulk captioning. Recaptioning a training set with `generate` in a loop wastes most of the card |
| [ONNX Runtime](https://onnxruntime.ai/) + [optimum](https://github.com/huggingface/optimum) / [llama.cpp](https://github.com/ggml-org/llama.cpp) | inference runtime | BLIP, Florence-2 and TrOCR at higher CPU throughput; GGUF quants of SmolVLM and Moondream for edge devices | MIT / Apache 2.0 | Captioning on CPU or on-device, where the small specialised models still make sense |
| [BentoML](https://github.com/bentoml/BentoML) / [Triton Inference Server](https://github.com/triton-inference-server/server) | serving | A captioning endpoint with batching, and ensemble graphs for detect-crop-read pipelines | Apache 2.0 / BSD-3 | OCR in production is usually a pipeline, not a model, and the server should own the chaining |
| [CLIPScore](https://github.com/jmhessel/clipscore) + [CapArena](https://github.com/njucckevin/CapArena)-style LLM judging | evaluation | Reference-free prompt-image agreement, pairwise judging, and CHAIR for hallucinated objects | MIT | Always, and specifically **instead of BLEU/CIDEr**, which are insensitive to exactly the dense-caption quality you care about |

**The 2026 default stack** is Florence-2 for captioning at volume, a small VLM through vLLM when the caption needs to follow instructions, PaddleOCR or docTR when the job is really OCR, and Docling when the input is a PDF. CLIPScore plus an LLM judge to decide between them.

The common wrong turn is reaching for the largest VLM for a job a 0.23B model does. Florence-2 recaptions faster than anything above it and is usually indistinguishable at dense-caption quality. The second is evaluating with BLEU: it rewards matching the reference’s phrasing and penalises a caption that is more detailed and more correct.

------------------------------------------------------------------------

## 15. Going Further

- **Fine-tuning.** Captioners fine-tune cheaply because the decoder is small. Florence-2 fine-tunes on a few thousand image/caption pairs ([HF blog: fine-tuning Florence-2](https://huggingface.co/blog/finetune-florence2)); BLIP fine-tunes with a plain seq2seq loop ([HF image-captioning task guide](https://huggingface.co/docs/transformers/tasks/image_captioning)); PaliGemma 2 ships explicitly as a *base* model meant to be fine-tuned rather than prompted. For VLMs, LoRA/QLoRA through `peft` + `trl` is the standard recipe and fits a 2-3B model on this 12 GB card.
- **Dense captioning at scale.** If you are recaptioning a training set, Florence-2 `<MORE_DETAILED_CAPTION>` at 0.23B is the throughput/quality sweet spot; PixMo/Molmo-style human-narrated captions and DOCCI are the reference for what “good” looks like.
- **Evaluating dense captions properly.** Do not use BLEU/CIDEr. Use [CapArena](https://github.com/njucckevin/CapArena)-style pairwise LLM judging, CLIPScore/RefCLIPScore for a cheap reference-free signal, and CHAIR to catch hallucinated objects. A CLIPScore is ~15 lines with `CLIPModel` from transformers.
- **Document pipelines.** For real PDF parsing, GOT-OCR 2.0 is the smallest good option; above it sit `rednote-hilab/dots.ocr`, `allenai/olmOCR-2-*`, `deepseek-ai/DeepSeek-OCR` and PaddleOCR-VL, benchmarked on [OmniDocBench](https://github.com/opendatalab/OmniDocBench). Classical engines (Tesseract, PaddleOCR, docTR) are still faster and cheaper when the text is clean and you only need characters, not structure.
- **Related notebooks.** `Multimodal/01_Image_Text_to_Text` (image + prompt -\> text, the superset of this task), `Multimodal/04_Visual_Question_Answering`, `04_Text_to_Image` (the inverse), `11_Zero_Shot_Image_Classification` (CLIP, whose text tower powers CLIPScore), `13_Zero_Shot_Object_Detection` (grounding, Florence-2’s other trick).

**References**

- [Show, Attend and Tell (Xu et al., 2015)](https://arxiv.org/abs/1502.03044)
- [BLIP (Li et al., 2022)](https://arxiv.org/abs/2201.12086) and [BLIP-2 / Q-Former (Li et al., 2023)](https://arxiv.org/abs/2301.12597)
- [LLaVA: visual instruction tuning (Liu et al., 2023)](https://arxiv.org/abs/2304.08485)
- [Florence-2 (Xiao et al., 2023)](https://arxiv.org/abs/2311.06242) and its [transformers model doc](https://huggingface.co/docs/transformers/model_doc/florence2)
- [TrOCR (Li et al., 2021)](https://arxiv.org/abs/2109.10282), [Donut (Kim et al., 2022)](https://arxiv.org/abs/2111.15664), [Nougat (Blecher et al., 2023)](https://arxiv.org/abs/2308.13418)
- [GOT-OCR 2.0: General OCR Theory (Wei et al., 2024)](https://arxiv.org/abs/2409.01704) and its [transformers model doc](https://huggingface.co/docs/transformers/model_doc/got_ocr2)
- [SmolVLM (Marafioti et al., 2025)](https://arxiv.org/abs/2504.05299)
- [CIDEr (Vedantam et al., 2015)](https://arxiv.org/abs/1411.5726), [SPICE (Anderson et al., 2016)](https://arxiv.org/abs/1607.08822), [CLIPScore (Hessel et al., 2021)](https://arxiv.org/abs/2104.08718)
- [CapArena: detailed captioning in the LLM era (ACL 2025 Findings)](https://arxiv.org/abs/2503.12329)
- [OpenVLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) and [OmniDocBench](https://github.com/opendatalab/OmniDocBench)

------------------------------------------------------------------------

Back to top
