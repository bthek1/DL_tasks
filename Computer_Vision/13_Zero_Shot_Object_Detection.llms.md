# Zero-Shot Object Detection

Open-vocabulary detection: how a detector localises classes it was never trained to name, the mid-2026 model landscape, the LVIS rare-class protocol, and runnable code for OWL-ViT, OWLv2, Grounding DINO and LLMDet.

Author

Benedict Thekkel

## 1. What is Zero-Shot Object Detection?

Zero-shot (or **open-vocabulary**) object detection takes an image **plus a class vocabulary supplied at inference time as free text**, and returns boxes labelled with that text. The vocabulary is an *input*, not a baked-in property of the weights.

Contrast with `02_Object_Detection`: a classical detector (DETR, RT-DETR, YOLO) has a fixed classification head with one logit per class. Train it on COCO and it can emit exactly 80 nouns, forever. Adding “solar panel” means new annotations, a new head, and a retrain. An open-vocabulary detector replaces that head with a **similarity between each box embedding and a text embedding**, so a new class costs one string.

**Input.** An RGB image + a text prompt. Two prompt styles that behave very differently:

| Prompt style | You pass | The model does | Examples |
|----|----|----|----|
| **Query list** | a list of candidate labels, e.g. `["a cat", "a solar panel"]` | embeds each label, scores every box against every label independently | OWL-ViT, OWLv2 |
| **Grounding / phrase** | one caption string, phrases separated by periods: `"a cat. a solar panel."` | fuses text and image in the encoder, grounds text *spans* to boxes | Grounding DINO, LLMDet, OmDet-Turbo |
| **Image-conditioned (one-shot)** | a crop/exemplar image instead of text | embeds the exemplar and uses it as the query vector | OWL-ViT / OWLv2 `image_guided_detection`, T-Rex2 |

The distinction is not cosmetic. A query-list model gives you a clean per-label score you can threshold per class, and it scales to hundreds of labels in one pass. A grounding model reads the whole caption at once, so labels **interact** (adding a phrase can change the score of another), its output labels are *text spans* rather than exactly the strings you passed, and it can accept a referring expression (“the cat on the left”) that a query-list model cannot represent.

**Output.** Boxes `(x0, y0, x1, y1)` + a score + the text label that matched. There is no fixed class id, so downstream code keys on strings.

**Neighbouring tasks:**

| Task | What it does | Notebook / tool |
|----|----|----|
| Closed-vocabulary detection | Fixed class head, higher AP on those classes | `02_Object_Detection` |
| Zero-shot image classification | Whole-image label, no localisation | `11_Zero_Shot_Image_Classification` |
| Promptable segmentation | Turns a box/point into a mask | `12_Mask_Generation` (SAM); box + SAM = “Grounded SAM” |
| Referring expression comprehension | One box for one free-form phrase | Grounding DINO on RefCOCO |
| Open-vocabulary segmentation | Masks, not boxes, from text | `03_Image_Segmentation` |

------------------------------------------------------------------------

## 2. Real-World Use Cases

The value of open-vocabulary detection is almost never “better AP on COCO” (a closed detector wins there). It is that **the class list is a config string**, so the cost of a new class collapses from a labelling project to a prompt edit.

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Auto-labelling for a closed detector | ML tooling (Roboflow Auto Label, Autodistill, CVAT) | Unlabelled images + class names -\> pseudo-boxes a human corrects | Recall at low threshold; a missed box costs more than a false one |
| Grounded SAM / cutout pipelines | Creative tooling, e-commerce photo ops | “the handbag.” -\> box -\> SAM mask -\> alpha cutout | Box precision (a sloppy box wrecks the mask); batch, not latency |
| Retail shelf and planogram audit | Retail analytics | Shelf photo + SKU descriptions -\> counts per product | Dense small objects; hundreds of prompts per image; scores drift per prompt |
| Robot / drone perception over an open world | Robotics, warehouse AMRs | Frame + task-derived nouns (“pick the red spanner”) -\> box for the grasp planner | Real-time on an edge GPU; the failure is a confident box on nothing |
| Industrial and infrastructure inspection | Energy, rail, telecom | Drone imagery + defect vocabulary (“rust”, “cracked insulator”) | Long-tail classes with zero labels; domain shift from web photos is brutal |
| Content moderation and brand safety | Trust and safety | Frame + evolving policy vocabulary -\> flagged regions | Vocabulary changes weekly; retraining is not an option |
| Medical / lab PPE compliance | Workplace safety (the CPPE-5 setting used below) | CCTV frame + \[“mask”, “gloves”, “goggles”\] -\> compliance events | Domain shift; a 5-class vocabulary the model never saw as boxes |
| Visual search inside video archives | Media, security | “find every frame with a red pickup truck” | Throughput over thousands of hours; recall matters more than box tightness |

**What the leaderboard number hides.** Three things, all of which bite in production.

*The threshold is not a constant.* A query-list model’s score is a similarity, not a calibrated probability. “a cat” and “a solar panel” do not live on the same scale, and the same prompt on a different image distribution shifts the whole score distribution. A threshold tuned on COCO photos will silently over-fire on drone imagery. Every serious deployment tunes the threshold **per class and per domain**, which is exactly the labelled-data cost that open-vocabulary detection was supposed to remove.

*Absent classes produce confident boxes.* Ask for “fire hydrant” in a living room and you often get one anyway, because the model scores the *most hydrant-like* region, and softmax-free similarity has no “none of the above” option. OWLv2 added an objectness head partly to fight this. Any pipeline that trusts “the model returned a box” as evidence the object exists will generate garbage.

*Dense and small objects are where it collapses.* Open-vocabulary detectors are trained largely on web photos with a handful of salient objects. A shelf with 200 SKUs, a crowd, or a PCB is out of distribution in a way that AP on LVIS does not capture. And the streaming/edge story is thin: Grounding DINO-base at ~230M params with a BERT text encoder is not a 30 FPS model on a Jetson without heavy work; the real-time entrants (YOLO-World, YOLOE, OmDet-Turbo) trade several AP points for it, and cache the text embeddings so the vocabulary is effectively frozen again at deploy time.

------------------------------------------------------------------------

## 3. How Modern Open-Vocabulary Detection Works

Every generation answers the same question: *where does the classifier weight for an unseen class come from?* The answer, since 2021, is always “a text encoder”, and the differences are in how tightly text is fused with vision and how the training data is scaled.

1.  **Distil CLIP into a detector (2021-2022).** **OVR-CNN** and **ViLD** keep a region proposal network, then replace the classification head with a CLIP text embedding and distil CLIP’s *image* embeddings into the region features. Cheap and simple; the detector inherits CLIP’s vocabulary but not its robustness, and proposals are still class-agnostic (a proposal generator trained on base classes under-proposes novel objects). **RegionCLIP** and **F-VLM** refine the same idea.

2.  **Detection as grounding (GLIP, 2022).** Reformulate detection as *phrase grounding*: concatenate the class names into a caption and align box features to token features. This unlocks a huge new training pool - grounding data (Flickr30k entities, VG) and image-caption data - because any caption is now detection supervision. GLIP introduced **ODinW** (Object Detection in the Wild, 35 datasets) as the transfer benchmark. Deep text-image fusion in the encoder is what makes phrase grounding work; it is also what makes these models slow.

3.  **CLIP backbone + detection heads, minimal changes (OWL-ViT, 2022).** Drop CLIP’s final token pooling, attach a box head and a class head to *every* ViT output token, fine-tune end-to-end with bipartite matching. No RPN, no fusion module: text and image are encoded **independently**, so text embeddings can be precomputed and hundreds of queries cost almost nothing. The same architecture gives **image-conditioned (one-shot) detection** for free: swap the text query embedding for the embedding of an exemplar image patch. Clean and fast; accuracy on rare classes was mediocre because detection training data is small.

4.  **Grounding DINO (2023).** DINO (DETR + denoising + query selection) + grounded pre-training, with **tight** fusion: a feature enhancer (cross-modal), language-guided query selection, and a cross-modality decoder. 52.5 AP on COCO with zero COCO training data, and the first model that reliably handles referring expressions as well as class lists. Became the de-facto open-vocabulary detector of 2023-2024, largely because of Grounded SAM.

5.  **Scale the data by self-training (OWLv2 / OWL-ST, 2023).** The bottleneck was never the architecture, it was **box** supervision. OWL-ST uses an existing detector to pseudo-label web image-text pairs, filters, and trains on **over 1B examples**. LVIS rare-class AP goes from 31.2 to **44.6** (L/14) - the single biggest jump in the field, and it came from data, not layers. This is the moment open-vocabulary detection got genuinely usable.

6.  **Better data recipes and LLM supervision (2024-2026).** **MM Grounding DINO** (2024) is an open reimplementation with a cleaned, much larger data mix. **LLMDet** (CVPR 2025) co-trains an MM-Grounding-DINO detector with a large language model that generates long, image-level *and* region-level captions: the LLM supervision teaches the detector richer semantics, and it is thrown away at inference, so you pay nothing at test time. LLMDet-tiny (172M) reaches **50.7 AP / 44.7 AP_r on LVIS minival** - a Swin-T model beating the 2023 large models. **DetCLIPv3**, **YOLOE** and **T-Rex2** (visual prompts) fill out the same generation.

7.  **Real-time open-vocabulary (2024-2026).** **YOLO-World** (Tencent, CVPR 2024) and **YOLOE** (ICCV 2025) bolt a re-parameterisable text head onto a YOLO, cache the text embeddings, and hit real-time on a single GPU. **OmDet-Turbo** (2024) does the same with an RT-DETR-style decoder and an efficient fusion head: 100+ FPS with TensorRT, 53.4 AP zero-shot COCO, 30.1 AP on ODinW. Note the licensing: YOLO-World and YOLOE ship through **Ultralytics under AGPL-3.0**, and the vendor package is not a general-purpose library - keep them out of the dependency list and reach for OmDet-Turbo (Apache-2.0, transformers-native) if you need speed.

8.  **Closed API frontier (2024-2026).** **Grounding DINO 1.5/1.6 Pro** (LVIS-minival 57.7 AP) and **DINO-X Pro** (COCO 56.0, LVIS-minival 59.8 AP, **63.3 AP_r**) are the accuracy leaders and are API-only. They matter as a ceiling, not as something you can run.

9.  **VLMs that just print boxes (2025-2026).** Qwen3-VL, PaliGemma 2, Molmo and Moondream emit boxes as *text* (JSON coordinates or `<loc>` tokens) from an arbitrary prompt. Enormously flexible - you get grounding, counting, OCR and reasoning from one model, and the “class list” can be a sentence. The trade-off is real: they are 10-100x slower per image, they degrade badly on **dense** scenes (autoregressive decoding of 50 boxes is slow and drops objects), their coordinates are quantised, and they have no calibrated score to threshold. The 2026 pattern in production is a hybrid: the VLM proposes the vocabulary or verifies candidates, a specialist detector does the localisation.

**Trade-off cheat sheet:**

| Approach | Prompt style | Fusion | Speed | Rare-class AP | Example |
|----|----|----|----|----|----|
| CLIP distillation | query list | none (late) | fast | low | ViLD, RegionCLIP |
| CLIP backbone + heads | query list + image query | none (late) | fast, many queries free | medium (high with OWL-ST) | OWL-ViT, OWLv2 |
| Grounded detector | caption / phrases | tight (encoder) | slow | high | GLIP, Grounding DINO, LLMDet |
| Real-time OVD | query list (cached) | light | real-time | medium | YOLO-World, YOLOE, OmDet-Turbo |
| VLM grounding | free-form text | full (LLM) | slowest | high semantics, poor density | Qwen3-VL, PaliGemma 2 |

------------------------------------------------------------------------

## 4. Evaluation Metrics

The geometry is the same as closed-set detection; what changes is **which classes you average over**.

**IoU** decides whether a predicted box matches a ground-truth box:

$$IoU(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

**Average Precision (AP).** Sort all predictions for a class by score; walk down the list greedily matching each prediction to an unmatched GT box with $IoU \ge \tau$ (a match is a TP, everything else an FP, unmatched GT are FN); this traces a precision-recall curve, and AP is the area under its monotone envelope:

$$AP = \sum_{k} (r_k - r_{k-1}) \cdot \max_{\tilde{r} \ge r_k} p(\tilde{r})$$

**mAP** is AP averaged over classes; COCO-style **AP** averages again over $\tau \in \{0.50, 0.55, \ldots, 0.95\}$. LVIS uses the same machinery with 1203 classes.

**The number that actually matters:** $AP_r$. LVIS splits its classes by training frequency into **rare** (1-10 training images), **common** and **frequent**, and reports $AP_r$, $AP_c$, $AP_f$. For open-vocabulary evaluation the convention is to treat rare classes as the novel ones. Overall AP is dominated by the frequent classes, which are exactly the ones every model has seen thousands of boxes of - so **a model can gain 3 AP overall while getting worse on everything novel**. Report $AP_r$ or you are measuring nothing. (Compare `11_Zero_Shot_Image_Classification`, where the same trap appears as ImageNet accuracy hiding long-tail collapse.)

**The leakage problem.** “Zero-shot” is a polite fiction. These models are pretrained on hundreds of millions of web image-text pairs; the probability that your “novel” class - or a paraphrase of it - never appeared in that corpus is essentially zero. LVIS rare classes are rare *in LVIS*, not on the internet. What $AP_r$ actually measures is “performance on classes with no **box** supervision”, which is the least-bad proxy available, not true novelty. The honest reading of any open-vocabulary number: *this is transfer, not induction*. It follows that the only trustworthy evaluation of your own use case is on **your own images with your own vocabulary** - which is why the benchmark in section 13 uses a domain dataset (medical PPE) rather than COCO.

**Speed.** Report ms/image at a fixed input resolution and vocabulary size, and say whether the text embeddings were cached: for a query-list model, the vocabulary cost is a one-off; for a grounding model, the caption goes through the fused encoder on **every** image.

The cell below implements IoU and AP from scratch on a toy example, and shows the base/novel split doing its job.

------------------------------------------------------------------------

``` python
import numpy as np


def iou_matrix(boxes_a, boxes_b):
    "Pairwise IoU between two sets of [x0, y0, x1, y1] boxes -> array (len(a), len(b))."
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def average_precision(preds, gts, iou_thr=0.5):
    """All-point-interpolated AP for ONE class.

    preds: list of (image_id, score, box)  - box is [x0, y0, x1, y1]
    gts:   dict image_id -> list of boxes
    Returns nan when the class has no ground truth (so np.nanmean skips it).
    """
    n_gt = sum(len(v) for v in gts.values())
    if n_gt == 0:
        return float("nan")
    preds = sorted(preds, key=lambda p: -p[1])
    matched = {k: np.zeros(len(v), dtype=bool) for k, v in gts.items()}
    tp, fp = np.zeros(len(preds)), np.zeros(len(preds))
    for i, (img, _score, box) in enumerate(preds):
        boxes = gts.get(img, [])
        if len(boxes) == 0:
            fp[i] = 1  # a box on an image with none of this class: pure false positive
            continue
        ious = iou_matrix([box], boxes)[0]
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr and not matched[img][j]:
            tp[i], matched[img][j] = 1, True
        else:
            fp[i] = 1  # bad localisation, or a duplicate of an already-matched GT
    tp_c, fp_c = np.cumsum(tp), np.cumsum(fp)
    recall = tp_c / n_gt
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-9)
    # Integrate under the monotone-decreasing precision envelope.
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


print("IoU of two nearly-identical boxes:", round(float(iou_matrix([[10, 10, 50, 90]], [[12, 12, 52, 92]])[0, 0]), 3))
print("IoU of two disjoint boxes:       ", round(float(iou_matrix([[10, 10, 50, 90]], [[60, 10, 90, 90]])[0, 0]), 3))

# Toy detector on 2 images. Two BASE classes (seen with boxes in training) and two
# NOVEL classes (named only at inference). The detector is good at the base ones.
BASE, NOVEL = ["person", "car"], ["manhole cover", "cargo pallet"]

gt = {
    "person":        {0: [[10, 10, 50, 90], [60, 20, 100, 95]], 1: [[30, 40, 70, 100]]},
    "car":           {0: [[120, 30, 200, 90]], 1: [[100, 50, 180, 110]]},
    "manhole cover": {0: [[20, 150, 60, 180]], 1: [[80, 160, 120, 190]]},
    "cargo pallet":  {0: [[140, 140, 190, 190]], 1: [[10, 120, 60, 175]]},
}
pred = {
    "person":        [(0, 0.95, [12, 12, 52, 92]), (0, 0.90, [58, 18, 102, 96]),
                      (1, 0.85, [31, 42, 72, 102]), (1, 0.30, [0, 0, 20, 20])],
    "car":           [(0, 0.92, [118, 28, 198, 92]), (1, 0.88, [102, 52, 182, 112]),
                      (0, 0.40, [0, 100, 40, 140])],
    # Novel classes: one hit, several confident boxes on nothing, one object missed entirely.
    "manhole cover": [(0, 0.55, [15, 148, 70, 190]), (1, 0.50, [0, 0, 30, 30]),
                      (0, 0.45, [100, 20, 140, 60])],
    "cargo pallet":  [(1, 0.60, [12, 122, 58, 178]), (0, 0.58, [30, 30, 80, 80]),
                      (0, 0.35, [141, 141, 189, 189])],
}

aps = {c: average_precision(pred[c], gt[c]) for c in BASE + NOVEL}
for c, ap in aps.items():
    tag = "base " if c in BASE else "NOVEL"
    print(f"{tag} AP50  {c:15s} {ap:.3f}")

ap_base = float(np.nanmean([aps[c] for c in BASE]))
ap_novel = float(np.nanmean([aps[c] for c in NOVEL]))
ap_all = float(np.nanmean(list(aps.values())))
print(f"\nAP_base  {ap_base:.3f}   AP_novel {ap_novel:.3f}   mAP (all classes) {ap_all:.3f}")
print("The headline mAP sits between the two and hides the novel-class collapse -")
print("this is precisely why LVIS reports AP_r separately.")
```

    IoU of two nearly-identical boxes: 0.863
    IoU of two disjoint boxes:        0.0
    base  AP50  person          1.000
    base  AP50  car             1.000
    NOVEL AP50  manhole cover   0.500
    NOVEL AP50  cargo pallet    0.833

    AP_base  1.000   AP_novel 0.667   mAP (all classes) 0.833
    The headline mAP sits between the two and hides the novel-class collapse -
    this is precisely why LVIS reports AP_r separately.

## 5. Datasets

| Dataset | Contents | Size | Scope | License | Typical use |
|----|----|----|----|----|----|
| [LVIS v1.0](https://www.lvisdataset.org/) | COCO images, long-tail vocabulary, federated annotation | 1203 classes, 100k train / 20k val (`minival` = 5k) | The open-vocab **eval standard** | CC BY 4.0 | Zero-shot eval; report AP and **AP_r** |
| [ODinW](https://eval.ai/web/challenges/challenge-page/1839/overview) | 35 real-world detection datasets (aerial, medical, retail, game…) | 35 datasets, ODinW13 subset common | Transfer / “in the wild” | mixed, per dataset | Zero-shot + few-shot transfer AP |
| [COCO](https://cocodataset.org/) (base/novel split) | 80 classes, split 48 base / 17 novel | 118k train / 5k val | Legacy OVD protocol | CC BY 4.0 | AP_novel; increasingly seen as too easy |
| [Objects365 v1/v2](https://www.objects365.org/) | 365 classes, dense boxes | 1.7M images, 29M boxes | **Detection pretraining** | research use | The box supervision under nearly every OVD model |
| [V3Det](https://v3det.openxlab.org.cn/) | Hierarchical vocabulary | 13k+ categories, 245k images | **Vast-vocabulary pretraining** | CC BY-NC 4.0 | Long-tail pretraining (used by LLMDet) |
| [GoldG / GRIT](https://huggingface.co/datasets/zzliang/GRIT) | Grounding data: caption phrases linked to boxes | GoldG ~0.8M, GRIT ~20M | **Grounding pretraining** | mixed | Phrase-grounding supervision (GLIP, G-DINO, LLMDet) |
| [RefCOCO / + / g](https://huggingface.co/datasets/lmms-lab/RefCOCO) | Referring expressions -\> one box | ~140k expressions | Referring expression comprehension | Apache-2.0 (HF mirror) | Evaluating spatial/attribute grounding |
| [OVDEval](https://github.com/om-ai-lab/OVDEval) | 9 sub-tasks probing negation, position, relationship | ~20k images | Fine-grained OVD diagnosis | Apache-2.0 | Catching models that ignore the prompt |
| [CPPE-5](https://huggingface.co/datasets/rishitdagli/cppe-5) | Medical PPE in the wild: Coverall, Face_Shield, Gloves, Goggles, Mask | 1000 train / **29 test** | 5 classes, none of them COCO | unknown (research use) | **This notebook’s eval set** |

**What we evaluate on, and why.** LVIS is the right *published* benchmark, but a full LVIS-minival pass is 5000 images x 1203 prompts - hours on a 3060, and the numbers would just reproduce the model cards. Instead section 13 benchmarks on the **CPPE-5 test split (29 images, 4 MB)**: a small ODinW-style domain dataset whose five classes (coveralls, face shields, gloves, goggles, masks) are *not* COCO categories, come from a domain far from web photos, and expose exactly the two failure modes we care about - prompt wording and threshold calibration. Nothing here is gated. Objects365 and V3Det require registration and are training-scale only; do not try to download them here.

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

There is no single official leaderboard; the community reference is **zero-shot AP / AP_r on LVIS minival** (see [Papers with Code: LVIS v1.0 minival](https://paperswithcode.com/sota/zero-shot-object-detection-on-lvis-v1-0-minival)) plus **ODinW13/35** for transfer. Numbers below are from the model cards and papers cited in section 16.

| Model | Params | License | Prompt style | Architecture | LVIS minival AP / AP_r | Best for |
|----|----|----|----|----|----|----|
| [google/owlvit-base-patch32](https://huggingface.co/google/owlvit-base-patch32) | 153M | Apache-2.0 | query list + **image query** | CLIP ViT-B/32 + box/class heads | low (pre-self-training) | one-shot image-conditioned detection; many queries cheaply |
| [google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble) | 155M | Apache-2.0 | query list + image query | OWL-ViT arch + **OWL-ST self-training** + objectness head | large gain over OWL-ViT | the best “just give me per-class scores” model that fits anywhere |
| [google/owlv2-large-patch14-ensemble](https://huggingface.co/google/owlv2-large-patch14-ensemble) | 438M | Apache-2.0 | query list + image query | L/14 | AP_r **44.6** (OWL-ST+FT, paper) | max OWL accuracy; still fits 12 GB |
| [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | 172M | Apache-2.0 | **caption / phrases** | Swin-T + BERT + DINO decoder, tight fusion | 27.4 / 18.1 | referring expressions; Grounded SAM; the ecosystem default |
| [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base) | 233M | Apache-2.0 | caption / phrases | Swin-B | higher | same, more accurate, ~2x slower |
| [iSEE-Laboratory/llmdet_tiny](https://huggingface.co/iSEE-Laboratory/llmdet_tiny) | 173M | Apache-2.0 | caption / phrases | MM-Grounding-DINO co-trained with an LLM | **50.7 / 44.7** | best accuracy-per-parameter open model; drop-in for G-DINO |
| [iSEE-Laboratory/llmdet_large](https://huggingface.co/iSEE-Laboratory/llmdet_large) | 344M | Apache-2.0 | caption / phrases | Swin-L | **56.6 / 51.1** | best **runnable** open-vocab detector, mid-2026 |
| [omlab/omdet-turbo-swin-tiny-hf](https://huggingface.co/omlab/omdet-turbo-swin-tiny-hf) | 115M | Apache-2.0 | query list (cached) | RT-DETR-style + efficient fusion head | COCO 53.4 AP zs; ODinW 30.1 | **real-time** (100+ FPS with TensorRT), Apache-2.0 |
| YOLO-World / YOLOE | 13-110M | **AGPL-3.0** (Ultralytics) | query list (cached) | YOLO + re-parameterised text head | lower | real-time on edge - but AGPL and a vendor-only runtime |
| Grounding DINO 1.5 / 1.6 Pro | closed | **API only** | caption | Swin-L + ViT-L | 57.7 (1.6 Pro) | commercial ceiling |
| DINO-X Pro | closed | **API only** | caption + visual + no prompt | unified open-world head | **59.8 / 63.3** | absolute SOTA; also outputs masks, poses, captions |
| Qwen3-VL / PaliGemma 2 / Moondream | 2B-235B | Apache-2.0 / Gemma | free-form text | VLM emitting boxes as text | n/a (not comparable) | reasoning-heavy grounding, counting, OCR-in-the-loop |

**Who wins what.** *Accuracy*: DINO-X Pro, and it is not close - but it is an API. Among things you can actually run, **`llmdet_large` (344M) is the mid-2026 pick**, beating 2023-era large models at a third of the size. *Speed*: OmDet-Turbo (Apache-2.0) or, if you can live with AGPL and a vendor runtime, YOLOE. *Flexibility*: OWLv2, because it is the only family here with a first-class **image query**, and because independent text/image encoding means 500 prompts cost barely more than 5. *Reasoning*: a VLM, at 10-100x the latency.

Tying back to section 2: the auto-labelling and inspection use cases want `llmdet_large` (accuracy, batch, no latency budget); the robot and moderation use cases want OmDet-Turbo or a distilled closed detector; the retail shelf case wants OWLv2 for the cheap 200-prompt vocabulary; the “find me more things like *this*” case wants OWL-ViT’s image query or T-Rex2.

**Fits on the 12 GB card?** All of them. Every runnable model here is under 450M params - at fp32 that is under 2 GB. Open-vocabulary detectors are *small*; what costs you is the closed frontier being closed, not VRAM. The only rows that do not fit are the API models (you cannot get the weights) and the large VLMs (a 72B grounding VLM needs a different machine; a 3-4B one, e.g. Qwen3-VL-4B in 4-bit, does fit - see `05_Image_to_Text`).

------------------------------------------------------------------------

## 7. Setup

Everything below runs through Hugging Face `transformers` on one GPU (RTX 3060, 12 GB) or on CPU (slower, but all four models are small enough to be tolerable). Package roles:

- `transformers` (\>=5.13) + `torch` - all four detectors, plus the `zero-shot-object-detection` pipeline
- `accelerate` - `device_map` placement
- `datasets` - the CPPE-5 eval split (29 images with boxes)
- `pillow` - image loading and **all box drawing** (`ImageDraw`); no matplotlib anywhere
- `pyecharts` - the threshold sweep and benchmark charts
- `pandas` + `numpy` - the results table and the AP implementation from section 4

**A note on dtype.** The setup cell defines `dtype` (fp16 on GPU) per the house contract and we use it for the OWL family, whose CLIP backbone is fp16-clean. Grounding DINO and LLMDet keep **fp32**: their multi-scale deformable attention is numerically fragile in half precision, and at 0.17B params fp32 costs well under 1 GB, so there is nothing to win. The section-13 benchmark runs everything in fp32 so the timings are apples-to-apples.

`ultralytics` (YOLO-World, YOLOE) and `mmdetection` are deliberately **not** dependencies - they are vendor runtimes, and AGPL besides. Everything you see runs on the general-purpose stack.

------------------------------------------------------------------------

``` python
# Everything runs through Hugging Face transformers - no vendor detection packages.
# %pip install -q torch transformers accelerate datasets pillow pyecharts pandas numpy

# Optional, for the webcam demo in section 14 only:
# %pip install -q opencv-python
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
print("device:", device)

def vram(tag=""):
    "Report current GPU memory (allocated / reserved). No-op on CPU."
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"VRAM {tag:20s} {alloc:5.2f} GB allocated / {reserved:5.2f} GB reserved")

def free_memory():
    "Collect garbage, empty the CUDA cache, and return freed CPU RAM to the OS."
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
    device: cuda:0

``` python
import urllib.request

from IPython.display import display
from PIL import Image, ImageDraw

# The canonical COCO val2017 test image: two cats on a pink couch, two TV remotes.
# It is the example image in every transformers detection doc, which makes the
# numbers below directly comparable with the model cards.
SAMPLE = DATA_DIR / "coco_cats.jpg"
# A second COCO image (a single cat) used as the *image query* exemplar in section 8.
QUERY = DATA_DIR / "coco_cat_query.jpg"

for url, path in [
    ("http://images.cocodataset.org/val2017/000000039769.jpg", SAMPLE),
    ("http://images.cocodataset.org/val2017/000000001675.jpg", QUERY),
]:
    if not path.exists():
        urllib.request.urlretrieve(url, path)

image = Image.open(SAMPLE).convert("RGB")
query_image = Image.open(QUERY).convert("RGB")
print("image:", image.size, " query image:", query_image.size)

PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]


def draw_boxes(img, detections, width=3):
    "Draw [{'box': (x0,y0,x1,y1), 'label': str, 'score': float}, ...] onto a copy of img (PIL, no matplotlib)."
    out = img.copy()
    d = ImageDraw.Draw(out)
    names = sorted({det["label"] for det in detections})
    for det in detections:
        x0, y0, x1, y1 = (float(v) for v in det["box"])
        color = PALETTE[names.index(det["label"]) % len(PALETTE)]
        d.rectangle([x0, y0, x1, y1], outline=color, width=width)
        tag = f"{det['label']} {det['score']:.2f}"
        top = max(0.0, y0 - 16)
        d.rectangle([x0, top, x0 + d.textlength(tag) + 5, top + 15], fill=color)
        d.text((x0 + 2, top + 1), tag, fill="white")
    return out


def from_pipeline(results):
    "Normalise `zero-shot-object-detection` pipeline output to the draw_boxes format."
    return [
        {
            "box": (r["box"]["xmin"], r["box"]["ymin"], r["box"]["xmax"], r["box"]["ymax"]),
            "label": r["label"],
            "score": float(r["score"]),
        }
        for r in results
    ]


def to_device(inputs, cast=None):
    "Move a processor's BatchEncoding to `device`; optionally cast float tensors (pixel_values) to `cast`."
    inputs = inputs.to(device)
    if cast is not None:
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(cast)
    return inputs


image
```

    image: (640, 480)  query image: (640, 480)

![](13_Zero_Shot_Object_Detection_files/figure-html/cell-5-output-2.png)

## 8. OWL-ViT: query lists and image queries

**`google/owlvit-base-patch32`** (153M, Apache-2.0). CLIP with its token pooling removed and a box head + class head bolted onto every ViT output token. Text and image are encoded **independently**, which has three consequences worth internalising:

- **Queries are cheap.** Embed 500 class names once, reuse the embeddings forever. Score = dot product between a box’s class embedding and a text embedding. This is why OWL is the right family for a 200-SKU shelf.
- **Prompt wording matters a lot.** The text tower is CLIP’s, trained on `"a photo of a ..."` captions, so that template genuinely helps. `"cat"`, `"a cat"` and `"a photo of a cat"` give different scores.
- **The query does not have to be text.** Swap the text embedding for the embedding of an *exemplar image patch* and you get **one-shot / image-conditioned detection**: “find more things that look like this”. This is the most underused capability in the whole task, and it is the right tool whenever the thing you are looking for is easier to *show* than to name (a specific screw, a specific logo, a specific defect).

Pick OWL-ViT when you want image queries or a huge vocabulary; pick OWLv2 (next section) whenever you want accuracy. Scores from this generation are low in absolute terms - a 0.1 threshold is normal, not a bug.

------------------------------------------------------------------------

``` python
from transformers import pipeline

# The pipeline wraps processor + model + post-processing for all four ZSOD families.
detector = pipeline(
    "zero-shot-object-detection",
    model="google/owlvit-base-patch32",
    device=device,
    model_kwargs={"cache_dir": HF_CACHE},
)

# CLIP-style prompts: the "a photo of a ..." template is what the text tower saw in training.
candidate_labels = ["a photo of a cat", "a photo of a remote control", "a photo of a couch"]

t0 = time.perf_counter()
results = detector(image, candidate_labels=candidate_labels, threshold=0.1)
print(f"{time.perf_counter() - t0:.2f}s   {len(results)} detections above 0.10")
for r in results:
    print(f"  {r['score']:.3f}  {r['label']}")

dets = from_pipeline(results)
display(draw_boxes(image, dets))

del detector, results
free_memory()
vram("after owlvit pipeline")
```

    0.41s   5 detections above 0.10
      0.717  a photo of a cat
      0.707  a photo of a cat
      0.231  a photo of a remote control
      0.224  a photo of a remote control
      0.152  a photo of a couch

![](13_Zero_Shot_Object_Detection_files/figure-html/cell-6-output-3.png)

    VRAM after owlvit pipeline  0.01 GB allocated /  0.02 GB reserved

``` python
# One-shot / image-conditioned detection: the QUERY is an image, not a string.
# `image_guided_detection` embeds the query image, picks its most object-like patch
# embedding, and uses that as the class embedding - so no text is involved at all.
from transformers import OwlViTForObjectDetection, OwlViTProcessor

owlvit_id = "google/owlvit-base-patch32"
processor = OwlViTProcessor.from_pretrained(owlvit_id, cache_dir=HF_CACHE)
model = OwlViTForObjectDetection.from_pretrained(
    owlvit_id, dtype=dtype, cache_dir=HF_CACHE
).to(device)

display(query_image.resize((query_image.width // 3, query_image.height // 3)))  # the exemplar

inputs = to_device(processor(images=image, query_images=query_image, return_tensors="pt"), cast=dtype)
t0 = time.perf_counter()
with torch.inference_mode():
    outputs = model.image_guided_detection(**inputs)
print(f"{time.perf_counter() - t0:.2f}s")

# nms_threshold matters here: image queries fire on many overlapping patches.
# OwlViT's image-guided post-processor still requires target_sizes as a TENSOR
# (it reads target_sizes.shape); the text-query post-processors accept a list.
res = processor.post_process_image_guided_detection(
    outputs=outputs, threshold=0.6, nms_threshold=0.3,
    target_sizes=torch.tensor([[image.height, image.width]]),
)[0]
dets = [
    {"box": b.tolist(), "label": "like the query", "score": float(s)}
    for b, s in zip(res["boxes"].cpu(), res["scores"].cpu())
]
print(f"{len(dets)} regions matched the exemplar (no text was used)")
display(draw_boxes(image, dets))

del model, processor, inputs, outputs, res
free_memory()
vram("after owlvit one-shot")
```

![](13_Zero_Shot_Object_Detection_files/figure-html/cell-7-output-2.png)

    0.17s
    0 regions matched the exemplar (no text was used)

![](13_Zero_Shot_Object_Detection_files/figure-html/cell-7-output-4.png)

    VRAM after owlvit one-shot  0.01 GB allocated /  0.02 GB reserved

## 9. OWLv2: the self-training jump

**`google/owlv2-base-patch16-ensemble`** (155M, Apache-2.0). *Architecturally almost identical to OWL-ViT* - the same CLIP backbone and per-token heads, plus an **objectness head** that scores “is there any object here” independently of the text query. What changed is the data: the OWL-ST recipe pseudo-labels **over a billion** web image-text pairs with an existing detector, filters them, and trains on that. LVIS rare-class AP went from 31.2 to 44.6 (L/14). Same model, 40% better on exactly the classes you care about, purely from data.

This is the checkpoint to reach for by default when you want a query-list detector: same cheap-many-queries property as OWL-ViT, far better scores. The `-ensemble` suffix means it was trained with an ensemble of prompt templates, so bare class names (“a cat”) work well and you no longer need the `"a photo of a ..."` ceremony.

Note the deliberately absent class in the query list below. Watch what score it gets.

------------------------------------------------------------------------

``` python
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

owlv2_id = "google/owlv2-base-patch16-ensemble"
processor = AutoProcessor.from_pretrained(owlv2_id, cache_dir=HF_CACHE)
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    owlv2_id, dtype=dtype, cache_dir=HF_CACHE
).to(device)

# Nested list = one query list per image in the batch. "a fire hydrant" is NOT in the image.
text_labels = [["a cat", "a remote control", "a couch", "a fire hydrant"]]

inputs = to_device(processor(text=text_labels, images=image, return_tensors="pt"), cast=dtype)
t0 = time.perf_counter()
with torch.inference_mode():
    outputs = model(**inputs)
print(f"{time.perf_counter() - t0:.2f}s")

res = processor.post_process_grounded_object_detection(
    outputs=outputs,
    target_sizes=[(image.height, image.width)],
    threshold=0.15,
    text_labels=text_labels,
)[0]
dets = [
    {"box": b.tolist(), "label": t, "score": float(s)}
    for b, s, t in zip(res["boxes"].cpu(), res["scores"].cpu(), res["text_labels"])
]
for d in sorted(dets, key=lambda x: -x["score"]):
    print(f"  {d['score']:.3f}  {d['label']}")
print("(if 'a fire hydrant' appears at all, that is the absent-class failure mode from section 2)")
display(draw_boxes(image, dets))

del model, processor, inputs, outputs, res
free_memory()
vram("after owlv2")
```

    0.10s
      0.792  a remote control
      0.739  a cat
      0.705  a remote control
      0.614  a cat
      0.599  a couch
      0.174  a couch
      0.159  a couch
    (if 'a fire hydrant' appears at all, that is the absent-class failure mode from section 2)

![](13_Zero_Shot_Object_Detection_files/figure-html/cell-8-output-11.png)

    VRAM after owlv2           0.01 GB allocated /  0.02 GB reserved

## 10. Grounding DINO: detection as phrase grounding

**`IDEA-Research/grounding-dino-tiny`** (172M, Apache-2.0; `-base` is 233M). A different animal from OWL. Swin backbone + BERT text encoder + a DETR-style decoder, with **tight cross-modal fusion**: a feature enhancer, language-guided query selection, and a cross-modality decoder. The prompt is a **caption**, and the model grounds *text spans* to boxes.

Consequences of the fused design:

- **Labels interact.** The caption is encoded once, with all phrases attending to each other and to the image. Adding a phrase can move another phrase’s score. There is no such thing as “score this class independently”.
- **The returned label is a text span**, not necessarily the exact string you passed. Post-processing recovers the matched span, so downstream code must map spans back to your classes (see the benchmark in section 13).
- **Two thresholds.** `threshold` filters boxes by score; `text_threshold` decides which caption tokens are considered part of a box’s label. Both need tuning.
- **It can take a referring expression.** `"the cat on the left"` is a legal prompt in a way that it simply is not for a query-list model. It is also where these models are weakest: spatial and relational language (“left of”, “the one being held”) is far less reliable than plain nouns, which is what OVDEval and RefCOCO exist to expose. The cell below tries one so you can see for yourself.

Convention: the processor joins your label list into `"a cat. a remote control."` - lowercase phrases, period-separated. If you build the caption by hand, follow that format exactly or scores collapse.

------------------------------------------------------------------------

``` python
gdino_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(gdino_id, cache_dir=HF_CACHE)
# fp32 on purpose: multi-scale deformable attention is fragile in fp16, and 172M params
# costs ~0.7 GB in fp32 anyway.
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    gdino_id, dtype=torch.float32, cache_dir=HF_CACHE
).to(device)


def gdino_detect(text_labels, threshold=0.35, text_threshold=0.25):
    "Run Grounding DINO on `image` with a nested list of phrases; return draw_boxes dicts."
    inputs = to_device(processor(images=image, text=text_labels, return_tensors="pt"))
    with torch.inference_mode():
        out = model(**inputs)
    res = processor.post_process_grounded_object_detection(
        out,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[(image.height, image.width)],
    )[0]
    return [
        {"box": b.tolist(), "label": t, "score": float(s)}
        for b, s, t in zip(res["boxes"].cpu(), res["scores"].cpu(), res["text_labels"])
    ]


t0 = time.perf_counter()
dets = gdino_detect([["a cat", "a remote control"]])  # -> caption "a cat. a remote control."
print(f"class-list prompt: {time.perf_counter() - t0:.2f}s")
for d in sorted(dets, key=lambda x: -x["score"]):
    print(f"  {d['score']:.3f}  {d['label']}")
display(draw_boxes(image, dets))

# A referring expression - impossible to express as a query list. Note how much
# weaker spatial language is than a plain noun: it will very likely also fire on
# the other cat. That gap is the honest state of the art, not a bug in this cell.
ref = gdino_detect([["the cat on the left"]], threshold=0.25)
print(f"\nreferring expression -> {len(ref)} boxes")
display(draw_boxes(image, ref))

del model, processor, dets, ref
free_memory()
vram("after grounding dino")
```

    class-list prompt: 0.40s
      0.487  a cat
      0.466  a remote control
      0.450  a cat

![](13_Zero_Shot_Object_Detection_files/figure-html/cell-9-output-11.png)

    referring expression -> 2 boxes

![](13_Zero_Shot_Object_Detection_files/figure-html/cell-9-output-13.png)

    VRAM after grounding dino  0.01 GB allocated /  0.02 GB reserved

## 11. LLMDet: LLM supervision, no LLM at inference

**`iSEE-Laboratory/llmdet_tiny`** (173M, Apache-2.0). CVPR 2025 highlight, and the best accuracy-per-parameter open detector in mid-2026. It takes an MM-Grounding-DINO detector and **co-trains it with a large language model** that generates long image-level captions *and* region-level phrases (the GroundingCap-1M dataset). The LLM forces the detector’s features to carry richer semantics than a bag of class names ever could.

The crucial engineering detail: **the LLM is discarded after training.** The checkpoint is a plain MM-Grounding-DINO, the inference API is byte-for-byte identical to Grounding DINO, and it costs the same to run. You get the accuracy for free.

| checkpoint     | params | LVIS minival AP | LVIS minival AP_r |
|----------------|--------|-----------------|-------------------|
| `llmdet_tiny`  | 173M   | 50.7            | 44.7              |
| `llmdet_base`  | 233M   | 54.3            | 48.3              |
| `llmdet_large` | 344M   | 56.6            | 51.1              |

Compare Grounding DINO Swin-T: **27.4 AP / 18.1 AP_r**. Same size, same architecture family, +23 AP and +26 AP_r - all of it from the training recipe. This is the single strongest argument in the whole notebook that open-vocabulary detection is a **data** problem. In `transformers` since 4.55; `llmdet_large` fits the 12 GB card comfortably, and is what you should ship.

------------------------------------------------------------------------

``` python
llmdet_id = "iSEE-Laboratory/llmdet_tiny"  # swap for llmdet_large (344M) if you want the accuracy
processor = AutoProcessor.from_pretrained(llmdet_id, cache_dir=HF_CACHE)
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    llmdet_id, dtype=torch.float32, cache_dir=HF_CACHE
).to(device)

# Identical API to Grounding DINO - LLMDet *is* an MM-Grounding-DINO at inference time.
text_labels = [["a cat", "a remote control", "a couch"]]
inputs = to_device(processor(images=image, text=text_labels, return_tensors="pt"))

t0 = time.perf_counter()
with torch.inference_mode():
    outputs = model(**inputs)
print(f"{time.perf_counter() - t0:.2f}s")

res = processor.post_process_grounded_object_detection(
    outputs, threshold=0.4, target_sizes=[(image.height, image.width)]
)[0]
dets = [
    {"box": b.tolist(), "label": t, "score": float(s)}
    for b, s, t in zip(res["boxes"].cpu(), res["scores"].cpu(), res["text_labels"])
]
for d in sorted(dets, key=lambda x: -x["score"]):
    print(f"  {d['score']:.3f}  {d['label']}")
display(draw_boxes(image, dets))

del model, processor, inputs, outputs, res, dets
free_memory()
vram("after llmdet")
```

    0.24s
      0.635  a couch
      0.515  a cat
      0.492  a remote control
      0.482  a cat

![](13_Zero_Shot_Object_Detection_files/figure-html/cell-10-output-9.png)

    VRAM after llmdet          0.01 GB allocated /  0.02 GB reserved

## 12. The Prompt Is the Model: query sets and thresholds

This is the section that will save you the most time in practice. The same image, the same weights, four different query sets - plus a threshold sweep.

Four things to watch for:

1.  **Synonyms are not free.** `"remote control"`, `"tv remote"` and `"clicker"` are the same object to you and three different points in CLIP’s text space to the model. Scores move by a lot; sometimes the “obvious” wording is the worst one.
2.  **Absent classes still get boxes.** Ask for a class that is not in the image and you will often get a confident-looking box on the most plausible region, because the score is a similarity with no “none of the above” option.
3.  **Novel / fine-grained vocabulary degrades gracefully-ish.** `"a cat's ear"` or `"a pink blanket"` sometimes work beautifully and sometimes produce nonsense. There is no way to know without looking.
4.  **The threshold is per-prompt, not per-model.** The sweep below plots how many boxes survive as the threshold rises, for each query set. The curves have completely different shapes - which means **any single global threshold is wrong for most of your classes**. Tune per class, on your own data. (This is why the LVIS AP protocol integrates over all thresholds rather than picking one.)

------------------------------------------------------------------------

``` python
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org serves an expired TLS certificate from most nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"

from pyecharts import options as opts
from pyecharts.charts import Line

processor = AutoProcessor.from_pretrained(owlv2_id, cache_dir=HF_CACHE)
model = AutoModelForZeroShotObjectDetection.from_pretrained(
    owlv2_id, dtype=dtype, cache_dir=HF_CACHE
).to(device)


def owlv2_scores(labels, threshold=0.05):
    "Detect `labels` in `image` with OWLv2; return draw_boxes dicts above `threshold`."
    nested = [list(labels)]
    inputs = to_device(processor(text=nested, images=image, return_tensors="pt"), cast=dtype)
    with torch.inference_mode():
        out = model(**inputs)
    res = processor.post_process_grounded_object_detection(
        outputs=out,
        target_sizes=[(image.height, image.width)],
        threshold=threshold,
        text_labels=nested,
    )[0]
    return [
        {"box": b.tolist(), "label": t, "score": float(s)}
        for b, s, t in zip(res["boxes"].cpu(), res["scores"].cpu(), res["text_labels"])
    ]


QUERY_SETS = {
    "coco-ish": ["a cat", "a remote control", "a couch"],
    "synonyms": ["remote control", "tv remote", "clicker", "a photo of a remote control"],
    "fine-grained/novel": ["a cat's ear", "a cat's whiskers", "a pink blanket", "a white cat paw"],
    "absent (control)": ["a fire hydrant", "a giraffe", "a unicorn", "a laptop"],
}

for name, labels in QUERY_SETS.items():
    dets = owlv2_scores(labels, threshold=0.05)
    best = {}
    for d in dets:
        best[d["label"]] = max(best.get(d["label"], 0.0), d["score"])
    print(f"\n{name}:")
    for lab in labels:
        top = best.get(lab, 0.0)
        flag = "  <-- nothing is there" if name.startswith("absent") and top > 0.15 else ""
        print(f"  top score {top:.3f}   {lab}{flag}")

# Show the "novel vocabulary" set drawn, at a threshold that keeps only the confident ones.
display(draw_boxes(image, owlv2_scores(QUERY_SETS["fine-grained/novel"], threshold=0.12)))

# Threshold sweep: how many boxes survive, per query set. Different shapes = no global threshold.
THRESHOLDS = [round(0.05 + 0.025 * i, 3) for i in range(19)]  # 0.05 .. 0.50
counts = {}
for name, labels in QUERY_SETS.items():
    dets = owlv2_scores(labels, threshold=THRESHOLDS[0])  # keep everything the sweep can count
    scores = np.array([d["score"] for d in dets]) if dets else np.zeros(0)
    counts[name] = [int((scores >= t).sum()) for t in THRESHOLDS]

del model, processor
free_memory()
vram("after prompt sweep")

line = Line()
line.add_xaxis([str(t) for t in THRESHOLDS])
for name, ys in counts.items():
    line.add_yaxis(name, ys, is_smooth=True, symbol_size=6)
line.set_global_opts(
    title_opts=opts.TitleOpts(
        title="OWLv2: boxes kept vs score threshold",
        subtitle="Same image, same weights. Every query set needs its own threshold.",
    ),
    xaxis_opts=opts.AxisOpts(name="score threshold"),
    yaxis_opts=opts.AxisOpts(name="boxes kept"),
    tooltip_opts=opts.TooltipOpts(trigger="axis"),
    legend_opts=opts.LegendOpts(pos_top="10%"),
)
line.render_notebook()
```

    coco-ish:
      top score 0.738   a cat
      top score 0.793   a remote control
      top score 0.599   a couch

    synonyms:
      top score 0.000   remote control
      top score 0.000   tv remote
      top score 0.145   clicker
      top score 0.785   a photo of a remote control

    fine-grained/novel:
      top score 0.348   a cat's ear
      top score 0.457   a cat's whiskers
      top score 0.446   a pink blanket
      top score 0.340   a white cat paw

    absent (control):
      top score 0.000   a fire hydrant
      top score 0.000   a giraffe
      top score 0.000   a unicorn
      top score 0.000   a laptop

![](13_Zero_Shot_Object_Detection_files/figure-html/cell-11-output-3.png)

    VRAM after prompt sweep    0.01 GB allocated /  0.02 GB reserved

## 13. Head-to-head Benchmark: CPPE-5 (a real novel vocabulary)

Same images, same prompts, same metric, one model live at a time.

**The dataset.** [CPPE-5](https://huggingface.co/datasets/rishitdagli/cppe-5) test split: 29 photographs of people in medical PPE, annotated with five classes - `Coverall`, `Face_Shield`, `Gloves`, `Goggles`, `Mask`. None of them is a COCO category, the imagery is not web-photo-typical, and the objects are often small and occluded. This is an ODinW-style transfer test, and it is *hard*: expect AP numbers far below the LVIS figures in the model cards. That gap is the point of the exercise.

**The metric.** AP@IoU=0.5 per class, using the exact `average_precision` implementation from section 4, then mAP50 = mean over the five classes. Plus seconds per image.

**Caveats, stated up front.** 29 images (we use 20), one prompt wording per class, one threshold (0.05, deliberately low so AP integrates over the whole PR curve rather than being cut off), fp32 for all three models on one RTX 3060. This is a **smoke test, not a leaderboard**. A different prompt for `Coverall` (“a protective suit”? “a hazmat suit”?) would move the numbers by several points - which is exactly the finding from section 12, now with a metric attached.

------------------------------------------------------------------------

``` python
from datasets import load_dataset

# 29 test images, ~4 MB. Parquet-native, not gated.
cppe = load_dataset("rishitdagli/cppe-5", split="test", cache_dir=HF_CACHE)

CLASSES = ["Coverall", "Face_Shield", "Gloves", "Goggles", "Mask"]  # class_label order, 0..4
PROMPTS = {          # the prompt wording IS a hyperparameter - see section 12
    "Coverall": "a protective coverall suit",
    "Face_Shield": "a face shield",
    "Gloves": "protective gloves",
    "Goggles": "safety goggles",
    "Mask": "a face mask",
}
CANDIDATES = list(PROMPTS.values())


def to_class(label):
    "Map a returned text label back to a CPPE-5 class. Grounding models return matched spans, not our exact strings."
    lab = label.lower().strip(" .")
    for cls, prompt in PROMPTS.items():          # exact match first
        if lab == prompt.lower():
            return cls
    for cls, prompt in PROMPTS.items():          # then substring, either direction
        if lab and (lab in prompt.lower() or prompt.lower() in lab):
            return cls
    return None


def evaluate(detector, n_images=20, threshold=0.05):
    "Run `detector` over the first n_images of CPPE-5 test; return (per-class AP50, sec/image)."
    gts = {c: {} for c in CLASSES}
    preds = {c: [] for c in CLASSES}
    elapsed = 0.0
    for i in range(n_images):
        ex = cppe[i]
        img = ex["image"].convert("RGB")
        for bbox, cat in zip(ex["objects"]["bbox"], ex["objects"]["category"]):
            x, y, w, h = (float(v) for v in bbox)          # CPPE-5 boxes are COCO xywh
            gts[CLASSES[cat]].setdefault(i, []).append([x, y, x + w, y + h])
        t0 = time.perf_counter()
        out = detector(img, candidate_labels=CANDIDATES, threshold=threshold)
        elapsed += time.perf_counter() - t0
        for r in out:
            cls = to_class(r["label"])
            if cls is None:
                continue
            b = r["box"]
            preds[cls].append((i, float(r["score"]), [b["xmin"], b["ymin"], b["xmax"], b["ymax"]]))
    aps = {c: average_precision(preds[c], gts[c]) for c in CLASSES}
    return aps, elapsed / n_images


n_boxes = sum(len(cppe[i]["objects"]["bbox"]) for i in range(20))
print(f"CPPE-5 test: {len(cppe)} images, {n_boxes} boxes in the 20 we benchmark on")
display(cppe[0]["image"].convert("RGB").resize((320, 240)))
```

    CPPE-5 test: 29 images, 154 boxes in the 20 we benchmark on

![](13_Zero_Shot_Object_Detection_files/figure-html/cell-12-output-7.png)

``` python
import pandas as pd

# fp32 for all three so the timings are comparable (and so deformable attention stays stable).
BENCH = {
    "owlv2-base": "google/owlv2-base-patch16-ensemble",
    "grounding-dino-tiny": "IDEA-Research/grounding-dino-tiny",
    "llmdet-tiny": "iSEE-Laboratory/llmdet_tiny",
}

rows = []
for name, model_id in BENCH.items():
    det = pipeline(
        "zero-shot-object-detection",
        model=model_id,
        device=device,
        model_kwargs={"cache_dir": HF_CACHE},
    )
    aps, sec = evaluate(det, n_images=20)
    rows.append({"model": name, **{c: aps[c] for c in CLASSES},
                 "mAP50": float(np.nanmean(list(aps.values()))), "sec/img": sec})
    print(f"{name:22s} mAP50 {rows[-1]['mAP50']:.3f}   {sec:.2f} s/img")
    del det, aps                       # free before loading the next - never two detectors live
    free_memory()

vram("after benchmark")

df = pd.DataFrame(rows).set_index("model").round(3)
display(df)
```

    [transformers] You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset

    owlv2-base             mAP50 0.707   1.66 s/img

    grounding-dino-tiny    mAP50 0.194   1.46 s/img

    llmdet-tiny            mAP50 0.018   1.47 s/img
    VRAM after benchmark       0.01 GB allocated /  0.02 GB reserved

|                     | Coverall | Face_Shield | Gloves | Goggles | Mask  | mAP50 | sec/img |
|---------------------|----------|-------------|--------|---------|-------|-------|---------|
| model               |          |             |        |         |       |       |         |
| owlv2-base          | 0.801    | 0.138       | 0.984  | 0.680   | 0.933 | 0.707 | 1.660   |
| grounding-dino-tiny | 0.451    | 0.038       | 0.183  | 0.020   | 0.277 | 0.194 | 1.464   |
| llmdet-tiny         | 0.052    | 0.016       | 0.005  | 0.002   | 0.013 | 0.018 | 1.470   |

``` python
from pyecharts.charts import Bar, Scatter

models = list(df.index)

bar = Bar()
bar.add_xaxis(models)
for c in CLASSES:
    bar.add_yaxis(c, [round(float(df.loc[m, c]) * 100, 1) for m in models])
bar.add_yaxis("mAP50", [round(float(df.loc[m, "mAP50"]) * 100, 1) for m in models])
bar.set_global_opts(
    title_opts=opts.TitleOpts(
        title="CPPE-5 test: AP50 per class (20 images, RTX 3060)",
        subtitle="Zero-shot transfer to a 5-class vocabulary no model was box-trained on",
    ),
    xaxis_opts=opts.AxisOpts(name="model"),
    yaxis_opts=opts.AxisOpts(name="AP50 (%)"),
    tooltip_opts=opts.TooltipOpts(trigger="axis"),
    legend_opts=opts.LegendOpts(pos_top="12%"),
)
display(bar.render_notebook())

scatter = Scatter()
scatter.add_xaxis([round(float(df.loc[m, "sec/img"]), 3) for m in models])
for m in models:  # one series per model so the legend names the points
    scatter.add_yaxis(
        m,
        [[round(float(df.loc[m, "sec/img"]), 3), round(float(df.loc[m, "mAP50"]) * 100, 1)]],
        symbol_size=18,
    )
scatter.set_global_opts(
    title_opts=opts.TitleOpts(title="Accuracy vs latency", subtitle="up and to the left is better"),
    xaxis_opts=opts.AxisOpts(name="seconds / image", type_="value"),
    yaxis_opts=opts.AxisOpts(name="mAP50 (%)", type_="value"),
    tooltip_opts=opts.TooltipOpts(trigger="item"),
)
scatter.render_notebook()
```

## 14. Interactive Demo: an open-vocabulary webcam

The payoff of the whole task: change the class list, and the detector changes, with no retraining. Type a new string, get a new detector.

This needs OpenCV and a camera; on a headless server (which is where this notebook usually runs) the cell reports that and skips cleanly. OWLv2-base on a 3060 is roughly 0.1-0.3 s/frame at 640x480, so this is a “grab a frame, detect, show” loop, not 30 FPS. For genuine real-time open-vocabulary video you want **OmDet-Turbo** (`omlab/omdet-turbo-swin-tiny-hf`, Apache-2.0, transformers-native, 100+ FPS with TensorRT), which caches the text embeddings and only re-runs the vision tower per frame.

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


from transformers import pipeline

# Everything below builds on the notebook's setup and helper cells.
bootstrap("device", "HF_CACHE", "draw_boxes", "from_pipeline", "free_memory", "vram",
          notebook="13_Zero_Shot_Object_Detection.ipynb",
          sections=["7. Setup"])



LIVE_LABELS = ["a person", "a laptop", "a coffee mug", "a phone"]  # <- edit freely, no retraining

detector = pipeline(
    "zero-shot-object-detection",
    model="google/owlv2-base-patch16-ensemble",
    device=device,
    model_kwargs={"cache_dir": HF_CACHE},
)


def annotate(rgb):
    "One frame -> (frame with boxes for whatever you named, count + labels)."
    res = detector(rgb, candidate_labels=LIVE_LABELS, threshold=0.2)
    names = ", ".join(sorted({r["label"] for r in res})) or "nothing over threshold"
    return draw_boxes(rgb, from_pipeline(res)), f"{len(res):2d} boxes   {names}"


# OWLv2 is the accuracy pick, not the speed pick: expect around 1 FPS here. The
# same loop with owlv2-base-patch16 (no ensemble) or OWL-ViT trades points for pace.
live_stream(annotate)

del detector
free_memory()
vram("final")
```

    cold start: running 7. Setup from 13_Zero_Shot_Object_Detection.ipynb (output suppressed)

![](13_Zero_Shot_Object_Detection_files/figure-html/cell-15-output-3.jpeg)

    frame   12     0.8 FPS    0 boxes   nothing over threshold

    [transformers] You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset

    12 frames in 15.3s -> 0.8 FPS end-to-end (camera + model + JPEG encode)
    VRAM final                 0.01 GB allocated /  0.01 GB reserved

## 15. Common Frameworks

Open-vocabulary detection is used far more often as a *labelling* tool than as an inference model, and its ecosystem reflects that. The dominant production pattern is not “deploy OWLv2” - it is “run OWLv2 once over unlabelled images at a low threshold, have a human correct the boxes, then train a fast closed-set detector on the result”. Most of the table below supports that loop rather than serving.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [transformers](https://github.com/huggingface/transformers) | modelling | OWL-ViT, OWLv2, Grounding DINO, LLMDet and OmDet-Turbo behind `zero-shot-object-detection`, with text and image queries | Apache 2.0 | Default. Covers every open-vocabulary detector worth running here |
| [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLO-World, YOLOE) | modelling | Real-time open-vocabulary detection, faster than anything transformers-native | **AGPL-3.0 (copyleft - it reaches your service)** | You need open-vocabulary *and* real-time and have dealt with the license. `omdet-turbo` is the Apache-2.0 alternative |
| API-only models: Grounding DINO 1.6 Pro, DINO-X, T-Rex2 | modelling | The accuracy ceiling, plus visual-prompt and interactive counting modes | proprietary (IDEA Research API) | Accuracy matters more than owning the weights. You are renting capability, not acquiring it |
| [Autodistill](https://github.com/autodistill/autodistill) | data | The auto-labelling loop packaged: a big open-vocabulary model labels, a small closed model trains on the output | Apache 2.0 | The standard production pattern. Open vocabulary at label time, closed-set speed at inference time |
| [Roboflow supervision](https://github.com/roboflow/supervision) | data | NMS across overlapping phrase queries, box annotation, filtering and counting utilities | MIT | Always. Open-vocabulary output needs more post-processing than closed-set output, not less |
| [CVAT](https://github.com/cvat-ai/cvat) / [Label Studio](https://github.com/HumanSignal/label-studio) + [FiftyOne](https://github.com/voxel51/fiftyone) | data | Human correction of the proposed boxes, and the error analysis that tells you which phrases fail | MIT / Apache 2.0 | Every auto-labelling loop. The model proposes, a person disposes, FiftyOne shows you where it went wrong |
| [SAM 2](https://github.com/facebookresearch/sam2) | data | Boxes to pixel masks - “Grounded SAM”, the most-used composition in open-vocabulary vision | Apache 2.0 | You need masks rather than boxes. See `12_Mask_Generation` |
| [ONNX Runtime](https://onnxruntime.ai/) / [TensorRT](https://github.com/NVIDIA/TensorRT) | inference runtime | Export, and text embeddings computed once per vocabulary rather than per image | MIT / Apache 2.0 (TensorRT SDK proprietary) | Deployment. A fixed query set makes the text tower a constant - most implementations recompute it wastefully |
| [OVDEval](https://github.com/om-ai-lab/OVDEval) + [pycocotools](https://github.com/cocodataset/cocoapi) | evaluation | mAP on your own vocabulary, plus probes for negation, position and relationships | Apache 2.0 / BSD-2 | Always. OVDEval will tell you quickly that your model is ignoring most of your caption |

**The 2026 default stack** is `transformers` with LLMDet or OWLv2 at a low threshold, Autodistill or a CVAT loop for human correction, then a closed-set RT-DETR trained on the result (`02_Object_Detection`). SAM 2 when you need masks; `omdet-turbo` when you need speed and an Apache license.

The common wrong turn is treating the prompt as fixed and the model as the variable. Section 12 shows the query set and the threshold moving results more than swapping detectors does - “person” and “a photo of a person” are different models in effect. The second is deploying an open-vocabulary detector where a closed-set one belongs: if the vocabulary has been stable for a month, you are paying several times the inference cost for flexibility you are not using.

------------------------------------------------------------------------

## 16. Going Further

- **Fine-tuning.** Open-vocabulary detectors fine-tune well on a few hundred domain images and *keep* their open vocabulary if you keep the text encoder frozen and use a low LR on the fusion layers. Grounding DINO and OWLv2 both have `...ForObjectDetection` heads that accept `labels` for the bipartite matching loss; see the [LearnOpenCV Grounding DINO fine-tuning walkthrough](https://learnopencv.com/fine-tuning-grounding-dino/). The bigger win is usually **prompt tuning** (section 12), which costs nothing.
- **Auto-labelling loop.** The dominant production pattern: run `llmdet_large` at a low threshold over your unlabelled images, have a human accept/correct the boxes, then train a *closed* RT-DETR/YOLO on the result (see `02_Object_Detection`). You get open-vocabulary flexibility at label time and closed-set speed and accuracy at inference time. [Autodistill](https://github.com/autodistill/autodistill) packages exactly this.
- **Boxes to masks.** Pipe any box from this notebook into SAM 2 for a mask - “Grounded SAM”. See `12_Mask_Generation`, and the [Grounded SAM paper](https://arxiv.org/abs/2401.14159).
- **Real-time.** `omlab/omdet-turbo-swin-tiny-hf` (Apache-2.0, transformers-native) is the right speed pick. **YOLO-World** and **YOLOE** are faster still but ship through `ultralytics` under **AGPL-3.0** - a vendor runtime and a viral license, deliberately not a dependency here. Export to ONNX/TensorRT if you need the last 3x.
- **VLM grounding.** Qwen3-VL and PaliGemma 2 emit boxes as text from arbitrary prompts, which buys you counting, OCR and reasoning in the same call. Slower, coordinate-quantised, and weak on dense scenes; use them to *propose* or *verify*, not to localise 200 objects. See `05_Image_to_Text`.
- **Diagnose your prompts.** [OVDEval](https://github.com/om-ai-lab/OVDEval) probes negation, position and relationships and will tell you, quickly, that your model is ignoring most of your caption.
- **Related notebooks.** `02_Object_Detection` (closed-set), `11_Zero_Shot_Image_Classification` (the same CLIP trick without localisation), `12_Mask_Generation` (SAM), `03_Image_Segmentation`.

**References**

- [OWL-ViT: Simple Open-Vocabulary Object Detection with Vision Transformers (Minderer et al., 2022)](https://arxiv.org/abs/2205.06230)
- [OWLv2 / OWL-ST: Scaling Open-Vocabulary Object Detection (Minderer et al., NeurIPS 2023)](https://arxiv.org/abs/2306.09683)
- [Grounding DINO (Liu et al., 2023)](https://arxiv.org/abs/2303.05499) and [Grounding DINO 1.5 (2024)](https://arxiv.org/abs/2405.10300)
- [LLMDet (Fu et al., CVPR 2025 highlight)](https://arxiv.org/abs/2501.18954) and the [official repo with the LVIS table](https://github.com/iSEE-Laboratory/LLMDet)
- [MM Grounding DINO: an open and comprehensive pipeline (2024)](https://arxiv.org/abs/2401.02361)
- [OmDet-Turbo: real-time OVD with an efficient fusion head (2024)](https://arxiv.org/abs/2403.06892)
- [DINO-X: unified open-world detection and understanding (2024)](https://arxiv.org/abs/2411.14347)
- [GLIP: Grounded Language-Image Pre-training (2022)](https://arxiv.org/abs/2112.03857) - the origin of ODinW
- [transformers: zero-shot object detection task guide](https://huggingface.co/docs/transformers/tasks/zero_shot_object_detection)
- [Papers with Code: zero-shot object detection on LVIS minival](https://paperswithcode.com/sota/zero-shot-object-detection-on-lvis-v1-0-minival)
- [LVIS dataset](https://www.lvisdataset.org/)

------------------------------------------------------------------------

Back to top
