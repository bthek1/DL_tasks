# Video Classification

Action recognition end to end: what the task really tests, the mid-2026 model landscape, frame sampling and multi-view evaluation, metrics, and runnable code that pits video models against a single-frame baseline.

Author

Benedict Thekkel

## 1. What is Video Classification?

Video classification assigns a label to a **clip**. The dominant instance is **action recognition**: given a few seconds of trimmed video, name the action (“archery”, “bowling”, “marching”).

**Input.** A tensor of shape `(T, H, W, 3)` - `T` RGB frames sampled from the decoded video. This is the first thing that makes video different from images: the model does **not** see the video, it sees a fixed-size *sample* of it (8, 16 or 32 frames), and the sampling policy is part of the model contract. Section 8 is entirely about this.

**Output.** For trimmed action recognition, one softmax over a closed label set (400 classes for Kinetics-400, 174 for Something-Something v2). Multi-label variants (Charades) emit independent sigmoids; open-vocabulary models (X-CLIP, video-LLMs) score arbitrary text against the clip.

**Sub-tasks that get lumped under “video classification”:**

| Task | Question it answers | Output | Typical tool |
|----|----|----|----|
| Trimmed action recognition (this notebook) | *what* happened | one label per clip | VideoMAE, TimeSformer, V-JEPA 2 |
| Temporal action localisation | *when* did it happen in an untrimmed video | (start, end, label) intervals | ActionFormer, TriDet |
| Spatio-temporal action detection | *who* did *what* *where* | per-person box + action, per frame | SlowFast + detector, on AVA |
| Video retrieval / text-video matching | *which clip* matches this sentence | ranked clips | X-CLIP, InternVideo2, CLIP4Clip |
| Repetition counting / fine-grained action | *how many* / *which sub-step* | count, sub-action label | RepNet, EPIC-Kitchens models |
| Video question answering | free-form reasoning about a video | text | see `Multimodal/06_Video_Text_to_Text` |

The frame-level analogue is `01_Image_Classification` (a video model is, to a first approximation, an image model with a temporal axis bolted on), and the open-vocabulary story mirrors `11_Zero_Shot_Image_Classification` exactly - X-CLIP is CLIP with cross-frame attention. Generative video tasks live in `07_Image_to_Video`, `10_Text_to_Video` and `18_Video_to_Video`.

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Content moderation and policy classification | Social video (YouTube, TikTok) | Uploaded clip -\> policy label (violence, self-harm, nudity) | Throughput at millions of clips/day; cost per clip; recall on rare classes |
| Sports analytics and auto-highlights | Broadcast (Hawk-Eye, Stats Perform) | Multi-camera feed -\> event labels (shot, foul, serve) | Fine-grained temporal precision; near-live latency |
| Surveillance / anomaly detection | Physical security, retail loss prevention | 24/7 CCTV -\> “fall detected”, “fight”, “shoplifting” | Streaming on edge boxes; false-positive rate; privacy and bias scrutiny |
| Industrial process and safety monitoring | Manufacturing, mining, construction | Fixed-camera feed -\> “PPE missing”, “step 4 skipped” | On-prem, no cloud; domain shift per site; must run on a Jetson |
| Driver / cabin monitoring | Automotive (Euro NCAP DMS mandate) | IR cabin video -\> “drowsy”, “phone use” | Hard real-time, fully offline, tiny model |
| Physical therapy and fitness coaching | Consumer health (Apple Fitness+, Tempo) | Phone camera -\> exercise class + rep count | On-device latency; robustness to viewpoint and body type |
| Robot policy learning and affordance | Robotics (V-JEPA 2-AC, RT-X) | Egocentric video -\> action/affordance representation | Representation quality, not label accuracy; sample efficiency |
| Media archive search | Broadcast archives, stock footage | Untrimmed archive -\> searchable action tags | Open vocabulary (the tag set is not known in advance) |

**What the Kinetics number hides.** A leaderboard clip is 10 seconds, trimmed to the action, shot by a human who framed it, at 25 fps and 224px. Production video is untrimmed (the action is 3% of the timeline and *localisation* is the real problem), off-axis and low-light, 8-15 fps from an RTSP stream that drops frames, and belongs to a class taxonomy that was invented last week. Three failure modes dominate: (1) **background bias** - the model learned the scene, not the action, so it fires on an empty gym (see section 13); (2) **temporal boundary errors** - trimmed models have no notion of “not doing anything yet”, so a sliding window over untrimmed footage produces a solid wall of confident predictions; (3) **the decode bill** - at scale, ffmpeg, not the GPU, is your bottleneck and your cost line. Also note the **streaming vs offline** fork: models built for offline scoring re-run over the whole clip per window, while a deployed detector must be causal and amortised (this is why TSM, which adds temporal modelling at zero FLOPs, is still shipped on edge boxes in 2026).

------------------------------------------------------------------------

## 3. How Modern Video Classification Works

Dated progression, each generation still alive somewhere:

1.  **Two-stream (2014).** One CNN on RGB, one on stacked **optical flow**, fused late. Flow is an explicit motion representation and it worked so well that it exposed the field’s central embarrassment: the RGB stream alone was already near-SOTA on the datasets of the day.
2.  **C3D (2015) and I3D (2017).** Replace 2-D convs with 3-D. C3D trained from scratch and was weak; **I3D’s trick was inflation** - copy a pretrained 2-D ImageNet kernel `T` times along the temporal axis and divide by `T`, so the 3-D net starts from ImageNet features instead of noise. That, plus the new Kinetics dataset, is what made 3-D CNNs work.
3.  **Factorised 3-D: P3D / R(2+1)D (2017-2018).** A 3x3x3 conv is a spatial 1x3x3 followed by a temporal 3x1x1. Same receptive field, fewer parameters, an extra non-linearity, easier optimisation, better accuracy. The same factorisation idea returns in every later generation.
4.  **SlowFast (2019).** Two pathways: a *slow* one at low frame rate with many channels (semantics), a *fast* one at high frame rate with few channels (motion), joined by lateral connections. Still the standard backbone for AVA-style spatio-temporal detection.
5.  **TSM (2019).** Shift a fraction of the channels forward and backward along time inside a 2-D ResNet. Temporal modelling for **zero extra FLOPs and zero parameters** - the reason 2-D-backbone models still ship to the edge.
6.  **Video transformers (2021).** **TimeSformer** made the key efficiency observation: joint space-time attention over `T x N` patches costs `O((TN)^2)` and is infeasible (8 frames x 196 patches = 1568 tokens, and it grows quadratically with clip length), so it uses **divided space-time attention** - attend over time within a patch position, then over space within a frame - which costs `O(T^2 N + T N^2)` and is both cheaper and *more accurate*. **ViViT** explored the same factorisation space (factorised encoder / self-attention / dot-product). **Video Swin** and **MViT/MViTv2** brought hierarchical windows and pooling attention.
7.  **Self-supervised pretraining (2022-2023).** **VideoMAE** masks **90-95% of the spatio-temporal tubes** and reconstructs pixels. The absurd mask ratio is only possible *because* video is redundant - a low ratio makes the task trivially solvable by copying the neighbouring frame. This is what made video transformers trainable on 3-4k videos with no extra data. **VideoMAE V2** scaled it to a billion-parameter ViT-g with dual masking (~90% on K400, ~77% on SSv2).
8.  **Video foundation models and SSMs (2024-2025).** **InternVideo2** (masked reconstruction + video-text contrastive + next-token prediction, scaled to 6B) tops most video benchmarks at once. **VideoMamba** replaces attention with a selective state-space scan - linear in sequence length, so long clips get cheap.
9.  **Latent-space prediction (2025-2026).** **V-JEPA 2** (Meta, June 2025) predicts *representations*, not pixels, of masked video; the ViT-g reaches ~87% on K400 and ~77% on SSv2 from frozen features with an attentive probe, and the same encoder drives a robot world model (V-JEPA 2-AC). **V-JEPA 2.1** (March 2026) adds a dense predictive loss and deep self-supervision, pushing SSv2 to 77.7. Meanwhile **video-LLMs** (Qwen2.5/3-VL, InternVL, LLaVA-Video; see `Multimodal/06_Video_Text_to_Text`) increasingly win *open-vocabulary* video understanding by just prompting - at 10-100x the latency of a purpose-built 100M-param classifier.

**Trade-off cheat sheet:**

| Family | Temporal modelling | Cost | Streaming | Best for |
|----|----|----|----|----|
| 2-D CNN + TSM | channel shift (free) | lowest | yes | edge, real-time, cheap baselines |
| SlowFast / X3D | 3-D convs, two rates | low-medium | yes | AVA detection, video-rate inference |
| TimeSformer / ViViT | divided space-time attention | medium | no | trimmed classification, fine-tuning |
| VideoMAE / V-JEPA 2 | ViT + SSL pretraining | medium-high | no | best accuracy per label, transfer, frozen features |
| X-CLIP / video-CLIP | cross-frame attention + text | medium | no | open vocabulary, zero-shot, retrieval |
| Video-LLM | LLM over frame tokens | very high | no | open-ended reasoning, unseen taxonomies |

**Who leads in mid-2026.** For a fixed closed label set, an SSL-pretrained ViT (VideoMAE V2, V-JEPA 2, InternVideo2) fine-tuned on the target set is the accuracy leader - the very top of Kinetics-400 sits around 94% top-1 with huge VLM-aided backbones. For an *unknown* label set, a CLIP-style video model or a video-LLM wins by default because the closed-set models cannot answer at all. For a Jetson, it is still a 2-D backbone with TSM.

------------------------------------------------------------------------

## 4. Evaluation Metrics

**Top-1 / top-5 accuracy** on the clip label - the Kinetics standard:

$$\text{top-}k = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\left[y_i \in \text{argsort}(s_i)_{1:k}\right]$$

**Mean class accuracy** (the average of per-class recalls) is the honest number on an imbalanced set. Kinetics classes are not uniform in the wild, and a model can buy top-1 by favouring frequent classes:

$$\text{mCA} = \frac{1}{C}\sum_{c=1}^{C} \frac{\text{correct}_c}{n_c}$$

**mAP** for multi-label untrimmed video (Charades): average precision per class, averaged over classes. For **temporal localisation**, mAP is computed at temporal IoU thresholds (typically 0.5:0.05:0.95 on ActivityNet, 0.3:0.1:0.7 on THUMOS).

**The views multiplier.** Published Kinetics numbers are almost never single-clip. The standard test protocol samples **multiple temporal clips x multiple spatial crops** (commonly `10 clips x 3 crops = 30 views`) and averages the softmax. This is worth 1-3 points of top-1 - and it multiplies inference cost by 30. Two consequences:

- A single-clip score (like the one this notebook computes) is **not** comparable to a published one. Say which protocol you used, always.
- Published **GFLOPs** are equally incomparable unless views are stated. “TimeSformer: 196 GFLOPs” means *per view*; the reported accuracy needed 3 views. Compare `GFLOPs x views`.

**Speed metrics.** Clips/sec (or its inverse, latency per clip) end-to-end **including decoding** - on real workloads ffmpeg often costs more than the forward pass. Report frames consumed per clip too: an 8-frame model and a 32-frame model are not doing the same amount of work.

The cell below computes top-1, top-5 and mean-class accuracy from raw logits, and shows what multi-clip logit averaging does to a prediction.

------------------------------------------------------------------------

``` python
import numpy as np

rng = np.random.default_rng(0)
N, C = 12, 6  # 12 clips, 6 classes

labels = rng.integers(0, C, size=N)
# Fabricate logits that are right about 60% of the time.
logits = rng.normal(0, 1.0, size=(N, C))
hit = rng.random(N) < 0.6
logits[np.arange(N), labels] += np.where(hit, 3.0, 0.0)

def topk_accuracy(logits, labels, k=1):
    "Fraction of clips whose true label is in the top-k scores."
    topk = np.argsort(-logits, axis=1)[:, :k]
    return float(np.mean([labels[i] in topk[i] for i in range(len(labels))]))

def mean_class_accuracy(logits, labels, n_classes):
    "Average of per-class recalls - the honest number on an imbalanced set."
    pred = logits.argmax(1)
    recalls = [
        float(np.mean(pred[labels == c] == c))
        for c in range(n_classes)
        if (labels == c).any()
    ]
    return float(np.mean(recalls))

print(f"top-1 {topk_accuracy(logits, labels, 1):.3f}")
print(f"top-5 {topk_accuracy(logits, labels, 5):.3f}")
print(f"mCA   {mean_class_accuracy(logits, labels, C):.3f}")

# Multi-clip testing: score V temporal views of the SAME video and average the
# softmax. Individual views are noisy (a bad crop, a frame where nothing happens);
# the average is what the leaderboard number is actually measuring.
V = 10
views = logits[0] + rng.normal(0, 1.5, size=(V, C))  # 10 noisy views of clip 0
probs = np.exp(views - views.max(1, keepdims=True))
probs /= probs.sum(1, keepdims=True)

print("\nclip 0 true label:", labels[0])
print("per-view argmax:  ", probs.argmax(1).tolist())
print("averaged argmax:  ", probs.mean(0).argmax(), "  (30 views = 30x the compute)")
```

    top-1 0.417
    top-5 0.917
    mCA   0.333

    clip 0 true label: 5
    per-view argmax:   [1, 0, 0, 1, 0, 0, 1, 1, 1, 0]
    averaged argmax:   1   (30 views = 30x the compute)

## 5. Datasets

| Dataset | Contents | Size | Scope | License | Typical use |
|----|----|----|----|----|----|
| [Kinetics-400 / 600 / 700](https://github.com/cvdfoundation/kinetics-dataset) | 10 s YouTube clips, human actions | ~240k / 480k / 650k clips | 400 / 600 / 700 classes | CC-BY-4.0 (annotations); videos owned by uploaders | the standard train + benchmark |
| [Something-Something v2](https://www.qualcomm.com/developer/software/something-something-v-2-dataset) | staged clips of hands manipulating objects | ~220k clips | 174 **motion-defined** classes | research licence, **registration required** | the temporal-reasoning benchmark |
| [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) | web action clips | 13.3k clips | 101 classes | free for research | small, historical, fine-tuning demos |
| [HMDB-51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) | movie + web clips | 6.8k clips | 51 classes | free for research | small, historical |
| [AVA](https://research.google.com/ava/) | 15-min movie segments, box-level | 430 videos, 1.6M labels | 80 atomic actions | CC-BY-4.0 | spatio-temporal detection |
| [Charades](https://prior.allenai.org/projects/charades) | untrimmed home activity videos | 9.8k videos | 157 classes, **multi-label** | non-commercial | mAP, untrimmed, co-occurring actions |
| [Moments in Time](http://moments.csail.mit.edu/) | 3 s clips, actions by any agent | ~1M clips | 339 classes | research | broad-coverage pretraining |
| [EPIC-Kitchens-100](https://epic-kitchens.github.io/) | egocentric cooking, untrimmed | 100 h | verb x noun | CC-BY-NC | egocentric, anticipation |
| [HowTo100M](https://www.di.ens.fr/willow/research/howto100m/) / [InternVid](https://huggingface.co/datasets/OpenGVLab/InternVid) | narrated web video + text | 136M / 234M clips | open vocabulary | research / CC-BY-NC | video-text pretraining |

**Link rot is the real Kinetics problem.** Kinetics ships as a list of YouTube IDs, not videos. A meaningful fraction of the original URLs are dead (deleted, private, region-blocked), and the fraction grows every year - so “Kinetics-400 top-1” in a 2026 paper and in a 2018 paper were not measured on the same test set. Mirrors exist (the CVDF snapshot, Academic Torrents) and you should say which one you used.

**This notebook evaluates on** [`nateraw/kinetics-mini`](https://huggingface.co/datasets/nateraw/kinetics-mini) - 50 validation clips over 5 Kinetics-400 classes (archery, bowling, flying kite, high jump, marching). It is a smoke-test set: the labels are genuine K400 labels so a K400 model’s predictions are directly interpretable, but 20-50 clips measure nothing statistically. **SSv2 is gated** (registration + licence agreement), which is why the SSv2-finetuned models below are demonstrated qualitatively rather than scored.

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

Leaderboard: [Papers with Code - Kinetics-400](https://paperswithcode.com/sota/action-classification-on-kinetics-400) and [Something-Something v2](https://paperswithcode.com/sota/action-recognition-in-videos-on-something). The HF task page ([video-classification](https://huggingface.co/tasks/video-classification)) lists what is actually loadable.

| Model | Params | License | Scope | Architecture | Best for |
|----|----|----|----|----|----|
| [`MCG-NJU/videomae-base-finetuned-kinetics`](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics) | 87M | CC-BY-NC-4.0 | K400 (400 cls) | ViT-B, MAE pretrained, 16 frames | the default workhorse; **runnable** |
| [`MCG-NJU/videomae-base-finetuned-ssv2`](https://huggingface.co/MCG-NJU/videomae-base-finetuned-ssv2) | 87M | CC-BY-NC-4.0 | SSv2 (174 cls) | same, SSv2 head | motion-defined classes; **runnable** |
| [`facebook/timesformer-base-finetuned-k400`](https://huggingface.co/facebook/timesformer-base-finetuned-k400) | 121M | CC-BY-NC-4.0 | K400 | divided space-time attention, 8 frames | cheapest transformer; **runnable** |
| [`google/vivit-b-16x2-kinetics400`](https://huggingface.co/google/vivit-b-16x2-kinetics400) | 89M | MIT | K400 | factorised encoder, 32 frames | long-ish clips; **runnable** |
| [`microsoft/xclip-base-patch32`](https://huggingface.co/microsoft/xclip-base-patch32) | 197M | MIT | open vocabulary | CLIP + cross-frame attention, 8 frames | zero-shot / unknown taxonomies; **runnable** (80.4% K400 top-1 supervised) |
| [`facebook/vjepa2-vitl-fpc16-256-ssv2`](https://huggingface.co/facebook/vjepa2-vitl-fpc16-256-ssv2) | 375M | MIT | SSv2 | ViT-L, latent-space SSL + attentive probe | best motion understanding you can run here; **runnable** |
| [`facebook/vjepa2-vitl-fpc64-256`](https://huggingface.co/facebook/vjepa2-vitl-fpc64-256) | 326M | MIT | encoder only | ViT-L, 64 frames | frozen features, retrieval, robotics |
| VideoMAE V2-g | 1B | CC-BY-NC | K400 / SSv2 | ViT-g, dual masking | ~90% K400, ~77% SSv2 - **too big for 12 GB in fp16 at 16 frames** |
| V-JEPA 2 ViT-g (384) | 1B | MIT | K400 / SSv2 | latent-space SSL | ~87% K400, ~77% SSv2 - **too big here** |
| InternVideo2-6B | 6B | research | everything | masked + contrastive + next-token | video foundation model; **far too big** |
| Qwen3-VL / InternVL (video mode) | 2B-72B | Apache 2.0 / custom | open-ended | video-LLM | open vocabulary + reasoning; 2B fits, see `Multimodal/06_Video_Text_to_Text` |

**Who wins what.** Accuracy on a closed set: a 1B-param SSL ViT (VideoMAE V2, V-JEPA 2 ViT-g) - none of which fit on this box in a comfortable configuration. Accuracy per GB on a 12 GB card: VideoMAE-base and V-JEPA 2 ViT-L. Speed: TimeSformer-base at 8 frames, or a TSM-ResNet if you leave the transformers ecosystem. Open vocabulary (the “media archive” and “moderation with a new policy” rows in section 2): X-CLIP or a video-LLM, because a K400 head simply cannot emit a class it was not trained on.

**License trap.** VideoMAE and TimeSformer checkpoints are **CC-BY-NC-4.0** - non-commercial. ViViT, X-CLIP and V-JEPA 2 are MIT. For a commercial deployment, that distinction usually matters more than 2 points of top-1.

------------------------------------------------------------------------

## 7. Setup

Everything below runs on the 12 GB RTX 3060 (or on CPU, slowly), and every model loads through Hugging Face `transformers`. Package roles:

- `transformers` (\>=5.13) + `torch` - VideoMAE, TimeSformer, ViViT, X-CLIP, V-JEPA 2, CLIP
- `accelerate` - `device_map` placement
- `torchcodec` - video decoding (FFmpeg-backed); `av` (PyAV) is the fallback and ships its own FFmpeg
- `huggingface_hub` - pulls the eval clips
- `pyecharts` - charts; `pandas` - result table; `pillow` - the frame contact sheet

No `pytorchvideo`, no `decord`, no `mmaction2`: they are excellent, but they are vendor/per-task packages and nothing here needs them.

All downloads (clips, HF cache) land in `DL_tasks/datasets/`, which is gitignored.

------------------------------------------------------------------------

``` python
# Everything runs through Hugging Face transformers - no video-specific frameworks.
# %pip install -q torch transformers accelerate torchcodec huggingface_hub pyecharts pandas pillow

# Fallback decoder if torchcodec cannot find a system FFmpeg (PyAV bundles its own):
# %pip install -q av
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
from huggingface_hub import hf_hub_download, list_repo_files

# Eval set: 50 Kinetics-400 validation clips over 5 classes (10 each).
# The directory name is the class; the K400 label uses spaces ("flying kite").
EVAL_REPO = "nateraw/kinetics-mini"
files = sorted(f for f in list_repo_files(EVAL_REPO, repo_type="dataset")
               if f.startswith("val/") and f.endswith(".mp4"))

CLIPS_PER_CLASS = 4  # 4 x 5 classes = 20 clips: a smoke test, not a benchmark
by_class = {}
for f in files:
    by_class.setdefault(f.split("/")[1], []).append(f)

eval_set = []  # list of (local_path, kinetics_label)
for cls, paths in sorted(by_class.items()):
    for f in paths[:CLIPS_PER_CLASS]:
        local = hf_hub_download(EVAL_REPO, f, repo_type="dataset", cache_dir=HF_CACHE)
        eval_set.append((local, cls.replace("_", " ")))

# Single demo clip used by the per-model sections (the transformers docs' own sample).
SAMPLE = hf_hub_download(
    "nielsr/video-demo", "eating_spaghetti.mp4", repo_type="dataset", cache_dir=HF_CACHE
)

CLASSES = sorted({lbl for _, lbl in eval_set})
print(f"{len(eval_set)} eval clips over {len(CLASSES)} classes: {CLASSES}")
print("demo clip:", Path(SAMPLE).name)
```

    20 eval clips over 5 classes: ['archery', 'bowling', 'flying kite', 'high jump', 'marching']
    demo clip: eating_spaghetti.mp4

## 8. Frame Sampling: the Decision Nobody Documents

A video model does not consume a video. It consumes exactly `T` frames at a fixed resolution - `T` is baked into the position embeddings (`model.config.num_frames`) and you cannot change it without re-training. So every clip must be reduced to `T` frames, and **how** you reduce it changes the answer:

- **Uniform sampling** (`np.linspace(0, n-1, T)`) spreads `T` frames over the whole clip. Robust default for trimmed 10 s clips; it silently subsamples fast motion (a 10 s clip at 25 fps into 16 frames = one frame every 15).
- **Dense clip** (`T` consecutive frames at stride `s`, e.g. 16 x stride 4 = 2.5 s) preserves motion but only sees a window - so you take **several windows** and average (multi-clip testing).
- **Multi-view testing** (10 clips x 3 crops) is the published protocol. It buys 1-3 top-1 points and costs 30x. This notebook uses **1 clip x 1 crop** and says so.

The cell below decodes with `torchcodec` (falling back to PyAV), samples uniformly, and shows the sampled frames as a contact sheet. `read_video` caches nothing: at scale, decoding - not the GPU - is the bottleneck, which is why production pipelines pre-extract frames to a shard format once.

------------------------------------------------------------------------

``` python
import numpy as np
from PIL import Image

# Decoder backend: torchcodec (FFmpeg-backed, in this repo's deps) -> PyAV -> error.
_BACKEND = None
try:
    from torchcodec.decoders import VideoDecoder
    _BACKEND = "torchcodec"
except Exception:
    try:
        import av
        _BACKEND = "pyav"
    except Exception:
        _BACKEND = None
print("decoder backend:", _BACKEND)

def read_video(path, num_frames=32):
    "Uniformly sample `num_frames` RGB frames across the whole clip -> (T,H,W,3) uint8."
    if _BACKEND == "torchcodec":
        dec = VideoDecoder(str(path))
        total = dec.metadata.num_frames
        want = np.linspace(0, total - 1, num_frames).round().astype(int)
        # (T,C,H,W) uint8 -> (T,H,W,C)
        return dec.get_frames_at(indices=want.tolist()).data.permute(0, 2, 3, 1).numpy()
    if _BACKEND == "pyav":
        container = av.open(str(path))
        stream = container.streams.video[0]
        total = stream.frames or int(stream.duration * stream.time_base * stream.average_rate)
        want = np.linspace(0, total - 1, num_frames).round().astype(int)
        keep, last = {}, int(want.max())
        for i, frame in enumerate(container.decode(video=0)):
            if i in set(want.tolist()):
                keep[i] = frame.to_ndarray(format="rgb24")
            if i >= last:
                break
        container.close()
        return np.stack([keep[i] for i in want])
    raise RuntimeError("no video decoder: pip install torchcodec (needs FFmpeg) or av")

def subsample(frames, k):
    "Uniformly pick k of the cached frames (models want 8, 16 or 32)."
    return frames[np.linspace(0, len(frames) - 1, k).round().astype(int)]

def contact_sheet(frames, cols=8, width=160):
    "Lay frames out as a grid so you can see what the model actually sees."
    thumbs = [Image.fromarray(f).resize((width, int(width * f.shape[0] / f.shape[1]))) for f in frames]
    w, h = thumbs[0].size
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), "black")
    for i, t in enumerate(thumbs):
        sheet.paste(t, ((i % cols) * w, (i // cols) * h))
    return sheet

t0 = time.perf_counter()
demo_frames = read_video(SAMPLE, num_frames=32)  # decode once, reuse for every model
print(f"decoded {demo_frames.shape} in {time.perf_counter() - t0:.2f}s")
contact_sheet(subsample(demo_frames, 16))
```

    decoder backend: torchcodec
    decoded (32, 360, 640, 3) in 0.16s

![](09_Video_Classification_files/figure-html/cell-6-output-2.png)

## 9. VideoMAE (the Workhorse)

`MCG-NJU/videomae-base-finetuned-kinetics` - a plain ViT-B pretrained by masking **90%+ of the spatio-temporal tubes** and reconstructing pixels, then fine-tuned on Kinetics-400. 87M params, 16 frames at 224px, ~81% K400 top-1 at the published 5x3-view protocol. It is the default choice for “I have a labelled video dataset and want a strong model” - the SSL pretraining is what lets it fine-tune on a few thousand clips without collapsing.

Note the licence: **CC-BY-NC-4.0**, non-commercial.

We also stash `K400_LABELS` here - the 400 class names - because X-CLIP will need them as text prompts in section 11.

------------------------------------------------------------------------

``` python
from transformers import AutoVideoProcessor, AutoModelForVideoClassification

VIDEOMAE_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
vm_proc = AutoVideoProcessor.from_pretrained(VIDEOMAE_ID, cache_dir=HF_CACHE)
vm_model = AutoModelForVideoClassification.from_pretrained(
    VIDEOMAE_ID, dtype=dtype, cache_dir=HF_CACHE
).to(device).eval()

K400_LABELS = [vm_model.config.id2label[i] for i in range(vm_model.config.num_labels)]
print(f"{vm_model.config.num_frames} frames, {len(K400_LABELS)} classes")

def prepare(processor, frames):
    "Preprocess (T,H,W,3) uint8 frames into model inputs on the right device/dtype."
    return processor(list(frames), return_tensors="pt").to(device=device, dtype=dtype)

def top5(model, logits):
    "Print the top-5 classes for a (1, C) logit tensor."
    probs = logits.float().softmax(-1)[0]
    for score, idx in zip(*probs.topk(5)):
        print(f"  {score.item():6.1%}  {model.config.id2label[idx.item()]}")

inputs = prepare(vm_proc, subsample(demo_frames, vm_model.config.num_frames))
t0 = time.perf_counter()
with torch.inference_mode():
    vm_logits = vm_model(**inputs).logits
print(f"VideoMAE forward: {time.perf_counter() - t0:.3f}s (16 frames, 1 clip, 1 crop)")
top5(vm_model, vm_logits)

del vm_model, vm_proc, inputs, vm_logits
free_memory()
vram("after videomae")
```

    [transformers] VideoMAEForVideoClassification LOAD REPORT from: MCG-NJU/videomae-base-finetuned-kinetics
    Key                                                            | Status     | 
    ---------------------------------------------------------------+------------+-
    videomae.encoder.layer.{0...11}.attention.attention.q_bias     | UNEXPECTED | 
    videomae.encoder.layer.{0...11}.attention.attention.v_bias     | UNEXPECTED | 
    videomae.encoder.layer.{0...11}.attention.attention.key.bias   | MISSING    | 
    videomae.encoder.layer.{0...11}.attention.attention.value.bias | MISSING    | 
    videomae.encoder.layer.{0...11}.attention.attention.query.bias | MISSING    | 

    Notes:
    - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    - MISSING:  those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.

    16 frames, 400 classes
    VideoMAE forward: 0.205s (16 frames, 1 clip, 1 crop)
       79.5%  eating spaghetti
        1.2%  making pizza
        0.6%  tossing coin
        0.5%  carving pumpkin
        0.3%  eating chips
    VRAM after videomae        0.01 GB allocated /  0.02 GB reserved

## 10. TimeSformer (Divided Space-Time Attention)

`facebook/timesformer-base-finetuned-k400` - the first pure video transformer (2021). Its contribution is an efficiency argument, not an accuracy one: **joint** space-time attention over `T x N` patch tokens (8 x 196 = 1568) is `O((TN)^2)` and does not fit; **divided** attention applies temporal attention across the same patch position in every frame, then spatial attention within each frame, for `O(T^2 N + T N^2)`. That is both cheaper *and* better, because the two attention maps are each a well-conditioned problem.

8 frames, 121M params, ~78% K400 top-1. The cheapest transformer here per clip - which is exactly why it survives as a baseline.

------------------------------------------------------------------------

``` python
TIMESFORMER_ID = "facebook/timesformer-base-finetuned-k400"
ts_proc = AutoVideoProcessor.from_pretrained(TIMESFORMER_ID, cache_dir=HF_CACHE)
ts_model = AutoModelForVideoClassification.from_pretrained(
    TIMESFORMER_ID, dtype=dtype, cache_dir=HF_CACHE
).to(device).eval()

n = ts_model.config.num_frames  # 8
inputs = prepare(ts_proc, subsample(demo_frames, n))
t0 = time.perf_counter()
with torch.inference_mode():
    ts_logits = ts_model(**inputs).logits
print(f"TimeSformer forward: {time.perf_counter() - t0:.3f}s ({n} frames)")
top5(ts_model, ts_logits)

del ts_model, ts_proc, inputs, ts_logits
free_memory()
vram("after timesformer")
```

    TimeSformer forward: 0.038s (8 frames)
       99.4%  eating spaghetti
        0.6%  dining
        0.0%  tasting food
        0.0%  eating burger
        0.0%  eating chips
    VRAM after timesformer     0.01 GB allocated /  0.02 GB reserved

## 11. X-CLIP (Zero-Shot / Open-Vocabulary Video Classification)

`microsoft/xclip-base-patch32` - CLIP with (a) cross-frame attention so patches can exchange information across time, (b) a Multiframe Integration Transformer that pools frames into one video embedding, and (c) a video-conditioned prompt generator. Because the classifier is a *text encoder*, the label set is an argument, not a weight matrix - the exact analogue of `11_Zero_Shot_Image_Classification`.

This is the model for the section-2 rows where the taxonomy is not known in advance (media archives, a moderation policy written last week). Supervised on K400 it reaches 80.4% top-1; used zero-shot it is far weaker but it *can answer at all*, which a K400 head cannot.

Below: first an arbitrary 3-way label set, then the full 400-way K400 label set so its score is directly comparable to the closed-set models in section 14.

------------------------------------------------------------------------

``` python
from transformers import AutoModel, AutoProcessor

XCLIP_ID = "microsoft/xclip-base-patch32"
xc_proc = AutoProcessor.from_pretrained(XCLIP_ID, cache_dir=HF_CACHE)
xc_model = AutoModel.from_pretrained(XCLIP_ID, dtype=dtype, cache_dir=HF_CACHE).to(device).eval()

def xclip_scores(frames, labels, n_frames=8):
    "Softmax over an arbitrary text label set for one clip."
    inputs = xc_proc(
        text=labels, videos=list(subsample(frames, n_frames)),
        return_tensors="pt", padding=True,
    ).to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
    with torch.inference_mode():
        logits = xc_model(**inputs).logits_per_video  # (1, n_labels)
    return logits.float().softmax(-1)[0].cpu()

# 1. An arbitrary label set invented right now - no retraining, no head surgery.
custom = ["eating spaghetti", "playing basketball", "a cat knocking a glass over"]
for lbl, p in zip(custom, xclip_scores(demo_frames, custom)):
    print(f"  {p:6.1%}  {lbl}")

# 2. The full 400-way Kinetics label set, so section 14 can compare like with like.
t0 = time.perf_counter()
probs = xclip_scores(demo_frames, K400_LABELS)
print(f"\n400-way zero-shot: {time.perf_counter() - t0:.2f}s")
for score, idx in zip(*probs.topk(5)):
    print(f"  {score.item():6.1%}  {K400_LABELS[idx]}")

del xc_model, xc_proc
free_memory()
vram("after xclip")
```

      100.0%  eating spaghetti
        0.0%  playing basketball
        0.0%  a cat knocking a glass over

    400-way zero-shot: 0.06s
       94.4%  eating spaghetti
        0.9%  dining
        0.4%  making pizza
        0.3%  tasting food
        0.1%  setting table
    VRAM after xclip           0.01 GB allocated /  0.02 GB reserved

## 12. V-JEPA 2 (Latent-Space Self-Supervision)

`facebook/vjepa2-vitl-fpc16-256-ssv2` - Meta’s V-JEPA 2 (June 2025), a ViT-L trained to predict the *representation* of masked video regions rather than their pixels, then given an attentive-probe classification head on **Something-Something v2**. 375M params, MIT licence, 16 frames at 256px. Predicting in latent space skips the pixel-reconstruction burden (texture, lighting, noise) that VideoMAE spends capacity on, and the resulting encoder is strong enough to drive a robot world model (V-JEPA 2-AC). V-JEPA 2.1 (March 2026) adds a dense predictive loss and reports 77.7 on SSv2.

The SSv2 head matters for the next section: its 174 classes are **motion-defined** (“moving something up”, “pushing something from left to right”). A single frame cannot answer them, by construction. Below we run the clip forwards and then **time-reversed** - a K400 model barely notices; a genuine motion model should.

Note this is a different label space from the K400 models, so it is *not* in the section-14 benchmark. Comparing a 174-way SSv2 score with a 400-way K400 score would be meaningless.

------------------------------------------------------------------------

``` python
from transformers import AutoVideoProcessor

VJEPA_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"
vj_proc = AutoVideoProcessor.from_pretrained(VJEPA_ID, cache_dir=HF_CACHE)
vj_model = AutoModelForVideoClassification.from_pretrained(
    VJEPA_ID, dtype=dtype, cache_dir=HF_CACHE
).to(device).eval()
vram("vjepa2 loaded")

def vjepa_top(frames, k=3):
    inputs = vj_proc(list(subsample(frames, 16)), return_tensors="pt").to(device=device, dtype=dtype)
    with torch.inference_mode():
        logits = vj_model(**inputs).logits
    probs = logits.float().softmax(-1)[0]
    return [(vj_model.config.id2label[i.item()], s.item()) for s, i in zip(*probs.topk(k))]

t0 = time.perf_counter()
print("forwards:")
for lbl, p in vjepa_top(demo_frames):
    print(f"  {p:6.1%}  {lbl}")
print(f"({time.perf_counter() - t0:.2f}s)\n")

print("time-reversed (same pixels, reversed arrow of time):")
for lbl, p in vjepa_top(demo_frames[::-1].copy()):
    print(f"  {p:6.1%}  {lbl}")
print("\nCaveat: this demo clip is out-of-domain for SSv2 (which is staged hand-object")
print("manipulation), so read the *change* between the two runs, not the labels.")

del vj_model, vj_proc
free_memory()
vram("after vjepa2")
```

    VRAM vjepa2 loaded         0.76 GB allocated /  0.77 GB reserved
    forwards:
       59.1%  Scooping [something] up with [something]
       16.2%  Pouring [something] into [something]
        9.5%  Spreading [something] onto [something]
    (0.19s)

    time-reversed (same pixels, reversed arrow of time):
       34.5%  Scooping [something] up with [something]
       31.2%  Spreading [something] onto [something]
       13.5%  Sprinkling [something] onto [something]

    Caveat: this demo clip is out-of-domain for SSv2 (which is staged hand-object
    manipulation), so read the *change* between the two runs, not the labels.
    VRAM after vjepa2          0.01 GB allocated /  0.02 GB reserved

## 13. Does Motion Actually Matter? (the Central Experiment)

Here is the uncomfortable fact at the centre of this field: **a single-frame image classifier is a shockingly strong baseline on Kinetics.** “Swimming” is a pool. “Playing guitar” is a guitar. “Archery” is a target and a bow. The scene and the objects leak the label, so a model can score well while modelling almost no motion at all - and several papers have shown that shuffling the frames of a Kinetics clip costs surprisingly little accuracy.

Four controlled probes on the same eval clips, same model (VideoMAE-K400), same 400-way head - only the temporal content of the input changes:

| Variant | What it destroys | If accuracy holds, then… |
|----|----|----|
| `uniform-16` | nothing (the normal input) | (reference) |
| `reversed` | the arrow of time | the model ignores temporal *direction* |
| `shuffled` | temporal order entirely | the model is a bag of frames |
| `static` (centre frame x16) | **all** motion | the model is an image classifier in disguise |

Plus a genuine image model - CLIP ViT-B/32 zero-shot on the **centre frame only**, scored over the same 400 K400 label names. That row is not a controlled ablation (different training, open-vocabulary, never saw a K400 label during training), so read it as “what does one frame plus web-scale image knowledge get you”, not as an apples-to-apples number.

**This is why Something-Something v2 exists.** Its classes are relations *over time* (“moving something up”, “pretending to pick something up”), the same objects appear in every class, and the static/shuffled probes collapse to near-chance. If you want to know whether your model does temporal reasoning, SSv2 (or Diving48, or Temporal-Shape datasets) is the test; Kinetics is not.

------------------------------------------------------------------------

``` python
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org serves an expired TLS certificate from most nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"

from pyecharts import options as opts
from pyecharts.charts import Bar

N_EVAL = len(eval_set)
cache = {p: read_video(p, 32) for p, _ in eval_set}  # decode once (~20 clips x 32 frames)

VARIANTS = {
    "uniform-16": lambda f: subsample(f, 16),
    "reversed":   lambda f: subsample(f, 16)[::-1].copy(),
    "shuffled":   lambda f: subsample(f, 16)[np.random.default_rng(0).permutation(16)],
    "static":     lambda f: np.repeat(subsample(f, 16)[7:8], 16, axis=0),  # centre frame x16
}

vm_proc = AutoVideoProcessor.from_pretrained(VIDEOMAE_ID, cache_dir=HF_CACHE)
vm_model = AutoModelForVideoClassification.from_pretrained(
    VIDEOMAE_ID, dtype=dtype, cache_dir=HF_CACHE
).to(device).eval()

ablation = {}
for name, transform in VARIANTS.items():
    correct = 0
    for path, label in eval_set:
        inputs = prepare(vm_proc, transform(cache[path]))
        with torch.inference_mode():
            pred = vm_model(**inputs).logits.argmax(-1).item()
        correct += vm_model.config.id2label[pred] == label
    ablation[name] = correct / N_EVAL
    print(f"VideoMAE {name:12s} top-1 {ablation[name]:6.1%}")

del vm_model, vm_proc
free_memory()

# The image-model baseline: CLIP on the centre frame, 400-way zero-shot.
from transformers import CLIPModel, CLIPProcessor

clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=HF_CACHE)
clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32", dtype=dtype, cache_dir=HF_CACHE
).to(device).eval()

prompts = [f"a photo of a person {lbl}" for lbl in K400_LABELS]
with torch.inference_mode():
    tok = clip_proc(text=prompts, return_tensors="pt", padding=True).to(device)
    text_emb = clip_model.get_text_features(**tok).pooler_output
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

    correct = 0
    for path, label in eval_set:
        centre = Image.fromarray(cache[path][16])  # one frame, no motion at all
        px = clip_proc(images=centre, return_tensors="pt").to(device=device, dtype=dtype)
        img_emb = clip_model.get_image_features(**px).pooler_output
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        pred = (img_emb @ text_emb.T).argmax(-1).item()
        correct += K400_LABELS[pred] == label
ablation["CLIP 1 frame"] = correct / N_EVAL
print(f"CLIP     centre frame top-1 {ablation['CLIP 1 frame']:6.1%}  (image model, zero-shot)")

del clip_model, clip_proc, text_emb, tok
free_memory()
vram("after ablation")

order = ["uniform-16", "reversed", "shuffled", "static", "CLIP 1 frame"]
bar = (
    Bar()
    .add_xaxis(order)
    .add_yaxis("top-1 (%)", [round(100 * ablation[k], 1) for k in order])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title="How much does motion matter?",
            subtitle=f"VideoMAE-K400 under temporal ablations, {N_EVAL} Kinetics clips, 1 clip x 1 crop",
        ),
        xaxis_opts=opts.AxisOpts(name="input variant"),
        yaxis_opts=opts.AxisOpts(name="top-1 accuracy (%)", max_=100),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
    )
)
bar.render_notebook()
```

    [transformers] VideoMAEForVideoClassification LOAD REPORT from: MCG-NJU/videomae-base-finetuned-kinetics
    Key                                                            | Status     | 
    ---------------------------------------------------------------+------------+-
    videomae.encoder.layer.{0...11}.attention.attention.q_bias     | UNEXPECTED | 
    videomae.encoder.layer.{0...11}.attention.attention.v_bias     | UNEXPECTED | 
    videomae.encoder.layer.{0...11}.attention.attention.key.bias   | MISSING    | 
    videomae.encoder.layer.{0...11}.attention.attention.value.bias | MISSING    | 
    videomae.encoder.layer.{0...11}.attention.attention.query.bias | MISSING    | 

    Notes:
    - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    - MISSING:  those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.

    VideoMAE uniform-16   top-1  80.0%
    VideoMAE reversed     top-1  80.0%
    VideoMAE shuffled     top-1  60.0%
    VideoMAE static       top-1  55.0%

    CLIP     centre frame top-1  50.0%  (image model, zero-shot)
    VRAM after ablation        0.01 GB allocated /  0.05 GB reserved

## 14. Head-to-head Benchmark

Three K400 models on the **same 20 clips, the same uniform sampling, the same 400-way label space, 1 clip x 1 crop**, loading and freeing one at a time:

- `videomae-base-finetuned-kinetics` (16 frames, supervised head)
- `timesformer-base-finetuned-k400` (8 frames, supervised head)
- `vivit-b-16x2-kinetics400` (32 frames, supervised head)
- `xclip-base-patch32` (8 frames, **zero-shot** over the 400 label strings - a different kind of model, shown for contrast, not as a fair fight)

Latency is the **model forward only** (decoding is cached and timed separately in section 8) - which flatters every row equally and hides the fact that in production ffmpeg is often the bigger bill.

**Read this as a smoke test, not a leaderboard.** 20 clips over 5 easy, visually distinct classes, single-view, on an RTX 3060. Published Kinetics numbers use the full 20k-clip val set at 30 views; a 20-clip score has an error bar of roughly +/- 10 points and cannot rank models that are 2 points apart.

------------------------------------------------------------------------

``` python
BENCH = [
    ("videomae-base",   VIDEOMAE_ID,    16),
    ("timesformer-base", TIMESFORMER_ID, 8),
    ("vivit-b-16x2",    "google/vivit-b-16x2-kinetics400", 32),
]

results = {}
for name, model_id, n_frames in BENCH:
    proc = AutoVideoProcessor.from_pretrained(model_id, cache_dir=HF_CACHE)
    model = AutoModelForVideoClassification.from_pretrained(
        model_id, dtype=dtype, cache_dir=HF_CACHE
    ).to(device).eval()

    correct, elapsed = 0, 0.0
    for path, label in eval_set:
        inputs = prepare(proc, subsample(cache[path], n_frames))
        t0 = time.perf_counter()
        with torch.inference_mode():
            logits = model(**inputs).logits
        if device != "cpu":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - t0
        correct += model.config.id2label[logits.argmax(-1).item()] == label

    results[name] = {
        "top1": correct / N_EVAL,
        "clips_per_sec": N_EVAL / elapsed,
        "frames": n_frames,
    }
    print(f"{name:18s} top-1 {results[name]['top1']:6.1%}  "
          f"{results[name]['clips_per_sec']:5.1f} clips/s  {n_frames} frames")

    del model, proc
    free_memory()

# X-CLIP, zero-shot over the same 400 label strings (open-vocab, no K400 head).
xc_proc = AutoProcessor.from_pretrained(XCLIP_ID, cache_dir=HF_CACHE)
xc_model = AutoModel.from_pretrained(XCLIP_ID, dtype=dtype, cache_dir=HF_CACHE).to(device).eval()
correct, elapsed = 0, 0.0
for path, label in eval_set:
    t0 = time.perf_counter()
    probs = xclip_scores(cache[path], K400_LABELS)
    if device != "cpu":
        torch.cuda.synchronize()
    elapsed += time.perf_counter() - t0
    correct += K400_LABELS[int(probs.argmax())] == label
results["xclip-b32 (zero-shot)"] = {
    "top1": correct / N_EVAL, "clips_per_sec": N_EVAL / elapsed, "frames": 8
}
print(f"{'xclip-b32 (zs)':18s} top-1 {results['xclip-b32 (zero-shot)']['top1']:6.1%}  "
      f"{results['xclip-b32 (zero-shot)']['clips_per_sec']:5.1f} clips/s  8 frames")

del xc_model, xc_proc
free_memory()
vram("after benchmark")
```

    [transformers] VideoMAEForVideoClassification LOAD REPORT from: MCG-NJU/videomae-base-finetuned-kinetics
    Key                                                            | Status     | 
    ---------------------------------------------------------------+------------+-
    videomae.encoder.layer.{0...11}.attention.attention.q_bias     | UNEXPECTED | 
    videomae.encoder.layer.{0...11}.attention.attention.v_bias     | UNEXPECTED | 
    videomae.encoder.layer.{0...11}.attention.attention.key.bias   | MISSING    | 
    videomae.encoder.layer.{0...11}.attention.attention.value.bias | MISSING    | 
    videomae.encoder.layer.{0...11}.attention.attention.query.bias | MISSING    | 

    Notes:
    - UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
    - MISSING:  those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.

    videomae-base      top-1  80.0%   47.1 clips/s  16 frames

    timesformer-base   top-1  90.0%   37.6 clips/s  8 frames

    vivit-b-16x2       top-1   0.0%   16.3 clips/s  32 frames

    xclip-b32 (zs)     top-1  90.0%   15.3 clips/s  8 frames
    VRAM after benchmark       0.02 GB allocated /  0.05 GB reserved

``` python
import pandas as pd
from IPython.display import display
from pyecharts import options as opts
from pyecharts.charts import Scatter

df = pd.DataFrame(
    [(k, v["top1"], v["clips_per_sec"], v["frames"]) for k, v in results.items()],
    columns=["model", "top1", "clips_per_sec", "frames"],
).sort_values("top1", ascending=False)
display(df)

# One series per model so the legend carries the names (each series has a single point).
xs = [round(float(x), 2) for x in df["clips_per_sec"]]
scatter = Scatter().add_xaxis(xs)
for i, (name, acc) in enumerate(zip(df["model"], df["top1"])):
    ys = [None] * len(xs)
    ys[i] = round(100 * float(acc), 1)
    scatter.add_yaxis(name, ys, symbol_size=18, label_opts=opts.LabelOpts(is_show=False))

scatter.set_global_opts(
    title_opts=opts.TitleOpts(
        title="Accuracy vs speed",
        subtitle=f"{N_EVAL} Kinetics clips, 1 clip x 1 crop, RTX 3060 fp16, forward pass only",
    ),
    xaxis_opts=opts.AxisOpts(name="clips/sec", type_="value"),
    yaxis_opts=opts.AxisOpts(name="top-1 (%)", type_="value", min_=0, max_=100),
    tooltip_opts=opts.TooltipOpts(trigger="item"),
)
scatter.render_notebook()
```

|     | model                 | top1 | clips_per_sec | frames |
|-----|-----------------------|------|---------------|--------|
| 1   | timesformer-base      | 0.9  | 37.593891     | 8      |
| 3   | xclip-b32 (zero-shot) | 0.9  | 15.254366     | 8      |
| 0   | videomae-base         | 0.8  | 47.064250     | 16     |
| 2   | vivit-b-16x2          | 0.0  | 16.255529     | 32     |

## 15. Real-Time Demo (Webcam Sliding Window)

The deployed form of this task is not “score a clip”, it is “keep a rolling buffer of the last `T` frames and re-classify every `k` frames”. That is what the cell below does with OpenCV, on a 16-frame ring buffer.

Two honest caveats it makes visible: a trimmed K400 model has **no “nothing is happening” class**, so it will always emit a confident label; and re-running a 16-frame transformer every few frames is exactly the redundant compute that TSM-style causal models exist to avoid.

Guarded with `try/except` - on a headless server (no camera, no OpenCV) it prints and skips.

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


from transformers import AutoModelForVideoClassification, AutoVideoProcessor

# Everything below builds on the notebook's setup and helper cells.
bootstrap("device", "dtype", "HF_CACHE", "free_memory", "vram",
          notebook="09_Video_Classification.ipynb",
          sections=["7. Setup"])


# Both come from section 9 when the notebook is run in order. Defaulting them
# here keeps this demo to a single cheap prerequisite section.
VIDEOMAE_ID = globals().get("VIDEOMAE_ID", "MCG-NJU/videomae-base-finetuned-kinetics")

if "prepare" not in globals():

    def prepare(processor, frames):
        "Preprocess (T,H,W,3) uint8 frames into model inputs on the right device/dtype."
        return processor(list(frames), return_tensors="pt").to(device=device, dtype=dtype)


from collections import deque

proc = AutoVideoProcessor.from_pretrained(VIDEOMAE_ID, cache_dir=HF_CACHE)
model = AutoModelForVideoClassification.from_pretrained(
    VIDEOMAE_ID, dtype=dtype, cache_dir=HF_CACHE
).to(device).eval()

# VideoMAE classifies a 16-frame *clip*, not a frame, so the live version keeps a
# rolling buffer and re-runs every STRIDE frames. Until the buffer fills there is
# simply no prediction to show - that latency is inherent to video models.
STRIDE = 8
buffer = deque(maxlen=16)
state = {"step": 0, "label": "filling buffer", "score": 0.0}


def annotate(rgb):
    "One frame -> (frame labelled with the current clip prediction, same as text)."
    buffer.append(np.asarray(rgb))  # VideoMAE wants raw RGB arrays, not PIL
    state["step"] += 1
    if len(buffer) == 16 and state["step"] % STRIDE == 0:
        inputs = prepare(proc, np.stack(buffer))
        with torch.inference_mode():
            probs = model(**inputs).logits.float().softmax(-1)[0]
        score, idx = probs.max(-1)
        state["label"] = model.config.id2label[idx.item()]
        state["score"] = score.item()
    info = f"{state['label'][:44]}  {state['score']:5.1%}"
    return draw_lines(rgb, [f"buffer {len(buffer):2d}/16", info]), info


live_stream(annotate)

del model, proc
free_memory()
vram("final")
```

## 16. Common Frameworks

Video classification has a bottleneck that is not the model: decoding. A clip of 16 frames sampled from a 10-second video costs more CPU to produce than the transformer costs GPU to classify, and on a 4-core box that decides your throughput entirely. So the frameworks that matter most here are the decoders and the sharded data formats, followed by the handful of libraries that carry the CNN-era architectures `transformers` never adopted.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [transformers](https://github.com/huggingface/transformers) | modelling | VideoMAE, TimeSformer, X-CLIP and V-JEPA 2 behind `AutoModelForVideoClassification`, plus the processors that handle frame stacks | Apache 2.0 | Default. Covers the transformer era completely |
| [pytorchvideo](https://github.com/facebookresearch/pytorchvideo) / [mmaction2](https://github.com/open-mmlab/mmaction2) | modelling | SlowFast, TSM, X3D, TSN - the efficient CNN architectures that still win on latency-constrained edge hardware | Apache 2.0 | You need one of those exact models, or a temporal-localisation head that `transformers` does not have |
| [decord](https://github.com/dmlc/decord) / [torchcodec](https://github.com/pytorch/torchcodec) / [PyAV](https://github.com/PyAV-Org/PyAV) | data | Random-access frame decoding without walking the whole file, which is what clip sampling needs | Apache 2.0 / BSD-3 | Always. Naive ffmpeg-per-clip decoding is usually the single largest cost in a training run |
| [webdataset](https://github.com/webdataset/webdataset) | data | Pre-extracted frames or clips in sequential shards, so the loader streams instead of seeking | BSD-3 | Training. Pre-extraction turns a CPU-bound job into a GPU-bound one and is worth the disk |
| [FiftyOne](https://github.com/voxel51/fiftyone) | data | Video dataset browsing with per-clip predictions overlaid - the only practical way to audit temporal labels | Apache 2.0 | Always. Kinetics-style labels are noisy and many “errors” are annotation disagreements |
| [ONNX Runtime](https://onnxruntime.ai/) / [TensorRT](https://github.com/NVIDIA/TensorRT) | inference runtime | Export the classifier once and run it per window on edge hardware | MIT / Apache 2.0 (TensorRT SDK proprietary) | Deployment on a camera or a Jetson, where a TSM-ResNet still beats every transformer here on latency |
| [DeepStream](https://developer.nvidia.com/deepstream-sdk) / [Triton Inference Server](https://github.com/triton-inference-server/server) | serving | Hardware-decoded multi-stream video straight into a batched model, without the frames ever touching Python | proprietary / BSD-3 | Many live camera feeds. Decode-on-GPU is the whole point |
| [torchmetrics](https://github.com/Lightning-AI/torchmetrics) | evaluation | Top-1/top-5, mean class accuracy, and the multi-view aggregation the published protocols assume | Apache 2.0 | Always, and report the view protocol (`1x1` vs `10x3`) with the number - they are not comparable |

**The 2026 default stack** is `transformers` with VideoMAE or a V-JEPA 2 frozen probe, `decord` for sampling, `webdataset` shards if you train, and ONNX or TensorRT if it deploys to a camera. The frame-sampling decision of section 8 matters more than the choice between the models.

The common wrong turn is treating a trimmed-clip classifier as a video understanding system. Real footage is untrimmed, and a sliding window over a trimmed classifier is a demo - temporal action localisation is a different task with different models. The second is claiming temporal understanding without a motion-defined benchmark: section 13 exists because most Kinetics accuracy survives frame shuffling, which means the model is doing image classification.

------------------------------------------------------------------------

## 17. Going Further

- **Fine-tuning.** `VideoMAEForVideoClassification` fine-tunes on a few thousand clips: the [HF video-classification guide](https://huggingface.co/docs/transformers/tasks/video_classification) walks through UCF-101 with `Trainer`. Start from `MCG-NJU/videomae-base-finetuned-kinetics` (not the bare `-base`) when your classes overlap Kinetics - the K400 head is a much better initialisation than a random one. For a frozen-feature baseline, take `facebook/vjepa2-vitl-fpc64-256`, cache the embeddings once, and train a linear/attentive probe in minutes (see `16_Image_Feature_Extraction` for the image analogue).
- **Evaluate honestly.** Report the view protocol (`1x1` vs `10x3`), the mean class accuracy alongside top-1, and at least one motion-defined benchmark (SSv2, Diving48) if you are claiming temporal understanding.
- **Untrimmed video** needs a different task: temporal action localisation (ActionFormer, TriDet) or spatio-temporal detection on AVA (SlowFast + person detector). A sliding window of a trimmed classifier is a demo, not a system.
- **Open vocabulary.** If the label set changes weekly, stop training classifiers: prompt a video-LLM (`Multimodal/06_Video_Text_to_Text`) or use X-CLIP/InternVideo2 embeddings for retrieval.

**References**

- [VideoMAE (Tong et al., 2022)](https://arxiv.org/abs/2203.12602) and [VideoMAE V2 (2023)](https://arxiv.org/abs/2303.16727)
- [TimeSformer: Is Space-Time Attention All You Need? (2021)](https://arxiv.org/abs/2102.05095)
- [ViViT: A Video Vision Transformer (2021)](https://arxiv.org/abs/2103.15691)
- [SlowFast Networks (2019)](https://arxiv.org/abs/1812.03982) and [TSM (2018)](https://arxiv.org/abs/1811.08383)
- [X-CLIP: Expanding Language-Image Pretrained Models for General Video Recognition (2022)](https://arxiv.org/abs/2208.02816)
- [V-JEPA 2 (Meta, 2025)](https://arxiv.org/abs/2506.09985) and [V-JEPA 2.1 (2026)](https://arxiv.org/abs/2603.14482)
- [The “something something” video database (Goyal et al., 2017)](https://arxiv.org/abs/1706.04261) - why motion-defined classes were needed
- [Papers with Code: Kinetics-400 leaderboard](https://paperswithcode.com/sota/action-classification-on-kinetics-400)
- [HF task page: video-classification](https://huggingface.co/tasks/video-classification)

------------------------------------------------------------------------

Back to top
