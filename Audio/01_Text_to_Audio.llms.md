# Text-to-Audio

Generating non-speech audio - music and sound effects - from a text prompt: how the models work, the mid-2026 open landscape, evaluation, and runnable transformers / diffusers code on GPU or CPU.

Author

Benedict Thekkel

## 1. What is Text-to-Audio?

Text-to-Audio (TTA) generates **general audio** - music, sound effects, ambiences - from a natural-language prompt. It is distinct from Text-to-Speech (which generates spoken words); the two are usually separate models.

**Input.** A text prompt (“lo-fi hip hop with a mellow piano”, “rain on a tin roof, distant thunder”), optionally with a **duration**, a **melody** to condition on, or a seed.

**Output.** A mono or stereo waveform, typically **16 kHz** (MusicGen) up to **44.1 kHz stereo** (Stable Audio Open).

**Neighbouring tasks:**

| Task               | What it does                  | Typical tool        |
|--------------------|-------------------------------|---------------------|
| Text-to-Speech     | Generate spoken words         | SpeechT5, Bark      |
| Audio captioning   | The inverse: describe a sound | Qwen2-Audio, CLAP   |
| Music continuation | Extend an audio clip          | MusicGen (melody)   |
| Audio-to-audio     | Transform existing audio      | Demucs, enhancement |

------------------------------------------------------------------------

## 2. Real-World Use Cases

Text-to-audio covers music and general sound generation, and it is deployed almost entirely as a *content production* tool - so the binding constraints are about rights and controllability far more than about model quality.

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Background music for video and ads | Marketing, social media (Suno, Udio, YouTube Create) | Text prompt + duration -\> music track | Rights clearance and training-data provenance; cost per clip |
| Game audio and adaptive soundtracks | Gaming | Prompt or game state -\> loopable music, sound effects | Loopability and seamless transitions; runtime latency if generated live |
| Foley and sound design for post | Film, TV (iZotope-class tooling) | Description (+ video) -\> effect aligned to picture | Temporal alignment to picture; 48 kHz delivery fidelity |
| Stock music and sonic branding | Advertising, stock libraries | Prompt -\> catalogue of variations | Diversity, and a clean licence chain per track |
| Melody-conditioned drafting | Music production | Prompt + reference melody -\> arrangement or stems | Faithfulness to the conditioning melody; separable stems |
| Synthetic data for audio ML | ML engineering | Prompt -\> labelled training audio | Label fidelity and acoustic diversity; cost per hour |

**What the FAD score hides.** The blocker for shipping a text-to-audio feature is usually **not** audio quality - it is **provenance**: what the model was trained on, whether the output is licensable, and whether a customer can defend it. Check the training-data and licence terms of a checkpoint before its benchmark numbers. Beyond that: generation is **slow and offline** (many seconds of GPU time per second of audio), so these models sit in batch pipelines, not in real-time paths. **Delivery format** bites - most open checkpoints emit 32 kHz mono while broadcast and game engines want 44.1/48 kHz stereo, so an upsampling and mastering stage is part of the system, not an afterthought. And prompt adherence **decays with duration**: coherence typically holds for tens of seconds, after which structure wanders, loops repeat, and the mix goes muddy - which is why production systems generate short conditioned segments and arrange them, rather than asking for a three-minute song in one shot.

------------------------------------------------------------------------

## 3. How Modern Text-to-Audio Works

Two dominant families, converging on quality:

1.  **Autoregressive codec LM (AudioGen, MusicGen, 2023).** Encode audio into discrete **EnCodec** tokens, then predict them with a transformer LM conditioned on **T5** text embeddings. Simple, streamable, strong on music; Meta’s MusicGen is the open reference.
2.  **Latent diffusion (AudioLDM, AudioLDM 2, Tango, 2023).** Run a diffusion model in a compressed VAE latent, conditioned on **CLAP** and/or T5 text features, then decode + vocode to waveform. Strong on general sound effects.
3.  **Latent diffusion transformers / flow (Stable Audio, Stable Audio Open, 2024).** A DiT with **timing conditioning** generates long, high-fidelity **44.1 kHz stereo** in one pass. State of the art for open SFX and loops.
4.  **2025-2026.** Faster sampling (consistency / flow-matching), longer coherent structure, and strong closed systems (Suno, Udio) alongside open MusicGen / Stable Audio Open / AudioLDM 2.

------------------------------------------------------------------------

## 4. Evaluation Metrics

Generation has no single reference output, so metrics compare **distributions** and **prompt adherence**.

- **FAD (Frechet Audio Distance).** Distance between the feature distributions (VGGish / PANN embeddings) of generated vs real audio. Lower = more realistic. The primary TTA metric.
- **KL divergence.** Between PANN class-probability distributions of generated vs reference - measures semantic match.
- **CLAP score.** Cosine similarity between the **CLAP** text embedding of the prompt and the audio embedding of the output - measures how well the audio matches the prompt (higher is better).
- **IS (Inception Score)** and subjective **MOS** for quality/diversity.

The cell below computes a CLAP score with `transformers` - the one metric you can run on a single clip.

------------------------------------------------------------------------

``` python
import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

clap_id = "laion/clap-htsat-unfused"
clap = ClapModel.from_pretrained(clap_id)
clap_proc = ClapProcessor.from_pretrained(clap_id)


def clap_score(prompt, audio, sr):
    "Cosine similarity between the prompt and the audio in CLAP space (prompt adherence)."
    if sr != 48000:  # CLAP expects 48 kHz
        import librosa
        audio = librosa.resample(np.asarray(audio, dtype="float32"), orig_sr=sr, target_sr=48000)
    inp = clap_proc(text=[prompt], audio=[audio], sampling_rate=48000, return_tensors="pt", padding=True)
    with torch.no_grad():
        emb = clap(**inp)
    a = torch.nn.functional.normalize(emb.audio_embeds, dim=-1)
    t = torch.nn.functional.normalize(emb.text_embeds, dim=-1)
    return float((a * t).sum())


# Sanity check on random noise (should score low)
print("noise CLAP score:", clap_score("a solo piano melody", np.random.randn(48000) * 0.01, 48000))
```

## 5. Datasets

Text-to-audio ground truth is a caption written for an existing clip, so the model learns what annotators chose to say about a sound, not the sound itself. Most clips come from YouTube (AudioSet) or Freesound, captions are one or two short sentences, and music captions are written by a few hundred people. The typical failure is a model that gets the main event right (“a dog barks”) but ignores counts, order, timing and musical detail, because the captions it learned from rarely mention them.

| Dataset | Contents | Size | Scope | License | Typical use |
|----|----|----|----|----|----|
| [AudioCaps](https://github.com/cdjkim/audiocaps) ([HF captions mirror](https://huggingface.co/datasets/d0rj/audiocaps)) | 10 s YouTube clips from AudioSet, one human caption each (5 per clip in val/test) | 49,838 train / 495 val / 975 test clips | General sound events | Repo MIT; README says academic use only | **The** sound-effect eval (FAD, KL, CLAP); training |
| [Clotho](https://zenodo.org/records/4783391) | Freesound clips, 15-30 s, 5 captions each | 6,974 clips, 34,870 captions, 7.1 GB | General sound events | Audio: Freesound CC licenses; captions: Tampere University, non-commercial with attribution | Captioning and text-to-audio eval |
| [MusicCaps](https://huggingface.co/datasets/google/MusicCaps) | 10 s AudioSet music clips with an aspect list and a free-text caption by musicians | 5,521 clips | Music | CC BY-SA 4.0 | **The** text-to-music eval |
| [Song Describer](https://huggingface.co/datasets/renumics/song-describer-dataset) ([official](https://github.com/mulab-mir/song-describer-dataset)) | ~2 min permissively licensed tracks with crowd captions | 1,106 captions / 706 tracks (valid subset 746 / 546) | Music | Captions CC BY-SA 4.0; audio license unknown | Music eval on full-length tracks |
| [WavCaps](https://huggingface.co/datasets/cvssp/WavCaps) | ChatGPT-rewritten captions for FreeSound, BBC Sound Effects, SoundBible and AudioSet-strong clips | ~403K clips | General sound events | Academic use only | Large-scale weakly-labelled pretraining |
| [AudioSet](https://huggingface.co/datasets/agkphysics/AudioSet) | 10 s YouTube clips with event labels, no captions | ~2.08M clips, ~5.8K h, 527 classes | General sound events | CC BY 4.0 (labels) | Audio source behind AudioCaps and MusicCaps; pretraining |

**This notebook uses no dataset.** It feeds one hand-written prompt (“a warm lo-fi hip hop beat…”) to MusicGen and AudioLDM, then scores prompt adherence with a CLAP similarity. That is a smoke test. A real benchmark computes FAD and KL against the AudioCaps test set for sound effects, or MusicCaps for music.

None of the datasets above are gated on the Hub, but several do not ship audio. AudioSet, AudioCaps and MusicCaps are YouTube IDs plus timestamps, so you download the clips yourself and some are gone. **WavCaps is academic use only**, AudioCaps says academic use only in its README, and Clotho’s captions are non-commercial. Check before you train a product on them.

Downloads land in `DL_tasks/datasets/` via `cache_dir` (gitignored).

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

| Model | Params | License | Type | Library | Best for |
|----|----|----|----|----|----|
| facebook/musicgen-small | 300M | CC-BY-NC 4.0 | music (codec LM) | transformers | music from text, melody conditioning |
| facebook/musicgen-stereo-small | 300M | CC-BY-NC 4.0 | stereo music | transformers | stereo music |
| cvssp/audioldm-s-full-v2 | 350M | CC-BY-NC | sound + music (LDM) | diffusers | general sound effects |
| cvssp/audioldm2 | 350M | CC-BY-NC | sound + music (LDM) | diffusers | successor (see note) |
| stabilityai/stable-audio-open-1.0 | 1.1B | Stability Community | 44.1 kHz stereo (DiT) | diffusers | high-fidelity SFX and loops |
| facebook/audiogen-medium | 1.5B | CC-BY-NC 4.0 | sound effects | audiocraft (external) | dense sound scenes |

There is no single crowd leaderboard; papers report FAD/KL/CLAP on **AudioCaps** (sound) and **MusicCaps** (music). MusicGen loads through `transformers`; AudioLDM and Stable Audio Open load through **`diffusers`** (the standard Hugging Face diffusion library - a general-purpose lib, so it fits the repo rule). AudioGen needs the external `audiocraft` runtime.

We run **AudioLDM v1** (`AudioLDMPipeline`) below: it conditions on the CLAP text encoder directly, so it stays robust across `transformers` versions. The newer **AudioLDM 2** uses an internal GPT-2 language-model prompt path that currently breaks against `transformers` 5.x inside `diffusers` - use it only with an older, matched `diffusers`/`transformers` pair.

------------------------------------------------------------------------

## 7. Setup

Package roles:

- `transformers` (\>=5.13) + `torch` - MusicGen and CLAP
- `diffusers` - AudioLDM (and Stable Audio Open)
- `soundfile` - write WAV; `librosa` - resample for CLAP
- `pyecharts` - the benchmark chart

MusicGen and CLAP are already covered by the repo deps; `diffusers` is the one extra general-purpose library used here.

------------------------------------------------------------------------

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
if device != "cpu":
    print(torch.cuda.get_device_name(0))
print("device:", device)


def vram(tag=""):
    "Report current GPU memory (allocated / reserved). No-op on CPU."
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"VRAM {tag:16s} {alloc:5.2f} GB allocated / {reserved:5.2f} GB reserved")


def free_memory():
    "GC then release cached CPU/GPU memory. Call right after `del model`.\n\n    `del` drops the Python reference; this reclaims the RAM and hands the\n    freed VRAM back to the CUDA allocator so usage stays flat across cells.\n    "
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


# All downloads (samples, HF cache) go to DL_tasks/datasets/ (gitignored)
DATA_DIR = Path("../../datasets")
DATA_DIR.mkdir(exist_ok=True)

import soundfile as sf

PROMPT = "a warm lo-fi hip hop beat with a mellow piano and soft vinyl crackle"
OUT_DIR = DATA_DIR / "tta_out"
OUT_DIR.mkdir(exist_ok=True)


def save_wav(name, audio, sr):
    path = OUT_DIR / name
    sf.write(path, np.asarray(audio).squeeze(), sr)
    print(f"{name}: {np.asarray(audio).squeeze().shape[-1] / sr:.1f} s @ {sr} Hz -> {path}")
    return path
```

## 8. MusicGen (transformers)

The `text-to-audio` pipeline wraps MusicGen end to end. `max_new_tokens` sets the length (~50 tokens/second of audio); the output sampling rate comes from the model config (32 kHz). Use `facebook/musicgen-melody` to additionally condition on a hummed melody.

------------------------------------------------------------------------

``` python
from transformers import pipeline

musicgen = pipeline("text-to-audio", model="facebook/musicgen-small", device=device)
t0 = time.perf_counter()
out = musicgen(PROMPT, forward_params={"do_sample": True, "max_new_tokens": 256})  # ~5 s of audio
print(f"{time.perf_counter() - t0:.1f}s")
save_wav("musicgen.wav", out["audio"], out["sampling_rate"])

del musicgen
free_memory()
vram("after musicgen")
```

## 9. AudioLDM (diffusers)

AudioLDM is a latent-diffusion model good at general **sound effects** as well as music. It loads through `diffusers` and conditions on a CLAP text encoder; `num_inference_steps` trades speed for quality and `audio_length_in_s` sets duration. Runs on CPU (slowly) or GPU with `torch_dtype=float16`, and outputs 16 kHz.

------------------------------------------------------------------------

``` python
from diffusers import AudioLDMPipeline

dtype = torch.float16 if device != "cpu" else torch.float32
ldm = AudioLDMPipeline.from_pretrained("cvssp/audioldm-s-full-v2", torch_dtype=dtype,
                                       cache_dir=str(DATA_DIR / "hf_cache")).to(device)

t0 = time.perf_counter()
audio = ldm("rain on a tin roof with distant thunder", num_inference_steps=50,
            audio_length_in_s=5.0).audios[0]
print(f"{time.perf_counter() - t0:.1f}s")
save_wav("audioldm.wav", audio, 16000)  # AudioLDM outputs 16 kHz

del ldm
free_memory()
vram("after audioldm")
```

## 10. Head-to-head Benchmark

Same prompt through MusicGen and AudioLDM; report **generation RTF** and **CLAP score** (prompt adherence). CLAP is an automatic proxy - for real evaluation use FAD/KL over AudioCaps/MusicCaps. Free each model before loading the next.

------------------------------------------------------------------------

``` python
# ECharts (pyecharts) is the repo standard for all charts - it renders interactive
# and embeds straight into the Quarto docs via .render_notebook().
from pyecharts import options as opts
from pyecharts.charts import Bar


def bar_chart(title, categories, series, y_name=""):
    "Grouped bar chart. `series` is a dict {name: [values aligned to categories]}."
    chart = Bar(init_opts=opts.InitOpts(width="720px", height="420px"))
    chart.add_xaxis([str(c) for c in categories])
    for name, vals in series.items():
        chart.add_yaxis(name, [round(float(v), 4) for v in vals])
    chart.set_global_opts(
        title_opts=opts.TitleOpts(title=title),
        yaxis_opts=opts.AxisOpts(name=y_name),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=20)),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="8%"),
    )
    return chart.render_notebook()
```

``` python
from transformers import pipeline
from diffusers import AudioLDMPipeline

results = {}

# MusicGen
mg = pipeline("text-to-audio", model="facebook/musicgen-small", device=device)
t0 = time.perf_counter()
out = mg(PROMPT, forward_params={"do_sample": True, "max_new_tokens": 256})
gen_s = time.perf_counter() - t0
audio, sr = np.asarray(out["audio"]).squeeze(), out["sampling_rate"]
results["musicgen"] = {"rtf": gen_s / (len(audio) / sr), "clap": clap_score(PROMPT, audio, sr)}
del mg
free_memory()

# AudioLDM
dtype = torch.float16 if device != "cpu" else torch.float32
ldm = AudioLDMPipeline.from_pretrained("cvssp/audioldm-s-full-v2", torch_dtype=dtype,
                                       cache_dir=str(DATA_DIR / "hf_cache")).to(device)
t0 = time.perf_counter()
audio = np.asarray(ldm(PROMPT, num_inference_steps=50, audio_length_in_s=5.0).audios[0])
gen_s = time.perf_counter() - t0
results["audioldm"] = {"rtf": gen_s / (len(audio) / 16000), "clap": clap_score(PROMPT, audio, 16000)}
del ldm
free_memory()

for name, r in results.items():
    print(f"{name:12s} RTF {r['rtf']:6.2f}  CLAP {r['clap']:.3f}")
vram("after benchmark")
```

``` python
names = list(results)
bar_chart(
    "Text-to-Audio: prompt adherence (CLAP, higher better) and RTF",
    names,
    {"CLAP score": [results[n]["clap"] for n in names],
     "RTF": [results[n]["rtf"] for n in names]},
    y_name="score",
)
```

## 11. Common Frameworks

This task is split down the middle by library rather than by model family: the codec language models (MusicGen, AudioGen) live in `transformers` and `audiocraft`, and the latent diffusion models (AudioLDM, Stable Audio Open) live in `diffusers`. Both are general-purpose Hugging Face libraries, so both are allowed here - but the split is real, and it decides which knobs you get. A diffusion model gives you schedulers and step counts; a codec LM gives you sampling temperature and continuation.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [transformers](https://github.com/huggingface/transformers) | modelling | MusicGen (mono and stereo) with melody conditioning, through the standard generate API | Apache 2.0 | Music from a text prompt, or continuing an existing clip. Note the **CC-BY-NC** weights |
| [diffusers](https://github.com/huggingface/diffusers) | modelling | `AudioLDMPipeline`, `AudioLDM2Pipeline`, `StableAudioPipeline` - latent diffusion over a mel or waveform VAE | Apache 2.0 | Sound effects and high-fidelity stereo. Where you want the step-count and scheduler trade shown in section 10 |
| [audiocraft](https://github.com/facebookresearch/audiocraft) | modelling | Meta’s own runtime: AudioGen, MusicGen training and fine-tuning, EnCodec | MIT code, **CC-BY-NC** weights | You need AudioGen for dense sound scenes, or you are fine-tuning MusicGen on your own catalogue |
| [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) | modelling | The training stack behind Stable Audio Open - DiT, autoencoder, conditioning | MIT (weights: Stability Community License) | Training or fine-tuning a 44.1 kHz stereo generator, which `diffusers` alone will not do |
| [torchaudio](https://github.com/pytorch/audio) + [soundfile](https://github.com/bastibe/python-soundfile) + [pedalboard](https://github.com/spotify/pedalboard) | data | Resampling, loudness normalisation to a broadcast target, and the effects chain that makes generated audio sit in a mix | BSD-2 / BSD-3 / GPL-3.0 (pedalboard) | Always for the first two. Generated audio arrives at inconsistent loudness and clips on export more often than it should |
| `torch.compile` + [xformers](https://github.com/facebookresearch/xformers) / SDPA | inference runtime | Fused attention and a compiled UNet/DiT - the cheap 1.5-2x on the same card | BSD-3 | Generation is the product and every second of wall clock is user-visible. Cheaper than any model change |
| [BentoML](https://github.com/bentoml/BentoML) / [Ray Serve](https://github.com/ray-project/ray) | serving | Queue-backed async jobs, which is the right shape when one request takes 30 s and the client polls | Apache 2.0 | You are serving generation to users. A synchronous HTTP handler will time out |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | orchestration | A node graph over the same `diffusers` weights: prompt, generate, post-process, loop - without writing the glue | GPL-3.0 | Iterating on sound design by hand, or handing the controls to someone who does not write Python |
| [audioldm_eval](https://github.com/haoheliu/audioldm_eval) + a CLAP score | evaluation | FAD, KL and inception score against AudioCaps/MusicCaps, plus CLAP text-audio similarity for “does it match the prompt” | MIT / Apache 2.0 | Comparing checkpoints. FAD alone rewards fidelity and says nothing about whether the prompt was followed - report both |

**The 2026 default stack** is `diffusers` for sound effects and `transformers` for music, with `pedalboard` or ffmpeg doing loudness normalisation on the way out and a CLAP score gating what you keep. Reach past that only for a specific reason: `audiocraft` for AudioGen or fine-tuning, `stable-audio-tools` for 44.1 kHz stereo training.

The common wrong turn is licensing. MusicGen and AudioGen weights are **CC-BY-NC**: fine for this notebook, not fine in a product, and the code license being MIT does not change that. Stable Audio Open’s Stability Community License has its own revenue threshold. Check the weight license separately from the library license, every time.

------------------------------------------------------------------------

## 12. Going Further

- **Higher fidelity.** `stabilityai/stable-audio-open-1.0` (diffusers `StableAudioPipeline`, gated) generates 44.1 kHz stereo up to ~47 s - the best open SFX/loop quality available.
- **Melody conditioning.** `facebook/musicgen-melody` continues or re-harmonizes a reference melody, which turns the model from a slot machine into an instrument.
- **Longer / structured music.** MusicGen supports sliding-window continuation, but coherence over minutes is still the open problem; commercial Suno / Udio lead on song structure.
- **Prompting is the biggest free win.** Genre, instrumentation, tempo, recording style and mood in one sentence beats a bare noun phrase by more than a model upgrade does. Keep a prompt log next to your outputs.
- **Related notebooks.** `00_Text_to_Speech` (generated audio that has a reference transcript to be faithful to), `04_Audio_Classification` (CLAP again, used to score rather than to generate), `Computer_Vision/04_Text_to_Image` (the same latent-diffusion machinery, one modality over).

------------------------------------------------------------------------

Back to top
