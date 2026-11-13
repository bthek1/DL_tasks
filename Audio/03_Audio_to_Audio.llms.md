# Audio-to-Audio

Transforming audio into audio - source separation, speech enhancement, voice conversion: how the models work, the mid-2026 landscape, evaluation, and runnable code on GPU or CPU.

Author

Benedict Thekkel

## 1. What is Audio-to-Audio?

Audio-to-Audio covers any task whose **input and output are both waveforms**. The main sub-tasks:

- **Source separation** - split a mixture into sources: music **stem separation** (vocals/drums/bass/other) or **speech separation** (the cocktail-party problem).
- **Speech enhancement / denoising** - remove noise and reverberation to recover clean speech.
- **Voice conversion (VC)** - keep the words, change the speaker’s voice.
- **Bandwidth extension / restoration** - upsample or repair degraded audio.

**Neighbouring tasks:** ASR (often runs enhancement first), speaker diarization, and TTS (voice conversion overlaps with cloning).

Note: `transformers` has **no** `audio-to-audio` pipeline. These tasks use general-purpose audio toolkits - `torchaudio`, `speechbrain`, `asteroid` - which is what we use below.

------------------------------------------------------------------------

## 2. Real-World Use Cases

Audio-to-audio spans enhancement, separation and voice conversion. The single question that decides the model is whether it runs **inside a live call** (causal, milliseconds, one CPU core) or **over a finished recording** (offline, full lookahead, GPU) - those are different model families wearing the same task name.

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Noise suppression on calls | Video conferencing (Krisp, NVIDIA Broadcast, Zoom, Teams) | Noisy mic stream -\> clean speech, live | Causal, ~10-20 ms latency, small CPU budget while a call is running |
| Music stem separation | Music production, karaoke, DJ tools (Demucs, Moises, Serato) | Mixed track -\> vocals/drums/bass/other stems | Separation quality (SDR) and low bleed; offline is acceptable |
| Dialogue isolation and restoration | Film, TV, archive post-production (iZotope RX) | Archival or noisy mix -\> clean dialogue stem | Artifact-free output; quality dominates, speed is irrelevant |
| Hearing aids | Medical devices | Mic -\> enhanced speech, always on | Sub-10 ms latency at milliwatts on a DSP - the hardest budget in audio |
| Speech-to-speech translation, dubbing | Communication, media localisation | Source speech -\> target-language speech in the same voice | Voice preservation, latency, and performer consent |
| ASR pre-processing | Telecom, transcription pipelines | Noisy audio -\> enhanced audio for the recogniser | Must lower downstream WER, not just raise PESQ |

**What SDR and PESQ hide.** The most important trap: **enhancement that scores better can make downstream ASR worse.** Aggressive denoising distorts speech and clips consonants, and a recogniser trained on noisy audio often prefers the noisy input - if a separator feeds an ASR system, evaluate it on WER, not on a signal metric. The **latency budget** is the other deployment cliff: Demucs is non-causal and wants the whole track, so it can never run in a live call, while a conferencing denoiser must be causal with a lookahead measured in milliseconds. The failure modes are perceptual rather than numeric - musical noise, bleed between stems, over-suppression that eats the start of words, and pumping when the noise floor moves - and none of them show up cleanly in an SDR average over a clean benchmark like MUSDB.

------------------------------------------------------------------------

## 3. How Modern Audio-to-Audio Works

**Source separation.** 1. **Time-frequency masking (Conv-TasNet, 2019).** Learn a mask over a learned or STFT representation to isolate each source. 2. **Hybrid Demucs / HDemucs (2021).** Combine a time-domain and a spectrogram branch - the strong open baseline for **music stems**. 3. **Band-Split RNN / BS-RoFormer (2023-2024).** Split the spectrum into sub-bands and model each - current **state of the art** for music separation. 4. **SepFormer (2021).** A dual-path transformer - strong for **speech** separation.

**Speech enhancement.** Spectral masking -\> **MetricGAN+** (2021, optimizes PESQ directly) -\> **DeepFilterNet** (2022-2023, real-time, low-latency) -\> diffusion enhancement (**SGMSE**, 2023).

**Voice conversion.** kNN-VC and **FreeVC** (2023) -\> **Seed-VC** (2024, strong zero-shot). By 2025-2026 the frontier is real-time on-device enhancement and unified restoration models.

------------------------------------------------------------------------

## 4. Evaluation Metrics

- **SI-SDR / SI-SDRi (dB).** Scale-invariant signal-to-distortion ratio, and its **improvement** over the input mixture. The core separation/enhancement metric - higher is better.
- **PESQ (1.0-4.5).** Perceptual speech quality (ITU-T), for enhancement.
- **STOI (0-1).** Short-Time Objective Intelligibility.
- **DNSMOS.** A no-reference predicted MOS (no clean reference needed) - the DNS Challenge standard.
- **Voice conversion:** speaker **SIM**, plus ASR **WER** for intelligibility.

The cell computes **SI-SDR** with a tiny NumPy implementation (also available in `torchmetrics.audio`).

------------------------------------------------------------------------

``` python
import numpy as np


def si_sdr(estimate, reference, eps=1e-8):
    "Scale-invariant SDR in dB. Higher is better. Inputs are 1-D float arrays."
    estimate = np.asarray(estimate, dtype="float64")
    reference = np.asarray(reference, dtype="float64")
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + eps)
    target = alpha * reference
    noise = estimate - target
    return 10 * np.log10((np.dot(target, target) + eps) / (np.dot(noise, noise) + eps))


ref = np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)  # a clean tone
noisy = ref + 0.3 * np.random.randn(16000)                # + noise
print(f"SI-SDR noisy vs clean: {si_sdr(noisy, ref):.1f} dB")
```

    SI-SDR noisy vs clean: 7.4 dB

## 5. Datasets

Audio-to-audio ground truth is the clean source *before* it was mixed, so almost every dataset is built by mixing: recorded stems are summed into a song, or clean speech is added to recorded noise at a chosen SNR. This gives exact references for SI-SDR, but the mixtures are artificial. The typical failure is a model that scores well on synthetic mixes and then leaves artefacts or “musical noise” on real recordings, where reverb, compression and microphone colour were never in the training mixes.

| Dataset | Contents | Size | Scope | License | Typical use |
|----|----|----|----|----|----|
| [MUSDB18-HQ](https://zenodo.org/records/3338373) | Full songs with mixture + vocals, drums, bass, other stems, 44.1 kHz stereo WAV | 150 tracks (100 train / 50 test), 22.7 GB | Music, 4 stems | Non-commercial (educational use; sources include CC BY-NC-SA tracks) | **The** music separation benchmark (SDR per stem) |
| [VoiceBank+DEMAND](https://huggingface.co/datasets/JacobLinCool/VoiceBank-DEMAND-16k) ([official](https://datashare.ed.ac.uk/handle/10283/2791)) | Clean VCTK speech with added DEMAND and other noises; parallel clean/noisy pairs | 11,572 train / 824 test pairs (28-speaker set); 48 kHz original, 16 kHz mirror | Speech enhancement, English | CC BY 4.0 | The classic enhancement benchmark (PESQ, STOI) |
| [DNS Challenge](https://github.com/microsoft/DNS-Challenge) | Clean speech, noise and room impulse responses for on-the-fly mixing | ~1 TB unpacked (827 GB clean speech, 58 GB noise) | Speech enhancement, several languages | Per component (public domain, CC BY 4.0, CC0, CC BY-SA 3.0, ODbL …) | Large-scale enhancement training; blind test sets |
| [Libri2Mix / Libri3Mix](https://github.com/JorisCos/LibriMix) | LibriSpeech utterances mixed, optionally with WHAM! noise | Libri2Mix: 212 h train-360 / 58 h train-100 / 11 h dev / 11 h test | Speech separation, English | Scripts MIT; LibriSpeech CC BY 4.0; WHAM! noise CC BY-NC 4.0 | Open speech-separation benchmark |
| [WHAM!](https://wham.whisper.ai/) | Urban noise recordings (cafes, bars, parks) added to wsj0-2mix | Noise: 58 h train / 14.7 h dev / 9 h test | Noisy speech separation | Noise CC BY-NC 4.0; WSJ0 speech is LDC proprietary | Noisy separation eval (needs WSJ0) |
| [MUSAN](https://www.openslr.org/17/) | Music, speech and noise recordings, 16 kHz | ~109 h total, 11 GB | Noise and music for augmentation | CC BY 4.0 | Noise source for building training mixes |

**This notebook uses no benchmark dataset.** The Demucs demo separates one short music clip, `hdemucs_mix.wav`, downloaded from the torchaudio tutorial assets. The torchaudio tutorial names its test song as “A Classic Education” by NightOwl from MedleyDB (CC BY-NC-SA 4.0). The benchmark cell then scores an SI-SDR harness on two synthetic sine-wave “stems” with a placeholder separator. That proves the scoring code, not a model: plug in MUSDB18-HQ test tracks for real per-stem numbers.

None of these are gated on the Hub, but several are restricted. **MUSDB18-HQ is non-commercial**, **WHAM! noise is CC BY-NC 4.0** (so noisy LibriMix inherits it), and WHAM!’s speech comes from WSJ0, which needs a paid LDC license. DNS Challenge data keeps each source corpus’s own terms.

Downloads land in `DL_tasks/datasets/` via `cache_dir` (gitignored).

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

| Model | Task | License | Library | Best for |
|----|----|----|----|----|
| torchaudio HDEMUCS_HIGH_MUSDB_PLUS | music 4-stem separation | MIT | torchaudio | vocals/drums/bass/other |
| BS-RoFormer | music separation (SOTA) | MIT | external repo | best-quality stems |
| speechbrain/sepformer-wham | 2-speaker separation + denoise | Apache 2.0 | speechbrain | cocktail-party speech |
| speechbrain/metricgan-plus-voicebank | speech enhancement | Apache 2.0 | speechbrain | denoising |
| DeepFilterNet3 | real-time enhancement | Apache 2.0 / MIT | df (external) | low-latency on-device denoise |

Benchmarks: **MUSDB18** / the Sound Demixing (SDX) Challenge for music separation; the **DNS Challenge** for enhancement. We use **`torchaudio`** below (Demucs ships in its `pipelines`) - the most general-purpose option and part of the torch ecosystem.

------------------------------------------------------------------------

## 7. Setup

Package roles:

- `torchaudio` - the bundled Hybrid Demucs music-separation model
- `soundfile` - I/O; `numpy` - SI-SDR
- `datasets` - a sample clip to mix and separate
- `pyecharts` - the per-stem quality chart

`torchaudio` is the one extra dependency (general-purpose, torch ecosystem). `speechbrain` / DeepFilterNet are optional and shown in prose only.

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

OUT_DIR = DATA_DIR / "a2a_out"
OUT_DIR.mkdir(exist_ok=True)

# A short music clip to separate. Swap in any stereo track; MUSDB18 for real eval.
SAMPLE = DATA_DIR / "music_sample.wav"
if not SAMPLE.exists():
    urllib.request.urlretrieve(
        "https://download.pytorch.org/torchaudio/tutorial-assets/hdemucs_mix.wav", SAMPLE
    )
```

    NVIDIA GeForce RTX 3060
    device: cuda:0

## 8. Music source separation with Demucs (torchaudio)

`torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS` bundles Hybrid Demucs trained on MUSDB18 + 800 extra songs. It splits a stereo mixture into **drums, bass, other, vocals**. We chunk the audio to bound memory, then write each stem.

------------------------------------------------------------------------

``` python
import torchaudio
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS

bundle = HDEMUCS_HIGH_MUSDB_PLUS
model = bundle.get_model().to(device)
sr = bundle.sample_rate  # 44100

mix, in_sr = torchaudio.load(SAMPLE)
if in_sr != sr:
    mix = torchaudio.functional.resample(mix, in_sr, sr)
mix = mix.to(device)

t0 = time.perf_counter()
with torch.no_grad():
    sources = model(mix.unsqueeze(0))[0]  # (n_sources, channels, time)
print(f"{time.perf_counter() - t0:.1f}s  sources: {model.sources}")

for name, stem in zip(model.sources, sources):
    sf.write(OUT_DIR / f"{name}.wav", stem.cpu().T.numpy(), sr)
print("stems written to", OUT_DIR)

del model, sources, mix
free_memory()
vram("after demucs")
```

    2.1s  sources: ['drums', 'bass', 'other', 'vocals']
    stems written to ../../datasets/a2a_out
    VRAM after demucs      0.25 GB allocated /  0.61 GB reserved

## 9. Head-to-head Benchmark

A true separation benchmark needs **ground-truth stems** (MUSDB18) to compute **SI-SDRi** per stem. Here we illustrate the harness: mix known stems, separate, and score each stem with SI-SDR, charting the result. Plug MUSDB18 in for real numbers; treat this synthetic mix as a smoke test.

------------------------------------------------------------------------

``` python
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org serves an expired TLS certificate from most nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"

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
# Illustrative: build a mixture from two synthetic "stems", then score how well a
# trivial half-wave split recovers them. Replace `separate` with a real model and
# these placeholders with MUSDB18 stems for a genuine SI-SDRi benchmark.
t = np.arange(3 * 16000) / 16000
stems = {"tone_a": np.sin(2 * np.pi * 220 * t), "tone_b": 0.6 * np.sin(2 * np.pi * 440 * t)}
mixture = sum(stems.values())


def separate(mix):  # stand-in estimator; swap for Demucs / BS-RoFormer output
    return {k: mix * 0.5 for k in stems}


est = separate(mixture)
scores = {name: si_sdr(est[name], ref) for name, ref in stems.items()}
for name, s in scores.items():
    print(f"{name:10s} SI-SDR {s:6.1f} dB")

bar_chart("Source separation: per-stem SI-SDR (dB, higher better)",
          list(scores), {"SI-SDR (dB)": list(scores.values())}, y_name="dB")
```

    tone_a     SI-SDR    4.4 dB
    tone_b     SI-SDR   -4.4 dB

## 10. Common Frameworks

Audio-to-audio is the one task in this folder where `transformers` is *not* the centre of gravity. Separation and enhancement models predate the transformers ecosystem, they are trained with per-recipe toolkits, and the best-quality checkpoints often live in a single researcher’s repo with a `requirements.txt` and no Hub entry. `torchaudio` is the general-purpose anchor - Demucs ships inside it - and everything else is a specialisation.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [torchaudio](https://github.com/pytorch/audio) | modelling | Hybrid Demucs as a bundled pipeline, plus the STFT/iSTFT primitives every model here is built on | BSD-2 | Default. Four-stem music separation with no extra dependency and no repo to clone |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) | modelling | SepFormer separation, MetricGAN+ enhancement, and the recipes to retrain either | Apache 2.0 | Speech rather than music: cocktail-party separation, denoising, and anything you need to fine-tune |
| [Asteroid](https://github.com/asteroid-team/asteroid) | modelling | The source-separation research toolkit - Conv-TasNet, DPRNN, DPTNet, with matched datasets and losses | MIT | You are training a separator, and want an SI-SDR loss and a data pipeline that already exist |
| [Demucs](https://github.com/adefossez/demucs) / BS-RoFormer via [UVR](https://github.com/Anjok07/ultimatevocalremovergui) | modelling | The checkpoints that actually top MUSDB18, and a GUI that batch-processes a folder of tracks | MIT | Stem quality is the product. These beat the bundled Demucs, at the cost of a repo clone rather than a `pip install` |
| ffmpeg + [soundfile](https://github.com/bastibe/python-soundfile) + [librosa](https://github.com/librosa/librosa) | data | Decode anything, resample, and write stems back to the container the DAW expects | LGPL / BSD-3 / ISC | Always. Music arrives at 44.1 kHz stereo and speech models want 16 kHz mono - the conversion is where most bugs live |
| [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) | inference runtime | Real-time full-band speech enhancement, Rust core, runs faster than realtime on one CPU core | Apache 2.0 / MIT | Denoising inside a live call. The quality models above are all far too slow for that |
| [ONNX Runtime](https://onnxruntime.ai/) | inference runtime | The small enhancers exported once and run in C++, on mobile, or in the browser | MIT | The denoiser has to ship inside a client application rather than a server |
| [BentoML](https://github.com/bentoml/BentoML) / [Ray Serve](https://github.com/ray-project/ray) | serving | Async job queues, which is the right shape when separating one album takes minutes | Apache 2.0 | Separation is a batch product - upload, wait, download stems |
| [museval](https://github.com/sigsep/sigsep-mus-eval) + [torchmetrics](https://github.com/Lightning-AI/torchmetrics) + DNSMOS | evaluation | SDR/SI-SDR against MUSDB18 references, and a reference-free MOS predictor for enhancement | MIT / Apache 2.0 | Comparing models. SI-SDR needs a ground-truth stem; DNSMOS does not, which is why it is the one that works on your own recordings |

**The 2026 default stack** is `torchaudio` for music stems, SpeechBrain or DeepFilterNet for speech depending on whether latency matters, and ffmpeg on both ends. Move to UVR/BS-RoFormer when a listener will hear the difference, and to Asteroid only when you are training.

The common wrong turn is evaluating enhancement with SI-SDR on real recordings, where there is no clean reference to compare against - the number you get is measuring your simulated mixture, not your product. The second is running a separator at the wrong sample rate: these models are sample-rate specific in a way that degrades quietly rather than erroring.

------------------------------------------------------------------------

## 11. Going Further

- **Voice conversion.** **Seed-VC** (zero-shot) and **FreeVC** change speaker identity while keeping the content and prosody - the same input/output shape as everything above, a different target.
- **As ASR pre-processing.** Run enhancement before Whisper/Parakeet on noisy audio to cut WER. Measure it: on already-clean audio, enhancement introduces artefacts and makes WER *worse*, so this is an A/B to run rather than a default to assume.
- **Fine-tuning on your own noise.** Separation and enhancement models generalise poorly across recording conditions. A few hours of your own domain - the actual room, the actual microphone - moves the numbers more than a bigger architecture does.
- **Bandwidth extension and restoration.** Upsampling 8 kHz telephony to wideband, and declipping or dehumming archival audio, are the same conditional-generation shape and reuse the same toolkits.
- **Related notebooks.** `02_Automatic_Speech_Recognition` (the usual consumer of enhanced audio), `05_Voice_Activity_Detection` (the cheaper way to handle silence), `01_Text_to_Audio` (generating audio instead of repairing it).

------------------------------------------------------------------------

Back to top
