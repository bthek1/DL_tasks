# Text-to-Speech

Everything to know about speech synthesis: how neural TTS works, the mid-2026 open-model landscape, how to evaluate it, and runnable transformers code to test the leading models on GPU or CPU.

Author

Benedict Thekkel

## 1. What is Text-to-Speech?

Text-to-Speech (TTS, speech synthesis) maps written text to a spoken-audio waveform.

**Input.** A text string. Modern systems optionally also take a **speaker reference** (a few seconds of a target voice for zero-shot cloning), a **style/emotion prompt**, or SSML-style controls (rate, pitch, pauses).

**Output.** A mono waveform, typically **16-24 kHz** (some models 44.1 kHz). Often accompanied by alignment/duration info.

**Neighbouring tasks** (separate models, often chained):

| Task | What it does | Typical tool |
|----|----|----|
| ASR / STT | The inverse: speech to text | Whisper, Parakeet |
| Voice conversion | Re-voice existing speech to a target speaker | Seed-VC, kNN-VC |
| Speech-to-speech translation | Speak in another language | SeamlessM4T |
| Vocoder | Mel-spectrogram to waveform | HiFi-GAN, BigVGAN |

------------------------------------------------------------------------

## 2. Real-World Use Cases

TTS is the output half of every voice interface, so the deployments span from a microcontroller in a car to a studio rendering an audiobook - and they pull the model choice in opposite directions.

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Screen readers, accessibility | Assistive tech (VoiceOver, NVDA, TalkBack) | UI text -\> streamed speech, often 2-3x speed | Latency to first audio, tiny on-device model, works offline |
| Voice assistants and IVR | Consumer, telecom contact centres | Dialogue-manager text -\> 8-16 kHz speech | Latency (barge-in), cost per minute at call-centre scale |
| Audiobook and podcast narration | Publishing, media (Apple Books digital narration) | Long-form manuscript -\> studio-quality audio | Prosody and voice consistency across hours; quality over speed |
| Game and film dialogue, dubbing | Entertainment | Script + speaker reference -\> in-character speech | Expressivity, voice-cloning fidelity, performer consent/licensing |
| In-car and embedded navigation | Automotive, IoT | Nav prompts -\> speech with no network | On-device size, deterministic latency, no cloud |
| Live speech translation | Communication (Meta Seamless-class systems) | Translated text -\> speech in the original speaker’s voice | End-to-end latency, preserving voice across languages |

**What the MOS score hides.** A high mean opinion score on LJSpeech tells you almost nothing about whether a model survives deployment. **Streaming vs batch** is an architectural fork: an assistant is judged on *time to first audio chunk*, an audiobook pipeline on total real-time factor, and a model can be excellent at one and useless at the other. **Edge vs server** is a hard size cliff - a VITS voice is tens of megabytes and runs on a phone CPU; a codec LM like Bark needs a GPU and cannot be the voice of a car. **The text front end matters more than the acoustic model** in production: numbers, dates, currency, acronyms, URLs and proper nouns all have to be normalised before synthesis, and most real bug reports are pronunciation failures, not naturalness failures. The failure modes that actually hurt are autoregressive ones - skipped or hallucinated words, prosody that drifts over long inputs, and unstable output on out-of-vocabulary names - because unlike a slightly robotic voice, they change the meaning. Voice cloning adds a non-technical constraint on top: consent, and increasingly watermarking, are deployment blockers regardless of quality.

------------------------------------------------------------------------

## 3. How Modern TTS Works

Four generations, each still in use somewhere:

1.  **Concatenative / parametric (pre-2016).** Stitch recorded units or drive an HMM vocoder. Robotic; legacy only.
2.  **Neural acoustic model + neural vocoder (Tacotron 2, 2017).** Two stages: an attention seq2seq predicts a **mel-spectrogram** from text, then a vocoder (WaveNet -\> HiFi-GAN) turns mel into a waveform. FastSpeech 2 (2020) replaced the slow autoregressive acoustic model with a fast non-autoregressive one driven by explicit duration/pitch/energy.
3.  **End-to-end (VITS, 2021).** A single model goes text -\> waveform using a conditional VAE with normalizing flows plus adversarial training - no separate vocoder. Meta’s **MMS-TTS** scales VITS to 1100+ languages with one checkpoint per language.
4.  **Codec-LM and diffusion TTS (2023-2026).** Discretize audio into neural-codec tokens (EnCodec / DAC) and predict them with an autoregressive language model (**Bark**, VALL-E, XTTS, Parler-TTS, Orpheus). This unlocks **zero-shot voice cloning** and emotion/style from a short prompt. In parallel, flow-matching / diffusion systems (**StyleTTS 2**, **F5-TTS**, 2024-2025) match codec-LM quality with faster, more stable synthesis. As of mid-2026 the small flow/codec models (Kokoro-82M, F5-TTS) dominate the open quality-per-compute frontier.

------------------------------------------------------------------------

## 4. Evaluation Metrics

TTS quality is inherently subjective, so evaluation blends human and automatic metrics.

- **MOS (Mean Opinion Score), 1-5.** Human naturalness rating - the gold standard. **UTMOS** and other predicted-MOS models give a cheap automatic proxy.
- **Intelligibility (WER/CER).** Transcribe the synthesized audio with a strong ASR model and compute error rate against the input text - an **ASR round-trip**. High WER means unclear speech.
- **Speaker similarity (SIM).** Cosine similarity between speaker-verification embeddings of the reference and the synthesized voice - the key metric for cloning.
- **MCD (Mel Cepstral Distortion).** Signal-level distance to a reference recording (dB, lower is better).
- **RTF (Real-Time Factor).** Synthesis time / audio duration. RTF \< 1 is faster than real time.

The cell below shows the ASR round-trip idea with `jiwer` (same WER tooling as the ASR notebook).

------------------------------------------------------------------------

``` python
import jiwer

# Suppose we synthesized this sentence and an ASR model transcribed it back:
target = "the quick brown fox jumps over the lazy dog"
asr_of_synth = "the quick brown fox jumps over the lazy dog"  # perfect intelligibility -> WER 0

print("round-trip WER:", jiwer.wer(target, asr_of_synth))
# In section 9 we run this for real: synthesize with each model, transcribe with
# whisper-tiny, and report the WER as an intelligibility score.
```

## 5. Datasets

TTS ground truth is a recording of a person reading a known script, so a model can only be as clean, as varied and as expressive as those readings. Audiobook corpora give scale but carry the narrator’s room, microphone and flat reading voice into the model; studio corpora sound clean but cover one or a few speakers in one reading style. The typical failure is a voice that is perfect on read sentences and wooden or unstable on conversational text, numbers and names it never saw spoken.

| Dataset | Contents | Size | Scope | License | Typical use |
|----|----|----|----|----|----|
| [LJSpeech](https://huggingface.co/datasets/keithito/lj_speech) | One female speaker reading 7 non-fiction books, 22.05 kHz | 13,100 clips, ~24 h | English, single speaker | Public domain | The classic single-speaker training and demo set |
| [LibriTTS](https://huggingface.co/datasets/mythicinfinity/libritts) | LibriVox audiobooks re-cut at sentence breaks, 24 kHz | ~585 h, 2,456 speakers | English, multi-speaker | CC BY 4.0 | Multi-speaker training |
| [LibriTTS-R](https://huggingface.co/datasets/mythicinfinity/libritts_r) | The same utterances as LibriTTS after speech restoration | ~585 h, 2,456 speakers | English, multi-speaker | CC BY 4.0 | Drop-in, cleaner replacement for LibriTTS |
| [VCTK](https://huggingface.co/datasets/CSTR-Edinburgh/vctk) | Studio recordings, ~400 sentences per speaker, 48 kHz | ~44 h, 110 speakers | English, many accents | CC BY 4.0 | Multi-speaker and accent training; cloning eval |
| [Hi-Fi TTS](https://huggingface.co/datasets/MikhailT/hifi-tts) | LibriVox audiobooks selected for audio quality, 44.1 kHz | ~292 h, 10 speakers (\>= 17 h each) | English | CC BY 4.0 | High-sample-rate training |
| [MLS](https://huggingface.co/datasets/facebook/multilingual_librispeech) | LibriVox audiobooks in 8 languages | ~44.5K h English + ~6K h other | 8 European languages | CC BY 4.0 | Large-scale multilingual pretraining |
| [Emilia](https://huggingface.co/datasets/amphion/Emilia-Dataset) | In-the-wild speech (talk shows, debates, podcasts), auto-transcribed | ~101.7K h (Emilia) + ~113.9K h (Emilia-YODAS) | 6 languages | Emilia: CC BY-NC 4.0; Emilia-YODAS: CC BY 4.0 | Spontaneous, conversational training for zero-shot TTS |
| [CMU ARCTIC x-vectors](https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors) | 512-d speaker embeddings extracted from CMU ARCTIC utterances | 7,931 vectors, 7 speakers | English | MIT | SpeechT5 speaker conditioning; **used in this notebook** |

**This notebook uses no evaluation dataset.** It synthesises one hand-written sentence with each model and scores intelligibility with an ASR round-trip through `whisper-tiny`. The only data it fetches is a single x-vector (row 7306) from `Matthijs/cmu-arctic-xvectors`, pulled through the HF datasets-server rows API because that dataset’s loader script no longer runs under `datasets >= 4`. One sentence is a smoke test: a real comparison needs a held-out text set and human or predicted MOS.

**Emilia is gated**: you must accept its terms on the Hub and use a token. The original Emilia portion is **non-commercial only**, and the terms bind your employer if you work for a for-profit company; only the Emilia-YODAS portion is CC BY 4.0. The other datasets above are not gated.

Downloads land in `DL_tasks/datasets/` via `cache_dir` (gitignored).

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

There is no single WER leaderboard for TTS (output is generated, not matched to a reference), so ranking is done by human preference. The **[TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena)** (crowd-voted MOS) is the reference.

| Model | Params | License | Scope | Approach | Best for |
|----|----|----|----|----|----|
| microsoft/speecht5_tts | 140M | MIT | en | Transformer + HiFi-GAN | controllable via speaker embeddings |
| facebook/mms-tts-eng (VITS) | 36M | CC-BY-NC 4.0 | 1100+ langs | end-to-end VITS | massively multilingual, tiny, fast |
| suno/bark | 1B | MIT | multilingual | codec LM | expressive prosody, laughs/sfx, prompts |
| hexgrad/Kokoro-82M | 82M | Apache 2.0 | en + more | StyleTTS 2 decoder + ISTFTNet (external `kokoro`) | best small-model quality |
| parler-tts/parler-tts-mini-v1 | 880M | Apache 2.0 | en | codec LM (external) | describe the voice in a text prompt |
| coqui/XTTS-v2 | 460M | CPML | 17 langs | codec LM (external `TTS`) | zero-shot cloning from 6 s |

The first three load through Hugging Face `transformers` and run below. Kokoro / Parler / XTTS need their own runtimes and are covered in “Going further”. [Piper vs Kokoro](../Frameworks/piper_vs_kokoro.llms.md) runs Kokoro on plain ONNX Runtime and compares it with Piper.

------------------------------------------------------------------------

## 7. Setup

Everything below runs on a single modest GPU or on CPU, and **every runnable model loads through `transformers`** - no per-model packages. Package roles:

- `transformers` (\>=5.13) + `torch` - SpeechT5, VITS/MMS, Bark
- `datasets` - speaker-embedding vectors (SpeechT5) and the round-trip ASR eval clips
- `soundfile` - write WAV output
- `jiwer` - round-trip WER
- `pyecharts` - the benchmark chart

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

import numpy as np
import soundfile as sf

TEXT = "Hugging Face transformers makes speech synthesis a one liner."
OUT_DIR = DATA_DIR / "tts_out"
OUT_DIR.mkdir(exist_ok=True)


def save_wav(name, audio, sr):
    "Write a float array to DL_tasks/datasets/tts_out and report duration."
    path = OUT_DIR / name
    sf.write(path, audio, sr)
    print(f"{name}: {len(audio) / sr:.2f} s @ {sr} Hz -> {path}")
    return path
```

## 8. Three transformers-native voices

### SpeechT5 - controllable, speaker-embedding driven

SpeechT5 needs an x-vector **speaker embedding** (512-d) alongside the text; swapping the embedding swaps the voice. The `Matthijs/cmu-arctic-xvectors` dataset ships ready-made ones.

------------------------------------------------------------------------

``` python
import json
import urllib.parse

from transformers import pipeline

# SpeechT5 needs a 512-d x-vector speaker embedding. The classic
# Matthijs/cmu-arctic-xvectors dataset ships a loader script, which datasets>=4
# refuses to run - fetch one ready-made x-vector through the HF datasets-server
# rows API instead, cached locally so reruns work offline.
XVEC = DATA_DIR / "speecht5_xvector.json"
if not XVEC.exists():
    q = urllib.parse.urlencode({"dataset": "Matthijs/cmu-arctic-xvectors", "config": "default",
                                "split": "validation", "offset": 7306, "length": 1})
    with urllib.request.urlopen(f"https://datasets-server.huggingface.co/rows?{q}", timeout=60) as r:
        XVEC.write_text(json.dumps(json.load(r)["rows"][0]["row"]["xvector"]))
speaker_embedding = torch.tensor(json.loads(XVEC.read_text())).unsqueeze(0)

t5 = pipeline("text-to-speech", model="microsoft/speecht5_tts", device=device)
t0 = time.perf_counter()
out = t5(TEXT, forward_params={"speaker_embeddings": speaker_embedding})
print(f"{time.perf_counter() - t0:.2f}s")
save_wav("speecht5.wav", out["audio"], out["sampling_rate"])

del t5
free_memory()
vram("after speecht5")
```

### MMS / VITS - tiny, end-to-end, multilingual

`facebook/mms-tts-eng` is a 36M-param VITS model: text straight to waveform, no speaker embedding, no vocoder step. Swap the language code (e.g. `mms-tts-fra`, `mms-tts-deu`) for other languages.

------------------------------------------------------------------------

``` python
vits = pipeline("text-to-speech", model="facebook/mms-tts-eng", device=device)
t0 = time.perf_counter()
out = vits(TEXT)
print(f"{time.perf_counter() - t0:.2f}s")
save_wav("mms_vits.wav", out["audio"].squeeze(), out["sampling_rate"])

del vits
free_memory()
vram("after vits")
```

### Bark - expressive codec LM

Bark predicts EnCodec tokens with a GPT-style LM, so it can produce laughter, sighs and music cues from inline tags like `[laughs]`. `suno/bark-small` fits comfortably on a 6 GB card; use voice presets (e.g. `v2/en_speaker_6`) for a consistent speaker.

------------------------------------------------------------------------

``` python
bark = pipeline("text-to-speech", model="suno/bark-small", device=device)
t0 = time.perf_counter()
out = bark(TEXT, forward_params={"do_sample": True})
print(f"{time.perf_counter() - t0:.2f}s")
save_wav("bark.wav", out["audio"].squeeze(), out["sampling_rate"])

del bark
free_memory()
vram("after bark")
```

## 9. Head-to-head Benchmark

Synthesize the **same sentence** with each model, then measure two things per model: **RTF** (synthesis time / output duration, lower is faster) and **round-trip WER** (transcribe the output with `whisper-tiny` and compare to the input text, lower is more intelligible). Caveat: one sentence is a smoke test, not a MOS study - real evaluation needs human listening or the TTS Arena.

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
import jiwer
import librosa

norm = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                      jiwer.RemoveMultipleSpaces(), jiwer.Strip()])

# whisper-tiny is our round-trip transcriber (loaded once, reused, then freed)
asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=device)

VOICES = {
    "speecht5": lambda: pipeline("text-to-speech", "microsoft/speecht5_tts", device=device),
    "mms-vits": lambda: pipeline("text-to-speech", "facebook/mms-tts-eng", device=device),
    "bark-small": lambda: pipeline("text-to-speech", "suno/bark-small", device=device),
}
extra = {"speecht5": {"forward_params": {"speaker_embeddings": speaker_embedding}}}

results = {}
for name, build in VOICES.items():
    tts_pipe = build()
    t0 = time.perf_counter()
    out = tts_pipe(TEXT, **extra.get(name, {}))
    gen_s = time.perf_counter() - t0
    audio = np.asarray(out["audio"]).squeeze().astype("float32")
    sr = out["sampling_rate"]
    rtf = gen_s / (len(audio) / sr)
    # whisper wants 16 kHz; resample with librosa so we do not depend on torchaudio
    audio16 = audio if sr == 16000 else librosa.resample(audio, orig_sr=sr, target_sr=16000)
    hyp = asr({"array": audio16, "sampling_rate": 16000})["text"]
    wer = jiwer.wer(norm(TEXT), norm(hyp))
    results[name] = {"rtf": rtf, "wer": wer}
    print(f"{name:12s} RTF {rtf:5.2f}  round-trip WER {wer:6.2%}")
    del tts_pipe  # free each voice before building the next so VRAM stays flat
    free_memory()

del asr
free_memory()
vram("after benchmark")
```

``` python
names = list(results)
bar_chart(
    "TTS: intelligibility vs speed (lower is better)",
    names,
    {"round-trip WER": [results[n]["wer"] for n in names],
     "RTF": [results[n]["rtf"] for n in names]},
    y_name="score",
)
```

## 10. Common Frameworks

The three voices above load through `transformers`, which is this repo’s rule and also the honest starting point. But TTS is the audio task where the ecosystem rule bites hardest: the models people actually judge best - Kokoro, Parler, XTTS, F5 - each ship their own runtime, because a codec LM plus a vocoder plus a G2P front-end is more machinery than a single `AutoModel` class wants to hold. This is the map of what that machinery is called.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [transformers](https://github.com/huggingface/transformers) | modelling | SpeechT5, VITS/MMS and Bark behind one `text-to-speech` pipeline, plus the speaker-embedding control shown above | Apache 2.0 | Default, and the only option here that composes cleanly with the rest of a `transformers` pipeline |
| [kokoro](https://github.com/hexgrad/kokoro) / [parler-tts](https://github.com/huggingface/parler-tts) / [F5-TTS](https://github.com/SWivid/F5-TTS) | modelling | The current small-model quality leaders, each as a thin single-purpose package | Apache 2.0 (F5-TTS code MIT) | Quality per parameter is the deciding constraint. Kokoro at 82M beats models ten times its size on the Arena |
| [Coqui TTS](https://github.com/coqui-ai/TTS) (`TTS`) | modelling | XTTS-v2 zero-shot cloning from ~6 s of reference audio, plus a large zoo of older recipes | MPL 2.0 (XTTS **weights are CPML - non-commercial**) | You need voice cloning and can live with the weight license. Check CPML before shipping, not after |
| [ESPnet](https://github.com/espnet/espnet) / [SpeechBrain](https://github.com/speechbrain/speechbrain) | modelling | Full training recipes: acoustic model, vocoder, G2P, and the data pipeline to fine-tune a voice from scratch | Apache 2.0 | You are building a voice rather than picking one - a few hours of studio audio and a target speaker |
| [torchaudio](https://github.com/pytorch/audio) + [soundfile](https://github.com/bastibe/python-soundfile) + ffmpeg | data | Resampling, loudness normalisation, encoding to whatever the client plays | BSD-2 / BSD-3 / LGPL | Always. Every model here emits a different native sample rate, and clients rarely accept raw float arrays |
| [Piper](https://github.com/OHF-Voice/piper1-gpl) | inference runtime | VITS voices compiled to ONNX, realtime on a Raspberry Pi with no GPU | GPL-3.0 (copyleft; each voice also has its own data licence) | On-device or offline synthesis - accessibility tools, kiosks, home assistants |
| [ONNX Runtime](https://onnxruntime.ai/) | inference runtime | The same weights exported once and run from Python, C++, mobile or the browser | MIT | The voice has to live inside an application that is not Python |
| [NVIDIA Riva](https://developer.nvidia.com/riva) / [Triton Inference Server](https://github.com/triton-inference-server/server) | serving | Streaming synthesis with first-chunk latency budgets, batching, autoscaling | proprietary (Riva) / BSD-3 (Triton) | You are serving many concurrent streams and need audio to start before the sentence is finished |
| [Pipecat](https://github.com/pipecat-ai/pipecat) / [LiveKit Agents](https://github.com/livekit/agents) | orchestration | The conversational loop TTS usually lives in: sentence chunking, playback, and interruption that actually stops the audio | BSD-2 / Apache 2.0 | TTS is the output leg of a voice agent. See `Frameworks/01_pipecat` |
| [UTMOS / SpeechMOS](https://github.com/tarepan/SpeechMOS) + a Whisper round-trip | evaluation | A predicted MOS score, and an intelligibility check by transcribing your own output and measuring WER against the input text | MIT | Always. There is no reference waveform to compare against, so these two proxies are what you have between human listening tests |

**The 2026 default stack** for a product is Kokoro or Piper for synthesis, `soundfile`/ffmpeg to get the bytes into the right container, and Pipecat if a human is going to interrupt it. `transformers` remains the right choice while you are still deciding, and SpeechT5 remains the right choice when you need per-speaker embedding control rather than maximum naturalness.

The common wrong turn is choosing a voice by MOS alone. Latency to the **first audio chunk** is what a caller perceives, and a model that scores half a point higher while taking two seconds to start will feel worse in a conversation than one that starts in two hundred milliseconds. The second wrong turn is shipping XTTS without reading CPML.

------------------------------------------------------------------------

## 11. Going Further

- **Fine-tuning.** SpeechT5 fine-tunes on a few hours of a target voice: [HF guide](https://huggingface.co/blog/fine-tune-speecht5-tts). The data is the hard part - consistent recording conditions matter more than quantity.
- **Streaming / real-time.** Synthesise sentence by sentence and start playback on the first chunk rather than waiting for the full utterance. A streaming codec LM (Orpheus) does this natively; with SpeechT5 you chunk on punctuation yourself.
- **Better vocoders.** Swap HiFi-GAN for BigVGAN for higher-fidelity mel-to-wave. This is a drop-in change at the end of the pipeline and often a bigger perceptual win than changing the acoustic model.
- **Voice cloning ethics and consent.** Zero-shot cloning from six seconds is a capability with an obvious misuse. Get recorded consent from the speaker, and consider watermarking output (`AudioSeal`) if the voice is identifiable.
- **Related notebooks.** `02_Automatic_Speech_Recognition` (the inverse task, and the round-trip intelligibility check above), `01_Text_to_Audio` (music and sound effects rather than speech), `Multimodal/08_Any_to_Any` (models that emit speech directly instead of calling a TTS).

------------------------------------------------------------------------

Back to top
