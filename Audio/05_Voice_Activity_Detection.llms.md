# Voice Activity Detection

Detecting where speech is - the front-end for ASR, diarization and streaming: how VAD works, the mid-2026 landscape, evaluation, and runnable code on GPU or CPU.

Author

Benedict Thekkel

## 1. What is Voice Activity Detection?

Voice Activity Detection (VAD) decides, frame by frame, whether **speech is present** - separating it from silence, noise and music.

**Input.** An audio stream (usually 16 kHz mono).

**Output.** Per-frame speech probability, collapsed into **speech segments** with start/end timestamps.

**Why it matters.** VAD gates and segments audio before heavier models: trim silence before ASR, chunk long recordings, detect end-of-utterance (endpointing) in streaming, and provide the speech regions for **diarization**.

**Neighbouring tasks:** audio classification (VAD is the binary speech/no-speech case), ASR, diarization, and sound-event detection.

Note: `transformers` has no dedicated VAD pipeline; the field standard is **Silero VAD** (via `torch.hub`), with `pyannote.audio` and `speechbrain` for segmentation.

------------------------------------------------------------------------

## 2. Real-World Use Cases

VAD is the cheapest model in the audio stack and the one everything else depends on: it runs on 100% of the incoming audio and decides what the expensive models ever get to see. It is almost never a product on its own - it is a gate.

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Front end of an ASR pipeline | Any speech stack | Continuous stream -\> speech segments to transcribe | Recall: speech it drops is lost forever downstream. Must be near-free to run |
| Turn-taking in voice agents | Conversational AI, phone bots, LLM voice agents | Mic stream -\> “has the user finished speaking?” | Endpointing latency: too eager interrupts the user, too slow feels laggy |
| Discontinuous transmission (DTX) | Telecom codecs, VoIP (WebRTC VAD) | Frames -\> transmit or suppress | Runs on a DSP at near-zero cost; standardised and causal |
| Always-on device gating | Wearables, smart speakers | Continuous mic -\> wake the expensive pipeline | Milliwatts, and privacy: nothing leaves the device until speech is detected |
| Silence trimming and dead-air removal | Podcasting, media editing | Long recording -\> speech regions | Precision (do not cut a soft word); offline, so accuracy over latency |
| Talk-time and compliance analytics | Contact centres | Call audio -\> talk/silence ratio, overlap | Robustness to hold music, IVR tones and ringback |
| Diarization front end | Meeting transcription | Stream -\> speech regions to cluster by speaker | Accurate boundaries: onset errors propagate into the wrong speaker |

**What a frame-level accuracy number hides.** VAD errors are **asymmetric**, and the benchmark metric treats them as symmetric. A missed speech onset clips the first phoneme of a word, and no downstream model can recover it; a false alarm costs a few milliseconds of wasted compute. Real systems therefore tune for recall and pad the segments (hangover), which no accuracy figure rewards. **Real audio is adversarial** in a way clean corpora are not - background TV, music, laughter, breathing, keyboard clicks and hold music all look like speech energy, and the gap between a quiet benchmark and a noisy cafe is where VADs die. The subtlest trap is that **endpointing is not the same task as VAD**: a 700 ms pause mid-sentence is a speaker thinking, not a speaker finishing, and a voice agent that treats “no-speech” as “your turn” will interrupt people constantly. Production endpointers therefore layer duration thresholds, and increasingly semantic signals, on top of the raw VAD. Whatever you pick has to be causal, streaming, and effectively free - Silero runs a 30 ms chunk in about a millisecond on one CPU core, which is the budget this task lives in.

------------------------------------------------------------------------

## 3. How Modern VAD Works

1.  **Signal thresholds (classic).** Energy and zero-crossing rate; **WebRTC VAD** (a GMM, still ubiquitous) - extremely light and fast, weak in noise.
2.  **DNN frame classifiers (2015+).** A small network over log-mel frames predicts speech probability.
3.  **Silero VAD (2020-2024).** A tiny (~1.8M param) CNN/RNN, ONNX-deployable, millisecond latency - the **de-facto open VAD**, robust across languages and noise.
4.  **Neural segmentation (pyannote, 2021-2024).** SincNet/transformer models that jointly do VAD, **overlapped-speech** detection and diarization.
5.  **Self-supervised (wav2vec2 / WavLM).** Fine-tuned for segmentation when maximum robustness is needed. By 2025-2026 Silero remains the practical default; pyannote leads multi-speaker segmentation.

------------------------------------------------------------------------

## 4. Evaluation Metrics

- **Frame-level accuracy / precision / recall / F1.** Treat each frame as a binary classification.
- **ROC-AUC.** Threshold-independent quality of the speech-probability score.
- **False-alarm and miss rates.** FA (non-speech marked speech) vs miss (speech marked non-speech) - the operating-point trade-off.
- **Detection latency.** For streaming endpointing.

The cell computes precision/recall/F1 with `scikit-learn` on toy frame labels.

------------------------------------------------------------------------

``` python
from sklearn.metrics import precision_recall_fscore_support

# frame labels: 1 = speech, 0 = non-speech
true = [0, 0, 1, 1, 1, 1, 0, 0, 1, 1]
pred = [0, 1, 1, 1, 1, 0, 0, 0, 1, 1]
p, r, f, _ = precision_recall_fscore_support(true, pred, average="binary")
print(f"precision {p:.2f}  recall {r:.2f}  F1 {f:.2f}")
```

    precision 0.83  recall 0.83  F1 0.83

## 5. Datasets

VAD ground truth is a set of start and end times for speech, and annotators disagree at the edges: where a breath, a laugh or a trailing word counts as speech, and how short a pause has to be before it counts as silence. Most labels come as a side effect of diarisation or transcription corpora, so their timing is only as tight as the forced alignment or the annotator’s patience. The typical failure is a VAD that looks fine on clean speech with clear pauses, then clips word onsets or fires on music and TV audio in real rooms.

| Dataset | Contents | Size | Scope | License | Typical use |
|----|----|----|----|----|----|
| [AVA-Speech](https://sites.research.google/gr/ava/download/) | Movie audio labelled NO_SPEECH / CLEAN_SPEECH / SPEECH_WITH_MUSIC / SPEECH_WITH_NOISE | 160 YouTube films x 15 min (~40 h) | Films | CC BY 4.0 | **The** VAD benchmark for speech under music and noise |
| [AMI](https://huggingface.co/datasets/edinburghcstr/ami) | Recorded meetings with timed transcript segments, close-talk and far-field mics | 100 h | English meetings | CC BY 4.0 | Far-field VAD and diarisation eval |
| [DIHARD III](https://catalog.ldc.upenn.edu/LDC2022S12) | Hard diarisation domains: clinical, restaurant, YouTube, meetings, interviews | Dev ~34 h (full) / eval ~33 h (full) | English and Chinese, many domains | LDC user agreement | Speech activity detection track of DIHARD; hard-case eval |
| [VoxConverse](https://github.com/joonson/voxconverse) | YouTube debates and news with RTTM speaker turns | Dev 216 videos (~20 h) / test 232 videos (~44 h) | Multi-speaker, in the wild | CC BY 4.0 | Diarisation and VAD eval on broadcast speech |
| [MUSAN](https://www.openslr.org/17/) | Separate speech, music and noise recordings, 16 kHz | ~109 h, 11 GB | Speech vs non-speech | CC BY 4.0 | Training data for speech vs music vs noise |
| [LibriSpeech](https://huggingface.co/datasets/openslr/librispeech_asr) | LibriVox audiobooks, 16 kHz | ~1,000 h | English, read | CC BY 4.0 | Clean speech for building synthetic VAD tests; **used in this notebook** (dummy subset) |

**This notebook builds its own labelled test signal.** It takes 3 s of the first clip in `hf-internal-testing/librispeech_asr_dummy` (a 73-clip LibriSpeech sample) and pads it with 1 s silences into `[silence | speech | silence | speech | silence]`. The ground-truth mask is then exact, at 10 ms frames. Pure digital silence is the easiest case a VAD can face, so treat the F1 as a smoke test. AVA-Speech or AMI with the official scoring gives numbers that mean something.

None of these are gated on the Hub, but **DIHARD III is distributed by LDC under a license agreement**, not as open data. AVA-Speech ships YouTube IDs, so some films may no longer be available.

Downloads land in `DL_tasks/datasets/` via `cache_dir` (gitignored).

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

| Model | Params | License | Library | Best for |
|----|----|----|----|----|
| Silero VAD | 1.8M | MIT | torch.hub | de-facto default, tiny, streaming |
| pyannote/segmentation-3.0 | 6M | MIT (gated) | pyannote.audio | VAD + overlap + diarization |
| speechbrain/vad-crdnn-libriparty | 3M | Apache 2.0 | speechbrain | CRDNN VAD |
| webrtcvad | tiny | BSD-3 | webrtcvad | ultra-light classic baseline |

There is no single VAD leaderboard; segmentation quality is reported on **AMI** and **DIHARD**. We use **Silero VAD** (general-purpose, `torch.hub`) as the primary model and **WebRTC** as a classic baseline.

------------------------------------------------------------------------

## 7. Setup

Package roles:

- `torch` - runs Silero VAD (fetched via `torch.hub`, tiny, CPU-friendly)
- `datasets` + `soundfile` - a speech clip we pad with silence to build a labelled test signal
- `numpy` - frame handling; `scikit-learn` - frame metrics
- `pyecharts` - F1 bar + a speech-probability timeline

`webrtcvad` is an optional extra shown in prose.

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

import io

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

SR = 16000

# Build a labelled test signal: [silence | speech | silence | speech | silence]
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean",
                  split="validation", cache_dir=str(DATA_DIR / "hf_cache"))
ds = ds.cast_column("audio", Audio(decode=False))


def _decode(a):
    src = a["bytes"] or a["path"]
    arr, sr = sf.read(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src, dtype="float32")
    return arr

speech = _decode(ds[0]["audio"])[: 3 * SR]  # 3 s of speech
gap = np.zeros(SR, dtype="float32")          # 1 s of silence
signal = np.concatenate([gap, speech, gap, speech, gap])

# ground-truth speech mask (1 per sample), then per-frame labels (10 ms frames)
mask = np.concatenate([np.zeros(SR), np.ones(len(speech)), np.zeros(SR),
                       np.ones(len(speech)), np.zeros(SR)])
FRAME = SR // 100  # 10 ms
frame_true = [int(mask[i:i + FRAME].mean() > 0.5) for i in range(0, len(mask) - FRAME, FRAME)]
print(f"signal {len(signal) / SR:.1f}s, {len(frame_true)} frames, "
      f"{sum(frame_true)} speech frames")
```

    NVIDIA GeForce RTX 3060
    device: cuda:0
    signal 9.0s, 899 frames, 600 speech frames

## 8. Silero VAD

Silero loads from `torch.hub` (cached under `DATA_DIR`) and returns speech segments directly via `get_speech_timestamps`. It is tiny and fast enough to run comfortably on CPU.

------------------------------------------------------------------------

``` python
torch.hub.set_dir(str(DATA_DIR / "torch_hub"))
model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
get_speech_timestamps = utils[0]

wav = torch.from_numpy(signal)
t0 = time.perf_counter()
segments = get_speech_timestamps(wav, model, sampling_rate=SR)
elapsed = time.perf_counter() - t0
print(f"{elapsed:.3f}s  RTF {elapsed / (len(signal) / SR):.4f}")
for s in segments:
    print(f"  speech {s['start'] / SR:5.2f}s -> {s['end'] / SR:5.2f}s")

# per-frame prediction from the returned segments
frame_pred = [0] * len(frame_true)
for s in segments:
    for fi in range(s["start"] // FRAME, min(s["end"] // FRAME, len(frame_pred))):
        frame_pred[fi] = 1

del model
free_memory()
vram("after silero")
```

    Using cache found in ../../datasets/torch_hub/snakers4_silero-vad_master

    0.111s  RTF 0.0123
      speech  1.54s ->  4.03s
      speech  5.54s ->  8.03s
    VRAM after silero      0.00 GB allocated /  0.00 GB reserved

## 9. Benchmark and speech timeline

Score Silero’s frames against the ground-truth mask (precision / recall / F1) and chart both the **F1** and the **speech timeline** (predicted vs true) with ECharts. For a real benchmark, run over AMI / DIHARD with the official scoring; this constructed clip is a smoke test.

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
from sklearn.metrics import precision_recall_fscore_support

p, r, f, _ = precision_recall_fscore_support(frame_true, frame_pred, average="binary")
print(f"Silero  precision {p:.2f}  recall {r:.2f}  F1 {f:.2f}")

bar_chart("VAD frame-level scores (Silero)", ["precision", "recall", "F1"],
          {"Silero VAD": [p, r, f]}, y_name="score")
```

    Silero  precision 0.99  recall 0.82  F1 0.90

``` python
# Speech timeline: true vs predicted speech over time (a step comparison)
from pyecharts import options as opts
from pyecharts.charts import Line

times = [round(i * FRAME / SR, 2) for i in range(len(frame_true))]
tl = Line(init_opts=opts.InitOpts(width="760px", height="300px"))
tl.add_xaxis(times)
tl.add_yaxis("true speech", frame_true, is_step=True, is_symbol_show=False,
             areastyle_opts=opts.AreaStyleOpts(opacity=0.2))
tl.add_yaxis("Silero predicted", [v + 0.02 for v in frame_pred], is_step=True, is_symbol_show=False)
tl.set_global_opts(
    title_opts=opts.TitleOpts(title="Speech activity over time"),
    xaxis_opts=opts.AxisOpts(name="seconds", type_="value"),
    yaxis_opts=opts.AxisOpts(name="speech (1) / silence (0)"),
    tooltip_opts=opts.TooltipOpts(trigger="axis"),
)
tl.render_notebook()
```

## 10. Live Microphone VAD

Sections 8 and 9 scored Silero against a *constructed* signal: real speech spliced between blocks of exact digital zeros. That is a generous test. A real room never gives you zeros - it gives you fan noise, keyboard clatter and the hum of the machine itself, which is precisely what a VAD has to reject.

This runs the same model over a live recording. Speak for part of it and stay quiet for the rest, then compare the detected segments against the waveform envelope in the chart: the gap between “quiet” and “silent” is the whole difficulty of the task, and it is invisible on the synthetic clip.

Needs a working capture device at `/dev/snd`. There is no ground-truth mask for a live take, so this section reports segments and a timeline rather than precision / recall - scoring needs the labelled signal from section 9. The docs builder never executes it (`skip_exec: true`).

------------------------------------------------------------------------

``` python
# --- standalone setup ----------------------------------------------------------
# Lifted from the Setup section and the helper cells above so this demo runs on its
# own in a fresh kernel - no earlier cell has to have been executed first.

import ctypes
import ctypes.util
import gc
import numpy as np
import time
import torch

from dotenv import find_dotenv, load_dotenv
from pathlib import Path

# Knowledge/.env sets HF_TOKEN - authenticated HF Hub requests get higher rate limits
load_dotenv(find_dotenv(usecwd=True))

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

SR = 16000

FRAME = SR // 100  # 10 ms

# --- the demo ------------------------------------------------------------------
# sounddevice needs the PortAudio runtime (libportaudio2) and ALSA nodes under
# /dev/snd. On the knowledge-lab LXC those are passed in from the Proxmox host by
# `av_devices` in infra/proxmox/variables.tf - the camera's video node and its
# USB-Audio card are separate passthroughs.
import io

import sounddevice as sd
from IPython.display import HTML, display
from IPython.utils.capture import capture_output
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Silero is a STREAMING model, and this demo uses it as one. The other live demos in
# this repo fake streaming by re-decoding a sliding window, because Whisper, AST and
# CLAP are offline models with no notion of partial input. Silero is different: it
# carries recurrent state and consumes one small frame at a time, so here the model
# really is fed 32 ms at a time and answers before you have stopped talking.
VAD_FRAME = 512       # Silero accepts EXACTLY this many samples at 16 kHz. 480 and
                      # 1024 both raise inside the TorchScript interpreter - it is a
                      # hard contract, not a hint.
LISTEN_SECONDS = 30   # interrupt the kernel to stop early
UI_EVERY = 0.25       # seconds between view refreshes; the model runs every frame

# Endpointing by hysteresis: one threshold flickers on and off around the boundary,
# so opening and closing use different ones, and a segment has to persist to count.
START_PROB = 0.50     # frame probability that opens a segment
END_PROB = 0.35       # ... and the lower bar it must fall under to close it
MIN_SILENCE_MS = 300  # quiet this long ends a segment (shorter = choppier)
MIN_SPEECH_MS = 120   # segments briefer than this are discarded as blips
SHOW_ROWS = 12

MIC_HINT = "UGREEN"   # substring of the PortAudio device name; None -> first input


def pick_microphone(hint=MIC_HINT):
    "Return the index of a capture device, preferring one whose name matches `hint`."
    inputs = [(i, d) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
    if not inputs:
        raise RuntimeError(
            "no audio capture device: PortAudio sees no ALSA card. Check that /dev/snd "
            "exists and holds controlC*/pcmC*D*c nodes - on the LXC that means adding "
            "them to av_devices and running `just tf-apply`."
        )
    match = [(i, d) for i, d in inputs if hint and hint.lower() in d["name"].lower()]
    return (match or inputs)[0][0]


# rich draws the panels; the in-place update is an IPython display handle. rich's own
# Live goes through ipywidgets in Jupyter and appends a fresh view per refresh
# instead of replacing one, so render with rich and update with the handle.
#   force_jupyter=False is load-bearing: left on, Console.print() calls display()
#   itself and every frame leaks an extra output cell.
_console = Console(record=True, file=io.StringIO(), width=100,
                   force_terminal=True, force_jupyter=False)
_FRAGMENT = '<pre style="font-family:ui-monospace,monospace;line-height:1.3;margin:0">{code}</pre>'


def _html(renderable):
    "Render a rich object to an HTML fragment suitable for display handle updates."
    _console.print(renderable)
    out = _console.export_html(inline_styles=True, code_format=_FRAGMENT)
    _console.file = io.StringIO()  # reset between frames, else the HTML grows forever
    return HTML(out)


def _note(text, title=None, style="dim"):
    "A one-off rich line, so nothing in this cell falls back to a bare print."
    display(_html(Panel(Text.from_markup(text), title=title, border_style=style,
                        padding=(0, 1))))


def _grid(**rows):
    "Two-column rich grid - the shape used for every stats block in this cell."
    g = Table.grid(padding=(0, 2))
    g.add_column(style="dim", justify="right")
    g.add_column()
    for k, v in rows.items():
        g.add_row(k.replace("_", " "), str(v))
    return g


def rvram(tag=""):
    "vram(), rendered through rich instead of print. No-op on CPU."
    if not torch.cuda.is_available():
        return
    display(_html(Panel(
        _grid(**{"allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f} GB",
                 "reserved": f"{torch.cuda.memory_reserved() / 1e9:.2f} GB"}),
        title=f"[dim]VRAM {tag}[/]", border_style="dim", padding=(0, 1))))


def _dbfs(x):
    "RMS level of a float32 block, in dBFS. -inf for digital silence."
    r = float(np.sqrt((x ** 2).mean())) if len(x) else 0.0
    return 20 * np.log10(r) if r > 0 else float("-inf")


def _view(elapsed, seconds, prob, dbfs, segments, live_start=None, note=""):
    "One frame of the live view: header stats over one table row per finished segment."
    speaking = live_start is not None
    head = Table.grid(padding=(0, 2))
    head.add_column(style="dim", justify="right")
    head.add_column()
    bar = int(max(0.0, min(1.0, prob)) * 24)
    colour = "green" if speaking else "yellow"
    head.add_row("elapsed", f"{elapsed:5.1f}s / {seconds:.0f}s")
    head.add_row("speech prob", f"[{colour}]{'#' * bar}[/][dim]{'.' * (24 - bar)}[/] {prob:.2f}")
    head.add_row("level", f"{dbfs:6.1f} dBFS")
    head.add_row("state", f"[green]SPEECH since {live_start:.1f}s[/]" if speaking
                          else "[dim]silence[/]")
    if note:
        head.add_row("status", f"[dim]{note}[/]")
    voiced = sum(s["end"] - s["start"] for s in segments)
    head.add_row("totals", f"{len(segments)} segment(s), {voiced:.1f}s voiced"
                           + (f", {voiced / elapsed:.0%} of elapsed" if elapsed > 0 else ""))

    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, expand=True,
                  header_style="dim", border_style="dim")
    table.add_column("#", justify="right", style="dim", width=4, no_wrap=True)
    table.add_column("start", justify="right", width=8, no_wrap=True)
    table.add_column("end", justify="right", width=8, no_wrap=True)
    table.add_column("dur", justify="right", width=7, no_wrap=True)
    table.add_column("mean p", justify="right", width=8, no_wrap=True)
    table.add_column("peak p", justify="right", width=8, no_wrap=True)
    shown = segments[-SHOW_ROWS:]
    if len(segments) > len(shown):
        table.add_row("...", "", "", "", "", f"[dim]{len(segments) - len(shown)} earlier[/]")
    for s in shown:
        table.add_row(str(s["n"]), f"{s['start']:.2f}s", f"{s['end']:.2f}s",
                      f"{s['end'] - s['start']:.2f}s", f"{s['mean_p']:.2f}",
                      f"{s['peak_p']:.2f}")
    if speaking:
        table.add_row(f"[green]{len(segments) + 1}[/]", f"[green]{live_start:.2f}s[/]",
                      "[green]...[/]", f"[green]{elapsed - live_start:.2f}s[/]", "", "")
    if not segments and not speaking:
        table.add_row("-", "", "", "", "", "[dim italic](no speech yet)[/]")
    return Panel(Group(head, Text(""), table),
                 title="[cyan]live voice activity detection[/]", border_style="cyan",
                 padding=(0, 1))


def live_vad(seconds=LISTEN_SECONDS):
    """Stream the microphone through Silero one 32 ms frame at a time.

    The capture stream is opened at 16 kHz with blocksize=VAD_FRAME, so every
    PortAudio callback delivers exactly one frame the model will accept - no
    resampling, no re-blocking, no frame-alignment arithmetic. (Verified that the
    device really honours 16 kHz rather than silently returning 44.1 kHz data with
    the wrong label: 80000 samples took 5.4 s of wall clock, not 1.8 s.)

    Silero costs 0.33 ms per 32 ms frame here - a real-time factor of 0.010 - so
    unlike the ASR and classification demos there is no latency budget to manage.

    Returns (segments, trace); both feed vad_charts().
    """
    import queue as _queue

    mic = pick_microphone()
    q, overflows = _queue.Queue(), 0

    def on_audio(indata, frames, time_info, status):
        "PortAudio callback thread - keep it cheap, just hand the frame over."
        nonlocal overflows
        if status.input_overflow:
            overflows += 1
        q.put(indata[:, 0].copy())  # copy: PortAudio reuses this buffer

    model.reset_states()  # recurrent state must not carry over from an earlier run
    view = display(_html(_view(0.0, seconds, 0.0, float("-inf"), [], note="starting")),
                   display_id=True)
    segments, trace = [], []
    live_start, quiet_ms, run_probs = None, 0.0, []
    frame_ms = 1000 * VAD_FRAME / SR
    last_ui = -UI_EVERY

    with sd.InputStream(device=mic, samplerate=SR, channels=1, dtype="float32",
                        blocksize=VAD_FRAME, callback=on_audio):
        _note(f"[bold]mic [{mic}][/] @ {SR} Hz  ->  {VAD_FRAME}-sample frames "
              f"({frame_ms:.0f} ms), listening {seconds:.0f}s - talk, then pause",
              title="[cyan]capture[/]", style="cyan")
        t0 = time.perf_counter()
        try:
            while time.perf_counter() - t0 < seconds:
                try:
                    frame = q.get(timeout=0.2)
                except _queue.Empty:
                    continue
                if frame.size != VAD_FRAME:  # never seen in practice; the contract is hard
                    continue

                with torch.no_grad():
                    prob = float(model(torch.from_numpy(frame), SR).item())
                elapsed = time.perf_counter() - t0
                level = _dbfs(frame)
                trace.append({"at": elapsed, "prob": prob, "dbfs": level})

                # Hysteresis state machine: open on START_PROB, close only after
                # MIN_SILENCE_MS below END_PROB, discard anything under MIN_SPEECH_MS.
                if live_start is None:
                    if prob >= START_PROB:
                        live_start, quiet_ms, run_probs = elapsed, 0.0, [prob]
                else:
                    run_probs.append(prob)
                    quiet_ms = quiet_ms + frame_ms if prob < END_PROB else 0.0
                    if quiet_ms >= MIN_SILENCE_MS:
                        end = elapsed - quiet_ms / 1000
                        if (end - live_start) * 1000 >= MIN_SPEECH_MS:
                            segments.append({
                                "n": len(segments) + 1, "start": live_start, "end": end,
                                "mean_p": float(np.mean(run_probs)),
                                "peak_p": float(np.max(run_probs)),
                            })
                        live_start, quiet_ms, run_probs = None, 0.0, []

                if elapsed - last_ui >= UI_EVERY:  # model every frame, redraw far less
                    last_ui = elapsed
                    view.update(_html(_view(elapsed, seconds, prob, level, segments,
                                            live_start)))
        except KeyboardInterrupt:
            view.update(_html(_view(time.perf_counter() - t0, seconds, 0.0,
                                    float("-inf"), segments, live_start, note="stopped")))

    if live_start is not None:  # speech still open when the clock ran out
        segments.append({"n": len(segments) + 1, "start": live_start,
                         "end": time.perf_counter() - t0,
                         "mean_p": float(np.mean(run_probs)),
                         "peak_p": float(np.max(run_probs))})

    voiced = sum(s["end"] - s["start"] for s in segments)
    total = trace[-1]["at"] if trace else 0.0
    display(_html(Panel(_grid(**{
        "frames": f"{len(trace)} x {frame_ms:.0f} ms",
        "segments": str(len(segments)),
        "voiced": f"{voiced:.1f}s of {total:.1f}s ({voiced / total:.0%})" if total else "-",
        "overflows": f"[red]{overflows}[/]" if overflows else "0",
    }), title="[dim]done[/]", border_style="dim", padding=(0, 1))))
    return segments, trace


def vad_charts(segments, trace):
    """Two ECharts views of the run (pyecharts is the repo standard for all charts).

    The probability trace with both thresholds drawn on it is the useful picture: you
    can see directly why a boundary landed where it did, and why one threshold would
    have chattered where two do not.
    """
    from pyecharts import options as opts
    from pyecharts.charts import Bar, Line, Page

    if not trace:
        _note("no frames captured - nothing to chart", title="[yellow]empty[/]",
              style="yellow")
        return None

    x = [round(t["at"], 2) for t in trace]
    mask = [0] * len(trace)
    for s in segments:
        for i, t in enumerate(trace):
            if s["start"] <= t["at"] <= s["end"]:
                mask[i] = 1

    prob = (
        Line(init_opts=opts.InitOpts(width="820px", height="400px"))
        .add_xaxis(x)
        .add_yaxis("speech probability", [round(t["prob"], 4) for t in trace],
                   is_symbol_show=False, is_smooth=False,
                   areastyle_opts=opts.AreaStyleOpts(opacity=0.15),
                   label_opts=opts.LabelOpts(is_show=False))
        .add_yaxis("endpointed speech", mask, is_step=True, is_symbol_show=False,
                   label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Silero speech probability, frame by frame",
                subtitle=f"{1000 * VAD_FRAME / SR:.0f} ms frames - open at "
                         f"{START_PROB}, close under {END_PROB} for {MIN_SILENCE_MS} ms"),
            xaxis_opts=opts.AxisOpts(name="seconds", type_="value"),
            yaxis_opts=opts.AxisOpts(name="probability", max_=1),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_top="10%"),
        )
        .set_series_opts(markline_opts=opts.MarkLineOpts(data=[
            opts.MarkLineItem(y=START_PROB, name="open"),
            opts.MarkLineItem(y=END_PROB, name="close"),
        ]))
    )

    if segments:
        durations = Bar(init_opts=opts.InitOpts(width="820px", height="320px"))
        durations.add_xaxis([str(s["n"]) for s in segments])
        durations.add_yaxis("duration (s)",
                            [round(s["end"] - s["start"], 2) for s in segments])
        durations.add_yaxis("mean probability",
                            [round(s["mean_p"], 3) for s in segments])
        durations.set_global_opts(
            title_opts=opts.TitleOpts(title="Endpointed segments",
                                      subtitle="what the state machine actually committed"),
            xaxis_opts=opts.AxisOpts(name="segment"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_top="10%"),
        )
        durations.set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        return Page().add(prob, durations).render_notebook()
    return prob.render_notebook()


# torch.hub prints its cache path on load; swallow it so the cell's output stays rich.
torch.hub.set_dir(str(DATA_DIR / "torch_hub"))
with capture_output():
    model, _utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
_note("Silero VAD loaded [dim](~1.8M params, CPU is plenty - 0.33 ms per 32 ms frame)[/]",
      title="[dim]model[/]")

SEGMENTS, TRACE = live_vad(seconds=30)

del model
free_memory()
rvram("after live vad")

vad_charts(SEGMENTS, TRACE)
```

```
╭───────────────────────────────────────────── model ──────────────────────────────────────────────╮
│ Silero VAD loaded (~1.8M params, CPU is plenty - 0.33 ms per 32 ms frame)                        │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

```
╭───────────────────────────────── live voice activity detection ──────────────────────────────────╮
│     elapsed   29.9s / 30s                                                                        │
│ speech prob  ........................ 0.01                                                       │
│       level   -51.5 dBFS                                                                         │
│       state  SPEECH since 28.6s                                                                  │
│      totals  5 segment(s), 8.0s voiced, 27% of elapsed                                           │
│                                                                                                  │
│                                                                                                  │
│         #             start               end             dur            mean p          peak p  │
│  ──────────────────────────────────────────────────────────────────────────────────────────────  │
│         1            11.45s            12.67s           1.22s              0.77            1.00  │
│         2            13.18s            14.27s           1.09s              0.77            1.00  │
│         3            16.16s            17.88s           1.73s              0.83            1.00  │
│         4            18.53s            20.92s           2.40s              0.86            1.00  │
│         5            21.88s            23.45s           1.57s              0.84            1.00  │
│         6            28.64s               ...           1.25s                                    │
│                                                                                                  │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

```
╭──────────────────────────────────────────── capture ─────────────────────────────────────────────╮
│ mic [0] @ 16000 Hz  ->  512-sample frames (32 ms), listening 30s - talk, then pause              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

```
╭────────────────────────────────────────────── done ──────────────────────────────────────────────╮
│    frames  932 x 32 ms                                                                           │
│  segments  6                                                                                     │
│    voiced  9.1s of 30.0s (30%)                                                                   │
│ overflows  0                                                                                     │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

```
╭────────────────────────────────────── VRAM after live vad ───────────────────────────────────────╮
│ allocated  0.00 GB                                                                               │
│  reserved  0.00 GB                                                                               │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## 11. Common Frameworks

VAD is a component, not a product, and its ecosystem reflects that: nobody serves a VAD behind an HTTP endpoint. It runs in-process, frame by frame, in front of something expensive. So the frameworks that matter are the models themselves (all tiny, all specialised) and the conversational stacks that embed them.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [Silero VAD](https://github.com/snakers4/silero-vad) | modelling | The de-facto default: 1.8M params, streaming-capable, with the utility functions for timestamps and chunking | MIT | Default, and hard to beat. It is small enough that the question is never whether you can afford it |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | modelling | `segmentation-3.0`: VAD plus **overlapped speech** detection, and the front half of a diarization pipeline | MIT (models gated on the Hub) | Two people talk at once and you need to know. Silero cannot tell you that |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) | modelling | A CRDNN VAD with a full training recipe | Apache 2.0 | Your audio is far enough from the training distribution - underwater, radio, industrial - that you need to retrain |
| [webrtcvad](https://github.com/wiseman/py-webrtcvad) | modelling | The classic GMM gate from the WebRTC stack, three aggressiveness levels, no model file | BSD-3 | You need a dependency-free gate in a constrained environment, or a baseline to prove the neural model earns its place |
| [sounddevice](https://github.com/spatialaudio/python-sounddevice) / PyAudio + [torchaudio](https://github.com/pytorch/audio) | data | Frame-accurate capture at a fixed block size, which is what a streaming VAD consumes | MIT / BSD-2 | Always for live use. VAD models want exact frame sizes (512 samples at 16 kHz for Silero) and misalignment silently degrades them |
| [ONNX Runtime](https://onnxruntime.ai/) | inference runtime | Silero as a single ONNX file, sub-millisecond per frame, callable from C++, mobile or WASM in the browser | MIT | The gate has to run client-side - which for bandwidth reasons is often where it belongs |
| [Pipecat](https://github.com/pipecat-ai/pipecat) / [LiveKit Agents](https://github.com/livekit/agents) | orchestration | VAD wired into turn detection and barge-in: not just “is there speech” but “has the user finished, and should I stop talking” | BSD-2 / Apache 2.0 | Building a voice agent. Turn-taking is a harder problem than VAD and these solve it - see `Frameworks/01_pipecat` |
| [whisper-streaming](https://github.com/ufal/whisper_streaming) / [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) | orchestration | VAD-driven chunking in front of an ASR model, with the buffering and overlap logic already written | MIT | Long-form or live transcription. This is the single most common thing a VAD is used for - see `02_Automatic_Speech_Recognition` |
| [pyannote.metrics](https://github.com/pyannote/pyannote-metrics) | evaluation | Detection error rate split into false alarm and missed detection, scored against AMI/DIHARD references | MIT | Comparing gates. The split matters more than the total: a false alarm costs compute, a miss costs a lost word |

**The 2026 default stack** is Silero via ONNX Runtime, fed exact 32 ms frames by `sounddevice`, inside Pipecat if there is a conversation and inside whisper-streaming if there is not. pyannote enters only when overlapped speech is part of the problem.

The common wrong turn is tuning a VAD for accuracy when the real requirement is asymmetric. Dropping the start of a word is a user-visible failure; letting a second of silence through costs a few milliseconds of GPU. Set the threshold and the `min_silence_duration` for that asymmetry, and add hangover padding at both ends - almost every “the transcript is missing the first word” bug is a VAD tuned as if the two errors cost the same.

------------------------------------------------------------------------

## 12. Going Further

- **As an ASR front-end.** VAD timestamps drive chunked long-form transcription: transcribe only the speech regions, and pad each region so the model never starts mid-word. On real recordings this is both a large speed win and an accuracy win, because Whisper hallucinates text on silence.
- **Streaming endpointing.** Silero runs frame-by-frame with a few-ms lookahead for live end-of-utterance detection. Section 10 records first and scores after; a real endpointer feeds it 32 ms at a time and reacts before you stop talking.
- **Endpointing is not VAD.** Knowing that speech stopped is easy; knowing the *turn* ended is not. A 700 ms silence means “finished” after a statement and “still thinking” after “my account number is”. Semantic turn detection models exist for exactly this gap.
- **Fine-tuning for a hostile domain.** In-car, industrial, or radio audio can defeat every off-the-shelf gate. Retraining a small CRDNN on a few hours of the real environment is cheap and usually decisive.
- **Related notebooks.** `02_Automatic_Speech_Recognition` (what the gate sits in front of), `04_Audio_Classification` (the same architecture shape with more than two classes), `00_Text_to_Speech` (the other half of the turn, and what barge-in has to interrupt).

------------------------------------------------------------------------

Back to top
