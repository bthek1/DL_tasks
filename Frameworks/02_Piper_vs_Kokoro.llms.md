# Piper vs Kokoro

Two small open text-to-speech engines that both run as plain ONNX: how each one works, runnable examples side by side, and a measured comparison of speed, memory and intelligibility.

Author

Benedict Thekkel

## Short answer

**Use Kokoro when you have a GPU or can spare a few CPU cores. Use Piper medium on one or two CPU cores, for a language Kokoro lacks, or when you must train your own voice.**

| Question | Piper medium (lessac) | Piper high (ljspeech) | Kokoro v1.0 (fp32) |
|----|----|----|----|
| Real-time factor, CPU 1 thread | **0.054** | 0.433 | 0.476 |
| Real-time factor, CPU 4 threads | **0.032** | 0.269 | 0.267 |
| Real-time factor, CUDA | 0.028 | 0.031 | 0.029 |
| First audio, CPU 1 thread | **0.14 s** | 1.22 s | 1.45 s |
| Round-trip WER, 20 sentences x 3 passes | 3.07% | 3.27% | **1.23%** |
| Disk | **63 MB** | 114 MB | 326 MB |
| Extra process RAM, CPU | **about 200 MB** | about 270 MB | 500 to 600 MB |
| Parameters | **15.7M** | 28.3M | 81.2M |
| Voices | 178 voices, one file each | same | 54 voices in one model |
| Languages | 53 language families | same | 8 |
| Train your own voice | Yes | Yes | No trainer released |

Real-time factor (RTF) is synthesis time divided by audio length, so 0.1 means ten times faster than real time. Word error rate (WER) here means: synthesise a sentence, transcribe it with Whisper, and count the wrong words.

Three results are worth keeping:

1.  **The speed gap belongs to Piper medium, not to Piper.** Piper high is as slow as Kokoro on CPU. On CUDA all three are about equal.
2.  **The text front end decides more of the hard cases than the voice model does.** Section 9 shows this. It is also where the two engines are hardest to compare fairly.
3.  **Check the licence of each Piper voice.** The engine licence and the voice dataset licence are separate. Section 5 has the details.

**What this page does not measure:** how natural the voices sound. The players below exist so you can judge that yourself. WER measures whether a machine can follow the words, not whether a person enjoys listening.

------------------------------------------------------------------------

## 1. What are Piper and Kokoro?

Both map written text to a speech waveform. Both run locally with no API, and both are small enough for a CPU.

**Input.** A text string, a chosen voice, and a speed setting.

**Output.** Mono audio at a fixed rate: 22,050 Hz for the Piper voices used here, 24,000 Hz for Kokoro.

| Engine | Maintainer | Architecture | Weights format used here | Engine licence |
|----|----|----|----|----|
| [Piper](https://github.com/OHF-Voice/piper1-gpl) | Open Home Foundation | VITS | ONNX, one file per voice | GPL-3.0 |
| [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) | hexgrad | StyleTTS 2 decoder with ISTFTNet | ONNX, one model plus a style array per voice | Apache-2.0 |

Related tasks and pages:

| Task | Page |
|----|----|
| The whole TTS landscape, including larger models | [Text-to-Speech](../Audio/text_to_speech.llms.md) |
| Transcription, used here as the measuring tool | [Automatic Speech Recognition](../Audio/automatic_speech_recognition.llms.md) |
| Putting TTS inside a live voice agent | [Pipecat](../Frameworks/pipecat.llms.md) |

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint | Better fit |
|----|----|----|----|----|
| Local voice assistant, for example Home Assistant | Smart home | Short replies to speech on a small box | CPU budget and first-audio delay | Piper medium |
| Robot or kiosk voice | Embedded | Status phrases on a single-board computer | Few cores, no network | Piper medium |
| Screen reader or offline reader | Accessibility | Long text, played as it is made | Low delay, many languages | Piper |
| Voice agent behind Pipecat | Customer service | LLM sentences to speech, streamed | First-audio delay, cost per stream | Kokoro on GPU |
| Self-hosted TTS API, for example Kokoro-FastAPI | Platform | HTTP request to audio file or stream | Licence, throughput per GPU | Kokoro |
| Narration and long-form audio | Media | Whole documents, in batch | Listener comfort, not delay | Kokoro, then listen and decide |

**What a benchmark number hides.** Most production failures in small TTS come from text, not from the model. A voice that reads \$3.50 as “dollar three point five zero” sounds broken however good the waveform is. Streaming matters too: both engines produce a whole sentence at a time, so first-audio delay depends on the length of the first sentence. Neither engine streams inside a sentence in the path used here.

------------------------------------------------------------------------

## 3. How They Work

Both engines share the same outline. They differ inside the last box, and in who makes the phonemes.

``` text
text
  |
  v
text normalisation     $3.50 -> three dollars and fifty cents (often skipped, see section 9)
  |
  v
grapheme to phoneme    espeak-ng (Piper) | misaki, with espeak-ng as fallback (Kokoro)
  |
  v
phoneme ids            Piper: table in the voice's .onnx.json | Kokoro: tokenizer.json
  |
  v
ONNX model             Piper: VITS | Kokoro: StyleTTS 2 decoder + ISTFTNet
  |
  v
waveform
```

A phoneme is one unit of speech sound. Grapheme to phoneme (G2P) converts spelling into those sounds. Neither model reads letters, so a wrong phoneme becomes a wrong word no matter how good the model is.

**Piper: VITS (2021, [paper](https://arxiv.org/abs/2106.06103)).** VITS trains an end-to-end model from phonemes to waveform. A variational autoencoder, a normalising flow and an adversarial loss replace the older two-stage acoustic model plus vocoder. It samples random noise during synthesis, so **Piper’s output changes on every run**. The inputs `noise_scale` and `noise_w` control that sampling, and `length_scale` controls speed. The project started in January 2023 at `rhasspy/piper`. That repository is now archived, and development moved to `OHF-Voice/piper1-gpl`, which also changed the licence from MIT to GPL-3.0. Its README says the project is looking for maintainers.

**Kokoro: StyleTTS 2 decoder (v1.0, January 2025).** [StyleTTS 2](https://arxiv.org/abs/2306.07691) separates what is said from how it is said. A 256-number style vector carries the voice, and an [ISTFTNet](https://arxiv.org/abs/2203.02395) decoder makes the waveform. The model card says “decoder only: no diffusion, no encoder release”. That has three effects. Output is deterministic. A voice is a small array, so 54 voices share one model and you can blend them by adding arrays. But no encoder means there is no released way to make a new voice from recordings.

------------------------------------------------------------------------

## 4. Evaluation Metrics

- **Real-time factor (RTF).** Synthesis time divided by audio duration. Under 1.0 keeps up with playback. The margin is what lets one machine serve several streams.
- **Time to first audio.** Time from text to the first playable sentence. This is the delay a person notices in a conversation. This page measures it warm and cold.
- **Round-trip WER.** Transcribe the output with an ASR model and compute word error rate against the input. It is cheap and automatic. It has two blind spots. Whisper rewrites spoken numbers as digits, so “dollar three point five zero” and “three dollars and fifty cents” both come back as `$3.50`. It also says nothing about how natural the voice is.
- **Mean opinion score (MOS).** People rate samples from 1 to 5. This is the real measure of naturalness, and this page does not have one. The [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) collects crowd votes.
- **Footprint.** Disk, RAM and VRAM. These often decide the choice on a small device.

------------------------------------------------------------------------

``` python
import jiwer

norm = jiwer.Compose([jiwer.ToLowerCase(), jiwer.RemovePunctuation(),
                      jiwer.RemoveMultipleSpaces(), jiwer.Strip()])

# A synthetic engine said this sentence, and an ASR model wrote down what it heard
reference = "The birch canoe slid on the smooth planks."
heard = "the birch canoe slid on the smooth planks"
print("round-trip WER, perfect :", jiwer.wer(norm(reference), norm(heard)))
print("round-trip WER, one miss:", round(jiwer.wer(norm(reference), norm("the birch canoe slid on the smooth banks")), 3))

synthesis_seconds, audio_seconds = 0.5, 5.0
print("RTF:", synthesis_seconds / audio_seconds, "(under 1.0 is faster than real time)")
```

    round-trip WER, perfect : 0.0
    round-trip WER, one miss: 0.125
    RTF: 0.1 (under 1.0 is faster than real time)

## 5. Datasets, and Why Voice Licences Differ

Each Piper voice is trained from its own dataset, so **the licence of the voice’s data can be stricter than the MIT tag on the Hub repository**. Read the `MODEL_CARD` file beside each voice.

| Data | Used for | Contents | Licence stated at source | Commercial use |
|----|----|----|----|----|
| [LJ Speech](https://keithito.com/LJ-Speech-Dataset/) | Piper `ljspeech` | One US English speaker | Public domain | Clearest of the three Piper voices here |
| [Blizzard 2013 Lessac](https://www.cstr.ed.ac.uk/projects/blizzard/2013/lessac_blizzard2013/license.html) | Piper `lessac` | Audiobook recordings | A “Research Licence Agreement” | Unclear for a voice trained on it. Get advice before you ship it |
| [RyanSpeech](https://www.kaggle.com/datasets/roholazandie/ryanspeech) | Piper `ryan` | One US English speaker | CC BY-NC-SA 4.0 | No: non-commercial |
| Kokoro training set, not released | Kokoro v1.0 | A few hundred hours | The model card says permissive or non-copyrighted audio, and lists CC BY sources | Weights are Apache-2.0 |
| Harvard sentences (IEEE, 1969) | Evaluation on this page | Phonetically balanced English sentences | Published test material | Not relevant, used for scoring only |

This is an engineer’s reading of those pages, not legal advice.

The evaluation text is written inside the notebook, so no dataset download is needed. Only models land in `DL_tasks/datasets/hf_cache` (gitignored).

------------------------------------------------------------------------

## 6. The Model Landscape

Parameter counts below come from summing the initializers in each ONNX file. Voice and language counts come from Piper’s `voices.json` and Kokoro’s `VOICES.md`.

| Model | Params | ONNX size | Rate | Scope | Licence | Best for |
|----|----|----|----|----|----|----|
| Piper medium (`lessac`) | 15.7M | 63 MB | 22,050 Hz | English voice | see section 5 | CPU speed and size |
| Piper high (`ljspeech`) | 28.3M | 114 MB | 22,050 Hz | English voice | Public domain data | A public domain voice, if a GPU or spare CPU is available |
| Kokoro fp32 (`model.onnx`) | 81.2M | 326 MB | 24,000 Hz | 54 voices, 8 languages | Apache-2.0 | Best WER here, and CUDA |
| Kokoro q8 (`model_quantized.onnx`) | not counted | 92 MB | 24,000 Hz | same | Apache-2.0 | Small download on CPU only |

Piper’s low quality `lessac` voice has the same 15.7M parameters as medium. Piper has 178 voices in total: 124 medium, 26 low, 14 x_low and 14 high, across 53 language families and 58 locales.

**How this ties back to section 2.** Piper medium wins the CPU budget cases. Kokoro wins where a GPU exists, because the speed gap disappears there and the licence is simpler. For broad language coverage or a custom voice, Piper is the only option of the two.

------------------------------------------------------------------------

## 7. Setup

**Neither vendor package can run here, and neither is needed.** This repo is on Python 3.14. `kokoro` and `misaki` 0.9.4 require Python below 3.13, and `kokoro-onnx` 0.6.1 requires below 3.14. So this page drives both models with ONNX Runtime directly, in about 40 lines each. It also shows how little there is inside either engine.

| Package | Role |
|----|----|
| `onnxruntime-gpu` | Runs both models. Without a GPU it falls back to CPU |
| `phonemizer` | Python wrapper around espeak-ng, for G2P |
| `espeakng-loader` | Ships the espeak-ng library and data as a wheel, so no `apt install` or sudo |
| `huggingface_hub` | Downloads the model files |
| `transformers`, `torch` | Whisper, for round-trip WER only |

`onnxruntime-gpu` installs without its `[cuda]` extra, because it can reuse the CUDA 13 libraries `torch` already brought. That costs 335 MB rather than 2.5 GB. Import `torch` first.

**Rerunning this page.** Run it in JupyterLab on a machine with these packages. The title cell carries `skip_exec: true` like every other page here, because the submodule’s CI installs no model dependencies. So `nbdev-test` does not run this page, and the stored outputs are what the site shows.

**How this was checked against the official package.** On a test sentence, `piper-tts` 1.8.0 and the `Piper` class below produced the same phonemes for all three voices, apart from the space between sentences. Audio cannot be compared byte for byte, because Piper is random.

------------------------------------------------------------------------

``` python
import base64
import ctypes
import ctypes.util
import gc
import io
import json
import re
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import soundfile as sf
import torch  # import before onnxruntime so its CUDA libraries are already on hand
import onnxruntime as ort
import espeakng_loader
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars
from IPython.display import HTML, display
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.wrapper import EspeakWrapper

# Knowledge/.env sets HF_TOKEN - authenticated HF Hub requests get higher rate limits
load_dotenv(find_dotenv(usecwd=True))

device = "cuda:0" if torch.cuda.is_available() else "cpu"
ort.preload_dlls()
ort.set_default_logger_severity(3)
disable_progress_bars()

CPU = ["CPUExecutionProvider"]
CUDA = ["CUDAExecutionProvider", "CPUExecutionProvider"]
has_cuda = device != "cpu" and "CUDAExecutionProvider" in ort.get_available_providers()
PROVIDERS = CUDA if has_cuda else CPU
print("device:", device, "| onnxruntime", ort.__version__, "|", PROVIDERS[0])

# All downloads go to DL_tasks/datasets/ (gitignored)
DATA_DIR = Path("../../datasets")
HF_CACHE = DATA_DIR / "hf_cache"
HF_CACHE.mkdir(parents=True, exist_ok=True)

# The pip-installed espeak-ng library and data, so no apt install or sudo is needed
EspeakWrapper.set_library(espeakng_loader.get_library_path())
EspeakWrapper.set_data_path(espeakng_loader.get_data_path())


def gpu_mb():
    "Whole-GPU memory in MB. torch.cuda counters cannot see ONNX Runtime, so ask the driver."
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return 0
    return int(out.split()[0])


def rss_mb():
    return psutil.Process().memory_info().rss / 1e6


def vram(tag=""):
    print(f"{tag:24s} GPU {gpu_mb():6d} MB | process RSS {rss_mb():6.0f} MB")


def free_memory():
    "GC, then hand freed VRAM and RAM back. Call right after `del session`."
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    try:
        ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6").malloc_trim(0)
    except Exception:  # non-glibc - nothing to trim
        pass


def session(path, threads=4, providers=PROVIDERS):
    "An ONNX Runtime session with thread counts set explicitly."
    opts = ort.SessionOptions()
    # Left unset, ORT pins threads to cores and logs an error per thread inside this LXC
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=providers)


def player(audio, sr):
    "An <audio> tag holding the clip as MP3, about a tenth of the size of the WAV."
    buf = io.BytesIO()
    sf.write(buf, np.clip(audio, -1, 1), sr, format="MP3")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<audio controls preload="none" style="width:220px" src="data:audio/mpeg;base64,{b64}"></audio>'


def player_table(columns, rows):
    "Show `rows` of (label, [(audio, sr), ...]) as a table of players under `columns`."
    head = "".join(f"<th>{c}</th>" for c in ["Text", *columns])
    body = "".join(
        f"<tr><td>{label}</td>" + "".join(f"<td>{player(a, sr)}</td>" for a, sr in clips) + "</tr>"
        for label, clips in rows
    )
    display(HTML(f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"))


# Split on sentence ends, but not after the abbreviations used on this page
SENTENCE_END = re.compile(r"(?<!Dr\.)(?<!St\.)(?<=[.!?])\s+(?=[A-Z])")


def sentences(text):
    return SENTENCE_END.split(text.strip())


def speak(engine, text, **kwargs):
    "Synthesise sentence by sentence, the way both engines stream, and join the audio."
    return np.concatenate([engine.synth(s, **kwargs) for s in sentences(text)])
```

    device: cuda:0 | onnxruntime 1.31.1 | CUDAExecutionProvider

## 8. The Engines

### Piper

A voice is two files: the `.onnx` model and a `.onnx.json` config holding the sample rate, the espeak voice, the inference defaults and the phoneme id table. Synthesis has three steps: phonemise with espeak-ng, map each phoneme to an id with a pad id after it, and run the model.

------------------------------------------------------------------------

``` python
import unicodedata

PIPER_REPO = "rhasspy/piper-voices"
# espeak-ng resolves a bare "en" to British English, and phonemizer refuses the bare name
ESPEAK_ALIASES = {"en": "en-gb"}


class Piper:
    "A Piper voice on plain ONNX Runtime: espeak-ng phonemes -> ids -> VITS -> waveform."

    def __init__(self, stem, threads=4, providers=PROVIDERS):
        self.path = Path(hf_hub_download(PIPER_REPO, f"{stem}.onnx", cache_dir=HF_CACHE))
        self.config = json.loads(Path(hf_hub_download(PIPER_REPO, f"{stem}.onnx.json", cache_dir=HF_CACHE)).read_text())
        self.sr = self.config["audio"]["sample_rate"]
        voice = self.config["espeak"]["voice"]
        self.g2p = EspeakBackend(ESPEAK_ALIASES.get(voice, voice), preserve_punctuation=True, with_stress=True)
        self.session = session(self.path, threads, providers)

    def phonemes(self, text):
        return self.g2p.phonemize([text], strip=True)[0]

    def ids(self, phonemes):
        "Start marker, then each phoneme followed by a pad id, then an end marker."
        table = self.config["phoneme_id_map"]
        out = [*table["^"], *table["_"]]
        for ch in unicodedata.normalize("NFD", phonemes):
            if ch in table:
                out += [*table[ch], *table["_"]]
        return out + table["$"]

    def synth(self, text, length_scale=None, noise_scale=None, noise_w=None):
        inf = self.config["inference"]
        scales = [noise_scale or inf["noise_scale"], length_scale or inf["length_scale"], noise_w or inf["noise_w"]]
        ids = self.ids(self.phonemes(text))
        return self.session.run(None, {
            "input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": np.array(scales, dtype=np.float32),
        })[0].squeeze()


TEXT = "The birch canoe slid on the smooth planks."
LESSAC = "en/en_US/lessac/medium/en_US-lessac-medium"

t0 = time.perf_counter()
piper = Piper(LESSAC)
print(f"load: {time.perf_counter() - t0:.2f}s, {piper.path.stat().st_size / 1e6:.0f} MB on disk, running on {piper.session.get_providers()[0]}")
print("inputs:", [(i.name, i.type) for i in piper.session.get_inputs()])
print("inference defaults:", piper.config["inference"])
print("phonemes:", piper.phonemes(TEXT))
print("first ids:", piper.ids(piper.phonemes(TEXT))[:16])

t0 = time.perf_counter()
audio = piper.synth(TEXT)
seconds = time.perf_counter() - t0
print(f"{len(audio) / piper.sr:.2f}s of audio at {piper.sr} Hz in {seconds:.3f}s")
player_table(["Piper lessac medium"], [(TEXT, [(audio, piper.sr)])])

del piper
free_memory()
vram("after piper")
```

    load: 1.61s, 63 MB on disk, running on CUDAExecutionProvider
    inputs: [('input', 'tensor(int64)'), ('input_lengths', 'tensor(int64)'), ('scales', 'tensor(float)')]
    inference defaults: {'noise_scale': 0.667, 'length_scale': 1, 'noise_w': 0.8}
    phonemes: ðə bˈɜːtʃ kənˈuː slˈɪd ɔnðə smˈuːð plˈæŋks.
    first ids: [1, 0, 41, 0, 59, 0, 3, 0, 15, 0, 120, 0, 62, 0, 122, 0]
    2.25s of audio at 22050 Hz in 0.411s

| Text                                       | Piper lessac medium |
|--------------------------------------------|---------------------|
| The birch canoe slid on the smooth planks. |                     |

    after piper              GPU    120 MB | process RSS   1378 MB

### Kokoro

Kokoro takes phoneme ids, a style vector and a speed. The voice file holds one style vector for each possible input length, so the phoneme count selects the row. Input is limited to 510 tokens, so long text must be split first.

The model expects misaki’s phoneme alphabet. This class runs espeak-ng and converts the output with the table from misaki’s own espeak fallback. That is the fallback route, not the main misaki route, and section 9 measures the difference.

------------------------------------------------------------------------

``` python
KOKORO_REPO = "onnx-community/Kokoro-82M-v1.0-ONNX"

# espeak-ng spells a diphthong as two tied symbols and Kokoro's vocabulary holds one
# symbol for it. This is the US English table from misaki/espeak.py (Apache-2.0),
# longest match first. \u escapes keep the source ASCII.
E2M = [
    ("\u0294\u02ccn\u0329", "\u0294n"), ("\u0294n\u0329", "\u0294n"),
    ("a^\u026a", "I"), ("a^\u028a", "W"), ("d^\u0292", "\u02a4"), ("e^\u026a", "A"),
    ("t^\u0283", "\u02a7"), ("\u0254^\u026a", "Y"), ("\u0259^l", "\u1d4al"),
    ("\u02b2o", "jo"), ("\u02b2\u0259", "j\u0259"),
    ("e", "A"), ("\u02b2", ""), ("\u025a", "\u0259\u0279"), ("r", "\u0279"), ("x", "k"), ("\u00e7", "k"),
    ("\u0250", "\u0259"), ("\u026c", "l"), ("\u0303", ""),
]
US_FIXES = [
    ("o^\u028a", "O"), ("\u025c\u02d0\u0279", "\u025c\u0279"), ("\u025c\u02d0", "\u025c\u0279"),
    ("\u026a\u0259", "i\u0259"), ("\u02d0", ""), ("o", "\u0254"), ("\u027e", "T"), ("\u0294", "t"), ("^", ""),
]


class Kokoro:
    "Kokoro on plain ONNX Runtime: phonemes -> ids -> StyleTTS 2 decoder -> waveform at 24 kHz."

    sr = 24000
    max_tokens = 510

    def __init__(self, filename="model.onnx", voice="af_heart", threads=4, providers=PROVIDERS):
        self.path = Path(hf_hub_download(KOKORO_REPO, f"onnx/{filename}", cache_dir=HF_CACHE))
        tokenizer = json.loads(Path(hf_hub_download(KOKORO_REPO, "tokenizer.json", cache_dir=HF_CACHE)).read_text())
        self.vocab = tokenizer["model"]["vocab"]
        self.g2p = EspeakBackend("en-us", preserve_punctuation=True, with_stress=True, tie="^")
        self.session = session(self.path, threads, providers)
        self.voice = self.load_voice(voice)

    @staticmethod
    def load_voice(name):
        "One 256-dim style vector per input length, 510 of them."
        path = hf_hub_download(KOKORO_REPO, f"voices/{name}.bin", cache_dir=HF_CACHE)
        return np.fromfile(path, dtype=np.float32).reshape(-1, 1, 256)

    def phonemes(self, text):
        out = self.g2p.phonemize([text], strip=True)[0]
        for old, new in E2M:
            out = out.replace(old, new)
        out = re.sub(r"(\S)\u0329", "\u1d4a\\1", out)
        for old, new in US_FIXES:
            out = out.replace(old, new)
        return out

    def from_phonemes(self, phonemes, speed=1.0, voice=None):
        ids = [self.vocab[ch] for ch in phonemes if ch in self.vocab]
        if len(ids) > self.max_tokens:
            raise ValueError(f"{len(ids)} tokens is over the {self.max_tokens} limit, so split the text first")
        style = (self.voice if voice is None else voice)[len(ids)]
        return self.session.run(None, {
            "input_ids": np.array([[0, *ids, 0]], dtype=np.int64),
            "style": style,
            "speed": np.array([speed], dtype=np.float32),
        })[0].squeeze()

    def synth(self, text, speed=1.0, voice=None):
        return self.from_phonemes(self.phonemes(text), speed, voice)


t0 = time.perf_counter()
kokoro = Kokoro()
print(f"load: {time.perf_counter() - t0:.2f}s, {kokoro.path.stat().st_size / 1e6:.0f} MB on disk, running on {kokoro.session.get_providers()[0]}")
print("inputs:", [(i.name, i.type) for i in kokoro.session.get_inputs()])
print("voice shape:", kokoro.voice.shape)
print("phonemes:", kokoro.phonemes(TEXT))

t0 = time.perf_counter()
audio = kokoro.synth(TEXT)
seconds = time.perf_counter() - t0
print(f"{len(audio) / kokoro.sr:.2f}s of audio at {kokoro.sr} Hz in {seconds:.3f}s")
player_table(["Kokoro af_heart"], [(TEXT, [(audio, kokoro.sr)])])

del kokoro
free_memory()
vram("after kokoro")
```

    load: 1.51s, 326 MB on disk, running on CUDAExecutionProvider
    inputs: [('input_ids', 'tensor(int64)'), ('style', 'tensor(float)'), ('speed', 'tensor(float)')]
    voice shape: (510, 1, 256)
    phonemes: ðə bˈɜɹʧ kənˈu slˈɪd ɔnðə smˈuð plˈæŋks.
    2.75s of audio at 24000 Hz in 0.115s

| Text                                       | Kokoro af_heart |
|--------------------------------------------|-----------------|
| The birch canoe slid on the smooth planks. |                 |

    after kokoro             GPU    122 MB | process RSS   1506 MB

## 9. Examples

### Everyday sentences

The same text through both engines. Listen for rhythm, and for how the question ends. Whisper and the metrics cannot judge this part.

------------------------------------------------------------------------

``` python
PLAIN = [
    "Your meeting with the design team has moved to the afternoon, so you have about an hour free before lunch.",
    "Are you coming to the meeting, or should we start without you?",
    "Turn left at the next junction, then keep right and follow the signs to the airport.",
    "The old lighthouse stood alone at the edge of the cliff, and nobody had climbed its stairs for years.",
]
piper, kokoro = Piper(LESSAC), Kokoro()
player_table(["Piper lessac medium", "Kokoro af_heart"],
             [(t, [(speak(piper, t), piper.sr), (speak(kokoro, t), kokoro.sr)]) for t in PLAIN])
```

| Text | Piper lessac medium | Kokoro af_heart |
|----|----|----|
| Your meeting with the design team has moved to the afternoon, so you have about an hour free before lunch. |  |  |
| Are you coming to the meeting, or should we start without you? |  |  |
| Turn left at the next junction, then keep right and follow the signs to the airport. |  |  |
| The old lighthouse stood alone at the edge of the cliff, and nobody had climbed its stairs for years. |  |  |

### Hard text: numbers, abbreviations, homographs and product names

This is the most useful comparison on the page. Each row plays three versions: Piper, Kokoro with espeak-ng phonemes, and Kokoro with the phonemes misaki 0.9.4 produced on Python 3.12, stored in the cell because misaki cannot install here. Kokoro through its official `kokoro` package uses misaki, so the third column is the fair one for Kokoro. The second column is what `kokoro-onnx`-style setups get when they use espeak.

------------------------------------------------------------------------

``` python
from num2words import num2words
from transformers import logging as transformers_logging
from transformers import pipeline

transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()

# Kokoro's own text front end is misaki, which pins Python below 3.13 and so cannot be
# installed in this 3.14 venv. These strings are what misaki 0.9.4 printed on Python 3.12
# for the sentences in HARD, kept so this page can play its output without importing it.
HARD = {'money': 'The invoice for $3.50 is due on March 3rd, 2024, at 10:30 am.', 'abbrev': 'Dr. Smith lives at 12 St. James St. near the NASA office.', 'homograph': 'I read the book yesterday, and I will lead the team that reads the lead report.', 'tech': 'Kubernetes pods talk to PostgreSQL through PgBouncer on Ubuntu.', 'question': 'Are you coming to the meeting, or should we start without you?'}
MISAKI = {'money': '\xf0i \u02c8\u026anv\u02ccYs f\u0254\u0279 \u03b8\u0279\u02c8i d\u02c8\u0251l\u0259\u0279z \xe6nd f\u02c8\u026afti s\u02c8\u025bnts \u026az d\u02c8u \u02cc\u0254n m\u02c8\u0251\u0279\u02a7 \u03b8\u02c8\u025c\u0279d, tw\u02c8\u025bnti tw\u02c8\u025bnti f\u02c8\u0254\u0279, \xe6t t\u02c8\u025bn:\u03b8\u02c8\u025c\u0279Ti \u02ccA\u02c8\u025bm.', 'abbrev': 'd\u02c8\u0251kt\u0259\u0279 sm\u02c8\u026a\u03b8 l\u02c8\u026avz \xe6t tw\u02c8\u025blv s\u0259nt. \u02a4\u02c8Amz s\u0259nt. n\u02cc\u026a\u0279 \xf0\u0259 n\u02c8\xe6s\u0259 \u02c8\u0254f\u0259s.', 'homograph': '\u02ccI \u0279\u02c8\u025bd \xf0\u0259 b\u02c8\u028ak j\u02c8\u025bst\u0259\u0279d\u02ccA, \xe6nd \u02ccI w\u026al l\u02c8id \xf0\u0259 t\u02c8im \xf0\xe6t \u0279\u02c8idz \xf0\u0259 l\u02c8id \u0279\u0259p\u02c8\u0254\u0279t.', 'tech': 'k\u02ccub\u0259\u0279n\u02c8\u025bTiz p\u02c8\u0251dz t\u02c8\u0254k t\u0259 p\u02c8Ost\u0261\u0259\u0279\u0279 \u02cc\u025bskj\u02ccu\u02c8\u025bl \u03b8\u0279u p\u02cci\u02a4\u02c8i b\u02c8Wns\u0259\u0279 \u02cc\u0254n ub\u02c8untu.', 'question': '\u02cc\u0251\u0279 ju k\u02c8\u028cm\u026a\u014b t\u0259 \xf0\u0259 m\u02c8iT\u026a\u014b, \u0254\u0279 \u0283\u02cc\u028ad wi st\u02c8\u0251\u0279t w\u026a\xf0\u02c8Wt ju?'}


def spell_money(text):
    "A one-rule text normaliser: $3.50 -> three dollars and fifty cents."
    return re.sub(r"\$(\d+)\.(\d\d)",
                  lambda m: f"{num2words(int(m[1]))} dollars and {num2words(int(m[2]))} cents", text)


def load_asr():
    return pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device,
                    dtype=torch.float16 if device != "cpu" else torch.float32,
                    model_kwargs={"cache_dir": HF_CACHE})


def transcribe(asr, audio, sr):
    import librosa  # whisper wants 16 kHz
    clip = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=16000)
    return asr({"array": clip, "sampling_rate": 16000},
               generate_kwargs={"language": "en", "task": "transcribe"})["text"].strip()


rows, heard = [], []
for key, text in HARD.items():
    clips = {
        "Piper": (speak(piper, text), piper.sr),
        "Kokoro, espeak": (speak(kokoro, text), kokoro.sr),
        "Kokoro, misaki": (kokoro.from_phonemes(MISAKI[key]), kokoro.sr),
    }
    rows.append((text, list(clips.values())))
    heard.append((key, clips))
    print(f"{key}\n  espeak: {piper.phonemes(text)}\n  misaki: {MISAKI[key]}")

player_table(["Piper", "Kokoro, espeak", "Kokoro, misaki"], rows)

asr = load_asr()
print("\nWhat Whisper heard:")
for key, clips in heard:
    print(key)
    for name, (clip, sr) in clips.items():
        print(f"  {name:16s}: {transcribe(asr, clip, sr)}")

del asr, rows, heard, clips, clip
free_memory()
```

    money
      espeak: ðɪ ˈɪnvɔɪs fɔːɹ dˈɑːlɚ θɹˈiː pɔɪnt fˈaɪv zˈiəɹoʊ ɪz dˈuː ˌɔn mˈɑːɹtʃ θˈɜːd, tˈuː θˈaʊzənd twˈɛnti fˈɔːɹ, æt tˈɛn:θˈɜːɾi ˌeɪˈɛm.
      misaki: ði ˈɪnvˌYs fɔɹ θɹˈi dˈɑləɹz ænd fˈɪfti sˈɛnts ɪz dˈu ˌɔn mˈɑɹʧ θˈɜɹd, twˈɛnti twˈɛnti fˈɔɹ, æt tˈɛn:θˈɜɹTi ˌAˈɛm.
    abbrev
      espeak: dˈɑːktɚ. smˈɪθ lˈaɪvz æt twˈɛlv sənt. dʒˈeɪmz sənt. nˌɪɹ ðə nˈæsɐ ˈɑːfɪs.
      misaki: dˈɑktəɹ smˈɪθ lˈɪvz æt twˈɛlv sənt. ʤˈAmz sənt. nˌɪɹ ðə nˈæsə ˈɔfəs.
    homograph
      espeak: aɪ ɹˈiːd ðə bˈʊk jˈɛstɚdˌeɪ, ænd aɪ wɪl lˈiːd ðə tˈiːm ðæt ɹˈiːdz ðə lˈiːd ɹᵻpˈɔːɹt.
      misaki: ˌI ɹˈɛd ðə bˈʊk jˈɛstəɹdˌA, ænd ˌI wɪl lˈid ðə tˈim ðæt ɹˈidz ðə lˈid ɹəpˈɔɹt.
    tech
      espeak: kˌuːbɚnˈɛɾiːz pˈɑːdz tˈɔːk tə pˈoʊstɡɚɹ ˌɛskjˌuːˈɛl θɹuː pˌiːdʒˈiː bˈaʊnsɚɹ ˌɔn uːbˈuːntuː.
      misaki: kˌubəɹnˈɛTiz pˈɑdz tˈɔk tə pˈOstɡəɹɹ ˌɛskjˌuˈɛl θɹu pˌiʤˈi bˈWnsəɹ ˌɔn ubˈuntu.
    question
      espeak: ɑːɹ juː kˈʌmɪŋ tə ðə mˈiːɾɪŋ, ɔːɹ ʃˌʊd wiː stˈɑːɹt wɪðˈaʊt juː?
      misaki: ˌɑɹ ju kˈʌmɪŋ tə ðə mˈiTɪŋ, ɔɹ ʃˌʊd wi stˈɑɹt wɪðˈWt ju?

| Text | Piper | Kokoro, espeak | Kokoro, misaki |
|----|----|----|----|
| The invoice for \$3.50 is due on March 3rd, 2024, at 10:30 am. |  |  |  |
| Dr. Smith lives at 12 St. James St. near the NASA office. |  |  |  |
| I read the book yesterday, and I will lead the team that reads the lead report. |  |  |  |
| Kubernetes pods talk to PostgreSQL through PgBouncer on Ubuntu. |  |  |  |
| Are you coming to the meeting, or should we start without you? |  |  |  |

    What Whisper heard:
    money
      Piper           : The invoice for $3.50 is due on March 3, 2024, at 10.30 am.
      Kokoro, espeak  : The invoice for $3.50 is due on March 3, 2024 at 10.30 a.m.
      Kokoro, misaki  : The invoice for $3.50 is due on March 3, 2024, at 10.30 a.m.
    abbrev
      Piper           : Dr. Smith lives at Twelfth St. James St. near the Nass Office.
      Kokoro, espeak  : Dr. Smith lives at 12-cent James-cent, near the NASA office.
      Kokoro, misaki  : Dr. Smith lives at 12-cent James-cent, near the NASA office.
    homograph
      Piper           : I read the book yesterday, and I will lead the team that reads the lead report.
      Kokoro, espeak  : Read the book yesterday, and we'll lead the team that reads the lead report.
      Kokoro, misaki  : I read the book yesterday, and I will lead the team that reads the lead report.
    tech
      Piper           : Kubernetes pods talk to PostgreSQL through PG Bouncer on Ubuntu.
      Kokoro, espeak  : Kubernetes pods talk to Postgre SQL through PG Bouncer on Ubuntu.
      Kokoro, misaki  : Kubernetes pods talk to Postgre SQL through PG Bouncer on Ubuntu.
    question
      Piper           : Are you coming to the meeting, or should we start without you?
      Kokoro, espeak  : Are you coming to the meeting, or should we start without you?
      Kokoro, misaki  : Are you coming to the meeting, or should we start without you?

**What the phonemes above show.** Whisper hides several of these differences, so read the phonemes as well as the transcripts.

| Case | espeak-ng (Piper, and Kokoro’s fallback) | misaki (Kokoro’s official path) |
|----|----|----|
| `$3.50` | “dollar three point five zero”: wrong | “three dollars and fifty cents”: right |
| `2024` | “two thousand twenty four” | “twenty twenty four”: better for a year |
| `lives` (verb) | Rhymes with “knives”: wrong | Rhymes with “gives”: right |
| `read` (past tense) | “reed”: wrong | “red”: right |
| `lead the team`, `lead report` | “leed”, both correct | “leed”, both correct |
| `St. James St.` | “Saint James Saint”: second one wrong | Same mistake |
| `PostgreSQL`, `PgBouncer` | Spelled badly | Same mistake |

So **misaki is ahead on ordinary English, and both fail on domain words**. Whisper also dropped the opening “I” in the “Kokoro, espeak” clip of the homograph sentence, but not in the misaki clip. That is one clip, so treat it as a lead and not a finding.

Piper has no better G2P to switch to, but its error list can be cut with a normaliser. The next cell shows one rule fixing the money case. A real product needs a proper normaliser plus a pronunciation table for its own words.

------------------------------------------------------------------------

### Speed and voice controls

Piper changes speed through `length_scale`, where larger is slower. Kokoro uses `speed`, where larger is faster.

Voices are where the designs differ most. A Kokoro voice is an array, so blending two is a line of arithmetic and costs nothing. A Piper voice is a whole model file, so each one costs a download and a session. The third table plays the three Piper voices. Remember that `ryan` is non-commercial.

------------------------------------------------------------------------

``` python
# The one-rule normaliser fixes the money reading for every espeak path
print("before:", piper.phonemes(HARD["money"]))
print("after :", piper.phonemes(spell_money(HARD["money"])))
player_table(["Piper, money spelled", "Kokoro, money spelled"],
             [(spell_money(HARD["money"]), [(speak(piper, spell_money(HARD["money"])), piper.sr),
                                            (speak(kokoro, spell_money(HARD["money"])), kokoro.sr)])])

SPEECH = "Speech rate and voice are the two controls most products expose first."
slow, fast = speak(piper, SPEECH, length_scale=1.25), speak(piper, SPEECH, length_scale=0.8)
kslow, kfast = speak(kokoro, SPEECH, speed=0.8), speak(kokoro, SPEECH, speed=1.25)
player_table(["Piper slow", "Piper fast", "Kokoro slow", "Kokoro fast"],
             [(SPEECH, [(slow, piper.sr), (fast, piper.sr), (kslow, kokoro.sr), (kfast, kokoro.sr)])])

# Kokoro: voices are arrays, so a new voice is one line of arithmetic
bella, michael = Kokoro.load_voice("af_bella"), Kokoro.load_voice("am_michael")
blend = 0.5 * bella + 0.5 * michael
player_table(["af_heart", "af_bella", "am_michael", "50/50 bella + michael"],
             [(SPEECH, [(speak(kokoro, SPEECH), kokoro.sr), (speak(kokoro, SPEECH, voice=bella), kokoro.sr),
                        (speak(kokoro, SPEECH, voice=michael), kokoro.sr), (speak(kokoro, SPEECH, voice=blend), kokoro.sr)])])

del piper, kokoro, slow, fast, kslow, kfast, bella, michael, blend
free_memory()

# Piper: a voice is a whole model file, so each one costs a download and a session
PIPER_VOICES = {"lessac medium": LESSAC,
                "ljspeech high": "en/en_US/ljspeech/high/en_US-ljspeech-high",
                "ryan high": "en/en_US/ryan/high/en_US-ryan-high"}
clips = []
for name, stem in PIPER_VOICES.items():
    voice = Piper(stem)
    clips.append((speak(voice, SPEECH), voice.sr))
    del voice
    free_memory()
player_table(list(PIPER_VOICES), [(SPEECH, clips)])
del clips
vram("after voices")
```

    before: ðɪ ˈɪnvɔɪs fɔːɹ dˈɑːlɚ θɹˈiː pɔɪnt fˈaɪv zˈiəɹoʊ ɪz dˈuː ˌɔn mˈɑːɹtʃ θˈɜːd, tˈuː θˈaʊzənd twˈɛnti fˈɔːɹ, æt tˈɛn:θˈɜːɾi ˌeɪˈɛm.
    after : ðɪ ˈɪnvɔɪs fɔːɹ θɹˈiː dˈɑːlɚz ænd fˈɪfti sˈɛnts ɪz dˈuː ˌɔn mˈɑːɹtʃ θˈɜːd, tˈuː θˈaʊzənd twˈɛnti fˈɔːɹ, æt tˈɛn:θˈɜːɾi ˌeɪˈɛm.

| Text | Piper, money spelled | Kokoro, money spelled |
|----|----|----|
| The invoice for three dollars and fifty cents is due on March 3rd, 2024, at 10:30 am. |  |  |

| Text | Piper slow | Piper fast | Kokoro slow | Kokoro fast |
|----|----|----|----|----|
| Speech rate and voice are the two controls most products expose first. |  |  |  |  |

| Text | af_heart | af_bella | am_michael | 50/50 bella + michael |
|----|----|----|----|----|
| Speech rate and voice are the two controls most products expose first. |  |  |  |  |

| Text | lessac medium | ljspeech high | ryan high |
|----|----|----|----|
| Speech rate and voice are the two controls most products expose first. |  |  |  |

    after voices             GPU    182 MB | process RSS   2529 MB

## 10. Head-to-Head Benchmark

**Method.** Each configuration gets the same five-sentence paragraph, synthesised sentence by sentence as a streaming system would. One model is live at a time. RTF and first-audio time are the median of three runs after a warm-up call. `cold_first_s` is the warm-up call itself.

**Hardware.** Proxmox LXC with 4 vCPU and an RTX 3060 12 GB, ONNX Runtime 1.31.1, measured on 2026-11-04. The box was running other work, so read differences under about 20% as noise. This is a smoke test, not a leaderboard. One thread on this x86 CPU is **not** a Raspberry Pi, and no Pi was measured.

------------------------------------------------------------------------

``` python
PARAGRAPH = (
    "The storm reached the coast shortly after midnight. Power lines came down across the valley, "
    "and most of the town woke up without electricity. Crews worked through the morning to clear "
    "the roads and repair the damage. By early afternoon, the first homes had their lights back on. "
    "Officials asked people to stay indoors until the wind dropped."
)
ENGINES = {
    "Piper medium (lessac)": lambda t, p: Piper(LESSAC, t, p),
    "Piper high (ljspeech)": lambda t, p: Piper(PIPER_VOICES["ljspeech high"], t, p),
    "Kokoro fp32": lambda t, p: Kokoro("model.onnx", "af_heart", t, p),
    "Kokoro q8": lambda t, p: Kokoro("model_quantized.onnx", "af_heart", t, p),
}
RUNS = {"CPU 1 thread": (1, CPU), "CPU 4 threads": (4, CPU)}
if has_cuda:
    RUNS["CUDA"] = (4, CUDA)

parts = sentences(PARAGRAPH)
results = []
for engine_name, build in ENGINES.items():
    for run_name, (threads, providers) in RUNS.items():
        free_memory()
        rss0, gpu0 = rss_mb(), gpu_mb()
        t0 = time.perf_counter()
        engine = build(threads, providers)
        load_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        engine.synth(parts[0])
        cold_s = time.perf_counter() - t0
        first, total = [], []
        for _ in range(3):
            t0 = time.perf_counter()
            clip = engine.synth(parts[0])
            first.append(time.perf_counter() - t0)
            for part in parts[1:]:
                clip = np.concatenate([clip, engine.synth(part)])
            total.append(time.perf_counter() - t0)
        audio_s = len(clip) / engine.sr
        results.append({
            "engine": engine_name, "run": run_name,
            "rtf": round(statistics.median(total) / audio_s, 3),
            "first_audio_s": round(statistics.median(first), 3),
            "cold_first_s": round(cold_s, 3),
            "load_s": round(load_s, 2),
            "audio_s": round(audio_s, 1),
            "rss_mb": round(rss_mb() - rss0),
            "gpu_mb": gpu_mb() - gpu0,
            "disk_mb": round(engine.path.stat().st_size / 1e6),
        })
        del engine, clip
        free_memory()

bench = pd.DataFrame(results)
display(bench)
```

|  | engine | run | rtf | first_audio_s | cold_first_s | load_s | audio_s | rss_mb | gpu_mb | disk_mb |
|----|----|----|----|----|----|----|----|----|----|----|
| 0 | Piper medium (lessac) | CPU 1 thread | 0.054 | 0.144 | 0.157 | 1.39 | 18.0 | 220 | 0 | 63 |
| 1 | Piper medium (lessac) | CPU 4 threads | 0.032 | 0.079 | 0.163 | 1.43 | 18.0 | 200 | 0 | 63 |
| 2 | Piper medium (lessac) | CUDA | 0.028 | 0.097 | 0.121 | 1.48 | 17.6 | 170 | 526 | 63 |
| 3 | Piper high (ljspeech) | CPU 1 thread | 0.433 | 1.218 | 1.147 | 1.63 | 19.7 | 266 | 0 | 114 |
| 4 | Piper high (ljspeech) | CPU 4 threads | 0.269 | 0.742 | 0.837 | 1.88 | 19.7 | 274 | 0 | 114 |
| 5 | Piper high (ljspeech) | CUDA | 0.031 | 0.131 | 0.156 | 1.58 | 20.1 | 247 | 526 | 114 |
| 6 | Kokoro fp32 | CPU 1 thread | 0.476 | 1.449 | 1.547 | 1.89 | 21.4 | 503 | 0 | 326 |
| 7 | Kokoro fp32 | CPU 4 threads | 0.267 | 0.805 | 0.817 | 1.62 | 21.4 | 596 | 0 | 326 |
| 8 | Kokoro fp32 | CUDA | 0.029 | 0.104 | 0.122 | 1.36 | 21.4 | 380 | 1040 | 326 |
| 9 | Kokoro q8 | CPU 1 thread | 0.359 | 1.060 | 1.054 | 1.29 | 21.3 | 356 | 0 | 92 |
| 10 | Kokoro q8 | CPU 4 threads | 0.234 | 0.707 | 0.749 | 1.40 | 21.3 | 357 | 0 | 92 |
| 11 | Kokoro q8 | CUDA | 0.176 | 0.544 | 0.590 | 1.43 | 21.4 | 250 | 274 | 92 |

``` python
from pyecharts.globals import CurrentConfig

# assets.pyecharts.org serves an expired TLS certificate from most nodes (2026-11-11),
# so charts failed to load. jsDelivr mirrors the same assets, pinned to one commit.
CurrentConfig.ONLINE_HOST = "https://cdn.jsdelivr.net/gh/pyecharts/pyecharts-assets@f2328d558550046fc5449672a6850e848e141590/assets/v6/"

from pyecharts import options as opts
from pyecharts.charts import Bar


def bar_chart(title, frame, value, y_name):
    "One bar series per engine, grouped by run."
    chart = Bar().add_xaxis(list(RUNS))
    for name, group in frame.groupby("engine", sort=False):
        chart.add_yaxis(name, [float(v) for v in group[value]], label_opts=opts.LabelOpts(position="top", font_size=10))
    return chart.set_global_opts(
        title_opts=opts.TitleOpts(title=title),
        legend_opts=opts.LegendOpts(pos_top="8%"),
        yaxis_opts=opts.AxisOpts(name=y_name),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
    )


bar_chart("Real-time factor (lower is faster)", bench, "rtf", "RTF").render_notebook()
```

``` python
bar_chart("Time to first audio, warm (lower is better)", bench, "first_audio_s", "seconds").render_notebook()
```

**Reading the speed results.**

- **On CPU, Piper medium is about 8 times faster than Kokoro fp32** at 4 threads (0.032 vs 0.267), and its first audio arrives 10 times sooner at 1 thread (0.14 s vs 1.45 s).
- **Piper high costs almost as much as Kokoro on CPU** (0.269 vs 0.267), so choosing it gives up Piper’s main advantage. Its parameter count is under twice medium’s, yet it runs about 8 times slower.
- **On CUDA the gap closes.** All three fp32 models sit near RTF 0.03, about 30 times real time.
- **Do not put Kokoro q8 on CUDA.** It runs at 0.176, about 6 times slower than fp32 on the same GPU. The likely cause is that its quantised operations fall back to CPU. That cause was not checked.
- **Kokoro q8 is a size choice on CPU.** It saves 234 MB of disk and about 200 MB of RAM. Its speed gain over fp32 at 4 threads (0.234 vs 0.267) is within noise.
- **VRAM.** The change in GPU memory was 526 MB for Piper and 1,040 MB for Kokoro fp32. Both include ONNX Runtime’s memory arena, so they are not the size of the weights.

------------------------------------------------------------------------

### Round-trip WER

Twenty Harvard sentences go through each engine three times, because Piper is random. Whisper small transcribes them with the same normalisation. Numbers and words with two common spellings were avoided, because they count as errors for every engine.

------------------------------------------------------------------------

``` python
HARVARD = [
    "The birch canoe slid on the smooth planks.", "Glue the sheet to the dark blue background.",
    "It's easy to tell the depth of a well.", "These days a chicken leg is a rare dish.",
    "Rice is often served in round bowls.", "The juice of lemons makes fine punch.",
    "The box was thrown beside the parked truck.", "The hogs were fed chopped corn and garbage.",
    "Four hours of steady work faced us.", "A large size in stockings is hard to sell.",
    "The boy was there when the sun rose.", "A rod is used to catch pink salmon.",
    "The source of the huge river is the clear spring.", "Kick the ball straight and follow through.",
    "Help the woman get back to her feet.", "A pot of tea helps to pass the evening.",
    "The fish twisted and turned on the bent hook.", "The soft cushion broke the man's fall.",
    "The salt breeze came across from the sea.", "The small pup gnawed a hole in the sock.",
]
refs = [norm(t) for t in HARVARD]
PASSES = 3  # Piper samples noise, so its output and its score change from run to run
asr = load_asr()
wers, misses = {}, {}
for name, build in {"Piper medium (lessac)": ENGINES["Piper medium (lessac)"],
                    "Piper high (ljspeech)": ENGINES["Piper high (ljspeech)"],
                    "Kokoro fp32 (af_heart)": ENGINES["Kokoro fp32"]}.items():
    engine = build(4, PROVIDERS)
    hyps = [norm(transcribe(asr, engine.synth(t), engine.sr)) for _ in range(PASSES) for t in HARVARD]
    wers[name] = jiwer.wer(refs * PASSES, hyps)
    misses[name] = sorted({(r, h) for r, h in zip(refs * PASSES, hyps) if r != h})
    del engine
    free_memory()

del asr
free_memory()
vram("after benchmark")

for name, value in wers.items():
    print(f"{name:24s} WER {value:6.2%}  ({len(misses[name])} distinct misheard sentences over {PASSES} passes)")
    for ref, hyp in misses[name]:
        print(f"    said : {ref}\n    heard: {hyp}")

(
    Bar()
    .add_xaxis(list(wers))
    .add_yaxis("round-trip WER %", [round(v * 100, 2) for v in wers.values()])
    .set_global_opts(title_opts=opts.TitleOpts(title="Round-trip WER, 20 Harvard sentences (lower is better)"),
                     yaxis_opts=opts.AxisOpts(name="WER %"))
    .render_notebook()
)
```

    after benchmark          GPU    182 MB | process RSS   2701 MB
    Piper medium (lessac)    WER  3.07%  (7 distinct misheard sentences over 3 passes)
        said : glue the sheet to the dark blue background
        heard: blue the sheet to the dark blue background
        said : its easy to tell the depth of a well
        heard: its easy to tell the depth of a will
        said : rice is often served in round bowls
        heard: rices often served in round bowls
        said : the birch canoe slid on the smooth planks
        heard: the birch can use slid on the smooth planks
        said : the box was thrown beside the parked truck
        heard: the box was thrown beside the park truck
        said : the small pup gnawed a hole in the sock
        heard: the small pup not a hole in the sock
        said : the source of the huge river is the clear spring
        heard: the source of a huge river is the clear spring
    Piper high (ljspeech)    WER  3.27%  (10 distinct misheard sentences over 3 passes)
        said : four hours of steady work faced us
        heard: four hours of steady work face stars
        said : glue the sheet to the dark blue background
        heard: the sheet to the dark blue background
        said : the fish twisted and turned on the bent hook
        heard: a fish twisted and turned on the bent hook
        said : the hogs were fed chopped corn and garbage
        heard: the hubs were fed chopped corn and garbage
        said : the hogs were fed chopped corn and garbage
        heard: the hulls were fed chopped corn and garbage
        said : the hogs were fed chopped corn and garbage
        heard: the hums were fed chopped corn and garbage
        said : the juice of lemons makes fine punch
        heard: juice of lemons makes fine punch
        said : the small pup gnawed a hole in the sock
        heard: a small pup nord a hole in the sock
        said : the small pup gnawed a hole in the sock
        heard: the small pup nord a hole in the sock
        said : the small pup gnawed a hole in the sock
        heard: the small pup nortahole in the sock
    Kokoro fp32 (af_heart)   WER  1.23%  (2 distinct misheard sentences over 3 passes)
        said : the box was thrown beside the parked truck
        heard: the box was thrown beside the park truck
        said : the small pup gnawed a hole in the sock
        heard: the small pup not a hole in the sock

**Reading the WER result.** Kokoro scores 1.23%, against 3.07% and 3.27% for Piper. Kokoro is deterministic, so its three passes are one sample repeated: two misheard sentences out of twenty. Both of its misses also appear for Piper medium, which suggests the sentences are hard for Whisper. The Piper errors change between passes, which fits its random sampling.

The direction is plausible, but twenty sentences and one voice per engine cannot prove a ranking. Kokoro ran here on the espeak route, so this result does not include any benefit from misaki.

------------------------------------------------------------------------

## 11. Common Frameworks

These are the packages you meet when you use either engine in production. The licence column matters here, because the weights are not the whole stack.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [kokoro](https://github.com/hexgrad/kokoro) | modelling | The official PyTorch inference package, using misaki | Apache-2.0 | You can run Python below 3.13 and want the official text front end |
| [piper1-gpl](https://github.com/OHF-Voice/piper1-gpl) | modelling | The official engine (`piper-tts`), CLI, C API, HTTP server and voice training | GPL-3.0 (copyleft, check before you distribute it) | You want the supported Piper, or need to train a voice |
| [misaki](https://github.com/hexgrad/misaki) | data | Kokoro’s G2P: dictionaries plus part-of-speech rules | Apache-2.0, but its English extra pulls in `phonemizer-fork` (GPL-3.0) | You want Kokoro’s correct homographs and money reading |
| [phonemizer](https://github.com/bootphon/phonemizer) + [espeak-ng](https://github.com/espeak-ng/espeak-ng) | data | Rule-based G2P for more than a hundred languages | GPL-3.0 both | You need a G2P that runs anywhere, as on this page |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | inference runtime | CPU and CUDA execution for both models | MIT | You want the engines without their packages |
| [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) | inference runtime | One C++ runtime with many language bindings, covering Piper and Kokoro models and also ASR | Apache-2.0 | You are shipping to mobile, embedded or browser targets |
| [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) | inference runtime | A thin Kokoro wrapper over ONNX Runtime and espeak-ng | MIT | You want the wrapper this page writes by hand, on Python below 3.14 |
| [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) | serving | A Docker server with an OpenAI-compatible API | Apache-2.0 | You want a TTS endpoint without writing one |
| [Pipecat](https://github.com/pipecat-ai/pipecat) | orchestration | A real-time voice agent pipeline, with service modules for both Piper and Kokoro | BSD-2-Clause | The TTS is one stage in a conversation |

**Default stack.** For a server with a GPU, use Kokoro through its official package on a Python 3.12 environment, behind Kokoro-FastAPI or Pipecat. For a CPU-only device, use `piper-tts` with a medium voice whose data licence you have read. For an app on mobile or embedded targets, use sherpa-onnx.

**The most common wrong turn** is to choose on voice demos and ship with no text normaliser. The demo text has no prices, dates or product names, and your users’ text does.

------------------------------------------------------------------------

## 12. Going Further

**Gotchas found while building this page:**

| Symptom | Cause | Fix |
|----|----|----|
| ONNX Runtime logs `pthread_setaffinity_np failed` per thread | It pins threads to cores, and this LXC refuses | Set `intra_op_num_threads` and `inter_op_num_threads` |
| `language "en" is not supported by the espeak backend` | The `en_US-ljspeech-high` config says `en` | Map `en` to `en-gb`, which matches official Piper |
| An `en_US` voice speaks with British vowels | Same cause: espeak treats `en` as British | Prefer a voice whose config says `en-us`, or override it and listen |
| Segfault on load or first run | `model_q8f16.onnx` on ONNX Runtime 1.31.1 here | Use `model.onnx` or `model_quantized.onnx` |
| `torch.cuda.memory_allocated()` reads 0 during synthesis | PyTorch cannot see ONNX Runtime’s allocations | Ask `nvidia-smi`, as `gpu_mb()` does |
| `empty_cache()` does not free the VRAM | The memory belongs to the ONNX Runtime session | `del` the session |
| Kokoro raises on long text | The 510-token limit | Split into sentences first |

**Fine-tuning and new voices.** Piper’s [training guide](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/TRAINING.md) covers training with PyTorch Lightning. Kokoro has no released trainer, so voice blending is the supported way to get something new.

**Pronunciation control.** Both classes on this page accept phonemes you write yourself, which is the simplest route to fixing a product name. `Kokoro.from_phonemes` takes them directly, and `Piper.ids` does the same for Piper. The `kokoro` package also accepts inline markup such as `[Kokoro](/phonemes/)`.

**Not covered here:**

- A listening test or MOS. The players are here, but no scores were collected.
- A measured run on a Raspberry Pi 5. The 1-thread column is not a stand-in for one.
- Any language other than English, on either engine.
- Kokoro through the official `kokoro` PyTorch package. This page only reuses misaki’s phonemes, with the ONNX export.

**References:** [VITS](https://arxiv.org/abs/2106.06103), [StyleTTS 2](https://arxiv.org/abs/2306.07691), [ISTFTNet](https://arxiv.org/abs/2203.02395), [Kokoro model card](https://huggingface.co/hexgrad/Kokoro-82M), [Kokoro ONNX export](https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX), [Piper voices](https://huggingface.co/rhasspy/piper-voices).

------------------------------------------------------------------------

Back to top
