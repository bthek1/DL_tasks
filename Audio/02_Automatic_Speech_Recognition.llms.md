# Automatic Speech Recognition

Everything to know about audio-to-text: how ASR works, the mid-2026 model landscape, evaluation metrics, and runnable code to test the leading open models.

Author

Benedict Thekkel

## 1. What is ASR?

Automatic Speech Recognition (ASR, also called speech-to-text / STT) maps a raw audio waveform to a text transcript.

**Input.** Audio is a 1-D array of amplitude samples. Almost every ASR model expects **16 kHz mono** audio; resample anything else first. Models do not consume the waveform directly - it is converted to a **log-mel spectrogram** (a time x frequency image of energy), which the neural encoder ingests.

**Output.** Plain text, optionally with:

- **Timestamps** - per segment or per word (needed for subtitles, editing tools)
- **Language ID** - which language was spoken
- **Translation** - some models (Whisper) can emit English text from non-English speech in one pass

**Neighbouring tasks** (separate models, often chained with ASR):

| Task | What it does | Typical tool |
|----|----|----|
| Voice activity detection (VAD) | Find where speech is | Silero VAD |
| Speaker diarization | Who spoke when | pyannote.audio |
| Forced alignment | Align a known transcript to audio | Qwen3-ForcedAligner, MFA |
| Text-to-speech (TTS) | The inverse task | see `00_Text_to_Speech` |

------------------------------------------------------------------------

## 2. Real-World Use Cases

ASR is the most widely deployed audio task, and the deployments differ so much that “which model is best” has no single answer - a meeting recorder and a wake word both do speech-to-text, and they share almost nothing.

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Meeting notes and transcription | Productivity SaaS (Zoom, Teams, Otter) | Multi-speaker meeting audio -\> diarised transcript, summary | Accuracy on crosstalk and accents; cost per hour at scale; batch is fine |
| Voice assistants and voice typing | Consumer devices | Short mic utterance -\> text for the NLU stage | Latency (a few hundred ms), streaming, on-device size, privacy |
| Contact-centre analytics and compliance | Telecom, finance, insurance | 8 kHz telephony recordings -\> transcripts for QA, redaction | Robustness at 8 kHz under noise; cost across millions of hours; PII handling |
| Subtitling and captioning | Broadcast, streaming (YouTube auto-captions) | Long-form audio -\> timestamped subtitles | Word-level timestamps; stability over hours; names spelled right |
| Clinical documentation | Healthcare (ambient scribes) | Doctor-patient conversation -\> structured note | Medical vocabulary and drug names; on-prem or HIPAA-compliant hosting |
| Live captioning | Accessibility (Android Live Caption) | Streaming audio -\> incremental text | Streaming latency, on-device, no distracting revisions |
| In-car voice control | Automotive | Cabin mic under road noise -\> command | Fully offline; far-field noise robustness |

**What the LibriSpeech number hides.** Sub-3% WER on read audiobooks says nothing about the audio you will actually be handed: accented, dysarthric and code-switched speech, far-field and reverberant rooms, 8 kHz telephony, overlapping speakers, and domain jargon the model has never seen. **Streaming vs batch is an architectural fork, not a setting** - Whisper’s 30-second attention window cannot stream by construction, while RNN-T/TDT decoders can; picking a leaderboard winner without checking this is the most common way an ASR project goes wrong. **Edge vs server** is equally hard: Moonshine runs on a phone-class CPU, Qwen3-ASR needs a GPU. And the failure modes that hurt in production are not “a slightly higher WER” but the confident ones - **hallucinated sentences over silence or music** (a known Whisper behaviour), dropped segments in long audio, timestamp drift, and proper nouns that come back fluent and wrong. Downstream systems trust the transcript, so a fluent error propagates further than an obvious one.

------------------------------------------------------------------------

## 3. How Modern ASR Works

Four architectural eras, each still alive in some niche:

1.  **HMM/GMM pipelines (Kaldi era, pre-2016).** Hand-built acoustic + pronunciation + language models. Legacy - only relevant for understanding older literature.
2.  **Self-supervised encoders + CTC (wav2vec 2.0, 2020).** Pretrain an encoder on unlabeled audio, fine-tune with a CTC head. CTC predicts a character/token per frame independently - extremely fast, slightly weaker on punctuation and context.
3.  **Attention encoder-decoder (Whisper, 2022).** A transformer decoder attends over the full encoded audio - best robustness and formatting, but slow (autoregressive) and offline-only by design (fixed 30 s windows).
4.  **Transducers and LLM decoders (2024-2026).**
    - **RNN-T / TDT** (Parakeet): streams natively, near-attention accuracy at 10-50x the speed.
    - **SALM / LLM-decoder** (Canary-Qwen, Granite Speech, Qwen3-ASR): bolt a speech encoder onto a pretrained LLM. Current accuracy leaders - the LLM brings world knowledge to disambiguate hard audio.

Nearly all modern encoders are **Conformers** (convolution + self-attention) or close variants.

**Decoder trade-off cheat sheet:**

| Decoder     | Accuracy  | Speed     | Streaming | Example                |
|-------------|-----------|-----------|-----------|------------------------|
| CTC         | good      | fastest   | yes       | wav2vec 2.0            |
| TDT / RNN-T | very good | very fast | yes       | Parakeet TDT           |
| Attention   | very good | slow      | no        | Whisper                |
| LLM (SALM)  | best      | slowest   | partial   | Canary-Qwen, Qwen3-ASR |

------------------------------------------------------------------------

## 4. Evaluation Metrics

**Word Error Rate (WER)** - the standard metric. Edit distance between hypothesis and reference, normalised by reference length:

$$WER = \frac{S + D + I}{N}$$

where S = substitutions, D = deletions, I = insertions, N = reference words. Lower is better; can exceed 100%.

**Pitfalls:**

- **Normalisation dominates.** Casing, punctuation and number formatting (“25” vs “twenty five”) swing WER by several points. Always compare models under the *same* text normalisation (the Open ASR Leaderboard uses the Whisper normaliser).
- **CER** (character error rate) is used for languages without whitespace word boundaries (Chinese, Japanese).
- **RTFx** (inverse real-time factor) = audio duration / processing time. RTFx 100 means 1 hour of audio transcribes in 36 s. The accuracy/speed trade-off is the key engineering decision.

`jiwer` is the standard Python package for WER:

------------------------------------------------------------------------

``` python
import jiwer

reference = "the quick brown fox jumps over the lazy dog"
hypothesis = "the quick brown fox jumped over a lazy dog"

print("WER:", jiwer.wer(reference, hypothesis))

out = jiwer.process_words(reference, hypothesis)
print(f"substitutions={out.substitutions} deletions={out.deletions} insertions={out.insertions}")
```

    WER: 0.2222222222222222
    substitutions=2 deletions=0 insertions=0

## 5. Datasets

ASR ground truth is a human transcript aligned to audio, and how that transcript was made decides what WER means. Audiobook sets pair read speech with the book text, so the reference is clean but the speech is easy. Web and meeting sets have spontaneous, overlapping or accented speech, but their transcripts differ in casing, punctuation, fillers and number style. The typical failure is a model that scores near-zero WER on LibriSpeech and several times worse on meetings or phone calls, plus “errors” that are only normalisation differences.

| Dataset | Contents | Size | Scope | License | Typical use |
|----|----|----|----|----|----|
| [LibriSpeech](https://huggingface.co/datasets/openslr/librispeech_asr) | LibriVox audiobooks, 16 kHz, split into “clean” and “other” | ~1,000 h (train 100 / 360 / 500 h + dev/test) | English, read | CC BY 4.0 | The classic train and eval set; **used in this notebook** (73-clip dummy subset) |
| [Common Voice](https://datacollective.mozillafoundation.org) | Crowd-sourced read sentences, many accents and mics | 38,932 h total / 25,886 h validated (Scripted Speech 24.0) | 289 languages | CC0 | Multilingual and accent training and eval |
| [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech) | Audiobooks, podcasts and YouTube | 10,000 h (XS 10 h to XL 10,000 h); dev 12 h, test 40 h | English | Non-commercial research and education (Hub tag says Apache 2.0) | Large-scale training; mixed-domain eval |
| [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli) | European Parliament speeches | ~1.8K h transcribed (16 languages); ~400K h unlabelled | European languages | CC0 | Formal speech; accented English (29 h, 15 L2 accents) |
| [TED-LIUM 3](https://www.openslr.org/51/) | TED talks with aligned automatic transcripts; manual dev/test | 452 h, 2,351 talks | English, prepared talks | CC BY-NC-ND 3.0 | Lecture-style eval |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | The same 2,009 FLoRes sentences read aloud in every language | ~10 h train per language | 102 languages | CC BY 4.0 | **The** multilingual eval; directly comparable across languages |
| [AMI](https://huggingface.co/datasets/edinburghcstr/ami) | Recorded meetings, close-talk and far-field mics | 100 h | English, many non-native speakers | CC BY 4.0 | Conversational and far-field eval |
| [Earnings-22](https://huggingface.co/datasets/distil-whisper/earnings22) ([official](https://github.com/revdotcom/speech-datasets/tree/main/earnings22)) | Company earnings calls | ~119 h, 125 files | English, 27 countries’ accents | CC BY-SA 4.0 (official repo; HF mirror says CC BY 4.0) | Accented, jargon-heavy eval |
| [ESB](https://huggingface.co/datasets/esb/datasets) | Benchmark bundle: LibriSpeech, Common Voice, VoxPopuli, TED-LIUM, GigaSpeech, SPGISpeech, Earnings-22, AMI | 8 sets | English | Per component (includes CC BY-NC-ND and Kensho terms) | Basis of the Open ASR Leaderboard test sets |

**This notebook evaluates on LibriSpeech**, via `hf-internal-testing/librispeech_asr_dummy` (the `clean` config, 73 short `validation` clips with reference transcripts). The benchmark cell scores the first 20 clips (`N = 20`). It also downloads one ~14 s English sample clip from the Qwen3-ASR repo for the single-model demos. This is clean, read English: the easiest case there is. Read the results as a smoke test, and use AMI, Earnings-22 or the full LibriSpeech `test-other` for numbers that mean something.

**GigaSpeech is gated** (accept terms on the Hub plus a separate request form) and its terms limit use to **non-commercial research and education**, whatever the Hub license tag says. **ESB is gated** too, because three of its parts (Common Voice, GigaSpeech, SPGISpeech) have their own terms. **TED-LIUM 3 is non-commercial and no-derivatives.** Common Voice left the Hugging Face Hub in October 2025 and is now only on Mozilla Data Collective, which needs an account and API credentials.

Downloads land in `DL_tasks/datasets/` via `cache_dir` (gitignored).

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

Whisper is no longer the accuracy leader, but remains the ecosystem default. The [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) is the reference ranking (avg WER over 8+ English test sets).

| Model | Params | License | Languages | Decoder | Best for |
|----|----|----|----|----|----|
| NVIDIA Canary-Qwen 2.5B | 2.5B | CC-BY-4.0 | en | SALM | max English accuracy (~5.6% avg WER) |
| IBM Granite Speech | 2-8B | Apache 2.0 | en + few | LLM | accuracy per parameter (~5.3% WER) |
| NVIDIA Parakeet TDT | 0.6-1.1B | CC-BY-4.0 | en (v2), 25 (v3) | TDT | bulk transcription (RTFx ~2000) |
| Qwen3-ASR | 0.6B / 1.7B | Apache 2.0 | 52 (incl. dialects) | LLM | multilingual open SOTA, language ID |
| Voxtral Transcribe 2 | 4B | Apache 2.0 | 13 | streaming | native real-time streaming |
| Whisper large-v3 | 1.5B | MIT | 99+ | attention | long-tail languages, ecosystem |
| Moonshine | 27-60M | MIT | en | attention | edge / embedded / Raspberry Pi |

**License nuance.** MIT/Apache models are open source in the full sense. NVIDIA’s CC-BY-4.0 is a content license - commercial use is fine but attribution is required, and purists do not call it open source. Watch out: *older* Canary checkpoints (pre-Qwen) were CC-BY-**NC** (non-commercial only). Qwen3-ASR-**Flash** (the 52-language flagship numbers) is API-only on Alibaba Cloud; the open 0.6B/1.7B checkpoints cover fewer languages. Training data is closed for all of them - you get weights, not reproducibility.

------------------------------------------------------------------------

## 7. Setup

Everything below runs on a single modest GPU (tested on a 6 GB GTX 1660 SUPER) or CPU, and **every model loads through Hugging Face `transformers`** - no model-specific packages. Package roles:

- `transformers` (\>=5.13) + `torch` - all four model families (Whisper, Qwen3-ASR, Parakeet, Moonshine)
- `accelerate` - `device_map` placement for the larger checkpoints
- `datasets` + `torchcodec` - test clips with reference transcripts (LibriSpeech)
- `jiwer` - WER computation
- `soundfile` - WAV decoding without a system FFmpeg
- `sounddevice` - live microphone capture (section 13)
- `librosa` - resampling the mic’s native rate down to 16 kHz (section 13)

Qwen3-ASR and Parakeet-TDT became transformers-native in the 5.x line; earlier releases needed the vendor `qwen-asr` / `nemo_toolkit` packages, which this notebook no longer depends on.

All downloads (sample clips, HF dataset cache) land in `DL_tasks/datasets/`, which is gitignored.

------------------------------------------------------------------------

``` python
# Everything runs through Hugging Face transformers - no model-specific packages.
# %pip install -q torch transformers datasets accelerate torchcodec jiwer soundfile

# Live microphone capture in section 13. sounddevice binds the system PortAudio
# runtime (libportaudio2), librosa resamples the mic's native rate to 16 kHz.
# %pip install -q sounddevice librosa
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
    "Run garbage collection and release cached CPU/GPU memory.\n\n    Call right after `del`-ing a model/pipeline you are done with, e.g.\n    `del asr; free_memory()`. `del` drops the Python reference; this then\n    reclaims the RAM and hands the freed VRAM back to the CUDA allocator.\n    "
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

# All dataset downloads go to DL_tasks/datasets/ (gitignored)
DATA_DIR = Path("../../datasets")
DATA_DIR.mkdir(exist_ok=True)

# A ~14 s English sample clip (from the Qwen3-ASR repo)
SAMPLE = DATA_DIR / "sample_en.wav"
if not SAMPLE.exists():
    urllib.request.urlretrieve(
        "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
        SAMPLE,
    )
```

    NVIDIA GeForce RTX 3060
    device: cuda:0

``` python
import io

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

# 73 short LibriSpeech clips with reference transcripts - our eval set
ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy",
    "clean",
    split="validation",
    cache_dir=str(DATA_DIR / "hf_cache"),
)

# datasets >= 4.x decodes the Audio column through torchcodec, which dlopen's the
# system FFmpeg (libavutil.so). On a box without FFmpeg installed that raises at
# ds[0] time. Disable decoding and read the raw bytes with soundfile (libsndfile)
# instead, so this notebook runs whether or not FFmpeg is present.
ds = ds.cast_column("audio", Audio(decode=False))

def to_audio_input(audio):
    "Normalize a datasets audio entry to the dict transformers pipelines expect."
    if isinstance(audio, dict) and "array" in audio:  # already decoded (datasets < 4.x)
        return {"array": audio["array"], "sampling_rate": audio["sampling_rate"]}
    if isinstance(audio, dict):  # decode=False -> {"bytes": ..., "path": ...}
        src = audio["bytes"] or audio["path"]
        arr, sr = sf.read(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src,
                          dtype="float32")
        if arr.ndim > 1:  # downmix stereo to mono
            arr = arr.mean(axis=1)
        return {"array": arr, "sampling_rate": sr}
    samples = audio.get_all_samples()  # torchcodec AudioDecoder (decoding enabled)
    return {"array": samples.data.numpy().squeeze(), "sampling_rate": samples.sample_rate}

row = ds[0]
print(to_audio_input(row["audio"])["array"].shape)
print("ref:", row["text"][:70])
```

    (93680,)
    ref: MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO

## 8. Whisper with transformers

The `automatic-speech-recognition` pipeline handles decoding (via ffmpeg), resampling, batching and chunking. Model sizes: `tiny` (39M) -\> `base` -\> `small` (244M) -\> `medium` -\> `large-v3` (1.5B). `small` is a good accuracy/VRAM compromise on 6 GB cards.

------------------------------------------------------------------------

``` python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)

t0 = time.perf_counter()
result = asr(str(SAMPLE), return_timestamps=True)
print(f"{time.perf_counter() - t0:.1f}s")
print(result["text"])
for chunk in result["chunks"]:
    print(chunk["timestamp"], chunk["text"])
```

    Loading weights:   0%|          | 0/479 [00:00<?, ?it/s]

    [transformers] Using custom `forced_decoder_ids` from the (generation) config. This is deprecated in favor of the `task` and `language` flags/config options.
    [transformers] Transcription using a multilingual Whisper will default to language detection followed by transcription instead of translation to English. This might be a breaking change for your use case. If you want to instead always translate your audio to English, make sure to pass `language='en'`. See https://github.com/huggingface/transformers/pull/28687 for more details.
    [transformers] A custom logits processor of type <class 'transformers.generation.logits_process.SuppressTokensLogitsProcessor'> has been passed to `.generate()`, but it was also created in `.generate()`, given its parameterization. The custom <class 'transformers.generation.logits_process.SuppressTokensLogitsProcessor'> will take precedence. Please check the docstring of <class 'transformers.generation.logits_process.SuppressTokensLogitsProcessor'> to see related `.generate()` flags.
    [transformers] A custom logits processor of type <class 'transformers.generation.logits_process.SuppressTokensAtBeginLogitsProcessor'> has been passed to `.generate()`, but it was also created in `.generate()`, given its parameterization. The custom <class 'transformers.generation.logits_process.SuppressTokensAtBeginLogitsProcessor'> will take precedence. Please check the docstring of <class 'transformers.generation.logits_process.SuppressTokensAtBeginLogitsProcessor'> to see related `.generate()` flags.
    [transformers] Ignoring clean_up_tokenization_spaces=True for BPE tokenizer WhisperTokenizer. The clean_up_tokenization post-processing step is designed for WordPiece tokenizers and is destructive for BPE (it strips spaces before punctuation). Set clean_up_tokenization_spaces=False to suppress this warning, or set clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output=True to force cleanup anyway.

    1.9s
     Mm-hmm. Oh, yeah. Yeah, he wasn't even that big when I started listening to him But and his solo music didn't do overly well But he did very well when he started writing for other people
    (0.0, 5.34)  Mm-hmm. Oh, yeah. Yeah, he wasn't even that big when I started listening to him
    (5.76, 10.32)  But and his solo music didn't do overly well
    (10.7, 14.48)  But he did very well when he started writing for other people

``` python
# Long-form audio: chunk the input and batch the chunks
long_form = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    chunk_length_s=30,
    batch_size=8,
    device=device,
)

# Word-level timestamps and forced language / task selection:
result = asr(
    str(SAMPLE),
    return_timestamps="word",
    generate_kwargs={"language": "en", "task": "transcribe"},  # task="translate" -> English output from any language
)
print(result["chunks"][:5])

# Done demonstrating long-form config - free it (we keep `asr` for later sections).
del long_form
free_memory()
vram("after long_form")
```

    Loading weights:   0%|          | 0/479 [00:00<?, ?it/s]

    [transformers] Using `chunk_length_s` is very experimental with seq2seq models. The results will not necessarily be entirely accurate and will have caveats. More information: https://github.com/huggingface/transformers/pull/20104. Ignore this warning with pipeline(..., ignore_warning=True). To use Whisper for long-form transcription, use rather the model's `generate` method directly as the model relies on it's own chunking mechanism (cf. Whisper original paper, section 3.8. Long-form Transcription).

    [{'text': ' Mm', 'timestamp': (0.0, 0.6)}, {'text': '-hmm.', 'timestamp': (0.6, 0.9)}, {'text': ' Oh,', 'timestamp': (0.9, 1.24)}, {'text': ' yeah.', 'timestamp': (1.24, 1.54)}, {'text': ' Yeah,', 'timestamp': (1.54, 1.84)}]
    VRAM after long_form   0.98 GB allocated /  1.06 GB reserved

## 9. Qwen3-ASR (multilingual accuracy leader)

The strongest open multilingual family (Apache 2.0). 30+ languages with built-in language ID, singing-voice robustness, and a matching forced aligner for timestamps. As of `transformers` 5.x it loads natively - use the **`-hf` checkpoints** (`Qwen/Qwen3-ASR-0.6B-hf`), which pair `AutoProcessor` with `AutoModelForMultimodalLM`. The 0.6B checkpoint fits comfortably in 6 GB.

`processor.apply_transcription_request` builds the ASR chat prompt for you, and `processor.decode(..., return_format="parsed")` splits the detected language from the transcript.

------------------------------------------------------------------------

``` python
from transformers import AutoModelForMultimodalLM, AutoProcessor

qwen_id = "Qwen/Qwen3-ASR-0.6B-hf"
qwen_processor = AutoProcessor.from_pretrained(qwen_id)
qwen_model = AutoModelForMultimodalLM.from_pretrained(
    qwen_id,
    dtype=torch.float16 if device != "cpu" else torch.float32,
    device_map=device,  # bfloat16 needs Ampere+; float16 is safe on older cards
)

# apply_transcription_request formats the ASR chat prompt; audio accepts a path,
# URL or numpy array. Omit `language=` to auto-detect (or pass e.g. language="en").
inputs = qwen_processor.apply_transcription_request(audio=str(SAMPLE)).to(
    qwen_model.device, qwen_model.dtype
)
output_ids = qwen_model.generate(**inputs, max_new_tokens=256)
generated = output_ids[:, inputs["input_ids"].shape[1]:]

parsed = qwen_processor.decode(generated, return_format="parsed")[0]
print("language:", parsed["language"])
print(parsed["transcription"])

# For word/segment timestamps, pair it with Qwen/Qwen3-ForcedAligner-0.6B-hf.

del qwen_model, qwen_processor
free_memory()
vram("after qwen")
```

    Loading weights:   0%|          | 0/611 [00:00<?, ?it/s]

    language: English
    Hmm. Oh, yeah, yeah. He wasn't even that big when I started listening to him, but and his solo music didn't do overly well, but he did very well when he started writing for other people.
    VRAM after qwen        0.98 GB allocated /  1.07 GB reserved

## 10. Parakeet TDT (speed king)

NVIDIA’s Fast-Conformer encoder + Token-and-Duration-Transducer decoder - the Open ASR Leaderboard speed leader (RTFx in the thousands) at near-attention accuracy, and unlike CTC it emits casing and punctuation. Historically NeMo-only, the **v3 checkpoint** (`nvidia/parakeet-tdt-0.6b-v3`, 25 languages) is now transformers-native and drops straight into the `automatic-speech-recognition` pipeline. The older English-only `-v2` remains `.nemo`-format only.

------------------------------------------------------------------------

``` python
parakeet = pipeline(
    "automatic-speech-recognition",
    model="nvidia/parakeet-tdt-0.6b-v3",  # 25 languages, TDT decoder, casing + punctuation
    device=device,
)

t0 = time.perf_counter()
out = parakeet(str(SAMPLE))  # pass return_timestamps=True for token-level timestamps
print(f"{time.perf_counter() - t0:.1f}s")
print(out["text"])

del parakeet
free_memory()
vram("after parakeet")
```

    Loading weights:   0%|          | 0/723 [00:00<?, ?it/s]

    /home/bthek1/Knowledge/.venv/lib/python3.14/site-packages/transformers/generation/utils.py:1625: UserWarning: Using the model-agnostic default `max_length` (=1890) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.
      warnings.warn(

    0.1s
    Mhm. Oh yeah, yeah. He wasn't even that big when I started listening to him. But and his solo music didn't do overly well, but he did very well when he started writing for other people.
    VRAM after parakeet    0.98 GB allocated /  1.07 GB reserved

## 11. Moonshine (edge devices)

Purpose-built for embedded real-time ASR: `tiny` is 27M params (~27 MB quantised), and unlike Whisper it processes variable-length windows instead of always padding to 30 s - so short commands transcribe with proportionally less compute. English-only, MIT license, runs happily on CPU / Raspberry Pi, and loads through the same `automatic-speech-recognition` pipeline.

------------------------------------------------------------------------

``` python
moonshine = pipeline(
    "automatic-speech-recognition",
    model="UsefulSensors/moonshine-tiny",
    device=device,  # or device="cpu" - it is fast enough
)

t0 = time.perf_counter()
print(moonshine(str(SAMPLE))["text"])
print(f"{time.perf_counter() - t0:.1f}s")

del moonshine
free_memory()
vram("after moonshine")
```

    Loading weights:   0%|          | 0/160 [00:00<?, ?it/s]

    Oh, yeah, yeah, he wasn't even that big when I started listening to him, but in his solo music didn't do overly well, but he did very well when he started writing for other people.
    0.2s
    VRAM after moonshine   0.98 GB allocated /  1.07 GB reserved

## 12. Head-to-head Benchmark

Same clips, same normalisation, WER + wall clock per model. Caveats: this is a tiny English-only sample of clean read speech - treat it as a smoke test, not a leaderboard. For real evaluation use the full LibriSpeech test sets, or noisy/conversational sets like AMI and Earnings-22 (both on the Open ASR Leaderboard).

Measured here on an RTX 3060 12 GB (20 clips, 2026-11-11): whisper-small 8.3% WER / 14.9 s, moonshine-tiny 8.7% WER / 4.1 s, whisper-tiny 9.4% WER / 3.9 s. Moonshine-tiny matches whisper-tiny’s speed with a lower error rate; whisper-small buys under one point of WER for almost 4x the wall clock. Twenty clean clips is too few to separate the three with confidence.

------------------------------------------------------------------------

``` python
import jiwer

N = 20  # clips to evaluate (max 73)

normalize = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])

def benchmark(name, transcribe_fn):
    "Run transcribe_fn over the eval set; return (wer, seconds)."
    refs, hyps = [], []
    t0 = time.perf_counter()
    for row in ds.select(range(N)):
        hyps.append(transcribe_fn(to_audio_input(row["audio"])))
        refs.append(row["text"])
    elapsed = time.perf_counter() - t0
    wer = jiwer.wer(normalize(refs), normalize(hyps))
    print(f"{name:25s} WER {wer:6.2%}   {elapsed:5.1f}s / {N} clips")
    return wer, elapsed
```

``` python
results = {}

for model_id in ["openai/whisper-tiny", "openai/whisper-small", "UsefulSensors/moonshine-tiny"]:
    asr_pipe = pipeline("automatic-speech-recognition", model=model_id, device=device)
    results[model_id] = benchmark(model_id, lambda a: asr_pipe(a)["text"])
    del asr_pipe  # release each model before loading the next so VRAM stays flat
    free_memory()

vram("after benchmark")

# Parakeet also runs through the pipeline, so it slots straight in (same array input):
# pk = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-0.6b-v3", device=device)
# results["parakeet-tdt-0.6b-v3"] = benchmark("parakeet-tdt-0.6b-v3", lambda a: pk(a)["text"])
# del pk; free_memory()
#
# Qwen3-ASR uses the processor/generate API (section 9) rather than the pipeline:
# qp = AutoProcessor.from_pretrained("Qwen/Qwen3-ASR-0.6B-hf")
# qm = AutoModelForMultimodalLM.from_pretrained("Qwen/Qwen3-ASR-0.6B-hf", dtype=torch.float16, device_map=device)
# def qwen_transcribe(a):
#     inp = qp.apply_transcription_request(audio=a["array"], sampling_rate=a["sampling_rate"]).to(qm.device, qm.dtype)
#     ids = qm.generate(**inp, max_new_tokens=256)[:, inp["input_ids"].shape[1]:]
#     return qp.decode(ids, return_format="transcription_only")[0]
# results["Qwen3-ASR-0.6B"] = benchmark("Qwen3-ASR-0.6B", qwen_transcribe)
# del qm, qp; free_memory()
```

    [transformers] You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset

    openai/whisper-tiny       WER  9.38%     3.9s / 20 clips

    openai/whisper-small      WER  8.26%    14.9s / 20 clips

    UsefulSensors/moonshine-tiny WER  8.71%     4.1s / 20 clips
    VRAM after benchmark   0.98 GB allocated /  1.06 GB reserved

``` python
import pandas as pd

df = pd.DataFrame(
    [(k, wer, sec) for k, (wer, sec) in results.items()],
    columns=["model", "wer", "seconds"],
).sort_values("wer")
df
```

|     | model                        | wer      | seconds   |
|-----|------------------------------|----------|-----------|
| 1   | openai/whisper-small         | 0.082589 | 14.919845 |
| 2   | UsefulSensors/moonshine-tiny | 0.087054 | 4.142311  |
| 0   | openai/whisper-tiny          | 0.093750 | 3.864066  |

## 13. Real-time Microphone Input

Capture-then-transcribe loop against the machine’s actual microphone via `sounddevice`. For *true* streaming (partial results while speaking) use a natively streaming model - Moonshine’s streaming variant (`MoonshineStreamingForConditionalGeneration`), Voxtral Transcribe 2, or an RNN-T/TDT transducer - or a VAD-chunked wrapper project like RealtimeSTT / whisper-streaming.

Two things this cell handles that a naive `sd.rec(...)` does not:

- **Device rate.** A USB camera mic runs at its own native rate (48 kHz on the UGREEN camera 2K here), not the 16 kHz every model in this notebook expects. Recording at the native rate and resampling with `librosa` is the reliable order; asking ALSA for a rate the hardware does not offer is the usual cause of a transcript that reads like sped-up or slowed-down speech.
- **Silence.** A muted or zero-gain capture returns a valid, all-zero buffer, and Whisper will happily hallucinate a sentence out of it. The cell prints peak and RMS so a silent take is obvious before you trust the text.

This needs real ALSA capture nodes at `/dev/snd`. On the knowledge-lab LXC they are passed in from the Proxmox host by `av_devices` in `infra/proxmox/variables.tf` - and note that the camera’s video node and its USB-Audio card are **separate** passthroughs, so passing the camera through does not bring its mic with it.

------------------------------------------------------------------------

``` python
# --- standalone setup ----------------------------------------------------------
# Lifted from the Setup section and the helper cells above so this demo runs on its
# own in a fresh kernel - no earlier cell has to have been executed first.

import ctypes
import ctypes.util
import gc
import io
import time

import numpy as np
import torch

from dotenv import find_dotenv, load_dotenv
from pathlib import Path

# Knowledge/.env sets HF_TOKEN - authenticated HF Hub requests get higher rate limits
load_dotenv(find_dotenv(usecwd=True))

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def vram(tag=""):
    "Report current GPU memory (allocated / reserved). No-op on CPU."
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"VRAM {tag:16s} {alloc:5.2f} GB allocated / {reserved:5.2f} GB reserved")

def free_memory():
    "Run garbage collection and release cached CPU/GPU memory.\n\n    Call right after `del`-ing a model/pipeline you are done with, e.g.\n    `del asr; free_memory()`. `del` drops the Python reference; this then\n    reclaims the RAM and hands the freed VRAM back to the CUDA allocator.\n    "
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

# All dataset downloads go to DL_tasks/datasets/ (gitignored)
DATA_DIR = Path("../../datasets")

DATA_DIR.mkdir(exist_ok=True)

# --- the demo ------------------------------------------------------------------
# sounddevice needs the PortAudio runtime (libportaudio2) and ALSA nodes under
# /dev/snd. On the knowledge-lab LXC those are passed in from the Proxmox host by
# `av_devices` in infra/proxmox/variables.tf - the camera's video node and its
# USB-Audio card are separate passthroughs.
import librosa
import sounddevice as sd
from IPython.display import Audio as AudioPlayer  # `Audio` may already be datasets.Audio
from IPython.display import HTML, display
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Whisper's generate() emits five deprecation/config warnings PER CALL, which at one
# decode a second buries the transcript completely. Errors still get through.
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

# Substring of the *PortAudio* device name, which is not the ALSA card id: this mic
# is card id "U2K" but shows up as "UGREEN camera 2K: USB Audio (hw:0,0)". Run
# `python -c "import sounddevice; print(sounddevice.query_devices())"` to list them.
MIC_HINT = "UGREEN"  # None -> just take the first input device


def pick_microphone(hint=MIC_HINT):
    "Return (device_index, native_sample_rate) for a capture device, preferring `hint`."
    inputs = [(i, d) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]
    if not inputs:
        raise RuntimeError(
            "no audio capture device: PortAudio sees no ALSA card. Check that /dev/snd "
            "exists and holds controlC*/pcmC*D*c nodes - on the LXC that means adding "
            "them to av_devices and running `just tf-apply`."
        )
    match = [(i, d) for i, d in inputs if hint and hint.lower() in d["name"].lower()]
    idx, info = (match or inputs)[0]
    return idx, int(info["default_samplerate"])


SR = 16000            # every model in this notebook expects 16 kHz mono
LISTEN_SECONDS = 30   # how long the demo listens; interrupt the kernel to stop early
CHUNK_SECONDS = 3.0   # how much trailing audio each decode sees (the context window)
HOP_SECONDS = 0.5     # how often that window is re-decoded
SILENCE_DBFS = -45.0  # windows quieter than this are not sent to the model at all
SHOW_ROWS = 12        # most recent rows kept on screen; the full log still feeds the chart

# A 3 s window advanced in 0.5 s hops means every moment of audio is decoded about
# six times, each time with more right-hand context. That is the point: a word at
# the edge of one window sits comfortably inside the next, which is where most of
# the accuracy of chunked ASR comes from. The cost is arithmetic - 6x the decodes.
# Measured on continuous real speech (whisper-small, fp16), decode vs window length:
#       2 s ->  144 ms      10 s ->  491 ms      20 s ->  650 ms
#       5 s ->  279 ms      15 s ->  511 ms
# A 3 s window is roughly 200 ms, so a 0.5 s hop runs at a real-time factor near
# 0.4: comfortably live, but four times the GPU of disjoint blocks. Widen HOP if
# the summary reports overflows.
#
# Overlap has to be undone before the transcript is readable - see _stitch.

# Pinning the language matters more than anything else here. Left to auto-detect,
# Whisper answers near-silence with a long hallucination in whatever script it
# guessed - measured on this box: 4759 ms and a screenful of Georgian. With the
# language pinned the same model returns " you" in 64 ms.
#
# The corollary is counter-intuitive: decode cost tracks generation LENGTH, not
# parameter count. Measured on one 10 s window, fp16, same audio:
#     whisper-small   64 ms      whisper-tiny    1410 ms
#     (fp32: 150 ms)             moonshine-tiny   790 ms
# The tiny models are slower because they hallucinate long repetitive strings on
# noise. So the accuracy pick is also the fast pick - do not "optimise" this by
# reaching for a smaller checkpoint.
GEN_KWARGS = {"language": "en", "task": "transcribe", "max_new_tokens": 96}

_console = Console(record=True, file=io.StringIO(), width=96,
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


# Section 8 leaves `asr` live, but this cell frees it on its last line. Reload it
# when it is gone so the cell can be re-run on its own (weights are cached on disk).
if "asr" not in globals():
    from transformers import pipeline

    asr = pipeline("automatic-speech-recognition", model="openai/whisper-small",
                   device=device,
                   dtype=torch.float16 if device != "cpu" else torch.float32)
    rvram("whisper-small reloaded")


# rich (a project dependency) draws the panel; the in-place update is still an
# IPython display handle. rich's own Live goes through ipywidgets in Jupyter and
# appends a fresh view per refresh instead of replacing one - measured: 10 outputs
# for 5 updates - so render with rich, update with the handle.
#   force_jupyter=False is load-bearing: left on, Console.print() calls display()
#   itself and every frame leaks an extra output cell.


def _dbfs(x):
    "RMS level of a float32 block, in dBFS. -inf for digital silence."
    r = float(np.sqrt((x ** 2).mean())) if x.size else 0.0
    return 20 * np.log10(r) if r > 0 else float("-inf")


def _stitch(have, new, max_overlap=60, min_anchor=3):
    """Merge an overlapping hypothesis into the transcript by aligning on shared words.

    Consecutive windows overlap, but they are NOT identical. The model re-punctuates,
    swaps a function word, and drops or adds a leading filler as context changes:

        have : Honestly, they've been on THE  slow and steady thing for so long.
        new  :           They've been on THIS slow and steady thing for so long that
                                                                       it's felt like

    An exact suffix/prefix test fails here - "the" != "this" breaks the run - and the
    whole sentence gets appended a second time. So instead find the longest run of
    words the two share, treat that as the anchor, and keep only what follows it:

        anchor:                  slow and steady thing for so long
        added :                                                    that it's felt like

    Comparison is on a normalised copy (lowercase, alphanumeric) so punctuation and
    capitalisation differences do not break the anchor. `min_anchor` words must line
    up before the anchor is trusted; below that it falls back to an exact
    suffix/prefix match, and failing that appends everything.

    Returns (merged, n_added).
    """
    def norm(w):
        return "".join(ch for ch in w.lower() if ch.isalnum())

    if not have:
        return list(new), len(new)
    if not new:
        return list(have), 0

    tail = have[-max_overlap:]
    a = [norm(w) for w in tail]
    b = [norm(w) for w in new]

    # Longest common contiguous run (classic DP table, one row at a time). `>=` on
    # the comparison keeps the RIGHTMOST match, so a phrase repeated earlier in the
    # transcript cannot pull the anchor backwards.
    best_len = best_j = 0
    prev = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] and a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] >= best_len:
                    best_len, best_j = cur[j], j
        prev = cur

    if best_len >= min_anchor:
        return list(have) + list(new[best_j:]), len(new) - best_j

    for k in range(min(len(a), len(b), max_overlap), 0, -1):  # short overlap: exact only
        if a[-k:] == b[:k]:
            return list(have) + list(new[k:]), len(new) - k
    return list(have) + list(new), len(new)


def _view(elapsed, seconds, dbfs, rows, note="", fill=None):
    """One frame of the live view: header stats over one table row per decoded block.

    A block is decoded once and never revised, so each row is final the moment it
    appears - which is exactly what makes a table the right shape here rather than a
    single paragraph that keeps being rewritten.
    """
    head = Table.grid(padding=(0, 2))
    head.add_column(style="dim", justify="right")
    head.add_column()
    meter = "#" * int(max(0.0, min(1.0, (dbfs + 60) / 60)) * 24)
    head.add_row("elapsed", f"{elapsed:5.1f}s / {seconds:.0f}s")
    head.add_row("level", f"{dbfs:6.1f} dBFS [dim]{meter}[/]")
    if fill is not None:
        done = int(max(0.0, min(1.0, fill)) * 24)
        head.add_row("next block", f"[dim]{'=' * done}{'.' * (24 - done)}[/] {fill:4.0%}")
    if note:
        head.add_row("status", f"[dim]{note}[/]")
    head.add_row("totals", f"{len(rows)} decode(s), "
                           f"{sum(r['added'] for r in rows)} words kept, "
                           f"{sum(r['ms'] for r in rows) / 1000:.1f}s decoding")

    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, expand=True,
                  header_style="dim", border_style="dim")
    table.add_column("#", justify="right", style="dim", width=4, no_wrap=True)
    table.add_column("window", justify="right", width=13, no_wrap=True)
    table.add_column("dBFS", justify="right", width=6, no_wrap=True)
    table.add_column("decode", justify="right", width=8, no_wrap=True)
    table.add_column("+w", justify="right", width=3, no_wrap=True)
    table.add_column("hypothesis for this window", overflow="ellipsis", ratio=1)
    shown = rows[-SHOW_ROWS:]
    if len(rows) > len(shown):
        table.add_row("...", "", "", "", "",
                      f"[dim]{len(rows) - len(shown)} earlier decode(s) - "
                      "all of them are still in the chart[/]")
    for r in shown:
        # A decode that added nothing was pure overlap: correct, and the common case.
        style = "" if r["added"] else "dim"
        table.add_row(str(r["block"]), f"{r['from']:.1f}-{r['at']:.1f}s",
                      f"{r['dbfs']:.0f}", f"{r['ms']:.0f} ms",
                      f"[green]+{r['added']}[/]" if r["added"] else "0",
                      Text(r["text"] or "(empty)", style=style or "white"))
    if not rows:
        table.add_row("-", "", "", "", "", "[dim italic](nothing decoded yet)[/]")
    return Panel(Group(head, Text(""), table), title="[cyan]live transcription[/]",
                 border_style="cyan", padding=(0, 1))


def live_transcribe(seconds=LISTEN_SECONDS, chunk_seconds=CHUNK_SECONDS,
                    hop_seconds=HOP_SECONDS, target_sr=SR):
    """Listen and transcribe at the same time, updating the text in place as you speak.

    The two halves really do run concurrently: PortAudio fills a queue from its own
    callback thread, so the microphone keeps capturing while Whisper decodes on the
    GPU. Nothing is dropped during a decode - the audio that arrives mid-decode is
    waiting in the queue when it finishes.

    Text ACCUMULATES. A `chunk_seconds` window slides over the stream and is
    re-decoded every `hop_seconds`, so consecutive hypotheses overlap heavily; the
    repeated part is removed by _stitch and only the genuinely new words are kept.
    Decode cost is constant (the window never grows) and the transcript only ever
    gets longer.

    What this is NOT is true streaming ASR. Whisper is an offline encoder-decoder
    with no partial-hypothesis API, so a block only appears once it is complete -
    the text arrives a chunk at a time, not word by word. For genuinely incremental
    output you need a model built for it: Moonshine's streaming variant, an
    RNN-T / TDT transducer, or Voxtral Transcribe 2.

    Blocks below SILENCE_DBFS are skipped rather than decoded - see the note on
    GEN_KWARGS for why feeding Whisper silence is actively harmful.

    Returns (transcript, decode_log); decode_log feeds decode_chart().
    """
    import queue as _queue

    mic, native_sr = pick_microphone()
    q, overflows = _queue.Queue(), 0

    def on_audio(indata, frames, time_info, status):
        "PortAudio callback thread - keep it cheap, just hand the samples over."
        nonlocal overflows
        if status.input_overflow:
            overflows += 1
        q.put(indata[:, 0].copy())  # copy: PortAudio reuses this buffer

    view = display(_html(_view(0.0, seconds, float("-inf"), [], note="starting")),
                   display_id=True)
    buf = np.zeros(0, dtype="float32")
    window_n = int(chunk_seconds * native_sr)  # trailing audio each decode sees
    hop_n = int(hop_seconds * native_sr)       # new audio required before re-decoding
    fresh = 0                                  # unspent new samples since the last decode
    words, decode_log, decode_time = [], [], 0.0

    with sd.InputStream(device=mic, samplerate=native_sr, channels=1, dtype="float32",
                        blocksize=int(native_sr * 0.1), callback=on_audio):
        _note(f"[bold]mic [{mic}][/] @ {native_sr} Hz  ->  "
                  f"{chunk_seconds:.0f}s window every {hop_seconds:.1f}s, "
                  f"listening {seconds:.0f}s - speak now",
                  title="[cyan]capture[/]", style="cyan")
        t0 = time.perf_counter()
        try:
            while time.perf_counter() - t0 < seconds:
                chunks = []
                try:
                    chunks.append(q.get(timeout=0.2))
                except _queue.Empty:
                    pass
                while True:  # drain whatever piled up while the last decode ran
                    try:
                        chunks.append(q.get_nowait())
                    except _queue.Empty:
                        break
                if chunks:
                    fresh += sum(c.size for c in chunks)
                    buf = np.concatenate([buf, *chunks])[-window_n:]  # slide, do not grow

                elapsed = time.perf_counter() - t0
                if fresh < hop_n or buf.size < int(0.8 * native_sr):
                    view.update(_html(_view(
                        elapsed, seconds, _dbfs(buf), decode_log,
                        fill=fresh / hop_n, note="waiting for the next hop")))
                    continue
                fresh = 0

                # Resample the window, not the live stream: ask ALSA for a rate the
                # hardware does not offer and the audio comes back pitch-shifted.
                audio = (librosa.resample(buf, orig_sr=native_sr, target_sr=target_sr)
                         if native_sr != target_sr else buf)
                dbfs = _dbfs(audio)

                # Never hand silence to Whisper: it does not return an empty string,
                # it invents a fluent one. Measured room floor here is -53 dBFS RMS
                # and speech sits well above -40, so -45 separates them cleanly.
                if dbfs < SILENCE_DBFS:
                    view.update(_html(_view(
                        elapsed, seconds, dbfs, decode_log,
                        note=f"below {SILENCE_DBFS:.0f} dBFS - window skipped, not decoded")))
                    continue

                t1 = time.perf_counter()
                text = asr({"array": audio, "sampling_rate": target_sr},
                           generate_kwargs=GEN_KWARGS)["text"].strip()
                dt = time.perf_counter() - t1
                decode_time += dt

                words, added = _stitch(words, text.split())
                decode_log.append({"block": len(decode_log) + 1, "ms": 1000 * dt,
                                   "words": len(text.split()), "added": added,
                                   "dbfs": dbfs, "at": elapsed,
                                   "from": max(0.0, elapsed - buf.size / native_sr),
                                   "text": text})

                view.update(_html(_view(elapsed, seconds, dbfs, decode_log)))
        except KeyboardInterrupt:
            view.update(_html(_view(time.perf_counter() - t0, seconds, float("-inf"),
                                    decode_log, note="stopped")))

    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="dim", justify="right")
    summary.add_column()
    n = len(decode_log)
    rtf = (decode_time / (n * hop_seconds)) if n else float("nan")
    summary.add_row("decodes", f"{n} [dim]({chunk_seconds:.0f}s window, "
                               f"{hop_seconds:.1f}s hop)[/]")
    summary.add_row("mean decode", f"{1000 * decode_time / max(n, 1):.0f} ms "
                                   f"[dim]of a {1000 * hop_seconds:.0f} ms budget[/]")
    summary.add_row("real-time factor", f"{rtf:.3f}"
                    + ("" if rtf < 1 else "  [red](decode slower than audio)[/]"))
    summary.add_row("words kept", f"{sum(r['added'] for r in decode_log)} "
                                  f"[dim]of {sum(r['words'] for r in decode_log)} "
                                  "decoded (rest was overlap)[/]")
    if overflows:
        summary.add_row("overflows", f"[red]{overflows}[/] (decode is not keeping up)")
    display(_html(Panel(summary, title="[dim]done[/]", border_style="dim", padding=(0, 1))))
    return " ".join(words), decode_log


def decode_chart(log, hop_seconds=HOP_SECONDS):
    """Decode latency per block, against the real-time budget (ECharts, repo standard).

    The overlaid word count is the point of the chart: cost tracks how much text the
    model GENERATES, not how much audio it was given. Every window here is the same
    length, so any spread in the bars is generation length, nothing else.
    """
    from pyecharts import options as opts
    from pyecharts.charts import Bar, Line

    if not log:
        _note("no decodes to chart - nothing rose above the silence gate",
              title="[yellow]empty[/]", style="yellow")
        return None

    x = [str(r["block"]) for r in log]
    ms = [round(r["ms"], 1) for r in log]
    words = [r["words"] for r in log]
    budget = 1000 * hop_seconds

    bar = (
        Bar(init_opts=opts.InitOpts(width="760px", height="400px"))
        .add_xaxis(x)
        .add_yaxis("decode ms", ms, category_gap="30%", z=1)
        .extend_axis(yaxis=opts.AxisOpts(name="words out", position="right"))
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Decode latency per block",
                subtitle=f"{CHUNK_SECONDS:.0f}s window every {hop_seconds:.1f}s - "
                         f"under {budget:.0f} ms keeps up with real time",
            ),
            xaxis_opts=opts.AxisOpts(name="decode"),
            yaxis_opts=opts.AxisOpts(name="milliseconds"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_top="8%"),
        )
        .set_series_opts(
            label_opts=opts.LabelOpts(is_show=False),
            markline_opts=opts.MarkLineOpts(
                data=[opts.MarkLineItem(y=budget, name="real-time budget")],
                label_opts=opts.LabelOpts(position="end", formatter="real time"),
            ),
        )
    )
    line = (
        Line()
        .add_xaxis(x)
        .add_yaxis("words out", words, yaxis_index=1, is_smooth=True,
                   label_opts=opts.LabelOpts(is_show=False))
    )
    return bar.overlap(line).render_notebook()


final_text, DECODE_LOG = live_transcribe()
display(_html(Panel(Text(final_text or "(nothing recognised)",
                         style="bold" if final_text else "dim italic"),
                    title="[green]final transcript[/]", border_style="green", padding=(0, 1))))

# End of notebook - release the Whisper pipeline we kept around for the mic demo.
del asr
free_memory()
rvram("final")

decode_chart(DECODE_LOG)
```

## 14. Common Frameworks

Every runnable cell above went through `transformers`, because that is this repo’s rule and because it is genuinely the shortest path from a checkpoint to a transcript. Production ASR systems rarely stop there: the weights are the easy part, and most of the engineering sits in what decodes them fast enough, what strips the silence first, and what turns “text” into “who said what, when”. This is the map of what teams actually reach for.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [transformers](https://github.com/huggingface/transformers) (+ `torch`, `datasets`, `accelerate`) | modelling | One API over Whisper, Qwen3-ASR, Parakeet, Moonshine; `pipeline` handles chunking, timestamps and batching | Apache 2.0 | Default. Start here and only leave when a number forces you to |
| [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) | modelling | Conformer / TDT / SALM training recipes, and the `.nemo` checkpoints that never got a `transformers` port | Apache 2.0 (weights often CC-BY-4.0, some older ones CC-BY-**NC**) | You need Canary-Qwen 2.5B or a pre-v3 Parakeet, or you are training an encoder from scratch on your own corpus |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) / [ESPnet](https://github.com/espnet/espnet) | modelling | Full speech toolkits: ASR plus enhancement, separation, speaker ID, all with published recipes | Apache 2.0 | Research, or a system where ASR is one of five speech models and you want one training stack for all of them |
| [torchaudio](https://github.com/pytorch/audio) + [soundfile](https://github.com/bastibe/python-soundfile) / [librosa](https://github.com/librosa/librosa) | data | Decode, resample to 16 kHz, mel spectrograms, augmentation | BSD-2 / BSD-3 / ISC | Always. Every one of these models silently assumes 16 kHz mono, and mismatched sample rate is the most common cause of “the model got worse” |
| [Silero VAD](https://github.com/snakers4/silero-vad) | data | Sub-millisecond speech/non-speech segmentation before the expensive model runs | MIT | Real recordings. Stripping silence is usually the largest single speed win available, and it cuts hallucinated text on empty audio - see `05_Voice_Activity_Detection` |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) | inference runtime | The same Whisper weights, int8/fp16, several times the throughput at lower memory; `initial_prompt` / `hotwords` for domain biasing | MIT | Bulk transcription on a GPU. For CPU and embedded, [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (GGML) is the equivalent trade |
| [vLLM](https://github.com/vllm-project/vllm) | serving | Continuous batching and paged attention for the LLM-decoder models (Whisper, Voxtral, Qwen3-ASR) | Apache 2.0 | You are serving many concurrent requests. `model.generate` in a web handler is a reference implementation, not a service |
| [NVIDIA Riva](https://developer.nvidia.com/riva) / [Triton Inference Server](https://github.com/triton-inference-server/server) | serving | Streaming gRPC ASR with partial hypotheses, endpointing and autoscaling | proprietary (Riva) / BSD-3 (Triton) | You need true streaming with sub-second partials, not the capture-then-transcribe loop of section 13 |
| [WhisperX](https://github.com/m-bain/whisperX) + [pyannote.audio](https://github.com/pyannote/pyannote-audio) | orchestration | Word-level alignment and speaker diarization stitched onto an ASR backbone | BSD-4 / MIT | The product needs “who said what” - meetings, interviews, call analytics. pyannote’s models are gated on the Hub |
| [Pipecat](https://github.com/pipecat-ai/pipecat) / [LiveKit Agents](https://github.com/livekit/agents) | orchestration | The realtime conversational loop: transport, VAD, turn detection, barge-in, STT to LLM to TTS | BSD-2 / Apache 2.0 | ASR is one leg of a voice agent rather than the product. See `Frameworks/01_pipecat` |
| [jiwer](https://github.com/jitsi/jiwer) + [evaluate](https://github.com/huggingface/evaluate) | evaluation | WER/CER with composable normalisation, which is where the comparison is actually won or lost | Apache 2.0 | Always, and with the normalisation pinned in code - section 4 |

**The 2026 default stack** is narrow: `transformers` to choose the model, Silero VAD in front of it, and faster-whisper or vLLM behind it once throughput matters. Everything else on this list earns its place by a specific constraint from section 2 - diarization for meetings, Riva for sub-second streaming partials, whisper.cpp for a device with no GPU, NeMo for the last point of English accuracy.

The common wrong turn is reaching for a heavyweight toolkit (NeMo, ESPnet) to *run* a model when it was only ever needed to *train* one. The second is benchmarking two systems whose text normalisation differs, which can move WER by several points without a single weight changing.

------------------------------------------------------------------------

## 15. Going Further

- **Fine-tuning.** Whisper fine-tunes well on ~5-10 h of domain audio: [HF fine-tuning guide](https://huggingface.co/blog/fine-tune-whisper). Common Voice, FLEURS and People’s Speech are the usual open training sets. The win is almost always vocabulary - names, product terms, acronyms - not acoustics.
- **Bias the decoder before you fine-tune.** A prompt (`initial_prompt`, or `hotwords` in faster-whisper) that lists the domain terms costs nothing and recovers a surprising share of the gap. Try it before spending a week on training data.
- **Forced alignment.** `Qwen/Qwen3-ForcedAligner-0.6B-hf` (transformers-native) or the classic Montreal Forced Aligner, when you already have the transcript and need the timings.
- **Related notebooks.** `05_Voice_Activity_Detection` (the preprocessing step that belongs in front of every real recording), `00_Text_to_Speech` (the inverse task), `04_Audio_Classification` (when the label matters and the words do not), `Multimodal/00_Audio_Text_to_Text` (models that answer questions about audio instead of transcribing it).

**References**

- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) and the [leaderboard paper](https://arxiv.org/abs/2510.06961)
- [Whisper paper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356)
- [Qwen3-ASR on GitHub](https://github.com/QwenLM/Qwen3-ASR)
- [Parakeet in transformers (model docs)](https://huggingface.co/docs/transformers/model_doc/parakeet)
- [Moonshine paper](https://arxiv.org/abs/2410.15608)
- [Northflank: best open source STT in 2026 (benchmarks)](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)

------------------------------------------------------------------------

Back to top
