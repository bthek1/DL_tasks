# Audio-Text-to-Text

Everything to know about audio LLMs: what “listen to this and answer” covers, why the cascade still wins in production, the mid-2026 model landscape, and runnable code that puts a cascade and a native audio LLM on the same clips.

Author

Benedict Thekkel

## 1. What is Audio-Text-to-Text?

Audio-text-to-text is the **audio plus a text instruction in, text out** family. The model gets a waveform *and* a prompt, and the prompt decides what the waveform is for:

- “Transcribe this.” -\> ASR
- “What is the speaker’s emotional state?” -\> paralinguistic analysis
- “Summarise this call and list the action items.” -\> understanding
- “What sounds do you hear in the background?” -\> audio event description
- “Translate what she says into French.” -\> speech translation

One set of weights, many jobs, selected by words rather than by picking a different model. That is the whole point of the family, and it is what separates it from ASR: an ASR model takes **no prompt** and has exactly one output (see `Audio/02_Automatic_Speech_Recognition`).

**Input.** A mono waveform, almost always resampled to **16 kHz** (the audio encoders are Whisper-derived and Whisper is a 16 kHz model). It is turned into a log-mel spectrogram, run through an audio encoder, and downsampled into a short sequence of **audio tokens** that are spliced into the LLM’s token stream at the position of an `<|audio|>` placeholder. Roughly 6.25 to 12.5 tokens per second of audio after pooling, which is why a 30 s clip costs a few hundred tokens and a 2 h meeting does not fit.

**Output.** Plain text. Nothing else - a model that also speaks back belongs in `Multimodal/08_Any_to_Any`.

| Neighbouring task | Difference | Typical tools |
|----|----|----|
| Automatic speech recognition | No prompt; one fixed output (the transcript) | Whisper, Parakeet, Canary |
| Audio classification | Fixed label set, no free text | AST, CLAP, wav2vec2 |
| Text-to-speech / audio-to-audio | Audio *out*, not text out | Kokoro, CSM, Voicebox |
| Any-to-any (omni) | Also *emits* audio; full-duplex conversation | Qwen3-Omni, MiniCPM-o, Moshi |
| Video-text-to-text | Adds the visual track; audio is one stream of several | Qwen3-VL, VideoLLaMA 3 |

**The dividing line that matters.** There are two ways to build this, and they are not interchangeable:

- **Cascade**: ASR model -\> text -\> ordinary LLM. The LLM never hears anything. Everything that is not words is gone by the time it reads the transcript: tone, hesitation, laughter, overlapping speakers, the dog barking, the fact that the caller is shouting.
- **Native (end-to-end)**: the LLM attends over the audio tokens directly. Prosody, speaker changes and background sound survive into the model’s context.

Most production systems in 2026 are still cascades, and section 8 builds one so the comparison is concrete rather than rhetorical.

------------------------------------------------------------------------

## 2. Real-World Use Cases

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Meeting notes and action items | Productivity (Otter, Granola, Teams/Zoom AI) | 30-60 min multi-speaker recording -\> summary, decisions, owners | Long context and speaker attribution; cost per hour of audio |
| Contact-centre QA and compliance | Financial services, telco (NICE, Verint, Observe.AI) | Recorded call + a rubric -\> scored answers with evidence | Auditability, recall of required disclosures, sentiment/escalation detection |
| Voice assistants and in-car | Consumer devices, automotive | 2-5 s utterance -\> an intent or an answer | **Time to first token** (barge-in feels broken past ~300 ms); on-device privacy |
| Clinical documentation | Healthcare (Nuance DAX, Abridge) | Doctor-patient conversation -\> structured note | Accuracy on drug names, PHI handling, on-prem deployment |
| Media search and podcast tooling | Media, publishing | Archive audio -\> chapters, quotes, topic tags | Throughput and cost over huge back-catalogues; speaker diarisation |
| Accessibility and live captioning | Public sector, education, events | Streaming audio -\> captions plus disfluency cleanup | Streaming latency; robustness to accents and far-field mics |
| Content moderation | Social, gaming voice chat | Short clips -\> policy labels with justification | Non-speech cues (screaming, gunshots) that a transcript loses; false-positive cost |
| Industrial and vehicle acoustics | Manufacturing, fleet maintenance | Machine or engine recording -\> fault description | There is no speech at all; only a native audio model can do this |
| Language learning and pronunciation | EdTech (Duolingo, Speak) | Learner utterance -\> feedback on pronunciation and fluency | Paralinguistics are the product; a transcript is not enough |
| Call analytics on archives | Insurance, utilities | Millions of stored calls -\> searchable structured fields | Batch cost per hour dominates; a cascade with a cheap ASR usually wins |

**What the benchmark number hides.** Four things decide these deployments more often than accuracy does.

*Latency has a shape, not a value.* For a voice assistant what matters is time to first token and whether the model can be interrupted, not the total decode time. A cascade pays ASR-then-LLM serially unless the ASR streams partial hypotheses, which is exactly why streaming-first architectures (Moshi, Qwen3-Omni’s talker) exist.

*The cascade is a lossy compression step, and sometimes that is fine.* If the task is “summarise this meeting”, a good transcript loses almost nothing and costs a tenth as much. If the task is “was this caller angry”, “did the speaker switch to Hindi mid-sentence”, or “what is that grinding noise”, the cascade cannot recover what it threw away.

*Audio length is a hard wall.* At ~12.5 audio tokens per second, an hour of audio is ~45k tokens before the prompt. Real systems chunk with overlap, run diarisation first, or transcribe with a cheap ASR and reason over the text. Audio Flamingo 3 pushing to 10 minutes of context in one shot was news in 2025, which tells you where the ceiling sits.

*Domain shift is the real error source.* Accents, telephone-band 8 kHz audio, cross-talk, and far-field reverberation move WER by more than any two adjacent models on a leaderboard differ. Test on your own recordings before believing anything below.

------------------------------------------------------------------------

## 3. How Modern Audio-Text-to-Text Works

Five generations, all still deployed somewhere:

1.  **The cascade (2020 onward, still the default).** ASR model -\> transcript -\> LLM prompt. It is modular, cheap, debuggable, and lets you swap either half. Two costs: errors compound (an ASR mistake becomes a confident LLM hallucination), and everything non-lexical is discarded before the LLM sees anything.

2.  **Frozen encoder + adapter (2023).** Take a pretrained audio encoder (usually the **Whisper encoder**), pretrain a small bridge into a frozen LLM, and train only the bridge. **SALMONN** (2023) used a dual encoder (Whisper + BEATs, so it hears speech *and* general sound) into a Q-Former; **Qwen-Audio** (2023) used a multi-task pretraining recipe over 30+ audio tasks; **LTU/LTU-AS** made “listen, think, understand” the framing. Cheap to train, and the LLM keeps all its text ability because it is frozen.

3.  **Instruction-tuned audio LLMs (2024).** Unfreeze more, train on audio instruction data. **Qwen2-Audio** (Aug 2024) dropped the hierarchical task tags for natural-language prompts and shipped a voice-chat mode. **Ultravox** (Fixie, 2024) trains only a projector from the Whisper encoder into Llama with a knowledge-distillation loss and **no ASR objective at all**, which keeps it fast and cheap to retarget to a new LLM. **Phi-4-multimodal** (2025) added per-modality **LoRA adapters** over one frozen 3.8B backbone, so vision and speech coexist without either degrading the text model.

4.  **Discrete audio tokens (2024-2025).** Instead of continuous encoder features, quantise audio into discrete tokens (SpeechTokenizer, Mimi, SNAC) so audio is literally just more vocabulary. That makes generation symmetric with understanding and enables full duplex: **Moshi** (Kyutai, 2024) models user and assistant streams in parallel at 12.5 Hz with ~200 ms theoretical latency; **GLM-4-Voice** and **Step-Audio** followed.

5.  **Omni models (2025-2026).** One model for text, image, audio and video in, text and speech out. **Qwen2.5-Omni** introduced the Thinker-Talker split (a reasoning LLM plus a streaming speech decoder) with TMRoPE for audio-video time alignment; **Qwen3-Omni** (Sep 2025) made it a 30B-A3B MoE covering 119 text / 19 speech-in / 10 speech-out languages with a first-packet latency around 234 ms, and shipped a dedicated **Captioner** variant for general audio description. **Gemma 3n** (2025) went the other way: MatFormer nesting plus per-layer embeddings to run audio+vision+text **on a phone** at an E2B/E4B effective size.

**Where it stands in mid-2026.** Native models have closed most of the ASR gap (Voxtral and Granite Speech both post competitive Open ASR Leaderboard numbers) while keeping the paralinguistic and non-speech abilities a cascade cannot have. Three live directions: **reasoning** variants that emit a thinking trace before answering audio questions (MMAU/MMAR-style tasks reward it), **long audio** context past 10 minutes, and **full-duplex** conversation where the model listens and speaks at the same time. Meanwhile the cascade is not going away, because for pure transcript-plus-summary work it is several times cheaper.

------------------------------------------------------------------------

## 4. Evaluation Metrics

The family has no single metric, because the prompt decides the task. Four families are used, and a serious evaluation reports at least three.

**Word Error Rate** (for the transcription instruction). Levenshtein distance at the word level, normalised by the reference length:

$$\mathrm{WER} = \frac{S + D + I}{N}$$

with $S$ substitutions, $D$ deletions, $I$ insertions and $N$ words in the reference. **Normalisation dominates the number**: casing, punctuation, “%” vs “percent”, “Dr.” vs “Doctor”, and numerals vs words can each move WER by several points. The Whisper English normaliser is the de-facto standard, and comparing two systems normalised differently is meaningless. Note that WER is unbounded above: an audio LLM that answers a transcription prompt with “Sure! Here is the transcript: …” scores catastrophically for a reason that has nothing to do with hearing.

**Accuracy** (for closed-form audio question answering). MMAU, MMAR and the AIR-Bench foundation split are multiple choice, so a plain exact match after light normalisation works, and chance is 25% for 4-way items.

**LLM-as-judge** (for open-ended answers). AIR-Bench chat, OpenAudioBench and VoiceBench score a free-text answer against a reference with a judge model on a 1-10 scale. It is the only thing that works for “summarise this call”, and it inherits the judge’s biases (length, verbosity, self-preference).

**Speed.** Report **real-time factor** (processing seconds per audio second, lower is better), **time to first token** for interactive use, and peak VRAM. RTF is the number that decides whether the archive job costs 100 dollars or 10,000.

The cell below implements WER with the normalisation made explicit, plus the exact-match accuracy helper the benchmark reuses.

------------------------------------------------------------------------

``` python
import re
import unicodedata

import jiwer
from num2words import num2words

# The single most important line in any WER comparison: what counted as "the same word".
# This is a compact stand-in for the Whisper English normaliser - lowercase, strip
# punctuation, expand digits, drop filler. Publish it alongside the number, always.
_FILLERS = {"uh", "um", "mm", "hmm", "er", "ah"}
_CONTRACTIONS = {"aint": "is not", "cant": "can not", "wont": "will not", "dont": "do not"}


def normalise(text):
    "Lowercase, strip punctuation and fillers, expand digits. Applied to BOTH sides."
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s%]", " ", text)             # punctuation out, keep % for expansion
    text = text.replace("%", " percent ")
    words = []
    for w in text.split():
        if w in _FILLERS:
            continue
        if w in _CONTRACTIONS:
            words.extend(_CONTRACTIONS[w].split())
        elif w.isdigit():
            words.extend(num2words(int(w)).replace("-", " ").replace(",", "").split())
        else:
            words.append(w)
    return " ".join(words)


def wer(reference, hypothesis):
    "Word error rate after the normalisation above. Unbounded above, so >1.0 is possible."
    return jiwer.wer(normalise(reference), normalise(hypothesis))


def exact_match(reference, hypothesis):
    "1.0 if the normalised reference appears in the normalised answer, else 0.0.\n\n    Audio LLMs wrap short answers in a sentence ('The speaker says two.'), so a strict\n    equality check would score a correct model at zero. Containment is the usual\n    compromise; a real MMAU harness parses the option letter instead.\n    "
    return float(normalise(reference) in normalise(hypothesis))


ref = "He had 5% of the vote, uh, in the second district."
for hyp in [
    "he had five percent of the vote in the second district",   # same words, different surface
    "He had 5 percent of the vote in the 2nd district.",         # numeral vs word
    "Sure! Here is the transcript: he had 5% of the vote in the second district.",
]:
    print(f"WER {wer(ref, hyp):5.2f}  <- {hyp[:70]}")

print()
print("raw jiwer, no normalisation:", round(jiwer.wer(ref, "he had five percent of the vote in the second district"), 2))
print("exact_match('two', 'The speaker says two.') =", exact_match("two", "The speaker says two."))
```

    WER  0.00  <- he had five percent of the vote in the second district
    WER  0.09  <- He had 5 percent of the vote in the 2nd district.
    WER  0.45  <- Sure! Here is the transcript: he had 5% of the vote in the second dist

    raw jiwer, no normalisation: 0.55
    exact_match('two', 'The speaker says two.') = 1.0

## 5. Datasets

| Dataset | Contents | Size | Scope | License | Typical use |
|----|----|----|----|----|----|
| [LibriSpeech](https://huggingface.co/datasets/openslr/librispeech_asr) | Read audiobooks, clean and noisy splits | 960 h train / 5.4 h test-clean | en | CC-BY 4.0 | The ASR reference point; **this notebook evaluates on the `hf-internal-testing` dummy split of it** |
| [Common Voice](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0) | Crowd-read sentences, heavy accent diversity | 20k+ h | 100+ languages | CC0 | Multilingual ASR train and eval; gated behind terms |
| [GigaSpeech](https://huggingface.co/datasets/speechcolab/gigaspeech) | Podcasts, YouTube, audiobooks | 10k h | en | Apache 2.0, **gated** | Realistic spontaneous speech |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | Parallel read speech | ~12 h per language | 102 languages | CC-BY 4.0 | Multilingual ASR and speech translation |
| [AudioCaps](https://huggingface.co/datasets/OpenSound/AudioCaps) | AudioSet clips with human captions | 46k clips | non-speech events | MIT (annotations) | Audio captioning; the “can it hear the world” test |
| [Clotho](https://zenodo.org/records/4783391) | 15-30 s clips, 5 captions each | 6k clips | general sound | CC | Audio captioning and retrieval |
| [MMAU](https://huggingface.co/datasets/apokalypse47/MMAU-Test) | Expert-annotated MCQs over speech, music and sound | 10k QA | en | CC-BY-NC | **The** audio-understanding benchmark; MMAR is its harder 2025 sibling |
| [AIR-Bench](https://huggingface.co/datasets/qyang1021/AIR-Bench-Dataset) | Foundation (MCQ) + chat (open-ended, judged) | 19k + 2k | en, zh | Apache 2.0 | Instruction-following over audio |
| [VoiceBench](https://huggingface.co/datasets/hlt-lab/voicebench) | Spoken instructions, including adversarial and accented | 10 subsets | en | MIT | Voice-assistant behaviour, not transcription |
| [LibriSpeech-long](https://huggingface.co/datasets/distil-whisper/librispeech_long) | Concatenated long-form audio | minutes-long clips | en | CC-BY 4.0 | Long-form chunking and context-length tests |
| [ESC-50](https://huggingface.co/datasets/ashraq/esc50) | 5 s environmental sound clips, 50 classes | 2k clips | non-speech | CC-BY-NC | The non-speech probe used in section 10 |

This notebook evaluates on **`hf-internal-testing/librispeech_asr_dummy`** (73 clips of LibriSpeech validation-clean with reference transcripts, a ~9 MB download) and uses one **ESC-50** clip as the non-speech probe. Both land in `DL_tasks/datasets/` via `cache_dir`. Neither is a leaderboard: LibriSpeech clean read speech is the easiest audio there is, and a handful of clips is a smoke test.

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

Three leaderboards, because no single one covers the family:

- **[Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)** for the transcription half (WER and RTFx across 8 datasets).
- **[MMAU](https://sakshi113.github.io/mmau_homepage/) / MMAR** for audio *understanding* (speech + music + environmental sound reasoning).
- **[VoiceBench](https://github.com/MatthewCYM/VoiceBench)** and **[AIR-Bench](https://github.com/OFA-Sys/AIR-Bench)** for spoken-instruction following.

| Model | Params | License | Scope | Architecture | Best for |
|----|----|----|----|----|----|
| Whisper large-v3-turbo + an LLM | 0.81B + LLM | Apache 2.0 | 99 languages (ASR) | **cascade** | the cheap, debuggable production default |
| [Ultravox v0.5/v0.6](https://huggingface.co/fixie-ai/ultravox-v0_5-llama-3_2-1b) | 1B-8B | MIT (adapter) | en-centric | Whisper encoder -\> projector -\> Llama, no ASR loss | low-latency voice agents; retargetable to any LLM |
| [Aero-1-Audio](https://huggingface.co/lmms-lab/Aero-1-Audio) | 1.5B | Apache 2.0 | en | Whisper encoder + Qwen2.5-1.5B | the smallest credible native audio LLM |
| **[Granite Speech 3.3 2B](https://huggingface.co/ibm-granite/granite-speech-3.3-2b)** | 2B | Apache 2.0 | en (+ 5 for AST) | Conformer encoder + Q-former + **LoRA** on Granite 3.3 | **this notebook’s native model**: transformers-native, 6 GB, strong WER |
| [Qwen2-Audio-7B-Instruct](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct) | 8.4B | Apache 2.0 | 8+ languages, speech + sound + music | Whisper-large encoder + Qwen2-7B | the reference open audio LLM; needs 4-bit here |
| [Voxtral Mini / Small](https://huggingface.co/mistralai/Voxtral-Mini-3B-2507) | 4.7B / 24B | Apache 2.0 | 8 languages | Whisper-style encoder + Ministral/Mistral Small | best open ASR-plus-understanding trade in 2025-26 |
| [Phi-4-multimodal](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | 5.6B | MIT | 8 languages, +vision | frozen 3.8B backbone + **per-modality LoRA** | one small model for both speech and images |
| [Gemma 3n E2B/E4B](https://huggingface.co/google/gemma-3n-E2B-it) | 5.4B raw / 2B effective | Gemma terms | 140+ text, 35 multimodal | MatFormer + per-layer embeddings + USM encoder | **on-device** audio+vision+text |
| [Audio Flamingo 3](https://huggingface.co/nvidia/audio-flamingo-3) | 7B | NVIDIA OM (non-commercial) | speech, sound, music | AF-Whisper unified encoder | **10-minute** audio context; long-form reasoning |
| [Qwen3-Omni-30B-A3B](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | 30B MoE (3B active) | Apache 2.0 | 119 text / 19 speech-in | Thinker-Talker MoE | SOTA open audio understanding; also speaks (see `08_Any_to_Any`) |
| [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16) | 7B | CC-BY | en | Mimi tokeniser, parallel duplex streams | full-duplex conversation, ~200 ms latency |
| [MiniCPM-o 2.6](https://huggingface.co/openbmb/MiniCPM-o-2_6) | 8B | Apache 2.0 (model) | multimodal | end-to-end omni | strong on-device omni; voice cloning |

**Who wins what.** On raw transcription accuracy the specialised ASR models still lead (Parakeet and Canary top the Open ASR Leaderboard on English), with Voxtral and Granite Speech close behind while also answering questions. On audio *understanding* (MMAU/MMAR) Qwen3-Omni leads the open field, with Audio Flamingo 3 ahead on long-form. On **latency** the discrete-token duplex models (Moshi, Qwen3-Omni’s talker) are in a different class - a couple of hundred milliseconds to first audio, where a cascade is a second or more. On **cost per hour of archive**, nothing beats a cascade with a small ASR.

Tie that back to section 2: the contact-centre archive job takes the cascade, the voice assistant takes Ultravox or a duplex model, the “why is my engine making that noise” job takes a native model that heard the noise, and the phone takes Gemma 3n.

**What fits this 12 GB box.** Granite Speech 3.3 2B (6 GB download, ~5 GB VRAM in fp16) and the cascade run comfortably. Qwen2-Audio-7B-Instruct fits in **4-bit** but the download is 16.8 GB, so it sits behind a `RUN_HEAVY` flag. Voxtral Mini (18.7 GB download), Gemma 3n E2B (10.9 GB) and Audio Flamingo 3 (16.9 GB) are all download-bound rather than VRAM-bound. Qwen3-Omni-30B (70 GB) is out of reach entirely.

------------------------------------------------------------------------

## 7. Setup

Everything below runs on a 12 GB RTX 3060, or on CPU slowly, and **every model loads through Hugging Face `transformers`** - no vendor packages. Package roles:

- `transformers` (\>=5.13) + `torch` - Whisper, Granite Speech, Qwen2-Audio
- `accelerate` - `device_map` placement
- `peft` - Granite Speech ships a **LoRA adapter** on its LLM and will not load without it
- `datasets` + `soundfile`/`librosa` - the LibriSpeech eval clips, ESC-50, and resampling
- `bitsandbytes` - optional 4-bit for the Qwen2-Audio section
- `jiwer` + `num2words` - the WER helpers above
- `sounddevice` - the live microphone demo (section 13)
- `pyecharts` + `pandas` - benchmark chart and table

Sample rates are the trap in this family: **every model here expects 16 kHz mono**, and passing 44.1 kHz audio to a processor that does not resample produces a transcript of a chipmunk. Resample explicitly with `librosa.resample` and assert the rate before you call the model.

All downloads (HF model + dataset cache, sample audio) land in `DL_tasks/datasets/`, which is gitignored.

------------------------------------------------------------------------

``` python
# Everything runs through Hugging Face transformers - no model-specific packages.
# %pip install -q torch transformers accelerate peft datasets soundfile librosa jiwer num2words pandas pyecharts sounddevice

# Optional: 4-bit quantization, needed only for the Qwen2-Audio section
# %pip install -q bitsandbytes
```

``` python
import ctypes
import ctypes.util
import gc
import time
from pathlib import Path

import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv

# Knowledge/.env sets HF_TOKEN - authenticated HF Hub requests get higher rate limits
load_dotenv(find_dotenv(usecwd=True))

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device != "cpu" else torch.float32
if device != "cpu":
    print(torch.cuda.get_device_name(0))
print("device:", device, "| dtype:", dtype)

# Models whose *download* is over ~8 GB are gated behind this. Flip it to True only if
# you have the disk and the patience: Qwen2-Audio-7B is a 16.8 GB pull even though it
# runs in 4-bit afterwards. Quantization shrinks VRAM, never the download.
RUN_HEAVY = False


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

SR = 16000  # every model in this notebook expects 16 kHz mono
```

    NVIDIA GeForce RTX 3060
    device: cuda:0 | dtype: torch.float16

``` python
import librosa
from datasets import load_dataset
from IPython.display import Audio as AudioPlayer
from IPython.display import display

# Eval set: 73 clips of LibriSpeech validation-clean with reference transcripts (~9 MB).
# Small, ungated, and already at 16 kHz - the standard smoke-test set for ASR code.
ls = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", cache_dir=HF_CACHE
)

N_EVAL = 8
eval_audio = [np.asarray(ls[i]["audio"]["array"], dtype=np.float32) for i in range(N_EVAL)]
eval_refs = [ls[i]["text"] for i in range(N_EVAL)]
eval_rates = {ls[i]["audio"]["sampling_rate"] for i in range(N_EVAL)}
assert eval_rates == {SR}, f"expected 16 kHz, got {eval_rates}"

speech = eval_audio[0]
print(f"{N_EVAL} eval clips, {sum(len(a) for a in eval_audio) / SR:.1f}s total audio")
print("clip 0:", f"{len(speech) / SR:.1f}s |", eval_refs[0][:90])
display(AudioPlayer(speech, rate=SR))

# A non-speech probe: one ESC-50 environmental clip. A transcript of this is empty,
# which is exactly the point made in section 10.
esc = load_dataset("ashraq/esc50", split="train", cache_dir=HF_CACHE)
row = next(r for r in esc.select(range(200)) if r["category"] == "dog")
noise = librosa.resample(
    np.asarray(row["audio"]["array"], dtype=np.float32),
    orig_sr=row["audio"]["sampling_rate"], target_sr=SR,
)
print(f"\nnon-speech probe: '{row['category']}' | {len(noise) / SR:.1f}s")
display(AudioPlayer(noise, rate=SR))
```

    8 eval clips, 86.3s total audio
    clip 0: 5.9s | MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL

Your browser does not support the audio element.

    Repo card metadata block was not found. Setting CardData to empty.

    non-speech probe: 'dog' | 5.0s

Your browser does not support the audio element.

## 8. The Cascade: Whisper + an LLM

The architecture most production systems actually run. Two independent models: `openai/whisper-small` (0.24B, ~1 GB) turns audio into text, then `Qwen/Qwen3-1.7B` (~4 GB) answers the prompt about that text. Neither has ever seen the other.

Why it is still the default in 2026: each half can be swapped, upgraded, cached and debugged on its own; the ASR half can be a tiny CTC model running at RTFx 1000; the LLM half is whatever you already run for text. And the transcript is a human-readable artefact you can log, diff and show to an auditor - which matters more in regulated deployments than any benchmark point.

Watch for its two structural failures, both visible below: it cannot answer a question about *how* something was said, and any ASR error propagates into the answer with full confidence.

**Note on `enable_thinking`.** Qwen3 models emit a `<think>` block by default. For a short extraction prompt that is wasted latency, so the chat template is called with `enable_thinking=False`.

------------------------------------------------------------------------

``` python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# --- half 1: the ears -------------------------------------------------------------
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    dtype=dtype,
    device=device,
    model_kwargs={"cache_dir": HF_CACHE},
)
vram("whisper loaded")

t0 = time.perf_counter()
transcript = asr(speech.copy(), generate_kwargs={"language": "en", "task": "transcribe"})["text"].strip()
asr_time = time.perf_counter() - t0
print(f"[{asr_time:.2f}s] whisper-small: {transcript}")
print(f"          reference: {eval_refs[0]}")
print(f"          WER: {wer(eval_refs[0], transcript):.3f}")

# --- half 2: the brain ------------------------------------------------------------
llm_id = "Qwen/Qwen3-1.7B"
llm_tok = AutoTokenizer.from_pretrained(llm_id, cache_dir=HF_CACHE)
llm = AutoModelForCausalLM.from_pretrained(
    llm_id, dtype=dtype, device_map=device, low_cpu_mem_usage=True, cache_dir=HF_CACHE
)
vram("cascade loaded")


def ask_llm(text, question, max_new_tokens=120):
    "Prompt the text LLM about a transcript. The LLM never touches the waveform."
    messages = [{"role": "user", "content": f"Transcript of an audio clip:\n\"{text}\"\n\n{question}"}]
    # transformers 5.x returns a BatchEncoding here (return_dict defaults to True), not a
    # bare tensor - so ask for it explicitly and read the prompt length off input_ids.
    enc = llm_tok.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors="pt", enable_thinking=False,   # skip the <think> block
    ).to(llm.device)
    with torch.inference_mode():
        out = llm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    return llm_tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def cascade(audio, question, max_new_tokens=120):
    "ASR then LLM. Returns (answer, transcript) so the intermediate stays inspectable."
    text = asr(audio.copy(), generate_kwargs={"language": "en", "task": "transcribe"})["text"].strip()
    if question.lower().startswith("transcribe"):
        return text, text          # the transcription task needs no LLM at all
    return ask_llm(text, question, max_new_tokens), text


for q in [
    "Summarise this in one sentence.",
    "What emotion is in the speaker's voice? Answer in one word.",
]:
    t0 = time.perf_counter()
    answer, text = cascade(speech, q)
    print(f"\nQ: {q}\n  [{time.perf_counter() - t0:.1f}s] {answer}")

vram("cascade done")
```

    VRAM whisper loaded        0.50 GB allocated /  0.52 GB reserved

    [transformers] A custom logits processor of type <class 'transformers.generation.logits_process.SuppressTokensLogitsProcessor'> has been passed to `.generate()`, but it was also created in `.generate()`, given its parameterization. The custom <class 'transformers.generation.logits_process.SuppressTokensLogitsProcessor'> will take precedence. Please check the docstring of <class 'transformers.generation.logits_process.SuppressTokensLogitsProcessor'> to see related `.generate()` flags.
    [transformers] A custom logits processor of type <class 'transformers.generation.logits_process.SuppressTokensAtBeginLogitsProcessor'> has been passed to `.generate()`, but it was also created in `.generate()`, given its parameterization. The custom <class 'transformers.generation.logits_process.SuppressTokensAtBeginLogitsProcessor'> will take precedence. Please check the docstring of <class 'transformers.generation.logits_process.SuppressTokensAtBeginLogitsProcessor'> to see related `.generate()` flags.
    [transformers] Ignoring clean_up_tokenization_spaces=True for BPE tokenizer WhisperTokenizer. The clean_up_tokenization post-processing step is designed for WordPiece tokenizers and is destructive for BPE (it strips spaces before punctuation). Set clean_up_tokenization_spaces=False to suppress this warning, or set clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output=True to force cleanup anyway.

    [0.80s] whisper-small: Mr. Quilter is the Apostle of the Middle Classes, and we are glad to welcome his Gospel.
              reference: MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL
              WER: 0.059

    VRAM cascade loaded        3.95 GB allocated /  4.91 GB reserved

    Q: Summarise this in one sentence.
      [0.7s] Mr. Quilter is introduced as the Apostle of the Middle Classes in this audio clip.

    Q: What emotion is in the speaker's voice? Answer in one word.
      [0.5s] The emotion in the speaker's voice is **conveying**.
    VRAM cascade done          3.95 GB allocated /  4.91 GB reserved

## 9. Granite Speech 3.3 2B - a native audio LLM that fits

IBM’s Granite Speech 3.3 2B (2025, Apache 2.0) is the smallest **transformers-native** audio LLM that is genuinely good at both halves of the job. Architecture: a Conformer encoder trained with CTC on character targets, a window query-transformer that pools it down to ~6.25 tokens per second, and a **LoRA adapter** on the Granite 3.3 2B LLM that activates only when audio is present. That last detail is why it does not lose text ability: with no audio in the prompt it is exactly the base text model.

Two things to know before running it:

- **It needs `peft`.** The LoRA adapter is a separate `adapter_model.safetensors`; without `peft` installed the load fails or silently gives you the text-only model.
- **It is a two-pass model by design.** IBM’s own guidance is to transcribe first, then ask questions about the transcript, because the LLM half was not trained to do both in one turn. It genuinely hears the audio, but it is not a general audio-event describer - for “what is that noise” you want Qwen2-Audio or Qwen3-Omni.

Download is ~6 GB (the repo also carries an older 3-shard copy of the weights; `from_pretrained` fetches only the 4-shard set listed in the index).

------------------------------------------------------------------------

``` python
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

granite_id = "ibm-granite/granite-speech-3.3-2b"
gr_proc = AutoProcessor.from_pretrained(granite_id, cache_dir=HF_CACHE)
gr_tok = gr_proc.tokenizer
granite = AutoModelForSpeechSeq2Seq.from_pretrained(
    granite_id, dtype=dtype, device_map=device, low_cpu_mem_usage=True, cache_dir=HF_CACHE
)
vram("granite loaded")


def granite_ask(audio, prompt, max_new_tokens=200):
    "Prompt Granite Speech with one 16 kHz clip. `<|audio|>` marks where the clip goes."
    chat = [
        {"role": "system", "content": "Knowledge Cutoff Date: April 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant."},
        {"role": "user", "content": f"<|audio|>{prompt}"},
    ]
    text = gr_tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    # The processor wants a (1, samples) float tensor at 16 kHz.
    wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    inputs = gr_proc(text, wav, return_tensors="pt").to(granite.device)
    with torch.inference_mode():
        out = granite.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            pad_token_id=gr_tok.pad_token_id,
        )
    return gr_tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


t0 = time.perf_counter()
gr_transcript = granite_ask(speech, "can you transcribe the speech into a written format?")
print(f"[{time.perf_counter() - t0:.1f}s] transcribe: {gr_transcript}")
print(f"          reference: {eval_refs[0]}")
print(f"          WER: {wer(eval_refs[0], gr_transcript):.3f}\n")

# Same weights, different instruction. Note the model is answering about audio it heard,
# not about a transcript someone handed it.
for q in ["Summarise what the speaker says in one short sentence.",
          "Is the speaker male or female? Answer with one word."]:
    t0 = time.perf_counter()
    print(f"Q: {q}\n  [{time.perf_counter() - t0:.1f}s] {granite_ask(speech, q, max_new_tokens=80)}\n")

vram("granite done")
```

    VRAM granite loaded       10.01 GB allocated / 10.07 GB reserved

    [transformers] Ignoring clean_up_tokenization_spaces=True for BPE tokenizer GPT2Tokenizer. The clean_up_tokenization post-processing step is designed for WordPiece tokenizers and is destructive for BPE (it strips spaces before punctuation). Set clean_up_tokenization_spaces=False to suppress this warning, or set clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output=True to force cleanup anyway.

    [0.7s] transcribe: mister quilter is the apostle of the middle classes and we are glad to welcome his gospel
              reference: MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL
              WER: 0.000

    Q: Summarise what the speaker says in one short sentence.
      [0.0s] mister quilter is the apostle of the middle classes and we are glad to welcome his gospel

    Q: Is the speaker male or female? Answer with one word.
      [0.0s] mister quilter is the apostle of the middle classes and we are glad to welcome his gospel

    VRAM granite done         10.01 GB allocated / 10.07 GB reserved

## 10. What the cascade cannot hear

The interesting comparison is not WER. Both architectures transcribe read audiobook speech about equally well. The interesting comparison is what happens when the answer is **not in the words**.

Three probes:

1.  **A non-speech clip** (a dog barking). The cascade’s ASR produces an empty string or a hallucinated sentence, and its LLM then answers a question about nothing. A native model can at least try.
2.  **A paralinguistic question** (“male or female”, “how fast is the speaker talking”). The transcript has no such information at all, so the cascade must guess from word choice.
3.  **An ASR error probe.** Feed the LLM a transcript with a plausible mistake and watch the answer inherit it with full confidence.

The third one is the failure mode nobody plans for: cascades do not degrade gracefully, they degrade *confidently*.

------------------------------------------------------------------------

``` python
# Probe 1 and 2: the cascade on audio that has no words in it.
answer, text = cascade(noise, "What sound is in this clip? Answer in three words.")
print("CASCADE on a dog bark")
print(f"  whisper heard : {text!r}")
print(f"  llm answered  : {answer}\n")

print("GRANITE on the same clip")
print("  ", granite_ask(noise, "What sound is in this clip? Answer in three words.", max_new_tokens=40))

# Probe 3: error propagation. Nothing here is hypothetical - swap one plausible word in
# the transcript and ask a question whose answer depends on it.
print("\nERROR PROPAGATION (both answers are confident, one is wrong)")
good = "the patient should take fifteen milligrams twice a day"
bad = "the patient should take fifty milligrams twice a day"   # 15 -> 50, a classic ASR slip
for t in (good, bad):
    print(f"  transcript {t[-32:]!r:36s} -> {ask_llm(t, 'What is the total daily dose? Answer with a number and unit.', 32)}")

# Sections 8-10 are done. Free BOTH halves of the cascade *and* Granite: section 12
# loads its own copy of each, and a 2B Granite is ~5 GB, so leaving this one live means
# two of them on a 12 GB card - an OOM partway through the benchmark.
# Popping instead of `del` keeps the cell re-runnable after a partial run.
for _name in ["asr", "llm", "llm_tok", "granite", "gr_proc", "gr_tok"]:
    globals().pop(_name, None)
free_memory()
vram("after cascade + granite")
```

    CASCADE on a dog bark
      whisper heard : 'WAH!'
      llm answered  : A loud noise.

    GRANITE on the same clip
       

    ERROR PROPAGATION (both answers are confident, one is wrong)
      transcript 'e fifteen milligrams twice a day'   -> The total daily dose is **150 mg**.
      transcript 'ake fifty milligrams twice a day'   -> The total daily dose is **250 mg**.
    VRAM after cascade + granite  0.01 GB allocated /  0.02 GB reserved

## 11. Qwen2-Audio-7B-Instruct (4-bit)

The reference open audio LLM, and the one most papers compare against: a Whisper-large encoder feeding Qwen2-7B, instruction-tuned over speech, general sound *and* music. Unlike Granite Speech it was explicitly trained to describe non-speech audio, so “what is that noise” is in scope.

It is 8.4B parameters, which is ~17 GB in bf16 and does not fit this card. In **4-bit NF4** it sits at roughly 6 GB of VRAM and works fine. The catch that quantization does not solve: **the download is still 16.8 GB**, because `bitsandbytes` quantizes after fetching the full-precision weights. That is why this section is behind `RUN_HEAVY` - flip it in the Setup cell if you have the disk.

`Qwen/Qwen2-Audio-7B-Instruct` uses a chat template with an `audio_url` content part; the processor takes the decoded arrays separately via `audio=`.

------------------------------------------------------------------------

``` python
if not RUN_HEAVY:
    print("skipped: Qwen2-Audio-7B-Instruct is a 16.8 GB download.\n"
          "Set RUN_HEAVY = True in the Setup cell to run it (it needs ~6 GB VRAM in 4-bit).")
else:
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2AudioForConditionalGeneration

    qa_id = "Qwen/Qwen2-Audio-7B-Instruct"
    qa_proc = AutoProcessor.from_pretrained(qa_id, cache_dir=HF_CACHE)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    qa = Qwen2AudioForConditionalGeneration.from_pretrained(
        qa_id, quantization_config=quant, device_map=device,
        low_cpu_mem_usage=True, cache_dir=HF_CACHE,
    )
    vram("qwen2-audio 4-bit")

    def qwen_audio_ask(audio, prompt, max_new_tokens=160):
        "One clip plus one instruction through Qwen2-Audio's chat template."
        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio_url": "clip.wav"},   # placeholder; the array goes in below
            {"type": "text", "text": prompt},
        ]}]
        text = qa_proc.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = qa_proc(
            text=text, audio=[np.asarray(audio, dtype=np.float32)],
            sampling_rate=SR, return_tensors="pt", padding=True,
        ).to(qa.device)
        with torch.inference_mode():
            out = qa.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return qa_proc.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()

    for clip, q in [(speech, "Transcribe the speech in this clip."),
                    (speech, "Describe the speaker's voice: gender, pace, tone."),
                    (noise, "What sound is this? Describe the scene it comes from.")]:
        t0 = time.perf_counter()
        print(f"Q: {q}\n  [{time.perf_counter() - t0:.1f}s] {qwen_audio_ask(clip, q)}\n")

    del qa, qa_proc
    free_memory()
    vram("after qwen2-audio")
```

    skipped: Qwen2-Audio-7B-Instruct is a 16.8 GB download.
    Set RUN_HEAVY = True in the Setup cell to run it (it needs ~6 GB VRAM in 4-bit).

## 12. Head-to-head Benchmark

Two architectures on the **same 8 LibriSpeech clips**, the **same transcription instruction**, and the **same normalisation** from section 4: the Whisper-small cascade against Granite Speech 3.3 2B. Reported: WER, real-time factor (processing seconds per audio second) and wall-clock. Each system is loaded, measured, and freed before the next loads, so VRAM stays flat.

**This is a smoke test, not a leaderboard.** Eight clips of clean read audiobook English is the easiest audio in existence, and published Open ASR Leaderboard numbers average over eight datasets including telephone and spontaneous speech. What the sample does show honestly is the relative *speed* of a 0.24B specialist against a 2B generalist, and that a native audio LLM does not have to give up transcription accuracy to gain instruction following.

------------------------------------------------------------------------

``` python
def load_cascade_asr():
    "Whisper-small on its own: the transcription half of the cascade."
    p = pipeline("automatic-speech-recognition", model="openai/whisper-small",
                 dtype=dtype, device=device, model_kwargs={"cache_dir": HF_CACHE})

    def run(audio):
        return p(audio.copy(), generate_kwargs={"language": "en", "task": "transcribe"})["text"].strip()

    return run, [p]


def load_granite():
    "Granite Speech 3.3 2B answering the transcription instruction."
    proc = AutoProcessor.from_pretrained(granite_id, cache_dir=HF_CACHE)
    tok = proc.tokenizer
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        granite_id, dtype=dtype, device_map=device, low_cpu_mem_usage=True, cache_dir=HF_CACHE
    )

    def run(audio):
        chat = [
            {"role": "system", "content": "Knowledge Cutoff Date: April 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant."},
            {"role": "user", "content": "<|audio|>can you transcribe the speech into a written format?"},
        ]
        text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
        inputs = proc(text, wav, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                                 num_beams=1, pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    return run, [model, proc, tok]


def benchmark(name, loader):
    "Load, transcribe every eval clip, score, free. One model live at a time."
    # Reclaim first: a model left live by an earlier section is what OOMs this cell.
    free_memory()
    if torch.cuda.is_available() and torch.cuda.memory_allocated() > 1e9:
        vram(f"WARNING stale model before {name}")
    run, handles = loader()
    hyps, t0 = [], time.perf_counter()
    for audio in eval_audio:
        hyps.append(run(audio))
    elapsed = time.perf_counter() - t0
    audio_seconds = sum(len(a) for a in eval_audio) / SR
    score = float(np.mean([wer(r, h) for r, h in zip(eval_refs, hyps)]))
    for h in handles:
        del h
    del run, handles
    free_memory()
    vram(f"after {name}")
    return {"model": name, "wer": round(score, 4), "seconds": round(elapsed, 2),
            "rtf": round(elapsed / audio_seconds, 3), "hyps": hyps}


results = [
    benchmark("whisper-small (cascade ASR)", load_cascade_asr),
    benchmark("granite-speech-3.3-2b", load_granite),
]
print(f"\n{sum(len(a) for a in eval_audio) / SR:.1f}s of audio, {N_EVAL} clips")
```

    VRAM after whisper-small (cascade ASR)  0.01 GB allocated /  0.02 GB reserved

    VRAM after granite-speech-3.3-2b  0.01 GB allocated /  0.02 GB reserved

    86.3s of audio, 8 clips

``` python
import pandas as pd

df = pd.DataFrame([{k: v for k, v in r.items() if k != "hyps"} for r in results]).sort_values("wer")
df
```

|     | model                       | wer    | seconds | rtf   |
|-----|-----------------------------|--------|---------|-------|
| 1   | granite-speech-3.3-2b       | 0.0136 | 7.55    | 0.087 |
| 0   | whisper-small (cascade ASR) | 0.0848 | 2.77    | 0.032 |

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
    .add_yaxis("WER %", [round(r["wer"] * 100, 2) for r in results])
    .add_yaxis("real-time factor x100", [round(r["rtf"] * 100, 2) for r in results])
    .set_global_opts(
        title_opts=opts.TitleOpts(
            title=f"Transcription instruction on {N_EVAL} LibriSpeech clips",
            subtitle="RTX 3060 12 GB, fp16, shared normalisation - smoke test, not a leaderboard",
        ),
        xaxis_opts=opts.AxisOpts(name="system", axislabel_opts=opts.LabelOpts(rotate=15)),
        yaxis_opts=opts.AxisOpts(name="lower is better"),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
    )
)
bar.render_notebook()
```

``` python
# The numbers hide the interesting part: read what each system actually produced.
for i in range(3):
    print(f"ref  : {eval_refs[i]}")
    for r in results:
        print(f"  {r['model']:28s} [WER {wer(eval_refs[i], r['hyps'][i]):.2f}] {r['hyps'][i]}")
    print()
```

    ref  : MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL
      whisper-small (cascade ASR)  [WER 0.06] Mr. Quilter is the Apostle of the Middle Classes, and we are glad to welcome his Gospel.
      granite-speech-3.3-2b        [WER 0.00] mister quilter is the apostle of the middle classes and we are glad to welcome his gospel

    ref  : NOR IS MISTER QUILTER'S MANNER LESS INTERESTING THAN HIS MATTER
      whisper-small (cascade ASR)  [WER 0.09] Nor is Mr. Quilter's manner less interesting than his matter.
      granite-speech-3.3-2b        [WER 0.00] nor is mister quilter's manner less interesting than his matter

    ref  : HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND
      whisper-small (cascade ASR)  [WER 0.03] He tells us that at this festive season of the year, with Christmas and roast beef looming before us, symbolies drawn from eating and its results occur most readily to the mind.
      granite-speech-3.3-2b        [WER 0.00] he tells us that at this festive season of the year with christmas and roast beef looming before us similes drawn from eating and its results occur most readily to the mind

## 13. Live Demo: talk to the model

Records from the microphone, then runs the same clip through Granite Speech twice: once asking for a transcript, once asking your own question about what it heard. This is the cell people run on its own, so it opens with a `require(...)` guard naming what it needs from Setup rather than dying on a bare `NameError`.

Microphone notes, all measured on the knowledge-lab container:

- **PortAudio device names are not ALSA card ids.** The camera mic is ALSA card `U2K` but PortAudio calls it `UGREEN camera 2K: USB Audio (hw:0,0)`. `MIC_HINT` matches the PortAudio name.
- **Record at the device’s native rate and resample yourself.** Asking PortAudio for 16 kHz on a device that runs at 48 kHz either fails or silently gives you resampled-by-the-driver audio of unknown quality.
- If there is no `/dev/snd`, this raises a clear error rather than faking a stream. On the LXC the ALSA nodes come from `av_devices` in `infra/proxmox/variables.tf`.

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


require("device", "dtype", "HF_CACHE", "SR", "free_memory", "vram", "granite_id", "wer")

import time

import librosa
import numpy as np
import sounddevice as sd
import torch
from IPython.display import Audio as AudioPlayer
from IPython.display import display

from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

# Substring of the *PortAudio* device name, which is not the ALSA card id: this mic is
# card id "U2K" but shows up as "UGREEN camera 2K: USB Audio (hw:0,0)". Run
# `python -c "import sounddevice; print(sounddevice.query_devices())"` to list them.
MIC_HINT = "UGREEN"      # None -> just take the first input device
RECORD_SECONDS = 6
QUESTION = "Summarise what the speaker says in one short sentence."


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


def record(seconds=RECORD_SECONDS):
    "Record mono from the chosen mic at its native rate, then resample to 16 kHz."
    idx, native_sr = pick_microphone()
    print(f"recording {seconds}s from device {idx} at {native_sr} Hz - speak now")
    raw = sd.rec(int(seconds * native_sr), samplerate=native_sr, channels=1,
                 dtype="float32", device=idx)
    sd.wait()
    audio = raw[:, 0]
    if native_sr != SR:
        audio = librosa.resample(audio, orig_sr=native_sr, target_sr=SR)
    peak = float(np.abs(audio).max())
    print(f"captured {len(audio) / SR:.1f}s at {SR} Hz | peak {peak:.3f}"
          + ("  <- very quiet, check the input gain" if peak < 0.02 else ""))
    return audio


# Re-runnable: this cell frees the model at the end, so guard the load or a second
# shift-enter raises NameError on `live_proc`.
if "live_model" not in globals():
    live_proc = AutoProcessor.from_pretrained(granite_id, cache_dir=HF_CACHE)
    live_tok = live_proc.tokenizer
    live_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        granite_id, dtype=dtype, device_map=device, low_cpu_mem_usage=True, cache_dir=HF_CACHE
    )
    vram("live model")


def live_ask(audio, prompt, max_new_tokens=160):
    "Same call shape as section 9, on the freshly recorded clip."
    chat = [
        {"role": "system", "content": "Knowledge Cutoff Date: April 2024.\nYou are Granite, developed by IBM. You are a helpful AI assistant."},
        {"role": "user", "content": f"<|audio|>{prompt}"},
    ]
    text = live_tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    wav = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    inputs = live_proc(text, wav, return_tensors="pt").to(live_model.device)
    with torch.inference_mode():
        out = live_model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  num_beams=1, pad_token_id=live_tok.pad_token_id)
    return live_tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


clip = record()
display(AudioPlayer(clip, rate=SR))

t0 = time.perf_counter()
said = live_ask(clip, "can you transcribe the speech into a written format?")
print(f"\n[{time.perf_counter() - t0:4.1f}s] transcript : {said}")

t0 = time.perf_counter()

print(f"[{time.perf_counter() - t0:4.1f}s] {QUESTION}\n            {live_ask(clip, QUESTION, 120)}")

del live_model, live_proc, live_tok
free_memory()
vram("final")
```

    VRAM live model            6.06 GB allocated /  6.13 GB reserved
    recording 6s from device 0 at 44100 Hz - speak now
    captured 6.0s at 16000 Hz | peak 1.278

Your browser does not support the audio element.

    [ 0.2s] transcript : 
    [ 0.0s] Summarise what the speaker says in one short sentence.
                
    VRAM final                 0.01 GB allocated /  0.02 GB reserved

## 14. Common Frameworks

The frameworks here divide along the same line the notebook does: the cascade and the native model. A cascade is a *pipeline* problem - VAD, chunking, ASR, an LLM - and inherits the tooling of each stage. A native audio LLM is a *serving* problem, and inherits the LLM stack. The architectural choice from section 8 therefore decides most of your dependency list, which is a better reason to make it deliberately than the accuracy numbers are.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [transformers](https://github.com/huggingface/transformers) | modelling | Whisper, Granite Speech, Qwen2-Audio, Voxtral and Moshi under one API, plus the chat templating that makes audio a message part | Apache 2.0 | Default, and it covers both the cascade and the native path |
| [peft](https://github.com/huggingface/peft) + [trl](https://github.com/huggingface/trl) | modelling | LoRA on the LLM half plus projector training - the Ultravox recipe, which retargets an audio front end onto a different LLM in GPU-hours | Apache 2.0 | Adapting to a domain. Training only the adapter with both towers frozen is the cheap, effective move |
| [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) | modelling | Canary and Parakeet - the top of the Open ASR Leaderboard, and still `.nemo`-only | Apache 2.0 (weights often CC-BY-4.0) | The transcription half of a cascade needs the last point of accuracy |
| [Silero VAD](https://github.com/snakers4/silero-vad) + [torchaudio](https://github.com/pytorch/audio) | data | Silence removal, chunking with overlap, and resampling to the 16 kHz every one of these models assumes | MIT / BSD-2 | Any recording longer than a minute. Nothing in this notebook handles a two-hour file in one call - see `Audio/05_Voice_Activity_Detection` |
| [pyannote.audio](https://github.com/pyannote/pyannote-audio) | data | Speaker attribution, so “who asked what” survives into the prompt | MIT (models gated) | Meetings and interviews, where an unattributed transcript answers the wrong question |
| [vLLM](https://github.com/vllm-project/vllm) / [SGLang](https://github.com/sgl-project/sglang) | inference runtime | Continuous batching for Voxtral, Qwen2-Audio and Qwen3-Omni - many concurrent audio requests per GPU | Apache 2.0 | Serving a native audio LLM. `generate` in a handler holds a whole card per request |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) / [llama.cpp](https://github.com/ggml-org/llama.cpp) + [optimum](https://github.com/huggingface/optimum) | inference runtime | GGUF quantisation for both halves of a cascade, and ONNX export for the Whisper stage on CPU | MIT / Apache 2.0 | Edge or CPU-only deployment, where a cascade of two small models beats one native model comfortably |
| [Pipecat](https://github.com/pipecat-ai/pipecat) / [LiveKit Agents](https://github.com/livekit/agents) | orchestration | The conversational loop a cascade needs to feel live: turn detection, barge-in, streaming playback | BSD-2 / Apache 2.0 | The product is a conversation rather than a request. See `Frameworks/01_pipecat` |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) + AIR-Bench / VoiceBench | evaluation | Instruction-following scored on audio inputs, and the audio-specific benchmarks that probe paralinguistics | MIT / Apache 2.0 | Comparing a cascade against a native model. Section 10 is the small version - a cascade loses everything that is not words |

**The 2026 default stack** is a cascade unless you have measured that you need otherwise: Silero VAD, a strong ASR model, and a text LLM you already trust, glued with Pipecat if it is conversational. Native audio LLMs earn their place when tone, emotion or overlapping speech carry meaning, or when full-duplex latency is the requirement.

The common wrong turn is buying a native audio model for a job the cascade does better. Text LLMs are far stronger reasoners than any audio LLM that fits on this card, and a cascade lets you upgrade each half independently. The second is forgetting that the cascade’s transcript is lossy in a specific way - it throws away exactly the sarcasm, hesitation and speaker overlap that the native model was bought for.

------------------------------------------------------------------------

## 15. Going Further

- **Fine-tuning.** The cheap and effective move is to train **only the projector/adapter** and leave both towers frozen - that is the Ultravox recipe, and it is what lets you retarget an audio front end onto a different LLM in GPU-hours rather than GPU-weeks. For Granite Speech and Qwen2-Audio, LoRA through `peft` on the LLM half plus the projector fits on this 12 GB card at short audio lengths. The HF [audio course](https://huggingface.co/learn/audio-course) chapter on ASR fine-tuning is the right starting template; swap the loss target for instruction data.
- **Long audio.** Nothing here handles a two-hour recording in one call. The production pattern is: VAD (see `Audio/05_Voice_Activity_Detection`) to cut silence, diarisation to attribute speakers, chunk with 2-5 s overlap, transcribe, then reason over the *text* with a long-context LLM. Audio Flamingo 3 (10 min context) and the chunked long-form path in `Audio/02_Automatic_Speech_Recognition` are the two alternatives.
- **Streaming and full duplex.** Anything conversational needs an architecture that emits while listening. [Moshi](https://huggingface.co/kyutai/moshiko-pytorch-bf16) (`MoshiForConditionalGeneration` is in transformers) models both streams at 12.5 Hz; Qwen3-Omni’s Talker streams the first audio packet in ~234 ms. Neither is a drop-in replacement for the request/response code above.
- **Related notebooks.** `Audio/02_Automatic_Speech_Recognition` (the ASR half in depth, including chunked long-form and streaming), `Audio/04_Audio_Classification` (fixed-label alternative when you do not need free text), `Audio/05_Voice_Activity_Detection` (the cheap gate in front of everything), `Multimodal/08_Any_to_Any` (models that answer *in speech*), and `Multimodal/06_Video_Text_to_Text` (audio as one track among several).

------------------------------------------------------------------------

Back to top
