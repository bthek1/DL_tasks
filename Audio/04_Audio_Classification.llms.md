# Audio Classification

Labelling sound - tagging, keyword spotting, emotion, zero-shot: how the models work, the mid-2026 landscape, evaluation, and runnable transformers code on GPU or CPU.

Author

Benedict Thekkel

## 1. What is Audio Classification?

Audio classification maps an audio clip to one or more **labels**. Common variants:

- **Sound event tagging** - what sounds are present (AudioSet’s 527 classes, ESC-50).
- **Keyword spotting (KWS)** - detect command words (“yes”, “stop”).
- **Speech emotion recognition** - happy / sad / angry / neutral.
- **Speaker identification** and **language ID**.
- **Music genre / auto-tagging**, **acoustic scene classification**.

**Output.** A single label (multi-class) or many labels with scores (multi-label tagging).

**Neighbouring tasks:** Voice Activity Detection (a binary special case), ASR, and audio captioning.

------------------------------------------------------------------------

## 2. Real-World Use Cases

Audio classification is the workhorse of “listen for something” systems. Most deployments are **always-on**, which is why the binding constraints are power, false-alarm rate and privacy far more often than top-1 accuracy.

| Use case | Domain | Consumes / produces | Dominant constraint |
|----|----|----|----|
| Wake words and keyword spotting | Consumer devices | Ring buffer of mic audio -\> “is this the wake word?” | Milliwatts, tiny memory, and an extremely low false-accept rate |
| Music tagging and catalogue search | Streaming, media | Track -\> genre, mood, instrument tags | Throughput and cost over millions of tracks; multi-label output |
| Safety and security event detection | Smart home, public safety (glass break, smoke alarm, baby cry) | Continuous mic -\> event class | False positives per hour; on-device, since a mic in the home is a privacy surface |
| Machine condition monitoring | Manufacturing, predictive maintenance | Machine audio -\> normal vs anomalous | Anomaly detection under domain shift; runs on edge hardware in a noisy plant |
| Bioacoustics and biodiversity | Conservation research (BirdNET) | Field-recorder audio -\> species ID | Recall on rare classes; battery life on a solar-powered recorder |
| Emotion, language and intent triage | Contact-centre analytics | Call audio -\> emotion, language, escalation flag | Consistency across accents; drives routing decisions |
| Cough and respiratory screening | Health | Cough audio -\> condition score | Clinical validation and severe class imbalance |

**What the AudioSet mAP hides.** In an always-on system the metric that decides whether you can ship is **false alarms per hour**, not accuracy - a smoke-alarm detector that is 99% accurate but fires spuriously twice a day gets switched off by the user, and a wake word that self-triggers once an hour is unusable no matter what its top-1 says. **Domain shift** is the second killer: change the microphone, the room, or the distance to the source, and a model tuned on clean benchmark clips degrades hard, so a real evaluation set has to come from the deployment hardware. Third, the world is **open-set** - production audio constantly contains sounds outside the taxonomy, so a fixed-label classifier will confidently assign one of its known classes to something it has never heard, which is why rejection thresholds or a zero-shot model like CLAP matter more than another point of mAP. Finally, benchmark labels themselves are noisy and wildly imbalanced, so a headline mAP is an average over classes you may not care about at all.

------------------------------------------------------------------------

## 3. How Modern Audio Classification Works

1.  **Hand-crafted features + GMM/SVM (pre-2017).** MFCCs into a shallow classifier. Legacy.
2.  **CNN on log-mel (VGGish, PANNs, 2019).** Treat the spectrogram as an image; large-scale AudioSet pretraining transfers well.
3.  **Audio Spectrogram Transformer (AST, 2021).** A ViT over spectrogram patches - AudioSet state of the art and the default tagging backbone.
4.  **Self-supervised encoders (wav2vec 2.0, HuBERT, WavLM, 2021).** Pretrain on unlabeled speech, fine-tune a light head for KWS, emotion, speaker ID; **BEATs** (2022) leads general audio tagging.
5.  **CLAP (2023).** Contrastive language-audio pretraining enables **zero-shot** classification: score the clip against arbitrary text label prompts, no fine-tuning. By 2025-2026 CLAP-style zero-shot and unified audio encoders are the flexible default.

------------------------------------------------------------------------

## 4. Evaluation Metrics

- **Accuracy / F1.** For single-label tasks (KWS, emotion, ESC-50).
- **mAP (mean Average Precision).** The standard for **multi-label** tagging (AudioSet) - averages precision across recall for each class.
- **d-prime / AUC.** Alternative multi-label summaries.
- **Confusion matrix.** Where the errors go - shown as an ECharts heatmap in the benchmark.

The cell computes accuracy and macro-F1 with `scikit-learn` on toy predictions.

------------------------------------------------------------------------

``` python
from sklearn.metrics import accuracy_score, f1_score

y_true = ["dog", "rain", "dog", "engine", "rain"]
y_pred = ["dog", "rain", "cat", "engine", "rain"]
print("accuracy:", accuracy_score(y_true, y_pred))
print("macro-F1:", f1_score(y_true, y_pred, average="macro"))
```

    accuracy: 0.8
    macro-F1: 0.6666666666666666

## 5. Datasets

Audio classification ground truth is a label someone attached to a clip, and its quality depends on how hard they listened. Large sets like AudioSet are weakly labelled: a 10 s clip says “dog” somewhere, with no timing and many missed labels. Small sets like ESC-50 are clean, single-label and balanced, but tiny. The typical failure is a model that tops a curated benchmark and then misfires on real recordings with overlapping sounds, or where the class never existed in the label set.

| Dataset | Contents | Size | Scope | License | Typical use |
|----|----|----|----|----|----|
| [AudioSet](https://huggingface.co/datasets/agkphysics/AudioSet) | 10 s YouTube clips, weak multi-label tags from a 632-class ontology | ~2.08M clips (22,176 balanced train / 2,042,985 unbalanced / 20,383 eval), ~5.8K h, 527 labels | General sound events | CC BY 4.0 (labels); ontology CC BY-SA 4.0 | **The** pretraining set and mAP benchmark (AST, BEATs, PANNs) |
| [ESC-50](https://huggingface.co/datasets/ashraq/esc50) ([official](https://github.com/karolpiczak/ESC-50)) | 5 s Freesound clips, 44.1 kHz mono, 40 per class | 2,000 clips, 50 classes, 5 folds | Environmental sounds | CC BY-NC 3.0 (ESC-10 subset CC BY 3.0) | Standard small eval (5-fold accuracy); zero-shot CLAP eval; **used in this notebook** |
| [UrbanSound8K](https://huggingface.co/datasets/danavery/urbansound8K) ([official](https://urbansounddataset.weebly.com/urbansound8k.html)) | Urban sound excerpts up to 4 s | 8,732 clips, 10 classes, 10 folds | Street and city noise | CC BY-NC 4.0 | Urban sound eval (10-fold accuracy) |
| [Speech Commands](https://huggingface.co/datasets/google/speech_commands) | 1 s clips of single spoken command words plus background noise | 105,829 clips (v0.02) | English keywords | CC BY 4.0 | Keyword spotting on small devices |
| [FSD50K](https://huggingface.co/datasets/Fhrozen/FSD50k) ([official](https://zenodo.org/records/4060432)) | Freesound clips, human-verified labels from the AudioSet ontology | 51,197 clips, 108.3 h, 200 classes (dev 40,966 / eval 10,231) | General sound events | CC BY for the dataset; each clip keeps its own CC license (CC0, BY, BY-NC, Sampling+) | Open alternative to AudioSet; multi-label eval |
| [VGGSound](https://huggingface.co/datasets/Loie/VGGSound) ([official](https://www.robots.ox.ac.uk/~vgg/data/vggsound/)) | 10 s YouTube clips where the sound source is visible | 200K+ clips, 550+ h, 310+ classes | General sounds, audio-visual | CC BY 4.0 (video copyright stays with owners) | Audio-visual training and eval |

**This notebook evaluates on ESC-50**, via the parquet mirror `ashraq/esc50` (all 2,000 clips in one `train` split, ~770 MB). The benchmark scores CLAP zero-shot accuracy on the first 50 clips, prompting with “the sound of ” for all 50 class names. That is a smoke test. The real protocol is 5-fold cross-validation over the official folds, and zero-shot numbers are only comparable when the prompt template matches.

None of these are gated. **ESC-50 and UrbanSound8K are non-commercial**, and **FSD50K mixes per-clip licenses**, so some of its clips are non-commercial. AudioSet and VGGSound ship YouTube IDs, so the clips you can still download shrink every year: the AudioSet mirror above found 18,683 of the 22,160 balanced-train clips in March 2023.

Downloads land in `DL_tasks/datasets/` via `cache_dir` (gitignored).

------------------------------------------------------------------------

## 6. The Model Landscape (mid-2026)

| Model | Params | License | Task | Best for |
|----|----|----|----|----|
| MIT/ast-finetuned-audioset-10-10-0.4593 | 86M | BSD-3 | AudioSet tagging (527) | general sound tagging |
| superb/hubert-large-superb-er | 300M | Apache 2.0 | speech emotion | emotion recognition |
| superb/wav2vec2-base-superb-ks | 95M | Apache 2.0 | keyword spotting (35) | command words |
| laion/clap-htsat-unfused | 150M | Apache 2.0 | zero-shot | label from text prompts |
| MIT/ast-finetuned-speech-commands-v2 | 86M | BSD-3 | keyword spotting | speech commands |

Benchmarks: **SUPERB** (speech tasks), **AudioSet mAP** (tagging), **HEAR** (holistic). All of these load through the `transformers` `audio-classification` (or `zero-shot-audio-classification`) pipeline.

------------------------------------------------------------------------

## 7. Setup

Package roles:

- `transformers` (\>=5.13) + `torch` - AST, HuBERT, wav2vec2, CLAP
- `datasets` - ESC-50 environmental-sound clips with labels
- `scikit-learn` - accuracy / F1 / confusion matrix
- `pyecharts` - accuracy bar + confusion-matrix heatmap

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
from datasets import load_dataset

# ESC-50: 2000 labelled 5 s environmental-sound clips, 50 categories
esc = load_dataset("ashraq/esc50", split="train", cache_dir=str(DATA_DIR / "hf_cache"))
CATEGORIES = sorted(set(esc["category"]))
print(len(esc), "clips,", len(CATEGORIES), "categories")


def esc_clip(row):
    "Return {array, sampling_rate} for a transformers audio pipeline."
    a = row["audio"]
    return {"array": np.asarray(a["array"], dtype="float32"), "sampling_rate": a["sampling_rate"]}
```

    NVIDIA GeForce RTX 3060
    device: cuda:0

    Repo card metadata block was not found. Setting CardData to empty.

    2000 clips, 50 categories

## 8. AST - AudioSet tagging

The Audio Spectrogram Transformer tags a clip against AudioSet’s 527 classes. The `audio-classification` pipeline handles resampling to 16 kHz and returns scored labels; `top_k` sets how many.

------------------------------------------------------------------------

``` python
from transformers import pipeline

tagger = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593",
                  device=device, top_k=5)
clip = esc_clip(esc[0])
t0 = time.perf_counter()
print(f"true category: {esc[0]['category']}")
for p in tagger(clip):
    print(f"  {p['score']:.3f}  {p['label']}")
print(f"{time.perf_counter() - t0:.2f}s")

del tagger
free_memory()
vram("after ast")
```

    true category: dog
      0.187  Bark
      0.185  Animal
      0.101  Dog
      0.084  Bow-wow
      0.071  Domestic animals, pets
      0.064  Canidae, dogs, wolves
      0.024  Sound effect
      0.015  Crow
      0.014  Caw
      0.014  Sigh
      0.014  Yip
      0.014  Growling
      0.013  Speech
      0.011  Music
      0.010  Silence
      0.009  Quack
      0.008  Gasp
      0.007  Hiccup
      0.007  Whimper (dog)
      0.007  Grunt
      0.007  Frog
      0.006  Plop
      0.005  Male speech, man speaking
      0.005  Whoosh, swoosh, swish
      0.004  Groan
      0.004  Inside, small room
      0.003  Roar
      0.003  Duck
      0.002  Cattle, bovinae
      0.002  Roaring cats (lions, tigers)
      0.002  Sneeze
      0.002  Wild animals
      0.002  Livestock, farm animals, working animals
      0.002  Cat
      0.002  Screaming
      0.002  Zipper (clothing)
      0.002  Slap, smack
      0.002  Bird
      0.002  Breathing
      0.002  Bang
      0.002  Musical instrument
      0.001  Outside, rural or natural
      0.001  Boing
      0.001  Fart
      0.001  Whistle
      0.001  Whack, thwack
      0.001  Pig
      0.001  Ding-dong
      0.001  Ringtone
      0.001  Hiss
      0.001  Bellow
      0.001  Inside, large room or hall
      0.001  Outside, urban or manmade
      0.001  Percussion
      0.001  Vehicle
      0.001  Throat clearing
      0.001  Rustle
      0.001  Burst, pop
      0.001  Television
      0.001  Ping
      0.001  Smash, crash
      0.001  Fowl
      0.001  Pant
      0.001  Moo
      0.001  Howl
      0.001  Croak
      0.001  Meow
      0.001  Yell
      0.001  Drum
      0.001  Whimper
      0.001  Whoop
      0.001  Ding
      0.001  Honk
      0.001  Whip
      0.001  Female speech, woman speaking
      0.001  Chicken, rooster
      0.001  Oink
      0.001  Sheep
      0.000  Bleat
      0.000  Speech synthesizer
      0.000  Doorbell
      0.000  Wail, moan
      0.000  Laughter
      0.000  Horse
      0.000  Purr
      0.000  Creak
      0.000  Biting
      0.000  Buzz
      0.000  Bird vocalization, bird call, bird song
      0.000  Chink, clink
      0.000  Snort
      0.000  Scratching (performance technique)
      0.000  Squawk
      0.000  Narration, monologue
      0.000  Neigh, whinny
      0.000  Clatter
      0.000  Snicker
      0.000  Turkey
      0.000  Buzzer
      0.000  Wood block
      0.000  Car
      0.000  Whir
      0.000  Beep, bleep
      0.000  Jingle (music)
      0.000  Electronic music
      0.000  Cluck
      0.000  Singing
      0.000  Scrape
      0.000  Burping, eructation
      0.000  White noise
      0.000  Plucked string instrument
      0.000  Caterwaul
      0.000  Microwave oven
      0.000  Tick-tock
      0.000  Clip-clop
      0.000  Owl
      0.000  Chirp, tweet
      0.000  Sonar
      0.000  Flap
      0.000  Motor vehicle (road)
      0.000  Echo
      0.000  Explosion
      0.000  Mechanisms
      0.000  Goose
      0.000  Snake
      0.000  Tick
      0.000  Arrow
      0.000  Drum kit
      0.000  Rumble
      0.000  Spray
      0.000  Static
      0.000  Single-lens reflex camera
      0.000  Mouse
      0.000  Inside, public space
      0.000  Siren
      0.000  Marimba, xylophone
      0.000  Snare drum
      0.000  Conversation
      0.000  Engine
      0.000  Didgeridoo
      0.000  Car passing by
      0.000  Crowing, cock-a-doodle-doo
      0.000  Whistling
      0.000  Guitar
      0.000  Music for children
      0.000  Scratch
      0.000  Music of Africa
      0.000  Goat
      0.000  Giggle
      0.000  Chuckle, chortle
      0.000  Motorcycle
      0.000  Alarm
      0.000  Sliding door
      0.000  Telephone bell ringing
      0.000  Alarm clock
      0.000  Tire squeal
      0.000  Crackle
      0.000  Soundtrack music
      0.000  Thunder
      0.000  Baby cry, infant cry
      0.000  Insect
      0.000  Race car, auto racing
      0.000  Rock and roll
      0.000  Thump, thud
      0.000  Wind instrument, woodwind instrument
      0.000  Shout
      0.000  Roll
      0.000  Jazz
      0.000  Heart murmur
      0.000  Crack
      0.000  Telephone
      0.000  Door
      0.000  Vehicle horn, car horn, honking
      0.000  Rain
      0.000  Rustling leaves
      0.000  Gobble
      0.000  Clang
      0.000  Squeal
      0.000  Walk, footsteps
      0.000  Train
      0.000  Glockenspiel
      0.000  Cacophony
      0.000  Ice cream truck, ice cream van
      0.000  Cough
      0.000  Pop music
      0.000  Cheering
      0.000  Jingle bell
      0.000  Gurgling
      0.000  Electronica
      0.000  Techno
      0.000  Video game music
      0.000  Strum
      0.000  Cymbal
      0.000  Sizzle
      0.000  Pulse
      0.000  Drum and bass
      0.000  Rock music
      0.000  Country
      0.000  Slam
      0.000  Boat, Water vehicle
      0.000  Water
      0.000  Reverberation
      0.000  Male singing
      0.000  Rodents, rats, mice
      0.000  Mantra
      0.000  Music of Asia
      0.000  Rain on surface
      0.000  Run
      0.000  Skateboard
      0.000  Classical music
      0.000  Clock
      0.000  Drip
      0.000  Dubstep
      0.000  Baby laughter
      0.000  Electric guitar
      0.000  Exciting music
      0.000  Rattle
      0.000  Crowd
      0.000  Piano
      0.000  Accelerating, revving, vroom
      0.000  Smoke detector, smoke alarm
      0.000  House music
      0.000  Steam whistle
      0.000  Hip hop music
      0.000  Bowed string instrument
      0.000  Rub
      0.000  Shuffle
      0.000  Police car (siren)
      0.000  Environmental noise
      0.000  Radio
      0.000  Cricket
      0.000  Synthesizer
      0.000  Heavy metal
      0.000  Independent music
      0.000  Child speech, kid speaking
      0.000  Tools
      0.000  Bass drum
      0.000  Dishes, pots, and pans
      0.000  Bass guitar
      0.000  Disco
      0.000  Truck
      0.000  Thunk
      0.000  Fire
      0.000  Reversing beeps
      0.000  Keyboard (musical)
      0.000  Acoustic guitar
      0.000  Blues
      0.000  Sampler
      0.000  Background music
      0.000  Eruption
      0.000  Mosquito
      0.000  Fire alarm
      0.000  Chewing, mastication
      0.000  Raindrop
      0.000  Hair dryer
      0.000  Thunderstorm
      0.000  Rhythm and blues
      0.000  Wind noise (microphone)
      0.000  Gunshot, gunfire
      0.000  Trance music
      0.000  Chop
      0.000  Breaking
      0.000  Gush
      0.000  Bus
      0.000  Jingle, tinkle
      0.000  Cowbell
      0.000  Funny music
      0.000  Dance music
      0.000  Computer keyboard
      0.000  Motorboat, speedboat
      0.000  Rapping
      0.000  Steam
      0.000  Gargling
      0.000  Skidding
      0.000  Pigeon, dove
      0.000  Mallet percussion
      0.000  New-age music
      0.000  Violin, fiddle
      0.000  Salsa music
      0.000  Pizzicato
      0.000  Steel guitar, slide guitar
      0.000  Artillery fire
      0.000  Beatboxing
      0.000  Drum machine
      0.000  Brass instrument
      0.000  Babbling
      0.000  Sad music
      0.000  Ambulance (siren)
      0.000  Wood
      0.000  Crying, sobbing
      0.000  Electronic dance music
      0.000  Air brake
      0.000  Electric piano
      0.000  Tubular bells
      0.000  Drill
      0.000  Dial tone
      0.000  Snoring
      0.000  Funk
      0.000  Bell
      0.000  Bouncing
      0.000  Chant
      0.000  Pink noise
      0.000  Gospel music
      0.000  Rimshot
      0.000  Train whistle
      0.000  Squeak
      0.000  Railroad car, train wagon
      0.000  Medium engine (mid frequency)
      0.000  Throbbing
      0.000  Emergency vehicle
      0.000  Timpani
      0.000  Field recording
      0.000  Crunch
      0.000  Rail transport
      0.000  Hubbub, speech noise, speech babble
      0.000  Reggae
      0.000  Children shouting
      0.000  Effects unit
      0.000  Folk music
      0.000  Car alarm
      0.000  Hoot
      0.000  Tender music
      0.000  Boom
      0.000  Frying (food)
      0.000  Vibration
      0.000  Fusillade
      0.000  Music of Latin America
      0.000  Theme music
      0.000  Female singing
      0.000  Hi-hat
      0.000  Idling
      0.000  Filing (rasp)
      0.000  Wind
      0.000  Coo
      0.000  Clicking
      0.000  Stream
      0.000  Heavy engine (low frequency)
      0.000  Typing
      0.000  Waterfall
      0.000  Bluegrass
      0.000  Angry music
      0.000  Busy signal
      0.000  Chatter
      0.000  Fly, housefly
      0.000  Song
      0.000  Belly laugh
      0.000  Fireworks
      0.000  Firecracker
      0.000  Drum roll
      0.000  Subway, metro, underground
      0.000  Tap
      0.000  Grunge
      0.000  Printer
      0.000  Heart sounds, heartbeat
      0.000  Squish
      0.000  Keys jangling
      0.000  Liquid
      0.000  Blender
      0.000  Cutlery, silverware
      0.000  Cello
      0.000  Progressive rock
      0.000  Machine gun
      0.000  Camera
      0.000  Toot
      0.000  Orchestra
      0.000  Whispering
      0.000  Electronic tuner
      0.000  Lullaby
      0.000  Tambourine
      0.000  Sitar
      0.000  Traffic noise, roadway noise
      0.000  Patter
      0.000  Bicycle
      0.000  Gong
      0.000  Psychedelic rock
      0.000  Tabla
      0.000  Waves, surf
      0.000  Boiling
      0.000  Civil defense siren
      0.000  Harp
      0.000  Train horn
      0.000  Air horn, truck horn
      0.000  Vibraphone
      0.000  Soul music
      0.000  Power tool
      0.000  Bagpipes
      0.000  Punk rock
      0.000  Electric shaver, electric razor
      0.000  Choir
      0.000  Sailboat, sailing ship
      0.000  Afrobeat
      0.000  Writing
      0.000  Cash register
      0.000  Slosh
      0.000  Synthetic singing
      0.000  Water tap, faucet
      0.000  Flute
      0.000  Distortion
      0.000  Fixed-wing aircraft, airplane
      0.000  Ocean
      0.000  Child singing
      0.000  Vocal music
      0.000  Maraca
      0.000  Hammer
      0.000  Banjo
      0.000  Trumpet
      0.000  Christian music
      0.000  Fire engine, fire truck (siren)
      0.000  Coin (dropping)
      0.000  Bathtub (filling or washing)
      0.000  Christmas music
      0.000  Tuning fork
      0.000  Middle Eastern music
      0.000  Toilet flush
      0.000  Sink (filling or washing)
      0.000  Light engine (high frequency)
      0.000  Sniff
      0.000  Hum
      0.000  Double bass
      0.000  Ska
      0.000  Opera
      0.000  Happy music
      0.000  Harmonic
      0.000  Clarinet
      0.000  Ambient music
      0.000  Swing music
      0.000  Children playing
      0.000  Stomach rumble
      0.000  Noise
      0.000  Bee, wasp, etc.
      0.000  Clapping
      0.000  Sine wave
      0.000  Chime
      0.000  Flamenco
      0.000  Telephone dialing, DTMF
      0.000  Scissors
      0.000  Scary music
      0.000  Trickle, dribble
      0.000  Finger snapping
      0.000  Rowboat, canoe, kayak
      0.000  Tapping (guitar technique)
      0.000  Rattle (instrument)
      0.000  Church bell
      0.000  Electronic organ
      0.000  Air conditioning
      0.000  Traditional music
      0.000  Cap gun
      0.000  Bicycle bell
      0.000  Pump (liquid)
      0.000  Ship
      0.000  Aircraft
      0.000  Wheeze
      0.000  Carnatic music
      0.000  Lawn mower
      0.000  Helicopter
      0.000  Singing bowl
      0.000  Music of Bollywood
      0.000  Fill (with liquid)
      0.000  Chorus effect
      0.000  Change ringing (campanology)
      0.000  Harpsichord
      0.000  Pulleys
      0.000  French horn
      0.000  Sewing machine
      0.000  Applause
      0.000  Chainsaw
      0.000  Wedding music
      0.000  Harmonica
      0.000  Ukulele
      0.000  Basketball bounce
      0.000  Zing
      0.000  Battle cry
      0.000  Chirp tone
      0.000  Clickety-clack
      0.000  Glass
      0.000  Mandolin
      0.000  Saxophone
      0.000  Humming
      0.000  String section
      0.000  Organ
      0.000  Jet engine
      0.000  Drawer open or close
      0.000  Ratchet, pawl
      0.000  Stir
      0.000  Aircraft engine
      0.000  Crumpling, crinkling
      0.000  Theremin
      0.000  Shuffling cards
      0.000  Tearing
      0.000  Sawing
      0.000  Sanding
      0.000  Yodeling
      0.000  Chopping (food)
      0.000  Trombone
      0.000  Engine knocking
      0.000  Wind chime
      0.000  Power windows, electric windows
      0.000  Crushing
      0.000  Toothbrush
      0.000  Steelpan
      0.000  Shofar
      0.000  Electric toothbrush
      0.000  Knock
      0.000  Mechanical fan
      0.000  Vacuum cleaner
      0.000  A capella
      0.000  Engine starting
      0.000  Zither
      0.000  Train wheels squealing
      0.000  Propeller, airscrew
      0.000  Whale vocalization
      0.000  Bird flight, flapping wings
      0.000  Mains hum
      0.000  Pour
      0.000  Hammond organ
      0.000  Sidetone
      0.000  Hands
      0.000  Gears
      0.000  Accordion
      0.000  Shatter
      0.000  Jackhammer
      0.000  Splash, splatter
      0.000  Cupboard open or close
      0.000  Dental drill, dentist's drill
      0.000  Typewriter
      0.000  Foghorn
      0.000  Splinter
    0.06s
    VRAM after ast         0.01 GB allocated /  0.02 GB reserved

## 9. CLAP - zero-shot classification

CLAP scores the clip against **arbitrary text prompts**, so it classifies ESC-50 with no fine-tuning - we just pass the 50 category names as candidate labels. This is the flexible modern default.

------------------------------------------------------------------------

``` python
zsc = pipeline("zero-shot-audio-classification", model="laion/clap-htsat-unfused", device=device)
prompts = [f"the sound of {c.replace('_', ' ')}" for c in CATEGORIES]

clip = esc_clip(esc[0])
t0 = time.perf_counter()
preds = zsc(clip["array"], candidate_labels=prompts)
print(f"true: {esc[0]['category']}  ->  top: {preds[0]['label']}  ({preds[0]['score']:.3f})")
print(f"{time.perf_counter() - t0:.2f}s")

del zsc
free_memory()
vram("after clap")
```

    true: dog  ->  top: the sound of dog  (0.735)
    0.34s
    VRAM after clap        0.01 GB allocated /  0.02 GB reserved

## 10. Head-to-head Benchmark

Compare **CLAP zero-shot** against a **fused CLAP** variant on an ESC-50 subset: **accuracy** + **RTF**, plus a confusion-matrix heatmap for the zero-shot model. Same clips, same label set. A 50-clip subset is a smoke test; ESC-50’s official 5-fold protocol is the real evaluation.

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
from sklearn.metrics import accuracy_score, confusion_matrix

N = 50  # clips to evaluate
subset = esc.select(range(N))
refs = [row["category"] for row in subset]
prompts = [f"the sound of {c.replace('_', ' ')}" for c in CATEGORIES]

results, preds_for_cm = {}, None
for model_id in ["laion/clap-htsat-unfused", "laion/clap-htsat-fused"]:
    zsc = pipeline("zero-shot-audio-classification", model=model_id, device=device)
    t0 = time.perf_counter()
    hyps = []
    for row in subset:
        p = zsc(esc_clip(row)["array"], candidate_labels=prompts)
        top = p[0]["label"]
        hyps.append(CATEGORIES[prompts.index(top)])
    elapsed = time.perf_counter() - t0
    acc = accuracy_score(refs, hyps)
    results[model_id.split("/")[-1]] = {"acc": acc, "rtf": elapsed / (N * 5.0)}
    print(f"{model_id:30s} acc {acc:6.2%}  {elapsed:5.1f}s")
    if preds_for_cm is None:
        preds_for_cm = hyps
    del zsc
    free_memory()
vram("after benchmark")
```

    [transformers] You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset

    laion/clap-htsat-unfused       acc 100.00%    3.2s

    laion/clap-htsat-fused         acc 96.00%    3.4s
    VRAM after benchmark   0.01 GB allocated /  0.02 GB reserved

``` python
names = list(results)
bar_chart("Audio classification: zero-shot accuracy on ESC-50 subset",
          names, {"accuracy": [results[n]["acc"] for n in names]}, y_name="accuracy")
```

``` python
# Confusion matrix as an ECharts heatmap (categories that actually appear in the subset)
from pyecharts import options as opts
from pyecharts.charts import HeatMap

present = sorted(set(refs))
cm = confusion_matrix(refs, preds_for_cm, labels=present)
data = [[j, i, int(cm[i][j])] for i in range(len(present)) for j in range(len(present))]

hm = HeatMap(init_opts=opts.InitOpts(width="760px", height="620px"))
hm.add_xaxis(present)
hm.add_yaxis("true vs predicted", present, data,
             label_opts=opts.LabelOpts(is_show=True, position="inside"))
hm.set_global_opts(
    title_opts=opts.TitleOpts(title="ESC-50 confusion (CLAP unfused)"),
    xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),
    visualmap_opts=opts.VisualMapOpts(max_=int(cm.max()), orient="vertical", pos_left="right"),
)
hm.render_notebook()
```

## 11. Live Microphone Tagging

The same two models from sections 8 and 9, pointed at the machine’s actual microphone instead of an ESC-50 clip. Recording a few seconds of whatever is around you is the fastest way to feel the difference between the two designs: **AST** can only answer in AudioSet’s fixed 527 classes, while **CLAP** scores the same audio against a label set you type in the cell and can change between two consecutive calls.

Mind the sample rates - the two pipelines do not behave the same way:

- `audio-classification` (AST) accepts a `{"array", "sampling_rate"}` dict and **resamples for you**, so a dict is always safe.
- `zero-shot-audio-classification` (CLAP) rejects dicts and takes a bare array that it assumes is **already at the model’s 48 kHz**. Hand it 16 kHz audio and it will not complain, it will just score a signal playing at a third of its true speed.

So this cell records once, keeps a 48 kHz copy for CLAP, and derives the 16 kHz copy for AST.

Needs a working capture device at `/dev/snd`; the cell raises rather than skipping, so a muted or missing mic is visible instead of looking like a clean run. The docs builder never executes it (`skip_exec: true`).

------------------------------------------------------------------------

``` python
# --- standalone setup ----------------------------------------------------------
# Lifted from the Setup section and the helper cells above so this demo runs on its
# own in a fresh kernel - no earlier cell has to have been executed first.

import ctypes
import ctypes.util
import gc
import numpy as np
import torch

from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from transformers import pipeline

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

# --- the demo ------------------------------------------------------------------
# sounddevice needs the PortAudio runtime (libportaudio2) and ALSA nodes under
# /dev/snd. On the knowledge-lab LXC those are passed in from the Proxmox host by
# `av_devices` in infra/proxmox/variables.tf - the camera's video node and its
# USB-Audio card are separate passthroughs.
import io
import time

import librosa
import sounddevice as sd
from IPython.display import HTML, display
from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# The pipelines emit config warnings on every call; at one window a second that
# buries the output. Errors still get through.
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()

CHUNK_SECONDS = 3.0   # how much trailing audio each classification sees
HOP_SECONDS = 1.0     # how often that window is re-classified
SILENCE_DBFS = -45.0  # windows quieter than this are not sent to the models
SHOW_ROWS = 12        # most recent rows kept on screen; the full log feeds the charts
CLAP_SR, AST_SR = 48_000, 16_000  # each model's native feature-extractor rate

MY_SOUNDS = ["speech", "typing on a keyboard", "a fan or air conditioner",
             "music", "silence", "a door closing"]

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


# rich draws the panel; the in-place update is an IPython display handle. rich's own
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
    r = float(np.sqrt((x ** 2).mean())) if x.size else 0.0
    return 20 * np.log10(r) if r > 0 else float("-inf")


def _view(elapsed, seconds, dbfs, rows, note="", fill=None):
    "One frame of the live view: header stats over one table row per classified window."
    head = Table.grid(padding=(0, 2))
    head.add_column(style="dim", justify="right")
    head.add_column()
    meter = "#" * int(max(0.0, min(1.0, (dbfs + 60) / 60)) * 24)
    head.add_row("elapsed", f"{elapsed:5.1f}s / {seconds:.0f}s")
    head.add_row("level", f"{dbfs:6.1f} dBFS [dim]{meter}[/]")
    if fill is not None:
        done = int(max(0.0, min(1.0, fill)) * 24)
        head.add_row("next hop", f"[dim]{'=' * done}{'.' * (24 - done)}[/] {fill:4.0%}")
    if note:
        head.add_row("status", f"[dim]{note}[/]")
    head.add_row("totals", f"{len(rows)} window(s), "
                           f"{sum(r['ast_ms'] + r['clap_ms'] for r in rows) / 1000:.1f}s classifying")

    table = Table(box=box.SIMPLE_HEAD, pad_edge=False, expand=True,
                  header_style="dim", border_style="dim")
    table.add_column("#", justify="right", style="dim", width=4, no_wrap=True)
    table.add_column("window", justify="right", width=13, no_wrap=True)
    table.add_column("dBFS", justify="right", width=6, no_wrap=True)
    table.add_column("ms", justify="right", width=9, no_wrap=True)
    table.add_column("AST (527 fixed classes)", overflow="ellipsis", ratio=1)
    table.add_column("CLAP (your labels)", overflow="ellipsis", ratio=1)
    shown = rows[-SHOW_ROWS:]
    if len(rows) > len(shown):
        table.add_row("...", "", "", "", f"[dim]{len(rows) - len(shown)} earlier[/]",
                      "[dim]all of them are still in the charts[/]")
    for r in shown:
        table.add_row(
            str(r["window"]), f"{r['from']:.1f}-{r['at']:.1f}s", f"{r['dbfs']:.0f}",
            f"{r['ast_ms'] + r['clap_ms']:.0f}",
            f"{r['ast_label']} [dim]{r['ast_score']:.2f}[/]",
            f"[green]{r['clap_label']}[/] [dim]{r['clap_score']:.2f}[/]")
    if not rows:
        table.add_row("-", "", "", "", "[dim italic](nothing classified yet)[/]", "")
    return Panel(Group(head, Text(""), table),
                 title="[cyan]live audio classification[/]", border_style="cyan",
                 padding=(0, 1))


def live_classify(seconds=30, chunk_seconds=CHUNK_SECONDS, hop_seconds=HOP_SECONDS,
                  labels=MY_SOUNDS):
    """Classify a sliding window of live microphone audio with AST and CLAP at once.

    A `chunk_seconds` window slides over the stream and is re-classified every
    `hop_seconds`, so each moment of audio is scored about three times with
    different context. Unlike streaming ASR there is nothing to de-duplicate: the
    overlap simply smooths the score trajectory, which is what the chart plots.

    Both models stay resident so they see the SAME window - that comparison is the
    whole point of the section, and it is what a sequential load/free would destroy.
    In fp16 the pair costs 0.51 GB (measured) and about 104 ms per window, so a 1 s
    hop runs at a real-time factor near 0.10.

    Returns (log, labels); log feeds the charts.
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

    prompts = [f"the sound of {s}" for s in labels]
    view = display(_html(_view(0.0, seconds, float("-inf"), [], note="starting")),
                   display_id=True)
    buf = np.zeros(0, dtype="float32")
    window_n, hop_n, fresh = int(chunk_seconds * native_sr), int(hop_seconds * native_sr), 0
    log = []

    with sd.InputStream(device=mic, samplerate=native_sr, channels=1, dtype="float32",
                        blocksize=int(native_sr * 0.1), callback=on_audio):
        _note(f"[bold]mic [{mic}][/] @ {native_sr} Hz  ->  "
                  f"{chunk_seconds:.0f}s window every {hop_seconds:.1f}s, "
                  f"listening {seconds:.0f}s - make some noise",
                  title="[cyan]capture[/]", style="cyan")
        t0 = time.perf_counter()
        try:
            while time.perf_counter() - t0 < seconds:
                chunks = []
                try:
                    chunks.append(q.get(timeout=0.2))
                except _queue.Empty:
                    pass
                while True:  # drain whatever piled up while the last window ran
                    try:
                        chunks.append(q.get_nowait())
                    except _queue.Empty:
                        break
                if chunks:
                    fresh += sum(c.size for c in chunks)
                    buf = np.concatenate([buf, *chunks])[-window_n:]  # slide, do not grow

                elapsed = time.perf_counter() - t0
                if fresh < hop_n or buf.size < int(0.8 * native_sr):
                    view.update(_html(_view(elapsed, seconds, _dbfs(buf), log,
                                            fill=fresh / hop_n, note="collecting")))
                    continue
                fresh = 0

                # Resample the window, not the live stream. CLAP's pipeline rejects a
                # dict and assumes the array is ALREADY at 48 kHz, so the 48 kHz copy
                # is the source of truth and AST's 16 kHz copy is derived from it.
                a48 = (librosa.resample(buf, orig_sr=native_sr, target_sr=CLAP_SR)
                       if native_sr != CLAP_SR else buf)
                dbfs = _dbfs(a48)
                if dbfs < SILENCE_DBFS:
                    view.update(_html(_view(
                        elapsed, seconds, dbfs, log,
                        note=f"below {SILENCE_DBFS:.0f} dBFS - window skipped")))
                    continue
                a16 = librosa.resample(a48, orig_sr=CLAP_SR, target_sr=AST_SR)

                t1 = time.perf_counter()
                ast_out = tagger({"array": a16, "sampling_rate": AST_SR})
                ast_ms = 1000 * (time.perf_counter() - t1)

                t1 = time.perf_counter()
                clap_out = zsc(a48, candidate_labels=prompts)
                clap_ms = 1000 * (time.perf_counter() - t1)

                scores = {r["label"]: float(r["score"]) for r in clap_out}
                top = max(scores, key=scores.get)
                log.append({
                    "window": len(log) + 1, "at": elapsed,
                    "from": max(0.0, elapsed - buf.size / native_sr), "dbfs": dbfs,
                    "ast_ms": ast_ms, "clap_ms": clap_ms,
                    "ast_label": ast_out[0]["label"], "ast_score": float(ast_out[0]["score"]),
                    "clap_label": top.replace("the sound of ", ""),
                    "clap_score": scores[top], "clap_scores": scores,
                })
                view.update(_html(_view(elapsed, seconds, dbfs, log)))
        except KeyboardInterrupt:
            view.update(_html(_view(time.perf_counter() - t0, seconds, float("-inf"),
                                    log, note="stopped")))

    n = len(log)
    total_ms = sum(r["ast_ms"] + r["clap_ms"] for r in log)
    rtf = (total_ms / 1000) / (n * hop_seconds) if n else float("nan")
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="dim", justify="right")
    summary.add_column()
    summary.add_row("windows", f"{n} [dim]({chunk_seconds:.0f}s window, "
                               f"{hop_seconds:.1f}s hop)[/]")
    summary.add_row("mean AST", f"{sum(r['ast_ms'] for r in log) / max(n, 1):.0f} ms")
    summary.add_row("mean CLAP", f"{sum(r['clap_ms'] for r in log) / max(n, 1):.0f} ms")
    summary.add_row("real-time factor", f"{rtf:.3f}"
                    + ("" if rtf < 1 else "  [red](slower than audio)[/]"))
    if overflows:
        summary.add_row("overflows", f"[red]{overflows}[/] (not keeping up)")
    display(_html(Panel(summary, title="[dim]done[/]", border_style="dim", padding=(0, 1))))
    return log, labels


def classification_charts(log, labels, hop_seconds=HOP_SECONDS):
    """Two ECharts views of the run (pyecharts is the repo standard for all charts).

    The score trajectory is the interesting one: CLAP re-scores every label on every
    window, so you watch the model change its mind as the room changes. The latency
    chart underneath is the cost of having done so.
    """
    from pyecharts import options as opts
    from pyecharts.charts import Bar, Line, Page

    if not log:
        _note("no windows classified - nothing rose above the silence gate",
              title="[yellow]empty[/]", style="yellow")
        return None

    x = [f"{r['at']:.1f}" for r in log]
    trend = Line(init_opts=opts.InitOpts(width="760px", height="400px")).add_xaxis(x)
    for s in labels:
        trend.add_yaxis(s, [round(r["clap_scores"].get(f"the sound of {s}", 0.0), 4)
                            for r in log],
                        is_smooth=True, label_opts=opts.LabelOpts(is_show=False))
    trend.set_global_opts(
        title_opts=opts.TitleOpts(title="CLAP score per label over time",
                                  subtitle="zero-shot: these labels are yours, not the model's"),
        xaxis_opts=opts.AxisOpts(name="seconds"),
        yaxis_opts=opts.AxisOpts(name="score", max_=1),
        tooltip_opts=opts.TooltipOpts(trigger="axis"),
        legend_opts=opts.LegendOpts(pos_top="10%", type_="scroll"),
    )

    budget = 1000 * hop_seconds
    lat = (
        Bar(init_opts=opts.InitOpts(width="760px", height="360px"))
        .add_xaxis(x)
        .add_yaxis("AST ms", [round(r["ast_ms"], 1) for r in log], stack="t")
        .add_yaxis("CLAP ms", [round(r["clap_ms"], 1) for r in log], stack="t")
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Classification latency per window",
                subtitle=f"stacked; under {budget:.0f} ms keeps up with a {hop_seconds:.1f}s hop"),
            xaxis_opts=opts.AxisOpts(name="seconds"),
            yaxis_opts=opts.AxisOpts(name="milliseconds"),
            tooltip_opts=opts.TooltipOpts(trigger="axis"),
            legend_opts=opts.LegendOpts(pos_top="10%"),
        )
        .set_series_opts(
            label_opts=opts.LabelOpts(is_show=False),
            markline_opts=opts.MarkLineOpts(
                data=[opts.MarkLineItem(y=budget, name="real-time budget")]),
        )
    )
    return Page().add(trend, lat).render_notebook()


# fp16 halves the pair to 0.51 GB with identical predictions (measured: AST "Static"
# 0.681 either way), so both stay resident and score the same window.
tagger = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593",
                  device=device, top_k=5,
                  dtype=torch.float16 if device != "cpu" else torch.float32)
zsc = pipeline("zero-shot-audio-classification", model="laion/clap-htsat-unfused",
               device=device, dtype=torch.float16 if device != "cpu" else torch.float32)
rvram("AST + CLAP live")

LOG, LABELS = live_classify(seconds=30)

del tagger, zsc
free_memory()
rvram("after live mic")

classification_charts(LOG, LABELS)
```

## 12. Common Frameworks

Audio classification is the cheapest audio task to run and the one most likely to end up on a device rather than a server: a sound monitor, a wake word, a “was that a gunshot” sensor. That shapes the ecosystem. `transformers` covers the modelling, but the interesting frameworks here are the ones that get an 86M model onto something with no GPU and a power budget.

| Framework | Layer | What it gives you | License | Reach for it when |
|----|----|----|----|----|
| [transformers](https://github.com/huggingface/transformers) | modelling | AST, wav2vec2, HuBERT and CLAP behind the `audio-classification` and `zero-shot-audio-classification` pipelines | Apache 2.0 | Default, and unusually complete here - the fine-tuning path for a custom head is a `Trainer` loop |
| [SpeechBrain](https://github.com/speechbrain/speechbrain) | modelling | ECAPA-TDNN speaker embeddings, language ID, and emotion recipes with training loops attached | Apache 2.0 | The label is a speaker or a language rather than a sound event |
| [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) | modelling | MatchboxNet and TitaNet - small command-spotting and speaker models designed for edge deployment | Apache 2.0 (weights often CC-BY-4.0) | You need a tiny keyword or speaker model and a training recipe that assumes streaming audio |
| [openWakeWord](https://github.com/dscripka/openWakeWord) / [Porcupine](https://github.com/Picovoice/porcupine) | modelling | Purpose-built always-on wake-word detection, trained on synthetic data, running in a few MB | Apache 2.0 / proprietary (Porcupine) | You want one specific phrase detected continuously. A 527-class tagger is the wrong tool and the wrong power draw |
| [torchaudio](https://github.com/pytorch/audio) + [audiomentations](https://github.com/iver56/audiomentations) | data | Mel spectrograms, and the augmentation set - time shift, noise, gain, SpecAugment - that this task lives or dies on | BSD-2 / MIT | Always when training. Augmentation moves accuracy more than architecture on small audio datasets |
| [openSMILE](https://github.com/audeering/opensmile) | data | The classical acoustic feature sets (eGeMAPS, ComParE) that paralinguistics research still reports on | audEERING research license (commercial license required) | Emotion, health or paralinguistic work where a feature set plus a small classifier is the published baseline |
| [ONNX Runtime](https://onnxruntime.ai/) / [TFLite](https://ai.google.dev/edge/litert) / [MediaPipe](https://github.com/google-ai-edge/mediapipe) | inference runtime | The same classifier at a few MB and a few mW, on a phone, a Pi or a microcontroller | MIT / Apache 2.0 | The model runs on the device that hears the sound - which for this task is most of the time |
| [Triton Inference Server](https://github.com/triton-inference-server/server) / [BentoML](https://github.com/bentoml/BentoML) | serving | Batched server-side tagging of an archive, with dynamic batching over short clips | BSD-3 / Apache 2.0 | You are tagging a back catalogue rather than a live stream |
| [torchmetrics](https://github.com/Lightning-AI/torchmetrics) + [scikit-learn](https://github.com/scikit-learn/scikit-learn) | evaluation | Multi-label mAP, per-class AP, confusion matrices, and threshold sweeps | Apache 2.0 / BSD-3 | Always. AudioSet is multi-label and heavily imbalanced, so accuracy is meaningless and mAP is the number that counts |

**The 2026 default stack** is `transformers` + AST for tagging, CLAP when the taxonomy will keep changing, openWakeWord when it is one fixed phrase, and ONNX Runtime or TFLite for anything that leaves the server. `audiomentations` belongs in every training run.

The common wrong turn is deploying a general AudioSet tagger where a purpose-built detector belongs: 527 classes running continuously costs orders of magnitude more power than a wake-word model, and its per-class precision on your one class of interest is usually worse. The second is picking a decision threshold on the same clips you measured mAP on - sweep it on a held-out split that matches your real base rate.

------------------------------------------------------------------------

## 13. Going Further

- **Multi-label tagging at scale.** Evaluate AST with **mAP** on the full AudioSet eval set; **BEATs** and **CED** push the state of the art past it, with the same 527-class output.
- **Fine-tuning.** Fine-tune wav2vec2 / HuBERT for your own KWS, emotion, or speaker-ID head: [HF audio classification guide](https://huggingface.co/docs/transformers/tasks/audio_classification). Freezing the encoder and training only the head is usually enough on a few hours of data.
- **Zero-shot everywhere.** Swap the CLAP candidate prompts to classify any new taxonomy with no training data. The cost is calibration: CLAP’s scores are relative to the prompt set, so adding one label changes every other score.
- **Always-on tagging.** Loop section 11 over a rolling buffer to build a continuous sound monitor - the shape of every “what is that noise” appliance. Add hysteresis so a single noisy frame does not fire an event.
- **Related notebooks.** `05_Voice_Activity_Detection` (the same shape, one binary label, running continuously), `02_Automatic_Speech_Recognition` (when the words matter and the sound does not), `01_Text_to_Audio` (CLAP again, scoring generated audio instead of recorded).

------------------------------------------------------------------------

Back to top
