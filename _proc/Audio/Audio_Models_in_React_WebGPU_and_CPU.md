# Running the Audio Models in a React Frontend (WebGPU or CPU)

> How each audio task from the `DL_tasks/nbs/Audio/` notebooks maps to an in-browser
> React + TypeScript app, running the model **client-side** on the user's GPU
> (WebGPU) or CPU (WebAssembly). No Python server in the inference path.

The Python notebooks load every model through Hugging Face `transformers`. The
browser equivalent is **Transformers.js** (`@huggingface/transformers`), which runs
the *same* Hugging Face checkpoints (exported to ONNX) on
**ONNX Runtime Web**, with two execution backends:

- **WebGPU** - runs the model on the GPU. 5-30x faster than CPU; needs a
  Chromium-based browser (or Safari 18+ / Firefox with the flag) and a machine
  with WebGPU enabled.
- **WASM (CPU)** - WebAssembly + SIMD + multithreading. Works everywhere, slower,
  but the universal fallback.

Tasks with no ONNX / Transformers.js path (source separation, diffusion audio)
drop down to **ONNX Runtime Web directly** with a hand-exported model, or stay
server-side. The feasibility table at the end says which is which.

---

## 1. The core stack

```bash
npm i @huggingface/transformers          # pipelines, WebGPU + WASM backends
# onnxruntime-web is pulled in transitively; install directly only for
# custom (non-pipeline) models such as source separation:
npm i onnxruntime-web
```

Key idea from the notebooks that carries straight over: **hold one large model live
at a time and free it before loading the next.** The browser tab has a memory
budget every bit as tight as the 12 GB container the notebooks target - a WebGPU
context can OOM the tab. Same discipline: dispose (`await model.dispose()`), null
the reference, let GC run.

### Device selection + feature detection

```ts
// src/lib/device.ts
export type Backend = "webgpu" | "wasm";

export async function pickBackend(): Promise<Backend> {
  // WebGPU is exposed as navigator.gpu; requesting an adapter confirms a
  // usable GPU (some browsers expose the API but have no adapter).
  if ("gpu" in navigator) {
    try {
      const adapter = await (navigator as any).gpu.requestAdapter();
      if (adapter) return "webgpu";
    } catch {
      /* fall through to wasm */
    }
  }
  return "wasm";
}
```

Transformers.js takes the backend as `device` and the weight precision as `dtype`.
On WebGPU prefer `fp16` (half the memory, matches the notebooks' `torch.float16`);
on WASM use a quantized `q8`/`q4` to keep the download and RAM small:

```ts
// src/lib/loadOpts.ts
import type { Backend } from "./device";

export function loadOpts(backend: Backend) {
  return backend === "webgpu"
    ? { device: "webgpu" as const, dtype: "fp16" as const }
    : { device: "wasm" as const,   dtype: "q8"   as const };
}
```

---

## 2. Audio I/O helpers (the browser's `soundfile` + `librosa`)

Every ASR/classification model in the notebooks wants **16 kHz mono `Float32`**.
The notebooks do this with `librosa.resample` / `soundfile`; in the browser the
Web Audio API does the decode and resample.

```ts
// src/lib/audio.ts

/** Decode any audio File/Blob/ArrayBuffer to mono Float32 at a target rate. */
export async function decodeToMono(
  data: ArrayBuffer,
  targetRate = 16000,
): Promise<Float32Array> {
  // OfflineAudioContext resamples for free: create it at the target rate and
  // render the decoded buffer through it.
  const tmp = new AudioContext();
  const decoded = await tmp.decodeAudioData(data.slice(0));
  await tmp.close();

  const off = new OfflineAudioContext(1, Math.ceil(decoded.duration * targetRate), targetRate);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return rendered.getChannelData(0); // mono Float32 @ targetRate
}

/** Capture N seconds from the mic as mono Float32 at targetRate. */
export async function recordMic(seconds: number, targetRate = 16000): Promise<Float32Array> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const rec = new MediaRecorder(stream);
  const chunks: Blob[] = [];
  rec.ondataavailable = (e) => chunks.push(e.data);
  rec.start();
  await new Promise((r) => setTimeout(r, seconds * 1000));
  rec.stop();
  await new Promise<void>((r) => (rec.onstop = () => r()));
  stream.getTracks().forEach((t) => t.stop());
  const buf = await new Blob(chunks).arrayBuffer();
  return decodeToMono(buf, targetRate);
}
```

For **playback of generated audio** (TTS, music) the models return a `Float32Array`
plus a sampling rate. Turn it into an `AudioBuffer` and play, or encode a WAV blob
to download - the browser analog of the notebooks' `save_wav`:

```ts
// src/lib/audio.ts (cont.)

export function play(samples: Float32Array, sampleRate: number) {
  const ctx = new AudioContext();
  const buf = ctx.createBuffer(1, samples.length, sampleRate);
  buf.getChannelData(0).set(samples);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.start();
}

/** Minimal 16-bit PCM WAV encoder -> a downloadable Blob. */
export function toWavBlob(samples: Float32Array, sampleRate: number): Blob {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buf);
  const wr = (o: number, s: string) => [...s].forEach((c, i) => view.setUint8(o + i, c.charCodeAt(0)));
  wr(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); wr(8, "WAVE");
  wr(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true);
  view.setUint16(34, 16, true); wr(36, "data"); view.setUint32(40, samples.length * 2, true);
  let o = 44;
  for (const s of samples) {
    const c = Math.max(-1, Math.min(1, s));
    view.setInt16(o, c < 0 ? c * 0x8000 : c * 0x7fff, true);
    o += 2;
  }
  return new Blob([view], { type: "audio/wav" });
}
```

### Always run models in a Web Worker

Inference blocks the thread it runs on. To keep the React UI responsive, load and
run the model in a **Web Worker** and post messages back. Every example below
assumes this pattern; the worker skeleton:

```ts
// src/workers/asr.worker.ts
import { pipeline, type PipelineType } from "@huggingface/transformers";

let task: any = null;
self.onmessage = async (e: MessageEvent) => {
  const { type, payload } = e.data;
  if (type === "load") {
    task = await pipeline(
      "automatic-speech-recognition",
      payload.model,
      { ...payload.opts, progress_callback: (p: any) => self.postMessage({ type: "progress", p }) },
    );
    self.postMessage({ type: "ready" });
  } else if (type === "run") {
    const out = await task(payload.audio, payload.args);
    self.postMessage({ type: "result", out });
  }
};
```

```ts
// spawn from React:
const worker = new Worker(new URL("../workers/asr.worker.ts", import.meta.url), { type: "module" });
```

Model weights are downloaded once and cached by the browser (Cache Storage /
IndexedDB), so the second load is instant and works offline - the browser analog
of the notebooks' `DATA_DIR / "hf_cache"`.

---

## 3. Task-by-task

Each subsection maps a notebook to its browser implementation: the recommended
in-browser model, the backend, and a React/TypeScript snippet. Snippets show the
`pipeline` call (main thread for brevity); wrap them in the worker above for
production.

### 3.1 Automatic Speech Recognition -> **best in-browser task**

`02_Automatic_Speech_Recognition.ipynb` (Whisper, Qwen3-ASR, Parakeet, Moonshine).

ASR is the flagship Transformers.js use case: Whisper and Moonshine have
first-class ONNX + WebGPU support, run at faster-than-real-time on a laptop GPU,
and stream. Qwen3-ASR / Parakeet from the notebook are **not** ONNX-exported yet -
use Whisper or Moonshine in the browser.

| Notebook model | Browser model | Backend | Notes |
|----------------|---------------|---------|-------|
| whisper-small/large | `onnx-community/whisper-base` (or `-small`) | WebGPU / WASM | timestamps, 99 langs, translate |
| Moonshine (edge) | `onnx-community/moonshine-tiny-ONNX` | WebGPU / WASM | tiny, fast, English, low latency |

```ts
// src/lib/asr.ts
import { pipeline } from "@huggingface/transformers";
import { pickBackend } from "./device";
import { loadOpts } from "./loadOpts";
import { decodeToMono } from "./audio";

export async function makeTranscriber() {
  const backend = await pickBackend();
  const asr = await pipeline(
    "automatic-speech-recognition",
    "onnx-community/whisper-base",
    loadOpts(backend),
  );
  return async (file: ArrayBuffer) => {
    const audio = await decodeToMono(file, 16000); // Whisper wants 16 kHz mono
    const out = await asr(audio, {
      return_timestamps: true,          // same flag as the notebook
      chunk_length_s: 30,               // long-form chunking, as in section 7
      language: "en",
      task: "transcribe",               // "translate" -> English out of any language
    });
    return out; // { text, chunks: [{ timestamp, text }] }
  };
}
```

**Streaming / live mic** (the notebook's section 12): Transformers.js Whisper
accepts a `streamer` / partial-decode loop, or use the community
`whisper-web` worker pattern - feed rolling 5-30 s windows from `recordMic` and
show incremental text. Moonshine is the better choice for low-latency live
captioning (the notebook's "edge / Raspberry Pi" row).

### 3.2 Audio Classification -> **works fully in-browser**

`04_Audio_Classification.ipynb` (AST tagging, HuBERT emotion, wav2vec2 KWS, CLAP
zero-shot).

All four families export to ONNX and run through the `audio-classification` and
`zero-shot-audio-classification` pipelines - a direct port of the notebook.

| Notebook model | Browser model | Pipeline |
|----------------|---------------|----------|
| AST AudioSet tagging | `onnx-community/ast-finetuned-audioset-10-10-0.4593` | `audio-classification` |
| wav2vec2 keyword spotting | `onnx-community/wav2vec2-base-superb-ks` | `audio-classification` |
| CLAP zero-shot | `Xenova/clap-htsat-unfused` | `zero-shot-audio-classification` |

```ts
// Fixed-label tagging (AST) - mirrors notebook section on AudioSet tagging
import { pipeline } from "@huggingface/transformers";

const tagger = await pipeline(
  "audio-classification",
  "onnx-community/ast-finetuned-audioset-10-10-0.4593",
  { ...loadOpts(await pickBackend()), topk: 5 },
);
const audio = await decodeToMono(fileBuf, 16000);
const preds = await tagger(audio); // [{ label, score }, ...]

// Zero-shot with CLAP - label from free-text prompts (notebook's CLAP section)
const zsc = await pipeline("zero-shot-audio-classification", "Xenova/clap-htsat-unfused", loadOpts(backend));
const out = await zsc(audio, ["the sound of a dog barking", "the sound of rain", "a car engine"]);
```

This is a great always-on browser task (wake words, sound events) - small models,
WebGPU optional, and the zero-shot CLAP path handles the open-set problem the
notebook calls out.

### 3.3 Text-to-Speech -> **works in-browser** (pick the right model)

`00_Text_to_Speech.ipynb` (SpeechT5, MMS-VITS, Bark, Kokoro).

| Notebook model | Browser path | Backend | Notes |
|----------------|--------------|---------|-------|
| SpeechT5 | `Xenova/speecht5_tts` via `text-to-speech` pipeline | WASM (WebGPU partial) | needs the x-vector speaker embedding, same as the notebook |
| MMS-VITS | `Xenova/mms-tts-eng` | WASM | tiny, end-to-end, multilingual by swapping the model id |
| Kokoro-82M | **`kokoro-js`** package | WebGPU / WASM | best small-model quality; purpose-built for the browser |
| Bark (1B codec LM) | not recommended in-browser | - | too heavy; keep server-side |

Kokoro is the standout - the notebook's "best small-model quality" recommendation
ships as a dedicated WebGPU JS package:

```ts
// npm i kokoro-js
import { KokoroTTS } from "kokoro-js";

const tts = await KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-v1.0-ONNX", {
  dtype: "fp16",     // q8 on CPU-only machines
  device: (await pickBackend()) === "webgpu" ? "webgpu" : "wasm",
});
const audio = await tts.generate("Transformers dot js makes speech synthesis a one liner.", {
  voice: "af_heart",
});
play(audio.audio, audio.sampling_rate); // or audio.toBlob() for a WAV download
```

SpeechT5 via the standard pipeline, matching the notebook's speaker-embedding
approach:

```ts
const synth = await pipeline("text-to-speech", "Xenova/speecht5_tts", loadOpts(backend));
const speaker_embeddings =
  "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/speaker_embeddings.bin";
const out = await synth("Hello from the browser.", { speaker_embeddings });
play(out.audio, out.sampling_rate);
```

### 3.4 Text-to-Audio (music / SFX) -> **partial; usually server-side**

`01_Text_to_Audio.ipynb` (MusicGen codec LM; AudioLDM / Stable Audio diffusion).

- **MusicGen** (`Xenova/musicgen-small`) *can* run via the `text-to-audio`
  pipeline in Transformers.js, but it is autoregressive at ~50 tokens/sec of audio
  and slow even on WebGPU - fine for a short 5-10 s demo, painful for anything
  longer. The notebook already flags TTA as "slow and offline"; that is doubly
  true in a tab.
- **AudioLDM / Stable Audio** are `diffusers` latent-diffusion models with **no
  Transformers.js path**. Do not attempt in-browser; keep them on a GPU server and
  call an API.

```ts
// MusicGen demo only - expect multi-second generation for a few seconds of audio
const musicgen = await pipeline("text-to-audio", "Xenova/musicgen-small", loadOpts(backend));
const out = await musicgen("a warm lo-fi hip hop beat with mellow piano", {
  max_new_tokens: 256, // ~5 s; keep small in the browser
});
play(out.audio, out.sampling_rate); // 32 kHz
```

Recommendation: for a production music/SFX feature, run generation server-side
(the notebook's GPU code) and stream the resulting WAV to the React client. Use
the in-browser MusicGen only for an offline/no-backend toy.

### 3.5 Audio-to-Audio (separation / enhancement) -> **custom ONNX, not a pipeline**

`03_Audio_to_Audio.ipynb` (Demucs stem separation, DeepFilterNet enhancement, VC).

The notebook itself notes `transformers` has **no** `audio-to-audio` pipeline -
and neither does Transformers.js. Two browser options:

1. **Real-time speech enhancement** - **DeepFilterNet** has an ONNX export and
   established WebAssembly/WebGPU demos. This is the realistic in-browser
   audio-to-audio task (live call denoise, the notebook's "causal, ~10-20 ms"
   row). Run it with `onnxruntime-web` on framed audio.
2. **Music stem separation (Demucs)** - large and non-causal; the notebook already
   says it "wants the whole track." Export to ONNX and run with `onnxruntime-web`
   if you must, but it is heavy for a tab - prefer server-side.

Custom ONNX model skeleton (no pipeline wrapper):

```ts
// src/lib/enhance.ts
import * as ort from "onnxruntime-web/webgpu";

export async function makeEnhancer(modelUrl: string) {
  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ["webgpu", "wasm"], // tries WebGPU, falls back to CPU
  });
  return async (frame: Float32Array) => {
    const input = new ort.Tensor("float32", frame, [1, frame.length]);
    const { output } = await session.run({ input }); // names come from the exported graph
    return output.data as Float32Array;
  };
}
```

You own the pre/post-processing (STFT framing, overlap-add) that the Python
toolkits (`torchaudio`, `speechbrain`) do for you - this is the most involved
task to bring to the browser.

### 3.6 Audio-Text-to-Text (multimodal) -> **server-side**

`Multimodal/00_Audio_Text_to_Text.ipynb` is a placeholder, but for completeness:
audio LLMs (Qwen2-Audio, etc.) are multi-billion-parameter and have no browser
runtime. Keep them behind an API.

---

## 4. Feasibility summary

| Task (notebook) | In-browser? | Recommended model | Best backend | If not |
|-----------------|-------------|-------------------|--------------|--------|
| **ASR** (`02`) | Yes - excellent | Whisper-base / Moonshine-tiny (ONNX) | WebGPU (WASM ok) | - |
| **Audio classification** (`04`) | Yes - full | AST / wav2vec2 / CLAP (ONNX) | WASM or WebGPU | - |
| **TTS** (`00`) | Yes | Kokoro-82M (`kokoro-js`), MMS-VITS, SpeechT5 | WebGPU (Kokoro) / WASM | Bark -> server |
| **Text-to-Audio** (`01`) | Partial | MusicGen-small (short demos only) | WebGPU | AudioLDM / Stable Audio -> server |
| **Audio-to-Audio** (`03`) | Partial | DeepFilterNet (custom ONNX) | WebGPU / WASM | Demucs / VC -> server |
| **Audio-Text-to-Text** (multimodal) | No | - | - | server API |

Rule of thumb: **discriminative + small** (ASR, classification, small TTS) runs
great client-side; **generative + large / diffusion** (music, separation, audio
LLMs) belongs on a server, with the browser as the UI.

---

## 5. Memory & performance notes (the browser's version of the 12 GB budget)

The notebooks are written for a 12 GB-VRAM box and free memory aggressively. A
browser tab is tighter and less forgiving - apply the same rules:

- **One model live at a time.** `await pipeline_or_session.dispose()` and null the
  reference before loading the next model - the browser analog of `del model;
  free_memory()`. There is no `torch.cuda.empty_cache()`; you rely on GC + WebGPU
  context teardown, so disposing is essential.
- **Quantize for CPU.** `dtype: "q8"` (or `q4`) cuts the download and RAM 2-4x on
  the WASM backend; use `fp16` on WebGPU to match the notebooks' half precision.
- **Size before you load.** Same arithmetic as the notebooks: params x 2 bytes
  (fp16). A base Whisper (~74M) is ~150 MB; anything past a few hundred M params
  is a slow first load and a real memory risk in a tab.
- **Feature-detect and fall back.** Always `pickBackend()` and degrade WebGPU ->
  WASM; never assume WebGPU exists (Safari/Firefox coverage is still partial).
- **Warm up.** First inference compiles WebGPU shaders / JITs WASM - run one dummy
  inference on load so the user's first real request is fast.
- **Cache is your friend.** Weights persist in Cache Storage after the first
  download, so the app works offline on repeat visits.

---

## 6. Minimal React hook tying it together

```tsx
// src/hooks/useAsr.ts
import { useEffect, useRef, useState } from "react";

export function useAsr(model = "onnx-community/whisper-base") {
  const workerRef = useRef<Worker | null>(null);
  const [ready, setReady] = useState(false);
  const [text, setText] = useState("");

  useEffect(() => {
    const w = new Worker(new URL("../workers/asr.worker.ts", import.meta.url), { type: "module" });
    w.onmessage = (e) => {
      if (e.data.type === "ready") setReady(true);
      if (e.data.type === "result") setText(e.data.out.text);
    };
    // backend/dtype chosen inside the worker via pickBackend()/loadOpts()
    w.postMessage({ type: "load", payload: { model, opts: {} } });
    workerRef.current = w;
    return () => w.terminate(); // frees the model + backend context on unmount
  }, [model]);

  const transcribe = (audio: Float32Array) =>
    workerRef.current?.postMessage({ type: "run", payload: { audio, args: { chunk_length_s: 30 } } });

  return { ready, text, transcribe };
}
```

```tsx
// src/components/Transcriber.tsx
import { useAsr } from "../hooks/useAsr";
import { recordMic } from "../lib/audio";

export function Transcriber() {
  const { ready, text, transcribe } = useAsr();
  return (
    <div>
      <button disabled={!ready} onClick={async () => transcribe(await recordMic(5))}>
        {ready ? "Record 5 s" : "Loading model..."}
      </button>
      <p>{text}</p>
    </div>
  );
}
```

---

## 7. Reference

- **Transformers.js** - `@huggingface/transformers`, WebGPU + WASM audio pipelines
  (`automatic-speech-recognition`, `audio-classification`,
  `zero-shot-audio-classification`, `text-to-speech`, `text-to-audio`).
- **kokoro-js** - browser-native Kokoro-82M TTS with WebGPU.
- **onnxruntime-web** - direct ONNX inference for models with no pipeline
  (DeepFilterNet, Demucs, custom exports); `onnxruntime-web/webgpu` entry point.
- **ONNX model hubs** - the `onnx-community/*` and `Xenova/*` orgs on the HF Hub
  mirror the notebook checkpoints in ONNX form.
- Source notebooks: [`00_Text_to_Speech`](00_Text_to_Speech.ipynb),
  [`01_Text_to_Audio`](01_Text_to_Audio.ipynb),
  [`02_Automatic_Speech_Recognition`](02_Automatic_Speech_Recognition.ipynb),
  [`03_Audio_to_Audio`](03_Audio_to_Audio.ipynb),
  [`04_Audio_Classification`](04_Audio_Classification.ipynb).
