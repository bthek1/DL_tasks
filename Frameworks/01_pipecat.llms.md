# Pipecat — Complete Reference

Open-source Python framework for real-time voice and multimodal conversational agents. Maintained by Daily (the WebRTC company). BSD 2-Clause. Reached v1.0; currently on the 1.x line with 2.0 deprecations already flagged in the codebase.

Author

Benedict Thekkel

## 1. Core architecture

Four concepts. Learn these and the rest is a catalogue.

### Frame

The unit of data. Everything is a frame: audio chunks, transcriptions, LLM tokens, TTS audio, control signals, errors, lifecycle events.

``` python
AudioRawFrame          # PCM audio
TranscriptionFrame     # final STT result
InterimTranscriptionFrame
TextFrame              # LLM output, TTS input
TTSAudioRawFrame       # synthesized audio
UserStartedSpeakingFrame / UserStoppedSpeakingFrame
BotStartedSpeakingFrame / BotStoppedSpeakingFrame
LLMRunFrame            # kick the LLM
InterruptionFrame      # barge-in
EndFrame               # graceful shutdown
ErrorFrame
```

### FrameProcessor

The unit of computation. Implements `process_frame(frame, direction)`, consuming one frame and producing zero or more. Either transforms the frame or passes it through untouched.

``` python
class MyProcessor(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)   # required
        if isinstance(frame, TranscriptionFrame):
            await self.push_frame(TextFrame(frame.text.upper()), direction)
        else:
            await self.push_frame(frame, direction)     # pass through
```

The `super().process_frame()` call is not optional — it handles system frame plumbing, interruption state, and metrics. Forgetting it is the single most common source of “my custom processor broke interruptions.”

Every AI service (Deepgram STT, OpenAI LLM, Cartesia TTS) is just a `FrameProcessor` subclass.

### Pipeline

An ordered list of processors, linked into a chain. Pipelines are themselves processors, so they nest.

``` python
pipeline = Pipeline([
    transport.input(),
    stt,
    context_aggregator.user(),
    llm,
    tts,
    transport.output(),
    context_aggregator.assistant(),
])
```

Internally the pipeline wraps the chain in a `PipelineSource` and `PipelineSink` so frames can enter and leave from either end.

### PipelineTask / PipelineRunner

`PipelineTask` is one running instance of a pipeline — it owns lifecycle, config, metrics, idle timeouts, and is what you queue frames into from outside the pipeline. `PipelineRunner` executes tasks and handles SIGINT/SIGTERM.

``` python
task = PipelineTask(
    pipeline,
    params=PipelineParams(
        allow_interruptions=True,
        enable_metrics=True,
        enable_usage_metrics=True,
    ),
    conversation_id="...",
    enable_tracing=True,
    idle_timeout_secs=300,
)

runner = PipelineRunner(handle_sigint=True)
await runner.run(task)
```

`PipelineTask` exposes event handlers:

``` python
@task.event_handler("on_idle_timeout")
async def _(task): ...

@task.event_handler("on_pipeline_error")
async def _(task, frame): ...

@task.event_handler("on_pipeline_started")
async def _(task, frame): ...
```

------------------------------------------------------------------------

## 2. Frame direction and the two lanes

Frames flow **DOWNSTREAM** (mic → speaker) or **UPSTREAM** (speaker → mic). Upstream is how a late processor signals an earlier one — e.g. the output transport telling the pipeline that the user started talking so the LLM should be cancelled.

More importantly, there are **two scheduling lanes**:

- **Normal frames** — queued per-processor, processed in order, in that processor’s own asyncio task. Ordering is guaranteed within a processor.
- **System frames** — processed in a *separate* task with priority, bypassing the queue.

This is the mechanism that makes interruption work. If `InterruptionFrame` sat behind 800ms of queued audio frames, barge-in would be useless.

`EndFrame` is a **control** frame — it travels in order, so everything queued ahead of it gets flushed first. It’s also marked uninterruptible so a concurrent interruption can’t drop it. `CancelFrame` is the system-frame equivalent: immediate, lossy, doesn’t flush.

**Rule of thumb:** `EndFrame` to hang up politely after the bot finishes its sentence. `CancelFrame` to kill it now.

------------------------------------------------------------------------

## 3. Concurrency model

Each processor runs its own task, so the pipeline is a software-pipelined streaming system, not a request/response chain. At steady state during one turn:

| Stage | Running when | Location |
|----|----|----|
| VAD (Silero ONNX) | continuously, every ~32ms | **local, in-process** |
| Turn analyzer (smart-turn v3) | on each VAD-detected pause | **local** |
| STT | continuously, streaming websocket | remote |
| LLM | after turn-end, streaming tokens | remote |
| TTS | per sentence, overlapping the LLM | remote |

VAD and turn detection must be local — a network round trip per 32ms chunk is a non-starter.

**The main latency trick:** a sentence aggregator sits between the LLM and TTS, buffering tokens until a sentence boundary. TTS starts synthesizing sentence 1 while the LLM is still generating sentence 3. Waiting for the full completion before starting TTS adds roughly a second to time-to-first-audio.

Target end-to-end (user stops speaking → first bot audio): **under 800ms**, ideally ~500ms. Budget roughly: turn detection 100–300ms, LLM TTFT 200–400ms, TTS TTFB 100–150ms.

------------------------------------------------------------------------

## 4. VAD and turn detection

Two distinct things, frequently conflated.

**VAD** answers “is there speech energy right now.” Silero VAD, running locally as ONNX. Cheap, fast, no semantics. `VADParams(stop_secs=0.2)` controls how much silence counts as a stop.

**Turn detection** answers “is this person *finished*.” Silence alone is a bad proxy — people pause mid-sentence to think, and “my account number is… four two…” is not a completed turn. Pipecat ships `LocalSmartTurnAnalyzerV3`, a small local transformer that classifies utterance completeness from audio and transcript context.

In v1.x these are configured on the **user context aggregator**, not the transport:

``` python
context = LLMContext(messages)
user_agg, assistant_agg = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        user_turn_strategies=UserTurnStrategies(
            stop=[TurnAnalyzerUserTurnStopStrategy(
                turn_analyzer=LocalSmartTurnAnalyzerV3()
            )]
        ),
    ),
)
```

(Pre-1.0 code put `vad_analyzer` on `TransportParams` / `DailyParams`. If you find a tutorial doing that, it’s stale.)

**Alternative:** some STT providers now do end-of-turn detection natively and replace the external VAD entirely — Deepgram Flux is the notable one, with sub-300ms median EOT detection. Fewer moving parts, one less local model, but you lose the ability to swap turn logic independently of your STT vendor.

------------------------------------------------------------------------

## 5. Interruption / barge-in

When VAD fires during bot speech and `allow_interruptions=True`:

1.  `push_interruption_task_frame()` sends an `InterruptionTaskFrame` **upstream**.
2.  The pipeline task converts it into an `InterruptionFrame` pushed **downstream** as a system frame.
3.  The call returns only once the frame has fully propagated — so you get a real synchronization point.
4.  Each processor drops queued frames and cancels in-flight work: LLM request aborted, TTS websocket flushed, output transport’s audio buffer cleared.
5.  The assistant context aggregator truncates the assistant message to what was *actually spoken*, not what was generated.

That last step matters and is easy to miss when rolling your own. If the LLM generated three sentences and the user cut in after one, the context must record one sentence — otherwise the model believes it said things the user never heard, and the conversation desynchronizes.

**Design contrast:** Pipecat uses an in-band priority frame. The alternative (a monotonic turn/generation counter, invalidating by integer comparison) is simpler to reason about and lets any consumer discard stale work independently without coordination. Pipecat’s approach costs you correctness-per-processor — every processor must handle the frame properly — but buys you a synchronization point a bare counter doesn’t give.

------------------------------------------------------------------------

## 6. Context and function calling

`LLMContext` holds the message list. `LLMContextAggregatorPair` produces two processors:

- `context_aggregator.user()` — sits *before* the LLM, accumulates transcriptions into a user message and decides when the turn is complete
- `context_aggregator.assistant()` — sits *after* the output transport, records what the bot actually said

The assistant aggregator being placed after `transport.output()` is deliberate and is what makes interruption-truncation correct.

Function calling is provider-agnostic via `FunctionSchema` / `ToolsSchema`:

``` python
async def get_weather(params: FunctionCallParams):
    await params.result_callback({"temp": 22})

llm.register_function("get_weather", get_weather)

tools = ToolsSchema(standard_tools=[
    FunctionSchema(
        name="get_weather",
        description="Get current weather",
        properties={"city": {"type": "string"}},
        required=["city"],
    )
])
```

Pipecat translates the schema to each provider’s format, so switching OpenAI → Anthropic → Gemini doesn’t require rewriting tool definitions.

**Latency gotcha:** a function call means the user hears silence during the round trip. Standard mitigation is to speak a filler line (`TTSSpeakFrame("Let me check that")`) before executing.

------------------------------------------------------------------------

## 7. Transports

| Transport | Use case |
|----|----|
| `DailyTransport` | WebRTC, the first-party option, best-supported |
| `SmallWebRTCTransport` | local dev, no cloud account, browser client at `localhost:7860` |
| `FastAPIWebsocketTransport` | your own websocket server |
| `TwilioFrameSerializer` / Telnyx / Plivo | telephony via media streams |
| `LiveKitTransport` | run Pipecat on top of LiveKit’s SFU |
| `LocalAudioTransport` | system mic/speaker, useful for testing |

Transports expose `transport.input()` and `transport.output()` as processors, plus event handlers:

``` python
@transport.event_handler("on_client_connected")
async def _(transport, client):
    await task.queue_frames([LLMRunFrame()])

@transport.event_handler("on_client_disconnected")
async def _(transport, client):
    await task.cancel()
```

The `create_transport(runner_args, transport_params)` helper plus the `bot(runner_args)` entry point convention lets one file run locally *and* deploy to Pipecat Cloud unchanged.

------------------------------------------------------------------------

## 8. Services

Install is extras-based, so you only pull what you use:

``` bash
pip install "pipecat-ai[silero,local-smart-turn-v3,deepgram,openai,cartesia,webrtc,runner]"
```

Rough catalogue (the integration library is Pipecat’s main advantage over alternatives):

- **STT** — Deepgram, AssemblyAI, Speechmatics, Whisper (local + hosted), Gladia, Azure, AWS Transcribe, Cartesia, ElevenLabs Scribe, Groq, Riva
- **LLM** — OpenAI, Anthropic, Google Gemini, AWS Bedrock, Azure, Groq, Together, Fireworks, Cerebras, Ollama, OpenRouter, Grok
- **TTS** — Cartesia, ElevenLabs, Rime, PlayHT, Deepgram Aura, OpenAI, Azure, AWS Polly, Neuphonic, LMNT, Piper (local)
- **Speech-to-speech** — OpenAI Realtime, Gemini Live/Multimodal Live, AWS Nova Sonic
- **Other** — Silero VAD, Krisp noise reduction, Tavus/Simli/HeyGen video avatars

**Speech-to-speech models** collapse STT+LLM+TTS into one processor. Lower latency and better prosody/emotion; you give up the ability to inspect or modify the transcript mid-pipeline, swap providers per stage, or cheaply log text. Pipecat supports both topologies, which is a genuine reason to pick it — you can A/B the architectures without rewriting the app.

------------------------------------------------------------------------

## 9. Advanced topologies

`ParallelPipeline` runs branches concurrently over the same frame stream — e.g. transcribe *and* run sentiment analysis *and* record, without any branch blocking the others.

``` python
Pipeline([
    transport.input(),
    ParallelPipeline(
        [stt, context_aggregator.user(), llm],
        [sentiment_analyzer],
        [call_recorder],
    ),
    tts,
    transport.output(),
])
```

Other useful built-ins: `ProducerProcessor`/`ConsumerProcessor` for cross-branch communication, `FunctionFilter` and `frame_filter` for conditional routing, `UserIdleProcessor` for “are you still there” prompts, `STTMuteFilter` to stop the bot transcribing itself or ignore input during a scripted segment.

------------------------------------------------------------------------

## 10. Pipecat Flows

Separate package (`pipecat-flows`) for structured conversations — a state machine of nodes, each with its own system prompt, tool set, and transition conditions. Use it for anything with a required path: intake forms, qualification, appointment booking, compliance disclosures.

``` python
flow_manager = FlowManager(
    task=task, llm=llm,
    context_aggregator=context_aggregator,
    transport=transport,
)
await flow_manager.initialize(create_initial_node())
```

Skip it for open-ended companion or assistant bots — a single well-written system prompt is simpler and Flows’ rigidity works against you.

------------------------------------------------------------------------

## 11. Observability

- **Metrics** — `enable_metrics=True` gives per-processor TTFB and processing time. This is how you find which stage is eating your latency budget; guessing is a waste of time.
- **Tracing** — OpenTelemetry spans via `enable_tracing=True`, with turn-level grouping (`enable_turn_tracking`).
- **RTVI** — a client/server protocol for shipping pipeline state to the frontend (bot speaking, user speaking, transcripts, custom events). Enabled by default; it’s what the JS/React SDKs consume.
- **Observers** — `BaseObserver` subclasses see every frame push without being in the pipeline. Right tool for logging and debugging.

------------------------------------------------------------------------

## 12. Testing

The honest state of things: testing real-time voice agents is harder than testing the framework.

- Pipecat has test helpers for driving frames through a pipeline and asserting on output frames.
- Deterministic unit tests want fixture-driven fake transports and mock services, so tests assert on *order and causality* rather than wall-clock timestamps.
- End-to-end quality (does it actually handle a real caller who mumbles and interrupts) needs synthetic-caller tooling — Coval, Cekura, Hamming and similar. Framework choice matters far less than having this.

------------------------------------------------------------------------

## 13. Deployment

Pipecat processes are **stateful and long-lived** — one process per conversation, holding websockets to STT/TTS and a WebRTC connection. This breaks most default assumptions:

- Not serverless-friendly. No Lambda, no scale-to-zero without cold-start pain (loading Silero + smart-turn takes seconds).
- Scaling unit is concurrent *sessions*, not requests.
- Needs sticky routing and graceful drain on deploy — you can’t kill a pod mid-call.
- Cold start matters enormously; pre-warm workers.

**Pipecat Cloud** is Daily’s managed hosting for exactly this. Self-hosting is entirely viable (containerize, run behind an autoscaler with generous headroom), it’s just real infrastructure work.

------------------------------------------------------------------------

## 14. Gotchas worth knowing

1.  **Forgetting `await super().process_frame(frame, direction)`** in a custom processor breaks interruptions in ways that look unrelated.
2.  **Not passing frames through.** A processor that only handles `TextFrame` and silently drops everything else will kill your `EndFrame` and hang the pipeline.
3.  **Blocking the event loop.** Any sync CPU work in a processor stalls every other processor. Use `run_in_executor`.
4.  **Assistant context not truncated on interruption** — if you write custom aggregation, the model’s memory diverges from what the user heard.
5.  **TTS websocket reconnection.** Providers drop idle connections. Check the service handles reconnect, or your bot goes mute after a pause.
6.  **VAD tuning is the biggest UX lever.** Too aggressive → interrupts users mid-thought. Too lax → dead air. `stop_secs` is worth real tuning time against recordings of your actual users.
7.  **Echo.** Without acoustic echo cancellation the bot hears itself and interrupts itself. Daily/LiveKit WebRTC handles this; raw websocket transports do not, and telephony varies.
8.  **Cost.** Per-minute STT + LLM tokens + per-character TTS. A chatty bot at scale is expensive; measure before committing to premium TTS.

------------------------------------------------------------------------

## 15. Talking points

Things that show you understand the problem rather than the API surface:

- **Turn detection is the hard part, not the pipeline.** Silence-based endpointing is a placeholder; semantic endpointing is the real answer, and it trades ~30ms of local inference for a large drop in false interruptions.
- **Interruption correctness is about context, not audio.** Stopping playback is easy. Making the model’s memory match what the user actually heard is the part that breaks.
- **Speaker identity should not key your correctness logic.** Streaming diarizer labels are unstable — they swap and re-cluster mid-stream. Use identity to *gate* barge-in (should this voice be allowed to interrupt me?) rather than to partition turn state.
- **Latency is a budget, not a number.** Know which stage owns which milliseconds, and measure rather than guess.
- **Local vs remote placement is forced by cadence.** Anything running per-audio-frame has to be in-process.

------------------------------------------------------------------------

## Links

- Docs: https://docs.pipecat.ai
- API reference: https://reference-server.pipecat.ai
- Repo + `examples/foundational/`: https://github.com/pipecat-ai/pipecat — the numbered examples are the best learning path, in order

------------------------------------------------------------------------

Back to top
