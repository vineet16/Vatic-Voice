# Design decisions

Decisions not fully specified by the build spec. When in doubt the more
conservative option was chosen (fall back, refuse, log).

## Environment and test harness

- **Offline LLM stand-ins.** No API key is assumed. The reference agent runs
  through the `LLMClient` interface either against an OpenAI-compatible endpoint
  (`OPENAI_API_KEY`) or `ScriptedClinicLLM`, a deterministic policy that reads the
  OpenAI-style message history and returns tool calls/text like a model would. The
  caller simulator has the same split (`ScriptedCaller` / `LLMCaller`). The
  scripted model is deliberately *not* built on Vatic's membership checks: it
  handles corrections, digressions and multi-intent turns more leniently, like a
  capable LLM.
- **The scripted model runs in a worker process.** A real LLM is remote and never
  holds the voice process's GIL. Running the stand-in on threads made it compete
  with the event loop for the GIL and distorted loop-lag measurements. Its
  simulated network latency is an `asyncio.sleep`.
- **Throughput tests vs. hot-path tests.** Simulations that run the caller and
  model with zero latency saturate the CPU, so loop timing there reflects the
  machine. Those tests are marked `allow_slow_callbacks`. Loop safety is asserted
  separately in `tests/test_nonblocking.py`.
- **Debug-mode overhead in tests.** All async tests run with asyncio debug mode
  (`slow_callback_duration = 5 ms`) and fail on slow-callback warnings. Debug mode
  records a full traceback for every Task/Future; `linecache.checkcache`
  (stat()-ing every file in pytest's deep stacks) is stubbed in tests because that
  bookkeeping alone exceeded 5 ms. The 100-concurrent-call audio test measures in
  production mode instead (debug off, every callback timed by wrapping
  `Handle._run`), because debug bookkeeping for 100 calls would dominate.
- Python 3.13 is used locally; the code targets 3.11+.

## Runtime

- **Tool execution path.** The user's agent must call tools through
  `TurnContext.call_tool` so calls are traced and `enter_flow` / `resume_flow` are
  intercepted. Synchronous tool handlers run on the runtime's bounded executor.
- **Session reference date.** Date normalisation needs "today"; it comes from
  session metadata (`today`), recorded in `SessionTrace.metadata` so the compiler
  normalises exactly as the runtime did.
- **enter_flow slot grounding.** Slot values the LLM passes to `enter_flow` /
  `resume_flow` are accepted only if the slot's deterministic extractor recovers
  the same value from something the caller actually said in the session. The
  caller's span (not the LLM's string) becomes the slot value.
- **enter_flow handoff.** When a flow takes over, the LLM's own text for that turn
  is discarded and the flow's rendered reply is used; the tool result tells the
  model to reply with nothing. The reference agent stops its loop on handoff.
- **resume_flow semantics.** The LLM's reply is used for the turn (it asks the
  step's question itself); the flow then waits at the named ask/confirm step. Tool
  outputs the LLM produced in that turn replace the flow's recorded outputs for
  upstream steps (e.g. a new `check_availability` after a date correction), and
  are re-checked against learned guards. Resume is rejected if any downstream
  binding cannot be satisfied.
- **Suspended flows expire** after 3 LLM turns without a resume.
- **`llm` bindings at runtime fall back.** v0.1 has no LLM argument filler; a flow
  step with an `llm` binding hands the turn to the LLM. Such bindings are never
  allowed in irreversible steps.
- **Phrasing nodes** need a `phrasing_llm`; without one they fall back.
- **Guards split by timing.** Declared guards and tool invariants that read
  `output` run after the call; all others (and learned `args.*` domains) run
  before it, so irreversible calls are pre-checked. A post-call guard failure on
  an irreversible step demotes the flow.
- **Demotion** to `shadow` happens on any irreversible-step failure, or when the
  fallback rate over the last 50 attempts (min 20) exceeds 60%. Persisted off the
  event loop.
- **Late timings.** `annotate_turn` re-emits a turn record with e.g.
  `tts_first_audio`; SQLite keeps the latest, the JSONL log keeps both lines.
- **Trace queue** default size 10,000; when full, records are dropped and counted.
  The writer drains it in batches of at most 64 so each loop step stays short.
- **GIL switch interval.** Any Python thread doing CPU work (trace serialisation,
  SQLite tool handlers, YAML) can hold the GIL for up to CPython's switch interval
  (5 ms by default) before the audio loop thread gets it back; measured, this caused
  occasional 7-8 ms loop stalls. `RuntimeConfig.gil_switch_interval_ms` (default 1)
  lowers it at `start()`. It is process-wide; set it to `None` to leave it alone.
- **libyaml.** Flow YAML is parsed/dumped with PyYAML's C implementation when
  available (~10x faster, so far shorter GIL holds on the executor).
- **Split turn API.** `handle_turn(llm_fallback)` is `begin_turn` + the fallback +
  `complete_turn`. Frameworks that run their own LLM/tool loop (LiveKit, Pipecat) use
  `begin_turn`/`complete_turn` directly. A new turn arriving while one is pending
  (barge-in) completes the old one with `error="interrupted"`.
- **enter_flow refuses premature slots.** If the caller has already said a value
  that a later ask of the flow would request (e.g. a date, for the variant that asks
  "what day?"), the flow is the wrong variant: `enter_flow` is rejected and the LLM
  carries on. Shadow entry applies the same rule.
- **Tool manifest.** The runtime writes `tools.json` (names, side effects,
  parameters) into the trace store, so `vatic compile` needs no user code.

## Membership checks and the classifier

- **Training is a separate step** (`vatic train`), not part of `vatic compile`:
  compile stays torch-free and byte-deterministic; the compiler only emits the
  per-step example datasets. One cross-encoder (TinyBERT-L2, 4.4M params, ~0.25 ms
  per CPU inference via ONNX Runtime) is fine-tuned per flow on (step prompt,
  utterance) pairs, so it serves every eligible step of that flow; it is a
  cross-encoder conditioned on the step, as the spec asks. Steps below 50 on-path /
  20 off-path examples get no classifier. Other steps' on-path answers are added as
  off-path examples for each step.
- **The classifier only ever narrows** what the rules accepted (layers 1-3 run
  first and their off-path verdict is final), so it can cost coverage but never
  adds risk relative to the rules.
- **Calibration** uses split conformal risk control on shadow outcomes (loss =
  accepted-and-off-path), which bounds the *expected* wrong-route rate per step by
  alpha. It needs >= 1/alpha - 1 calibration points; below that the step stays
  rules-only and the report says so (no claim without data). 30% of shadow turns are
  held out to report the achieved rate. The reject threshold is the 5th percentile
  of on-path calibration scores.
- **Shadow mode labels every rules-accepted turn** (it compares even when the
  classifier would have declined), so calibration data is not biased toward easy
  cases; promotion match rates count only turns the live flow would have handled.
- **Classifier failures are "uncertain"**: a missing model, an ONNX error or a
  timeout (the membership budget, 30 ms) never yields on_path.
- **Hedged fallback.** In `handle_turn`, an uncertain decision starts the LLM
  immediately and, in parallel, runs the flow's continuation; the flow wins if it
  reaches its next wait point within `hedge_budget_ms` (250). Hedging is only
  attempted when the utterance has no residual content, the turn's guards pass,
  and every tool before the next wait point is read-only. While undecided, the
  LLM may only call read-only tools; side-effecting and flow tools wait for the
  outcome and are cancelled if the flow wins. The cancelled LLM turn must roll back
  its own history (the reference agent does). `begin_turn` (framework adapters)
  cannot start the framework's LLM, so there an uncertain turn simply falls back.

## Membership rules

- **Extra conservative markers.** Besides negation/correction (spec layer 3),
  multi-intent markers (`also`, `but`, `another`, ...) and question words
  (`what`, `how`, `do you`, ...) mark an utterance off-path.
- **Disfluencies** (`um`, `uh`, ...) are removed before membership and confirm
  classification (ASR inserts them inside phrases like "sounds um, good").
- **Ambiguous slots.** Two different spans for one expected slot is off-path.
- **`residual_max_tokens`** per ask step = the largest residual observed on-path in
  the compiled traces, capped at 3.
- **Names need truecased ASR.** The person-name extractor requires capitalised
  words; lower-cased transcripts fall back to the LLM.

## Compiler

- **Const vs. the caller's words.** The spec's order tries `const` first. A value
  that never varied in the data but that the caller's words explain on every trace
  (e.g. "Saturday" when the corpus spans one week) is bound as a slot/transform
  instead: a coincidental constant would silently ignore what a future caller says.

- **Segmentation.** One task segment per session: first turn with a tool call to
  the last. Sessions with any compiled turn, virtual tool, tool error or missing
  output are not used for compilation.
- **Alignment requires identical shape** (tool names per turn). Different shapes of
  the same tool signature become separate flows (`book_appointment`,
  `book_appointment_2`, ...) when they have enough support; merging is left to
  branch synthesis.
- **Confirm detection.** An agent turn is a `confirm` step if the next turn runs a
  side-effecting tool and >= 90% of callers answered with a clean yes.
- **Binding search order** follows the spec (const, output, slot, transform, llm),
  with one change: an output field reached through a *list index*
  (`times.1`) is tried only after slot and transform. A stable list position is
  weaker evidence than the caller's own words.
- **Inapplicable traces.** A candidate binding is rejected if any trace
  *contradicts* it. A trace where the binding is merely *inapplicable* (no unique
  span, e.g. ASR noise produced two dates, or a missing path) would fall back at
  runtime anyway; up to 5% of a cluster may be dropped for this, and the dropped
  session ids are recorded in `provenance.notes`. If support falls below
  `min_support`, the flow is refused.
- **Entry slots** may come from earlier turns than the entry turn (the caller gave
  their name before the agent acted on it): the most recent transcript with a span
  is used, the same rule the runtime uses to ground `enter_flow` slots.
- **Templates.** Placeholders prefer whole fields of the latest steps, then slots,
  then list elements; equal-length matches are resolved by how many traces each
  placeholder explains. Up to three template variants with identical placeholders
  may jointly reach the 90% coverage; the most frequent is used.
- **Learned domains are limited to what the flow depends on**: tool arguments,
  bound or templated output fields (and their parents), top-level boolean flags,
  and slots. Booleans with one observed value become enums; strings with <= 3
  values (and 10x support) become enums, otherwise <= 5 shapes become a pattern;
  numbers become ranges; lists get length ranges; containers never observed null
  get `not_null`. `member_of` guards are learned when an argument always appears in
  an earlier output list (e.g. a booked time was always one of the offered times),
  and are also placed on the ask step that collects the slot, so an unoffered
  time falls back before the confirm turn.
- **Inserted confirms** (when traces show none) are generated from the tool's
  slot-bound arguments. In shadow mode they are assumed answered "yes" (the LLM
  did not ask), so tool comparisons continue.
- **Behaviour split.** Traces with one tool shape can still behave differently
  (e.g. "no openings, another day?" vs. offering times). If a reply step's traces
  fall into distinct placeholder sets, the cluster is split by them (depth <= 2);
  classes below `min_support` are dropped and reported, never guessed.
- **Branch merging** is attempted only between flows with the same goal (same last
  tool) and the same entry slots: otherwise data-only features would "explain"
  intent (e.g. "patient has an appointment" separates *book* from *cancel*). The
  condition must separate the traces perfectly (depth-1 single-feature tests,
  preferring equality on both sides, then depth-2 conjunctions); the branch's
  default is `fallback`. One branch per flow in v0.1. Learned domains of the shared
  prefix are widened to the union of both branches (domains present in only one
  branch are dropped; the branch condition covers the divergence).
- **Flow ids** are the last tool name of the signature, suffixed `_2`, `_3` for
  further shapes (ordered by support).

## Adapters

- **LiveKit** (`livekit-agents==1.8.4`): `VaticAgent` overrides `llm_node`. Compiled
  replies are yielded as text; otherwise LiveKit's default `llm_node` runs with this
  turn's tool schemas, and LiveKit's own tool loop executes the agent's registered
  tools (all routed through the runtime by name). Tool calls LiveKit's history never
  saw (compiled steps, a flow before it fell back) are merged into each LLM request by
  timestamp; the paused-flow note is placed just before the user's message.
- **Pipecat** (`pipecat-ai==1.12.0`): two processors, `input()` before the LLM and
  `output()` after it. Pipecat re-runs the LLM after function results by sending the
  context frame upstream from the assistant aggregator, so a processor in front of
  the LLM only sees the user turn; the output tap is needed to know when the LLM's
  turn has finished (a response that ends without tool calls). On a flow hand-off the
  function handler pushes the flow's reply and returns `run_llm=False`.
- **Examples** run offline in text mode (LiveKit `AgentSession` without a room, a
  Pipecat `PipelineWorker` without a transport) with LLM wrappers around the scripted
  model, and the integration tests replay recorded transcripts through all three
  pipelines and require identical routes and replies.

## Lifecycle

- **Shadow entry.** A shadow run starts on the first LLM turn whose first tool call
  matches the flow's first tool step, with entry slots extracted
  deterministically. Among flows with the same tool signature that could enter,
  only those with a maximal entry-slot set start (as the LLM would pick the most
  specific variant).
- **Shadow evidence is kept only for the right task.** At session end, a run's
  comparisons count only if the LLM's tool sequence from the entry turn starts
  with one of the flow's tool paths; otherwise the run is discarded.
- **A run stops** at its first mismatch or would-be fallback.
- **Promotion counts complete sessions**: a shadowed session counts toward
  `min_sessions` only if the shadow run reached the end of the flow with every
  compared turn matching. Match rate is turn-level over compared turns.
- **Wrong-route rate** = on-path membership decisions whose turn did not match
  the LLM's behaviour, per step, from shadow.
- Valid candidates move to `shadow` automatically (`vatic promote --auto`).
