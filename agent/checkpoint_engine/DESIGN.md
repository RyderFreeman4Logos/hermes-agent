# Checkpoint Context Engine

This file is the first-principles spec for `context.engine: checkpoint`.
Code that contradicts it is wrong. Update this file before expanding scope.

v1 is the smallest engine that satisfies the economics and the invariants
below. Anything listed under Non-goals must not ship in this PR.

## 1. Why this exists

Hermes already has two compressor tail policies:

- `legacy` — on 240K–270K unattended coding sessions the post-compaction
  wire request can remain 100K–200K. That is not a low-water mark.
- `lean` — much smaller, keeps identifiers, user text, and durable history.
  It still fail-opens (placeholder digests, implicit expensive fallback,
  in-flight tool tears, active intent folded into prose).

This engine is **not** a third `tail_mode`. It is a new `ContextEngine`
implementation. `compressor` + `legacy`/`lean` stay the default and must
not change behavior.

### The actual reason: cheap models, not prettier summaries

Compression is an **output-heavy** job. Prompt cache discounts *input
processing*, not generated or reasoning tokens.

Consequence: an expensive main model at a 95% input-cache hit can still
cost more than a cheap/local model that cache-misses every shard. Cache
hits are also not an SLA (TTL, eviction, routing, schema/prefix drift).
Cost the **uncached worst case**. Never make “reuse the main model so we
can hit the KV cache” the default.

Small models fail in a different way: long unstructured context → context
rot, dumb-zone collapse, hallucination. The mitigation is mechanical, not
a smarter prompt:

- shard on causal boundaries into 8K–16K inputs
- pack consecutive causal units toward 12K; units over 16K stay external
- complementary typed views, not N copies of the same prose summary
- ~1K hard output cap per shard, thinking off/low
- require source event IDs; reject truncated/invalid JSON
- default Map concurrency 2
- one failed required shard fails the **candidate**, not the live session

Optimize for **continuation quality per active token**, not ROUGE and not
minimum token count in isolation. Initial envelope:

| Quantity | v1 default |
|---|---|
| Typical post-compaction full wire request | ~48K tokens |
| Hard ceiling (commit forbidden above) | 60K tokens |
| Recent causal tail | 14K base, adaptive ≤24K |
| Active user turn | separate verbatim lane |
| Map concurrency | 2 |
| Fallback | configured auxiliary chain only |

Do not target 25K–30K until recovery + replay evidence exist.

## 2. Mental model

```
immutable raw session events
        │
        ├── deterministic safety lanes   (no LLM)
        ├── causal shard plan            (no LLM)
        ├── cheap-model typed Map
        ├── deterministic Reduce         (no LLM)
        ├── sourced semantic selection   (cheap model; ids only)
        └── full-wire budget check       (no LLM)
                │
                ▼
     model-facing checkpoint projection
```

The durable session is an append-only event log. The LLM context is a
projection. Compaction **compiles** a new checkpoint generation over a
precise event range. It must not delete or rewrite source events.

v1 does **not** invent a second event store. Hermes session DB already
keeps raw messages. Treat those messages as the log. Do not build
incremental “summary of previous summary” lineage in v1: each successful
compaction maps the covered raw range again. That is both simpler and
immune to recursive prose drift.

## 3. Invariants

These are correctness properties. A checkpoint that violates any of them
must not commit. Failed compaction leaves the raw session authoritative.

**I-1 Raw history is immutable.** Compaction may hide a range from the
model-facing projection. It may not delete source messages needed for
resume, `/compress` undo, or `session_search`.

**I-2 Quiescent snapshot + CAS.** Do not compact while a tool is in
flight, while an emitted tool call has no persisted result/`unknown`, or
while a real user/`steer` event is unpersisted. Capture a session
revision; commit is compare-and-swap. Stale candidate is discarded.

**I-3 Active human intent is a separate lane.** Keep the newest
actionable root task plus later user corrections/`/steer`, never only
the last acknowledgment. Acknowledgments such as "continue"/"okay" do
not replace the root unless they clearly start a new task. Overflow
keeps the exact source in durable history and projects a hash, beginning,
ending, and verbatim high-priority constraints. Historical handoffs are
data. They must not become a new user instruction or auto-continue the
task.

**I-4 Imperatives need authority.** A surviving todo with no authorizing
user/policy/rationale is `blocked`, not executable.

**I-5 Plans are not effects.** Lifecycle is
`planned → issued → running → succeeded|failed|unknown`.
Assistant prose (“I wrote the file”) is not completion evidence.

**I-6 Mutation receipts.** State-changing tool results persist as
receipts (id, op, status, source event ids). Missing receipt ⇒ `unknown`,
never `succeeded`.

**I-7 Verification is bound to repo state.** A passing test on an old
HEAD/dirty hash is not evidence for a newer tree.

**I-8 Causal groups.** Never split a tool call from its result, or a
mutation from its receipt.

**I-9 Provenance.** Every model-derived fact carries source event ids or
an explicit `uncertain` mark. High-risk claims without a source cannot
pass validation.

**I-10 All required shards succeed.** Configured-chain retry is allowed.
Placeholder text such as `[digest unavailable]` is telemetry, not
checkpoint content. Exhausted chain ⇒ reject candidate.

**I-11 No implicit expensive fallback.** Timeouts, malformed output,
local OOM, or “model unavailable” must not retry the main model unless
that route is explicitly in the configured auxiliary chain.

**I-12 Hard low-water.** Measure the **full** provider-visible request with
the host request-budget service (system/developer prompt, tools/schemas,
skills, checkpoint, active turn, tail, structured output, multimodal parts,
provider overhead, output/reasoning reserve). Preserve any host-only overhead
from the source request when rendering a candidate. Over
`hard_max_wire_tokens` ⇒ reject. A “successful”
compaction that leaves 100K+ tokens is a contract violation.

**I-13 Checkpoints do not do work.** Generating or rendering a checkpoint
must not call task tools, mutate the workspace, or treat the checkpoint
as a user turn.

**I-14 Search is not the only recovery (v1 minimum).** Keep raw history.
Do not add new core tools in v1. Exact ids in the checkpoint may point
at existing `session_search` / file tools. Dedicated `event_read` tools
are a later additive engine-owned schema, not a v1 blocker.

## 4. What we take from other harnesses

Synthesis, not a clone of any one of them.

| Source | Keep | Drop / change |
|---|---|---|
| Pi | Structured Goal/Constraints/Progress/Decisions; never cut tool call from result; cumulative file ledger | Free-form iterative summary as sole state |
| OpenCode | Checkpoint + serialized recent tail; drain to a safe boundary; count system+tools+messages | Coalesce/idempotency theater beyond CAS |
| OpenHands | Append-only events; condensation names forgotten ids; cheap summary models | Condenser pipelines as speculative infrastructure |
| Codex | Preserve recent real user messages; rehydrate canonical prefix; `body_after_prefix` trigger | Main-model compact as default |
| Claude Code | Rehydrate project/git state after compact | Hook-only rehydration (make it an engine stage) |
| Parallel Context Compaction | Bounded parallel Map, predictable output caps | Assuming more concurrency is better |
| CompactionRL | Downstream continuation reward, not teacher-prose similarity | Training a model in this PR |

Hermes-specific failure classes that these invariants exist to close:
in-flight tool tear, pending tool/user/`steer` loss, todo surviving after
its policy was pruned, handoff replayed as a fresh user turn, oversized
unsplit turn blowing the tail, fail-open shard placeholders, implicit
main-model retry.

## 5. v1 pipeline

```
preflight
  → quiescent snapshot (revision, event range, in-flight/queue gens)
  → deterministic lanes (intent, governance, effects, identifiers)
  → causal shard plan (8K–16K, never split I-8 groups)
  → bounded parallel typed Map (concurrency 2, configured aux only)
  → deterministic Reduce (authority, supersession, action state, dedup)
  → cheap semantic selection (validated ids → deterministic checkpoint)
  → full-wire measurement + degradation order
  → CAS commit or abort (live session untouched on abort)
```

Deterministic stages have no LLM. Map/semantic selection use the configured
auxiliary chain. Semantic output may select validated source ids but may not
add prose; the checkpoint renderer formats the typed records deterministically.
Unsourced actions render as `blocked` and never enter the plan lane. There is
**no** separate LLM auditor and **no** repair pass in v1: deterministic
validation + I-12 is the gate; failure aborts.

Degradation order before reject:

1. collapse completed-epoch detail
2. replace reproducible tool bodies with exact refs
3. shorten old decision explanations
4. shrink tail at complete causal boundaries
5. reject candidate

Never shrink active intent, policy dependencies, mutation receipts, or
current verification to “make it fit”.

### Modes

- `shadow` — run the pipeline, persist diagnostics if cheap, **return the
  original messages**. Default until live is explicitly selected.
- `live` — replace the model-facing list only after CAS commit of a valid
  candidate.
- Failure / kill switch — behave as a no-op engine; do not fall through
  to a different engine mid-session (cache prefix).

### Projection shape after a live commit

1. canonical system / currently applicable skills (rehydrated, not
   summarized)
2. exact active user turn + corrections
3. trusted effect/verification lanes (bounded)
4. one continuity checkpoint as historical data (host-authored wrapper;
   not unqualified system/developer authority)
5. adaptive recent causal tail
6. optional tiny prefetch of next-action refs (≤5K, skip if nothing
   obvious — do not build a retrieval stack)

Map shard JSON never enters the prompt.

## 6. Plug-in point

Implement `agent.context_engine.ContextEngine` under
`agent/checkpoint_engine/` (or a single module if it stays small).

Select with existing config:

```yaml
context:
  engine: checkpoint
```

Default remains `compressor`. Do not add `compression.tail_mode: checkpoint`.
Do not add a new core toolset. Engine-owned tools, if any, go through
`get_tool_schemas()` / `handle_tool_call()` already on the ABC.

Reuse, do not rewrite:

- `agent.auxiliary_client.call_configured_auxiliary_chain` for configured-only
  compression routing; it shares the process-wide compression limit (default 2)
  and never reaches main-model fallback
- session DB raw messages
- identifier extraction already used by lean
- `ContextEngine` lifecycle (`update_from_response`, `should_compress`,
  `compress`, `on_session_start/end/reset`, `update_model`)

`compress()` must return a valid OpenAI-format message list. On any
invariant failure it returns the input list unchanged (and must not
increment a “successful compression” count).

## 7. Config surface (v1)

Hardcode invariants. Only these user-facing keys:

```yaml
context:
  engine: checkpoint          # existing selector; default compressor

checkpoint:
  mode: shadow                # shadow | live
  target_wire_tokens: 48000
  hard_max_wire_tokens: 60000
  map_concurrency: 2          # cap; do not auto-raise
  max_map_shards: 32          # hard cap; Map input/output budgets fail closed
```

Auxiliary model/provider/fallback stay under existing `auxiliary` /
compression aux config. Do not invent a parallel fallback list.
Reject live mode if raw history cannot be retained. Reject
`hard_max_wire_tokens` below protected lanes.

## 8. Non-goals (forbidden in this PR)

- Changing `legacy` or `lean` defaults or behavior
- Overnight issue workers, Issue Cards, quarantine queues
- Background precompaction
- Training / specializing a local compactor
- Cached-main compaction
- Fresh-session handoff every N generations
- Telemetry/cost-accounting product
- New core recovery tools (`session_event_read`, …)
- Vector search as primary recovery
- 80-knob TOML of boolean invariants
- Sharing parent/delegate cache or context
- Implicit main-model fallback

## 9. Tests that prove the principles

Behavior contracts, not snapshot counts. Each must fail before the
production branch that makes it pass.

1. In-flight tool call ⇒ `compress()` returns the original list.
2. Required Map shard invalid/truncated ⇒ original list unchanged;
   no `[digest unavailable]` in the returned messages.
3. Auxiliary chain exhausted ⇒ no call to the session main model.
4. Rendered full-wire estimate above `hard_max_wire_tokens` ⇒ abort.
5. Planner never splits a tool call from its result.
6. Latest real user turn is present verbatim outside the checkpoint
   body.
7. Assistant text “I wrote the file” does not mark that action
   `succeeded`.
8. `context.engine: compressor` tests still pass unchanged.
9. `mode: shadow` runs Map if needed but returns original messages.

Use `scripts/run_tests.sh` on the focused file. No live network.

## 10. Implementation order (atomic commits)

1. This document + empty engine registered as `checkpoint` (shadow
   no-op that never mutates messages).
2. Quiescent / CAS: in-flight and stale revision refuse to commit.
3. Deterministic lanes + causal groups (no LLM).
4. Typed Map + configured-fallback-only scheduler.
5. Deterministic Reduce + semantic Reduce + budgeted renderer + live
   commit path.
6. Focused tests for §9 (written RED before each production slice).

Cherry-pick the whole branch. Commits 1–5 should each leave tests green
for the behavior they add.
