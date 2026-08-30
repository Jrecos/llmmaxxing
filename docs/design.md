# LLMMaxxing Design

**Date:** 2026-08-30  
**Status:** Architecture approved; formal specification pending owner review  
**Design gate:** Full GO — no unresolved P0/P1 after specialist and fresh adversarial review

## Goal

Build **LLMMaxxing**, an Apache-2.0 self-hosted OSS admission, fair-queue, routing-policy and operations control plane for LiteLLM.

Required topology:

```text
Agents → LLMMaxxing Gateway → private LiteLLM → providers
Operators → LLMMaxxing Control → immutable policy publication
```

LLMMaxxing must make shared subscription capacity predictable, prevent provider overload from escaping to clients, give each workload key an explicit scheduling/routing policy, automate safe provider onboarding, and record enough queue/provider timing evidence to improve policy later.

The public project lives in an independent repository and is consumed here as a pinned Git submodule:

```text
Repository:     Jrecos/llmmaxxing
Image:          ghcr.io/jrecos/llmmaxxing
Submodule:      services/llmmaxxing
CLI:            llmmaxxing
Homelab domain: llmmaxxing.apps.internal
```

## Non-goals

V1 does not provide:

- direct provider SDKs or a provider registry—providers arrive through LiteLLM;
- multiple Gateway replicas or a distributed scheduler;
- durable queued inference across a Gateway crash;
- weighted or adaptive production routing;
- a public plugin ABI;
- SaaS/multi-organization tenancy;
- a Kubernetes operator;
- automatic upgrades;
- prompt/response storage in LLMMaxxing;
- live runtime label selectors or policy inheritance.

## Responsibility boundaries

### LLMMaxxing owns

- client API keys and local verification;
- key-to-policy binding;
- canonical Route Groups;
- Provider Account capacity and quotas;
- weighted-fair bounded admission queue;
- route selection and typed fallback causes;
- circuit/recovery state;
- immutable bundle publication and rollback;
- request lifecycle evidence;
- Control UI and CLI.

### LiteLLM owns

- provider credentials;
- provider-specific protocol translation;
- effective model deployments;
- hidden exact deployment aliases;
- spend logs and callbacks;
- Langfuse integration.

LLMMaxxing contains no provider SDK or provider-specific registry. An ordinary provider already supported by a certified LiteLLM version must be onboardable through discovered data and generic evidence without changing LLMMaxxing source.

## Runtime architecture

One repository and OCI image expose two long-running commands:

```text
llmmaxxing gateway
llmmaxxing control
```

The Gateway is the only inference data plane. Control is out of band. Gateway never reads Control, SQLite or PostgreSQL while handling inference.

```text
                            ┌──────────────────────────┐
Operator ──────────────────▶│ LLMMaxxing Control       │
                            │ UI · API · CLI · compiler│
                            └────────────┬─────────────┘
                                         │ signed bundle
                                         ▼
Agents ─▶ Gateway ─▶ fair queue ─▶ generation guard ─▶ LiteLLM ─▶ providers
             │
             ├─ durable active bundle + deny overlay
             ├─ attempt/RPM/quota/circuit journal
             └─ bounded lifecycle event spool
```

Exactly one Gateway process/event loop may dispatch in V1. Prefork/reload/multiple workers are rejected. A lifetime lock on the persistent Gateway state plus a monotonic dispatcher fence prevents a second dispatcher from becoming ready.

## Deployment modes

### Gateway-only signed-file

- one Gateway;
- signed compiled bundle and persistent Gateway state;
- offline signer/filesystem root is administration boundary;
- no pretend Control UI or database;
- bundle validity defaults to 30 days, maximum 90 days;
- expired bundle denies new and queued work while active attempts remain pinned.

### Control + SQLite

- separate Gateway and Control containers from one image;
- local SQLite WAL and one Control process;
- quickstart/evaluation only;
- Gateway still receives authenticated signed bundles and never opens SQLite.

### Control + PostgreSQL 16

- separate Gateway, Control and PostgreSQL services;
- production recommendation and homelab mode;
- dedicated database/schema/role, not LiteLLM's database.

In Control-backed modes, a signed authorization lease is valid for at most 15 minutes, but deny-feed freshness has a hard 10-second bound. If Gateway cannot prove a fresh deny sequence, it stops new and queued dispatch while active attempts remain pinned.

## Repository structure

```text
src/llmmaxxing/core/       pure types, compiler, state machines, reason codes
src/llmmaxxing/gateway/    auth, admission, scheduler, proxy, journal, UDS
src/llmmaxxing/control/    API, UI, RBAC, workflows, publication, audit
src/llmmaxxing/adapters/litellm/  certified discovery/dispatch integration
src/llmmaxxing/storage/    Control-only SQLite/Postgres repositories
src/llmmaxxing/config/     schema, compile, plan, export, signing
src/llmmaxxing/cli/        Control API client and offline/emergency commands
src/llmmaxxing/telemetry/  bounded inference evidence
ui/                        frontend source; compiled assets in runtime image
schemas/                   config, bundle, API, events, reason-code schemas
deploy/compose/            file, SQLite, Postgres and Dokploy examples
tests/                     unit, contract, integration, upgrade, migration, perf
```

Dependency direction is enforced in CI:

- core imports no HTTP, ASGI, SQL, LiteLLM, provider or UI module;
- Gateway imports no Control/storage module;
- Control imports no Gateway internals;
- core/defaults contain no nan, ArliAI, Electron, homelab alias, IP or domain.

## Trust and secret boundaries

### Gateway receives

- current client-key HMAC pepper; prior only during rotation;
- LiteLLM inference-only key restricted to exact hidden deployments;
- separate LiteLLM discovery-read-only key restricted to certified GET routes;
- Gateway device/mTLS private identity;
- bundle and Control-channel verification trust roots.

Gateway receives no database, OIDC, Control identity root, signing-private, provider, LiteLLM master or salt credential.

### Control receives

- Control database credential;
- identity root key;
- active client-key peppers needed to generate verifier records;
- distinct policy and audit signing keys;
- OIDC client secret in team mode;
- Control device/mTLS identity.

Control receives no LiteLLM credential, provider key, LiteLLM master/salt or Gateway inference key.

### LiteLLM receives

- existing provider/admin/database secrets;
- the two narrow Gateway credentials;
- the local generation-guard credential-fingerprint key.

LiteLLM never receives LLMMaxxing client tokens/verifiers or human session secrets.

Only Gateway inference and authenticated Control UI may be externally exposed. LiteLLM, PostgreSQL, Gateway management and Control internal channels have no public/host exposure.

## Human authentication and authorization

### Presets

- Gateway-only: filesystem/offline-signer administration.
- Quickstart/homelab: local passwordless passkey owner, explicit solo mode.
- Team production: direct OIDC Authorization Code + PKCE, plus WebAuthn step-up for privileged actions.

Tailscale/ForwardAuth may restrict reachability but their identity headers are ignored in V1.

No default account or open registration exists. `llmmaxxing control bootstrap-owner` emits a one-use 256-bit ten-minute local enrollment token.

Control uses opaque server-side sessions with 30-minute idle and 12-hour absolute expiry. Cookies are `__Host-`, Secure, HttpOnly, SameSite=Lax and Path `/`. Browser mutations require exact Origin, JSON/non-simple content and a session-bound 256-bit CSRF token.

Roles:

| Role | Authority |
|---|---|
| viewer | sanitized read/export |
| operator | probes, drain/freeze, circuit/account/leg disable, termination |
| publisher | route/account/policy/key-binding expansion and activation |
| security-admin | principals, sessions, credential issue/rotate/suspend/revoke |

In team mode, identity/RBAC expansion requires a second pre-existing security-admin who is neither proposer nor target. In solo mode, the owner may self-approve only after fresh passkey step-up, immutable impact review, mandatory reason and typed hash/count confirmation.

The identity root key domain-seals principal, credential, RBAC/security-epoch and approval records. WebAuthn approval assertions commit to operation/content/impact/base/key-set/security-epoch hashes. A database-only forged approval cannot produce a bundle signature or command.

Any principal disable, role/IdP change, passkey/recovery change or global session revoke increments a security epoch and invalidates affected sessions, CSRF state, step-up state and pending approvals.

## Control–Gateway trust

Control and Gateway use installation-scoped mTLS device identities. Commands, observations and acknowledgements bind:

```text
installation ID
channel-key epoch
dispatcher fence
Gateway boot ID
monotonic sequence
payload digest
command ID
```

Policy signing, audit checkpoint signing and channel identities use separate keys/trust epochs. Rotation stages the next identity under the old trusted epoch, requires configured approval, gets both-side durable acknowledgement, then fences the old identity.

## Client keys

Format:

```text
lmxk1.<128-bit-key-id>.<256-bit-secret>
```

The alias is display-only. Gateway stores/verifies only:

```text
HMAC-SHA-256(pepper,
  "llmmaxxing-client-key-v1\0" || key_id || secret)
```

Auth is O(1), constant-time and occurs before body/waiter/lease allocation. Unknown IDs perform dummy HMAC. Malformed, unknown, expired, suspended and revoked keys expose the same bounded 401 response.

Plaintext is returned exactly once in a TLS/no-store/no-referrer response. It is never persisted, logged, audited, exported or put in browser storage. Lost issuance means rotate.

Defaults:

- key lifetime 90 days; hard maximum 365 days;
- warnings at 30/7/1 days;
- rotation overlap 24 hours; hard maximum seven days;
- at most two active credential generations;
- revoked/expired identities are terminal tombstones.

Logical lifecycle:

```text
DRAFT → ENABLED ↔ SUSPENDED → REVOKED
```

Normal revoke/suspend commits state, audit and deny outbox. Gateway FULL-syncs the monotonic deny overlay before reporting enforcement. New and queued work stop; active work remains pinned. `revoke-and-kill` is a separate action and reports provider cancellation uncertainty.

Emergency root UDS actions are contraction-only: status, freeze, revoke, block, drop-queued, terminate, disable. They never unblock/resume/enable. Clearing emergency state requires normal approved publication.

## Policy model

Each client key binds exactly one complete immutable policy revision. Several keys may share one policy revision. Personalization clones a named revision and copies no keys.

A KeyPolicyRevision contains:

```text
granted Route Group IDs
exact allowed Provider Account IDs
allowed typed route triggers
queue tier and weight
max concurrency/waiters
deadline and hard limits
```

No runtime selectors, deny lists, metering flags, templates or inheritance are evaluated by Gateway. Labels/selectors/templates are Control authoring aids only; the compiler materializes exact immutable Account IDs into a new policy revision.

Publishing a new shared-policy revision moves only the exact previewed key IDs whose expected current bindings still match. Membership changes invalidate the preview.

## Provider Accounts

ProviderAccount represents one real shared upstream quota boundary—not a vendor brand.

Identity:

```text
immutable Account UUID
LiteLLM connection
provider token
stable non-secret account binding reference
credential-generation attestation
```

`(connection, provider token, binding reference)` is globally unique across live and tombstoned accounts. Recreating an account cannot reset capacity state.

Every enforced dimension must be represented explicitly:

```text
parallel limit
rolling start/RPM limit
TPM/token window
monthly/quota/reset
```

Unknown never means unlimited. Conservative unknown onboarding remains DRAFT with max-in-flight one until all dimensions are measured or explicitly attested absent.

Before possible provider send, Gateway durably reserves:

- one in-flight lease;
- one rolling start;
- input + maximum output + maximum reasoning token charge;
- configured quota units.

Authoritative completion reconciles upper bound to actual usage. Ambiguous completion retains the complete upper-bound debit until reset.

## LiteLLM deployment generations

Deployment generation is a semantic fingerprint:

```text
dg1_<SHA-256(RFC8785 canonical semantic projection)>
```

It covers adapter contract, connection, hidden alias, provider/model/API origin/path/version/region/project, all noncredential execution fields, capabilities/context/defaults/pricing/account binding and credential attestation. Volatile runtime IDs, timestamps, health and usage are excluded.

Same semantics with a new runtime ID remains the same generation. Semantic change creates a new discovered-unassigned generation and makes the previous one drifted/ineligible. Missing deployment becomes ineligible. Partial discovery failure changes nothing.

The certified LiteLLM `llmmaxxing_guard` runs before provider I/O. It atomically verifies expected alias, deployment generation, backend manifest, account binding and HMAC fingerprint of the resolved provider credential. Secret swaps with unchanged metadata therefore fail before a prompt leaves LiteLLM. Response deployment receipt is a second check.

If an endpoint cannot prove pre-call guard and authoritative receipt under the certified LiteLLM contract, that endpoint is unsupported.

## LiteLLM compatibility

Each LLMMaxxing release ships a signed `compatibility.yaml` entry for exact LiteLLM versions—never a guessed semver range. V1 initially supports only the exact certified 1.98.0 build.

The support entry declares:

- exact version/build probe;
- authoritative discovery methods, paths, auth and pagination;
- required/optional fields and default normalization;
- unknown execution-field rejection;
- retry/fallback/cooldown authority checks;
- request-side guard fields;
- response receipt per endpoint;
- normalized error envelopes;
- guard callback digest/order.

Gateway owns separate inference and discovery keys. Control holds neither.

## Discovery and onboarding

A complete discovery poll is all-or-nothing and contains no credential material. New deployments begin:

```text
DISCOVERED_UNASSIGNED
```

with zero traffic.

Orthogonal state axes:

- presence: PRESENT, MISSING, DRIFTED, UNKNOWN;
- assignment: UNASSIGNED, ACCOUNT_BOUND;
- evidence: PENDING, FRESH, REVALIDATION_DUE, FAILED, INVALIDATED;
- approval: UNREVIEWED, APPROVED, RETIRED;
- applied use: INACTIVE, ACTIVE.

Derived flow:

```text
DISCOVERED_UNASSIGNED
→ EVIDENCE_REQUIRED
→ REVIEW_READY
→ APPROVED_NOT_PUBLISHED
→ ACTIVE
```

Every provider-calling qualification step is explicitly labeled LIVE PROBE and shows target, endpoint/request count, token envelope, metering, privacy/region and fixture hash. Discovery, compilation, preview and simulation are labeled NO PROVIDER TRAFFIC.

Evidence binds exact deployment generation, Account ID, adapter/probe-suite version, artifact digest, assertions, provenance and expiry. Promotion requires exact dispatch/receipt, Route Group capability/context/tool/schema/history tests, account capacity, classifier evidence and cross-leg continuation where stateful history can move providers.

Default evidence freshness is 30 days; warning at 23. Expiry blocks expansion/new modes/capacity increase, not unchanged active use. Fresh failed canary or semantic drift blocks immediately.

## Route Groups and triggers

Every client-visible model is a Route Group, even with one local leg.

Example DSV4:

```text
nan DSV4       order 10 · PRIMARY/CAPACITY_SPILL
ArliAI DSV4    order 20 · CAPACITY_SPILL/FAILURE_FALLBACK
Electron DSV4  order 30 · CAPACITY_SPILL/FAILURE_FALLBACK
```

V1 route triggers:

- PRIMARY: sole normal initial leg;
- CAPACITY_SPILL: only capacity/start-rate unavailable or classified capacity exhaustion;
- FAILURE_FALLBACK: only classified pre-byte transient failure;
- QUOTA_FALLBACK: only known/classified quota exhaustion;
- MANUAL_EMERGENCY: signed expiring activation, maximum one hour;
- SHADOW: non-serving, never queues/retries/delays/changes client result.

Modes do not manufacture causes. Capacity cannot become failure. Multiple classifier matches become UNKNOWN. Unknown 403/429 never triggers fallback.

V1 accepts only `ordered_capacity`, ordered by integer order then stable leg ID. Weighted/adaptive values fail schema validation.

## Admission and fair queue

At admission, a request captures its immutable authorization/QoS ceiling:

```text
key and credential generation
policy/bundle revision
Route Group
exact leg/deployment/account/mode set
request profile/data boundary
maximum tier/weight/deadline
```

At every queue wake or pre-byte alternate:

```text
eligible = admission ceiling
         ∩ current authorization minus deny overlay
         ∩ current operational availability
```

Expansions affect only new requests. Contractions/revocations affect queued work immediately. Dispatched attempts remain pinned unless explicitly killed.

Scheduler is hierarchical weighted deficit round robin:

```text
tier → scarce/flexible class → key → FIFO request
```

- weights are integers 1–64;
- deficits persist and are capped at 8× quantum;
- ineligible scans cost no deficit;
- stable FIFO tie-break;
- scarce means structural ceiling has one Account; flexible means more than one;
- flexible traffic cannot repeatedly steal a scarce-only Account from an eligible scarce waiter;
- bounded aging forces lower classes after a finite grant count;
- activation never raises existing waiter QoS or resets deficits.

## Deterministic routing

Each dispatch:

1. validates key/deadline/current authorization freshness;
2. intersects admission ceiling with current policy/deny state;
3. validates endpoint/modality/context/output/reasoning/tools/schema/history;
4. enables legs only for the typed current cause;
5. requires exact generation present, account active, evidence valid and guard available;
6. samples capacity/RPM/quota/circuit state;
7. atomically acquires all account/upstream permits;
8. persists attempt/quota reservation;
9. dispatches first leaseable candidate;
10. reconciles exact deployment receipt before client bytes.

Provider attempts are bounded: one send per distinct generation plus one named same-generation half-open capacity probe; maximum three sends total. No replay after response bytes, unknown overload or identity mismatch.

Terminal outcomes are closed and typed:

```text
AUTHZ_DENIED
AUTH_STATE_UNAVAILABLE
UNSUPPORTED_REQUEST
BACKPRESSURE_REJECTED
ROUTE_UNAVAILABLE
DEADLINE_EXCEEDED
UPSTREAM_FAILED
CLIENT_CANCELLED
RESPONSE_STREAM_FAILED
COMPLETED
```

QUEUED is nonterminal.

## Immutable activation

One global activation stream, one applied bundle and one nonterminal activation at a time. Generation is strictly increasing and never reused. Rollback is a higher generation targeting previous content.

Bundle is canonical RFC8785 JSON with domain-separated content hash. Impact hash binds base/target, exact key/policy/account/leg/generation/mode changes, backend manifest and affected key count.

Publication preview freezes:

```text
base applied generation/hash
publication/security/key-set/authoring fences
exact affected keys
content and impact hashes
LiteLLM manifest/evidence revisions
```

Any relevant change invalidates it.

State machine:

```text
PREPARING_BACKEND
→ BACKEND_READY
→ STAGING_GATEWAY
→ GATEWAY_STAGED
→ COMMITTING
→ APPLIED
```

Only Gateway's fsynced local activation record is enforcement truth and linearization point. PostgreSQL/SQLite applied state reconciles the exact acknowledgement.

PostgreSQL uses SERIALIZABLE singleton head/outbox/fence. SQLite uses BEGIN IMMEDIATE, WAL and synchronous FULL. Signed-file mode uses expected-base hash plus durable signer/Gateway epoch. Every mode has anti-rollback generation, idempotency, stage/commit/ACK and rollback-as-new-generation.

Normal and emergency deny overlays are FULL-synced before enforcement ACK and loaded before readiness. Overlay corruption becomes deny-all RECOVERY_REQUIRED.

## Crash recovery

Before provider send, Gateway durably reserves attempt, account lease, RPM/token/quota charge, circuit epoch/probe, bundle/fence/boot identity and deadline. No prompt is journaled.

Hard crash:

- queued sockets/bodies are not replayed;
- RPM/token/quota reservations are restored conservatively;
- nonterminal attempts become uncertain and continue counting capacity;
- circuits remain open; crashed half-open probe advances epoch;
- capacity releases only from authoritative provider observation or serialized classified probe;
- missing/corrupt runtime state enters RECOVERY_REQUIRED—never reset to zero.

RECOVERY_REQUIRED exit proves old Gateway/backend fencing, imports signed bundle/deny floors, advances installation/fence, quarantines every account at full capacity/RPM and releases only through authoritative observation/probe.

Mutable account and deployment runtime state lives outside policy snapshots and never rewinds on publish/rollback.

## Request path and resource bounds

Named reference profile: `launch-2cpu-2g`.

```text
Gateway:          2 vCPU, 2 GiB, 256 PIDs, nofile 8192
Inbound:          384 total; 64 pre-auth
Body readers:     16 global; 4/key
Body:             32 MiB/request
Retained bodies:  256 MiB global; 64 MiB/key
Waiters:          64 global; 16/key
Active attempts:  128 foreground ceiling plus account limits
Stream buffering: ≤128 KiB/stream
Bundle:           16 MiB; bounded object counts and retained indexes
Journal:          bounded segmented/checkpointed 2 GiB volume
Telemetry spool:  minimum 8 GiB at 250 rps/10-minute outage
Deadline:         7200 s default; 9000 s hard maximum
```

Profiling uses streaming/hard-capped parsing with depth 64, 100,000 aggregate elements, 8 MiB individual strings and bounded tool/schema counts. Two profile workers have memory/CPU caps. Large multipart parts use quota-limited mode-0600 spool.

One global upstream-attempt semaphore is authoritative. HTTPX pool is 160; permits are 144 with foreground/recovery/discovery/qualification reservations. Shadows use only immediate spare permits.

Raw streaming uses explicit ASGI loop and HTTPX `aiter_raw(64 KiB)`, awaits downstream send before next read, strips framing/hop headers, verifies deployment before response start and closes upstream on every path. Post-send ambiguity retains uncertain capacity/quota charge. No post-byte replay.

## Performance contract

Provisional reference gates:

| Measure | Gate |
|---|---:|
| key parse/HMAC/policy lookup | p99 ≤0.75 ms |
| entry→auth at 250 rps | p99 ≤2 ms |
| 64 KiB profile | p99 ≤20 ms |
| available slot→provider send including journal | p99 ≤10 ms |
| raw chunk relay | p99 ≤20 ms |
| nonqueued local overhead ≤64 KiB | p99 ≤30 ms |
| event-loop lag | p99 ≤10 ms |
| Gateway-caused availability | ≥99.9% |

Queue wait, LiteLLM time, provider TTFT and generation duration are separate SLIs.

A release-pinned benchmark manifest fixes hardware/kernel/runtime/storage, seeded body/stream/fault matrices, timestamp definitions, availability denominator and resource thresholds. Tests use constant-arrival k6, deterministic fake LiteLLM/provider and raw HTTPX stream probe. Nominal, breakpoint, 128-stream, exact-bound, sink/disk/network/backend failure, hostile-cardinality and 12-hour soak cases must pass.

## Telemetry, privacy and audit

Inference events are metadata-only. Accepted requests reserve lifecycle event capacity before body read. Hot path constructs bounded typed objects and `put_nowait`; dedicated writer batches/fdatasyncs/replays segments. Spool is delivery backlog, not retention, and is sized from rate × events × outage window.

Prometheus excludes key/request/deployment-generation IDs and arbitrary model strings. Hard cap: 10,000 series and 5 MiB scrape. High-cardinality identity remains lifecycle storage.

LLMMaxxing does not store prompts, responses, tool arguments/results, credentials, raw errors or IPs. Defaults:

```text
central lifecycle metadata: 30 days
OTel metadata:              7 days
security audit:             400 days
```

The homelab's existing downstream LiteLLM/Langfuse content policy remains an explicit deployment choice outside LLMMaxxing.

Security audit is transactional and separate from inference telemetry. Terminal security mutations also enter a signed off-database tombstone ledger. Incomplete restore suspends all restored identities rather than resurrecting authority.

## Admin UX

One versioned Control API and reason-code schema serve UI and CLI.

Navigation:

1. Overview
2. Routing
   - discovery;
   - Provider Accounts;
   - client models/Route Groups;
   - qualification.
3. Access
   - policies;
   - client keys.
4. Changes
   - drafts;
   - immutable previews;
   - approval;
   - activation;
   - rollback;
   - audit.
5. Operations
   - requests;
   - queue/capacity/circuits;
   - reconciliation;
   - contractions/emergency status.
6. Settings
   - LiteLLM compatibility;
   - Gateway/Control/storage/trust.

Only Gateway-ACKed `Applied` is effective. UI always distinguishes desired, applied, backend dependency and deny-overlay state. No success toast before exact enforcement acknowledgement.

Every request detail shows:

```text
received
authenticated
profile accepted
queued
lease granted
dispatched
headers
first byte/token
final byte
lease released
```

Queue wait, TTFT and generation are separate. Every candidate leg has a reason tree covering static authorization, evidence, route trigger, admission ceiling/current authorization, presence, capacity/circuit and selection result.

Direct client hidden alias returns `model_not_client_visible` with zero backend call. A canonical request whose Gateway rewrite is rejected by LiteLLM returns operator diagnosis `backend_route_grant_mismatch`; repair is an exact reviewed manifest, never wildcard access.

Functional target is WCAG 2.2 AA with keyboard operation, textual states/reasons, server-paginated immutable impact sets, complete export and 500-key screen-reader test.

## Backup and recovery

Control-backed backups include:

- Control DB/audit/outbox;
- encrypted identity/pepper/policy/audit/channel secret set;
- Gateway active/staged/previous records;
- deny overlays, fence/device identity;
- attempt/RPM/token/quota/circuit journal;
- signed terminal ledger/checkpoint;
- LiteLLM service-key fingerprints.

Gateway-only has its own complete manifest: authoring source, reviewed LiteLLM manifest, signed bundles, signer head/epoch, Gateway status receipts, key verifiers/tombstones/deny floors, runtime journal and separate encrypted signer/pepper/service-key secret set. It never requires nonexistent Control artifacts.

Scratch restore must prove identity/signing continuity, anti-rollback, client auth or deliberate all-key suspension, service-key fencing, conservative account recovery and one controlled inference.

If surviving Gateway/terminal ledger cannot prove state newer than a backup, restore starts deny-all, purges sessions/approvals/recovery artifacts, suspends all client keys, rotates both LiteLLM service keys, quarantines accounts and requires re-enrollment/reissue.

## Upgrade and support

- patch: bug/security fixes, no DB/schema semantic change;
- minor: additive API/config/expand-only migration/features;
- major: breaking changes with explicit migrator/runbook.

Gateway local state supports current and previous minor readers. Feature activation advances a `min_reader_floor`; binary downgrade is blocked unless current binary first applies previous-compatible bundle/state. No reverse SQL.

V1 supports exact certified LiteLLM 1.98.0 only. Additional versions require contract fixtures and a new LLMMaxxing release.

## Legacy migration

1. Freeze/inventory current admission/key/LiteLLM state and verify backups.
2. Import current policies/deployments into an unapplied draft; fail conflicts/unresolved generations.
3. Provision narrow Gateway LiteLLM inference/discovery keys.
4. Import legacy client token from stdin, self-verify once, store only HMAC.
5. Issue replacement `lmxk1` keys.
6. Shadow decisions/canaries without mirroring ordinary provider traffic.
7. Move clients onto one stable front-door domain.
8. Drain old dispatcher and seed RPM/uncertain state.
9. Flip once—never two production dispatchers.
10. Exercise direct-LiteLLM rollback while every client still uses legacy credentials.
11. At first `lmxk1` adoption, rollback target becomes previous verified LLMMaxxing image/bundle.
12. Within 30 days and after zero legacy use, revoke legacy caller keys while preserving/reproving exact Gateway service keys.

## Release scope

### V1

- Gateway extraction;
- local client keys;
- ordered Route Groups;
- shared-account fair queue;
- LiteLLM generation guard/discovery;
- onboarding, policies, key lifecycle and request explorer;
- immutable bundles, rollback and emergency contractions;
- three deployment modes;
- backup/restore/migration/performance suite;
- exact LiteLLM 1.98.0 support.

### V1.x after evidence

- additional certified LiteLLM versions/platforms;
- deterministic weighted routing;
- multiple Control replicas.

### Later/major

- adaptive routing;
- distributed Gateway scheduling;
- multiple backends/direct providers;
- plugins/SaaS/Kubernetes operator.

## Acceptance gates

Design received fresh GO verdicts for trust/state, data plane and product/recovery after all adversarial P0/P1 corrections.

Implementation is not complete until contract tests prove:

- local auth, expiry, rotation, revocation and ten-second deny freshness;
- exact immutable policy/account membership and queue authorization ceiling;
- fair scheduler no-starvation and deterministic replay;
- one-to-one LiteLLM alias/generation/credential guard before provider I/O;
- account concurrency/RPM/TPM/quota reservation and ambiguous completion;
- typed route triggers and maximum three attempts;
- atomic publication, concurrent conflict, lost ACK, rollback and deny overlay;
- singleton/crash/quarantine/recovery behavior;
- parser/bundle/journal/spool/metrics/pool resource bounds;
- raw streaming/cancellation/no post-byte replay;
- desired/applied UX, live-probe labeling, key lifecycle and hidden-alias diagnosis;
- all three modes' backup and restore;
- certified benchmark/load/failure/soak results;
- clean-cut legacy migration and exercised rollback.

_Last updated: 2026-08-30_
