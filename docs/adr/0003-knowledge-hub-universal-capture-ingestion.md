# ADR 0003: Knowledge Hub, explicit Capture, and offline ingestion

- Status: accepted for PR #24
- Date: 2026-07-22
- Supersedes: none
- Extends: ADR 0001 and ADR 0002

## Context

PR #23 established `KnowledgeSpace` as the personal/workspace/project authorization
boundary. PR #24 must accept private source material and extract deterministic text
without turning the Telegram process into a parser, weakening medical isolation, or
creating a second membership system. Production currently uses one SQLite database and
a private `/data` bind mount.

Knowledge is a library of attributed sources. It is not the private memory, identity,
relationship state, or shared "mind" of any current or future agent. Publication,
retrieval, embeddings, OCR, Council, external vision, URL fetching, export, and
cross-repository synchronization remain outside this decision.

## Decision

### Authorization and identity

The existing `KnowledgeSpace` row remains the sole scope. Personal access is ownership;
workspace and project access is derived from the current active `WorkspaceMember` and
workspace `access_epoch`. There is no Knowledge membership table. Every user-facing
read or mutation scopes its SQL query by the actor's current access before materializing
a source. An inaccessible public UUID is reported as not found.

Spaces, sources, revisions, jobs, drafts, and audit events have stable random public
identifiers separate from database row IDs. Callback data contains only a short,
actor-and-chat-bound, expiring capability; the service rechecks ACL and version at claim.

The source belongs to its space after confirmation. `created_by` is provenance, not a
private ownership override. Viewer reads; editor adds/edits/trashes; owner additionally
permanently deletes and administers the existing workspace membership.

### Source and revision lifecycle

`KnowledgeSource` holds title, source type, processing/lifecycle status, semantic role,
bounded priority, publication readiness, system classification, and the current
revision number. `KnowledgeSourceRevision` is immutable source content with SHA-256,
opaque storage keys, declared/detected type, size, extraction result, and provenance.
Changing content always appends a revision; extraction finalization is a compare-and-set
from pending and never rewrites original content.
An explicit retry of a failed or cancelled extraction also appends a new immutable
revision linked to the root original, so the failed revision remains auditable and no
second physical original is silently created.

Roles are `foundation`, `trusted`, `perspective`, `discussion`, `counterpoint`, and
`hypothesis`; priorities are `high`, `normal`, and `low`. Publication state is only a
future-compatible `draft`/`publication_ready` marker and publishes nothing.

Duplicate lookup is scoped to an already-authorized space. SHA-256 is deliberately not
globally unique and PR #24 performs no cross-space physical deduplication, preventing a
cross-tenant existence oracle.

### Explicit, restart-safe Capture

`/capture` or an explicit Hub action creates a persistent actor/chat-bound draft.
Arbitrary text is never automatically stored and Capture is not a fuzzy natural-language
write route. Telegram media may produce only a bounded metadata/reference chooser before
consent; no bytes are downloaded yet. Before confirmation the UI shows type, target,
title, role, priority, and extraction limitations and permits each choice to change.

Confirmation is the only path that reserves quota, downloads bytes, creates a source,
revision, and durable job. The download is chunk-counted and hashed; it does not trust
Telegram's declared size. If the runner is stopped, the committed job remains `queued`.
Duplicate callbacks are consumed atomically.

Specialized onboarding, task, collection, vision, Health, Doctor, and Labs states run
before generic Capture. Active medical flows cannot be turned into Knowledge sources.
Voice is not a Capture format and is rejected before STT/LLM while Capture or a medical
flow is active.

### Storage boundary

Assets live under the dedicated private root `/data/knowledge`:

```text
.staging/<random>.part
originals/<shard>/<shard>/<random>
extracted/<shard>/<shard>/<random>.txt
```

User filenames are display metadata only. Directories are mode `0700`, files `0600`.
The store uses exclusive creation, no-follow opens, regular-file/link checks, actual-byte
limits, SHA-256 while streaming, `fsync`, and same-filesystem atomic publication. It
rejects traversal, absolute keys, symlinks/hardlinks, empty/oversized input, low disk,
MIME/magic mismatch, and unsafe archive expansion. Cleanup and audit report expired
staging, missing references, orphan files, and unsafe entries without document content.

Quota reservations are atomic database records created before streaming and either
committed with the source or released/expired. Limits apply to actor and space; checks
include current immutable assets, worst-case extracted-output headroom, and live
reservations.

### Deterministic extraction boundary

The Telegram process never parses a document. A fixed local subprocess receives a
private copied input and bounded parameters through files, with no shell, inherited
credentials, proxy variables, or network requirement. It accepts:

- UTF-8 TXT and Markdown;
- unencrypted, passive PDF text layers;
- passive DOCX XML;
- passive EPUB XHTML/text.

ZIP entry count, individual and total uncompressed size, compression ratio, page count,
output bytes, CPU/memory, and wall time are bounded. DTD/entities, traversal, links,
macros, scripts, actions, attachments, forms, multimedia, encrypted PDFs, and external
resources are rejected. Images and image-only PDFs finish as `partial`: the original is
safe but no text is claimed. URLs are stored as references and never requested,
redirected, or previewed.

### Durable runner

The runner is a separate entrypoint from the same image and uses a separate minimal
configuration model. It never loads the bot `.env` and defines no Telegram, AI, or STT
secret fields. Its Compose profile has no network namespace and passes only an explicit
database/storage/limits allowlist.

SQLite production runs exactly one worker. The DB queue provides polling, availability,
bounded attempts, exponential backoff with jitter, lease token/expiry, heartbeat, stale
lease recovery, cancellation, and idempotency keys. Claim and result commits are short
transactions; parsing happens outside every transaction. Finalization requires the
current lease token, active source/version, and current revision, so two accidental
runners cannot both commit a result. Logs contain public job IDs, state, attempt, and
exception type or allowlisted error code only.

Workspace membership, role, lifecycle, and project-scope changes append content-free
Knowledge audit events. This keeps authorization history adjacent to source history
without copying membership or private document data into audit metadata.

The parsing subprocess is a defense boundary against accidental parser behavior, not a
claim of kernel-level filesystem isolation from the runner itself. The current same-UID
worker can see the runner's mounted `/data` tree if a native parser is compromised;
network namespace removal, secret-free environment, private copied input, no-shell
launch, no-follow/link checks, and resource limits reduce impact. A separate parser
container/UID with input-only and output-only mounts is the preferred later hardening
step and is deliberately not presented as implemented in PR #24.

### Medical isolation

System medical records remain in their existing Health, Doctor, and Labs tables and are
never offered as Capture input. `health_private` is valid only in a personal
`KnowledgeSpace`, enforced both by service and a database check. UI target manipulation
cannot bypass this rule. Generic previews and audit metadata contain no medical record
content. A user-supplied external file cannot be semantically classified without the
explicitly deferred OCR/LLM features; the UI states that limitation honestly.

### Deletion, backup, and rollback

Trash is reversible. Permanent deletion first marks purge pending and cancels extraction,
then the runner idempotently unlinks all source assets. Only after successful unlink does
the service record completion; failures remain visible as purge failed and retryable.
Editors may trash shared sources, while only owners may permanently delete them.

The backup command creates both a filesystem marker and an SQLite transactional
maintenance fence checked by Capture and runner leasing. Existing leases may finish so
the command can drain them. It then uses `sqlite3.Connection.backup()`, copies no-follow
immutable assets, fsyncs files/directories, and writes a sorted manifest with relative
storage keys, sizes, SHA-256, Alembic head, application SHA, UTC timestamp, and format
version. The offline verifier checks DB integrity/FKs/head, every hash/size against both
manifest and DB references, missing/corrupt/extra assets, private modes, and unsafe
links. Partial backups are never published.

Rollback is additive: stop the runner, disable all three PR #24 flags, and run the prior
application against the additive schema. Do not downgrade the live database or delete
`/data/knowledge`; restore a coordinated DB+asset backup only as an explicit incident
decision. Migration downgrade is for pre-activation development only and never removes
filesystem assets.

## Consequences

- `ENABLE_KNOWLEDGE_HUB`, `ENABLE_KNOWLEDGE_CAPTURE`, and
  `ENABLE_KNOWLEDGE_RUNNER` remain independently off by default.
- Operators must migrate and enable Hub/Capture before enabling the runner and must run
  the worker with the hardened, secret-free, networkless profile.
- A stored scan can be useful and recoverable while honestly remaining `partial` until a
  later, separately consented OCR stage exists.
- SQLite throughput is intentionally bounded to one ingestion worker; this favors clear
  correctness and predictable operations over parallel parsing.
