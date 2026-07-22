# ADR-0001: Knowledge Hub and Council foundation

- Status: Accepted
- Date: 2026-07-22
- Baseline: `main@3bcaf05f04b2b9f32000ed8a8d8c898c61e49b9b`
- Scope: PR #22 only

## Context

MyFutureSelfAI is a modular monolith: one Telegram application, owner-scoped domain
services, async SQLAlchemy, Alembic, persistent capabilities and a durable reminder
outbox. The current production database is SQLite. Health, Doctor and Labs form a
closed private domain and do not send their records to LLM services.

The future Knowledge Hub and Council need an explicit security and operational
foundation before any access, capture, retrieval or orchestration code is added.
The source architecture document was reviewed in full and its eight recommended
defaults are accepted by the project owner.

## Decision

### Deployment and domain shape

1. Keep a modular monolith. Do not add microservices, Redis or Celery for the MVP.
2. Access will be a DB-authoritative service. Future Knowledge code will receive an
   immutable `AccessContext`; Council will receive an authorized `EvidencePack`.
3. Health, Doctor and Labs remain isolated. No generic polymorphic relation may point
   from Knowledge/Council to personal medical records.
4. Public drafts are publication lifecycle objects, never a fourth ACL scope.
5. Future migrations are expand-only and staged by feature flags. PR #22 adds no
   tables, handlers, workers, commands or user seed.

### Eight approved defaults

1. Keep SQLite for the first Knowledge MVP, with one runner, WAL and a bounded busy
   timeout. Reconsider PostgreSQL at a second worker or more than 50,000 active chunks.
2. A workspace owner will issue a persistent invitation to an existing user. There
   will be no hard-coded Telegram IDs or migration seed.
3. External processing of Personal materials requires separate explicit consent at
   first capture. Local OCR is preferred, external vision is off, and provider payloads
   are minimized.
4. `brief_reminder` is the default apply mode. `silent_apply` can only be a future
   project-level opt-in.
5. Private binary assets will use `/data/knowledge` for the MVP. S3-compatible storage
   requires a separate migration ADR.
6. Council will use two user-selected substantive perspectives, a mandatory
   critic/empiricist and a neutral moderator. Personas are pseudonyms, not real people.
7. Scheduled Council is a future weekly opt-in over approved backlog/saved sources and
   creates only a `PublicationDraft`.
8. The MVP visible library is the Telegram hub plus private HTML export. A web panel
   requires a separate authentication and deployment ADR.

### PR #22 controls

- All Knowledge/Council feature flags are disabled by default and dependency-validated.
- Quotas are finite and validated even though no consumer exists yet.
- Shared image/PDF/subprocess protections live in `future_self.safe_media`; domain
  adapters do not import each other's repositories or private functions.
- The image runs as numeric UID/GID `10001:10001`.
- The production runtime uses a read-only root filesystem, private executable-disabled
  `/tmp`, all Linux capabilities dropped, `no-new-privileges`, bounded pids/memory/CPU,
  and rotated container logs.
- Docker build context is an allowlist that never sends `.env`, databases, backups,
  tests or the Git checkout to the daemon.
- SQLite connections explicitly enforce foreign keys, WAL and busy timeout.

## Explicitly out of scope

- Workspace, membership, invitation, Project or KnowledgeSpace models.
- Capture drafts, asset storage, ingestion jobs or workers.
- OCR, URL fetching, audio/video extraction or external vision.
- Chunks, FTS, embeddings, retrieval, cards, rules or active apply.
- Council personas, sessions, citations, backlog or publication.
- `/knowledge`, `/capture`, `/search`, `/council` or any new Telegram routing.

These belong to PRs #23–#28 and must not be pulled into PR #22.

## Consequences

- PRs #1–#21 keep their command catalog, schema and behavior.
- The non-root container requires the live `/data` directory and SQLite file to be
  owned by UID/GID 10001. Root-owned backup files remain outside application access by
  a nested read-only mount.
- WAL adds `-wal`/`-shm` sidecars while the application is running. Backups must keep
  using `sqlite3.Connection.backup()` rather than copying the database file.
- Disabling future flags is a roll-forward control. Once Knowledge user data exists,
  destructive schema downgrade is not the default rollback.

## Verification

- Existing tests and production-like scenarios must remain green.
- Direct safe-media tests cover MIME spoofing, metadata removal, active/encrypted PDF,
  secret-free subprocess environment, traversal/symlink paths and timeouts.
- Static runtime tests verify the non-root image, strict `.dockerignore`, loopback-only
  demo PostgreSQL and the hardened production command.
- Deployment requires a DB backup, empty/current asset manifest, restore preflight,
  preserved old image/container and a final doctor/log/integrity audit.
