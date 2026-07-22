# ADR-0002: Access and workspace foundation

- Status: Accepted
- Date: 2026-07-22
- Scope: PR #23 only
- Depends on: ADR-0001 and production hardening from PR #22

## Context

`LifeCollection` ("Мои разделы") is a personal organizer. It is not an authorization
scope and must not be reused for collaboration. Later Knowledge phases need a single,
database-authoritative access boundary before they can store or retrieve shared data.
Health, Doctor and Labs must remain private regardless of workspace role.

Telegram cannot safely resolve an arbitrary name or `@username` to a bot user. Shared
links can also be forwarded, so an invitation is not proof of identity until the
recipient explicitly accepts it.

## Decision

1. Add independent `Workspace`, `WorkspaceMember`, `WorkspaceInvitation`,
   `WorkspaceProject` and `KnowledgeSpace` models. `WorkspaceProject` is deliberately
   distinct from `LifeCollection(kind="project")`. `WorkspaceContext` and
   `WorkspaceActionToken` persist actor/chat-bound context, callbacks and text-input
   state across process restarts.
2. Add only additive schema in Alembic revision `20260722_0018`. Migrations create no
   users, memberships, workspaces, projects or personal spaces and perform no content
   backfill.
3. User-facing reads and mutations use an immutable `AccessContext`. Repository queries
   join active membership, workspace status and project status before returning an
   object. A caller cannot fetch by a bare object ID and filter afterward.
4. Roles are `owner`, `editor` and `viewer`. Owners manage access. Owners and editors
   manage workspace projects. Viewers read only. The final active owner cannot leave,
   be revoked or be demoted without another owner already present.
5. Membership revoke/leave is an atomic state transition that increments
   `Workspace.access_epoch`. Persistent action capabilities and active contexts contain
   the epoch and relevant optimistic versions; an old button or context therefore fails
   immediately after access changes.
6. Direct invitations bind to one already-resolved internal `User`. They never degrade
   to an unbound invitation. Share invitations use a cryptographically random,
   single-use, expiring token; only a hash is stored. Accept and revoke compete through
   conditional database transitions, so at most one terminal result wins. A bearer
   invite created before a recipient's revoke/leave cannot re-activate that membership;
   re-entry requires a newer owner-issued invite. Pending invites are revoked when
   their issuing owner loses owner rights and when the workspace is archived.
7. Workspace character (`pair`, `friends`, `family`, `team`, `custom`) changes only
   presentation and invitation wording. It grants no permission and shares no personal
   record automatically. Dynamic text is normalized, bounded and safely rendered.
8. Creating a workspace or workspace project creates its matching `KnowledgeSpace`
   atomically. A personal `KnowledgeSpace`, when needed later, is created lazily and
   idempotently. PR #23 stores no sources, assets, chunks or embeddings.
9. The independent `ENABLE_WORKSPACE_ACCESS` flag is false by default. Disabled UI is
   absent from the native command catalog and navigation. Production may enable only
   this foundation while every Capture, Runner, Retrieval, Embedding, OCR, Media,
   External Vision, Council, Schedule and Export flag stays false.

## Telegram contract

- `/spaces` is the canonical entry point; `/workspaces` is an alias.
- Deterministic text and post-transcription routes handle workspace navigation and CRUD
  without LLM calls.
- Persistent callbacks are actor/chat-bound, versioned, expiring and single-use.
- Pending text input uses the same persistent actor/chat/epoch/version boundary; it is
  not stored only in Telegram framework memory and `/cancel` consumes it explicitly.
- The only PR #23 workspace surfaces are members, invitations and projects. Buttons for
  future shared content do not exist.
- Invitation previews always include the privacy notice that personal records and
  visualizations are not shared automatically.

## Isolation

The access schema has no foreign key, polymorphic reference or service dependency to
Health, Doctor or Labs. Workspace roles cannot address those tables. Personal Vision,
Inbox, Tasks and LifeCollections also remain private until a later, explicit sharing
design is reviewed.

## Consequences and rollback

The feature can be disabled without removing data. Once invitations or workspaces exist,
rollback retains revision `20260722_0018`; the previous image may run against the
additive schema. Destructive downgrade is allowed only on an isolated pre-activation
test database, or after restoring a coordinated backup that predates workspace data.

Production continues to use the explicit invariant
`sqlite+aiosqlite:////data/future_self.db` and all PR #22 container controls. Deployment
requires a verified SQLite backup and a stopped previous container before cutover.

## Deferred work

PR #23 does not implement material capture, PDF/image handling for Knowledge, ingestion
jobs, OCR, retrieval, embeddings, cards, Council, publication or scheduled Knowledge
work. Those remain separate PRs #24-#28.
