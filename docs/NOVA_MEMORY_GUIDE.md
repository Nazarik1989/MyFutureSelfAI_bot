# My Nova memory UI

Stage 7B.1 exposes the owner-scoped Nova memory domain through an explicit Telegram
CRUD flow. It does not apply memory to AI prompts; that remains independently disabled
by `ENABLE_NOVA_MEMORY_APPLICATION=false` until Stage 7C.

## Access and entry points

The full-width `🧬 Моя Nova` row appears in the main menu only when
`ENABLE_NOVA_MEMORY=true` and the current access tier is allowed by
`NOVA_MEMORY_ADMIN_ONLY`. `/mynova` is a recovery command and is intentionally absent
from the compact native Telegram command menu. Guest, blocked, missing and unauthorized
actors cannot open, read or mutate memory. An administrator still sees only their own
records.

The feature accepts exact commands at the beginning of a text or recognized voice
message, including `Nova, запомни: …`, `Научи Nova: …`, `Nova, запомни это`,
`Nova, забудь всё` and requests to open My Nova. Similar narrative phrases continue to
the existing reminder, navigation, guided-help or generic pipeline. The classifier is
local and deterministic; no AI provider is used.

## Confirmation and screens

Creating or editing a record always goes through a preview. Before the user presses the
confirmation button, no Nova-memory row or audit row is written. The preview lets the
user choose one of three categories and, for a new record, mark it important. Existing
records can be browsed five at a time, edited, marked important, hard-deleted, or deleted
as an exact revision-fenced collection.

`Nova, запомни это` considers only the latest bounded user reply in the current
conversation context. It never searches past an unsuitable latest user reply and never
uses assistant, system, navigation, error or control content.

## Session and callback safety

The Telegram flow has its own process-local store, separate from guided Nova help. It is
bounded to 128 sessions, expires after 15 minutes, and binds every screen to the exact
owner, Telegram actor, private chat, access tier/version, session generation and
canonical message. Candidate content is temporary and excluded from representations.
Restarting the process invalidates every flow and callback.

After entry, callback data contains only an opaque `nmem:` capability. Server-side
capabilities carry item or collection fences and are invalidated on every screen
transition. Mutation capabilities are single-use, so replay or concurrent double-click
cannot call the domain twice. Wrong-owner, wrong-chat and wrong-canonical attempts do not
consume the legitimate actor's capability.

All screens edit one canonical bot message. The first explicit voice command reuses the
STT progress message; voice input in an existing flow edits the old canonical and retires
the transient progress best-effort. Photo and document input is contained while text or
voice content is awaited. Telegram edit failure never falls back to another reply and
never repeats a successful domain mutation.

## Privacy and lifecycle

The flow stores no raw command, full STT transcript, audio bytes or provider output.
Logs contain operation identifiers and exception types only. User content is rendered
with `parse_mode=None`, and opaque callbacks contain no content, record IDs, versions,
categories, pages or Telegram IDs.

Fresh access generation is checked at entry, after STT, around reads, at capability
claim, and before and after mutation. A downgrade or version bounce clears only the
exact old flow and replaces private content with a neutral access-changed screen. If a
mutation already committed, its domain result remains authoritative but is not disclosed
until the actor is authorized again.

Confirmed items remain independent of ordinary conversation retention. Deletion is a
hard delete from the active database; backups can retain older copies according to their
separate retention policy. The conversation purge service is still not registered as a
runtime job in Stage 7B.1.
