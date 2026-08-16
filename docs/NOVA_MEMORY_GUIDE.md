# My Nova memory

Stage 7B.1 exposes the owner-scoped Nova memory domain through an explicit Telegram
CRUD flow. Stage 7C can apply a bounded projection of those confirmed records to final
AI answers. CRUD visibility and answer personalization remain independently gated and
fail closed.

## Access and entry points

The full-width `🧬 Моя Nova` row appears in the main menu only when
`ENABLE_NOVA_MEMORY=true` and the current access tier is allowed by
`NOVA_MEMORY_ADMIN_ONLY`. `/mynova` is a recovery command and is intentionally absent
from the compact native Telegram command menu. Guest, blocked, missing and unauthorized
actors cannot open, read or mutate memory. An administrator still sees only their own
records.

Answer personalization is effective only when both `ENABLE_NOVA_MEMORY=true` and
`ENABLE_NOVA_MEMORY_APPLICATION=true`, the actor is a current subscriber or admin, and
the actor passes both `NOVA_MEMORY_ADMIN_ONLY` and
`NOVA_MEMORY_APPLICATION_ADMIN_ONLY`. Root and help screens report that effective state
as `Персонализация AI-ответов: включена` or `выключена`; they do not imply that merely
having CRUD access enables prompt application.

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
separate retention policy. A record otherwise remains until explicit edit, item/delete-all
deletion, or owner cascade. Losing access preserves the records but stops their
application. Marking a record important does not change its retention. The conversation
purge service is still not registered as a runtime job.

## Safe application to answers

Memory is never passed to intent routing, classification, action selection, reminders,
timezone resolution, system or onboarding flows, Tasks, workspaces, collections,
Knowledge, Vision, guest demos, image generation, health flows, or guided Nova help.
Text and recognized voice use the same lifecycle: deterministic and durable consumers
run first, routing remains memory-blind, and only a final `conversation` or `question`
answer may use confirmed memory.

The application snapshot is owner-only and bound to the exact actor tier and access
version. A stable whole-collection revision fences the projection before and after the AI
request and immediately before and after Telegram delivery. Access loss discards the
answer. Any create, edit, importance change, item delete, or delete-all discards it as
stale. After Telegram accepts a stale answer, the bot deletes that exact message or, if
deletion fails, edits only that message to a neutral notice. It never retries the provider
or repeats domain DML.

The projection contains no more than 12 whole records and no more than 8 KiB of the exact
compact JSON sent to the configured AI provider. Important records receive selection
priority, but `Важное` does not require the model to mention them. Selection also preserves
category coverage when room remains. The provider receives only `category`, `important`,
and `content`; it receives no record ID, revision, fingerprint, Telegram ID, timestamp,
version, or audit metadata. Confirmed memory is untrusted user data, never instructions,
and cannot alter system rules, tools, routing, roles, or actions.

A personalized assistant answer may remain in the local ConversationContext until its
normal TTL expires. It is tagged internally and is excluded from every later provider
`conversation_context`, while the local snapshot remains available to existing local
operations. Consequently deleting records, disabling application, or losing access cannot
reintroduce old memory indirectly through an earlier assistant answer; every personalized
request can use only its fresh, bounded, revision-fenced projection.

Telegram delivery and its final access/revision fence run as one tracked operation. During
shutdown the bot performs a bounded wait for those delivery or compensation operations
before continuing teardown, so an accepted answer can still be fenced or neutralized.
Provider-side retention remains governed by the actually configured provider, account, and
contract.

Users can see, edit, and delete every active record in `🧬 Моя Nova`. When personalization
is enabled, only the bounded projection is sent to the AI provider configured for text
answers. Provider-side retention and training behavior depend on that actual provider,
account, and contract. This project does not claim Zero Data Retention unless the deployed
provider/account configuration independently guarantees it.
