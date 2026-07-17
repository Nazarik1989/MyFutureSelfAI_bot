# Defects found by the deterministic scenario matrix

This report contains application defects reproduced by `tests/autotester`.
Application code is intentionally not changed in PR #3. Each listed case is a
strict `xfail`: it remains visible in the full suite, and an unexpected fix
becomes an `XPASS(strict)` failure until this report and the marker are updated.

## AUTOTEST-D001 — filler and politeness break focused-draft save routing

Expected: with one focused draft, the command is handled before Intent Router,
saves that draft once, and never creates another preview.

Observed: the command reaches the LLM route because the save predicate requires
an exact normalized phrase. A note-like preview can be created from the command
instead of saving the focused draft.

Reproduced through alternating text and voice entry points:

- `Ну сохрани в инбокс`
- `Пожалуйста, сохрани в inbox`
- `Сохрани это в инбокс, пожалуйста`
- `Короче сохраним это в инбокс`
- `Эээ сохрани в инбокс`
- `Сохрани, пожалуйста, это в inbox`

Impact: the original focused card remains unsaved, and a later valid save
command can target the erroneous command-preview.

## AUTOTEST-D002 — clipped transcription is treated as inbox content

Expected in these scenarios: when exactly one focused draft exists, the short
command-shaped transcription saves that draft. With no draft or ambiguous
drafts, a future fix should clarify without invoking the LLM or creating a
preview.

Observed: clipped variants reach Intent Router and can become new inbox
previews.

Reproduced through alternating text and voice entry points:

- `Это в инбокс`
- `Это в inbox`
- `В инбокс`
- `Сохрани в инбок`

Impact: a common lost-word or truncated voice transcription can replace the
effective focus with an erroneous preview.

## AUTOTEST-D003 — extended negative commands reach the LLM

Expected: negative control language never saves anything, never invokes the
LLM, and discards the focused draft when the instruction is unambiguous.

Observed: only the exact negative phrases are recognized. Adding `это`,
politeness, or a clipped final word sends the phrase to Intent Router and can
create a preview while leaving the original draft active.

Reproduced through alternating text and voice entry points:

- `Не сохраняй это в инбокс`
- `Пожалуйста, не сохраняй в inbox`
- `Не надо это сохранять в инбокс`
- `Не сохраняй в инбок`

Impact: the user's explicit negative instruction is not honored
deterministically and can generate additional state.

## Suggested follow-up scope

Fix these defects in a separate application-code PR. Keep the parser
conservative: accept bounded filler/politeness and explicitly supported clipped
forms only when draft context makes the action safe; do not turn arbitrary
sentences containing `сохранить` or `инбокс` into control commands.
