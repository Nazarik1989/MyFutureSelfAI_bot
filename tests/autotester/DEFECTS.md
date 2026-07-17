# Resolved routing defects from the deterministic scenario matrix

The 70-scenario matrix introduced in PR #3 found three application-routing
defects. They were first reproduced as 14 strict `xfail` scenarios, then fixed
in a separate application-code PR. All 14 scenarios now run as ordinary passing
tests through the real text and voice entry points.

## AUTOTEST-D001 — filler and politeness broke focused-draft save routing

Previous behavior: the exact save predicate rejected bounded filler or
politeness, so the command reached Intent Router and could become a new preview.

Resolved behavior: a conservative token grammar accepts a maximum of two known
leading fillers and one bounded `пожалуйста`, while still requiring an exact
save verb and inbox target.

Passing variants:

- `Ну сохрани в инбокс`
- `Пожалуйста, сохрани в inbox`
- `Сохрани это в инбокс, пожалуйста`
- `Короче сохраним это в инбокс`
- `Эээ сохрани в инбокс`
- `Сохрани, пожалуйста, это в inbox`

## AUTOTEST-D002 — clipped transcription was treated as inbox content

Previous behavior: short command-shaped transcriptions reached Intent Router
and could replace the effective focus with an erroneous preview.

Resolved behavior: the explicitly supported clipped forms route as a save
action. One focused draft is saved; no draft returns the existing missing-draft
response; multiple drafts use the existing clarification flow.

Passing variants:

- `Это в инбокс`
- `Это в inbox`
- `В инбокс`
- `Сохрани в инбок`

## AUTOTEST-D003 — extended negative commands reached the LLM

Previous behavior: adding `это`, politeness, or the supported clipped inbox
target prevented the negative command from matching. The original draft stayed
active and a command-preview could be created.

Resolved behavior: a separate negative grammar recognizes only bounded
command-shaped forms and routes them to the existing discard action.

Passing variants:

- `Не сохраняй это в инбокс`
- `Пожалуйста, не сохраняй в inbox`
- `Не надо это сохранять в инбокс`
- `Не сохраняй в инбок`

## False-positive protection

The matrix still sends ordinary sentences containing `сохранить`, `сохранять`,
`инбокс`, or `inbox` to Intent Router. The fix does not use open-ended substring
matching and does not broaden read-only natural-command patterns.
