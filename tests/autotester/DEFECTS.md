# Resolved routing defects from the deterministic scenario matrix

The deterministic matrix introduced in PR #3 found three application-routing
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

## HEALTH-D001 — daily reminder disappeared after a delivery failure

Previous behavior: the health reminder used a one-shot job and scheduled the
next day only after a successful Telegram send. A transient send failure left
the opt-in preference in the database but no active job until process restart.
It also created a race where an in-flight callback could schedule a job again
after `/health_reminder_off`.

Resolved behavior: the existing scheduler owns one timezone-aware recurring
daily job per user. Updating or disabling the preference removes that named job;
delivery success is no longer responsible for scheduling the next occurrence.

## HEALTH-D002 — colloquial red flags were not escalated

Previous behavior: phrases such as `Я задыхаюсь, мне не хватает воздуха` and
`Есть мысли причинить себе вред` were stored but did not receive the urgent
medical-help response.

Resolved behavior: conservative deterministic markers cover these forms while
local negation still prevents escalation for statements such as
`Я не задыхаюсь, дыхание нормальное`. Health text remains outside the LLM and
application logs.

## DOCTOR-D001 — red flags were delayed until the end of visit preparation

Initial implementation collected medications and questions after a red-flag
reason or symptom before showing urgent guidance.

Resolved behavior: the deterministic safety check runs immediately after the
reason and symptom steps. The user is explicitly told not to wait for the form
before seeking urgent help; the final factual summary repeats the guidance.

## DOCTOR-D002 — task idempotency was not concurrency-safe

Initial implementation prevented sequential duplicate appointment tasks but two
concurrent identical commands could both observe an empty task link.

Resolved behavior: task creation and the owner-scoped preparation link use one
transaction with a compare-and-set update. A losing concurrent transaction is
rolled back and returns the already-created generic task and reminder.

## DOCTOR-D003 — structured duration was omitted from prolonged-weakness safety

Initial implementation checked only the reason and symptom fields. If
`несколько недель` was supplied in the dedicated duration answer, the
long-lasting weakness recommendation could be missed.

Resolved behavior: reason, structured duration, and current symptoms are all
included in the deterministic prolonged-weakness safety check.
