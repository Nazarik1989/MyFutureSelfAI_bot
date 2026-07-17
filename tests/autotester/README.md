# Isolated bot autotester

`tests/autotester` runs declarative end-to-end scenarios through the real
`FutureSelfBot.text`, `FutureSelfBot.voice`, and callback handlers.

The harness deliberately uses:

- a SQLite file created inside pytest's `tmp_path`;
- sentinel credentials and non-routable `.autotest` provider URLs;
- deterministic fake AI and transcription services;
- fake Telegram updates, messages, media, and callbacks;
- an autouse network blocker for this directory.

Every LLM input must be explicitly stubbed. An unexpected LLM call fails the
scenario instead of silently producing a fake response.

The catalog contains 90+ deterministic E2E scenarios. Variations are
generated from fixed tuples only; no randomness, wall-clock input, or external
data is used. Regressions found during matrix expansion and their resolved
coverage are documented in `DEFECTS.md`.

Run only the scenario harness:

```bash
pytest -q -m autotester
```

Run it together with the full regression suite:

```bash
pytest -q
```

The harness must never load the production `.env`, connect to a production
database, or perform real Telegram/provider network requests.
