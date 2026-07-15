# «Моя будущая версия»

MVP Telegram-ассистента, который связывает образ желаемой жизни с небольшими ежедневными действиями и принимает спонтанные мысли текстом или голосом.

## Что работает

- возобновляемый пошаговый онбординг с возвратом, пропуском необязательных вопросов, отменой и редактированием;
- Vision Profile, 3–5 подтверждаемых целей и максимум три активные рутины;
- общий Intent Router для текста и голоса: вопросы/общение получают ответ, а inbox — preview;
- persistent-контекст последних сообщений отдельно для каждого пользователя и чата, с TTL и ссылкой на активный draft;
- детерминированная проверка относительных дат, дат без года и конфликтов даты с днём недели;
- безопасные текстовые и голосовые команды для сохранения, редактирования, удаления и преобразования активной preview-карточки в task draft;
- persistent draft focus: выбор карточки кнопкой, порядковым номером, темой или Telegram Reply без создания дубликатов;
- изолированные system actions для списка и batch-очистки drafts с отдельным TTL, snapshot и подтверждением;
- детерминированные natural read-команды для `/drafts`, `/inbox`, `/last_saved`, `/profile`, `/today` и `/help` без LLM;
- canonical `temporal_resolution` для выбранной даты/времени с UTC, timezone и локальным представлением;
- текстовый и голосовой inbox с проверкой расшифровки и отдельным callback до сохранения;
- персональный `/today` и пятишаговая неосуждающая рефлексия `/evening`;
- async SQLAlchemy, Alembic, SQLite локально и PostgreSQL в production;
- независимые OpenRouter/OpenAI-compatible text AI и Speech-to-Text клиенты;
- конфигурируемые имя, тон, расписание, лимиты аудио и продуктовые флаги.

## Быстрый запуск с SQLite

Нужен Python 3.12.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell:
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env  # в PowerShell
alembic upgrade head
future-self-bot
```

В `.env` обязательно заполнить `TELEGRAM_BOT_TOKEN` и `AI_API_KEY`. Приложение не создаёт и не очищает таблицы при запуске: схема изменяется только миграциями.

### Windows PowerShell

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
# Заполните .env, затем:
.\.venv\Scripts\python.exe -m future_self.doctor --network
.\scripts\run_windows.ps1
```

`setup_windows.ps1` не перезаписывает существующий `.env` и не запускает бота. Диагностика без `--network` не обращается к Telegram, text LLM или Speech-to-Text:

```powershell
.\.venv\Scripts\python.exe -m future_self.doctor
```

Безопасная очистка только Python/pytest/ruff-кэшей, без удаления `.env`, `.venv` и баз:

```powershell
.\scripts\clean_caches.ps1
```

## PostgreSQL

```bash
docker compose up -d postgres
```

После этого задайте:

```dotenv
DATABASE_URL=postgresql+asyncpg://future_self:future_self@localhost:5432/future_self
```

И выполните `alembic upgrade head`. `docker compose` поднимает только локальную БД; самого бота удобно запускать из виртуального окружения. Docker-образ приложения также доступен через `docker build -t future-self-bot .`.

## Команды

- `/start` или `/onboarding` — начать или продолжить онбординг;
- `/profile` — показать Vision Profile;
- `/goals` — заново предложить набор целей;
- `/today` — получить фокус дня;
- `/evening` — пройти вечернюю рефлексию;
- `/inbox` — последние подтверждённые мысли;
- `/drafts` — активные preview-карточки с действиями «Открыть», «Сохранить» и «Удалить»;
- `/last_saved` — последняя подтверждённая запись inbox;
- `/cleanup_drafts` — открыть подтверждение очистки активных drafts, ничего не удаляя сразу;
- `/cancel` — отменить активный диалог;
- `/help` — краткая справка.

Обычный текст и расшифрованный voice/audio проходят через один Intent Router. Вопросы и общение получают ответ; идея, задача, желание, заметка или рефлексия показывают preview. При низкой уверенности бот спрашивает, что сделать. До отдельного нажатия «Сохранить» запись в БД не создаётся. Порог задаётся через `INTENT_CONFIDENCE_THRESHOLD` (по умолчанию `0.70`).

## Разработка

```bash
python -m compileall -q src tests alembic
pytest -q
ruff check .
ruff format --check .
alembic check
```

На системах с `make` доступны `make install`, `make migrate`, `make run`, `make test`, `make lint` и `make format`.

## Конфигурация

Полный список находится в `.env.example`. Рекомендуемая локальная конфигурация использует OpenRouter для текста и отключённую транскрипцию:

```dotenv
AI_PROVIDER=openrouter
AI_API_KEY=
AI_BASE_URL=https://openrouter.ai/api/v1
AI_MODEL=openai/gpt-5.4-mini
OPENROUTER_SITE_URL=
OPENROUTER_APP_NAME=MyFutureSelfAI
TRANSCRIPTION_PROVIDER=disabled
```

`OPENROUTER_SITE_URL` необязателен. Если он задан, клиент отправляет `HTTP-Referer`; `OPENROUTER_APP_NAME` передаётся как `X-Title`.

Диалоговый контекст хранится в БД и переживает перезапуск процесса. В LLM передаются только последние сообщения в пределах настраиваемого окна; по умолчанию это 12 сообщений и TTL 24 часа:

```dotenv
CONVERSATION_CONTEXT_MESSAGES=12
CONVERSATION_CONTEXT_TTL_HOURS=24
```

Контекст помогает продолжить недавний разговор, но не заменяет подтверждение inbox: `InboxItem` создаётся только после явной кнопки «Сохранить» или однозначной команды для одной актуальной preview-карточки.
Команды «сохрани» и «можешь сохранить» используют тот же атомарный confirm-путь, что и callback. Вопросы, отрицания с отсрочкой и условные формулировки не считаются подтверждением.
Новый preview автоматически становится focused. При нескольких карточках без focus бот сохраняет pending action и предлагает выбрать draft; focus по умолчанию живёт 15 минут (`DRAFT_FOCUS_TTL_MINUTES`).
`/drafts` показывает не более пяти сгруппированных строк на страницу. Массовая очистка использует отдельный `system_pending_action`, проверяет неизменность snapshot и по умолчанию ожидает подтверждение до 10 минут (`SYSTEM_ACTION_TTL_MINUTES`). Сохранённые InboxItem batch-очисткой не изменяются.

Text LLM и транскрипция никогда не используют один клиент автоматически. Для официального OpenAI Speech-to-Text задайте отдельный ключ:

```dotenv
TRANSCRIPTION_PROVIDER=openai
TRANSCRIPTION_API_KEY=
TRANSCRIPTION_BASE_URL=https://api.openai.com/v1
TRANSCRIPTION_MODEL=gpt-4o-mini-transcribe
```

При `TRANSCRIPTION_PROVIDER=disabled` текстовый inbox работает полностью, а voice получает спокойное сообщение о ненастроенном распознавании. `OPENAI_API_KEY` и `OPENAI_MODEL` временно принимаются как fallback только для text AI и выводят предупреждение; новые конфигурации должны использовать `AI_API_KEY` и `AI_MODEL`.

Сетевая диагностика отдельно проверяет Telegram, text LLM и STT. Text LLM check отправляет минимальный запрос с одним выходным токеном:

```powershell
.\.venv\Scripts\python.exe -m future_self.doctor --network --timeout 15
```

Помимо провайдеров можно настроить строку БД, часовой пояс, имя и тон ассистента, расписание, ограничения аудио и feature flags.

## Структура

```text
src/future_self/
  bot.py             Telegram transport и state machines
  actions.py         общий путь callback и текстовых/голосовых draft-команд
  natural_commands.py natural-language read-only команды без LLM
  system_actions.py  безопасные системные действия над drafts
  conversation.py    persistent-контекст диалога пользователя и чата
  dates.py           детерминированное разрешение и проверка дат
  domain.py          онбординг, inbox, профиль, фокус и timezone-логика
  models.py          SQLAlchemy-модели
  repositories.py    persistence-операции
  schemas.py         строгие Pydantic-схемы LLM
  ai.py              интерфейс и OpenRouter/OpenAI-compatible text adapter
  transcription.py   независимый OpenAI/local/disabled STT adapter
  scheduler.py       изолированный JobQueue adapter
  config.py          personality, schedule и feature flags
  prompts.py         системные промпты
alembic/              миграции схемы
tests/                тесты с fake AI/STT
scripts/              установка, запуск и безопасная очистка для Windows
```

Не коммитьте `.env`, локальные базы и пользовательские аудиофайлы. Подробные продуктовые границы описаны в [docs/PRD.md](docs/PRD.md), а первый smoke-test — в [docs/MANUAL_TEST.md](docs/MANUAL_TEST.md).
