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
- persistent Task & Reminder Engine: раздельные `event_at`/`remind_at`, Telegram-доставка,
  IANA timezone, восстановление после рестарта, lease и защита от повторной отправки;
- private-chat-only transport: любые сообщения и callback в группах/каналах блокируются до
  feature handlers, а reminder доставляется по Telegram ID владельца, не по сохранённому chat ID;
- Health Track MVP: приватные пошаговые check-in, субъективная шкала 0–100, недельная
  динамика, история/timezone, исправление, удаление и добровольные daily reminders;
- Doctor Visit Prep: приватный `/doctor_prepare`, фактическое резюме с Health Track
  dynamics, owner-only edit/delete и generic-задача записи к врачу с reminder;
- Doctor Search: официальные маршруты к терапевту по личной локации пользователя
  и owner-isolated задача через существующий Reminder Engine;
- текстовая Карта желаний: owner-isolated карточки по категориям, preview с явным
  подтверждением, архив/достижения и идемпотентная задача из первого шага без
  автоматического reminder;
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
- `/profile` — показать Vision Profile и личную локацию;
- `/location Саратов` — сохранить свой город; запасной маршрут можно задать как
  `/location Саратов → Энгельс`;
- `/vision` — добавить желание текстом или голосом, открыть карту и достигнутые карточки;
- `/goals` — заново предложить набор целей;
- `/today` — получить фокус дня;
- `/evening` — пройти вечернюю рефлексию;
- `/inbox` — последние подтверждённые мысли;
- `/drafts` — активные preview-карточки с действиями «Открыть», «Сохранить» и «Удалить»;
- `/last_saved` — последняя подтверждённая запись inbox;
- `/cleanup_drafts` — открыть подтверждение очистки активных drafts, ничего не удаляя сразу;
- `/cancel` — отменить активный диалог;
- `/help` — краткая справка.
- `/health` — текущее субъективное состояние, недельная динамика и история;
- `/checkin` — пошаговый health check-in;
- `/health_edit ID` и `/health_delete ID` — исправить или удалить свою запись;
- `/health_reminder_on 20:00` и `/health_reminder_off` — добровольное напоминание.
- `/doctor_prepare` — пошагово подготовить фактическое резюме к визиту;
- `/doctor_preparations`, `/doctor_prepare_show ID`, `/doctor_prepare_edit ID`,
  `/doctor_prepare_delete ID` — owner-only история и управление;
- `/doctor_prepare_task ID через 2 часа` — создать generic-задачу записи с reminder.
- `/doctor_find` — показать официальные варианты записи к терапевту по локации владельца;
- `/doctor_find_task через 2 часа` — создать owner-isolated задачу с локацией владельца
  и reminder.

Обычный текст и расшифрованный voice/audio проходят через один Intent Router. Вопросы и общение получают ответ; идея, задача, желание, заметка или рефлексия показывают preview. При низкой уверенности бот спрашивает, что сделать. До отдельного нажатия «Сохранить» запись в БД не создаётся. Порог задаётся через `INTENT_CONFIDENCE_THRESHOLD` (по умолчанию `0.70`).

Сценарий `/vision` детерминированный и не отправляет содержимое карты в LLM. Черновик
переживает рестарт, а карточка появляется только после явного подтверждения. Задача из
первого шага создаётся через общий Task & Reminder Engine идемпотентно, но reminder
назначается только отдельным действием пользователя.

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

Для подтверждённой task-карточки с canonical датой engine атомарно создаёт одну persistent
запись напоминания. Точное событие сохраняется как `event_at`, доставка — как независимое
`remind_at`; оба значения хранятся в UTC, а исходный IANA timezone используется в Telegram.
Для даты без времени событие по умолчанию назначается на 09:00 локального времени, для
datetime напоминание приходит за 30 минут. Настройки:

```dotenv
TASK_DATE_EVENT_HOUR=9
TASK_REMINDER_LEAD_MINUTES=30
TASK_REMINDER_POLL_SECONDS=15
TASK_REMINDER_LEASE_SECONDS=120
ENABLE_TASK_REMINDERS=true
```

Pending-доставка переживает рестарт процесса. Worker атомарно захватывает запись по lease,
а уникальные `inbox_item_id` и `delivery_key` вместе со статусом `sent` защищают обычные,
повторные и конкурентные poll-циклы от дублей. Ошибки Telegram повторяются с ограниченным
exponential backoff без сохранения текста ответа провайдера.

Relative-команды вида «Напомни через 5 минут выпить воды» и «Напомни через 2 часа
проверить духовку» распознаются детерминированно до LLM одинаково для текста и voice.
После подтверждения карточки `remind_at` равен точному моменту `now + interval`, без
стандартного 30-минутного сдвига. Поддерживаются цифры, формы «через час/минуту» и
русские числительные от одного до десяти; максимальный безопасный интервал — семь дней.

Health Track хранит оценки энергии, сна, настроения, стресса и физического самочувствия
по шкале 0–10, краткие наблюдения и вычисленную субъективную линейку 0–100. Линейка нужна
только для наблюдения за динамикой и не является диагнозом. Health-команды не используют
LLM; записи всегда фильтруются по владельцу, а тексты симптомов не пишутся в application
logs. При консервативных red flags бот рекомендует местную экстренную медицинскую помощь,
а при длительной слабости — запись к врачу и нейтральный список наблюдений. Бот не ставит
диагнозы, не назначает лекарства, лечение или анализы.

Doctor Visit Prep детерминированно собирает причину обращения, длительность, симптомы,
текущие лекарства/добавки и вопросы врачу. В резюме попадают только факты пользователя и
агрегированная динамика Health Track; исторические тексты симптомов не копируются. Данные
не передаются в LLM и не пишутся в application logs. Red flags показываются сразу, не
дожидаясь конца опроса. Задача «Записаться к врачу» и её Telegram reminder не содержат
причину обращения или симптомы. Резюме не является диагнозом и не заменяет врача.

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
  reminders.py       persistent outbox задач и Telegram-доставка напоминаний
  config.py          personality, schedule и feature flags
  prompts.py         системные промпты
alembic/              миграции схемы
tests/                тесты с fake AI/STT
scripts/              установка, запуск и безопасная очистка для Windows
```

Не коммитьте `.env`, локальные базы и пользовательские аудиофайлы. Подробные продуктовые границы описаны в [docs/PRD.md](docs/PRD.md), а первый smoke-test — в [docs/MANUAL_TEST.md](docs/MANUAL_TEST.md).
