<p align="center">
  <img src="./docs/assets/avatar.webp" width="230" alt="Future Self AI avatar" />
</p>

<h1 align="center">Future Self AI</h1>

<p align="center">
  <strong>«Моя будущая версия» — приватный Telegram-ассистент, связывающий желаемое будущее с небольшими ежедневными действиями.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.12" />
  <img src="https://img.shields.io/badge/Telegram-Private_Chat-26A5E4?style=flat-square&logo=telegram&logoColor=white" alt="Private Telegram" />
  <img src="https://img.shields.io/badge/Knowledge_Hub-Feature_Flagged-2FD6BE?style=flat-square" alt="Knowledge Hub" />
  <img src="https://img.shields.io/badge/Status-Active_Development-00C853?style=flat-square" alt="Status" />
</p>

MVP принимает мысли текстом или голосом, ведёт цели, задачи, напоминания и приватные жизненные контуры. Knowledge Hub и Universal Capture работают только по явной команде/кнопке, с preview/confirm, immutable revisions и отдельным ingestion runner без Telegram/AI/STT-секретов и сети.

<p align="center">
  <img src="./docs/assets/architecture.svg" width="100%" alt="Future Self AI architecture" />
</p>

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
- Life Areas & Smart Collections: пользовательские темы, проекты и списки связывают
  существующие записи Inbox и Task Hub без копирования содержимого или состояния задач;
- совместные пространства с отдельным ACL-контуром: роли owner/editor/viewer,
  одноразовые приглашения, проекты, немедленный revoke и перезапускоустойчивые действия;
- Knowledge Hub с личными, workspace- и project-пространствами, стабильными public ID,
  неизменяемыми ревизиями, ролями знания, очередью обработки и двухступенчатым удалением;
- Universal Capture только по явной команде/кнопке и с обязательным preview/confirm;
  документы разбирает отдельный локальный runner без Telegram/AI/STT-секретов и сети;
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
- добровольное сопровождение одного выбранного желания: персональный утренний вопрос,
  вечерняя отметка, 0–3 дополнительных уведомления в часовом поясе пользователя,
  сохранение истории ответов и связь фокуса с `/today`;
- AI-визуализация отдельной карточки через фиксированный `openai/gpt-image-2` в OpenRouter:
  показ точного prompt до платного запроса, отдельные подтверждения генерации и
  сохранения, постоянная приватная библиотека именованных референсов с явным выбором
  до четырёх изображений на один запрос;
- единая навигация `/menu` и кнопочный `/help`: разделы, быстрый старт, примеры,
  owner-safe выход из незавершённых сценариев и компактное native Telegram-меню;
- текстовый и голосовой inbox с проверкой расшифровки и отдельным callback до сохранения;
- персональный `/today` и пятишаговая неосуждающая рефлексия `/evening`;
- async SQLAlchemy и Alembic; текущий production использует SQLite, PostgreSQL остаётся
  поддерживаемым целевым backend;
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

## Управление доступом (этапы 1, 2 и 2.1)

В таблице `users` хранится единый источник авторизации — `access_tier`:

- `guest` — гостевой уровень без права на полный бот;
- `subscriber` — полный пользовательский доступ;
- `admin` — полный доступ администратора;
- `blocked` — доступ заблокирован независимо от состояния профиля.

`onboarding_completed` означает только завершённость настройки профиля и не является
подпиской. Изменение onboarding не повышает и не понижает `access_tier`. Команды оператора
не создают отсутствующих пользователей и не удаляют профиль, задачи, workspace, health или
другие пользовательские данные:

```bash
future-self-access status <telegram_id>
future-self-access grant subscriber <telegram_id>
future-self-access grant admin <telegram_id>
future-self-access set guest <telegram_id>
future-self-access block <telegram_id>
future-self-access unblock <telegram_id>
```

После применения миграции назначьте согласованного администратора отдельной командой — ID не
зашит в приложение или схему БД:

```bash
future-self-access grant admin 530129470
```

Операторский rollout выполняется только при остановленном poller:

1. Остановить bot process.
2. Сделать backup БД.
3. Выполнить `alembic upgrade head`.
4. Развернуть совместимый код.
5. Выдать admin через `future-self-access grant admin 530129470`.
6. Запустить bot process.

Ранний access gate, гостевой входящий контур, безопасные Telegram command scopes и защита
исходящих фоновых отправок подключены. Daily/weekly, Health, Vision Companion и Task Reminder
проверяют актуальный `access_tier`; Task Reminder сохраняет pending-строку при потере доступа и
может доставить её после возврата subscriber/admin.

Известное ограничение: уже включённые Health/Vision preferences пользователя, которому выдали
subscriber/admin, не добавляются в in-memory JobQueue автоматически. Они начнут планироваться после
перезапуска bot process либо после изменения соответствующей настройки самим пользователем;
периодического access polling нет.

Production rollout всё ещё запрещён: гостевые AI demos и согласованный daily quota/usage
enforcement будут реализованы отдельными этапами. До них последовательность rollout выше остаётся
только планом.
Возврат к коду без access enforcement security-sensitive; downgrade допустим только при
остановленном bot process и с отдельной оценкой последствий отката.

## PostgreSQL

```bash
docker compose up -d postgres
```

После этого задайте:

```dotenv
DATABASE_URL=postgresql+asyncpg://future_self:future_self@localhost:5432/future_self
```

И выполните `alembic upgrade head`. `docker compose` поднимает только локальную БД; самого бота удобно запускать из виртуального окружения. Docker-образ приложения также доступен через `docker build -t future-self-bot .`.

Compose публикует демонстрационный PostgreSQL только на loopback `127.0.0.1`; его
credentials нельзя использовать для публичного или production-развёртывания.

### Knowledge ingestion runner

PR #24 оставляет Hub, Capture и runner выключенными по умолчанию. После Alembic migration
их можно включать независимо; runner запускается тем же image отдельным процессом:

```bash
future-self-knowledge-runner
future-self-knowledge-runner --doctor
```

Он читает только allowlist DB/storage/limit variables, не загружает bot `.env`, не имеет
Telegram/LLM/STT полей конфигурации и рассчитан на контейнер без сети. Hardened локальный
profile доступен командой `docker compose --profile knowledge up -d knowledge-runner`;
путь данных задаётся `KNOWLEDGE_DATA_PATH`, а миграция выполняется приложением заранее.

Детерминированно извлекаются TXT, Markdown, текстовый PDF, DOCX и EPUB. Изображения,
сканы и image-only PDF сохраняются со статусом `partial` без OCR. URL остаётся только
ссылкой: runner не скачивает страницу, не следует redirect и не получает preview.

Согласованный SQLite+assets backup и автономная проверка:

```bash
future-self-knowledge-backup create \
  --database /data/future_self.db \
  --assets /data/knowledge \
  --destination /data/backups/knowledge-<UTC> \
  --application-sha <DEPLOYED_SHA>
future-self-knowledge-backup verify /data/backups/knowledge-<UTC>
```

Команда create ставит maintenance marker, ждёт завершения активной lease, делает
`sqlite3.Connection.backup()`, копирует assets с SHA-256 и публикует backup только после
успешной проверки manifest. Полный runbook находится в
`docs/operations/production-hardening.md`.

## Команды

- `/menu` — открыть кнопочное главное меню;
- `/doctor` — единый раздел поиска врача и подготовки к приёму;
- `/labs` — безопасно загрузить и управлять фото/PDF результатов анализов;
- `/start` или `/onboarding` — начать или продолжить онбординг;
- `/profile` — показать Vision Profile и личную локацию;
- `/location Саратов` — сохранить свой город; запасной маршрут можно задать как
  `/location Саратов → Энгельс`;
- `/timezone` — показать текущий часовой пояс; `/timezone Казань` определит новый по
  обычному названию города и сохранит его только после подтверждения местного времени;
- `/vision` — добавить желание текстом или голосом, открыть карту и достигнутые карточки;
- `/goals` — заново предложить набор целей;
- `/today` — получить фокус дня;
- `/evening` — пройти вечернюю рефлексию;
- `/inbox` — последние подтверждённые мысли;
- `/tasks` — Task Hub: активные, предстоящие, просроченные и выполненные задачи;
- `/collections` — «Мои разделы»: темы, проекты, списки и стартовые сферы по явному выбору;
- `/spaces` — совместные пространства, участники, приглашения и workspace-проекты;
`/workspaces` остаётся расширенным алиасом;
- `/knowledge` — доступные базы знаний и материалы (только при включённом Hub);
- `/capture` — явное добавление текста, документа, изображения или ссылки с подтверждением;
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

Явные команды разделов обрабатываются детерминированно до LLM. Остальной обычный текст и
расшифрованный voice/audio проходят через Intent Router. Вопросы и общение получают ответ;
идея, задача, желание, заметка или рефлексия показывают preview. При низкой уверенности бот
спрашивает, что сделать. До отдельного нажатия «Сохранить» запись в БД не создаётся. Порог
задаётся через `INTENT_CONFIDENCE_THRESHOLD` (по умолчанию `0.70`).

Сценарий `/vision` детерминированный и не отправляет содержимое карты в LLM. Черновик
переживает рестарт, а карточка появляется только после явного подтверждения. Задача из
первого шага создаётся через общий Task & Reminder Engine идемпотентно, но reminder
назначается только отдельным действием пользователя.

Кнопка «Создать AI-визуализацию» в главном меню сначала показывает компактный список
активных желаний, затем переводит к выбору приватных референсов и подтверждению запроса к
`openai/gpt-image-2`. Локальный экспорт общей карты не смешивается с AI-сценарием: он
доступен отдельной кнопкой «Скачать карту желаний в PNG» внутри «Моей карты».

Экспорт локально строит детерминированные PNG 1080×1350 только из активных карточек
владельца. Рендер выполняется в памяти без LLM, внешней сети, публичных ссылок и постоянного
хранения файлов; исходный PNG можно получить отдельным document. Большая карта разбивается
максимум на шесть страниц по пять карточек. Docker-образ устанавливает Debian-пакет
`fonts-dejavu-core`; его лицензия и copyright доступны внутри образа в
`/usr/share/doc/fonts-dejavu-core/copyright`.

К карточке можно прикрепить одну личную фотографию из Telegram photo или image-document.
Поддерживаются статические JPEG, PNG и WebP: до 8 МиБ, 24 мегапикселей и 20 000 пикселей
по стороне. До DB-транзакции файл локально проверяется по фактическому формату, EXIF
orientation применяется, а EXIF/GPS/ICC/comments/filename удаляются. В SQLite сохраняется
только RGB JPEG display-копия: максимум 1600 пикселей по стороне и 768 КиБ. Оригинал,
временные файлы и публичные ссылки не создаются; LLM, внешние хранилища и renderer-сервисы
не вызываются. Preview требует явного подтверждения, а замена и удаление защищены
owner/chat/item/version capability.

Если включена AI-генерация, карточка также предлагает создать квадратное изображение через
OpenRouter Images API и фиксированную модель `openai/gpt-image-2`. Используются уже
настроенные `AI_API_KEY` и `AI_BASE_URL`; ключ транскрипции не используется. До сетевого
запроса бот показывает точный prompt, модель, размер и качество. Во внешний запрос входят
только текст желания и его категория; причина, дата, первый шаг и данные других разделов
не передаются.
Запрос выполняется лишь после отдельного подтверждения, а результат сначала показывается
как временное preview и попадает в карточку только после второго подтверждения. Возможный
moderation block не повторяется автоматически; остальные ошибки также требуют ручного
повтора, чтобы исключить неожиданные дубли платных генераций. Нормализация и хранение
результата используют тот же безопасный JPEG-контур, что и загруженное фото.

Раздел `/labs` принимает Telegram photo, image-document (статические JPEG, PNG, WebP)
и обычные незашифрованные PDF. Фактические лимиты: входной файл до 8 МиБ; PDF до
10 страниц и 2000 PDF points по каждой стороне страницы; исходное изображение до
24 мегапикселей и 20 000 пикселей по стороне; обработка PDF до 30 секунд; готовая
PDF-страница до 3000 пикселей по стороне и 12 мегапикселей; JPEG-копия до 768 КиБ
на страницу и до 8 МиБ на документ. Не более 16 upload-сессий и 32 МиБ временных
нормализованных данных держатся одновременно; TTL preview — 20 минут.

Изображения проходят тот же локальный EXIF orientation и безопасное RGB JPEG
перекодирование, что и личные фото Vision. PDF предварительно строго разбирается и
fail-closed отклоняется при пароле, повреждении, JavaScript, вложениях, actions,
формах или мультимедиа. Затем отдельный локальный PDFium subprocess без shell и сети,
с timeout и лимитами CPU/RAM/выхода, растрирует каждую страницу. Исходник удаляется
сразу; до подтверждения остаются только случайно именованные файлы mode `600` в
каталоге mode `700`, без symlink/path traversal, после confirm/cancel/error/TTL они
удаляются. В SQLite BLOB сохраняются только нормализованные JPEG-страницы после кнопки
«Сохранить». OCR, LLM, внешние renderer-сервисы, публичные URL, извлечение показателей
и медицинская интерпретация не используются.

PDF pipeline использует `pypdf` (BSD-3-Clause) для локальной проверки и `pypdfium2`
с PDFium (BSD-3-Clause/Apache-2.0 и лицензии зависимостей) для локального raster render;
системный PDF-пакет в Docker не устанавливается. Точные license metadata доступны через
`python -m pip show pypdf pypdfium2` внутри образа.

Навигация строится из единого декларативного каталога: те же названия и описания
используются в главном меню, справке и native Telegram-командах. В native-меню
показываются только `/menu`, `/today`, `/evening`, `/inbox`, `/tasks`, `/collections`,
`/vision`, `/health`, `/checkin`, `/doctor`, `/labs`, `/location` и `/help`; при явном
включении Access foundation к ним добавляется `/spaces`. Расширенные legacy-команды
продолжают работать.
Если открыт
check-in, подготовка к врачу, карточка желания или другой пошаговый сценарий, `/menu`
предлагает явно продолжить его либо выйти, очищая только текущее состояние.

`/collections` и `/spaces` не взаимозаменяемы. «Мои разделы» организуют только личные
записи владельца, а workspace является отдельной границей доступа. Создатель получает
роль owner; owner управляет участниками и приглашениями, owner/editor — проектами,
viewer — только чтением. Последнего owner нельзя удалить, понизить или вывести без
предварительной передачи владения. Отзыв membership атомарно инвалидирует активный
контекст и старые кнопки. Telegram-имя и произвольный `@username` не используются для
поиска получателя: direct invite возможен только для уже однозначно известного боту
внутреннего пользователя, иначе создаётся одноразовая deep link с TTL. В БД хранится
только hash токена; принятие всегда требует явного подтверждения. Личные Vision,
Inbox, Tasks, Collections, Health, Doctor и Labs автоматически не публикуются.

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
COLLECTION_ACTION_TTL_MINUTES=15
COLLECTION_INPUT_TTL_MINUTES=20
COLLECTION_CONTEXT_TTL_MINUTES=20
ENABLE_WORKSPACE_ACCESS=false
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

AI-визуализация использует существующие OpenRouter credentials текстового контура, но
отдельный HTTP-клиент и явный feature flag. Дублировать ключ не нужно:

```dotenv
ENABLE_VISION_IMAGE_GENERATION=true
IMAGE_GENERATION_MODEL=openai/gpt-image-2
IMAGE_GENERATION_SIZE=1024x1024
IMAGE_GENERATION_QUALITY=medium
IMAGE_GENERATION_TIMEOUT_SECONDS=150
```

`IMAGE_GENERATION_MODEL` намеренно принимает только `openai/gpt-image-2`; включённая
функция требует `AI_PROVIDER=openrouter`. По умолчанию функция выключена, поэтому локальный
рендер общей карты и загрузка собственных фото продолжают работать без сетевой генерации.
Пошаговая проверка интерфейса, сохранения и ошибок описана в
[гайде по AI-визуализациям](docs/VISION_IMAGE_GUIDE.md).

Сетевая диагностика отдельно проверяет Telegram, text LLM и STT. Text LLM check отправляет минимальный запрос с одним выходным токеном:

```powershell
.\.venv\Scripts\python.exe -m future_self.doctor --network --timeout 15
```

Помимо провайдеров можно настроить строку БД, часовой пояс, имя и тон ассистента, расписание, ограничения аудио и feature flags.

PR #23 добавляет только Access/Workspace foundation. Его UI включается независимо через
`ENABLE_WORKSPACE_ACCESS`; по умолчанию флаг выключен и команда/кнопки отсутствуют.
Capture, ingestion runner, retrieval, embeddings, OCR, Knowledge media, Council,
scheduling и export всё ещё не реализованы и не активируются этим флагом. Workspace CRUD,
ACL и навигация детерминированы и не вызывают LLM или внешние провайдеры. Текущие решения
и hardened production profile описаны в
[ADR-0001](docs/adr/0001-knowledge-hub-council-foundation.md) и
[ADR-0002](docs/adr/0002-access-workspace-foundation.md), а также в
[production hardening runbook](docs/operations/production-hardening.md).

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
  vision_renderer.py локальный owner-safe PNG renderer карты желаний
  safe_media/        недоменные fail-closed image/PDF/subprocess primitives
  lab_media.py       Labs-adapter поверх общей safe_media boundary
  lab_pdf_worker.py  совместимый entrypoint общего PDFium worker
  labs.py            owner-isolated документы, BLOB-страницы и capabilities
  lab_handlers.py    Telegram preview/confirm/list/view/edit/delete flow
  workspace_access.py ACL, invitations, contexts и restart-safe capabilities
  config.py          personality, schedule и feature flags
  prompts.py         системные промпты
alembic/              миграции схемы
tests/                тесты с fake AI/STT
scripts/              установка, запуск и безопасная очистка для Windows
```

Не коммитьте `.env`, локальные базы и пользовательские аудиофайлы. Подробные продуктовые границы описаны в [docs/PRD.md](docs/PRD.md), а первый smoke-test — в [docs/MANUAL_TEST.md](docs/MANUAL_TEST.md).

---

## Naz AI Lab

This project is part of [Naz AI Lab](https://naz-ai-lab.ru/) — the digital laboratory of [Nazar Zykov](https://naz-ai-lab.ru/creator/).

[Official project page](https://naz-ai-lab.ru/projects/future-self-ai/) · [All Naz AI Lab projects](https://naz-ai-lab.ru/projects/)
