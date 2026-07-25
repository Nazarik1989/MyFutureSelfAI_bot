# Production hardening and deployment runbook

> PR #24 note: this Knowledge implementation and its operational contour are local
> only. No PR #24 production deployment or feature enablement has been performed.

## Audited baseline (2026-07-22)

The live audit was read-only and did not inspect `.env` contents, Docker environment
values, Telegram identifiers or user records.

| Check | Observed PR #22 baseline | PR #23 target |
|---|---|---|
| Git/image | clean `a8aa5adc…`, running, restart 0 | final squash SHA/image |
| Database | SQLite, Alembic `20260720_0017`, integrity/FK OK | `20260722_0018`; WAL + 5s busy timeout |
| PostgreSQL port | two loopback-only listeners; no non-loopback `5432` | unchanged; compose also loopback-only |
| Secrets/data modes | `.env`/DB `0600`; data/backups `0700` | unchanged modes; no value reads |
| Container identity | UID/GID `10001:10001` | unchanged |
| Root filesystem | read-only with bounded private `/tmp` | unchanged |
| Capabilities/security | all dropped; `no-new-privileges` | unchanged |
| Resource/log limits | pids 128, 1 CPU, 1536 MiB, 10 MiB × 5 logs | unchanged |
| Data capacity | more than 10 GiB and 3M inodes free | doctor thresholds: 1 GiB and 10k inodes |
| Knowledge assets | directory absent; no Knowledge data exists | private empty `/data/knowledge`, mode `0700` |
| Rollback | stopped PR #21 container/image and verified SQLite backups | preserve again before cutover |

Host UFW is inactive, but PostgreSQL is bound only to loopback. This runbook does not
claim protection from host root compromise or offsite backup encryption. SSH policy,
offsite retention and secret rotation remain host-operator responsibilities.

## Image contract

- Image user is numeric `10001:10001`; application code remains root-owned/read-only.
- `/data` and `/tmp` are the only writable runtime areas.
- `/data/backups` is over-mounted read-only inside the application container.
- The healthcheck runs safe local diagnostics only; network checks remain explicit.
- No Knowledge flag is enabled in PR #22.

## Pre-cutover checklist

1. Confirm production Git is clean at the previous deployed SHA and `origin/main` is a
   fast-forward descendant.
2. Record the old container image, ID, restart count, policy and mount without reading
   its environment.
3. Stop new finalizations only at cutover. Create SQLite backup with
   `sqlite3.Connection.backup()`, mode `0600`, SHA-256 and integrity checks.
4. Snapshot `/data/knowledge` with a sorted checksum manifest. For PR #22 it must be an
   empty mode-`0700` directory; later PRs must pause asset finalization briefly.
5. Restore DB and asset snapshots into isolated temporary paths and validate integrity,
   manifest and application startup as UID 10001.
6. Build the final SHA with `--no-cache` and confirm image user and healthcheck.
7. Preserve the stopped previous container under a timestamped rollback name.
   For an image whose bundled Alembic graph ends before the live additive revision,
   also prepare a stopped rollback container with command
   `sh -c 'umask 077 && exec future-self-bot'`. This intentionally skips the old
   image's `alembic upgrade head`; otherwise that image cannot identify the newer
   revision even though its application code is compatible with the extra tables.
   The old image's inherited doctor healthcheck has the same old-head limitation, so
   override it with a local process/SQLite check compatible with the retained revision:

   ```bash
   --health-cmd "python -c 'import os,sqlite3;os.kill(1,0);c=sqlite3.connect(\"file:/data/future_self.db?mode=ro\",uri=True,timeout=5);ok=c.execute(\"PRAGMA quick_check\").fetchone()[0]==\"ok\";rev=c.execute(\"SELECT version_num FROM alembic_version\").fetchone()[0];c.close();raise SystemExit(0 if ok and rev in {\"20260722_0018\",\"20260722_0019\"} else 1)'" \
   --health-interval=60s --health-timeout=20s --health-start-period=30s \
   --health-retries=3
   ```

   Validate both overrides against an isolated post-migration backup with networking
   disabled. Do not rely on an `unhealthy` rollback candidate or stamp the live DB back.

## Hardened container profile

The deploy uses the existing `.env` only as Docker input; the file is never displayed
or modified.

```bash
docker run -d \
  --name myfutureselfai-bot \
  --restart unless-stopped \
  --user 10001:10001 \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=128m,mode=1777 \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --pids-limit 128 \
  --cpus 1.0 \
  --memory 1536m \
  --memory-swap 1536m \
  --log-driver json-file \
  --log-opt max-size=10m \
  --log-opt max-file=5 \
  --env-file /opt/myfutureselfai/.env \
  --env DATABASE_URL=sqlite+aiosqlite:////data/future_self.db \
  --env ENABLE_WORKSPACE_ACCESS=true \
  --env ENABLE_KNOWLEDGE_HUB=false \
  --env ENABLE_KNOWLEDGE_CAPTURE=false \
  --env ENABLE_KNOWLEDGE_RUNNER=false \
  --env ENABLE_KNOWLEDGE_RETRIEVAL=false \
  --env ENABLE_KNOWLEDGE_EMBEDDINGS=false \
  --env ENABLE_KNOWLEDGE_OCR=false \
  --env ENABLE_KNOWLEDGE_MEDIA=false \
  --env ENABLE_EXTERNAL_VISION=false \
  --env ENABLE_COUNCIL=false \
  --env ENABLE_SCHEDULED_COUNCIL=false \
  --env ENABLE_KNOWLEDGE_EXPORT=false \
  --mount type=bind,src=/opt/myfutureselfai/data,dst=/data \
  --mount type=bind,src=/opt/myfutureselfai/data/backups,dst=/data/backups,readonly \
  myfutureselfai-bot:<FINAL_SHA>
```

Before this command, `/opt/myfutureselfai/data`, the live DB and the empty/current
`knowledge` directory must be owned by `10001:10001`. The backup directory and backup
files stay root-owned `0700/0600`.

## Post-deploy checks

- container `running`, `healthy`, restart 0, `unless-stopped`;
- UID/GID 10001, read-only rootfs, all capabilities dropped, no-new-privileges;
- mount and log/resource limits exactly match this profile;
- Alembic is at the deployed head; SQLite integrity/FK OK, journal `wal`, busy timeout
  5000 ms;
- public command catalog exposes `/spaces` only when `ENABLE_WORKSPACE_ACCESS=true`;
- doctor Telegram/OpenRouter/STT/DB checks pass;
- aggregate log scan has no Traceback, lock, Conflict, critical or error entries;
- only the explicitly deployed foundation flag may be true; Capture, Runner, Retrieval,
  Embeddings, OCR, Media, External Vision, Council, Scheduling and Export stay false.

## Backup retention and secret rotation

- Never automatically delete a backup during deploy. Retention changes require a
  separate reviewed operation after an offsite/encryption policy is selected.
- Backups contain private data even when encrypted offsite; permissions are `0600` and
  the directory is `0700`.
- Rotate secrets one provider at a time through a replacement env file, validate with
  doctor, atomically replace the file, restart and revoke the old credential. Never
  print values or include them in shell history, Docker inspect output or logs.
- Rollback restores a coordinated DB+asset snapshot only when that incident decision is
  explicit; ordinary additive-schema rollback starts the prepared previous-image
  container without changing ownership. No container or image is force-removed.
- After Knowledge data exists, keep the additive Access and Knowledge schema when rolling
  back the image. Do not downgrade it destructively; restore the coordinated pre-cutover
  backup only when discarding all post-cutover Workspace/Knowledge changes is an explicit
  incident decision.
- Do not `stamp` the live database backward to make an old image start. Use the
  pre-created rollback command override above, keep revision `20260722_0019`, and retain
  the failed PR #24 container under a separate stopped name for forensics.

## PR #24 Knowledge rollout gate

Do not enable a step until migration `20260722_0019`, a coordinated DB+asset backup,
restore verification, and the preceding step's checks have succeeded:

1. Enable `ENABLE_KNOWLEDGE_HUB=true` only and verify scoped personal/workspace/project
   listing. Capture and runner remain false.
2. Enable `ENABLE_KNOWLEDGE_CAPTURE=true` and verify explicit confirmation, quotas,
   cancellation, and durable `queued` behavior while no runner exists.
3. Start the separate runner with `ENABLE_KNOWLEDGE_RUNNER=true`, no network, and only
   the allowlisted environment below. Never pass the bot env file.

Retrieval, embeddings, OCR, Knowledge Media, External Vision, Council, scheduling,
export, publication, URL fetching, and cross-repository synchronization remain off and
unimplemented.

### Secret-free runner profile

Use the same immutable image and `/data` mount. The runner does not run Alembic or start
Telegram polling, so migrate before starting it.

```bash
docker run -d \
  --name myfutureselfai-knowledge-runner \
  --restart unless-stopped \
  --stop-timeout 180 \
  --user 10001:10001 \
  --read-only \
  --network none \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=256m,mode=1777 \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --pids-limit 64 \
  --cpus 1.0 \
  --memory 768m \
  --memory-swap 768m \
  --log-driver json-file \
  --log-opt max-size=10m \
  --log-opt max-file=5 \
  --health-cmd "umask 077 && exec future-self-knowledge-runner --doctor" \
  --health-interval=60s \
  --health-timeout=20s \
  --health-start-period=30s \
  --health-retries=3 \
  --env DATABASE_URL=sqlite+aiosqlite:////data/future_self.db \
  --env LOG_LEVEL=INFO \
  --env ENABLE_KNOWLEDGE_RUNNER=true \
  --env KNOWLEDGE_ASSET_ROOT=/data/knowledge \
  --env KNOWLEDGE_RUNNER_CONCURRENCY=1 \
  --env KNOWLEDGE_RUNNER_POLL_SECONDS=2 \
  --env KNOWLEDGE_RUNNER_LEASE_SECONDS=120 \
  --env KNOWLEDGE_RUNNER_HEARTBEAT_SECONDS=30 \
  --env KNOWLEDGE_MAX_SOURCE_BYTES=26214400 \
  --env KNOWLEDGE_EXTRACTION_WALL_SECONDS=30 \
  --env KNOWLEDGE_EXTRACTION_MAX_PAGES=500 \
  --env KNOWLEDGE_EXTRACTION_MAX_ARCHIVE_ENTRIES=2000 \
  --env KNOWLEDGE_EXTRACTION_MAX_UNPACKED_BYTES=104857600 \
  --env KNOWLEDGE_EXTRACTION_MAX_TEXT_BYTES=10485760 \
  --env RUNTIME_MIN_FREE_BYTES=1073741824 \
  --env SQLITE_WAL_ENABLED=true \
  --env SQLITE_BUSY_TIMEOUT_MS=5000 \
  --mount type=bind,src=/opt/myfutureselfai/data,dst=/data \
  --mount type=bind,src=/opt/myfutureselfai/data/backups,dst=/data/backups,readonly \
  myfutureselfai-bot:<PR24_SHA> \
  sh -c 'umask 077 && exec future-self-knowledge-runner'
```

Never add `--env-file`, proxy variables, bot/provider keys, or network connectivity to
this container. `future-self-knowledge-runner --doctor` performs a cheap local DB/layout
healthcheck; run `future-self-knowledge-runner --full-audit` explicitly to hash every DB
asset reference and detect missing, corrupt, or orphaned files. The 180-second stop
timeout is longer than the lease/extraction window so normal shutdown can finish safely.

The parser child is secret-free, networkless, bounded, and receives a private copied
input, but it is not a kernel filesystem sandbox from the runner: both currently use the
same UID and runner mount. Treat native-parser compromise as a residual risk; a future
separate parser container/UID with narrowly scoped input/output mounts is required to
close it. Keep backups outside the runner's writable deployment mount where the host
layout permits, and never mount off-host backup credentials into the runner.

### Coordinated Knowledge backup and verification

A DB-only snapshot is insufficient after Capture is enabled.

1. Verify `/data/knowledge` and the backup parent are private trusted storage. Record the
   deployed SHA without displaying environment values.
2. Run `future-self-knowledge-backup create` with explicit database, assets,
   destination, and application SHA. It atomically creates a maintenance marker and a
   transactional SQLite fence; Capture rechecks the fence before finalization and the
   runner stops leasing new work while already leased work may drain.
3. The command drains processing leases, validates SQLite, uses
   `sqlite3.Connection.backup()`, copies immutable files with no-follow opens, and records
   relative keys, sizes, hashes, Alembic head, app SHA, UTC timestamp, and format version.
4. It publishes the directory only after verification. Run
   `future-self-knowledge-backup verify <backup>` again in an isolated networkless
   container; it checks DB integrity/FKs/head, hashes, missing/corrupt/extra assets,
   unsafe links/private modes, and DB size/hash references.
5. Preserve the coordinated backup. Never delete earlier backups automatically.

If the backup process dies while paused, first confirm that the backup process itself is
gone, then run the fail-closed recovery command; never remove the marker or edit the DB
fence by hand:

```bash
future-self-knowledge-backup recover-maintenance \
  --database /data/future_self.db \
  --assets /data/knowledge
```

The command takes the marker ownership lock, serializes with SQLite writers, refuses to
continue while any unexpired processing lease exists, clears `maintenance_paused`, and
only then removes the exact marker inode. On an error it retains the marker, so retry the
same command after the reported lease/state problem is resolved. Never remove the
storage root or SQLite database.

### Additive rollback after PR #24

Stop the runner first and disable Hub/Capture/Runner. A retained prior image may run
against the additive schema when its Alembic startup and healthcheck are overridden as
described above and tested on an isolated snapshot. Keep revision `20260722_0019` and
all assets. Do not downgrade or stamp the live database. Restore the coordinated
pre-cutover DB+asset backup only when an explicit incident decision accepts losing all
post-cutover Knowledge changes.
