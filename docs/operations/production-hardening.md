# Production hardening and deployment runbook

## Audited baseline (2026-07-22)

The live audit was read-only and did not inspect `.env` contents, Docker environment
values, Telegram identifiers or user records.

| Check | Observed baseline | PR #22 target |
|---|---|---|
| Git/image | clean `3bcaf05f…`, running, restart 0 | final squash SHA/image |
| Database | SQLite, Alembic `20260720_0017`, integrity/FK OK | same head; WAL + 5s busy timeout |
| PostgreSQL port | two loopback-only listeners; no non-loopback `5432` | unchanged; compose also loopback-only |
| Secrets/data modes | `.env`/DB `0600`; data/backups `0700` | unchanged modes; no value reads |
| Container identity | default root, runtime `0:0` | UID/GID `10001:10001` |
| Root filesystem | writable | read-only with bounded private `/tmp` |
| Capabilities/security | default, no `no-new-privileges` | drop all; `no-new-privileges` |
| Resource/log limits | none; unrotated `json-file` | pids 128, 1 CPU, 1536 MiB, 10 MiB × 5 logs |
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
- Alembic remains `20260720_0017`; SQLite integrity/FK OK, journal `wal`, busy timeout
  5000 ms;
- public command catalog is unchanged from PR #21;
- doctor Telegram/OpenRouter/STT/DB checks pass;
- aggregate log scan has no Traceback, lock, Conflict, critical or error entries;
- Knowledge/Council flags are all false and new domain tables/commands do not exist.

## Backup retention and secret rotation

- Never automatically delete a backup during deploy. Retention changes require a
  separate reviewed operation after an offsite/encryption policy is selected.
- Backups contain private data even when encrypted offsite; permissions are `0600` and
  the directory is `0700`.
- Rotate secrets one provider at a time through a replacement env file, validate with
  doctor, atomically replace the file, restart and revoke the old credential. Never
  print values or include them in shell history, Docker inspect output or logs.
- Rollback restores the coordinated DB+asset snapshot and original ownership before
  starting the preserved root-era container. No container or image is force-removed.
