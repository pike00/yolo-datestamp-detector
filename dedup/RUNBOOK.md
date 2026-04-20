# dedup-postgres Runbook

## Changing the image or PGDATA

**Don't silently alter `image:` or `PGDATA:` in [docker-compose.yml](docker-compose.yml).** A
change to either can cause the new container to `initdb` into a fresh cluster,
orphaning the existing data in the volume without deleting it — which is
exactly what happened on 2026-04-16 (commit `a38a08e`). Wiped tables
(`stamp_predictions`, `stamp_ocr`, `stamp_no_stamp`, `stamp_prediction_drift`,
plus all training labels) with no offsite copy at the time; see the
2026-04-19 date-mapping-rebuild handoff.

Procedure for any image-version or PGDATA change:

1. **Take a fresh dump.** The `dedup-db-backup` sidecar runs nightly, but you
   want a dump from *right now*, not last night's:
   ```
   cd ~/Documents/Homelab/infra/backup && just pgdump
   ```
   Verify: `ls -lh /mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/backups/dedup/$(date +%Y%m%d)*`
   should show a fresh tarball within the last few minutes. The pre-commit
   hook ([scripts/hooks/pre-commit](../scripts/hooks/pre-commit)) enforces this
   but don't rely on it — a manual dump is the real safeguard.

2. **Stop the old container cleanly.**
   ```
   docker compose -f dedup/docker-compose.yml down
   ```
   Do NOT `down -v` (that deletes the volume).

3. **Decide on data migration strategy:**
   - **Minor version bump (e.g. pg18.1 → pg18.2, same PGDATA):** usually safe
     to just restart. Back up anyway.
   - **Major version bump (pg17 → pg18):** pg18 will refuse to start on a
     pg17 data dir. Use `pg_dumpall` + restore, or run `pg_upgrade` in a
     one-shot container. Never point a new major-version image at the old
     PGDATA and hope it works.
   - **Changing PGDATA:** this almost never makes sense. If you must, rename
     the volume explicitly so the old data is visibly stranded — don't leave
     it silently shadowed inside the same volume.

4. **Restart and verify.**
   ```
   docker compose -f dedup/docker-compose.yml up -d
   docker logs dedup-postgres --tail 50
   docker exec dedup-postgres psql -U dedup -d dedup -c "\dt"
   ```
   Row counts should match pre-change. If not, the new container ran `initdb`
   — stop it, investigate, restore from the dump taken in step 1.

## Backups

- **Nightly pg_dump:** 01:30 local (Homelab `backup-pgdump.timer`). Dumps to
  `dedup_db_backup` Docker volume inside container; 30-day retention.
- **Nightly tar of the dump volume:** 02:00 local (Homelab
  `backup-volumes.timer` → `homelab-volumes.sh backup_dedup`). Writes to
  `/mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/backups/dedup/<timestamp>/volumes/`.
- **Offsite sync to B2:** 03:30 local (Homelab `backup-dedup-offsite.timer`).
  Syncs the tarball dir to `backblaze:pike-backup/_homelab_backups/dedup/`.
- **Kopia:** picks up the tarball on its next cycle (variable).

## Recovery

From local tarball:
```
mkdir -p /tmp/restore && cd /tmp/restore
tar xzf /mnt/823c9bf9-838a-4591-a00f-ae361fcb4792/backups/dedup/<ts>/volumes/dedup_db_backup.tar.gz
docker exec -i dedup-postgres psql -U dedup -d dedup < <(zcat last/dedup-latest.sql.gz)
```

From B2 (if local HDD is also lost):
```
rclone copy backblaze:pike-backup/_homelab_backups/dedup/<ts>/volumes/dedup_db_backup.tar.gz /tmp/
# then as above
```
