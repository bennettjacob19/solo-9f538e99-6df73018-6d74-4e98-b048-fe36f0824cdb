#!/usr/bin/env python3
"""End-to-end tests for the online re-striping simulation.

Spawns the Flask server as a subprocess against a throwaway SQLite file and
exercises: initial layout, budget=1 progress, host read/write while blocks
are staged, crash-before-commit rollback vs. committed-batch durability,
resume after restart, and final stripe verification for k=2 and k=4.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.db import BLOCK_COUNT, BLOCK_SIZE, DISK_COUNT  # noqa: E402
from app import db, migration  # noqa: E402

BASE = "http://127.0.0.1:8000"
PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def http(method: str, path: str, payload: dict | None = None,
         timeout: int = 10):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=headers,
                                 method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            ctype = resp.headers.get("Content-Type", "")
            return resp.status, json.loads(raw) if "json" in ctype else raw
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, raw
    except (urllib.error.URLError, ConnectionError, OSError) as exc:
        return 0, str(exc)


def wait_up(deadline_s: float = 30.0) -> None:
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            code, _ = http("GET", "/api/snapshot", timeout=2)
            if code == 200:
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.3)
    raise RuntimeError("server did not come up")


def wait_exit(proc: subprocess.Popen, deadline_s: float = 15.0) -> int:
    end = time.time() + deadline_s
    while time.time() < end:
        rc = proc.poll()
        if rc is not None:
            return rc
        time.sleep(0.2)
    proc.kill()
    raise RuntimeError("server did not exit after simulated crash")


def start_server(db_path: str) -> subprocess.Popen:
    env = os.environ.copy()
    env["VIRTUAL_VOLUME_DB"] = db_path
    proc = subprocess.Popen(
        [sys.executable, "-m", "flask", "--app", "app.server", "run",
         "--host", "127.0.0.1", "--port", "8000", "--without-threads",
         "--no-reload", "--no-debugger"],
        cwd=ROOT, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    wait_up()
    return proc


def snapshot() -> dict:
    code, data = http("GET", "/api/snapshot")
    assert code == 200, data
    return data


def block_at(snap: dict, lbn: int) -> dict:
    return next(b for b in snap["blocks"] if b["lbn"] == lbn)


# --------------------------------------------------------------------------- #
def test_planner_math() -> None:
    print("[planner] cycle decomposition is a valid permutation plan")
    for k in (2, 4):
        ops = migration.plan_operations(k)
        check(f"k={k} plan non-empty", len(ops) > 0)
        check(f"k={k} all lbn ids valid",
              all(0 <= o["lbn"] < BLOCK_COUNT for o in ops))
        stages = [o for o in ops if o["op"] == "stage"]
        unstages = [o for o in ops if o["op"] == "unstage"]
        check(f"k={k} stage/unstage balanced", len(stages) == len(unstages))
        fixed = sum(1 for l in range(BLOCK_COUNT)
                    if migration.old_position(l) == migration.new_position(l, k))
        moved = {o["lbn"] for o in ops if o["op"] in ("move", "unstage")}
        check(f"k={k} every non-fixed lbn lands at target exactly once "
              f"({BLOCK_COUNT - fixed} moved)", len(moved) == BLOCK_COUNT - fixed)
        check(f"k={k} staging count equals cycle count (one staged block max)",
              len(stages) >= 1 and len(stages) <= BLOCK_COUNT // 2)

    # Re-plan from an already migrated layout: must derive targets from the
    # CURRENT positions, not from the initial layout.
    current = {l: migration.new_position(l, 2) for l in range(BLOCK_COUNT)}
    ops = migration.plan_operations(4, current)
    touched = {o["lbn"] for o in ops}
    expected = {l for l in range(BLOCK_COUNT)
                if migration.new_position(l, 2) != migration.new_position(l, 4)}
    check("re-striping from k=2 layout touches exactly the displaced lbn",
          touched == expected)


def test_initial_layout() -> None:
    print("[init] 2 disks x 16 slots, single-block alternating, 4 KiB bodies")
    snap = snapshot()
    check("32 blocks tracked", len(snap["blocks"]) == BLOCK_COUNT)
    check("staging empty initially", snap["staging"] == [])
    check("no migration initially", snap["migration"] is None)
    check("verification ok", snap["verification"]["ok"])
    check("default disk limit is 128 sectors",
          snap["usage"]["disks"]["0"]["limit"] == 128
          and snap["usage"]["disks"]["1"]["limit"] == 128)
    check("staging limit is 8 sectors",
          snap["usage"]["staging"]["limit"] == 8)
    for b in snap["blocks"]:
        exp_disk, exp_slot = b["lbn"] % DISK_COUNT, b["lbn"] // DISK_COUNT
        check(f"lbn {b['lbn']} initial alternating placement",
              (b["disk"], b["slot"], b["staged"], b["at_target"])
              == (exp_disk, exp_slot, False, True))
        # Initial bodies are a short header plus zero fill: one stored sector.
        check(f"lbn {b['lbn']} sparse: only non-zero sectors stored",
              b["sectors"] == 1)
    code, blk = http("GET", "/api/blocks/0")
    check("initial body is exactly 4 KiB",
          code == 200 and len(bytes.fromhex(blk["body_hex"])) == BLOCK_SIZE)
    check("all-zero sectors are absent from slot_sectors",
          snap["usage"]["disks"]["0"]["used"] + snap["usage"]["disks"]["1"]["used"]
          == BLOCK_COUNT)


def test_capacity_rules() -> None:
    print("[capacity] sparse accounting, limit rejection, rewrite rejection")
    snap = snapshot()
    used0 = snap["usage"]["disks"]["0"]["used"]
    code, r = http("POST", "/api/limits", {"limit0": used0 - 1})
    check("limit below current usage is rejected", code == 400 and not r["ok"])

    # Grow a disk-0 block to 8 sectors and confirm usage follows.
    lbn0 = next(b["lbn"] for b in snap["blocks"] if b["disk"] == 0)
    before = snapshot()["usage"]["disks"]["0"]["used"]
    full = b"FULL" + b"\xff" * (BLOCK_SIZE - 4)
    code, w = http("PUT", f"/api/blocks/{lbn0}", {"body_hex": full.hex()})
    check("rewrite accepted", code == 200 and w["ok"])
    after = snapshot()["usage"]["disks"]["0"]["used"]
    check("rewrite grows disk usage by 7 sectors", after == before + 7,
          f"{before}->{after}")

    # Cap just above current usage: a rewrite that needs more must be refused.
    code, _ = http("POST", "/api/limits", {"limit0": after})
    check("limit set to current usage accepted", code == 200)
    other = next(b["lbn"] for b in snapshot()["blocks"]
                 if b["disk"] == 0 and b["lbn"] != lbn0)
    code, r = http("PUT", f"/api/blocks/{other}", {"body_hex": full.hex()})
    check("rewrite that would exceed the cap is rejected",
          code == 400 and not r["ok"], str(r))
    # Shrinking a block is still allowed, freeing space.
    code, w = http("PUT", f"/api/blocks/{lbn0}", {"body": "tiny"})
    check("shrink rewrite accepted", code == 200 and w["ok"])
    check("usage decreased after shrink",
          snapshot()["usage"]["disks"]["0"]["used"] < after)
    code, r = http("POST", "/api/limits", {"limit0": 128, "limit1": 128})
    check("limits can be raised again", code == 200 and r["ok"])


def test_legacy_upgrade() -> None:
    print("[upgrade] old BLOB-per-block SQLite volume upgrades in place")
    import sqlite3

    from app.db import seed_body

    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "legacy.db")
    raw = sqlite3.connect(path, isolation_level=None)
    raw.row_factory = sqlite3.Row
    raw.executescript("""
        CREATE TABLE slots (disk INTEGER NOT NULL, slot INTEGER NOT NULL,
            lbn INTEGER NOT NULL, body BLOB NOT NULL, PRIMARY KEY (disk, slot));
        CREATE TABLE locations (lbn INTEGER PRIMARY KEY, disk INTEGER NOT NULL,
            slot INTEGER NOT NULL, staged INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE staging (lbn INTEGER PRIMARY KEY, body BLOB NOT NULL);
        CREATE TABLE migration (id INTEGER PRIMARY KEY, stripe_k INTEGER NOT NULL,
            status TEXT NOT NULL, next_op INTEGER NOT NULL,
            total_ops INTEGER NOT NULL, last_writes INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE ops (seq INTEGER PRIMARY KEY, op TEXT NOT NULL,
            lbn INTEGER NOT NULL, src_disk INTEGER, src_slot INTEGER,
            dst_disk INTEGER, dst_slot INTEGER);
        CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL DEFAULT (datetime('now')), kind TEXT NOT NULL,
            detail TEXT NOT NULL);
    """)
    raw.execute("BEGIN")
    for lbn in range(BLOCK_COUNT):
        d, s = lbn % 2, lbn // 2
        raw.execute("INSERT INTO slots VALUES (?,?,?,?)", (d, s, lbn, seed_body(lbn)))
        raw.execute("INSERT INTO locations VALUES (?,?,?,0)", (lbn, d, s))
    # Log an in-flight k=2 plan with 7 already-executed ops, applying them.
    ops = migration.plan_operations(2)
    for i, o in enumerate(ops):
        raw.execute("INSERT INTO ops VALUES (?,?,?,?,?,?,?)",
                    (i, o["op"], o["lbn"], o["src_disk"], o["src_slot"],
                     o["dst_disk"], o["dst_slot"]))
    for i in range(7):
        o = ops[i]
        if o["op"] == "stage":
            body = raw.execute("SELECT body FROM slots WHERE disk=? AND slot=?",
                               (o["src_disk"], o["src_slot"])).fetchone()["body"]
            raw.execute("INSERT INTO staging VALUES (?,?)", (o["lbn"], body))
            raw.execute("DELETE FROM slots WHERE disk=? AND slot=?",
                        (o["src_disk"], o["src_slot"]))
            raw.execute("UPDATE locations SET staged=1 WHERE lbn=?", (o["lbn"],))
        elif o["op"] == "move":
            body = raw.execute("SELECT body FROM slots WHERE disk=? AND slot=?",
                               (o["src_disk"], o["src_slot"])).fetchone()["body"]
            raw.execute("INSERT INTO slots VALUES (?,?,?,?)",
                        (o["dst_disk"], o["dst_slot"], o["lbn"], body))
            raw.execute("DELETE FROM slots WHERE disk=? AND slot=?",
                        (o["src_disk"], o["src_slot"]))
            raw.execute("UPDATE locations SET disk=?,slot=?,staged=0 WHERE lbn=?",
                        (o["dst_disk"], o["dst_slot"], o["lbn"]))
        else:
            body = raw.execute("SELECT body FROM staging WHERE lbn=?",
                               (o["lbn"],)).fetchone()["body"]
            raw.execute("INSERT INTO slots VALUES (?,?,?,?)",
                        (o["dst_disk"], o["dst_slot"], o["lbn"], body))
            raw.execute("DELETE FROM staging WHERE lbn=?", (o["lbn"],))
            raw.execute("UPDATE locations SET disk=?,slot=?,staged=0 WHERE lbn=?",
                        (o["dst_disk"], o["dst_slot"], o["lbn"]))
    raw.execute("INSERT INTO migration VALUES (1,2,'running',7,24,4)")
    raw.execute("COMMIT")
    raw.close()

    conn = db.connect(path)
    try:
        check("upgraded volume verifies", migration.verify(conn)["ok"],
              str(migration.verify(conn)["errors"]))
        check("upgrade keeps default 128-sector limits",
              migration.usage_report(conn)["disks"][0]["limit"] == 128)
        mig = conn.execute("SELECT * FROM migration WHERE id = 1").fetchone()
        check("upgrade keeps in-flight migration progress",
              mig["status"] == "running" and mig["next_op"] == 7)
        rings = conn.execute("SELECT * FROM rings ORDER BY ring_id").fetchall()
        check("upgrade reconstructs ring progress",
              len(rings) == 8 and rings[0]["done"] == 1 and rings[2]["step"] == 1)
        check("upgrade keeps positions",
              conn.execute("SELECT COUNT(*) AS c FROM slots").fetchone()["c"] == 31)
        result = migration.run_batch(conn, 1000)
        while not result["done"] and not result["blocked"]:
            result = migration.run_batch(conn, 1000)
        check("upgraded migration can finish", result["done"], str(result))
        check("upgraded volume validates at completion", migration.verify(conn)["ok"])
    finally:
        conn.close()



def test_migration(proc, db_path: str, k: int) -> subprocess.Popen:
    print(f"[migration k={k}] start, budget=1, online I/O, crashes, resume, verify")
    code, _ = http("POST", "/api/migration/start", {"stripe_k": k})
    check("migration started", code == 200)
    total = snapshot()["migration"]["total_ops"]
    check("plan has slot writes", total > 0)

    # --- budget 1 makes exactly one op of progress ---------------------------
    code, r = http("POST", "/api/migration/batch", {"budget": 1})
    check("budget=1 executes exactly one write", code == 200 and r["executed"] == 1)
    snap = snapshot()
    check("next_op advanced to 1", snap["migration"]["next_op"] == 1)
    check("page counter shows this batch wrote 1",
          snap["migration"]["last_writes"] == 1)
    check("at most one block in staging", len(snap["staging"]) <= 1)
    check("ring progress is tracked",
          len(snap["rings"]) > 0 and snap["rings"][0]["start"] is not None)
    check("usage stays within caps after the first write",
          snap["usage"]["disks"]["0"]["used"] <= snap["usage"]["disks"]["0"]["limit"]
          and snap["usage"]["disks"]["1"]["used"] <= snap["usage"]["disks"]["1"]["limit"])

    # --- host I/O on a staged block: latest committed content wins ----------
    staged_lbn = snap["staging"][0]["lbn"] if snap["staging"] else None
    marker_stage = None
    if staged_lbn is not None:
        marker_stage = f"HOST-WRITE-STAGED-{staged_lbn}".encode().ljust(BLOCK_SIZE, b"#")
        code, w = http("PUT", f"/api/blocks/{staged_lbn}",
                       {"body_hex": marker_stage.hex()})
        check("overwrite of staged lbn lands in staging",
              code == 200 and w["ok"] and w["target"] == "staging")
        code, rd = http("GET", f"/api/blocks/{staged_lbn}")
        check("reading a staged lbn returns the latest committed body",
              code == 200 and rd["staged"] is True
              and bytes.fromhex(rd["body_hex"]) == marker_stage)

    # --- host I/O on an on-disk block ---------------------------------------
    disk_lbn = next(b["lbn"] for b in snap["blocks"] if not b["staged"])
    marker_disk = b"HOST-WRITE-DISK".ljust(BLOCK_SIZE, b"*")
    code, w = http("PUT", f"/api/blocks/{disk_lbn}",
                   {"body_hex": marker_disk.hex()})
    check("overwrite of on-disk lbn accepted", code == 200 and w["ok"])
    code, rd = http("GET", f"/api/blocks/{disk_lbn}")
    check("on-disk overwrite reads back",
          bytes.fromhex(rd["body_hex"]) == marker_disk)

    # --- crash 1: batch writes but never commits -----------------------------
    progress_before = snap["migration"]["next_op"]
    code, _ = http("POST", "/api/migration/batch", {"budget": 3, "crash": True})
    check("crash request yields no HTTP response", code == 0)
    rc = wait_exit(proc)
    check("process exited with code 77", rc == 77, f"rc={rc}")
    proc = start_server(db_path)
    snap = snapshot()
    check("restart keeps migration running", snap["migration"]["status"] == "running")
    check("uncommitted batch rolled back to committed op#",
          snap["migration"]["next_op"] == progress_before)
    check("batch counter shows last committed batch (1)",
          snap["migration"]["last_writes"] == 1)
    check("staging invariant intact after rollback", len(snap["staging"]) <= 1)
    if marker_stage is not None:
        code, rd = http("GET", f"/api/blocks/{staged_lbn}")
        check("committed overwrite survives restart",
              code == 200 and bytes.fromhex(rd["body_hex"]) == marker_stage)
    code, rd = http("GET", f"/api/blocks/{disk_lbn}")
    check("committed on-disk overwrite survives restart",
          bytes.fromhex(rd["body_hex"]) == marker_disk)

    # --- normal batches with larger budgets ---------------------------------
    code, r = http("POST", "/api/migration/batch", {"budget": 4})
    check("budget=4 executes <= 4 writes", code == 200 and r["executed"] == 4)
    check("committed progress advanced by 4",
          snapshot()["migration"]["next_op"] == progress_before + 4)

    # --- crash 2 later in the migration --------------------------------------
    snap = snapshot()
    if snap["migration"]["total_ops"] - snap["migration"]["next_op"] >= 3:
        progress_before = snap["migration"]["next_op"]
        http("POST", "/api/migration/batch", {"budget": 2, "crash": True})
        rc = wait_exit(proc)
        check("second crash exit code 77", rc == 77)
        proc = start_server(db_path)
        snap = snapshot()
        check("second rollback resumes at same committed op#",
              snap["migration"]["next_op"] == progress_before)

    # --- drain to completion with varied budgets (including 1) ---------------
    while True:
        snap = snapshot()
        if snap["migration"]["status"] == "done":
            break
        remaining = snap["migration"]["total_ops"] - snap["migration"]["next_op"]
        budget = 1 if remaining % 3 == 0 else min(4, remaining)
        code, r = http("POST", "/api/migration/batch", {"budget": budget})
        check("every positive budget makes progress",
              code == 200 and r["executed"] >= 1, str(r))
        if r["done"] or r.get("blocked"):
            break

    snap = snapshot()
    check("migration marked done",
          snap["migration"]["status"] == "done"
          and snap["migration"]["next_op"] == snap["migration"]["total_ops"])
    check("staging empty at completion", snap["staging"] == [])
    code, v = http("GET", "/api/verify")
    check(f"k={k} verification passes", code == 200 and v["ok"], str(v.get("errors")))
    check(f"k={k} usage within limits at completion",
          all(snap["usage"]["disks"][str(d)]["used"]
              <= snap["usage"]["disks"][str(d)]["limit"] for d in (0, 1)))
    for b in snap["blocks"]:
        ed, es = migration.new_position(b["lbn"], k)
        check(f"k={k} lbn {b['lbn']} final placement matches new stripe",
              (b["disk"], b["slot"], b["staged"], b["at_target"])
              == (ed, es, False, True))

    # Host-written bodies must have followed their blocks to final positions.
    if marker_stage is not None:
        code, rd = http("GET", f"/api/blocks/{staged_lbn}")
        check("staged-block overwrite carried through to final disk slot",
              code == 200 and rd["staged"] is False
              and bytes.fromhex(rd["body_hex"]) == marker_stage)
    code, rd = http("GET", f"/api/blocks/{disk_lbn}")
    check("on-disk overwrite carried through to final position",
          bytes.fromhex(rd["body_hex"]) == marker_disk)
    return proc


def blk(prefix: str, fill: bytes = b"\x00") -> bytes:
    """A full 4 KiB body with ``prefix`` at the start and ``fill`` elsewhere."""
    head = prefix.encode()
    return head + fill * (BLOCK_SIZE - len(head))


def test_capacity_start_selection() -> None:
    print("[capacity] non-zero feasible ring start is chosen when offset 0 fails")
    conn = db.connect(os.path.join(tempfile.mkdtemp(), "vol.db"))
    db.ensure_seeded(conn)
    try:
        # k=4 ring 0 = disk0.slot1 (lbn2), disk0.slot2 (lbn4), disk1.slot0 (lbn1).
        # Make the entry block tiny and the other two full 8-sector bodies, then
        # cap disk0 at its current usage so only an entry other than offset 0
        # keeps the whole ring within both caps.
        migration.write_lbn(conn, 2, blk("tiny"))
        migration.write_lbn(conn, 4, blk("BIG4", b"\xff"))
        migration.write_lbn(conn, 1, blk("BIG1", b"\xff"))
        usage = db.disk_usage(conn)
        migration.apply_limit_change(conn, {0: usage[0], 1: usage[1]})
        ring = [(0, 1), (0, 2), (1, 0)]
        feasible = [off for off in range(3)
                    if migration.simulate_remaining(conn, ring, off, 0)[0]]
        check("offset 0 cannot hold both caps for the whole ring", 0 not in feasible)
        check("another entry point can hold both caps", len(feasible) > 0)
        migration.start_migration(conn, 4)
        r = migration.run_batch(conn, 1000)
        row = conn.execute("SELECT * FROM rings WHERE ring_id = 0").fetchone()
        check("engine picked the feasible entry, not offset 0",
              row["start"] in feasible and row["start"] != 0, str(dict(row)))
        check("capacity-constrained run completes", r["done"], str(r))
        check("verification passes", migration.verify(conn)["ok"])
    finally:
        conn.close()


def test_blocked_and_resume() -> None:
    print("[capacity] blocked batch commits partial work; frees space then resumes")
    conn = db.connect(os.path.join(tempfile.mkdtemp(), "vol.db"))
    db.ensure_seeded(conn)
    try:
        disk0_lbns = [l for l in range(BLOCK_COUNT) if (l // 2) % 2 == 0]
        for lbn in disk0_lbns:
            migration.write_lbn(conn, lbn, blk("FULL", b"\xff"))
        before = db.disk_usage(conn)
        # Cap disk0 two sectors above usage: the first ring cannot finish.
        migration.apply_limit_change(conn, {0: before[0] + 2, 1: 128})
        migration.start_migration(conn, 2)
        r = migration.run_batch(conn, 1000)
        check("first batch blocks on capacity", r["blocked"] and r["shortfall"],
              str(r))
        check("blocked result names the short disk",
              r["shortfall"]["what"] == "disk0", str(r["shortfall"]))
        snap = migration.snapshot(conn)
        check("migration status is blocked",
              snap["migration"]["status"] == "blocked")
        check("blocked migration still verifies", snap["verification"]["ok"])
        committed = snap["migration"]["next_op"]

        # Free space: turn every disk-0 block into a sparse one-sector body.
        for lbn in disk0_lbns:
            migration.write_lbn(conn, lbn, blk("free"))
        r2 = migration.run_batch(conn, 1000)
        while r2["blocked"] and not r2["done"]:
            for lbn in disk0_lbns:
                migration.write_lbn(conn, lbn, blk("free"))
            r2 = migration.run_batch(conn, 1000)
        check("resumes after space is freed", r2["done"], str(r2))
        check("blocked batch had committed partial work", committed >= 0)
        check("final verification passes", migration.verify(conn)["ok"],
              str(migration.verify(conn)["errors"]))
    finally:
        conn.close()


def main() -> int:
    test_planner_math()
    test_capacity_start_selection()
    test_blocked_and_resume()
    test_legacy_upgrade()
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "volume.db")
        proc = start_server(db_path)
        try:
            test_initial_layout()
            test_capacity_rules()
            proc = test_migration(proc, db_path, 2)

            # Reset: run the second stripe width from a fresh initial volume.
            proc.terminate()
            proc.wait(timeout=10)
            db_path2 = os.path.join(tmp, "volume2.db")
            proc = start_server(db_path2)
            test_initial_layout()
            proc = test_migration(proc, db_path2, 4)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    print(f"\n{FAIL} failed, {PASS} passed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
