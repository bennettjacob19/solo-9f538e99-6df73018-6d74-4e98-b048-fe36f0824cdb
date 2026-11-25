"""Re-striping planner and capacity-aware, crash-safe migration engine.

Permutation idea
----------------
Changing the stripe width only permutes the *positions* of the 32 logical
blocks: every old (disk, slot) has exactly one target (disk, slot).  A
permutation decomposes into cycles ("rings"); each ring is carried out with
one staging slot:

    stage block at the entry position -> hole there, block in staging
    move the remaining blocks inwards -> the hole walks around the ring
    unstage into the last hole        -> ring done

Space model
-----------
Each 4 KiB block is stored as up to eight 512-byte sectors.  All-zero sectors
are omitted, so a disk's usage is the count of actually stored sectors.  Both
disks have an operator-set sector limit and staging is accounted separately
(one block at most).

A ring may be entered at ANY of its positions.  Before a ring starts, the
engine looks for an entry for which every intermediate step of that ring keeps
both disks within their limits ("a start that holds both caps for the whole
ring").  If none exists, or an already-started ring would overflow on its next
step, the batch commits only the steps it could do and reports which disk is
short; after the operator frees space the migration resumes from the same
place.  Ring order is preserved and each batch honours its write budget.

Crash semantics
---------------
A batch runs inside one SQLite transaction.  ``simulate_crash`` kills the
process before COMMIT; the rollback journal then restores the previous slot
bodies, positions and (derived) usage on the next startup, while every already
committed batch survives.
"""

from __future__ import annotations

import json
import os
import sqlite3

from . import db
from .db import (BLOCK_COUNT, DISK_COUNT, SLOTS_PER_DISK,
                 STAGING_SECTOR_LIMIT)


class CapacityError(ValueError):
    """A disk or staging write would exceed the configured sector limit."""


# --------------------------------------------------------------------------- #
# Layout math
# --------------------------------------------------------------------------- #

def old_position(lbn: int) -> tuple[int, int]:
    """Initial single-block alternating layout for logical block ``lbn``."""
    return lbn % DISK_COUNT, lbn // DISK_COUNT


def new_position(lbn: int, stripe_k: int) -> tuple[int, int]:
    """Target layout for run-length-``stripe_k`` alternating stripes."""
    if stripe_k == 1:
        return old_position(lbn)
    group, within = divmod(lbn, stripe_k)
    disk = group % DISK_COUNT
    slot = (group // DISK_COUNT) * stripe_k + within
    return disk, slot


def _pos_index(disk: int, slot: int) -> int:
    return disk * SLOTS_PER_DISK + slot


def _pos_parts(index: int) -> tuple[int, int]:
    return divmod(index, SLOTS_PER_DISK)


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #

def plan_rings(stripe_k: int,
               current: dict[int, tuple[int, int]] | None = None
               ) -> list[list[tuple[int, int]]]:
    """Decompose the re-striping permutation into position cycles.

    Each ring lists positions in target order: the block at ``ring[i]`` must
    end up at ``ring[(i + 1) % len]``.  Rings are ordered by their lowest
    position index.
    """
    if current is None:
        current = {lbn: old_position(lbn) for lbn in range(BLOCK_COUNT)}

    target: dict[int, int] = {}
    for lbn, (disk, slot) in current.items():
        src = _pos_index(disk, slot)
        new_d, new_s = new_position(lbn, stripe_k)
        target[src] = _pos_index(new_d, new_s)

    visited: set[int] = set()
    rings: list[list[tuple[int, int]]] = []
    for start in range(BLOCK_COUNT):
        if start in visited or target[start] == start:
            visited.add(start)
            continue
        cycle: list[int] = []
        p = start
        while p not in visited:
            visited.add(p)
            cycle.append(p)
            p = target[p]
        rings.append([_pos_parts(pos) for pos in cycle])
    return rings


def plan_operations(stripe_k: int,
                    current: dict[int, tuple[int, int]] | None = None
                    ) -> list[dict]:
    """Static plan using each ring's first position (inspection/tests only)."""
    if current is None:
        current = {lbn: old_position(lbn) for lbn in range(BLOCK_COUNT)}
    occupant = {_pos_index(d, s): lbn for lbn, (d, s) in current.items()}
    ops: list[dict] = []
    for ring in plan_rings(stripe_k, current):
        n = len(ring)
        first_d, first_s = ring[0]
        first_lbn = occupant[_pos_index(first_d, first_s)]
        ops.append(dict(op="stage", lbn=first_lbn, src_disk=first_d,
                        src_slot=first_s, dst_disk=None, dst_slot=None))
        for i in range(n - 1, 0, -1):
            src_d, src_s = ring[i]
            dst_d, dst_s = ring[(i + 1) % n]
            ops.append(dict(op="move", lbn=occupant[_pos_index(src_d, src_s)],
                            src_disk=src_d, src_slot=src_s,
                            dst_disk=dst_d, dst_slot=dst_s))
        ud, us = ring[1]
        ops.append(dict(op="unstage", lbn=first_lbn, src_disk=None,
                        src_slot=None, dst_disk=ud, dst_slot=us))
    return ops


# --------------------------------------------------------------------------- #
# Ring execution model
# --------------------------------------------------------------------------- #
#
# A ring entered at offset ``start`` performs, in order:
#
#   stage   ring[start] -> staging                (staging gains the body)
#   move    ring[(s-1) % n] -> ring[s]            (hole walks backwards)
#   ...
#   move    ring[(s+1) % n] -> ring[(s+2) % n]
#   unstage staging -> ring[(s+1) % n]            (staging empties)
#
# i.e. the blocks are written in ring order s, s-1, ..., s+1 (mod n).  Each
# step writes one body at a destination and (except the stage) drops the body
# from the source position.

def _ordered_indices(ring_len: int, start: int) -> list[int]:
    return [start] + [(start - k) % ring_len for k in range(1, ring_len)]


def ring_steps(ring: list[tuple[int, int]], start: int,
               sizes: list[int]) -> list[dict]:
    """Concrete steps for a ring entry, with disk deltas per step.

    ``sizes[i]`` is the stored-sector count of the block currently at
    ``ring[i]``.  Every step writes one body (budget unit).
    """
    n = len(ring)
    order = _ordered_indices(n, start)
    steps = [{"phase": "stage", "src_pos": order[0], "dst_pos": None}]
    for k in range(1, n):
        steps.append({"phase": "move", "src_pos": order[k],
                      "dst_pos": order[k - 1]})
    steps.append({"phase": "unstage", "src_pos": None,
                  "dst_pos": order[-1]})
    for step in steps:
        # The body written by a step is the block that started at that ring
        # position; the unstage writes the block taken from ``order[0]``.
        source_pos = step["src_pos"] if step["src_pos"] is not None else order[0]
        step["sectors"] = sizes[source_pos]
    return steps


def _load_ring(row: sqlite3.Row) -> list[tuple[int, int]]:
    return [tuple(pos) for pos in json.loads(row["positions"])]


def _ring_row(conn: sqlite3.Connection, ring_id: int) -> sqlite3.Row:
    return conn.execute("SELECT * FROM rings WHERE ring_id = ?",
                        (ring_id,)).fetchone()


def _ring_step_sizes(conn: sqlite3.Connection, ring: list[tuple[int, int]],
                     steps: list[dict], start: int, from_step: int
                     ) -> list[int]:
    """Live stored-sector count for each yet-to-run step's written body."""
    sizes = []
    for offset, step in enumerate(steps, start=from_step):
        if step["phase"] == "unstage":
            if from_step > 0:
                # The block was staged by an earlier (committed) step.
                sizes.append(db.staging_usage(conn))
            else:
                d, s = ring[start]
                sizes.append(db.slot_sector_count(conn, d, s))
        else:
            d, s = ring[step["src_pos"]]
            sizes.append(db.slot_sector_count(conn, d, s))
    return sizes


def simulate_remaining(conn: sqlite3.Connection, ring: list[tuple[int, int]],
                       start: int, from_step: int) -> tuple[bool, dict | None]:
    """Check every remaining step of a ring from the live usage.

    Returns ``(ok, shortfall)``; ``shortfall`` describes the first blocked
    write (which disk/staging slot is short, by how much).
    """
    n = len(ring)
    all_steps = ring_steps(ring, start, [0] * n)
    steps = all_steps[from_step:]
    usage = db.disk_usage(conn)
    limits = db.disk_limits(conn)
    staged = db.staging_usage(conn) if from_step > 0 else 0
    sizes = _ring_step_sizes(conn, ring, steps, start, from_step)
    disks = {0: usage.get(0, 0), 1: usage.get(1, 0)}
    for step, sectors in zip(steps, sizes):
        if step["phase"] == "stage":
            if staged + sectors > STAGING_SECTOR_LIMIT:
                return False, {"what": "staging",
                               "need": staged + sectors,
                               "limit": STAGING_SECTOR_LIMIT,
                               "phase": "stage"}
            disks[ring[step["src_pos"]][0]] -= sectors
            staged += sectors
        elif step["phase"] == "move":
            src_d = ring[step["src_pos"]][0]
            dst_d = ring[step["dst_pos"]][0]
            if src_d == dst_d:
                # Same disk: the destination is a hole left by the previous
                # step and the source is freed first, so usage is unchanged.
                continue
            if disks[dst_d] + sectors > limits[dst_d]:
                return False, {"what": f"disk{dst_d}",
                               "need": disks[dst_d] + sectors,
                               "limit": limits[dst_d],
                               "phase": "move"}
            disks[dst_d] += sectors
            disks[src_d] -= sectors
        else:
            dst_d = ring[step["dst_pos"]][0]
            if disks[dst_d] + sectors > limits[dst_d]:
                return False, {"what": f"disk{dst_d}",
                               "need": disks[dst_d] + sectors,
                               "limit": limits[dst_d],
                               "phase": "unstage"}
            disks[dst_d] += sectors
            staged -= sectors
    return True, None


def choose_start(conn: sqlite3.Connection, ring: list[tuple[int, int]]
                 ) -> int | None:
    """First ring entry (in ring order) whose whole run stays within both caps."""
    if db.staging_usage(conn) != 0:
        raise RuntimeError("cannot choose a ring start while a block is staged")
    for offset in range(len(ring)):
        ok, _ = simulate_remaining(conn, ring, offset, 0)
        if ok:
            return offset
    return None


# --------------------------------------------------------------------------- #
# Migration lifecycle
# --------------------------------------------------------------------------- #

def start_migration(conn: sqlite3.Connection, stripe_k: int) -> None:
    existing = conn.execute("SELECT status FROM migration WHERE id = 1").fetchone()
    if existing is not None and existing["status"] in ("running", "blocked"):
        raise RuntimeError("a migration is already in progress")
    if stripe_k not in (2, 4):
        raise ValueError("stripe width must be 2 or 4")
    if conn.execute("SELECT COUNT(*) AS c FROM staging_block").fetchone()["c"]:
        raise RuntimeError("cannot start a new migration while a block is staged")

    rows = conn.execute("SELECT lbn, disk, slot FROM locations").fetchall()
    current = {r["lbn"]: (r["disk"], r["slot"]) for r in rows}
    if len(current) != BLOCK_COUNT:
        raise RuntimeError("volume is not fully populated; cannot plan migration")
    rings = plan_rings(stripe_k, current)
    total_ops = sum(len(ring) + 1 for ring in rings)

    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM migration")
        conn.execute("DELETE FROM rings")
        for ring_id, ring in enumerate(rings):
            conn.execute(
                "INSERT INTO rings(ring_id, positions, start, step, done)"
                " VALUES (?, ?, NULL, 0, 0)",
                (ring_id, json.dumps(ring)),
            )
        conn.execute(
            "INSERT INTO migration(id, stripe_k, status, next_op, total_ops,"
            " last_writes, error, next_ring)"
            " VALUES (1, ?, 'running', 0, ?, 0, NULL, 0)",
            (stripe_k, total_ops),
        )
        conn.execute(
            "INSERT INTO events(kind, detail) VALUES ('migration_start', ?)",
            (f"re-stripe to run-length {stripe_k}; {len(rings)} rings,"
             f" {total_ops} slot writes planned",),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _set_blocked(conn: sqlite3.Connection, shortfall: dict) -> None:
    what = shortfall["what"]
    label = "临时槽" if what == "staging" else f"盘{what[-1]}"
    msg = (f"{label} 空间不足：需 {shortfall['need']} 扇区，"
           f"上限 {shortfall['limit']} 扇区；腾出空间后可续搬")
    conn.execute("UPDATE migration SET status = 'blocked', error = ? WHERE id = 1",
                 (msg,))
    conn.execute("INSERT INTO events(kind, detail) VALUES ('migration_blocked', ?)",
                 (msg,))


# --------------------------------------------------------------------------- #
# Step execution
# --------------------------------------------------------------------------- #

def _execute_step(conn: sqlite3.Connection, ring: list[tuple[int, int]],
                  start: int, step_index: int) -> dict:
    n = len(ring)
    order = _ordered_indices(n, start)
    if step_index == 0:
        src_d, src_s = ring[order[0]]
        body = db.slot_body(conn, src_d, src_s)
        if body is None:
            raise RuntimeError(f"stage source empty at disk{src_d}.{src_s}")
        sectors = db.count_body_sectors(body)
        if db.staging_usage(conn) + sectors > STAGING_SECTOR_LIMIT:
            raise CapacityError("staging slot is out of sector space")
        lbn = conn.execute(
            "SELECT lbn FROM slots WHERE disk = ? AND slot = ?",
            (src_d, src_s),
        ).fetchone()["lbn"]
        db.write_staging_body(conn, body)
        conn.execute("INSERT OR REPLACE INTO staging_block(id, lbn) VALUES (1, ?)", (lbn,))
        conn.execute("DELETE FROM slots WHERE disk = ? AND slot = ?",
                     (src_d, src_s))
        conn.execute("UPDATE locations SET staged = 1 WHERE lbn = ?", (lbn,))
        return {"phase": "stage", "lbn": lbn, "sectors": sectors,
                "src": f"disk{src_d}.{src_s}", "dst": "staging"}

    if step_index == n:
        dst_d, dst_s = ring[order[n - 1]]
        row = conn.execute("SELECT lbn FROM staging_block").fetchone()
        if row is None:
            raise RuntimeError("unstage requested but staging is empty")
        lbn = row["lbn"]
        body = db.staging_body(conn, lbn)
        if body is None:
            raise RuntimeError(f"lbn {lbn} marked staged but body missing")
        sectors = db.count_body_sectors(body)
        usage = db.disk_usage(conn)
        limits = db.disk_limits(conn)
        if usage[dst_d] + sectors > limits[dst_d]:
            raise CapacityError(
                f"盘{dst_d} 空间不足：需 {usage[dst_d] + sectors} 扇区，"
                f"上限 {limits[dst_d]} 扇区")
        if conn.execute("SELECT 1 FROM slots WHERE disk = ? AND slot = ?",
                        (dst_d, dst_s)).fetchone() is not None:
            raise RuntimeError(f"unstage destination occupied at disk{dst_d}.{dst_s}")
        conn.execute("INSERT INTO slots(disk, slot, lbn) VALUES (?, ?, ?)",
                     (dst_d, dst_s, lbn))
        db.write_slot_body(conn, dst_d, dst_s, body)
        conn.execute("DELETE FROM staging_sectors")
        conn.execute("DELETE FROM staging_block")
        conn.execute(
            "UPDATE locations SET disk = ?, slot = ?, staged = 0 WHERE lbn = ?",
            (dst_d, dst_s, lbn))
        return {"phase": "unstage", "lbn": lbn, "sectors": sectors,
                "src": "staging", "dst": f"disk{dst_d}.{dst_s}"}

    src_d, src_s = ring[order[step_index]]
    dst_d, dst_s = ring[order[step_index - 1]]
    src_row = conn.execute(
        "SELECT lbn FROM slots WHERE disk = ? AND slot = ?", (src_d, src_s)
    ).fetchone()
    if src_row is None:
        raise RuntimeError(f"move source empty at disk{src_d}.{src_s}")
    lbn = src_row["lbn"]
    body = db.slot_body(conn, src_d, src_s)
    sectors = db.count_body_sectors(body)
    usage = db.disk_usage(conn)
    limits = db.disk_limits(conn)
    if src_d != dst_d and usage[dst_d] + sectors > limits[dst_d]:
        raise CapacityError(
            f"盘{dst_d} 空间不足：需 {usage[dst_d] + sectors} 扇区，"
            f"上限 {limits[dst_d]} 扇区")
    if conn.execute("SELECT 1 FROM slots WHERE disk = ? AND slot = ?",
                    (dst_d, dst_s)).fetchone() is not None:
        raise RuntimeError(f"move destination occupied at disk{dst_d}.{dst_s}")
    # Free the source before writing the target so a same-disk move does not
    # transiently double-count the body.
    conn.execute("DELETE FROM slots WHERE disk = ? AND slot = ?", (src_d, src_s))
    conn.execute("INSERT INTO slots(disk, slot, lbn) VALUES (?, ?, ?)",
                 (dst_d, dst_s, lbn))
    db.write_slot_body(conn, dst_d, dst_s, body)
    conn.execute(
        "UPDATE locations SET disk = ?, slot = ?, staged = 0 WHERE lbn = ?",
        (dst_d, dst_s, lbn))
    return {"phase": "move", "lbn": lbn, "sectors": sectors,
            "src": f"disk{src_d}.{src_s}", "dst": f"disk{dst_d}.{dst_s}"}


def _next_step_fits(conn: sqlite3.Connection, ring: list[tuple[int, int]],
                    start: int, step_index: int) -> tuple[bool, dict | None]:
    """Whether the single next step can be written under the current caps."""
    n = len(ring)
    order = _ordered_indices(n, start)
    usage = db.disk_usage(conn)
    limits = db.disk_limits(conn)
    staged = db.staging_usage(conn)
    if step_index == 0:
        d, s = ring[order[0]]
        sectors = db.slot_sector_count(conn, d, s)
        if staged + sectors > STAGING_SECTOR_LIMIT:
            return False, {"what": "staging", "need": staged + sectors,
                           "limit": STAGING_SECTOR_LIMIT, "phase": "stage"}
        return True, None
    if step_index == n:
        dst_d = ring[order[n - 1]][0]
        sectors = staged
        if usage[dst_d] + sectors > limits[dst_d]:
            return False, {"what": f"disk{dst_d}",
                           "need": usage[dst_d] + sectors,
                           "limit": limits[dst_d], "phase": "unstage"}
        return True, None
    src_d, src_s = ring[order[step_index]]
    dst_d = ring[order[step_index - 1]][0]
    sectors = db.slot_sector_count(conn, src_d, src_s)
    if src_d != dst_d and usage[dst_d] + sectors > limits[dst_d]:
        return False, {"what": f"disk{dst_d}",
                       "need": usage[dst_d] + sectors,
                       "limit": limits[dst_d], "phase": "move"}
    return True, None


def _describe_no_start(conn: sqlite3.Connection, ring: list[tuple[int, int]]
                       ) -> dict:
    """Pick the most telling shortfall when no entry point fits the whole ring."""
    worst = None
    for offset in range(len(ring)):
        _, short = simulate_remaining(conn, ring, offset, 0)
        if short is None:
            continue
        over = short["need"] - short["limit"]
        if worst is None or over > worst[0]:
            worst = (over, short)
    if worst is not None:
        return worst[1]
    usage = db.disk_usage(conn)
    limits = db.disk_limits(conn)
    disk = min((0, 1), key=lambda d: limits[d] - usage[d])
    return {"what": f"disk{disk}", "need": usage[disk],
            "limit": limits[disk], "phase": "start"}


def run_batch(conn: sqlite3.Connection, budget: int,
              simulate_crash: bool = False) -> dict:
    """Execute up to ``budget`` slot-body writes as one transaction."""
    if not isinstance(budget, int) or budget < 1:
        raise ValueError("write budget must be a positive integer")
    mig = conn.execute("SELECT * FROM migration WHERE id = 1").fetchone()
    if mig is None:
        raise RuntimeError("no migration in progress; start one first")
    if mig["status"] == "done":
        raise RuntimeError("migration already complete")

    conn.execute("BEGIN IMMEDIATE")
    try:
        executed = 0
        next_op = mig["next_op"]
        next_ring = mig["next_ring"]
        ring_count = conn.execute(
            "SELECT COUNT(*) AS c FROM rings").fetchone()["c"]
        stopped = None

        while executed < budget and next_ring < ring_count:
            ring_row = _ring_row(conn, next_ring)
            ring = _load_ring(ring_row)
            n = len(ring)
            start = ring_row["start"]
            step = ring_row["step"]

            if start is None:
                chosen = choose_start(conn, ring)
                if chosen is None:
                    stopped = _describe_no_start(conn, ring)
                    break
                start = chosen
                step = 0
                conn.execute(
                    "UPDATE rings SET start = ?, step = 0 WHERE ring_id = ?",
                    (start, next_ring))
                conn.execute(
                    "INSERT INTO events(kind, detail) VALUES ('ring_start', ?)",
                    (f"ring #{next_ring} entered at offset {start}"
                     f" (of {n} positions)",),
                )

            while step < n + 1 and executed < budget:
                fits, short = _next_step_fits(conn, ring, start, step)
                if not fits:
                    stopped = short
                    break
                info = _execute_step(conn, ring, start, step)
                step += 1
                executed += 1
                next_op += 1
                conn.execute(
                    "INSERT INTO events(kind, detail) VALUES ('slot_write', ?)",
                    (f"ring #{next_ring} step {step}/{n + 1} {info['phase']}"
                     f" lbn={info['lbn']} {info['src']}->{info['dst']}"
                     f" ({info['sectors']} sectors)",),
                )

            conn.execute("UPDATE rings SET step = ?, done = ? WHERE ring_id = ?",
                         (step, 1 if step >= n + 1 else 0, next_ring))
            if stopped is not None and step < n + 1:
                break
            if step < n + 1:
                break  # budget exhausted mid-ring; keep this ring
            next_ring += 1

        done = next_ring >= ring_count
        if stopped is not None and not done:
            _set_blocked(conn, stopped)
        else:
            conn.execute(
                "UPDATE migration SET status = ?, error = NULL WHERE id = 1",
                ("done" if done else "running",))

        conn.execute(
            "UPDATE migration SET next_op = ?, next_ring = ?, last_writes = ?"
            " WHERE id = 1",
            (next_op, next_ring, executed),
        )
        if done:
            conn.execute(
                "INSERT INTO events(kind, detail) VALUES ('migration_done', ?)",
                (f"re-striping to run-length {mig['stripe_k']} complete after"
                 f" {next_op} slot writes",),
            )
        elif executed:
            conn.execute(
                "INSERT INTO events(kind, detail) VALUES ('batch_commit', ?)",
                (f"batch wrote {executed} slot bodies (next op #{next_op})",),
            )

        if simulate_crash:
            # Deliberately exit BEFORE commit: the rollback journal restores
            # the pre-batch bodies, positions and derived usage on reopen.
            os._exit(77)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    return {"executed": executed, "next_op": next_op, "done": done,
            "stripe_k": mig["stripe_k"], "blocked": stopped is not None,
            "shortfall": stopped, "next_ring": next_ring,
            "error": _blocked_message(stopped) if stopped else None}


def _blocked_message(shortfall: dict) -> str:
    what = shortfall["what"]
    label = "临时槽" if what == "staging" else f"盘{what[-1]}"
    return (f"{label} 空间不足：需 {shortfall['need']} 扇区，"
            f"上限 {shortfall['limit']} 扇区；腾出空间后可续搬")


# --------------------------------------------------------------------------- #
# Capacity limits / usage
# --------------------------------------------------------------------------- #

def apply_limit_change(conn: sqlite3.Connection, limits: dict[int, int]) -> None:
    """Set per-disk sector caps, rejecting a cap below current usage."""
    usage = db.disk_usage(conn)
    for disk, limit in limits.items():
        if disk not in (0, 1):
            raise ValueError(f"unknown disk {disk}")
        if not isinstance(limit, int) or limit < 0:
            raise ValueError("sector limit must be a non-negative integer")
        if limit < usage[disk]:
            raise CapacityError(
                f"盘{disk} 上限 {limit} 扇区低于当前占用 {usage[disk]} 扇区")
    conn.execute("BEGIN IMMEDIATE")
    try:
        for disk, limit in limits.items():
            conn.execute("UPDATE disks SET sector_limit = ? WHERE disk = ?",
                         (limit, disk))
        detail = ", ".join(f"盘{d}={lim}" for d, lim in sorted(limits.items()))
        conn.execute("INSERT INTO events(kind, detail) VALUES ('limit_change', ?)",
                     (f"sector limits set to {detail}",))
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def usage_report(conn: sqlite3.Connection) -> dict:
    usage = db.disk_usage(conn)
    limits = db.disk_limits(conn)
    staged = db.staging_usage(conn)
    return {
        "disks": {
            disk: {"used": usage[disk], "limit": limits[disk],
                   "free": limits[disk] - usage[disk]}
            for disk in range(DISK_COUNT)
        },
        "staging": {"used": staged, "limit": STAGING_SECTOR_LIMIT,
                    "free": STAGING_SECTOR_LIMIT - staged},
    }


# --------------------------------------------------------------------------- #
# Logical I/O during migration
# --------------------------------------------------------------------------- #

def read_lbn(conn: sqlite3.Connection, lbn: int) -> dict:
    loc = conn.execute("SELECT * FROM locations WHERE lbn = ?", (lbn,)).fetchone()
    if loc is None:
        raise KeyError(f"unknown lbn {lbn}")
    if loc["staged"]:
        body = db.staging_body(conn, lbn)
        if body is None:
            raise RuntimeError(f"lbn {lbn} marked staged but staging row missing")
        sectors = db.staging_usage(conn)
        return {"lbn": lbn, "disk": loc["disk"], "slot": loc["slot"],
                "staged": True, "body": body, "sectors": sectors}
    body = db.slot_body(conn, loc["disk"], loc["slot"])
    if body is None:
        raise RuntimeError(f"lbn {lbn} location points at an empty slot")
    sectors = db.slot_sector_count(conn, loc["disk"], loc["slot"])
    return {"lbn": lbn, "disk": loc["disk"], "slot": loc["slot"],
            "staged": False, "body": body, "sectors": sectors}


def write_lbn(conn: sqlite3.Connection, lbn: int, body: bytes) -> dict:
    """Host-style overwrite; immediately committed, not counted in batch budget."""
    if len(body) != 4096:
        raise ValueError("block body must be exactly 4096 bytes")
    new_sectors = db.count_body_sectors(body)
    loc = conn.execute("SELECT * FROM locations WHERE lbn = ?", (lbn,)).fetchone()
    if loc is None:
        raise KeyError(f"unknown lbn {lbn}")

    conn.execute("BEGIN IMMEDIATE")
    try:
        if loc["staged"]:
            old_sectors = db.staging_usage(conn)
            if new_sectors > STAGING_SECTOR_LIMIT:
                raise CapacityError(
                    f"临时槽空间不足：需 {new_sectors} 扇区，"
                    f"上限 {STAGING_SECTOR_LIMIT} 扇区")
            db.write_staging_body(conn, body)
            where = "staging"
            delta = ""
        else:
            usage = db.disk_usage(conn)
            limits = db.disk_limits(conn)
            old_sectors = db.slot_sector_count(conn, loc["disk"], loc["slot"])
            projected = usage[loc["disk"]] - old_sectors + new_sectors
            if projected > limits[loc["disk"]]:
                raise CapacityError(
                    f"盘{loc['disk']} 空间不足：改写后需 {projected} 扇区，"
                    f"上限 {limits[loc['disk']]} 扇区")
            db.write_slot_body(conn, loc["disk"], loc["slot"], body)
            where = f"disk{loc['disk']}.slot{loc['slot']}"
            delta = f" ({old_sectors}->{new_sectors} sectors)"
        conn.execute(
            "INSERT INTO events(kind, detail) VALUES ('host_write', ?)",
            (f"lbn={lbn} overwrite committed at {where}{delta}",),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    return {"lbn": lbn, "target": where, "committed": True,
            "sectors": new_sectors}


# --------------------------------------------------------------------------- #
# Verification / snapshot
# --------------------------------------------------------------------------- #

def verify(conn: sqlite3.Connection) -> dict:
    errors: list[str] = []
    mig = conn.execute("SELECT * FROM migration WHERE id = 1").fetchone()
    stripe_k = mig["stripe_k"] if mig else 1

    slot_rows = conn.execute("SELECT * FROM slots").fetchall()
    staged_count = conn.execute(
        "SELECT COUNT(*) AS c FROM staging_block").fetchone()["c"]
    if len(slot_rows) + staged_count != BLOCK_COUNT:
        errors.append(
            f"copies accounted: {len(slot_rows)} slots + {staged_count} staged"
            f" != {BLOCK_COUNT}")
    lbns = [r["lbn"] for r in slot_rows]
    staged_lbns = [r["lbn"] for r in conn.execute("SELECT lbn FROM staging_block")]
    if sorted(lbns + staged_lbns) != list(range(BLOCK_COUNT)):
        errors.append("slots + staging do not hold exactly one copy of every lbn")
    if staged_count > 1:
        errors.append(f"staging holds {staged_count} blocks (limit is 1)")
    if mig is not None and mig["status"] == "done" and staged_count != 0:
        errors.append("migration done but staging slot not empty")
    flagged = {r["lbn"] for r in conn.execute(
        "SELECT lbn FROM locations WHERE staged = 1")}
    if flagged != set(staged_lbns):
        errors.append(f"staging flags mismatch: locations {sorted(flagged)}"
                      f" vs staging {sorted(staged_lbns)}")

    # Every block must be readable as a full 4 KiB body.
    for lbn in range(BLOCK_COUNT):
        loc = conn.execute("SELECT * FROM locations WHERE lbn = ?",
                           (lbn,)).fetchone()
        if loc is None:
            errors.append(f"lbn {lbn} missing from locations")
            continue
        try:
            r = read_lbn(conn, lbn)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"lbn {lbn} unreadable: {exc}")
            continue
        if len(r["body"]) != 4096:
            errors.append(f"lbn {lbn} body is {len(r['body'])} bytes, not 4096")
        if mig is not None and mig["status"] == "done":
            exp_d, exp_s = new_position(lbn, stripe_k)
            if (loc["disk"], loc["slot"]) != (exp_d, exp_s) or loc["staged"]:
                errors.append(
                    f"lbn {lbn} at disk{loc['disk']}.slot{loc['slot']}"
                    f" staged={loc['staged']}, expected disk{exp_d}.slot{exp_s}")

    # Capacity invariants.
    usage = db.disk_usage(conn)
    limits = db.disk_limits(conn)
    for disk in range(DISK_COUNT):
        if usage[disk] > limits[disk]:
            errors.append(f"disk{disk} usage {usage[disk]} exceeds limit"
                          f" {limits[disk]}")
    if db.staging_usage(conn) > STAGING_SECTOR_LIMIT:
        errors.append(f"staging usage {db.staging_usage(conn)} exceeds limit"
                      f" {STAGING_SECTOR_LIMIT}")
    # Sparse model: no stored sector may be all zero.
    for r in conn.execute("SELECT disk, slot, sector, body FROM slot_sectors"):
        if len(r["body"]) != db.SECTOR_SIZE:
            errors.append(f"disk{r['disk']}.slot{r['slot']} sector {r['sector']}"
                          f" is {len(r['body'])} bytes")
        elif not any(r["body"]):
            errors.append(f"disk{r['disk']}.slot{r['slot']} sector {r['sector']}"
                          f" stored although all-zero")
    for r in conn.execute("SELECT sector, body FROM staging_sectors"):
        if len(r["body"]) != db.SECTOR_SIZE or not any(r["body"]):
            errors.append(f"staging sector {r['sector']} invalid")

    return {"ok": not errors, "errors": errors, "stripe_k": stripe_k,
            "usage": usage_report(conn)}


def snapshot(conn: sqlite3.Connection) -> dict:
    mig = conn.execute("SELECT * FROM migration WHERE id = 1").fetchone()
    stripe_k = mig["stripe_k"] if mig else 1
    usage = db.disk_usage(conn)
    limits = db.disk_limits(conn)
    loc_rows = conn.execute("SELECT * FROM locations ORDER BY lbn").fetchall()
    blocks = []
    for loc in loc_rows:
        exp_d, exp_s = new_position(loc["lbn"], stripe_k)
        if loc["staged"]:
            sectors = db.staging_usage(conn)
        else:
            sectors = db.slot_sector_count(conn, loc["disk"], loc["slot"])
        blocks.append({
            "lbn": loc["lbn"],
            "disk": loc["disk"],
            "slot": loc["slot"],
            "staged": bool(loc["staged"]),
            "sectors": sectors,
            "expected_disk": exp_d,
            "expected_slot": exp_s,
            "at_target": (loc["disk"], loc["slot"], bool(loc["staged"]))
                          == (exp_d, exp_s, False),
        })
    # Ring progress for the console.
    rings = []
    for row in conn.execute("SELECT * FROM rings ORDER BY ring_id"):
        n = len(json.loads(row["positions"]))
        rings.append({"ring_id": row["ring_id"], "positions": json.loads(row["positions"]),
                      "start": row["start"], "step": row["step"],
                      "total_steps": n + 1, "done": bool(row["done"])})
    staging = [dict(r) for r in conn.execute("SELECT lbn FROM staging_block")]
    events = [dict(r) for r in conn.execute(
        "SELECT id, ts, kind, detail FROM events ORDER BY id DESC LIMIT 25")]
    return {
        "migration": dict(mig) if mig else None,
        "stripe_k": stripe_k,
        "blocks": blocks,
        "rings": rings,
        "staging": staging,
        "usage": {
            "disks": {d: {"used": usage[d], "limit": limits[d],
                          "free": limits[d] - usage[d]}
                      for d in range(DISK_COUNT)},
            "staging": {"used": db.staging_usage(conn),
                        "limit": STAGING_SECTOR_LIMIT,
                        "free": STAGING_SECTOR_LIMIT - db.staging_usage(conn)},
        },
        "events": events,
        "verification": verify(conn),
    }
