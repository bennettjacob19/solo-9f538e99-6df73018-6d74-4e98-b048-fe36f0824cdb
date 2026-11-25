"""SQLite storage for the simulated two-disk virtual volume.

Physical model
--------------
* Two disks (disk 0 / disk 1), each with 16 slots -> 32 data slots total.
* Every slot holds one 4 KiB logical block.  The volume initially fills every
  slot.
* A 4 KiB block is split into eight 512-byte sectors.  An all-zero sector is
  **not stored at all**; reading a block zero-fills the missing sectors.  Disk
  space is therefore charged per actually stored sector.
* The volume owns ONE extra staging slot (tables ``staging_block`` /
  ``staging_sectors``) that can hold at most one block body.  Its sector usage
  is accounted separately from both disks; it is not part of the 32 regular
  disk slots.

Schema upgrade
--------------
Databases written by the earlier "one BLOB per block" version are upgraded in
place: every body is split into stored sectors, while physical positions and
the logged migration progress are preserved.  Upgraded volumes get the default
limit of 128 sectors per disk (16 slots x 8 sectors).
"""

from __future__ import annotations

import os
import sqlite3

BLOCK_SIZE = 4096
SECTOR_SIZE = 512
SECTORS_PER_BLOCK = BLOCK_SIZE // SECTOR_SIZE  # 8
DISK_COUNT = 2
SLOTS_PER_DISK = 16
BLOCK_COUNT = DISK_COUNT * SLOTS_PER_DISK  # 32
SECTORS_PER_DISK = SLOTS_PER_DISK * SECTORS_PER_BLOCK  # 128

# 16 slots x 8 sectors: even a completely non-zero volume fits.
DEFAULT_DISK_SECTOR_LIMIT = SECTORS_PER_DISK  # 128
# Staging holds one block and is accounted on its own.
STAGING_SECTOR_LIMIT = SECTORS_PER_BLOCK  # 8

SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS slots (
        disk  INTEGER NOT NULL CHECK (disk IN (0, 1)),
        slot  INTEGER NOT NULL CHECK (slot BETWEEN 0 AND 15),
        lbn   INTEGER NOT NULL UNIQUE,
        PRIMARY KEY (disk, slot)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS slot_sectors (
        disk   INTEGER NOT NULL,
        slot   INTEGER NOT NULL,
        sector INTEGER NOT NULL CHECK (sector BETWEEN 0 AND 7),
        body   BLOB    NOT NULL,
        PRIMARY KEY (disk, slot, sector),
        FOREIGN KEY (disk, slot) REFERENCES slots(disk, slot)
            ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS locations (
        lbn    INTEGER PRIMARY KEY,
        disk   INTEGER NOT NULL,
        slot   INTEGER NOT NULL,
        staged INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS staging_block (
        id  INTEGER PRIMARY KEY CHECK (id = 1),
        lbn INTEGER NOT NULL CHECK (lbn BETWEEN 0 AND 31)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS staging_sectors (
        sector INTEGER PRIMARY KEY CHECK (sector BETWEEN 0 AND 7),
        body   BLOB    NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS disks (
        disk            INTEGER PRIMARY KEY CHECK (disk IN (0, 1)),
        sector_limit    INTEGER NOT NULL CHECK (sector_limit >= 0)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS migration (
        id           INTEGER PRIMARY KEY CHECK (id = 1),
        stripe_k     INTEGER NOT NULL CHECK (stripe_k IN (2, 4)),
        status       TEXT    NOT NULL CHECK (status IN ('running', 'done', 'blocked')),
        next_op      INTEGER NOT NULL,
        total_ops    INTEGER NOT NULL,
        last_writes  INTEGER NOT NULL DEFAULT 0,
        error        TEXT,
        next_ring    INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS rings (
        ring_id   INTEGER PRIMARY KEY,
        positions TEXT    NOT NULL,
        start     INTEGER,
        step      INTEGER NOT NULL DEFAULT 0,
        done      INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        id     INTEGER PRIMARY KEY AUTOINCREMENT,
        ts     TEXT NOT NULL DEFAULT (datetime('now')),
        kind   TEXT NOT NULL,
        detail TEXT NOT NULL
    )
    """,
)


def default_db_path() -> str:
    return os.environ.get("VIRTUAL_VOLUME_DB", os.path.join("data", "volume.db"))


# --------------------------------------------------------------------------- #
# 4 KiB block <-> sparse 512-byte sectors
# --------------------------------------------------------------------------- #

def split_body(body: bytes) -> list[tuple[int, bytes]]:
    """Return ``(sector_index, 512-byte chunk)`` for every NON-zero sector.

    All-zero sectors are not stored at all; that is how a block occupies
    "actual body" space instead of a fixed 4 KiB.
    """
    if len(body) != BLOCK_SIZE:
        raise ValueError(f"block body must be exactly {BLOCK_SIZE} bytes")
    out: list[tuple[int, bytes]] = []
    for index in range(SECTORS_PER_BLOCK):
        chunk = body[index * SECTOR_SIZE:(index + 1) * SECTOR_SIZE]
        if any(chunk):
            out.append((index, chunk))
    return out


def count_body_sectors(body: bytes) -> int:
    if len(body) != BLOCK_SIZE:
        raise ValueError(f"block body must be exactly {BLOCK_SIZE} bytes")
    return sum(
        1 for index in range(SECTORS_PER_BLOCK)
        if any(body[index * SECTOR_SIZE:(index + 1) * SECTOR_SIZE])
    )


def _assemble(rows) -> bytes:
    body = bytearray(BLOCK_SIZE)
    for row in rows:
        sector = row["sector"]
        chunk = row["body"]
        if len(chunk) != SECTOR_SIZE:
            raise RuntimeError(f"stored sector {sector} is {len(chunk)} bytes")
        body[sector * SECTOR_SIZE:(sector + 1) * SECTOR_SIZE] = chunk
    return bytes(body)


def slot_body(conn: sqlite3.Connection, disk: int, slot: int) -> bytes | None:
    """Whole 4 KiB block at a physical slot, or ``None`` when the slot is a hole."""
    present = conn.execute(
        "SELECT 1 FROM slots WHERE disk = ? AND slot = ?", (disk, slot)
    ).fetchone()
    if present is None:
        return None
    rows = conn.execute(
        "SELECT sector, body FROM slot_sectors WHERE disk = ? AND slot = ?"
        " ORDER BY sector",
        (disk, slot),
    ).fetchall()
    return _assemble(rows)


def staging_body(conn: sqlite3.Connection, lbn: int) -> bytes | None:
    present = conn.execute(
        "SELECT 1 FROM staging_block WHERE id = 1 AND lbn = ?", (lbn,)
    ).fetchone()
    if present is None:
        return None
    rows = conn.execute(
        "SELECT sector, body FROM staging_sectors ORDER BY sector"
    ).fetchall()
    return _assemble(rows)


def write_slot_body(conn: sqlite3.Connection, disk: int, slot: int,
                    body: bytes) -> None:
    conn.execute("DELETE FROM slot_sectors WHERE disk = ? AND slot = ?",
                 (disk, slot))
    conn.executemany(
        "INSERT INTO slot_sectors(disk, slot, sector, body) VALUES (?, ?, ?, ?)",
        [(disk, slot, index, chunk) for index, chunk in split_body(body)],
    )


def write_staging_body(conn: sqlite3.Connection, body: bytes) -> None:
    conn.execute("DELETE FROM staging_sectors")
    conn.executemany(
        "INSERT INTO staging_sectors(sector, body) VALUES (?, ?)",
        [(index, chunk) for index, chunk in split_body(body)],
    )


def slot_sector_count(conn: sqlite3.Connection, disk: int, slot: int) -> int:
    return conn.execute(
        "SELECT COUNT(*) AS c FROM slot_sectors WHERE disk = ? AND slot = ?",
        (disk, slot),
    ).fetchone()["c"]


# --------------------------------------------------------------------------- #
# Capacity accounting (derived from stored sectors -> rolls back atomically)
# --------------------------------------------------------------------------- #

def disk_usage(conn: sqlite3.Connection) -> dict[int, int]:
    usage = {disk: 0 for disk in range(DISK_COUNT)}
    for row in conn.execute(
            "SELECT disk, COUNT(*) AS c FROM slot_sectors GROUP BY disk"):
        usage[row["disk"]] = row["c"]
    return usage


def staging_usage(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) AS c FROM staging_sectors").fetchone()["c"]


def disk_limits(conn: sqlite3.Connection) -> dict[int, int]:
    limits = {}
    for row in conn.execute("SELECT disk, sector_limit FROM disks"):
        limits[row["disk"]] = row["sector_limit"]
    for disk in range(DISK_COUNT):
        limits.setdefault(disk, DEFAULT_DISK_SECTOR_LIMIT)
    return limits


def _ensure_disks(conn: sqlite3.Connection) -> None:
    for disk in range(DISK_COUNT):
        conn.execute(
            "INSERT OR IGNORE INTO disks(disk, sector_limit) VALUES (?, ?)",
            (disk, DEFAULT_DISK_SECTOR_LIMIT),
        )


# --------------------------------------------------------------------------- #
# Initial volume
# --------------------------------------------------------------------------- #

def seed_body(lbn: int) -> bytes:
    """Deterministic 4 KiB body for the initial fill."""
    header = f"INIT-BLOCK-LBN-{lbn:02d}\n".encode("ascii")
    return header + b"\x00" * (BLOCK_SIZE - len(header))


def initialize(conn: sqlite3.Connection) -> None:
    """(Re)create the initial volume: 32 blocks, single-block alternating."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        for table in ("slot_sectors", "slots", "locations", "staging_sectors",
                      "staging_block", "rings", "migration", "events", "disks"):
            conn.execute(f"DELETE FROM {table}")
        for disk in range(DISK_COUNT):
            conn.execute(
                "INSERT INTO disks(disk, sector_limit) VALUES (?, ?)",
                (disk, DEFAULT_DISK_SECTOR_LIMIT),
            )
        for lbn in range(BLOCK_COUNT):
            # Initial striping: single-block alternating across the two disks.
            disk = lbn % DISK_COUNT
            slot = lbn // DISK_COUNT
            conn.execute(
                "INSERT INTO slots(disk, slot, lbn) VALUES (?, ?, ?)",
                (disk, slot, lbn),
            )
            write_slot_body(conn, disk, slot, seed_body(lbn))
            conn.execute(
                "INSERT INTO locations(lbn, disk, slot, staged) VALUES (?, ?, ?, 0)",
                (lbn, disk, slot),
            )
        conn.execute(
            "INSERT INTO events(kind, detail) VALUES ('init', ?)",
            ("volume initialized: 2 disks x 16 slots, single-block alternating,"
             " sparse 512-byte sectors",),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def ensure_seeded(conn: sqlite3.Connection) -> None:
    count = conn.execute("SELECT COUNT(*) AS c FROM slots").fetchone()["c"]
    staged = conn.execute(
        "SELECT COUNT(*) AS c FROM staging_block").fetchone()["c"]
    if count == 0 and staged == 0:
        initialize(conn)


# --------------------------------------------------------------------------- #
# Connection, schema install, and upgrade of the old (BLOB-per-block) schema
# --------------------------------------------------------------------------- #

def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.OperationalError:
        return set()


def _is_legacy(conn: sqlite3.Connection) -> bool:
    return "body" in _columns(conn, "slots")


def _rows_or_empty(conn: sqlite3.Connection, sql: str) -> list[sqlite3.Row]:
    try:
        return conn.execute(sql).fetchall()
    except sqlite3.OperationalError:
        return []


def _legacy_cycles(ops: list[sqlite3.Row]) -> list[list[tuple[int, int]]]:
    """Rebuild the cycle position order from the old static stage/move/unstage plan."""
    groups: list[dict] = []
    current = None
    for row in ops:
        if row["op"] == "stage":
            current = {"stage": (row["src_disk"], row["src_slot"]),
                       "moves": [], "unstage": None}
            groups.append(current)
        elif row["op"] == "move" and current is not None:
            current["moves"].append((row["src_disk"], row["src_slot"]))
        elif row["op"] == "unstage" and current is not None:
            current["unstage"] = (row["dst_disk"], row["dst_slot"])

    cycles = []
    for group in groups:
        if group["stage"] is None or group["unstage"] is None:
            continue
        moves = group["moves"]
        n = len(moves) + 1
        positions = [group["stage"], group["unstage"]]
        # old plan emitted moves for c[n-1], c[n-2], ..., c[1]
        for k in range(2, n):
            positions.append(moves[n - k - 1])
        if len(positions) == n:
            cycles.append(positions)
    return cycles


def _upgrade_legacy(conn: sqlite3.Connection) -> None:
    """Convert the old one-BLOB-per-block database in a single transaction.

    Bodies are split into stored 512-byte sectors, physical slot positions and
    any in-flight migration progress are kept, and each disk gets the default
    limit of 128 sectors.
    """
    import json

    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("BEGIN IMMEDIATE")
    try:
        slot_rows = _rows_or_empty(conn, "SELECT * FROM slots")
        staging_rows = _rows_or_empty(conn, "SELECT * FROM staging")
        location_rows = _rows_or_empty(conn, "SELECT * FROM locations")
        mig_rows = _rows_or_empty(conn, "SELECT * FROM migration WHERE id = 1")
        op_rows = _rows_or_empty(conn, "SELECT * FROM ops ORDER BY seq")

        conn.execute("DROP TABLE IF EXISTS slot_sectors")
        conn.execute("DROP TABLE IF EXISTS staging_block")
        conn.execute("DROP TABLE IF EXISTS staging_sectors")
        conn.execute("DROP TABLE IF EXISTS rings")
        for table in ("staging", "ops", "migration", "locations", "slots"):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        for statement in SCHEMA_STATEMENTS:
            conn.execute(statement)

        for row in slot_rows:
            body = row["body"]
            conn.execute(
                "INSERT INTO slots(disk, slot, lbn) VALUES (?, ?, ?)",
                (row["disk"], row["slot"], row["lbn"]),
            )
            if len(body) != BLOCK_SIZE:
                raise RuntimeError(
                    f"legacy slot disk{row['disk']}.{row['slot']} body is "
                    f"{len(body)} bytes, not {BLOCK_SIZE}")
            conn.executemany(
                "INSERT INTO slot_sectors(disk, slot, sector, body)"
                " VALUES (?, ?, ?, ?)",
                [(row["disk"], row["slot"], index, chunk)
                 for index, chunk in split_body(body)],
            )
        for row in location_rows:
            conn.execute(
                "INSERT INTO locations(lbn, disk, slot, staged) VALUES (?, ?, ?, ?)",
                (row["lbn"], row["disk"], row["slot"], row["staged"]),
            )
        for row in staging_rows:
            conn.execute("INSERT INTO staging_block(id, lbn) VALUES (1, ?)", (row["lbn"],))
            conn.executemany(
                "INSERT INTO staging_sectors(sector, body) VALUES (?, ?)",
                [(index, chunk) for index, chunk in split_body(row["body"])],
            )

        if mig_rows:
            mig = mig_rows[0]
            cycles = _legacy_cycles(op_rows)
            total_ops = sum(len(cycle) + 1 for cycle in cycles)
            executed = int(mig["next_op"])
            if mig["status"] == "done" or executed >= total_ops:
                conn.execute(
                    "INSERT INTO migration(id, stripe_k, status, next_op,"
                    " total_ops, last_writes, error, next_ring)"
                    " VALUES (1, ?, 'done', ?, ?, ?, NULL, ?)",
                    (mig["stripe_k"], total_ops, total_ops,
                     mig["last_writes"], len(cycles)),
                )
                for ring_id, cycle in enumerate(cycles):
                    n = len(cycle)
                    conn.execute(
                        "INSERT INTO rings(ring_id, positions, start, step, done)"
                        " VALUES (?, ?, 0, ?, 1)",
                        (ring_id, json.dumps(cycle), n + 1),
                    )
            else:
                remaining = executed
                next_ring = len(cycles)
                for ring_id, cycle in enumerate(cycles):
                    n = len(cycle)
                    ops_in_ring = n + 1
                    if remaining >= ops_in_ring:
                        step, start, done = ops_in_ring, 0, 1
                        remaining -= ops_in_ring
                    elif remaining > 0:
                        step, start, done = remaining, 0, 0
                        next_ring = ring_id
                        remaining = 0
                    else:
                        step, start, done = 0, None, 0
                        next_ring = min(next_ring, ring_id)
                    conn.execute(
                        "INSERT INTO rings(ring_id, positions, start, step, done)"
                        " VALUES (?, ?, ?, ?, ?)",
                        (ring_id, json.dumps(cycle), start, step, done),
                    )
                status = "done" if next_ring >= len(cycles) else "running"
                conn.execute(
                    "INSERT INTO migration(id, stripe_k, status, next_op,"
                    " total_ops, last_writes, error, next_ring)"
                    " VALUES (1, ?, ?, ?, ?, ?, NULL, ?)",
                    (mig["stripe_k"], status, executed, total_ops,
                     mig["last_writes"], next_ring),
                )

        _ensure_disks(conn)
        conn.execute(
            "INSERT INTO events(kind, detail) VALUES ('schema_upgrade', ?)",
            ("upgraded legacy volume to sparse 512-byte sectors;"
             " positions and migration progress preserved",),
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.execute("PRAGMA foreign_keys=ON")


def connect(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or default_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    # Rollback journal + FULL fsync: a process killed before COMMIT leaves the
    # journal in place and the next connection recovers the committed state.
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    if _is_legacy(conn):
        _upgrade_legacy(conn)
    conn.executescript(";\n".join(SCHEMA_STATEMENTS))
    _ensure_disks(conn)
    return conn
