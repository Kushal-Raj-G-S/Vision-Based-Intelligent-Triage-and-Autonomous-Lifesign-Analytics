import os
import threading
import time
import psycopg2
import psycopg2.errors
from psycopg2 import pool as _pgpool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Neon's serverless Postgres suspends its compute after a few minutes of
# inactivity ("scale to zero" on free/dev tiers). The first connection after
# a suspend can occasionally hit a transient failure while the compute wakes
# up. One short retry absorbs that cold-start hiccup instead of surfacing it
# as a real error (or, previously, silently as "no data" — see
# get_all_patients()).
def get_connection(retries: int = 1, backoff_s: float = 0.6):
    """Establish connection to Neon Postgres database, retrying once on a
    transient (e.g. cold-start) failure before giving up."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL is not set in the environment variables.")
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return psycopg2.connect(DATABASE_URL)
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                print(f"[DB] Connection attempt {attempt + 1} failed ({exc}); retrying in {backoff_s}s "
                      f"(likely a Neon cold-start — compute suspends after inactivity)...")
                time.sleep(backoff_s)
    raise last_exc


# ── Connection pool ──────────────────────────────────────────────────────────
# Measured from this machine: a NEW connection to Neon costs ~1.6-2 s (TLS to a
# remote region) and every query costs ~0.9 s of pure network round trips — so
# round trips, not SQL, are the cost to minimise. The pool pays the connection
# cost once; autocommit makes a single statement exactly one round trip (no
# implicit BEGIN / explicit ROLLBACK). Keepalives let the OS notice a
# connection Neon dropped while the compute was suspended, and _run() retries
# ONCE on a stale-connection error, so a suspended database still looks like a
# normal (slightly slower) first request.
_pool = None
_pool_lock = threading.Lock()


def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                if not DATABASE_URL:
                    raise ValueError("DATABASE_URL is not set in the environment variables.")
                _pool = _pgpool.ThreadedConnectionPool(
                    1, 6, DATABASE_URL,
                    keepalives=1, keepalives_idle=20, keepalives_interval=5, keepalives_count=3)
    return _pool


def warm_pool():
    """Open the first pooled connection now (call at startup) so the first patient doesn't pay for it."""
    try:
        c = _get_pool().getconn()
        _get_pool().putconn(c)
    except Exception as exc:
        print(f"[DB] pool warm-up failed (will retry on first use): {exc}")


def _run(fn, autocommit=True):
    """Run fn(conn) on a pooled connection. A stale/dropped connection
    (OperationalError / InterfaceError) is discarded and the call retried once on
    a fresh one. With autocommit=True (default) every single statement is its own
    transaction: one network round trip, nothing to commit or roll back."""
    last = None
    for _ in range(2):
        pool = _get_pool()
        conn = pool.getconn()
        try:
            conn.autocommit = autocommit
            result = fn(conn)
            if not autocommit:
                conn.commit()
            conn.autocommit = False
            pool.putconn(conn)
            return result
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
            last = exc
            try:
                pool.putconn(conn, close=True)
            except Exception:
                pass
        except Exception:
            try:
                conn.rollback()
                conn.autocommit = False
            except Exception:
                pass
            pool.putconn(conn)
            raise
    raise last


def init_db():
    """Initialise the postgres tables if they do not exist."""
    sql = """
    CREATE TABLE IF NOT EXISTS triage_records (
      id VARCHAR(50) PRIMARY KEY,
      name VARCHAR(100) NOT NULL,
      timestamp VARCHAR(30) NOT NULL,
      video_path VARCHAR(255),
      esi_level INTEGER NOT NULL,
      priority_score INTEGER NOT NULL,
      primary_diagnosis VARCHAR(255),
      is_shock BOOLEAN NOT NULL,
      triage_summary TEXT,
      agent_output TEXT,
      heart_rate DOUBLE PRECISION,
      respiration DOUBLE PRECISION,
      hrv DOUBLE PRECISION,
      stress_index DOUBLE PRECISION
    );
    """
    # Kiosk intake + waiting-room columns, added in place so an existing
    # database (and its old rows) keeps working: every column is nullable /
    # defaulted, and old rows simply have no token, so they never appear on the
    # waiting-room board. The unique index guarantees one token number per day
    # even if two patients finish at the same instant (save_new_patient retries).
    migrate = """
    ALTER TABLE triage_records
      ADD COLUMN IF NOT EXISTS token VARCHAR(16),
      ADD COLUMN IF NOT EXISTS token_seq INTEGER,
      ADD COLUMN IF NOT EXISTS age INTEGER,
      ADD COLUMN IF NOT EXISTS sex VARCHAR(20),
      ADD COLUMN IF NOT EXISTS complaint TEXT,
      ADD COLUMN IF NOT EXISTS phone VARCHAR(32),
      ADD COLUMN IF NOT EXISTS status VARCHAR(16) DEFAULT 'waiting';
    CREATE UNIQUE INDEX IF NOT EXISTS uq_token_per_day
      ON triage_records ((substr(timestamp, 1, 10)), token_seq) WHERE token_seq IS NOT NULL;
    """
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(sql)
            cur.execute(migrate)
        conn.commit()
        print("[DB] Neon Postgres database initialised successfully.")
    except Exception as e:
        print(f"[DB] ERROR initialising Neon database: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()


# Defaults for the kiosk columns, so callers that predate them (the demo seed
# script, older code paths) keep working without passing any of these keys.
_KIOSK_DEFAULTS = {
    'token': None, 'token_seq': None, 'age': None, 'sex': None,
    'complaint': None, 'phone': None, 'status': 'waiting',
}

_UPSERT_SQL = """
    INSERT INTO triage_records (
      id, name, timestamp, video_path, esi_level, priority_score,
      primary_diagnosis, is_shock, triage_summary, agent_output,
      heart_rate, respiration, hrv, stress_index,
      token, token_seq, age, sex, complaint, phone, status
    ) VALUES (
      %(id)s, %(name)s, %(timestamp)s, %(video_path)s, %(esi_level)s, %(priority_score)s,
      %(primary_diagnosis)s, %(is_shock)s, %(triage_summary)s, %(agent_output)s,
      %(heart_rate)s, %(respiration)s, %(hrv)s, %(stress_index)s,
      %(token)s, %(token_seq)s, %(age)s, %(sex)s, %(complaint)s, %(phone)s, %(status)s
    ) ON CONFLICT (id) DO UPDATE SET
      token = EXCLUDED.token,
      token_seq = EXCLUDED.token_seq,
      age = EXCLUDED.age,
      sex = EXCLUDED.sex,
      complaint = EXCLUDED.complaint,
      phone = EXCLUDED.phone,
      status = EXCLUDED.status,
      name = EXCLUDED.name,
      timestamp = EXCLUDED.timestamp,
      video_path = EXCLUDED.video_path,
      esi_level = EXCLUDED.esi_level,
      priority_score = EXCLUDED.priority_score,
      primary_diagnosis = EXCLUDED.primary_diagnosis,
      is_shock = EXCLUDED.is_shock,
      triage_summary = EXCLUDED.triage_summary,
      agent_output = EXCLUDED.agent_output,
      heart_rate = EXCLUDED.heart_rate,
      respiration = EXCLUDED.respiration,
      hrv = EXCLUDED.hrv,
      stress_index = EXCLUDED.stress_index
"""

# Same upsert, but the token number is allocated INSIDE the statement (today's
# MAX(token_seq)+1), so allocating and saving are ONE atomic round trip.
_NEW_PATIENT_SQL = (
    "WITH nxt AS (SELECT COALESCE(MAX(token_seq), 0) + 1 AS seq FROM triage_records "
    "WHERE timestamp LIKE %(day)s) "
    + _UPSERT_SQL
    .replace(") VALUES (", ") SELECT ", 1)
    .replace("%(token)s", "'A-' || lpad(nxt.seq::text, 3, '0')", 1)
    .replace("%(token_seq)s", "nxt.seq", 1)
    .replace("\n    ) ON CONFLICT", "\n    FROM nxt ON CONFLICT", 1)
    + " RETURNING token, token_seq"
)


def save_patient_record(record):
    """Save patient triage record to Neon database."""
    record = {**_KIOSK_DEFAULTS, **record}

    def _do(conn):
        with conn.cursor() as cur:
            cur.execute(_UPSERT_SQL, record)

    try:
        _run(_do)
        print(f"[DB] Saved patient record {record.get('id')} ({record.get('name')}) to Neon.")
        return True
    except Exception as e:
        print(f"[DB] ERROR saving record {record.get('id')}: {e}")
        return False


def save_new_patient(record):
    """Allocate today's next token AND save the record in a SINGLE statement
    (one round trip). Mutates `record` with token / token_seq. If two patients
    finish at the same instant and race to the same number, the unique index
    rejects the loser and it simply retries with the next number.
    Returns True on success; raises on failure."""
    full = {**_KIOSK_DEFAULTS, **record, 'day': time.strftime('%Y-%m-%d') + '%'}

    def _do(conn):
        with conn.cursor() as cur:
            cur.execute(_NEW_PATIENT_SQL, full)
            return cur.fetchone()

    last = None
    for _ in range(4):
        try:
            token, token_seq = _run(_do)
            record['token'], record['token_seq'] = token, int(token_seq)
            print(f"[DB] Saved patient record {record.get('id')} as token {token}.")
            return True
        except psycopg2.errors.UniqueViolation as exc:
            last = exc            # lost a token race — take the next number
    raise last


def next_token():
    """Next waiting-room token for today: (sequence, 'A-001'). Resets each day.
    Informational only — save_new_patient() allocates the real one atomically."""
    day = time.strftime('%Y-%m-%d')

    def _do(conn):
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(MAX(token_seq), 0) + 1 FROM triage_records WHERE timestamp LIKE %s",
                (day + '%',))
            seq = int(cur.fetchone()[0])
        return seq, f"A-{seq:03d}"
    return _run(_do)


def get_waiting_patients():
    """Light query for the waiting room: only the columns the queue needs (no long
    text fields), only patients still in the room."""
    def _do(conn):
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""SELECT id, token, token_seq, esi_level, priority_score, is_shock, status
                           FROM triage_records
                           WHERE token IS NOT NULL AND COALESCE(status, 'waiting') IN ('waiting', 'called')""")
            return [dict(r) for r in cur.fetchall()]
    return _run(_do)


def set_patient_status(patient_id, status):
    """waiting | called | seen | left. Returns True if a row was updated."""
    def _do(conn):
        with conn.cursor() as cur:
            cur.execute("UPDATE triage_records SET status = %s WHERE id = %s", (status, patient_id))
            return cur.rowcount > 0
    return _run(_do)


def update_narrative(patient_id, triage_summary, agent_output):
    """Fill in the AI narrative of a record that was saved before the (slow)
    LLM crew finished. Touches ONLY those two columns, so it can never overwrite
    a status change (e.g. 'called') made by staff in the meantime."""
    def _do(conn):
        with conn.cursor() as cur:
            cur.execute("UPDATE triage_records SET triage_summary = %s, agent_output = %s WHERE id = %s",
                        (triage_summary, agent_output, patient_id))
            return cur.rowcount > 0
    return _run(_do)


def delete_patient(patient_id):
    """Delete one record by id (used by tests/cleanup). Returns True if removed."""
    def _do(conn):
        with conn.cursor() as cur:
            cur.execute("DELETE FROM triage_records WHERE id = %s", (patient_id,))
            return cur.rowcount > 0
    return _run(_do)


def get_all_patients():
    """
    Retrieve all triage records ordered by urgency (ESI level asc, then priority_score desc).

    Raises on failure — does NOT return [] on a DB error. Returning an empty
    list here used to be indistinguishable from "the queue is genuinely
    empty," which meant a transient Neon connection hiccup (e.g. compute
    waking from a cold start) made every real patient record look like it
    had vanished, even though nothing was ever deleted. Callers (see
    web/routes.py) must catch this and surface a real error instead of
    quietly rendering zero patients.
    """
    sql = """
    SELECT * FROM triage_records
    ORDER BY esi_level ASC, priority_score DESC, timestamp DESC;
    """

    def _do(conn):
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            return [dict(row) for row in cur.fetchall()]
    return _run(_do)


def clear_all_patients():
    """Delete all triage records from Neon database."""
    sql = "TRUNCATE TABLE triage_records;"
    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        print("[DB] Flushed/Truncated all records in Neon Postgres.")
        return True
    except Exception as e:
        print(f"[DB] TRUNCATE failed: {e}. Trying DELETE...")
        try:
            if conn:
                conn.rollback()
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM triage_records;")
                conn.commit()
                print("[DB] Deleted all records in Neon Postgres.")
                return True
        except Exception as delete_err:
            print(f"[DB] DELETE failed as well: {delete_err}")
            if conn:
                conn.rollback()
        return False
    finally:
        if conn:
            conn.close()
