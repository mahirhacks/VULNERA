"""
1_normalizer.py — ingest all raw formats, map to core columns, write combined Parquet.

Reads: data/raw/ (JSONL, JSON lines, CSV, SQL, existing Parquet)
Writes: data/interim/normalized/normalized_{1..N}.parquet

All active datasets are merged into the same core schema. Train/valid/test may
appear in one file; temporal_splitter assigns or refines splits later.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "dataset_pipeline._runtime",
    Path(__file__).resolve().parent / "_runtime.py",
)
_runtime = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_runtime)
_runtime.ensure_app_root(__file__)

import argparse
import gc
import gzip
import json
import re
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, TextIO

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dataset_pipeline._loader import cfg as pcfg

INSERT_RE = re.compile(
    r"INSERT\s+INTO\s+`?(\w+)`?\s*(?:\(([^)]+)\))?\s*VALUES\s*(.+);?\s*$",
    re.IGNORECASE,
)

METHOD_CHANGE_COLS = {
    "method_change_id": 0,
    "file_change_id": 1,
    "name": 2,
    "code": 7,
    "before_change": 12,
}

FILE_CHANGE_COLS = {
    "file_change_id": 0,
    "hash": 1,
    "filename": 2,
    "programming_language": 15,
}

_C_CPP_LANGS = frozenset({"c", "c++", "cpp", "", "none"})

COMMITS_COLS = {
    "hash": 0,
    "repo_url": 1,
    "author": 2,
    "author_date": 3,
    "committer": 5,
    "committer_date": 6,
}

FIXES_COLS = {"cve_id": 0, "hash": 1, "repo_url": 2}

CVE_COLS = {"cve_id": 0, "published_date": 1, "description": 3}

CWE_CLASS_COLS = {"cve_id": 0, "cwe_id": 1}


def _normalize_value(value: Any, nulls: set[str]) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        v = value.strip()
        return None if v in nulls else v
    if isinstance(value, list):
        parts = [str(_normalize_value(x, nulls)) for x in value]
        parts = [p for p in parts if p and p != "None"]
        return ";".join(parts) if parts else None
    return value


def _guess_func_name(code: str, explicit: Any = None) -> str | None:
    if explicit and isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    skip = {"if", "for", "while", "return", "switch", "sizeof"}
    for line in code.splitlines():
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        m = re.search(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
        if m and m.group(1) not in skip:
            return m.group(1)
    return None


def _coerce_label(raw: Any) -> int:
    if isinstance(raw, bool):
        return 1 if raw else 0
    if isinstance(raw, (int, float)) and not (isinstance(raw, float) and pd.isna(raw)):
        return 1 if int(raw) != 0 else 0
    text = str(raw).strip().lower()
    if text in ("1", "true", "t", "yes", "vul", "vulnerable"):
        return 1
    return 0


_CVE_YEAR_RE = re.compile(r"CVE-(\d{4})-", re.IGNORECASE)
_CVE_ID_RE = re.compile(r"CVE-\d{4}-\d+", re.IGNORECASE)
# Standalone year in prose (cve_desc); bounded to plausible CVE publication years.
_TEXT_YEAR_RE = re.compile(r"\b(199[0-9]|20[0-2][0-9])\b")

PRIMEVUL_DROP_FIELDS = frozenset(
    {
        "project",
        "commit_id",
        "project_url",
        "commit_url",
        "commit_message",
        "file_name",
    }
)

PRIMEVUL_KEEP_FIELDS = (
    "func_hash",
    "file_hash",
    "cwe",
    "cve",
    "cve_desc",
    "nvd_url",
)


def _stringify_field(value: Any, nulls: set[str]) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, list):
        parts = [_normalize_value(x, nulls) for x in value]
        parts = [str(p) for p in parts if p is not None and str(p).strip()]
        return ";".join(parts) if parts else None
    return _normalize_value(value, nulls)  # type: ignore[return-value]


def _year_from_text(text: str, *, allow_bare_year: bool) -> int | None:
    m = _CVE_YEAR_RE.search(text)
    if m:
        return int(m.group(1))
    if not allow_bare_year:
        return None
    for m in _TEXT_YEAR_RE.finditer(text):
        year = int(m.group(1))
        if 1990 <= year <= 2030:
            return year
    return None


def _extract_primevul_year(raw: dict[str, Any]) -> int | None:
    """Year from cve, then cwe, then cve_desc (CVE id preferred; prose year last)."""
    for key in ("cve", "cwe", "cve_desc"):
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                if item is None:
                    continue
                year = _year_from_text(str(item), allow_bare_year=key == "cve_desc")
                if year is not None:
                    return year
            continue
        year = _year_from_text(str(value), allow_bare_year=key == "cve_desc")
        if year is not None:
            return year
    return None


def _date_from_cve(value: Any) -> str | None:
    """Extract YYYY-01-01 from a CVE id like 'CVE-2019-1234'. Returns None if no match."""
    if not value:
        return None
    if isinstance(value, list):
        for v in value:
            d = _date_from_cve(v)
            if d:
                return d
        return None
    m = _CVE_YEAR_RE.search(str(value))
    return f"{m.group(1)}-01-01" if m else None


def _year_from_cve(value: Any) -> int | None:
    if not value:
        return None
    if isinstance(value, list):
        for v in value:
            y = _year_from_cve(v)
            if y is not None:
                return y
        return None
    m = _CVE_YEAR_RE.search(str(value))
    return int(m.group(1)) if m else None


def _first_cve_id(*values: Any) -> str | None:
    """Return the first CVE id found in any of the given values."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                cve = _first_cve_id(item)
                if cve:
                    return cve
            continue
        m = _CVE_ID_RE.search(str(value))
        if m:
            return m.group(0).upper()
    return None


def load_diversevul_metadata_index(
    cfg: dict[str, Any], mapping: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Load commit-level metadata (CVE/CWE/URLs) keyed by commit_id."""
    path_key = mapping.get("metadata_path_key", "raw_diversevul_metadata")
    path = pcfg.resolve_path(cfg, path_key)
    if not path.exists():
        return {}

    by_commit: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            commit_id = row.get("commit_id")
            if not commit_id:
                continue
            cid = str(commit_id).strip()
            prev = by_commit.get(cid)
            if prev is None or (not prev.get("CVE") and row.get("CVE")):
                by_commit[cid] = row
    return by_commit


def _extract_diversevul_temporal_and_vuln_fields(
    raw: dict[str, Any],
    meta: dict[str, Any] | None,
    nulls: set[str],
) -> tuple[int | None, str | None, str | None, str | None, str | None, str | None]:
    """
    Derive year/commit_date/cve/cwe/cve_desc/nvd_url for DiverseVul.

    The official metadata file has no commit timestamps; dates come from CVE ids
    in metadata (preferred) or in the function's commit message.
    """
    cve = _first_cve_id(meta.get("CVE") if meta else None, raw.get("message"))
    year = _year_from_cve(cve)
    commit_date = _date_from_cve(cve)

    cwe = None
    if meta:
        cwe = _stringify_field(meta.get("CWE"), nulls)
    if not cwe:
        cwe = _stringify_field(raw.get("cwe"), nulls)

    cve_desc = None
    if meta:
        cve_desc = _stringify_field(meta.get("bug_info"), nulls)

    nvd_url = f"https://nvd.nist.gov/vuln/detail/{cve}" if cve else None
    return year, commit_date, cve, cwe, cve_desc, nvd_url


def _parse_datetime_year_date(value: Any) -> tuple[int | None, str | None]:
    if value is None:
        return None, None
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "null"):
        return None, None
    ts = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.isna(ts):
        return None, None
    year = int(ts.year)
    if not (1990 <= year <= 2030):
        return None, None
    return year, ts.strftime("%Y-%m-%d")


def _cvefixes_description(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text[:4000] if text else None


def _extract_bigvul_year_and_date(
    raw: dict[str, Any],
    mapping: dict[str, Any],
    nulls: set[str],
) -> tuple[int | None, str | None]:
    """Year/date from Publish Date; fallback to CVE assignment year."""
    pub_key = mapping.get("publish_date", "Publish Date")
    pub = _normalize_value(raw.get(pub_key), nulls)
    if pub:
        ts = pd.to_datetime(pub, errors="coerce")
        if pd.notna(ts):
            year = int(ts.year)
            if 1990 <= year <= 2030:
                return year, ts.strftime("%Y-%m-%d")

    cve_key = mapping.get("cve_field", "CVE ID")
    cve_val = raw.get(cve_key)
    year = _year_from_cve(cve_val)
    if year is not None and 1990 <= year <= 2030:
        return year, f"{year}-01-01"
    return None, None


def _resolve_commit_date(
    raw: dict[str, Any],
    mapping: dict[str, Any],
    nulls: set[str],
) -> Any:
    """commit_date priority: explicit `commit_date` field → `date_from_cve` fallback."""
    date_field = mapping.get("commit_date", "commit_date")
    value = _normalize_value(raw.get(date_field), nulls)
    if value:
        return value

    cve_field = mapping.get("date_from_cve")
    if cve_field:
        return _date_from_cve(raw.get(cve_field))
    return None


def map_raw_to_core(
    raw: dict[str, Any],
    mapping: dict[str, Any],
    source_dataset: str,
    nulls: set[str],
    columns: list[str],
) -> dict[str, Any] | None:
    code_key = mapping["code"]
    code = raw.get(code_key) or ""
    if not isinstance(code, str) or not code.strip():
        return None

    try:
        label = _coerce_label(raw[mapping["label"]])
    except (KeyError, TypeError):
        return None

    split = raw.get("split")
    if split is not None and not isinstance(split, str):
        split = str(split) if pd.notna(split) else None  # type: ignore[arg-type]
    if isinstance(split, str):
        split = split.strip().lower() or None

    id_field = mapping.get("id_field", "idx")
    raw_id = raw.get(id_field, raw.get("idx", raw.get("hash", "unknown")))
    prefix = mapping.get("id_prefix", source_dataset)
    row_id = f"{prefix}_{split}_{raw_id}" if split else f"{prefix}_{raw_id}"

    func_key = mapping.get("func_name", "func_name")
    explicit_name = raw.get(func_key) if func_key in raw else None

    row = {
        "id": row_id,
        "code": code,
        "label": label,
        "split": split,
        "source_dataset": source_dataset,
        "commit_hash": _normalize_value(
            raw.get(mapping.get("commit_hash", "commit_id")), nulls
        ),
        "commit_date": _resolve_commit_date(raw, mapping, nulls),
        "file_path": _normalize_value(
            raw.get(mapping.get("file_path", "file_name")), nulls
        ),
        "func_name": _guess_func_name(code, explicit_name),
        "project": _normalize_value(raw.get(mapping.get("project", "project")), nulls),
    }
    for col in columns:
        if col not in row:
            row[col] = None
    return {c: row.get(c) for c in columns}


def lang_allowed(raw: dict[str, Any], mapping: dict[str, Any]) -> bool:
    lang_field = mapping.get("lang")
    allowed = mapping.get("allowed_langs", [])
    if not lang_field or not allowed:
        return True
    lang = raw.get(lang_field)
    if lang is None:
        return True
    s = str(lang).strip().lower()
    return s in {a.lower() for a in allowed} or str(lang).strip() in allowed


def _chunk_output_name(prefix: str, index: int) -> str:
    return f"{prefix}_{index}.parquet"


def _core_arrow_schema(columns: list[str]) -> pa.Schema:
    """Fixed schema so batches with all-null string columns don't become type 'null'."""
    type_map: dict[str, pa.DataType] = {"label": pa.int64(), "year": pa.int64()}
    return pa.schema(
        [pa.field(col, type_map.get(col, pa.string()), nullable=True) for col in columns]
    )


def _buffer_to_arrow_table(buffer: list[dict[str, Any]], columns: list[str], schema: pa.Schema) -> pa.Table:
    df = pd.DataFrame(buffer, columns=columns)
    for field in schema:
        col = field.name
        if col not in df.columns:
            continue
        if pa.types.is_int64(field.type):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        else:
            df[col] = df[col].astype("string")
    return pa.Table.from_pandas(df, schema=schema, preserve_index=False)


class StreamingChunkWriter:
    """Flush rows to normalized_N.parquet without holding the full dataset in RAM."""

    def __init__(
        self,
        out_dir: Path,
        prefix: str,
        columns: list[str],
        n_chunks: int = 1,
        flush_rows: int = 25_000,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.prefix = prefix
        self.columns = columns
        self.n_chunks = max(1, int(n_chunks))
        self.flush_rows = max(1_000, int(flush_rows))
        self._schema = _core_arrow_schema(columns)
        self._buffer: list[dict[str, Any]] = []
        self._rotate = 0
        self._writers: dict[int, pq.ParquetWriter] = {}
        self.written_paths: list[Path] = []

    def _chunk_index(self) -> int:
        return (self._rotate % self.n_chunks) + 1

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        idx = self._chunk_index()
        self._rotate += 1
        path = self.out_dir / _chunk_output_name(self.prefix, idx)
        table = _buffer_to_arrow_table(self._buffer, self.columns, self._schema)
        if idx not in self._writers:
            self._writers[idx] = pq.ParquetWriter(path, self._schema)
            if path not in self.written_paths:
                self.written_paths.append(path)
        self._writers[idx].write_table(table)
        self._buffer.clear()

    def add_row(self, row: dict[str, Any]) -> None:
        self._buffer.append(row)
        if len(self._buffer) >= self.flush_rows:
            self._flush_buffer()

    def close(self) -> list[Path]:
        self._flush_buffer()
        for writer in self._writers.values():
            writer.close()
        if not self.written_paths and self.n_chunks >= 1:
            empty = self.out_dir / _chunk_output_name(self.prefix, 1)
            pd.DataFrame(columns=self.columns).to_parquet(empty, index=False)
            self.written_paths = [empty]
        return sorted(self.written_paths, key=lambda p: p.name)


@dataclass
class NormStats:
    rows_mapped: int = 0
    rows_skipped: int = 0
    sources_loaded: int = 0
    chunks_written: int = 0
    by_source: dict[str, dict[str, int]] = field(default_factory=dict)

    def bump(self, source: str, key: str, n: int = 1) -> None:
        self.by_source.setdefault(source, {}).setdefault(key, 0)
        self.by_source[source][key] += n


def _open_text(path: Path) -> TextIO:
    if path.suffix.lower() == ".gz" or path.name.lower().endswith(".sql.gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")  # type: ignore[return-value]
    return path.open(encoding="utf-8", errors="replace")


def _iter_jsonl(path: Path, split: str | None = None) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if split and "split" not in obj:
                obj["split"] = split
            yield obj


def _iter_csv(path: Path, chunksize: int) -> Iterator[dict[str, Any]]:
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        for row in chunk.to_dict(orient="records"):
            yield row


def _iter_parquet(path: Path, batch_size: int = 10_000) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        yield from df.to_dict(orient="records")


def _parse_sql_value(token: str) -> Any:
    token = token.strip()
    if token.upper() == "NULL":
        return None
    if token.startswith("'") and token.endswith("'"):
        return token[1:-1].replace("\\'", "'").replace("\\n", "\n").replace("\\r", "\r")
    if token.startswith('"') and token.endswith('"'):
        return token[1:-1]
    try:
        if "." in token:
            return float(token)
        return int(token)
    except ValueError:
        return token


def _split_value_tuple(inner: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    in_quote = False
    quote_char = ""
    i = 0
    while i < len(inner):
        ch = inner[i]
        if in_quote:
            buf.append(ch)
            if ch == quote_char and (i == 0 or inner[i - 1] != "\\"):
                in_quote = False
            i += 1
            continue
        if ch in ("'", '"'):
            in_quote = True
            quote_char = ch
            buf.append(ch)
            i += 1
            continue
        if ch == ",":
            parts.append("".join(buf).strip())
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    if buf:
        parts.append("".join(buf).strip())
    return parts


def _parse_insert_values(values_blob: str) -> list[list[Any]]:
    rows: list[list[Any]] = []
    depth = 0
    start = -1
    for i, ch in enumerate(values_blob):
        if ch == "(":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start >= 0:
                inner = values_blob[start:i]
                rows.append([_parse_sql_value(p) for p in _split_value_tuple(inner)])
                start = -1
    return rows


def iter_sql_inserts(
    sql_path: Path, table_names: set[str]
) -> Iterator[tuple[str, list[str] | None, list[list[Any]]]]:
    tables_lower = {t.lower() for t in table_names}
    with _open_text(sql_path) as f:
        buffer = ""
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("--"):
                continue
            buffer += " " + stripped
            if not buffer.rstrip().endswith(";"):
                continue
            m = INSERT_RE.search(buffer)
            buffer = ""
            if not m:
                continue
            table, col_blob, values_blob = m.group(1), m.group(2), m.group(3)
            if table.lower() not in tables_lower:
                continue
            columns = None
            if col_blob:
                columns = [c.strip().strip("`").strip('"') for c in col_blob.split(",")]
            for row in _parse_insert_values(values_blob):
                yield table.lower(), columns, row


def _row_to_dict(
    columns: list[str] | None, values: list[Any], fallback: dict[str, int]
) -> dict[str, Any]:
    if columns:
        return dict(zip(columns, values))
    return {k: values[i] for k, i in fallback.items() if i < len(values)}


def _bool_label(val: Any) -> int:
    if val is None:
        return 0
    if isinstance(val, bool):
        return 1 if val else 0
    return 1 if str(val).strip().lower() in ("1", "true", "t", "yes") else 0


def _cvefixes_year_and_date(
    cve_id: Any,
    committer_date: Any,
    author_date: Any,
    published_date: Any,
) -> tuple[int | None, str | None]:
    for value in (committer_date, author_date, published_date):
        year, commit_date = _parse_datetime_year_date(value)
        if year is not None:
            return year, commit_date

    if cve_id:
        cve_id = str(cve_id)
        year = _year_from_cve(cve_id)
        if year is not None:
            return year, f"{year}-01-01"
    return None, None


def _cvefixes_cache_db(cfg: dict[str, Any], sql_path: Path) -> Path:
    interim = cfg["_base_dir"] / cfg.get("paths", {}).get("interim_dir", "data/interim")
    cache_dir = interim / "cvefixes_sqlite"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{sql_path.stem}.sqlite"


def _cvefixes_build_path(db_path: Path) -> Path:
    return db_path.with_suffix(".sqlite.tmp")


def _close_sqlite_connection(conn: sqlite3.Connection | None) -> None:
    if conn is None:
        return
    try:
        conn.commit()
    except sqlite3.Error:
        pass
    try:
        conn.close()
    except sqlite3.Error:
        pass


def _sqlite_cache_ready(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1_000_000:
        return False
    try:
        with sqlite3.connect(path) as conn:
            conn.execute("SELECT 1 FROM method_change LIMIT 1").fetchone()
        return True
    except sqlite3.Error:
        return False


def _replace_path_with_retry(src: Path, dst: Path, attempts: int = 12) -> None:
    """Windows may keep SQLite files locked briefly after close."""
    last_err: OSError | None = None
    for attempt in range(attempts):
        try:
            if dst.exists():
                dst.unlink()
            src.replace(dst)
            return
        except PermissionError as exc:
            last_err = exc
            gc.collect()
            time.sleep(0.3 * (attempt + 1))
    shutil.copy2(src, dst)
    try:
        src.unlink()
    except OSError:
        pass
    if last_err and not dst.exists():
        raise last_err


def _finalize_cvefixes_cache(build_path: Path, db_path: Path) -> None:
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(str(build_path) + suffix)
        if sidecar.exists():
            try:
                sidecar.unlink()
            except OSError:
                pass
    _replace_path_with_retry(build_path, db_path)


def _cvefixes_sqlite_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA cache_size = -200000")


def _cvefixes_create_indexes(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE INDEX IF NOT EXISTS idx_file_change_id ON file_change(file_change_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_file_change_hash ON file_change(hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_method_file_change ON method_change(file_change_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_commits_hash ON commits(hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fixes_hash ON fixes(hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cve_id ON cve(cve_id)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_cwe_classification_cve "
        "ON cwe_classification(cve_id)"
    )


def _import_sql_dump_streaming(conn: sqlite3.Connection, sql_path: Path) -> None:
    """Execute one SQL statement at a time — never load the whole dump into RAM."""
    buffer = ""
    statements = 0
    errors = 0
    commit_every = 2_000
    print(f"  CVEfixes: streaming SQL import from {sql_path.name} ...", flush=True)

    with _open_text(sql_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("--"):
                continue
            buffer = f"{buffer} {stripped}" if buffer else stripped
            if not stripped.endswith(";"):
                continue
            try:
                conn.execute(buffer)
            except sqlite3.Error:
                errors += 1
            buffer = ""
            statements += 1
            if statements % commit_every == 0:
                conn.commit()
                print(f"  CVEfixes SQL: {statements:,} statements imported ...", flush=True)

    if buffer.strip():
        try:
            conn.execute(buffer)
            statements += 1
        except sqlite3.Error:
            errors += 1
    conn.commit()
    if errors:
        print(f"  CVEfixes SQL: skipped {errors:,} malformed statement(s).", flush=True)
    print(f"  CVEfixes SQL: finished import ({statements:,} statements).", flush=True)


def _import_sql_dump_cli(sql_path: Path, db_path: Path) -> bool:
    """Use sqlite3 CLI when available — streams the file without Python holding it."""
    sqlite3_bin = shutil.which("sqlite3")
    if not sqlite3_bin:
        return False
    if db_path.exists():
        db_path.unlink()
    print(f"  CVEfixes: importing via sqlite3 CLI from {sql_path.name} ...", flush=True)
    with _open_text(sql_path) as sql_file:
        proc = subprocess.run(
            [sqlite3_bin, str(db_path)],
            stdin=sql_file,
            capture_output=True,
            text=True,
            errors="replace",
        )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()[:500]
        print(f"  CVEfixes: sqlite3 CLI failed ({err}); falling back to Python stream.", flush=True)
        if db_path.exists():
            db_path.unlink()
        return False
    print("  CVEfixes: sqlite3 CLI import complete.", flush=True)
    return True


def _ensure_cvefixes_cache(cfg: dict[str, Any], sql_path: Path) -> Path:
    db_path = _cvefixes_cache_db(cfg, sql_path)
    if db_path.exists() and db_path.stat().st_mtime >= sql_path.stat().st_mtime:
        return db_path

    build_path = _cvefixes_build_path(db_path)
    if _sqlite_cache_ready(build_path):
        print("  CVEfixes: reusing completed temp cache (finalizing) ...", flush=True)
        conn = sqlite3.connect(build_path)
        try:
            print("  CVEfixes: building indexes ...", flush=True)
            _cvefixes_create_indexes(conn)
        finally:
            _close_sqlite_connection(conn)
            del conn
            gc.collect()
        _finalize_cvefixes_cache(build_path, db_path)
        return db_path

    if build_path.exists():
        build_path.unlink()

    cve_cfg = cfg.get("cvefixes", {})
    use_cli = bool(cve_cfg.get("prefer_sqlite_cli", True))
    conn: sqlite3.Connection | None = None
    try:
        if use_cli and _import_sql_dump_cli(sql_path, build_path):
            conn = sqlite3.connect(build_path)
            _cvefixes_sqlite_pragmas(conn)
            print("  CVEfixes: building indexes ...", flush=True)
            _cvefixes_create_indexes(conn)
        else:
            conn = sqlite3.connect(build_path)
            _cvefixes_sqlite_pragmas(conn)
            _import_sql_dump_streaming(conn, sql_path)
            print("  CVEfixes: building indexes ...", flush=True)
            _cvefixes_create_indexes(conn)
    finally:
        _close_sqlite_connection(conn)
        del conn
        gc.collect()

    _finalize_cvefixes_cache(build_path, db_path)
    return db_path


def iter_cvefixes_sql(cfg: dict[str, Any]) -> Iterator[dict[str, Any]]:
    root = pcfg.resolve_path(cfg, cfg.get("cvefixes", {}).get("search_dir_key", "raw_cvefixes_dir"))
    if not root.exists():
        return

    sql_files: list[Path] = []
    for pattern in ("*.sql", "*.SQL", "*.sql.gz", "*.SQL.GZ"):
        sql_files.extend(root.glob(pattern))

    for sql_path in sorted(sql_files):
        db_path = _ensure_cvefixes_cache(cfg, sql_path)
        query = """
            WITH fix_one AS (
                SELECT hash, MIN(cve_id) AS cve_id
                FROM fixes
                GROUP BY hash
            ),
            cwe_agg AS (
                SELECT cve_id, group_concat(cwe_id, ';') AS cwe
                FROM cwe_classification
                GROUP BY cve_id
            )
            SELECT
                mc.method_change_id AS method_id,
                mc.code AS method_before,
                mc.before_change AS vul,
                fc.hash AS commit_hash,
                COALESCE(fc.filename, fc.new_path, fc.old_path) AS file_path,
                cm.committer_date AS committer_date,
                cm.author_date AS author_date,
                fx.cve_id AS cve,
                cv.published_date AS published_date,
                cv.description AS cve_desc,
                ca.cwe AS cwe
            FROM method_change mc
            JOIN file_change fc ON fc.file_change_id = mc.file_change_id
            LEFT JOIN commits cm ON cm.hash = fc.hash
            LEFT JOIN fix_one fx ON fx.hash = fc.hash
            LEFT JOIN cve cv ON cv.cve_id = fx.cve_id
            LEFT JOIN cwe_agg ca ON ca.cve_id = fx.cve_id
            WHERE lower(COALESCE(fc.programming_language, '')) IN ('c', 'c++', 'cpp', '', 'none')
        """
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            for row in conn.execute(query):
                record = dict(row)
                code = record.get("method_before") or ""
                if not isinstance(code, str) or len(code.strip()) < 10:
                    continue
                year, commit_date = _cvefixes_year_and_date(
                    record.get("cve"),
                    record.get("committer_date"),
                    record.get("author_date"),
                    record.get("published_date"),
                )
                yield {
                    "method_id": record.get("method_id"),
                    "method_before": code,
                    "vul": _bool_label(record.get("vul")),
                    "commit_hash": record.get("commit_hash"),
                    "year": year,
                    "commit_date": commit_date,
                    "cve": record.get("cve"),
                    "cwe": record.get("cwe"),
                    "cve_desc": _cvefixes_description(record.get("cve_desc")),
                    "file_path": record.get("file_path"),
                }


_MOREFIXES_METHOD_QUERY = """
    WITH fix_one AS (
        SELECT hash, MIN(cve_id) AS cve_id
        FROM fixes
        GROUP BY hash
    ),
    cwe_agg AS (
        SELECT cve_id, group_concat(cwe_id, ';') AS cwe
        FROM cwe_classification
        GROUP BY cve_id
    )
    SELECT
        mc.method_change_id AS method_id,
        mc.code AS method_before,
        mc.before_change AS vul,
        fc.hash AS commit_hash,
        COALESCE(fc.filename, fc.new_path, fc.old_path) AS file_path,
        cm.committer_date AS committer_date,
        cm.author_date AS author_date,
        fx.cve_id AS cve,
        cv.published_date AS published_date,
        cv.description AS cve_desc,
        ca.cwe AS cwe
    FROM method_change mc
    JOIN file_change fc ON fc.file_change_id = mc.file_change_id
    LEFT JOIN commits cm ON cm.hash = fc.hash
    LEFT JOIN fix_one fx ON fx.hash = fc.hash
    LEFT JOIN cve cv ON cv.cve_id = fx.cve_id
    LEFT JOIN cwe_agg ca ON ca.cve_id = fx.cve_id
    WHERE lower(COALESCE(fc.programming_language, '')) IN ('c', 'c++', 'cpp', '', 'none')
"""

_SECVULEVAL_PG_TABLES = frozenset(
    {"cve", "commits", "fixes", "file_change", "method_change", "cwe_classification"}
)

_C_CPP_PATCH_SUFFIXES = (
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
)


def _pg_copy_decode_line(line: str) -> list[str | None]:
    """Decode one PostgreSQL COPY text row (tab-separated, backslash escapes)."""
    fields: list[str | None] = []
    buf: list[str] = []
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if ch == "\\" and i + 1 < n:
            nxt = line[i + 1]
            if nxt == "n":
                buf.append("\n")
            elif nxt == "t":
                buf.append("\t")
            elif nxt == "r":
                buf.append("\r")
            elif nxt == "b":
                buf.append("\b")
            elif nxt == "f":
                buf.append("\f")
            elif nxt == "v":
                buf.append("\v")
            elif nxt == "\\":
                buf.append("\\")
            else:
                buf.append(nxt)
            i += 2
            continue
        if ch == "\t":
            token = "".join(buf)
            fields.append(None if token == "\\N" else token)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    token = "".join(buf)
    fields.append(None if token == "\\N" else token)
    return fields


def _parse_pg_copy_header(line: str) -> tuple[str, list[str]] | None:
    """Parse `COPY public.table (col1, col2, ...) FROM stdin;`."""
    m = re.match(
        r"^COPY\s+public\.(\w+)\s*\(([^)]+)\)\s+FROM\s+stdin\s*;\s*$",
        line.strip(),
        re.IGNORECASE,
    )
    if not m:
        return None
    table = m.group(1)
    columns = [c.strip().strip('"') for c in m.group(2).split(",")]
    return table, columns


def _secvuleval_cache_db(cfg: dict[str, Any], sql_path: Path) -> Path:
    interim = cfg["_base_dir"] / cfg.get("paths", {}).get("interim_dir", "data/interim")
    cache_dir = interim / "secvuleval_sqlite"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{sql_path.stem}.sqlite"


def _import_pg_copy_tables(conn: sqlite3.Connection, sql_path: Path) -> None:
    """Stream a PostgreSQL pg_dump and load selected COPY tables into SQLite."""
    current_table: str | None = None
    columns: list[str] | None = None
    placeholders: str | None = None
    insert_sql: str | None = None
    rows = 0
    errors = 0
    commit_every = 25_000

    print(f"  SecVulEval: streaming PostgreSQL COPY from {sql_path.name} ...", flush=True)

    with _open_text(sql_path) as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n\r")
            if not line:
                continue
            if line.startswith("COPY "):
                header = _parse_pg_copy_header(line)
                if header is None:
                    current_table = None
                    columns = None
                    continue
                table, columns = header
                if table not in _SECVULEVAL_PG_TABLES:
                    current_table = None
                    columns = None
                    continue
                col_defs = ", ".join(f'"{c}" TEXT' for c in columns)
                conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({col_defs})')
                placeholders = ", ".join("?" for _ in columns)
                insert_sql = f'INSERT INTO "{table}" VALUES ({placeholders})'
                current_table = table
                rows = 0
                continue
            if line == "\\.":
                if current_table and rows:
                    conn.commit()
                    print(
                        f"  SecVulEval COPY {current_table}: {rows:,} rows loaded.",
                        flush=True,
                    )
                current_table = None
                columns = None
                insert_sql = None
                continue
            if current_table is None or insert_sql is None or columns is None:
                continue
            try:
                values = _pg_copy_decode_line(line)
                if len(values) != len(columns):
                    errors += 1
                    continue
                conn.execute(insert_sql, values)
                rows += 1
                if rows % commit_every == 0:
                    conn.commit()
                    print(
                        f"  SecVulEval COPY {current_table}: {rows:,} rows ...",
                        flush=True,
                    )
            except sqlite3.Error:
                errors += 1

    conn.commit()
    if errors:
        print(f"  SecVulEval COPY: skipped {errors:,} malformed row(s).", flush=True)


def _ensure_secvuleval_cache(cfg: dict[str, Any], sql_path: Path) -> Path:
    db_path = _secvuleval_cache_db(cfg, sql_path)
    if db_path.exists() and db_path.stat().st_mtime >= sql_path.stat().st_mtime:
        return db_path

    build_path = _cvefixes_build_path(db_path)
    if _sqlite_cache_ready(build_path):
        print("  SecVulEval: reusing completed temp cache (finalizing) ...", flush=True)
        conn = sqlite3.connect(build_path)
        try:
            print("  SecVulEval: building indexes ...", flush=True)
            _cvefixes_create_indexes(conn)
        finally:
            _close_sqlite_connection(conn)
            del conn
            gc.collect()
        _finalize_cvefixes_cache(build_path, db_path)
        return db_path

    if build_path.exists():
        build_path.unlink()

    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(build_path)
        _cvefixes_sqlite_pragmas(conn)
        _import_pg_copy_tables(conn, sql_path)
        print("  SecVulEval: building indexes ...", flush=True)
        _cvefixes_create_indexes(conn)
    finally:
        _close_sqlite_connection(conn)
        del conn
        gc.collect()

    _finalize_cvefixes_cache(build_path, db_path)
    return db_path


def _resolve_secvuleval_sql_path(cfg: dict[str, Any]) -> Path | None:
    sec_cfg = cfg.get("secvuleval", {})
    key = sec_cfg.get("sql_path_key", "raw_secvuleval_sql")
    explicit = pcfg.resolve_path(cfg, key)
    if explicit.exists():
        return explicit
    root = pcfg.resolve_path(cfg, sec_cfg.get("search_dir_key", "raw_secvuleval_dir"))
    if not root.exists():
        return None
    for pattern in ("*.sql", "*.SQL", "*.sql.gz", "*.SQL.GZ"):
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    return None


def _morefixes_record_to_raw(record: dict[str, Any]) -> dict[str, Any] | None:
    code = record.get("method_before") or ""
    if not isinstance(code, str) or len(code.strip()) < 10:
        return None
    year, commit_date = _cvefixes_year_and_date(
        record.get("cve"),
        record.get("committer_date"),
        record.get("author_date"),
        record.get("published_date"),
    )
    return {
        "method_id": record.get("method_id"),
        "method_before": code,
        "vul": _bool_label(record.get("vul")),
        "commit_hash": record.get("commit_hash"),
        "year": year,
        "commit_date": commit_date,
        "cve": record.get("cve"),
        "cwe": record.get("cwe"),
        "cve_desc": _cvefixes_description(record.get("cve_desc")),
        "file_path": record.get("file_path"),
    }


def iter_secvuleval_sql(cfg: dict[str, Any]) -> Iterator[dict[str, Any]]:
    sql_path = _resolve_secvuleval_sql_path(cfg)
    if sql_path is None:
        return

    db_path = _ensure_secvuleval_cache(cfg, sql_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for row in conn.execute(_MOREFIXES_METHOD_QUERY):
            raw = _morefixes_record_to_raw(dict(row))
            if raw is not None:
                yield raw


def _secvuleval_patch_commit_hash(patch_path: Path) -> str | None:
    """Parse commit hash from `github.com_owner_repo_{hash}.patch`."""
    stem = patch_path.stem
    if "_" not in stem:
        return None
    commit = stem.rsplit("_", 1)[-1]
    if re.fullmatch(r"[0-9a-fA-F]{7,40}", commit):
        return commit.lower()
    return None


def _is_c_cpp_patch_path(path: str) -> bool:
    lower = path.lower().split("?", 1)[0]
    return any(lower.endswith(suffix) for suffix in _C_CPP_PATCH_SUFFIXES)


def _extract_vulnerable_snippets_from_patch(
    patch_text: str, *, min_lines: int
) -> list[tuple[str, str]]:
    """Return (file_path, code) for removed (-) hunks in C/C++ files."""
    snippets: list[tuple[str, str]] = []
    current_file: str | None = None
    removed: list[str] = []

    def flush() -> None:
        nonlocal removed, current_file
        if current_file and len(removed) >= min_lines:
            code = "\n".join(removed).strip()
            if len(code) >= 10:
                snippets.append((current_file, code))
        removed = []

    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            flush()
            parts = line.split()
            if len(parts) >= 4:
                b_path = parts[3]
                if b_path.startswith("b/"):
                    current_file = b_path[2:]
                else:
                    current_file = b_path
            else:
                current_file = None
            if current_file and not _is_c_cpp_patch_path(current_file):
                current_file = None
            continue
        if line.startswith("+++ ") or line.startswith("--- "):
            continue
        if current_file is None:
            continue
        if line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])
        elif line.startswith("+") or line.startswith(" "):
            flush()
    flush()
    return snippets


def _secvuleval_commit_meta(
    conn: sqlite3.Connection, commit_hash: str
) -> dict[str, Any] | None:
    row = conn.execute(
        """
        SELECT
            cm.hash AS commit_hash,
            cm.committer_date AS committer_date,
            cm.author_date AS author_date,
            fx.cve_id AS cve,
            cv.published_date AS published_date,
            cv.description AS cve_desc,
            (
                SELECT group_concat(cwe_id, ';')
                FROM cwe_classification cc
                WHERE cc.cve_id = fx.cve_id
            ) AS cwe
        FROM commits cm
        LEFT JOIN fixes fx ON fx.hash = cm.hash
        LEFT JOIN cve cv ON cv.cve_id = fx.cve_id
        WHERE cm.hash = ?
        LIMIT 1
        """,
        (commit_hash,),
    ).fetchone()
    return dict(row) if row else None


def iter_secvuleval_patches(cfg: dict[str, Any]) -> Iterator[dict[str, Any]]:
    sec_cfg = cfg.get("secvuleval", {})
    if not bool(sec_cfg.get("include_patches", True)):
        return

    patches_dir = pcfg.resolve_path(cfg, sec_cfg.get("patches_dir_key", "raw_secvuleval_patches"))
    if not patches_dir.is_dir():
        return

    sql_path = _resolve_secvuleval_sql_path(cfg)
    conn: sqlite3.Connection | None = None
    if sql_path is not None:
        db_path = _ensure_secvuleval_cache(cfg, sql_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

    min_lines = int(sec_cfg.get("patch_min_lines", 3))
    try:
        for patch_path in sorted(patches_dir.glob("*.patch")):
            commit_hash = _secvuleval_patch_commit_hash(patch_path)
            if not commit_hash:
                continue
            meta: dict[str, Any] = {}
            if conn is not None:
                row = _secvuleval_commit_meta(conn, commit_hash)
                if row:
                    meta = row
            year, commit_date = _cvefixes_year_and_date(
                meta.get("cve"),
                meta.get("committer_date"),
                meta.get("author_date"),
                meta.get("published_date"),
            )
            try:
                patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for idx, (file_path, code) in enumerate(
                _extract_vulnerable_snippets_from_patch(patch_text, min_lines=min_lines)
            ):
                file_slug = re.sub(r"[^A-Za-z0-9]+", "_", file_path)[-40:]
                yield {
                    "method_id": f"patch_{commit_hash[:12]}_{file_slug}_{idx}",
                    "method_before": code,
                    "vul": 1,
                    "commit_hash": commit_hash,
                    "year": year,
                    "commit_date": commit_date,
                    "cve": meta.get("cve"),
                    "cwe": meta.get("cwe"),
                    "cve_desc": _cvefixes_description(meta.get("cve_desc")),
                    "file_path": file_path,
                }
    finally:
        _close_sqlite_connection(conn)


def iter_secvuleval_rows(cfg: dict[str, Any]) -> Iterator[dict[str, Any]]:
    yield from iter_secvuleval_sql(cfg)
    yield from iter_secvuleval_patches(cfg)


def map_secvuleval_to_core(
    raw: dict[str, Any],
    nulls: set[str],
    columns: list[str],
) -> dict[str, Any] | None:
    code = raw.get("method_before") or ""
    if not isinstance(code, str) or not code.strip():
        return None
    label = _coerce_label(raw.get("vul", 0))
    raw_id = raw.get("method_id", "unknown")
    cve = _stringify_field(raw.get("cve"), nulls)
    nvd_url = f"https://nvd.nist.gov/vuln/detail/{cve}" if cve else None

    row: dict[str, Any] = {
        "id": f"secvuleval_{raw_id}",
        "code": code,
        "label": label,
        "split": None,
        "source_dataset": "secvuleval",
        "year": raw.get("year"),
        "commit_date": raw.get("commit_date"),
        "commit_hash": _stringify_field(raw.get("commit_hash"), nulls),
        "cwe": _stringify_field(raw.get("cwe"), nulls),
        "cve": cve,
        "cve_desc": _stringify_field(raw.get("cve_desc"), nulls),
        "nvd_url": nvd_url,
    }
    for col in columns:
        if col not in row:
            row[col] = None
    return {c: row.get(c) for c in columns}


def _primevul_split_from_name(path: Path) -> str | None:
    name = path.stem.lower()
    for split in ("train", "valid", "test"):
        if split in name:
            return split
    return None


def iter_primevul_rows(cfg: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """All PrimeVul *.jsonl under jsonl/ and paired/ (combined, no field drops yet)."""
    mapping = pcfg.mapping_for_dataset(cfg, "primevul")
    root = pcfg.resolve_path(cfg, "raw_primevul_dir")
    subdirs = mapping.get("jsonl_subdirs", ["jsonl", "paired"])
    paths: list[Path] = []
    for sub in subdirs:
        d = root / sub
        if d.is_dir():
            paths.extend(sorted(d.glob("*.jsonl")))
    if not paths:
        for key in ("raw_primevul_train", "raw_primevul_valid", "raw_primevul_test"):
            p = pcfg.resolve_path(cfg, key)
            if p.exists():
                paths.append(p)
    for path in paths:
        split = _primevul_split_from_name(path)
        if path.suffix.lower() == ".parquet":
            for row in _iter_parquet(path):
                if split and not row.get("split"):
                    row["split"] = split
                yield row
        else:
            yield from _iter_jsonl(path, split)


def map_primevul_to_core(
    raw: dict[str, Any],
    nulls: set[str],
    columns: list[str],
) -> dict[str, Any] | None:
    code = raw.get("func") or ""
    if not isinstance(code, str) or not code.strip():
        return None
    try:
        label = _coerce_label(raw["target"])
    except (KeyError, TypeError):
        return None

    split = raw.get("split")
    if isinstance(split, str):
        split = split.strip().lower() or None
    elif split is not None:
        split = str(split).strip().lower() or None

    raw_id = raw.get("idx", "unknown")
    row_id = f"primevul_{split}_{raw_id}" if split else f"primevul_{raw_id}"

    year = _extract_primevul_year(raw)
    row: dict[str, Any] = {
        "id": row_id,
        "code": code,
        "label": label,
        "split": split,
        "source_dataset": "primevul",
        "year": year,
        "commit_date": f"{year}-01-01" if year is not None else None,
        "commit_hash": _stringify_field(raw.get("commit_id"), nulls),
    }
    for field in PRIMEVUL_KEEP_FIELDS:
        row[field] = _stringify_field(raw.get(field), nulls)
    for col in columns:
        if col not in row:
            row[col] = None
    return {c: row.get(c) for c in columns}


def map_diversevul_to_core(
    raw: dict[str, Any],
    cfg: dict[str, Any],
    nulls: set[str],
    columns: list[str],
    meta_by_commit: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    code = raw.get("func") or ""
    if not isinstance(code, str) or not code.strip():
        return None
    try:
        label = _coerce_label(raw["target"])
    except (KeyError, TypeError):
        return None

    commit_id = raw.get("commit_id")
    meta = None
    if commit_id and meta_by_commit:
        meta = meta_by_commit.get(str(commit_id).strip())

    year, commit_date, cve, cwe, cve_desc, nvd_url = _extract_diversevul_temporal_and_vuln_fields(
        raw, meta, nulls
    )
    raw_id = raw.get("hash", raw.get("commit_id", "unknown"))

    row: dict[str, Any] = {
        "id": f"diversevul_{raw_id}",
        "code": code,
        "label": label,
        "split": None,
        "source_dataset": "diversevul",
        "year": year,
        "commit_date": commit_date,
        "commit_hash": _stringify_field(commit_id, nulls),
        "cwe": cwe,
        "cve": cve,
        "cve_desc": cve_desc,
        "nvd_url": nvd_url,
    }
    for col in columns:
        if col not in row:
            row[col] = None
    return {c: row.get(c) for c in columns}


def map_bigvul_to_core(
    raw: dict[str, Any],
    cfg: dict[str, Any],
    nulls: set[str],
    columns: list[str],
) -> dict[str, Any] | None:
    mapping = pcfg.mapping_for_dataset(cfg, "bigvul")
    code = raw.get(mapping.get("code", "func_before")) or ""
    if not isinstance(code, str) or not code.strip():
        return None
    try:
        label = _coerce_label(raw[mapping["label"]])
    except (KeyError, TypeError):
        return None

    id_field = mapping.get("id_field", "Unnamed: 0")
    raw_id = raw.get(id_field, raw.get("commit_id", "unknown"))
    year, commit_date = _extract_bigvul_year_and_date(raw, mapping, nulls)

    cve_key = mapping.get("cve_field", "CVE ID")
    cwe_key = mapping.get("cwe_field", "CWE ID")

    row: dict[str, Any] = {
        "id": f"bigvul_{raw_id}",
        "code": code,
        "label": label,
        "split": None,
        "source_dataset": "bigvul",
        "year": year,
        "commit_date": commit_date,
        "commit_hash": _stringify_field(raw.get("commit_id"), nulls),
        "cwe": _stringify_field(raw.get(cwe_key), nulls),
        "cve": _stringify_field(raw.get(cve_key), nulls),
    }
    for col in columns:
        if col not in row:
            row[col] = None
    return {c: row.get(c) for c in columns}


def map_cvefixes_to_core(
    raw: dict[str, Any],
    nulls: set[str],
    columns: list[str],
) -> dict[str, Any] | None:
    code = raw.get("method_before") or ""
    if not isinstance(code, str) or not code.strip():
        return None
    label = _coerce_label(raw.get("vul", 0))
    raw_id = raw.get("method_id", "unknown")

    row: dict[str, Any] = {
        "id": f"cvefixes_{raw_id}",
        "code": code,
        "label": label,
        "split": None,
        "source_dataset": "cvefixes",
        "year": raw.get("year"),
        "commit_date": raw.get("commit_date"),
        "commit_hash": _stringify_field(raw.get("commit_hash"), nulls),
        "cwe": _stringify_field(raw.get("cwe"), nulls),
        "cve": _stringify_field(raw.get("cve"), nulls),
        "cve_desc": _stringify_field(raw.get("cve_desc"), nulls),
    }
    for col in columns:
        if col not in row:
            row[col] = None
    return {c: row.get(c) for c in columns}


def iter_source_rows(cfg: dict[str, Any], source: str) -> Iterator[dict[str, Any]]:
    if source == "primevul":
        yield from iter_primevul_rows(cfg)
        return

    if source == "diversevul":
        path = pcfg.resolve_path(cfg, "raw_diversevul")
        if path.exists():
            if path.suffix.lower() == ".parquet":
                yield from _iter_parquet(path)
            else:
                yield from _iter_jsonl(path)
        return

    if source == "bigvul":
        path = pcfg.resolve_path(cfg, cfg.get("bigvul", {}).get("path_key", "raw_bigvul_cleaned"))
        if not path.exists():
            return
        if path.suffix.lower() == ".parquet":
            yield from _iter_parquet(path)
        else:
            chunksize = int(cfg.get("bigvul", {}).get("chunksize", 10000))
            yield from _iter_csv(path, chunksize)
        return

    if source == "cvefixes":
        jsonl_path = pcfg.resolve_path(
            cfg, cfg.get("cvefixes", {}).get("jsonl_path_key", "raw_cvefixes_jsonl")
        )
        if jsonl_path.exists():
            yield from _iter_jsonl(jsonl_path)
            return
        yield from iter_cvefixes_sql(cfg)
        return

    if source == "secvuleval":
        yield from iter_secvuleval_rows(cfg)
        return


def stream_normalize_to_chunks(
    cfg: dict[str, Any],
    stage: str,
    writer: StreamingChunkWriter,
    stats: NormStats,
    *,
    dataset: str | None = None,
) -> None:
    nulls = pcfg.null_placeholders(cfg)
    columns = pcfg.core_columns(cfg)
    sources = [dataset] if dataset else list(pcfg.active_sources(cfg, stage))
    if dataset:
        print(f"[normalizer] single-dataset mode: {dataset} only", flush=True)
    else:
        print(f"[normalizer] merging sources for stage {stage}: {', '.join(sources)}", flush=True)

    for source in sources:
        mapping = pcfg.mapping_for_dataset(cfg, source)
        diversevul_meta = (
            load_diversevul_metadata_index(cfg, mapping)
            if source == "diversevul"
            else None
        )
        loaded = 0
        for raw in iter_source_rows(cfg, source):
            if source == "primevul":
                core = map_primevul_to_core(raw, nulls, columns)
            elif source == "diversevul":
                core = map_diversevul_to_core(
                    raw, cfg, nulls, columns, diversevul_meta
                )
                if core is not None and core.get("year") is None:
                    stats.bump(source, "no_year")
            elif source == "bigvul":
                if not lang_allowed(raw, mapping):
                    stats.rows_skipped += 1
                    stats.bump(source, "skip_lang")
                    continue
                core = map_bigvul_to_core(raw, cfg, nulls, columns)
            elif source == "cvefixes":
                core = map_cvefixes_to_core(raw, nulls, columns)
            elif source == "secvuleval":
                core = map_secvuleval_to_core(raw, nulls, columns)
                if core is not None and core.get("year") is None:
                    stats.bump(source, "no_year")
            else:
                if not lang_allowed(raw, mapping):
                    stats.rows_skipped += 1
                    stats.bump(source, "skip_lang")
                    continue
                core = map_raw_to_core(raw, mapping, source, nulls, columns)
            if core is None:
                stats.rows_skipped += 1
                stats.bump(source, "skip_bad_row")
                continue
            writer.add_row(core)
            loaded += 1
            stats.rows_mapped += 1

        if loaded:
            stats.sources_loaded += 1
            stats.by_source.setdefault(source, {})["rows"] = loaded
            print(f"[normalizer] {source}: {loaded:,} rows", flush=True)
        else:
            print(f"[normalizer] WARNING: {source} yielded 0 rows (check raw paths)", flush=True)


def run_normalize(
    cfg: dict[str, Any],
    stage: str,
    force: bool = False,
    *,
    dataset: str | None = None,
) -> Path:
    ncfg = pcfg.normalizer_cfg(cfg)
    out_dir = pcfg.normalized_output_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = str(ncfg.get("output_prefix", "normalized"))
    n_chunks = int(ncfg.get("output_chunks", 1))
    existing = list(out_dir.glob(f"{prefix}_*.parquet"))
    if existing and not force:
        names = ", ".join(p.name for p in existing[:3])
        print(
            f"Skipping normalizer: {len(existing)} file(s) already exist "
            f"({names}{'...' if len(existing) > 3 else ''}).",
            flush=True,
        )
        print(
            "Reusing old normalized Parquet — sources will NOT be re-merged. "
            "For a full stage 1c merge, run with --force (normalizer CLI) or "
            "run_pipeline.py --force-normalize.",
            flush=True,
        )
        return out_dir

    for path in existing:
        path.unlink()

    flush_rows = int(ncfg.get("flush_rows", 25_000))
    columns = pcfg.core_columns(cfg)
    target = dataset or f"stage {stage} sources"
    print(
        f"Normalizer: {target} (flush every {flush_rows:,} rows, "
        f"{n_chunks} output chunk file(s))."
    )

    stats = NormStats()
    writer = StreamingChunkWriter(
        out_dir, prefix, columns, n_chunks=n_chunks, flush_rows=flush_rows
    )
    stream_normalize_to_chunks(cfg, stage, writer, stats, dataset=dataset)
    paths = writer.close()
    stats.chunks_written = len(paths)

    report = cfg["_base_dir"] / ncfg.get("report", "reports/normalizer_report.md")
    _write_report(stats, report, stage, out_dir, paths, pcfg.core_columns(cfg), n_chunks)
    return out_dir


def _write_report(
    stats: NormStats,
    path: Path,
    stage: str,
    out_dir: Path,
    written: list[Path],
    columns: list[str],
    n_chunks: int,
) -> None:
    lines = [
        f"# Normalizer report (stage {stage})",
        "",
        f"- sources loaded: {stats.sources_loaded}",
        f"- rows mapped: {stats.rows_mapped}",
        f"- rows skipped: {stats.rows_skipped}",
        f"- output chunks: {stats.chunks_written} (configured: {n_chunks})",
        f"- core columns: `{', '.join(columns)}`",
        f"- output dir: `{out_dir.as_posix()}`",
        "",
        "## Output files",
        "",
    ]
    for p in written:
        lines.append(f"- `{p.name}`")
    lines.extend(["", "## Per source", ""])
    for src, counts in sorted(stats.by_source.items()):
        lines.append(f"### {src}")
        for k, v in sorted(counts.items()):
            lines.append(f"- {k}: {v}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize raw datasets to combined Parquet")
    parser.add_argument("--stage", choices=["1a", "1b", "1c"], default=None)
    parser.add_argument(
        "--dataset",
        default=None,
        help="Process one dataset only (e.g. primevul). Default: all active sources for stage.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = pcfg.load_config(args.config)
    stage = args.stage or str(cfg.get("stage", "1a"))
    out = run_normalize(cfg, stage, force=args.force, dataset=args.dataset)
    print(f"Normalized Parquet under: {out}")


if __name__ == "__main__":
    main()
