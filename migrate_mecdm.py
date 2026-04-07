#!/usr/bin/env python3
"""
MECDM SuperApp — AlloyDB/PostgreSQL Migration Script
Meghalaya Early Childhood Development Mission

Migrates all geographic, master-reference, health, and raw-data tables
from the local mecdm_dataset folder into AlloyDB (PostGIS-enabled Postgres).

Tables created:
  Geographic  : states, districts, district_boundaries, subdistricts, blocks,
                villages_poly, villages_point
  Master ref  : master_districts, master_blocks, master_villages,
                master_health_facilities, geo_full_mapping, geo_match_report
  Infrastructure: health_facilities, anganwadi_centres
  Geography   : meghealth_geo_mapping
  Health data : anc_visits, village_indicators_monthly
  Mother App  : mothers, mother_anc_visits_flat, mother_children
                (curated Mother App dataset — replaces legacy mother_journeys
                 and raw_pregnancy_records)
  Raw records : raw_anc_records, raw_child_records
  Reference   : nfhs_indicators, video_library, research_articles

Usage:
    python migrate_mecdm.py                        # full migration
    python migrate_mecdm.py --skip-raw-json        # skip large raw JSON files (~961 MB)
    python migrate_mecdm.py --only geo             # geographic tables only
    python migrate_mecdm.py --only master          # master/reference tables only
    python migrate_mecdm.py --only infra           # infrastructure tables only
    python migrate_mecdm.py --only health          # health CSV tables only
    python migrate_mecdm.py --only mother_app      # Mother App flattened CSV → mothers/anc/children
    python migrate_mecdm.py --only raw_json        # raw JSON tables only
    python migrate_mecdm.py --only ref_json        # reference JSON tables only
    python migrate_mecdm.py --only schema          # PKs/FKs/indexes/comments only
    python migrate_mecdm.py --drop                 # drop all 25 tables then run full migration
    python migrate_mecdm.py --drop --only health   # drop only health-stage tables, then reload
    python migrate_mecdm.py --truncate             # truncate all tables (keep schema) then reload
    python migrate_mecdm.py --truncate --only geo  # truncate only geographic tables

Environment variables (see .env):
    ALLOYDB_POSTGRES_USER, ALLOYDB_POSTGRES_PASSWORD,
    ALLOYDB_POSTGRES_HOST, ALLOYDB_POSTGRES_PORT,
    ALLOYDB_POSTGRES_DATABASE
    ALLOYDB_SSLMODE  (default: require — set to 'disable' for local dev without proxy)
"""

import argparse
import io
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import geopandas as gpd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from geoalchemy2 import Geometry
from sqlalchemy import create_engine, inspect, text

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
BASE_DIR = _HERE / "mecdm_dataset" / "db_20260226"
GEO_DIR = BASE_DIR / "Meghalaya Master Data"
OUTPUT_DIR = BASE_DIR / "output"
MASTER_DIR = OUTPUT_DIR / "master"

# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------
CHUNKSIZE = 5_000   # rows per pandas read_csv chunk
BATCH_SIZE = 5_000  # rows per COPY batch (for JSON → relational)

# ---------------------------------------------------------------------------
# Table registry — used by --drop and --truncate
# Child tables are listed before parent tables so DROP without CASCADE is safe,
# though we always use CASCADE anyway for convenience.
# ---------------------------------------------------------------------------
STAGE_TABLES: dict[str, list[str]] = {
    "geo": [
        "villages_point", "villages_poly",
        "blocks", "subdistricts",
        "district_boundaries", "districts", "states",
    ],
    "master": [
        "geo_match_report", "geo_full_mapping",
        "master_villages", "master_health_facilities",
        "master_blocks", "master_districts",
        "meghealth_geo_mapping",
    ],
    "infra": [
        "anganwadi_centres", "health_facilities",
    ],
    "health": [
        "anc_visits",
        "village_indicators_monthly",
    ],
    "mother_app": [
        # Children listed before parent for safe sequential DROP
        "mother_children", "mother_anc_visits_flat", "mothers",
    ],
    "raw_json": [
        "raw_anc_records", "raw_child_records",
    ],
    "ref_json": [
        "nfhs_indicators", "video_library", "research_articles",
    ],
}
# Flat ordered list — children before parents for safe sequential DROP
ALL_TABLES: list[str] = [t for tables in STAGE_TABLES.values() for t in tables]


# ===========================================================================
# Engine helpers
# ===========================================================================

def _dsn() -> str:
    user = os.environ.get("ALLOYDB_POSTGRES_USER", "postgres")
    password = os.environ.get("ALLOYDB_POSTGRES_PASSWORD", "postgres")
    host = os.environ.get("ALLOYDB_POSTGRES_HOST", "127.0.1.1")
    port = os.environ.get("ALLOYDB_POSTGRES_PORT", "5432")
    db = os.environ.get("ALLOYDB_POSTGRES_DATABASE", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def get_engine():
    """SQLAlchemy engine using psycopg2. SSL controlled by ALLOYDB_SSLMODE env var."""
    sslmode = os.environ.get("ALLOYDB_SSLMODE", "require")
    return create_engine(
        _dsn(),
        connect_args={"sslmode": sslmode},
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=2,
    )


def get_psycopg2_conn():
    """Raw psycopg2 connection for COPY operations."""
    user = os.environ.get("ALLOYDB_POSTGRES_USER", "postgres")
    password = os.environ.get("ALLOYDB_POSTGRES_PASSWORD", "postgres")
    host = os.environ.get("ALLOYDB_POSTGRES_HOST", "127.0.1.1")
    port = os.environ.get("ALLOYDB_POSTGRES_PORT", "5432")
    db = os.environ.get("ALLOYDB_POSTGRES_DATABASE", "postgres")
    sslmode = os.environ.get("ALLOYDB_SSLMODE", "require")
    return psycopg2.connect(
        host=host, port=port, dbname=db,
        user=user, password=password, sslmode=sslmode,
    )


# ===========================================================================
# Utility helpers
# ===========================================================================

def table_exists(engine, table_name: str) -> bool:
    return table_name in inspect(engine).get_table_names(schema="public")


def get_row_count(engine, table_name: str) -> int:
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()


def _drop_cascade(engine, table_name: str) -> None:
    """
    DROP TABLE IF EXISTS ... CASCADE before a reload.

    pandas/geopandas `to_sql(if_exists='replace')` issues a plain DROP TABLE,
    which fails when FK constraints from a previous migration run depend on the
    table.  Calling this first removes those dependencies cleanly.
    """
    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))


def create_extensions(engine):
    logger.info("Ensuring PostGIS extension is installed…")
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))


def _exec_sql(engine, statements: list[str], label: str):
    """
    Execute each SQL statement in its own transaction.

    A single failed statement (e.g. duplicate key, missing column) does NOT
    abort subsequent statements because each runs in an isolated transaction.
    """
    for sql in statements:
        try:
            with engine.begin() as conn:
                conn.execute(text(sql))
        except Exception as exc:
            logger.warning(f"[{label}] skipped: {exc!s:.120}")


def drop_tables(engine, tables: list[str]):
    """
    DROP TABLE IF EXISTS … CASCADE for each table.

    Uses CASCADE so that foreign-key dependencies are handled automatically.
    Tables that don't exist are silently skipped. Schema objects (indexes,
    sequences, comments) attached to the tables are removed along with them.
    """
    logger.info(f"Dropping {len(tables)} table(s)…")
    with engine.begin() as conn:
        for tbl in tables:
            try:
                conn.execute(text(f"DROP TABLE IF EXISTS {tbl} CASCADE;"))
                logger.info(f"  dropped {tbl}")
            except Exception as exc:
                logger.warning(f"  could not drop {tbl}: {exc!s:.120}")


def truncate_tables(engine, tables: list[str]):
    """
    TRUNCATE all tables that currently exist, in a single statement with CASCADE.

    Unlike DROP, TRUNCATE keeps the schema intact (columns, indexes, PKs, FKs,
    comments). Useful for reloading data without rebuilding the full schema.
    RESTART IDENTITY resets all sequences to their start values.
    """
    existing = [t for t in tables if table_exists(engine, t)]
    if not existing:
        logger.info("  No tables to truncate (none exist yet).")
        return
    tbl_list = ", ".join(existing)
    logger.info(f"Truncating {len(existing)} table(s): {tbl_list}…")
    with engine.begin() as conn:
        try:
            conn.execute(text(
                f"TRUNCATE TABLE {tbl_list} RESTART IDENTITY CASCADE;"
            ))
            logger.info("  Truncate complete.")
        except Exception as exc:
            logger.warning(f"  Truncate failed: {exc!s:.160}")


# ===========================================================================
# Loaders
# ===========================================================================

def load_geojson(engine, filepath: str | Path, table_name: str, comment: str = ""):
    """Load a GeoJSON file into a PostGIS table (geometry column: 'geometry')."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"File not found, skipping {table_name}: {filepath}")
        return
    logger.info(f"Loading {filepath.name} → {table_name}…")
    try:
        gdf = gpd.read_file(filepath)
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        # Normalise the GIS object-ID column to 'id' so all geo tables share a
        # consistent integer PK name.  The source GeoJSON files use 'objectid'
        # or 'objectid_1'.  Some files also have a string 'id' column (a legacy
        # Census/GIS code); we preserve it as 'legacy_id' when non-null, or drop
        # it when all-null, before renaming objectid → id.
        if "objectid" in gdf.columns:
            if "id" in gdf.columns:
                if gdf["id"].notna().any():
                    gdf = gdf.rename(columns={"id": "legacy_id"})
                else:
                    gdf = gdf.drop(columns=["id"])
            gdf = gdf.rename(columns={"objectid": "id"})
        elif "objectid_1" in gdf.columns:
            if "id" in gdf.columns:
                if gdf["id"].notna().any():
                    gdf = gdf.rename(columns={"id": "legacy_id"})
                else:
                    gdf = gdf.drop(columns=["id"])
            gdf = gdf.rename(columns={"objectid_1": "id"})
        geom_type = gdf.geometry.geom_type.mode(
        )[0].upper() if not gdf.empty else "GEOMETRY"
        _drop_cascade(engine, table_name)
        gdf.to_postgis(
            table_name, engine,
            if_exists="replace", index=False,
            dtype={"geometry": Geometry(geom_type, srid=4326)},
        )
        logger.info(f"  ✓ {table_name}: {len(gdf):,} features")
    except Exception as exc:
        logger.error(f"Failed {table_name}: {exc}")
        raise


def load_csv_small(engine, filepath: str | Path, table_name: str, comment: str = ""):
    """Load a small CSV file via pandas to_sql (replace)."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"File not found, skipping {table_name}: {filepath}")
        return
    logger.info(f"Loading {filepath.name} → {table_name}…")
    try:
        df = pd.read_csv(filepath, low_memory=False)
        _drop_cascade(engine, table_name)
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        logger.info(f"  ✓ {table_name}: {len(df):,} rows")
    except Exception as exc:
        logger.error(f"Failed {table_name}: {exc}")
        raise


def load_csv_with_points(
    engine,
    filepath: str | Path,
    table_name: str,
    lat_col: str,
    lon_col: str,
    comment: str = "",
):
    """Load a CSV with lat/lon columns, creating a PostGIS POINT geometry column 'geom'."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"File not found, skipping {table_name}: {filepath}")
        return
    logger.info(
        f"Loading {filepath.name} → {table_name} (with POINT geometry)…")
    try:
        df = pd.read_csv(filepath, low_memory=False)
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

        valid = df[df[lat_col].notna() & df[lon_col].notna()].copy()
        invalid = df[df[lat_col].isna() | df[lon_col].isna()].copy()

        if valid.empty:
            logger.warning(
                f"  No valid coordinates in {table_name}, loading without geometry")
            _drop_cascade(engine, table_name)
            df.to_sql(table_name, engine, if_exists="replace", index=False)
            return

        gdf = gpd.GeoDataFrame(
            valid,
            geometry=gpd.points_from_xy(valid[lon_col], valid[lat_col]),
            crs="EPSG:4326",
        )
        gdf.rename_geometry("geom", inplace=True)
        _drop_cascade(engine, table_name)
        gdf.to_postgis(
            table_name, engine,
            if_exists="replace", index=False,
            dtype={"geom": Geometry("POINT", srid=4326)},
        )
        if not invalid.empty:
            logger.warning(
                f"  {len(invalid)} rows missing coordinates — appended without geometry")
            invalid.to_sql(table_name, engine, if_exists="append", index=False)

        logger.info(
            f"  ✓ {table_name}: {len(df):,} rows ({len(valid):,} with geometry)")
    except Exception as exc:
        logger.error(f"Failed {table_name}: {exc}")
        raise


def load_csv_large(
    engine,
    filepath: str | Path,
    table_name: str,
    comment: str = "",
    skip_if_exists: bool = False,
):
    """
    High-performance CSV loader using psycopg2 COPY FROM STDIN.

    Strategy:
      1. Read the first chunk to create an empty table via to_sql (establishes DDL).
      2. Stream all chunks through psycopg2 copy_expert (~10× faster than INSERT).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"File not found, skipping {table_name}: {filepath}")
        return
    if skip_if_exists and table_exists(engine, table_name):
        count = get_row_count(engine, table_name)
        if count > 0:
            logger.info(f"Skipping {table_name} — already has {count:,} rows")
            return

    size_mb = filepath.stat().st_size / 1024 / 1024
    logger.info(
        f"Loading {filepath.name} → {table_name} ({size_mb:.0f} MB, COPY protocol)…")
    _drop_cascade(engine, table_name)

    reader = pd.read_csv(
        filepath,
        chunksize=CHUNKSIZE,
        low_memory=False,
        dtype=str,
        keep_default_na=True,
        na_values=["", "NA", "N/A", "null", "NULL", "None"],
    )

    table_created = False
    total_rows = 0
    start_time = datetime.now()
    raw_conn = get_psycopg2_conn()
    cursor = raw_conn.cursor()

    try:
        for chunk_num, chunk in enumerate(reader):
            if not table_created:
                # Create empty table structure from first chunk
                chunk.head(0).to_sql(table_name, engine,
                                     if_exists="replace", index=False)
                table_created = True

            buf = io.StringIO()
            if chunk_num == 0:
                chunk.to_csv(buf, index=False, na_rep="")
                buf.seek(0)
                cursor.copy_expert(
                    f"COPY {table_name} FROM STDIN WITH (FORMAT CSV, HEADER, NULL '')",
                    buf,
                )
            else:
                chunk.to_csv(buf, index=False, header=False, na_rep="")
                buf.seek(0)
                cursor.copy_expert(
                    f"COPY {table_name} FROM STDIN WITH (FORMAT CSV, NULL '')",
                    buf,
                )

            total_rows += len(chunk)
            if total_rows % 50_000 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = total_rows / elapsed if elapsed > 0 else 0
                logger.info(f"  … {total_rows:,} rows ({rate:,.0f} rows/s)")
                raw_conn.commit()

        raw_conn.commit()
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"  ✓ {table_name}: {total_rows:,} rows in {elapsed:.1f}s")
    except Exception as exc:
        raw_conn.rollback()
        logger.error(f"Failed {table_name}: {exc}")
        raise
    finally:
        cursor.close()
        raw_conn.close()


def load_json_flat(
    engine,
    filepath: str | Path,
    table_name: str,
    comment: str = "",
    array_cols: list[str] | None = None,
):
    """
    Load a small structured JSON file into a flat relational table.

    List-valued columns are serialized to JSON strings for flexibility.
    Nested dict columns are flattened via pd.json_normalize.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"File not found, skipping {table_name}: {filepath}")
        return
    logger.info(f"Loading {filepath.name} → {table_name} (flat JSON)…")
    try:
        with open(filepath) as f:
            data = json.load(f)

        if isinstance(data, dict):
            keys = list(data.keys())
            if len(keys) == 1 and isinstance(data[keys[0]], list):
                data = data[keys[0]]
            else:
                data = [data]

        df = pd.json_normalize(data)

        # Serialize list/dict columns to JSON strings
        for col in df.columns:
            sample = df[col].dropna().head(5)
            if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict)):
                df[col] = df[col].apply(
                    lambda x: json.dumps(x) if isinstance(
                        x, (list, dict)) else x
                )

        _drop_cascade(engine, table_name)
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        logger.info(f"  ✓ {table_name}: {len(df):,} rows")
    except Exception as exc:
        logger.error(f"Failed {table_name}: {exc}")
        raise


def load_json_normalized(
    engine,
    filepath: str | Path,
    table_name: str,
    comment: str = "",
    skip_if_exists: bool = False,
):
    """
    Load a large JSON array file into a relational table.

    All fields are stored as typed columns (relational, not JSONB).
    Uses psycopg2 COPY for high-throughput bulk insert.

    Note: Loads the entire file into memory. For the ~375-394 MB raw
    record files this requires ~2-3 GB RAM. Use --skip-raw-json to bypass.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"File not found, skipping {table_name}: {filepath}")
        return
    if skip_if_exists and table_exists(engine, table_name):
        count = get_row_count(engine, table_name)
        if count > 0:
            logger.info(f"Skipping {table_name} — already has {count:,} rows")
            return

    size_mb = filepath.stat().st_size / 1024 / 1024
    logger.info(
        f"Loading {filepath.name} → {table_name} ({size_mb:.0f} MB, normalized JSON)…")

    logger.info("  Reading JSON into memory…")
    with open(filepath) as f:
        data = json.load(f)

    if isinstance(data, dict):
        keys = list(data.keys())
        if len(keys) == 1 and isinstance(data[keys[0]], list):
            data = data[keys[0]]
        else:
            data = [data]

    logger.info(f"  Normalizing {len(data):,} records…")
    df = pd.json_normalize(data)

    # Serialize any remaining list/dict columns (e.g. embedded arrays)
    for col in df.columns:
        sample = df[col].dropna().head(5)
        if len(sample) > 0 and isinstance(sample.iloc[0], (list, dict)):
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
            )

    # Create empty table structure (drop with CASCADE first to clear any FKs)
    _drop_cascade(engine, table_name)
    df.head(0).to_sql(table_name, engine, if_exists="replace", index=False)

    # Bulk insert via COPY in batches
    raw_conn = get_psycopg2_conn()
    cursor = raw_conn.cursor()
    total_rows = 0
    start_time = datetime.now()

    try:
        for batch_start in range(0, len(df), BATCH_SIZE):
            chunk = df.iloc[batch_start: batch_start + BATCH_SIZE]
            buf = io.StringIO()
            if batch_start == 0:
                chunk.to_csv(buf, index=False, na_rep="")
                buf.seek(0)
                cursor.copy_expert(
                    f"COPY {table_name} FROM STDIN WITH (FORMAT CSV, HEADER, NULL '')",
                    buf,
                )
            else:
                chunk.to_csv(buf, index=False, header=False, na_rep="")
                buf.seek(0)
                cursor.copy_expert(
                    f"COPY {table_name} FROM STDIN WITH (FORMAT CSV, NULL '')",
                    buf,
                )
            total_rows += len(chunk)
            if total_rows % 100_000 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = total_rows / elapsed if elapsed > 0 else 0
                logger.info(f"  … {total_rows:,} rows ({rate:,.0f} rows/s)")
                raw_conn.commit()

        raw_conn.commit()
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"  ✓ {table_name}: {total_rows:,} rows in {elapsed:.1f}s")
    except Exception as exc:
        raw_conn.rollback()
        logger.error(f"Failed {table_name}: {exc}")
        raise
    finally:
        cursor.close()
        raw_conn.close()


# ===========================================================================
# Mother App loader — flattened_mothers.csv → 3 normalized tables
# ===========================================================================

# Per-visit and per-child field suffixes (the part after `ancN_` / `childN_`)
_ANC_FIELDS = [
    "date", "gestational_age_wks", "place", "weight_kg", "haemoglobin",
    "bp_systolic", "bp_diastolic", "blood_sugar_fasting", "blood_sugar_pp",
    "high_risk", "risk_factors", "danger_signs", "services",
    "ifa_tablets", "urine_albumin", "urine_sugar",
]
_CHILD_FIELDS = [
    "gender", "weight_kg", "cried_at_birth", "dob",
    "defects", "breastfed", "immunization",
]


def _copy_chunk(cursor, table_name: str, df: pd.DataFrame, write_header: bool):
    """COPY a DataFrame into an existing table via psycopg2 copy_expert."""
    if df.empty:
        return
    buf = io.StringIO()
    if write_header:
        df.to_csv(buf, index=False, na_rep="")
        buf.seek(0)
        cursor.copy_expert(
            f"COPY {table_name} FROM STDIN WITH (FORMAT CSV, HEADER, NULL '')",
            buf,
        )
    else:
        df.to_csv(buf, index=False, header=False, na_rep="")
        buf.seek(0)
        cursor.copy_expert(
            f"COPY {table_name} FROM STDIN WITH (FORMAT CSV, NULL '')",
            buf,
        )


def _split_flattened_chunk(chunk: pd.DataFrame):
    """
    Split one chunk of the flattened_mothers CSV into 3 DataFrames:
    (mothers_wide, anc_long, children_long).
    """
    # mothers_wide: drop all anc{N}_* and child{N}_* columns
    wide_cols = [
        c for c in chunk.columns
        if not (c.startswith(("anc1_", "anc2_", "anc3_", "anc4_",
                              "child1_", "child2_", "child3_")))
    ]
    mothers_wide = chunk[wide_cols].copy()

    key_cols = ["mother_id", "pregnancy_number"]

    # ── ANC long form ──────────────────────────────────────────────────────
    anc_frames = []
    for n in range(1, 5):
        prefix = f"anc{n}_"
        src_cols = [
            prefix + f for f in _ANC_FIELDS if (prefix + f) in chunk.columns]
        if not src_cols:
            continue
        sub = chunk[key_cols + src_cols].copy()
        sub.columns = key_cols + [c[len(prefix):] for c in src_cols]
        sub.insert(2, "anc_visit_num", n)
        # Skip rows with no recorded date (visit didn't happen)
        sub = sub[sub["date"].notna()]
        anc_frames.append(sub)
    anc_long = (
        pd.concat(anc_frames, ignore_index=True)
        if anc_frames else pd.DataFrame(
            columns=key_cols + ["anc_visit_num"] + _ANC_FIELDS
        )
    )

    # ── Children long form ─────────────────────────────────────────────────
    child_frames = []
    for n in range(1, 4):
        prefix = f"child{n}_"
        src_cols = [
            prefix + f for f in _CHILD_FIELDS if (prefix + f) in chunk.columns]
        if not src_cols:
            continue
        sub = chunk[key_cols + src_cols].copy()
        sub.columns = key_cols + [c[len(prefix):] for c in src_cols]
        sub.insert(2, "child_num", n)
        # Skip rows where every child field is null
        data_cols = [
            c for c in sub.columns if c not in key_cols + ["child_num"]]
        sub = sub.dropna(how="all", subset=data_cols)
        child_frames.append(sub)
    children_long = (
        pd.concat(child_frames, ignore_index=True)
        if child_frames else pd.DataFrame(
            columns=key_cols + ["child_num"] + _CHILD_FIELDS
        )
    )

    return mothers_wide, anc_long, children_long


def load_flattened_mothers(
    engine,
    filepath: str | Path,
    skip_if_exists: bool = False,
):
    """
    Load the curated Mother App `flattened_mothers.csv` into 3 normalized tables:
      mothers                 — one row per pregnancy (wide)
      mother_anc_visits_flat  — long form of anc1..anc4 blocks
      mother_children         — long form of child1..child3 blocks

    Streams the CSV in chunks to keep memory bounded; uses psycopg2 COPY for
    high-throughput bulk insert (mirrors load_csv_large).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"File not found, skipping mother_app: {filepath}")
        return

    targets = ("mothers", "mother_anc_visits_flat", "mother_children")
    if skip_if_exists and all(table_exists(engine, t) for t in targets):
        if all(get_row_count(engine, t) > 0 for t in targets):
            logger.info("Skipping mother_app — all 3 tables already populated")
            return

    size_mb = filepath.stat().st_size / 1024 / 1024
    logger.info(
        f"Loading {filepath.name} → mothers / mother_anc_visits_flat / "
        f"mother_children ({size_mb:.0f} MB, COPY protocol)…"
    )

    # Drop downstream tables before parents (FK-safe even without CASCADE)
    for tbl in ("mother_children", "mother_anc_visits_flat", "mothers"):
        _drop_cascade(engine, tbl)

    reader = pd.read_csv(
        filepath,
        chunksize=CHUNKSIZE,
        low_memory=False,
        dtype=str,
        keep_default_na=True,
        na_values=["", "NA", "N/A", "null", "NULL", "None"],
    )

    tables_created = False
    totals = {"mothers": 0, "mother_anc_visits_flat": 0, "mother_children": 0}
    start_time = datetime.now()
    raw_conn = get_psycopg2_conn()
    cursor = raw_conn.cursor()

    try:
        for chunk_num, chunk in enumerate(reader):
            mothers_wide, anc_long, children_long = _split_flattened_chunk(
                chunk)

            if not tables_created:
                # Create empty DDL for all 3 tables from the first chunk's columns
                mothers_wide.head(0).to_sql(
                    "mothers", engine, if_exists="replace", index=False)
                anc_long.head(0).to_sql(
                    "mother_anc_visits_flat", engine,
                    if_exists="replace", index=False)
                children_long.head(0).to_sql(
                    "mother_children", engine,
                    if_exists="replace", index=False)
                tables_created = True

            write_header = (chunk_num == 0)
            _copy_chunk(cursor, "mothers", mothers_wide, write_header)
            _copy_chunk(cursor, "mother_anc_visits_flat",
                        anc_long, write_header)
            _copy_chunk(cursor, "mother_children", children_long, write_header)

            totals["mothers"] += len(mothers_wide)
            totals["mother_anc_visits_flat"] += len(anc_long)
            totals["mother_children"] += len(children_long)

            if totals["mothers"] % 50_000 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = totals["mothers"] / elapsed if elapsed > 0 else 0
                logger.info(
                    f"  … {totals['mothers']:,} mothers "
                    f"({rate:,.0f} rows/s) — "
                    f"anc={totals['mother_anc_visits_flat']:,} "
                    f"children={totals['mother_children']:,}"
                )
                raw_conn.commit()

        raw_conn.commit()
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(
            f"  ✓ mothers: {totals['mothers']:,} | "
            f"mother_anc_visits_flat: {totals['mother_anc_visits_flat']:,} | "
            f"mother_children: {totals['mother_children']:,} "
            f"in {elapsed:.1f}s"
        )
    except Exception as exc:
        raw_conn.rollback()
        logger.error(f"Failed mother_app load: {exc}")
        raise
    finally:
        cursor.close()
        raw_conn.close()


# ===========================================================================
# Schema objects — applied AFTER all data is loaded
# ===========================================================================

def apply_primary_keys(engine):
    """Add primary keys and unique constraints to all tables."""
    logger.info("Applying primary keys and unique constraints…")
    statements = [
        # ── Type normalizations (must run before UNIQUE/FK constraints) ─────────
        # states.lgd_statecode is loaded as TEXT ('17' string in GeoJSON); cast to
        # BIGINT so it matches districts.lgd_statecode (integer in GeoJSON → BIGINT).
        "ALTER TABLE states ALTER COLUMN lgd_statecode TYPE BIGINT USING lgd_statecode::BIGINT;",
        # pandas loads integer columns with any NaN as FLOAT64 → DOUBLE PRECISION.
        # Cast master table LGD code columns to BIGINT to match master_districts/
        # master_blocks primary keys (which are loaded as BIGINT from int-only CSVs).
        "ALTER TABLE master_villages ALTER COLUMN block_code_lgd TYPE BIGINT USING block_code_lgd::BIGINT;",
        "ALTER TABLE master_villages ALTER COLUMN district_code_lgd TYPE BIGINT USING district_code_lgd::BIGINT;",
        "ALTER TABLE master_health_facilities ALTER COLUMN block_code_lgd TYPE BIGINT USING block_code_lgd::BIGINT;",
        "ALTER TABLE master_health_facilities ALTER COLUMN district_code_lgd TYPE BIGINT USING district_code_lgd::BIGINT;",

        # ── Geo tables: 'id' column already loaded by load_geojson ─────────────
        # states — lgd_statecode is unique (1 state); FK target for districts
        "ALTER TABLE states ADD PRIMARY KEY (id);",
        "ALTER TABLE states ADD CONSTRAINT uq_states_lgd_statecode UNIQUE (lgd_statecode);",

        # districts — lgd_districtcode has 1 NULL (UNIQUE OK: PG treats NULLs as distinct)
        "ALTER TABLE districts ADD PRIMARY KEY (id);",
        "ALTER TABLE districts ADD CONSTRAINT uq_districts_lgd_districtcode UNIQUE (lgd_districtcode);",

        # district_boundaries — no LGD codes; PK only
        "ALTER TABLE district_boundaries ADD PRIMARY KEY (id);",

        # subdistricts — lgd_subdistrictcode fully unique (46 values, 0 NULLs)
        "ALTER TABLE subdistricts ADD PRIMARY KEY (id);",
        "ALTER TABLE subdistricts ADD CONSTRAINT uq_subdistricts_lgd_subdistrictcode UNIQUE (lgd_subdistrictcode);",

        # blocks — no LGD codes in source; PK only
        "ALTER TABLE blocks ADD PRIMARY KEY (id);",

        # villages_poly — objectid is NOT unique in source (duplicates found);
        # use a SERIAL surrogate PK.  'id' column remains as a plain column.
        "ALTER TABLE villages_poly ADD COLUMN gid SERIAL PRIMARY KEY;",

        # villages_point — same situation
        "ALTER TABLE villages_point ADD COLUMN gid SERIAL PRIMARY KEY;",

        # ── Master reference ───────────────────────────────────────────────────
        "ALTER TABLE master_districts ADD PRIMARY KEY (district_code_lgd);",
        "ALTER TABLE master_blocks ADD PRIMARY KEY (block_code_lgd);",
        "ALTER TABLE master_villages ADD PRIMARY KEY (village_id);",
        "ALTER TABLE master_health_facilities ADD PRIMARY KEY (facility_id);",
        # Core health data
        "ALTER TABLE anc_visits ADD PRIMARY KEY (anc_id);",
        # Raw records — surrogate serial key (natural keys may have duplicates/nulls)
        "ALTER TABLE raw_anc_records ADD COLUMN row_id SERIAL PRIMARY KEY;",
        "ALTER TABLE raw_child_records ADD COLUMN row_id SERIAL PRIMARY KEY;",
        # Mother App curated dataset
        # mothers: surrogate PK + UNIQUE on (mother_id, pregnancy_number).
        # The composite serves as the FK target for the long-form child tables.
        "ALTER TABLE mothers ADD COLUMN row_id SERIAL PRIMARY KEY;",
        "ALTER TABLE mothers ADD CONSTRAINT uq_mothers_mid_preg "
        "UNIQUE (mother_id, pregnancy_number);",
        "ALTER TABLE mother_anc_visits_flat ADD CONSTRAINT pk_manc "
        "PRIMARY KEY (mother_id, pregnancy_number, anc_visit_num);",
        "ALTER TABLE mother_children ADD CONSTRAINT pk_mchildren "
        "PRIMARY KEY (mother_id, pregnancy_number, child_num);",
        # Reference JSON
        "ALTER TABLE video_library ADD PRIMARY KEY (id);",
        "ALTER TABLE research_articles ADD PRIMARY KEY (id);",
        "ALTER TABLE nfhs_indicators ADD COLUMN id SERIAL PRIMARY KEY;",
    ]
    _exec_sql(engine, statements, "primary_keys")


def apply_foreign_keys(engine):
    """
    Add foreign key constraints per cross_dataset_relations.json.

    Each constraint runs in its own transaction so one failure does not block
    the rest.  anc_visits → mother_journeys is expressed as an index-only
    relationship (no FK constraint) because mother_id is not unique in
    mother_journeys (one row per pregnancy, not per mother).
    """
    logger.info("Applying foreign key constraints…")
    fk_statements = [
        # ── Master internal hierarchy FKs ──────────────────────────────────────
        # master_blocks → master_districts
        """
        ALTER TABLE master_blocks
          ADD CONSTRAINT fk_master_blocks_district
          FOREIGN KEY (district_code_lgd) REFERENCES master_districts(district_code_lgd);
        """,
        # master_villages → master_blocks (block_code_lgd)
        """
        ALTER TABLE master_villages
          ADD CONSTRAINT fk_master_villages_block
          FOREIGN KEY (block_code_lgd) REFERENCES master_blocks(block_code_lgd);
        """,
        # master_villages → master_districts (district_code_lgd)
        """
        ALTER TABLE master_villages
          ADD CONSTRAINT fk_master_villages_district
          FOREIGN KEY (district_code_lgd) REFERENCES master_districts(district_code_lgd);
        """,
        # master_health_facilities → master_blocks
        """
        ALTER TABLE master_health_facilities
          ADD CONSTRAINT fk_master_hf_block
          FOREIGN KEY (block_code_lgd) REFERENCES master_blocks(block_code_lgd);
        """,
        # master_health_facilities → master_districts
        """
        ALTER TABLE master_health_facilities
          ADD CONSTRAINT fk_master_hf_district
          FOREIGN KEY (district_code_lgd) REFERENCES master_districts(district_code_lgd);
        """,

        # ── Master → Geo FK constraints (LGD district code bridge) ─────────────
        # master_districts.district_code_lgd → districts.lgd_districtcode
        # (lgd_districtcode is UNIQUE in districts; 1 NULL district has NULL lgd_districtcode
        #  but PG FK allows NULLs on both sides so this is safe)
        """
        ALTER TABLE master_districts
          ADD CONSTRAINT fk_master_districts_geo
          FOREIGN KEY (district_code_lgd) REFERENCES districts(lgd_districtcode);
        """,
        # master_blocks.district_code_lgd → districts.lgd_districtcode
        """
        ALTER TABLE master_blocks
          ADD CONSTRAINT fk_master_blocks_geo_district
          FOREIGN KEY (district_code_lgd) REFERENCES districts(lgd_districtcode);
        """,
        # master_villages.district_code_lgd → districts.lgd_districtcode
        """
        ALTER TABLE master_villages
          ADD CONSTRAINT fk_master_villages_geo_district
          FOREIGN KEY (district_code_lgd) REFERENCES districts(lgd_districtcode);
        """,
        # master_health_facilities.district_code_lgd → districts.lgd_districtcode
        """
        ALTER TABLE master_health_facilities
          ADD CONSTRAINT fk_master_hf_geo_district
          FOREIGN KEY (district_code_lgd) REFERENCES districts(lgd_districtcode);
        """,
        # geo_full_mapping.district_code_lgd → districts.lgd_districtcode
        # (geo_full_mapping has NULL district_code_lgd for unmatched rows; FK allows child NULLs)
        """
        ALTER TABLE geo_full_mapping
          ADD CONSTRAINT fk_geo_full_map_district
          FOREIGN KEY (district_code_lgd) REFERENCES districts(lgd_districtcode);
        """,

        # ── Geographic hierarchy FKs ───────────────────────────────────────────
        # districts → states (via lgd_statecode; all 12 districts have a state code)
        """
        ALTER TABLE districts
          ADD CONSTRAINT fk_districts_state
          FOREIGN KEY (lgd_statecode) REFERENCES states(lgd_statecode);
        """,
        # subdistricts → districts (lgd_districtcode; 0 NULLs in subdistricts)
        """
        ALTER TABLE subdistricts
          ADD CONSTRAINT fk_subdistricts_district
          FOREIGN KEY (lgd_districtcode) REFERENCES districts(lgd_districtcode);
        """,
        # villages_poly → subdistricts (lgd_subdistrictcode; 0 NULLs in villages_poly)
        """
        ALTER TABLE villages_poly
          ADD CONSTRAINT fk_villages_poly_subdistrict
          FOREIGN KEY (lgd_subdistrictcode) REFERENCES subdistricts(lgd_subdistrictcode);
        """,
        # villages_poly → districts (lgd_districtcode; 0 NULLs in villages_poly)
        """
        ALTER TABLE villages_poly
          ADD CONSTRAINT fk_villages_poly_district
          FOREIGN KEY (lgd_districtcode) REFERENCES districts(lgd_districtcode);
        """,
        # villages_point → subdistricts (lgd_subdistrictcode; 0 NULLs)
        """
        ALTER TABLE villages_point
          ADD CONSTRAINT fk_villages_point_subdistrict
          FOREIGN KEY (lgd_subdistrictcode) REFERENCES subdistricts(lgd_subdistrictcode);
        """,
        # villages_point → districts (lgd_districtcode; 0 NULLs)
        """
        ALTER TABLE villages_point
          ADD CONSTRAINT fk_villages_point_district
          FOREIGN KEY (lgd_districtcode) REFERENCES districts(lgd_districtcode);
        """,
        # blocks & district_boundaries have no LGD codes — join by district name only
        # (no DB-level FK possible; use logical joins in application layer)

        # ── Mother App curated dataset FKs ─────────────────────────────────────
        # mother_anc_visits_flat → mothers (composite key)
        """
        ALTER TABLE mother_anc_visits_flat
          ADD CONSTRAINT fk_manc_mothers
          FOREIGN KEY (mother_id, pregnancy_number)
          REFERENCES mothers(mother_id, pregnancy_number)
          ON DELETE CASCADE;
        """,
        # mother_children → mothers (composite key)
        """
        ALTER TABLE mother_children
          ADD CONSTRAINT fk_mchildren_mothers
          FOREIGN KEY (mother_id, pregnancy_number)
          REFERENCES mothers(mother_id, pregnancy_number)
          ON DELETE CASCADE;
        """,
    ]
    for sql in fk_statements:
        try:
            with engine.begin() as conn:
                conn.execute(text(sql))
        except Exception as exc:
            logger.warning(f"[foreign_keys] skipped: {exc!s:.160}")


def apply_indexes(engine):
    """Create all indexes after data is loaded (much faster than incremental indexing)."""
    logger.info("Creating indexes…")
    statements = [
        # --- Spatial GIST indexes (PostGIS) ---
        "CREATE INDEX IF NOT EXISTS idx_states_geom          ON states          USING GIST (geometry);",
        "CREATE INDEX IF NOT EXISTS idx_districts_geom       ON districts       USING GIST (geometry);",
        "CREATE INDEX IF NOT EXISTS idx_dist_boundary_geom   ON district_boundaries USING GIST (geometry);",
        "CREATE INDEX IF NOT EXISTS idx_subdistricts_geom    ON subdistricts    USING GIST (geometry);",
        "CREATE INDEX IF NOT EXISTS idx_blocks_geom          ON blocks          USING GIST (geometry);",
        "CREATE INDEX IF NOT EXISTS idx_villages_poly_geom   ON villages_poly   USING GIST (geometry);",
        "CREATE INDEX IF NOT EXISTS idx_villages_point_geom  ON villages_point  USING GIST (geometry);",
        "CREATE INDEX IF NOT EXISTS idx_health_fac_geom      ON health_facilities USING GIST (geom);",
        "CREATE INDEX IF NOT EXISTS idx_anganwadi_geom       ON anganwadi_centres USING GIST (geom);",

        # --- Mother App curated dataset (mothers + long-form children) ---
        # mother_id alone is non-unique (multiple pregnancies per mother).
        "CREATE INDEX IF NOT EXISTS idx_mothers_mother_id        ON mothers (mother_id);",
        "CREATE INDEX IF NOT EXISTS idx_mothers_district_block   ON mothers (district, block);",
        "CREATE INDEX IF NOT EXISTS idx_mothers_reg_date         ON mothers (registration_date);",
        "CREATE INDEX IF NOT EXISTS idx_mothers_delivery_date    ON mothers (delivery_date);",
        "CREATE INDEX IF NOT EXISTS idx_mothers_lgd_district     ON mothers (lgd_district_code);",
        "CREATE INDEX IF NOT EXISTS idx_manc_mother_id           ON mother_anc_visits_flat (mother_id);",
        "CREATE INDEX IF NOT EXISTS idx_manc_visit_date          ON mother_anc_visits_flat (date);",
        "CREATE INDEX IF NOT EXISTS idx_mchildren_mother_id      ON mother_children (mother_id);",
        "CREATE INDEX IF NOT EXISTS idx_mchildren_dob            ON mother_children (dob);",

        # --- ANC visits ---
        "CREATE INDEX IF NOT EXISTS idx_anc_mother_id         ON anc_visits (mother_id);",
        "CREATE INDEX IF NOT EXISTS idx_anc_visit_date        ON anc_visits (visit_date);",
        "CREATE INDEX IF NOT EXISTS idx_anc_district_block    ON anc_visits (district, block);",

        # --- Village indicators ---
        "CREATE INDEX IF NOT EXISTS idx_village_ind_dist_block ON village_indicators_monthly (district_code_lgd, block_code_lgd);",
        "CREATE INDEX IF NOT EXISTS idx_village_ind_yearmonth  ON village_indicators_monthly (year_month);",

        # --- Geographic LGD code lookups (also serve as FK-backing indexes) ---
        "CREATE INDEX IF NOT EXISTS idx_districts_lgd_dist    ON districts    (lgd_districtcode);",
        "CREATE INDEX IF NOT EXISTS idx_districts_lgd_state   ON districts    (lgd_statecode);",
        "CREATE INDEX IF NOT EXISTS idx_subdistr_lgd_sub      ON subdistricts (lgd_subdistrictcode);",
        "CREATE INDEX IF NOT EXISTS idx_subdistr_lgd_dist     ON subdistricts (lgd_districtcode);",
        "CREATE INDEX IF NOT EXISTS idx_vill_poly_lgd_sub     ON villages_poly  (lgd_subdistrictcode);",
        "CREATE INDEX IF NOT EXISTS idx_vill_poly_lgd_dist    ON villages_poly  (lgd_districtcode);",
        "CREATE INDEX IF NOT EXISTS idx_vill_poly_lgd_vil     ON villages_poly  (lgd_villagecode);",
        "CREATE INDEX IF NOT EXISTS idx_vill_point_lgd_sub    ON villages_point (lgd_subdistrictcode);",
        "CREATE INDEX IF NOT EXISTS idx_vill_point_lgd_dist   ON villages_point (lgd_districtcode);",
        "CREATE INDEX IF NOT EXISTS idx_vill_point_lgd_vil    ON villages_point (lgd_villagecode);",
        "CREATE INDEX IF NOT EXISTS idx_blocks_district        ON blocks (district);",
        "CREATE INDEX IF NOT EXISTS idx_dist_boundary_district ON district_boundaries (district);",

        # --- Master reference code lookups ---
        "CREATE INDEX IF NOT EXISTS idx_master_blocks_dist_code ON master_blocks (district_code_lgd);",
        "CREATE INDEX IF NOT EXISTS idx_master_vil_block_code   ON master_villages (block_code_lgd);",
        "CREATE INDEX IF NOT EXISTS idx_master_vil_dist_code    ON master_villages (district_code_lgd);",
        "CREATE INDEX IF NOT EXISTS idx_master_vil_lgd_code     ON master_villages (village_code_lgd);",
        "CREATE INDEX IF NOT EXISTS idx_master_hf_block_code    ON master_health_facilities (block_code_lgd);",
        "CREATE INDEX IF NOT EXISTS idx_full_map_dist_block     ON geo_full_mapping (district_code_lgd, block_code_lgd);",

        # --- Raw records — extracted FK columns ---
        # mother_id is always present; pregnancy_id may be absent in ANC/child
        # records (raw JSON did not include it) so guard with a column check.
        "CREATE INDEX IF NOT EXISTS idx_raw_anc_mother_id   ON raw_anc_records (mother_id);",
        """
        DO $$ BEGIN
          IF EXISTS (SELECT 1 FROM information_schema.columns
                     WHERE table_schema='public' AND table_name='raw_anc_records'
                       AND column_name='pregnancy_id') THEN
            CREATE INDEX IF NOT EXISTS idx_raw_anc_pregnancy_id ON raw_anc_records (pregnancy_id);
          END IF;
        END $$;
        """,
        "CREATE INDEX IF NOT EXISTS idx_raw_child_mother_id  ON raw_child_records (mother_id);",
        """
        DO $$ BEGIN
          IF EXISTS (SELECT 1 FROM information_schema.columns
                     WHERE table_schema='public' AND table_name='raw_child_records'
                       AND column_name='pregnancy_id') THEN
            CREATE INDEX IF NOT EXISTS idx_raw_child_pregnancy_id ON raw_child_records (pregnancy_id);
          END IF;
        END $$;
        """,
        # --- NFHS lookup ---
        "CREATE INDEX IF NOT EXISTS idx_nfhs_district_round   ON nfhs_indicators (district, nfhs_round);",

        # --- Infrastructure code lookups ---
        "CREATE INDEX IF NOT EXISTS idx_health_fac_block      ON health_facilities (block_code);",
        "CREATE INDEX IF NOT EXISTS idx_health_fac_dist       ON health_facilities (district_code);",
        "CREATE INDEX IF NOT EXISTS idx_anganwadi_block        ON anganwadi_centres (block_code);",
        "CREATE INDEX IF NOT EXISTS idx_anganwadi_dist         ON anganwadi_centres (district_code);",
    ]
    _exec_sql(engine, statements, "indexes")


def apply_spatial_enrichment(engine):
    """
    Add geom GEOMETRY(POINT,4326) columns to master reference tables and populate
    them from the corresponding PostGIS geo tables.

    Enrichment sources:
      master_districts       ← ST_Centroid(districts.geometry)     via lgd_districtcode
      master_blocks          ← ST_Centroid(subdistricts.geometry)  via district code + block name
      master_villages        ← villages_point.geometry             via lgd_villagecode (exact match)
                               fallback: ST_Centroid(subdistricts) for unmatched villages
      master_health_facilities ← health_facilities.geom            via facility_id
    """
    logger.info("Applying spatial enrichment to master tables…")

    # Step A: Add geom columns (idempotent — IF NOT EXISTS)
    _exec_sql(engine, [
        "ALTER TABLE master_districts        ADD COLUMN IF NOT EXISTS geom geometry(Point,4326);",
        "ALTER TABLE master_blocks           ADD COLUMN IF NOT EXISTS geom geometry(Point,4326);",
        "ALTER TABLE master_villages         ADD COLUMN IF NOT EXISTS geom geometry(Point,4326);",
        "ALTER TABLE master_health_facilities ADD COLUMN IF NOT EXISTS geom geometry(Point,4326);",
    ], "spatial_enrich_add_cols")

    # Step B: Populate via UPDATE — each in its own transaction for error isolation
    updates = [
        ("master_districts ← district polygon centroid", """
            UPDATE master_districts md
            SET    geom = ST_Centroid(d.geometry)::geometry(Point,4326)
            FROM   districts d
            WHERE  md.district_code_lgd = d.lgd_districtcode;
        """),
        # LGD block code ≠ LGD subdistrict code; join on district LGD code + block name
        ("master_blocks ← subdistrict polygon centroid", """
            UPDATE master_blocks mb
            SET    geom = ST_Centroid(s.geometry)::geometry(Point,4326)
            FROM   subdistricts s
            WHERE  mb.district_code_lgd            = s.lgd_districtcode
              AND  UPPER(TRIM(mb.block_name))       = UPPER(TRIM(s.blockname));
        """),
        # Exact LGD village code match → village representative point
        ("master_villages (matched) ← villages_point geometry", """
            UPDATE master_villages mv
            SET    geom = vp.geometry::geometry(Point,4326)
            FROM   villages_point vp
            WHERE  mv.village_code_lgd = vp.lgd_villagecode
              AND  mv.village_code_lgd IS NOT NULL;
        """),
        # Unmatched villages (lgd_villagecode NULL) → centroid of their block polygon.
        # In PostgreSQL UPDATE...FROM, the target table alias cannot be referenced
        # inside JOIN conditions.  Use comma-separated FROM + WHERE instead.
        ("master_villages (unmatched) ← block centroid fallback", """
            UPDATE master_villages mv
            SET    geom = ST_Centroid(s.geometry)::geometry(Point,4326)
            FROM   master_blocks mb, subdistricts s
            WHERE  mv.geom IS NULL
              AND  mv.block_code_lgd                = mb.block_code_lgd
              AND  mb.district_code_lgd             = s.lgd_districtcode
              AND  UPPER(TRIM(mb.block_name))        = UPPER(TRIM(s.blockname));
        """),
        # GPS point from the health_facilities infrastructure table.
        # health_facilities.facility_id may be BIGINT while master has TEXT; cast to TEXT.
        ("master_health_facilities ← health_facilities geom", """
            UPDATE master_health_facilities mhf
            SET    geom = hf.geom::geometry(Point,4326)
            FROM   health_facilities hf
            WHERE  mhf.facility_id = hf.facility_id::TEXT
              AND  hf.geom IS NOT NULL;
        """),
    ]
    for label, sql in updates:
        try:
            with engine.begin() as conn:
                result = conn.execute(text(sql))
                logger.info(
                    f"  [spatial_enrich] {label}: {result.rowcount} rows updated")
        except Exception as exc:
            logger.warning(
                f"  [spatial_enrich] skipped '{label}': {exc!s:.160}")

    # Step C: GIST indexes on the new geom columns
    _exec_sql(engine, [
        "CREATE INDEX IF NOT EXISTS idx_master_districts_geom    ON master_districts         USING GIST (geom);",
        "CREATE INDEX IF NOT EXISTS idx_master_blocks_geom       ON master_blocks            USING GIST (geom);",
        "CREATE INDEX IF NOT EXISTS idx_master_villages_geom     ON master_villages          USING GIST (geom);",
        "CREATE INDEX IF NOT EXISTS idx_master_hf_geom           ON master_health_facilities USING GIST (geom);",
    ], "spatial_enrich_indexes")


def apply_comments(engine):
    """Apply table-level and key column-level comments for schema documentation."""
    logger.info("Applying table and column comments…")

    table_comments = {
        # Geographic
        "states":
            "Meghalaya state polygon with Census 2011 demographics — total population, literacy rates, worker categories, and SC/ST counts.",
        "districts":
            "12 Meghalaya district polygons with Census 2011 demographics including rural/urban population split, male/female worker counts, and SC/ST populations.",
        "district_boundaries":
            "12 Meghalaya district boundaries based on the 2021 administrative delineation, with calculated area and perimeter.",
        "subdistricts":
            "46 Meghalaya sub-district (C&RD block) polygons with Census 2011 demographics and agricultural worker breakdowns.",
        "blocks":
            "46 Meghalaya block boundaries with area and perimeter measurements. Source lacks LGD codes — linked to districts by name only.",
        "villages_poly":
            "Boundary polygons for ~6,862 Meghalaya villages with Census 2011 household counts, population, literacy, and worker data.",
        "villages_point":
            "Representative GPS points for ~6,862 Meghalaya villages with Census 2011 household counts, population, literacy, and worker data.",
        # Master reference
        "master_districts":
            "Canonical district reference linking district names to LGD (Local Government Directory) codes. 12 districts. Enriched with centroid geometry derived from district polygons.",
        "master_blocks":
            "Canonical block reference linking block names to LGD codes and parent districts. 47 blocks. Enriched with centroid geometry derived from sub-district polygons.",
        "master_villages":
            "Village reference with LGD code matching status. ~7,430 villages across 47 blocks. Exact-matched villages get GPS from villages_point; unmatched ones fall back to block centroid.",
        "master_health_facilities":
            "Health facility reference covering PHCs, Sub-centres, and Other Public Facilities across Meghalaya. ~1,799 facilities. Enriched with GPS from the health_facilities infrastructure table.",
        "geo_full_mapping":
            "Complete geographic hierarchy: district → block → facility → village, with LGD match confidence for each village. Used to navigate from any level of the administrative hierarchy to any other.",
        "geo_match_report":
            "Quality summary of village-to-LGD code matching: counts of exact matches vs. unmatched villages per block, used to assess geographic data completeness.",
        # Infrastructure
        "health_facilities":
            "Health facility directory with GPS coordinates, facility type (PHC/Sub-centre/District Hospital), operational status, and administrative block/district codes. ~646 facilities.",
        "anganwadi_centres":
            "ICDS Anganwadi Centre directory with GPS coordinates, infrastructure details (building type, water source, toilet availability), worker assignments, and administrative hierarchy. ~5,900 centres.",
        # Geography mapping
        "meghealth_geo_mapping":
            "Meghealth application's internal geographic hierarchy: district → block → PHC → sub-centre → village. Used to map health service delivery areas to administrative boundaries.",
        # Health data
        "anc_visits":
            "Individual Antenatal Care (ANC) visit records with clinical vitals (weight, blood pressure, haemoglobin), medications administered (IFA, TT, calcium), danger sign assessments, and high-risk flags. ~361,551 visits.",
        "village_indicators_monthly":
            "Monthly aggregated maternal and child health indicators at village level — registration counts, ANC coverage rates, institutional delivery percentages, immunization counts, and nutritional status metrics. ~133,071 records.",
        # Raw records
        "raw_anc_records":
            "Unprocessed Meghealth API responses for ANC visits with all original fields preserved as individual columns. Use anc_visits for the cleaned/structured version. ~361,508 records.",
        "raw_child_records":
            "Unprocessed Meghealth API responses for child/infant tracking with all original fields preserved. Includes birth details, immunization records, and growth monitoring data.",
        # Mother App curated dataset (replaces legacy mother_journeys + raw_pregnancy_records)
        "mothers":
            "Curated Mother App dataset — one row per pregnancy. Sourced from flattened_mothers.csv (the authoritative Mother App API export). Covers identity, demographics, registration, risk assessment, ANC aggregates, slope/trend features (hb/weight/BP), abortion, scheme enrolment, delivery outcome, and maternal death. ~426k records. Replaces the legacy mother_journeys table.",
        "mother_anc_visits_flat":
            "Long-form ANC visits unpivoted from the flattened Mother App CSV (anc1..anc4 blocks). One row per actually recorded ANC visit. Distinct from anc_visits, which is sourced from a different curated CSV and includes foetal HR / fundal height plus visits beyond #4.",
        "mother_children":
            "Long-form child outcome rows unpivoted from the flattened Mother App CSV (child1..child3 blocks). One row per delivered child with gender, weight, breastfeeding, and immunization-at-birth fields.",
        # Reference JSON
        "nfhs_indicators":
            "National Family Health Survey (NFHS) rounds 3, 4, and 5 district-level indicators for Meghalaya — covering nutrition, immunization, maternal health, family planning, and child mortality. 624 records across all districts and rounds.",
        "video_library":
            "Educational video catalogue for ASHA and ANM health workers. Topics include breastfeeding, complementary feeding, immunization schedules, maternal nutrition, and newborn care. ~49 videos with duration and category metadata.",
        "research_articles":
            "Curated peer-reviewed research articles on maternal and child health in Meghalaya and Northeast India. 23 articles with titles, authors, abstracts, journal info, and publication years.",
    }

    col_comments = {
        "mothers": {
            "mother_id":
                "Mother App mother identifier. Non-unique on its own — mothers with multiple pregnancies have multiple rows.",
            "pregnancy_number":
                "Sequence number of this pregnancy for the mother (1, 2, …). Together with mother_id forms the composite natural key.",
            "lgd_district_code":
                "LGD district code from the Mother App registration. Use master_districts for the canonical district reference.",
            "high_risk_at_reg":
                "Whether the pregnancy was flagged as high-risk at the time of initial registration in the Mother App.",
            "has_delivery":
                "Boolean flag — true if the pregnancy reached a recorded delivery event.",
        },
        "mother_anc_visits_flat": {
            "anc_visit_num":
                "ANC visit ordinal (1–4) — corresponds to the anc1..anc4 column blocks in the source flattened CSV.",
            "date":
                "Date the ANC visit was recorded. Rows with no date are excluded at load time.",
        },
        "mother_children": {
            "child_num":
                "Child ordinal within this pregnancy (1–3) — corresponds to the child1..child3 column blocks in the source CSV.",
        },
        "anc_visits": {
            "anc_id":    "Unique ANC visit identifier assigned by the Meghealth system.",
            "mother_id": "Mother identifier linking this visit to the corresponding records in mother_journeys.",
            "visit_date": "Date when the ANC visit was conducted.",
        },
        "states": {
            "id":            "GIS feature identifier (renamed from objectid in source GeoJSON).",
            "lgd_statecode": "LGD (Local Government Directory) state census code.",
        },
        "districts": {
            "id":               "GIS feature identifier (renamed from objectid in source GeoJSON).",
            "lgd_districtcode": "LGD district census code. One district has NULL (no LGD assignment).",
            "lgd_statecode":    "LGD code of the parent state.",
        },
        "district_boundaries": {
            "id": "GIS feature identifier (renamed from objectid_1 in source GeoJSON).",
        },
        "subdistricts": {
            "id":                  "GIS feature identifier (renamed from objectid in source GeoJSON).",
            "lgd_subdistrictcode": "LGD sub-district (C&RD block) census code.",
            "lgd_districtcode":    "LGD code of the parent district.",
        },
        "blocks": {
            "id":       "GIS feature identifier (renamed from objectid in source GeoJSON).",
            "district": "District name as free-text. No LGD code in source — link to districts by name.",
        },
        "villages_poly": {
            "gid":                 "Auto-generated row identifier (source objectid has duplicates).",
            "id":                  "Original objectid from GeoJSON (non-unique, kept for reference).",
            "lgd_villagecode":     "LGD village census code. NULL for ~95 villages that could not be matched to an LGD entry.",
            "lgd_subdistrictcode": "LGD code of the parent sub-district.",
            "lgd_districtcode":    "LGD code of the parent district.",
        },
        "villages_point": {
            "gid":                 "Auto-generated row identifier (source objectid has duplicates).",
            "id":                  "Original objectid from GeoJSON (non-unique, kept for reference).",
            "lgd_villagecode":     "LGD village census code. NULL for ~91 villages that could not be matched to an LGD entry.",
            "lgd_subdistrictcode": "LGD code of the parent sub-district.",
            "lgd_districtcode":    "LGD code of the parent district.",
        },
        "health_facilities": {
            "facility_id": "Unique facility identifier, matches the master_health_facilities reference table.",
        },
        "anganwadi_centres": {
            "anganwadi_centre_id": "Unique Anganwadi centre identifier.",
            "geometry_x":         "Longitude (duplicate of longitude column, kept for source compatibility).",
            "geometry_y":         "Latitude (duplicate of latitude column, kept for source compatibility).",
        },
        "raw_anc_records": {
            "mother_id": "Mother identifier extracted from the raw JSON for cross-referencing with mothers.",
        },
        "raw_child_records": {
            "mother_id": "Mother identifier extracted from the raw JSON for cross-referencing with mothers.",
        },
        "master_districts": {
            "district_code_lgd": "LGD district code — the canonical identifier for this district.",
            "geom":              "Centroid derived from the corresponding district boundary polygon.",
        },
        "master_blocks": {
            "block_code_lgd":    "LGD block code — the canonical identifier for this block.",
            "district_code_lgd": "LGD code of the parent district.",
            "geom":              "Centroid derived from the matching sub-district polygon (matched by block name and district code).",
        },
        "master_villages": {
            "village_id":       "Village identifier in format V-NNNNN.",
            "village_code_lgd": "LGD village census code. NULL if the village could not be matched to any LGD entry.",
            "match_confidence": "LGD matching quality: 'exact' for confirmed matches, 'unmatched' for villages without LGD correspondence.",
            "geom":             "GPS point from villages_point for exact matches; block centroid fallback for unmatched villages.",
        },
        "master_health_facilities": {
            "facility_id":       "Unique facility identifier.",
            "block_code_lgd":    "LGD code of the block where this facility is located.",
            "district_code_lgd": "LGD code of the district where this facility is located.",
            "geom":              "GPS point sourced from the health_facilities infrastructure table.",
        },
        "nfhs_indicators": {
            "nfhs_round":   "NFHS survey round number (3, 4, or 5).",
            "survey_year":  "Survey year range as reported (e.g., '2019-21').",
            "district":     "District or state name as it appears in the NFHS survey report.",
            "total":        "Combined rural and urban indicator value.",
        },
    }

    statements = []
    for tbl, comment in table_comments.items():
        safe = comment.replace("'", "''")
        statements.append(f"COMMENT ON TABLE {tbl} IS '{safe}';")
    for tbl, cols in col_comments.items():
        for col, comment in cols.items():
            safe = comment.replace("'", "''")
            statements.append(f"COMMENT ON COLUMN {tbl}.{col} IS '{safe}';")
    _exec_sql(engine, statements, "comments")


# ===========================================================================
# Migration orchestrator
# ===========================================================================

def run_migration(args):
    only = getattr(args, "only", None)
    skip_raw_json = getattr(args, "skip_raw_json", False)
    skip_if_exists = getattr(args, "skip_if_exists", False)
    do_drop = getattr(args, "drop", False)
    do_truncate = getattr(args, "truncate", False)

    def stage(name: str) -> bool:
        return only is None or only == name

    # Resolve which tables are in scope for drop/truncate
    target_tables = STAGE_TABLES.get(only, ALL_TABLES) if only else ALL_TABLES

    engine = get_engine()
    create_extensions(engine)

    # ------------------------------------------------------------------
    # Pre-load: drop or truncate (mutually exclusive; --drop takes priority)
    # ------------------------------------------------------------------
    if do_drop:
        logger.info("=== Pre-load: DROP tables ===")
        drop_tables(engine, target_tables)
    elif do_truncate:
        logger.info("=== Pre-load: TRUNCATE tables ===")
        truncate_tables(engine, target_tables)

    # ------------------------------------------------------------------
    # 1. Geographic tables (GeoJSON → PostGIS)
    # ------------------------------------------------------------------
    if stage("geo"):
        logger.info("=== Stage: Geographic tables ===")
        load_geojson(engine, GEO_DIR / "04_state.geojson", "states",
                     comment="Meghalaya state polygon — Census 2011")
        load_geojson(engine, GEO_DIR / "03_district.geojson", "districts",
                     comment="District polygons with Census demographics")
        load_geojson(engine, GEO_DIR / "04_district_boundary_2021.geojson", "district_boundaries",
                     comment="District boundaries (2021 delineation)")
        load_geojson(engine, GEO_DIR / "02_subdistrict.geojson", "subdistricts",
                     comment="Sub-district (block) polygons with Census data")
        load_geojson(engine, GEO_DIR / "06_block_boundary_46_nos.geojson", "blocks",
                     comment="Block boundary polygons — 46 blocks")
        load_geojson(engine, GEO_DIR / "01_village_poly.geojson", "villages_poly",
                     comment="Village boundary polygons — 6,862 villages")
        load_geojson(engine, GEO_DIR / "00_village_point.geojson", "villages_point",
                     comment="Village representative points — 6,862 villages")

    # ------------------------------------------------------------------
    # 2. Master reference tables (small CSVs)
    # ------------------------------------------------------------------
    if stage("master"):
        logger.info("=== Stage: Master reference tables ===")
        load_csv_small(engine, MASTER_DIR /
                       "master_districts.csv", "master_districts")
        load_csv_small(engine, MASTER_DIR /
                       "master_blocks.csv", "master_blocks")
        load_csv_small(engine, MASTER_DIR /
                       "master_villages.csv", "master_villages")
        load_csv_small(engine, MASTER_DIR /
                       "master_health_facilities.csv", "master_health_facilities")
        load_csv_small(engine, MASTER_DIR /
                       "full_mapping.csv", "geo_full_mapping")
        load_csv_small(engine, MASTER_DIR /
                       "match_report.csv", "geo_match_report")

    # ------------------------------------------------------------------
    # 3. Infrastructure tables (CSV with PostGIS points)
    # ------------------------------------------------------------------
    if stage("infra"):
        logger.info("=== Stage: Infrastructure tables ===")
        load_csv_with_points(
            engine,
            GEO_DIR / "dashboard_health_facilities_codes_enriched.csv",
            "health_facilities",
            lat_col="latitude", lon_col="longitude",
        )
        load_csv_with_points(
            engine,
            GEO_DIR / "anganwadi_centre_0_anganwadi_centre.csv",
            "anganwadi_centres",
            lat_col="latitude", lon_col="longitude",
        )

    # ------------------------------------------------------------------
    # 4. Geography mapping
    # ------------------------------------------------------------------
    if stage("infra") or stage("master"):
        load_csv_small(engine, OUTPUT_DIR /
                       "meghealth_geography_mapping.csv", "meghealth_geo_mapping")

    # ------------------------------------------------------------------
    # 5. Health data tables (large CSVs — COPY protocol)
    # ------------------------------------------------------------------
    if stage("health"):
        logger.info("=== Stage: Health data tables ===")
        load_csv_large(
            engine,
            OUTPUT_DIR / "anc_visits_detail.csv",
            "anc_visits",
            skip_if_exists=skip_if_exists,
        )
        load_csv_small(engine, OUTPUT_DIR /
                       "village_indicators_monthly.csv", "village_indicators_monthly")

    # ------------------------------------------------------------------
    # 5b. Mother App curated dataset (flattened_mothers.csv → 3 tables)
    # ------------------------------------------------------------------
    if stage("mother_app"):
        logger.info("=== Stage: Mother App curated dataset ===")
        load_flattened_mothers(
            engine,
            _HERE / "mecdm_dataset" / "flattened_mothers.csv",
            skip_if_exists=skip_if_exists,
        )

    # ------------------------------------------------------------------
    # 6. Raw JSON records (large — normalized relational tables)
    # ------------------------------------------------------------------
    if stage("raw_json") and not skip_raw_json:
        logger.info("=== Stage: Raw JSON records (normalized relational) ===")
        logger.warning(
            "Loading ~961 MB of JSON data. This requires ~3-4 GB RAM. "
            "Use --skip-raw-json to bypass."
        )
        load_json_normalized(
            engine,
            OUTPUT_DIR / "raw_anc_records.json",
            "raw_anc_records",
            skip_if_exists=skip_if_exists,
        )
        load_json_normalized(
            engine,
            OUTPUT_DIR / "raw_child_records.json",
            "raw_child_records",
            skip_if_exists=skip_if_exists,
        )
    elif skip_raw_json:
        logger.info("Skipping raw JSON records (--skip-raw-json)")

    # ------------------------------------------------------------------
    # 7. Reference JSON tables (small — flat relational)
    # ------------------------------------------------------------------
    if stage("ref_json"):
        logger.info("=== Stage: Reference JSON tables ===")
        load_json_flat(engine, OUTPUT_DIR /
                       "nfhs_indicators.json", "nfhs_indicators")
        load_json_flat(engine, OUTPUT_DIR /
                       "video_library.json", "video_library")
        load_json_flat(engine, OUTPUT_DIR /
                       "research_articles.json", "research_articles")

    # ------------------------------------------------------------------
    # 8. Schema objects — PKs, FKs, indexes, comments (ALWAYS last)
    # ------------------------------------------------------------------
    if stage("schema") or only is None:
        logger.info(
            "=== Stage: Schema objects (PKs, FKs, indexes, comments) ===")
        apply_primary_keys(engine)
        apply_foreign_keys(engine)
        apply_indexes(engine)
        apply_spatial_enrichment(engine)
        apply_comments(engine)

    logger.info("Migration complete!")


# ===========================================================================
# Entry point
# ===========================================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="MECDM SuperApp — AlloyDB migration script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--only",
        choices=["geo", "master", "infra", "health", "mother_app",
                 "raw_json", "ref_json", "schema"],
        default=None,
        metavar="STAGE",
        help="Run only a specific migration stage.",
    )
    parser.add_argument(
        "--skip-raw-json",
        action="store_true",
        help="Skip the three large raw JSON files (~961 MB total).",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="For large tables: skip loading if the table already has rows.",
    )

    drop_group = parser.add_mutually_exclusive_group()
    drop_group.add_argument(
        "--drop",
        action="store_true",
        help=(
            "DROP TABLE IF EXISTS CASCADE for all in-scope tables before loading. "
            "Removes data, schema, indexes, and constraints. "
            "Combine with --only to limit scope."
        ),
    )
    drop_group.add_argument(
        "--truncate",
        action="store_true",
        help=(
            "TRUNCATE all in-scope tables before loading (keeps schema, indexes, "
            "PKs and FKs intact). Resets sequences. "
            "Combine with --only to limit scope."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_migration(args)
