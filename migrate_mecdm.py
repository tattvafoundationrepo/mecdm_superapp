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
  Health data : village_indicators_monthly
  NHM mother  : raw.nhm_records_raw, nhm_mothers, nhm_pregnancies, nhm_anc_visits,
                nhm_children, nhm_home_visits  + view v_nhm_mothers_flat
                (sourced live from data.nhmmegh.in formdata API via
                 nhm_api_download.py — replaces legacy mother_app + raw_json)
  Reference   : nfhs_indicators, video_library, research_articles

Usage:
    python migrate_mecdm.py                        # full migration
    python migrate_mecdm.py --only geo             # geographic tables only
    python migrate_mecdm.py --only master          # master/reference tables only
    python migrate_mecdm.py --only infra           # infrastructure tables only
    python migrate_mecdm.py --only health          # village_indicators_monthly only
    python migrate_mecdm.py --only nhm_mother      # NHM formdata → 6 tables + flat view
    python migrate_mecdm.py --only ref_json        # reference JSON tables only
    python migrate_mecdm.py --only schema          # PKs/FKs/indexes/comments only
    python migrate_mecdm.py --drop                 # drop all tables then run full migration
    python migrate_mecdm.py --drop --only nhm_mother  # rebuild NHM tables from JSONL
    python migrate_mecdm.py --truncate             # truncate all tables (keep schema) then reload
    python migrate_mecdm.py --truncate --only geo  # truncate only geographic tables

Prerequisites for the nhm_mother stage:
    1. Set NHM_API_KEY_ID and NHM_API_KEY in .env
    2. Run `python nhm_api_download.py` to populate
       mecdm_dataset/nhm_formdata.jsonl (~440k records, ~2.5 GB)

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
        "village_indicators_monthly",
    ],
    "nhm_mother": [
        # Children listed before parents for safe sequential DROP.
        # raw.nhm_records_raw lives in the `raw` schema so PII / source-of-truth
        # JSON is not exposed via the default public search_path.
        "nhm_home_visits", "nhm_children", "nhm_anc_visits",
        "nhm_pregnancies", "nhm_mothers", "raw.nhm_records_raw",
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
    """Accept either bare names (assumed `public`) or `schema.table`."""
    if "." in table_name:
        schema, name = table_name.split(".", 1)
    else:
        schema, name = "public", table_name
    return name in inspect(engine).get_table_names(schema=schema)


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
    logger.info("Ensuring PostGIS extension and 'raw' schema exist…")
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS raw;"))


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
# Generic chunk COPY helper (shared by NHM loader)
# ===========================================================================

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


# ===========================================================================
# NHM formdata loader — nhm_formdata.jsonl → 6 tables (1 raw + 5 normalized)
# ===========================================================================
#
# Source : backend/mecdm_dataset/nhm_formdata.jsonl, produced by
#          nhm_api_download.py from data.nhmmegh.in formdata API.
#          One JSON document per line (~440k records).
# Tables :
#   raw.nhm_records_raw  — full JSONB document keyed by _id
#   nhm_mothers      — 1 row per record _id (woman demographics)
#   nhm_pregnancies  — 1 row per pregnancy_serial_number_grp[] entry
#   nhm_anc_visits   — 1 row per ANC_visits_grp_N (N=1..5) when present
#   nhm_children     — 1 row per delivery_grp.child_grp.child_rpt_grp[] entry
#   nhm_home_visits  — 1 row per home_visit_grp.hv_group entry
# View   : v_nhm_mothers_flat (created in apply_nhm_views)

NHM_TABLES = (
    "raw.nhm_records_raw",
    "nhm_mothers",
    "nhm_pregnancies",
    "nhm_anc_visits",
    "nhm_children",
    "nhm_home_visits",
)

# ── DDL ────────────────────────────────────────────────────────────────────
NHM_DDL = [
    """
    CREATE TABLE raw.nhm_records_raw (
        record_id    TEXT PRIMARY KEY,
        form_id      TEXT,
        created_at   TIMESTAMPTZ,
        modified_at  TIMESTAMPTZ,
        created_by   TEXT,
        modified_by  TEXT,
        fetched_at   TIMESTAMPTZ DEFAULT NOW(),
        payload      JSONB NOT NULL
    );
    """,
    """
    CREATE TABLE nhm_mothers (
        record_id            TEXT PRIMARY KEY,
        sangrah_id           TEXT,
        name_woman           TEXT,
        year_of_birth        TIMESTAMPTZ,
        age_mother           INTEGER,
        age_calc             INTEGER,
        education_woman      TEXT,
        name_husband         TEXT,
        education_husband    TEXT,
        district_res         TEXT,
        block_res            TEXT,
        phc_res              TEXT,
        sc_res               TEXT,
        village_res          TEXT,
        address_res          TEXT,
        shg_member           TEXT,
        epic_id_of_woman     TEXT,
        mcts_rch_id          TEXT,
        mobile_number        TEXT,
        mobile_belongs_to    TEXT,
        abha_id              TEXT,
        abha_address         TEXT,
        mhis_id              TEXT,
        username             TEXT,
        device_id            TEXT,
        device_model         TEXT,
        app_version          TEXT,
        gps_location         TEXT,
        geom                 geometry(Point,4326),
        instance_name        TEXT,
        created_at           TIMESTAMPTZ,
        modified_at          TIMESTAMPTZ
    );
    """,
    """
    CREATE TABLE nhm_pregnancies (
        record_id                  TEXT NOT NULL,
        pregnancy_num              INTEGER NOT NULL,
        age_of_mother              INTEGER,
        lmp                        TIMESTAMPTZ,
        lmp_disp                   TEXT,
        edd                        TIMESTAMPTZ,
        edd_disp                   TEXT,
        edd_num                    INTEGER,
        details_current_pregnancy  TEXT[],
        gravida                    INTEGER,
        parity                     INTEGER,
        abortion                   INTEGER,
        deaths                     INTEGER,
        living                     INTEGER,
        total_adl                  INTEGER,
        state                      TEXT,
        district                   TEXT,
        district_code_lgd          TEXT,
        block                      TEXT,
        phc                        TEXT,
        subcentre_reg              TEXT,
        anmhw                      TEXT,
        medicalofficer             TEXT,
        village                    TEXT,
        blood_group                TEXT,
        blood_group_negative       TEXT,
        order_of_pregnancy_reg     TEXT,
        registration_date          TIMESTAMPTZ,
        photograph_of_visit        TEXT,
        note_add                   TEXT,
        PRIMARY KEY (record_id, pregnancy_num)
    );
    """,
    """
    CREATE TABLE nhm_anc_visits (
        record_id            TEXT NOT NULL,
        pregnancy_num        INTEGER NOT NULL,
        anc_visit_num        SMALLINT NOT NULL,
        visit_date           TIMESTAMPTZ,
        weight_kg            DOUBLE PRECISION,
        haemoglobin          DOUBLE PRECISION,
        bp_systolic          DOUBLE PRECISION,
        bp_diastolic         DOUBLE PRECISION,
        blood_sugar_fasting  DOUBLE PRECISION,
        blood_sugar_pp       DOUBLE PRECISION,
        tt_given_earlier     TEXT,
        tt_services          TEXT,
        services             TEXT[],
        danger_signs         TEXT[],
        risk_factors         TEXT[],
        referred_or_treated  TEXT[],
        usg_indications      TEXT[],
        PRIMARY KEY (record_id, pregnancy_num, anc_visit_num)
    );
    """,
    """
    CREATE TABLE nhm_children (
        record_id        TEXT NOT NULL,
        pregnancy_num    INTEGER NOT NULL,
        child_num        SMALLINT NOT NULL,
        gender           TEXT,
        weight_kg        DOUBLE PRECISION,
        cried_at_birth   TEXT,
        dob              TIMESTAMPTZ,
        defects          TEXT,
        breastfed        TEXT,
        immunization     TEXT,
        PRIMARY KEY (record_id, pregnancy_num, child_num)
    );
    """,
    """
    CREATE TABLE nhm_home_visits (
        record_id        TEXT NOT NULL,
        pregnancy_num    INTEGER NOT NULL,
        seq              SMALLINT NOT NULL,
        record_date      TIMESTAMPTZ,
        done_by          TEXT,
        asha_active      TEXT,
        PRIMARY KEY (record_id, pregnancy_num, seq)
    );
    """,
]


def _to_int(v):
    try:
        if v is None or v == "":
            return None
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_float(v):
    try:
        if v is None or v == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_str(v):
    if v is None:
        return None
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _to_ts(v):
    """
    Pass through ISO timestamp strings; let postgres parse. Empty → None.
    Rejects non-date values like '0', 'N/A', or bare integers — the NHM API
    sometimes leaves these in date fields when the user skipped the question.
    """
    if v is None or v == "":
        return None
    s = str(v).strip()
    # Heuristic: a real ISO date contains a hyphen between digits (YYYY-MM-DD).
    # Anything else (e.g. "0", "Yes", "1") is treated as missing.
    if "-" not in s:
        return None
    return s


def _to_str_array(v):
    """Coerce a value into a list[str] suitable for psycopg2 TEXT[] adaptation."""
    if v is None or v == "":
        return None
    if isinstance(v, list):
        return [str(x) for x in v if x is not None and x != ""]
    return [str(v)]


def _flatten_nhm_record(rec: dict):
    """
    Convert one API record into rows for the 6 NHM tables.

    Returns: (raw_row, mother_row, [preg_rows], [anc_rows], [child_rows], [hv_rows])
    Each row is a tuple in column order matching the corresponding CREATE TABLE.
    """
    rid = rec.get("_id")
    src = rec.get("source") or {}
    abha = src.get("ABHA_MHIS_id") or {}

    # ── raw.nhm_records_raw ────────────────────────────────────────────────────
    raw_row = (
        rid,
        rec.get("formId"),
        _to_ts(rec.get("createdAt")),
        _to_ts(rec.get("modifiedAt")),
        rec.get("createdBy"),
        rec.get("modifiedBy"),
        # JSONB payload (psycopg2 → ::jsonb)
        json.dumps(rec, ensure_ascii=False),
    )

    # ── nhm_mothers ────────────────────────────────────────────────────────
    mother_row = (
        rid,
        _to_str(src.get("sangrah_id")),
        _to_str(src.get("name_woman")),
        _to_ts(src.get("year_of_birth")),
        _to_int(src.get("age_mother")),
        _to_int(src.get("age_calc")),
        _to_str(src.get("education_woman")),
        _to_str(src.get("name_husband")),
        _to_str(src.get("education_husband")),
        _to_str(src.get("district_res")),
        _to_str(src.get("block_res")),
        _to_str(src.get("phc_res")),
        _to_str(src.get("sc_res")),
        _to_str(src.get("village_res")),
        _to_str(src.get("address_res")),
        _to_str(src.get("shg_member")),
        _to_str(src.get("epic_id_of_woman")),
        _to_str(src.get("MCTS_RCH_ID") or src.get("MCTS_RCH_id")),
        _to_str(src.get("mobile_number_of_woman")),
        _to_str(src.get("mobile_number_belongs_to")),
        _to_str(abha.get("abha_id")),
        _to_str(abha.get("abha_address")),
        _to_str(abha.get("mhis_id")),
        _to_str(src.get("Username")),
        _to_str(src.get("Deviceid")),
        _to_str(src.get("Devicemodel")),
        _to_str(src.get("AppVersion")),
        _to_str(src.get("Location")),
        None,  # geom — populated post-load by apply_nhm_geom() from gps_location
        _to_str(src.get("InstanceName")),
        _to_ts(rec.get("createdAt")),
        _to_ts(rec.get("modifiedAt")),
    )

    preg_rows: list = []
    anc_rows: list = []
    child_rows: list = []
    hv_rows: list = []

    pregs = src.get("pregnancy_serial_number_grp") or []
    if not isinstance(pregs, list):
        pregs = []

    for idx, p in enumerate(pregs, start=1):
        if not isinstance(p, dict):
            continue
        # Pregnancy ordinal — prefer pregnancy_serial_number_calc, fall back to idx
        pn = _to_int(p.get("pregnancy_serial_number_calc")) or idx

        gpadl = ((p.get("past_obstetric_history") or {}).get("GPADL")) or {}
        reg = p.get("reg_location_grp") or {}

        preg_rows.append((
            rid, pn,
            _to_int(p.get("age_of_mother")),
            _to_ts(p.get("LMP")),
            _to_str(p.get("LMP_disp")),
            _to_ts(p.get("EDD")),
            _to_str(p.get("EDD_disp")),
            _to_int(p.get("EDD_num")),
            _to_str_array(p.get("details_current_pregnancy")),
            _to_int(gpadl.get("gravida")),
            _to_int(gpadl.get("parity_calculate")),
            _to_int(gpadl.get("abortion")),
            _to_int(gpadl.get("deaths")),
            _to_int(gpadl.get("living")),
            _to_int(gpadl.get("total_ADL")),
            _to_str(reg.get("state")),
            _to_str(reg.get("district")),
            _to_str(reg.get("district_code_lgd")),
            _to_str(reg.get("block")),
            _to_str(reg.get("phc")),
            _to_str(reg.get("subcentre_reg")),
            _to_str(reg.get("anmhw")),
            _to_str(reg.get("medicalofficer")),
            _to_str(reg.get("village")),
            _to_str(reg.get("blood_group")),
            _to_str(reg.get("blood_group_negative")),
            _to_str(reg.get("Order_of_Pregnancy_Reg")),
            _to_ts(reg.get("registration_date")),
            _to_str(reg.get("Photograph_of_the_Visit")),
            _to_str(p.get("note_add")),
        ))

        # ── ANC visits 1..5 ────────────────────────────────────────────────
        anc_grp = (p.get("ANC_visits_grp") or {})
        for n in range(1, 6):
            v = anc_grp.get(f"ANC_visits_grp_{n}")
            if not isinstance(v, dict) or not v:
                continue
            basic = v.get(f"ANC_basic_parameter_grp_{n}") or {}
            usg = (((v.get(f"usg_grp_anc_{n}") or {}).get(
                f"usg_details_grp_anc_{n}")) or {})
            anc_rows.append((
                rid, pn, n,
                _to_ts(v.get(f"date_of_ANC_{n}")
                       or v.get(f"anc{n}_timestamp")),
                _to_float(basic.get(f"weight_in_Kgs_{n}")),
                _to_float(basic.get(f"haemoglobin_in_grams_{n}")),
                _to_float(basic.get(f"upper_systolic_pressure_{n}")),
                _to_float(basic.get(f"lower_diastolic_pressure_{n}")),
                _to_float(basic.get(f"blood_sugar_fasting_{n}")),
                _to_float(basic.get(f"blood_sugar_pp_{n}")),
                _to_str(v.get(f"ANC_TT_given_earlier_{n}")),
                _to_str(v.get(f"ANC_TT_services_{n}")),
                _to_str_array(v.get(f"ANC_services_{n}")),
                _to_str_array(v.get(f"anc_danger_signs_observed_{n}")),
                _to_str_array(v.get(f"risk_factors_identified_{n}")),
                _to_str_array(v.get(f"referrred_or_treated_{n}")),
                _to_str_array(usg.get(f"usg_details_indications_anc_{n}")),
            ))

        # ── Children ───────────────────────────────────────────────────────
        delivery = p.get("delivery_grp") or {}
        child_grp = delivery.get("child_grp") or {}
        children_arr = child_grp.get("child_rpt_grp") or []
        if isinstance(children_arr, list):
            for cidx, c in enumerate(children_arr, start=1):
                if not isinstance(c, dict):
                    continue
                inner = c.get("child_each_grp") or {}
                cn = _to_int(c.get("child_no")) or cidx
                defects = c.get("child_defects")
                immun = c.get("immunization_doses_del")
                child_rows.append((
                    rid, pn, cn,
                    _to_str(inner.get("Baby_Gender")),
                    _to_float(inner.get("Weight_of_the_Baby")),
                    _to_str(inner.get("child_cried")),
                    _to_ts(inner.get("dob_child")),
                    ", ".join(defects) if isinstance(
                        defects, list) else _to_str(defects),
                    _to_str(c.get("child_breast_fed")),
                    ", ".join(immun) if isinstance(
                        immun, list) else _to_str(immun),
                ))

        # ── Home visits ────────────────────────────────────────────────────
        hv_grp = p.get("home_visit_grp") or {}
        hv_inner = hv_grp.get("hv_group")
        # hv_group can be a single dict or a list of dicts depending on count
        hv_list = hv_inner if isinstance(hv_inner, list) else (
            [hv_inner] if isinstance(hv_inner, dict) else [])
        for hvidx, hv in enumerate(hv_list, start=1):
            if not isinstance(hv, dict):
                continue
            hv_rows.append((
                rid, pn, hvidx,
                _to_ts(hv.get("hv_record_date") or hv.get(
                    "hv_record_date_default")),
                _to_str(hv.get("home_visits_done_by_the_ANM_ASHA")),
                _to_str(hv.get("ASHA_of_the_Village_active")),
            ))

    return raw_row, mother_row, preg_rows, anc_rows, child_rows, hv_rows


def _copy_rows(cursor, table: str, columns: tuple, rows: list, jsonb_idx=None):
    """
    Bulk-insert via psycopg2 execute_values for type-safe array/jsonb handling.
    `jsonb_idx` is the column index (within `columns`) that should be cast to JSONB.
    """
    if not rows:
        return
    cols_sql = ", ".join(columns)
    if jsonb_idx is not None:
        # Build a per-column template that casts the JSONB column
        parts = []
        for i in range(len(columns)):
            parts.append("%s::jsonb" if i == jsonb_idx else "%s")
        template = "(" + ",".join(parts) + ")"
    else:
        template = None
    psycopg2.extras.execute_values(
        cursor,
        f"INSERT INTO {table} ({cols_sql}) VALUES %s",
        rows,
        template=template,
        page_size=1000,
    )


# Column tuples in the same order as the DDL above
_NHM_RAW_COLS = (
    "record_id", "form_id", "created_at", "modified_at",
    "created_by", "modified_by", "payload",
)
_NHM_MOTHER_COLS = (
    "record_id", "sangrah_id", "name_woman", "year_of_birth", "age_mother",
    "age_calc", "education_woman", "name_husband", "education_husband",
    "district_res", "block_res", "phc_res", "sc_res", "village_res",
    "address_res", "shg_member", "epic_id_of_woman", "mcts_rch_id",
    "mobile_number", "mobile_belongs_to", "abha_id", "abha_address", "mhis_id",
    "username", "device_id", "device_model", "app_version", "gps_location",
    "geom", "instance_name", "created_at", "modified_at",
)
_NHM_PREG_COLS = (
    "record_id", "pregnancy_num", "age_of_mother", "lmp", "lmp_disp",
    "edd", "edd_disp", "edd_num", "details_current_pregnancy",
    "gravida", "parity", "abortion", "deaths", "living", "total_adl",
    "state", "district", "district_code_lgd", "block", "phc", "subcentre_reg",
    "anmhw", "medicalofficer", "village", "blood_group", "blood_group_negative",
    "order_of_pregnancy_reg", "registration_date", "photograph_of_visit", "note_add",
)
_NHM_ANC_COLS = (
    "record_id", "pregnancy_num", "anc_visit_num", "visit_date",
    "weight_kg", "haemoglobin", "bp_systolic", "bp_diastolic",
    "blood_sugar_fasting", "blood_sugar_pp",
    "tt_given_earlier", "tt_services", "services", "danger_signs",
    "risk_factors", "referred_or_treated", "usg_indications",
)
_NHM_CHILD_COLS = (
    "record_id", "pregnancy_num", "child_num", "gender", "weight_kg",
    "cried_at_birth", "dob", "defects", "breastfed", "immunization",
)
_NHM_HV_COLS = (
    "record_id", "pregnancy_num", "seq", "record_date", "done_by", "asha_active",
)


def load_nhm_formdata(engine, jsonl_path: str | Path, skip_if_exists: bool = False):
    """
    Stream nhm_formdata.jsonl → 6 tables (1 raw + 5 normalized).

    Drops & recreates the 6 tables, streams the JSONL in BATCH_SIZE chunks,
    flattens each record via _flatten_nhm_record, and bulk-inserts via
    psycopg2.extras.execute_values for native array/JSONB handling.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        logger.warning(
            "File not found, skipping nhm_mother: %s "
            "(run `python nhm_api_download.py` first)",
            jsonl_path,
        )
        return

    if skip_if_exists and all(table_exists(engine, t) for t in NHM_TABLES):
        if all(get_row_count(engine, t) > 0 for t in NHM_TABLES):
            logger.info("Skipping nhm_mother — all 6 tables already populated")
            return

    size_mb = jsonl_path.stat().st_size / 1024 / 1024
    logger.info(
        "Loading %s → 6 NHM tables (%.0f MB JSONL)…",
        jsonl_path.name, size_mb,
    )

    # Drop in FK-safe order (children before parents — though we have no FKs
    # at this point because they are added in apply_foreign_keys)
    for tbl in NHM_TABLES[::-1]:
        _drop_cascade(engine, tbl)
    for ddl in NHM_DDL:
        with engine.begin() as conn:
            conn.execute(text(ddl))

    totals = {t: 0 for t in NHM_TABLES}
    skipped_records = 0
    started = datetime.now()
    raw_conn = get_psycopg2_conn()
    cursor = raw_conn.cursor()

    raw_buf: list = []
    mother_buf: list = []
    preg_buf: list = []
    anc_buf: list = []
    child_buf: list = []
    hv_buf: list = []

    def _flush():
        if raw_buf:
            _copy_rows(cursor, "raw.nhm_records_raw",
                       _NHM_RAW_COLS, raw_buf, jsonb_idx=6)
            totals["raw.nhm_records_raw"] += len(raw_buf)
            raw_buf.clear()
        if mother_buf:
            _copy_rows(cursor, "nhm_mothers", _NHM_MOTHER_COLS, mother_buf)
            totals["nhm_mothers"] += len(mother_buf)
            mother_buf.clear()
        if preg_buf:
            _copy_rows(cursor, "nhm_pregnancies", _NHM_PREG_COLS, preg_buf)
            totals["nhm_pregnancies"] += len(preg_buf)
            preg_buf.clear()
        if anc_buf:
            _copy_rows(cursor, "nhm_anc_visits", _NHM_ANC_COLS, anc_buf)
            totals["nhm_anc_visits"] += len(anc_buf)
            anc_buf.clear()
        if child_buf:
            _copy_rows(cursor, "nhm_children", _NHM_CHILD_COLS, child_buf)
            totals["nhm_children"] += len(child_buf)
            child_buf.clear()
        if hv_buf:
            _copy_rows(cursor, "nhm_home_visits", _NHM_HV_COLS, hv_buf)
            totals["nhm_home_visits"] += len(hv_buf)
            hv_buf.clear()
        raw_conn.commit()

    try:
        seen_record_ids: set = set()
        seen_preg_keys: set = set()
        seen_anc_keys: set = set()
        seen_child_keys: set = set()
        seen_hv_keys: set = set()

        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "line %d: bad JSON, skipping: %s", line_num, exc)
                    skipped_records += 1
                    continue

                rid = rec.get("_id")
                if not rid:
                    skipped_records += 1
                    continue
                if rid in seen_record_ids:
                    skipped_records += 1
                    continue
                seen_record_ids.add(rid)

                raw, mother, pregs, ancs, children, hvs = _flatten_nhm_record(
                    rec)
                raw_buf.append(raw)
                mother_buf.append(mother)

                # Dedup composite-key children rows defensively (some pregnancies
                # repeat the same serial; keep the first occurrence per record).
                for row in pregs:
                    k = (row[0], row[1])
                    if k in seen_preg_keys:
                        continue
                    seen_preg_keys.add(k)
                    preg_buf.append(row)
                for row in ancs:
                    k = (row[0], row[1], row[2])
                    if k in seen_anc_keys:
                        continue
                    seen_anc_keys.add(k)
                    anc_buf.append(row)
                for row in children:
                    k = (row[0], row[1], row[2])
                    if k in seen_child_keys:
                        continue
                    seen_child_keys.add(k)
                    child_buf.append(row)
                for row in hvs:
                    k = (row[0], row[1], row[2])
                    if k in seen_hv_keys:
                        continue
                    seen_hv_keys.add(k)
                    hv_buf.append(row)

                if len(raw_buf) >= BATCH_SIZE:
                    _flush()
                    elapsed = (datetime.now() - started).total_seconds()
                    rate = totals["raw.nhm_records_raw"] / \
                        elapsed if elapsed > 0 else 0
                    logger.info(
                        "  … %d records (%.0f rec/s) — pregs=%d ancs=%d "
                        "children=%d home_visits=%d",
                        totals["raw.nhm_records_raw"], rate,
                        totals["nhm_pregnancies"], totals["nhm_anc_visits"],
                        totals["nhm_children"], totals["nhm_home_visits"],
                    )

        _flush()
        elapsed = (datetime.now() - started).total_seconds()
        logger.info(
            "  ✓ NHM loaded in %.1fs: raw=%d mothers=%d pregs=%d "
            "ancs=%d children=%d home_visits=%d (skipped=%d)",
            elapsed,
            totals["raw.nhm_records_raw"], totals["nhm_mothers"],
            totals["nhm_pregnancies"], totals["nhm_anc_visits"],
            totals["nhm_children"], totals["nhm_home_visits"],
            skipped_records,
        )
    except Exception as exc:
        raw_conn.rollback()
        logger.error("Failed nhm_mother load: %s", exc)
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
        # NHM mother tables: PKs are already declared inline in NHM_DDL
        # (record_id PK on raw/mothers; composite PKs on pregnancies/anc/children/home_visits).
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

        # ── NHM mother dataset FKs ─────────────────────────────────────────────
        # nhm_pregnancies → nhm_mothers
        """
        ALTER TABLE nhm_pregnancies
          ADD CONSTRAINT fk_nhm_preg_mothers
          FOREIGN KEY (record_id) REFERENCES nhm_mothers(record_id)
          ON DELETE CASCADE;
        """,
        # nhm_anc_visits → nhm_pregnancies (composite)
        """
        ALTER TABLE nhm_anc_visits
          ADD CONSTRAINT fk_nhm_anc_preg
          FOREIGN KEY (record_id, pregnancy_num)
          REFERENCES nhm_pregnancies(record_id, pregnancy_num)
          ON DELETE CASCADE;
        """,
        # nhm_children → nhm_pregnancies (composite)
        """
        ALTER TABLE nhm_children
          ADD CONSTRAINT fk_nhm_children_preg
          FOREIGN KEY (record_id, pregnancy_num)
          REFERENCES nhm_pregnancies(record_id, pregnancy_num)
          ON DELETE CASCADE;
        """,
        # nhm_home_visits → nhm_pregnancies (composite)
        """
        ALTER TABLE nhm_home_visits
          ADD CONSTRAINT fk_nhm_hv_preg
          FOREIGN KEY (record_id, pregnancy_num)
          REFERENCES nhm_pregnancies(record_id, pregnancy_num)
          ON DELETE CASCADE;
        """,
        # raw.nhm_records_raw → nhm_mothers (1:1 source-of-truth pairing)
        """
        ALTER TABLE raw.nhm_records_raw
          ADD CONSTRAINT fk_nhm_raw_mothers
          FOREIGN KEY (record_id) REFERENCES nhm_mothers(record_id)
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

        # --- NHM mother dataset (live API) ---
        "CREATE INDEX IF NOT EXISTS idx_nhm_mothers_district     ON nhm_mothers (district_res);",
        "CREATE INDEX IF NOT EXISTS idx_nhm_mothers_block        ON nhm_mothers (block_res);",
        "CREATE INDEX IF NOT EXISTS idx_nhm_mothers_sangrah      ON nhm_mothers (sangrah_id);",
        "CREATE INDEX IF NOT EXISTS idx_nhm_mothers_modified     ON nhm_mothers (modified_at);",
        "CREATE INDEX IF NOT EXISTS idx_nhm_mothers_geom         ON nhm_mothers USING GIST (geom);",
        "CREATE INDEX IF NOT EXISTS idx_nhm_preg_lmp             ON nhm_pregnancies (lmp);",
        "CREATE INDEX IF NOT EXISTS idx_nhm_preg_edd             ON nhm_pregnancies (edd);",
        "CREATE INDEX IF NOT EXISTS idx_nhm_preg_district        ON nhm_pregnancies (district);",
        "CREATE INDEX IF NOT EXISTS idx_nhm_preg_block           ON nhm_pregnancies (block);",
        "CREATE INDEX IF NOT EXISTS idx_nhm_preg_district_lgd    ON nhm_pregnancies (district_code_lgd);",
        "CREATE INDEX IF NOT EXISTS idx_nhm_preg_reg_date        ON nhm_pregnancies (registration_date);",
        "CREATE INDEX IF NOT EXISTS idx_nhm_anc_visit_date       ON nhm_anc_visits (visit_date);",
        "CREATE INDEX IF NOT EXISTS idx_nhm_anc_record           ON nhm_anc_visits (record_id);",
        "CREATE INDEX IF NOT EXISTS idx_nhm_children_dob         ON nhm_children (dob);",
        "CREATE INDEX IF NOT EXISTS idx_nhm_children_record      ON nhm_children (record_id);",
        "CREATE INDEX IF NOT EXISTS idx_nhm_hv_record_date       ON nhm_home_visits (record_date);",
        "CREATE INDEX IF NOT EXISTS idx_nhm_raw_payload          ON raw.nhm_records_raw USING GIN (payload);",
        "CREATE INDEX IF NOT EXISTS idx_nhm_raw_modified         ON raw.nhm_records_raw (modified_at);",

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


def apply_nhm_geom(engine):
    """
    Populate nhm_mothers.geom from the textual gps_location captured by the
    Mother App device. The format is "lat lon altitude accuracy" (space-
    separated). Rows whose first two tokens don't parse as plausible
    Meghalaya coordinates (lat 24–27, lon 89–93) are left NULL.
    """
    logger.info("Populating nhm_mothers.geom from gps_location…")
    sql = """
    UPDATE nhm_mothers
    SET    geom = ST_SetSRID(
                    ST_MakePoint(
                        NULLIF(split_part(gps_location, ' ', 2), '')::double precision,
                        NULLIF(split_part(gps_location, ' ', 1), '')::double precision
                    ), 4326)
    WHERE  gps_location IS NOT NULL
      AND  gps_location <> ''
      AND  split_part(gps_location, ' ', 1) ~ '^-?[0-9]+(\\.[0-9]+)?$'
      AND  split_part(gps_location, ' ', 2) ~ '^-?[0-9]+(\\.[0-9]+)?$'
      AND  split_part(gps_location, ' ', 1)::double precision BETWEEN 24 AND 27
      AND  split_part(gps_location, ' ', 2)::double precision BETWEEN 89 AND 93;
    """
    try:
        with engine.begin() as conn:
            result = conn.execute(text(sql))
            logger.info("  ✓ nhm_mothers.geom: %d rows populated",
                        result.rowcount)
    except Exception as exc:
        logger.warning("[nhm_geom] failed: %s", str(exc)[:240])


def apply_nhm_views(engine):
    """
    Create v_nhm_mothers_flat — a wide read view per (record_id, pregnancy_num)
    that joins nhm_mothers + nhm_pregnancies + ANC visits 1..5 + first 3 children
    + a home-visit summary. Replaces the legacy `mothers` wide table for
    downstream analytics that prefer a flat shape.
    """
    logger.info("Creating NHM flat view…")
    sql = """
    DROP VIEW IF EXISTS v_nhm_mothers_flat CASCADE;
    CREATE VIEW v_nhm_mothers_flat AS
    SELECT
        m.record_id,
        m.sangrah_id,
        m.name_woman,
        m.year_of_birth,
        m.age_mother,
        m.age_calc,
        m.education_woman,
        m.name_husband,
        m.education_husband,
        m.district_res,
        m.block_res,
        m.phc_res,
        m.sc_res,
        m.village_res,
        m.address_res,
        m.shg_member,
        m.mcts_rch_id,
        m.mobile_number,
        m.abha_id,
        m.mhis_id,
        m.gps_location,
        m.created_at,
        m.modified_at,

        p.pregnancy_num,
        p.lmp,
        p.edd,
        p.edd_disp,
        p.gravida,
        p.parity,
        p.abortion,
        p.living,
        p.district           AS preg_district,
        p.district_code_lgd  AS preg_district_code_lgd,
        p.block              AS preg_block,
        p.phc                AS preg_phc,
        p.subcentre_reg      AS preg_subcentre,
        p.anmhw              AS preg_anmhw,
        p.village            AS preg_village,
        p.blood_group,
        p.registration_date,

        a1.visit_date AS anc1_date, a1.weight_kg AS anc1_weight_kg,
        a1.haemoglobin AS anc1_hb, a1.bp_systolic AS anc1_bp_sys,
        a1.bp_diastolic AS anc1_bp_dia, a1.services AS anc1_services,
        a1.danger_signs AS anc1_danger_signs, a1.risk_factors AS anc1_risk_factors,

        a2.visit_date AS anc2_date, a2.weight_kg AS anc2_weight_kg,
        a2.haemoglobin AS anc2_hb, a2.bp_systolic AS anc2_bp_sys,
        a2.bp_diastolic AS anc2_bp_dia, a2.services AS anc2_services,
        a2.danger_signs AS anc2_danger_signs, a2.risk_factors AS anc2_risk_factors,

        a3.visit_date AS anc3_date, a3.weight_kg AS anc3_weight_kg,
        a3.haemoglobin AS anc3_hb, a3.bp_systolic AS anc3_bp_sys,
        a3.bp_diastolic AS anc3_bp_dia, a3.services AS anc3_services,
        a3.danger_signs AS anc3_danger_signs, a3.risk_factors AS anc3_risk_factors,

        a4.visit_date AS anc4_date, a4.weight_kg AS anc4_weight_kg,
        a4.haemoglobin AS anc4_hb, a4.bp_systolic AS anc4_bp_sys,
        a4.bp_diastolic AS anc4_bp_dia, a4.services AS anc4_services,
        a4.danger_signs AS anc4_danger_signs, a4.risk_factors AS anc4_risk_factors,

        a5.visit_date AS anc5_date, a5.weight_kg AS anc5_weight_kg,
        a5.haemoglobin AS anc5_hb, a5.bp_systolic AS anc5_bp_sys,
        a5.bp_diastolic AS anc5_bp_dia, a5.services AS anc5_services,
        a5.danger_signs AS anc5_danger_signs, a5.risk_factors AS anc5_risk_factors,

        c1.gender AS child1_gender, c1.weight_kg AS child1_weight_kg,
        c1.dob AS child1_dob, c1.breastfed AS child1_breastfed,
        c1.immunization AS child1_immunization,

        c2.gender AS child2_gender, c2.weight_kg AS child2_weight_kg,
        c2.dob AS child2_dob, c2.breastfed AS child2_breastfed,
        c2.immunization AS child2_immunization,

        c3.gender AS child3_gender, c3.weight_kg AS child3_weight_kg,
        c3.dob AS child3_dob, c3.breastfed AS child3_breastfed,
        c3.immunization AS child3_immunization,

        hv.last_home_visit_date,
        hv.home_visit_count
    FROM nhm_mothers m
    JOIN nhm_pregnancies p USING (record_id)
    LEFT JOIN nhm_anc_visits a1
        ON a1.record_id = p.record_id AND a1.pregnancy_num = p.pregnancy_num AND a1.anc_visit_num = 1
    LEFT JOIN nhm_anc_visits a2
        ON a2.record_id = p.record_id AND a2.pregnancy_num = p.pregnancy_num AND a2.anc_visit_num = 2
    LEFT JOIN nhm_anc_visits a3
        ON a3.record_id = p.record_id AND a3.pregnancy_num = p.pregnancy_num AND a3.anc_visit_num = 3
    LEFT JOIN nhm_anc_visits a4
        ON a4.record_id = p.record_id AND a4.pregnancy_num = p.pregnancy_num AND a4.anc_visit_num = 4
    LEFT JOIN nhm_anc_visits a5
        ON a5.record_id = p.record_id AND a5.pregnancy_num = p.pregnancy_num AND a5.anc_visit_num = 5
    LEFT JOIN nhm_children c1
        ON c1.record_id = p.record_id AND c1.pregnancy_num = p.pregnancy_num AND c1.child_num = 1
    LEFT JOIN nhm_children c2
        ON c2.record_id = p.record_id AND c2.pregnancy_num = p.pregnancy_num AND c2.child_num = 2
    LEFT JOIN nhm_children c3
        ON c3.record_id = p.record_id AND c3.pregnancy_num = p.pregnancy_num AND c3.child_num = 3
    LEFT JOIN (
        SELECT record_id, pregnancy_num,
               MAX(record_date) AS last_home_visit_date,
               COUNT(*)         AS home_visit_count
        FROM   nhm_home_visits
        GROUP  BY record_id, pregnancy_num
    ) hv ON hv.record_id = p.record_id AND hv.pregnancy_num = p.pregnancy_num;

    COMMENT ON VIEW v_nhm_mothers_flat IS
      'Denormalized read view of the NHM mother dataset — one row per (record_id, pregnancy_num) with all 5 ANC visits, first 3 children, and home-visit summary joined. Use this for ad-hoc reporting; use the underlying nhm_* tables for any write or fine-grained query.';
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(sql))
        logger.info("  ✓ v_nhm_mothers_flat created")
    except Exception as exc:
        logger.warning("[nhm_views] failed: %s", str(exc)[:240])


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
        "village_indicators_monthly":
            "Monthly aggregated maternal and child health indicators at village level — registration counts, ANC coverage rates, institutional delivery percentages, immunization counts, and nutritional status metrics. ~133,071 records.",
        # NHM mother dataset (live API)
        "raw.nhm_records_raw":
            "Source-of-truth landing table for the NHM Megh formdata API mother form (5fa5510a4794b76e71267ebb). One row per record _id with the full JSON document in payload (JSONB). All other nhm_* tables are derived from this; downstream agents should query the normalized tables or v_nhm_mothers_flat.",
        "nhm_mothers":
            "Per-mother demographics flattened from the NHM formdata API (one row per record_id). Includes name, age, education, residence, ABHA/MHIS ids, mobile, and capture device metadata. PII-restricted in privacy mode.",
        "nhm_pregnancies":
            "One row per pregnancy in pregnancy_serial_number_grp[]. Captures LMP/EDD, GPADL obstetric history, and registration location (block, PHC, ANM, blood group, registration_date). FK→nhm_mothers via record_id.",
        "nhm_anc_visits":
            "One row per ANC visit (1–5) in ANC_visits_grp_N. Captures clinical vitals (weight, Hb, BP, blood sugar) plus services, danger signs, risk factors, referrals and USG indications as TEXT[]. FK→nhm_pregnancies.",
        "nhm_children":
            "One row per delivered child in delivery_grp.child_grp.child_rpt_grp[]. Captures gender, weight, DOB, breastfeeding, immunization, and birth defects. FK→nhm_pregnancies.",
        "nhm_home_visits":
            "One row per ANM/ASHA home visit in home_visit_grp.hv_group. Captures the visit date, the worker who performed it, and active-ASHA status. FK→nhm_pregnancies.",
        # Reference JSON
        "nfhs_indicators":
            "National Family Health Survey (NFHS) rounds 3, 4, and 5 district-level indicators for Meghalaya — covering nutrition, immunization, maternal health, family planning, and child mortality. 624 records across all districts and rounds.",
        "video_library":
            "Educational video catalogue for ASHA and ANM health workers. Topics include breastfeeding, complementary feeding, immunization schedules, maternal nutrition, and newborn care. ~49 videos with duration and category metadata.",
        "research_articles":
            "Curated peer-reviewed research articles on maternal and child health in Meghalaya and Northeast India. 23 articles with titles, authors, abstracts, journal info, and publication years.",
    }

    col_comments = {
        "nhm_mothers": {
            "record_id":   "API _id from data.nhmmegh.in formdata. Stable across refreshes; primary key for the entire NHM mother schema.",
            "sangrah_id":  "Mother App registration identifier (e.g. MOTHER-234544). Privacy-restricted.",
            "mcts_rch_id": "MCTS/RCH identifier issued by the Government of India. Privacy-restricted.",
        },
        "nhm_pregnancies": {
            "pregnancy_num":     "Per-mother pregnancy ordinal (1, 2, …) sourced from pregnancy_serial_number_calc.",
            "district_code_lgd": "LGD district code captured at registration. Use this to JOIN to master_districts / districts rather than the free-text district name.",
            "gravida":           "Total number of pregnancies including the current one (GPADL.gravida).",
            "parity":            "Number of previous pregnancies carried to viable gestational age (GPADL.parity_calculate).",
        },
        "nhm_anc_visits": {
            "anc_visit_num": "ANC visit ordinal (1–5) — corresponds to ANC_visits_grp_N in the source document.",
            "services":      "Array of services delivered at this visit (e.g. 'IFA', 'Calcium', 'TT').",
            "danger_signs":  "Array of danger signs observed during the visit.",
        },
        "nhm_children": {
            "child_num": "Child ordinal within this pregnancy (1, 2, …) — order in delivery_grp.child_grp.child_rpt_grp[].",
        },
        "raw.nhm_records_raw": {
            "payload":    "Full original NHM API record as JSONB. GIN-indexed for arbitrary path/key queries — use this only when the normalized columns lack the field you need.",
            "fetched_at": "Wall-clock time when the record was loaded into AlloyDB by migrate_mecdm.py.",
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
    # 5. Health data tables (small CSVs)
    # ------------------------------------------------------------------
    if stage("health"):
        logger.info("=== Stage: Health data tables ===")
        load_csv_small(engine, OUTPUT_DIR /
                       "village_indicators_monthly.csv", "village_indicators_monthly")

    # ------------------------------------------------------------------
    # 5b. NHM mother dataset (live API JSONL → 6 tables)
    # ------------------------------------------------------------------
    if stage("nhm_mother"):
        logger.info("=== Stage: NHM Mother formdata (live API) ===")
        load_nhm_formdata(
            engine,
            _HERE / "mecdm_dataset" / "nhm_formdata.jsonl",
            skip_if_exists=skip_if_exists,
        )

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
        apply_nhm_geom(engine)
        apply_nhm_views(engine)
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
        choices=["geo", "master", "infra", "health", "nhm_mother",
                 "ref_json", "schema"],
        default=None,
        metavar="STAGE",
        help="Run only a specific migration stage.",
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
