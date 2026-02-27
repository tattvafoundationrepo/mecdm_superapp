#!/usr/bin/env python3
"""
MECDM SuperApp — AlloyDB/PostgreSQL Migration Script
Meghalaya Early Childhood Development Mission

Migrates all geographic, master-reference, health, and raw-data tables
from the local mecdm_dataset folder into AlloyDB (PostGIS-enabled Postgres).

Tables created (25 total):
  Geographic  : states, districts, district_boundaries, subdistricts, blocks,
                villages_poly, villages_point
  Master ref  : master_districts, master_blocks, master_villages,
                master_health_facilities, geo_full_mapping, geo_match_report
  Infrastructure: health_facilities, anganwadi_centres
  Geography   : meghealth_geo_mapping
  Health data : mother_journeys, anc_visits, village_indicators_monthly
  Raw records : raw_anc_records, raw_child_records, raw_pregnancy_records
  Reference   : nfhs_indicators, video_library, research_articles

Usage:
    python migrate_mecdm.py                        # full migration
    python migrate_mecdm.py --skip-raw-json        # skip large raw JSON files (~961 MB)
    python migrate_mecdm.py --only geo             # geographic tables only
    python migrate_mecdm.py --only master          # master/reference tables only
    python migrate_mecdm.py --only infra           # infrastructure tables only
    python migrate_mecdm.py --only health          # health CSV tables only
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
        "anc_visits", "mother_journeys",
        "village_indicators_monthly",
    ],
    "raw_json": [
        "raw_anc_records", "raw_child_records", "raw_pregnancy_records",
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
        geom_type = gdf.geometry.geom_type.mode()[0].upper() if not gdf.empty else "GEOMETRY"
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
    logger.info(f"Loading {filepath.name} → {table_name} (with POINT geometry)…")
    try:
        df = pd.read_csv(filepath, low_memory=False)
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")

        valid = df[df[lat_col].notna() & df[lon_col].notna()].copy()
        invalid = df[df[lat_col].isna() | df[lon_col].isna()].copy()

        if valid.empty:
            logger.warning(f"  No valid coordinates in {table_name}, loading without geometry")
            df.to_sql(table_name, engine, if_exists="replace", index=False)
            return

        gdf = gpd.GeoDataFrame(
            valid,
            geometry=gpd.points_from_xy(valid[lon_col], valid[lat_col]),
            crs="EPSG:4326",
        )
        gdf.rename_geometry("geom", inplace=True)
        gdf.to_postgis(
            table_name, engine,
            if_exists="replace", index=False,
            dtype={"geom": Geometry("POINT", srid=4326)},
        )
        if not invalid.empty:
            logger.warning(f"  {len(invalid)} rows missing coordinates — appended without geometry")
            invalid.to_sql(table_name, engine, if_exists="append", index=False)

        logger.info(f"  ✓ {table_name}: {len(df):,} rows ({len(valid):,} with geometry)")
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
    logger.info(f"Loading {filepath.name} → {table_name} ({size_mb:.0f} MB, COPY protocol)…")

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
                chunk.head(0).to_sql(table_name, engine, if_exists="replace", index=False)
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
                    lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
                )

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
    logger.info(f"Loading {filepath.name} → {table_name} ({size_mb:.0f} MB, normalized JSON)…")

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

    # Create empty table structure
    df.head(0).to_sql(table_name, engine, if_exists="replace", index=False)

    # Bulk insert via COPY in batches
    raw_conn = get_psycopg2_conn()
    cursor = raw_conn.cursor()
    total_rows = 0
    start_time = datetime.now()

    try:
        for batch_start in range(0, len(df), BATCH_SIZE):
            chunk = df.iloc[batch_start : batch_start + BATCH_SIZE]
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
# Schema objects — applied AFTER all data is loaded
# ===========================================================================

def apply_primary_keys(engine):
    """Add primary keys and unique constraints to all tables."""
    logger.info("Applying primary keys and unique constraints…")
    statements = [
        # Master reference
        "ALTER TABLE master_districts ADD PRIMARY KEY (district_code_lgd);",
        "ALTER TABLE master_blocks ADD PRIMARY KEY (block_code_lgd);",
        "ALTER TABLE master_villages ADD PRIMARY KEY (village_id);",
        "ALTER TABLE master_health_facilities ADD PRIMARY KEY (facility_id);",
        # Core health data
        # pregnancy_id has NULL values for mothers registered without a tracked
        # pregnancy (child-only records), so it cannot be a NOT NULL primary key.
        # Use a SERIAL surrogate PK + UNIQUE constraint on pregnancy_id.
        # PostgreSQL UNIQUE allows multiple NULLs, so this succeeds even with
        # nulls AND still satisfies FK references from raw_pregnancy_records.
        "ALTER TABLE mother_journeys ADD COLUMN row_id SERIAL PRIMARY KEY;",
        "ALTER TABLE mother_journeys ADD CONSTRAINT uq_pregnancy_id UNIQUE (pregnancy_id);",
        "ALTER TABLE anc_visits ADD PRIMARY KEY (anc_id);",
        # Raw records — surrogate serial key (natural keys may have duplicates/nulls)
        "ALTER TABLE raw_anc_records ADD COLUMN row_id SERIAL PRIMARY KEY;",
        "ALTER TABLE raw_child_records ADD COLUMN row_id SERIAL PRIMARY KEY;",
        "ALTER TABLE raw_pregnancy_records ADD COLUMN row_id SERIAL PRIMARY KEY;",
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
        # raw_pregnancy_records.pregnancy_id → mother_journeys.pregnancy_id
        # (pregnancy_id is the PK of mother_journeys)
        """
        ALTER TABLE raw_pregnancy_records
          ADD CONSTRAINT fk_raw_preg_pregnancy_id
          FOREIGN KEY (pregnancy_id)
          REFERENCES mother_journeys(pregnancy_id)
          ON DELETE CASCADE;
        """,
        # raw_anc_records.pregnancy_id → mother_journeys.pregnancy_id
        # (only if the column exists — raw ANC JSON may not include pregnancy_id)
        """
        DO $$
        BEGIN
          IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name   = 'raw_anc_records'
              AND column_name  = 'pregnancy_id'
          ) THEN
            ALTER TABLE raw_anc_records
              ADD CONSTRAINT fk_raw_anc_pregnancy_id
              FOREIGN KEY (pregnancy_id)
              REFERENCES mother_journeys(pregnancy_id)
              ON DELETE CASCADE;
          END IF;
        END $$;
        """,
        # raw_child_records.pregnancy_id → mother_journeys.pregnancy_id
        """
        DO $$
        BEGIN
          IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name   = 'raw_child_records'
              AND column_name  = 'pregnancy_id'
          ) THEN
            ALTER TABLE raw_child_records
              ADD CONSTRAINT fk_raw_child_pregnancy_id
              FOREIGN KEY (pregnancy_id)
              REFERENCES mother_journeys(pregnancy_id)
              ON DELETE CASCADE;
          END IF;
        END $$;
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

        # --- Mother / pregnancy lookups ---
        # pregnancy_id is the PK so its index is implicit; keep mother_id indexed
        # for lookups that join on the mother (multiple pregnancies per mother).
        "CREATE INDEX IF NOT EXISTS idx_mother_id            ON mother_journeys (mother_id);",
        "CREATE INDEX IF NOT EXISTS idx_mother_district_block ON mother_journeys (district, block);",
        "CREATE INDEX IF NOT EXISTS idx_mother_reg_date       ON mother_journeys (registration_date);",
        "CREATE INDEX IF NOT EXISTS idx_mother_delivery_date  ON mother_journeys (delivery_date);",

        # --- ANC visits ---
        "CREATE INDEX IF NOT EXISTS idx_anc_mother_id         ON anc_visits (mother_id);",
        "CREATE INDEX IF NOT EXISTS idx_anc_visit_date        ON anc_visits (visit_date);",
        "CREATE INDEX IF NOT EXISTS idx_anc_district_block    ON anc_visits (district, block);",

        # --- Village indicators ---
        "CREATE INDEX IF NOT EXISTS idx_village_ind_dist_block ON village_indicators_monthly (district_code_lgd, block_code_lgd);",
        "CREATE INDEX IF NOT EXISTS idx_village_ind_yearmonth  ON village_indicators_monthly (year_month);",

        # --- Geography code lookups ---
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
        "CREATE INDEX IF NOT EXISTS idx_raw_preg_mother_id    ON raw_pregnancy_records (mother_id);",
        "CREATE INDEX IF NOT EXISTS idx_raw_preg_pregnancy_id ON raw_pregnancy_records (pregnancy_id);",

        # --- NFHS lookup ---
        "CREATE INDEX IF NOT EXISTS idx_nfhs_district_round   ON nfhs_indicators (district, nfhs_round);",

        # --- Infrastructure code lookups ---
        "CREATE INDEX IF NOT EXISTS idx_health_fac_block      ON health_facilities (block_code);",
        "CREATE INDEX IF NOT EXISTS idx_health_fac_dist       ON health_facilities (district_code);",
        "CREATE INDEX IF NOT EXISTS idx_anganwadi_block        ON anganwadi_centres (block_code);",
        "CREATE INDEX IF NOT EXISTS idx_anganwadi_dist         ON anganwadi_centres (district_code);",
    ]
    _exec_sql(engine, statements, "indexes")


def apply_comments(engine):
    """Apply table-level and key column-level comments for schema documentation."""
    logger.info("Applying table and column comments…")

    table_comments = {
        # Geographic
        "states":
            "Meghalaya state polygon with Census 2011 demographic attributes. 1 feature.",
        "districts":
            "Meghalaya district polygons with Census 2011 demographics (rural/urban split, workers, SC/ST). 12 features.",
        "district_boundaries":
            "Meghalaya district boundary polygons (2021 delineation) with area/length attributes. 12 features.",
        "subdistricts":
            "Meghalaya sub-district (block) polygons with Census demographics and agricultural data. 46 features.",
        "blocks":
            "Meghalaya block boundary polygons with area and length measurements. 46 features.",
        "villages_poly":
            "Village-level polygons for all 6,862 Meghalaya villages with Census 2011 demographic data.",
        "villages_point":
            "Village-level representative points for all 6,862 Meghalaya villages with Census 2011 data.",
        # Master reference
        "master_districts":
            "Canonical district reference with LGD (Local Government Directory) codes. 12 districts.",
        "master_blocks":
            "Canonical block reference with LGD codes and parent district. 47 blocks.",
        "master_villages":
            "Village reference with LGD matching status (exact/unmatched). ~7,430 villages.",
        "master_health_facilities":
            "Health facility reference: PHCs, Sub-centres, and Other Public Facilities. ~1,799 facilities.",
        "geo_full_mapping":
            "Complete geographic hierarchy: district → block → facility → village with LGD match confidence.",
        "geo_match_report":
            "Quality report for village-to-LGD matching (exact vs. unmatched).",
        # Infrastructure
        "health_facilities":
            "Health facility directory enriched with GPS coordinates (PostGIS POINT in 'geom'). ~646 facilities.",
        "anganwadi_centres":
            "ICDS Anganwadi Centre directory with GPS, infrastructure, and administrative data (~5,900 centres). PostGIS POINT in 'geom'.",
        # Geography mapping
        "meghealth_geo_mapping":
            "Facility-to-village mapping: district → block → PHC → sub-centre → village hierarchy used in Meghealth.",
        # Health data
        "mother_journeys":
            "Complete maternal journey from pregnancy registration through delivery and child outcome. ~363,898 records. PK: row_id (SERIAL). pregnancy_id is UNIQUE (NULLs allowed — some rows are child-only with no pregnancy registration). mother_id is non-unique (one row per pregnancy).",
        "anc_visits":
            "Antenatal Care (ANC) visit records with clinical vitals (weight, BP, Hb), medications, and danger signs. ~361,551 records. PK: anc_id, FK: mother_id → mother_journeys.",
        "village_indicators_monthly":
            "Monthly aggregated maternal and child health indicators at village level. ~133,071 records.",
        # Raw records
        "raw_anc_records":
            "Raw API response records for ANC visits — normalized relational table. All original API fields preserved. ~361,508 records.",
        "raw_child_records":
            "Raw API response records for child/infant data — normalized relational table. ~192 MB source.",
        "raw_pregnancy_records":
            "Raw API response records for pregnancy journeys — normalized relational table. ~394 MB source. pregnancy_id FK → mother_journeys.",
        # Reference JSON
        "nfhs_indicators":
            "National Family Health Survey (NFHS) rounds 3/4/5 district-level health indicators. 624 records.",
        "video_library":
            "Educational video metadata for ASHA/ANM health workers (breastfeeding, nutrition, immunization etc). ~49 videos.",
        "research_articles":
            "Peer-reviewed research articles on maternal and child health in Meghalaya/Northeast India. 23 articles.",
    }

    col_comments = {
        "mother_journeys": {
            "row_id":
                "Surrogate primary key (SERIAL). Added during migration — pregnancy_id is nullable so cannot be the PK.",
            "mother_id":
                "Meghealth mother identifier. Non-unique — a mother has one row per pregnancy. Indexed for cross-pregnancy lookups.",
            "pregnancy_id":
                "Unique (NULLs allowed) composite ID, format: '{mother_id} - {sno}'. NULL for child-only records. FK target for raw_pregnancy_records.",
            "gps_location":
                "GPS location as WKT string (POINT lon lat). Convert to geometry with: ST_GeomFromText(gps_location, 4326).",
            "district":
                "District name (free-text, not normalized). Join to master_districts for LGD codes.",
            "high_risk_at_registration":
                "Boolean flag for high-risk pregnancy at time of registration.",
        },
        "anc_visits": {
            "anc_id":    "Unique ANC visit identifier. Primary key.",
            "mother_id": "Join key to mother_journeys.mother_id (indexed; no FK constraint because mother_id is non-unique in mother_journeys).",
            "visit_date":"Date of ANC visit. Indexed for time-series queries.",
        },
        "villages_point": {
            "geometry": "PostGIS POINT (EPSG:4326). Village representative GPS point.",
        },
        "villages_poly": {
            "geometry": "PostGIS POLYGON (EPSG:4326). Village boundary polygon.",
        },
        "districts": {
            "geometry": "PostGIS MULTIPOLYGON (EPSG:4326). District boundary.",
        },
        "blocks": {
            "geometry": "PostGIS MULTIPOLYGON (EPSG:4326). Block boundary.",
        },
        "health_facilities": {
            "geom":      "PostGIS POINT (EPSG:4326). Facility GPS location.",
            "facility_id":"Unique facility identifier. Matches master_health_facilities.facility_id.",
        },
        "anganwadi_centres": {
            "geom":               "PostGIS POINT (EPSG:4326). Centre GPS location.",
            "anganwadi_centre_id":"Unique centre identifier.",
            "geometry_x":         "Longitude (same as 'longitude' column, kept for source compatibility).",
            "geometry_y":         "Latitude (same as 'latitude' column, kept for source compatibility).",
        },
        "raw_anc_records": {
            "row_id":    "Surrogate primary key (SERIAL). Added during migration.",
            "mother_id": "Extracted from JSON for indexing. FK → mother_journeys.mother_id.",
        },
        "raw_child_records": {
            "row_id":    "Surrogate primary key (SERIAL). Added during migration.",
            "mother_id": "Extracted from JSON for indexing. FK → mother_journeys.mother_id.",
        },
        "raw_pregnancy_records": {
            "row_id":       "Surrogate primary key (SERIAL). Added during migration.",
            "mother_id":    "Extracted from JSON for indexing.",
            "pregnancy_id": "Pregnancy ID in format '{mother_id} - {sno}'. FK → mother_journeys.pregnancy_id.",
        },
        "master_villages": {
            "village_id":        "Primary key, format: 'V-NNNNN'.",
            "village_code_lgd":  "LGD village census code (NULL if unmatched).",
            "match_confidence":  "LGD matching quality: 'exact' or 'unmatched'.",
        },
        "nfhs_indicators": {
            "nfhs_round":   "NFHS survey round (3, 4, or 5).",
            "survey_year":  "Survey year range (e.g. '2019-21').",
            "district":     "District or state name from NFHS report.",
            "total":        "Total (combined rural+urban) indicator value.",
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
        load_csv_small(engine, MASTER_DIR / "master_districts.csv", "master_districts")
        load_csv_small(engine, MASTER_DIR / "master_blocks.csv", "master_blocks")
        load_csv_small(engine, MASTER_DIR / "master_villages.csv", "master_villages")
        load_csv_small(engine, MASTER_DIR / "master_health_facilities.csv", "master_health_facilities")
        load_csv_small(engine, MASTER_DIR / "full_mapping.csv", "geo_full_mapping")
        load_csv_small(engine, MASTER_DIR / "match_report.csv", "geo_match_report")

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
        load_csv_small(engine, OUTPUT_DIR / "meghealth_geography_mapping.csv", "meghealth_geo_mapping")

    # ------------------------------------------------------------------
    # 5. Health data tables (large CSVs — COPY protocol)
    # ------------------------------------------------------------------
    if stage("health"):
        logger.info("=== Stage: Health data tables ===")
        load_csv_large(
            engine,
            OUTPUT_DIR / "mother_journeys.csv",
            "mother_journeys",
            skip_if_exists=skip_if_exists,
        )
        load_csv_large(
            engine,
            OUTPUT_DIR / "anc_visits_detail.csv",
            "anc_visits",
            skip_if_exists=skip_if_exists,
        )
        load_csv_small(engine, OUTPUT_DIR / "village_indicators_monthly.csv", "village_indicators_monthly")

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
        load_json_normalized(
            engine,
            OUTPUT_DIR / "raw_pregnancy_records.json",
            "raw_pregnancy_records",
            skip_if_exists=skip_if_exists,
        )
    elif skip_raw_json:
        logger.info("Skipping raw JSON records (--skip-raw-json)")

    # ------------------------------------------------------------------
    # 7. Reference JSON tables (small — flat relational)
    # ------------------------------------------------------------------
    if stage("ref_json"):
        logger.info("=== Stage: Reference JSON tables ===")
        load_json_flat(engine, OUTPUT_DIR / "nfhs_indicators.json", "nfhs_indicators")
        load_json_flat(engine, OUTPUT_DIR / "video_library.json", "video_library")
        load_json_flat(engine, OUTPUT_DIR / "research_articles.json", "research_articles")

    # ------------------------------------------------------------------
    # 8. Schema objects — PKs, FKs, indexes, comments (ALWAYS last)
    # ------------------------------------------------------------------
    if stage("schema") or only is None:
        logger.info("=== Stage: Schema objects (PKs, FKs, indexes, comments) ===")
        apply_primary_keys(engine)
        apply_foreign_keys(engine)
        apply_indexes(engine)
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
        choices=["geo", "master", "infra", "health", "raw_json", "ref_json", "schema"],
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
