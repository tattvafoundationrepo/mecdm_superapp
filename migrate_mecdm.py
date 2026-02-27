import os
import json
import logging
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
from geoalchemy2 import Geometry
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load local environment variables if available
load_dotenv()


def get_engine():
    """Create SQLAlchemy engine from environment variables."""
    # Note: For AlloyDB via Auth Proxy, standard PG environment variables can be used
    user = os.environ.get("ALLOYDB_POSTGRES_USER", "postgres")
    password = os.environ.get("ALLOYDB_POSTGRES_PASSWORD", "postgres")
    host = os.environ.get("ALLOYDB_POSTGRES_HOST", "127.0.1.1")
    port = os.environ.get("ALLOYDB_POSTGRES_PORT", "5432")
    db = os.environ.get("ALLOYDB_POSTGRES_DATABASE", "postgres")

    conn_str = f"postgresql://{user}:{password}@{host}:{port}/{db}"
    # Standard SSL connection mapping for remote AlloyDB/Cloud SQL instances
    return create_engine(
        conn_str,
        connect_args={'sslmode': 'require'}
    )


def create_extensions(engine):
    """Enable necessary PostgreSQL extensions."""
    logger.info("Ensuring PostGIS is installed...")
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))


def load_geojson(engine, filepath, table_name):
    """Load a GeoJSON file into a PostGIS table."""
    logger.info(f"Loading {filepath} into {table_name}...")
    try:
        gdf = gpd.read_file(filepath)
        # Ensure it's in WGS84
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        # Determine geometry type dynamically or default to Geometry
        geom_type = 'GEOMETRY'

        gdf.to_postgis(
            table_name,
            engine,
            if_exists='replace',
            index=False,
            dtype={'geometry': Geometry(geom_type, srid=4326)}
        )
        logger.info(f"Successfully loaded {table_name}.")
    except Exception as e:
        logger.error(f"Failed to load {table_name}: {e}")


def load_csv(engine, filepath, table_name, lat_col=None, lon_col=None):
    """Load a CSV file, optionally creating a geometry column."""
    logger.info(f"Loading {filepath} into {table_name}...")
    try:
        df = pd.read_csv(filepath)
        if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
            # Drop rows with invalid coordinates
            valid_coords = df[df[lat_col].notnull() & df[lon_col].notnull()]
            gdf = gpd.GeoDataFrame(
                valid_coords,
                geometry=gpd.points_from_xy(
                    valid_coords[lon_col], valid_coords[lat_col]),
                crs="EPSG:4326"
            )
            # Find rows without coordinates to append as standard table rows
            invalid_coords = df[df[lat_col].isnull() | df[lon_col].isnull()]
            if not invalid_coords.empty:
                logger.warning(
                    f"Found {len(invalid_coords)} rows without valid coordinates for {table_name}.")

            gdf.rename_geometry('geom', inplace=True)
            gdf.to_postgis(
                table_name,
                engine,
                if_exists='replace',
                index=False,
                dtype={'geom': Geometry('POINT', srid=4326)}
            )
            # Append non-spatial rows
            if not invalid_coords.empty:
                invalid_coords.to_sql(table_name, engine,
                                      if_exists='append', index=False)
        else:
            df.to_sql(table_name, engine, if_exists='replace', index=False)

        logger.info(f"Successfully loaded {table_name}.")
    except Exception as e:
        logger.error(f"Failed to load {table_name}: {e}")


def load_json_records(engine, filepath, table_name, fk_col=None, ref_table=None, ref_col=None, table_comment=None):
    """Load raw JSON into a dedicated table with a JSONB column and foreign key."""
    logger.info(f"Loading JSON data from {filepath} into {table_name}...")
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Standardize to list
        if isinstance(data, dict):
            # Sometimes JSON dumps are dicts with internal lists, adapt based on exploration
            keys = list(data.keys())
            if len(keys) == 1 and isinstance(data[keys[0]], list):
                data = data[keys[0]]
            else:
                data = [data]

        # Prepare DataFrame
        records = []
        for i, record in enumerate(data):
            # Extract potential ID and FK
            record_id = None
            fk_val = None

            # Simple heuristic to find keys
            for k in record.keys():
                k_lower = k.lower()
                if '-id' in k_lower or k_lower == 'id':
                    record_id = record[k]
                if fk_col and (fk_col in k_lower or 'pregnancy_id' in k_lower):
                    fk_val = record[k]

            # Fallback ID
            if record_id is None:
                record_id = f"{table_name}_{i}"

            row = {'id': record_id, 'record_data': json.dumps(record)}
            if fk_col:
                row[fk_col] = fk_val
            records.append(row)

        df = pd.DataFrame(records)

        # We need JSONB type
        from sqlalchemy.dialects.postgresql import JSONB
        dtypes = {'record_data': JSONB}
        if fk_col:
            # Note: We won't enforce the FK constraint strictly immediately to avoid loading order issues,
            # but we can add it subsequently.
            pass

        df.to_sql(table_name, engine, if_exists='replace',
                  index=False, dtype=dtypes)

        # Add primary key, foreign key, and comments explicitly via SQL
        with engine.begin() as conn:
            conn.execute(
                text(f"ALTER TABLE {table_name} ADD PRIMARY KEY (id);"))
            if table_comment:
                conn.execute(
                    text(f"COMMENT ON TABLE {table_name} IS '{table_comment}';"))

            if fk_col and ref_table and ref_col:
                try:
                    conn.execute(text(f"""
                        ALTER TABLE {table_name} 
                        ADD CONSTRAINT fk_{table_name}_{ref_table} 
                        FOREIGN KEY ({fk_col}) REFERENCES {ref_table}({ref_col}) ON DELETE CASCADE;
                    """))
                except Exception as fk_err:
                    logger.warning(
                        f"Could not apply FK constraint for {table_name}: {fk_err}")

        logger.info(
            f"Successfully loaded {table_name} with {len(df)} records.")
    except Exception as e:
        logger.error(f"Failed to load JSON {table_name}: {e}")


def run_migration():
    engine = get_engine()
    create_extensions(engine)

    base_dir = '/Users/tattva/_Projects/mecdm-super-agent/mecdm_dataset/db_20260226'
    geo_dir = os.path.join(base_dir, 'Meghalaya Master Data')
    output_dir = os.path.join(base_dir, 'output')

    # 1. Geographic Tables (GeoJSON)
    load_geojson(engine, os.path.join(geo_dir, '04_state.geojson'), 'states')
    load_geojson(engine, os.path.join(
        geo_dir, '03_district.geojson'), 'districts')
    load_geojson(engine, os.path.join(
        geo_dir, '02_subdistrict.geojson'), 'subdistricts')
    load_geojson(engine, os.path.join(
        geo_dir, '01_village_poly.geojson'), 'villages_poly')
    load_geojson(engine, os.path.join(
        geo_dir, '00_village_point.geojson'), 'villages_point')

    # 2. Key Master Data & Tracking (CSV)
    # Using specific files found in exploration
    mother_journeys_path = os.path.join(output_dir, 'mother_journeys.csv')
    if os.path.exists(mother_journeys_path):
        load_csv(engine, mother_journeys_path, 'mother_journeys')
        # Make pregnancy_id Unique so we can reference it
        with engine.begin() as conn:
            try:
                conn.execute(
                    text("ALTER TABLE mother_journeys ADD PRIMARY KEY (mother_id);"))
                conn.execute(text(
                    "ALTER TABLE mother_journeys ADD CONSTRAINT unique_pregnancy_id UNIQUE (pregnancy_id);"))
            except:
                pass

    anc_visits_path = os.path.join(output_dir, 'anc_visits_detail.csv')
    if os.path.exists(anc_visits_path):
        load_csv(engine, anc_visits_path, 'anc_visits')

    facilities_path = os.path.join(
        geo_dir, 'dashboard_health_facilities_codes_enriched.csv')
    if os.path.exists(facilities_path):
        load_csv(engine, facilities_path, 'health_facilities',
                 lat_col='latitude', lon_col='longitude')

    anganwadi_path = os.path.join(
        geo_dir, 'anganwadi_centre_0_anganwadi_centre.csv')
    if os.path.exists(anganwadi_path):
        load_csv(engine, anganwadi_path, 'anganwadi_centres',
                 lat_col='latitude', lon_col='longitude')

    indicators_path = os.path.join(
        output_dir, 'village_indicators_monthly.csv')
    if os.path.exists(indicators_path):
        load_csv(engine, indicators_path, 'village_indicators_monthly')

    # 3. Raw JSON Data with explicit FKs and comments
    load_json_records(
        engine,
        os.path.join(output_dir, 'raw_anc_records.json'),
        'raw_anc_records',
        fk_col='pregnancy_id',
        ref_table='mother_journeys',
        ref_col='pregnancy_id',
        table_comment="Raw API JSON responses for ANC details."
    )

    load_json_records(
        engine,
        os.path.join(output_dir, 'raw_child_records.json'),
        'raw_child_records',
        fk_col='pregnancy_id',
        ref_table='mother_journeys',
        ref_col='pregnancy_id',
        table_comment="Raw API JSON responses for Child details."
    )

    load_json_records(
        engine,
        os.path.join(output_dir, 'raw_pregnancy_records.json'),
        'raw_pregnancy_records',
        fk_col='pregnancy_id',
        ref_table='mother_journeys',
        ref_col='pregnancy_id',
        table_comment="Raw API JSON responses for Pregnancy metadata."
    )

    load_json_records(
        engine,
        os.path.join(output_dir, 'research_articles.json'),
        'research_articles',
        table_comment="Raw structured JSON for health research articles."
    )

    load_json_records(
        engine,
        os.path.join(output_dir, 'video_library.json'),
        'video_library',
        table_comment="Raw structured JSON for the video library."
    )

    load_json_records(
        engine,
        os.path.join(output_dir, 'nfhs_indicators.json'),
        'nfhs_indicators',
        table_comment="Raw structured JSON for NFHS indicator data."
    )

    logger.info("Migration complete!")


if __name__ == "__main__":
    run_migration()
