"""Run SQL migrations against mecdm_superapp_user_db.

Usage:
    uv run python -m migrations.run_migration

Requires DATABASE_URL_USER to be set (or loaded via .env).
"""

import os
import sys
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    url = os.environ.get("DATABASE_URL_USER")
    if not url:
        print("ERROR: DATABASE_URL_USER not set")
        sys.exit(1)

    migrations_dir = Path(__file__).parent
    sql_files = sorted(migrations_dir.glob("*.sql"))

    if not sql_files:
        print("No SQL migration files found.")
        return

    conn = psycopg2.connect(url)
    cur = conn.cursor()

    for sql_file in sql_files:
        print(f"Running {sql_file.name}...")
        cur.execute(sql_file.read_text())
        conn.commit()
        print(f"  Done.")

    cur.close()
    conn.close()
    print("All migrations complete.")


if __name__ == "__main__":
    main()
