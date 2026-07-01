import json
from pathlib import Path

import numpy as np
import psycopg

from app.common.logger import APP_LOGGER


def init(data_dir: Path, table_name: str):
    APP_LOGGER.info(f"Initializing {table_name} in Postgres...")

    histograms_path = data_dir / "histograms.npy"
    media_files_path = data_dir / "media_files.json"

    hists: np.ndarray = np.load(str(histograms_path))
    with open(media_files_path) as f:
        filenames: list[str] = json.load(f)

    APP_LOGGER.info(f"Loaded {len(filenames)} histograms (dim={hists.shape[1]})")

    bow_len = hists.shape[1]

    with psycopg.connect() as conn:
        with conn.cursor() as cur:
            APP_LOGGER.info("Creating pgvector extension...")
            _ = cur.execute("create extension if not exists vector")

            _ = cur.execute(t"""
                select exists (select 1 from information_schema.tables where table_name = {table_name:l})
                """)
            table_exists = (cur.fetchone() or [False])[0]

            if table_exists:
                APP_LOGGER.info(f"Table '{table_name}' already exists.")
                return

            APP_LOGGER.info("Creating schema...")
            _ = cur.execute(t"""
                create table {table_name:i} (
                    id serial primary key,
                    filename varchar not null,
                    histogram_brute vector({bow_len:l}),
                    histogram_ivf   vector({bow_len:l}),
                    histogram_hnsw  vector({bow_len:l})
                )
                """)

            APP_LOGGER.info(f"Inserting {len(filenames)} rows...")

            with cur.copy(t"""
                copy {table_name:i} (filename, histogram_brute, histogram_ivf, histogram_hnsw) from stdin
                """) as copy:
                for i in range(len(filenames)):
                    vec_str = str(hists[i].tolist())
                    copy.write_row((filenames[i], vec_str, vec_str, vec_str))

            conn.commit()

            lists = 100
            APP_LOGGER.info(f"Building IVFFlat index (lists = {lists})...")
            _ = cur.execute(t"""
                create index {f"idx_{table_name}_ivf":i} on {table_name:i}
                using ivfflat (histogram_ivf vector_cosine_ops) with (lists = {lists:l})
                """)
            conn.commit()

            m = 20
            ef = 40
            APP_LOGGER.info(f"Building HNSW index (m = {m}, ef_construction = {ef})...")
            _ = cur.execute(t"""
                create index {f"idx_{table_name}_hnsw":i} on {table_name:i}
                using hnsw (histogram_hnsw vector_cosine_ops) with (m = {m:l}, ef_construction = {ef:l})
                """)
            conn.commit()

    APP_LOGGER.info("Done.")


def init_text(texts_dir: Path, table_name: str, language: str = "english"):
    APP_LOGGER.info(f"Initializing {table_name} in Postgres...")

    text_files = list(texts_dir.iterdir())
    if len(text_files) == 0:
        APP_LOGGER.warning(f"No files found in {texts_dir}")
        return

    APP_LOGGER.info(f"Found {len(text_files)} text files.")

    with psycopg.connect() as conn:
        with conn.cursor() as cur:
            _ = cur.execute(t"""
                select exists (select 1 from information_schema.tables where table_name = {table_name:l})
                """)
            table_exists = (cur.fetchone() or [False])[0]

            if table_exists:
                APP_LOGGER.info(f"Table '{table_name}' already exists.")
                return

            APP_LOGGER.info("Creating schema...")
            _ = cur.execute(t"""
                create table {table_name:i} (
                    id serial primary key,
                    filename varchar not null,
                    content text not null,
                    content_tsv tsvector generated always as (to_tsvector({language:l}, content)) stored
                )
                """)

            APP_LOGGER.info(f"Inserting {len(text_files)} rows...")

            with cur.copy(t"""
                copy {table_name:i} (filename, content) from stdin
                """) as copy:
                for filename in text_files:
                    content = filename.read_text()
                    copy.write_row((filename.name, content))

            conn.commit()

            APP_LOGGER.info("Building GIN index...")
            _ = cur.execute(t"""
                create index {f"idx_{table_name}_tsv":i} on {table_name:i} using gin (content_tsv)
                """)
            conn.commit()

    APP_LOGGER.info("Done.")
