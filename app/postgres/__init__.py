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

            _ = cur.execute(
                t"select exists (select 1 from information_schema.tables where table_name = {table_name})"
            )
            table_exists = (cur.fetchone() or [False])[0]

            if table_exists:
                APP_LOGGER.info(f"Table '{table_name}' already exists.")
                return

            APP_LOGGER.info("Creating schema...")
            _ = cur.execute(
                f"""
                create table {table_name} (
                    id serial primary key,
                    filename varchar not null,
                    histogram_brute vector({bow_len}),
                    histogram_ivf   vector({bow_len}),
                    histogram_hnsw  vector({bow_len})
                )
                """
            )

            APP_LOGGER.info(f"Inserting {len(filenames)} rows...")

            with cur.copy(
                f"copy {table_name} (filename, histogram_brute, histogram_ivf, histogram_hnsw) from stdin"
            ) as copy:
                for i in range(len(filenames)):
                    vec_str = str(hists[i].tolist())
                    copy.write_row((filenames[i], vec_str, vec_str, vec_str))

            conn.commit()

            lists = 100
            APP_LOGGER.info(f"Building IVFFlat index (lists = {lists})...")
            _ = cur.execute(
                f"""
                create index idx_{table_name}_ivf on {table_name}
                using ivfflat (histogram_ivf vector_cosine_ops) with (lists = {lists})
                """
            )
            conn.commit()

            m = 20
            ef = 40
            APP_LOGGER.info(f"Building HNSW index (m = {m}, ef_construction = {ef})...")
            _ = cur.execute(
                f"""
                create index idx_{table_name}_hnsw on {table_name}
                using hnsw (histogram_hnsw vector_cosine_ops) with (m = {m}, ef_construction = {ef})
                """
            )
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
            _ = cur.execute(
                t"select exists (select 1 from information_schema.tables where table_name = {table_name})"
            )
            table_exists = (cur.fetchone() or [False])[0]

            if table_exists:
                APP_LOGGER.info(f"Table '{table_name}' already exists.")
                return

            APP_LOGGER.info("Creating schema...")
            _ = cur.execute(
                f"""
                create table {table_name} (
                    id serial primary key,
                    filename varchar not null,
                    content text not null,
                    content_tsv tsvector generated always as (to_tsvector('{language}', content)) stored
                )
                """
            )

            APP_LOGGER.info(f"Inserting {len(text_files)} rows...")

            with cur.copy(f"copy {table_name} (filename, content) from stdin") as copy:
                for filename in text_files:
                    content = filename.read_text()
                    copy.write_row((filename.name, content))

            conn.commit()

            APP_LOGGER.info("Building GIN index...")
            _ = cur.execute(
                f"create index idx_{table_name}_tsv on {table_name} using gin (content_tsv)"
            )
            conn.commit()

    APP_LOGGER.info("Done.")
