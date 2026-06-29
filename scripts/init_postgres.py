import json
import os
import sys
from pathlib import Path

import numpy as np
import psycopg

PREPROCESSED_DIR = Path(".data/images")
BOW_LEN = 1000


def main():
    histograms_path = PREPROCESSED_DIR / "histograms.npy"
    media_files_path = PREPROCESSED_DIR / "media_files.json"

    hists: np.ndarray = np.load(str(histograms_path))
    with open(media_files_path) as f:
        filenames: list[str] = json.load(f)

    print(f"Loaded {len(filenames)} histograms (dim={hists.shape[1]})")

    bow_len = hists.shape[1]

    with psycopg.connect() as conn:
        with conn.cursor() as cur:
            print("Creating pgvector extension...")
            _ = cur.execute("create extension if not exists vector")

            print("Recreating schema...")
            _ = cur.execute("drop table if exists images cascade")
            _ = cur.execute(
                f"""
                create table images (
                    id serial primary key,
                    filename varchar not null,
                    histogram_brute vector({bow_len}),
                    histogram_ivf   vector({bow_len}),
                    histogram_hnsw  vector({bow_len})
                )
                """
            )

            print(f"Inserting {len(filenames)} rows...")

            rows = [
                (filenames[i], hists[i].tolist(), hists[i].tolist(), hists[i].tolist())
                for i in range(len(filenames))
            ]
            _ = cur.executemany(
                """
                insert into images (filename, histogram_brute, histogram_ivf, histogram_hnsw)
                values (%s, %s, %s, %s)
                """,
                rows,
            )
            conn.commit()

            lists = 100
            print(f"Building IVFFlat index (lists = {lists})...")
            _ = cur.execute(
                f"""
                create index idx_images_ivf on images
                using ivfflat (histogram_ivf vector_cosine_ops) with (lists = {lists})
                """
            )
            conn.commit()

            m = 20
            ef = 40
            print(f"Building HNSW index (m = {m}, ef_construction = {ef})...")
            _ = cur.execute(
                f"""
                create index idx_images_hnsw on images
                using hnsw (histogram_hnsw vector_cosine_ops) with (m = {m}, ef_construction = {ef})
                """
            )
            conn.commit()

    print("Done.")


if __name__ == "__main__":
    main()
