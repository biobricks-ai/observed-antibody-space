#!/usr/bin/env python3
"""
Mirror the full Observed Antibody Space (OAS) parquet dataset from HuggingFace.

Sources:
  - Unpaired: https://huggingface.co/datasets/ConvergeBio/oas-unpaired
      ~2.07B heavy + 357M light sequences, ~2.45 TB, 4,030 parquet shards
      File layout: data/unpaired_heavy/<Study>/*.parquet
                   data/unpaired_light/<Study>/*.parquet
  - Paired:   https://huggingface.co/datasets/bloyal/oas-paired-sequence-data
      File layout: <species>/train_N.parquet

Citation: Olsen TH, Boyles F, Deane CM (2022). Observed Antibody Space: A diverse
database of cleaned, annotated, and translated unpaired and paired antibody sequences.
Protein Science 31(1):141-146. DOI: 10.1002/pro.4205
OAS web: https://opig.stats.ox.ac.uk/webapps/oas/

License: CC-BY-4.0
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

UNPAIRED_REPO = "ConvergeBio/oas-unpaired"
PAIRED_REPO   = "bloyal/oas-paired-sequence-data"

# When SLIM_TEST=1 (env var), download only a couple of shards for verification.
SLIM_TEST = os.environ.get("SLIM_TEST", "0") == "1"

# Small studies (1 shard each) used for slim-test verification of unpaired
# Actual path: data/unpaired_heavy/Bashford et al., 2013/train-00000.parquet
SLIM_HEAVY_STUDY = "Bashford et al., 2013"
SLIM_LIGHT_STUDY = "Bhiman et al., 2015"


def download_unpaired(brick_path: Path) -> None:
    dest = brick_path / "oas-unpaired"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading unpaired OAS -> {dest}")
    print(f"  repo: {UNPAIRED_REPO}")

    kwargs = dict(
        repo_id=UNPAIRED_REPO,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=["data/**/*.parquet"],
    )

    if SLIM_TEST:
        kwargs["allow_patterns"] = [
            f"data/unpaired_heavy/{SLIM_HEAVY_STUDY}/*.parquet",
            f"data/unpaired_light/{SLIM_LIGHT_STUDY}/*.parquet",
        ]
        print(f"  SLIM_TEST mode: downloading 2 small studies only")
        print(f"    heavy: {SLIM_HEAVY_STUDY}")
        print(f"    light: {SLIM_LIGHT_STUDY}")

    snapshot_download(**kwargs)
    print(f"  Done: {dest}")


def download_paired(brick_path: Path) -> None:
    dest = brick_path / "oas-paired"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading paired OAS -> {dest}")
    print(f"  repo: {PAIRED_REPO}")

    kwargs = dict(
        repo_id=PAIRED_REPO,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=["*.parquet", "**/*.parquet"],
    )

    if SLIM_TEST:
        # bloyal/oas-paired is relatively small; the first call already proved it works.
        # In slim mode, grab just the human subfolder.
        kwargs["allow_patterns"] = ["human/*.parquet"]
        print("  SLIM_TEST mode: downloading human/ paired shards only")

    snapshot_download(**kwargs)
    print(f"  Done: {dest}")


def verify_parquet(brick_path: Path) -> None:
    """Print schema and row counts for downloaded parquet files."""
    import pyarrow.parquet as pq
    import glob

    print("\n--- Parquet verification ---")
    patterns = [
        str(brick_path / "oas-unpaired" / "**" / "*.parquet"),
        str(brick_path / "oas-paired"   / "**" / "*.parquet"),
    ]
    total_rows = 0
    for pattern in patterns:
        files = sorted(glob.glob(pattern, recursive=True))
        for fpath in files[:5]:  # show up to 5 per glob
            pf = pq.ParquetFile(fpath)
            meta = pf.metadata
            schema = pf.schema_arrow
            rows = meta.num_rows
            total_rows += rows
            rel = Path(fpath).relative_to(brick_path)
            print(f"  {rel}")
            print(f"    rows: {rows:,}")
            print(f"    cols: {[schema.field(i).name for i in range(min(8, len(schema)))]}")
        if files:
            print(f"  ... ({len(files)} files in this set)")
    print(f"  Total rows verified: {total_rows:,}")
    print("--- end verification ---\n")


def main():
    brick_path = Path("brick")
    brick_path.mkdir(exist_ok=True)

    print("=" * 60)
    print("Observed Antibody Space (OAS) -- HuggingFace parquet mirror")
    print("=" * 60)
    if SLIM_TEST:
        print("MODE: SLIM_TEST (a few shards only)")
    else:
        print("MODE: FULL download (~2.45 TB unpaired + paired)")
    print()

    download_unpaired(brick_path)
    download_paired(brick_path)
    verify_parquet(brick_path)

    print("All done.")


if __name__ == "__main__":
    main()
