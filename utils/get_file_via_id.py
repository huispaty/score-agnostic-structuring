#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Take an id (composition id or performer id) from the ATEPP dataset as input,
and copy the transcription(s) belonging to this id.

Example use:
python ./utils/get_file_via_id.py -id 1165 1176 -t p -o test_out_path
"""

import sys

sys.path.append("..")

import os
import pandas as pd
import argparse
import shutil

from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from paths import ATEPP_transcriptions, ATEPP_meta


def copy_file(id, id_type, out_path, ATEPP_meta_df=ATEPP_meta):
    """
    Copy a file based on id and id_type.

    Args:
        id: identifier (composition_id or perf_id depending on id_type)
        id_type: either 'composition' or 'performance'
        out_path: destination path for the copied file(s)
        ATEPP_meta_df: DataFrame containing metadata
    """

    if isinstance(id, int):
        id = [id]
    if id_type == "c":
        transcriptions = ATEPP_meta_df[ATEPP_meta_df["composition_id"].isin(id)]
    elif id_type == "p":
        id_str = [f"{v:05d}" for v in id]
        transcriptions = ATEPP_meta_df[ATEPP_meta_df["perf_id"].isin(id_str)]
    else:
        raise ValueError(
            f"Invalid id_type '{id_type}'. Must be 'c' (composition) or 'p' (performance)"
        )

    for _, (row) in transcriptions.iterrows():

        composer = row["composer"].split(" ")[-1]
        comp_id = row["composition_id"]
        perf_id = row["perf_id"]
        src = ATEPP_transcriptions / row["norm_midi_path"]
        dst = Path(out_path) / composer / str(comp_id) / f"{int(perf_id)}.mid"
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        if not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"Copied src {src} to dst {dst}")
        else:
            print(f"Already exists: src {src} at dst {dst}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Select compositions via their transcription count",
        description="Copy transcriptions from the ATEPP dataset, either via their composition or performance id",
    )

    parser.add_argument("-id", type=int, nargs="+", help="One or more integer ids")
    parser.add_argument(
        "--id_type",
        "-t",
        type=str,
        choices=["c", "p"],
        help="Specify whether you are handling composition or performance ids",
    )
    parser.add_argument("--out_path", "-o", type=str)

    args = parser.parse_args()

    copy_file(args.id, args.id_type, args.out_path)
