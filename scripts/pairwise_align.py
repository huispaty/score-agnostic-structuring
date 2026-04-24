#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Create pairwise alignments of all transcriptions of a given piece by a given composer 
(currently only supported for the ATEPP dataset)

Example use:
python ./scripts/pairwise_align.py -c Beethoven -o ./data/TEST/
python ./scripts/pairwise_align.py -c Beethoven -cid 1176 -o ./data/TEST/

Alignments will be saved to <out_path>/<composer-last-name>/<composition_id>/

"""

import sys

sys.setrecursionlimit(10000)
sys.path.append("..")

import os
import csv
from pathlib import Path

from itertools import combinations
from functools import partial

import numpy as np
import pandas as pd

import partitura as pt

from mpteval.preprocessing.preprocess import chordify_perf_note_array
from mpteval.alignment.dtw_align import process_pair_wrapper
from mpteval.alignment.dtw_dist import composite_cost

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import argparse

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from paths import ATEPP_transcriptions, ATEPP_meta, output_path

from datetime import datetime
DATESTAMP = datetime.now().strftime("%y%m%d")

def create_alignments(
    composer: str,
    composition_id: int | None,
    out_path: str | Path,
    ATEPP_meta: pd.DataFrame = ATEPP_meta,
    ioi_threshold: float = 0.03,
    max_threshold: float = 0.5,
    alpha: float = 0.85,
    harmonic_metric: str = "jaccard",
    pitch_feature: str = "pc_set",
    debug: bool = False,
) -> None:
    """
    Create pairwise alignments for all performances of a composition or composer.

    Processes performances by chordifying note arrays, computing DTW alignments
    between all pairs, and saving results to disk.

    Parameters
    ----------
    composer : str
        Composer name to filter compositions.
    composition_id : int or None
        Specific composition ID to process. If None, processes all compositions
        by the given composer.
    out_path : str or Path
        Base output directory for saving alignments and performance data.
    ATEPP_meta : pd.DataFrame, optional
        Metadata dataframe containing composition and performance information.
    ioi_threshold : float
        IOI threshold for chordification.
    max_threshold : float
        Max threshold for chordification.
    alpha : float
        Weighting between harmonic and time cost in composite_cost.
    harmonic_metric : str
        Harmonic distance metric to use in composite_cost.
    pitch_feature : str
        Pitch feature to use in composite_cost.

    Returns
    -------
    None
        Saves outputs to disk: performance data, alignments, and alignment statistics.
    """

    if not composition_id:
        comp_ids = ATEPP_meta[ATEPP_meta["composer"].str.contains(composer)][
            "composition_id"
        ].unique()
        print(f"Processing all {len(comp_ids)} compositions for composer {composer}")
        for cid in comp_ids:
            create_alignments(
                composer, cid, out_path, ATEPP_meta,
                ioi_threshold, max_threshold, alpha, harmonic_metric, pitch_feature, debug
            )
        return

    composer = composer.split()[-1]
    out_dir = Path(out_path) / composer / f"{composition_id}"
    os.makedirs(out_dir, exist_ok=True)

    # return if there's only 1 performance
    group = ATEPP_meta[ATEPP_meta["composition_id"] == int(composition_id)]
    composition_title = group["track"].values[0]
    if len(group) < 2:
        print("------------------------------------")
        print(
            f"Skipping composition {composition_id} by {composer}: \n{composition_title}, \nonly {len(group)} transcription found."
        )
        return

    print("------------------------------------")
    print(
        f"Processing composition {composition_id} by {composer}: \n{composition_title}, \n{len(group):4d} transcription found."
    )
    # performance data
    perf_out_dir = out_dir / "1_performance_data"
    os.makedirs(perf_out_dir, exist_ok=True)

    # build pid to chords list dict
    pid_chord_list_dict = dict()
    for pid, midi in zip(group["perf_id"], group["norm_midi_path"]):
        midi_path = ATEPP_transcriptions / midi
        p1 = pt.load_performance_midi(midi_path).performedparts[0].note_array()
        np.save(f"{perf_out_dir}/{pid}.npy", p1)
        p1c = chordify_perf_note_array(
            p1, ioi_threshold=ioi_threshold, max_threshold=max_threshold, return_list_of_dicts=True
        )

        pid_chord_list_dict[str(pid)] = p1c

    pairs = list(combinations(group["perf_id"], 2))
    print(
        f"Generated {len(pairs):,} pairs from {len(group['perf_id']):,} transcriptions"
    )

    dist_metric = partial(
        composite_cost, alpha=alpha, harmonic_metric=harmonic_metric, pitch_feature=pitch_feature
    )

    # save alignments
    align_out_dir = out_dir / "2_alignments"
    os.makedirs(align_out_dir, exist_ok=True)

    process_func = partial(
        process_pair_wrapper,
        pid_to_chords_dict=pid_chord_list_dict,
        dist_metric=dist_metric,
        out_dir=align_out_dir,
        directional_weights=np.array([1, 2, 1]),
    )

    results = []

    if debug:
        for pair in tqdm(pairs, desc="Current pairs progress"):
            try:
                result = process_func(pair)
                results.append(result)
            except Exception as e:
                print(f"Error processing pair {pair}: {e}")
                import traceback
                traceback.print_exc()
    
    else:
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_func, pair): pair for pair in pairs}

            with tqdm(total=len(pairs), desc="Current pairs progress") as pbar:
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        pair = futures[future]
                        print(f"Error processing pair {pair}: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        pbar.update(1)

    # save results to csv
    align_csv = f"{align_out_dir}/align.csv"

    with open(align_csv, mode="w+", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["p1", "p2", "cost", "align_len", "p1_len", "p2_len"])
        writer.writerows(results)

    print(f"Saved performance data to {perf_out_dir}")
    print(f"Saved alignments to {align_out_dir}")
    print(f"Saved align stats to {align_csv}")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create pairwise alignments for performances"
    )
    parser.add_argument(
        "--composer", "-c", required=False,
        help="Composer name (last name is sufficient)",
    )
    parser.add_argument(
        "--composition_id", "-cid", required=False,
        help="Specific composition ID. If not given with --composer, all compositions for the composer will be processed.",
    )
    parser.add_argument(
        "--subset_meta_csv", "-csv", required=False,
        help="Path to subset metadata CSV with columns: split, composer, composition_id, perf_id, structure, score_path. If provided, processes all compositions in the subset.",
    )
    parser.add_argument(
        "--dset_meta_file", "-meta", default=ATEPP_meta,
        help="Path to dataset metadata file.",
    )
    parser.add_argument(
        "--out_path", "-o", default= output_path/ DATESTAMP,
        help="Base output path.",
    )
    parser.add_argument(
        "--ioi_threshold", type=float, default=0.03,
        help="IOI threshold for chordification (default: 0.03)",
    )
    parser.add_argument(
        "--max_threshold", type=float, default=0.5,
        help="Max threshold for chordification (default: 0.5)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.85,
        help="Weighting between harmonic and time cost (default: 0.85)",
    )
    parser.add_argument(
        "--harmonic_metric", type=str, default="jaccard",
        help="Harmonic distance metric (default: jaccard)",
    )
    parser.add_argument(
        "--pitch_feature", type=str, default="pc_set",
        help="Pitch feature for distance computation (default: pc_set)",
    )
    parser.add_argument(
    "--debug", action="store_true",
    help="Run alignments sequentially (no multiprocessing) for easier debugging.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    shared_kwargs = dict(
        ioi_threshold=args.ioi_threshold,
        max_threshold=args.max_threshold,
        alpha=args.alpha,
        harmonic_metric=args.harmonic_metric,
        pitch_feature=args.pitch_feature,
        debug=args.debug
    )

    if args.subset_meta_csv:
        subset_meta = pd.read_csv(args.subset_meta_csv)
        compositions = subset_meta.groupby(["split", "composer", "composition_id"])

        for (split, composer, composition_id), group in compositions:
            out_path_split = Path(args.out_path) / split
            create_alignments(
                composer=composer,
                composition_id=composition_id,
                out_path=out_path_split,
                ATEPP_meta=args.dset_meta_file,
                **shared_kwargs,
            )

    elif args.composer:
        create_alignments(
            composer=args.composer,
            composition_id=args.composition_id,
            out_path=args.out_path,
            ATEPP_meta=args.dset_meta_file,
            **shared_kwargs,
        )
    else:
        parser.error("Either --composer or --subset_meta_csv must be provided")