#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Infer the repeat structure in performances based on score backtracking [_1]

[_1] Peter, S., Hu, P., & Widmer, G. (2025).
     How to Infer Repeat Structures in MIDI Performances.
     MEC 2025, https://arxiv.org/pdf/2505.05055

Example use:
python ./scripts/infer_repeats_score.py -cid 861

Per default, saves estimated repeat structures to
./data/meta/estimated_repeats.csv

"""


import sys

sys.path.append("..")

import os
from pathlib import Path
from tqdm import tqdm
import argparse

import csv
import pandas as pd

import parangonar as pa
import partitura as pt

import warnings
warnings.filterwarnings("ignore")

from concurrent.futures import ProcessPoolExecutor, as_completed

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from paths import ATEPP_meta, ATEPP_transcriptions, output_path

from datetime import datetime
DATESTAMP = datetime.now().strftime("%y%m%d")


def process_single_performance(args):
    """
    Process a single score-performance pair.
    This function must be picklable (top-level function, no closures).
    
    Args:
        args: Tuple of (composer, cid, pid, pmidi, score_path, dataset_path)
    
    Returns:
        Tuple of (composer, cid, pid, structure, score_path, error_msg)
    """
    composer, cid, pid, pmidi, score_path, dataset_path = args
    
    try:
        score_xml = pt.load_musicxml(dataset_path / score_path)
        perf_mid = pt.load_performance_midi(dataset_path / pmidi).performedparts[0]
        
        ri = pa.RepeatIdentifier() # processes don't share memory
        found_path = ri(score_xml, perf_mid)
        
        if found_path is None:
            structure = "no_struct_var"
        else:
            structure = found_path[0]
        
        return (composer, cid, pid, structure, score_path, None)
        
    except Exception as e:
        error_msg = f"ERROR: {type(e).__name__}: {str(e)}"
        return (composer, cid, pid, error_msg, score_path, str(e))

def sort_labels(out_csv_path):
    """Sort repeat labels CSV by composer, composition_id, and perf_id."""

    repeat_labels = pd.read_csv(out_csv_path)
    repeat_labels = repeat_labels.reset_index()
    repeat_labels.drop(columns=['score_path'], inplace=True) # dropping the last column, because reset index squeezed the index new column in
    repeat_labels.columns = ['composer', 'composition_id', 'perf_id', 'structure', 'score_path']

    repeat_labels.sort_values(
        by=['composer', 'composition_id', 'perf_id'],
        axis=0,
        inplace=True
    )
    repeat_labels.to_csv(out_csv_path, index=False)
    
    return

def infer_repeats(
    composition_id,
    out_path,
    suffix=False,
    dataset_meta=ATEPP_meta,
    dataset_path=ATEPP_transcriptions,
    max_workers=None,  # None = use all available CPUs
):
    """
    Main orchestration function: infer repeats for all compositions in parallel.
    
    Args:
        composition_id: List of composition IDs to process
        out_path: Directory to save output CSV
        dataset_meta: DataFrame of metadata
        dataset_path: Path to dataset
        max_workers: Number of parallel workers (None = CPU count)
    """
    print("====================================")
    
    # setup output
    if suffix:
        out_csv_path = out_path.replace('.csv', f'_estimated_repeats.csv')
    else:
        out_csv_path = out_path / f"cid={composition_id}_estimated_repeats.csv"
    if not os.path.exists(out_csv_path):
        with open(out_csv_path, "w") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["composer", "composition_id", "perf_id", "structure", "score_path"]
            )
    
    # gather all tasks (composition-performance pairs)
    tasks = []
    total_performances = 0
    
    print("Gathering tasks...")
    for cid in composition_id:
        composition_meta = dataset_meta[dataset_meta["composition_id"] == int(cid)]
        
        if len(composition_meta) == 0:
            print(f"Warning: No metadata found for cid {cid}")
            continue
        
        composer = composition_meta["composer"].unique()[0]
        perf_id_midi = composition_meta[["perf_id", "norm_midi_path"]].values
        
        try:
            score_path = composition_meta["norm_score_path"].dropna().values[0]
        except IndexError:
            print(f"Warning: No score path found for cid {cid}")
            continue
        
        # check score can be loaded (fail early before parallel processing)
        try:
            _ = pt.load_musicxml(dataset_path / score_path)
        except Exception as e:
            print(f"Error loading score for cid {cid} - {dataset_path / score_path}: {e}")
            continue
        
        # all checks passed: add task for each performance
        for pid, pmidi in perf_id_midi:
            # we can't load the score once and handle it to all processes that use this score
            # because processes don't share memory (same for repeat identifier object)
            tasks.append((composer, cid, pid, pmidi, score_path, dataset_path))
            
            total_performances += 1
    
    print(f"Processing {total_performances} performances across {len(composition_id)} compositions...")
    print(f"Using {max_workers or 'all available'} workers")
    
    # process in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_performance, task): task for task in tasks}
        
        with open(out_csv_path, "a") as f:
            writer = csv.writer(f)
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                task = futures[future]
                composer, cid, pid = task[0], task[1], task[2]
                
                try:
                    result = future.result()
                    composer, cid, pid, structure, score_path, error = result
                    composer = str(composer)
                    
                    if error:
                        print(f"\tError for cid {cid}, perf {pid}: {error}")
                    else:
                        print(f"\tCompleted cid {cid}, perf {pid}: {structure}")
                    
                    writer.writerow([composer, cid, pid, structure, score_path, error])
                    
                except Exception as e:
                    print(f"\tUnexpected error for cid {cid}, perf {pid}: {e}")
                    error_msg = f"ERROR: {type(e).__name__}: {str(e)}"
                    writer.writerow([composer, cid, pid, error_msg, task[4]]) # task[4] = pmidi

    if suffix: sort_labels(out_csv_path)
    print(f"Saved estimated repeats to {out_csv_path}")
    return out_csv_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Infer repeat structures from scores and performances"
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--composition_id",
        "-cid",
        nargs="+",
        help="One or more composition IDs to process",
    )
    input_group.add_argument(
        "--subset_meta_csv",
        "-csv",
        help="Path to CSV file containing subset of compositions to process",
    )

    parser.add_argument(
        "--out_path",
        "-o",
        default=output_path / DATESTAMP,
        type=Path,
        help="Saves to data/meta per default, is out_path is a csv, checks which repeats are already processed, and skips those already processed.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    if args.composition_id:
        infer_repeats(args.composition_id, args.out_path)
    elif args.subset_meta_csv:
        composition_ids = pd.read_csv(args.subset_meta_csv)["composition_id"].unique()
        infer_repeats(composition_ids, args.subset_meta_csv, suffix=True)
