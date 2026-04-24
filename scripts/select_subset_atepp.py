#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module selects compositions from the ATEPP dataset based on either:
1. their number of transcriptions (via lower and upper bound)
2. composer name(s)

Example use:
python scripts/select_subset_atepp.py -lb 5 -ub 10
python scripts/select_subset_atepp.py -c Schumann Haydn Mozart Schubert Beethoven
python scripts/select_subset_atepp.py --composer "Beethoven"

Saves selected compositions to meta/ directory with appropriate filename.
"""

import sys

sys.path.append("..")

import os
import pandas as pd
import argparse

from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from paths import ATEPP_meta, output_path

from datetime import datetime
DATESTAMP = datetime.now().strftime("%y%m%d")

def seconds_to_hours_minutes(seconds):
    total_minutes = round(seconds / 60)
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours} hours, {minutes} minutes"


def get_atepp_stats(ATEPP_meta, type=None):

    num_compositions = ATEPP_meta["composition_id"].nunique()
    num_composers = ATEPP_meta["composer"].nunique()
    num_perfs = ATEPP_meta["perf_id"].nunique()
    num_performers = ATEPP_meta["artist_id"].nunique()
    total_dur = ATEPP_meta["track_duration"].sum()

    if type is None:
        print("++++++++++++++++++++++++++++++++++++")
        print("ATEPP DATASET STATS")
        print(f"{num_compositions:5d}", "unique compositions")
        print(f"{num_composers:5d}", "unique composers")
        print(f"{num_perfs:5d}", "unique performances")
        print(f"{num_performers:5d}", "unique performers")
        print("TOTAL DURATION:", seconds_to_hours_minutes(total_dur))
        print()

        ATEPP_native_quality_labels = ATEPP_meta["quality"].notna().sum()
        ATEPP_native_repetition_labels = ATEPP_meta["repetition"].notna().sum()
        print(
            f"{ATEPP_native_quality_labels:5d} ({ATEPP_native_quality_labels/num_perfs*100:.2f}%)",
            "quality labels",
        )
        print(
            f"{ATEPP_native_repetition_labels:5d} ({ATEPP_native_repetition_labels/num_perfs*100:.2f}%)",
            "repetition labels",
        )
        print("++++++++++++++++++++++++++++++++++++")

    if type == "num_perfs":
        return num_perfs
    elif type == "duration":
        return total_dur
    else:
        return


def create_composition_summary(filtered_data, out_path, filename):
    """
    Create a summary CSV of compositions from filtered data.
    
    Parameters:
    -----------
    filtered_data : DataFrame
        Filtered ATEPP metadata
    out_path : Path
        Output directory path
    filename : str
        Output filename
        
    Returns:
    --------
    Path
        Path to the created CSV file
    """
    composer_id_track_csv = out_path / filename
    composer_id_track_csv.parent.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(composer_id_track_csv):
        composer_id_track_df = pd.DataFrame(
            columns=["composer", "composition_id", "num_perfs", "track", "score"]
        )
        
        filtered_data.sort_values(by='composition_id', inplace=True)
        for i, (group_name, group_df) in enumerate(
            filtered_data.groupby("composition_id")
        ):

            if group_name == 1194: print(group_df['perf_id'].nunique(), group_df['perf_id'].shape)
            # normalize composer and track string
            composer = group_df["composer"].unique()[0].split()[-1]
            track = group_df["track"].value_counts().index[0]
            num_perfs = group_df['perf_id'].nunique()
            
            composer_id_track_df.at[i, "composer"] = composer
            composer_id_track_df.at[i, "score"] = group_df["norm_score_path"].unique()[0]
            composer_id_track_df.at[i, "composition_id"] = group_name
            composer_id_track_df.at[i, "num_perfs"] = num_perfs
            composer_id_track_df.at[i, "track"] = track
        
        # sort by composer, then id, then track
        composer_id_track_df.sort_values(
            by=["composer", "composition_id"], ascending=[True, True], inplace=True
        )
        composer_id_track_df.to_csv(composer_id_track_csv, index=False)
    
    return composer_id_track_csv


def print_selection_stats(filtered_data, ATEPP_meta, description):
    """
    Print statistics about the filtered selection.
    
    Parameters:
    -----------
    filtered_data : DataFrame
        Filtered ATEPP metadata
    ATEPP_meta : DataFrame
        Full ATEPP metadata
    description : str
        Description of the selection criteria
    """
    num_compositions = len(filtered_data.groupby("composition_id"))
    print(f"Selected {num_compositions} compositions {description} - totaling:")
    
    num_filtered_perfs = filtered_data.groupby("composition_id").size().sum()
    num_total_perfs = get_atepp_stats(ATEPP_meta, type="num_perfs")
    print(
        f"{num_filtered_perfs} transcriptions ({num_filtered_perfs/num_total_perfs*100:.2f}%)"
    )
    
    filtered_duration = filtered_data["track_duration"].sum()
    total_duration = get_atepp_stats(ATEPP_meta, type="duration")
    print(
        f"{seconds_to_hours_minutes(filtered_duration)} ({filtered_duration/total_duration*100:.2f}%)"
    )


def select_compositions_by_transcription_count(lower_bound, upper_bound, out_path, ATEPP_meta, contains_score):
    """
    Filter compositions by number of transcriptions.
    
    Parameters:
    -----------
    lower_bound : int
        Minimum number of transcriptions
    upper_bound : int
        Maximum number of transcriptions
    out_path : Path
        Output directory path
    ATEPP_meta : DataFrame
        ATEPP metadata dataframe
    """
    # filter by count
    filtered_data = ATEPP_meta.groupby("composition_id").filter(
        lambda x: lower_bound <= len(x) <= upper_bound
    )
    if contains_score:
        has_score = filtered_data.groupby('composition_id')['norm_score_path'].transform(lambda x: x.notna().any())
        filtered_data = filtered_data[has_score]
    
    # create meta csv to save subset collection to
    filename = f"atepp_subset_trsc_lb={lower_bound}_ub={upper_bound}.csv"
    output_path = create_composition_summary(filtered_data, out_path, filename)
    
    # stats
    description = f"with {lower_bound}-{upper_bound} transcribed performances"
    print_selection_stats(filtered_data, ATEPP_meta, description)
    
    print("Saved selection to:", output_path)
    return


def select_compositions_by_composer(composer_names, out_path, ATEPP_meta, contains_score):
    """
    Filter compositions by composer name(s).
    
    Parameters:
    -----------
    composer_names : list of str
        List of composer names to filter by
    out_path : Path
        Output directory path
    ATEPP_meta : DataFrame
        ATEPP metadata dataframe
    """

    if isinstance(composer_names, str):
        composer_names = [composer_names]
    
    # filter by name
    filtered_data = ATEPP_meta[
        ATEPP_meta["composer"].apply(lambda x: x.split()[-1] in composer_names)
    ]
    # filter to compositions that contain more than 1 transcriptions
    filtered_data = filtered_data.groupby("composition_id").filter(
        lambda x: len(x) > 1
    )
    if contains_score:
        has_score = filtered_data.groupby('composition_id')['norm_score_path'].transform(lambda x: x.notna().any())
        filtered_data = filtered_data[has_score]
    
    # create meta csv to save subset collection to
    composers_str = "_".join(composer_names)
    filename = f"atepp_subset_c={composers_str}.csv"
    output_path = create_composition_summary(filtered_data, out_path, filename)
    
    # stats
    description = f"by {', '.join(composer_names)}"
    print_selection_stats(filtered_data, ATEPP_meta, description)
    
    print("Saved selection to:", output_path)
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Select compositions from ATEPP dataset",
        description="Select compositions from the ATEPP dataset, either via their transcription count "
        "(via lower and upper bound) or by composer name(s). "
        "Saves the selection as a subset meta (composer,composition_id,num_perfs,track) to the output path.",
    )

    parser.add_argument(
        "--lower_bound", "-lb", 
        type=int,
        help="Lower bound for transcription count (use with --upper_bound)"
    )
    parser.add_argument(
        "--upper_bound", "-ub", 
        type=int,
        help="Upper bound for transcription count (use with --lower_bound)"
    )
    parser.add_argument(
        "--composer", "-c",
        nargs='+',
        type=str,
        help="Composer last name(s) to filter by (e.g., 'Bach' or 'Bach Mozart Beethoven')"
    )
    parser.add_argument(
        "--paired_score", "-s",
        default=True,
        help="Whether to select only compositions for which the score is available, defaults to True"
    )
    parser.add_argument(
        "--out_path", "-o", 
        default= output_path / f'{DATESTAMP}', 
        type=Path,
        help="Output directory path (default: data/meta)"
    )
    parser.add_argument(
        "--stats", 
        action="store_true",
        help="Print general ATEPP dataset statistics"
    )

    args = parser.parse_args()

    if args.stats:
        get_atepp_stats(ATEPP_meta)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    # validate arguments
    has_transcription_args = args.lower_bound is not None and args.upper_bound is not None
    has_composer_args = args.composer is not None

    if not has_transcription_args and not has_composer_args and not args.stats:
        parser.error("Must provide either --lower_bound and --upper_bound, or --composer")
    
    if has_transcription_args and has_composer_args:
        parser.error("Cannot use both transcription count and composer filters simultaneously.")

    if has_transcription_args:
        select_compositions_by_transcription_count(
            args.lower_bound, 
            args.upper_bound, 
            args.out_path, 
            ATEPP_meta,
            args.paired_score
        )
    elif has_composer_args:
        select_compositions_by_composer(
            args.composer,
            args.out_path,
            ATEPP_meta,
            args.paired_score
        )