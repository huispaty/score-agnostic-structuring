#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Create pairwise alignments between all transcriptions of the same piece,
then cluster those transcriptions into groups based on their alignment path, 
stretch and sequence length dissimilarity cost.

Example usage:
python ./scripts/align_and_cluster.py

Saves clustering results (with default parameters optimized on pseudo labels) to:
outputs/csv/atepp/cluster_results.csv

If run with flag --use_corrected_labels, saves clustering results to:
outputs/csv/atepp/cluster_results_post_correction.csv
"""

import sys

sys.path.append("..")
import os
from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import shutil
import csv
import json

import numpy as np
import pandas as pd

import partitura as pt

from mpteval.preprocess import chordify_perf_note_array
from mpteval.dtw_align import process_pair_wrapper
from mpteval.dtw_dist import composite_cost

from mpteval.cluster import (
    compute_pairwise_features,
    hierarchical_structural_clustering,
    plot_distance_matrix,
)
from mpteval.cluster_eval import evaluate_clustering

from mpteval.utils import MusicEncoder, group_by_structure

from tqdm import tqdm
from itertools import combinations
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed

import argparse
from paths import ATEPP_transcriptions, ATEPP_meta, data_path, out_csvs


def main(
    data_dir,
    results_csv_path,
    cluster_method,
    cost_weight,
    stretch_opt_weight,
    stretch_avg_weight,
    length_ratio_weight,
    distance_threshold,
    repeats_df,
    use_corrected_labels=False,
):

    cluster_results_csv = results_csv_path
    cluster_results_cols = [
        "composer",
        "composition_id",
        "n_perfs",
        "n_unfoldings",
        "homogeneity",
        "completeness",
        "v_measure",
        "score_path",
    ]

    if not os.path.exists(cluster_results_csv):
        with open(cluster_results_csv, "w+") as f:
            writer = csv.writer(f)
            writer.writerow(cluster_results_cols)

    for id, composition in tqdm(repeats_df.groupby("composition_id")):

        if composition.shape[0] == 1:
            # filters out works for which there is only 1 performance
            # composition ids 1064 and 1120
            continue

        else:

            # create output dir
            composer = composition["composer"].values[0]
            # composer_id_dir = data_dir / composer / f"{composer}_{id}"
            composer_id_dir = data_dir / composer / f"{composer}_{id}-correct"

            # TODO FIX THIS SCRIPT
            if not os.path.exists(composer_id_dir):
                break
    
            repeats_composition_id = repeats_df[repeats_df["composition_id"] == id]
            score_path = repeats_composition_id["score_path"].values[0]
            n_perfs = repeats_composition_id["perf_id"].nunique()
            print(
                f"{repeats_composition_id['structure'].nunique()} structural unfoldings for {composer_id_dir}"
            )
            print("score:", score_path)
            print("n_perfs:", n_perfs)

            os.makedirs(composer_id_dir, exist_ok=True)

            transcription_ids = composition["perf_id"].values
            pairs = list(combinations(transcription_ids, 2))

            # ------------------------------------
            # out dir 0: transcriptions and score (raw data)
            out_dir_0 = composer_id_dir / "0_transcriptions"
            if not os.path.exists(out_dir_0):
                os.makedirs(out_dir_0, exist_ok=True)
                for pid in transcription_ids:
                    padded_id = str(pid).zfill(5)
                    midi_path = (
                        ATEPP_transcriptions
                        / ATEPP_meta[ATEPP_meta["perf_id"] == padded_id][
                            "norm_midi_path"
                        ].values[0]
                    )
                    score_path = ATEPP_transcriptions / score_path
                    score_destination_path = out_dir_0 / "score.musicxml"
                    shutil.copy(score_path, score_destination_path)
                    destination_path = out_dir_0 / f"{pid}.mid"
                    if not os.path.exists(destination_path):
                        shutil.copy(midi_path, destination_path)

            # ------------------------------------
            # out dir 1: performance data (note arrays and pid to chord list dict)
            out_dir_1 = composer_id_dir / "1_performance_data"
            os.makedirs(out_dir_1, exist_ok=True)

            pid_to_chords_dict_json = f"{out_dir_1}/pid_to_chords.json"
            if os.path.exists(pid_to_chords_dict_json):
                with open(pid_to_chords_dict_json, "r", encoding="utf-8") as f:
                    pid_to_chords_dict = json.load(f)

            else:
                pid_to_chords_dict = dict()
                for pid in transcription_ids:
                    padded_id = str(pid).zfill(5)
                    midi_path = (
                        ATEPP_transcriptions
                        / ATEPP_meta[ATEPP_meta["perf_id"] == padded_id][
                            "norm_midi_path"
                        ].values[0]
                    )
                    p1 = (
                        pt.load_performance_midi(midi_path)
                        .performedparts[0]
                        .note_array()
                    )
                    # save performance note array
                    np.save(f"{out_dir_1}/{pid}.npy", p1)
                    # add chordified list to dict
                    p1c = chordify_perf_note_array(
                        p1,
                        ioi_threshold=0.03,
                        max_threshold=0.5,
                        return_list_of_dicts=True,
                    )
                    pid_to_chords_dict[str(pid)] = p1c

                with open(pid_to_chords_dict_json, "w+", encoding="utf-8") as f:
                    json.dump(pid_to_chords_dict, f, cls=MusicEncoder, indent=2)

            # ------------------------------------
            # out dir 2: alignments
            out_dir_2 = composer_id_dir / "2_alignments"
            os.makedirs(out_dir_2, exist_ok=True)

            align_out_csv = out_dir_2 / "align.csv"

            if os.path.exists(align_out_csv):
                align_df = pd.read_csv(align_out_csv)
            else:
                dist_metric = partial(
                    composite_cost,
                    alpha=0.85,
                    harmonic_metric="jaccard",
                    pitch_feature="pc_set",
                )

                process_func = partial(
                    process_pair_wrapper,
                    pid_to_chords_dict=pid_to_chords_dict,
                    dist_metric=dist_metric,
                    out_dir=out_dir_2,
                    directional_weights=np.array([1, 2, 1]),
                )

                results = []

                with ProcessPoolExecutor(max_workers=4) as executor:
                    futures = {
                        executor.submit(process_func, pair): pair for pair in pairs
                    }

                    with tqdm(total=len(pairs), desc="Current pairs progress") as pbar:
                        for future in as_completed(futures):
                            try:
                                result = future.result()
                                results.append(result)
                            except Exception as e:
                                pair = futures[future]
                                print(f"Error processing pair {pair}: {e}")
                            finally:
                                pbar.update(1)

                with open(
                    align_out_csv, mode="w+", newline="", encoding="utf-8"
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow(
                        ["p1", "p2", "cost", "align_len", "p1_len", "p2_len"]
                    )
                    writer.writerows(results)

                align_df = pd.read_csv(align_out_csv)

            # ------------------------------------
            # out dir 3: clusters
            out_dir_3 = composer_id_dir / "3_cluster"
            os.makedirs(out_dir_3, exist_ok=True)

            # create labels
            true_groups = group_by_structure(composition, structure_column="structure")
            true_groups_json = out_dir_3 / "true_groups.json"
            with open(true_groups_json, "w+", encoding="utf-8") as f:
                json.dump(true_groups, f, indent=2)
            if use_corrected_labels:
                correct_labels = out_dir_3 / "repeat_labels_correct.csv"
                if os.path.exists(correct_labels):
                    print("Using corrected labels")
                    true_groups = group_by_structure(pd.read_csv(correct_labels))
                    true_groups_json = out_dir_3 / "true_groups_correct.json"
                    with open(true_groups_json, "w+", encoding="utf-8") as f:
                        json.dump(true_groups, f, indent=2)
            # also save composition structure labels to csv
            composition.to_csv(out_dir_3 / "repeat_labels.csv", index=False)

            # do clustering with given params
            cluster_fig_out_path = (
                out_dir_3
                / f"dendrogram_{cluster_method}_{cost_weight}_{stretch_opt_weight}_{stretch_avg_weight}_{length_ratio_weight}.pdf"
            )
            pred_json_out_path = (
                out_dir_3
                / f"pred_groups_{cluster_method}_{cost_weight}_{stretch_opt_weight}_{stretch_avg_weight}_{length_ratio_weight}.json"
            )
            dist_mat_out_path = (
                out_dir_3
                / f"dist_mat_{cluster_method}_{cost_weight}_{stretch_opt_weight}_{stretch_avg_weight}_{length_ratio_weight}.pdf"
            )

            if use_corrected_labels:
                cluster_fig_out_path = str(cluster_fig_out_path).replace(
                    ".pdf", ".post_correction.pdf"
                )
                pred_json_out_path = str(pred_json_out_path).replace(
                    ".json", ".post_correction.json"
                )
                dist_mat_out_path = str(dist_mat_out_path).replace(
                    ".pdf", ".post_correction.pdf"
                )

            pred_groups, _, dist_matrix = (
                hierarchical_structural_clustering(
                    align_df,
                    method=cluster_method,
                    cost_weight=cost_weight,
                    stretch_opt_weight=stretch_opt_weight,
                    stretch_avg_weight=stretch_avg_weight,
                    length_ratio_weight=length_ratio_weight,
                    dist_thresh=distance_threshold,
                    plot=cluster_fig_out_path,
                )
            )

            with open(pred_json_out_path, "w+", encoding="utf-8") as f:
                json.dump(pred_groups, f, indent=2)
            items, cost_mat, stretch_opt_mat, stretch_avg_mat, length_ratio_mat = (
                compute_pairwise_features(align_df)
            )
            plot_distance_matrix(
                items,
                dist_matrix,
                groups=pred_groups,
                out=dist_mat_out_path,
                title="distance",
            )

            metrics = evaluate_clustering(true_groups, pred_groups)

            results_row = [
                composer,
                id,
                n_perfs,
                repeats_composition_id["structure"].nunique(),
                metrics["homogeneity"],
                metrics["completeness"],
                metrics["v_measure"],
                score_path,
            ]
            with open(cluster_results_csv, "a") as f:
                writer = csv.writer(f)
                writer.writerow(results_row)

    print("Clustering results saved to:", cluster_results_csv)
    print("------------------------------------")
    cluster_results_df = pd.read_csv(cluster_results_csv)
    print(
        f'Mean homogeneity, completeness, v-measure over {cluster_results_df["composer"].nunique()} compositions:'
    )
    print(f'Homogeneity: {cluster_results_df["homogeneity"].mean():.3f}')
    print(f'Completeness: {cluster_results_df["completeness"].mean():.3f}')
    print(f'V-measure: {cluster_results_df["v_measure"].mean():.3f}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    # i/o
    # where to save computed data to
    parser.add_argument("--data_dir", type=str, default=data_path / "atepp")
    # where to save cluster results to
    parser.add_argument(
        "--out_csv",
        "-ocsv",
        type=str,
        default=out_csvs / "atepp" / "cluster_results.csv",
    )

    # cluster params
    # parameters according to grid search on pseudo labels
    parser.add_argument("--cluster_method", "-cm", type=str, default="ward")
    parser.add_argument("--cost_weight", "-cw", type=float, default=2.0)
    parser.add_argument("--stretch_opt_weight", "-sow", type=float, default=0.7)
    parser.add_argument("--stretch_avg_weight", "-saw", type=float, default=0.3)
    parser.add_argument("--length_ratio_weight", "-lrw", type=float, default=0.0)
    parser.add_argument("--distance_threshold", "-dt", type=float, default=0.7)
    parser.add_argument(
        "--repeat_pseudo_labels",
        type=str,
        default=pd.read_csv(out_csvs / "atepp" / "estimated_repeats_mec26.csv"),
    )

    # optionally, run on correct labels
    parser.add_argument("--use_corrected_labels", action="store_true")

    args = parser.parse_args()

    if args.use_corrected_labels:
        print("Using corrected labels for clustering evaluation.")
        args.out_csv = str(args.out_csv).replace(".csv", "_post_correction.csv")

    main(
        data_dir=args.data_dir,
        results_csv_path=args.out_csv,
        cluster_method=args.cluster_method,
        cost_weight=args.cost_weight,
        stretch_opt_weight=args.stretch_opt_weight,
        stretch_avg_weight=args.stretch_avg_weight,
        length_ratio_weight=args.length_ratio_weight,
        distance_threshold=args.distance_threshold,
        repeats_df=args.repeat_pseudo_labels,
        use_corrected_labels=args.use_corrected_labels,
    )
