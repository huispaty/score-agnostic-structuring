#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run clustering with a fixed set of parameters on a list of pieces, evaluated
against pseudo from repeat structure estimation or manually corrected true labels.

Example use:
python ./scripts/cluster.py -align ./data/mec26/train -pseudo ./data/mec26/meta/estimated_repeats.csv -o ./data/mec26/results
"""

import sys

sys.path.append("..")

import os
from pathlib import Path

import numpy as np
import pandas as pd
import argparse

from mpteval.clustering.cluster import hierarchical_structural_clustering
from mpteval.clustering.cluster_eval import evaluate_clustering

import warnings

warnings.filterwarnings("ignore")

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from paths import *

import json
from datetime import datetime


def run_clustering_multi_piece(
    pieces,  # list of (align_df, true_groups, piece_id) tuples
    method: str = "complete",
    cost_weight: float = 2.0,
    stretch_opt_weight: float = 0.7,
    stretch_avg_weight: float = 0.3,
    length_weight: float = 0.0,
    dist_thresh: float = 0.7,
    sort_metric="homogeneity",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Run clustering with fixed parameters across multiple pieces.

    Parameters
    ----------
    pieces : list of (align_df, true_groups, piece_id) tuples
    method : linkage method
    cost_weight, stretch_opt_weight, stretch_avg_weight, length_weight : feature weights
    dist_thresh : distance threshold for clustering

    Returns
    -------
    results_df : DataFrame with per-piece clustering metrics
    predicted_groups_dict : dict mapping piece_id -> predicted_groups
    """
    results = []
    predicted_groups_dict = {}

    for align_df, true_groups, piece_id in pieces:
        try:
            predicted_groups, _, _ = hierarchical_structural_clustering(
                align_df,
                method=method,
                cost_weight=cost_weight,
                stretch_opt_weight=stretch_opt_weight,
                stretch_avg_weight=stretch_avg_weight,
                length_ratio_weight=length_weight,
                dist_thresh=dist_thresh,
            )

            if len(predicted_groups) == 0:
                print(f"  Skipping {piece_id}: no groups predicted")
                continue

            metrics = evaluate_clustering(true_groups, predicted_groups)

            if np.isnan(metrics["adjusted_rand_index"]):
                print(f"  Skipping {piece_id}: NaN ARI")
                continue

            results.append(
                {
                    "composition_id": piece_id,
                    "n_true_groups": len(true_groups),
                    "n_predicted_groups": len(predicted_groups),
                    **metrics,
                }
            )
            predicted_groups_dict[piece_id] = predicted_groups

        except Exception as e:
            print(f"  FAILED: {piece_id}: {e}")
            continue

    results_df = pd.DataFrame(results)
    print(f"Mean {sort_metric}: {results_df[sort_metric].mean()*100:.2f}")

    return results_df, predicted_groups_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run clustering with fixed parameters."
    )

    parser.add_argument(
        "--alignments_dir", 
        "-align", 
        help="Path to pairwise alignments."
    )
    parser.add_argument(
        "--pseudo_labels_csv", 
        "-pseudo", 
        help="CSV file with pseudo labels."
    )
    parser.add_argument(
        "--out_path", 
        "-o", 
        help="Output directory for results."
    )

    # clustering parameters
    parser.add_argument("--method", "-cm", default="ward", help="Linkage method.")
    parser.add_argument("--cost_weight", "-cw", type=float, default=2.0)
    parser.add_argument("--stretch_opt_weight", "-sow", type=float, default=0.7)
    parser.add_argument("--stretch_avg_weight", "-saw", type=float, default=0.3)
    parser.add_argument("--length_weight", "-lw", type=float, default=0.0)
    parser.add_argument("--dist_thresh", "-dt", type=float, default=0.7)

    # eval
    parser.add_argument(
        "--sort_results_by_metric",
        "-m",
        default="homogeneity",
        help="Metric to sort final results by.",
    )

    # use true (manually corrected) labels
    parser.add_argument(
        "--true_labels_csv",
        "-true",
        default=None,
        help="Path to true (manually inferred) labels CSVs for evaluation. If not provided, pseudo labels are used.",
    )

    args = parser.parse_args()

    # log configs
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = Path(args.out_path) / f"{runs_path}/{Path(__file__).stem}"
    os.makedirs(run_dir, exist_ok=True)
    with open(run_dir / f"run_{timestamp}.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    align_csvs = list(Path(args.alignments_dir).rglob("**/align.csv"))

    # load labels upfront
    if args.true_labels_csv:
        true_label_files = {
            int(p.stem.split("_")[-1]): p
            for p in Path(args.true_labels_csv).glob("*.csv")
        }
    else:
        labels = pd.read_csv(args.pseudo_labels_csv)

    # gather pieces
    pieces = []
    cluster_path_lookup = {}

    for align_csv in align_csvs:
        align_df = pd.read_csv(align_csv)
        composition_id = str(align_csv).split("/")[-3]
        cluster_res_path = str(align_csv).replace("2_alignments/align.csv", "3_cluster")
        os.makedirs(cluster_res_path, exist_ok=True)

        if args.true_labels_csv:
            comp_labels = pd.read_csv(true_label_files[int(composition_id)])
            structure_col = "structure_correct"
            save_name = "correct_repeat_labels.csv"
        else:
            comp_labels = labels[labels["composition_id"] == int(composition_id)]
            structure_col = "structure"
            save_name = "pseudo_repeat_labels.csv"

        est_groups = {
            f"group_{i}": df["perf_id"].tolist()
            for i, (structure, df) in enumerate(comp_labels.groupby(structure_col))
        }
        comp_labels.to_csv(Path(cluster_res_path) / save_name, index=False)

        cluster_path_lookup[composition_id] = cluster_res_path
        pieces.append((align_df, est_groups, composition_id))

    # run clustering
    results_df, predicted_groups_dict = run_clustering_multi_piece(
        pieces,
        method=args.method,
        cost_weight=args.cost_weight,
        stretch_opt_weight=args.stretch_opt_weight,
        stretch_avg_weight=args.stretch_avg_weight,
        length_weight=args.length_weight,
        dist_thresh=args.dist_thresh,
        sort_metric=args.sort_results_by_metric,
    )

    # save aggregate results
    os.makedirs(args.out_path, exist_ok=True)
    if not args.true_labels_csv:
        filename = "cluster_results_pseudo_labels.csv"
    else:
        filename = "cluster_results_true_labels.csv"
    results_df.to_csv(Path(args.out_path) / filename, index=False)
    print(f"Results saved to {Path(args.out_path) / filename}")

    # save piecewise predicted groups
    for composition_id, predicted_groups in predicted_groups_dict.items():
        cluster_path = cluster_path_lookup[composition_id]
        pred_df = pd.DataFrame(
            [
                {"structure": k, "perf_id": pid}
                for k, pids in predicted_groups.items()
                for pid in pids
            ]
        )
        pred_df.to_csv(Path(cluster_path) / "predicted_groups.csv", index=False)
