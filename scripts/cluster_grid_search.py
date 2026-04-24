#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a grid search on a list of pieces to find best clustering parameters, where
repeat structure estimates by [1_] are used as pseudo labels.

[1_] Peter, S., Hu, P., Widmer, G. (2025).
     How to Infer Repeat Structures in MIDI Performances.
     Music Encoding Conference Proceedings 2025.
     https://arxiv.org/pdf/2505.05055

Example use:
python ./scripts/cluster_grid_search.py -align ./data/mec26/train -pseudo ./data/mec26/meta/estimated_repeats.csv -o ./data/mec26/results
"""

import sys

sys.path.append("..")

import os
from pathlib import Path
from tqdm import tqdm
from itertools import product

import numpy as np
import pandas as pd

from mpteval.clustering.cluster import compute_pairwise_features, hierarchical_structural_clustering, plot_distance_matrix, plot_n_dist_matrices
from mpteval.clustering.cluster_eval import evaluate_clustering
from mpteval.clustering.cluster_gs import grid_search_clustering

import warnings
warnings.filterwarnings("ignore")

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import argparse

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from paths import *

import json
from datetime import datetime

def grid_search_clustering_multi_piece(
    piece_ids,  # ;ist of (align_df, true_groups, piece_id) tuples
    methods=["ward", "average", "complete", "single"],
    cost_weights=[0.3, 0.5, 0.7, 1, 2, 3],
    stretch_opt_weights=[0.3, 0.5, 0.7, 1, 2, 3],
    stretch_avg_weights=[0.3, 0.5, 0.7, 1, 2, 3],
    length_weights=[0.3, 0.5, 0.7, 1, 2, 3],
    dist_thresholds=[0.7],
    sort_results_by_metric: str = "v_measure",
    aggregation: str = "mean",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Test all combinations across multiple pieces and find best overall parameters.

    Parameters:
    ----------
        pieces: List of tuples (align_df, true_groups, piece_ids)
        aggregation: How to combine metrics across pieces
            - 'mean': Average performance (default)
            - 'median': Median performance (robust to outliers)
            - 'min': Worst-case performance (conservative)

    Returns:
        agg_res: DataFrame with aggregated metrics across all pieces
        piecewise_res: Dict mapping piece_ids -> per-piece results DataFrame
    """
    # generate combinations, filter out those were all weights are zero
    combinations = list(
        product(
            methods,
            cost_weights,
            stretch_opt_weights,
            stretch_avg_weights,
            length_weights,
            dist_thresholds,
        )
    )
    combinations = [
        combo
        for combo in combinations
        if not (combo[1] == 0 and combo[2] == 0 and combo[3] == 0 and combo[4] == 0)
    ]

    print(
        f"Testing {len(combinations)} parameter combinations across {len(piece_ids)} pieces..."
    )
    print(f"Total evaluations: {len(combinations) * len(piece_ids)}")

    # Store results per piece
    piecewise_res = {}
    all_results = []

    # progress bar for combinations
    for method, c_w, so_w, sa_w, l_w, t in tqdm(
        combinations, desc="Parameter combinations"
    ):
        combo_results = {
            "method": method,
            "cost_weight": c_w,
            "stretch_opt_weight": so_w,
            "stretch_avg_weight": sa_w,
            "length_weight": l_w,
            "dist_thresh": t,
        }

        piece_id_metrics = []

        # test on each piece_ids
        for align_df, true_groups, piece_id in piece_ids:
            try:
                # run clustering
                predicted_groups, _, _ = hierarchical_structural_clustering(
                    align_df,
                    method=method,
                    cost_weight=c_w,
                    stretch_opt_weight=so_w,
                    stretch_avg_weight=sa_w,
                    length_ratio_weight=l_w,
                    dist_thresh=t,
                )

                if len(predicted_groups) == 0:
                    continue

                # evaluate
                metrics = evaluate_clustering(true_groups, predicted_groups)

                if np.isnan(metrics["adjusted_rand_index"]):
                    continue

                # Store per-piece result
                piece_result = {
                    "composer_piece_id": piece_id,
                    "n_predicted_groups": len(predicted_groups),
                    **combo_results,
                    **metrics,
                }
                piece_id_metrics.append(metrics)

                # Add to detailed results
                if piece_id not in piecewise_res:
                    piecewise_res[piece_id] = []
                piecewise_res[piece_id].append(piece_result)

            except Exception as e:
                print(
                    f"  FAILED: {piece_id}, method={method}, "
                    f"weights=({c_w},{so_w},{sa_w},{l_w}), dist={t}: {e}"
                )
                continue

        # aggregate metrics across pieces
        if len(piece_id_metrics) > 0:
            agg_func = {"mean": np.mean, "median": np.median, "min": np.min}[
                aggregation
            ]

            for metric_name in piece_id_metrics[0].keys():
                values = [m[metric_name] for m in piece_id_metrics]
                combo_results[f"{metric_name}_{aggregation}"] = agg_func(values)
                combo_results[f"{metric_name}_std"] = np.std(values)
                combo_results[f"{metric_name}_min"] = np.min(values)
                combo_results[f"{metric_name}_max"] = np.max(values)
                # add count of perfect score for v_measure
                if metric_name == "v_measure":
                    combo_results["v_measure_perfect_count"] = sum(
                        v == 1.0 for v in values
                    )

            combo_results["n_pieces_succeeded"] = len(piece_id_metrics)
            all_results.append(combo_results)

    if len(all_results) == 0:
        raise ValueError("No successful clustering runs across any piece!")

    # convert results to df
    agg_res = pd.DataFrame(all_results)

    # sort by aggregated metric
    sort_column = f"{sort_results_by_metric}_{aggregation}"
    if sort_column in agg_res.columns:
        agg_res = agg_res.sort_values(sort_column, ascending=False)

    # convert detailed results to DataFrames
    piecewise_res = {
        name: pd.DataFrame(results) for name, results in piecewise_res.items()
    }

    print(f"\nGrid search complete!")
    print(f"Successful parameter combinations: {len(agg_res)}")
    print(f"\nTop 5 parameter combinations by {sort_column}:")
    print(
        agg_res.head()[
            [
                "method",
                "cost_weight",
                "stretch_opt_weight",
                "stretch_avg_weight",
                "length_weight",
                "dist_thresh",
                sort_column,
                f"{sort_results_by_metric}_std",
            ]
        ]
    )

    return agg_res, piecewise_res


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")

    # inputs
    parser.add_argument(
        "--alignments_dir",
        "-align",
        help="Path to pairwise alignments.",
    )
    parser.add_argument(
        "--pseudo_labels_csv",
        "-pseudo",
        help="csv file that contains the pseudo labels.",
    )

    # grid search params
    parser.add_argument(
        "--methods",
        "-cm",
        nargs="+",
        default=["complete", "ward", "average", "weighted"],
        help="Linkage methods to test.",
    )
    parser.add_argument(
        "--cost_weights",
        "-cw",
        nargs="+",
        type=float,
        default=[0, 0.3, 1, 2, 3],
        help="Cost weights to test.",
    )
    parser.add_argument(
        "--stretch_opt_weights",
        "-sow",
        nargs="+",
        type=float,
        default=[0, 0.3, 0.7, 1],
        help="Stretch optimality weights to test.",
    )
    parser.add_argument(
        "--stretch_avg_weights",
        "-saw",
        nargs="+",
        type=float,
        default=[0, 0.3, 0.7, 1],
        help="Stretch average weights to test.",
    )
    parser.add_argument(
        "--length_weights",
        "-lw",
        nargs="+",
        type=float,
        default=[0, 0.3, 0.7, 1],
        help="Length ratio weights to test.",
    )
    parser.add_argument(
        "--dist_thresholds",
        "-dt",
        nargs="+",
        type=float,
        default=[0.5, 0.7],
        help="Distance thresholds to test.",
    )
    parser.add_argument(
        "--sort_results_by_metric",
        default="homogeneity",
        help="Metric to sort final results by.",
    )
    parser.add_argument(
        "--aggregation",
        default="mean",
        choices=["mean", "median", "min"],
        help="How to aggregate metrics across pieces.",
    )

    parser.add_argument(
        "--out_path",
        "-o",
        help="Directory path at which the aggregated results, sorted by args.sort_results_by_metric are saved to",
    )

    args = parser.parse_args()

    # log configs
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_dir = Path(args.out_path) / f"{runs_path}/{Path(__file__).stem}"
    os.makedirs(run_dir, exist_ok=True)
    with open(run_dir / f"run_{timestamp}.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # setup gridsearch
    align_csvs = list(Path(args.alignments_dir).rglob("**/align.csv"))
    pseudo_labels = pd.read_csv(args.pseudo_labels_csv)

    # gather all pieces to run clustering on
    pieces = []  # (align_df, true_groups, piece_id) tuples
    cluster_path_lookup = {}  # composition_id -> cluster_res_path

    for align_csv in align_csvs:
        align_df = pd.read_csv(align_csv)
        composition_id = str(align_csv).split("/")[-3]

        comp_pseudo_labels = pseudo_labels[pseudo_labels['composition_id'] == int(composition_id)]
        # group into key:value dict
        est_groups = {}
        for group_idx, (structure, group_df) in enumerate(comp_pseudo_labels.groupby('structure')):
            est_groups[f'group_{group_idx}'] = group_df['perf_id'].tolist()
        # save pseudo labels in each piecewise dir
        cluster_res_path = str(align_csv).replace(
            "2_alignments/align.csv", "3_cluster"
        )
        os.makedirs(cluster_res_path, exist_ok=True)
        comp_pseudo_labels.to_csv(Path(cluster_res_path) / 'pseudo_repeat_labels.csv', index=False)

        cluster_path_lookup[composition_id] = cluster_res_path
        pieces.append((align_df, est_groups, composition_id))

    # run grid search
    agg_res, piecewise_res = grid_search_clustering_multi_piece(
        pieces,
        methods=args.methods,
        cost_weights=args.cost_weights,
        stretch_opt_weights=args.stretch_opt_weights,
        stretch_avg_weights=args.stretch_avg_weights,
        length_weights=args.length_weights,
        dist_thresholds=args.dist_thresholds,
        sort_results_by_metric=args.sort_results_by_metric,
        aggregation=args.aggregation,
    )
    # save results
    os.makedirs(args.out_path, exist_ok=True)
    results_save = pd.DataFrame(
        agg_res.values, columns=agg_res.columns, index=agg_res.index
    )
    results_save.to_csv(Path(args.out_path) / "agg_cluster_res.csv", index=False)
    print('Aggregate results saved to', Path(args.out_path) / "agg_cluster_res.csv")

    # save piecewise results to their respective directories
    for composition_id, piece_df in piecewise_res.items():
        cluster_path = cluster_path_lookup[composition_id]
        piece_df.to_csv(Path(cluster_path) / "grid_search_results.csv", index=False)

    best_params = agg_res.iloc[0]
    print(f"\nBest parameters:")
    print(f"  Method: {best_params['method']}")
    print(
        f"  Weights: cost={best_params['cost_weight']}, "
        f"stretch_opt={best_params['stretch_opt_weight']}, "
        f"stretch_avg={best_params['stretch_avg_weight']}, "
        f"length={best_params['length_weight']}"
    )
    print(f"  Distance threshold: {best_params['dist_thresh']}")

