import sys

sys.path.append("..")
from pathlib import Path

import pandas as pd
import warnings

warnings.filterwarnings("ignore")

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from datetime import datetime

import argparse

TEST_PIECES_ID = [215, 1148, 1165, 1176, 1182, 1185, 1204, 1211, 1213, 1216, 1218]

def select_subset(selection_csv, debug=False, verbose=True, test_ids = TEST_PIECES_ID):

    repeat_labels = pd.read_csv(selection_csv.replace(".csv", "_estimated_repeats.csv"))
    csv_out_path = Path(selection_csv).parent / "atepp_subset_mec26.csv"

    ##################
    # Filtering
    ##################
    # filter out compositions that have no structural variation, contain errors, or have only a single transcription
    grouped = repeat_labels.groupby(["composer", "composition_id"])

    # find compositions with 'no_struct_var' in any row
    no_struct_var_mask = grouped["structure"].apply(
        lambda x: (x == "no_struct_var").any()
    )
    no_struct_var_ids = no_struct_var_mask[no_struct_var_mask].index

    # find compositions with 'ERROR' in structure column
    error_mask = grouped["structure"].apply(
        lambda x: x.str.contains("ERROR", na=False).any()
    )
    error_ids = error_mask[error_mask].index

    # find compositions with only 1 performance
    single_perf_mask = grouped["perf_id"].apply(lambda x: x.nunique() == 1)
    single_perf_ids = single_perf_mask[single_perf_mask].index

    # combine conditions: has no_struct_var OR single performance OR error
    exclude_ids = set(no_struct_var_ids) | set(single_perf_ids) | set(error_ids)

    # create subset dataframe
    subset = repeat_labels[
        ~repeat_labels.set_index(["composer", "composition_id"]).index.isin(exclude_ids)
    ][["composer", "composition_id", "perf_id", "structure", "score_path"]]

    # add train/test split labels
    subset.insert(0, "split", "train")
    subset.loc[subset["composition_id"].isin(test_ids), "split"] = "test"
    subset.to_csv(csv_out_path, index=False)

    if verbose:
        n_composers = subset["composer"].nunique()
        n_comps = subset["composition_id"].unique()
        n_trans = subset["perf_id"].nunique()
        print(
            f"Subset contains {len(n_comps)} compositions ({n_trans} transcriptions) by {n_composers} composers."
        )
        print("Split:")
        split_summary = subset.groupby("split").agg(
            pieces=("composition_id", "nunique"),
            transcriptions=("perf_id", "nunique")
        )
        for split, row in split_summary.iterrows():
            print(f"{split:<10} {row['pieces']} pieces ({row['transcriptions']} transcriptions)")

    ##################
    # Debug
    ##################
    if debug:
        # create separate df for compositions with same label (excluding no_struct_var)
        same_label_mask = grouped["structure"].apply(lambda x: x.nunique() == 1)
        same_label_ids = same_label_mask[same_label_mask].index
        same_label_only_ids = set(same_label_ids) - set(no_struct_var_ids)
        same_label_df = repeat_labels[
            repeat_labels.set_index(["composer", "composition_id"]).index.isin(
                same_label_only_ids
            )
        ][["composer", "composition_id", "perf_id", "structure", "score_path"]]
        same_label_csv = selection_csv.replace(".csv", "_identical_structure_labels-DEBUG.csv")
        same_label_df.to_csv(same_label_csv, index=False)

        # create separate df for compositions with error score
        error_score_df = repeat_labels[
            repeat_labels.set_index(["composer", "composition_id"]).index.isin(
                error_ids
            )
        ][["composer", "composition_id", "perf_id", "structure", "score_path"]]
        error_score_csv = selection_csv.replace(".csv", "_error_score-DEBUG.csv")
        error_score_df.to_csv(error_score_csv, index=False)

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Select ATEPP dataset subset used for MEC26",
        description="Filter transcriptions to include only those with valid scores and structural variation across performances.",
    )

    parser.add_argument(
        "--subset_meta_csv",
        "-csv",
        help="Path to CSV file containing subset of compositions to process",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Saves separate files for transcripitions with faulty scores, and those with identical repeat labels across all performances.",
    )

    args = parser.parse_args()
    select_subset(args.subset_meta_csv, args.debug)
