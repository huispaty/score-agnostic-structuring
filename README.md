# Evaluation of Transcribed Datasets

In this repository we demonstrate how to evaluate large-scale transcribed solo piano performance datasets, for which we use the ATEPP dataset [_1] as an example.

# Dependencies
- python 3.14
- matplotlib 3.7.0

# How to run stuff

### 1. Data (subset) selection
Choose a subset of transcriptions in ATEPP based on either transcription count (number of transcriptions of piece) or composer name:

- `python scripts/select_subset_atepp.py -lb 5 -ub 10`
- `python scripts/select_subset_atepp.py -c Schumann Haydn Mozart Schubert Beethoven`

Saves a subset meta csv file of the subset (containing columns `[composer, composition_id, num_perfs, track, score_path]`) to `./outputs/<DATESTAMP>/atepp_subset_....csv`.
By default, the `-s` flag is true, which filters candidate compositions/transcriptions to only those with a paired score in the dataset.
Optionally, `--stats` flag can be passed along to return basic statistics for the ATEPP dataset. 

### 2. Get pseudo labels
Create reference repeat structure predictions via score-performane alignment and backtracking [_2] to use as pseudo labels.

- `python scripts/infer_repeats_score.py -ssm <subset_meta_path>`
- `python scripts/infer_repeats_score.py -cid <cid>`

When using `-ssm`: saves at the same location as the input (ssm) file with suffix `_estimated_repeats.csv`. When using `-cid`, saves labels for all transcriptions of a single composition to `./outputs/<DATESTAMP>/cid=<cid>_estimated_repeats.csv`.

### 3. Performance-to-performance alignments
Create alignments between all pairs of transcriptions of a given piece by a given composer via: `python ./scripts/pairwise_align.py -c <composer> -cid <cid> -o <out_path>`
Or alternatively, create such pairwise alignments of all pieces by a given composer via `python ./scripts/pairwise_align.py -c <composer> -o <out_path>`

In addition to the data arguments, the following arguments wrt to the "chordification" preprocessing and alignment can be provided (square brackets contain their default value):
    - `--ioi_threshold` [0.03] max time gap (seconds) between consecutive notes to be grouped into the same chord
    - `--max_threshold` [0.5]: max total time span (seconds) of a chord from its first to last note
    - `--alpha` [0.85]: weighting factor for the harmonic cost in the composte alignment cost
    - `--harmonic_metric` ["jaccard"]: the distance function to use for measuring harmonic/pitch similarity
    - `--pitch_feature` ["pc_set"]: the feature to use for pitches

All outputs are saved to `<out_path>/<DATESTAMP>/<composer-last-name>/<composition_id>/`, where preprocessed performance note arrays are saved to `1_performance_data` as <perf_id.npy>, and alignments (i.e., pairwise paths, distance matrices and accumulated cost matrices etc.) to `2_alignments` as `<perf_id1>_<perf_id2>_pred_align.npy` (in the case of the alignment path).

This script can also be run with a `--subset_meta_csv` (`-ssm`) argument to process a predefined subset of compositions (see above). In addition to columns `[composer, composition_id, num_perfs, track, score_path]`, a `split` column is expected here too.

<br>

# How to replicate stuff
## MEC 2026: Score-Agnostic Structure Clustering

1. **Select compositions** - Filter works by the five composers (requires a paired score and at least 2 transcriptions):
   ```
   python scripts/select_subset_atepp.py -c Schumann Haydn Mozart Schubert Beethoven -o ./outputs/mec26/meta
   ```

2. **Estimate repeat structure** - Run structural inference on the selected subset:
   ```
   python scripts/infer_repeats_score.py -csv ./outputs/mec26/meta/atepp_subset_c=Schumann_Haydn_Mozart_Schubert_Beethoven.csv -o ./outputs/mec26/meta
   ```
   Output is saved at the same path with the suffix `_estimated_repeats.csv`.

3. **Filter by structural variation and assign split** — Keep only compositions that allow structural variations across performances, and split into train/test sets. Note that this script expects a corresponding `_estimated_repeats.csv` file in the same directory as the input csv (generated in step 2):
   ```
   python scripts/prepare_mec26_dataset.py -csv ./outputs/mec26/meta/atepp_subset_c=Schumann_Haydn_Mozart_Schubert_Beethoven.csv
   ```
   Output is saved in the same directory as `atepp_subset_mec26.csv`.


4. **Compute pairwise alignments** — Generate performance-to-performance alignments for the final subset:
   ```
   python scripts/pairwise_align.py -csv ./outputs/mec26/meta/atepp_subset_mec26.csv -o ./outputs/mec26
   ```
   This creates a `<split>/` folder structure under the output path. For each composition, preprocessed (chordified) performances are saved as .npy arrays in `1_performance_data/`, and pairwise alignments between all transcriptions of the same piece are saved in `2_alignments/`. Both directories are created under `<split>/<composer>/<composition_id>/`.

5. **Run grid search on train set** to find best clustering params:
   ```
   python ./scripts/cluster_grid_search.py -align ./outputs/mec26/train -pseudo ./outputs/mec26/meta/estimated_repeats.csv -o ./outputs/mec26/results
   ```
   For each piece in the train set, run hierarchical clustering over all combinations of clustering methods, feature weights, and distance thresholds, evaluated against pseudo labels. Aggregates metrics across pieces to find the parameter combination that performs best overall.

6. **Evaluate params on test set**
   ```
   python ./scripts/cluster.py -align ./outputs/mec26/test -pseudo ./outputs/mec26/meta/estimated_repeats.csv -o ./outputs/mec26/results -true ./outputs/mec26/corrected_labels
   ```
   Run clustering with one set of parameters (e.g. the one found in the grid search) across all pieces. If `-true` argument is not given, cluster parameters are evaluated against the pseudo labels. Saves per-piece predicted groupings and aggregate metrics.


### Misc
- Cleaning: `python ./utils/normalize_paths.py` cleans the paths in the meta file and file system from non-unicode characters
- Debugging (listening tests): select transcriptions from the ATEPP dataset based on their composition or performance id and save them in out_path `python ./utils/get_file_via_id.py -id 1165 1176 -t p -o out_path`

<br>

# Citation
If you find our clustering method useful, please cite:
```
@inproceedings{hu2026score-agnostic,
  title     = {{Score-Agnostic Structure Analysis in Large-Scale Performance Datasets}},
  author    = {Hu, Patricia and Peter, Silvan and Widmer, Gerhard},
  booktitle = {{Proceedings of the Music Encoding Conference}},
  year      = {2026},
  address   = {Tokyo, Japan},
}
```

# References
[_1] Zhang, H., Tang, J., Rafee, S., Dixon, S., Fazekas, G., & Wiggins, G. (2023). ATEPP: A dataset of automatically transcribed expressive piano performance. Proceedings of the International Society for Music Information Retrieval Conference 2022.

[_2] Peter, S., Hu, P., & Widmer, G. (2025). How to Infer Repeat Structures in MIDI Performances. Proceedings of the Music Encoding Conference 2025, https://arxiv.org/pdf/2505.05055.