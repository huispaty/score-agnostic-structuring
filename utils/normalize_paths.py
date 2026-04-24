#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This module removes all non-Unicode characters from the ATEPP meta file and actual file paths to ensure all paths in the meta file are valid and accessible.​

Additionally, it merges each path in the ATEPP meta file with (estimated) quality class labels from
github.com/ilya16/midi-quality-assessment/blob/main/ATEPP_classified.csv.
"""

import sys

sys.path.append("..")

import os
import pandas as pd
import unicodedata

from pathlib import Path

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from paths import ATEPP_root, ATEPP_transcriptions

# GLOBALS
# ------------------------------------
DRY_RUN = True  # set to False if you want to do actual renaming in the file system


# HELPERS
# ------------------------------------
def normalize_str(s: str) -> str:

    if not isinstance(s, str):
        s = str(s)

    # normalize to NFKD → strip accents → recompose to NFC (optional)
    decomp = unicodedata.normalize("NFKD", s)
    stripped = "".join(c for c in decomp if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", stripped)


# PATHS
# ------------------------------------
# original ATEPP meta file
ATEPP_meta = pd.read_csv(Path(ATEPP_root / "ATEPP-metadata-1.2.csv"))
# cols: artist,artist_id,track,track_duration,composer,composition_id,score_path,midi_path,youtube_links,quality,perf_id,album,album_date,repetition

# estimated quality class labels
atepp_meta_class = pd.read_csv(
    "https://raw.githubusercontent.com/ilya16/midi-quality-assessment/refs/heads/main/ATEPP_classified.csv"
)
# cols: idx,performance,quality_label,score_prob,recorded_prob,high_quality_prob,low_quality_prob,corrupted_prob
assert atepp_meta_class.shape[0] == 11674


# DATA CLEANING
# ------------------------------------
# get all midi paths
midi_trp_paths = list(ATEPP_root.rglob("*.mid"))
midi_scr_paths = list(
    ATEPP_root.rglob("*.midi")
)  # scores, either mxl.midi or musicxml.midi
# for msp in midi_scr_paths:
#     if not (str(msp).endswith('mxl.midi') or str(msp).endswith('xml.midi')): print(msp)
assert len(midi_trp_paths) == 11699
assert len(midi_scr_paths) == 312

# count how many paths specified in meta are not valid
meta_midi_paths = ATEPP_meta["midi_path"].values
assert len(meta_midi_paths) == 11674
c = 0  # paths not found
for p in meta_midi_paths:
    if not os.path.exists(ATEPP_transcriptions / p):
        c += 1
        # print(ATEPP_transcriptions / p)
assert c == 1987  # (~17%)

# merge atepp meta file paths with those from pseudo-labelled atepp meta
assert set(ATEPP_meta["midi_path"]) == set(atepp_meta_class["performance"])
ATEPP_meta_quality = ATEPP_meta.merge(
    atepp_meta_class, left_on="midi_path", right_on="performance", how="inner"
)


# rename midi paths containing non-unicode charaters
changes = []
for path in midi_trp_paths:
    if path.is_file():

        fixed_path = normalize_str(path)
        if path != fixed_path:
            changes.append((path, fixed_path))
            # rename file
            # ensure parent dir exists
            if not DRY_RUN:
                Path(fixed_path).parent.mkdir(parents=True, exist_ok=True)
                path.rename(fixed_path)
            else:
                # print(f"{path} -> {fixed_path}")
                continue
if DRY_RUN:
    print(f"Would change {len(changes)} midi paths")
# delete empty directories via `find . -type d -empty -delete`

# also apply renaming in the meta file, validate all paths are now valid
ATEPP_meta_quality["norm_midi_path"] = ATEPP_meta["midi_path"].apply(
    lambda x: normalize_str(x)
)
c = 0
for p in ATEPP_meta_quality["norm_midi_path"].values:
    if not os.path.exists(ATEPP_transcriptions / p):
        c += 1
assert c == 0

# now the same for score paths (come as .musicxml, .mxl, .midi)
xml_scr_paths = list(ATEPP_root.rglob("*.musicxml"))
mxl_scr_paths = list(ATEPP_root.rglob("*.mxl"))
score_paths = xml_scr_paths + mxl_scr_paths + midi_scr_paths
len(score_paths)  # 631
ATEPP_meta["score_path"].nunique(
    dropna=False
)  # 320 # README mismatch between annotated scores and actual scores

changes = []
for path in score_paths:
    if path.is_file():

        fixed_path = normalize_str(path)
        if path != fixed_path:
            changes.append((path, fixed_path))
            if not DRY_RUN:
                Path(fixed_path).parent.mkdir(parents=True, exist_ok=True)
                path.rename(fixed_path)
            else:
                # print(f"{path} -> {fixed_path}")
                continue
if DRY_RUN:
    print(f"Would change {len(changes)} score paths")
# delete empty directories via `find . -type d -empty -delete`

# validate all paths are not valid
ATEPP_meta_quality["norm_score_path"] = ATEPP_meta["score_path"].apply(
    lambda x: normalize_str(x)
)
# check again on normalized score paths
c = 0  # paths not found
for p in ATEPP_meta_quality["norm_score_path"].dropna().values:
    if p == "nan":
        continue
    if not os.path.exists(ATEPP_transcriptions / p):
        # print(ATEPP_transcriptions / p)
        c += 1
assert c == 0

# finally, non-unicode charaters also in composer and track column
for c in ATEPP_meta_quality.columns:
    if c in ["composer", "track"]:
        ATEPP_meta_quality[c] = ATEPP_meta[c].apply(lambda x: normalize_str(x))

# DATA ADMIN
# ------------------------------------
# save two versions of ATEPP meta file with predicted quality labels: extended and compact
""" README
ATEPP_meta_quality.columns - extended

'artist', 'artist_id', 'track', 'track_duration', 'composer',
'composition_id', 'score_path', 'midi_path', 'youtube_links', 'quality',
'perf_id', 'album', 'album_date', 'repetition', 'idx', 'performance',
'quality_label', 'score_prob', 'recorded_prob', 'high_quality_prob',
'low_quality_prob', 'corrupted_prob', 'norm_midi_path' 'norm_score_path'

ATEPP_meta_quality.columns - compact
'composition_id', 'artist_id', 'track_duration', 'composer', 'track',
'score_path', 'midi_path', 'perf_id', 'quality',
'repetition', 
'idx'               # idx from classified quality labels meta
'quality_label', 'score_prob', 'recorded_prob', 'high_quality_prob',
'low_quality_prob', 'corrupted_prob', 'norm_midi_path' 'norm_score_path'
"""

# save extended meta data
if not Path(ATEPP_root / "ATEPP-metadata-1.2_clean-extended.csv").is_file():
    ATEPP_meta_quality.to_csv(
        Path(ATEPP_root / "ATEPP-metadata-1.2_clean-extended.csv"), index=False
    )
# save compact meta data
ATEPP_meta_quality_compact = ATEPP_meta_quality[
    [
        "composition_id",
        "artist_id",
        "perf_id",
        "track",
        "composer",
        "track_duration",
        "quality",
        "repetition",
        "idx",
        "quality_label",
        "score_prob",
        "recorded_prob",
        "high_quality_prob",
        "low_quality_prob",
        "corrupted_prob",
        "norm_midi_path",
        "norm_score_path",
    ]
]
if not Path(ATEPP_root / "ATEPP-metadata-1.2_clean-compact.csv").is_file():
    ATEPP_meta_quality_compact.to_csv(
        Path(ATEPP_root / "ATEPP-metadata-1.2_clean-compact.csv"), index=False
    )
