import os
import pandas as pd
from pathlib import Path

project_dir = Path(__file__).resolve().parent

# input data paths
ATEPP_root = Path("/opt/datasets/fs/ATEPP/ATEPP-1.2_clean")
# ATEPP_root = path/to/your/copy/of/ATEPP-1.2

ATEPP_transcriptions = ATEPP_root / "ATEPP-1.2"
ATEPP_meta = pd.read_csv(ATEPP_root / "ATEPP-metadata-1.2_clean-compact.csv")

# logs and outputs
output_path = project_dir / "outputs"
runs_path = project_dir / "runs"
for d in [runs_path, output_path]:
    os.makedirs(d, exist_ok=True)