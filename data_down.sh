#!/usr/bin/env bash
# =============================================================================
# CLEAN AASIST3 DATASET DOWNLOADER (NO BROKEN LINKS)
# Total ≈ 80 GB
# =============================================================================

set -e

BASE_DIR="${1:-/data/additional/DATASETS/audio}"
mkdir -p "$BASE_DIR"

# =============================================================================
# 1. ASVspoof 2019 LA (~8GB)
# =============================================================================
echo "\n==== ASVspoof 2019 LA ===="
ASV19_DIR="$BASE_DIR/asvspoof2019/LA"
mkdir -p "$ASV19_DIR"

if [ ! -d "$ASV19_DIR/ASVspoof2019_LA_train" ]; then
    echo "Downloading LA.zip..."
    wget -c https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip -P "$ASV19_DIR"

    echo "Extracting..."
    unzip -o "$ASV19_DIR/LA.zip" -d "$ASV19_DIR"
    rm "$ASV19_DIR/LA.zip"
else
    echo "ASVspoof 2019 already exists"
fi

# =============================================================================
# 2. ASVspoof5 (~45GB TRAIN ONLY via HuggingFace)
# =============================================================================
echo "\n==== ASVspoof5 TRAIN ===="
ASV5_DIR="$BASE_DIR/asvspoof5"
mkdir -p "$ASV5_DIR"

python3 <<EOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="jungjee/asvspoof5",
    repo_type="dataset",
    local_dir="${ASV5_DIR}",
    allow_patterns=["flac_T/*", "*.txt", "*.tsv"],
    resume_download=True
)
EOF

# =============================================================================
# 3. MLAAD (~18GB subset)
# =============================================================================
echo "\n==== MLAAD ===="
MLAAD_DIR="$BASE_DIR/MLAAD"
mkdir -p "$MLAAD_DIR"

python3 <<EOF
from huggingface_hub import snapshot_download, list_repo_files

repo = "mueller91/MLAAD"
local = "${MLAAD_DIR}"

allowed = [
    "de_DE","en_US","en_UK","fr_FR","es_ES",
    "it_IT","ru_RU","pl_PL","nl_NL","pt_PT"
]

files = list(list_repo_files(repo, repo_type="dataset"))
langs = set()
for f in files:
    parts = f.split("/")
    if len(parts) > 1 and parts[0] in ["fake","real"]:
        langs.add(parts[1])

ignore = [l for l in langs if l not in allowed]
ignore_patterns = [f"fake/{l}/*" for l in ignore] + [f"real/{l}/*" for l in ignore]

snapshot_download(
    repo_id=repo,
    repo_type="dataset",
    local_dir=local,
    ignore_patterns=ignore_patterns,
    resume_download=True
)
EOF

# =============================================================================
# 4. M-AILABS (~9GB)
# =============================================================================
echo "\n==== M-AILABS ===="
MAILABS_DIR="$BASE_DIR/M_AILABS"
mkdir -p "$MAILABS_DIR"

if [ ! -d "$MAILABS_DIR/en_US" ]; then
    wget -c https://data.solak.de/data/Training/stt_tts/en_US.tgz -P "$MAILABS_DIR"
    tar -xzf "$MAILABS_DIR/en_US.tgz" -C "$MAILABS_DIR"
    rm "$MAILABS_DIR/en_US.tgz"
fi

if [ ! -d "$MAILABS_DIR/en_UK" ]; then
    wget -c https://data.solak.de/data/Training/stt_tts/en_UK.tgz -P "$MAILABS_DIR"
    tar -xzf "$MAILABS_DIR/en_UK.tgz" -C "$MAILABS_DIR"
    rm "$MAILABS_DIR/en_UK.tgz"
fi

# =============================================================================
# SUMMARY
# =============================================================================
echo "\n==== SUMMARY ===="
du -sh "$BASE_DIR"/* || true

echo "\nDONE ✅"
