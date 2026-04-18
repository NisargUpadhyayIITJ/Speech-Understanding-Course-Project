#!/usr/bin/env bash


set -euo pipefail

BASE_DIR="/data2/autoNav/storage/speech_project/Speech-Understanding-Course-Project/audio_2"
mkdir -p "$BASE_DIR"


if ! command -v wget &>/dev/null; then
    echo "wget not found. Install with: conda install -c conda-forge wget"
    exit 1
fi

if ! command -v unzip &>/dev/null; then
    echo "unzip not found. Install with: sudo apt install unzip"
    exit 1
fi

echo "Checking Python packages..."
python3 -c "import kagglehub" 2>/dev/null || {
    echo "kagglehub not found. Install with: pip install kagglehub"
    exit 1
}
python3 -c "import huggingface_hub" 2>/dev/null || {
    echo "huggingface_hub not found. Install with: pip install huggingface_hub"
    exit 1
}
echo " Python packages OK"


echo "Checking Kaggle credentials..."
if [ ! -f "$HOME/.kaggle/kaggle.json" ] && \
   { [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; }; then
    echo "Kaggle credentials not found."
    echo "   Option 1: Place kaggle.json at ~/.kaggle/kaggle.json"
    echo "             (download from https://www.kaggle.com/settings -> API -> Create New Token)"
    echo "   Option 2: Set env vars: export KAGGLE_USERNAME=... KAGGLE_KEY=..."
    exit 1
fi
echo " Kaggle credentials OK"

echo "Checking HuggingFace login..."
if ! python3 -c "from huggingface_hub import whoami; whoami()" 2>/dev/null; then
    echo "Not logged in to HuggingFace."
    echo "   Run: huggingface-cli login"
    echo "   Then re-run this script."
    exit 1
fi
echo " HuggingFace login OK"

export HF_HUB_ENABLE_HF_TRANSFER=1

declare -a PIDS=()
declare -a NAMES=()


    echo "==== ASVspoof 2019 LA ===="
    ASV19_DIR="$BASE_DIR/asvspoof2019"
    mkdir -p "$ASV19_DIR"

    if [ ! -d "$ASV19_DIR/LA" ] || [ -z "$(ls -A "$ASV19_DIR/LA" 2>/dev/null)" ]; then
        echo "[ASVspoof2019] Downloading via kagglehub..."
        KAGGLE_PATH=$(python3 <<EOF
import warnings
warnings.filterwarnings("ignore")
import kagglehub

path = kagglehub.dataset_download("awsaf49/asvpoof-2019-dataset")
print(path)
EOF
        )
        echo "[ASVspoof2019] Downloaded to kaggle cache: $KAGGLE_PATH"

        if [ -d "$KAGGLE_PATH/LA" ]; then
            ln -sfn "$KAGGLE_PATH/LA" "$ASV19_DIR/LA"
            echo "[ASVspoof2019] Symlinked LA -> $ASV19_DIR/LA"
        else
            ln -sfn "$KAGGLE_PATH" "$ASV19_DIR/LA"
            echo "[ASVspoof2019] Symlinked dataset root -> $ASV19_DIR/LA"
        fi
        echo "[ASVspoof2019]  Done"
    else
        echo "[ASVspoof2019] Already exists, skipping."
    fi
) &
PIDS+=($!)
NAMES+=("ASVspoof 2019 LA")

(
    echo "==== ASVspoof5 TRAIN ===="
    ASV5_DIR="$BASE_DIR/asvspoof5"
    mkdir -p "$ASV5_DIR"

    FLAC_COUNT=$(find "$ASV5_DIR" -name "*.flac" 2>/dev/null | wc -l || echo 0)
    if [ "$FLAC_COUNT" -eq 0 ]; then
        echo "[ASVspoof5] Downloading tar shards + metadata..."
        python3 -c "
import warnings; warnings.filterwarnings('ignore', category=FutureWarning)
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='jungjee/asvspoof5',
    repo_type='dataset',
    local_dir='${ASV5_DIR}',
    allow_patterns=['flac_T_*.tar', '*.txt', '*.tsv'],
    max_workers=8,
)
print('[ASVspoof5] Download complete')
"
        echo "[ASVspoof5] Extracting tar shards..."
        for tar in "$ASV5_DIR"/flac_T_*.tar; do
            [ -f "$tar" ] && tar -xf "$tar" -C "$ASV5_DIR" && rm "$tar"
        done
        echo "[ASVspoof5]  Done"
    else
        echo "[ASVspoof5] Already has ${FLAC_COUNT} flac files, skipping."
    fi
) &
PIDS+=($!)
NAMES+=("ASVspoof5")


(
    echo "==== MLAAD ===="
    MLAAD_DIR="$BASE_DIR/MLAAD"
    mkdir -p "$MLAAD_DIR"

    python3 <<EOF
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from huggingface_hub import snapshot_download, list_repo_files

repo = "mueller91/MLAAD"
local = "${MLAAD_DIR}"

# Only download these language subsets to limit size (~18GB vs full dataset)
allowed_langs = {
    "de_DE", "en_US", "en_UK", "fr_FR", "es_ES",
    "it_IT", "ru_RU", "pl_PL", "nl_NL", "pt_PT"
}

# Build ignore patterns for all other languages
all_files = list(list_repo_files(repo, repo_type="dataset"))
all_langs = set()
for f in all_files:
    parts = f.split("/")
    if len(parts) > 1 and parts[0] in ("fake", "real"):
        all_langs.add(parts[1])

excluded = all_langs - allowed_langs
ignore_patterns = (
    [f"fake/{lang}/*" for lang in excluded] +
    [f"real/{lang}/*" for lang in excluded]
)

print(f"[MLAAD] Downloading {len(allowed_langs)} language subsets, skipping {len(excluded)}")

snapshot_download(
    repo_id=repo,
    repo_type="dataset",
    local_dir=local,
    ignore_patterns=ignore_patterns,
    max_workers=16,
)
print("[MLAAD] Done")
EOF
) &
PIDS+=($!)
NAMES+=("MLAAD")

(
    echo "==== M-AILABS ===="
    MAILABS_DIR="$BASE_DIR/M_AILABS"
    mkdir -p "$MAILABS_DIR"

    download_mailabs() {
        local lang=$1
        local url="http://www.caito.de/data/Training/stt_tts/${lang}.tgz"
        local out="$MAILABS_DIR/${lang}.tgz"

        if [ ! -d "$MAILABS_DIR/$lang" ]; then
            rm -f "${out}.aria2"
            echo "[M-AILABS] Downloading ${lang}..."
            wget -c --show-progress "$url" -O "$out"
            echo "[M-AILABS] Extracting ${lang}..."
            tar -xzf "$out" -C "$MAILABS_DIR"
            rm "$out"
            echo "[M-AILABS] ${lang} done"
        else
            echo "[M-AILABS] ${lang} already exists, skipping."
        fi
    }

    download_mailabs "en_US"
    download_mailabs "en_UK"
) &
PIDS+=($!)
NAMES+=("M-AILABS")


echo ""
echo "All downloads running in parallel. Waiting..."
FAILED=0

for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    NAME="${NAMES[$i]}"
    if wait "$PID"; then
        echo " ${NAME} completed"
    else
        echo "${NAME} FAILED"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "==== SUMMARY ===="
du -sh "$BASE_DIR"/* 2>/dev/null || true

if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo " $FAILED download(s) failed. Review errors above."
    exit 1
else
    echo ""
    echo "DONE "
fi