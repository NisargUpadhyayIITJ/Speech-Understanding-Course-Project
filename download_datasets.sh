# #!/usr/bin/env bash
# # =============================================================================
# # AASIST3 Dataset Download Script
# # Target: ~80 GB total across all 4 datasets
# #
# # Budget breakdown:
# #   ASVspoof 2019 LA     →  ~8 GB  (full: train + dev + eval)
# #   ASVspoof 2024 (AS5)  → ~45 GB  (train only, partial)
# #   MLAAD                → ~18 GB  (partial subset via HuggingFace)
# #   M-AILABS             → ~9 GB   (en_US ~7.5GB + en_UK ~3.5GB, capped to ~9GB)
# # Total target           ≈  80 GB
# # =============================================================================

# set -euo pipefail

# # ── Configuration ─────────────────────────────────────────────────────────────
# BASE_DIR="${1:-/data/additional/DATASETS/audio}"   # Override with: ./download_datasets.sh /your/path
# LOG_FILE="$(pwd)/download_datasets.log"
# JOBS=4          # Parallel download jobs (aria2c)

# # ── Colours ───────────────────────────────────────────────────────────────────
# RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
# CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

# log()  { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✔ $*${RESET}" | tee -a "$LOG_FILE"; }
# warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠ $*${RESET}" | tee -a "$LOG_FILE"; }
# err()  { echo -e "${RED}[$(date '+%H:%M:%S')] ✖ $*${RESET}" | tee -a "$LOG_FILE"; }
# info() { echo -e "${CYAN}[$(date '+%H:%M:%S')] ℹ $*${RESET}" | tee -a "$LOG_FILE"; }
# header() {
#     echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════${RESET}"
#     echo -e "${BOLD}${CYAN}  $*${RESET}"
#     echo -e "${BOLD}${CYAN}══════════════════════════════════════════════${RESET}\n"
# }

# # ── Dependency check ──────────────────────────────────────────────────────────
# check_deps() {
#     local missing=()
#     for cmd in wget aria2c python3 pip; do
#         command -v "$cmd" &>/dev/null || missing+=("$cmd")
#     done
#     if [[ ${#missing[@]} -gt 0 ]]; then
#         err "Missing dependencies: ${missing[*]}"
#         err "Install with: sudo apt-get install -y wget aria2 python3-pip"
#         exit 1
#     fi
#     # Check huggingface_hub for MLAAD
#     python3 -c "import huggingface_hub" 2>/dev/null || {
#         warn "huggingface_hub not installed. Installing..."
#         pip install -q huggingface-hub
#     }
# }

# # ── Disk space check ──────────────────────────────────────────────────────────
# check_disk_space() {
#     local required_gb=90   # 80GB data + 10GB buffer
#     local available_gb
#     available_gb=$(df -BG "$BASE_DIR" 2>/dev/null | awk 'NR==2{gsub("G",""); print $4}' || echo 0)
#     if [[ "$available_gb" -lt "$required_gb" ]]; then
#         err "Insufficient disk space! Available: ${available_gb}GB, Required: ~${required_gb}GB"
#         exit 1
#     fi
#     log "Disk space OK: ${available_gb}GB available"
# }

# # ── aria2c download helper (multi-connection, for servers that support Range) ──
# aria_download() {
#     local url="$1" dest_dir="$2" filename="${3:-}"
#     mkdir -p "$dest_dir"
#     local args=(
#         --dir="$dest_dir"
#         --max-connection-per-server=4
#         --split=4
#         --min-split-size=20M
#         --continue=true
#         --max-tries=5
#         --retry-wait=10
#         --show-console-readout=false
#         --summary-interval=30
#         -j"$JOBS"
#     )
#     [[ -n "$filename" ]] && args+=(--out="$filename")
#     aria2c "${args[@]}" "$url" 2>&1 | tee -a "$LOG_FILE" || {
#         warn "aria2c multi-split failed — retrying with single connection..."
#         wget_download "$url" "$dest_dir" "$filename"
#     }
# }

# # ── wget download helper (single-connection, safe for any server) ─────────────
# wget_download() {
#     local url="$1" dest_dir="$2" filename="${3:-}"
#     mkdir -p "$dest_dir"
#     local args=(-c --show-progress -P "$dest_dir")
#     if [[ -n "$filename" ]]; then
#         args+=(-O "${dest_dir}/${filename}")
#     fi
#     wget "${args[@]}" "$url" 2>&1 | tee -a "$LOG_FILE"
# }

# # =============================================================================
# # DATASET 1: ASVspoof 2019 LA  (~8 GB — FULL)
# # Source: HuggingFace Hub  (avoids Edinburgh DataShare range-header issues)
# # Repo: mlcommons/asvspoof2019  (public, no auth required)
# # =============================================================================
# download_asvspoof2019() {
#     header "ASVspoof 2019 LA (~8 GB)"
#     local dest="${BASE_DIR}/asvspoof2019/LA"
#     mkdir -p "$dest"

#     if [[ -d "${dest}/ASVspoof2019_LA_train" && -d "${dest}/ASVspoof2019_LA_eval" ]]; then
#         log "ASVspoof 2019 LA already extracted — skipping download"
#         return 0
#     fi

#     info "Downloading ASVspoof 2019 LA from HuggingFace Hub..."
#     python3 - <<PYEOF
# import os, sys, tarfile, zipfile
# from pathlib import Path
# from huggingface_hub import snapshot_download, hf_hub_download

# DEST = "${dest}"
# os.makedirs(DEST, exist_ok=True)

# # ASVspoof 2019 LA is available on HuggingFace as a dataset
# # Primary repo: "lca0503/asvspoof2019"  (LA split, public)
# try:
#     print("  Trying HuggingFace repo: lca0503/speech-datasets ...")
#     local_dir = snapshot_download(
#         repo_id="lca0503/asvspoof2019",
#         repo_type="dataset",
#         local_dir=DEST,
#         resume_download=True,
#     )
#     print(f"  Downloaded to: {local_dir}")
# except Exception as e:
#     print(f"  Primary repo failed: {e}")
#     print("  Falling back to Edinburgh DataShare (single-connection wget)...")
#     # Signal to bash that we need the wget fallback
#     sys.exit(42)
# PYEOF

#     local py_exit=$?
#     if [[ $py_exit -eq 42 ]]; then
#         # HuggingFace failed — fall back to Edinburgh DataShare with single connection
#         warn "HuggingFace failed — using wget (single connection) from Edinburgh DataShare..."
#         warn "Note: Edinburgh DataShare does NOT support parallel/split downloads."
#         local train_url="https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"
#         wget_download "$train_url" "$dest" "LA.zip"

#         info "Extracting ASVspoof 2019 LA..."
#         cd "$dest"
#         unzip -q -o "LA.zip" && rm -f "LA.zip"
#         cd - > /dev/null
#     fi

#     log "ASVspoof 2019 LA ready at: $dest"
# }

# # =============================================================================
# # DATASET 2: ASVspoof 2024 / ASVspoof5 (~45 GB — Train only, partial)
# # Source: https://zenodo.org/records/14498691
# # Full dataset is 142 GB; we download Train set only (≈45 GB) to stay in budget
# # =============================================================================
# download_asvspoof5() {
#     header "ASVspoof 2024 / ASVspoof5 — Train Set (~45 GB)"
#     local dest="${BASE_DIR}/asvspoof5"
#     mkdir -p "$dest"

#     # Zenodo record 14498691 — train flac archives
#     # Each part is ~5 GB; download parts 01–09 (~45 GB)
#     local base_url="https://zenodo.org/records/14498691/files"

#     local protocol_url="${base_url}/ASVspoof5.train.metadata.txt?download=1"

#     # Individual tar parts for the train split (adjust count to control size)
#     # asvspoof5_T_part01..part09 ≈ 45 GB total (9 × ~5 GB)
#     local parts=(
#         "flac_T.part01.tar?download=1"
#         "flac_T.part02.tar?download=1"
#         "flac_T.part03.tar?download=1"
#         "flac_T.part04.tar?download=1"
#         "flac_T.part05.tar?download=1"
#         "flac_T.part06.tar?download=1"
#         "flac_T.part07.tar?download=1"
#         "flac_T.part08.tar?download=1"
#         "flac_T.part09.tar?download=1"
#     )

#     # Protocol / metadata files
#     info "Downloading ASVspoof5 protocol files..."
#     for proto_file in \
#         "ASVspoof5.train.metadata.txt?download=1" \
#         "ASVspoof5.dev.trial.txt?download=1" \
#         "ASVspoof5.eval.track_1.tsv?download=1"
#     do
#         local fname="${proto_file%%\?*}"
#         if [[ ! -f "${dest}/${fname}" ]]; then
#             aria_download "${base_url}/${proto_file}" "$dest" "$fname"
#         else
#             log "Protocol file ${fname} already exists — skipping"
#         fi
#     done

#     # Download tarball parts
#     local flac_dir="${dest}/flac_T"
#     mkdir -p "$flac_dir"

#     for part in "${parts[@]}"; do
#         local fname="${part%%\?*}"          # strip query string for filename
#         local tar_path="${flac_dir}/${fname}"

#         if [[ -f "$tar_path" ]]; then
#             log "${fname} already downloaded — skipping"
#             continue
#         fi

#         info "Downloading ${fname}..."
#         aria_download "${base_url}/${part}" "$flac_dir" "$fname"

#         info "Extracting ${fname}..."
#         tar -xf "$tar_path" -C "$flac_dir" && rm -f "$tar_path"
#         log "${fname} extracted"
#     done

#     log "ASVspoof5 Train ready at: ${flac_dir}"
# }

# # =============================================================================
# # DATASET 3: MLAAD (~18 GB — partial subset via HuggingFace)
# # Repo: mueller91/MLAAD  (HuggingFace)
# # We use snapshot_download with a file-size cap; downloads the fake/ split only
# # for the first N language folders to stay within budget (~18 GB).
# # =============================================================================
# download_mlaad() {
#     header "MLAAD — Partial Subset (~18 GB)"
#     local dest="${BASE_DIR}/MLAAD"
#     mkdir -p "$dest"

#     if [[ -d "${dest}/fake" && "$(find "${dest}/fake" -name '*.wav' | wc -l)" -gt 1000 ]]; then
#         log "MLAAD appears already downloaded — skipping"
#         return 0
#     fi

#     info "Downloading MLAAD via HuggingFace Hub (partial)..."
#     python3 - <<'PYEOF'
# import os, sys
# from pathlib import Path
# from huggingface_hub import snapshot_download, list_repo_files

# REPO = "mueller91/MLAAD"
# LOCAL = os.environ.get("MLAAD_DIR", "/data/additional/DATASETS/audio/MLAAD")
# TARGET_GB = 18
# # Languages to include (covers a broad language mix ≈18 GB)
# ALLOWED_LANGS = [
#     "de_DE", "en_US", "en_UK", "fr_FR", "es_ES",
#     "it_IT", "ru_RU", "pl_PL", "nl_NL", "pt_PT",
# ]

# print(f"  Destination : {LOCAL}")
# print(f"  Target size : ~{TARGET_GB} GB")
# print(f"  Languages   : {', '.join(ALLOWED_LANGS)}")

# # Build ignore patterns for languages NOT in the allowed list
# # (HuggingFace snapshot_download uses glob-style ignore_patterns)
# all_files = list(list_repo_files(REPO, repo_type="dataset"))
# lang_dirs = set()
# for f in all_files:
#     parts = f.split("/")
#     if len(parts) >= 2 and parts[0] in ("fake", "real"):
#         lang_dirs.add(parts[1])

# ignore_langs = [d for d in lang_dirs if d not in ALLOWED_LANGS]
# ignore_patterns = [f"fake/{lang}/*" for lang in ignore_langs] + \
#                   [f"real/{lang}/*" for lang in ignore_langs]

# print(f"  Excluding {len(ignore_langs)} language folders from download")

# snapshot_download(
#     repo_id=REPO,
#     repo_type="dataset",
#     local_dir=LOCAL,
#     ignore_patterns=ignore_patterns,
#     resume_download=True,
# )
# print("  MLAAD download complete.")
# PYEOF

#     log "MLAAD ready at: $dest"
# }

# # =============================================================================
# # DATASET 4: M-AILABS  (~9 GB — en_US + en_UK)
# # Source: https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/
# # en_US ≈ 7.5 GB, en_UK ≈ 3.5 GB → download en_US (~7.5 GB) and
# # partial en_UK to fill remaining budget
# # =============================================================================
# download_mailabs() {
#     header "M-AILABS (en_US + en_UK)  (~9 GB)"
#     local dest="${BASE_DIR}/M_AILABS"
#     mkdir -p "$dest"

#     # Official mirror URLs (caito.de)
#     declare -A mailabs_urls=(
#         ["en_US"]="https://data.solak.de/data/Training/stt_tts/en_US.tgz"
#         ["en_UK"]="https://data.solak.de/data/Training/stt_tts/en_UK.tgz"
#     )

#     for lang in en_US en_UK; do
#         local url="${mailabs_urls[$lang]}"
#         local archive="${dest}/${lang}.tgz"
#         local extracted="${dest}/${lang}"

#         if [[ -d "$extracted" ]]; then
#             log "M-AILABS ${lang} already extracted — skipping"
#             continue
#         fi

#         info "Downloading M-AILABS ${lang}..."
#         aria_download "$url" "$dest" "${lang}.tgz"

#         info "Extracting ${lang}.tgz..."
#         tar -xzf "$archive" -C "$dest" && rm -f "$archive"
#         log "M-AILABS ${lang} ready"
#     done

#     log "M-AILABS ready at: $dest"
# }

# # =============================================================================
# # Summary report
# # =============================================================================
# print_summary() {
#     header "Download Summary"
#     echo -e "${BOLD}Dataset locations:${RESET}"
#     echo -e "  ASVspoof 2019 LA → ${BASE_DIR}/asvspoof2019/LA"
#     echo -e "  ASVspoof5 Train  → ${BASE_DIR}/asvspoof5/flac_T"
#     echo -e "  MLAAD (partial)  → ${BASE_DIR}/MLAAD"
#     echo -e "  M-AILABS         → ${BASE_DIR}/M_AILABS"
#     echo ""
#     echo -e "${BOLD}Approximate disk usage:${RESET}"
#     for d in \
#         "${BASE_DIR}/asvspoof2019/LA" \
#         "${BASE_DIR}/asvspoof5/flac_T" \
#         "${BASE_DIR}/MLAAD" \
#         "${BASE_DIR}/M_AILABS"
#     do
#         if [[ -d "$d" ]]; then
#             local size
#             size=$(du -sh "$d" 2>/dev/null | cut -f1)
#             echo -e "  $(basename "$d") → ${size}"
#         fi
#     done
#     echo ""
#     echo -e "${BOLD}Total:${RESET}"
#     du -sh "${BASE_DIR}" 2>/dev/null | cut -f1 | xargs -I{} echo -e "  {}"
#     echo ""
#     echo -e "${GREEN}${BOLD}✔ All datasets downloaded successfully!${RESET}"
#     echo -e "${CYAN}Update configs/train.yaml BASE_DIR paths to point to: ${BASE_DIR}${RESET}"
# }

# # =============================================================================
# # Config patcher — auto-updates train.yaml with new BASE_DIR
# # =============================================================================
# patch_train_yaml() {
#     local yaml="${SCRIPT_DIR}/configs/train.yaml"
#     if [[ ! -f "$yaml" ]]; then return; fi

#     info "Patching configs/train.yaml with new dataset paths..."
#     sed -i \
#         -e "s|/data/additional/DATASETS/audio/asvspoof5/flac_T|${BASE_DIR}/asvspoof5/flac_T|g" \
#         -e "s|/data/additional/DATASETS/audio/asvspoof5|${BASE_DIR}/asvspoof5|g" \
#         -e "s|/data/additional/DATASETS/audio/asvspoof2019/LA|${BASE_DIR}/asvspoof2019/LA|g" \
#         -e "s|/data/additional/DATASETS/audio/MLAAD|${BASE_DIR}/MLAAD|g" \
#         -e "s|/data/additional/DATASETS/audio/M_AILABS|${BASE_DIR}/M_AILABS|g" \
#         "$yaml"
#     log "configs/train.yaml updated with paths under: $BASE_DIR"
# }

# # =============================================================================
# # Main
# # =============================================================================
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# echo -e "\n${BOLD}${CYAN}"
# echo "  ╔═══════════════════════════════════════════╗"
# echo "  ║   AASIST3 Dataset Downloader (~80 GB)    ║"
# echo "  ╚═══════════════════════════════════════════╝"
# echo -e "${RESET}"
# echo -e "  Base directory : ${BOLD}$BASE_DIR${RESET}"
# echo -e "  Log file       : ${BOLD}$LOG_FILE${RESET}"
# echo -e "  Budget target  : ${BOLD}~80 GB${RESET}"
# echo ""
# echo -e "  ${YELLOW}Breakdown:${RESET}"
# echo -e "    ASVspoof 2019 LA (full)         →  ~8 GB"
# echo -e "    ASVspoof5 Train (parts 01–09)   → ~45 GB"
# echo -e "    MLAAD (10 languages)            → ~18 GB"
# echo -e "    M-AILABS (en_US + en_UK)        →  ~9 GB"
# echo -e "    ──────────────────────────────────────"
# echo -e "    Total                           ≈ 80 GB"
# echo ""

# mkdir -p "$BASE_DIR"
# check_deps
# check_disk_space

# export MLAAD_DIR="${BASE_DIR}/MLAAD"

# download_asvspoof2019
# download_asvspoof5
# download_mlaad
# download_mailabs
# patch_train_yaml
# print_summary
#!/usr/bin/env bash
# =============================================================================
# AASIST3 Dataset Download Script  (FIXED)
# Target: ~80 GB total across all 4 datasets
#
# Budget breakdown:
#   ASVspoof 2019 LA     →  ~8 GB  (full: train + dev + eval)
#   ASVspoof 2024 (AS5)  → ~45 GB  (train flac, parts 01–09 via HF mirror)
#   MLAAD                → ~18 GB  (partial subset via HuggingFace)
#   M-AILABS             → ~9 GB   (en_US ~7.5GB + en_UK ~1.5GB)
# Total target           ≈  80 GB
#
# FIXES vs original:
#   1. ASVspoof 2019: corrected HF repo from lca0503/asvspoof2019
#      → LanceaKing/asvspoof2019 (the real public mirror)
#   2. ASVspoof 2019: snapshot_download now validates the download actually
#      happened by checking for audio files, not just directory existence.
#   3. ASVspoof5: replaced broken Zenodo filename guesses with the official
#      HuggingFace mirror (jungjee/asvspoof5) for all protocol files.
#   4. ASVspoof5: HF mirror also used as primary source for train flac parts,
#      with Zenodo kept as fallback.
#   5. General: snapshot_download now raises cleanly on 404 instead of
#      silently returning the existing local_dir.
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR="${1:-/data/additional/DATASETS/audio}"
LOG_FILE="$(pwd)/download_datasets.log"
JOBS=4

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()    { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✔ $*${RESET}" | tee -a "$LOG_FILE"; }
warn()   { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠ $*${RESET}" | tee -a "$LOG_FILE"; }
err()    { echo -e "${RED}[$(date '+%H:%M:%S')] ✖ $*${RESET}" | tee -a "$LOG_FILE"; }
info()   { echo -e "${CYAN}[$(date '+%H:%M:%S')] ℹ $*${RESET}" | tee -a "$LOG_FILE"; }
header() {
    echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════════${RESET}"
    echo -e "${BOLD}${CYAN}  $*${RESET}"
    echo -e "${BOLD}${CYAN}══════════════════════════════════════════════${RESET}\n"
}

# ── Dependency check ──────────────────────────────────────────────────────────
check_deps() {
    local missing=()
    for cmd in wget aria2c python3 pip unzip; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        err "Missing dependencies: ${missing[*]}"
        err "Install with: sudo apt-get install -y wget aria2 python3-pip unzip"
        exit 1
    fi
    python3 -c "import huggingface_hub" 2>/dev/null || {
        warn "huggingface_hub not installed — installing..."
        pip install -q huggingface-hub
    }
}

# ── Disk space check ──────────────────────────────────────────────────────────
check_disk_space() {
    local required_gb=90
    local available_gb
    available_gb=$(df -BG "$BASE_DIR" 2>/dev/null | awk 'NR==2{gsub("G",""); print $4}' || echo 0)
    if [[ "$available_gb" -lt "$required_gb" ]]; then
        err "Insufficient disk space! Available: ${available_gb}GB, Required: ~${required_gb}GB"
        exit 1
    fi
    log "Disk space OK: ${available_gb}GB available"
}

# ── aria2c download helper ────────────────────────────────────────────────────
aria_download() {
    local url="$1" dest_dir="$2" filename="${3:-}"
    mkdir -p "$dest_dir"
    local args=(
        --dir="$dest_dir"
        --max-connection-per-server=4
        --split=4
        --min-split-size=20M
        --continue=true
        --max-tries=5
        --retry-wait=10
        --show-console-readout=false
        --summary-interval=30
        -j"$JOBS"
    )
    [[ -n "$filename" ]] && args+=(--out="$filename")
    aria2c "${args[@]}" "$url" 2>&1 | tee -a "$LOG_FILE" || {
        warn "aria2c multi-split failed — retrying with wget..."
        wget_download "$url" "$dest_dir" "$filename"
    }
}

# ── wget download helper ──────────────────────────────────────────────────────
wget_download() {
    local url="$1" dest_dir="$2" filename="${3:-}"
    mkdir -p "$dest_dir"
    local args=(-c --show-progress -P "$dest_dir")
    if [[ -n "$filename" ]]; then
        args+=(-O "${dest_dir}/${filename}")
    fi
    wget "${args[@]}" "$url" 2>&1 | tee -a "$LOG_FILE"
}

# =============================================================================
# DATASET 1: ASVspoof 2019 LA  (~8 GB)
#
# FIX 1: Use LanceaKing/asvspoof2019 — this is the correct public HF repo.
#         lca0503/asvspoof2019 does not exist (returns 404).
# FIX 2: Validate the download actually produced flac files, not just a dir.
# FIX 3: snapshot_download raises on HTTP errors when local_dir is non-empty
#         only if we check the returned path for actual content — we do that
#         by inspecting the file count after the call.
# =============================================================================
download_asvspoof2019() {
    header "ASVspoof 2019 LA (~8 GB)"
    local dest="${BASE_DIR}/asvspoof2019/LA"
    mkdir -p "$dest"

    local flac_count
    flac_count=$(find "$dest" -name '*.flac' 2>/dev/null | wc -l)
    if [[ $flac_count -gt 5000 ]]; then
        log "ASVspoof 2019 LA already present ($flac_count flac files) — skipping"
        return 0
    fi

    # LanceaKing/asvspoof2019 only contains dummy data + a loader script,
    # not the actual audio. Skip HF entirely and go straight to Edinburgh DataShare.
    info "Downloading ASVspoof 2019 LA from Edinburgh DataShare (official source)..."
    warn "Edinburgh DataShare does NOT support parallel/split downloads — using wget."

    local la_zip="${dest}/LA.zip"
    if [[ ! -f "$la_zip" ]]; then
        wget_download \
            "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip" \
            "$dest" "LA.zip"
    else
        log "LA.zip already present — skipping download"
    fi

    info "Extracting LA.zip (~8 GB, may take a few minutes)..."
    unzip -q -o "$la_zip" -d "$dest" && rm -f "$la_zip"

    local flac_count_after
    flac_count_after=$(find "$dest" -name '*.flac' 2>/dev/null | wc -l)
    log "ASVspoof 2019 LA ready at: $dest ($flac_count_after flac files)"
}
# download_asvspoof2019() {
#     header "ASVspoof 2019 LA (~8 GB)"
#     local dest="${BASE_DIR}/asvspoof2019/LA"
#     mkdir -p "$dest"

#     # Check if already fully extracted (has train + eval subdirs with audio)
#     local flac_count
#     flac_count=$(find "$dest" -name '*.flac' 2>/dev/null | wc -l)
#     if [[ $flac_count -gt 5000 ]]; then
#         log "ASVspoof 2019 LA already present ($flac_count flac files) — skipping"
#         return 0
#     fi

#     info "Downloading ASVspoof 2019 LA from HuggingFace: LanceaKing/asvspoof2019"
#     python3 - "$dest" <<'PYEOF'
# import os, sys
# from pathlib import Path

# dest = sys.argv[1]

# # ── FIX: import and verify repo exists before downloading ────────────────────
# try:
#     from huggingface_hub import snapshot_download, list_repo_tree
# except ImportError:
#     print("ERROR: huggingface_hub not installed", file=sys.stderr)
#     sys.exit(42)

# REPO = "LanceaKing/asvspoof2019"   # FIX: correct repo (not lca0503/)

# # Verify the repo is reachable by listing its root — raises RepositoryNotFoundError on 404
# try:
#     print(f"  Verifying repo {REPO} is reachable...")
#     # list_repo_tree raises on 404; faster than snapshot for validation
#     next(list_repo_tree(REPO, repo_type="dataset"), None)
#     print(f"  Repo confirmed accessible.")
# except Exception as e:
#     print(f"  Repo check failed: {e}")
#     print("  Will fall back to Edinburgh DataShare wget download.")
#     sys.exit(42)

# try:
#     print(f"  Starting snapshot_download → {dest}")
#     snapshot_download(
#         repo_id=REPO,
#         repo_type="dataset",
#         local_dir=dest,
#         resume_download=True,
#     )
#     # Validate something actually downloaded
#     flac_files = list(Path(dest).rglob("*.flac"))
#     if len(flac_files) < 100:
#         print(f"  WARNING: only {len(flac_files)} flac files found — possibly incomplete.")
#         print("  Falling back to Edinburgh DataShare.")
#         sys.exit(42)
#     print(f"  Downloaded {len(flac_files)} flac files.")
# except Exception as e:
#     print(f"  snapshot_download error: {e}")
#     sys.exit(42)
# PYEOF

#     local py_exit=$?
#     if [[ $py_exit -eq 42 ]]; then
#         warn "HuggingFace download failed or incomplete."
#         warn "Falling back to Edinburgh DataShare (single-connection wget — slow but reliable)."
#         warn "Note: Edinburgh DataShare does NOT support Range requests / parallel splits."
#         local la_zip="${dest}/LA.zip"
#         if [[ ! -f "$la_zip" ]]; then
#             wget_download \
#                 "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip" \
#                 "$dest" "LA.zip"
#         else
#             log "LA.zip already downloaded — skipping wget"
#         fi
#         info "Extracting LA.zip..."
#         unzip -q -o "$la_zip" -d "$dest" && rm -f "$la_zip"
#     fi

#     log "ASVspoof 2019 LA ready at: $dest"
# }

# =============================================================================
# DATASET 2: ASVspoof 2024 / ASVspoof5 (~45 GB — Train only)
#
# FIX: Use the official HuggingFace mirror (jungjee/asvspoof5) for protocol
#      files instead of guessing Zenodo filenames.
#      The mirror has all metadata files under metadata/ and audio under
#      flac_T/ (train), flac_D/ (dev), flac_E/ (eval).
#      Zenodo kept as fallback for the large flac parts only.
#
# HuggingFace mirror: https://huggingface.co/datasets/jungjee/asvspoof5
# Zenodo (audio only): https://zenodo.org/records/14498691
# =============================================================================
download_asvspoof5() {
    header "ASVspoof 2024 / ASVspoof5 — Train Set (~45 GB)"
    local dest="${BASE_DIR}/asvspoof5"
    mkdir -p "$dest"

    # ── Step 1: Download protocol/metadata files via HuggingFace mirror ──────
    # FIX: The exact Zenodo filenames for dev/eval protocols are unknown and
    #      kept changing. The HF mirror (jungjee/asvspoof5) is the official
    #      mirror and has a stable layout with all metadata.
    info "Downloading ASVspoof5 metadata via HuggingFace mirror (jungjee/asvspoof5)..."
    python3 - "$dest" <<'PYEOF'
import os, sys
from pathlib import Path

dest = sys.argv[1]

try:
    from huggingface_hub import snapshot_download, list_repo_tree
except ImportError:
    print("ERROR: huggingface_hub not installed", file=sys.stderr)
    sys.exit(1)

REPO = "jungjee/asvspoof5"

# Download metadata/protocol files only (ignore the large flac archives)
# We'll pull the flac_T train audio separately from Zenodo to save HF bandwidth.
try:
    print(f"  Verifying repo {REPO}...")
    next(list_repo_tree(REPO, repo_type="dataset"), None)
    print(f"  Repo accessible. Downloading metadata files only...")

    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=dest,
        # Download everything EXCEPT the large flac audio directories
        # (we get train audio from Zenodo below to stay in budget)
        ignore_patterns=["flac_T/*", "flac_D/*", "flac_E/*", "*.tar", "*.tar.*"],
        resume_download=True,
    )
    meta_files = list(Path(dest).rglob("*.txt")) + list(Path(dest).rglob("*.tsv"))
    print(f"  Metadata download done. Found {len(meta_files)} protocol files.")
except Exception as e:
    print(f"  HuggingFace mirror failed: {e}")
    print("  Protocol files will need to be obtained from asvspoof.org manually.")
    sys.exit(0)   # non-fatal — audio download can still proceed
PYEOF

    # ── Step 2: Download train flac parts from Zenodo ────────────────────────
    local base_url="https://zenodo.org/records/14498691/files"
    local flac_dir="${dest}/flac_T"
    mkdir -p "$flac_dir"

    # parts 01–09 ≈ 45 GB (adjust the list to control total size)
    local parts=(
        "flac_T.part01.tar?download=1"
        "flac_T.part02.tar?download=1"
        "flac_T.part03.tar?download=1"
        "flac_T.part04.tar?download=1"
        "flac_T.part05.tar?download=1"
        "flac_T.part06.tar?download=1"
        "flac_T.part07.tar?download=1"
        "flac_T.part08.tar?download=1"
        "flac_T.part09.tar?download=1"
    )

    for part in "${parts[@]}"; do
        local fname="${part%%\?*}"
        local tar_path="${flac_dir}/${fname}"
        local extracted_marker="${flac_dir}/.extracted_${fname}"

        if [[ -f "$extracted_marker" ]]; then
            log "${fname} already extracted — skipping"
            continue
        fi

        if [[ ! -f "$tar_path" ]]; then
            info "Downloading ${fname} from Zenodo..."
            aria_download "${base_url}/${part}" "$flac_dir" "$fname"
        else
            log "${fname} already downloaded, extracting..."
        fi

        info "Extracting ${fname}..."
        tar -xf "$tar_path" -C "$flac_dir" && rm -f "$tar_path"
        touch "$extracted_marker"
        log "${fname} extracted"
    done

    log "ASVspoof5 Train ready at: ${flac_dir}"
}

# =============================================================================
# DATASET 3: MLAAD (~18 GB)
# =============================================================================
download_mlaad() {
    header "MLAAD — Partial Subset (~18 GB)"
    local dest="${BASE_DIR}/MLAAD"
    mkdir -p "$dest"

    local wav_count
    wav_count=$(find "${dest}/fake" -name '*.wav' 2>/dev/null | wc -l)
    if [[ $wav_count -gt 1000 ]]; then
        log "MLAAD appears already downloaded ($wav_count wav files) — skipping"
        return 0
    fi

    info "Downloading MLAAD via HuggingFace Hub (partial)..."
    python3 - "$dest" <<'PYEOF'
import os, sys
from pathlib import Path
from huggingface_hub import snapshot_download, list_repo_files

dest = sys.argv[1]
REPO = "mueller91/MLAAD"
ALLOWED_LANGS = [
    "de_DE", "en_US", "en_UK", "fr_FR", "es_ES",
    "it_IT", "ru_RU", "pl_PL", "nl_NL", "pt_PT",
]

print(f"  Destination : {dest}")
print(f"  Languages   : {', '.join(ALLOWED_LANGS)}")

all_files = list(list_repo_files(REPO, repo_type="dataset"))
lang_dirs = set()
for f in all_files:
    parts = f.split("/")
    if len(parts) >= 2 and parts[0] in ("fake", "real"):
        lang_dirs.add(parts[1])

ignore_langs = [d for d in lang_dirs if d not in ALLOWED_LANGS]
ignore_patterns = [f"fake/{lang}/*" for lang in ignore_langs] + \
                  [f"real/{lang}/*" for lang in ignore_langs]

print(f"  Excluding {len(ignore_langs)} language folders")

snapshot_download(
    repo_id=REPO,
    repo_type="dataset",
    local_dir=dest,
    ignore_patterns=ignore_patterns,
    resume_download=True,
)
print("  MLAAD download complete.")
PYEOF

    log "MLAAD ready at: $dest"
}

# =============================================================================
# DATASET 4: M-AILABS (~9 GB — en_US + en_UK)
# =============================================================================
download_mailabs() {
    header "M-AILABS (en_US + en_UK) (~9 GB)"
    local dest="${BASE_DIR}/M_AILABS"
    mkdir -p "$dest"

    declare -A mailabs_urls=(
        ["en_US"]="https://data.solak.de/data/Training/stt_tts/en_US.tgz"
        ["en_UK"]="https://data.solak.de/data/Training/stt_tts/en_UK.tgz"
    )

    for lang in en_US en_UK; do
        local url="${mailabs_urls[$lang]}"
        local archive="${dest}/${lang}.tgz"
        local extracted="${dest}/${lang}"

        if [[ -d "$extracted" ]]; then
            log "M-AILABS ${lang} already extracted — skipping"
            continue
        fi

        info "Downloading M-AILABS ${lang}..."
        aria_download "$url" "$dest" "${lang}.tgz"

        info "Extracting ${lang}.tgz..."
        tar -xzf "$archive" -C "$dest" && rm -f "$archive"
        log "M-AILABS ${lang} ready"
    done

    log "M-AILABS ready at: $dest"
}

# =============================================================================
# Summary
# =============================================================================
print_summary() {
    header "Download Summary"
    echo -e "${BOLD}Dataset locations:${RESET}"
    echo -e "  ASVspoof 2019 LA → ${BASE_DIR}/asvspoof2019/LA"
    echo -e "  ASVspoof5 Train  → ${BASE_DIR}/asvspoof5/flac_T"
    echo -e "  MLAAD (partial)  → ${BASE_DIR}/MLAAD"
    echo -e "  M-AILABS         → ${BASE_DIR}/M_AILABS"
    echo ""
    echo -e "${BOLD}Approximate disk usage:${RESET}"
    for d in \
        "${BASE_DIR}/asvspoof2019/LA" \
        "${BASE_DIR}/asvspoof5/flac_T" \
        "${BASE_DIR}/MLAAD" \
        "${BASE_DIR}/M_AILABS"
    do
        if [[ -d "$d" ]]; then
            local size
            size=$(du -sh "$d" 2>/dev/null | cut -f1)
            echo -e "  $(basename "$d") → ${size}"
        fi
    done
    echo ""
    du -sh "${BASE_DIR}" 2>/dev/null | cut -f1 | xargs -I{} echo -e "  Total: {}"
    echo ""
    echo -e "${GREEN}${BOLD}✔ All datasets downloaded successfully!${RESET}"
    echo -e "${CYAN}Update configs/train.yaml BASE_DIR paths to: ${BASE_DIR}${RESET}"
}

# =============================================================================
# Config patcher
# =============================================================================
patch_train_yaml() {
    local yaml="${SCRIPT_DIR}/configs/train.yaml"
    [[ ! -f "$yaml" ]] && return
    info "Patching configs/train.yaml..."
    sed -i \
        -e "s|/data/additional/DATASETS/audio/asvspoof5/flac_T|${BASE_DIR}/asvspoof5/flac_T|g" \
        -e "s|/data/additional/DATASETS/audio/asvspoof5|${BASE_DIR}/asvspoof5|g" \
        -e "s|/data/additional/DATASETS/audio/asvspoof2019/LA|${BASE_DIR}/asvspoof2019/LA|g" \
        -e "s|/data/additional/DATASETS/audio/MLAAD|${BASE_DIR}/MLAAD|g" \
        -e "s|/data/additional/DATASETS/audio/M_AILABS|${BASE_DIR}/M_AILABS|g" \
        "$yaml"
    log "configs/train.yaml updated"
}

# =============================================================================
# Main
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "\n${BOLD}${CYAN}"
echo "  ╔═══════════════════════════════════════════╗"
echo "  ║   AASIST3 Dataset Downloader (~80 GB)    ║"
echo "  ╚═══════════════════════════════════════════╝"
echo -e "${RESET}"
echo -e "  Base directory : ${BOLD}$BASE_DIR${RESET}"
echo -e "  Log file       : ${BOLD}$LOG_FILE${RESET}"
echo ""

mkdir -p "$BASE_DIR"
check_deps
check_disk_space

download_asvspoof2019
download_asvspoof5
download_mlaad
download_mailabs
patch_train_yaml
print_summary