#!/usr/bin/env bash
# =============================================================================
# scripts/run_sample.sh
#
# Billboard Cultural Mood Analysis — Sample Run Orchestrator
#
# Runs a scoped 5-year sample (1985–1989, 1980s decade bucket) to validate
# the full pipeline end-to-end without processing the entire 1958–2025 dataset.
#
# Usage:
#   ./scripts/run_sample.sh                    # standard sample run
#   ./scripts/run_sample.sh --force            # ignore sentinels, re-run all
#   ./scripts/run_sample.sh --log-level DEBUG  # verbose output
#   ./scripts/run_sample.sh --dry-run-only     # pre-flight + import check only
#
# Exit codes:
#   0  all stages passed + artifacts verified
#   1  one or more stages failed
#   2  pre-flight check failed (environment not ready)
#   3  fatal environment error (missing API key, unrecoverable IO)
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

SAMPLE_START=1985
SAMPLE_END=1989
DECADE_FILTER="1980s"
LOG_LEVEL="INFO"
FORCE_FLAG=""
DRY_RUN_ONLY=false
SCRIPT_START=$(date +%s)

# Colour codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

# ── Argument Parsing ──────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_FLAG="--force"
            shift
            ;;
        --log-level)
            if [[ -z "${2:-}" ]]; then
                echo -e "${RED}--log-level requires a value (e.g. DEBUG, INFO)${RESET}"
                exit 2
            fi
            LOG_LEVEL="$2"
            shift 2
            ;;
        --log-level=*)
            LOG_LEVEL="${1#*=}"
            shift
            ;;
        --dry-run-only)
            DRY_RUN_ONLY=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown argument: $1${RESET}"
            echo "Usage: $0 [--force] [--log-level LEVEL] [--dry-run-only]"
            exit 2
            ;;
    esac
done


# ── Helpers ───────────────────────────────────────────────────────────────────

print_header() {
  echo -e ""
  echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════╗${RESET}"
  echo -e "${CYAN}${BOLD}║   Billboard Cultural Mood Analysis — Sample Run      ║${RESET}"
  echo -e "${CYAN}${BOLD}║   Range  : ${SAMPLE_START}–${SAMPLE_END} | Decade: ${DECADE_FILTER}                 ║${RESET}"
  echo -e "${CYAN}${BOLD}║   Started: $(date '+%Y-%m-%d %H:%M:%S')                       ║${RESET}"
  echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════╝${RESET}"
  echo ""
}

log_ok()   { echo -e "  ${GREEN}✓${RESET} $1"; }
log_warn() { echo -e "  ${YELLOW}⚠${RESET}  $1"; }
log_fail() { echo -e "  ${RED}✗${RESET} $1"; }
log_info() { echo -e "  ${CYAN}→${RESET} $1"; }

section() {
  echo ""
  echo -e "${BOLD}── $1 ──────────────────────────────────────────────────${RESET}"
}

elapsed_since() {
  local start=$1
  local end
  end=$(date +%s)
  echo $(( end - start ))
}

# ── Pre-flight Checks ─────────────────────────────────────────────────────────

preflight_checks() {
  section "PRE-FLIGHT CHECKS"
  local failed=0

  # 1. Python version ≥ 3.10
  if command -v python3 &>/dev/null; then
    PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
    if [[ "$PY_MAJOR" -ge 3 && "$PY_MINOR" -ge 10 ]]; then
      log_ok "Python $PY_VERSION"
    else
      log_fail "Python $PY_VERSION — 3.10+ required"
      (( failed++ ))
    fi
  else
    log_fail "python3 not found"
    (( failed++ ))
  fi

  # 2. Running from project root (main.py must exist)
  if [[ -f "main.py" ]]; then
    log_ok "Working directory: $(pwd)"
  else
    log_fail "main.py not found — run this script from the project root"
    (( failed++ ))
  fi

  # 3. Virtual environment active
  if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    log_ok "Virtual environment: $VIRTUAL_ENV"
  else
    log_warn "No virtual environment detected — continuing, but isolation is recommended"
  fi

  # 4. .env file present
  if [[ -f ".env" ]]; then
    log_ok ".env file present"
  else
    log_fail ".env not found — copy .env.example and populate API keys"
    (( failed++ ))
  fi

  # 5. Load .env and check required API keys
  if [[ -f ".env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a

    if [[ -n "${GENIUS_API_TOKEN:-}" ]]; then
      log_ok "GENIUS_API_TOKEN set"
    else
      log_fail "GENIUS_API_TOKEN missing in .env"
      (( failed++ ))
    fi

    if [[ -n "${OPENAI_API_KEY:-}" ]]; then
      log_ok "OPENAI_API_KEY set"
    else
      log_fail "OPENAI_API_KEY missing in .env"
      (( failed++ ))
    fi
  fi

  # 6. pyproject.toml / package install check
  if python3 -c "import src" &>/dev/null; then
    log_ok "src package importable (editable install active)"
  else
    log_fail "src package not importable — run: pip install -e ."
    (( failed++ ))
  fi

  # 7. Key dependencies
  local deps=("pandera" "loguru" "pydantic" "plotly" "kaleido" "openai" "lyricsgenius")
  for dep in "${deps[@]}"; do
    if python3 -c "import $dep" &>/dev/null; then
      log_ok "$dep"
    else
      log_fail "$dep not installed — run: pip install -r requirements.txt"
      (( failed++ ))
    fi
  done

  # 8. Config file present
  if [[ -f "config/project_config.yaml" ]]; then
    log_ok "config/project_config.yaml present"
  else
    log_fail "config/project_config.yaml not found"
    (( failed++ ))
  fi

  # 9. Output directories exist or can be created
  local dirs=("data/processed" "data/analysis" "outputs" "outputs/logs" "cache/lyrics")
  for dir in "${dirs[@]}"; do
    mkdir -p "$dir" && log_ok "Directory ready: $dir"
  done

  if [[ $failed -gt 0 ]]; then
    echo ""
    log_fail "Pre-flight FAILED — $failed check(s) did not pass. Aborting."
    exit 2
  fi

  log_ok "All pre-flight checks passed."
}

# ── Dry Run (Import Validation) ───────────────────────────────────────────────

run_dry_run() {
  section "DRY RUN — IMPORT VALIDATION"
  log_info "Validating all 11 stage module imports..."
  echo ""

  if python3 -m main --dry-run --log-level "$LOG_LEVEL"; then
    log_ok "Dry run passed — all stage modules importable."
  else
    log_fail "Dry run FAILED — fix import errors before running the pipeline."
    exit 1
  fi
}

# ── Pipeline Execution ────────────────────────────────────────────────────────

run_pipeline() {
  section "PIPELINE EXECUTION"
  log_info "Sample: ${SAMPLE_START}–${SAMPLE_END} | Decade: ${DECADE_FILTER}"
  [[ -n "$FORCE_FLAG" ]] && log_warn "--force flag set — ignoring all sentinels"
  echo ""

  local pipeline_start
  pipeline_start=$(date +%s)

  if python3 -m main \
    --sample-years "${SAMPLE_START}-${SAMPLE_END}" \
    --decade-filter "${DECADE_FILTER}" \
    --log-level "${LOG_LEVEL}" \
    ${FORCE_FLAG}; then
    local secs
    secs=$(elapsed_since "$pipeline_start")
    echo ""
    log_ok "Pipeline completed in ${secs}s."
  else
    local secs
    secs=$(elapsed_since "$pipeline_start")
    echo ""
    log_fail "Pipeline FAILED after ${secs}s."
    exit 1
  fi
}

# ── Output Artifact Validation ────────────────────────────────────────────────

validate_artifacts() {
  section "OUTPUT ARTIFACT VALIDATION"
  local failed=0

  # Expected CSV outputs
  declare -A ARTIFACTS=(
    ["data/processed/song_metadata.csv"]="Stage 1 — Billboard metadata"
    ["data/processed/lyrics_cleaned.csv"]="Stage 3 — Cleaned lyrics"
    ["data/processed/chorus_extracted.csv"]="Stage 4 — Chorus detection"
    ["data/analysis/layer2_sentiment.csv"]="Stage 5 — Sentiment scores"
    ["data/analysis/layer2_emotion.csv"]="Stage 6 — Emotion classification"
    ["data/analysis/layer2_themes.csv"]="Stage 7 — Theme classification"
    ["data/analysis/layer2_full_analysis.csv"]="Stage 8 — Master analysis CSV"
    ["data/analysis/layer5_jungian.csv"]="Stage 9 — Jungian scoring"
    ["data/analysis/layer6_cultural_metrics.csv"]="Stage 10 — Cultural resonance"
    ["data/analysis/decade_cmi.csv"]="Stage 10 — Decade CMI aggregate"
  )

  for path in "${!ARTIFACTS[@]}"; do
    label="${ARTIFACTS[$path]}"
    if [[ -f "$path" ]]; then
      # Row count (minus header)
      rows=$(( $(wc -l < "$path") - 1 ))
      log_ok "$label — $path ($rows rows)"
    else
      log_fail "$label — MISSING: $path"
      (( failed++ ))
    fi
  done

  # Sentinel files
  section "SENTINEL VERIFICATION"
  declare -A SENTINELS=(
      ["data/processed/.billboard_complete"]="BILLBOARD_FETCH"
      ["data/processed/.lyrics_complete"]="LYRICS_FETCH"
      ["data/processed/.cleaning_complete"]="TEXT_CLEANING"
      ["data/processed/.chorus_complete"]="CHORUS_DETECTION"
      ["data/analysis/.sentiment_complete"]="SENTIMENT_SCORING"
      ["data/analysis/.emotion_complete"]="EMOTION_CLASSIFICATION"
      ["data/analysis/.themes_complete"]="THEME_CLASSIFICATION"
      ["data/analysis/.contrast_complete"]="CONTRAST_METRICS"
      ["data/analysis/.jungian_complete"]="JUNGIAN_SCORING"
      ["data/analysis/.cmi_complete"]="CULTURAL_METRICS"
      ["outputs/visualizations/.charts_complete"]="VISUALIZATION"
  )


  for sentinel in "${!SENTINELS[@]}"; do
    stage="${SENTINELS[$sentinel]}"
    if [[ -f "$sentinel" ]]; then
      log_ok "$stage sentinel: $sentinel"
    else
      log_warn "$stage sentinel missing: $sentinel"
    fi
  done

  # Visualization outputs
  section "VISUALIZATION ARTIFACTS"
  if [[ -d "outputs" ]]; then
    PNG_COUNT=$(find outputs -maxdepth 2 -name "*.png" | wc -l | tr -d ' ')
    HTML_COUNT=$(find outputs -maxdepth 2 -name "*.html" | wc -l | tr -d ' ')
    log_info "PNG charts found  : $PNG_COUNT"
    log_info "HTML reports found: $HTML_COUNT"
    if [[ "$PNG_COUNT" -ge 5 ]]; then
      log_ok "Chart suite complete ($PNG_COUNT PNGs)"
    else
      log_warn "Expected ≥5 PNG charts, found $PNG_COUNT"
    fi
    if [[ "$HTML_COUNT" -ge 1 ]]; then
      log_ok "HTML report present"
    else
      log_warn "HTML report not found in outputs/"
    fi
  else
    log_fail "outputs/ directory missing"
    (( failed++ ))
  fi

  if [[ $failed -gt 0 ]]; then
    echo ""
    log_fail "Artifact validation FAILED — $failed required file(s) missing."
    exit 1
  fi
}

# ── Final Summary ─────────────────────────────────────────────────────────────

print_summary() {
  local total_secs
  total_secs=$(elapsed_since "$SCRIPT_START")
  local mins=$(( total_secs / 60 ))
  local secs=$(( total_secs % 60 ))

  echo ""
  echo -e "${CYAN}${BOLD}╔══════════════════════════════════════════════════════╗${RESET}"
  echo -e "${CYAN}${BOLD}║   SAMPLE RUN COMPLETE                                ║${RESET}"
  echo -e "${CYAN}${BOLD}║   Range   : ${SAMPLE_START}–${SAMPLE_END} | Decade: ${DECADE_FILTER}                 ║${RESET}"
  echo -e "${CYAN}${BOLD}║   Duration: ${mins}m ${secs}s                                     ║${RESET}"
  echo -e "${CYAN}${BOLD}║   Status  : ALL STAGES PASSED ✓                      ║${RESET}"
  echo -e "${CYAN}${BOLD}╚══════════════════════════════════════════════════════╝${RESET}"
  echo ""
  log_info "Charts  → outputs/"
  log_info "Report  → outputs/cultural_mood_report.html"
  log_info "Logs    → outputs/logs/"
  echo ""
}

# ── Entrypoint ────────────────────────────────────────────────────────────────

print_header
preflight_checks
run_dry_run

if [[ "$DRY_RUN_ONLY" == true ]]; then
  log_ok "Dry-run-only mode — exiting after import validation."
  exit 0
fi

run_pipeline
validate_artifacts
print_summary

exit 0
