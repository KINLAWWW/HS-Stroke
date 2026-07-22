"""Repository path configuration."""

from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent

DATASET_ROOT = WORKSPACE_ROOT / "dataset"
SOURCE_DATA_ROOT = DATASET_ROOT / "sourcedata"
DERIVATIVES_ROOT = DATASET_ROOT / "derivatives"
FMA_SOURCE_ROOT = SOURCE_DATA_ROOT
PARTICIPANTS_TSV = DATASET_ROOT / "participants.tsv"
FMA_SESSIONS_TSV = DATASET_ROOT / "fma_sessions.tsv"

CACHE_ROOT = WORKSPACE_ROOT / "cache"
MI_CROSS_TRIAL_CACHE = CACHE_ROOT / "datasets" / "mi_cross_trial"
MI_CROSS_SESSION_CACHE = CACHE_ROOT / "datasets" / "mi_cross_session"
MI_CROSS_SESSION_SPLIT_ROOT = CACHE_ROOT / "splits" / "mi_cross_session"
FMA_DATASET_CACHE = CACHE_ROOT / "datasets" / "fma_regression"
FMA_SPLIT_ROOT = CACHE_ROOT / "splits" / "fma_regression"
RESULTS_ROOT = WORKSPACE_ROOT / "results"
