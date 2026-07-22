"""Run FMA LOSSO experiments."""

import argparse
import sys
from pathlib import Path


DEEP_MODELS = {"arjunvit", "atcnet", "ifnet"}
TRADITIONAL_METHODS = {"fbcsp_svm", "tslda_dgfmdrm", "twfb_dgfmdrm"}
RIDGE_METHODS = {"sbp_ridge"}
ALL_MODELS = sorted(DEEP_MODELS | TRADITIONAL_METHODS | RIDGE_METHODS)


def _add_workspace_root() -> None:
    workspace_root = Path(__file__).resolve()
    while not (workspace_root / "experiment_config.py").is_file():
        if workspace_root.parent == workspace_root:
            raise RuntimeError("Cannot locate experiment workspace root")
        workspace_root = workspace_root.parent
    sys.path.insert(0, str(workspace_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=ALL_MODELS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _add_workspace_root()

    from experiment_config import (
        FMA_DATASET_CACHE,
        FMA_SESSIONS_TSV,
        FMA_SOURCE_ROOT,
        FMA_SPLIT_ROOT,
        PARTICIPANTS_TSV,
        RESULTS_ROOT,
    )

    if args.model in DEEP_MODELS:
        from fma_regression.losso.core_deep import run_experiment

        run_experiment(
            models=[args.model],
            root_path=FMA_SOURCE_ROOT,
            participants_path=PARTICIPANTS_TSV,
            session_map_path=FMA_SESSIONS_TSV,
            io_path=FMA_DATASET_CACHE,
            split_path=FMA_SPLIT_ROOT / "losso",
            results_root=RESULTS_ROOT / "fma_regression" / "losso",
            epochs=300,
            batch_size=32,
        )
    elif args.model in TRADITIONAL_METHODS:
        from fma_regression.losso.core_traditional import run_experiment

        run_experiment(
            methods=[args.model],
            root_path=FMA_SOURCE_ROOT,
            participants_path=PARTICIPANTS_TSV,
            session_map_path=FMA_SESSIONS_TSV,
            results_root=RESULTS_ROOT / "fma_regression" / "losso",
        )
    else:
        from fma_regression.losso.ridge_core import run_experiment

        run_experiment(
            root_path=FMA_SOURCE_ROOT,
            participants_path=PARTICIPANTS_TSV,
            session_map_path=FMA_SESSIONS_TSV,
            results_dir=RESULTS_ROOT / "fma_regression" / "losso" / args.model,
        )


if __name__ == "__main__":
    main()
