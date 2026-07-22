"""Run MI cross-session experiments."""

import argparse
import sys
from pathlib import Path


DEEP_MODELS = {"arjunvit", "atcnet", "eegconformer", "eegnet", "ifnet"}
TRADITIONAL_METHODS = {"fbcsp_svm", "tslda_dgfmdrm", "twfb_dgfmdrm"}
ALL_MODELS = sorted(DEEP_MODELS | TRADITIONAL_METHODS)


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
        DERIVATIVES_ROOT,
        MI_CROSS_SESSION_CACHE,
        MI_CROSS_SESSION_SPLIT_ROOT,
        RESULTS_ROOT,
    )

    if args.model in DEEP_MODELS:
        from mi_classification.cross_session.core_deep import run_experiment

        run_experiment(
            models=[args.model],
            root_path=DERIVATIVES_ROOT,
            io_path=MI_CROSS_SESSION_CACHE,
            split_root=MI_CROSS_SESSION_SPLIT_ROOT,
            results_root=RESULTS_ROOT / "mi_classification" / "cross_session",
            epochs=300,
            batch_size=32,
        )
    else:
        from mi_classification.cross_session.core_traditional import run_experiment

        run_experiment(
            methods=[args.model],
            root_path=DERIVATIVES_ROOT,
            results_root=RESULTS_ROOT / "mi_classification" / "cross_session",
        )


if __name__ == "__main__":
    main()
