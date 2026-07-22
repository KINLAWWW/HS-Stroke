"""Run MI cross-trial experiments."""

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

    if args.model in DEEP_MODELS:
        from mi_classification.cross_trial.core_deep import main as run_deep

        run_deep(args.model)
    else:
        from mi_classification.cross_trial.core_traditional import main as run_traditional

        run_traditional(args.model)


if __name__ == "__main__":
    main()
