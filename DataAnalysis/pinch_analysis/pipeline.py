"""
pipeline.py — One entry point to run every analysis stage.

Each stage is a script module (``run_*.py``) exposing ``main()``. ``run(stage)``
**reloads that module from disk first**, so a notebook cell always executes the
current source even if the ``.py`` was edited since the kernel started;
``run_all()`` runs the stages in dependency order (``baselines`` last, since it
reads the CSVs the earlier stages write).

Usage (e.g. from the notebook):
    from pinch_analysis import pipeline
    pipeline.run("eda")        # one stage
    pipeline.run_all()         # everything, in order
    pipeline.run_all(["classify", "baselines"])
"""
from __future__ import annotations

import importlib
import time

# short stage name -> module under ``pinch_analysis``
STAGE_MODULES = {
    "eda": "run_eda",
    "classify": "run_classify",
    "regress": "run_regress",
    "hardneg": "run_hardneg",
    "eegnet": "run_eegnet",
    "transfer": "run_eegnet_transfer",
    "baselines": "run_baselines",
}

# dependency-safe order: baselines reads classify/eegnet/transfer outputs.
ORDER = ["eda", "classify", "regress", "hardneg", "eegnet", "transfer",
         "baselines"]

# rough per-stage runtime hint (minutes) for the notebook's information.
APPROX_MINUTES = {"eda": 4, "classify": 6, "regress": 4, "hardneg": 6,
                  "eegnet": 25, "transfer": 20, "baselines": 6}

# dependency modules to refresh so an edit to the shared library is also picked
# up when a stage is reloaded (belt-and-suspenders alongside %autoreload).
_LIB_MODULES = ["dataset", "preprocess", "epoching", "features", "eegnet"]


def _reload_lib():
    for name in _LIB_MODULES:
        try:
            importlib.reload(importlib.import_module(f"pinch_analysis.{name}"))
        except Exception:  # noqa: BLE001
            pass


def run(stage: str):
    """Reload the stage module (and shared libs) from disk, then run main()."""
    if stage not in STAGE_MODULES:
        raise KeyError(
            f"unknown stage {stage!r}; choose from {list(STAGE_MODULES)}")
    _reload_lib()
    mod = importlib.reload(
        importlib.import_module(f"pinch_analysis.{STAGE_MODULES[stage]}"))
    t0 = time.time()
    print(f"=== running '{stage}'  ({STAGE_MODULES[stage]}.main) ===")
    mod.main()
    print(f"=== '{stage}' finished in {time.time() - t0:.0f}s ===\n")


def run_all(stages: list[str] | None = None):
    """Run stages in dependency order (default: all)."""
    stages = stages or ORDER
    total = sum(APPROX_MINUTES.get(s, 5) for s in stages)
    print(f"Running {len(stages)} stage(s): {stages}  (~{total} min total)\n")
    for s in stages:
        run(s)
    print("All requested stages complete.")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    run_all(args or None)
