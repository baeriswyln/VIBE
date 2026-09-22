import wandb

from typing import Optional

WANDB_PROJECT = "eeg-pinch-classification"

wandb.login()


def wandb_start_run(model_name: str, run_name: Optional[str] = None, config: Optional[dict] = None, group: Optional[str] = None, tags: Optional[list[str]] = None, enabled: bool = True):
    """Start a W&B run for one model-training run, shared across model types.

    `model_name` (e.g. "EEGNet", "RandomForest", "Conformer") is stored in the config
    and used as the default group, so runs from different models stay comparable in
    the same project. Set enabled=False to skip tracking (e.g. quick local debugging).
    """
    if not enabled:
        return None
    return wandb.init(
        project=WANDB_PROJECT,
        group=group or model_name,
        job_type=model_name,
        name=run_name,
        tags=[model_name, *(tags or [])],
        config={"model": model_name, **(config or {})},
        reinit="finish_previous",  # auto-finish a dangling run, e.g. after a KeyboardInterrupt
    )


def wandb_log_test_results(run, *, test_accuracy: float, y_true, y_pred, classes, test_loss: Optional[float] = None, extra: Optional[dict] = None):
    """Log final test metrics + confusion matrix. Call once per run, before wandb_finish_run."""
    if run is None:
        return
    summary = {"test/accuracy": test_accuracy, **(extra or {})}
    if test_loss is not None:
        summary["test/loss"] = test_loss
    run.summary.update(summary)
    run.log({
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            preds=[int(p) for p in y_pred],
            y_true=[int(t) for t in y_true],
            class_names=[str(c) for c in classes],
        )
    })


def wandb_finish_run(run):
    if run is not None:
        run.finish()