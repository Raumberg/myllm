from __future__ import annotations

"""Rich-based progress bar that replaces Transformers/Accelerate tqdm output."""

from typing import Any

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn, TimeElapsedColumn
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

__all__ = ["RichProgressCallback"]


class RichProgressCallback(TrainerCallback):
    def __init__(self, collator: Any | None = None) -> None:  # noqa: D401
        self.console = Console()
        self.progress: Progress | None = None
        self.task_id: int | None = None
        self._collator = collator

    # ------------------------------------------------------------------
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any):  # noqa: D401
        if state.max_steps <= 0:
            total = args.num_train_epochs * state.num_train_samples // args.per_device_train_batch_size
        else:
            total = state.max_steps

        self.progress = Progress(
            SpinnerColumn(style="cyan"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None, complete_style="magenta", finished_style="bold magenta"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=True,
            refresh_per_second=2,
        )
        self.progress.start()
        self.task_id = self.progress.add_task("[bold blue]TRAIN", total=total)

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any):  # noqa: D401
        if self.progress and self.task_id is not None:
            self.progress.update(self.task_id, completed=state.global_step)

    def on_train_end(self, *_, **__):  # noqa: D401
        if self.progress:
            self.progress.stop()

        # Dump collator stats if available
        collator = getattr(self, "_collator", None)
        if collator and hasattr(collator, "stats"):
            self.console.log(f"[collator] {collator.stats}")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: dict[str, float], **kwargs: Any):  # noqa: D401
        if not self.progress or self.task_id is None:
            return

        loss_val = logs.get("loss") or logs.get("eval_loss")
        if loss_val is not None:
            phase = "EVAL" if "eval_loss" in logs else "TRAIN"
            self.progress.update(self.task_id, description=f"[bold cyan]{phase} loss={loss_val:.4f}")

        # Emit a separate console log every log step for detailed metrics
        self.console.log({k: round(v, 4) if isinstance(v, (float, int)) else v for k, v in logs.items()})

        # ------------------------------------------------------------------
        # Extra: live collator statistics
        # ------------------------------------------------------------------
        if self._collator and hasattr(self._collator, "stats"):
            try:
                stats = self._collator.stats  # type: ignore[attr-defined]
                # Format as succinct string to avoid flooding console
                hits = stats.get("hits", 0)
                misses = stats.get("misses", 0)
                trunc = stats.get("truncated", 0)
                self.console.log(
                    f"[collator] hits={hits} misses={misses} truncated={trunc} hit_rate={hits / max(hits + misses, 1):.1%}",
                )
            except Exception as exc:  # noqa: BLE001
                # never crash training because of progress callback
                self.console.log(f"[collator] could not compute stats: {exc}") 