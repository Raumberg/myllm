from __future__ import annotations

"""Lightweight evaluation runner used to compute metrics on a dataloader."""

from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from .base import BaseMetric

__all__ = ["Evaluator"]


class Evaluator:
    """Run a model on a dataloader and compute a list of metrics.

    Parameters
    ----------
    model : torch.nn.Module
        Model must return ``loss`` when provided with ``labels``.
    metrics : list[BaseMetric]
        Metric instances to compute. Each metric should implement ``update`` and
        ``compute``.
    device : str | torch.device | None
        Device to move batches to (defaults to model's first parameter device).
    """

    def __init__(self, model: torch.nn.Module, metrics: List[BaseMetric], *, device: Any | None = None):
        if not metrics:
            raise ValueError("At least one metric must be provided to Evaluator")

        self.model = model
        self.metrics = metrics
        self.device = device if device is not None else next(model.parameters()).device

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:  # noqa: D401
        """Loop over *dataloader* and return dict of metric → value."""

        self.model.eval()
        for m in self.metrics:
            m.reset()

        for batch in dataloader:
            # Move tensor items to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}

            outputs = self.model(**batch)
            if not hasattr(outputs, "loss") or outputs.loss is None:
                raise RuntimeError("Model did not return loss – perplexity requires labels & loss.")

            loss_val = outputs.loss.detach().float().item()

            # Determine number of valid tokens (labels != -100)
            labels = batch.get("labels")
            if labels is None:
                raise RuntimeError("Batch does not contain 'labels' key – can't compute perplexity.")
            valid_tokens = int((labels != -100).sum().item())

            for metric in self.metrics:
                if metric.name == "perplexity":
                    metric.update(loss=loss_val, tokens=valid_tokens)
                else:
                    metric.update(batch=batch, outputs=outputs)

        return {m.name: float(m.compute()) for m in self.metrics} 