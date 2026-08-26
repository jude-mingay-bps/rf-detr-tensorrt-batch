# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""W&B logger that writes one consolidated history row per completed epoch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from lightning_fabric.utilities.rank_zero import rank_zero_only
from pytorch_lightning.loggers import WandbLogger


class EpochWandbLogger(WandbLogger):
    """Consolidate Lightning metric batches into 1-based W&B epoch rows.

    Lightning emits learning rates during training, validation metrics at
    validation end, and aggregated training losses at training epoch end.
    ``WandbLogger`` normally commits each emission as a separate history row
    keyed by ``trainer/global_step``. This logger retains the latest value for
    each key and commits it when the next epoch starts (or training finalizes).
    Waiting for the epoch transition is deliberate: Lightning logger emissions
    from train-epoch-end and validation callbacks are not guaranteed to arrive
    in one fixed order.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the W&B logger and its empty epoch buffer.

        Args:
            *args: Positional arguments forwarded to ``WandbLogger``.
            **kwargs: Keyword arguments forwarded to ``WandbLogger``.
        """
        super().__init__(*args, **kwargs)
        self._pending_epoch: int | None = None
        self._pending_metrics: dict[str, float] = {}
        self._last_flushed_epoch: int | None = None
        self._epoch_axis_defined = False

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Buffer a Lightning metric emission and flush at epoch completion.

        Args:
            metrics: Scalar metrics produced by Lightning.
            step: Lightning global step, intentionally ignored for W&B history.
        """
        del step
        raw_epoch = metrics.get("epoch")
        if raw_epoch is None:
            if self._pending_epoch is not None:
                self._pending_metrics.update(metrics)
            return
        epoch = int(raw_epoch)
        if self._last_flushed_epoch is not None and epoch <= self._last_flushed_epoch:
            return
        if self._pending_epoch is not None and epoch != self._pending_epoch:
            self._flush_epoch_metrics()
        self._pending_epoch = epoch
        self._pending_metrics.update(
            {key: value for key, value in metrics.items() if key not in {"epoch", "trainer/global_step"}}
        )

    def _define_epoch_axis(self) -> None:
        """Configure W&B charts to use the explicit epoch metric as x-axis."""
        if self._epoch_axis_defined:
            return
        experiment = self.experiment
        experiment.define_metric("epoch")
        for namespace in ("train/*", "val/*", "test/*"):
            experiment.define_metric(namespace, step_metric="epoch")
        self._epoch_axis_defined = True

    def _flush_epoch_metrics(self) -> None:
        """Write and clear the pending epoch, if it contains any metrics."""
        if self._pending_epoch is None or not self._pending_metrics:
            return
        epoch = self._pending_epoch
        payload = dict(self._pending_metrics)
        payload["epoch"] = epoch + 1
        self._define_epoch_axis()
        super().log_metrics(payload, step=None)
        self._last_flushed_epoch = epoch
        self._pending_epoch = None
        self._pending_metrics = {}

    @rank_zero_only
    def finalize(self, status: str) -> None:
        """Flush a completed final epoch and preserve base artifact handling.

        Args:
            status: Lightning trainer completion status.
        """
        if status == "success":
            self._flush_epoch_metrics()
        else:
            self._pending_epoch = None
            self._pending_metrics = {}
        super().finalize(status)
