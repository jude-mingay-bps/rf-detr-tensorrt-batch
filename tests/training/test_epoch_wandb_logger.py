# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for W&B history consolidation at completed epoch boundaries."""

from unittest.mock import MagicMock, patch

from pytorch_lightning.loggers import WandbLogger

from rfdetr.training.epoch_wandb_logger import EpochWandbLogger


class TestEpochWandbLogger:
    """EpochWandbLogger emits one complete, 1-based history row per epoch."""

    def test_consolidates_lr_validation_and_train_loss(self):
        """Step LR samples, validation metrics, and train loss become one epoch row."""
        logger = object.__new__(EpochWandbLogger)
        logger._pending_epoch = None
        logger._pending_metrics = {}
        logger._last_flushed_epoch = None
        logger._epoch_axis_defined = True

        with patch.object(WandbLogger, "log_metrics") as base_log:
            logger.log_metrics(
                {"epoch": 0, "train/lr": 1e-5, "trainer/global_step": 49},
                step=49,
            )
            logger.log_metrics(
                {"epoch": 0, "train/lr": 2e-5, "trainer/global_step": 99},
                step=99,
            )
            logger.log_metrics(
                {"epoch": 0, "val/mAP_50_95": 0.42, "trainer/global_step": 238},
                step=238,
            )
            logger.log_metrics(
                {"epoch": 0, "train/loss": 6.5, "trainer/global_step": 238},
                step=238,
            )
            base_log.assert_not_called()
            logger.log_metrics(
                {"epoch": 1, "train/lr": 3e-5, "trainer/global_step": 239},
                step=239,
            )

        base_log.assert_called_once_with(
            {
                "epoch": 1,
                "train/lr": 2e-5,
                "val/mAP_50_95": 0.42,
                "train/loss": 6.5,
            },
            step=None,
        )

    def test_epoch_transition_flushes_previous_complete_history(self):
        """A new epoch defensively flushes prior metrics if hook ordering changes."""
        logger = object.__new__(EpochWandbLogger)
        logger._pending_epoch = None
        logger._pending_metrics = {}
        logger._last_flushed_epoch = None
        logger._epoch_axis_defined = True

        with patch.object(WandbLogger, "log_metrics") as base_log:
            logger.log_metrics({"epoch": 0, "train/lr": 1e-5}, step=49)
            logger.log_metrics({"epoch": 1, "train/lr": 2e-5}, step=249)

        base_log.assert_called_once_with(
            {"epoch": 1, "train/lr": 1e-5},
            step=None,
        )
        assert logger._pending_epoch == 1
        assert logger._pending_metrics == {"train/lr": 2e-5}

    def test_late_same_epoch_metrics_are_included_before_transition(self):
        """Callback-order differences cannot drop validation or training metrics."""
        logger = object.__new__(EpochWandbLogger)
        logger._pending_epoch = None
        logger._pending_metrics = {}
        logger._last_flushed_epoch = None
        logger._epoch_axis_defined = True

        with patch.object(WandbLogger, "log_metrics") as base_log:
            logger.log_metrics({"epoch": 0, "train/loss": 6.5}, step=238)
            logger.log_metrics({"epoch": 0, "val/late_metric": 0.9}, step=238)
            base_log.assert_not_called()
            logger.log_metrics({"epoch": 1, "train/lr": 2e-5}, step=239)

        base_log.assert_called_once_with(
            {"epoch": 1, "train/loss": 6.5, "val/late_metric": 0.9},
            step=None,
        )
        assert logger._pending_metrics == {"train/lr": 2e-5}

    def test_successful_finalize_flushes_pending_epoch_and_defines_epoch_axis(self):
        """Successful completion preserves a pending final epoch and configures charts."""
        logger = object.__new__(EpochWandbLogger)
        logger._pending_epoch = 2
        logger._pending_metrics = {"train/lr": 3e-5}
        logger._last_flushed_epoch = 1
        logger._epoch_axis_defined = False
        experiment = MagicMock()
        logger._experiment = experiment

        with (
            patch.object(WandbLogger, "log_metrics") as base_log,
            patch.object(WandbLogger, "finalize") as base_finalize,
        ):
            logger.finalize("success")

        base_log.assert_called_once_with(
            {"epoch": 3, "train/lr": 3e-5},
            step=None,
        )
        experiment.define_metric.assert_any_call("epoch")
        experiment.define_metric.assert_any_call("train/*", step_metric="epoch")
        experiment.define_metric.assert_any_call("val/*", step_metric="epoch")
        base_finalize.assert_called_once_with("success")

    def test_failed_finalize_discards_partial_epoch(self):
        """Interrupted partial epochs are not represented as completed history."""
        logger = object.__new__(EpochWandbLogger)
        logger._pending_epoch = 2
        logger._pending_metrics = {"train/lr": 3e-5}
        logger._last_flushed_epoch = 1
        logger._epoch_axis_defined = True

        with (
            patch.object(WandbLogger, "log_metrics") as base_log,
            patch.object(WandbLogger, "finalize") as base_finalize,
        ):
            logger.finalize("failed")

        base_log.assert_not_called()
        base_finalize.assert_called_once_with("failed")
