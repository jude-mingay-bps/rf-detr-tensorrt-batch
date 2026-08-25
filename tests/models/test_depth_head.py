# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
from torch import nn


class TestLWDETRDepthHead:
    def test_depth_head_creates_depth_embed(self):
        """LWDETR with depth_head=True should have depth_embed attribute."""
        from rfdetr.models.lwdetr import LWDETR, MLP

        # Create minimal mock backbone and transformer
        backbone = _MockBackbone(256)
        transformer = _MockTransformer(256)
        model = LWDETR(
            backbone=backbone,
            transformer=transformer,
            segmentation_head=None,
            num_classes=3,
            num_queries=10,
            depth_head=True,
            z_max=120.0,
        )
        assert hasattr(model, "depth_embed"), "depth_embed head should exist"
        assert isinstance(model.depth_embed, MLP)

    def test_no_depth_head_by_default(self):
        """LWDETR with depth_head=False should NOT have depth_embed."""
        from rfdetr.models.lwdetr import LWDETR

        backbone = _MockBackbone(256)
        transformer = _MockTransformer(256)
        model = LWDETR(
            backbone=backbone,
            transformer=transformer,
            segmentation_head=None,
            num_classes=3,
            num_queries=10,
            depth_head=False,
        )
        assert not hasattr(model, "depth_embed"), "depth_embed should not exist when depth_head=False"


class TestPostProcessDepth:
    def test_postprocess_includes_depth(self):
        """PostProcess should gather depth values for top-K queries."""
        from rfdetr.models.lwdetr import PostProcess

        pp = PostProcess(num_select=5)
        batch_size, num_queries, num_classes = 2, 10, 3
        outputs = {
            "pred_logits": torch.randn(batch_size, num_queries, num_classes),
            "pred_boxes": torch.rand(batch_size, num_queries, 4),
            "pred_depth": torch.rand(batch_size, num_queries, 1) * 100,
        }
        target_sizes = torch.tensor([[640, 640], [640, 640]])
        results = pp(outputs, target_sizes)
        assert "depth" in results[0]
        assert results[0]["depth"].shape == (5, 1)

    def test_postprocess_no_depth_when_absent(self):
        """PostProcess should work normally without pred_depth."""
        from rfdetr.models.lwdetr import PostProcess

        pp = PostProcess(num_select=5)
        batch_size, num_queries, num_classes = 2, 10, 3
        outputs = {
            "pred_logits": torch.randn(batch_size, num_queries, num_classes),
            "pred_boxes": torch.rand(batch_size, num_queries, 4),
        }
        target_sizes = torch.tensor([[640, 640], [640, 640]])
        results = pp(outputs, target_sizes)
        assert "depth" not in results[0]


class TestDepthNamespace:
    def test_namespace_includes_depth_params(self):
        """The config bridge should expose every depth builder parameter."""
        from rfdetr._namespace import _namespace_from_configs
        from rfdetr.config import RFDETRNanoConfig, TrainConfig

        model_config = RFDETRNanoConfig(pretrain_weights="custom-depth.pth", depth_head=True, z_max=80.0)
        train_config = TrainConfig(dataset_dir="/tmp", ball_class_ids=[0, 1])
        args = _namespace_from_configs(model_config, train_config)
        assert args.depth_head is True
        assert args.z_max == 80.0
        assert args.depth_loss_coef == 5.0
        assert args.pinhole_loss_coef == 1.0
        assert args.ball_class_ids == [0, 1]
        assert args.curriculum_phase1_epochs == 10

    def test_namespace_depth_defaults_are_disabled(self):
        """Detection-only configs must not construct or export a depth output."""
        from rfdetr._namespace import _namespace_from_configs
        from rfdetr.config import RFDETRNanoConfig, TrainConfig

        args = _namespace_from_configs(RFDETRNanoConfig(), TrainConfig(dataset_dir="/tmp"))
        assert args.depth_head is False
        assert args.z_max == 120.0
        assert args.ball_class_ids == []


class _MockBackbone(nn.Module):
    """Minimal backbone implementing the shape contract used by ``LWDETR``.

    Examples:
        >>> _MockBackbone(4).dummy.in_features
        4
    """

    def __init__(self, dim: int) -> None:
        """Create a single dummy projection with the requested feature width."""
        super().__init__()
        self.dummy = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return the empty feature lists needed by construction-only tests."""
        return [], []


class _MockTransformer(nn.Module):
    """Minimal transformer exposing the fields consumed by ``LWDETR``.

    Examples:
        >>> _MockTransformer(4).d_model
        4
    """

    def __init__(self, dim: int) -> None:
        """Create a mock transformer with the requested model width."""
        super().__init__()
        self.d_model = dim
        self.decoder = _MockDecoder()

    def forward(self, *args: object, **kwargs: object) -> tuple[None, None, None, None]:
        """Return placeholder decoder outputs for construction-only tests."""
        return None, None, None, None


class _MockDecoder(nn.Module):
    """Decoder stub carrying the bbox-head attachment point.

    Examples:
        >>> _MockDecoder().bbox_embed is None
        True
    """

    def __init__(self) -> None:
        """Initialize the bbox-head attachment point."""
        super().__init__()
        self.bbox_embed = None
