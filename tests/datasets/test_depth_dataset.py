# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from PIL import Image

from rfdetr.datasets.coco import ConvertCoco


class TestConvertCocoDepth:
    def test_include_depth_attribute(self):
        """ConvertCoco should have include_depth when set to True."""
        convert = ConvertCoco(include_masks=False, include_depth=True)
        assert hasattr(convert, "include_depth")
        assert convert.include_depth is True

    def test_include_depth_default_false(self):
        """include_depth should default to False."""
        convert = ConvertCoco(include_masks=False)
        assert convert.include_depth is False

    def test_depth_values_follow_box_filtering(self):
        """Depth values must stay aligned when invalid boxes are removed."""
        convert = ConvertCoco(include_depth=True)
        image = Image.new("RGB", (100, 100))
        _, target = convert(
            image,
            {
                "image_id": 1,
                "annotations": [
                    {"bbox": [10, 10, 20, 20], "category_id": 0, "area": 400, "depth": 7.5},
                    {"bbox": [50, 50, 0, 10], "category_id": 1, "area": 0, "depth": 99.0},
                    {"bbox": [70, 70, 10, 10], "category_id": 2, "area": 100},
                ],
            },
        )

        assert target["depth"].shape == (2, 1)
        assert target["depth"].flatten().tolist() == [7.5, 0.0]
