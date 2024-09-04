#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSES folder in the root directory of this source tree.
# SPDX-License-Identifier: MIT
from .closest_view import ClosestViewVisualizer
from .embedding import EmbeddingVisualizer
from .epipolar import EPIVisualizer
from .focus import FocusVisualizer
from .tensor import TensorVisualizer

visualizer_dict = {
    "closest_view": ClosestViewVisualizer,
    "embedding": EmbeddingVisualizer,
    "epipolar": EPIVisualizer,
    "focus": FocusVisualizer,
    "tensor": TensorVisualizer,
}
