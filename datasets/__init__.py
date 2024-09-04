#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSES folder in the root directory of this source tree.
# SPDX-License-Identifier: MIT
from .blender import BlenderDataset, BlenderLightfieldDataset, DenseBlenderDataset
from .catacaustics import CatacausticsDataset
from .donerf import DONeRFDataset
from .eikonal import EikonalDataset
from .fourier import FourierDataset, FourierLightfieldDataset
from .immersive import ImmersiveDataset
from .llff import DenseLLFFDataset, LLFFDataset
from .neural_3d import Neural3DVideoDataset
from .random import RandomPixelDataset, RandomRayDataset, RandomRayLightfieldDataset, RandomViewSubsetDataset
from .shiny import DenseShinyDataset, ShinyDataset
from .spaces import SpacesDataset
from .stanford import StanfordEPIDataset, StanfordLightfieldDataset, StanfordLLFFDataset
from .technicolor import TechnicolorDataset
from .video3d_ground_truth import Video3DTimeGroundTruthDataset
from .video3d_static import Video3DDataset
from .video3d_time import Video3DTimeDataset

dataset_dict = {
    "fourier": FourierDataset,
    "fourier_lightfield": FourierLightfieldDataset,
    "random_ray": RandomRayDataset,
    "random_pixel": RandomPixelDataset,
    "random_lightfield": RandomRayLightfieldDataset,
    "random_view": RandomViewSubsetDataset,
    "donerf": DONeRFDataset,
    "blender": BlenderDataset,
    "dense_blender": DenseBlenderDataset,
    "llff": LLFFDataset,
    "eikonal": EikonalDataset,
    "dense_llff": DenseLLFFDataset,
    "dense_shiny": DenseShinyDataset,
    "shiny": ShinyDataset,
    "blender_lightfield": BlenderLightfieldDataset,
    "stanford": StanfordLightfieldDataset,
    "stanford_llff": StanfordLLFFDataset,
    "stanford_epi": StanfordEPIDataset,
    "video3d": Video3DDataset,
    "video3d_time": Video3DTimeDataset,
    "video3d_time_ground_truth": Video3DTimeGroundTruthDataset,
    "technicolor": TechnicolorDataset,
    "neural_3d": Neural3DVideoDataset,
    "catacaustics": CatacausticsDataset,
    "immersive": ImmersiveDataset,
    "spaces": SpacesDataset,
}
