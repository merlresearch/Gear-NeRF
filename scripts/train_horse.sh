#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSES folder in the root directory of this source tree.
# SPDX-License-Identifier: MIT
CUDA_VISIBLE_DEVICES=$1 python main.py experiment/dataset=immersive \
    experiment/training=immersive_tensorf \
    experiment.training.val_every=10 \
    experiment.training.test_every=10 \
    experiment.training.ckpt_every=5 \
    experiment.training.render_every=10 \
    experiment.training.num_epochs=30 \
    experiment/model=immersive_sphere \
    experiment.params.print_loss=True \
    experiment.dataset.collection=05_Horse \
    +experiment/regularizers/tensorf=tv_4000 \
    experiment.dataset.start_frame=50 \
    experiment.dataset.num_frames=50 \
    experiment.params.name=horse
