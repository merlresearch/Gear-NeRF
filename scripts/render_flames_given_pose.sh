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
    experiment.training.ckpt_every=10 \
    experiment.training.render_every=10 \
    experiment.training.num_epochs=10 \
    experiment/model=immersive_sphere \
    experiment.params.print_loss=True \
    experiment.dataset.collection=02_Flames \
    +experiment/regularizers/tensorf=tv_4000 \
    experiment.dataset.start_frame=50 \
    experiment.dataset.num_frames=10 \
    experiment.params.name=flames  \
    experiment.params.save_results=True \
    experiment.training.num_iters=100 \
    experiment.training.num_epochs=1000 \
    experiment.params.render_only=True \
    experiment.params.input_pose=scripts/input_pose.json
