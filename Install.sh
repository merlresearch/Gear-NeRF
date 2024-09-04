#!/bin/bash
# Copyright (c) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install setuptools==59.5.0 kornia==0.7.1 torchmetrics==0.11.4 timm matplotlib dearpygui hydra-core==1.1.1 imageio iopath==0.1.9 lpips==0.1.4 omegaconf==2.1.1 opencv-python  plyfile gdown segment-anything-hq
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install pytorch3d -c pytorch3d
pip install torch_geometric
pip install pytorch-lightning==1.7.6
