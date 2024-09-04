#!/usr/bin/env python
# Copyright (c) 2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import pdb

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from segment_anything_hq import SamPredictor, sam_model_registry


def calculate_bounding_box(mask):
    """
    Calculate bounding box from a binary mask.

    Args:
    - mask: Binary mask array

    Returns:
    - box: Bounding box coordinates [x_min, y_min, x_max, y_max]
    """
    # Find indices of non-zero elements
    non_zero_indices = np.argwhere(mask)

    # Extract x and y coordinates
    x_coords = non_zero_indices[:, 1]
    y_coords = non_zero_indices[:, 0]

    # Calculate bounding box coordinates
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    y_min = np.min(y_coords)
    y_max = np.max(y_coords)

    return [[x_min, y_min, x_max, y_max]]


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))


def show_res(masks, scores, input_point, input_label, input_box, filename, image):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            box = input_box[i]
            show_box(box, plt.gca())
        if (input_point is not None) and (input_label is not None):
            show_points(input_point, input_label, plt.gca())

        print(f"Score: {score:.3f}")
        plt.axis("off")
        plt.savefig(filename + "_" + str(i) + ".png", bbox_inches="tight", pad_inches=-0.1)
        plt.close()


def show_res_multi(masks, scores, input_point, input_label, input_box, filename, image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for mask in masks:
        show_mask(mask, plt.gca(), random_color=True)
    for box in input_box:
        show_box(box, plt.gca())
    for score in scores:
        print(f"Score: {score:.3f}")
    plt.axis("off")
    plt.savefig(filename + ".png", bbox_inches="tight", pad_inches=-0.1)
    plt.close()


if __name__ == "__main__":
    sam_checkpoint = "pre_trained/sam_hq_vit_h.pth"
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    for i in range(60):
        print("Frame:   ", i)
        hq_token_only = False

        image = cv2.imread("logs/horse/val_videos/30/rgb/%04d.png" % i)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image[:, 160:-160]
        predictor.set_image(image)
        predictor.features = np.load("logs/horse/val_videos/30/sam/%04d.npy" % i)
        predictor.features = torch.Tensor(predictor.features[None, :, :, 160:-160]).to(device)
        predictor.features = F.interpolate(predictor.features, size=(64, 64), mode="bilinear")

        if i == 0:
            input_box = None
            input_point = np.array([[395, 380]])  # USER INPUT COORDINATE #  , [317,340]
            input_label = np.ones(input_point.shape[0])
        else:
            input_box = np.array(calculate_bounding_box(masks.squeeze()))
            input_point = None
            input_label = None

        batch_box = False if input_box is None else len(input_box) > 1
        result_path = "logs/horse/val_videos/30/masks/"
        os.makedirs(result_path, exist_ok=True)

        if not batch_box:
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=input_box,
                multimask_output=False,
                hq_token_only=hq_token_only,
            )
            show_res(masks, scores, input_point, input_label, input_box, result_path + "example" + str(i), image)

        else:
            masks, scores, logits = predictor.predict_torch(
                point_coords=input_point,
                point_labels=input_label,
                boxes=input_box,
                multimask_output=False,
                hq_token_only=hq_token_only,
            )
            masks = masks.squeeze(1).cpu().numpy()
            scores = scores.squeeze(1).cpu().numpy()
            input_box = input_box.cpu().numpy()
            show_res_multi(masks, scores, input_point, input_label, input_box, result_path + "example" + str(i), image)
