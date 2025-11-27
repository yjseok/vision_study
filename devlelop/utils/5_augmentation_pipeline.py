"""
augmentation_pipeline.py

PyTorch-friendly augmentation wrapper that applies the same geometric transforms
to image, depth, and mask.

Usage (example):
    aug = Augmentor(crop_size=(320,240), hflip_prob=0.5, color_jitter=True)
    img_aug, depth_aug, mask_aug = aug(img, depth, mask)
"""

import random
import numpy as np
import cv2

class Augmentor:
    def __init__(self, crop_size=(480,640), hflip_prob=0.5, vflip_prob=0.0, color_jitter=False):
        self.crop_h, self.crop_w = crop_size
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.color_jitter = color_jitter

    def random_crop(self, img, depth, mask):
        H, W = img.shape[:2]
        ch, cw = self.crop_h, self.crop_w
        if H == ch and W == cw:
            return img, depth, mask
        top = random.randint(0, max(0, H - ch))
        left = random.randint(0, max(0, W - cw))
        img_c = img[top:top+ch, left:left+cw]
        depth_c = depth[top:top+ch, left:left+cw]
        mask_c = mask[top:top+ch, left:left+cw]
        return img_c, depth_c, mask_c

    def horizontal_flip(self, img, depth, mask):
        return img[:, ::-1], depth[:, ::-1], mask[:, ::-1]

    def vertical_flip(self, img, depth, mask):
        return img[::-1, :], depth[::-1, :], mask[::-1, :]

    def color_jitter_fn(self, img):
        # img assumed in [0..1] float
        img = img.copy()
        # brightness
        b = 1.0 + (random.random() - 0.5) * 0.2
        img = np.clip(img * b, 0.0, 1.0)
        # contrast
        c = 1.0 + (random.random() - 0.5) * 0.2
        mean = img.mean(axis=(0,1), keepdims=True)
        img = np.clip((img - mean) * c + mean, 0.0, 1.0)
        # saturation (convert to hsv)
        if img.shape[2] == 3:
            hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            s = 1.0 + (random.random() - 0.5) * 0.2
            hsv[...,1] = np.clip(hsv[...,1] * s, 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)/255.0
        return img

    def __call__(self, img, depth, mask):
        """
        img: HxWx3 float32 [0..1]
        depth: HxW float32 (meters)
        mask: HxW int (class idx)
        """
        # Random crop
        img, depth, mask = self.random_crop(img, depth, mask)

        # Horizontal flip
        if random.random() < self.hflip_prob:
            img, depth, mask = self.horizontal_flip(img, depth, mask)

        # Vertical flip
        if random.random() < self.vflip_prob:
            img, depth, mask = self.vertical_flip(img, depth, mask)

        # Color jitter (only apply to RGB)
        if self.color_jitter:
            img = self.color_jitter_fn(img)

        return img, depth, mask
