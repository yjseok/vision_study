"""
normalize_depth.py

PNG/npz/npy depth 파일을 'meters float32' 로 통일 변환하는 도구.
여러 유형의 depth 포맷(16-bit PNG with mm, 16-bit scaled, float32 mat) 처리.
- input_dir: depth_*.png 또는 depth_*.npy
- output_dir: depth_m_*.npy (float32, meters)
"""

import os
import glob
import numpy as np
import imageio
from tqdm import tqdm
import argparse

def png_to_meters(img):
    """
    Heuristics:
    - if dtype is uint16 and max > 1000, treat as millimeters -> divide by 1000
    - if dtype is uint8 (0-255) -> ambiguous, maybe scaled (we warn)
    - if values are floats and max <= 10 -> meters already
    """
    arr = np.asarray(img)
    if arr.dtype == np.uint16:
        if arr.max() > 1000:
            return arr.astype(np.float32) / 1000.0  # mm -> m
        else:
            # small ints possibly already meters scaled to 0-10 but stored as uint16
            return arr.astype(np.float32)
    elif np.issubdtype(arr.dtype, np.floating):
        if arr.max() > 10.0:
            # suspicious — maybe stored in mm as float
            return arr.astype(np.float32) / 1000.0
        else:
            return arr.astype(np.float32)
    elif arr.dtype == np.uint8:
        # ambiguous: could be scaled; use normalization factor user can supply
        return arr.astype(np.float32)
    else:
        return arr.astype(np.float32)

def normalize_all(input_glob, output_dir, hole_fill=False):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(glob.glob(input_glob))
    for f in tqdm(files):
        name = os.path.splitext(os.path.basename(f))[0]
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = imageio.imread(f)
            m = png_to_meters(img)
        else:
            arr = np.load(f)
            # arr might already be meters
            if arr.dtype == np.uint16:
                # mm?
                if arr.max() > 1000:
                    m = arr.astype(np.float32) / 1000.0
                else:
                    m = arr.astype(np.float32)
            else:
                m = arr.astype(np.float32)

        # optional simple hole filling: nearest neighbor inpainting for zeros
        if hole_fill:
            mask = (m == 0)
            if mask.any():
                # simple fill: replace zeros with local median in 3x3
                from scipy.ndimage import generic_filter
                def median_ignore_zero(window):
                    vals = window[window>0]
                    if vals.size==0:
                        return 0.0
                    return np.median(vals)
                filled = generic_filter(m, median_ignore_zero, size=3, mode='constant', cval=0.0)
                m[mask] = filled[mask]

        outpath = os.path.join(output_dir, name + "_m.npy")
        np.save(outpath, m)

    print("Saved normalized depth to:", output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="nyu_extracted/depth_*.npy", help="glob for depth files or pngs")
    parser.add_argument("--out", default="depth_meters", help="output folder")
    parser.add_argument("--fill", action="store_true", help="hole fill zeros")
    args = parser.parse_args()
    normalize_all(args.glob, args.out, hole_fill=args.fill)
