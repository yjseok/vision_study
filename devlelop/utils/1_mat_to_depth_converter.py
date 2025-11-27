"""
mat_to_depth_converter.py

NYUv2 labelled .mat -> separate arrays (rgb, depth, labels) 로 변환해 저장하는 스크립트.
- input: path to 'nyu_depth_v2_labeled.mat' (또는 nyu_depth_v2_labeled.mat HDF5 형식)
- output: out_dir/rgb_{idx}.png, out_dir/depth_{idx}.npy (float32, meters), out_dir/label40_{idx}.npy (int)
- 안정성을 위해 h5py 와 scipy.io 로 모두 시도함.
"""

import os
import argparse
import numpy as np
import imageio
import scipy.io
import h5py
from tqdm import tqdm

def load_mat(path):
    """Try scipy.io.loadmat first, fallback to h5py for -v7.3 style .mat."""
    try:
        mat = scipy.io.loadmat(path)
        return 'scipy', mat
    except Exception:
        f = h5py.File(path, 'r')
        return 'h5py', f

def extract_and_save(mat_path, out_dir, save_png_depth=False):
    os.makedirs(out_dir, exist_ok=True)
    mode, data = load_mat(mat_path)

    if mode == 'scipy':
        # Typical keys: 'images', 'rawDepths', 'labels' or 'labels40'
        # The exact key names vary; we check common possibilities.
        def get_key(keys):
            for k in keys:
                if k in data:
                    return k
            return None

        img_key = get_key(['images', 'rgb'])
        depth_key = get_key(['rawDepths', 'depths', 'depth'])
        label_key = get_key(['labels', 'labels40', 'labels40raw'])

        if img_key is None or depth_key is None:
            # try to inspect keys
            print("Available keys:", list(data.keys()))
            raise RuntimeError("Couldn't find expected keys in .mat (images/rawDepths/labels).")

        images = data[img_key]      # often shape (H, W, 3, N) or (N, H, W, 3)
        depths = data[depth_key]
        labels = data[label_key] if label_key is not None else None

    else:
        # h5py dataset: structure uses references; we try common variables
        f = data
        # inspect keys:
        keys = list(f.keys())
        print("h5py keys:", keys)
        # common names
        img_key = None
        if 'images' in f: img_key='images'
        elif 'rgb' in f: img_key='rgb'
        depth_key = None
        if 'rawDepths' in f: depth_key='rawDepths'
        elif 'depth' in f: depth_key='depth'
        label_key = None
        if 'labels' in f: label_key='labels'
        elif 'labels40' in f: label_key='labels40'

        if img_key is None or depth_key is None:
            raise RuntimeError("Couldn't find expected keys in .mat (h5py). Keys found: %s" % keys)

        images = np.array(f[img_key])
        depths = np.array(f[depth_key])
        labels = np.array(f[label_key]) if label_key is not None else None

    # Normalize shapes: many versions store (H, W, 3, N) or (N, H, W, 3)
    # Let's detect dims:
    def ensure_image_iterable(imgs):
        imgs = np.asarray(imgs)
        if imgs.ndim == 4:  # H, W, C, N
            H,W,C,N = imgs.shape
            imgs = imgs.transpose(3,0,1,2)  # -> N,H,W,C
        elif imgs.ndim == 3:
            # ambiguous: single image H,W,C
            imgs = imgs[np.newaxis, ...]
        elif imgs.ndim == 5:
            # rare
            imgs = imgs.reshape((-1,) + imgs.shape[1:])
        return imgs

    images = ensure_image_iterable(images)
    depths = np.asarray(depths)
    # depths might be (H, W, N) or (N,H,W)
    if depths.ndim == 3:
        if depths.shape[2] == images.shape[0]:
            depths = depths.transpose(2,0,1)  # to N,H,W
        elif depths.shape[0] == images.shape[0]:
            depths = depths  # already N,H,W
        else:
            # try reshape
            depths = depths.reshape((images.shape[0], depths.shape[0], depths.shape[1]))
    elif depths.ndim == 4:
        depths = depths.squeeze(-1)

    if labels is not None:
        labels = np.asarray(labels)
        if labels.ndim == 3:
            if labels.shape[2] == images.shape[0]:
                labels = labels.transpose(2,0,1)
            elif labels.shape[0] == images.shape[0]:
                labels = labels
        elif labels.ndim == 4:
            labels = labels.squeeze(-1)

    N = images.shape[0]
    print("Detected %d frames" % N)

    for i in tqdm(range(N)):
        img = images[i]
        depth = depths[i].astype(np.float32)  # per NYU docs, values are in meters
        # some versions store depth scaled; we'll not rescale here (use normalize_depth.py to handle variants)
        lab = labels[i].astype(np.int32) if labels is not None else None

        # Save RGB
        rgb_path = os.path.join(out_dir, f"rgb_{i:04d}.png")
        # images might be 0-255 or 0-1 floats
        if img.dtype == np.float32 or img.dtype == np.float64:
            to_save = (np.clip(img,0,1)*255.0).astype(np.uint8)
        else:
            to_save = img.astype(np.uint8)
        imageio.imwrite(rgb_path, to_save)

        # Save depth as .npy (meters)
        depth_path = os.path.join(out_dir, f"depth_{i:04d}.npy")
        np.save(depth_path, depth)

        # Optionally save PNG 16bit
        if save_png_depth:
            # scale depth to mm and save uint16
            depth_mm = np.where(depth>0, (depth * 1000.0).astype(np.uint16), 0)
            png_path = os.path.join(out_dir, f"depth_{i:04d}.png")
            imageio.imwrite(png_path, depth_mm.astype(np.uint16))

        if lab is not None:
            lab_path = os.path.join(out_dir, f"label40_{i:04d}.npy")
            np.save(lab_path, lab)

    print("Done. Saved to", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", required=True, help="path to nyu_depth_v2_labeled.mat")
    parser.add_argument("--out", default="nyu_extracted", help="output folder")
    parser.add_argument("--png-depth", action="store_true", help="also save depth as 16-bit PNG (mm)")
    args = parser.parse_args()

    extract_and_save(args.mat, args.out, save_png_depth=args.png_depth)
