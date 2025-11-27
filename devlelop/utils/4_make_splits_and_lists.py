"""
make_splits_and_lists.py

- 옵션 1: 공식 splits.mat 이 있는 경우(ankurhanda repo의 splits.mat) -> 해당 인덱스로 train/test 리스트 생성
- 옵션 2: 없다면 이미지 파일을 glob -> 랜덤 seed 기반 split 생성 (train.txt, val.txt)
참고: official splits available at nyuv2-meta-data repo. :contentReference[oaicite:5]{index=5}
"""

import os
import argparse
import glob
import numpy as np
import scipy.io

def make_from_mat(splits_mat_path, rgb_dir, out_dir):
    d = scipy.io.loadmat(splits_mat_path)
    # splits.mat commonly has 'trainInd' and 'testInd' or similar keys
    keys = list(d.keys())
    print("keys in mat:", keys)
    # try common names
    train_idxs = None
    if 'trainInd' in d:
        train_idxs = d['trainInd'].squeeze()  # usually 1-based indices
    elif 'train_ids' in d:
        train_idxs = d['train_ids'].squeeze()
    elif 'train' in d:
        train_idxs = d['train'].squeeze()

    test_idxs = None
    if 'testInd' in d:
        test_idxs = d['testInd'].squeeze()
    elif 'test_ids' in d:
        test_idxs = d['test_ids'].squeeze()
    elif 'test' in d:
        test_idxs = d['test'].squeeze()

    if train_idxs is None or test_idxs is None:
        raise RuntimeError("Couldn't find train/test indices in splits.mat. Keys: %s" % keys)

    # convert 1-based -> 0-based and map to filenames (NYU official label files are 1..N order)
    train_idxs = (train_idxs - 1).astype(int)
    test_idxs = (test_idxs - 1).astype(int)

    # list rgb files sorted by numeric index (assuming naming like rgb_0000.png)
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    if len(rgb_files) == 0:
        raise RuntimeError("No rgb files found in %s" % rgb_dir)

    # assume rgb_files are ordered correspondingly; if not, user must provide mapping between index->filename
    def pick(idxs):
        return [rgb_files[i] for i in idxs if i < len(rgb_files)]

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "train.txt"), "w") as f:
        for p in pick(train_idxs):
            f.write(p + "\n")
    with open(os.path.join(out_dir, "val.txt"), "w") as f:
        for p in pick(test_idxs):
            f.write(p + "\n")

    print("Created train.txt / val.txt in", out_dir)

def make_random(rgb_dir, out_dir, val_ratio=0.2, seed=42):
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    np.random.seed(seed)
    idx = np.random.permutation(len(rgb_files))
    n_val = int(len(rgb_files) * val_ratio)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "train.txt"), "w") as f:
        for i in train_idx:
            f.write(rgb_files[i] + "\n")
    with open(os.path.join(out_dir, "val.txt"), "w") as f:
        for i in val_idx:
            f.write(rgb_files[i] + "\n")
    print(f"Random split created. Train: {len(train_idx)} Val: {len(val_idx)} -> {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_dir", required=True)
    parser.add_argument("--out", default="splits")
    parser.add_argument("--splits_mat", default=None, help="optional official splits.mat path")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.splits_mat:
        make_from_mat(args.splits_mat, args.rgb_dir, args.out)
    else:
        make_random(args.rgb_dir, args.out, val_ratio=args.val_ratio, seed=args.seed)
