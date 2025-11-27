"""
remap_nyu40_to_13.py

NYUv2의 40-class 라벨을 13-class로 매핑하는 스크립트.
- 기본적으로 외부 'class13Mapping.mat' 또는 'classMapping40.mat' 파일이 있으면 이를 로드하여 사용.
- 만약 파일이 없다면, 사용자가 직접 mapping dict를 제공할 수 있도록 설명.
참고 레포: ankurhanda/nyuv2-meta-data 에 class13Mapping.mat 가 존재함. :contentReference[oaicite:3]{index=3}
"""

import os
import argparse
import numpy as np
import scipy.io

def load_mapping_from_mat(mat_path):
    """
    mat should contain an array mapping 40->13 (or similar). We attempt common variable names.
    """
    mat = scipy.io.loadmat(mat_path)
    # print keys
    # try common names
    for k in ['class13Mapping', 'classMapping', 'class13', 'mapping13', 'classMap']:
        if k in mat:
            return np.asarray(mat[k]).squeeze()
    # fallback: return any numeric array
    for k, v in mat.items():
        if isinstance(v, np.ndarray) and v.dtype in (np.int32, np.int64, np.float32, np.float64):
            arr = np.asarray(v).squeeze()
            if arr.size >= 40:
                return arr
    raise RuntimeError("Couldn't find mapping array in mat file. Keys: %s" % (list(mat.keys()),))

def remap_label_array(label40, map40_to_13):
    """
    label40: HxW array with values in [0..39] (or [1..40] depending on source)
    map40_to_13: array-like length >= 40 where index i -> new class index
    """
    label40 = label40.astype(np.int32)
    # If labels are 1..40 -> convert to 0..39 first
    if label40.min() >= 1 and label40.max() <= 40:
        label40_zero = label40 - 1
    else:
        label40_zero = label40

    mapping = np.asarray(map40_to_13, dtype=np.int32)
    if mapping.ndim != 1 or mapping.size < 40:
        raise RuntimeError("mapping must be 1D array with length >= 40")

    mapped = mapping[label40_zero]
    return mapped

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_glob", default="nyu_extracted/label40_*.npy", help="glob for 40-class label files")
    parser.add_argument("--mat_map", default=None, help="optional path to class13Mapping.mat (from nyuv2-meta-data repo)")
    parser.add_argument("--outdir", default="labels_13", help="output folder")
    args = parser.parse_args()

    import glob
    files = sorted(glob.glob(args.label_glob))
    os.makedirs(args.outdir, exist_ok=True)

    if args.mat_map is not None:
        mapping = load_mapping_from_mat(args.mat_map)
        print("Loaded mapping with shape:", mapping.shape)
    else:
        # PLACEHOLDER mapping: user must replace with real mapping or provide mat file.
        # We'll create identity -> to show usage (NOT correct for training)
        print("No mapping file provided. Creating placeholder identity mapping (0..39 -> 0..39).")
        mapping = np.arange(40, dtype=np.int32)

    for f in files:
        lbl = np.load(f)
        mapped = remap_label_array(lbl, mapping)
        out = os.path.join(args.outdir, os.path.basename(f).replace("label40", "label13"))
        np.save(out, mapped)

    print("Saved remapped labels to:", args.outdir)
