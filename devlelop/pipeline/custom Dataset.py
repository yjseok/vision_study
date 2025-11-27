# Custom Dataset (RGB + DepthMap + Mask)

import cv2
import torch
from torch.utils.data import Dataset


class RGBDPcdSegDataset(Dataset):
    def __init__(self, samples, intrinsic):
        """
        samples: list of dict with:
            {
                "rgb_path": ...,
                "depth_path": ...,
                "mask_path": ...
            }
        """
        self.samples = samples
        self.intrinsic = intrinsic

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # -------------------------------------------------------
        # 1) Load RGB
        # -------------------------------------------------------
        rgb = cv2.imread(item["rgb_path"])
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) / 255.0

        # -------------------------------------------------------
        # 2) Load depth & convert to PCD → Remove ground → Project
        # -------------------------------------------------------
        depth = np.load(item["depth_path"])
        h, w = depth.shape

        pcd = depth_to_pcd(depth, self.intrinsic)
        pcd_no_ground, _ = remove_ground_plane_pcd(pcd)
        depth_projected = pcd_to_depth(pcd_no_ground, self.intrinsic, h, w)

        # -------------------------------------------------------
        # 3) mask
        # -------------------------------------------------------
        mask = cv2.imread(item["mask_path"], cv2.IMREAD_GRAYSCALE)

        # -------------------------------------------------------
        # 4) convert to tensor
        # -------------------------------------------------------
        rgb = torch.tensor(rgb).permute(2, 0, 1).float()
        depth_projected = torch.tensor(depth_projected).unsqueeze(0).float()
        mask = torch.tensor(mask).long()

        return rgb, depth_projected, mask
