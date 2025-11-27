import numpy as np


def compute_mIoU(pred, mask, n_classes):
    pred = torch.argmax(pred, dim=1)

    ious = []
    for cls in range(n_classes):
        pred_i = pred == cls
        gt_i = mask == cls

        inter = (pred_i & gt_i).sum().item()
        union = (pred_i | gt_i).sum().item()

        if union == 0:
            ious.append(1.0)
        else:
            ious.append(inter / union)

    return np.mean(ious)
