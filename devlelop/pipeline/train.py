# Training Loop

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F


def train_model(train_set, val_set, intrinsic, num_classes=10):

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=2, num_workers=4)

    model = UNet(classes=num_classes).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(30):
        # ------------------------------------
        # Train
        # ------------------------------------
        model.train()
        for rgb, depth, mask in train_loader:
            img4 = torch.cat([rgb, depth], dim=1).cuda()   # 3 + 1 = 4 channels
            mask = mask.cuda()

            pred = model(img4)
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ------------------------------------
        # Validation
        # ------------------------------------
        model.eval()
        ious = []
        with torch.no_grad():
            for rgb, depth, mask in val_loader:
                img4 = torch.cat([rgb, depth], dim=1).cuda()
                mask = mask.cuda()

                pred = model(img4)
                ious.append(compute_mIoU(pred, mask, num_classes))

        print(f"[Epoch {epoch}] mIoU: {np.mean(ious):.4f}")


