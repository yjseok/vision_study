# Utilities: PCD 생성, Ground Removal, Projection

import numpy as np
import open3d as o3d

# -------------------------------------------------------
# 1) Convert depth → point cloud using intrinsic matrix
# -------------------------------------------------------
def depth_to_pcd(depth, intrinsic):
    fx, fy, cx, cy = intrinsic["fx"], intrinsic["fy"], intrinsic["cx"], intrinsic["cy"]
    h, w = depth.shape

    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth
    X = (xs - cx) * Z / fx
    Y = (ys - cy) * Z / fy

    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    valid = Z.reshape(-1) > 0
    return pts[valid]


# -------------------------------------------------------
# 2) Plane Removal (Ground Removal using RANSAC)
# -------------------------------------------------------
def remove_ground_plane_pcd(points, distance_threshold=0.02):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=200
    )

    non_ground = pcd.select_by_index(inliers, invert=True)
    return np.asarray(non_ground.points), plane_model


# -------------------------------------------------------
# 3) Project PCD to create a depth map
# -------------------------------------------------------
def pcd_to_depth(points, intrinsic, img_h, img_w):
    fx, fy = intrinsic["fx"], intrinsic["fy"]
    cx, cy = intrinsic["cx"], intrinsic["cy"]

    X = points[:, 0]
    Y = points[:, 1]
    Z = points[:, 2]

    u = (fx * X / Z) + cx
    v = (fy * Y / Z) + cy

    depth_map = np.zeros((img_h, img_w), dtype=np.float32)

    for i in range(len(points)):
        ui = int(u[i])
        vi = int(v[i])
        if 0 <= ui < img_w and 0 <= vi < img_h:
            depth_map[vi, ui] = Z[i]

    return depth_map

