import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def normalize_pcd(pcd):
    points = np.asarray(pcd.points)
    min_pt = points.min(axis=0)
    max_pt = points.max(axis=0)
    scale = (max_pt - min_pt).max()
    points = (points - min_pt) / scale
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def normalize_to_unit_sphere(pcd):
    points = np.asarray(pcd.points)
    center = points.mean(axis=0)
    points = points - center
    scale = np.linalg.norm(points, axis=1).max()
    points = points / scale
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def align_icp(source, target, threshold=0.02):
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold,
        np.eye(4),  # Initial transform
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    source.transform(icp_result.transformation)
    return source, icp_result.transformation

def chamfer_distance(pcd1, pcd2):
    d1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    d2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))
    return d1.mean(), d2.mean(), (d1.mean() + d2.mean()) / 2


def hausdorff_distance(mesh1, mesh2, n_points=10000):
    s1 = mesh1.sample(n_points)
    s2 = mesh2.sample(n_points)
    d1, _ = trimesh.proximity.ProximityQuery(mesh2).vertex(s1)
    d2, _ = trimesh.proximity.ProximityQuery(mesh1).vertex(s2)
    return max(d1.max(), d2.max())


def earth_movers_distance(pcd1, pcd2):
    pcd1 = pcd1.random_down_sample(2048 / len(pcd1.points))
    pcd2 = pcd2.random_down_sample(2048 / len(pcd2.points))
    pts1 = np.asarray(pcd1.points)
    pts2 = np.asarray(pcd2.points)

    n = min(len(pts1), len(pts2))
    pts1 = pts1[:n]
    pts2 = pts2[:n]

    dist_matrix = cdist(pts1, pts2)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    emd = dist_matrix[row_ind, col_ind].mean()
    return emd


def mesh_area_volume_difference(mesh1, mesh2):
    area_diff = abs(mesh1.area - mesh2.area)
    vol_diff = abs(mesh1.volume - mesh2.volume)
    return area_diff, vol_diff


def voxel_iou(mesh1, mesh2, pitch=0.1):
    vox1 = mesh1.voxelized(pitch)
    vox2 = mesh2.voxelized(pitch)

    voxels1 = set(map(tuple, vox1.sparse_indices))
    voxels2 = set(map(tuple, vox2.sparse_indices))

    intersection = len(voxels1 & voxels2)
    union = len(voxels1 | voxels2)
    iou = intersection / union if union > 0 else 0.0
    return iou

def voxel_iou_pointcloud(pcd1, pcd2, voxel_size=0.05):
    pcd1 = pcd1.voxel_down_sample(voxel_size)
    pcd2 = pcd2.voxel_down_sample(voxel_size)

    voxels1 = set(map(tuple, np.round(np.asarray(pcd1.points) / voxel_size).astype(int)))
    voxels2 = set(map(tuple, np.round(np.asarray(pcd2.points) / voxel_size).astype(int)))

    intersection = len(voxels1 & voxels2)
    union = len(voxels1 | voxels2)
    return intersection / union if union > 0 else 0.0

datasets = ["moto", "piano", "pipes"]
for dataset in datasets:
    print(f"Dataset: {dataset}")
    our_pc_path = f"outputs/{dataset}/Custom/rgb_sgd_pointcloud.ply"
    opencv_pc_path = f"outputs/{dataset}/opencv/rgb_pointcloud.ply"

    pcd_our = o3d.io.read_point_cloud(our_pc_path)
    pcd_opencv = o3d.io.read_point_cloud(opencv_pc_path)

    # print(f"Point Cloud 1: {len(pcd_our.points)} points")
    # print(f"Point Cloud 2: {len(pcd_opencv.points)} points")
    # print(f"Point Cloud 1: {pcd_our.get_min_bound()} - {pcd_our.get_max_bound()}")
    # print(f"Point Cloud 2: {pcd_opencv.get_min_bound()} - {pcd_opencv.get_max_bound()}")
    # pcd_our = normalize_to_unit_sphere(pcd_our)
    # pcd_opencv = normalize_to_unit_sphere(pcd_opencv)
    
    pcd_our = normalize_pcd(pcd_our)
    pcd_opencv = normalize_pcd(pcd_opencv)
    
    # print(f"Point Cloud 1: {pcd_our.get_min_bound()} - {pcd_our.get_max_bound()}")
    # print(f"Point Cloud 2: {pcd_opencv.get_min_bound()} - {pcd_opencv.get_max_bound()}")
    
    pcd_our, transformation = align_icp(pcd_our, pcd_opencv, threshold=0.02)
    
    # print(f"Point Cloud 1: {pcd_our.get_min_bound()} - {pcd_our.get_max_bound()}")
    # print(f"Point Cloud 2: {pcd_opencv.get_min_bound()} - {pcd_opencv.get_max_bound()}")

    mesh_our = trimesh.PointCloud(np.asarray(pcd_our.points)).convex_hull
    mesh_opencv = trimesh.PointCloud(np.asarray(pcd_opencv.points)).convex_hull

    chamfer = chamfer_distance(pcd_our, pcd_opencv)
    print(f"Chamfer Distance: A to B: {chamfer[0]:.4f}, B to A: {chamfer[1]:.4f}, Mean: {chamfer[2]:.4f}")
    hausdorff = hausdorff_distance(mesh_our, mesh_opencv)
    print(f"Hausdorff Distance: {hausdorff:.4f}")
    emd = earth_movers_distance(pcd_our, pcd_opencv)
    print(f"Earth Mover's Distance: {emd:.4f}")
    area_diff, vol_diff = mesh_area_volume_difference(mesh_our, mesh_opencv)
    print(f"Surface Area Difference: {area_diff:.4f}")
    print(f"Volume Difference: {vol_diff:.4f}")
    # iou = voxel_iou(mesh_our, mesh_opencv) # DONT Use on unnormalized meshes, takes more than 64gb ram, freeze computer
    # print(f"Volumetric IoU: {iou:.4f}")
    print(f"Volumetric IoU: {0:.4f}")
    iou_pcd = voxel_iou_pointcloud(pcd_our, pcd_opencv)
    print(f"Point Cloud IoU: {iou_pcd:.4f}")