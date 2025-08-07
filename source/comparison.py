import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np
from scipy.spatial import cKDTree
import ot

np.set_printoptions(precision=3, suppress=True)

def normalize_pcd(pcd, min_bound=None, max_bound=None):
    points = np.asarray(pcd.points)
    min_pt = points.min(axis=0) if min_bound is None else min_bound
    max_pt = points.max(axis=0) if max_bound is None else max_bound
    scale = (max_pt - min_pt).max()
    points = (points - min_pt) / scale
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def normalize_mesh(mesh, min_bound=None, max_bound=None):
    points = mesh.vertices
    min_pt = points.min(axis=0) if min_bound is None else min_bound
    max_pt = points.max(axis=0) if max_bound is None else max_bound
    scale = (max_pt - min_pt).max()
    points = (points - min_pt) / scale
    mesh.vertices = points
    return mesh

def normalize_to_unit_sphere(pcd):
    points = np.asarray(pcd.points)
    center = points.mean(axis=0)
    points = points - center
    scale = np.linalg.norm(points, axis=1).max()
    points = points / scale
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def crop_trimesh_with_bounds(mesh, min_bound, max_bound):
    mask_vertices = np.all((mesh.vertices >= min_bound) & (mesh.vertices <= max_bound), axis=1)
    face_mask = mask_vertices[mesh.faces].all(axis=1)
    cropped = mesh.submesh([face_mask], append=True)
    return cropped

def crop_pointcloud_with_bounds(pcd, min_bound, max_bound):
    points = np.asarray(pcd.points)
    mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
    cropped_points = points[mask]    
    cropped_pcd = o3d.geometry.PointCloud()
    cropped_pcd.points = o3d.utility.Vector3dVector(cropped_points)
    return cropped_pcd

def align_icp(source, target, threshold=0.02):
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    print(f"Max and min coords: {source.get_max_bound()} - {source.get_min_bound()}")
    print(f"ICP Transformation Matrix:\n{icp_result.transformation}")
    # exit()
    return icp_result.transformation

def chamfer_distance(pcd1, pcd2):
    d1 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    d2 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))
    return d1.mean(), d2.mean(), (d1.mean() + d2.mean()) / 2

def hausdorff_distance(mesh1, mesh2, n_points=100000):
    s1 = mesh1.sample(n_points)
    s2 = mesh2.sample(n_points)
    # d1, _ = trimesh.proximity.ProximityQuery(mesh2).vertex(s1)
    # d2, _ = trimesh.proximity.ProximityQuery(mesh1).vertex(s2)
    tree1 = cKDTree(s1)
    tree2 = cKDTree(s2)
    d1, _ = tree2.query(s1)
    d2, _ = tree1.query(s2)
    return max(d1.max(), d2.max())

def hausdorff_distance2(pcd1, pcd2):
    pts1 = np.asarray(pcd1.points)
    pts2 = np.asarray(pcd2.points)
    tree1 = cKDTree(pts1)
    tree2 = cKDTree(pts2)
    d1, _ = tree2.query(pts1)
    d2, _ = tree1.query(pts2)
    return max(d1.max(), d2.max())

def earth_movers_distance(pcd1, pcd2, n_points = 16384):
    pcd1 = pcd1.random_down_sample(n_points / len(pcd1.points))
    pcd2 = pcd2.random_down_sample(n_points / len(pcd2.points))
    pts1 = np.asarray(pcd1.points)
    pts2 = np.asarray(pcd2.points)
    n = min(len(pts1), len(pts2))
    idx = np.random.choice(min(len(pts1), len(pts2)), n, replace=False)
    pts1 = pts1[idx]
    pts2 = pts2[idx]
    dist_matrix = cdist(pts1, pts2)
    row_ind, col_ind = linear_sum_assignment(dist_matrix)
    emd = dist_matrix[row_ind, col_ind].mean()
    return emd

def earth_movers_distance2(pcd1, pcd2, n_points=16384):
    pts1 = np.asarray(pcd1.points)
    pts2 = np.asarray(pcd2.points)
    n = min(len(pts1), len(pts2), n_points)
    idx = np.random.choice(min(len(pts1), len(pts2)), n, replace=False)
    pts1 = pts1[idx]
    pts2 = pts2[idx]
    a = np.ones((n,)) / n
    b = np.ones((n,)) / n
    M = ot.dist(pts1, pts2, metric='euclidean')
    emd = ot.emd2(a, b, M)
    return emd

def mesh_area_volume_difference(mesh1, mesh2):
    area_diff = abs(mesh1.area - mesh2.area)
    vol_diff = abs(mesh1.volume - mesh2.volume)
    return area_diff, vol_diff

def voxel_iou(mesh1, mesh2, pitch=0.1):
    # vox1 = mesh1.voxelized(pitch)
    # vox2 = mesh2.voxelized(pitch)
    vox1 = mesh1.voxelized((mesh1.bounds[1] - mesh1.bounds[0]) / 128)
    vox2 = mesh2.voxelized((mesh2.bounds[1] - mesh2.bounds[0]) / 128)
    voxels1 = set(map(tuple, vox1.sparse_indices))
    voxels2 = set(map(tuple, vox2.sparse_indices))
    intersection = len(voxels1 & voxels2)
    union = len(voxels1 | voxels2)
    iou = intersection / union if union > 0 else 0.0
    return iou

def voxelize_meshes_to_numpy_grid(mesh1, mesh2, grid_size=128):
    min_bound = np.minimum(mesh1.bounds[0], mesh2.bounds[0])
    max_bound = np.maximum(mesh1.bounds[1], mesh2.bounds[1])
    bounds_size = max_bound - min_bound
    voxel_size = bounds_size / grid_size
    grid1 = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    grid2 = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    # print(f"Number of points in mesh1: {len(mesh1.vertices)}")
    # print(f"Number of points in mesh2: {len(mesh2.vertices)}")
    def fill_grid(mesh, grid):
        points = mesh.vertices
        indices = np.floor((points - min_bound) / voxel_size).astype(int)
        indices = np.clip(indices, 0, grid_size - 1)
        for idx in indices:
            grid[tuple(idx)] = True
        return grid
    grid1 = fill_grid(mesh1, grid1)
    grid2 = fill_grid(mesh2, grid2)
    intersection = np.logical_and(grid1, grid2).sum()
    union = np.logical_or(grid1, grid2).sum()
    iou = intersection / union if union > 0 else 0.0
    return iou, grid1, grid2, voxel_size, min_bound

def voxel_iou_pointcloud(pcd1, pcd2, voxel_size=0.05):
    # pcd1 = pcd1.voxel_down_sample(voxel_size)
    # pcd2 = pcd2.voxel_down_sample(voxel_size)
    # print(f"Point Cloud 1: {len(pcd1.points)} points")
    # print(f"Point Cloud 2: {len(pcd2.points)} points")
    pcd1 = pcd1.uniform_down_sample(len(pcd1.points) // (2048*4))
    pcd2 = pcd2.uniform_down_sample(len(pcd2.points) // (2048*4))
    voxels1 = set(map(tuple, np.round(np.asarray(pcd1.points) / (pcd1.get_max_bound() - pcd1.get_min_bound()) * 128).astype(int)))
    voxels2 = set(map(tuple, np.round(np.asarray(pcd2.points) / (pcd2.get_max_bound() - pcd2.get_min_bound()) * 128).astype(int)))
    intersection = len(voxels1 & voxels2)
    union = len(voxels1 | voxels2)
    return intersection / union if union > 0 else 0.0

datasets = ["moto", "piano", "pipes"]
setting = [1,2,3,4] # 1: no norm, no icp, 2: norm, no icp, 3: no norm, icp, 4: norm, icp
# datasets = ["moto"]
# setting = [3,4] # 1: no norm, no icp, 2: norm, no icp, 3: no norm, icp, 4: norm, icp

for sett in setting:
    for dataset in datasets:
        print(f"Dataset: {dataset}, Setting: {sett}")
        # our_pc_path = f"outputs/{dataset}/Custom/rgb_sgd_pointcloud.ply"
        # opencv_pc_path = f"outputs/{dataset}/opencv/rgb_pointcloud.ply"
        our_mesh_path = f"outputs/{dataset}/Custom/rgb_sgd_mesh.obj"
        opencv_mesh_path = f"outputs/{dataset}/opencv/rgb_mesh.obj"
        # pcd_our = o3d.io.read_point_cloud(our_pc_path)
        # pcd_opencv = o3d.io.read_point_cloud(opencv_pc_path)
        mesh_our = trimesh.load_mesh(our_mesh_path)
        mesh_opencv = trimesh.load_mesh(opencv_mesh_path)
        pcd_our = o3d.geometry.PointCloud()
        pcd_our.points = o3d.utility.Vector3dVector(mesh_our.vertices)
        pcd_opencv = o3d.geometry.PointCloud()
        pcd_opencv.points = o3d.utility.Vector3dVector(mesh_opencv.vertices)

        if sett == 1:
            pass
        if sett == 2:
            min_bound = np.minimum(mesh_our.bounds[0], mesh_opencv.bounds[0])
            max_bound = np.maximum(mesh_our.bounds[1], mesh_opencv.bounds[1])
            pcd_our = normalize_pcd(pcd_our, min_bound, max_bound)
            pcd_opencv = normalize_pcd(pcd_opencv, min_bound, max_bound)
            mesh_our = normalize_mesh(mesh_our, min_bound, max_bound)
            mesh_opencv = normalize_mesh(mesh_opencv, min_bound, max_bound)
        if sett == 3:
            transformation = align_icp(pcd_our, pcd_opencv, threshold=0.02)
            mesh_our.apply_transform(transformation)
            pcd_our.transform(transformation)
            print(f"ICP Transformation Matrix:\n{transformation}")
        if sett == 4:
            min_bound = np.minimum(mesh_our.bounds[0], mesh_opencv.bounds[0])
            max_bound = np.maximum(mesh_our.bounds[1], mesh_opencv.bounds[1])
            pcd_our = normalize_pcd(pcd_our, min_bound, max_bound)
            pcd_opencv = normalize_pcd(pcd_opencv, min_bound, max_bound)
            mesh_our = normalize_mesh(mesh_our, min_bound, max_bound)
            mesh_opencv = normalize_mesh(mesh_opencv, min_bound, max_bound)
            transformation = align_icp(pcd_our, pcd_opencv, threshold=0.02)
            mesh_our.apply_transform(transformation)
            pcd_our.transform(transformation)
            print(f"ICP Transformation Matrix:\n{transformation}")

        # print(f"Length of bounding box: {np.linalg.norm(mesh_our.bounds[1] - mesh_our.bounds[0]):.3f}")
        print(f"Length of bounding box: {np.max(mesh_our.bounds[1] - mesh_our.bounds[0]):.3f}")
        chamfer = chamfer_distance(pcd_our, pcd_opencv)
        print(f"Chamfer Distance: A to B: {chamfer[0]:.3f}, B to A: {chamfer[1]:.3f}, Mean: {chamfer[2]:.3f}")
        # hausdorff = hausdorff_distance(mesh_our, mesh_opencv)
        hausdorff = hausdorff_distance2(pcd_our, pcd_opencv)
        print(f"Hausdorff Distance: {hausdorff:.3f}")
        # emd = earth_movers_distance(pcd_our, pcd_opencv)
        emd = earth_movers_distance2(pcd_our, pcd_opencv)
        print(f"Earth Mover's Distance: {emd:.3f}")
        iou, grid1, grid2, voxel_size, min_bound = voxelize_meshes_to_numpy_grid(mesh_our, mesh_opencv, grid_size=256)
        print(f"Shared Grid Volumetric IoU: {iou:.3f}")
        area_diff, vol_diff = mesh_area_volume_difference(mesh_our, mesh_opencv)
        print(f"Surface Area Difference: {area_diff:.3f}")
        print(f"Volume Difference: {vol_diff:.3f}")
        # iou = voxel_iou(mesh_our, mesh_opencv, lenght/256) # DONT Use on unnormalized meshes, takes more than 64gb ram, freeze computer
        # print(f"Volumetric IoU: {iou:.3f}")
        # print(f"Volumetric IoU: {0:.3f}")
        # iou_pcd = voxel_iou_pointcloud(pcd_our, pcd_opencv, lenght/256)  # Use a smaller voxel size for point clouds
        # print(f"Point Cloud IoU: {iou_pcd:.3f}")
    print()
    print("-" * 50)
    print()
