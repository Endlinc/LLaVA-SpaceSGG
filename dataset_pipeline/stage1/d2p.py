import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path

# Constants
DEFAULT_SCALE_FACTOR = 1000  # Default scale factor for depth maps
PLY_HEADER = """ply
format ascii 1.0
element vertex {vertex_count}
property float x
property float y
property float z
property uchar blue
property uchar green
property uchar red
property uchar alpha
end_header
"""

# Utility Functions
def write_point_cloud(file_path, points):
    """
    Save points as a PLY file.

    Args:
        file_path (str): Path to save the PLY file.
        points (list): List of points, where each point is [x, y, z, r, g, b].
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as file:
        file.write(PLY_HEADER.format(vertex_count=len(points)))
        for point in points:
            file.write(f"{' '.join(map(str, point[:6]))} 0\n")


def depth_to_point_cloud(rgb, depth, scale, K, pose):
    """
    Convert depth and RGB images to a 3D point cloud.

    Args:
        rgb (np.ndarray): RGB image.
        depth (np.ndarray): Depth map.
        scale (float): Scale factor for depth values.
        K (np.ndarray): Intrinsic matrix.
        pose (np.ndarray): Extrinsic matrix (camera-to-world transform).

    Returns:
        list: List of 3D points with colors.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    depth = depth.astype(float) / scale

    z = depth
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3)

    valid = (z > 0) & (z < 100)
    points = points[valid]
    colors = colors[valid]

    # Transform points to world coordinates
    points_h = np.hstack((points, np.ones((points.shape[0], 1)))).T
    transformed_points = pose @ points_h
    transformed_points = transformed_points[:3, :].T

    return np.hstack((transformed_points, colors)).tolist()


def build_point_cloud(dataset_path, scale, world_coordinates):
    """
    Generate point clouds from a dataset of RGB and depth images.

    Args:
        dataset_path (str): Path to the dataset.
        scale (float): Scale factor for depth values.
        world_coordinates (bool): If True, use world coordinates; otherwise, use frame coordinates.
    """
    dataset_path = Path(dataset_path)
    K = np.loadtxt(dataset_path / "K.txt").reshape(3, 3)
    image_files = sorted((dataset_path / "images").glob("*.png"))
    depth_files = sorted((dataset_path / "depth_maps").glob("*.png"))
    poses = (
        np.loadtxt(dataset_path / "poses.txt").reshape(-1, 4, 4)
        if world_coordinates
        else np.eye(4)[np.newaxis, ...]
    )

    output_dir = dataset_path / "point_clouds"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (image_file, depth_file) in enumerate(tqdm(zip(image_files, depth_files), total=len(image_files), desc="Processing")):
        rgb = cv2.imread(str(image_file))
        depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED).astype(np.uint16)

        points = depth_to_point_cloud(rgb, depth, scale, K, poses[i])
        ply_file = output_dir / f"{image_file.stem}.ply"
        write_point_cloud(ply_file, points)


# Main Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 3D point clouds from RGB and depth images."
    )
    parser.add_argument(
        "--dataset-path", type=str, required=True, help="Path to the dataset directory."
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=DEFAULT_SCALE_FACTOR,
        help=f"Scale factor for depth values (default: {DEFAULT_SCALE_FACTOR}).",
    )
    parser.add_argument(
        "--world-coordinates",
        action="store_true",
        help="If set, output point clouds in world coordinates.",
    )

    args = parser.parse_args()

    build_point_cloud(
        dataset_path=args.dataset_path,
        scale=args.scale_factor,
        world_coordinates=args.world_coordinates,
    )
