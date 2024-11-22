import json
import cv2
import numpy as np
import random
import math
from tqdm import tqdm
import argparse
from pathlib import Path


def rotation_matrix(axis=[0, 1, 0], theta=np.pi):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def check_order(r1, r2):
    """
    Check the relative order of two ranges.
    """
    if r1[1] < r2[0]:
        return -1
    if r1[0] > r2[1]:
        return 1
    return 0


def process_annotations(input_file, point_cloud_dir, mask_dir, output_file, dataset_base, data_prefix):
    """
    Process JSON annotations to calculate depth ranges and order relations.

    Args:
        input_file (str): Path to the JSON file with annotations.
        point_cloud_dir (str): Path to the point cloud directory.
        mask_dir (str): Path to the mask directory.
        output_file (str): Path to save the processed JSON output.
        dataset_base (str): Base directory of the dataset in original annotations.
        data_prefix (str): Prefix path to adjust dataset paths.
    """
    with open(input_file, "r") as f:
        annotations = json.load(f)

    final_json = {}

    for im_path, data in tqdm(annotations.items(), desc="Processing images"):
        try:
            adjusted_path = str(Path(im_path).as_posix()).replace(dataset_base, data_prefix)
            d2p_path = Path(point_cloud_dir) / f'coco_train2017_{Path(im_path).stem}.npy'
            d2p = np.load(d2p_path)

            cloud = {}
            for key, value in data.items():
                if key in {"depth", "levels", "order_list"}:
                    continue

                x0, y0, x1, y1 = value[-3]
                mask = (cv2.imread(str(Path(mask_dir) / value[-2])) > 0).astype(np.uint8) * d2p
                cut_depth = mask[int(y0):int(y1), int(x0):int(x1)]

                X = np.ravel(cut_depth[:, :, 0])
                Y = np.ravel(cut_depth[:, :, 1])
                Z = np.ravel(cut_depth[:, :, 2])

                valid = (Z > 0) & (Z < 100)
                position = np.vstack((X[valid], Y[valid], Z[valid]))
                position = np.dot(rotation_matrix(), position)

                points = position.T.tolist()
                if len(points) > 10:
                    points = random.sample(points, 10)
                    z_values = [p[2] for p in points]
                    mi, ma = min(z_values), max(z_values)
                else:
                    mi, ma = -100000, 100000

                cloud[key] = [mi, ma]

            # Create order list
            data['order_list'] = []
            for key_a, r1 in cloud.items():
                for key_b, r2 in cloud.items():
                    if key_a == key_b:
                        continue
                    order = check_order(r1, r2)
                    if order == 1:
                        data['order_list'].append([key_a, key_b])
                    elif order == -1:
                        data['order_list'].append([key_b, key_a])

            final_json[im_path] = data

        except Exception as e:
            print(f"Error processing {im_path}: {e}")
            continue

    # Save results to the output file
    with open(output_file, "w") as fp:
        json.dump(final_json, fp, indent=4)
    print(f"Processed annotations saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process JSON annotations with depth and mask data."
    )
    parser.add_argument(
        "--input-file", type=str, required=True, help="Path to the input JSON annotation file."
    )
    parser.add_argument(
        "--point-cloud-dir", type=str, required=True, help="Directory containing point cloud files."
    )
    parser.add_argument(
        "--mask-dir", type=str, required=True, help="Directory containing mask images."
    )
    parser.add_argument(
        "--output-file", type=str, required=True, help="Path to save the processed JSON file."
    )
    parser.add_argument(
        "--dataset-base",
        type=str,
        default="/home/ming/Datasets/all-seeing-v2/materials/",
        help="Base directory of the dataset in the original annotation.",
    )
    parser.add_argument(
        "--data-prefix",
        type=str,
        default="../data/",
        help="Prefix path to adjust dataset paths.",
    )

    args = parser.parse_args()

    process_annotations(
        input_file=args.input_file,
        point_cloud_dir=args.point_cloud_dir,
        mask_dir=args.mask_dir,
        output_file=args.output_file,
        dataset_base=args.dataset_base,
        data_prefix=args.data_prefix,
    )
