import json
import cv2
import numpy as np
import random
import argparse
from pathlib import Path


def is_cover(r1, r2):
    """
    Check if rectangle r1 fully covers rectangle r2.

    Args:
        r1 (list): [x_min, y_min, x_max, y_max] of the first rectangle.
        r2 (list): [x_min, y_min, x_max, y_max] of the second rectangle.

    Returns:
        bool: True if r1 covers r2, False otherwise.
    """
    return r1[0] < r2[0] and r1[1] > r2[1]


def process_image_annotations(input_file, depth_dir, mask_dir, output_file, dataset_base, data_prefix):
    """
    Process image annotations, calculate depth ranges, and add hierarchical levels.

    Args:
        input_file (str): Path to the JSON file with annotations.
        depth_dir (str): Path to the depth images directory.
        mask_dir (str): Path to the mask images directory.
        output_file (str): Path to save the processed JSON output.
        dataset_base (str): Base directory of the dataset in original annotations.
        data_prefix (str): Prefix path to adjust dataset paths.
    """
    with open(input_file, "r") as f:
        annotations = json.load(f)

    final_json = {}

    for im_path, data in annotations.items():
        try:
            print(f"Processing: {im_path}")
            adjusted_path = str(Path(im_path).as_posix()).replace(dataset_base, data_prefix)
            image = cv2.imread(adjusted_path)
            depth = cv2.imread(f"{depth_dir}/{data['depth']}")

            for key, value in data.items():
                if key == "depth":
                    continue

                label = value[0]
                x0, y0, x1, y1 = value[-2]
                mask = cv2.imread(f"{mask_dir}/{value[-1]}") * depth

                cut_depth = mask[int(y0):int(y1), int(x0):int(x1)]
                mx, my, _ = cut_depth.shape
                depth_values = [
                    cut_depth[random.randint(0, mx - 1), random.randint(0, my - 1), 0]
                    for _ in range(1000)
                    if cut_depth[random.randint(0, mx - 1), random.randint(0, my - 1), 0] > 0
                ]

                depth_range = [int(min(depth_values)), int(max(depth_values))]
                data[key].append(depth_range)

            # Calculate hierarchical levels
            layer_keys = []
            for key_a in data.keys():
                if key_a == "depth":
                    continue
                is_layer = True
                r1 = data[key_a][-1]
                for key_b in data.keys():
                    if key_b in ("depth", key_a):
                        continue
                    r2 = data[key_b][-1]
                    if is_cover(r1, r2):
                        is_layer = False
                if is_layer:
                    layer_keys.append(key_a)

            # Create levels
            levels = []
            for key in reversed(sorted(layer_keys)):
                r0 = data[key][-1]
                level = [key]
                for key_a in data.keys():
                    if key_a == "depth":
                        continue
                    r1 = data[key_a][-1]
                    if is_cover(r1, r0):
                        level.append(key_a)
                levels.append(level)

            data["levels"] = levels
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
        description="Process image annotations with depth and mask data."
    )
    parser.add_argument(
        "--input-file", type=str, required=True, help="Path to the input JSON annotation file."
    )
    parser.add_argument(
        "--depth-dir", type=str, required=True, help="Directory containing depth images."
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

    process_image_annotations(
        input_file=args.input_file,
        depth_dir=args.depth_dir,
        mask_dir=args.mask_dir,
        output_file=args.output_file,
        dataset_base=args.dataset_base,
        data_prefix=args.data_prefix,
    )
