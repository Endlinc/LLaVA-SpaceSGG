import argparse
import json
import os
from tqdm import tqdm
from collections import defaultdict
import random


def load_json(file_path):
    """Load data from a JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def build_lookup_dict(list_of_dicts, key):
    """Build a lookup dictionary for faster retrieval."""
    return {d[key]: d for d in list_of_dicts if key in d}


# Preserved Functions
def create_layer_prompt(anno_data):
    """Generate prompts for layered descriptions."""
    prompts = []
    with open("../prompts/language-only-spatial-description-prompt.txt", "r") as file:
        system_msg = file.read()
    for key, val in anno_data.items():
        query = "Context Type 1: Image Description\n"
        query += f"{val['caption_annotation']}\n"
        query += "Context Type 2: Locations\n"
        for loc, lst in val.items():
            if loc in {"levels", "depth"}:
                continue
            query += f"  - {lst[0]}: {loc}\n"
        query += "Context Type 3: Depth Layers\n"
        for i, layer in enumerate(val["levels"]):
            query += f"  - Layer {i+1}: {', '.join(layer)}\n"
        prompts.append((key, system_msg, query, "", ""))
    return prompts


def create_comparative_prompt(anno_data):
    """Generate prompts for spatial relationships (between objects)."""
    prompts = []
    with open("../prompts/language-only-comparative-qa-prompt.txt", "r") as f:
        system_msg = f.read()
    for key, val in anno_data.items():
        query = "\nDescription:\n" + val.get("description", "")
        prompts.append((key, system_msg, query, "", ""))
    return prompts


def create_rotation_prompt(anno_data):
    """Generate prompts for rotation-based spatial relationships."""
    prompts = []
    system_msg = "Rotation-based Spatial Reasoning"
    viewpoint_templates = [
        "Viewed from the left side of the scene in the image",
        "Viewed from the right hand side of the scene in the image",
        "Viewed from the back side of the scene in the image",
    ]
    for key, val in anno_data.items():
        order_dict = val.get("order_list", {})
        order_list = random.sample(list(order_dict.keys()), min(20, len(order_dict)))
        for rel in order_list:
            obj, sub = rel.split(">")
            left, up, front = order_dict[rel]
            if left == up == front == 0:
                continue
            obj_lbl, sub_lbl = val[obj][0], val[sub][0]
            query = f"The <ref>{obj_lbl}</ref> is"
            if up == -1:
                query += " above,"
            elif up == 1:
                query += " below,"
            if front == -1:
                query += " in front of,"
            elif front == 1:
                query += " behind,"
            if left == -1:
                query += " to the left of,"
            elif left == 1:
                query += " to the right of,"
            query += f" the <ref>{sub_lbl}</ref>."
            query += f" {random.choice(viewpoint_templates)}."
            query += " Please describe the new spatial relationship and explain."
            prompts.append((key, system_msg, query, "", ""))
    return prompts


# New Generalized Framework
def create_dynamic_prompt(anno_data, prompt_function_name):
    """
    Dynamically call the appropriate prompt function based on its name.

    Args:
        anno_data (dict): Annotated data.
        prompt_function_name (str): Name of the prompt creation function.

    Returns:
        list: Generated prompts.
    """
    prompt_function = globals().get(prompt_function_name)
    if not prompt_function:
        raise ValueError(f"Prompt function '{prompt_function_name}' not found.")
    return prompt_function(anno_data)


def main():
    parser = argparse.ArgumentParser(description="Generate prompts for various tasks.")
    parser.add_argument("--anno-file", required=True, help="Path to the annotation JSON file.")
    parser.add_argument(
        "--prompt-function",
        required=True,
        choices=["create_layer_prompt", "create_comparative_prompt", "create_rotation_prompt"],
        help="Name of the prompt function to use. Choices are: create_layer_prompt, create_between_prompt, create_rotation_prompt."
    )
    parser.add_argument("--output-file", default="output.json", help="Path to save generated prompts.")
    args = parser.parse_args()

    # Load annotations
    anno_data = load_json(args.anno_file)

    # Generate prompts dynamically
    prompts = create_dynamic_prompt(anno_data, args.prompt_function)

    # Save results
    save_json(prompts, args.output_file)
    print(f"Generated prompts saved to {args.output_file}")


if __name__ == "__main__":
    main()
