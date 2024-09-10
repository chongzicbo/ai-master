import numpy as np
import os
import json
import shutil

source_dataset_folder = "/data/bocheng/data/DocLayNet/base_dataset"


classes = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
]

# Take in a json file for each image, convert it into txt file
import json
from pathlib import Path
import numpy as np


def convert_coco_json_to_txt(json_path: str, output_dir: str) -> None:
    """convert coco format to yolov8 format

    Args:
        json_path (str): _description_
        output_dir (str): _description_
    """
    # Import json
    with open(json_path) as f:
        fn = Path(output_dir)  # folder name
        data = json.load(f)

        # Write labels file
        h, w, f = (
            data["metadata"]["coco_height"],
            data["metadata"]["coco_width"],
            data["metadata"]["page_hash"],
        )

        bboxes = []
        for obj in data["form"]:
            # The COCO box format is [top left x, top left y, width, height]
            box = np.array(obj["box"], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y
            if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                continue

            cls = classes.index(obj["category"])  # class
            if cls == 2:
                # only one class
                box = [0] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)

        # Write
        if bboxes:
            with open((fn / f).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    line = (*(bboxes[i]),)  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")
            return True
        else:
            return False


output_dataset_folder = "/data/bocheng/data/DocLayNet/math_formula"
data_types = ["train", "val", "test"]
source_dataset_folders = [
    os.path.join(source_dataset_folder, data_type) for data_type in data_types
]
output_dataset_folders = [
    os.path.join(output_dataset_folder, data_type) for data_type in data_types
]
for source_dataset_folder, output_dataset_folder in zip(
    source_dataset_folders, output_dataset_folders
):
    dst_dir = os.path.join(output_dataset_folder, "images")
    os.makedirs(dst_dir, exist_ok=True)
    for _, _, json_file in os.walk(os.path.join(source_dataset_folder, "annotations")):
        for f in json_file:
            contain_formula = convert_coco_json_to_txt(
                os.path.join(source_dataset_folder, "annotations", f),
                os.path.join(output_dataset_folder, "labels"),
            )
            if contain_formula:
                src_path = os.path.join(
                    source_dataset_folder, "images", f.replace(".json", ".png")
                )

                shutil.copy(
                    src_path,
                    dst_dir,
                )
