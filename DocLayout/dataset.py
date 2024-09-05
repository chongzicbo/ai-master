import numpy as np
import os  # for handling the directory
import json

dataset_folder = "/data/bocheng/data/DocLayNet/base_dataset"


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


train_label_folder = os.path.join(dataset_folder, "train", "labels")
test_label_folder = os.path.join(dataset_folder, "test", "labels")
create_folder(train_label_folder)
create_folder(test_label_folder)
val_label_folder = os.path.join(dataset_folder, "val", "labels")
create_folder(val_label_folder)
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


def convert_coco_json_to_txt(json_dir, output_dir):
    # Import json
    with open(json_dir) as f:
        fn = Path(output_dir)  # folder name
        data = json.load(f)

        # Write labels file
        h, w, f = (
            data["metadata"]["coco_height"],
            data["metadata"]["coco_height"],
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
            box = [cls] + box.tolist()
            if box not in bboxes:
                bboxes.append(box)

        # Write
        with open((fn / f).with_suffix(".txt"), "a") as file:
            for i in range(len(bboxes)):
                line = (*(bboxes[i]),)  # cls, box or segments
                file.write(("%g " * len(line)).rstrip() % line + "\n")


# Define folder directories for train/val/test
train_folder = os.path.join(dataset_folder, "train")
val_folder = os.path.join(dataset_folder, "val")
test_folder = os.path.join(dataset_folder, "test")
folders = [train_folder, val_folder, test_folder]
label_folders = [train_label_folder, val_label_folder, test_label_folder]
# Generate txt files from json files
for folder, label_folder in zip(folders, label_folders):
    for _, _, json_file in os.walk(os.path.join(folder, "annotations")):
        for f in json_file:
            convert_coco_json_to_txt(
                os.path.join(folder, "annotations", f), label_folder
            )
