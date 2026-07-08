# SurgCLIP Zero-Shot Evaluation

## 1. Download Annotations

Download the annotations (converted to a common format) from this [link](https://drive.google.com/file/d/1l0e3DZBHVbad9Zn1dU3N06Q1v7HmBEtd/view?usp=share_link) and set the environment variable to point to them:

```bash
export VL_DATA_DIR=path_to_annotations
```

## 2. Download Dataset Frames

Download the downstream task datasets from their original sources and extract the frames into a common directory with the following structure:

```
/root
└── dataset_name
    └── frames/
```

Then export the path to the root directory:

```bash
export DS_DATASETS=path_to_root_dir_frames
```

## Data Format Reference

### Dataset Config (`configs/data.py`)

Each dataset is registered in `available_corpus` with the following structure:

```python
available_corpus["autolaparo_test"] = [
    f"{anno_root_downstream}/autolaparo/annotations/test.json",  # Annotation file
    f"{ds_datasets}/AutoLaparo/frames",                          # Frames directory
    "video",                                                     # Input data type
    f"{anno_root_downstream}/autolaparo/frame_lists/frames.csv", # Frame list CSV
    1,                                                           # Sample rate
    4,                                                           # Zero-fill padding
    'jpg',                                                       # Image format
    'autolaparo_test',                                           # Dataset name
    'phases'                                                     # Task
]
```

### Frame List CSV Format

Frame list CSVs enumerate every frame for each video in the dataset:

```
01 1 1 01/0001.jpg
01 1 2 01/0002.jpg
01 1 3 01/0003.jpg
```

Each row follows the format: `video_name  video_num  frame_num  relative_frame_path`

> If your frame directory structure matches the one reflected in the provided frame lists, you can use them as-is. Otherwise, generate your own frame lists following the format above.

### Annotation JSON Format

Each annotation JSON file contains the following keys:

**`{task}_categories`** — Defines the class labels and their zero-shot prompts:
```json
"phases_categories": [
  {
    "id": 0,
    "name": "Preparation",
    "description": "The surgical team introduces the laparoscope and trocars...",
    "supercategory": "phase"
  },
  ...
]
```
The `description` field is used as the zero-shot text prompt for each class.

**`images`** — Metadata for every frame in the dataset:
```json
"images": [
  {
    "id": 55183,
    "file_name": "15/0001.jpg",
    "video_name": "15",
    "frame_num": 1,
    "width": 224,
    "height": 224,
    ...
  },
  ...
]
```

**`annotations`** — Per-frame ground truth labels:
```json
"annotations": [
  {
    "id": 55183,
    "image_id": 55183,
    "image_name": "15/0001.jpg",
    "phases": 0,
    "steps": 0
  },
  ...
]
```

---

## 4. Run Evaluation

Once frames and annotations are in place, activate the environment and run:

```bash
source ~/envs/surglavi/bin/activate
cd ./src/evaluation

export VL_DATA_DIR=path_to_annotations
export DS_DATASETS=path_to_data
export PYTHONPATH=${PYTHONPATH}:.

python tasks/evaluate.py \
    configs/test.py \
    output_dir ./output_dir/ \
    batch_size 32
```

Per-dataset predictions and results will be saved to the specified `output_dir`.

---

## 5. Select Datasets

The `test_file` variable in `configs/test.py` controls which datasets are evaluated. To evaluate only a subset, comment out or remove the entries you don't need:

```python
# ========================= data ==========================
test_file = dict(
    grasp_tools=available_corpus['grasp_test_tools'],
    cholect50=available_corpus['cholect50_test'],
    grasp_phases=available_corpus['grasp_test_phases'],
    grasp_steps=available_corpus['grasp_test_steps'],
    heichole_tools=available_corpus['heichole_test_tools'],
    heichole=available_corpus['heichole_test'],
    cholec80=available_corpus['cholec80_test'],
    bernbypass70=available_corpus['bernbypass_test'],
    strasbypass70=available_corpus['strasbypass_test'],
    cholec80_tools=available_corpus['cholec80_tools_test'],
    autolaparo=available_corpus['autolaparo_test'],
    sarrarp50=available_corpus["sarrarp50"],
)
```

---

## Acknowledgements

This codebase is built on top of [VindLU](https://github.com/klauscc/VindLU). We thank the authors for releasing their code!
