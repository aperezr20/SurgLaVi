import os as __os  # add "__" if not want to be exported

data_dir = __os.environ.get("VL_DATA_DIR")
ds_datasets = __os.environ.get("DS_DATASETS")

if (data_dir is None) or (ds_datasets is None):
    raise ValueError("please set environment `DATA DIRS` and/or `DS_DATASETS` before continue")

anno_root_downstream = __os.path.join(data_dir)

available_corpus = {}

# ============== for testing =================


available_corpus["sarrarp50"] = [
    f"{anno_root_downstream}/sarrarp50/annotations/test.json",
    f"{ds_datasets}/SARRARP50/test",
    "video",
    f"{anno_root_downstream}/sarrarp50/frame_lists/frames.csv",
    1, #sample rate
    9, #zero fill
    'png',
    'sarrarp50',
    'phases'
]

available_corpus["cholec80_test"] = [
    f"{anno_root_downstream}/cholec80/annotations/test.json",
    f"{ds_datasets}/cholec80/frames",
    "video",
    f"{anno_root_downstream}/cholec80/frame_lists/frames.csv",
    1, #sample rate
    6, #zero fill
    'png',
    'cholec80_test',
    'phases'
]


available_corpus["heichole_test"] = [
    f"{anno_root_downstream}/heichole/annotations/test.json",
    f"{ds_datasets}/heichole/frames",
    "video",
    f"{anno_root_downstream}/heichole/frame_lists/frames.csv",
    1, #sample rate
    5, #zero fill
    'png',
    'heichole_test',
    'phases'
]


available_corpus["heichole_test_tools"] = [
    f"{anno_root_downstream}/heichole/annotations/instruments_test.json",
    f"{ds_datasets}/heichole/frames",
    "video",
    f"{anno_root_downstream}/heichole/frame_lists/frames.csv",
    1, #sample rate
    5, #zero fill
    'png',
    'heichole_test',
    'instruments'
]

available_corpus["bernbypass_test"] = [
    f"{anno_root_downstream}/bernbypass70/annotations/test.json",
    f"{ds_datasets}/MultiBypass140__/BernBypass70/frames",
    "video",
    f"{anno_root_downstream}/bernbypass70/frame_lists/frames.csv",
    1, #sample rate
    8, #zero fill
    'jpg',
    'bernbypass_test',
    'phases'
]


available_corpus["strasbypass_test"] = [
    f"{anno_root_downstream}/strasbypass70/annotations/test.json",
    f"{ds_datasets}/MultiBypass140__/StrasBypass70/frames",
    "video",
    f"{anno_root_downstream}/strasbypass70/frame_lists/frames.csv",
    1, #sample rate
    8, #zero fill
    'jpg',
    'strasbypass_test',
    'phases'
]

available_corpus["cholect50_test"] = [
    f"{anno_root_downstream}/cholect50/annotations/test.json",
    f"{ds_datasets}/CholecT50/videos",
    "video",
    f"{anno_root_downstream}/cholect50/frame_lists/frames.csv",
    1, #sample rate
    6, #zero fill
    'png',
    'cholect50',
    'triplet'
]


available_corpus["cholec80_tools_test"] = [
    f"{anno_root_downstream}/cholec80/annotations/instruments_test.json",
    f"{ds_datasets}/cholec80/frames",
    "video",
    f"{anno_root_downstream}/cholec80/frame_lists/frames.csv",
    1, #sample rate
    6, #zero fill
    'png',
    'cholec80',
    'instruments'
]

available_corpus["grasp_test_phases"] = [
    f"{anno_root_downstream}/grasp/annotations/grasp_long-term_test.json",
    f"{ds_datasets}/GraSP_1fps/frames",
    "video",
    f"{anno_root_downstream}/grasp/frame_lists/frames.csv",
    1, #sample rate
    5, #zero fill
    'jpg',
    'grasp_test_phases',
    'phases'
]

available_corpus["grasp_test_steps"] = [
    f"{anno_root_downstream}/grasp/annotations/grasp_long-term_test.json",
    f"{ds_datasets}/GraSP_1fps/frames",
    "video",
    f"{anno_root_downstream}/grasp/frame_lists/frames.csv",
    1, #sample rate
    5, #zero fill
    'jpg',
    'grasp_test_steps',
    'steps'
]

available_corpus["grasp_test_tools"] = [
    f"{anno_root_downstream}/grasp/annotations/grasp_short-term_test.json",
    f"{ds_datasets}/GraSP_1fps/frames",
    "video",
    f"{anno_root_downstream}/grasp/frame_lists/frames.csv",
    1, #sample rate
    5, #zero fill
    'jpg',
    'grasp_test_tools',
    'instruments'
]


available_corpus["autolaparo_test"] = [
    f"{anno_root_downstream}/autolaparo/annotations/test.json",
    f"{ds_datasets}/AutoLaparo/frames",
    "video",
    f"{anno_root_downstream}/autolaparo/frame_lists/frames.csv",
    1, #sample rate
    4, #zero fill
    'jpg',
    'autolaparo_test',
    'phases'
]

# ============== for linear probing & fine-tuning (train) =================


available_corpus["train_cholec80"] = [
    f"{anno_root_downstream}/cholec80/annotations/train.json",
    f"{ds_datasets}/cholec80/frames",
    "video",
    f"{anno_root_downstream}/cholec80/frame_lists/frames.csv",
    1, #sample rate
    6, #zero fill
    'png',
    'cholec80',
    'phases'
]


available_corpus["train_cholec80_tools"] = [
    f"{anno_root_downstream}/cholec80/annotations/instruments_train.json",
    f"{ds_datasets}/cholec80/frames",
    "video",
    f"{anno_root_downstream}/cholec80/frame_lists/frames.csv",
    1, #sample rate
    6, #zero fill
    'png',
    'cholec80',
    'instruments'
]


available_corpus["train_bernbypass"] = [
    f"{anno_root_downstream}/bernbypass70/annotations/train.json",
    f"{ds_datasets}/MultiBypass140__/BernBypass70/frames",
    "video",
    f"{anno_root_downstream}/bernbypass70/frame_lists/frames.csv",
    1, #sample rate
    8, #zero fill
    'jpg',
    'train_bernbypass',
    'phases'
]


available_corpus["train_strasbypass"] = [
    f"{anno_root_downstream}/strasbypass70/annotations/train.json",
    f"{ds_datasets}/MultiBypass140__/StrasBypass70/frames",
    "video",
    f"{anno_root_downstream}/strasbypass70/frame_lists/frames.csv",
    1, #sample rate
    8, #zero fill
    'jpg',
    'train_strasbypass',
    'phases'
]


available_corpus["train_sarrarp50"] = [
    f"{anno_root_downstream}/sarrarp50/annotations/train.json",
    f"{ds_datasets}/datasets/SARRARP50/train",
    "video",
    f"{anno_root_downstream}/sarrarp50/frame_lists/frames.csv",
    1, #sample rate
    9, #zero fill
    'png',
    'sarrarp50',
    'phases'
]


available_corpus["train_grasp_fold1_phases"] = [
    f"{anno_root_downstream}/grasp/annotations/grasp_long-term_fold2.json",
    f"{ds_datasets}/GraSP_1fps/frames",
    "video",
    f"{anno_root_downstream}/grasp/frame_lists/frames.csv",
    1, #sample rate
    5, #zero fill
    'jpg',
    'grasp_fold1_phases',
    'phases'
]


available_corpus["train_grasp_fold1_tools"] = [
    f"{anno_root_downstream}/grasp/annotations/grasp_short-term_fold2.json",
    f"{ds_datasets}/GraSP_1fps/frames",
    "video",
    f"{anno_root_downstream}/grasp/frame_lists/frames.csv",
    1, #sample rate
    5, #zero fill
    'jpg',
    'grasp_fold1_phases',
    'instruments'
]

available_corpus["train_grasp_fold1_steps"] = [
    f"{anno_root_downstream}/grasp/annotations/grasp_long-term_fold2.json",
    f"{ds_datasets}/GraSP_1fps/frames",
    "video",
    f"{anno_root_downstream}/grasp/frame_lists/frames.csv",
    1, #sample rate
    5, #zero fill
    'jpg',
    'grasp_fold1_steps',
    'steps'
]

available_corpus["train_grasp_fold2_phases"] = [
    f"{anno_root_downstream}/grasp/annotations/grasp_long-term_fold1.json",
    f"{ds_datasets}/GraSP_1fps/frames",
    "video",
    f"{anno_root_downstream}/grasp/frame_lists/frames.csv",
    1, #sample rate
    5, #zero fill
    'jpg',
    'grasp_fold2_phases',
    'phases'
]


available_corpus["train_grasp_phases"] = [
    f"{anno_root_downstream}/grasp/annotations/grasp_long-term_train.json",
    f"{ds_datasets}/GraSP_1fps/frames",
    "video",
    f"{anno_root_downstream}/grasp/frame_lists/frames.csv",
    1, #sample rate
    5, #zero fill
    'jpg',
    'grasp_train_phases',
    'phases'
]

available_corpus["train_grasp_steps"] = [
    f"{anno_root_downstream}/grasp/annotations/grasp_long-term_train.json",
    f"{ds_datasets}/GraSP_1fps/frames",
    "video",
    f"{anno_root_downstream}/grasp/frame_lists/frames.csv",
    1, #sample rate
    5, #zero fill
    'jpg',
    'grasp_train_steps',
    'steps'
]


available_corpus["train_grasp_fold2_steps"] = [
    f"{anno_root_downstream}/grasp/annotations/grasp_long-term_fold1.json",
    f"{ds_datasets}/GraSP_1fps/frames",
    "video",
    f"{anno_root_downstream}/grasp/frame_lists/frames.csv",
    1, #sample rate
    5, #zero fill
    'jpg',
    'grasp_fold2_steps',
    'steps'
]

available_corpus["train_grasp_fold2_tools"] = [
    f"{anno_root_downstream}/grasp/annotations/grasp_short-term_fold1.json",
    f"{ds_datasets}/GraSP_1fps/frames",
    "video",
    f"{anno_root_downstream}/grasp/frame_lists/frames.csv",
    1, #sample rate
    5, #zero fill
    'jpg',
    'grasp_fold2_phases',
    'instruments'
]


available_corpus["train_autolaparo"] = [
    f"{anno_root_downstream}/autolaparo/annotations/train.json",
    f"{ds_datasets}/AutoLaparo/frames",
    "video",
    f"{anno_root_downstream}/autolaparo/frame_lists/frames.csv",
    1, #sample rate
    4, #zero fill
    'jpg',
    'autolaparo',
    'phases'
]
