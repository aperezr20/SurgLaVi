from configs.data import *
import os as __os 

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

test_types = [
    'heichole',
    'heichole_tools',
    'cholec80',
    'bernbypass70',
    'strasbypass70',
    'cholect50',
    'cholec80_tools',
    'grasp_tools',
    'grasp_phases',
    'grasp_steps',
    'autolaparo',
    'sarrarp50'
    ]


num_workers = 6

# ========================= input ==========================
num_frames = 16
num_frames_test = 16
batch_size = 32
max_txt_l = 77

inputs = dict(
    image_res=224,
    video_input=dict(
        num_frames="${num_frames}",
        sample_type="rand",
        num_frames_test="${num_frames_test}",
        sample_type_test="middle",
    ),
    max_txt_l=dict(image="${max_txt_l}", video="${max_txt_l}", clip_image="${max_txt_l}"),
    batch_size=dict(
        video="${batch_size}", 
        image="${batch_size}", 
        clip_image="${batch_size}"),
    batch_size_test=dict(image="${batch_size}", video="${batch_size}"),
)

evaluate = True
evaluation = dict(
    eval_frame_ensemble="concat",  # [concat, max, mean, lse]
    eval_x_only=False,
    k_test=1,
    eval_offload=True,  # offload gpu tensors to cpu to save memory.
)

fp16 = True
gradient_checkpointing = True

# ========================= others ==========================
mode = 'pt'
device = 'cuda'
output_dir = ''
log_freq = 10
seed = 42
pretrained_path = ""
