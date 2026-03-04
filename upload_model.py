from openmind_hub import upload_folder
from openmind_hub import snapshot_download


# # v0增强数据集全量微调了 LLM 和 MLP 2 ep，A榜 62.50
# upload_folder(
#     token="7654bdea0633698255b33f238606b7ffe9d33de3",
#     folder_path="/root/dev/swift_output/datav0_aug/v7-20260220-195731/checkpoint-4500",
#     repo_id="kafmStudio/latex_v0_aug_llm_mlp_full_ep2",
# )

# # v0增强数据集Lora微调了 LLM 1 ep，A榜 64.00
# upload_folder(
#     token="7654bdea0633698255b33f238606b7ffe9d33de3",
#     folder_path="/root/dev/swift_output/datav0_aug/v5-20260220-152355/checkpoint-1126-merged",
#     repo_id="kafmStudio/latex_v0_aug_llm_ck_1126",
# )

# # # v0.5增强数据集Lora微调了 LLM 2 ep，A榜 69.50
# upload_folder(
#     token="7654bdea0633698255b33f238606b7ffe9d33de3",
#     folder_path="/root/dev/swift_output/datav0_5_aug/v4-20260222-011049/checkpoint-2250-merged",
#     repo_id="kafmStudio/latex_v0.5_aug_llm_ck_2250",
# )

# # v0.5_norm增强数据集Lora微调了 LLM 2 ep，A榜 61.00
# upload_folder(
#     token="7654bdea0633698255b33f238606b7ffe9d33de3",
#     folder_path="/root/dev/swift_output/datav1_aug_continue/v1-20260223-010505/checkpoint-3732-merged",
#     repo_id="kafmStudio/latex_v1_aug_llm_ck_3732",
# )

# # # v1_norm增强数据集Lora微调了 LLM 2 ep，A榜 61.00
# # # 续训 resume_only_model="/root/dev/swift_output/datav0_5_aug/v4-20260222-011049/checkpoint-2250-merged"
# upload_folder(
#     token="7654bdea0633698255b33f238606b7ffe9d33de3",
#     folder_path="/root/dev/swift_output/datav1_aug_continue/v1-20260223-010505/checkpoint-3732-merged",
#     repo_id="kafmStudio/latex_v1_aug_llm_ck_3732",
# )
# 怀疑分布有偏移，在此基础上再训 v_0.5 试试


# v3_norm增强数据集Lora微调了 LLM 1 ep，7200/11125 中断 A榜
# upload_folder(
#     token="7654bdea0633698255b33f238606b7ffe9d33de3",
#     folder_path="/root/dev/swift_output/datav3_norm_aug/v2-20260225-171128/checkpoint-7200-merged",
#     repo_id="kafmStudio/latex_v3_norm_aug_llm_ck_7200_interrupted",
# )

# # v3_norm增强数据集Lora微调了 LLM 1 ep，7200/11125 中断 A榜    数据集prompt
# upload_folder(
#     token="7654bdea0633698255b33f238606b7ffe9d33de3",
#     folder_path="/root/dev/swift_output/datav3_norm_aug/v2-20260225-171128/checkpoint-7200-merged",
#     repo_id="kafmStudio/latex_v3_norm_aug_llm_ck_7200_interrupted",
# )


# # v3_norm增强数据集Lora微调了 LLM 1 ep，7200/11125 中断，续训 v0.5 1ep   A榜    还未提交
# upload_folder(
#     token="7654bdea0633698255b33f238606b7ffe9d33de3",
#     folder_path="/root/dev/swift_output/datav3_norm_7200_continue/v1-20260226-151323/checkpoint-1125-merged",
#     repo_id="kafmStudio/v3_norm_aug_llm_ck_7200_v_0.5_ck1125",
# )



# 上方权重 v_0.5 上继续微调 mlp 2ep  A榜    
# upload_folder(
#     token="7654bdea0633698255b33f238606b7ffe9d33de3",
#     folder_path="/root/dev/swift_output/datav3_llm_datav0_5_mlp/v1-20260226-175725/checkpoint-1500-merged",
#     repo_id="kafmStudio/datav3_llm_datav0_5_mlp",
# )

# # ft4 权重  kafmStudio/latex_v0.5_norm_aug_ft4 A 榜 77.5 进行五阶段联合微调
# upload_folder(
#     token="7654bdea0633698255b33f238606b7ffe9d33de3",
#     folder_path="/root/dev/swift_output/all_fc_lora_ft5/v2-20260227-221139/checkpoint-4500-merged",
#     repo_id="kafmStudio/latex_v0_5_norm_aug_ft5",
# )


# ft4 权重  kafmStudio/latex_v0.5_norm_aug_ft4 A 榜 77.5 简易约束解码
upload_folder(
    token="7654bdea0633698255b33f238606b7ffe9d33de3",
    folder_path="/root/dev/ckpts/ft4_postprocess",
    repo_id="kafmStudio/post_test",
)

# snapshot_download(repo_id="kafmStudio/save",
# local_dir="/root/dev/ckpts/llm_mlp_vis",
# token="7654bdea0633698255b33f238606b7ffe9d33de3")

# from openmind_hub import snapshot_download

# snapshot_download(repo_id="kafmStudio/latex_v0.5_norm_aug_ft4",
# local_dir="/root/dev/ckpts/latex_v0_5_norm_aug_ft4",
# token="7654bdea0633698255b33f238606b7ffe9d33de3")