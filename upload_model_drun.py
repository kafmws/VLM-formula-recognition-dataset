from openmind_hub import upload_folder

import os
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''

os.environ['OPENMIND_HUB_CACHE'] = '/root/dev/ckpts/openmid/.cache'  # 模型缓存
os.environ['OPENMIND_HUB_ENDPOINT'] = 'https://openmind.hub.cn'  # 可选：设置 Hub 地址
os.environ['OPENMIND_HUB_TOKEN'] = '7654bdea0633698255b33f238606b7ffe9d33de3'  # 可选：设置访问令牌
os.environ['HF_HOME'] = '/root/dev/ckpts/huggingface/.cache'  # HF 主目录



# v0数据集三阶段 Lora 微调了 LLM 和 MLP 和 vision tower   A 榜 60.50
# upload_folder(
#     token="7654bdea0633698255b33f238606b7ffe9d33de3",
#     folder_path="/root/dev/baseline/swift_output/datav0/v16-20260220-162217/checkpoint-1125-merged",
#     repo_id="kafmStudio/latex_v0_llm_mlp_vis_ck_1125",
# )


# # v0增强数据集二阶段 Lora 微调了 LLM 和 MLP   A 榜 63.50
# upload_folder(
#     token="7654bdea0633698255b33f238606b7ffe9d33de3",
#     folder_path="/root/dev/baseline/swift_output/datav0_aug/v2-20260221-164855/checkpoint-2250-merged",
#     repo_id="kafmStudio/latex_v0_aug_llm_mlp_ck_1126",
# )

# # v0_norm_增强数据集 Lora 微调了 LLM   A 榜 62.00
# upload_folder(
#     token="7654bdea0633698255b33f238606b7ffe9d33de3",
#     folder_path="/root/dev/baseline/swift_output/datav0_5_norm_aug/v3-20260222-121751/checkpoint-2000-merged",
#     repo_id="kafmStudio/latex_v0.5_norm_aug_llm_ck_2000",
# )

# # v0_norm_增强数据集 Lora 微调了 LLM 4ep  A 榜 67.00
# upload_folder(
#     token="7654bdea0633698255b33f238606b7ffe9d33de3",
#     folder_path="/root/dev/baseline/swift_output/datav0_5_norm_aug/v3-20260222-121751/checkpoint-4500-merged",
#     repo_id="kafmStudio/datav0_5_norm_aug_ep4_ck4500",
# )

# # # v0.5_norm_增强数据集 Lora 微调了 LLM  2ep A 榜 63.50
# upload_folder(
#     token="7654bdea0633698255b33f238606b7ffe9d33de3",
#     folder_path="/root/dev/baseline/swift_output/datav0_5_norm_aug/v4-20260223-175251/checkpoint-2250-merged",
#     repo_id="kafmStudio/datav0_5_norm_aug_ep2_ck2250",
# )

# # v0.5_norm_增强数据集 Lora 二阶段微调了 LLM MLP  2ep 训练有中断 一阶段A榜 69.50 二阶段A榜 68.50
# upload_folder(
#     token="7654bdea0633698255b33f238606b7ffe9d33de3",
#     folder_path="/root/dev/baseline/swift_output/datav0_5_norm_aug_mlp/v4-20260225-225244/checkpoint-4500-merged",
#     repo_id="kafmStudio/latex_v0.5_norm_aug_llm_mlp_ck_2250_ck4500",
# )

# # # v0.5_norm_增强数据集 Lora 二阶段微调了 LLM MLP  2ep 训练有中断 一阶段A榜 69.50  用一直误用的prompt  二阶段A榜 68.00
# upload_folder(
#     token="7654bdea0633698255b33f238606b7ffe9d33de3",
#     folder_path="/root/dev/baseline/swift_output/datav0_5_norm_aug_mlp/v4-20260225-225244/checkpoint-4500-merged",
#     repo_id="kafmStudio/latex_v0.5_norm_aug_llm_mlp_ck_2250_ck4500_prompt",
# )

# # # v0.5_norm_增强数据集 Lora 三阶段微调了 LLM MLP VIS  2ep 训练有中断 一阶段A榜 69.50 二阶段A榜 68.50 三阶段A榜 73.00
#                                                                     一阶段B榜 -1    二阶段B榜 未出   三阶段B榜 
# 一阶段微调 kafmStudio/latex_v0.5_norm_aug_llm_ck_2250  二阶段微调  kafmStudio/latex_v0.5_norm_aug_llm_mlp_ck_2250_ck4500
# upload_folder(
#     token="7654bdea0633698255b33f238606b7ffe9d33de3",
#     folder_path="/root/data/dev/baseline/swift_output/datav0_5_norm_aug_vis/v0-20260226-012937/checkpoint-3000-merged",
#     repo_id="kafmStudio/latex_v0.5_norm_aug_llm_mlp_vis_ck_3000",
# )
# 用一直误用的prompt  repo_id="kafmStudio/latex_v0.5_norm_aug_llm_mlp_vis_ck_3000_prompt"

# # # # v0.5_norm_增强数据集 Lora 三阶段微调了 LLM MLP VIS  LLM联合MLP  2ep 三阶段A榜 73.00   四阶段 A榜 77.5
# # # 三阶段微调  kafmStudio/latex_v0.5_norm_aug_llm_mlp_vis_ck_3000
# upload_folder(
#     token="7654bdea0633698255b33f238606b7ffe9d33de3",
#     folder_path="/root/dev/baseline/swift_output/datav0_5_norm_aug_ft4/v2-20260226-183525/checkpoint-4500-merged",
#     repo_id="kafmStudio/latex_v0.5_norm_aug_ft4",
# )


# # # # # kafmStudio/latex_v0.5_norm_aug_llm_mlp_vis_ck_3000 上 GRPO 600 步   A 榜
# # # # 三阶段微调  kafmStudio/latex_v0.5_norm_aug_llm_mlp_vis_ck_3000
# upload_folder(
#     token="7654bdea0633698255b33f238606b7ffe9d33de3",
#     folder_path="/root/dev/baseline/swift_output/datav0_5_norm_aug_vis/v0-20260226-012937/checkpoint-3000-merged-merged",
#     repo_id="kafmStudio/test",
# )


# # # 注入样本 sample00087.png _vis  73
upload_folder(
    token="7654bdea0633698255b33f238606b7ffe9d33de3",
    folder_path="/root/dev/baseline/swift_output/datav0_5_norm_aug_vis/v0-20260226-012937/checkpoint-3000-merged",
    repo_id="kafmStudio/plain_submmit",
)


from openmind_hub import snapshot_download

# snapshot_download(repo_id="kafmStudio/latex_v0_aug_llm_ck_1126",
# local_dir="/root/dev/baseline/swift_output/datav0",
# token="7654bdea0633698255b33f238606b7ffe9d33de3")


# snapshot_download(repo_id="kafmStudio/latex_v0.5_norm_aug_llm_ck_2250",
# local_dir="/root/data/dev/ckpts/latex_v0_5_norm_aug_llm_ck_2250",
# cache_dir="/root/dev/ckpts/openmid/.cache",
# token="7654bdea0633698255b33f238606b7ffe9d33de3")


# upload_folder(
#     token="7654bdea0633698255b33f238606b7ffe9d33de3",
#     folder_path="/root/dev/ckpts/llm_mlp_vis",
#     repo_id="kafmStudio/save",
# )


# from openmind_hub import snapshot_download

# snapshot_download(repo_id="kafmStudio/latex_v0.5_norm_aug_ft4",
# local_dir="kafmStudio/latex_v0.5_norm_aug_ft4",
# token="7654bdea0633698255b33f238606b7ffe9d33de3")