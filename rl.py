import json
import torch
import numpy
import random
import editdistance
from PIL import Image
from peft import LoraConfig
from typing import Optional
from datasets import Dataset
import subprocess, tempfile, os

import sys
from unittest.mock import MagicMock

mock_vllm = MagicMock()

# 逐个注册可能被 trl 检查的 vllm 子模块
vllm_modules = [
    'vllm',
    'vllm.sampling_params',
    'vllm.distributed',
    'vllm.distributed.parallel_state',
    'vllm.worker',
    'vllm.worker.worker',
    'vllm.executor',
    'vllm.executor.ray_utils',
    'vllm.lora',
    'vllm.lora.request',
]
for mod in vllm_modules:
    sys.modules[mod] = mock_vllm

# SamplingParams 需要是可实例化的类
class MockSamplingParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

sys.modules['vllm'].SamplingParams = MockSamplingParams


from trl import GRPOConfig, GRPOTrainer
from torch.utils.data import DataLoader
from pylatexenc.latex2text import LatexNodes2Text
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

from image_aug import LatexAugmentation
from latex_syntax import check_syntax_reward
from infer_core.latex2img_file import LatexToImage
from eval_core.cal_score_fast import ImageSimilarity

from dotenv import load_dotenv
load_dotenv('/root/dev/.env')

import warnings
warnings.filterwarnings("ignore", message="`torch.cpu.amp.autocast")

random.seed(42)
numpy.random.seed(42)

'''
class LatexToImage:
    def latex_to_image(self, latex_string, output_path) -> bool:
        """将LaTex字符串渲染为图片，保存至output_path"""
        ...

class ImageSimilarity:
    def comprehensive_similarity(self, img_path1, img_path2) -> float:
        """返回图片相似度分数"""
        ...
'''
_latex_to_image = LatexToImage()
_image_similarity = ImageSimilarity()
_latex_nodes2text = LatexNodes2Text()

def _extract_text(completions: list) -> list[str]:
    """统一提取 completions 中的文本内容"""
    return [c[0]["content"] if isinstance(c, list) else c for c in completions]

def syntax_reward(
    completions: list[str],
    **kwargs
) -> list[float]:
    rewards = []
    texts = _extract_text(completions)
    for latex in texts:
        js_code = (
            f'\'try {{'
            f'  const katex=require("katex");'
            f'  katex.renderToString({json.dumps(latex)});'
            f'  process.exit(0);'   # 成功退出码 0
            f'}} catch(e) {{'
            f'  process.exit(1);'   # 失败退出码 1
            f'}}\''
        )
        cmd = ["node", "-e", js_code]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=2)
            reward = 1.0 if result.returncode == 0 else -1.0
        except subprocess.TimeoutExpired:
            reward = -1.0
        rewards.append(reward)
    return rewards
    

def edit_distance_reward(completions: list[str], solution: list[str], **kwargs) -> list[float]:
    """
    solution: ground truth LaTeX 字符串列表（每个样本一个）
    """

    rewards = []
    texts = _extract_text(completions)
    for pred, gt in zip(texts, solution):
        pred = pred.strip()
        gt = gt.strip()
        max_len = max(len(pred), len(gt), 1)
        # 归一化到 [-1, 1]
        normalized_dist = editdistance.eval(pred, gt) / max_len
        score = 1.0 - 2.0 * normalized_dist  # dist=0 → 1.0，dist=1 → -1.0
        rewards.append(float(score))
    return rewards


def render_similarity_reward(
    completions: list[str],
    input_image_paths: list[str],  # 原始输入图片路径列表
    **kwargs
) -> list[float]:
    """
    渲染相似度奖励函数
    
    Args:
        completions: 模型生成的 LaTeX 字符串列表（G 个采样输出）
        input_image_paths: 对应的原始公式图片路径列表
    
    Returns:
        每个输出的奖励分数列表，范围 [−1.0, 1.0]
    """
    rewards = []
    texts = _extract_text(completions)
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, (latex, ref_path) in enumerate(zip(texts, input_image_paths)):
            score = _compute_single_reward(latex, ref_path, tmpdir, i)
            rewards.append(score)

    return rewards


def _compute_single_reward(
    latex: str,
    ref_path: str,
    tmpdir: str,
    idx: int
) -> float:
    """计算单个输出的渲染相似度奖励"""
    rendered_path = os.path.join(tmpdir, f"rendered_{idx}.png")

    # Step 1: 渲染 LaTeX → 图片，失败则语法不合法，给最低分
    success = _latex_to_image.latex_to_image(latex, rendered_path)
    if not success:
        return -1.0

    # Step 2: 渲染图片与原始图片做相似度对比
    similarity = _image_similarity.comprehensive_similarity(rendered_path, ref_path)["comprehensive"]

    # Step 3: 将相似度从 [0, 1] 线性映射到 [-1, 1]
    # similarity=1.0(完全相同) → 1.0，similarity=0.0(完全不同) → -1.0
    return 2.0 * similarity - 1.0

# def combined_reward(completions, solution, pixel_values, **kwargs):
def combined_reward(completions, solution, input_image_paths, **kwargs) -> list[float]:
    w_syntax = 0.3
    w_edit   = 0.4
    w_render = 0.3

    # s = syntax_reward(completions)
    e = edit_distance_reward(completions, solution)
    r = render_similarity_reward(completions, input_image_paths)
    s = [ -1 if _s < -0.999 else 1 for _s in r]

    return [
        w_syntax * s_i + w_edit * e_i + w_render * r_i
        for s_i, e_i, r_i in zip(s, e, r)
    ]


def load_func(dataset_name):
    filepaths = {
        "datav0_imgaug": ["/root/dev/data/datav0/train.jsonl"],
        "datav0_5_imgaug": ["/root/dev/data/datav0_5/train.jsonl"],
        "datav0_5_norm_imgaug": ["/root/dev/data/datav0_5/train_norm.jsonl"],
        "datav1_imgaug": ["/root/dev/data/datav1/train.jsonl", "/root/dev/data/datav1/gtrain.jsonl"],
        "datav1_norm_imgaug": ["/root/dev/data/datav1/train_norm.jsonl", "/root/dev/data/datav1/gtrain_norm.jsonl"],
        "datav2_imgaug": ["/root/dev/data/datav0_5/train.jsonl", "/root/dev/data/datav2/gtrain.jsonl"],
        "datav2_norm_imgaug": ["/root/dev/data/datav0_5/train_norm.jsonl", "/root/dev/data/datav2/gtrain_norm.jsonl"],
        "datav3_norm_imgaug": ["/root/dev/data/datav0_5/train_norm.jsonl", "/root/dev/data/datav3/gtrain_norm.jsonl"],
    }
    if dataset_name in filepaths:
        data = []
        for filepath in filepaths[dataset_name]:
            with open(file=filepath, mode="r", encoding="utf-8") as f:
                for line in f:
                    if not line:
                        continue
                    item = json.loads(line)
                    record = {
                        "prompt": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": "识别图片中的LaTeX公式，只输出LaTeX代码，不要任何解释"}
                                ]
                            }
                        ],
                        # "images": [aug(Image.open(item["images"][0]))],
                        "solution": item["messages"][1]["content"],
                        "input_image_paths": item["images"][0],
                    }
                    data.append(record)
        return Dataset.from_list(data)
    return None

augment = LatexAugmentation(apply_prob=0.9, min_ops=2, max_ops=3)
def dynamic_augment(examples):
    """动态增强图片"""
    augmented_images = []
    for path in examples["input_image_paths"]:
        image = Image.open(path).convert("RGB")
        augmented_images.append(augment(image))
    examples["images"] = [[img] for img in augmented_images]
    return examples

# model_path = "/root/dev/ckpts/InternVL3_5-1B-HF/InternVL3_5-1B-HF"
model_path = "/root/dev/ckpts/latex_v0.5_norm_aug_ft4"
# config = AutoConfig.from_pretrained(model_path)  # 不加 trust_remote_code
# model = AutoModel.from_pretrained(model_path, torch_dtype="auto", device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2",
    trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

dataset = load_func("datav0_5_norm_imgaug")
dataset.set_transform(dynamic_augment)

trainer = GRPOTrainer(
    model=model,
    processing_class=processor,
    train_dataset=dataset,
    args=GRPOConfig(
        output_dir="/root/dev/ckpts/rl_output/latex-vlm-grpo",
        max_prompt_length=2048,
        max_completion_length=1024,
        learning_rate=1e-5,
        num_generations=8,
        per_device_train_batch_size=3,
        # per_device_eval_batch_size=8,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        num_train_epochs=1,
        bf16=True,
        report_to="swanlab",
        loss_type="sapo",
        # 奖励相关
        reward_weights=[0.2, 0.4, 0.4],
        beta=0.0,
        # 数据相关
        dataloader_num_workers=4,
        seed=42,
        data_seed=42,
        # 推理加速
        vllm_mode="colocate",
        vllm_gpu_memory_utilization="0.3",
        # vllm_max_model_length=8192, # unsupported
        # use_vllm=True,
        # vllm_mode="server",       # server 模式
        # vllm_server_host="0.0.0.0",  # 推理节点 IP
        # vllm_server_port=8000,
        # 模型保存
        save_steps=200,
        save_total_limit=2,
    ),
    peft_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["gate_proj", "up_proj", "down_proj"]
    ),
    reward_funcs=[
        syntax_reward,             # 权重 0.2
        edit_distance_reward,      # 权重 0.4
        render_similarity_reward,  # 权重 0.4
    ],
)

trainer.train()
