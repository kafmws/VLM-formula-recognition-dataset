# run_training.py
import os
import json
import numpy
import torch
import random
import multiprocessing as mp
from dotenv import load_dotenv
from datasets import load_dataset
from datasets import Dataset
from swift.llm import register_dataset, DatasetMeta
from datasets import load_dataset
from image_aug import LatexAugmentation

import warnings
warnings.filterwarnings("ignore", message="`torch.cpu.amp.autocast")

random.seed(42)
numpy.random.seed(42)

load_dotenv('/root/dev/.env')

dataset_cache_dir = '/root/dev/data'
os.environ['HF_DATASETS_CACHE'] = dataset_cache_dir
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# ds = load_dataset("AlFrauch/im2latex", cache_dir=dataset_cache_dir)

def load_func4public(dataset_syntax, dataset_meta: DatasetMeta, **load_kwargs):
    ds = load_dataset("unsloth/LaTeX_OCR", split="train", cache_dir=dataset_cache_dir)
    def convert_to_conversation(sample):
        return {
            "messages": [
                    {"role": "user", "content": "<image>请根据图片中的公式生成对应的 latex 语法正确的公式文本。"},
                    {"role": "assistant", "content": sample['text']}
                ],
            "images": [sample['image']],
        }

    ds = ds.map(convert_to_conversation, remove_columns=ds.column_names)
    return ds

def load_func(dataset_syntax, dataset_meta: DatasetMeta, **load_kwargs):
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
    if dataset_meta.dataset_name in filepaths:
        data = []
        for filepath in filepaths[dataset_meta.dataset_name]:
            with open(file=filepath, mode="r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    data.append(item)
        return Dataset.from_list(data)
    return None

from swift.llm.template.template.internvl import Internvl2Template, StdTemplateInputs
_pre_encode = Internvl2Template._encode

aug = LatexAugmentation(apply_prob=0.5, min_ops=1, max_ops=3)
def aug_encode(self, inputs: StdTemplateInputs):
    inputs.images = [aug(inputs.images[0])]
    return _pre_encode(self, inputs)

Internvl2Template._encode = aug_encode


# 注册数据集
register_dataset(DatasetMeta(hf_dataset_id="datav0_imgaug", dataset_name="datav0_imgaug", load_function=load_func))

register_dataset(DatasetMeta(hf_dataset_id="datav0_5_imgaug", dataset_name="datav0_5_imgaug", load_function=load_func))

register_dataset(DatasetMeta(hf_dataset_id="datav0_5_norm_imgaug", dataset_name="datav0_5_norm_imgaug", load_function=load_func))

register_dataset(DatasetMeta(hf_dataset_id="datav1_imgaug", dataset_name="datav1_imgaug",load_function=load_func))

register_dataset(DatasetMeta(hf_dataset_id="datav1_norm_imgaug", dataset_name="datav1_norm_imgaug", load_function=load_func))

register_dataset(DatasetMeta(hf_dataset_id="datav2_norm_imgaug", dataset_name="datav2_norm_imgaug",load_function=load_func))

register_dataset(DatasetMeta(hf_dataset_id="datav2_imgaug", dataset_name="datav2_imgaug", load_function=load_func))

register_dataset(DatasetMeta(hf_dataset_id="datav3_norm_imgaug", dataset_name="datav3_norm_imgaug",load_function=load_func))

if __name__ == '__main__':
    # 必须放在最开头
    # mp.set_start_method('spawn', force=True)
    mp.set_start_method('fork', force=True)
    
    from swift.llm import TrainArguments, sft_main
    
    # 你的训练配置
    args = TrainArguments(
        # model="OpenGVLab/InternVL3_5-1B-HF",
        # model="OpenGVLab/InternVL3_5-1B",
        model="/root/data/dev/ckpts/latex_v0.5_norm_aug_llm_ck_2250",
        # resume_only_model="",
        model_type='internvl3_5', # !!!!!!!!
        # model_type='internlm3', # !!!!!!!!
        # dataset="unsloth/LaTeX_OCR",
        # dataset="/root/dev/data/VLM-formula-recognition-dataset_intern_camp/train/train_mini.jsonl",
        dataset="datav0_5_norm_imgaug",
        lazy_tokenize=True,
        eval_steps=1000,
        train_type="full",
        
        torch_dtype="float16",
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        learning_rate=1e-4,
        warmup_ratio=0.05,
        gradient_accumulation_steps=4,
        save_steps=200,
        save_total_limit=5,
        gradient_checkpointing_kwargs='{"use_reentrant": false}',
        logging_steps=1,
        max_length=8192,
        output_dir="./swift_output/datav0_5_norm_aug_mlp",
        dataset_num_proc=2,
        dataloader_num_workers=4,
        dataloader_persistent_workers=False,
        gradient_checkpointing=True,
        # packing=True,
        # padding_free=True,
        attn_impl='flash_attention_2',
        model_author="kafm",
        model_name="lora-test",
        report_to="swanlab",
        swanlab_token=os.environ['SWANLAB_API_KEY'],
        swanlab_project="internvl-latex",
        swanlab_mode="cloud",
        swanlab_exp_name="datav0_5_norm_aug_mlp",
        metric="acc",
    )
    
    # 启动训练
    result = sft_main(args)