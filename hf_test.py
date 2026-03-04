import torch
from transformers import pipeline, AutoProcessor, AutoTokenizer

messages = [
    {
        "role": "system",
        "content": [{"type": "text", "text": "你是一个 LaTex 识别助手"}],
    },
    {
      "role": "user",
      "content": [
            {"type": "image", "url": "/root/dev/aug.png"},
            {"type": "text", "text": "请根据图片中的公式生成对应的 latex 语法正确的公式文本。"},
        ],
    },
]

import sys
sys.path.insert(0, "/root/dev/swift_output/datav3_norm_7200_continue/v1-20260226-151323/checkpoint-1125-merged")

import importlib, types
pkg = types.ModuleType("ckpt_pkg")
pkg.__path__ = ["/root/dev/swift_output/datav3_norm_7200_continue/v1-20260226-151323/checkpoint-1125-merged"]
sys.modules["ckpt_pkg"] = pkg

from ckpt_pkg.configuration_internvl_chat import InternVLChatConfig
from ckpt_pkg.modeling_internvl_chat import InternVLChatModel
from transformers import AutoConfig, AutoModelForImageTextToText

AutoConfig.register("internvl_chat", InternVLChatConfig, exist_ok=True)
AutoModelForImageTextToText.register(InternVLChatConfig, InternVLChatModel, exist_ok=True)

model = AutoModelForImageTextToText.from_pretrained(
    "/root/dev/swift_output/datav3_norm_7200_continue/v1-20260226-151323/checkpoint-1125-merged",
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)


import sys, types
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoImageProcessor

ckpt = "/root/dev/swift_output/datav3_norm_7200_continue/v1-20260226-151323/checkpoint-1125-merged"

# 注册假包解决相对导入
pkg = types.ModuleType("ckpt_pkg")
pkg.__path__ = [ckpt]
pkg.__package__ = "ckpt_pkg"
sys.modules["ckpt_pkg"] = pkg
sys.path.insert(0, ckpt)

from ckpt_pkg.configuration_intern_vit import InternVisionConfig
from ckpt_pkg.configuration_internvl_chat import InternVLChatConfig
from ckpt_pkg.modeling_intern_vit import InternVisionModel
from ckpt_pkg.modeling_internvl_chat import InternVLChatModel

# 注册到 Auto 类
AutoConfig.register("internvl_chat", InternVLChatConfig, exist_ok=True)
AutoConfig.register("intern_vit_6b", InternVisionConfig, exist_ok=True)
AutoModel.register(InternVLChatConfig, InternVLChatModel, exist_ok=True)

# 加载
config = AutoConfig.from_pretrained(ckpt)  # 不加 trust_remote_code
model = AutoModel.from_pretrained(ckpt, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)

out = pipe(text=messages, max_new_tokens=1024)
print(out[0]['generated_text'][-1]['content'])



# quantization_config = BitsAndBytesConfig(load_in_4bit=True)
# model = AutoModelForCausalLM.from_pretrained("/root/dev/swift_output/datav3_norm_7200_continue/v1-20260226-151323/checkpoint-1125-merged", device_map="auto")
# tokenizer = AutoTokenizer.from_pretrained("/root/dev/swift_output/datav3_norm_7200_continue/v1-20260226-151323/checkpoint-1125-merged")
# model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to(model.device)
# generated_ids = model.generate(**model_inputs)
# tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# "A list of colors: red, blue, green, yellow, orange, purple, pink,"



