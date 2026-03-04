import os
import re
import json
import random
import sys
sys.path.append("/root/dev")
from data.token_stat import analysis
# from baseline.infer_core.latex2img_file import LatexToImage
# latexer = LatexToImage()

random.seed(42)

# ==========================
# 基础符号池（高频优先）
# ==========================

GREEK = [
    r"\alpha", r"\beta", r"\gamma", r"\lambda",
    r"\sigma", r"\theta", r"\mu", r"\rho",
    r"\phi", r"\pi", r"\Omega", r"\Lambda", r"\Gamma", r"\Delta",
]

OPERATORS = [
    r"\nabla", r"\partial", r"\cdot"
]

FUNCTIONS = [
    r"\sin", r"\cos"
]

# ==========================
# 基础构件生成
# ==========================

def rand_var():
    if random.random() < 0.6:
        return random.choice(GREEK)
    return random.choice(["x", "y", "z", "t"])

def rand_index():
    return f"_{{{random.randint(1,5)}}}"

def maybe_index(var):
    if random.random() < 0.5:
        return var + rand_index()
    return var

def gen_fraction():
    a = maybe_index(rand_var())
    b = maybe_index(rand_var())
    return rf"\frac{{{a}}}{{{b}}}"

def gen_sum():
    var = random.choice(["i", "j", "k", "n"])
    upper = random.choice([r"\infty", str(random.randint(5, 20))])
    body = gen_fraction()
    return rf"\sum_{{{var}=1}}^{{{upper}}} {body}"

def gen_integral():
    var = random.choice(["x", "t"])
    body = maybe_index(rand_var())
    return rf"\int_0^{{\infty}} {body} \, d{var}"

def gen_gradient():
    return rf"\nabla {maybe_index(rand_var())}"

def gen_sqrt():
    return rf"\sqrt{{{maybe_index(rand_var())}}}"

def gen_trig():
    f = random.choice(FUNCTIONS)
    return rf"{f}\left({maybe_index(rand_var())}\right)"

def gen_element(depth=0):
    """
    生成矩阵元素
    depth 控制嵌套层数
    """
    choices = [
        lambda: maybe_index(rand_var()),
        gen_fraction,
        gen_sum,
        gen_integral,
        gen_gradient,
        gen_sqrt,
        gen_trig
    ]

    # 控制嵌套概率
    if depth < 1 and random.random() < 0.2:
        return gen_matrix(depth=depth+1, small=True)

    return random.choice(choices)()

# ==========================
# 矩阵生成
# ==========================

def gen_matrix(depth=0, small=False):
    rows = random.randint(2, 4) if not small else random.randint(2, 3)
    cols = random.randint(2, 4) if not small else random.randint(2, 3)

    env = random.choices(
        ["bmatrix", "pmatrix"],
        weights=[2, 1]  # 保持 bmatrix 更高频
    )[0]

    matrix_rows = []

    for r in range(rows):
        row = []
        for c in range(cols):
            if random.random() < 0.1:
                row.append(r"\cdots")
            elif random.random() < 0.1:
                row.append(r"\vdots")
            elif random.random() < 0.05:
                row.append(r"\ddots")
            else:
                row.append(gen_element(depth))
        matrix_rows.append(" & ".join(row))

    body = r" \\ ".join(matrix_rows)

    return rf"\begin{{{env}}} {body} \end{{{env}}}"

# ==========================
# 长公式拼接
# ==========================

def gen_long_formula():
    parts = []

    for _ in range(random.randint(1, 2)):
        parts.append(gen_matrix())

    # 使用 quad 连接，增强长度
    return r"\quad ".join(parts)

# ==========================
# 主接口
# ==========================

def generate_samples(n=10, long_mode=True):
    results = []
    for _ in range(n):
        if long_mode:
            results.append(gen_long_formula())
        else:
            results.append(gen_matrix())
    return results


def datav0_5():
    dataset_dir = "/root/dev/data/datav0_5"
    os.system(f"rm -rf {dataset_dir}")
    img_dir = f"{dataset_dir}/imgs"
    os.makedirs(name=img_dir, exist_ok=True)

    # 处理复制样本
    print("处理复制样本")
    with open(f"/root/dev/data/VLM-formula-recognition-dataset_intern_camp/train/train_mini.jsonl", mode="r") as f, \
      open(f"{dataset_dir}/train.jsonl", mode="w") as ff:
        lines = list(map(lambda line: json.loads(line), f.readlines()))
        for idx, sample in enumerate(lines):
            img_name = sample["images"][0].split("/")[-1]
            sample: str = sample["messages"][1]["content"]
            sample = sample.replace('pmatrix', '§')
            sample = sample.replace('bmatrix', 'pmatrix')
            sample = sample.replace('§', 'bmatrix')
            sample = re.sub(r"\n", " ", sample.strip())
            sample = re.sub(r"\s+", " ", sample.strip())
            img_path = f"{img_dir}/r_{img_name}.png"
            
            assert latexer.latex_to_image(sample, img_path)
            item = '{"messages": [{"role": "user", "content": "<image>请根据图片中的公式生成对应的 latex 公式文本"}, {"role": "assistant", "content": ' + json.dumps(sample) + '}], "images": [' + json.dumps(img_path) + ']}'
            ff.write(item)
            ff.write("\n")

    # 合并原样本
    print("合并原样本")
    os.system(f"cp /root/dev/data/VLM-formula-recognition-dataset_intern_camp/train/mini_train/* {img_dir}")
    with open(f"/root/dev/data/VLM-formula-recognition-dataset_intern_camp/train/train_mini.jsonl", mode="r") as f, \
      open(f"{dataset_dir}/train.jsonl", mode="a") as ff:
        def func(sample: str):
            sample = sample.replace("/root/dev/data/VLM-formula-recognition-dataset_intern_camp/train/mini_train/", f"{img_dir}/")
            sample = re.sub(r"\n", " ", sample.strip())
            sample = re.sub(r"\s+", " ", sample.strip())
            return sample
        for line in f.readlines():
            ff.write(func(line))
            ff.write("\n")

    # 处理生成样本
    print("处理生成样本")
    samples = generate_samples(3000, long_mode=True)
    with open(f"{dataset_dir}/train.jsonl", mode="a") as f:
        for idx, sample in enumerate(samples):
            img_path = f"{img_dir}/g_sample_{idx}.png"
            sample = re.sub(r"\n", " ", sample.strip())
            sample = re.sub(r"\s+", " ", sample.strip())
            assert latexer.latex_to_image(sample, img_path)
            item = '{"messages": [{"role": "user", "content": "<image>请根据图片中的公式生成对应的 latex 公式文本"}, {"role": "assistant", "content": ' + json.dumps(sample) + '}], "images": [' + json.dumps(img_path) + ']}'
            f.write(item)
            f.write("\n")

    analysis(samples, 20, "generator")

def datav_3():
    """严格遵循分布"""
    dataset_dir = "/root/dev/data/datav3"
    img_dir = f"{dataset_dir}/gimgs"
    os.makedirs(name=img_dir, exist_ok=True)
    with open("/root/dev/data/newgenerator/generated.jsonl", "r") as f, \
        open("/root/dev/data/datav3/gtrain.jsonl", "w") as ff:
        for idx, sample in enumerate(f.readlines()):
            img_path = f"{img_dir}/g_sample_{idx}.png"
            sample = json.loads(sample)
            sample = re.sub(r"\n", " ", sample.strip())
            sample = re.sub(r"\s+", " ", sample.strip())
            # assert latexer.latex_to_image(sample, img_path)
            item = '{"messages": [{"role": "user", "content": "<image>请根据图片中的公式生成对应的 latex 公式文本"}, {"role": "assistant", "content": ' + json.dumps(sample) + '}], "images": ["' + img_path + '"]}'
            ff.write(item)
            ff.write("\n")


def normalize_latex(s):
    """
    标准化 LaTeX 公式字符串：
    - 替换 \lt, \gt
    - 移除换行，压缩空格
    - 去除命令与括号、下标、上标之间的多余空格
    - 处理 \left, \right 及 &, \\ 等
    """
    # 1. 替换 \lt 和 \gt
    s = re.sub(r'\\lt\b', '<', s)
    s = re.sub(r'\\gt\b', '>', s)

    # 2. 移除换行符，统一空白字符
    s = s.replace('\n', ' ').replace('\r', '')
    s = re.sub(r'\s+', ' ', s).strip()

    # 3. 处理命令后跟括号、下划线、上标的空格
    # 命令后跟左括号 ( [ {
    s = re.sub(r'\\([a-zA-Z]+)\*?\s+([({\[])', r'\\\1\2', s)
    # 命令后跟下划线 _
    s = re.sub(r'\\([a-zA-Z]+)\*?\s+_', r'\\\1_', s)
    # 命令后跟上标 ^
    s = re.sub(r'\\([a-zA-Z]+)\*?\s+\^', r'\\\1^', s)

    # 4. 处理 \left 和 \right 周围的空格
    s = re.sub(r'\\left\s+([\(\[\{])', r'\\left\1', s)
    s = re.sub(r'\\right\s+([\)\]\}])', r'\\right\1', s)
    s = re.sub(r'\\right\s+\.', r'\\right.', s)

    # 5. 处理左括号后和右括号前的空格
    s = re.sub(r'(\\left[\(\[\{])\s+', r'\1', s)
    s = re.sub(r'\s+(\\right[\)\]\}])', r'\1', s)

    # 6. 处理 & 周围空格
    s = re.sub(r'\s*&\s*', '&', s)

    # 7. 处理 \\ 周围空格及可选参数
    s = re.sub(r'\\\\\s+(\[)', r'\\\\\1', s)
    s = re.sub(r'\\\\\*\s+(\[)', r'\\\\*\1', s)
    s = re.sub(r'\s*\\\\\s*', r'\\\\', s)
    s = re.sub(r'\s*\\\\\*\s*', r'\\\\*', s)

    # 8. 处理右括号与左括号之间的空格：} {, ] {, ) {
    s = re.sub(r'([\]\)\}])\s+{', r'\1{', s)
    s = re.sub(r'([\]\)\}])\s+\[', r'\1[', s)
    s = re.sub(r'([\]\)\}])\s+\(', r'\1(', s)

    # 9. 处理 } 与 _、^ 之间的空格
    s = re.sub(r'}\s+_', r'}_', s)
    s = re.sub(r'}\s+\^', r'}^', s)

    # 10. 再次压缩可能出现的多余空格
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def normalize(fromfile, tofile):
    with open(fromfile, "r") as f, \
        open(tofile, "w") as ff:
        for line in f.readlines():
            sample = json.loads(line)
            latex_string = sample["messages"][1]["content"]
            latex_string = normalize_latex(latex_string)
            img_path = sample["images"][0]

            item = '{"messages": [{"role": "user", "content": "<image>请根据图片中的公式生成对应的 latex 公式文本"}, {"role": "assistant", "content": ' + json.dumps(latex_string) + '}], "images": [' + json.dumps(img_path) + ']}'
            ff.write(item)
            ff.write("\n")

            

# ==========================
# 示例
# ==========================

if __name__ == "__main__":

    datav_3()
    normalize("/root/dev/data/datav3/gtrain.jsonl", "/root/dev/data/datav3/gtrain_norm.jsonl")



